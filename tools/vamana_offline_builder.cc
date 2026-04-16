/**
 * Offline Vamana index builder for SHINE.
 *
 * Builds a Vamana graph on CPU, computes RaBitQ quantization,
 * and serializes to SHINE memory node shard format with fixed-size VamanaNode records.
 *
 * Steps:
 *   1. Read dataset from file
 *   2. Compute medoid (geometric median approximation)
 *   3. Generate random orthogonal matrix P (Eigen QR)
 *   4. Build Vamana graph: greedy insert with beam search + RobustPrune
 *   5. Quantize all vectors using RaBitQ
 *   6. Serialize to SHINE VamanaNode shard format
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <vector>

#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include <library/utils.hh>

#ifdef __AVX__
#include <x86intrin.h>
#endif

#include "common/index_path.hh"
#include "common/types.hh"
#include "gpu/gpu_kernel_launcher.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace po = boost::program_options;

namespace {

// ============================================================================
// Configuration
// ============================================================================

struct VamanaBuildConfig {
  filepath_t data_path{};
  filepath_t output_prefix{};
  filepath_t query_path{};       // for post-build recall test
  filepath_t groundtruth_path{}; // for post-build recall test
  u32 num_memory_nodes{1};
  u32 threads{0};
  u32 R{64};                // max out-degree
  u32 beam_width{128};      // beam width used by beam search during offline construction
  f64 alpha{1.2};           // RobustPrune diversity factor
  u32 rabitq_bits{1};       // bits per dimension
  i32 seed{1234};
  size_t max_vectors{std::numeric_limits<u32>::max()};
  bool ip_distance{false};
  bool no_gpu{false};
  i32 gpu_device{0};
};

// ============================================================================
// Dataset
// ============================================================================

struct Dataset {
  filepath_t source_file{};
  u32 dim{0};
  size_t total_vectors{0};
  vec<element_t> vectors;
  vec<node_t> ids;

  const float* vector(size_t i) const { return vectors.data() + i * dim; }
};

// ============================================================================
// Utilities
// ============================================================================

size_t effective_thread_count(u32 configured_threads) {
  const size_t detected = std::thread::hardware_concurrency();
  return configured_threads == 0 ? std::max<size_t>(detected, 1) : configured_threads;
}

str format_duration(std::chrono::steady_clock::duration duration) {
  const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
  const auto hours = seconds / 3600;
  const auto minutes = (seconds % 3600) / 60;
  const auto secs = seconds % 60;
  std::ostringstream os;
  if (hours > 0) os << hours << "h" << minutes << "m" << secs << "s";
  else if (minutes > 0) os << minutes << "m" << secs << "s";
  else os << secs << "s";
  return os.str();
}

class ProgressReporter {
public:
  ProgressReporter(str label, size_t total)
      : label_(std::move(label)),
        total_(std::max<size_t>(total, 1)),
        interactive_(::isatty(fileno(stderr)) != 0),
        start_(std::chrono::steady_clock::now()),
        last_render_(start_),
        thread_([this]() { run(); }) {}

  ~ProgressReporter() { finish(); }

  void increment(size_t value = 1) { current_.fetch_add(value, std::memory_order_relaxed); }

  void finish() {
    if (!finished_.exchange(true, std::memory_order_relaxed)) {
      current_.store(total_, std::memory_order_relaxed);
      if (thread_.joinable()) thread_.join();
    }
  }

private:
  void run() {
    while (!finished_.load(std::memory_order_relaxed)) {
      render(false);
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    render(true);
  }

  void render(bool done) {
    const size_t current = std::min(current_.load(std::memory_order_relaxed), total_);
    const double ratio = static_cast<double>(current) / static_cast<double>(total_);
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - start_;

    std::ostringstream os;
    os << label_ << " ";

    if (interactive_) {
      constexpr size_t bar_width = 28;
      const size_t filled = static_cast<size_t>(ratio * static_cast<double>(bar_width));
      os << "[";
      for (size_t i = 0; i < bar_width; ++i) os << (i < filled ? '=' : ' ');
      os << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << "% ";
      os << "(" << current << "/" << total_ << ") ";
      os << "elapsed " << format_duration(elapsed);
      if (current > 0 && current < total_) {
        const auto estimated = std::chrono::duration_cast<std::chrono::steady_clock::duration>(elapsed / ratio);
        os << " eta " << format_duration(estimated - elapsed);
      }
      std::cerr << '\r' << os.str();
      if (done) std::cerr << '\n';
      std::cerr.flush();
      return;
    }

    const size_t bucket = done ? 20 : static_cast<size_t>(ratio * 20.0);
    const auto log_interval = std::chrono::seconds(15);
    if (!done && bucket <= last_bucket_ && (now - last_render_) < log_interval) return;
    last_bucket_ = std::max(last_bucket_, bucket);
    last_render_ = now;

    os << static_cast<int>(ratio * 100.0) << "% (" << current << "/" << total_ << ") elapsed "
       << format_duration(elapsed);
    if (done) os << " done";
    std::cerr << os.str() << '\n';
  }

  const str label_;
  const size_t total_;
  const bool interactive_;
  const std::chrono::steady_clock::time_point start_;
  std::atomic<size_t> current_{0};
  std::atomic<bool> finished_{false};
  size_t last_bucket_{0};
  std::chrono::steady_clock::time_point last_render_;
  std::thread thread_;
};

template <class Function>
void parallel_for(size_t begin, size_t end, size_t num_threads, Function&& fn) {
  num_threads = effective_thread_count(num_threads);
  if (num_threads == 1 || end <= begin + 1) {
    for (size_t i = begin; i < end; ++i) fn(i, 0);
    return;
  }
  std::atomic<size_t> current{begin};
  std::exception_ptr last_exception;
  std::mutex exception_mutex;
  vec<std::thread> threads;
  threads.reserve(num_threads);
  for (size_t tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      for (;;) {
        const size_t i = current.fetch_add(1);
        if (i >= end) return;
        try { fn(i, tid); }
        catch (...) {
          std::lock_guard<std::mutex> lock(exception_mutex);
          if (!last_exception) last_exception = std::current_exception();
          current.store(end);
          return;
        }
      }
    });
  }
  for (auto& t : threads) t.join();
  if (last_exception) std::rethrow_exception(last_exception);
}

u32 read_u32(std::ifstream& input) {
  u32 value{};
  if (!input.read(reinterpret_cast<char*>(&value), sizeof(value)))
    lib_failure("failed to read u32 from dataset");
  return value;
}

filepath_t resolve_dataset_file(const filepath_t& input_path) {
  if (std::filesystem::is_regular_file(input_path)) return input_path;
  if (!std::filesystem::is_directory(input_path)) return input_path;
  static const vec<str> candidates = {"base.fbin", "base.u8bin", "base.i8bin", "base.bin"};
  for (const auto& c : candidates) {
    const filepath_t path = input_path / c;
    if (std::filesystem::exists(path)) return path;
  }
  lib_failure("unable to resolve dataset file under " + input_path.string());
  return {};
}

Dataset read_dataset(const VamanaBuildConfig& config) {
  Dataset dataset;
  dataset.source_file = resolve_dataset_file(config.data_path);

  std::ifstream input(dataset.source_file, std::ios::binary);
  lib_assert(input.good(), "dataset file does not exist: " + dataset.source_file.string());

  const str ext = dataset.source_file.extension().string();
  const bool is_float32 = ext == ".fbin" || ext == ".bin";
  const bool is_uint8 = ext == ".u8bin";
  const bool is_int8 = ext == ".i8bin";
  lib_assert(is_float32 || is_uint8 || is_int8, "unsupported dataset extension: " + ext);

  dataset.total_vectors = read_u32(input);
  dataset.dim = read_u32(input);

  const size_t num_vectors = std::min(dataset.total_vectors, config.max_vectors);
  lib_assert(num_vectors > 0, "dataset is empty");

  std::cerr << "reading dataset " << dataset.source_file
            << " (dim=" << dataset.dim << ", vectors=" << num_vectors
            << "/" << dataset.total_vectors << ")\n";

  dataset.vectors.resize(num_vectors * dataset.dim);
  dataset.ids.resize(num_vectors);
  std::iota(dataset.ids.begin(), dataset.ids.end(), 0);

  if (is_float32) {
    ProgressReporter progress{"Reading dataset", num_vectors};
    const size_t rows_per_chunk = std::max<size_t>(1, (8 * 1024 * 1024) / (dataset.dim * sizeof(element_t)));
    for (size_t row = 0; row < num_vectors; row += rows_per_chunk) {
      const size_t chunk_rows = std::min(rows_per_chunk, num_vectors - row);
      const size_t chunk_bytes = chunk_rows * dataset.dim * sizeof(element_t);
      if (!input.read(reinterpret_cast<char*>(dataset.vectors.data() + row * dataset.dim), chunk_bytes))
        lib_failure("failed to read float32 dataset payload");
      progress.increment(chunk_rows);
    }
    progress.finish();
  } else if (is_uint8) {
    vec<u8> raw(dataset.vectors.size());
    if (!input.read(reinterpret_cast<char*>(raw.data()), raw.size()))
      lib_failure("failed to read uint8 dataset payload");
    ProgressReporter progress{"Converting dataset", num_vectors};
    parallel_for(0, num_vectors, config.threads, [&](size_t row, size_t) {
      const size_t base = row * dataset.dim;
      for (size_t col = 0; col < dataset.dim; ++col)
        dataset.vectors[base + col] = static_cast<element_t>(raw[base + col]);
      progress.increment();
    });
    progress.finish();
  } else {
    vec<i8> raw(dataset.vectors.size());
    if (!input.read(reinterpret_cast<char*>(raw.data()), raw.size()))
      lib_failure("failed to read int8 dataset payload");
    ProgressReporter progress{"Converting dataset", num_vectors};
    parallel_for(0, num_vectors, config.threads, [&](size_t row, size_t) {
      const size_t base = row * dataset.dim;
      for (size_t col = 0; col < dataset.dim; ++col)
        dataset.vectors[base + col] = static_cast<element_t>(raw[base + col]);
      progress.increment();
    });
    progress.finish();
  }

  return dataset;
}

// ============================================================================
// Distance computation (CPU)
// ============================================================================

float l2_squared(const float* a, const float* b, u32 dim) {
#ifdef __AVX__
  __m256 sum = _mm256_setzero_ps();
  u32 i = 0;
  for (; i + 16 <= dim; i += 16) {
    __m256 v1 = _mm256_loadu_ps(a + i);
    __m256 v2 = _mm256_loadu_ps(b + i);
    __m256 d = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(d, d));
    v1 = _mm256_loadu_ps(a + i + 8);
    v2 = _mm256_loadu_ps(b + i + 8);
    d = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(d, d));
  }
  float __attribute__((aligned(32))) tmp[8];
  _mm256_store_ps(tmp, sum);
  float result = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < dim; ++i) {
    float d = a[i] - b[i];
    result += d * d;
  }
  return result;
#else
  float sum = 0.0f;
  for (u32 i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
#endif
}

float ip_distance(const float* a, const float* b, u32 dim) {
#ifdef __AVX__
  __m256 sum = _mm256_setzero_ps();
  u32 i = 0;
  for (; i + 16 <= dim; i += 16) {
    __m256 v1 = _mm256_loadu_ps(a + i);
    __m256 v2 = _mm256_loadu_ps(b + i);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
    v1 = _mm256_loadu_ps(a + i + 8);
    v2 = _mm256_loadu_ps(b + i + 8);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
  }
  float __attribute__((aligned(32))) tmp[8];
  _mm256_store_ps(tmp, sum);
  float dot = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < dim; ++i) dot += a[i] * b[i];
  return -dot;
#else
  float sum = 0.0f;
  for (u32 i = 0; i < dim; ++i) sum += a[i] * b[i];
  return -sum;
#endif
}

using DistFn = float(*)(const float*, const float*, u32);

// ============================================================================
// CPU Vamana graph builder
// ============================================================================

struct VamanaGraph {
  size_t num_nodes{0};
  u32 dim{0};
  u32 R{0};            // max out-degree
  size_t medoid{0};    // medoid node index

  // Adjacency lists: neighbors[i] = list of neighbor indices for node i
  vec<vec<u32>> neighbors;
  // Per-node mutex for concurrent reverse-edge updates
  vec<std::mutex> node_locks;

  void init(size_t n, u32 d, u32 max_degree) {
    num_nodes = n;
    dim = d;
    R = max_degree;
    neighbors.resize(n);
    node_locks = vec<std::mutex>(n);
  }
};

// Minimum batch size for GPU to be faster than CPU SIMD
static constexpr u32 GPU_BATCH_THRESHOLD = 16;

/**
 * Per-thread GPU context for accelerated distance computation.
 * Each thread gets its own CUDA stream + staging buffers.
 */
struct BuilderGpuContext {
  cudaStream_t stream{nullptr};
  cudaEvent_t event{nullptr};

  // Host pinned buffers
  float* h_query{nullptr};
  float* h_candidates{nullptr};
  float* h_distances{nullptr};
  float* h_candidate_dists{nullptr};
  u32*   h_pruned_indices{nullptr};
  u32*   h_pruned_count{nullptr};

  // Device buffers
  float* d_query{nullptr};
  float* d_candidates{nullptr};
  float* d_distances{nullptr};
  float* d_candidate_dists{nullptr};
  u32*   d_pruned_indices{nullptr};
  u32*   d_pruned_count{nullptr};

  u32 dim{0};
  u32 max_candidates{0};
  u32 max_R{0};

  void init(u32 dim_, u32 max_cand, u32 R) {
    dim = dim_;
    max_candidates = max_cand;
    max_R = R;

    stream = gpu::gpu_stream_create();
    event  = gpu::gpu_event_create();

    h_query           = static_cast<float*>(gpu::gpu_malloc_host(dim * sizeof(float)));
    h_candidates      = static_cast<float*>(gpu::gpu_malloc_host(max_cand * dim * sizeof(float)));
    h_distances       = static_cast<float*>(gpu::gpu_malloc_host(max_cand * sizeof(float)));
    h_candidate_dists = static_cast<float*>(gpu::gpu_malloc_host(max_cand * sizeof(float)));
    h_pruned_indices  = static_cast<u32*>(gpu::gpu_malloc_host(R * sizeof(u32)));
    h_pruned_count    = static_cast<u32*>(gpu::gpu_malloc_host(sizeof(u32)));

    d_query           = static_cast<float*>(gpu::gpu_malloc(dim * sizeof(float)));
    d_candidates      = static_cast<float*>(gpu::gpu_malloc(max_cand * dim * sizeof(float)));
    d_distances       = static_cast<float*>(gpu::gpu_malloc(max_cand * sizeof(float)));
    d_candidate_dists = static_cast<float*>(gpu::gpu_malloc(max_cand * sizeof(float)));
    d_pruned_indices  = static_cast<u32*>(gpu::gpu_malloc(R * sizeof(u32)));
    d_pruned_count    = static_cast<u32*>(gpu::gpu_malloc(sizeof(u32)));
  }

  void destroy() {
    if (!stream) return;
    gpu::gpu_free_host(h_query);
    gpu::gpu_free_host(h_candidates);
    gpu::gpu_free_host(h_distances);
    gpu::gpu_free_host(h_candidate_dists);
    gpu::gpu_free_host(h_pruned_indices);
    gpu::gpu_free_host(h_pruned_count);
    gpu::gpu_free(d_query);
    gpu::gpu_free(d_candidates);
    gpu::gpu_free(d_distances);
    gpu::gpu_free(d_candidate_dists);
    gpu::gpu_free(d_pruned_indices);
    gpu::gpu_free(d_pruned_count);
    gpu::gpu_event_destroy(event);
    gpu::gpu_stream_destroy(stream);
    stream = nullptr;
  }
};

/**
 * Compute medoid: the vector with minimum sum of distances to all others.
 * Uses sampling for large datasets.
 */
size_t compute_medoid(const Dataset& dataset, DistFn dist_fn) {
  const size_t n = dataset.ids.size();
  const u32 dim = dataset.dim;

  // For large datasets, sample to find approximate medoid
  const size_t sample_size = std::min<size_t>(n, 10000);
  vec<size_t> sample_indices(n);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  if (sample_size < n) {
    std::mt19937 rng(42);
    std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
    sample_indices.resize(sample_size);
  }

  // Compute centroid
  vec<float> centroid(dim, 0.0f);
  for (size_t idx : sample_indices) {
    const float* v = dataset.vector(idx);
    for (u32 d = 0; d < dim; ++d) centroid[d] += v[d];
  }
  for (u32 d = 0; d < dim; ++d) centroid[d] /= static_cast<float>(sample_size);

  // Find vector closest to centroid
  size_t best = 0;
  float best_dist = std::numeric_limits<float>::max();
  for (size_t i = 0; i < n; ++i) {
    float d = dist_fn(dataset.vector(i), centroid.data(), dim);
    if (d < best_dist) {
      best_dist = d;
      best = i;
    }
  }
  return best;
}

/**
 * Beam search from medoid to find nearest candidates for a query vector.
 * Thread-safe: reads neighbor lists under per-node locks.
 * Optionally uses GPU for batch distance computation.
 */
/**
 * Beam search from medoid to find nearest candidates for a query vector.
 * Thread-safe: reads neighbor lists under per-node locks.
 * Optionally uses GPU for batch distance computation.
 *
 * Returns ALL visited nodes with their distances (the full visited set V),
 * sorted by distance. The DiskANN paper uses V (not just the beam L) as
 * candidates for RobustPrune.
 */
vec<std::pair<float, u32>> beam_search(VamanaGraph& graph,
                                       const Dataset& dataset,
                                       const float* query,
                                       u32 beam_width,
                                       DistFn dist_fn,
                                       BuilderGpuContext* gpu_ctx = nullptr) {
  const u32 dim = dataset.dim;

  // all_visited: every node we computed distance for (returned to caller)
  vec<std::pair<float, u32>> all_visited;
  // beam: top beam_width candidates used for navigation
  vec<std::pair<float, u32>> beam;
  std::unordered_set<u32> visited;
  std::unordered_set<u32> expanded;

  float medoid_dist = dist_fn(query, dataset.vector(graph.medoid), dim);
  beam.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  all_visited.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  visited.insert(static_cast<u32>(graph.medoid));

  // Upload query to GPU once per search
  bool query_on_gpu = false;
  if (gpu_ctx) {
    std::memcpy(gpu_ctx->h_query, query, dim * sizeof(float));
    gpu::gpu_memcpy_h2d_async(gpu_ctx->d_query, gpu_ctx->h_query,
                               dim * sizeof(float), gpu_ctx->stream);
    gpu::gpu_stream_synchronize(gpu_ctx->stream);
    query_on_gpu = true;
  }

  while (true) {
    // Find the closest unexpanded node in the (sorted) beam
    ssize_t best_pos = -1;
    for (size_t i = 0; i < beam.size(); ++i) {
      if (expanded.count(beam[i].second) == 0) {
        best_pos = static_cast<ssize_t>(i);
        break;
      }
    }
    if (best_pos < 0) break;

    u32 best_node = beam[best_pos].second;
    expanded.insert(best_node);

    // Read neighbors under lock (thread-safe)
    vec<u32> nbrs;
    {
      std::lock_guard<std::mutex> lock(graph.node_locks[best_node]);
      nbrs = graph.neighbors[best_node];
    }

    // Collect unvisited neighbors
    vec<u32> unvisited;
    for (u32 nbr : nbrs) {
      if (visited.count(nbr)) continue;
      visited.insert(nbr);
      unvisited.push_back(nbr);
    }

    if (!unvisited.empty()) {
      if (gpu_ctx && query_on_gpu && unvisited.size() >= GPU_BATCH_THRESHOLD) {
        // GPU batch distance path
        const u32 batch = static_cast<u32>(std::min<size_t>(unvisited.size(), gpu_ctx->max_candidates));
        for (u32 j = 0; j < batch; ++j) {
          std::memcpy(gpu_ctx->h_candidates + static_cast<size_t>(j) * dim,
                       dataset.vector(unvisited[j]),
                       dim * sizeof(float));
        }
        gpu::gpu_memcpy_h2d_async(gpu_ctx->d_candidates, gpu_ctx->h_candidates,
                                   static_cast<size_t>(batch) * dim * sizeof(float), gpu_ctx->stream);
        gpu::launch_batch_l2_distances(gpu_ctx->stream, gpu_ctx->event,
                                        gpu_ctx->d_query, gpu_ctx->d_candidates,
                                        gpu_ctx->d_distances, batch, dim);
        gpu::gpu_memcpy_d2h_async(gpu_ctx->h_distances, gpu_ctx->d_distances,
                                   batch * sizeof(float), gpu_ctx->stream);
        gpu::gpu_stream_synchronize(gpu_ctx->stream);
        for (u32 j = 0; j < batch; ++j) {
          beam.push_back({gpu_ctx->h_distances[j], unvisited[j]});
          all_visited.push_back({gpu_ctx->h_distances[j], unvisited[j]});
        }
        // Handle any overflow beyond max_candidates with CPU
        for (size_t j = batch; j < unvisited.size(); ++j) {
          float d = dist_fn(query, dataset.vector(unvisited[j]), dim);
          beam.push_back({d, unvisited[j]});
          all_visited.push_back({d, unvisited[j]});
        }
      } else {
        // CPU SIMD path
        for (u32 nbr : unvisited) {
          float d = dist_fn(query, dataset.vector(nbr), dim);
          beam.push_back({d, nbr});
          all_visited.push_back({d, nbr});
        }
      }
    }

    std::sort(beam.begin(), beam.end());
    if (beam.size() > beam_width) beam.resize(beam_width);
  }

  std::sort(all_visited.begin(), all_visited.end());
  return all_visited;
}

/**
 * RobustPrune: select up to R diverse neighbors from sorted candidates.
 *
 * For each candidate p* (in order of increasing distance from source):
 *   Accept p* unless there exists an already-selected p' such that
 *   alpha * dist(p*, p') <= dist(source, p*)
 */
vec<u32> robust_prune(const Dataset& dataset,
                      u32 source,
                      const vec<std::pair<float, u32>>& sorted_candidates,
                      float alpha,
                      u32 R,
                      DistFn dist_fn) {
  const u32 dim = dataset.dim;
  vec<u32> selected;
  selected.reserve(R);

  for (const auto& [cand_dist, cand_id] : sorted_candidates) {
    if (cand_id == source) continue;
    if (selected.size() >= R) break;

    bool pruned = false;
    for (u32 sel_id : selected) {
      float d_sel_cand = dist_fn(dataset.vector(sel_id), dataset.vector(cand_id), dim);
      if (alpha * d_sel_cand <= cand_dist) {
        pruned = true;
        break;
      }
    }

    if (!pruned) {
      selected.push_back(cand_id);
    }
  }

  return selected;
}

/**
 * Build the Vamana graph using parallel insertion with optional GPU acceleration.
 *
 * Algorithm from DiskANN paper:
 *   1. Compute medoid
 *   2. Sequential warmup: insert first R*2 nodes (graph too sparse for parallelism)
 *   3. Parallel insert remaining nodes: beam search → RobustPrune → locked edge updates
 */
void build_vamana_graph(VamanaGraph& graph,
                        const Dataset& dataset,
                        const VamanaBuildConfig& config,
                        DistFn dist_fn,
                        BuilderGpuContext* gpu_contexts,
                        size_t num_gpu_contexts) {
  const size_t n = dataset.ids.size();
  const u32 dim = dataset.dim;
  const u32 R = config.R;
  const float alpha = static_cast<float>(config.alpha);
  const u32 beam_width = config.beam_width;
  const size_t num_threads = effective_thread_count(config.threads);
  const bool use_gpu = num_gpu_contexts > 0 && !config.ip_distance;

  graph.init(n, dim, R);

  std::cerr << "computing medoid...\n";
  graph.medoid = compute_medoid(dataset, dist_fn);
  std::cerr << "medoid: node " << graph.medoid << "\n";

  // Random insertion order
  vec<size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  const size_t seed = config.seed == -1 ? std::random_device{}() : static_cast<size_t>(config.seed);
  std::mt19937 rng(seed);
  std::shuffle(order.begin(), order.end(), rng);

  // Initialize graph with random R-regular directed edges (DiskANN Algorithm 2, step 1).
  // This provides baseline connectivity that the Vamana insertion pass refines.
  {
    std::cerr << "initializing random R-regular graph (degree=" << R << ")...\n";
    std::mt19937 init_rng(seed + 1);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    for (size_t i = 0; i < n; ++i) {
      graph.neighbors[i].reserve(R);
      while (graph.neighbors[i].size() < R) {
        size_t j = dist(init_rng);
        if (j == i) continue;
        bool dup = false;
        for (u32 k : graph.neighbors[i]) {
          if (k == static_cast<u32>(j)) { dup = true; break; }
        }
        if (!dup) graph.neighbors[i].push_back(static_cast<u32>(j));
      }
    }
  }

  // Construction uses alpha=1.0 for reliable connectivity in high dimensions.
  // Alpha > 1 causes aggressive pruning that disconnects the directed graph when
  // distances are concentrated (curse of dimensionality).
  const float build_alpha = 1.0f;
  if (alpha > 1.0f + 1e-6f) {
    std::cerr << "note: using alpha=1.0 for construction (config alpha="
              << alpha << " stored in metadata)\n";
  }

  ProgressReporter progress{"Building Vamana graph", n};

  // Node insertion logic
  auto insert_node = [&](size_t step, size_t tid) {
    const size_t node_idx = order[step];

    BuilderGpuContext* gpu_ctx = (use_gpu && tid < num_gpu_contexts) ? &gpu_contexts[tid] : nullptr;

    // Beam search to find candidate neighbors (returns ALL visited nodes)
    const float* query = dataset.vector(node_idx);
    auto candidates = beam_search(graph, dataset, query, beam_width, dist_fn, gpu_ctx);

    // Merge with existing neighbors (preserves connectivity from random init)
    {
      std::lock_guard<std::mutex> lock(graph.node_locks[node_idx]);
      for (u32 existing : graph.neighbors[node_idx]) {
        float d = dist_fn(query, dataset.vector(existing), dim);
        candidates.push_back({d, existing});
      }
    }
    std::sort(candidates.begin(), candidates.end());

    // Deduplicate
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.second == b.second; }),
        candidates.end());

    vec<u32> new_neighbors = robust_prune(
        dataset, static_cast<u32>(node_idx), candidates, build_alpha, R, dist_fn);

    // Set forward edges (lock own node)
    {
      std::lock_guard<std::mutex> lock(graph.node_locks[node_idx]);
      graph.neighbors[node_idx] = new_neighbors;
    }

    // Add reverse edges (lock each neighbor individually)
    for (u32 nbr : new_neighbors) {
      std::lock_guard<std::mutex> lock(graph.node_locks[nbr]);
      auto& nbr_list = graph.neighbors[nbr];

      // Check if already present
      bool already_present = false;
      for (u32 existing : nbr_list) {
        if (existing == static_cast<u32>(node_idx)) {
          already_present = true;
          break;
        }
      }
      if (already_present) continue;

      if (nbr_list.size() < R) {
        nbr_list.push_back(static_cast<u32>(node_idx));
      } else {
        // Need to prune: collect current neighbors + new node as candidates
        vec<std::pair<float, u32>> prune_candidates;
        prune_candidates.reserve(nbr_list.size() + 1);

        const float* nbr_vec = dataset.vector(nbr);
        for (u32 existing : nbr_list) {
          float d = dist_fn(nbr_vec, dataset.vector(existing), dim);
          prune_candidates.push_back({d, existing});
        }
        float d_new = dist_fn(nbr_vec, dataset.vector(node_idx), dim);
        prune_candidates.push_back({d_new, static_cast<u32>(node_idx)});

        std::sort(prune_candidates.begin(), prune_candidates.end());
        nbr_list = robust_prune(dataset, nbr, prune_candidates, build_alpha, R, dist_fn);
      }
    }

    progress.increment();
  };

  // Parallel construction (random init provides connectivity, no warmup needed)
  parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t step, size_t tid) {
    insert_node(step, tid);
  });

  progress.finish();

  // Print graph stats
  size_t total_edges = 0;
  size_t max_edges = 0;
  size_t min_edges = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < n; ++i) {
    total_edges += graph.neighbors[i].size();
    max_edges = std::max(max_edges, graph.neighbors[i].size());
    min_edges = std::min(min_edges, graph.neighbors[i].size());
  }
  std::cerr << "graph stats: avg_degree=" << (static_cast<double>(total_edges) / n)
            << " max=" << max_edges << " min=" << min_edges << "\n";

  // Quick in-memory recall sanity check
  {
    const size_t n_queries = std::min<size_t>(200, n);
    const u32 topk = 10;
    std::mt19937 sample_rng(42);
    vec<size_t> query_indices(n);
    std::iota(query_indices.begin(), query_indices.end(), 0);
    std::shuffle(query_indices.begin(), query_indices.end(), sample_rng);
    query_indices.resize(n_queries);

    size_t total_hits = 0;
    for (size_t qi = 0; qi < n_queries; ++qi) {
      const size_t qid = query_indices[qi];
      const float* qvec = dataset.vector(qid);

      // brute-force top-k
      vec<std::pair<float, u32>> all_dists;
      all_dists.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        if (i == qid) continue;
        all_dists.push_back({dist_fn(qvec, dataset.vector(i), dim), static_cast<u32>(i)});
      }
      std::partial_sort(all_dists.begin(),
                        all_dists.begin() + std::min<size_t>(topk, all_dists.size()),
                        all_dists.end());
      std::unordered_set<u32> gt;
      for (size_t i = 0; i < topk && i < all_dists.size(); ++i) gt.insert(all_dists[i].second);

      // beam_search on built graph
      auto results = beam_search(graph, dataset, qvec, beam_width, dist_fn);
      size_t hits = 0;
      for (size_t i = 0; i < topk && i < results.size(); ++i) {
        if (gt.count(results[i].second)) ++hits;
      }
      total_hits += hits;
    }
    double recall = static_cast<double>(total_hits) / static_cast<double>(n_queries * topk);
    std::cerr << "in-memory recall@" << topk << " (sample " << n_queries << "): " << recall << "\n";
  }
}

// ============================================================================
// RaBitQ quantization (CPU)
// ============================================================================

struct RaBitQState {
  Eigen::MatrixXf rotation_matrix;  // dim x dim, column-major
  vec<float> rotated_centroid;       // dim
  double t_const{0.0};              // scaling constant
  u32 dim{0};
  u32 bits_per_dim{0};
  u32 packed_bytes{0};              // (bits_per_dim * dim + 7) / 8
  u32 total_rabitq_bytes{0};        // packed_bytes + 8 (add + rescale)
};

constexpr std::array<float, 9> kTightStart = {
  0.0f,
  0.15f,
  0.20f,
  0.52f,
  0.59f,
  0.71f,
  0.75f,
  0.77f,
  0.81f
};

double best_rescale_factor(const float* abs_unit_residual, size_t dim, size_t bits_per_dim) {
  constexpr double kEps = 1e-5;
  constexpr int kNEnum = 10;

  const double max_o = *std::max_element(abs_unit_residual, abs_unit_residual + dim);
  if (max_o <= 0.0) {
    return 0.0;
  }

  const double t_end = static_cast<double>(((1u << bits_per_dim) - 1u) + kNEnum) / max_o;
  const double t_start = t_end * kTightStart.at(bits_per_dim);

  vec<int> cur_o_bar(dim);
  double sqr_denominator = static_cast<double>(dim) * 0.25;
  double numerator = 0.0;

  for (size_t i = 0; i < dim; ++i) {
    const int cur = static_cast<int>((t_start * abs_unit_residual[i]) + kEps);
    cur_o_bar[i] = cur;
    sqr_denominator += static_cast<double>(cur) * cur + cur;
    numerator += (static_cast<double>(cur) + 0.5) * abs_unit_residual[i];
  }

  std::priority_queue<std::pair<double, size_t>,
                      vec<std::pair<double, size_t>>,
                      std::greater<>> next_t;
  for (size_t i = 0; i < dim; ++i) {
    if (abs_unit_residual[i] > 0.0f) {
      next_t.emplace(static_cast<double>(cur_o_bar[i] + 1) / abs_unit_residual[i], i);
    }
  }

  double max_ip = 0.0;
  double best_t = 0.0;
  while (!next_t.empty()) {
    const auto [cur_t, update_id] = next_t.top();
    next_t.pop();

    cur_o_bar[update_id]++;
    const int update_o_bar = cur_o_bar[update_id];
    sqr_denominator += 2.0 * update_o_bar;
    numerator += abs_unit_residual[update_id];

    const double cur_ip = numerator / std::sqrt(sqr_denominator);
    if (cur_ip > max_ip) {
      max_ip = cur_ip;
      best_t = cur_t;
    }

    if (update_o_bar < static_cast<int>((1u << bits_per_dim) - 1u)) {
      const double t_next = static_cast<double>(update_o_bar + 1) / abs_unit_residual[update_id];
      if (t_next < t_end) {
        next_t.emplace(t_next, update_id);
      }
    }
  }

  return best_t;
}

double get_const_scaling_factors(size_t dim, size_t bits_per_dim, uint64_t seed) {
  constexpr size_t n_samples = 1000;
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  vec<float> sample(dim);
  vec<float> abs_unit(dim);
  double total = 0.0;

  for (size_t sample_id = 0; sample_id < n_samples; ++sample_id) {
    double norm_sqr = 0.0;
    for (size_t d = 0; d < dim; ++d) {
      sample[d] = normal(rng);
      norm_sqr += static_cast<double>(sample[d]) * sample[d];
    }
    const double norm = std::sqrt(norm_sqr);
    const double inv_norm = norm > 0.0 ? (1.0 / norm) : 0.0;
    for (size_t d = 0; d < dim; ++d) {
      abs_unit[d] = std::fabs(static_cast<float>(sample[d] * inv_norm));
    }
    total += best_rescale_factor(abs_unit.data(), dim, bits_per_dim);
  }

  return total / static_cast<double>(n_samples);
}

/**
 * Initialize RaBitQ: generate random orthogonal matrix via QR decomposition.
 */
RaBitQState init_rabitq(const Dataset& dataset, u32 bits_per_dim, int seed) {
  const u32 dim = dataset.dim;
  const size_t n = dataset.ids.size();

  RaBitQState state;
  state.dim = dim;
  state.bits_per_dim = bits_per_dim;
  state.packed_bytes = (bits_per_dim * dim + 7) / 8;
  state.total_rabitq_bytes = state.packed_bytes + 2 * sizeof(float);

  // Generate random matrix and compute QR decomposition for orthogonal P
  std::cerr << "generating rotation matrix (dim=" << dim << ")...\n";
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  Eigen::MatrixXf random_mat(dim, dim);
  for (u32 i = 0; i < dim; ++i)
    for (u32 j = 0; j < dim; ++j)
      random_mat(i, j) = normal(rng);

  Eigen::HouseholderQR<Eigen::MatrixXf> qr(random_mat);
  state.rotation_matrix = qr.householderQ() * Eigen::MatrixXf::Identity(dim, dim);
  Eigen::MatrixXf r = state.rotation_matrix.transpose() * random_mat;
  for (u32 i = 0; i < dim; ++i) {
    if (r(i, i) < 0.0f) {
      state.rotation_matrix.col(i) *= -1.0f;
    }
  }

  // Compute centroid
  std::cerr << "computing centroid...\n";
  Eigen::VectorXf centroid = Eigen::VectorXf::Zero(dim);
  for (size_t i = 0; i < n; ++i) {
    Eigen::Map<const Eigen::VectorXf> v(dataset.vector(i), dim);
    centroid += v;
  }
  centroid /= static_cast<float>(n);

  // Rotate centroid with the same transform later used by quantization.
  Eigen::VectorXf rot_centroid = state.rotation_matrix.transpose() * centroid;
  state.rotated_centroid.assign(rot_centroid.data(), rot_centroid.data() + dim);

  // Match Jasper's sampled constant scaling factor instead of the earlier simplified approximation.
  state.t_const = get_const_scaling_factors(dim, bits_per_dim, static_cast<uint64_t>(seed));

  return state;
}

/**
 * Quantize a single vector using RaBitQ.
 * Output: [packed_bits(packed_bytes) | add(4B) | rescale(4B)]
 */
void rabitq_quantize_vector(const float* vector,
                            const RaBitQState& state,
                            byte_t* output) {
  const u32 dim = state.dim;
  const u32 bits = state.bits_per_dim;
  constexpr double kEps = 1e-5;

  // Rotate vector: x' = P^T * x
  Eigen::Map<const Eigen::VectorXf> v(vector, dim);
  Eigen::VectorXf rotated = state.rotation_matrix.transpose() * v;

  // Subtract rotated centroid: delta = x' - c'
  vec<float> delta(dim);
  for (u32 i = 0; i < dim; ++i)
    delta[i] = rotated(i) - state.rotated_centroid[i];

  float l2_sqr = 0.0f;
  for (u32 i = 0; i < dim; ++i) l2_sqr += delta[i] * delta[i];
  const float l2_norm = std::sqrt(l2_sqr);

  vec<u8> quantized_vals(dim, 0);
  float ip_norm = 0.0f;
  const u32 magnitude_cap = (1u << (bits - 1)) - 1u;
  for (u32 i = 0; i < dim; ++i) {
    const float abs_o = l2_norm > 0.0f ? std::fabs(delta[i] / l2_norm) : 0.0f;
    int val = static_cast<int>((state.t_const * abs_o) + kEps);
    if (val >= static_cast<int>(1u << (bits - 1))) {
      val = static_cast<int>((1u << (bits - 1)) - 1u);
    }
    quantized_vals[i] = static_cast<u8>(val);
    ip_norm += (static_cast<float>(val) + 0.5f) * abs_o;
  }
  const float ip_norm_inv = ip_norm == 0.0f ? 1.0f : (1.0f / ip_norm);

  for (u32 i = 0; i < dim; ++i) {
    if (delta[i] >= 0.0f) {
      quantized_vals[i] = static_cast<u8>(quantized_vals[i] + (1u << (bits - 1)));
    } else {
      quantized_vals[i] = static_cast<u8>((~quantized_vals[i]) & magnitude_cap);
    }
  }

  // Pack bits into bytes
  std::memset(output, 0, state.packed_bytes);
  for (u32 i = 0; i < dim; ++i) {
    const u32 bit_idx = i * bits;
    const u32 byte_idx = bit_idx / 8;
    const u32 bit_off = bit_idx % 8;
    output[byte_idx] |= static_cast<byte_t>(quantized_vals[i] << bit_off);
    if (bit_off + bits > 8 && byte_idx + 1 < state.packed_bytes) {
      output[byte_idx + 1] |= static_cast<byte_t>(quantized_vals[i] >> (8 - bit_off));
    }
  }

  const float cb = -(static_cast<float>(1u << (bits - 1)) - 0.5f);
  float ip_resi_xucb = 0.0f;
  float ip_cent_xucb = 0.0f;
  for (u32 i = 0; i < dim; ++i) {
    const float xu_cb = static_cast<float>(quantized_vals[i]) + cb;
    ip_resi_xucb += delta[i] * xu_cb;
    ip_cent_xucb += state.rotated_centroid[i] * xu_cb;
  }
  if (ip_resi_xucb == 0.0f) {
    ip_resi_xucb = std::numeric_limits<float>::infinity();
  }

  const float add_factor = l2_sqr + 2.0f * l2_sqr * ip_cent_xucb / ip_resi_xucb;
  const float rescale_factor = ip_norm_inv * -2.0f * l2_norm;

  byte_t* trailer = output + state.packed_bytes;
  std::memcpy(trailer, &add_factor, sizeof(float));
  std::memcpy(trailer + sizeof(float), &rescale_factor, sizeof(float));
}

// ============================================================================
// Shard serialization
// ============================================================================

struct NodePlacement {
  u32 memory_node{0};
  u64 offset{0};
};

/**
 * Assign nodes to memory node shards using round-robin offset balancing.
 */
vec<NodePlacement> assign_nodes_to_shards(size_t num_vectors, u32 num_memory_nodes) {
  const size_t node_size = VamanaNode::total_size();
  // Align to 8 bytes
  const size_t aligned_size = (node_size + 7) & ~7ULL;

  vec<u64> shard_offsets(num_memory_nodes, 16);  // reserve 16B header: [free_ptr | medoid_ptr]
  vec<NodePlacement> placements(num_vectors);

  for (size_t i = 0; i < num_vectors; ++i) {
    // Pick shard with least data
    const auto min_it = std::min_element(shard_offsets.begin(), shard_offsets.end());
    const u32 shard = static_cast<u32>(std::distance(shard_offsets.begin(), min_it));

    placements[i] = {shard, *min_it};
    *min_it += aligned_size;
  }

  return placements;
}

void write_vamana_shards(const VamanaGraph& graph,
                         const Dataset& dataset,
                         const VamanaBuildConfig& config,
                         const RaBitQState& rabitq_state,
                         const vec<vec<byte_t>>& rabitq_data,
                         const filepath_t& output_prefix) {
  const size_t n = dataset.ids.size();
  const u32 dim = dataset.dim;
  const size_t node_size = VamanaNode::total_size();
  const size_t aligned_size = (node_size + 7) & ~7ULL;

  ProgressReporter progress{"Exporting Vamana shards", n + config.num_memory_nodes};

  const auto placements = assign_nodes_to_shards(n, config.num_memory_nodes);

  // Compute shard sizes
  vec<u64> shard_sizes(config.num_memory_nodes, 16);
  for (const auto& p : placements) {
    shard_sizes[p.memory_node] = std::max<u64>(shard_sizes[p.memory_node], p.offset + aligned_size);
  }

  // Allocate shard buffers
  vec<vec<byte_t>> shard_buffers(config.num_memory_nodes);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    shard_buffers[shard].assign(shard_sizes[shard], 0);
    // Write free_ptr at offset 0 (points past last node)
    *reinterpret_cast<u64*>(shard_buffers[shard].data()) = shard_sizes[shard];
  }

  // Write medoid_ptr at offset 8 on shard 0
  const RemotePtr medoid_ptr{placements[graph.medoid].memory_node, placements[graph.medoid].offset};
  *reinterpret_cast<u64*>(shard_buffers[0].data() + 8) = medoid_ptr.raw_address;

  // Serialize each node
  for (size_t i = 0; i < n; ++i) {
    const auto& placement = placements[i];
    byte_t* buf = shard_buffers[placement.memory_node].data() + placement.offset;

    // Header (8B)
    u64 header = 0;
    if (i == graph.medoid) header |= VamanaNode::HEADER_IS_MEDOID;
    *reinterpret_cast<u64*>(buf) = header;

    // ID (4B)
    *reinterpret_cast<u32*>(buf + VamanaNode::HEADER_SIZE) = dataset.ids[i];

    // Edge count (1B)
    const u8 edge_count = static_cast<u8>(std::min<size_t>(graph.neighbors[i].size(), config.R));
    *reinterpret_cast<u8*>(buf + VamanaNode::offset_edge_count()) = edge_count;

    // Padding (3B) - already zeroed

    // Vector (dim * 4B)
    std::memcpy(buf + VamanaNode::offset_vector(),
                dataset.vector(i),
                dim * sizeof(float));

    // RaBitQ data
    std::memcpy(buf + VamanaNode::offset_rabitq(),
                rabitq_data[i].data(),
                rabitq_state.total_rabitq_bytes);

    // Neighbors (R * 8B) — write active + zero rest
    auto* neighbor_buf = reinterpret_cast<u64*>(buf + VamanaNode::offset_neighbors());
    for (u8 j = 0; j < edge_count; ++j) {
      const u32 nbr = graph.neighbors[i][j];
      RemotePtr nbr_ptr{placements[nbr].memory_node, placements[nbr].offset};
      neighbor_buf[j] = nbr_ptr.raw_address;
    }
    // Remaining slots already zeroed

    progress.increment();
  }

  // Write shard files
  const filepath_t output_dir = output_prefix.parent_path();
  if (!output_dir.empty()) {
    std::filesystem::create_directories(output_dir);
  }

  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    const filepath_t shard_file = index_path::shard_file(output_prefix, shard + 1, config.num_memory_nodes);
    std::ofstream output(shard_file, std::ios::binary | std::ios::out);
    lib_assert(output.good(), "failed to open output shard file: " + shard_file.string());
    output.write(reinterpret_cast<const char*>(shard_buffers[shard].data()),
                 static_cast<std::streamsize>(shard_buffers[shard].size()));
    lib_assert(output.good(), "failed to write output shard file: " + shard_file.string());
    progress.increment();
  }

  // Write rotation matrix to a separate file
  {
    const filepath_t rot_file = filepath_t(output_prefix.string() + ".rotation.bin");
    std::ofstream out(rot_file, std::ios::binary);
    lib_assert(out.good(), "failed to open rotation matrix file: " + rot_file.string());
    // Write dim, then column-major matrix data
    u32 d = dim;
    out.write(reinterpret_cast<const char*>(&d), sizeof(u32));
    out.write(reinterpret_cast<const char*>(rabitq_state.rotation_matrix.data()),
              static_cast<std::streamsize>(dim * dim * sizeof(float)));
    // Write rotated centroid
    out.write(reinterpret_cast<const char*>(rabitq_state.rotated_centroid.data()),
              static_cast<std::streamsize>(dim * sizeof(float)));
    // Write t_const
    double tc = rabitq_state.t_const;
    out.write(reinterpret_cast<const char*>(&tc), sizeof(double));
    lib_assert(out.good(), "failed to write rotation matrix file");
  }

  // Write metadata
  nlohmann::json metadata{
    {"data_file", dataset.source_file.string()},
    {"output_prefix", output_prefix.string()},
    {"distance", config.ip_distance ? "ip" : "l2"},
    {"num_vectors", n},
    {"dim", dim},
    {"R", config.R},
    {"beam_width", config.beam_width},
    {"beam_width_construction", config.beam_width},
    {"alpha", config.alpha},
    {"rabitq_bits", config.rabitq_bits},
    {"num_memory_nodes", config.num_memory_nodes},
    {"medoid", {{"memory_node", medoid_ptr.memory_node()}, {"offset", medoid_ptr.byte_offset()}}},
    {"node_size", node_size},
    {"rabitq_size", rabitq_state.total_rabitq_bytes},
  };

  const filepath_t metadata_file = filepath_t(output_prefix.string() + ".meta.json");
  std::ofstream metadata_output(metadata_file);
  metadata_output << std::setw(2) << metadata << std::endl;
  progress.finish();
}

// ============================================================================
// CLI parsing
// ============================================================================

filepath_t default_vamana_prefix(const filepath_t& data_path, u32 R, u32 beam_width) {
  const filepath_t base = std::filesystem::is_regular_file(data_path) ? data_path.parent_path() : data_path;
  return base / "dump" / ("vamana_R" + std::to_string(R) + "_bw" + std::to_string(beam_width));
}

VamanaBuildConfig parse_configuration(int argc, char** argv) {
  VamanaBuildConfig config;

  po::options_description desc{"Vamana offline builder options"};
  desc.add_options()
    ("help,h", "Show help message")
    ("data-path,d", po::value<filepath_t>(&config.data_path), "Path to a dataset file or directory.")
    ("output-prefix,o", po::value<filepath_t>(&config.output_prefix),
     "Output prefix without _nodeX_ofN.dat suffix.")
    ("memory-nodes,n", po::value<u32>(&config.num_memory_nodes)->default_value(config.num_memory_nodes),
     "Number of output shards / memory nodes.")
    ("threads,t", po::value<u32>(&config.threads)->default_value(config.threads),
     "Number of threads. 0 = hardware concurrency.")
    ("R", po::value<u32>(&config.R)->default_value(config.R), "Maximum out-degree.")
    ("beam-width", po::value<u32>(&config.beam_width)->default_value(config.beam_width),
     "Beam width for beam search during offline construction.")
    ("beam-width-construction", po::value<u32>(&config.beam_width),
     "Alias for --beam-width. Offline builder only has a construction beam width.")
    ("ef-construction", po::value<u32>(&config.beam_width),
     "Alias for --beam-width in the offline builder.")
    ("alpha", po::value<f64>(&config.alpha)->default_value(config.alpha), "RobustPrune alpha parameter.")
    ("rabitq-bits", po::value<u32>(&config.rabitq_bits)->default_value(config.rabitq_bits),
     "Bits per dimension for RaBitQ (1, 2, 4, or 8).")
    ("seed", po::value<i32>(&config.seed)->default_value(config.seed), "PRNG seed.")
    ("max-vectors", po::value<size_t>(&config.max_vectors)->default_value(config.max_vectors),
     "Maximum number of vectors to read.")
    ("ip-dist", po::bool_switch(&config.ip_distance), "Use inner-product distance instead of L2.")
    ("no-gpu", po::bool_switch(&config.no_gpu), "Disable GPU acceleration.")
    ("gpu-device", po::value<i32>(&config.gpu_device)->default_value(config.gpu_device),
     "CUDA device ID (default 0).")
    ("query-path", po::value<filepath_t>(&config.query_path),
     "Path to query file (.fbin) for post-build recall test.")
    ("groundtruth-path", po::value<filepath_t>(&config.groundtruth_path),
     "Path to ground truth file (.bin) for post-build recall test.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  po::notify(vm);

  if (config.data_path.empty()) lib_failure("--data-path is required");
  if (config.num_memory_nodes == 0) lib_failure("--memory-nodes must be > 0");
  if (config.R == 0) lib_failure("--R must be > 0");
  if (config.rabitq_bits != 1 && config.rabitq_bits != 2 &&
      config.rabitq_bits != 4 && config.rabitq_bits != 8)
    lib_failure("--rabitq-bits must be 1, 2, 4, or 8");

  return config;
}

}  // namespace

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  const VamanaBuildConfig config = parse_configuration(argc, argv);
  const Dataset dataset = read_dataset(config);
  const filepath_t output_prefix =
      config.output_prefix.empty()
          ? default_vamana_prefix(dataset.source_file, config.R, config.beam_width)
          : config.output_prefix;

  std::cerr << "output prefix: " << output_prefix << "\n";
  std::cerr << "memory nodes: " << config.num_memory_nodes << "\n";
  std::cerr << "threads: " << effective_thread_count(config.threads) << "\n";
  std::cerr << "R=" << config.R << " construction_beam_width=" << config.beam_width
            << " alpha=" << config.alpha << " rabitq_bits=" << config.rabitq_bits << "\n";

  const auto build_start = std::chrono::steady_clock::now();

  // Initialize VamanaNode static storage
  VamanaNode::init_static_storage(dataset.dim, config.R, config.rabitq_bits);

  // Select distance function
  DistFn dist_fn = config.ip_distance ? ip_distance : l2_squared;

  // GPU initialization
  const size_t num_threads = effective_thread_count(config.threads);
  std::unique_ptr<BuilderGpuContext[]> gpu_contexts;
  size_t num_gpu_contexts = 0;

  if (!config.no_gpu && !config.ip_distance) {
    gpu::gpu_init(config.gpu_device);
    num_gpu_contexts = num_threads;
    gpu_contexts = std::make_unique<BuilderGpuContext[]>(num_gpu_contexts);
    for (size_t i = 0; i < num_gpu_contexts; ++i) {
      gpu_contexts[i].init(dataset.dim, config.beam_width, config.R);
    }
    std::cerr << "GPU: device " << config.gpu_device
              << " (" << num_gpu_contexts << " streams)\n";
  } else {
    std::cerr << "GPU: disabled"
              << (config.ip_distance ? " (IP distance not supported on GPU)" : "")
              << "\n";
  }

  // Step 1: Build Vamana graph
  VamanaGraph graph;
  build_vamana_graph(graph, dataset, config, dist_fn,
                     gpu_contexts.get(), num_gpu_contexts);

  // Optional: recall test with external queries and ground truth
  if (!config.query_path.empty() && !config.groundtruth_path.empty()) {
    std::cerr << "\n=== Recall Test ===\n";

    // Read queries (.fbin format: u32 num_queries, u32 dim, then float32 data)
    std::ifstream qfile(config.query_path, std::ios::binary);
    if (!qfile) lib_failure("cannot open query file: " + config.query_path.string());
    u32 n_queries, q_dim;
    qfile.read(reinterpret_cast<char*>(&n_queries), 4);
    qfile.read(reinterpret_cast<char*>(&q_dim), 4);
    if (q_dim != dataset.dim) lib_failure("query dim mismatch");
    vec<float> query_vecs(static_cast<size_t>(n_queries) * q_dim);
    qfile.read(reinterpret_cast<char*>(query_vecs.data()), query_vecs.size() * sizeof(float));
    qfile.close();
    std::cerr << "queries: " << n_queries << " x " << q_dim << "\n";

    // Read ground truth (.bin format: u32 n_queries, u32 topk, then u32 IDs)
    std::ifstream gtfile(config.groundtruth_path, std::ios::binary);
    if (!gtfile) lib_failure("cannot open groundtruth file: " + config.groundtruth_path.string());
    u32 gt_n, gt_k;
    gtfile.read(reinterpret_cast<char*>(&gt_n), 4);
    gtfile.read(reinterpret_cast<char*>(&gt_k), 4);
    if (gt_n != n_queries) lib_failure("groundtruth count mismatch");
    vec<u32> gt_ids(static_cast<size_t>(gt_n) * gt_k);
    gtfile.read(reinterpret_cast<char*>(gt_ids.data()), gt_ids.size() * sizeof(u32));
    gtfile.close();
    std::cerr << "ground truth: " << gt_n << " queries x top-" << gt_k << "\n";

    // Run beam_search for each query and compute recall
    const u32 search_beam = config.beam_width;
    for (u32 eval_k : {1u, 5u, 10u}) {
      if (eval_k > gt_k) continue;

      size_t total_hits = 0;
      for (u32 qi = 0; qi < n_queries; ++qi) {
        const float* qvec = query_vecs.data() + static_cast<size_t>(qi) * q_dim;
        auto results = beam_search(graph, dataset, qvec, search_beam, dist_fn);

        // Ground truth for this query
        const u32* gt_row = gt_ids.data() + static_cast<size_t>(qi) * gt_k;
        std::unordered_set<u32> gt_set(gt_row, gt_row + eval_k);

        size_t hits = 0;
        for (size_t i = 0; i < eval_k && i < results.size(); ++i) {
          if (gt_set.count(results[i].second)) ++hits;
        }
        total_hits += hits;
      }

      double recall = static_cast<double>(total_hits) / (static_cast<double>(n_queries) * eval_k);
      std::cerr << "recall@" << eval_k << " = " << std::fixed << std::setprecision(4) << recall
                << " (" << total_hits << "/" << (n_queries * eval_k) << ")\n";
    }
    std::cerr << "=== End Recall Test ===\n\n";
  }

  // Step 2: Initialize RaBitQ and quantize all vectors
  RaBitQState rabitq_state = init_rabitq(dataset, config.rabitq_bits, config.seed);

  vec<vec<byte_t>> rabitq_data(dataset.ids.size());
  {
    ProgressReporter progress{"Quantizing vectors (RaBitQ)", dataset.ids.size()};
    parallel_for(0, dataset.ids.size(), config.threads,
                 [&](size_t i, size_t) {
                   rabitq_data[i].resize(rabitq_state.total_rabitq_bytes);
                   rabitq_quantize_vector(dataset.vector(i), rabitq_state, rabitq_data[i].data());
                   progress.increment();
                 });
    progress.finish();
  }

  // Step 3: Serialize to shard files
  write_vamana_shards(graph, dataset, config, rabitq_state, rabitq_data, output_prefix);

  // GPU cleanup
  for (size_t i = 0; i < num_gpu_contexts; ++i) gpu_contexts[i].destroy();
  gpu_contexts.reset();
  if (num_gpu_contexts > 0) gpu::gpu_shutdown();

  const auto build_end = std::chrono::steady_clock::now();
  const auto seconds = std::chrono::duration_cast<std::chrono::duration<double>>(build_end - build_start).count();
  std::cerr << "offline build finished in " << seconds << " seconds\n";

  return EXIT_SUCCESS;
}
