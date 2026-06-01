#include "tools/vamana_offline/graph.hh"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_set>

#include <library/utils.hh>

#include "tools/vamana_offline/progress.hh"

#ifdef __AVX__
#include <x86intrin.h>
#endif

namespace tools::vamana_offline {

namespace {

class LocalIdSet {
public:
  explicit LocalIdSet(size_t expected_items) {
    size_t capacity = 1;
    while (capacity < expected_items * 2) capacity <<= 1;
    table_.assign(capacity, kEmpty);
    mask_ = capacity - 1;
  }

  bool contains(u32 value) const {
    size_t pos = hash(value) & mask_;
    for (;;) {
      const u32 current = table_[pos];
      if (current == kEmpty) return false;
      if (current == value) return true;
      pos = (pos + 1) & mask_;
    }
  }

  bool insert(u32 value) {
    size_t pos = hash(value) & mask_;
    for (;;) {
      const u32 current = table_[pos];
      if (current == value) return false;
      if (current == kEmpty) {
        table_[pos] = value;
        return true;
      }
      pos = (pos + 1) & mask_;
    }
  }

private:
  static constexpr u32 kEmpty = std::numeric_limits<u32>::max();

  static size_t hash(u32 value) {
    uint64_t x = value;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return static_cast<size_t>(x);
  }

  vec<u32> table_;
  size_t mask_{0};
};

}  // namespace

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
                                       BuilderGpuContext* gpu_ctx) {
  const u32 dim = dataset.dim;

  // all_visited: every node we computed distance for (returned to caller)
  vec<std::pair<float, u32>> all_visited;
  // beam: top beam_width candidates used for navigation
  vec<std::pair<float, u32>> beam;
  const size_t expected_seen = std::max<size_t>(1024, static_cast<size_t>(beam_width) * graph.R + graph.R + 1);
  LocalIdSet visited(expected_seen);
  LocalIdSet expanded(std::max<size_t>(1024, beam_width * 2));

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
      if (!expanded.contains(beam[i].second)) {
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
      if (!visited.insert(nbr)) continue;
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
  const bool deferred_reverse = config.offline_reverse_mode == "deferred";

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

  std::unique_ptr<u32[]> deferred_reverse_edges;
  std::unique_ptr<std::atomic<u32>[]> deferred_reverse_counts;
  if (deferred_reverse) {
    std::cerr << "offline reverse updates: deferred flat final merge/prune"
              << " (capacity=N*R edges, extra memory ~= N*R*4 + N*4 bytes)\n";
    deferred_reverse_edges.reset(new u32[n * static_cast<size_t>(R)]);
    deferred_reverse_counts.reset(new std::atomic<u32>[n]);
    parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t node, size_t) {
      deferred_reverse_counts[node].store(0, std::memory_order_relaxed);
    });
  } else {
    std::cerr << "offline reverse updates: immediate" << "\n";
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

    // Add reverse edges. In deferred mode, avoid pruning the same target node
    // many times; a final per-node merge/prune pass applies all reverse edges.
    for (u32 nbr : new_neighbors) {
      if (deferred_reverse) {
        const u32 slot = deferred_reverse_counts[nbr].fetch_add(1, std::memory_order_relaxed);
        if (slot < R) {
          deferred_reverse_edges[static_cast<size_t>(nbr) * R + slot] = static_cast<u32>(node_idx);
        }
        continue;
      }

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

  if (deferred_reverse) {
    ProgressReporter reverse_progress{"Applying deferred reverse edges", n};
    parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t node, size_t) {
      const u32 addition_count = std::min<u32>(deferred_reverse_counts[node].load(std::memory_order_relaxed), R);
      if (addition_count > 0) {
        std::lock_guard<std::mutex> lock(graph.node_locks[node]);
        auto& nbr_list = graph.neighbors[node];

        vec<u32> candidate_ids;
        candidate_ids.reserve(nbr_list.size() + addition_count);
        for (u32 existing : nbr_list) {
          if (existing != node) candidate_ids.push_back(existing);
        }
        const size_t base = node * static_cast<size_t>(R);
        for (u32 i = 0; i < addition_count; ++i) {
          const u32 incoming = deferred_reverse_edges[base + i];
          if (incoming != node) candidate_ids.push_back(incoming);
        }

        std::sort(candidate_ids.begin(), candidate_ids.end());
        candidate_ids.erase(std::unique(candidate_ids.begin(), candidate_ids.end()), candidate_ids.end());

        if (candidate_ids.size() <= R) {
          nbr_list = std::move(candidate_ids);
        } else {
          vec<std::pair<float, u32>> prune_candidates;
          prune_candidates.reserve(candidate_ids.size());
          const float* node_vec = dataset.vector(node);
          for (u32 candidate : candidate_ids) {
            const float d = dist_fn(node_vec, dataset.vector(candidate), dim);
            prune_candidates.push_back({d, candidate});
          }
          std::sort(prune_candidates.begin(), prune_candidates.end());
          nbr_list = robust_prune(dataset, static_cast<u32>(node), prune_candidates, build_alpha, R, dist_fn);
        }
      }
      reverse_progress.increment();
    });
    reverse_progress.finish();
    deferred_reverse_counts.reset();
    deferred_reverse_edges.reset();
  }

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
  if (!config.skip_sanity_check) {
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


}  // namespace tools::vamana_offline
