#include "gpu_search/persistent_engine.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "common/index_path.hh"
#include "gpu_search/navigation_bootstrapper.hh"
#ifdef DVSTOR_HAVE_GPUNETIO
#include "gpu/gpunetio_transport.hh"
#endif
#include "gpu_search/index_format.hh"
#include "gpu_search/memory_budget.hh"
#include "gpu_search/pq_index.hh"
#include "gpu_search/persistent_kernel.hh"
#include "vamana/anchor_index.hh"
#include "vamana/vamana_node.hh"

namespace gpu_search {
namespace {

static_assert(sizeof(DeviceShardRegion) == sizeof(format::ShardRegion));

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

u32 next_power_of_two(u32 value) {
  if (value >= (1u << 31)) return 1u << 31;
  return std::max<u32>(2, std::bit_ceil(value));
}

u64 align_up(u64 value, u64 alignment) {
  return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

template <class T>
void device_allocate(T*& pointer, size_t count, const char* operation) {
  if (count == 0) {
    pointer = nullptr;
    return;
  }
  if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(operation) + ": allocation size overflow");
  }
  const size_t bytes = count * sizeof(T);
  const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&pointer), bytes);
  if (status != cudaSuccess) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    (void)cudaMemGetInfo(&free_bytes, &total_bytes);
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status) +
      " requested=" + std::to_string(bytes) +
      " free=" + std::to_string(free_bytes) +
      " total=" + std::to_string(total_bytes));
  }
}

template <class T>
void device_free(T*& pointer) {
  if (pointer != nullptr) cudaFree(pointer);
  pointer = nullptr;
}

template <class T>
void mapped_host_allocate(T*& host_pointer, T*& device_pointer,
                          size_t count, const char* operation) {
  host_pointer = nullptr;
  device_pointer = nullptr;
  if (count == 0) return;
  if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(operation) + ": allocation size overflow");
  }
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&host_pointer),
                           count * sizeof(T),
                           cudaHostAllocMapped | cudaHostAllocPortable),
             operation);
  check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_pointer),
                                      host_pointer, 0),
             "cudaHostGetDevicePointer(delta staging)");
}

template <class T>
class MappedRing {
public:
  enum class Direction {
    host_to_device,
    device_to_host,
  };

  MappedRing(u32 capacity, Direction direction)
      : capacity_(next_power_of_two(capacity)), direction_(direction) {
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&enqueue_host_), sizeof(u64),
                             cudaHostAllocMapped), "cudaHostAlloc(ring enqueue)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&dequeue_host_), sizeof(u64),
                             cudaHostAllocMapped), "cudaHostAlloc(ring dequeue)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&sequences_host_),
                             static_cast<size_t>(capacity_) * sizeof(u64),
                             cudaHostAllocMapped), "cudaHostAlloc(ring sequences)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&entries_host_),
                             static_cast<size_t>(capacity_) * sizeof(T),
                             cudaHostAllocMapped), "cudaHostAlloc(ring entries)");
    *enqueue_host_ = 0;
    *dequeue_host_ = 0;
    for (u32 index = 0; index < capacity_; ++index) sequences_host_[index] = index;
    u64* enqueue_device = nullptr;
    u64* dequeue_device = nullptr;
    u64* sequences_device = nullptr;
    T* entries_device = nullptr;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&enqueue_device), enqueue_host_, 0),
               "cudaHostGetDevicePointer(ring enqueue)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dequeue_device), dequeue_host_, 0),
               "cudaHostGetDevicePointer(ring dequeue)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&sequences_device), sequences_host_, 0),
               "cudaHostGetDevicePointer(ring sequences)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&entries_device), entries_host_, 0),
               "cudaHostGetDevicePointer(ring entries)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_owned_position_), sizeof(u64)),
               "cudaMalloc(ring device position)");
    check_cuda(cudaMemset(device_owned_position_, 0, sizeof(u64)),
               "cudaMemset(ring device position)");
    if (direction_ == Direction::host_to_device) {
      dequeue_device = device_owned_position_;
    } else {
      enqueue_device = device_owned_position_;
    }
    device_view_ = {
      .enqueue_position = reinterpret_cast<unsigned long long*>(enqueue_device),
      .dequeue_position = reinterpret_cast<unsigned long long*>(dequeue_device),
      .sequences = reinterpret_cast<unsigned long long*>(sequences_device),
      .entries = entries_device,
      .capacity = capacity_,
      .mask = capacity_ - 1,
    };
  }

  ~MappedRing() {
    if (device_owned_position_ != nullptr) cudaFree(device_owned_position_);
    if (entries_host_ != nullptr) cudaFreeHost(entries_host_);
    if (sequences_host_ != nullptr) cudaFreeHost(sequences_host_);
    if (dequeue_host_ != nullptr) cudaFreeHost(dequeue_host_);
    if (enqueue_host_ != nullptr) cudaFreeHost(enqueue_host_);
  }

  bool try_push(const T& value) {
    std::atomic_ref<u64> enqueue(*enqueue_host_);
    const u64 position = enqueue.load(std::memory_order_relaxed);
    const u32 slot = static_cast<u32>(position) & (capacity_ - 1);
    std::atomic_ref<u64> sequence(sequences_host_[slot]);
    if (sequence.load(std::memory_order_acquire) != position) return false;
    entries_host_[slot] = value;
    sequence.store(position + 1, std::memory_order_release);
    enqueue.store(position + 1, std::memory_order_release);
    return true;
  }

  bool try_pop(T& value) {
    std::atomic_ref<u64> dequeue(*dequeue_host_);
    const u64 position = dequeue.load(std::memory_order_relaxed);
    const u32 slot = static_cast<u32>(position) & (capacity_ - 1);
    std::atomic_ref<u64> sequence(sequences_host_[slot]);
    if (sequence.load(std::memory_order_acquire) != position + 1) return false;
    value = entries_host_[slot];
    sequence.store(position + capacity_, std::memory_order_release);
    dequeue.store(position + 1, std::memory_order_release);
    return true;
  }

  DeviceRingView<T> device_view() const { return device_view_; }

private:
  u32 capacity_{};
  u64* enqueue_host_{};
  u64* dequeue_host_{};
  u64* sequences_host_{};
  T* entries_host_{};
  u64* device_owned_position_{};
  Direction direction_{};
  DeviceRingView<T> device_view_{};
};

struct AnchorTable {
  u32 dim{};
  std::vector<f32> vectors;
  std::vector<u32> handles;
  std::vector<u64> raw_pointers;
  std::vector<u32> shard_offsets;

  u32 count() const { return dim == 0 ? 0 : static_cast<u32>(vectors.size() / dim); }
};

AnchorTable load_anchor_table(const filepath_t& prefix, u32 expected_dim,
                              u32 expected_shards, const format::View& index_view) {
  AnchorTable result;
  const filepath_t path = index_path::anchor_file(prefix);
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    std::cerr << "[gpu-search] warning: no anchor sidecar; large deltas use a full scan\n";
    return result;
  }
  vamana::anchor::Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good() || header.magic != vamana::anchor::kMagic ||
      header.version != vamana::anchor::kVersion || header.dim != expected_dim ||
      header.shard_count != expected_shards || header.total_anchors > (1u << 24)) {
    throw std::runtime_error("invalid anchor sidecar for GPU delta buckets: " + path.string());
  }
  const VectorDType dtype = static_cast<VectorDType>(header.vector_dtype);
  if (vector_dtype_bytes(dtype, header.dim) != header.vector_bytes) {
    throw std::runtime_error("anchor sidecar vector layout mismatch");
  }
  result.dim = header.dim;
  result.vectors.reserve(static_cast<size_t>(header.total_anchors) * header.dim);
  result.shard_offsets.resize(header.shard_count + 1, 0);
  std::vector<byte_t> raw(header.vector_bytes);
  std::vector<f32> decoded(header.dim);
  for (u32 shard = 0; shard < header.shard_count; ++shard) {
    result.shard_offsets[shard] = result.count();
    vamana::anchor::ShardHeader shard_header;
    input.read(reinterpret_cast<char*>(&shard_header), sizeof(shard_header));
    if (!input.good() || shard_header.shard != shard ||
        shard_header.anchor_count > header.anchors_per_shard) {
      throw std::runtime_error("invalid anchor shard header");
    }
    input.seekg(static_cast<std::streamoff>(header.dim * sizeof(f32)), std::ios::cur);
    for (u32 index = 0; index < shard_header.anchor_count; ++index) {
      vamana::anchor::EntryHeader entry;
      input.read(reinterpret_cast<char*>(&entry), sizeof(entry));
      input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
      if (!input.good()) throw std::runtime_error("truncated anchor sidecar");
      u32 handle = UINT32_MAX;
      if (!format::remote_to_ordinal(index_view, RemotePtr{entry.rptr_raw}, handle)) {
        throw std::runtime_error("anchor sidecar contains a non-static GPU entry point");
      }
      decode_storage_vector_to_float(raw.data(), dtype, header.dim, decoded.data());
      result.vectors.insert(result.vectors.end(), decoded.begin(), decoded.end());
      result.handles.push_back(handle);
      result.raw_pointers.push_back(entry.rptr_raw);
    }
  }
  result.shard_offsets.back() = result.count();
  if (result.count() != header.total_anchors) {
    throw std::runtime_error("anchor sidecar count mismatch");
  }
  return result;
}

}  // namespace

struct PersistentSearchEngine::Impl {
  struct PendingQuery {
    u32 slot{};
    std::chrono::steady_clock::time_point submitted_at{};
    std::promise<service::QueryResult> promise;
  };

  struct PendingSubmission {
    QueryDescriptor descriptor{};
    std::chrono::steady_clock::time_point enqueued_at{};
  };

  std::string unhealthy_message() {
    std::lock_guard<std::mutex> lock(admission_mutex);
    return health_error.empty() ? "persistent GPU query engine is unhealthy" : health_error;
  }

  void reject_submission(const PendingSubmission& submission,
                         const std::string& message) {
    std::shared_ptr<PendingQuery> pending;
    {
      std::lock_guard<std::mutex> lock(pending_mutex);
      const auto iterator = pending_queries.find(submission.descriptor.request_id);
      if (iterator != pending_queries.end()) {
        pending = std::move(iterator->second);
        pending_queries.erase(iterator);
      }
    }
    if (!pending) return;
    pending->promise.set_exception(
      std::make_exception_ptr(std::runtime_error(message)));
    {
      std::lock_guard<std::mutex> lock(slot_mutex);
      free_slots.push_back(pending->slot);
    }
    slot_cv.notify_one();
    pending_count.fetch_sub(1, std::memory_order_release);
  }

  void mark_unhealthy(const std::string& message) {
    std::deque<PendingSubmission> rejected;
    {
      std::lock_guard<std::mutex> lock(admission_mutex);
      if (!healthy.load(std::memory_order_relaxed)) return;
      health_error = message;
      healthy.store(false, std::memory_order_release);
      rejected.swap(admission_queue);
    }
    admission_cv.notify_all();
    slot_cv.notify_all();
    for (const PendingSubmission& submission : rejected) {
      reject_submission(submission, message);
    }
    std::cerr << "[gpu-search] query engine entered fail-stop mode: "
              << message << '\n';
  }

  void reject_queued_submissions(const std::string& message) {
    std::deque<PendingSubmission> rejected;
    {
      std::lock_guard<std::mutex> lock(admission_mutex);
      rejected.swap(admission_queue);
    }
    for (const PendingSubmission& submission : rejected) {
      reject_submission(submission, message);
    }
  }

  void reject_all_pending(const std::string& message) {
    std::vector<std::shared_ptr<PendingQuery>> rejected;
    {
      std::lock_guard<std::mutex> lock(pending_mutex);
      rejected.reserve(pending_queries.size());
      for (auto& [request_id, pending] : pending_queries) {
        (void)request_id;
        rejected.push_back(std::move(pending));
      }
      pending_queries.clear();
    }
    if (rejected.empty()) return;
    {
      std::lock_guard<std::mutex> lock(slot_mutex);
      for (const auto& pending : rejected) free_slots.push_back(pending->slot);
    }
    for (const auto& pending : rejected) {
      try {
        pending->promise.set_exception(
          std::make_exception_ptr(std::runtime_error(message)));
      } catch (const std::future_error&) {
      }
    }
    pending_count.fetch_sub(rejected.size(), std::memory_order_release);
    slot_cv.notify_all();
  }

  void bind_cuda_device(const char* operation) const {
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice(GPU navigation)");
    if (current_device != static_cast<int>(config.gpu_device)) {
      check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)), operation);
    }
  }

  Impl(PersistentSearchEngine& owner,
       configuration::IndexConfiguration& config_in,
       Context& channel_context,
       ClientConnectionManager& connection_manager,
       const MemoryRegionTokens& remote_regions)
      : engine(owner), config(config_in),
        submissions(config.query_batch_max * 2,
                    MappedRing<QueryDescriptor>::Direction::host_to_device),
        completions(config.query_batch_max * 2,
                    MappedRing<CompletionDescriptor>::Direction::device_to_host),
        delta_submissions(8, MappedRing<DeltaPublishDescriptor>::Direction::host_to_device),
        delta_completions(8, MappedRing<DeltaPublishCompletion>::Direction::device_to_host) {
    bind_cuda_device("cudaSetDevice(GPU navigation construction)");
    if (config.gpu_traversal_beam_width > kPersistentMaxBeam ||
        config.gpu_final_rerank_width > kPersistentMaxExact ||
        config.R > kPersistentMaxGraphDegree) {
      throw std::invalid_argument("GPU navigation beam/exact/degree limit exceeded");
    }

    std::string load_error;
    bool used_anchor_entry_points = false;
    if (!format::synthesize_distributed_view(
          config.resolved_index_prefix(), index,
          format::SynthesisOptions{
            .entry_points = 0,
            .seed = static_cast<u64>(static_cast<u32>(config.seed)),
          },
          &used_anchor_entry_points, &load_error)) {
      throw std::runtime_error(load_error);
    }
    std::cerr << "[gpu-search] synthesized navigation manifest in memory from metadata"
              << (used_anchor_entry_points ? " and anchors\n" : "\n");
    if (!pq::read_model(index_path::navigation_model_file(
          config.resolved_index_prefix(), index.layout.pq_subquantizers),
          pq_model, &load_error)) {
      throw std::runtime_error(load_error);
    }
    if (index.layout.dim != config.dim || index.layout.graph_degree != config.R ||
        index.layout.num_shards != remote_regions.size() ||
        index.layout.num_shards > kPersistentMaxShards ||
        index.layout.pq_subquantizers != pq_model.subquantizers ||
        index.layout.pq_subquantizers > kPersistentMaxSubquantizers ||
        index.layout.pq_bits != pq_model.bits_per_code ||
        index.layout.code_bytes != pq_model.code_bytes() ||
        index.layout.model_checksum != pq_model.checksum() ||
        index.layout.graph_entry_bytes != VamanaNode::hot_graph_entry_size() ||
        index.layout.graph_shard_bits != VamanaNode::HOT_GRAPH_SHARD_BITS ||
        index.layout.vector_dtype != static_cast<u32>(config.resolved_vector_dtype()) ||
        index.entry_points.size() > kPersistentMaxEntryPoints) {
      throw std::runtime_error("GPU navigation manifest does not match runtime metadata");
    }
    const u64 max_merge_candidates =
      static_cast<u64>(config.gpu_traversal_beam_width) +
      static_cast<u64>(config.gpu_graph_prefetch_depth) * config.R;
    if (max_merge_candidates > kPersistentMaxMergeCandidates) {
      throw std::invalid_argument("GPU navigation prefetch/degree exceeds parallel top-k capacity");
    }

    anchor_table = load_anchor_table(config.resolved_index_prefix(), config.dim,
                                     index.layout.num_shards, index);
    for (u32 anchor = 0; anchor < anchor_table.raw_pointers.size(); ++anchor) {
      anchor_buckets_by_raw.emplace(anchor_table.raw_pointers[anchor], anchor);
    }
    entry_handles = index.entry_points;
    std::cerr << "[gpu-search] query routing="
              << (anchor_table.count() == 0 ? "static entry points" : "query-aware GPU anchors")
              << " anchors=" << anchor_table.count()
              << " seeds=" << config.gpu_entry_seed_count << '\n';
    query_slots = config.query_batch_max;
    result_capacity = std::max<u32>(config.k, config.gpu_final_rerank_width);
    exact_width = kPersistentMaxExact;
    code_bytes = index.layout.code_bytes;
    free_slots.resize(query_slots);
    for (u32 slot = 0; slot < query_slots; ++slot) free_slots[slot] = slot;

    node_record_bytes = static_cast<u32>(8 + VamanaNode::vector_bytes());
    const u64 engine_budget = static_cast<u64>(
      config.gpu_memory_limit_gb - config.gpu_memory_reserve_gb) << 30;
    size_t free_gpu_bytes = 0;
    size_t total_gpu_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes), "cudaMemGetInfo(GPU navigation budget)");
    const u64 runtime_reserve = static_cast<u64>(config.gpu_memory_reserve_gb) << 30;
    const u64 physically_available = free_gpu_bytes > runtime_reserve
      ? static_cast<u64>(free_gpu_bytes) - runtime_reserve : 0;
    const u64 usable_budget = std::min(engine_budget, physically_available);
    const auto budget = memory_budget::estimate(memory_budget::Request{
      .nodes = index.layout.num_nodes,
      .max_delta_vectors = config.max_vectors,
      .usable_bytes = usable_budget,
      .requested_cache_bytes = static_cast<u64>(config.gpu_adjacency_cache_mb) << 20,
      .requested_exact_cache_bytes = static_cast<u64>(config.gpu_exact_cache_mb) << 20,
      .delta_budget_bytes = static_cast<u64>(config.delta_budget_mb) << 20,
      .dim = config.dim,
      .pq_subquantizers = pq_model.subquantizers,
      .code_bytes = code_bytes,
      .vector_bytes = static_cast<u32>(VamanaNode::vector_bytes()),
      .query_slots = query_slots,
      .beam_width = config.gpu_traversal_beam_width,
      .graph_degree = config.R,
      .exact_width = exact_width,
      .exact_record_bytes = node_record_bytes,
      .anchor_count = anchor_table.count(),
      .shard_count = static_cast<u32>(index.shards.size()),
      .entry_point_count = static_cast<u32>(entry_handles.size()),
      .cache_ways = config.gpu_adjacency_cache_ways,
      .exact_cache_ways = config.gpu_exact_cache_ways,
    });
    if (!budget.fits) {
      throw std::runtime_error(
        "GPU navigation allocations exceed the configured memory budget; codes=" +
        std::to_string(budget.code_bytes) + " fixed=" +
        std::to_string(budget.fixed_bytes));
    }
    delta_capacity = budget.delta_capacity;
    delta_table_capacity = budget.delta_table_capacity;
    visited_capacity = budget.visited_capacity;
    graph_cache_sets = budget.cache_sets;
    graph_cache_slots = budget.cache_slots;
    graph_cache_bytes = static_cast<size_t>(budget.cache_payload_bytes);
    exact_cache_sets = budget.exact_cache_sets;
    exact_cache_slots = budget.exact_cache_slots;
    exact_cache_stride = budget.exact_cache_stride;
    exact_cache_bytes = static_cast<size_t>(budget.exact_cache_payload_bytes);
    const u64 invalidation_capacity = static_cast<u64>(config.storage_owner_batch_max) * config.R;
    if (invalidation_capacity > std::numeric_limits<u32>::max()) {
      throw std::runtime_error("GPU navigation graph invalidation capacity exceeds uint32");
    }
    graph_invalidation_capacity = static_cast<u32>(std::max<u64>(1, invalidation_capacity));
    explicit_gpu_bytes = budget.explicit_bytes;
    const u64 base_code_region_bytes = budget.code_bytes;
    const u64 exact_bytes = budget.exact_bytes;
    std::cerr << "[gpu-search] navigation budget codes=" << budget.code_bytes
              << " delta=" << budget.delta_bytes
              << " delta_capacity=" << budget.delta_capacity
              << " delta_codes=" << budget.delta_code_bytes
              << " adjacency_total=" << budget.cache_total_bytes
              << " exact_cache_total=" << budget.exact_cache_total_bytes
              << " explicit=" << explicit_gpu_bytes
              << " limit=" << engine_budget << " bytes\n";

    const size_t code_region_bytes = static_cast<size_t>(base_code_region_bytes);
    exact_region_offset = static_cast<size_t>(align_up(code_region_bytes, 256));
    exact_cache_offset = static_cast<size_t>(align_up(exact_region_offset + exact_bytes, 256));
    graph_cache_offset = static_cast<size_t>(
      align_up(exact_cache_offset + exact_cache_bytes, 512));
    const size_t remote_buffer_bytes = graph_cache_offset + graph_cache_bytes;
#ifdef DVSTOR_HAVE_GPUNETIO
    direct_transport = std::make_unique<gpu::GpuNetioPersistentTransport>(
      config, remote_buffer_bytes, channel_context, connection_manager, remote_regions);
    direct_view = direct_transport->view();
    if (direct_view.data == nullptr || direct_view.data_bytes < remote_buffer_bytes) {
      throw std::runtime_error("GPUNetIO returned an undersized GPU data region");
    }
    d_remote_buffer = direct_view.data;
    owns_remote_buffer = false;
#else
    throw std::runtime_error("GPU query engine requires DOCA GPUNetIO support");
#endif
    d_pq_codes = d_remote_buffer;
    d_exact_records = d_remote_buffer + exact_region_offset;
    d_exact_cache = d_remote_buffer + exact_cache_offset;
    d_graph_cache = d_remote_buffer + graph_cache_offset;

    NavigationBootstrapper bootstrap(
      config, channel_context, connection_manager, remote_regions,
      d_remote_buffer, remote_buffer_bytes);
    std::cerr << "[gpu-search] bootstrap=CPU-posted GPUDirect RDMA; "
                 "queries=strict GPU-initiated GPUNetIO\n";
    stream_codes_to_gpu(bootstrap);

    device_allocate(d_shards, index.shards.size(), "cudaMalloc(GPU navigation shards)");
    device_allocate(d_opq_matrix, pq_model.rotation.size(), "cudaMalloc(OPQ matrix)");
    device_allocate(d_pq_centroids, pq_model.centroids.size(), "cudaMalloc(PQ centroids)");
    device_allocate(d_entry_points, entry_handles.size(), "cudaMalloc(GPU navigation entries)");
    check_cuda(cudaMemcpy(d_shards, index.shards.data(),
                          index.shards.size() * sizeof(format::ShardRegion),
                          cudaMemcpyHostToDevice), "cudaMemcpy(GPU navigation shards)");
    if (!pq_model.rotation.empty()) {
      check_cuda(cudaMemcpy(d_opq_matrix, pq_model.rotation.data(),
                            pq_model.rotation.size() * sizeof(f32),
                            cudaMemcpyHostToDevice), "cudaMemcpy(OPQ matrix)");
    }
    check_cuda(cudaMemcpy(d_pq_centroids, pq_model.centroids.data(),
                          pq_model.centroids.size() * sizeof(f32),
                          cudaMemcpyHostToDevice), "cudaMemcpy(PQ centroids)");
    check_cuda(cudaMemcpy(d_entry_points, entry_handles.data(),
                          entry_handles.size() * sizeof(u32), cudaMemcpyHostToDevice),
               "cudaMemcpy(GPU navigation entries)");
    if (!anchor_table.vectors.empty()) {
      std::vector<f32> transposed_anchors(anchor_table.vectors.size());
      for (u32 anchor = 0; anchor < anchor_table.count(); ++anchor) {
        for (u32 dimension = 0; dimension < anchor_table.dim; ++dimension) {
          transposed_anchors[
            static_cast<size_t>(dimension) * anchor_table.count() + anchor] =
              anchor_table.vectors[
                static_cast<size_t>(anchor) * anchor_table.dim + dimension];
        }
      }
      device_allocate(d_anchor_vectors, anchor_table.vectors.size(),
                      "cudaMalloc(GPU navigation anchors)");
      check_cuda(cudaMemcpy(d_anchor_vectors, transposed_anchors.data(),
                            transposed_anchors.size() * sizeof(f32), cudaMemcpyHostToDevice),
                 "cudaMemcpy(GPU navigation anchors)");
      device_allocate(d_anchor_handles, anchor_table.handles.size(),
                      "cudaMalloc(GPU navigation anchor handles)");
      check_cuda(cudaMemcpy(d_anchor_handles, anchor_table.handles.data(),
                            anchor_table.handles.size() * sizeof(u32), cudaMemcpyHostToDevice),
                 "cudaMemcpy(GPU navigation anchor handles)");
      device_allocate(d_anchor_distances,
                      static_cast<size_t>(query_slots) * anchor_table.count(),
                      "cudaMalloc(GPU navigation anchor distances)");
      device_allocate(d_delta_bucket_heads, anchor_table.count(),
                      "cudaMalloc(GPU navigation delta buckets)");
      check_cuda(cudaMemset(d_delta_bucket_heads, 0xff,
                            static_cast<size_t>(anchor_table.count()) * sizeof(u32)),
                 "cudaMemset(GPU navigation delta buckets)");
    }

    device_allocate(d_queries, static_cast<size_t>(query_slots) * config.dim,
                    "cudaMalloc(GPU navigation queries)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&query_staging_host),
                             static_cast<size_t>(query_slots) * config.dim * sizeof(f32),
                             cudaHostAllocPortable), "cudaHostAlloc(GPU navigation query staging)");
    device_allocate(d_transformed_queries, static_cast<size_t>(query_slots) * config.dim,
                    "cudaMalloc(GPU transformed queries)");
    device_allocate(d_query_luts,
                    static_cast<size_t>(query_slots) * pq_model.subquantizers * 256,
                    "cudaMalloc(GPU PQ query LUTs)");
    device_allocate(d_visited, static_cast<size_t>(query_slots) * visited_capacity,
                    "cudaMalloc(GPU navigation visited)");

    device_allocate(d_graph_cache_keys, graph_cache_slots, "cudaMalloc(navigation cache keys)");
    device_allocate(d_graph_cache_generations, graph_cache_slots,
                    "cudaMalloc(navigation cache generations)");
    device_allocate(d_graph_cache_timestamps, graph_cache_slots,
                    "cudaMalloc(navigation cache timestamps)");
    device_allocate(d_graph_cache_states, graph_cache_slots, "cudaMalloc(navigation cache states)");
    device_allocate(d_graph_cache_readers, graph_cache_slots, "cudaMalloc(navigation cache readers)");
    device_allocate(d_graph_cache_victims, graph_cache_sets, "cudaMalloc(navigation cache victims)");
    device_allocate(d_graph_cache_generation, 1, "cudaMalloc(navigation cache generation)");
    delta_command_capacity = std::max({1u, config.storage_owner_batch_max,
                                       config.query_batch_max});
    mapped_host_allocate(graph_invalidation_keys_host, d_graph_invalidation_keys,
                         graph_invalidation_capacity,
                         "cudaHostAlloc(navigation graph invalidation staging)");
    mapped_host_allocate(delta_supersede_updates_host, d_delta_supersede_updates,
                         delta_command_capacity,
                         "cudaHostAlloc(navigation delta supersede staging)");
    mapped_host_allocate(delta_override_updates_host, d_delta_override_updates,
                         delta_command_capacity,
                         "cudaHostAlloc(navigation delta override staging)");
    check_cuda(cudaMemset(d_graph_cache_states, 0,
                          static_cast<size_t>(graph_cache_slots) * sizeof(u32)),
               "cudaMemset(navigation cache states)");
    check_cuda(cudaMemset(d_graph_cache_readers, 0,
                          static_cast<size_t>(graph_cache_slots) * sizeof(u32)),
               "cudaMemset(navigation cache readers)");
    check_cuda(cudaMemset(d_graph_cache_victims, 0,
                          static_cast<size_t>(graph_cache_sets) * sizeof(u32)),
               "cudaMemset(navigation cache victims)");
    const u64 initial_cache_generation = 1;
    check_cuda(cudaMemcpy(d_graph_cache_generation, &initial_cache_generation,
                          sizeof(initial_cache_generation), cudaMemcpyHostToDevice),
               "cudaMemcpy(navigation cache generation)");

    device_allocate(d_exact_cache_keys, exact_cache_slots,
                    "cudaMalloc(navigation exact-cache keys)");
    device_allocate(d_exact_cache_states, exact_cache_slots,
                    "cudaMalloc(navigation exact-cache states)");
    device_allocate(d_exact_cache_readers, exact_cache_slots,
                    "cudaMalloc(navigation exact-cache readers)");
    device_allocate(d_exact_cache_victims, exact_cache_sets,
                    "cudaMalloc(navigation exact-cache victims)");
    if (exact_cache_slots != 0) {
      check_cuda(cudaMemset(d_exact_cache_states, 0,
                            static_cast<size_t>(exact_cache_slots) * sizeof(u32)),
                 "cudaMemset(navigation exact-cache states)");
      check_cuda(cudaMemset(d_exact_cache_readers, 0,
                            static_cast<size_t>(exact_cache_slots) * sizeof(u32)),
                 "cudaMemset(navigation exact-cache readers)");
      check_cuda(cudaMemset(d_exact_cache_victims, 0,
                            static_cast<size_t>(exact_cache_sets) * sizeof(u32)),
                 "cudaMemset(navigation exact-cache victims)");
    }

    const size_t result_elements = static_cast<size_t>(query_slots) * result_capacity;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&result_ids_host),
                             result_elements * sizeof(u32),
                             cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAlloc(GPU navigation result ids)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_result_ids),
                                        result_ids_host, 0),
               "cudaHostGetDevicePointer(GPU navigation result ids)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&result_distances_host),
                             result_elements * sizeof(f32),
                             cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAlloc(GPU navigation result distances)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_result_distances),
                                        result_distances_host, 0),
               "cudaHostGetDevicePointer(GPU navigation result distances)");

    device_allocate(d_delta_records, delta_capacity, "cudaMalloc(navigation delta records)");
    device_allocate(d_delta_vectors,
                    static_cast<size_t>(delta_capacity) * VamanaNode::vector_bytes(),
                    "cudaMalloc(navigation delta vectors)");
    if (budget.delta_code_bytes !=
        static_cast<u64>(delta_capacity) * this->code_bytes) {
      throw std::logic_error("GPU delta-code budget does not match the PQ code width");
    }
    device_allocate(d_delta_pq_codes,
                    static_cast<size_t>(budget.delta_code_bytes),
                    "cudaMalloc(PQ delta codes)");
    mapped_host_allocate(delta_staging_records_host, d_delta_staging_records,
                         delta_command_capacity,
                         "cudaHostAlloc(navigation delta record staging)");
    mapped_host_allocate(delta_staging_vectors_host, d_delta_staging_vectors,
                         static_cast<size_t>(delta_command_capacity) *
                           VamanaNode::vector_bytes(),
                         "cudaHostAlloc(navigation delta vector staging)");
    device_allocate(d_delta_encode_scratch,
                    static_cast<size_t>(delta_command_capacity) * config.dim,
                    "cudaMalloc(navigation delta encode scratch)");
    device_allocate(d_delta_next, delta_capacity, "cudaMalloc(navigation delta links)");
    device_allocate(d_base_override_keys, delta_table_capacity,
                    "cudaMalloc(navigation override keys)");
    device_allocate(d_base_override_epochs, delta_table_capacity,
                    "cudaMalloc(navigation override epochs)");
    device_allocate(d_delta_remote_keys, delta_table_capacity,
                    "cudaMalloc(navigation delta remote keys)");
    device_allocate(d_delta_remote_slots, delta_table_capacity,
                    "cudaMalloc(navigation delta remote slots)");
    device_allocate(d_delta_count, 1, "cudaMalloc(navigation delta count)");
    clear_delta_device_state();

    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&stop_host), sizeof(u32),
                             cudaHostAllocMapped), "cudaHostAlloc(GPU navigation stop)");
    *stop_host = 0;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&stop_device), stop_host, 0),
               "cudaHostGetDevicePointer(GPU navigation stop)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&direct_disabled_host), sizeof(u32),
                             cudaHostAllocMapped),
               "cudaHostAlloc(GPU navigation direct failure flag)");
    *direct_disabled_host = 0;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&direct_disabled_device),
                                        direct_disabled_host, 0),
               "cudaHostGetDevicePointer(GPU navigation direct failure flag)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&direct_error_host), sizeof(i32),
                             cudaHostAllocMapped),
               "cudaHostAlloc(GPU navigation direct error)");
    *direct_error_host = 0;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&direct_error_device),
                                        direct_error_host, 0),
               "cudaHostGetDevicePointer(GPU navigation direct error)");
    check_cuda(cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking),
               "cudaStreamCreate(GPU navigation kernel)");
    check_cuda(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking),
               "cudaStreamCreate(GPU navigation transfer)");
    check_cuda(cudaStreamCreateWithFlags(&delta_stream, cudaStreamNonBlocking),
               "cudaStreamCreate(GPU navigation delta)");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, static_cast<int>(config.gpu_device)),
               "cudaGetDeviceProperties(GPU navigation)");
    gpu_clock_khz = static_cast<u64>(std::max(1, properties.clockRate));
    const u64 requested_blocks = static_cast<u64>(
      std::max(1, properties.multiProcessorCount)) * config.gpu_persistent_blocks_per_sm;
    const u64 useful_blocks = std::max<u32>(
      config.query_batch_min,
      std::min(config.num_threads, config.query_batch_target));
    kernel_blocks = static_cast<u32>(std::min({
      static_cast<u64>(query_slots), requested_blocks, useful_blocks}));

    kernel_params = PersistentKernelParams{
      .submissions = submissions.device_view(),
      .completions = completions.device_view(),
      .delta_submissions = delta_submissions.device_view(),
      .delta_completions = delta_completions.device_view(),
      .shards = d_shards,
      .num_shards = static_cast<u32>(index.shards.size()),
      .pq_codes = d_pq_codes,
      .opq_matrix = d_opq_matrix,
      .pq_centroids = d_pq_centroids,
      .entry_points = d_entry_points,
      .entry_point_count = static_cast<u32>(entry_handles.size()),
      .num_nodes = static_cast<u32>(index.layout.num_nodes),
      .medoid_ordinal = index.layout.medoid_ordinal,
      .dim = config.dim,
      .pq_subquantizers = pq_model.subquantizers,
      .pq_subvector_dim = pq_model.subvector_dim(),
      .pq_code_bytes = pq_model.code_bytes(),
      .graph_entry_bytes = index.layout.graph_entry_bytes,
      .graph_degree = index.layout.graph_degree,
      .graph_shard_bits = index.layout.graph_shard_bits,
      .node_meta_offset = static_cast<u32>(VamanaNode::offset_id()),
      .node_record_bytes = node_record_bytes,
      .vector_bytes = static_cast<u32>(VamanaNode::vector_bytes()),
      .vector_dtype = static_cast<u32>(config.resolved_vector_dtype()),
      .traversal_beam_width = config.gpu_traversal_beam_width,
      .final_rerank_width = config.gpu_final_rerank_width,
      .entry_seed_count = config.gpu_entry_seed_count,
      .exact_width = exact_width,
      .max_expansions = config.gpu_max_expansions,
      .prefetch_depth = config.gpu_graph_prefetch_depth,
      .visited_capacity = visited_capacity,
      .query_slots = query_slots,
      .direct_region_count = direct_view.remote_region_count,
      .direct_qps_per_node = direct_view.qps_per_node,
      .direct_local_mkey = direct_view.local_mkey,
      .direct_local_iova_base = direct_view.local_iova_base,
      .direct_timeout_ns = 20000000ULL,
      .direct_regions = reinterpret_cast<const DirectRemoteRegion*>(direct_view.remote_regions),
      .direct_qps = direct_view.qp_array,
      .direct_qp_locks = direct_view.qp_locks,
      .direct_dump = direct_view.dump,
      .direct_disabled = direct_disabled_device,
      .direct_error = direct_error_device,
      .delta_records = d_delta_records,
      .delta_vectors = d_delta_vectors,
      .delta_pq_codes = d_delta_pq_codes,
      .delta_staging_records = d_delta_staging_records,
      .delta_staging_vectors = d_delta_staging_vectors,
      .delta_encode_scratch = d_delta_encode_scratch,
      .delta_next = d_delta_next,
      .delta_bucket_heads = d_delta_bucket_heads,
      .delta_count = d_delta_count,
      .delta_capacity = delta_capacity,
      .base_override_keys = d_base_override_keys,
      .base_override_epochs = d_base_override_epochs,
      .base_override_capacity = delta_table_capacity,
      .delta_remote_keys = d_delta_remote_keys,
      .delta_remote_slots = d_delta_remote_slots,
      .delta_remote_capacity = delta_table_capacity,
      .delta_supersede_updates = d_delta_supersede_updates,
      .delta_override_updates = d_delta_override_updates,
      .graph_invalidation_keys = d_graph_invalidation_keys,
      .anchor_vectors = d_anchor_vectors,
      .anchor_handles = d_anchor_handles,
      .anchor_count = anchor_table.count(),
      .delta_anchor_probes = config.gpu_delta_anchor_probes,
      .anchor_distances = d_anchor_distances,
      .stop = stop_device,
      .graph_cache = d_graph_cache,
      .graph_cache_keys = d_graph_cache_keys,
      .graph_cache_generations = d_graph_cache_generations,
      .graph_cache_timestamps = d_graph_cache_timestamps,
      .graph_cache_states = d_graph_cache_states,
      .graph_cache_readers = d_graph_cache_readers,
      .graph_cache_victims = d_graph_cache_victims,
      .graph_cache_generation = d_graph_cache_generation,
      .graph_cache_sets = graph_cache_sets,
      .graph_cache_ways = config.gpu_adjacency_cache_ways,
      .graph_cache_ttl_ns = static_cast<u64>(config.gpu_graph_cache_ttl_us) * 1000,
      .transformed_queries = d_transformed_queries,
      .query_luts = d_query_luts,
      .visited_hash = d_visited,
      .exact_records = d_exact_records,
      .exact_cache = d_exact_cache,
      .exact_cache_stride = exact_cache_stride,
      .exact_cache_sets = exact_cache_sets,
      .exact_cache_ways = config.gpu_exact_cache_ways,
      .exact_cache_keys = d_exact_cache_keys,
      .exact_cache_states = d_exact_cache_states,
      .exact_cache_readers = d_exact_cache_readers,
      .exact_cache_victims = d_exact_cache_victims,
      .result_ids = d_result_ids,
      .result_distances = d_result_distances,
    };
    admission_thread = std::thread([this] { admission_loop(); });
    completion_thread = std::thread([this] { completion_loop(); });
    start_persistent_kernel();
    maintenance_thread = std::thread([this] { maintenance_loop(); });
  }

  void stream_codes_to_gpu(NavigationBootstrapper& source) {
    const u64 window_bytes = static_cast<u64>(config.gpu_bootstrap_window_mb) << 20;
    std::vector<NavigationRead> requests;
    std::vector<i32> statuses;
    requests.reserve(config.gpu_bootstrap_windows);
    u64 streamed = 0;
    for (const format::ShardRegion& shard : index.shards) {
      for (u64 offset = 0; offset < shard.code_bytes;) {
        requests.clear();
        for (u32 window = 0; window < config.gpu_bootstrap_windows &&
             offset < shard.code_bytes; ++window) {
          const u32 bytes = static_cast<u32>(std::min<u64>(
            window_bytes, shard.code_bytes - offset));
          requests.push_back(NavigationRead{
            .remote_offset = shard.code_remote_offset + offset,
            .destination_address = reinterpret_cast<u64>(d_pq_codes +
              shard.ordinal_base * code_bytes + offset),
            .bytes = bytes,
            .memory_node = static_cast<u16>(shard.memory_node),
          });
          offset += bytes;
        }
        statuses.assign(requests.size(), -EIO);
        source.read(requests, statuses);
        for (size_t request_index = 0; request_index < statuses.size(); ++request_index) {
          if (statuses[request_index] <= 0) {
            const NavigationRead& request = requests[request_index];
            throw std::runtime_error(
              "RDMA PQ code bootstrap failed: status=" +
              std::to_string(statuses[request_index]) + " shard=" +
              std::to_string(request.memory_node) + " remote_offset=" +
              std::to_string(request.remote_offset) + " bytes=" +
              std::to_string(request.bytes) + " destination=" +
              std::to_string(request.destination_address));
          }
        }
        for (const NavigationRead& request : requests) streamed += request.bytes;
      }
    }
    const u64 expected = index.layout.num_nodes * code_bytes;
    if (streamed != expected) throw std::runtime_error("GPU PQ code bootstrap size mismatch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(GPU PQ bootstrap)");

    struct AuditSample {
      u32 shard{};
      u64 slot{};
      u64 ordinal{};
    };
    std::vector<AuditSample> samples;
    samples.reserve(index.shards.size() * 3);
    for (const format::ShardRegion& shard : index.shards) {
      const std::array<u64, 3> shard_slots{0, shard.node_count / 2, shard.node_count - 1};
      for (size_t sample_index = 0; sample_index < shard_slots.size(); ++sample_index) {
        if (sample_index != 0 && shard_slots[sample_index] == shard_slots[sample_index - 1]) {
          continue;
        }
        const u64 slot = shard_slots[sample_index];
        samples.push_back(AuditSample{
          .shard = shard.memory_node,
          .slot = slot,
          .ordinal = shard.ordinal_base + slot,
        });
      }
    }
    std::vector<byte_t> authoritative(code_bytes);
    std::vector<byte_t> resident(code_bytes);
    for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
      const AuditSample& sample = samples[sample_index];
      const format::ShardRegion& shard = index.shards[sample.shard];
      requests.assign(1, NavigationRead{
        .remote_offset = shard.code_remote_offset + sample.slot * code_bytes,
        .destination_address = reinterpret_cast<u64>(d_exact_records),
        .bytes = code_bytes,
        .memory_node = static_cast<u16>(sample.shard),
      });
      statuses.assign(1, -EIO);
      source.read(requests, statuses);
      if (statuses.front() <= 0) {
        throw std::runtime_error(
          "GPU PQ ordinal audit RDMA read failed: shard=" +
          std::to_string(sample.shard) + " slot=" +
          std::to_string(sample.slot) + " status=" +
          std::to_string(statuses.front()));
      }
      check_cuda(cudaMemcpy(authoritative.data(), d_exact_records, authoritative.size(),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(GPU PQ audit source)");
      check_cuda(cudaMemcpy(
        resident.data(),
        d_pq_codes + sample.ordinal * code_bytes,
        resident.size(), cudaMemcpyDeviceToHost),
        "cudaMemcpy(GPU PQ audit resident)");
      if (!std::equal(resident.begin(), resident.end(), authoritative.begin())) {
        throw std::runtime_error(
          "GPU PQ ordinal mapping mismatch: shard=" +
          std::to_string(sample.shard) + " slot=" +
          std::to_string(sample.slot) + " ordinal=" +
          std::to_string(sample.ordinal));
      }
    }
    std::cerr << "[gpu-search] streamed " << streamed
              << " PQ bytes directly into final GPU storage; ordinal audit passed for "
              << samples.size() << " entries\n";
  }

  void clear_delta_device_state() {
    bind_cuda_device("cudaSetDevice(GPU navigation delta reset)");
    check_cuda(cudaMemset(d_delta_records, 0,
                          static_cast<size_t>(delta_capacity) * sizeof(DeviceDeltaRecord)),
               "cudaMemset(navigation delta records)");
    check_cuda(cudaMemset(d_delta_next, 0xff,
                          static_cast<size_t>(delta_capacity) * sizeof(u32)),
               "cudaMemset(navigation delta links)");
    check_cuda(cudaMemset(d_base_override_keys, 0xff,
                          static_cast<size_t>(delta_table_capacity) * sizeof(u32)),
               "cudaMemset(navigation override keys)");
    check_cuda(cudaMemset(d_base_override_epochs, 0,
                          static_cast<size_t>(delta_table_capacity) * sizeof(u64)),
               "cudaMemset(navigation override epochs)");
    check_cuda(cudaMemset(d_delta_remote_keys, 0,
                          static_cast<size_t>(delta_table_capacity) * sizeof(u64)),
               "cudaMemset(navigation remote keys)");
    check_cuda(cudaMemset(d_delta_remote_slots, 0xff,
                          static_cast<size_t>(delta_table_capacity) * sizeof(u32)),
               "cudaMemset(navigation remote slots)");
    check_cuda(cudaMemset(d_delta_count, 0, sizeof(u32)), "cudaMemset(navigation delta count)");
    if (d_delta_bucket_heads != nullptr) {
      check_cuda(cudaMemset(d_delta_bucket_heads, 0xff,
                            static_cast<size_t>(anchor_table.count()) * sizeof(u32)),
                 "cudaMemset(navigation delta buckets)");
    }
  }

  void start_persistent_kernel() {
    bind_cuda_device("cudaSetDevice(GPU navigation kernel start)");
    std::atomic_ref<u32>(*stop_host).store(0, std::memory_order_release);
    (void)cudaGetLastError();
    launch_persistent_search(kernel_stream, kernel_params, kernel_blocks, 256);
    check_cuda(cudaGetLastError(), "launch_persistent_search(navigation)");
    PersistentKernelParams control_params = kernel_params;
    control_params.submissions = {};
    control_params.completions = {};
    launch_persistent_search(delta_stream, control_params, 1, 256);
    check_cuda(cudaGetLastError(), "launch_persistent_search(delta control)");
    kernel_running = true;
    std::cerr << "[gpu-search] persistent CTAs=" << kernel_blocks
              << "+1-control threads/CTA=256 query_slots=" << query_slots << '\n';
  }

  void stop_persistent_kernel() {
    if (!kernel_running) return;
    bind_cuda_device("cudaSetDevice(GPU navigation kernel stop)");
    std::atomic_ref<u32>(*stop_host).store(1, std::memory_order_release);
    const cudaError_t query_status = cudaStreamSynchronize(kernel_stream);
    const cudaError_t control_status = cudaStreamSynchronize(delta_stream);
    kernel_running = false;
    check_cuda(query_status, "cudaStreamSynchronize(GPU navigation stop)");
    check_cuda(control_status, "cudaStreamSynchronize(GPU delta control stop)");
  }

  ~Impl() {
    const cudaError_t device_status = cudaSetDevice(static_cast<int>(config.gpu_device));
    if (device_status != cudaSuccess) {
      std::cerr << "[gpu-search] failed to bind CUDA device during teardown: "
                << cudaGetErrorString(device_status) << '\n';
    }
    accepting.store(false, std::memory_order_release);
    maintenance_shutdown.store(true, std::memory_order_release);
    maintenance_cv.notify_all();
    admission_cv.notify_all();
    slot_cv.notify_all();
    if (maintenance_thread.joinable()) maintenance_thread.join();
    shutdown.store(true, std::memory_order_release);
    admission_cv.notify_all();
    if (admission_thread.joinable()) admission_thread.join();
    reject_queued_submissions("persistent GPU query engine is stopping");
    const auto drain_deadline = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
    while (pending_count.load(std::memory_order_acquire) != 0 &&
           std::chrono::steady_clock::now() < drain_deadline) {
      std::this_thread::yield();
    }
    if (kernel_running) {
      std::atomic_ref<u32>(*stop_host).store(1, std::memory_order_release);
      if (kernel_stream != nullptr) cudaStreamSynchronize(kernel_stream);
      if (delta_stream != nullptr) cudaStreamSynchronize(delta_stream);
      kernel_running = false;
    }
    reject_all_pending("persistent GPU query engine stopped before completion");
    if (completion_thread.joinable()) completion_thread.join();
    if (delta_stream != nullptr) cudaStreamDestroy(delta_stream);
    if (transfer_stream != nullptr) cudaStreamDestroy(transfer_stream);
    if (kernel_stream != nullptr) cudaStreamDestroy(kernel_stream);
    if (direct_disabled_host != nullptr) cudaFreeHost(direct_disabled_host);
    if (direct_error_host != nullptr) cudaFreeHost(direct_error_host);
    if (stop_host != nullptr) cudaFreeHost(stop_host);
    if (result_distances_host != nullptr) cudaFreeHost(result_distances_host);
    if (result_ids_host != nullptr) cudaFreeHost(result_ids_host);
    if (delta_staging_vectors_host != nullptr) cudaFreeHost(delta_staging_vectors_host);
    if (delta_staging_records_host != nullptr) cudaFreeHost(delta_staging_records_host);
    if (delta_override_updates_host != nullptr) cudaFreeHost(delta_override_updates_host);
    if (delta_supersede_updates_host != nullptr) cudaFreeHost(delta_supersede_updates_host);
    if (graph_invalidation_keys_host != nullptr) cudaFreeHost(graph_invalidation_keys_host);
    device_free(d_delta_count);
    device_free(d_delta_remote_slots);
    device_free(d_delta_remote_keys);
    device_free(d_base_override_epochs);
    device_free(d_base_override_keys);
    device_free(d_delta_next);
    device_free(d_delta_pq_codes);
    device_free(d_delta_encode_scratch);
    device_free(d_delta_vectors);
    device_free(d_delta_records);
    device_free(d_graph_cache_generation);
    device_free(d_graph_cache_victims);
    device_free(d_graph_cache_states);
    device_free(d_graph_cache_readers);
    device_free(d_graph_cache_timestamps);
    device_free(d_graph_cache_generations);
    device_free(d_graph_cache_keys);
    device_free(d_exact_cache_victims);
    device_free(d_exact_cache_readers);
    device_free(d_exact_cache_states);
    device_free(d_exact_cache_keys);
    if (owns_remote_buffer) device_free(d_remote_buffer);
#ifdef DVSTOR_HAVE_GPUNETIO
    direct_transport.reset();
#endif
    device_free(d_visited);
    device_free(d_query_luts);
    device_free(d_transformed_queries);
    if (query_staging_host != nullptr) cudaFreeHost(query_staging_host);
    device_free(d_queries);
    device_free(d_delta_bucket_heads);
    device_free(d_anchor_distances);
    device_free(d_anchor_handles);
    device_free(d_anchor_vectors);
    device_free(d_entry_points);
    device_free(d_pq_centroids);
    device_free(d_opq_matrix);
    device_free(d_shards);
  }

  service::QueryResult search(VectorDType query_dtype, const byte_t* query_data, u32 k) {
    if (!accepting.load(std::memory_order_acquire)) {
      throw std::runtime_error("persistent GPU search engine is stopping");
    }
    if (!healthy.load(std::memory_order_acquire)) {
      throw std::runtime_error(unhealthy_message());
    }
    if (query_data == nullptr || k == 0 || k > result_capacity) {
      throw std::invalid_argument("invalid persistent GPU query");
    }
    u32 slot = 0;
    {
      std::unique_lock<std::mutex> lock(slot_mutex);
      slot_cv.wait(lock, [&] {
        return !free_slots.empty() || !accepting.load() || !healthy.load();
      });
      if (!accepting.load()) throw std::runtime_error("persistent GPU search engine stopped");
      if (!healthy.load()) {
        lock.unlock();
        throw std::runtime_error(unhealthy_message());
      }
      slot = free_slots.back();
      free_slots.pop_back();
    }
    f32* decoded = query_staging_host + static_cast<size_t>(slot) * config.dim;
    for (u32 dimension = 0; dimension < config.dim; ++dimension) {
      decoded[dimension] = vector_component_as_float(query_data, query_dtype, dimension);
    }
    const u64 request_id = next_request_id.fetch_add(1, std::memory_order_relaxed);
    const auto submitted_at = std::chrono::steady_clock::now();
    auto pending = std::make_shared<PendingQuery>();
    pending->slot = slot;
    pending->submitted_at = submitted_at;
    auto future = pending->promise.get_future();
    {
      std::lock_guard<std::mutex> lock(pending_mutex);
      pending_queries.emplace(request_id, pending);
      pending_count.fetch_add(1, std::memory_order_relaxed);
    }
    QueryDescriptor descriptor{
      .request_id = request_id,
      .snapshot_epoch = engine.delta_.published_epoch(),
      .query_device_address = reinterpret_cast<u64>(
        d_queries + static_cast<size_t>(slot) * config.dim),
      .result_device_address = reinterpret_cast<u64>(
        d_result_ids + static_cast<size_t>(slot) * result_capacity),
      .query_slot = slot,
      .result_capacity = result_capacity,
      .dim = static_cast<u16>(config.dim),
      .k = static_cast<u16>(k),
      .query_dtype = static_cast<u8>(VectorDType::float32),
    };
    bool rejected = false;
    std::string rejection_message;
    {
      std::lock_guard<std::mutex> lock(admission_mutex);
      if (!healthy.load(std::memory_order_relaxed)) {
        rejected = true;
        rejection_message = health_error;
      } else {
        admission_queue.push_back({.descriptor = descriptor, .enqueued_at = submitted_at});
      }
    }
    if (rejected) {
      reject_submission({.descriptor = descriptor, .enqueued_at = submitted_at},
                        rejection_message);
    } else {
      admission_cv.notify_one();
    }
    engine.telemetry_.queries_submitted.fetch_add(1, std::memory_order_relaxed);
    return future.get();
  }

  void admission_loop() {
    std::vector<PendingSubmission> batch;
    batch.reserve(config.query_batch_target);
    size_t submitted_count = 0;
    try {
      bind_cuda_device("cudaSetDevice(GPU navigation admission)");
      while (!shutdown.load(std::memory_order_acquire)) {
        batch.clear();
        submitted_count = 0;
        {
          std::unique_lock<std::mutex> lock(admission_mutex);
          admission_cv.wait(lock, [&] {
            return (!admission_queue.empty() && !admission_paused) ||
                   !healthy.load() || shutdown.load();
          });
          if (!healthy.load(std::memory_order_acquire) || shutdown.load()) return;
          if (admission_queue.empty() || admission_paused) continue;
          const auto deadline = admission_queue.front().enqueued_at +
            std::chrono::microseconds(config.query_batch_wait_us);
          admission_cv.wait_until(lock, deadline, [&] {
            return admission_queue.size() >= config.query_batch_min ||
                   admission_paused || shutdown.load();
          });
          if (shutdown.load()) return;
          if (admission_paused) continue;
          const size_t count = std::min<size_t>(
            admission_queue.size(), std::min(config.query_batch_target, config.query_batch_max));
          for (size_t index = 0; index < count; ++index) {
            batch.push_back(admission_queue.front());
            admission_queue.pop_front();
          }
          active_gpu_queries.fetch_add(count, std::memory_order_release);
        }
        if (batch.empty()) continue;
        for (const PendingSubmission& submission : batch) {
          const u32 slot = submission.descriptor.query_slot;
          check_cuda(cudaMemcpyAsync(
            d_queries + static_cast<size_t>(slot) * config.dim,
            query_staging_host + static_cast<size_t>(slot) * config.dim,
            static_cast<size_t>(config.dim) * sizeof(f32),
            cudaMemcpyHostToDevice, transfer_stream), "cudaMemcpyAsync(GPU navigation query)");
        }
        check_cuda(cudaStreamSynchronize(transfer_stream),
                   "cudaStreamSynchronize(GPU navigation query)");
        const auto admitted_at = std::chrono::steady_clock::now();
        u64 wait_ns = 0;
        for (PendingSubmission& submission : batch) {
          submission.descriptor.snapshot_epoch = engine.delta_.published_epoch();
          while (!submissions.try_push(submission.descriptor)) {
            if (shutdown.load(std::memory_order_acquire) ||
                !healthy.load(std::memory_order_acquire)) {
              throw std::runtime_error("GPU submission ring stopped making progress");
            }
            std::this_thread::yield();
          }
          ++submitted_count;
          wait_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            admitted_at - submission.enqueued_at).count());
        }
        engine.telemetry_.batches.fetch_add(1, std::memory_order_relaxed);
        engine.telemetry_.batch_queries.fetch_add(batch.size(), std::memory_order_relaxed);
        engine.telemetry_.submission_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
      }
    } catch (const std::exception& error) {
      for (size_t index = submitted_count; index < batch.size(); ++index) {
        reject_submission(batch[index], error.what());
      }
      const size_t rejected_count = batch.size() - submitted_count;
      if (rejected_count != 0) {
        active_gpu_queries.fetch_sub(rejected_count, std::memory_order_release);
        maintenance_cv.notify_all();
      }
      mark_unhealthy(std::string{"GPU admission failed: "} + error.what());
    } catch (...) {
      for (size_t index = submitted_count; index < batch.size(); ++index) {
        reject_submission(batch[index], "unknown GPU admission failure");
      }
      const size_t rejected_count = batch.size() - submitted_count;
      if (rejected_count != 0) {
        active_gpu_queries.fetch_sub(rejected_count, std::memory_order_release);
        maintenance_cv.notify_all();
      }
      mark_unhealthy("unknown GPU admission failure");
    }
  }

  void decode_mutation_payload(const DeltaMutation& mutation,
                               std::vector<f32>& decoded) const {
    std::fill(decoded.begin(), decoded.end(), 0.0f);
    if (mutation.kind == service::storage_owner::MutationKind::erase) return;
    if (mutation.vector.size() == static_cast<size_t>(config.dim) * sizeof(f32)) {
      std::memcpy(decoded.data(), mutation.vector.data(), mutation.vector.size());
    } else if (mutation.vector.size() ==
               vector_dtype_bytes(config.resolved_vector_dtype(), config.dim)) {
      decode_storage_vector_to_float(mutation.vector.data(), config.resolved_vector_dtype(),
                                     config.dim, decoded.data());
    } else {
      throw std::invalid_argument("GPU delta mutation vector has an invalid size");
    }
  }

  u32 nearest_anchor(const std::vector<f32>& vector, u64 remote_node) const {
    if (anchor_table.count() == 0) return 0;
    const u32 shard = static_cast<u32>(remote_node >> 48);
    if (shard >= index.shards.size()) return 0;
    const u32 begin = anchor_table.shard_offsets[shard];
    const u32 end = anchor_table.shard_offsets[shard + 1];
    if (begin == end) return 0;
    u32 best = begin;
    f32 best_distance = std::numeric_limits<f32>::max();
    for (u32 anchor = begin; anchor < end; ++anchor) {
      f32 distance = 0.0f;
      const f32* candidate = anchor_table.vectors.data() +
        static_cast<size_t>(anchor) * config.dim;
      for (u32 dimension = 0; dimension < config.dim; ++dimension) {
        const f32 difference = vector[dimension] - candidate[dimension];
        distance += difference * difference;
      }
      if (distance < best_distance) {
        best_distance = distance;
        best = anchor;
      }
    }
    return best;
  }

  u64 graph_cache_key(u64 raw) const {
    const u32 shard = static_cast<u32>(raw >> 48);
    const u64 node_offset = (raw << 16) >> 16;
    if (raw == 0 || shard >= index.shards.size()) {
      throw std::runtime_error("storage returned an invalid GPU graph-cache invalidation");
    }
    const format::ShardRegion& region = index.shards[shard];
    u64 graph_offset = 0;
    if (node_offset >= region.node_base_offset && region.node_stride != 0) {
      const u64 relative = node_offset - region.node_base_offset;
      if (relative % region.node_stride == 0 &&
          relative / region.node_stride < region.node_count) {
        graph_offset = region.graph_base_offset +
          (relative / region.node_stride) * index.layout.graph_entry_bytes;
      }
    }
    if (graph_offset == 0) {
      if (node_offset < region.dynamic_base_offset || region.dynamic_record_bytes == 0 ||
          (node_offset - region.dynamic_base_offset) % region.dynamic_record_bytes != 0) {
        throw std::runtime_error("storage returned a misaligned GPU graph-cache invalidation");
      }
      graph_offset = node_offset + region.dynamic_hot_offset;
    }
    return (static_cast<u64>(shard) << 48) | graph_offset;
  }

  std::vector<u64> graph_cache_keys(std::span<const u64> raw_nodes) const {
    std::vector<u64> keys;
    keys.reserve(raw_nodes.size());
    for (u64 raw : raw_nodes) keys.push_back(graph_cache_key(raw));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() > graph_invalidation_capacity) {
      throw std::runtime_error("GPU navigation graph invalidation batch exceeds capacity");
    }
    return keys;
  }

  void submit_delta_publication(const DeltaPublishDescriptor& descriptor) {
    const auto timeout = std::chrono::milliseconds(std::clamp<u32>(
      config.storage_owner_rpc_timeout_ms, 1000, 5000));
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!delta_submissions.try_push(descriptor)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("persistent GPU delta command queue is not making progress");
      }
      std::this_thread::yield();
    }

    DeltaPublishCompletion completion{};
    while (!delta_completions.try_pop(completion)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("persistent GPU delta publication timed out");
      }
      std::this_thread::yield();
    }
    if (completion.command_id != descriptor.command_id || completion.status != 0 ||
        completion.final_count != descriptor.final_count) {
      throw std::runtime_error(
        "persistent GPU delta publication failed: command=" +
        std::to_string(completion.command_id) + " status=" +
        std::to_string(completion.status) + " count=" +
        std::to_string(completion.final_count));
    }
  }

  void upload_records_locked(std::vector<DeltaMutation>& mutations, bool rebuilding,
                             std::span<const u64> invalidation_keys = {}) {
    const auto prepare_started = std::chrono::steady_clock::now();
    bind_cuda_device("cudaSetDevice(GPU navigation delta publication)");
    (void)cudaGetLastError();
    const u32 first_slot = static_cast<u32>(delta_records_host.size());
    if (first_slot + mutations.size() > delta_capacity) {
      throw std::runtime_error("GPU navigation delta live set exceeds its configured capacity");
    }
    const size_t vector_bytes = VamanaNode::vector_bytes();
    std::vector<DeviceDeltaRecord> records;
    std::vector<byte_t> vectors(static_cast<size_t>(mutations.size()) * vector_bytes);
    std::vector<byte_t> entries(
      static_cast<size_t>(mutations.size()) * code_bytes);
    records.reserve(mutations.size());
    std::vector<DeltaSupersedeUpdate> superseded_updates;
    std::vector<DeltaOverrideUpdate> override_updates;
    std::vector<f32> decoded(config.dim);
    std::vector<byte_t> entry(code_bytes);
    std::vector<f32> transformed(config.dim);
    for (size_t mutation_index = 0; mutation_index < mutations.size(); ++mutation_index) {
      DeltaMutation& mutation = mutations[mutation_index];
      bool decoded_ready = false;
      std::fill(entry.begin(), entry.end(), 0);
      if (rebuilding &&
          mutation.kind != service::storage_owner::MutationKind::erase) {
        decode_mutation_payload(mutation, decoded);
        decoded_ready = true;
        pq::encode(pq_model, decoded, entry, transformed);
      }
      const u32 slot = static_cast<u32>(delta_records_host.size());
      if (!rebuilding) {
        const auto previous = latest_delta_slot.find(mutation.id);
        if (previous != latest_delta_slot.end()) {
          delta_records_host[previous->second].superseded_epoch = mutation.epoch;
          if (previous->second >= first_slot) {
            records[previous->second - first_slot].superseded_epoch = mutation.epoch;
          } else {
            superseded_updates.push_back(DeltaSupersedeUpdate{
              .slot = previous->second,
              .epoch = mutation.epoch,
            });
          }
        }
      }
      const bool deleted = mutation.kind == service::storage_owner::MutationKind::erase;
      const u64 record_remote = mutation.remote_node != 0
        ? mutation.remote_node : mutation.old_remote_node;
      u32 bucket = 0;
      if (!deleted) {
        const auto hinted = anchor_buckets_by_raw.find(mutation.anchor_hint);
        if (hinted == anchor_buckets_by_raw.end()) {
          if (!decoded_ready) {
            decode_mutation_payload(mutation, decoded);
            decoded_ready = true;
          }
          bucket = nearest_anchor(decoded, record_remote);
        } else {
          bucket = hinted->second;
        }
      }
      DeviceDeltaRecord record{
        .id = mutation.id,
        .generation = std::max<u32>(1, mutation.generation),
        .flags = deleted ? kDeltaDeleted : 0u,
        .signature = 0,
        .epoch = mutation.epoch,
        .remote_node = record_remote,
        .anchor_bucket = bucket,
      };
      delta_records_host.push_back(record);
      records.push_back(record);
      latest_delta_slot[mutation.id] = slot;
      byte_t* stored_vector = vectors.data() + mutation_index * vector_bytes;
      if (deleted) {
        std::memset(stored_vector, 0, vector_bytes);
      } else if (mutation.vector.size() == vector_bytes) {
        std::memcpy(stored_vector, mutation.vector.data(), vector_bytes);
      } else {
        if (!decoded_ready) {
          decode_mutation_payload(mutation, decoded);
          decoded_ready = true;
        }
        encode_float_vector_to_storage(decoded.data(), config.dim,
                                       config.resolved_vector_dtype(), stored_vector);
      }
      std::copy(entry.begin(), entry.end(),
                entries.begin() + mutation_index * code_bytes);
      u32 ordinal = 0;
      if (format::remote_to_ordinal(index, RemotePtr{mutation.old_remote_node}, ordinal)) {
        const auto [it, inserted] = base_override_epochs.emplace(ordinal, mutation.epoch);
        if (inserted) {
          override_updates.push_back(DeltaOverrideUpdate{
            .ordinal = ordinal,
            .epoch = mutation.epoch,
          });
        } else if (mutation.epoch < it->second) {
          it->second = mutation.epoch;
          override_updates.push_back(DeltaOverrideUpdate{
            .ordinal = ordinal,
            .epoch = mutation.epoch,
          });
        }
      }
    }

    if (!rebuilding &&
        (records.size() > delta_command_capacity ||
         superseded_updates.size() > delta_command_capacity ||
         override_updates.size() > delta_command_capacity ||
         invalidation_keys.size() > graph_invalidation_capacity)) {
      throw std::runtime_error("GPU navigation delta control batch exceeds capacity");
    }

    if (!rebuilding) {
      std::memcpy(delta_staging_records_host, records.data(),
                  records.size() * sizeof(DeviceDeltaRecord));
      std::memcpy(delta_staging_vectors_host, vectors.data(), vectors.size());
      if (!superseded_updates.empty()) {
        std::memcpy(delta_supersede_updates_host, superseded_updates.data(),
                    superseded_updates.size() * sizeof(DeltaSupersedeUpdate));
      }
      if (!override_updates.empty()) {
        std::memcpy(delta_override_updates_host, override_updates.data(),
                    override_updates.size() * sizeof(DeltaOverrideUpdate));
      }
      if (!invalidation_keys.empty()) {
        std::memcpy(graph_invalidation_keys_host, invalidation_keys.data(),
                    invalidation_keys.size() * sizeof(u64));
      }
      const u32 count = static_cast<u32>(delta_records_host.size());
      const auto command_started = std::chrono::steady_clock::now();
      engine.telemetry_.publication_prepare_ns_total.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
          command_started - prepare_started).count()), std::memory_order_relaxed);
      submit_delta_publication(DeltaPublishDescriptor{
        .command_id = next_delta_command_id.fetch_add(1, std::memory_order_relaxed),
        .first_slot = first_slot,
        .record_count = static_cast<u32>(records.size()),
        .final_count = count,
        .invalidation_count = static_cast<u32>(invalidation_keys.size()),
        .superseded_count = static_cast<u32>(superseded_updates.size()),
        .override_count = static_cast<u32>(override_updates.size()),
      });
      engine.telemetry_.publication_command_ns_total.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - command_started).count()),
        std::memory_order_relaxed);
      engine.telemetry_.delta_live_entries.store(count, std::memory_order_relaxed);
      return;
    }

    if (!records.empty()) {
      check_cuda(cudaMemcpyAsync(d_delta_records + first_slot, records.data(),
                                 records.size() * sizeof(DeviceDeltaRecord),
                                 cudaMemcpyHostToDevice, delta_stream),
                 "cudaMemcpyAsync(navigation delta records)");
      check_cuda(cudaMemcpyAsync(
        d_delta_vectors + static_cast<size_t>(first_slot) * vector_bytes,
        vectors.data(), vectors.size(), cudaMemcpyHostToDevice, delta_stream),
        "cudaMemcpyAsync(navigation delta vectors)");
      check_cuda(cudaMemcpyAsync(
        d_delta_pq_codes + static_cast<size_t>(first_slot) * code_bytes,
        entries.data(), entries.size(), cudaMemcpyHostToDevice, delta_stream),
        "cudaMemcpyAsync(navigation delta codes)");
    }

    for (size_t index_in_batch = 0; index_in_batch < records.size(); ++index_in_batch) {
      const u32 slot = first_slot + static_cast<u32>(index_in_batch);
      const DeviceDeltaRecord& record = records[index_in_batch];
      if (record.remote_node != 0) {
        launch_insert_delta_remote(delta_stream, d_delta_remote_keys, d_delta_remote_slots,
                                   delta_table_capacity, record.remote_node, slot);
      }
      if ((record.flags & kDeltaDeleted) == 0 && d_delta_bucket_heads != nullptr) {
        launch_link_delta_bucket(delta_stream, d_delta_bucket_heads, d_delta_next,
                                 record.anchor_bucket, slot);
      }
    }
    for (const DeltaSupersedeUpdate& update : superseded_updates) {
      launch_supersede_delta_record(delta_stream, d_delta_records, update.slot,
                                    update.epoch);
    }
    if (rebuilding) {
      for (const auto& [ordinal, override_epoch] : base_override_epochs) {
        launch_insert_base_override(delta_stream, d_base_override_keys,
                                    d_base_override_epochs, delta_table_capacity,
                                    ordinal, override_epoch);
      }
    }
    const u32 count = static_cast<u32>(delta_records_host.size());
    launch_publish_delta_count(delta_stream, d_delta_count, count);
    check_cuda(cudaGetLastError(), "launch GPU navigation delta publication");
    check_cuda(cudaStreamSynchronize(delta_stream), "cudaStreamSynchronize(navigation delta publish)");
    engine.telemetry_.delta_live_entries.store(count, std::memory_order_relaxed);
  }

  void upload_mutations(std::vector<DeltaMutation>& mutations, u64 epoch,
                        std::span<const u64> invalidated_graph_nodes) {
    if (mutations.empty()) return;
    const std::vector<u64> invalidation_keys = graph_cache_keys(invalidated_graph_nodes);
    if (delta_records_host.size() + mutations.size() >
        static_cast<size_t>(delta_capacity) * 4 / 5) {
      compact_delta();
    }
    std::lock_guard<std::mutex> lock(delta_mutex);
    std::unordered_map<node_t, u32> batch_generations;
    for (DeltaMutation& mutation : mutations) {
      mutation.epoch = epoch;
      auto [iterator, inserted] = batch_generations.emplace(mutation.id, 0);
      if (inserted) {
        const auto version = engine.delta_.version(mutation.id);
        iterator->second = version ? version->generation : 0;
      }
      mutation.generation = std::max(
        mutation.generation, static_cast<u32>(iterator->second + 1));
      iterator->second = mutation.generation;
    }
    upload_records_locked(mutations, false, invalidation_keys);
  }

  void pause_admission_for_compaction() {
    {
      std::lock_guard<std::mutex> lock(admission_mutex);
      admission_paused = true;
    }
    admission_cv.notify_all();
    std::unique_lock<std::mutex> lock(maintenance_mutex);
    maintenance_cv.wait(lock, [&] {
      return active_gpu_queries.load(std::memory_order_acquire) == 0;
    });
  }

  void resume_admission_after_compaction() {
    {
      std::lock_guard<std::mutex> lock(admission_mutex);
      admission_paused = false;
    }
    admission_cv.notify_all();
  }

  void compact_delta() {
    std::lock_guard<std::mutex> compaction_lock(compaction_mutex);
    pause_admission_for_compaction();
    try {
      stop_persistent_kernel();
      std::lock_guard<std::mutex> delta_lock(delta_mutex);
      DeltaSnapshot snapshot = engine.delta_.begin_consolidation();
      if (snapshot.mutations.size() > delta_capacity) {
        throw std::runtime_error("GPU navigation live delta exceeds capacity");
      }
      clear_delta_device_state();
      delta_records_host.clear();
      latest_delta_slot.clear();
      upload_records_locked(snapshot.mutations, true);
      engine.delta_.mark_compacted();
      engine.telemetry_.delta_compactions.fetch_add(1, std::memory_order_relaxed);
      start_persistent_kernel();
    } catch (...) {
      const std::exception_ptr compaction_error = std::current_exception();
      std::exception_ptr restart_error;
      try {
        if (!kernel_running) start_persistent_kernel();
      } catch (...) {
        restart_error = std::current_exception();
      }
      resume_admission_after_compaction();
      if (restart_error != nullptr) std::rethrow_exception(restart_error);
      std::rethrow_exception(compaction_error);
    }
    resume_admission_after_compaction();
  }

  void maintenance_loop() {
    bind_cuda_device("cudaSetDevice(GPU navigation maintenance)");
    const auto period = std::chrono::milliseconds(config.merge_period_ms);
    while (!maintenance_shutdown.load(std::memory_order_acquire)) {
      {
        std::unique_lock<std::mutex> lock(maintenance_mutex);
        maintenance_cv.wait_for(lock, period, [&] { return maintenance_shutdown.load(); });
      }
      if (maintenance_shutdown.load()) break;
      bool compact = false;
      {
        std::lock_guard<std::mutex> lock(delta_mutex);
        const size_t live = engine.delta_.delta_size();
        compact = delta_records_host.size() > live + std::max<size_t>(128, live / 4) ||
          delta_records_host.size() >= static_cast<size_t>(delta_capacity) * 4 / 5;
      }
      if (!compact) continue;
      try {
        std::lock_guard<std::mutex> publish_lock(engine.mutation_publish_mutex_);
        compact_delta();
      } catch (const std::exception& error) {
        mark_unhealthy(std::string{"GPU delta compaction failed: "} + error.what());
        break;
      }
    }
  }

  void report_direct_path_failure() {
    if (direct_disabled_host == nullptr ||
        std::atomic_ref<u32>(*direct_disabled_host).load(std::memory_order_acquire) == 0) {
      return;
    }
    bool expected = false;
    if (!direct_failure_logged.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) return;
    const i32 direct_error = direct_error_host == nullptr
      ? 0 : std::atomic_ref<i32>(*direct_error_host).load(std::memory_order_acquire);
    std::cerr << "[gpu-search] GPUNetIO direct read failed with status=" << direct_error
              << "; strict GPUNetIO mode rejects the query\n";
    engine.telemetry_.direct_path_failures.fetch_add(1, std::memory_order_relaxed);
    mark_unhealthy("GPUNetIO direct read failed with status " +
                   std::to_string(direct_error));
  }

  void completion_loop() {
    while (!shutdown.load(std::memory_order_acquire) ||
           pending_count.load(std::memory_order_acquire) != 0) {
      CompletionDescriptor completion;
      if (!completions.try_pop(completion)) {
        std::this_thread::yield();
        continue;
      }
      report_direct_path_failure();
      std::shared_ptr<PendingQuery> pending;
      {
        std::lock_guard<std::mutex> lock(pending_mutex);
        const auto it = pending_queries.find(completion.request_id);
        if (it != pending_queries.end()) {
          pending = std::move(it->second);
          pending_queries.erase(it);
        }
      }
      if (!pending) {
        active_gpu_queries.fetch_sub(1, std::memory_order_release);
        maintenance_cv.notify_all();
        continue;
      }
      const auto completed_at = std::chrono::steady_clock::now();
      const u64 gpu_ns = completion.gpu_cycles * 1000000ULL / gpu_clock_khz;
      const auto phase_ns = [&](u64 cycles) {
        return cycles * 1000000ULL / gpu_clock_khz;
      };
      const u64 end_to_end_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          completed_at - pending->submitted_at).count());
      if (end_to_end_ns >= 10000000ULL &&
          slow_query_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
        std::cerr << "[gpu-search] slow query e2e_us=" << end_to_end_ns / 1000
                  << " gpu_us=" << gpu_ns / 1000
                  << " prepare_us=" << completion.prepare_cycles * 1000ULL / gpu_clock_khz
                  << " graph_us=" << completion.graph_cycles * 1000ULL / gpu_clock_khz
                  << " score_us=" << completion.score_cycles * 1000ULL / gpu_clock_khz
                  << " beam_us=" << completion.beam_cycles * 1000ULL / gpu_clock_khz
                  << " exact_us=" << completion.exact_cycles * 1000ULL / gpu_clock_khz
                  << " graph_reads=" << completion.remote_pages
                  << " graph_batches=" << completion.remote_batches
                  << " graph_hits=" << completion.cache_hits
                  << " exact_reads=" << completion.exact_vectors
                  << " exact_hits=" << completion.exact_cache_hits << '\n';
      }
      try {
        if (completion.status != 0) {
          const std::string message = "persistent GPU query failed with status " +
            std::to_string(completion.status);
          mark_unhealthy(message);
          throw std::runtime_error(message);
        }
        const size_t offset = static_cast<size_t>(pending->slot) * result_capacity;
        service::QueryResult result;
        result.reserve(completion.result_count);
        for (u32 index = 0; index < completion.result_count; ++index) {
          result.push_back({result_ids_host[offset + index],
                            result_distances_host[offset + index]});
        }
        pending->promise.set_value(std::move(result));
      } catch (...) {
        pending->promise.set_exception(std::current_exception());
      }
      {
        std::lock_guard<std::mutex> lock(slot_mutex);
        free_slots.push_back(pending->slot);
      }
      slot_cv.notify_one();
      pending_count.fetch_sub(1, std::memory_order_release);
      active_gpu_queries.fetch_sub(1, std::memory_order_release);
      maintenance_cv.notify_all();
      engine.telemetry_.queries_completed.fetch_add(1, std::memory_order_relaxed);
      engine.telemetry_.gpu_active_ns.fetch_add(gpu_ns, std::memory_order_relaxed);
      engine.telemetry_.gpu_prepare_ns.fetch_add(
        phase_ns(completion.prepare_cycles), std::memory_order_relaxed);
      engine.telemetry_.gpu_graph_ns.fetch_add(
        phase_ns(completion.graph_cycles), std::memory_order_relaxed);
      engine.telemetry_.gpu_score_ns.fetch_add(
        phase_ns(completion.score_cycles), std::memory_order_relaxed);
      engine.telemetry_.gpu_beam_ns.fetch_add(
        phase_ns(completion.beam_cycles), std::memory_order_relaxed);
      engine.telemetry_.gpu_exact_ns.fetch_add(
        phase_ns(completion.exact_cycles), std::memory_order_relaxed);
      engine.telemetry_.completion_wait_ns.fetch_add(end_to_end_ns,
                                                     std::memory_order_relaxed);
      if (completion.snapshot_epoch != 0) {
        engine.telemetry_.delta_queries.fetch_add(1, std::memory_order_relaxed);
      }
      engine.telemetry_.rdma_read_ops.fetch_add(
        static_cast<u64>(completion.exact_vectors) + completion.remote_pages,
        std::memory_order_relaxed);
      engine.telemetry_.rdma_read_bytes.fetch_add(
        static_cast<u64>(completion.exact_vectors) * node_record_bytes +
        static_cast<u64>(completion.remote_pages) * index.layout.graph_entry_bytes,
        std::memory_order_relaxed);
      if (completion.remote_pages > completion.remote_batches) {
        engine.telemetry_.rdma_merged_requests.fetch_add(
          completion.remote_pages - completion.remote_batches,
          std::memory_order_relaxed);
      }
      engine.telemetry_.exact_vector_reads.fetch_add(completion.exact_vectors,
                                                     std::memory_order_relaxed);
      engine.telemetry_.graph_page_requests.fetch_add(completion.remote_pages,
                                                      std::memory_order_relaxed);
      engine.telemetry_.graph_page_cache_hits.fetch_add(completion.cache_hits,
                                                        std::memory_order_relaxed);
      engine.telemetry_.exact_vector_cache_hits.fetch_add(completion.exact_cache_hits,
                                                          std::memory_order_relaxed);
    }
  }

  PersistentSearchEngine& engine;
  configuration::IndexConfiguration& config;
  format::View index;
  pq::Model pq_model;
  AnchorTable anchor_table;
  std::vector<u32> entry_handles;
  std::unordered_map<u64, u32> anchor_buckets_by_raw;
#ifdef DVSTOR_HAVE_GPUNETIO
  std::unique_ptr<gpu::GpuNetioPersistentTransport> direct_transport;
  gpu::GpuNetioPersistentView direct_view{};
#else
  struct EmptyDirectView {
    void** qp_array{};
    void* remote_regions{};
    u32 remote_region_count{};
    u32 qps_per_node{};
    u32 local_mkey{};
    u64 local_iova_base{};
    byte_t* data{};
    size_t data_bytes{};
    byte_t* dump{};
  } direct_view;
#endif
  MappedRing<QueryDescriptor> submissions;
  MappedRing<CompletionDescriptor> completions;
  MappedRing<DeltaPublishDescriptor> delta_submissions;
  MappedRing<DeltaPublishCompletion> delta_completions;
  u32 query_slots{};
  u32 result_capacity{};
  u32 exact_width{};
  u32 code_bytes{};
  u32 visited_capacity{};
  u32 node_record_bytes{};
  u32 delta_capacity{};
  u32 delta_table_capacity{};
  u32 graph_cache_sets{};
  u32 graph_cache_slots{};
  u32 exact_cache_sets{};
  u32 exact_cache_slots{};
  u32 exact_cache_stride{};
  u32 graph_invalidation_capacity{};
  u32 delta_command_capacity{};
  size_t graph_cache_bytes{};
  size_t exact_cache_bytes{};
  size_t exact_region_offset{};
  size_t exact_cache_offset{};
  size_t graph_cache_offset{};
  u64 explicit_gpu_bytes{};
  u64 gpu_clock_khz{1};
  DeviceShardRegion* d_shards{};
  byte_t* d_pq_codes{};
  f32* d_opq_matrix{};
  f32* d_pq_centroids{};
  u32* d_entry_points{};
  f32* d_anchor_vectors{};
  u32* d_anchor_handles{};
  f32* d_anchor_distances{};
  u32* d_delta_bucket_heads{};
  f32* d_queries{};
  f32* query_staging_host{};
  f32* d_transformed_queries{};
  f32* d_query_luts{};
  u32* d_visited{};
  byte_t* d_exact_records{};
  byte_t* d_exact_cache{};
  byte_t* d_remote_buffer{};
  byte_t* d_graph_cache{};
  u64* d_graph_cache_keys{};
  u64* d_graph_cache_generations{};
  u64* d_graph_cache_timestamps{};
  u32* d_graph_cache_states{};
  u32* d_graph_cache_readers{};
  u32* d_graph_cache_victims{};
  u64* d_graph_cache_generation{};
  u64* graph_invalidation_keys_host{};
  u64* d_graph_invalidation_keys{};
  u32* d_exact_cache_keys{};
  u32* d_exact_cache_states{};
  u32* d_exact_cache_readers{};
  u32* d_exact_cache_victims{};
  bool owns_remote_buffer{};
  u32* result_ids_host{};
  f32* result_distances_host{};
  u32* d_result_ids{};
  f32* d_result_distances{};
  DeviceDeltaRecord* d_delta_records{};
  byte_t* d_delta_vectors{};
  byte_t* d_delta_pq_codes{};
  f32* d_delta_encode_scratch{};
  DeviceDeltaRecord* delta_staging_records_host{};
  DeviceDeltaRecord* d_delta_staging_records{};
  byte_t* delta_staging_vectors_host{};
  byte_t* d_delta_staging_vectors{};
  u32* d_delta_next{};
  u32* d_base_override_keys{};
  u64* d_base_override_epochs{};
  u64* d_delta_remote_keys{};
  u32* d_delta_remote_slots{};
  DeltaSupersedeUpdate* delta_supersede_updates_host{};
  DeltaSupersedeUpdate* d_delta_supersede_updates{};
  DeltaOverrideUpdate* delta_override_updates_host{};
  DeltaOverrideUpdate* d_delta_override_updates{};
  u32* d_delta_count{};
  std::vector<DeviceDeltaRecord> delta_records_host;
  std::unordered_map<node_t, u32> latest_delta_slot;
  std::unordered_map<u32, u64> base_override_epochs;
  std::mutex delta_mutex;
  u32* stop_host{};
  u32* stop_device{};
  u32* direct_disabled_host{};
  u32* direct_disabled_device{};
  i32* direct_error_host{};
  i32* direct_error_device{};
  cudaStream_t kernel_stream{};
  cudaStream_t transfer_stream{};
  cudaStream_t delta_stream{};
  PersistentKernelParams kernel_params{};
  u32 kernel_blocks{};
  bool kernel_running{};
  std::atomic<bool> direct_failure_logged{false};
  std::atomic<u32> slow_query_logs{0};
  std::atomic<bool> accepting{true};
  std::atomic<bool> healthy{true};
  std::atomic<bool> shutdown{false};
  std::atomic<bool> maintenance_shutdown{false};
  std::atomic<u64> active_gpu_queries{0};
  std::atomic<u64> next_request_id{1};
  std::atomic<u64> next_delta_command_id{1};
  std::atomic<u64> pending_count{0};
  std::mutex admission_mutex;
  std::condition_variable admission_cv;
  std::deque<PendingSubmission> admission_queue;
  bool admission_paused{};
  std::string health_error;
  std::mutex slot_mutex;
  std::condition_variable slot_cv;
  std::vector<u32> free_slots;
  std::mutex pending_mutex;
  std::unordered_map<u64, std::shared_ptr<PendingQuery>> pending_queries;
  std::thread admission_thread;
  std::thread completion_thread;
  std::mutex compaction_mutex;
  std::mutex maintenance_mutex;
  std::condition_variable maintenance_cv;
  std::thread maintenance_thread;
};

PersistentSearchEngine::PersistentSearchEngine(
    configuration::IndexConfiguration& config,
    Context& channel_context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions)
    : delta_(1) {
  check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)),
             "cudaSetDevice(GPU navigation engine)");
  impl_ = std::make_unique<Impl>(*this, config, channel_context,
                                 connection_manager, remote_regions);
}

PersistentSearchEngine::~PersistentSearchEngine() = default;

service::QueryResult PersistentSearchEngine::search(
    VectorDType query_dtype, const byte_t* query_data, u32 k) {
  return impl_->search(query_dtype, query_data, k);
}

service::QueryResult PersistentSearchEngine::search(std::span<const element_t> query, u32 k) {
  return search(VectorDType::float32,
                reinterpret_cast<const byte_t*>(query.data()), k);
}

bool PersistentSearchEngine::publish_mutations(
    std::vector<DeltaMutation> mutations, u64 epoch,
    std::span<const u64> invalidated_graph_nodes) {
  std::lock_guard<std::mutex> publish_lock(mutation_publish_mutex_);
  if (mutations.empty() || epoch == 0) {
    throw std::invalid_argument("GPU mutation publication requires a non-empty epoch batch");
  }
  const size_t mutation_count = mutations.size();
  const auto publication_started = std::chrono::steady_clock::now();
  u64 publication_queue_ns = 0;
  for (const DeltaMutation& mutation : mutations) {
    if (mutation.enqueued_at == std::chrono::steady_clock::time_point{}) continue;
    publication_queue_ns += static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        publication_started - mutation.enqueued_at).count());
  }
  telemetry_.publication_queue_ns_total.fetch_add(publication_queue_ns,
                                                  std::memory_order_relaxed);
  try {
    impl_->upload_mutations(mutations, epoch, invalidated_graph_nodes);
  } catch (const std::exception& error) {
    impl_->mark_unhealthy(std::string{"GPU mutation publication failed: "} + error.what());
    throw;
  }
  const auto now = std::chrono::steady_clock::now();
  u64 visibility_ns_total = 0;
  u64 visibility_ns_max = 0;
  for (const DeltaMutation& mutation : mutations) {
    if (mutation.enqueued_at == std::chrono::steady_clock::time_point{}) continue;
    const u64 visibility_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        now - mutation.enqueued_at).count());
    visibility_ns_total += visibility_ns;
    visibility_ns_max = std::max(visibility_ns_max, visibility_ns);
  }
  try {
    if (!delta_.publish(std::move(mutations), epoch, now)) {
      impl_->mark_unhealthy("GPU mutation publication lost its coordinator epoch");
      return false;
    }
  } catch (const std::exception& error) {
    impl_->mark_unhealthy(std::string{"GPU epoch publication failed: "} + error.what());
    throw;
  }
  telemetry_.mutations_published.fetch_add(mutation_count, std::memory_order_relaxed);
  telemetry_.delta_publications.fetch_add(1, std::memory_order_relaxed);
  telemetry_.visibility_ns_total.fetch_add(visibility_ns_total,
                                           std::memory_order_relaxed);
  u64 current_max = telemetry_.visibility_ns_max.load(std::memory_order_relaxed);
  while (current_max < visibility_ns_max &&
         !telemetry_.visibility_ns_max.compare_exchange_weak(
           current_max, visibility_ns_max, std::memory_order_relaxed)) {}
  return true;
}

void PersistentSearchEngine::reset_telemetry() {
  telemetry_.reset();
  telemetry_.delta_live_entries.store(delta_.delta_size(), std::memory_order_relaxed);
}

}  // namespace gpu_search
