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
#include <cstddef>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <sstream>
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
constexpr u32 kDirectBatchQueueCapacity = 64;
constexpr u32 kCacheAdmissionWays = 4;
constexpr u32 kMaxCacheAdmissionSets = 1u << 18;
constexpr u32 kResidentRouteReady = 2;

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

  struct RetiredDeltaBatch {
    u64 query_ticket_barrier{};
    std::vector<u32> slots;
  };

  struct PendingStorageReclaimAck {
    u64 maintenance_sequence{};
    u64 query_ticket_barrier{};
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
    if (active_query_tickets != nullptr) {
      active_query_tickets[pending->slot].store(0, std::memory_order_release);
    }
    if (active_query_snapshots != nullptr) {
      active_query_snapshots[pending->slot].store(0, std::memory_order_release);
    }
    pending->promise.set_exception(
      std::make_exception_ptr(std::runtime_error(message)));
    {
      std::lock_guard<std::mutex> lock(slot_mutex);
      free_slots.push_back(pending->slot);
    }
    slot_cv.notify_one();
    pending_count.fetch_sub(1, std::memory_order_release);
    maintenance_cv.notify_all();
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
      for (const auto& pending : rejected) {
        if (active_query_tickets != nullptr) {
          active_query_tickets[pending->slot].store(0, std::memory_order_release);
        }
        if (active_query_snapshots != nullptr) {
          active_query_snapshots[pending->slot].store(0, std::memory_order_release);
        }
        free_slots.push_back(pending->slot);
      }
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
    maintenance_cv.notify_all();
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
        submissions(config.gpu_query_slots * 2,
                    MappedRing<QueryDescriptor>::Direction::host_to_device),
        completions(config.gpu_query_slots * 2,
                    MappedRing<CompletionDescriptor>::Direction::device_to_host),
        delta_submissions(8, MappedRing<DeltaPublishDescriptor>::Direction::host_to_device),
        delta_completions(8, MappedRing<DeltaPublishCompletion>::Direction::device_to_host) {
    bind_cuda_device("cudaSetDevice(GPU navigation construction)");
    compute_client_id = connection_manager.client_id;
    compute_client_count = connection_manager.num_total_clients;
    if (compute_client_count == 0 ||
        compute_client_count > format::kMaxComputeClients ||
        compute_client_id >= compute_client_count) {
      throw std::runtime_error("compute client identity exceeds storage reclaim capacity");
    }
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
      static_cast<u64>(std::min(config.gpu_graph_prefetch_depth,
                                kPersistentScoreChunk)) * config.R;
    if (max_merge_candidates > kPersistentMaxMergeCandidates) {
      throw std::invalid_argument("GPU navigation prefetch/degree exceeds parallel top-k capacity");
    }

    anchor_table = load_anchor_table(config.resolved_index_prefix(), config.dim,
                                     index.layout.num_shards, index);
    for (u32 anchor = 0; anchor < anchor_table.raw_pointers.size(); ++anchor) {
      anchor_buckets_by_raw.emplace(anchor_table.raw_pointers[anchor], anchor);
      anchor_graph_keys_host.push_back(
        graph_cache_key(anchor_table.raw_pointers[anchor]));
    }
    std::sort(anchor_graph_keys_host.begin(), anchor_graph_keys_host.end());
    anchor_graph_keys_host.erase(
      std::unique(anchor_graph_keys_host.begin(), anchor_graph_keys_host.end()),
      anchor_graph_keys_host.end());
    if (anchor_graph_keys_host.size() > std::numeric_limits<u32>::max()) {
      throw std::runtime_error("GPU anchor route table exceeds uint32 capacity");
    }
    entry_handles = index.entry_points;
    std::cerr << "[gpu-search] query routing="
              << (anchor_table.count() == 0 ? "static entry points" : "query-aware GPU anchors")
              << " anchors=" << anchor_table.count()
              << " seeds=" << config.gpu_entry_seed_count << '\n';
    query_slots = config.gpu_query_slots;
    result_capacity = std::max<u32>(config.k, config.gpu_final_rerank_width);
    exact_width = kPersistentMaxExact;
    code_bytes = index.layout.code_bytes;
    free_slots.resize(query_slots);
    for (u32 slot = 0; slot < query_slots; ++slot) free_slots[slot] = slot;
    active_query_tickets = std::make_unique<std::atomic<u64>[]>(query_slots);
    active_query_snapshots = std::make_unique<std::atomic<u64>[]>(query_slots);
    for (u32 slot = 0; slot < query_slots; ++slot) {
      active_query_tickets[slot].store(0, std::memory_order_relaxed);
      active_query_snapshots[slot].store(0, std::memory_order_relaxed);
    }

    node_record_bytes = static_cast<u32>(VamanaNode::size_until_vector_end());
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
    permanent_override_words = static_cast<u32>((index.layout.num_nodes + 31) / 32);
    visited_capacity = budget.visited_capacity;
    graph_cache_sets = budget.cache_sets;
    graph_cache_slots = budget.cache_slots;
    graph_cache_bytes = static_cast<size_t>(budget.cache_payload_bytes);
    exact_cache_sets = budget.exact_cache_sets;
    exact_cache_slots = budget.exact_cache_slots;
    exact_cache_stride = budget.exact_cache_stride;
    exact_cache_bytes = static_cast<size_t>(budget.exact_cache_payload_bytes);
    graph_admission_sets = std::min(graph_cache_sets, kMaxCacheAdmissionSets);
    exact_admission_sets = std::min(exact_cache_sets, kMaxCacheAdmissionSets);
    const u64 invalidation_capacity = static_cast<u64>(
      std::max(config.storage_owner_batch_max, config.gpu_query_slots)) * config.R;
    if (invalidation_capacity > std::numeric_limits<u32>::max()) {
      throw std::runtime_error("GPU navigation graph invalidation capacity exceeds uint32");
    }
    graph_invalidation_capacity = static_cast<u32>(std::max<u64>(1, invalidation_capacity));
    const u64 dynamic_code_scratch_bytes =
      static_cast<u64>(query_slots) * kPersistentMaxMergeCandidates * code_bytes;
    const u64 dynamic_request_scratch_bytes =
      static_cast<u64>(query_slots) * kPersistentMaxMergeCandidates *
      (sizeof(u32) + 2 * sizeof(u64));
    const u64 navigation_candidate_bytes =
      static_cast<u64>(query_slots) * kPersistentMaxMergeCandidates *
      (sizeof(u32) + sizeof(f32));
    const u64 estimated_direct_queue_count =
      static_cast<u64>(config.gpu_rdma_qps) * index.shards.size();
    const u64 direct_queue_bytes = estimated_direct_queue_count *
      (2 * sizeof(u64) + sizeof(DeviceRingView<DirectBatchDescriptor>) +
       static_cast<u64>(kDirectBatchQueueCapacity) *
         (sizeof(u64) + sizeof(DirectBatchDescriptor))) +
      static_cast<u64>(query_slots) * index.shards.size() * sizeof(i32);
    const u64 graph_scratch_bytes = static_cast<u64>(query_slots) *
      kPersistentMaxPrefetch * kPersistentGraphCacheLineBytes;
    const u64 cache_admission_bytes =
      static_cast<u64>(graph_admission_sets) *
        (kCacheAdmissionWays * sizeof(u64) + sizeof(u32)) +
      static_cast<u64>(exact_admission_sets) *
        (kCacheAdmissionWays * sizeof(u32) + sizeof(u32));
    const u64 route_graph_record_bytes =
      static_cast<u64>(anchor_graph_keys_host.size()) *
      index.layout.graph_entry_bytes;
    const u64 route_graph_metadata_bytes =
      static_cast<u64>(anchor_graph_keys_host.size()) *
      (sizeof(u64) + 2 * sizeof(u32));
    route_graph_bytes = route_graph_record_bytes + route_graph_metadata_bytes;
    const u64 additional_scratch_bytes =
      dynamic_code_scratch_bytes + dynamic_request_scratch_bytes +
      navigation_candidate_bytes + direct_queue_bytes + graph_scratch_bytes +
      cache_admission_bytes + route_graph_bytes;
    if (additional_scratch_bytes > usable_budget - budget.explicit_bytes) {
      throw std::runtime_error(
        "GPU navigation dynamic-code scratch exceeds the configured memory budget");
    }
    explicit_gpu_bytes = budget.explicit_bytes + additional_scratch_bytes;
    engine.telemetry_.gpu_memory_explicit_bytes.store(
      explicit_gpu_bytes, std::memory_order_relaxed);
    engine.telemetry_.gpu_memory_base_pq_bytes.store(
      budget.code_bytes, std::memory_order_relaxed);
    engine.telemetry_.gpu_memory_route_graph_bytes.store(
      route_graph_bytes, std::memory_order_relaxed);
    engine.telemetry_.gpu_memory_delta_reserved_bytes.store(
      budget.delta_bytes, std::memory_order_relaxed);
    engine.telemetry_.gpu_memory_graph_cache_bytes.store(
      budget.cache_total_bytes, std::memory_order_relaxed);
    engine.telemetry_.gpu_memory_exact_cache_bytes.store(
      budget.exact_cache_total_bytes, std::memory_order_relaxed);
    const u64 base_code_region_bytes = budget.code_bytes;
    const u64 exact_bytes = budget.exact_bytes;
    std::cerr << "[gpu-search] navigation budget codes=" << budget.code_bytes
              << " delta=" << budget.delta_bytes
              << " delta_capacity=" << budget.delta_capacity
              << " delta_codes=" << budget.delta_code_bytes
              << " permanent_overrides=" << budget.permanent_override_bytes
              << " adjacency_total=" << budget.cache_total_bytes
              << " exact_cache_total=" << budget.exact_cache_total_bytes
              << " dynamic_code_scratch=" << dynamic_code_scratch_bytes
              << " dynamic_request_scratch=" << dynamic_request_scratch_bytes
              << " navigation_candidates=" << navigation_candidate_bytes
              << " direct_queue_scratch=" << direct_queue_bytes
              << " graph_scratch=" << graph_scratch_bytes
              << " cache_admission=" << cache_admission_bytes
              << " anchor_route=" << route_graph_bytes
              << " explicit=" << explicit_gpu_bytes
              << " limit=" << engine_budget << " bytes\n";

    const size_t code_region_bytes = static_cast<size_t>(base_code_region_bytes);
    anchor_graph_region_offset = static_cast<size_t>(
      align_up(code_region_bytes, 512));
    dynamic_code_region_offset = static_cast<size_t>(align_up(
      anchor_graph_region_offset + route_graph_record_bytes, 256));
    exact_region_offset = static_cast<size_t>(align_up(
      dynamic_code_region_offset + dynamic_code_scratch_bytes, 256));
    graph_scratch_offset = static_cast<size_t>(align_up(
      exact_region_offset + exact_bytes, 512));
    exact_cache_offset = static_cast<size_t>(align_up(
      graph_scratch_offset + graph_scratch_bytes, 256));
    graph_cache_offset = static_cast<size_t>(
      align_up(exact_cache_offset + exact_cache_bytes, 512));
    control_region_offset = static_cast<size_t>(
      align_up(graph_cache_offset + graph_cache_bytes, 256));
    const size_t control_region_bytes =
      index.shards.size() * sizeof(format::StorageControlBlock);
    const size_t remote_buffer_bytes = control_region_offset + control_region_bytes;
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
    d_anchor_graph_records = d_remote_buffer + anchor_graph_region_offset;
    d_dynamic_code_records = d_remote_buffer + dynamic_code_region_offset;
    d_exact_records = d_remote_buffer + exact_region_offset;
    d_graph_scratch = d_remote_buffer + graph_scratch_offset;
    d_exact_cache = d_remote_buffer + exact_cache_offset;
    d_graph_cache = d_remote_buffer + graph_cache_offset;
    d_control_snapshots = reinterpret_cast<format::StorageControlBlock*>(
      d_remote_buffer + control_region_offset);

    control_bootstrapper = std::make_unique<NavigationBootstrapper>(
      config, channel_context, connection_manager, remote_regions,
      d_remote_buffer, remote_buffer_bytes);
    std::cerr << "[gpu-search] bootstrap=CPU-posted GPUDirect RDMA; "
                 "queries=strict GPU-initiated GPUNetIO\n";
    initialize_storage_reclaim_ack();
    stream_codes_to_gpu(*control_bootstrapper);
    stream_anchor_graph_to_gpu(*control_bootstrapper);

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
    const u32 anchor_graph_count =
      static_cast<u32>(anchor_graph_keys_host.size());
    device_allocate(d_anchor_graph_keys, anchor_graph_count,
                    "cudaMalloc(GPU anchor route keys)");
    device_allocate(d_anchor_graph_states, anchor_graph_count,
                    "cudaMalloc(GPU anchor route states)");
    device_allocate(d_anchor_graph_readers, anchor_graph_count,
                    "cudaMalloc(GPU anchor route readers)");
    anchor_graph_ready_states_host.assign(anchor_graph_count,
                                          kResidentRouteReady);
    if (anchor_graph_count != 0) {
      check_cuda(cudaMemcpy(d_anchor_graph_keys, anchor_graph_keys_host.data(),
                            anchor_graph_keys_host.size() * sizeof(u64),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(GPU anchor route keys)");
      check_cuda(cudaMemcpy(d_anchor_graph_states,
                            anchor_graph_ready_states_host.data(),
                            anchor_graph_ready_states_host.size() * sizeof(u32),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(GPU anchor route states)");
      check_cuda(cudaMemset(d_anchor_graph_readers, 0,
                            anchor_graph_keys_host.size() * sizeof(u32)),
                 "cudaMemset(GPU anchor route readers)");
      check_cuda(cudaHostAlloc(
                   reinterpret_cast<void**>(&anchor_graph_readers_host),
                   anchor_graph_keys_host.size() * sizeof(u32),
                   cudaHostAllocPortable),
                 "cudaHostAlloc(GPU anchor route reader snapshot)");
      check_cuda(cudaHostAlloc(
                   reinterpret_cast<void**>(&anchor_graph_validation_host),
                   index.layout.graph_entry_bytes,
                   cudaHostAllocPortable),
                 "cudaHostAlloc(GPU anchor route validation record)");
    }
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
      device_allocate(d_anchor_pq_codes,
                      static_cast<size_t>(anchor_table.count()) * code_bytes,
                      "cudaMalloc(GPU navigation anchor PQ codes)");
      launch_gather_anchor_codes(nullptr, d_pq_codes, d_anchor_handles,
                                 d_anchor_pq_codes, anchor_table.count(), code_bytes,
                                 static_cast<u32>(index.layout.num_nodes));
      check_cuda(cudaGetLastError(), "launch_gather_anchor_codes");
      check_cuda(cudaStreamSynchronize(nullptr),
                 "cudaStreamSynchronize(GPU navigation anchor PQ codes)");
      device_allocate(d_delta_bucket_heads, anchor_table.count(),
                      "cudaMalloc(GPU navigation delta buckets)");
      check_cuda(cudaMemset(d_delta_bucket_heads, 0xff,
                            static_cast<size_t>(anchor_table.count()) * sizeof(u32)),
                 "cudaMemset(GPU navigation delta buckets)");
    }

    query_input_stride = static_cast<size_t>(config.dim) * sizeof(f32);
    device_allocate(d_queries, static_cast<size_t>(query_slots) * config.dim,
                    "cudaMalloc(GPU decoded queries)");
    mapped_host_allocate(query_input_host, d_query_input,
                         static_cast<size_t>(query_slots) * query_input_stride,
                         "cudaHostAlloc(GPU navigation query input)");
    device_allocate(d_transformed_queries, static_cast<size_t>(query_slots) * config.dim,
                    "cudaMalloc(GPU transformed queries)");
    device_allocate(d_query_luts,
                    static_cast<size_t>(query_slots) * pq_model.subquantizers * 256,
                    "cudaMalloc(GPU PQ query LUTs)");
    device_allocate(d_navigation_candidate_handles,
                    static_cast<size_t>(query_slots) * kPersistentMaxMergeCandidates,
                    "cudaMalloc(GPU navigation candidate handles)");
    device_allocate(d_navigation_candidate_distances,
                    static_cast<size_t>(query_slots) * kPersistentMaxMergeCandidates,
                    "cudaMalloc(GPU navigation candidate distances)");
    device_allocate(d_visited, static_cast<size_t>(query_slots) * visited_capacity,
                    "cudaMalloc(GPU navigation visited)");
    const size_t dynamic_request_elements =
      static_cast<size_t>(query_slots) * kPersistentMaxMergeCandidates;
    device_allocate(d_dynamic_code_request_shards, dynamic_request_elements,
                    "cudaMalloc(dynamic PQ request shards)");
    device_allocate(d_dynamic_code_request_offsets, dynamic_request_elements,
                    "cudaMalloc(dynamic PQ request offsets)");
    device_allocate(d_dynamic_code_request_local_iovas, dynamic_request_elements,
                    "cudaMalloc(dynamic PQ request local IOVAs)");

    direct_batch_queue_count = direct_view.qps_per_node * direct_view.remote_region_count;
    if (direct_batch_queue_count == 0 ||
        direct_batch_queue_count != estimated_direct_queue_count) {
      throw std::runtime_error("GPUNetIO QP count does not match the GPU owner queues");
    }
    const size_t direct_queue_slots =
      static_cast<size_t>(direct_batch_queue_count) * kDirectBatchQueueCapacity;
    device_allocate(d_direct_batch_enqueue, direct_batch_queue_count,
                    "cudaMalloc(GPUNetIO owner enqueue positions)");
    device_allocate(d_direct_batch_dequeue, direct_batch_queue_count,
                    "cudaMalloc(GPUNetIO owner dequeue positions)");
    device_allocate(d_direct_batch_sequences, direct_queue_slots,
                    "cudaMalloc(GPUNetIO owner queue sequences)");
    device_allocate(d_direct_batch_entries, direct_queue_slots,
                    "cudaMalloc(GPUNetIO owner queue entries)");
    device_allocate(d_direct_batch_queues, direct_batch_queue_count,
                    "cudaMalloc(GPUNetIO owner queue views)");
    device_allocate(d_direct_batch_statuses,
                    static_cast<size_t>(query_slots) * index.shards.size(),
                    "cudaMalloc(GPUNetIO owner completion statuses)");
    check_cuda(cudaMemset(d_direct_batch_enqueue, 0,
                          static_cast<size_t>(direct_batch_queue_count) * sizeof(u64)),
               "cudaMemset(GPUNetIO owner enqueue positions)");
    check_cuda(cudaMemset(d_direct_batch_dequeue, 0,
                          static_cast<size_t>(direct_batch_queue_count) * sizeof(u64)),
               "cudaMemset(GPUNetIO owner dequeue positions)");
    std::vector<u64> direct_sequences(direct_queue_slots);
    std::vector<DeviceRingView<DirectBatchDescriptor>> direct_queues(
      direct_batch_queue_count);
    for (u32 queue = 0; queue < direct_batch_queue_count; ++queue) {
      const size_t queue_base = static_cast<size_t>(queue) * kDirectBatchQueueCapacity;
      for (u32 slot = 0; slot < kDirectBatchQueueCapacity; ++slot) {
        direct_sequences[queue_base + slot] = slot;
      }
      direct_queues[queue] = {
        .enqueue_position = reinterpret_cast<unsigned long long*>(
          d_direct_batch_enqueue + queue),
        .dequeue_position = reinterpret_cast<unsigned long long*>(
          d_direct_batch_dequeue + queue),
        .sequences = reinterpret_cast<unsigned long long*>(
          d_direct_batch_sequences + queue_base),
        .entries = d_direct_batch_entries + queue_base,
        .capacity = kDirectBatchQueueCapacity,
        .mask = kDirectBatchQueueCapacity - 1,
      };
    }
    check_cuda(cudaMemcpy(d_direct_batch_sequences, direct_sequences.data(),
                          direct_sequences.size() * sizeof(u64), cudaMemcpyHostToDevice),
               "cudaMemcpy(GPUNetIO owner queue sequences)");
    check_cuda(cudaMemcpy(d_direct_batch_queues, direct_queues.data(),
                          direct_queues.size() *
                            sizeof(DeviceRingView<DirectBatchDescriptor>),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(GPUNetIO owner queue views)");

    device_allocate(d_graph_cache_keys, graph_cache_slots, "cudaMalloc(navigation cache keys)");
    device_allocate(d_graph_cache_generations, graph_cache_slots,
                    "cudaMalloc(navigation cache generations)");
    device_allocate(d_graph_cache_timestamps, graph_cache_slots,
                    "cudaMalloc(navigation cache timestamps)");
    device_allocate(d_graph_cache_states, graph_cache_slots, "cudaMalloc(navigation cache states)");
    device_allocate(d_graph_cache_readers, graph_cache_slots, "cudaMalloc(navigation cache readers)");
    device_allocate(d_graph_cache_victims, graph_cache_sets, "cudaMalloc(navigation cache victims)");
    device_allocate(d_graph_admission_keys,
                    static_cast<size_t>(graph_admission_sets) * kCacheAdmissionWays,
                    "cudaMalloc(navigation admission keys)");
    device_allocate(d_graph_admission_victims, graph_admission_sets,
                    "cudaMalloc(navigation admission victims)");
    device_allocate(d_graph_cache_generation, 1, "cudaMalloc(navigation cache generation)");
    delta_command_capacity = std::max({1u, config.storage_owner_batch_max,
                                       config.gpu_query_slots});
    mapped_host_allocate(graph_invalidation_keys_host, d_graph_invalidation_keys,
                         graph_invalidation_capacity,
                         "cudaHostAlloc(navigation graph invalidation staging)");
    mapped_host_allocate(delta_supersede_updates_host, d_delta_supersede_updates,
                         delta_command_capacity,
                         "cudaHostAlloc(navigation delta supersede staging)");
    mapped_host_allocate(delta_override_updates_host, d_delta_override_updates,
                         delta_command_capacity,
                         "cudaHostAlloc(navigation delta override staging)");
    mapped_host_allocate(delta_durable_updates_host, d_delta_durable_updates,
                         delta_command_capacity,
                         "cudaHostAlloc(navigation delta durable staging)");
    if (graph_cache_slots != 0) {
      check_cuda(cudaMemset(d_graph_cache_states, 0,
                            static_cast<size_t>(graph_cache_slots) * sizeof(u32)),
                 "cudaMemset(navigation cache states)");
      check_cuda(cudaMemset(d_graph_cache_readers, 0,
                            static_cast<size_t>(graph_cache_slots) * sizeof(u32)),
                 "cudaMemset(navigation cache readers)");
      check_cuda(cudaMemset(d_graph_cache_victims, 0,
                            static_cast<size_t>(graph_cache_sets) * sizeof(u32)),
                 "cudaMemset(navigation cache victims)");
      check_cuda(cudaMemset(d_graph_admission_keys, 0xff,
                            static_cast<size_t>(graph_admission_sets) *
                              kCacheAdmissionWays * sizeof(u64)),
                 "cudaMemset(navigation admission keys)");
      check_cuda(cudaMemset(d_graph_admission_victims, 0,
                            static_cast<size_t>(graph_admission_sets) * sizeof(u32)),
                 "cudaMemset(navigation admission victims)");
    }
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
    device_allocate(d_exact_admission_keys,
                    static_cast<size_t>(exact_admission_sets) * kCacheAdmissionWays,
                    "cudaMalloc(navigation exact admission keys)");
    device_allocate(d_exact_admission_victims, exact_admission_sets,
                    "cudaMalloc(navigation exact admission victims)");
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
      check_cuda(cudaMemset(d_exact_admission_keys, 0xff,
                            static_cast<size_t>(exact_admission_sets) *
                              kCacheAdmissionWays * sizeof(u32)),
                 "cudaMemset(navigation exact admission keys)");
      check_cuda(cudaMemset(d_exact_admission_victims, 0,
                            static_cast<size_t>(exact_admission_sets) * sizeof(u32)),
                 "cudaMemset(navigation exact admission victims)");
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
    mapped_host_allocate(delta_staging_slots_host, d_delta_staging_slots,
                         delta_command_capacity,
                         "cudaHostAlloc(navigation delta slot staging)");
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
    device_allocate(d_delta_prev, delta_capacity,
                    "cudaMalloc(navigation delta reverse links)");
    device_allocate(d_delta_remote_positions, delta_capacity,
                    "cudaMalloc(navigation delta remote positions)");
    device_allocate(d_base_override_keys, delta_table_capacity,
                    "cudaMalloc(navigation override keys)");
    device_allocate(d_base_override_epochs, delta_table_capacity,
                    "cudaMalloc(navigation override epochs)");
    device_allocate(d_permanent_override_bits, permanent_override_words,
                    "cudaMalloc(navigation permanent override bits)");
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
    check_cuda(cudaStreamCreateWithFlags(&delta_stream, cudaStreamNonBlocking),
               "cudaStreamCreate(GPU navigation delta)");
    check_cuda(cudaStreamCreateWithFlags(&rdma_stream, cudaStreamNonBlocking),
               "cudaStreamCreate(GPU navigation RDMA owners)");
    check_cuda(cudaStreamCreateWithFlags(&route_refresh_stream,
                                         cudaStreamNonBlocking),
               "cudaStreamCreate(GPU anchor route refresh)");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, static_cast<int>(config.gpu_device)),
               "cudaGetDeviceProperties(GPU navigation)");
    gpu_clock_khz = static_cast<u64>(std::max(1, properties.clockRate));
    const u64 requested_blocks = static_cast<u64>(
      std::max(1, properties.multiProcessorCount)) * config.gpu_persistent_blocks_per_sm;
    const u64 useful_blocks = std::max<u64>(1, config.num_threads);
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
      .node_meta_offset = 0,
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
      .direct_batch_queues = d_direct_batch_queues,
      .direct_batch_statuses = d_direct_batch_statuses,
      .direct_batch_queue_count = direct_batch_queue_count,
      .direct_dump = direct_view.dump,
      .direct_disabled = direct_disabled_device,
      .direct_error = direct_error_device,
      .delta_records = d_delta_records,
      .delta_vectors = d_delta_vectors,
      .delta_pq_codes = d_delta_pq_codes,
      .delta_staging_slots = d_delta_staging_slots,
      .delta_staging_records = d_delta_staging_records,
      .delta_staging_vectors = d_delta_staging_vectors,
      .delta_encode_scratch = d_delta_encode_scratch,
      .delta_next = d_delta_next,
      .delta_prev = d_delta_prev,
      .delta_remote_positions = d_delta_remote_positions,
      .delta_bucket_heads = d_delta_bucket_heads,
      .delta_count = d_delta_count,
      .delta_capacity = delta_capacity,
      .base_override_keys = d_base_override_keys,
      .base_override_epochs = d_base_override_epochs,
      .base_override_capacity = delta_table_capacity,
      .permanent_override_bits = d_permanent_override_bits,
      .permanent_override_words = permanent_override_words,
      .delta_remote_keys = d_delta_remote_keys,
      .delta_remote_slots = d_delta_remote_slots,
      .delta_remote_capacity = delta_table_capacity,
      .delta_supersede_updates = d_delta_supersede_updates,
      .delta_override_updates = d_delta_override_updates,
      .delta_durable_updates = d_delta_durable_updates,
      .graph_invalidation_keys = d_graph_invalidation_keys,
      .anchor_vectors = d_anchor_vectors,
      .anchor_handles = d_anchor_handles,
      .anchor_pq_codes = d_anchor_pq_codes,
      .anchor_graph_keys = d_anchor_graph_keys,
      .anchor_graph_records = d_anchor_graph_records,
      .anchor_graph_states = d_anchor_graph_states,
      .anchor_graph_readers = d_anchor_graph_readers,
      .anchor_graph_count = anchor_graph_count,
      .anchor_count = anchor_table.count(),
      .delta_anchor_probes = config.gpu_delta_anchor_probes,
      .stop = stop_device,
      .graph_cache = d_graph_cache,
      .graph_scratch = d_graph_scratch,
      .graph_cache_keys = d_graph_cache_keys,
      .graph_cache_generations = d_graph_cache_generations,
      .graph_cache_timestamps = d_graph_cache_timestamps,
      .graph_cache_states = d_graph_cache_states,
      .graph_cache_readers = d_graph_cache_readers,
      .graph_cache_victims = d_graph_cache_victims,
      .graph_admission_keys = d_graph_admission_keys,
      .graph_admission_victims = d_graph_admission_victims,
      .graph_admission_sets = graph_admission_sets,
      .graph_cache_generation = d_graph_cache_generation,
      .graph_cache_sets = graph_cache_sets,
      .graph_cache_ways = config.gpu_adjacency_cache_ways,
      .graph_cache_ttl_ns = static_cast<u64>(
        config.gpu_graph_cache_ttl_us == 0
          ? config.update_visibility_us
          : std::min(config.gpu_graph_cache_ttl_us,
                     config.update_visibility_us)) * 1000,
      .decoded_queries = d_queries,
      .transformed_queries = d_transformed_queries,
      .query_luts = d_query_luts,
      .navigation_candidate_handles = d_navigation_candidate_handles,
      .navigation_candidate_distances = d_navigation_candidate_distances,
      .visited_hash = d_visited,
      .exact_records = d_exact_records,
      .dynamic_code_records = d_dynamic_code_records,
      .dynamic_code_request_shards = d_dynamic_code_request_shards,
      .dynamic_code_request_offsets = d_dynamic_code_request_offsets,
      .dynamic_code_request_local_iovas = d_dynamic_code_request_local_iovas,
      .exact_cache = d_exact_cache,
      .exact_cache_stride = exact_cache_stride,
      .exact_cache_sets = exact_cache_sets,
      .exact_cache_ways = config.gpu_exact_cache_ways,
      .exact_cache_keys = d_exact_cache_keys,
      .exact_cache_states = d_exact_cache_states,
      .exact_cache_readers = d_exact_cache_readers,
      .exact_cache_victims = d_exact_cache_victims,
      .exact_admission_keys = d_exact_admission_keys,
      .exact_admission_victims = d_exact_admission_victims,
      .exact_admission_sets = exact_admission_sets,
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

  void stream_anchor_graph_to_gpu(NavigationBootstrapper& source) {
    if (anchor_graph_keys_host.empty()) {
      std::cerr << "[gpu-search] anchor route graph disabled: no anchors\n";
      return;
    }
    constexpr size_t kBootstrapBatch = 4096;
    std::vector<NavigationRead> requests;
    std::vector<i32> statuses;
    requests.reserve(kBootstrapBatch);
    for (size_t begin = 0; begin < anchor_graph_keys_host.size();
         begin += kBootstrapBatch) {
      const size_t end = std::min(begin + kBootstrapBatch,
                                  anchor_graph_keys_host.size());
      requests.clear();
      for (size_t slot = begin; slot < end; ++slot) {
        const u64 key = anchor_graph_keys_host[slot];
        const u32 shard = static_cast<u32>(key >> 48);
        if (shard >= index.shards.size()) {
          throw std::runtime_error("anchor route graph key has an invalid shard");
        }
        requests.push_back(NavigationRead{
          .remote_offset = (key << 16) >> 16,
          .destination_address = reinterpret_cast<u64>(
            d_anchor_graph_records +
            slot * index.layout.graph_entry_bytes),
          .bytes = index.layout.graph_entry_bytes,
          .memory_node = static_cast<u16>(shard),
        });
      }
      statuses.assign(requests.size(), -EIO);
      source.read(requests, statuses);
      for (size_t request = 0; request < statuses.size(); ++request) {
        if (statuses[request] <= 0) {
          throw std::runtime_error(
            "anchor route graph bootstrap failed: slot=" +
            std::to_string(begin + request) + " status=" +
            std::to_string(statuses[request]));
        }
      }
    }

    const size_t audit_count = std::min<size_t>(15, anchor_graph_keys_host.size());
    std::vector<byte_t> record(index.layout.graph_entry_bytes);
    for (size_t audit = 0; audit < audit_count; ++audit) {
      const size_t slot = audit_count == 1 ? 0 :
        audit * (anchor_graph_keys_host.size() - 1) / (audit_count - 1);
      check_cuda(cudaMemcpy(
                   record.data(),
                   d_anchor_graph_records + slot * index.layout.graph_entry_bytes,
                   record.size(), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(anchor route graph audit)");
      const u16 expected = vamana::hot_graph::load_u16_le(record.data() + 2);
      const u16 actual = vamana::hot_graph::checksum16(record.data(), record.size());
      if (record[0] > index.layout.graph_degree || expected != actual) {
        throw std::runtime_error(
          "anchor route graph audit failed at slot " + std::to_string(slot));
      }
    }
    std::cerr << "[gpu-search] resident anchor route graph records="
              << anchor_graph_keys_host.size() << " bytes="
              << anchor_graph_keys_host.size() * index.layout.graph_entry_bytes
              << " audit=" << audit_count << '\n';
  }

  void clear_delta_device_state(cudaStream_t stream = nullptr) {
    bind_cuda_device("cudaSetDevice(GPU navigation delta reset)");
    check_cuda(cudaMemsetAsync(d_delta_records, 0,
                               static_cast<size_t>(delta_capacity) * sizeof(DeviceDeltaRecord),
                               stream),
               "cudaMemset(navigation delta records)");
    check_cuda(cudaMemsetAsync(d_delta_next, 0xff,
                               static_cast<size_t>(delta_capacity) * sizeof(u32), stream),
               "cudaMemset(navigation delta links)");
    check_cuda(cudaMemsetAsync(d_delta_prev, 0xff,
                               static_cast<size_t>(delta_capacity) * sizeof(u32), stream),
               "cudaMemset(navigation delta reverse links)");
    check_cuda(cudaMemsetAsync(d_delta_remote_positions, 0xff,
                               static_cast<size_t>(delta_capacity) * sizeof(u32), stream),
               "cudaMemset(navigation delta remote positions)");
    check_cuda(cudaMemsetAsync(d_base_override_keys, 0xff,
                               static_cast<size_t>(delta_table_capacity) * sizeof(u32), stream),
               "cudaMemset(navigation override keys)");
    check_cuda(cudaMemsetAsync(d_base_override_epochs, 0,
                               static_cast<size_t>(delta_table_capacity) * sizeof(u64), stream),
               "cudaMemset(navigation override epochs)");
    check_cuda(cudaMemsetAsync(d_permanent_override_bits, 0,
                               static_cast<size_t>(permanent_override_words) * sizeof(u32),
                               stream),
               "cudaMemset(navigation permanent override bits)");
    check_cuda(cudaMemsetAsync(d_delta_remote_keys, 0,
                               static_cast<size_t>(delta_table_capacity) * sizeof(u64), stream),
               "cudaMemset(navigation remote keys)");
    check_cuda(cudaMemsetAsync(d_delta_remote_slots, 0xff,
                               static_cast<size_t>(delta_table_capacity) * sizeof(u32), stream),
               "cudaMemset(navigation remote slots)");
    check_cuda(cudaMemsetAsync(d_delta_count, 0, sizeof(u32), stream),
               "cudaMemset(navigation delta count)");
    if (d_delta_bucket_heads != nullptr) {
      check_cuda(cudaMemsetAsync(d_delta_bucket_heads, 0xff,
                                 static_cast<size_t>(anchor_table.count()) * sizeof(u32),
                                 stream),
                 "cudaMemset(navigation delta buckets)");
    }
    check_cuda(cudaStreamSynchronize(stream),
               "cudaStreamSynchronize(navigation delta reset)");
  }

  void start_persistent_kernel() {
    bind_cuda_device("cudaSetDevice(GPU navigation kernel start)");
    std::atomic_ref<u32>(*stop_host).store(0, std::memory_order_release);
    (void)cudaGetLastError();
    PersistentKernelParams control_params = kernel_params;
    control_params.submissions = {};
    control_params.completions = {};
    launch_persistent_search(delta_stream, control_params, 1, 256);
    check_cuda(cudaGetLastError(), "launch_persistent_search(delta control)");
    launch_direct_read_owners(rdma_stream, kernel_params,
                              direct_batch_queue_count, 256);
    check_cuda(cudaGetLastError(), "launch_direct_read_owners(navigation)");
    PersistentKernelParams query_params = kernel_params;
    query_params.delta_submissions = {};
    query_params.delta_completions = {};
    launch_persistent_search(kernel_stream, query_params, kernel_blocks,
                             kPersistentQueryThreads);
    check_cuda(cudaGetLastError(), "launch_persistent_search(navigation)");
    kernel_running = true;
    std::cerr << "[gpu-search] persistent CTAs=" << kernel_blocks
              << "+1-control QP-owner-warps=" << direct_batch_queue_count
              << " threads/CTA=" << kPersistentQueryThreads
              << " query_slots=" << query_slots << '\n';
  }

  void stop_persistent_kernel() {
    if (!kernel_running) return;
    bind_cuda_device("cudaSetDevice(GPU navigation kernel stop)");
    std::atomic_ref<u32>(*stop_host).store(1, std::memory_order_release);
    const cudaError_t query_status = cudaStreamSynchronize(kernel_stream);
    const cudaError_t control_status = cudaStreamSynchronize(delta_stream);
    const cudaError_t rdma_status = cudaStreamSynchronize(rdma_stream);
    kernel_running = false;
    check_cuda(query_status, "cudaStreamSynchronize(GPU navigation stop)");
    check_cuda(control_status, "cudaStreamSynchronize(GPU delta control stop)");
    check_cuda(rdma_status, "cudaStreamSynchronize(GPU RDMA owner stop)");
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
      if (rdma_stream != nullptr) cudaStreamSynchronize(rdma_stream);
      kernel_running = false;
    }
    reject_all_pending("persistent GPU query engine stopped before completion");
    if (completion_thread.joinable()) completion_thread.join();
    if (route_refresh_stream != nullptr) cudaStreamDestroy(route_refresh_stream);
    if (rdma_stream != nullptr) cudaStreamDestroy(rdma_stream);
    if (delta_stream != nullptr) cudaStreamDestroy(delta_stream);
    if (kernel_stream != nullptr) cudaStreamDestroy(kernel_stream);
    if (direct_disabled_host != nullptr) cudaFreeHost(direct_disabled_host);
    if (direct_error_host != nullptr) cudaFreeHost(direct_error_host);
    if (stop_host != nullptr) cudaFreeHost(stop_host);
    if (result_distances_host != nullptr) cudaFreeHost(result_distances_host);
    if (result_ids_host != nullptr) cudaFreeHost(result_ids_host);
    if (delta_staging_vectors_host != nullptr) cudaFreeHost(delta_staging_vectors_host);
    if (delta_staging_records_host != nullptr) cudaFreeHost(delta_staging_records_host);
    if (delta_staging_slots_host != nullptr) cudaFreeHost(delta_staging_slots_host);
    if (delta_override_updates_host != nullptr) cudaFreeHost(delta_override_updates_host);
    if (delta_durable_updates_host != nullptr) cudaFreeHost(delta_durable_updates_host);
    if (delta_supersede_updates_host != nullptr) cudaFreeHost(delta_supersede_updates_host);
    if (graph_invalidation_keys_host != nullptr) cudaFreeHost(graph_invalidation_keys_host);
    if (anchor_graph_validation_host != nullptr) {
      cudaFreeHost(anchor_graph_validation_host);
    }
    if (anchor_graph_readers_host != nullptr) cudaFreeHost(anchor_graph_readers_host);
    device_free(d_delta_count);
    device_free(d_direct_batch_statuses);
    device_free(d_direct_batch_queues);
    device_free(d_direct_batch_entries);
    device_free(d_direct_batch_sequences);
    device_free(d_direct_batch_dequeue);
    device_free(d_direct_batch_enqueue);
    device_free(d_delta_remote_slots);
    device_free(d_delta_remote_keys);
    device_free(d_base_override_epochs);
    device_free(d_base_override_keys);
    device_free(d_permanent_override_bits);
    device_free(d_delta_remote_positions);
    device_free(d_delta_prev);
    device_free(d_delta_next);
    device_free(d_delta_pq_codes);
    device_free(d_delta_encode_scratch);
    device_free(d_delta_vectors);
    device_free(d_delta_records);
    device_free(d_graph_cache_generation);
    device_free(d_anchor_graph_readers);
    device_free(d_anchor_graph_states);
    device_free(d_anchor_graph_keys);
    device_free(d_graph_admission_victims);
    device_free(d_graph_admission_keys);
    device_free(d_graph_cache_victims);
    device_free(d_graph_cache_states);
    device_free(d_graph_cache_readers);
    device_free(d_graph_cache_timestamps);
    device_free(d_graph_cache_generations);
    device_free(d_graph_cache_keys);
    device_free(d_exact_cache_victims);
    device_free(d_exact_admission_victims);
    device_free(d_exact_admission_keys);
    device_free(d_exact_cache_readers);
    device_free(d_exact_cache_states);
    device_free(d_exact_cache_keys);
    control_bootstrapper.reset();
    if (owns_remote_buffer) device_free(d_remote_buffer);
#ifdef DVSTOR_HAVE_GPUNETIO
    direct_transport.reset();
#endif
    device_free(d_dynamic_code_request_local_iovas);
    device_free(d_dynamic_code_request_offsets);
    device_free(d_dynamic_code_request_shards);
    device_free(d_visited);
    device_free(d_navigation_candidate_distances);
    device_free(d_navigation_candidate_handles);
    device_free(d_query_luts);
    device_free(d_transformed_queries);
    if (query_input_host != nullptr) cudaFreeHost(query_input_host);
    device_free(d_queries);
    device_free(d_delta_bucket_heads);
    device_free(d_anchor_handles);
    device_free(d_anchor_pq_codes);
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
    if (query_data == nullptr || static_cast<u32>(query_dtype) > 2 ||
        k == 0 || k > result_capacity) {
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
    const size_t query_bytes = vector_dtype_bytes(query_dtype, config.dim);
    byte_t* query_slot = query_input_host + static_cast<size_t>(slot) * query_input_stride;
    std::memcpy(query_slot, query_data, query_bytes);
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
        d_query_input + static_cast<size_t>(slot) * query_input_stride),
      .result_device_address = reinterpret_cast<u64>(
        d_result_ids + static_cast<size_t>(slot) * result_capacity),
      .query_slot = slot,
      .result_capacity = result_capacity,
      .dim = static_cast<u16>(config.dim),
      .k = static_cast<u16>(k),
      .query_dtype = static_cast<u8>(query_dtype),
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
    batch.reserve(config.gpu_query_slots);
    size_t submitted_count = 0;
    try {
      bind_cuda_device("cudaSetDevice(GPU navigation admission)");
      while (!shutdown.load(std::memory_order_acquire)) {
        batch.clear();
        submitted_count = 0;
        {
          std::unique_lock<std::mutex> lock(admission_mutex);
          admission_cv.wait(lock, [&] {
            return !admission_queue.empty() ||
                   !healthy.load() || shutdown.load();
          });
          if (!healthy.load(std::memory_order_acquire) || shutdown.load()) return;
          if (admission_queue.empty()) continue;
          const size_t count = std::min<size_t>(
            admission_queue.size(), config.gpu_query_slots);
          for (size_t index = 0; index < count; ++index) {
            batch.push_back(admission_queue.front());
            admission_queue.pop_front();
          }
          active_gpu_queries.fetch_add(count, std::memory_order_release);
        }
        if (batch.empty()) continue;
        const auto admitted_at = std::chrono::steady_clock::now();
        u64 wait_ns = 0;
        {
          std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
          for (PendingSubmission& submission : batch) {
            submission.descriptor.snapshot_epoch = engine.delta_.published_epoch();
            const u64 query_ticket =
              next_query_ticket.fetch_add(1, std::memory_order_acq_rel);
            const u32 slot = submission.descriptor.query_slot;
            active_query_snapshots[slot].store(
              submission.descriptor.snapshot_epoch + 1,
              std::memory_order_release);
            active_query_tickets[slot].store(query_ticket, std::memory_order_release);
          }
        }
        for (PendingSubmission& submission : batch) {
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

  void refresh_anchor_graph_records(std::span<const u64> invalidation_keys) {
    if (invalidation_keys.empty() || anchor_graph_keys_host.empty()) return;
    std::vector<u32> route_slots;
    route_slots.reserve(invalidation_keys.size());
    for (u64 key : invalidation_keys) {
      const auto iterator = std::lower_bound(
        anchor_graph_keys_host.begin(), anchor_graph_keys_host.end(), key);
      if (iterator != anchor_graph_keys_host.end() && *iterator == key) {
        route_slots.push_back(static_cast<u32>(
          iterator - anchor_graph_keys_host.begin()));
      }
    }
    if (route_slots.empty()) return;

    const auto timeout = std::chrono::milliseconds(std::clamp<u32>(
      config.storage_owner_rpc_timeout_ms, 1000, 5000));
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      check_cuda(cudaMemcpyAsync(
                   anchor_graph_readers_host, d_anchor_graph_readers,
                   anchor_graph_keys_host.size() * sizeof(u32),
                   cudaMemcpyDeviceToHost, route_refresh_stream),
                 "cudaMemcpyAsync(anchor route readers)");
      check_cuda(cudaStreamSynchronize(route_refresh_stream),
                 "cudaStreamSynchronize(anchor route readers)");
      bool busy = false;
      for (u32 slot : route_slots) {
        if (anchor_graph_readers_host[slot] != 0) {
          busy = true;
          break;
        }
      }
      if (!busy) break;
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error(
          "anchor route graph refresh timed out waiting for active readers");
      }
      std::this_thread::yield();
    }

    std::vector<NavigationRead> requests;
    std::vector<i32> statuses(route_slots.size(), -EIO);
    requests.reserve(route_slots.size());
    for (u32 slot : route_slots) {
      const u64 key = anchor_graph_keys_host[slot];
      requests.push_back(NavigationRead{
        .remote_offset = (key << 16) >> 16,
        .destination_address = reinterpret_cast<u64>(
          d_anchor_graph_records +
          static_cast<size_t>(slot) * index.layout.graph_entry_bytes),
        .bytes = index.layout.graph_entry_bytes,
        .memory_node = static_cast<u16>(key >> 48),
      });
    }
    control_bootstrapper->read(requests, statuses);
    for (size_t request = 0; request < statuses.size(); ++request) {
      if (statuses[request] <= 0) {
        throw std::runtime_error(
          "anchor route graph refresh RDMA read failed: slot=" +
          std::to_string(route_slots[request]) + " status=" +
          std::to_string(statuses[request]));
      }
      check_cuda(cudaMemcpyAsync(
                   anchor_graph_validation_host,
                   d_anchor_graph_records +
                     static_cast<size_t>(route_slots[request]) *
                       index.layout.graph_entry_bytes,
                   index.layout.graph_entry_bytes, cudaMemcpyDeviceToHost,
                   route_refresh_stream),
                 "cudaMemcpyAsync(anchor route validation)");
      check_cuda(cudaStreamSynchronize(route_refresh_stream),
                 "cudaStreamSynchronize(anchor route validation)");
      const u16 expected = vamana::hot_graph::load_u16_le(
        anchor_graph_validation_host + 2);
      const u16 actual = vamana::hot_graph::checksum16(
        anchor_graph_validation_host, index.layout.graph_entry_bytes);
      if (anchor_graph_validation_host[0] > index.layout.graph_degree ||
          expected != actual) {
        throw std::runtime_error(
          "anchor route graph refresh produced an invalid record at slot " +
          std::to_string(route_slots[request]));
      }
    }

    check_cuda(cudaMemcpyAsync(
                 d_anchor_graph_states, anchor_graph_ready_states_host.data(),
                 anchor_graph_ready_states_host.size() * sizeof(u32),
                 cudaMemcpyHostToDevice, route_refresh_stream),
               "cudaMemcpyAsync(anchor route ready states)");
    check_cuda(cudaStreamSynchronize(route_refresh_stream),
               "cudaStreamSynchronize(anchor route ready states)");
    engine.telemetry_.graph_route_refreshes.fetch_add(
      route_slots.size(), std::memory_order_relaxed);
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

  void upload_records_locked(std::vector<DeltaMutation>& mutations,
                             std::span<const u64> invalidation_keys = {}) {
    const auto prepare_started = std::chrono::steady_clock::now();
    bind_cuda_device("cudaSetDevice(GPU navigation delta publication)");
    (void)cudaGetLastError();
    const size_t available_slots = free_delta_slots.size() +
      (delta_capacity - delta_records_host.size());
    if (mutations.size() > available_slots) {
      throw std::runtime_error("GPU navigation delta live set exceeds its configured capacity");
    }
    const size_t vector_bytes = VamanaNode::vector_bytes();
    std::vector<DeviceDeltaRecord> records;
    std::vector<u32> destination_slots;
    std::vector<byte_t> vectors(static_cast<size_t>(mutations.size()) * vector_bytes);
    records.reserve(mutations.size());
    destination_slots.reserve(mutations.size());
    std::unordered_map<u32, size_t> staged_record_indices;
    std::vector<DeltaSupersedeUpdate> superseded_updates;
    std::vector<DeltaOverrideUpdate> override_updates;
    std::vector<f32> decoded(config.dim);
    for (size_t mutation_index = 0; mutation_index < mutations.size(); ++mutation_index) {
      DeltaMutation& mutation = mutations[mutation_index];
      bool decoded_ready = false;
      u32 slot = UINT32_MAX;
      if (!free_delta_slots.empty()) {
        slot = free_delta_slots.back();
        free_delta_slots.pop_back();
      } else {
        slot = static_cast<u32>(delta_records_host.size());
        delta_records_host.emplace_back();
      }
      const auto previous = latest_delta_slot.find(mutation.id);
      if (previous != latest_delta_slot.end()) {
        DeviceDeltaRecord& previous_record = delta_records_host[previous->second];
        if (previous_record.superseded_epoch == 0 &&
            (previous_record.flags & kDeltaDeleted) == 0) {
          if ((previous_record.flags & kDeltaDurable) != 0) {
            --durable_delta_entries;
          } else {
            --mutable_delta_entries;
          }
        }
        previous_record.superseded_epoch = mutation.epoch;
        superseded_delta_slots[mutation.id].push_back(previous->second);
        const auto staged = staged_record_indices.find(previous->second);
        if (staged != staged_record_indices.end()) {
          records[staged->second].superseded_epoch = mutation.epoch;
        } else {
          superseded_updates.push_back(DeltaSupersedeUpdate{
            .slot = previous->second,
            .epoch = mutation.epoch,
          });
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
      u32 base_ordinal = kBaseOverrideEmpty;
      if (format::remote_to_ordinal(
            index, RemotePtr{mutation.old_remote_node}, base_ordinal)) {
        const auto [it, inserted] =
          base_override_epochs.emplace(base_ordinal, mutation.epoch);
        if (inserted) {
          override_updates.push_back(DeltaOverrideUpdate{
            .ordinal = base_ordinal,
            .epoch = mutation.epoch,
          });
        } else if (mutation.epoch < it->second) {
          it->second = mutation.epoch;
          override_updates.push_back(DeltaOverrideUpdate{
            .ordinal = base_ordinal,
            .epoch = mutation.epoch,
          });
        }
      } else {
        base_ordinal = kBaseOverrideEmpty;
      }
      DeviceDeltaRecord record{
        .id = mutation.id,
        .generation = std::max<u32>(1, mutation.generation),
        .flags = (deleted ? kDeltaDeleted : 0u) |
          (mutation.durable ? kDeltaDurable : 0u),
        .base_ordinal = base_ordinal,
        .epoch = mutation.epoch,
        .remote_node = record_remote,
        .anchor_bucket = bucket,
      };
      delta_records_host[slot] = record;
      records.push_back(record);
      destination_slots.push_back(slot);
      staged_record_indices.emplace(slot, records.size() - 1);
      latest_delta_slot[mutation.id] = slot;
      if (!deleted) {
        if (mutation.durable) {
          ++durable_delta_entries;
        } else {
          ++mutable_delta_entries;
        }
      }
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
    }

    if (records.size() > delta_command_capacity ||
        superseded_updates.size() > delta_command_capacity ||
        override_updates.size() > delta_command_capacity ||
        invalidation_keys.size() > graph_invalidation_capacity) {
      throw std::runtime_error("GPU navigation delta control batch exceeds capacity");
    }

    std::memcpy(delta_staging_records_host, records.data(),
                records.size() * sizeof(DeviceDeltaRecord));
    std::memcpy(delta_staging_slots_host, destination_slots.data(),
                destination_slots.size() * sizeof(u32));
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
      .record_count = static_cast<u32>(records.size()),
      .final_count = count,
      .invalidation_count = static_cast<u32>(invalidation_keys.size()),
      .superseded_count = static_cast<u32>(superseded_updates.size()),
      .override_count = static_cast<u32>(override_updates.size()),
    });
    refresh_anchor_graph_records(invalidation_keys);
    engine.telemetry_.publication_command_ns_total.fetch_add(
      static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - command_started).count()),
      std::memory_order_relaxed);
    engine.telemetry_.delta_physical_entries.store(
      count - free_delta_slots.size(), std::memory_order_relaxed);
    engine.telemetry_.delta_mutable_entries.store(
      mutable_delta_entries, std::memory_order_relaxed);
    engine.telemetry_.delta_durable_entries.store(
      durable_delta_entries, std::memory_order_relaxed);
  }

  size_t upload_mutations(std::vector<DeltaMutation>& mutations, u64 epoch,
                          std::span<const u64> invalidated_graph_nodes) {
    if (mutations.empty()) return 0;
    const std::vector<u64> invalidation_keys = graph_cache_keys(invalidated_graph_nodes);
    std::lock_guard<std::mutex> lock(delta_mutex);
    reclaim_retired_delta_slots_locked();
    const size_t active_slots = active_delta_slots_locked();
    const size_t hard_watermark = static_cast<size_t>(delta_capacity) * 9 / 10;
    if (active_slots + mutations.size() > hard_watermark) {
      throw MutationCapacityError(
        "bounded GPU update tier reached its hard watermark; "
        "storage maintenance has not retired updates quickly enough");
    }
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
    upload_records_locked(mutations, invalidation_keys);
    return invalidation_keys.size();
  }

  size_t active_delta_slots_locked() const {
    return delta_records_host.size() - free_delta_slots.size();
  }

  bool query_ticket_barrier_passed(u64 barrier) const {
    for (u32 slot = 0; slot < query_slots; ++slot) {
      const u64 ticket = active_query_tickets[slot].load(std::memory_order_acquire);
      if (ticket != 0 && ticket <= barrier) return false;
    }
    return true;
  }

  bool durable_snapshot_safe(u64 durable_epoch) const {
    for (u32 slot = 0; slot < query_slots; ++slot) {
      const u64 encoded_snapshot =
        active_query_snapshots[slot].load(std::memory_order_acquire);
      if (encoded_snapshot != 0 && encoded_snapshot - 1 < durable_epoch) {
        return false;
      }
    }
    return true;
  }

  void reclaim_retired_delta_slots_locked() {
    u64 reclaimed = 0;
    while (!retired_delta_batches.empty() &&
           query_ticket_barrier_passed(
             retired_delta_batches.front().query_ticket_barrier)) {
      RetiredDeltaBatch batch = std::move(retired_delta_batches.front());
      retired_delta_batches.pop_front();
      reclaimed += batch.slots.size();
      free_delta_slots.insert(free_delta_slots.end(),
                              batch.slots.begin(), batch.slots.end());
    }
    if (reclaimed != 0) {
      engine.telemetry_.delta_compactions.fetch_add(1, std::memory_order_relaxed);
    }
    engine.telemetry_.delta_physical_entries.store(
      active_delta_slots_locked(), std::memory_order_relaxed);
  }

  void validate_storage_control(const format::StorageControlBlock& control,
                                size_t shard) const {
    if (control.magic != format::kStorageControlMagic ||
        control.version != format::kStorageControlVersion ||
        control.header_bytes != sizeof(format::StorageControlBlock) ||
        control.shard_id != shard ||
        control.compute_client_count != compute_client_count ||
        control.dynamic_record_bytes != index.shards[shard].dynamic_record_bytes ||
        control.dynamic_hot_offset != index.shards[shard].dynamic_hot_offset ||
        control.dynamic_code_offset != index.shards[shard].dynamic_code_offset ||
        control.code_bytes != index.layout.code_bytes) {
      std::ostringstream message;
      message << "storage maintenance control mismatch for shard " << shard
              << ": expected{magic=0x" << std::hex
              << format::kStorageControlMagic << std::dec
              << ",version=" << format::kStorageControlVersion
              << ",header=" << sizeof(format::StorageControlBlock)
              << ",shard=" << shard
              << ",clients=" << compute_client_count
              << ",record=" << index.shards[shard].dynamic_record_bytes
              << ",hot=" << index.shards[shard].dynamic_hot_offset
              << ",dynamic_code=" << index.shards[shard].dynamic_code_offset
              << ",code=" << index.layout.code_bytes
              << "} actual{magic=0x" << std::hex << control.magic << std::dec
              << ",version=" << control.version
              << ",header=" << control.header_bytes
              << ",shard=" << control.shard_id
              << ",clients=" << control.compute_client_count
              << ",record=" << control.dynamic_record_bytes
              << ",hot=" << control.dynamic_hot_offset
              << ",dynamic_code=" << control.dynamic_code_offset
              << ",code=" << control.code_bytes
              << "}. Rebuild and restart every storage node from the current "
                 "dev branch before starting the compute node.";
      throw std::runtime_error(message.str());
    }
  }

  std::vector<format::StorageControlBlock> read_storage_controls() {
    if (control_bootstrapper == nullptr || index.shards.empty()) return {};
    std::vector<NavigationRead> requests(index.shards.size());
    std::vector<i32> statuses(index.shards.size(), -EIO);
    for (size_t shard = 0; shard < index.shards.size(); ++shard) {
      requests[shard] = NavigationRead{
        .remote_offset = index.shards[shard].control_remote_offset,
        .destination_address = reinterpret_cast<u64>(d_control_snapshots + shard),
        .bytes = sizeof(format::StorageControlBlock),
        .memory_node = static_cast<u16>(shard),
      };
    }
    control_bootstrapper->read(requests, statuses);
    std::vector<format::StorageControlBlock> controls(index.shards.size());
    check_cuda(cudaMemcpy(controls.data(), d_control_snapshots,
                          controls.size() * sizeof(format::StorageControlBlock),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(storage maintenance controls)");
    for (size_t shard = 0; shard < controls.size(); ++shard) {
      if (statuses[shard] <= 0) {
        throw std::runtime_error(
          "storage maintenance control read failed for shard " +
          std::to_string(shard));
      }
      validate_storage_control(controls[shard], shard);
    }
    return controls;
  }

  void write_storage_reclaim_acks(std::span<const u64> sequences) {
    if (sequences.size() != index.shards.size()) {
      throw std::invalid_argument("storage reclaim ACK cardinality mismatch");
    }
    std::vector<NavigationWrite> requests(index.shards.size());
    std::vector<i32> statuses(index.shards.size(), -EIO);
    for (size_t shard = 0; shard < index.shards.size(); ++shard) {
      u64* device_ack =
        &d_control_snapshots[shard].reclaim_ack_sequences[compute_client_id];
      check_cuda(cudaMemcpy(device_ack, &sequences[shard], sizeof(u64),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(storage reclaim ACK)");
      requests[shard] = NavigationWrite{
        .remote_offset = index.shards[shard].control_remote_offset +
          offsetof(format::StorageControlBlock, reclaim_ack_sequences) +
          static_cast<u64>(compute_client_id) * sizeof(u64),
        .source_address = reinterpret_cast<u64>(device_ack),
        .bytes = sizeof(u64),
        .memory_node = static_cast<u16>(shard),
      };
    }
    control_bootstrapper->write(requests, statuses);
    for (size_t shard = 0; shard < statuses.size(); ++shard) {
      if (statuses[shard] <= 0) {
        throw std::runtime_error(
          "storage reclaim ACK write failed for shard " +
          std::to_string(shard));
      }
    }
  }

  void initialize_storage_reclaim_ack() {
    (void)read_storage_controls();
    pending_storage_reclaim_acks.resize(index.shards.size());
    enqueued_reclaim_ack_sequences.assign(index.shards.size(), 0);
    published_reclaim_ack_sequences.assign(index.shards.size(), 0);
    const std::vector<u64> reset_sequences(index.shards.size(), 0);
    write_storage_reclaim_acks(reset_sequences);
    std::cerr << "[gpu-search] storage reclaim RCU client=" << compute_client_id
              << '/' << compute_client_count << " ACK reset complete\n";
  }

  void enqueue_storage_reclaim_barriers() {
    std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
    const u64 barrier = next_query_ticket.load(std::memory_order_acquire) - 1;
    for (size_t shard = 0; shard < safe_durable_sequences.size(); ++shard) {
      const u64 sequence = safe_durable_sequences[shard];
      if (sequence <= enqueued_reclaim_ack_sequences[shard]) continue;
      auto& queue = pending_storage_reclaim_acks[shard];
      if (!queue.empty() && queue.back().query_ticket_barrier == barrier) {
        queue.back().maintenance_sequence = sequence;
      } else {
        queue.push_back(PendingStorageReclaimAck{
          .maintenance_sequence = sequence,
          .query_ticket_barrier = barrier,
        });
      }
      enqueued_reclaim_ack_sequences[shard] = sequence;
    }
  }

  void publish_ready_storage_reclaim_acks() {
    if (!healthy.load(std::memory_order_acquire)) return;
    std::vector<u64> targets = published_reclaim_ack_sequences;
    bool advanced = false;
    for (size_t shard = 0; shard < pending_storage_reclaim_acks.size(); ++shard) {
      auto& queue = pending_storage_reclaim_acks[shard];
      while (!queue.empty() &&
             query_ticket_barrier_passed(queue.front().query_ticket_barrier)) {
        targets[shard] = queue.front().maintenance_sequence;
        queue.pop_front();
        advanced = true;
      }
    }
    if (!advanced) return;
    write_storage_reclaim_acks(targets);
    published_reclaim_ack_sequences = std::move(targets);
    engine.telemetry_.storage_reclaim_ack_writes.fetch_add(
      1, std::memory_order_relaxed);
    engine.telemetry_.storage_reclaim_ack_sequence.store(
      *std::min_element(published_reclaim_ack_sequences.begin(),
                        published_reclaim_ack_sequences.end()),
      std::memory_order_relaxed);
  }

  std::vector<DeltaMutation> retire_durable_delta() {
    if (control_bootstrapper == nullptr || index.shards.empty()) return {};
    const std::vector<format::StorageControlBlock> controls =
      read_storage_controls();
    if (durable_sequence_history.size() != index.shards.size()) {
      durable_sequence_history.resize(index.shards.size());
      observed_durable_sequences.assign(index.shards.size(), 0);
      safe_durable_sequences.assign(index.shards.size(), 0);
    }
    const auto now = std::chrono::steady_clock::now();
    const auto visibility_grace =
      std::chrono::microseconds(config.update_visibility_us);
    for (size_t shard = 0; shard < controls.size(); ++shard) {
      const auto& control = controls[shard];
      if (control.durable_maintenance_sequence > observed_durable_sequences[shard]) {
        observed_durable_sequences[shard] = control.durable_maintenance_sequence;
        durable_sequence_history[shard].emplace_back(
          control.durable_maintenance_sequence, now);
      }
      auto& history = durable_sequence_history[shard];
      while (!history.empty() && now - history.front().second >= visibility_grace) {
        safe_durable_sequences[shard] = history.front().first;
        history.pop_front();
      }
    }
    enqueue_storage_reclaim_barriers();
    return engine.delta_.retire_durable(
      safe_durable_sequences, delta_command_capacity);
  }

  void mark_durable_delta_records_locked(
      std::span<const DeltaMutation> retired) {
    std::vector<DeltaDurableUpdate> updates;
    std::vector<u32> retiring_slots;
    for (const DeltaMutation& mutation : retired) {
      std::vector<u32> retained_superseded;
      const auto superseded = superseded_delta_slots.find(mutation.id);
      if (superseded != superseded_delta_slots.end()) {
        retained_superseded.reserve(superseded->second.size());
        for (u32 slot : superseded->second) {
          if (slot < delta_records_host.size() &&
              delta_records_host[slot].epoch <= mutation.epoch) {
            retiring_slots.push_back(slot);
          } else {
            retained_superseded.push_back(slot);
          }
        }
        if (retained_superseded.empty()) {
          superseded_delta_slots.erase(superseded);
        } else {
          superseded->second = std::move(retained_superseded);
        }
      }
      const auto latest = latest_delta_slot.find(mutation.id);
      if (latest != latest_delta_slot.end() &&
          latest->second < delta_records_host.size() &&
          delta_records_host[latest->second].epoch <= mutation.epoch) {
        DeviceDeltaRecord& record = delta_records_host[latest->second];
        if ((record.flags & (kDeltaDeleted | kDeltaDurable)) == 0 &&
            record.superseded_epoch == 0) {
          if (mutable_delta_entries == 0) {
            throw std::runtime_error("GPU mutable delta accounting underflow");
          }
          --mutable_delta_entries;
        }
        retiring_slots.push_back(latest->second);
        latest_delta_slot.erase(latest);
      }
    }
    std::sort(retiring_slots.begin(), retiring_slots.end());
    retiring_slots.erase(
      std::unique(retiring_slots.begin(), retiring_slots.end()),
      retiring_slots.end());
    updates.reserve(retiring_slots.size());
    for (u32 slot : retiring_slots) {
      DeviceDeltaRecord& record = delta_records_host[slot];
      updates.push_back(DeltaDurableUpdate{
        .slot = slot,
        .epoch = record.epoch,
      });
    }
    for (size_t begin = 0; begin < updates.size(); begin += delta_command_capacity) {
      const size_t count = std::min<size_t>(
        delta_command_capacity, updates.size() - begin);
      std::memcpy(delta_durable_updates_host, updates.data() + begin,
                  count * sizeof(DeltaDurableUpdate));
      const u32 live_count = static_cast<u32>(delta_records_host.size());
      submit_delta_publication(DeltaPublishDescriptor{
        .command_id = next_delta_command_id.fetch_add(1, std::memory_order_relaxed),
        .final_count = live_count,
        .durable_count = static_cast<u32>(count),
      });
    }
    if (!retiring_slots.empty()) {
      for (u32 slot : retiring_slots) {
        DeviceDeltaRecord& record = delta_records_host[slot];
        record.flags |= kDeltaDurable;
        if (record.superseded_epoch == 0) record.superseded_epoch = record.epoch;
        if (record.base_ordinal != kBaseOverrideEmpty) {
          const auto override = base_override_epochs.find(record.base_ordinal);
          if (override != base_override_epochs.end() &&
              override->second <= record.epoch) {
            base_override_epochs.erase(override);
          }
        }
      }
      std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
      const u64 barrier = next_query_ticket.load(std::memory_order_acquire) - 1;
      retired_delta_batches.push_back(RetiredDeltaBatch{
        .query_ticket_barrier = barrier,
        .slots = std::move(retiring_slots),
      });
      reclaim_retired_delta_slots_locked();
    }
    engine.telemetry_.delta_mutable_entries.store(
      mutable_delta_entries, std::memory_order_relaxed);
    engine.telemetry_.delta_durable_entries.store(
      durable_delta_entries, std::memory_order_relaxed);
    engine.telemetry_.delta_entries_retired.fetch_add(
      updates.size(), std::memory_order_relaxed);
  }

  void maintenance_loop() {
    bind_cuda_device("cudaSetDevice(GPU navigation maintenance)");
    const auto period = std::chrono::milliseconds(std::max<u32>(
      1, std::min<u32>(config.merge_period_ms,
                       std::max<u32>(1, config.update_visibility_us / 1000))));
    while (!maintenance_shutdown.load(std::memory_order_acquire)) {
      {
        std::unique_lock<std::mutex> lock(maintenance_mutex);
        maintenance_cv.wait_for(lock, period, [&] { return maintenance_shutdown.load(); });
      }
      if (maintenance_shutdown.load()) break;
      std::vector<DeltaMutation> retired;
      try {
        std::lock_guard<std::mutex> publish_lock(engine.mutation_publish_mutex_);
        retired = retire_durable_delta();
        if (!retired.empty()) {
          pending_durable_retirements.insert(
            pending_durable_retirements.end(),
            std::make_move_iterator(retired.begin()),
            std::make_move_iterator(retired.end()));
        }
        std::vector<DeltaMutation> snapshot_safe;
        std::vector<DeltaMutation> deferred;
        snapshot_safe.reserve(pending_durable_retirements.size());
        deferred.reserve(pending_durable_retirements.size());
        {
          std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
          for (DeltaMutation& mutation : pending_durable_retirements) {
            if (snapshot_safe.size() < delta_command_capacity &&
                durable_snapshot_safe(mutation.epoch)) {
              snapshot_safe.push_back(std::move(mutation));
            } else {
              deferred.push_back(std::move(mutation));
            }
          }
        }
        pending_durable_retirements = std::move(deferred);
        {
          std::lock_guard<std::mutex> delta_lock(delta_mutex);
          if (!snapshot_safe.empty()) {
            mark_durable_delta_records_locked(snapshot_safe);
          }
          reclaim_retired_delta_slots_locked();
        }
        publish_ready_storage_reclaim_acks();
      } catch (const std::exception& error) {
        mark_unhealthy(std::string{"storage maintenance watermark failed: "} + error.what());
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
        if (completion.query_slot < query_slots) {
          active_query_tickets[completion.query_slot].store(
            0, std::memory_order_release);
          active_query_snapshots[completion.query_slot].store(
            0, std::memory_order_release);
        }
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
                  << " graph_rounds=" << completion.graph_rounds
                  << " graph_hits=" << completion.cache_hits
                  << " route_hits=" << completion.route_hits
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
        active_query_tickets[pending->slot].store(0, std::memory_order_release);
        active_query_snapshots[pending->slot].store(0, std::memory_order_release);
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
      engine.telemetry_.graph_dependency_rounds.fetch_add(
        completion.graph_rounds, std::memory_order_relaxed);
      engine.telemetry_.graph_page_cache_hits.fetch_add(completion.cache_hits,
                                                        std::memory_order_relaxed);
      engine.telemetry_.graph_route_hits.fetch_add(completion.route_hits,
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
  std::unique_ptr<NavigationBootstrapper> control_bootstrapper;
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
  u32 permanent_override_words{};
  u32 graph_cache_sets{};
  u32 graph_cache_slots{};
  u32 exact_cache_sets{};
  u32 exact_cache_slots{};
  u32 exact_cache_stride{};
  u32 graph_admission_sets{};
  u32 exact_admission_sets{};
  u32 graph_invalidation_capacity{};
  u32 delta_command_capacity{};
  u32 direct_batch_queue_count{};
  size_t graph_cache_bytes{};
  size_t exact_cache_bytes{};
  size_t anchor_graph_region_offset{};
  size_t dynamic_code_region_offset{};
  size_t exact_region_offset{};
  size_t exact_cache_offset{};
  size_t graph_cache_offset{};
  size_t graph_scratch_offset{};
  size_t control_region_offset{};
  u64 route_graph_bytes{};
  u64 explicit_gpu_bytes{};
  u64 gpu_clock_khz{1};
  DeviceShardRegion* d_shards{};
  byte_t* d_pq_codes{};
  f32* d_opq_matrix{};
  f32* d_pq_centroids{};
  u32* d_entry_points{};
  f32* d_anchor_vectors{};
  u32* d_anchor_handles{};
  u8* d_anchor_pq_codes{};
  std::vector<u64> anchor_graph_keys_host;
  std::vector<u32> anchor_graph_ready_states_host;
  u32* anchor_graph_readers_host{};
  byte_t* anchor_graph_validation_host{};
  u64* d_anchor_graph_keys{};
  u32* d_anchor_graph_states{};
  u32* d_anchor_graph_readers{};
  u32* d_delta_bucket_heads{};
  size_t query_input_stride{};
  f32* d_queries{};
  byte_t* query_input_host{};
  byte_t* d_query_input{};
  f32* d_transformed_queries{};
  f32* d_query_luts{};
  u32* d_navigation_candidate_handles{};
  f32* d_navigation_candidate_distances{};
  u32* d_visited{};
  byte_t* d_dynamic_code_records{};
  u32* d_dynamic_code_request_shards{};
  u64* d_dynamic_code_request_offsets{};
  u64* d_dynamic_code_request_local_iovas{};
  u64* d_direct_batch_enqueue{};
  u64* d_direct_batch_dequeue{};
  u64* d_direct_batch_sequences{};
  DirectBatchDescriptor* d_direct_batch_entries{};
  DeviceRingView<DirectBatchDescriptor>* d_direct_batch_queues{};
  i32* d_direct_batch_statuses{};
  byte_t* d_exact_records{};
  byte_t* d_exact_cache{};
  byte_t* d_remote_buffer{};
  byte_t* d_anchor_graph_records{};
  byte_t* d_graph_cache{};
  byte_t* d_graph_scratch{};
  format::StorageControlBlock* d_control_snapshots{};
  u64* d_graph_cache_keys{};
  u64* d_graph_cache_generations{};
  u64* d_graph_cache_timestamps{};
  u32* d_graph_cache_states{};
  u32* d_graph_cache_readers{};
  u32* d_graph_cache_victims{};
  u64* d_graph_admission_keys{};
  u32* d_graph_admission_victims{};
  u64* d_graph_cache_generation{};
  u64* graph_invalidation_keys_host{};
  u64* d_graph_invalidation_keys{};
  u32* d_exact_cache_keys{};
  u32* d_exact_cache_states{};
  u32* d_exact_cache_readers{};
  u32* d_exact_cache_victims{};
  u32* d_exact_admission_keys{};
  u32* d_exact_admission_victims{};
  bool owns_remote_buffer{};
  u32* result_ids_host{};
  f32* result_distances_host{};
  u32* d_result_ids{};
  f32* d_result_distances{};
  DeviceDeltaRecord* d_delta_records{};
  byte_t* d_delta_vectors{};
  byte_t* d_delta_pq_codes{};
  f32* d_delta_encode_scratch{};
  u32* delta_staging_slots_host{};
  u32* d_delta_staging_slots{};
  DeviceDeltaRecord* delta_staging_records_host{};
  DeviceDeltaRecord* d_delta_staging_records{};
  byte_t* delta_staging_vectors_host{};
  byte_t* d_delta_staging_vectors{};
  u32* d_delta_next{};
  u32* d_delta_prev{};
  u32* d_delta_remote_positions{};
  u32* d_base_override_keys{};
  u64* d_base_override_epochs{};
  u32* d_permanent_override_bits{};
  u64* d_delta_remote_keys{};
  u32* d_delta_remote_slots{};
  DeltaSupersedeUpdate* delta_supersede_updates_host{};
  DeltaSupersedeUpdate* d_delta_supersede_updates{};
  DeltaOverrideUpdate* delta_override_updates_host{};
  DeltaOverrideUpdate* d_delta_override_updates{};
  DeltaDurableUpdate* delta_durable_updates_host{};
  DeltaDurableUpdate* d_delta_durable_updates{};
  u32* d_delta_count{};
  std::vector<DeviceDeltaRecord> delta_records_host;
  std::vector<u32> free_delta_slots;
  std::unordered_map<node_t, std::vector<u32>> superseded_delta_slots;
  std::deque<RetiredDeltaBatch> retired_delta_batches;
  std::vector<DeltaMutation> pending_durable_retirements;
  std::unordered_map<node_t, u32> latest_delta_slot;
  std::unordered_map<u32, u64> base_override_epochs;
  std::vector<std::deque<std::pair<u64, std::chrono::steady_clock::time_point>>>
    durable_sequence_history;
  std::vector<u64> observed_durable_sequences;
  std::vector<u64> safe_durable_sequences;
  std::vector<std::deque<PendingStorageReclaimAck>> pending_storage_reclaim_acks;
  std::vector<u64> enqueued_reclaim_ack_sequences;
  std::vector<u64> published_reclaim_ack_sequences;
  std::mutex delta_mutex;
  size_t reserved_mutation_capacity{};
  u64 mutable_delta_entries{};
  u64 durable_delta_entries{};
  u32 compute_client_id{};
  u32 compute_client_count{};
  u32* stop_host{};
  u32* stop_device{};
  u32* direct_disabled_host{};
  u32* direct_disabled_device{};
  i32* direct_error_host{};
  i32* direct_error_device{};
  cudaStream_t kernel_stream{};
  cudaStream_t delta_stream{};
  cudaStream_t rdma_stream{};
  cudaStream_t route_refresh_stream{};
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
  std::atomic<u64> next_query_ticket{1};
  std::atomic<u64> next_request_id{1};
  std::atomic<u64> next_delta_command_id{1};
  std::atomic<u64> pending_count{0};
  std::mutex admission_mutex;
  std::condition_variable admission_cv;
  std::deque<PendingSubmission> admission_queue;
  std::string health_error;
  std::mutex slot_mutex;
  std::condition_variable slot_cv;
  std::vector<u32> free_slots;
  std::unique_ptr<std::atomic<u64>[]> active_query_tickets;
  std::unique_ptr<std::atomic<u64>[]> active_query_snapshots;
  std::mutex query_snapshot_mutex;
  std::mutex pending_mutex;
  std::unordered_map<u64, std::shared_ptr<PendingQuery>> pending_queries;
  std::thread admission_thread;
  std::thread completion_thread;
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
  size_t graph_cache_invalidations = 0;
  try {
    graph_cache_invalidations =
      impl_->upload_mutations(mutations, epoch, invalidated_graph_nodes);
  } catch (const MutationCapacityError&) {
    telemetry_.mutation_capacity_rejections.fetch_add(1, std::memory_order_relaxed);
    throw;
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
  telemetry_.graph_cache_invalidations.fetch_add(
    graph_cache_invalidations, std::memory_order_relaxed);
  telemetry_.visibility_ns_total.fetch_add(visibility_ns_total,
                                           std::memory_order_relaxed);
  telemetry_.delta_live_entries.store(delta_.delta_size(), std::memory_order_relaxed);
  u64 current_max = telemetry_.visibility_ns_max.load(std::memory_order_relaxed);
  while (current_max < visibility_ns_max &&
         !telemetry_.visibility_ns_max.compare_exchange_weak(
           current_max, visibility_ns_max, std::memory_order_relaxed)) {}
  return true;
}

bool PersistentSearchEngine::try_reserve_mutation_capacity(size_t mutation_count) {
  if (mutation_count == 0) return true;
  std::lock_guard<std::mutex> lock(impl_->delta_mutex);
  impl_->reclaim_retired_delta_slots_locked();
  const size_t active_slots = impl_->active_delta_slots_locked();
  const size_t hard_watermark = static_cast<size_t>(impl_->delta_capacity) * 9 / 10;
  if (mutation_count > hard_watermark ||
      active_slots > hard_watermark - mutation_count ||
      impl_->reserved_mutation_capacity >
        hard_watermark - mutation_count - active_slots) {
    telemetry_.mutation_capacity_rejections.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  impl_->reserved_mutation_capacity += mutation_count;
  const u64 reserved = static_cast<u64>(impl_->reserved_mutation_capacity);
  telemetry_.mutation_capacity_reserved.store(reserved, std::memory_order_relaxed);
  u64 current_max = telemetry_.mutation_capacity_reserved_max.load(
    std::memory_order_relaxed);
  while (current_max < reserved &&
         !telemetry_.mutation_capacity_reserved_max.compare_exchange_weak(
           current_max, reserved, std::memory_order_relaxed)) {}
  return true;
}

void PersistentSearchEngine::release_mutation_capacity(size_t mutation_count) {
  if (mutation_count == 0) return;
  std::lock_guard<std::mutex> lock(impl_->delta_mutex);
  if (mutation_count > impl_->reserved_mutation_capacity) {
    impl_->mark_unhealthy("GPU mutation capacity reservation accounting underflow");
    impl_->reserved_mutation_capacity = 0;
  } else {
    impl_->reserved_mutation_capacity -= mutation_count;
  }
  telemetry_.mutation_capacity_reserved.store(
    static_cast<u64>(impl_->reserved_mutation_capacity),
    std::memory_order_relaxed);
}

void PersistentSearchEngine::mark_committed_mutation_gap(
    const std::string& reason) {
  impl_->mark_unhealthy(
    "storage committed a mutation that is not GPU-visible: " + reason);
}

void PersistentSearchEngine::reset_telemetry() {
  telemetry_.reset();
  telemetry_.delta_live_entries.store(delta_.delta_size(), std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(impl_->delta_mutex);
  telemetry_.delta_physical_entries.store(
    impl_->active_delta_slots_locked(), std::memory_order_relaxed);
  telemetry_.delta_mutable_entries.store(
    impl_->mutable_delta_entries, std::memory_order_relaxed);
  telemetry_.delta_durable_entries.store(
    impl_->durable_delta_entries, std::memory_order_relaxed);
  telemetry_.mutation_capacity_reserved.store(
    impl_->reserved_mutation_capacity, std::memory_order_relaxed);
}

}  // namespace gpu_search
