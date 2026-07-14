#include "gpu_search/persistent_engine.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "common/index_path.hh"
#include "gpu_search/navigation_bootstrapper.hh"
#ifdef DVSTOR_HAVE_GPUNETIO
#include "gpu/gpunetio_transport.hh"
#endif
#include "gpu_search/index_format.hh"
#include "gpu_search/mapped_ring.hh"
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

  struct RetiredResidentPqBatch {
    u64 query_ticket_barrier{};
    std::vector<ResidentPqEraseUpdate> entries;
  };

  struct PendingStorageReclaimAck {
    u64 maintenance_sequence{};
    u64 query_ticket_barrier{};
  };

  struct DurableRetirement {
    node_t id{};
    service::storage_owner::MutationKind kind{
      service::storage_owner::MutationKind::insert};
    u64 epoch{};
    u64 remote_node{};
    u64 old_remote_node{};
  };

#include "gpu_search/persistent_engine/health.ipp"
#include "gpu_search/persistent_engine/construction.ipp"
#include "gpu_search/persistent_engine/lifecycle.ipp"
#include "gpu_search/persistent_engine/query_execution.ipp"
#include "gpu_search/persistent_engine/routing.ipp"
#include "gpu_search/persistent_engine/delta_publication.ipp"
#include "gpu_search/persistent_engine/storage_reclaim.ipp"
#include "gpu_search/persistent_engine/completion.ipp"
#include "gpu_search/persistent_engine/state.ipp"
};

PersistentSearchEngine::PersistentSearchEngine(
    configuration::IndexConfiguration& config,
    Context& channel_context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions)
    : delta_() {
  check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)),
             "cudaSetDevice(GPU navigation engine)");
  impl_ = std::make_unique<Impl>(*this, config, channel_context,
                                 connection_manager, remote_regions);
}

PersistentSearchEngine::~PersistentSearchEngine() {
  impl_.reset();
}

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
    if (!delta_.publish(std::move(mutations), epoch)) {
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
  const size_t active_resident_pq = impl_->active_resident_pq_slots_locked();
  const size_t resident_pq_hard_watermark =
    std::max<size_t>(1, static_cast<size_t>(impl_->resident_pq_capacity) * 95 / 100);
  if (mutation_count > hard_watermark ||
      active_slots > hard_watermark - mutation_count ||
      impl_->reserved_mutation_capacity >
        hard_watermark - mutation_count - active_slots ||
      mutation_count > resident_pq_hard_watermark ||
      active_resident_pq > resident_pq_hard_watermark - mutation_count ||
      impl_->reserved_mutation_capacity >
        resident_pq_hard_watermark - mutation_count - active_resident_pq) {
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
  telemetry_.resident_pq_capacity.store(
    impl_->resident_pq_capacity, std::memory_order_relaxed);
  telemetry_.resident_pq_entries.store(
    impl_->active_resident_pq_slots_locked(), std::memory_order_relaxed);
  telemetry_.resident_pq_peak_entries.store(
    impl_->active_resident_pq_slots_locked(), std::memory_order_relaxed);
  telemetry_.mutation_capacity_reserved.store(
    impl_->reserved_mutation_capacity, std::memory_order_relaxed);
}

}  // namespace gpu_search
