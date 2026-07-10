#include "gpu_search/persistent_engine.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "common/index_path.hh"
#ifdef DVSTOR_HAVE_GPUNETIO
#include "gpu/gpunetio_query_engine.hh"
#include "gpu/gpunetio_query_launcher.hh"
#endif
#include "gpu_search/index_format.hh"
#include "gpu_search/persistent_kernel.hh"
#include "gpu_search/remote_fetch_backend.hh"
#include "vamana/vamana_node.hh"

namespace gpu_search {
namespace {

static_assert(sizeof(DeviceNodeRecord) == sizeof(format::NodeRecord));

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

u32 next_power_of_two(u32 value) {
  return std::max<u32>(2, std::bit_ceil(value));
}

u64 align_up(u64 value, u64 alignment) {
  return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

template <class T>
class MappedRing {
public:
  explicit MappedRing(u32 capacity) : capacity_(next_power_of_two(capacity)) {
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
    for (u32 i = 0; i < capacity_; ++i) sequences_host_[i] = i;

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
  DeviceRingView<T> device_view_{};
};

template <class T>
void device_allocate(T*& pointer, size_t count, const char* operation) {
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), operation);
}

}  // namespace

struct PersistentSearchEngine::Impl {
  struct PendingQuery {
    u32 slot{};
    u32 capacity{};
    std::chrono::steady_clock::time_point submitted_at{};
    std::promise<service::QueryResult> promise;
  };

  struct PendingSubmission {
    QueryDescriptor descriptor{};
    std::chrono::steady_clock::time_point enqueued_at{};
  };

  Impl(PersistentSearchEngine& owner,
       configuration::IndexConfiguration& config_in,
       Context& channel_context,
       ClientConnectionManager& connection_manager,
       const MemoryRegionTokens& remote_regions)
      : engine(owner), config(config_in), submissions(config.query_batch_max * 2),
        completions(config.query_batch_max * 2),
        fetches(config.query_batch_max * kPersistentMaxExact * 2) {
    EngineKind engine_kind;
    RemoteBackendKind parsed_backend;
    if (!parse_engine_kind(config.search_engine, engine_kind) ||
        engine_kind != EngineKind::gpu_persistent ||
        !parse_remote_backend_kind(config.gpu_rdma_backend, parsed_backend)) {
      throw std::invalid_argument("invalid persistent search engine configuration");
    }
    backend_kind = parsed_backend;
    if (config.beam_width > kPersistentMaxBeam || config.rabitq_gate_max_width > kPersistentMaxExact) {
      throw std::invalid_argument("gpu_persistent supports beam-width <= 256 and rabitq-gate-max-width <= 128");
    }

    std::string load_error;
    if (!format::read_file(index_path::gpu_tiered_file(config.resolved_index_prefix()),
                           index, &load_error)) {
      throw std::runtime_error(load_error);
    }
    const bool centroid_matches =
      index.centroid.size() == VamanaNode::rabitq_centroid.size() &&
      std::equal(index.centroid.begin(), index.centroid.end(),
                 VamanaNode::rabitq_centroid.begin(), [](f32 stored, f32 configured) {
                   const f32 scale = std::max({1.0f, std::abs(stored), std::abs(configured)});
                   return std::abs(stored - configured) <= 1e-6f * scale;
                 });
    const bool shard_layout_matches = !index.shards.empty() &&
      std::all_of(index.shards.begin(), index.shards.end(), [](const format::ShardRegion& shard) {
        return shard.vector_region_offset ==
                 vamana::hot_graph::kNodeBaseOffset + VamanaNode::offset_vector() &&
               shard.vector_stride == VamanaNode::total_size();
      });
    if (index.header.dim != config.dim || index.header.graph_degree != config.R ||
        index.header.hot_degree != config.gpu_hot_degree ||
        index.header.num_shards != remote_regions.size() ||
        index.header.rabitq_code_bits != VamanaNode::rabitq_code_bits() ||
        index.header.rabitq_code_bits > kPersistentMaxCodeBits ||
        index.header.rabitq_entry_bytes != VamanaNode::rabitq_entry_size() ||
        index.entry_points.size() > kPersistentMaxEntryPoints ||
        index.header.vector_dtype != static_cast<u32>(config.resolved_vector_dtype()) ||
        !centroid_matches || !shard_layout_matches) {
      throw std::runtime_error("GPU tiered index metadata does not match runtime configuration");
    }

    query_slots = config.query_batch_max;
    result_capacity = std::max<u32>(config.k, config.rabitq_gate_max_width);
    exact_width = std::max<u32>(config.k, config.rabitq_gate_max_width);
    code_bits = index.header.rabitq_code_bits;
    visited_capacity = next_power_of_two(
      std::max<u32>(256, config.beam_width * index.header.hot_degree * 2));
    free_slots.resize(query_slots);
    for (u32 slot = 0; slot < query_slots; ++slot) free_slots[slot] = slot;

    device_allocate(d_nodes, index.nodes.size(), "cudaMalloc(GPU index nodes)");
    device_allocate(d_hot_neighbors, index.hot_neighbors.size(), "cudaMalloc(GPU hot neighbors)");
    device_allocate(d_rabitq_entries, index.rabitq_entries.size(), "cudaMalloc(GPU RaBitQ entries)");
    device_allocate(d_centroid, index.centroid.size(), "cudaMalloc(GPU centroid)");
    device_allocate(d_entry_points, index.entry_points.size(),
                    "cudaMalloc(GPU entry points)");
    check_cuda(cudaMemcpy(d_nodes, index.nodes.data(), index.nodes.size() * sizeof(format::NodeRecord),
                          cudaMemcpyHostToDevice), "cudaMemcpy(GPU index nodes)");
    check_cuda(cudaMemcpy(d_hot_neighbors, index.hot_neighbors.data(),
                          index.hot_neighbors.size() * sizeof(u32), cudaMemcpyHostToDevice),
               "cudaMemcpy(GPU hot neighbors)");
    check_cuda(cudaMemcpy(d_rabitq_entries, index.rabitq_entries.data(), index.rabitq_entries.size(),
                          cudaMemcpyHostToDevice), "cudaMemcpy(GPU RaBitQ entries)");
    check_cuda(cudaMemcpy(d_centroid, index.centroid.data(), index.centroid.size() * sizeof(f32),
                          cudaMemcpyHostToDevice), "cudaMemcpy(GPU centroid)");
    check_cuda(cudaMemcpy(d_entry_points, index.entry_points.data(),
                          index.entry_points.size() * sizeof(u32), cudaMemcpyHostToDevice),
               "cudaMemcpy(GPU entry points)");

    device_allocate(d_queries, static_cast<size_t>(query_slots) * config.dim,
                    "cudaMalloc(persistent queries)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&query_staging_host),
                             static_cast<size_t>(query_slots) * config.dim * sizeof(f32),
                             cudaHostAllocPortable),
               "cudaHostAlloc(persistent query staging)");
    device_allocate(d_rotated_queries, static_cast<size_t>(query_slots) * code_bits,
                    "cudaMalloc(persistent rotated queries)");
    device_allocate(d_query_luts,
                    static_cast<size_t>(query_slots) * (code_bits / 8) * 256,
                    "cudaMalloc(persistent RaBitQ query LUTs)");
    device_allocate(d_beam_ids, static_cast<size_t>(query_slots) * config.beam_width,
                    "cudaMalloc(persistent beam ids)");
    device_allocate(d_beam_distances, static_cast<size_t>(query_slots) * config.beam_width,
                    "cudaMalloc(persistent beam distances)");
    device_allocate(d_beam_expanded, static_cast<size_t>(query_slots) * config.beam_width,
                    "cudaMalloc(persistent beam expanded)");
    device_allocate(d_visited, static_cast<size_t>(query_slots) * visited_capacity,
                    "cudaMalloc(persistent visited)");
    const size_t exact_vector_bytes = static_cast<size_t>(query_slots) * exact_width *
      vector_dtype_bytes(config.resolved_vector_dtype(), config.dim);
    u64 graph_bytes = 0;
    for (const auto& shard : index.shards) graph_bytes += shard.graph_pages_bytes;
    if (config.gpu_cold_expansions != 0 && graph_bytes != 0) {
      size_t free_bytes = 0;
      size_t total_bytes = 0;
      check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo(graph cache)");
      (void)total_bytes;
      u64 requested_cache_bytes = static_cast<u64>(config.gpu_page_cache_mb) << 20;
      if (requested_cache_bytes == 0) {
        requested_cache_bytes = static_cast<u64>(
          static_cast<double>(free_bytes) * config.gpu_page_cache_ratio);
      }
      const u64 reserved_bytes = (static_cast<u64>(config.delta_budget_mb) << 20) +
        index.header.num_nodes * sizeof(u64) + (512ULL << 20);
      const u64 cache_headroom = free_bytes > reserved_bytes
        ? static_cast<u64>(free_bytes) - reserved_bytes : 0;
      requested_cache_bytes = std::min(requested_cache_bytes, cache_headroom);
      requested_cache_bytes = std::min<u64>(requested_cache_bytes, graph_bytes);
      graph_page_cache_slots = static_cast<u32>(std::min<u64>(
        requested_cache_bytes / index.header.page_bytes,
        std::numeric_limits<u32>::max()));
      if (graph_page_cache_slots != 0) {
        graph_page_cache_bytes = static_cast<size_t>(graph_page_cache_slots) *
          index.header.page_bytes;
      }
    }
    std::cerr << "[gpu-search] resident index nodes=" << index.header.num_nodes
              << " hot_edges=" << index.hot_neighbors.size()
              << " graph_cache=" << graph_page_cache_bytes
              << " bytes (" << graph_page_cache_slots << " pages)" << std::endl;
    const size_t exact_region_bytes = static_cast<size_t>(align_up(
      exact_vector_bytes, std::max<u32>(256, index.header.page_bytes)));
    const size_t remote_buffer_bytes = exact_region_bytes + graph_page_cache_bytes;
    if (parsed_backend == RemoteBackendKind::gpunetio) {
#ifdef DVSTOR_HAVE_GPUNETIO
      direct_transport = std::make_unique<gpu::GpuNetioPersistentTransport>(
        config, remote_buffer_bytes, channel_context, connection_manager, remote_regions);
      direct_view = direct_transport->view();
      if (direct_view.data == nullptr || direct_view.data_bytes < remote_buffer_bytes) {
        throw std::runtime_error("GPUNetIO transport returned an undersized GPU data region");
      }
      d_remote_buffer = direct_view.data;
      owns_remote_buffer = false;
#else
      throw std::runtime_error("DVSTOR was built without DOCA GPUNetIO support");
#endif
    } else {
      device_allocate(d_remote_buffer, remote_buffer_bytes,
                      "cudaMalloc(persistent remote buffer)");
      owns_remote_buffer = true;
    }
    d_exact_vectors = d_remote_buffer;
    d_graph_page_cache = graph_page_cache_bytes == 0
      ? nullptr : d_remote_buffer + exact_region_bytes;
    if (graph_page_cache_slots != 0) {
      device_allocate(d_graph_page_cache_keys, graph_page_cache_slots,
                      "cudaMalloc(graph page cache keys)");
      device_allocate(d_graph_page_cache_locks, graph_page_cache_slots,
                      "cudaMalloc(graph page cache locks)");
      check_cuda(cudaMemset(d_graph_page_cache_keys, 0xff,
                            static_cast<size_t>(graph_page_cache_slots) * sizeof(u64)),
                 "cudaMemset(graph page cache keys)");
      check_cuda(cudaMemset(d_graph_page_cache_locks, 0,
                            static_cast<size_t>(graph_page_cache_slots) * sizeof(u32)),
                 "cudaMemset(graph page cache locks)");
    }
    const size_t result_elements = static_cast<size_t>(query_slots) * result_capacity;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&result_ids_host),
                             result_elements * sizeof(u32),
                             cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAlloc(persistent result ids)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_result_ids),
                                        result_ids_host, 0),
               "cudaHostGetDevicePointer(persistent result ids)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&result_distances_host),
                             result_elements * sizeof(f32),
                             cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAlloc(persistent result distances)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_result_distances),
                                        result_distances_host, 0),
               "cudaHostGetDevicePointer(persistent result distances)");

    const u64 delta_budget_bytes = static_cast<u64>(config.delta_budget_mb) << 20;
    const u64 delta_record_bytes = sizeof(DeviceDeltaRecord) +
      static_cast<u64>(config.dim) * sizeof(f32) + index.header.rabitq_entry_bytes;
    delta_capacity = static_cast<u32>(std::min<u64>(
      config.max_vectors, std::max<u64>(1, delta_budget_bytes / delta_record_bytes)));
    device_allocate(d_delta_records, delta_capacity, "cudaMalloc(delta records)");
    device_allocate(d_delta_vectors, static_cast<size_t>(delta_capacity) * config.dim,
                    "cudaMalloc(delta vectors)");
    device_allocate(d_delta_rabitq,
                    static_cast<size_t>(delta_capacity) * index.header.rabitq_entry_bytes,
                    "cudaMalloc(delta RaBitQ)");
    device_allocate(d_base_override_epochs, index.header.num_nodes,
                    "cudaMalloc(base override epochs)");
    device_allocate(d_delta_count, 1, "cudaMalloc(delta count)");
    check_cuda(cudaMemset(d_delta_records, 0,
                          static_cast<size_t>(delta_capacity) * sizeof(DeviceDeltaRecord)),
               "cudaMemset(delta records)");
    check_cuda(cudaMemset(d_base_override_epochs, 0,
                          static_cast<size_t>(index.header.num_nodes) * sizeof(u64)),
               "cudaMemset(base override epochs)");
    check_cuda(cudaMemset(d_delta_count, 0, sizeof(u32)), "cudaMemset(delta count)");

    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&stop_host), sizeof(u32), cudaHostAllocMapped),
               "cudaHostAlloc(persistent stop)");
    *stop_host = 0;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&stop_device), stop_host, 0),
               "cudaHostGetDevicePointer(persistent stop)");
    fetch_status_stride = exact_width + 1;
    const size_t status_count = static_cast<size_t>(query_slots) * fetch_status_stride;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&fetch_status_host), status_count * sizeof(i32),
                             cudaHostAllocMapped), "cudaHostAlloc(fetch status)");
    std::fill(fetch_status_host, fetch_status_host + status_count, 0);
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&fetch_status_device),
                                        fetch_status_host, 0),
               "cudaHostGetDevicePointer(fetch status)");

    if (parsed_backend != RemoteBackendKind::gpunetio) {
      backend = create_remote_fetch_backend(parsed_backend, RemoteFetchBackendContext{
        .config = config,
        .channel_context = channel_context,
        .connection_manager = connection_manager,
        .remote_regions = remote_regions,
        .gpu_destination_base = d_exact_vectors,
        .gpu_destination_bytes = remote_buffer_bytes,
      });
    }

    check_cuda(cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(persistent kernel)");
    check_cuda(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(persistent transfer)");
    check_cuda(cudaStreamCreateWithFlags(&delta_stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(delta upload)");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, static_cast<int>(config.gpu_device)),
               "cudaGetDeviceProperties(persistent kernel)");
    gpu_clock_khz = static_cast<u64>(std::max(1, properties.clockRate));
    const u64 requested_blocks = static_cast<u64>(
      std::max(1, properties.multiProcessorCount)) * config.gpu_persistent_blocks_per_sm;
    const u32 blocks = static_cast<u32>(std::min<u64>(query_slots, requested_blocks));
    std::cerr << "[gpu-search] persistent_grid blocks=" << blocks
              << " threads=128 blocks_per_sm_target="
              << config.gpu_persistent_blocks_per_sm << std::endl;
    kernel_params = PersistentKernelParams{
      .submissions = submissions.device_view(),
      .completions = completions.device_view(),
      .fetches = fetches.device_view(),
      .nodes = reinterpret_cast<const DeviceNodeRecord*>(d_nodes),
      .hot_neighbors = d_hot_neighbors,
      .rabitq_entries = d_rabitq_entries,
      .centroid = d_centroid,
      .entry_points = d_entry_points,
      .entry_point_count = static_cast<u32>(index.entry_points.size()),
      .num_nodes = static_cast<u32>(index.header.num_nodes),
      .medoid_id = index.header.medoid_id,
      .dim = config.dim,
      .code_bits = code_bits,
      .code_storage_bytes = format::rabitq_code_storage_bytes(code_bits),
      .rabitq_entry_bytes = index.header.rabitq_entry_bytes,
      .vector_offset = static_cast<u32>(index.shards.front().vector_region_offset - 16),
      .vector_bytes = static_cast<u32>(
        vector_dtype_bytes(config.resolved_vector_dtype(), config.dim)),
      .vector_dtype = static_cast<u32>(config.resolved_vector_dtype()),
      .beam_width = config.beam_width,
      .exact_width = exact_width,
      .max_expansions = config.beam_width,
      .cold_expansions = config.gpu_cold_expansions,
      .visited_capacity = visited_capacity,
      .query_slots = query_slots,
      .graph_page_bytes = index.header.page_bytes,
      .id_encoding_bytes = index.header.id_encoding_bytes,
      .direct_backend = parsed_backend == RemoteBackendKind::gpunetio ? 1u : 0u,
      .direct_region_count = direct_view.remote_region_count,
      .direct_qps_per_node = direct_view.qps_per_node,
      .direct_local_mkey = direct_view.local_mkey,
      .direct_local_iova_base = direct_view.local_iova_base,
      .direct_regions = reinterpret_cast<const DirectRemoteRegion*>(direct_view.remote_regions),
      .direct_qps = direct_view.qp_array,
      .direct_dump = direct_view.dump,
      .delta_records = d_delta_records,
      .delta_vectors = d_delta_vectors,
      .delta_rabitq_entries = d_delta_rabitq,
      .base_override_epochs = d_base_override_epochs,
      .delta_count = d_delta_count,
      .delta_capacity = delta_capacity,
      .delta_signature_filter = config.gpu_delta_signature_filter ? 1u : 0u,
      .stop = stop_device,
      .fetch_status = fetch_status_device,
      .fetch_status_stride = fetch_status_stride,
      .graph_page_cache = d_graph_page_cache,
      .graph_page_cache_keys = d_graph_page_cache_keys,
      .graph_page_cache_locks = d_graph_page_cache_locks,
      .graph_page_cache_slots = graph_page_cache_slots,
      .rotated_queries = d_rotated_queries,
      .query_luts = d_query_luts,
      .beam_ids = d_beam_ids,
      .beam_distances = d_beam_distances,
      .beam_expanded = d_beam_expanded,
      .visited_hash = d_visited,
      .exact_vectors = d_exact_vectors,
      .result_ids = d_result_ids,
      .result_distances = d_result_distances,
    };
    if (parsed_backend != RemoteBackendKind::gpunetio) {
      fetch_thread = std::thread([this] { fetch_loop(); });
    }
    admission_thread = std::thread([this] { admission_loop(); });
    completion_thread = std::thread([this] { completion_loop(); });
    kernel_blocks = blocks;
    start_persistent_kernel();
    maintenance_thread = std::thread([this] { maintenance_loop(); });
  }

  void start_persistent_kernel() {
    std::atomic_ref<u32>(*stop_host).store(0, std::memory_order_release);
    launch_persistent_search(kernel_stream, kernel_params, kernel_blocks, 128);
    check_cuda(cudaPeekAtLastError(), "launch_persistent_search");
    kernel_running = true;
  }

  void stop_persistent_kernel() {
    if (!kernel_running) return;
    std::atomic_ref<u32>(*stop_host).store(1, std::memory_order_release);
    const cudaError_t status = cudaStreamSynchronize(kernel_stream);
    kernel_running = false;
    check_cuda(status, "cudaStreamSynchronize(persistent maintenance stop)");
  }

  ~Impl() {
    accepting.store(false, std::memory_order_release);
    maintenance_shutdown.store(true, std::memory_order_release);
    maintenance_cv.notify_all();
    admission_cv.notify_all();
    slot_cv.notify_all();
    if (maintenance_thread.joinable()) maintenance_thread.join();
    while (pending_count.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    if (kernel_running) {
      std::atomic_ref<u32>(*stop_host).store(1, std::memory_order_release);
      if (kernel_stream != nullptr) cudaStreamSynchronize(kernel_stream);
      kernel_running = false;
    }
    shutdown.store(true, std::memory_order_release);
    admission_cv.notify_all();
    if (admission_thread.joinable()) admission_thread.join();
    if (fetch_thread.joinable()) fetch_thread.join();
    if (completion_thread.joinable()) completion_thread.join();
    if (delta_stream != nullptr) cudaStreamDestroy(delta_stream);
    if (transfer_stream != nullptr) cudaStreamDestroy(transfer_stream);
    if (kernel_stream != nullptr) cudaStreamDestroy(kernel_stream);
    if (fetch_status_host != nullptr) cudaFreeHost(fetch_status_host);
    if (stop_host != nullptr) cudaFreeHost(stop_host);
    if (result_distances_host != nullptr) cudaFreeHost(result_distances_host);
    if (result_ids_host != nullptr) cudaFreeHost(result_ids_host);
    cudaFree(d_delta_count);
    cudaFree(d_base_override_epochs);
    cudaFree(d_delta_rabitq);
    cudaFree(d_delta_vectors);
    cudaFree(d_delta_records);
    cudaFree(d_graph_page_cache_locks);
    cudaFree(d_graph_page_cache_keys);
    if (owns_remote_buffer) cudaFree(d_remote_buffer);
#ifdef DVSTOR_HAVE_GPUNETIO
    direct_transport.reset();
#endif
    cudaFree(d_visited);
    cudaFree(d_beam_expanded);
    cudaFree(d_beam_distances);
    cudaFree(d_beam_ids);
    cudaFree(d_query_luts);
    cudaFree(d_rotated_queries);
    if (query_staging_host != nullptr) cudaFreeHost(query_staging_host);
    cudaFree(d_queries);
    cudaFree(d_entry_points);
    cudaFree(d_centroid);
    cudaFree(d_rabitq_entries);
    cudaFree(d_hot_neighbors);
    cudaFree(d_nodes);
  }

  service::QueryResult search(VectorDType query_dtype, const byte_t* query_data, u32 k) {
    if (!accepting.load(std::memory_order_acquire)) {
      throw std::runtime_error("persistent GPU search engine is stopping");
    }
    if (query_data == nullptr || k == 0 || k > result_capacity) {
      throw std::invalid_argument("invalid persistent GPU query");
    }
    u32 slot = 0;
    {
      std::unique_lock<std::mutex> lock(slot_mutex);
      slot_cv.wait(lock, [&] { return !free_slots.empty() || !accepting.load(); });
      if (!accepting.load()) throw std::runtime_error("persistent GPU search engine stopped");
      slot = free_slots.back();
      free_slots.pop_back();
    }

    f32* decoded = query_staging_host + static_cast<size_t>(slot) * config.dim;
    for (u32 d = 0; d < config.dim; ++d) {
      decoded[d] = vector_component_as_float(query_data, query_dtype, d);
    }

    const u64 request_id = next_request_id.fetch_add(1, std::memory_order_relaxed);
    const auto submitted_at = std::chrono::steady_clock::now();
    auto pending = std::make_shared<PendingQuery>();
    pending->slot = slot;
    pending->capacity = result_capacity;
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
    {
      std::lock_guard<std::mutex> lock(admission_mutex);
      admission_queue.push_back(PendingSubmission{
        .descriptor = descriptor,
        .enqueued_at = submitted_at,
      });
    }
    admission_cv.notify_one();
    engine.telemetry_.queries_submitted.fetch_add(1, std::memory_order_relaxed);
    return future.get();
  }

  void admission_loop() {
    std::vector<PendingSubmission> batch;
    batch.reserve(config.query_batch_target);
    while (!shutdown.load(std::memory_order_acquire)) {
      batch.clear();
      {
        std::unique_lock<std::mutex> lock(admission_mutex);
        admission_cv.wait(lock, [&] {
          return (!admission_queue.empty() && !admission_paused) ||
                 shutdown.load(std::memory_order_acquire);
        });
        if (admission_queue.empty() || admission_paused) continue;
        const auto deadline = admission_queue.front().enqueued_at +
          std::chrono::microseconds(config.query_batch_wait_us);
        admission_cv.wait_until(lock, deadline, [&] {
          return admission_queue.size() >= config.query_batch_min ||
                 admission_paused ||
                 shutdown.load(std::memory_order_acquire);
        });
        if (admission_paused) continue;
        const size_t count = std::min<size_t>(
          admission_queue.size(),
          std::min<u32>(config.query_batch_target, config.query_batch_max));
        for (size_t i = 0; i < count; ++i) {
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
          cudaMemcpyHostToDevice, transfer_stream),
          "cudaMemcpyAsync(persistent query batch)");
      }
      check_cuda(cudaStreamSynchronize(transfer_stream),
                 "cudaStreamSynchronize(persistent query batch)");
      const auto admitted_at = std::chrono::steady_clock::now();
      u64 wait_ns = 0;
      for (PendingSubmission& submission : batch) {
        submission.descriptor.snapshot_epoch = engine.delta_.published_epoch();
        while (!submissions.try_push(submission.descriptor)) std::this_thread::yield();
        wait_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
          admitted_at - submission.enqueued_at).count());
      }
      engine.telemetry_.batches.fetch_add(1, std::memory_order_relaxed);
      engine.telemetry_.batch_queries.fetch_add(batch.size(), std::memory_order_relaxed);
      engine.telemetry_.submission_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
    }
  }

  void encode_mutation_payload(const DeltaMutation& mutation,
                               std::vector<f32>& decoded,
                               std::vector<byte_t>& entry) const {
    std::fill(decoded.begin(), decoded.end(), 0.0f);
    std::fill(entry.begin(), entry.end(), 0);
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
    VamanaNode::RabitqCode code;
    f32 norm = 0.0f;
    f32 error = 0.0f;
    VamanaNode::compute_rabitq_entry(
      reinterpret_cast<const byte_t*>(decoded.data()), VectorDType::float32,
      code, norm, error);
    std::memcpy(entry.data(), code.data(), code.size());
    std::memcpy(entry.data() + VamanaNode::rabitq_code_storage_size(), &norm, sizeof(norm));
    std::memcpy(entry.data() + VamanaNode::rabitq_code_storage_size() + sizeof(norm),
                &error, sizeof(error));
  }

  u32 delta_signature(const std::vector<byte_t>& entry) const {
    u32 signature = 0;
    for (u32 bit = 0; bit < 16; ++bit) {
      const u32 source = bit * code_bits / 16;
      if ((entry[source >> 3] & static_cast<byte_t>(1u << (7u - (source & 7u)))) != 0) {
        signature |= 1u << bit;
      }
    }
    return signature;
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
      const DeltaSnapshot snapshot = engine.delta_.begin_consolidation();
      if (snapshot.mutations.size() > delta_capacity) {
        throw std::runtime_error("GPU delta has more live entries than its configured capacity");
      }

      std::vector<DeviceDeltaRecord> compact_records;
      compact_records.reserve(snapshot.mutations.size());
      std::vector<f32> compact_vectors;
      std::vector<byte_t> compact_entries;
      std::unordered_map<node_t, u32> compact_latest;
      std::unordered_set<node_t> compact_overrides;
      std::vector<node_t> merged_ids;
      std::vector<f32> decoded(config.dim);
      std::vector<byte_t> entry(index.header.rabitq_entry_bytes, 0);
      const u64 zero_epoch = 0;

      for (const DeltaMutation& mutation : snapshot.mutations) {
        const bool deleted = mutation.kind == service::storage_owner::MutationKind::erase;
        const bool merge_into_base = backend_kind != RemoteBackendKind::local &&
          mutation.id < index.header.num_nodes &&
          (deleted || mutation.remote_node != 0);
        encode_mutation_payload(mutation, decoded, entry);
        if (merge_into_base) {
          format::NodeRecord node = index.nodes[mutation.id];
          node.generation = std::max<u32>(1, mutation.generation);
          if (deleted) {
            node.flags |= format::kFlagDeleted;
          } else {
            node.flags &= ~format::kFlagDeleted;
            node.remote_node = mutation.remote_node;
            std::memcpy(index.rabitq_entries.data() +
                          static_cast<size_t>(mutation.id) * index.header.rabitq_entry_bytes,
                        entry.data(), entry.size());
            check_cuda(cudaMemcpyAsync(
              d_rabitq_entries +
                static_cast<size_t>(mutation.id) * index.header.rabitq_entry_bytes,
              index.rabitq_entries.data() +
                static_cast<size_t>(mutation.id) * index.header.rabitq_entry_bytes,
              entry.size(), cudaMemcpyHostToDevice, delta_stream),
              "cudaMemcpyAsync(base RaBitQ merge)");
          }
          index.nodes[mutation.id] = node;
          check_cuda(cudaMemcpyAsync(d_nodes + mutation.id, &index.nodes[mutation.id], sizeof(node),
                                     cudaMemcpyHostToDevice, delta_stream),
                     "cudaMemcpyAsync(base node merge)");
          check_cuda(cudaMemcpyAsync(d_base_override_epochs + mutation.id, &zero_epoch,
                                     sizeof(zero_epoch), cudaMemcpyHostToDevice, delta_stream),
                     "cudaMemcpyAsync(base override reset)");
          base_override_ids.erase(mutation.id);
          merged_ids.push_back(mutation.id);
          continue;
        }

        const u32 slot = static_cast<u32>(compact_records.size());
        DeviceDeltaRecord record{
          .id = mutation.id,
          .generation = std::max<u32>(1, mutation.generation),
          .flags = deleted ? kDeltaDeleted : 0u,
          .signature = deleted ? 0u : delta_signature(entry),
          .epoch = mutation.epoch,
        };
        compact_records.push_back(record);
        compact_vectors.insert(compact_vectors.end(), decoded.begin(), decoded.end());
        compact_entries.insert(compact_entries.end(), entry.begin(), entry.end());
        compact_latest[mutation.id] = slot;
        if (mutation.id < index.header.num_nodes) {
          compact_overrides.insert(mutation.id);
          check_cuda(cudaMemcpyAsync(d_base_override_epochs + mutation.id, &mutation.epoch,
                                     sizeof(mutation.epoch), cudaMemcpyHostToDevice, delta_stream),
                     "cudaMemcpyAsync(compact base override)");
        }
      }

      if (!compact_records.empty()) {
        check_cuda(cudaMemcpyAsync(d_delta_records, compact_records.data(),
                                   compact_records.size() * sizeof(DeviceDeltaRecord),
                                   cudaMemcpyHostToDevice, delta_stream),
                   "cudaMemcpyAsync(compact delta record batch)");
        check_cuda(cudaMemcpyAsync(d_delta_vectors, compact_vectors.data(),
                                   compact_vectors.size() * sizeof(f32),
                                   cudaMemcpyHostToDevice, delta_stream),
                   "cudaMemcpyAsync(compact delta vector batch)");
        check_cuda(cudaMemcpyAsync(d_delta_rabitq, compact_entries.data(),
                                   compact_entries.size(), cudaMemcpyHostToDevice, delta_stream),
                   "cudaMemcpyAsync(compact delta RaBitQ batch)");
      }

      check_cuda(cudaStreamSynchronize(delta_stream),
                 "cudaStreamSynchronize(delta compaction payload)");
      const u32 count = static_cast<u32>(compact_records.size());
      check_cuda(cudaMemcpyAsync(d_delta_count, &count, sizeof(count),
                                 cudaMemcpyHostToDevice, delta_stream),
                 "cudaMemcpyAsync(delta compact count)");
      check_cuda(cudaStreamSynchronize(delta_stream),
                 "cudaStreamSynchronize(delta compaction publish)");
      delta_records_host = std::move(compact_records);
      latest_delta_slot = std::move(compact_latest);
      base_override_ids = std::move(compact_overrides);
      if (!merged_ids.empty()) {
        engine.delta_.complete_partial_consolidation(
          merged_ids, snapshot.base_generation + 1, snapshot.epoch);
      } else {
        engine.delta_.mark_compacted();
      }
      engine.telemetry_.delta_compactions.fetch_add(1, std::memory_order_relaxed);
      engine.telemetry_.base_entries_merged.fetch_add(merged_ids.size(),
                                                      std::memory_order_relaxed);
      engine.telemetry_.delta_live_entries.store(count, std::memory_order_relaxed);
      start_persistent_kernel();
    } catch (...) {
      if (!kernel_running) start_persistent_kernel();
      resume_admission_after_compaction();
      throw;
    }
    resume_admission_after_compaction();
  }

  void maintenance_loop() {
    const auto period = std::chrono::milliseconds(config.merge_period_ms);
    while (!maintenance_shutdown.load(std::memory_order_acquire)) {
      {
        std::unique_lock<std::mutex> lock(maintenance_mutex);
        maintenance_cv.wait_for(lock, period, [&] {
          return maintenance_shutdown.load(std::memory_order_acquire);
        });
      }
      if (maintenance_shutdown.load(std::memory_order_acquire)) break;
      bool compact = false;
      {
        std::lock_guard<std::mutex> lock(delta_mutex);
        const size_t live = engine.delta_.delta_size();
        const size_t history = delta_records_host.size();
        compact = history > live + std::max<size_t>(128, live / 4) ||
          history >= static_cast<size_t>(delta_capacity) * 4 / 5 ||
          engine.delta_.should_consolidate(
            index.header.num_nodes,
            static_cast<size_t>(config.delta_budget_mb) << 20,
            config.delta_max_ratio, 0.8, period);
      }
      if (!compact) continue;
      try {
        std::lock_guard<std::mutex> publish_lock(engine.mutation_publish_mutex_);
        compact_delta();
      } catch (const std::exception& error) {
        std::cerr << "[gpu-search] delta compaction failed: " << error.what() << std::endl;
      }
    }
  }

  void upload_mutations(std::vector<DeltaMutation>& mutations, u64 epoch) {
    if (mutations.empty()) return;
    if (delta_records_host.size() + mutations.size() > delta_capacity) {
      compact_delta();
    }
    std::lock_guard<std::mutex> lock(delta_mutex);
    if (delta_records_host.size() + mutations.size() > delta_capacity) {
      throw std::runtime_error("GPU delta live set exceeds the configured capacity");
    }
    const u32 first_slot = static_cast<u32>(delta_records_host.size());
    std::vector<DeviceDeltaRecord> new_records;
    new_records.reserve(mutations.size());
    std::vector<f32> new_vectors(static_cast<size_t>(mutations.size()) * config.dim);
    std::vector<byte_t> new_entries(
      static_cast<size_t>(mutations.size()) * index.header.rabitq_entry_bytes);
    std::vector<u32> superseded_existing_slots;
    std::vector<u32> new_base_overrides;
    std::vector<f32> decoded(config.dim);
    std::vector<byte_t> entry(index.header.rabitq_entry_bytes, 0);
    for (size_t mutation_index = 0; mutation_index < mutations.size(); ++mutation_index) {
      DeltaMutation& mutation = mutations[mutation_index];
      mutation.epoch = epoch;
      encode_mutation_payload(mutation, decoded, entry);
      const u32 slot = static_cast<u32>(delta_records_host.size());
      auto latest = latest_delta_slot.find(mutation.id);
      if (latest != latest_delta_slot.end()) {
        DeviceDeltaRecord& previous = delta_records_host[latest->second];
        previous.superseded_epoch = epoch;
        if (latest->second >= first_slot) {
          new_records[latest->second - first_slot].superseded_epoch = epoch;
        } else {
          superseded_existing_slots.push_back(latest->second);
        }
      }

      DeviceDeltaRecord record{
        .id = mutation.id,
        .generation = mutation.generation == 0 ? 1u : mutation.generation,
        .flags = mutation.kind == service::storage_owner::MutationKind::erase
          ? kDeltaDeleted : 0u,
        .signature = mutation.kind == service::storage_owner::MutationKind::erase
          ? 0u : delta_signature(entry),
        .epoch = epoch,
      };
      delta_records_host.push_back(record);
      new_records.push_back(record);
      latest_delta_slot[mutation.id] = slot;
      std::copy(decoded.begin(), decoded.end(),
                new_vectors.begin() + static_cast<size_t>(mutation_index) * config.dim);
      std::copy(entry.begin(), entry.end(),
                new_entries.begin() +
                  static_cast<size_t>(mutation_index) * index.header.rabitq_entry_bytes);
      if (mutation.id < index.header.num_nodes &&
          base_override_ids.insert(mutation.id).second) {
        new_base_overrides.push_back(mutation.id);
      }
    }

    check_cuda(cudaMemcpyAsync(d_delta_records + first_slot, new_records.data(),
                               new_records.size() * sizeof(DeviceDeltaRecord),
                               cudaMemcpyHostToDevice, delta_stream),
               "cudaMemcpyAsync(delta record batch)");
    check_cuda(cudaMemcpyAsync(
      d_delta_vectors + static_cast<size_t>(first_slot) * config.dim,
      new_vectors.data(), new_vectors.size() * sizeof(f32),
      cudaMemcpyHostToDevice, delta_stream),
      "cudaMemcpyAsync(delta vector batch)");
    check_cuda(cudaMemcpyAsync(
      d_delta_rabitq + static_cast<size_t>(first_slot) * index.header.rabitq_entry_bytes,
      new_entries.data(), new_entries.size(), cudaMemcpyHostToDevice, delta_stream),
      "cudaMemcpyAsync(delta RaBitQ batch)");
    for (const u32 slot : superseded_existing_slots) {
      launch_supersede_delta_record(delta_stream, d_delta_records, slot, epoch);
    }
    for (const u32 id : new_base_overrides) {
      launch_publish_base_override(delta_stream, d_base_override_epochs, id, epoch);
    }
    const u32 count = static_cast<u32>(delta_records_host.size());
    launch_publish_delta_count(delta_stream, d_delta_count, count);
    check_cuda(cudaPeekAtLastError(), "launch GPU delta publication");
    check_cuda(cudaStreamSynchronize(delta_stream), "cudaStreamSynchronize(delta publish)");
    engine.telemetry_.delta_live_entries.store(count, std::memory_order_relaxed);
  }

  void fetch_loop() {
    std::vector<FetchDescriptor> batch;
    std::vector<i32> statuses;
    batch.reserve(std::max<u32>(config.query_batch_target * exact_width, 64));
    while (!shutdown.load(std::memory_order_acquire)) {
      batch.clear();
      FetchDescriptor request;
      if (!fetches.try_pop(request)) {
        std::this_thread::yield();
        continue;
      }
      batch.push_back(request);
      const size_t target = std::max<size_t>(64, config.query_batch_target * exact_width);
      while (batch.size() < target && fetches.try_pop(request)) batch.push_back(request);
      statuses.assign(batch.size(), -EIO);
      try {
        backend->fetch(batch, statuses);
      } catch (...) {
        std::fill(statuses.begin(), statuses.end(), -EIO);
      }
      u64 bytes = 0;
      u64 graph_pages = 0;
      u64 exact_vectors = 0;
      for (size_t i = 0; i < batch.size(); ++i) {
        bytes += batch[i].bytes;
        if (batch[i].kind == static_cast<u8>(FetchKind::graph_page)) {
          ++graph_pages;
        } else {
          ++exact_vectors;
        }
        std::atomic_ref<i32>(fetch_status_host[batch[i].sequence]).store(
          statuses[i], std::memory_order_release);
      }
      engine.telemetry_.rdma_read_ops.fetch_add(batch.size(), std::memory_order_relaxed);
      engine.telemetry_.rdma_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
      engine.telemetry_.graph_page_requests.fetch_add(graph_pages, std::memory_order_relaxed);
      engine.telemetry_.exact_vector_reads.fetch_add(exact_vectors, std::memory_order_relaxed);
    }
  }

  void completion_loop() {
    while (!shutdown.load(std::memory_order_acquire) ||
           pending_count.load(std::memory_order_acquire) != 0) {
      CompletionDescriptor completion;
      if (!completions.try_pop(completion)) {
        std::this_thread::yield();
        continue;
      }
      std::shared_ptr<PendingQuery> pending;
      {
        std::lock_guard<std::mutex> lock(pending_mutex);
        auto it = pending_queries.find(completion.request_id);
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
      try {
        if (completion.status != 0) {
          throw std::runtime_error("persistent GPU query failed with status " +
                                   std::to_string(completion.status));
        }
        const size_t result_offset = static_cast<size_t>(pending->slot) * result_capacity;
        std::vector<u32> ids(result_ids_host + result_offset,
                             result_ids_host + result_offset + completion.result_count);
        std::vector<f32> distances(result_distances_host + result_offset,
                                   result_distances_host + result_offset + completion.result_count);
        service::QueryResult result;
        result.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) result.push_back({ids[i], distances[i]});
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
      engine.telemetry_.gpu_active_ns.fetch_add(
        completion.gpu_cycles * 1000000ULL / gpu_clock_khz,
        std::memory_order_relaxed);
      engine.telemetry_.completion_wait_ns.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - pending->submitted_at).count()),
        std::memory_order_relaxed);
      if (completion.snapshot_epoch != 0) {
        engine.telemetry_.delta_queries.fetch_add(1, std::memory_order_relaxed);
      }
      if (backend_kind == RemoteBackendKind::gpunetio) {
        engine.telemetry_.rdma_read_ops.fetch_add(
          static_cast<u64>(completion.exact_vectors) + completion.remote_pages,
                                                  std::memory_order_relaxed);
        engine.telemetry_.rdma_read_bytes.fetch_add(
          static_cast<u64>(completion.exact_vectors) *
            vector_dtype_bytes(config.resolved_vector_dtype(), config.dim) +
            static_cast<u64>(completion.remote_pages) * index.header.page_bytes,
          std::memory_order_relaxed);
        engine.telemetry_.exact_vector_reads.fetch_add(completion.exact_vectors,
                                                       std::memory_order_relaxed);
        engine.telemetry_.graph_page_requests.fetch_add(completion.remote_pages,
                                                        std::memory_order_relaxed);
      }
      engine.telemetry_.graph_page_cache_hits.fetch_add(completion.cache_hits,
                                                        std::memory_order_relaxed);
    }
  }

  PersistentSearchEngine& engine;
  configuration::IndexConfiguration& config;
  format::View index;
  RemoteBackendKind backend_kind{RemoteBackendKind::local};
  std::unique_ptr<RemoteFetchBackend> backend;
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
  MappedRing<FetchDescriptor> fetches;
  u32 query_slots{};
  u32 result_capacity{};
  u32 exact_width{};
  u32 code_bits{};
  u32 visited_capacity{};
  u64 gpu_clock_khz{1};
  format::NodeRecord* d_nodes{};
  u32* d_hot_neighbors{};
  byte_t* d_rabitq_entries{};
  f32* d_centroid{};
  u32* d_entry_points{};
  f32* d_queries{};
  f32* query_staging_host{};
  f32* d_rotated_queries{};
  f32* d_query_luts{};
  u32* d_beam_ids{};
  f32* d_beam_distances{};
  u8* d_beam_expanded{};
  u32* d_visited{};
  byte_t* d_exact_vectors{};
  byte_t* d_remote_buffer{};
  byte_t* d_graph_page_cache{};
  u64* d_graph_page_cache_keys{};
  u32* d_graph_page_cache_locks{};
  u32 graph_page_cache_slots{};
  size_t graph_page_cache_bytes{};
  bool owns_remote_buffer{};
  u32* result_ids_host{};
  f32* result_distances_host{};
  u32* d_result_ids{};
  f32* d_result_distances{};
  DeviceDeltaRecord* d_delta_records{};
  f32* d_delta_vectors{};
  byte_t* d_delta_rabitq{};
  u64* d_base_override_epochs{};
  u32* d_delta_count{};
  u32 delta_capacity{};
  std::vector<DeviceDeltaRecord> delta_records_host;
  std::unordered_map<node_t, u32> latest_delta_slot;
  std::unordered_set<node_t> base_override_ids;
  std::mutex delta_mutex;
  u32* stop_host{};
  u32* stop_device{};
  i32* fetch_status_host{};
  i32* fetch_status_device{};
  u32 fetch_status_stride{};
  cudaStream_t kernel_stream{};
  cudaStream_t transfer_stream{};
  cudaStream_t delta_stream{};
  PersistentKernelParams kernel_params{};
  u32 kernel_blocks{};
  bool kernel_running{};
  std::atomic<bool> accepting{true};
  std::atomic<bool> shutdown{false};
  std::atomic<bool> maintenance_shutdown{false};
  std::atomic<u64> active_gpu_queries{0};
  std::atomic<u64> next_request_id{1};
  std::atomic<u64> pending_count{0};
  std::mutex admission_mutex;
  std::condition_variable admission_cv;
  std::deque<PendingSubmission> admission_queue;
  bool admission_paused{};
  std::mutex slot_mutex;
  std::condition_variable slot_cv;
  std::vector<u32> free_slots;
  std::mutex pending_mutex;
  std::unordered_map<u64, std::shared_ptr<PendingQuery>> pending_queries;
  std::thread admission_thread;
  std::thread fetch_thread;
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

bool PersistentSearchEngine::publish_mutations(std::vector<DeltaMutation> mutations, u64 epoch) {
  std::lock_guard<std::mutex> publish_lock(mutation_publish_mutex_);
  const size_t mutation_count = mutations.size();
  std::chrono::steady_clock::time_point oldest = std::chrono::steady_clock::now();
  for (const auto& mutation : mutations) {
    if (mutation.enqueued_at != std::chrono::steady_clock::time_point{}) {
      oldest = std::min(oldest, mutation.enqueued_at);
    }
  }
  impl_->upload_mutations(mutations, epoch);
  const auto now = std::chrono::steady_clock::now();
  if (!delta_.publish(std::move(mutations), epoch, now)) return false;
  const u64 visibility_ns = static_cast<u64>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(now - oldest).count());
  telemetry_.mutations_published.fetch_add(mutation_count, std::memory_order_relaxed);
  telemetry_.visibility_ns_total.fetch_add(visibility_ns * mutation_count,
                                           std::memory_order_relaxed);
  u64 current_max = telemetry_.visibility_ns_max.load(std::memory_order_relaxed);
  while (current_max < visibility_ns &&
         !telemetry_.visibility_ns_max.compare_exchange_weak(
           current_max, visibility_ns, std::memory_order_relaxed)) {}
  return true;
}

RemoteBackendKind PersistentSearchEngine::backend_kind() const {
  return impl_->backend_kind;
}

void PersistentSearchEngine::reset_telemetry() {
  telemetry_.reset();
  telemetry_.delta_live_entries.store(delta_.delta_size(), std::memory_order_relaxed);
}

}  // namespace gpu_search
