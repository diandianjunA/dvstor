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
#ifdef DVSTOR_HAVE_GPUNETIO
#include "gpu/gpunetio_query_engine.hh"
#include "gpu/gpunetio_query_launcher.hh"
#endif
#include "gpu_search/index_format.hh"
#include "gpu_search/memory_budget.hh"
#include "gpu_search/persistent_kernel.hh"
#include "gpu_search/remote_fetch_backend.hh"
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
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), operation);
}

template <class T>
void device_free(T*& pointer) {
  if (pointer != nullptr) cudaFree(pointer);
  pointer = nullptr;
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
  std::vector<u32> shard_offsets;

  u32 count() const { return dim == 0 ? 0 : static_cast<u32>(vectors.size() / dim); }
};

AnchorTable load_anchor_table(const filepath_t& prefix, u32 expected_dim,
                              u32 expected_shards) {
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
      decode_storage_vector_to_float(raw.data(), dtype, header.dim, decoded.data());
      result.vectors.insert(result.vectors.end(), decoded.begin(), decoded.end());
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

  void bind_cuda_device(const char* operation) const {
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice(GPU V4)");
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
        fetches(config.query_batch_max *
                  (kPersistentMaxExact + kPersistentMaxPrefetch) * 4,
                MappedRing<FetchDescriptor>::Direction::device_to_host) {
    EngineKind engine_kind;
    RemoteBackendKind parsed_backend;
    if (!parse_engine_kind(config.search_engine, engine_kind) ||
        engine_kind != EngineKind::gpu_persistent ||
        !parse_remote_backend_kind(config.gpu_rdma_backend, parsed_backend)) {
      throw std::invalid_argument("invalid persistent search engine configuration");
    }
    bind_cuda_device("cudaSetDevice(GPU V4 construction)");
    backend_kind = parsed_backend;
    if (config.beam_width > kPersistentMaxBeam ||
        config.rabitq_gate_max_width > kPersistentMaxExact ||
        config.R > kPersistentMaxGraphDegree) {
      throw std::invalid_argument("GPU V4 beam/exact/degree limit exceeded");
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
    std::cerr << "[gpu-search] synthesized V4 manifest in memory from metadata"
              << (used_anchor_entry_points ? " and anchors\n" : "\n");
    const bool centroid_matches = index.centroid.size() == VamanaNode::rabitq_centroid.size() &&
      std::equal(index.centroid.begin(), index.centroid.end(),
                 VamanaNode::rabitq_centroid.begin(), [](f32 stored, f32 configured) {
        const f32 scale = std::max({1.0f, std::abs(stored), std::abs(configured)});
        return std::abs(stored - configured) <= 1e-6f * scale;
      });
    if (index.header.dim != config.dim || index.header.graph_degree != config.R ||
        index.header.num_shards != remote_regions.size() ||
        index.header.rabitq_code_bits != VamanaNode::rabitq_code_bits() ||
        index.header.rabitq_code_bits > kPersistentMaxCodeBits ||
        index.header.rabitq_entry_bytes != VamanaNode::rabitq_entry_size() ||
        index.header.graph_entry_bytes != VamanaNode::hot_graph_entry_size() ||
        index.header.graph_shard_bits != VamanaNode::HOT_GRAPH_SHARD_BITS ||
        index.header.vector_dtype != static_cast<u32>(config.resolved_vector_dtype()) ||
        index.entry_points.size() > kPersistentMaxEntryPoints || !centroid_matches) {
      throw std::runtime_error("GPU V4 manifest does not match runtime storage metadata");
    }

    anchor_table = load_anchor_table(config.resolved_index_prefix(), config.dim,
                                     index.header.num_shards);
    query_slots = config.query_batch_max;
    result_capacity = std::max<u32>(config.k, config.rabitq_gate_max_width);
    const u64 scaled_gate_width = static_cast<u64>(config.rabitq_gate_max_width) *
      std::max<u32>(1, config.gpu_graph_prefetch_depth);
    exact_width = static_cast<u32>(std::min<u64>(
      kPersistentMaxExact, std::max<u64>(result_capacity, scaled_gate_width)));
    code_bits = index.header.rabitq_code_bits;
    free_slots.resize(query_slots);
    for (u32 slot = 0; slot < query_slots; ++slot) free_slots[slot] = slot;

    node_record_bytes = static_cast<u32>(8 + VamanaNode::vector_bytes());
    const u64 engine_budget = static_cast<u64>(
      config.gpu_memory_limit_gb - config.gpu_memory_reserve_gb) << 30;
    size_t free_gpu_bytes = 0;
    size_t total_gpu_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes), "cudaMemGetInfo(GPU V4 budget)");
    const u64 runtime_reserve = static_cast<u64>(config.gpu_memory_reserve_gb) << 30;
    const u64 physically_available = free_gpu_bytes > runtime_reserve
      ? static_cast<u64>(free_gpu_bytes) - runtime_reserve : 0;
    const u64 usable_budget = std::min(engine_budget, physically_available);
    const auto budget = memory_budget::estimate(memory_budget::Request{
      .nodes = index.header.num_nodes,
      .max_delta_vectors = config.max_vectors,
      .usable_bytes = usable_budget,
      .requested_cache_bytes = static_cast<u64>(config.gpu_adjacency_cache_mb) << 20,
      .requested_exact_cache_bytes = static_cast<u64>(config.gpu_exact_cache_mb) << 20,
      .delta_budget_bytes = static_cast<u64>(config.delta_budget_mb) << 20,
      .dim = config.dim,
      .code_bits = code_bits,
      .code_entry_bytes = index.header.rabitq_entry_bytes,
      .vector_bytes = static_cast<u32>(VamanaNode::vector_bytes()),
      .query_slots = query_slots,
      .beam_width = config.beam_width,
      .graph_degree = config.R,
      .exact_width = exact_width,
      .exact_record_bytes = node_record_bytes,
      .anchor_count = anchor_table.count(),
      .shard_count = static_cast<u32>(index.shards.size()),
      .entry_point_count = static_cast<u32>(index.entry_points.size()),
      .cache_ways = config.gpu_adjacency_cache_ways,
      .exact_cache_ways = config.gpu_exact_cache_ways,
    });
    if (!budget.fits) {
      throw std::runtime_error(
        "GPU V4 allocations exceed the configured memory budget; codes=" +
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
      throw std::runtime_error("GPU V4 graph invalidation capacity exceeds uint32");
    }
    graph_invalidation_capacity = static_cast<u32>(std::max<u64>(1, invalidation_capacity));
    explicit_gpu_bytes = budget.explicit_bytes;
    const u64 code_bytes = budget.code_bytes;
    const u64 exact_bytes = budget.exact_bytes;
    std::cerr << "[gpu-search] V4 budget codes=" << budget.code_bytes
              << " delta=" << budget.delta_bytes
              << " adjacency_total=" << budget.cache_total_bytes
              << " exact_cache_total=" << budget.exact_cache_total_bytes
              << " explicit=" << explicit_gpu_bytes
              << " limit=" << engine_budget << " bytes\n";

    const size_t code_region_bytes = static_cast<size_t>(code_bytes);
    exact_region_offset = static_cast<size_t>(align_up(code_region_bytes, 256));
    exact_cache_offset = static_cast<size_t>(align_up(exact_region_offset + exact_bytes, 256));
    graph_cache_offset = static_cast<size_t>(
      align_up(exact_cache_offset + exact_cache_bytes, 512));
    const size_t remote_buffer_bytes = graph_cache_offset + graph_cache_bytes;
    if (parsed_backend == RemoteBackendKind::gpunetio) {
#ifdef DVSTOR_HAVE_GPUNETIO
      direct_transport = std::make_unique<gpu::GpuNetioPersistentTransport>(
        config, remote_buffer_bytes, channel_context, connection_manager, remote_regions);
      direct_view = direct_transport->view();
      if (direct_view.data == nullptr || direct_view.data_bytes < remote_buffer_bytes) {
        throw std::runtime_error("GPUNetIO returned an undersized GPU V4 data region");
      }
      d_remote_buffer = direct_view.data;
      owns_remote_buffer = false;
#else
      throw std::runtime_error("DVSTOR was built without DOCA GPUNetIO support");
#endif
    } else {
      device_allocate(d_remote_buffer, remote_buffer_bytes, "cudaMalloc(GPU V4 remote buffer)");
      owns_remote_buffer = true;
    }
    d_rabitq_entries = d_remote_buffer;
    d_exact_records = d_remote_buffer + exact_region_offset;
    d_exact_cache = d_remote_buffer + exact_cache_offset;
    d_graph_cache = d_remote_buffer + graph_cache_offset;

    if (parsed_backend == RemoteBackendKind::gpunetio) {
      backend = create_gpunetio_fallback_backend(RemoteFetchBackendContext{
        .config = config,
        .channel_context = channel_context,
        .connection_manager = connection_manager,
        .remote_regions = remote_regions,
        .gpu_destination_base = d_remote_buffer,
        .gpu_destination_bytes = remote_buffer_bytes,
      });
      std::cerr << "[gpu-search] bootstrap=CPU-posted GPUDirect RDMA; "
                   "steady_state=strict GPU-initiated GPUNetIO\n";
    } else {
      backend = create_remote_fetch_backend(parsed_backend, RemoteFetchBackendContext{
        .config = config,
        .channel_context = channel_context,
        .connection_manager = connection_manager,
        .remote_regions = remote_regions,
        .gpu_destination_base = d_remote_buffer,
        .gpu_destination_bytes = remote_buffer_bytes,
      });
    }
    stream_codes_to_gpu(*backend);

    device_allocate(d_shards, index.shards.size(), "cudaMalloc(GPU V4 shards)");
    device_allocate(d_centroid, index.centroid.size(), "cudaMalloc(GPU V4 centroid)");
    device_allocate(d_entry_points, index.entry_points.size(), "cudaMalloc(GPU V4 entries)");
    check_cuda(cudaMemcpy(d_shards, index.shards.data(),
                          index.shards.size() * sizeof(format::ShardRegion),
                          cudaMemcpyHostToDevice), "cudaMemcpy(GPU V4 shards)");
    check_cuda(cudaMemcpy(d_centroid, index.centroid.data(),
                          index.centroid.size() * sizeof(f32), cudaMemcpyHostToDevice),
               "cudaMemcpy(GPU V4 centroid)");
    check_cuda(cudaMemcpy(d_entry_points, index.entry_points.data(),
                          index.entry_points.size() * sizeof(u32), cudaMemcpyHostToDevice),
               "cudaMemcpy(GPU V4 entries)");
    if (!anchor_table.vectors.empty()) {
      device_allocate(d_anchor_vectors, anchor_table.vectors.size(),
                      "cudaMalloc(GPU V4 anchors)");
      check_cuda(cudaMemcpy(d_anchor_vectors, anchor_table.vectors.data(),
                            anchor_table.vectors.size() * sizeof(f32), cudaMemcpyHostToDevice),
                 "cudaMemcpy(GPU V4 anchors)");
      device_allocate(d_anchor_distances,
                      static_cast<size_t>(query_slots) * anchor_table.count(),
                      "cudaMalloc(GPU V4 anchor distances)");
      device_allocate(d_delta_bucket_heads, anchor_table.count(),
                      "cudaMalloc(GPU V4 delta buckets)");
      check_cuda(cudaMemset(d_delta_bucket_heads, 0xff,
                            static_cast<size_t>(anchor_table.count()) * sizeof(u32)),
                 "cudaMemset(GPU V4 delta buckets)");
    }

    device_allocate(d_queries, static_cast<size_t>(query_slots) * config.dim,
                    "cudaMalloc(GPU V4 queries)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&query_staging_host),
                             static_cast<size_t>(query_slots) * config.dim * sizeof(f32),
                             cudaHostAllocPortable), "cudaHostAlloc(GPU V4 query staging)");
    device_allocate(d_rotated_queries, static_cast<size_t>(query_slots) * code_bits,
                    "cudaMalloc(GPU V4 rotated queries)");
    device_allocate(d_query_luts,
                    static_cast<size_t>(query_slots) * (code_bits / 8) * 256,
                    "cudaMalloc(GPU V4 query LUTs)");
    device_allocate(d_beam_handles, static_cast<size_t>(query_slots) * config.beam_width,
                    "cudaMalloc(GPU V4 beam handles)");
    device_allocate(d_beam_ids, static_cast<size_t>(query_slots) * config.beam_width,
                    "cudaMalloc(GPU V4 beam ids)");
    device_allocate(d_beam_distances, static_cast<size_t>(query_slots) * config.beam_width,
                    "cudaMalloc(GPU V4 beam distances)");
    device_allocate(d_beam_expanded, static_cast<size_t>(query_slots) * config.beam_width,
                    "cudaMalloc(GPU V4 beam state)");
    device_allocate(d_visited, static_cast<size_t>(query_slots) * visited_capacity,
                    "cudaMalloc(GPU V4 visited)");

    device_allocate(d_graph_cache_keys, graph_cache_slots, "cudaMalloc(V4 cache keys)");
    device_allocate(d_graph_cache_generations, graph_cache_slots,
                    "cudaMalloc(V4 cache generations)");
    device_allocate(d_graph_cache_timestamps, graph_cache_slots,
                    "cudaMalloc(V4 cache timestamps)");
    device_allocate(d_graph_cache_states, graph_cache_slots, "cudaMalloc(V4 cache states)");
    device_allocate(d_graph_cache_readers, graph_cache_slots, "cudaMalloc(V4 cache readers)");
    device_allocate(d_graph_cache_victims, graph_cache_sets, "cudaMalloc(V4 cache victims)");
    device_allocate(d_graph_cache_generation, 1, "cudaMalloc(V4 cache generation)");
    device_allocate(d_graph_invalidation_keys, graph_invalidation_capacity,
                    "cudaMalloc(V4 graph invalidation keys)");
    check_cuda(cudaMemset(d_graph_cache_states, 0,
                          static_cast<size_t>(graph_cache_slots) * sizeof(u32)),
               "cudaMemset(V4 cache states)");
    check_cuda(cudaMemset(d_graph_cache_readers, 0,
                          static_cast<size_t>(graph_cache_slots) * sizeof(u32)),
               "cudaMemset(V4 cache readers)");
    check_cuda(cudaMemset(d_graph_cache_victims, 0,
                          static_cast<size_t>(graph_cache_sets) * sizeof(u32)),
               "cudaMemset(V4 cache victims)");
    const u64 initial_cache_generation = 1;
    check_cuda(cudaMemcpy(d_graph_cache_generation, &initial_cache_generation,
                          sizeof(initial_cache_generation), cudaMemcpyHostToDevice),
               "cudaMemcpy(V4 cache generation)");

    device_allocate(d_exact_cache_keys, exact_cache_slots,
                    "cudaMalloc(V4 exact-cache keys)");
    device_allocate(d_exact_cache_states, exact_cache_slots,
                    "cudaMalloc(V4 exact-cache states)");
    device_allocate(d_exact_cache_readers, exact_cache_slots,
                    "cudaMalloc(V4 exact-cache readers)");
    device_allocate(d_exact_cache_victims, exact_cache_sets,
                    "cudaMalloc(V4 exact-cache victims)");
    check_cuda(cudaMemset(d_exact_cache_states, 0,
                          static_cast<size_t>(exact_cache_slots) * sizeof(u32)),
               "cudaMemset(V4 exact-cache states)");
    check_cuda(cudaMemset(d_exact_cache_readers, 0,
                          static_cast<size_t>(exact_cache_slots) * sizeof(u32)),
               "cudaMemset(V4 exact-cache readers)");
    check_cuda(cudaMemset(d_exact_cache_victims, 0,
                          static_cast<size_t>(exact_cache_sets) * sizeof(u32)),
               "cudaMemset(V4 exact-cache victims)");

    const size_t result_elements = static_cast<size_t>(query_slots) * result_capacity;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&result_ids_host),
                             result_elements * sizeof(u32),
                             cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAlloc(GPU V4 result ids)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_result_ids),
                                        result_ids_host, 0),
               "cudaHostGetDevicePointer(GPU V4 result ids)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&result_distances_host),
                             result_elements * sizeof(f32),
                             cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAlloc(GPU V4 result distances)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_result_distances),
                                        result_distances_host, 0),
               "cudaHostGetDevicePointer(GPU V4 result distances)");

    device_allocate(d_delta_records, delta_capacity, "cudaMalloc(V4 delta records)");
    device_allocate(d_delta_vectors, static_cast<size_t>(delta_capacity) * config.dim,
                    "cudaMalloc(V4 delta vectors)");
    device_allocate(d_delta_rabitq,
                    static_cast<size_t>(delta_capacity) * index.header.rabitq_entry_bytes,
                    "cudaMalloc(V4 delta codes)");
    device_allocate(d_delta_next, delta_capacity, "cudaMalloc(V4 delta links)");
    device_allocate(d_base_override_keys, delta_table_capacity,
                    "cudaMalloc(V4 override keys)");
    device_allocate(d_base_override_epochs, delta_table_capacity,
                    "cudaMalloc(V4 override epochs)");
    device_allocate(d_delta_remote_keys, delta_table_capacity,
                    "cudaMalloc(V4 delta remote keys)");
    device_allocate(d_delta_remote_slots, delta_table_capacity,
                    "cudaMalloc(V4 delta remote slots)");
    device_allocate(d_delta_count, 1, "cudaMalloc(V4 delta count)");
    clear_delta_device_state();

    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&stop_host), sizeof(u32),
                             cudaHostAllocMapped), "cudaHostAlloc(GPU V4 stop)");
    *stop_host = 0;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&stop_device), stop_host, 0),
               "cudaHostGetDevicePointer(GPU V4 stop)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&direct_disabled_host), sizeof(u32),
                             cudaHostAllocMapped),
               "cudaHostAlloc(GPU V4 direct fallback)");
    *direct_disabled_host = 0;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&direct_disabled_device),
                                        direct_disabled_host, 0),
               "cudaHostGetDevicePointer(GPU V4 direct fallback)");
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&direct_error_host), sizeof(i32),
                             cudaHostAllocMapped),
               "cudaHostAlloc(GPU V4 direct error)");
    *direct_error_host = 0;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&direct_error_device),
                                        direct_error_host, 0),
               "cudaHostGetDevicePointer(GPU V4 direct error)");
    fetch_status_stride = exact_width + config.gpu_graph_prefetch_depth;
    const size_t status_count = static_cast<size_t>(query_slots) * fetch_status_stride;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&fetch_status_host),
                             status_count * sizeof(i32), cudaHostAllocMapped),
               "cudaHostAlloc(GPU V4 fetch status)");
    std::fill(fetch_status_host, fetch_status_host + status_count, 0);
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&fetch_status_device),
                                        fetch_status_host, 0),
               "cudaHostGetDevicePointer(GPU V4 fetch status)");

    check_cuda(cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking),
               "cudaStreamCreate(GPU V4 kernel)");
    check_cuda(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking),
               "cudaStreamCreate(GPU V4 transfer)");
    check_cuda(cudaStreamCreateWithFlags(&delta_stream, cudaStreamNonBlocking),
               "cudaStreamCreate(GPU V4 delta)");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, static_cast<int>(config.gpu_device)),
               "cudaGetDeviceProperties(GPU V4)");
    gpu_clock_khz = static_cast<u64>(std::max(1, properties.clockRate));
    const u64 requested_blocks = static_cast<u64>(
      std::max(1, properties.multiProcessorCount)) * config.gpu_persistent_blocks_per_sm;
    kernel_blocks = static_cast<u32>(std::min<u64>(query_slots, requested_blocks));

    kernel_params = PersistentKernelParams{
      .submissions = submissions.device_view(),
      .completions = completions.device_view(),
      .fetches = fetches.device_view(),
      .shards = d_shards,
      .num_shards = static_cast<u32>(index.shards.size()),
      .rabitq_entries = d_rabitq_entries,
      .centroid = d_centroid,
      .entry_points = d_entry_points,
      .entry_point_count = static_cast<u32>(index.entry_points.size()),
      .num_nodes = static_cast<u32>(index.header.num_nodes),
      .medoid_ordinal = index.header.medoid_ordinal,
      .dim = config.dim,
      .code_bits = code_bits,
      .code_storage_bytes = format::rabitq_code_storage_bytes(code_bits),
      .rabitq_entry_bytes = index.header.rabitq_entry_bytes,
      .graph_entry_bytes = index.header.graph_entry_bytes,
      .graph_degree = index.header.graph_degree,
      .graph_shard_bits = index.header.graph_shard_bits,
      .node_meta_offset = static_cast<u32>(VamanaNode::offset_id()),
      .node_record_bytes = node_record_bytes,
      .vector_bytes = static_cast<u32>(VamanaNode::vector_bytes()),
      .vector_dtype = static_cast<u32>(config.resolved_vector_dtype()),
      .beam_width = config.beam_width,
      .exact_width = exact_width,
      .gate_width = config.rabitq_gate_width,
      .gate_max_width = config.rabitq_gate_max_width,
      .gate_margin = static_cast<f32>(config.rabitq_gate_margin),
      .warmup_exact_expansions = config.rabitq_warmup_exact_expansions,
      .audit_period = config.rabitq_audit_period,
      .max_expansions = config.beam_width,
      .prefetch_depth = config.gpu_graph_prefetch_depth,
      .visited_capacity = visited_capacity,
      .query_slots = query_slots,
      .direct_backend = parsed_backend == RemoteBackendKind::gpunetio ? 1u : 0u,
      .direct_region_count = direct_view.remote_region_count,
      .direct_qps_per_node = direct_view.qps_per_node,
      .direct_local_mkey = direct_view.local_mkey,
      .direct_local_iova_base = direct_view.local_iova_base,
      .direct_timeout_ns = 20000000ULL,
      .direct_regions = reinterpret_cast<const DirectRemoteRegion*>(direct_view.remote_regions),
      .direct_qps = direct_view.qp_array,
      .direct_dump = direct_view.dump,
      .direct_disabled = direct_disabled_device,
      .direct_error = direct_error_device,
      .delta_records = d_delta_records,
      .delta_vectors = d_delta_vectors,
      .delta_rabitq_entries = d_delta_rabitq,
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
      .anchor_vectors = d_anchor_vectors,
      .anchor_count = anchor_table.count(),
      .delta_anchor_probes = config.gpu_delta_anchor_probes,
      .anchor_distances = d_anchor_distances,
      .stop = stop_device,
      .fetch_status = fetch_status_device,
      .fetch_status_stride = fetch_status_stride,
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
      .rotated_queries = d_rotated_queries,
      .query_luts = d_query_luts,
      .beam_handles = d_beam_handles,
      .beam_ids = d_beam_ids,
      .beam_distances = d_beam_distances,
      .beam_expanded = d_beam_expanded,
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
    if (backend != nullptr) {
      fetch_thread = std::thread([this] { fetch_loop(); });
    }
    admission_thread = std::thread([this] { admission_loop(); });
    completion_thread = std::thread([this] { completion_loop(); });
    start_persistent_kernel();
    maintenance_thread = std::thread([this] { maintenance_loop(); });
  }

  void stream_codes_to_gpu(RemoteFetchBackend& source) {
    const u64 window_bytes = static_cast<u64>(config.gpu_bootstrap_window_mb) << 20;
    std::vector<FetchDescriptor> requests;
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
          requests.push_back(FetchDescriptor{
            .remote_offset = shard.code_remote_offset + offset,
            .destination_address = reinterpret_cast<u64>(d_rabitq_entries +
              shard.ordinal_base * index.header.rabitq_entry_bytes + offset),
            .bytes = bytes,
            .memory_node = static_cast<u16>(shard.memory_node),
            .kind = static_cast<u8>(FetchKind::code),
          });
          offset += bytes;
        }
        statuses.assign(requests.size(), -EIO);
        source.fetch(requests, statuses);
        for (size_t request_index = 0; request_index < statuses.size(); ++request_index) {
          if (statuses[request_index] <= 0) {
            const FetchDescriptor& request = requests[request_index];
            throw std::runtime_error(
              "RDMA V4 code bootstrap failed: status=" +
              std::to_string(statuses[request_index]) + " shard=" +
              std::to_string(request.memory_node) + " remote_offset=" +
              std::to_string(request.remote_offset) + " bytes=" +
              std::to_string(request.bytes) + " destination=" +
              std::to_string(request.destination_address));
          }
        }
        for (const FetchDescriptor& request : requests) streamed += request.bytes;
      }
    }
    const u64 expected = index.header.num_nodes * index.header.rabitq_entry_bytes;
    if (streamed != expected) throw std::runtime_error("GPU V4 code bootstrap size mismatch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(GPU V4 bootstrap)");

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
    std::vector<byte_t> authoritative(index.header.rabitq_entry_bytes);
    std::vector<byte_t> resident(index.header.rabitq_entry_bytes);
    for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
      const AuditSample& sample = samples[sample_index];
      const format::ShardRegion& shard = index.shards[sample.shard];
      requests.assign(1, FetchDescriptor{
        .remote_offset = shard.node_base_offset + sample.slot * shard.node_stride +
          VamanaNode::offset_rabitq_code(),
        .destination_address = reinterpret_cast<u64>(d_exact_records),
        .bytes = index.header.rabitq_entry_bytes,
        .memory_node = static_cast<u16>(sample.shard),
        .kind = static_cast<u8>(FetchKind::code),
      });
      statuses.assign(1, -EIO);
      source.fetch(requests, statuses);
      if (statuses.front() <= 0) {
        throw std::runtime_error(
          "GPU V4 RaBitQ ordinal audit RDMA read failed: shard=" +
          std::to_string(sample.shard) + " slot=" +
          std::to_string(sample.slot) + " status=" +
          std::to_string(statuses.front()));
      }
      check_cuda(cudaMemcpy(authoritative.data(), d_exact_records, authoritative.size(),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(GPU V4 RaBitQ audit source)");
      check_cuda(cudaMemcpy(
        resident.data(),
        d_rabitq_entries + sample.ordinal * index.header.rabitq_entry_bytes,
        resident.size(), cudaMemcpyDeviceToHost),
        "cudaMemcpy(GPU V4 RaBitQ audit resident)");
      if (!std::equal(resident.begin(), resident.end(), authoritative.begin())) {
        throw std::runtime_error(
          "GPU V4 RaBitQ ordinal mapping mismatch: shard=" +
          std::to_string(sample.shard) + " slot=" +
          std::to_string(sample.slot) + " ordinal=" +
          std::to_string(sample.ordinal));
      }
    }
    std::cerr << "[gpu-search] streamed " << streamed
              << " RaBitQ bytes directly into final GPU storage; ordinal audit passed for "
              << samples.size() << " entries\n";
  }

  void clear_delta_device_state() {
    bind_cuda_device("cudaSetDevice(GPU V4 delta reset)");
    check_cuda(cudaMemset(d_delta_records, 0,
                          static_cast<size_t>(delta_capacity) * sizeof(DeviceDeltaRecord)),
               "cudaMemset(V4 delta records)");
    check_cuda(cudaMemset(d_delta_next, 0xff,
                          static_cast<size_t>(delta_capacity) * sizeof(u32)),
               "cudaMemset(V4 delta links)");
    check_cuda(cudaMemset(d_base_override_keys, 0xff,
                          static_cast<size_t>(delta_table_capacity) * sizeof(u32)),
               "cudaMemset(V4 override keys)");
    check_cuda(cudaMemset(d_base_override_epochs, 0,
                          static_cast<size_t>(delta_table_capacity) * sizeof(u64)),
               "cudaMemset(V4 override epochs)");
    check_cuda(cudaMemset(d_delta_remote_keys, 0,
                          static_cast<size_t>(delta_table_capacity) * sizeof(u64)),
               "cudaMemset(V4 remote keys)");
    check_cuda(cudaMemset(d_delta_remote_slots, 0xff,
                          static_cast<size_t>(delta_table_capacity) * sizeof(u32)),
               "cudaMemset(V4 remote slots)");
    check_cuda(cudaMemset(d_delta_count, 0, sizeof(u32)), "cudaMemset(V4 delta count)");
    if (d_delta_bucket_heads != nullptr) {
      check_cuda(cudaMemset(d_delta_bucket_heads, 0xff,
                            static_cast<size_t>(anchor_table.count()) * sizeof(u32)),
                 "cudaMemset(V4 delta buckets)");
    }
  }

  void start_persistent_kernel() {
    bind_cuda_device("cudaSetDevice(GPU V4 kernel start)");
    std::atomic_ref<u32>(*stop_host).store(0, std::memory_order_release);
    (void)cudaGetLastError();
    launch_persistent_search(kernel_stream, kernel_params, kernel_blocks, 128);
    check_cuda(cudaGetLastError(), "launch_persistent_search(V4)");
    kernel_running = true;
  }

  void stop_persistent_kernel() {
    if (!kernel_running) return;
    bind_cuda_device("cudaSetDevice(GPU V4 kernel stop)");
    std::atomic_ref<u32>(*stop_host).store(1, std::memory_order_release);
    const cudaError_t status = cudaStreamSynchronize(kernel_stream);
    kernel_running = false;
    check_cuda(status, "cudaStreamSynchronize(GPU V4 stop)");
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
    backend.reset();
    if (delta_stream != nullptr) cudaStreamDestroy(delta_stream);
    if (transfer_stream != nullptr) cudaStreamDestroy(transfer_stream);
    if (kernel_stream != nullptr) cudaStreamDestroy(kernel_stream);
    if (fetch_status_host != nullptr) cudaFreeHost(fetch_status_host);
    if (direct_disabled_host != nullptr) cudaFreeHost(direct_disabled_host);
    if (direct_error_host != nullptr) cudaFreeHost(direct_error_host);
    if (stop_host != nullptr) cudaFreeHost(stop_host);
    if (result_distances_host != nullptr) cudaFreeHost(result_distances_host);
    if (result_ids_host != nullptr) cudaFreeHost(result_ids_host);
    device_free(d_delta_count);
    device_free(d_delta_remote_slots);
    device_free(d_delta_remote_keys);
    device_free(d_base_override_epochs);
    device_free(d_base_override_keys);
    device_free(d_delta_next);
    device_free(d_delta_rabitq);
    device_free(d_delta_vectors);
    device_free(d_delta_records);
    device_free(d_graph_cache_generation);
    device_free(d_graph_invalidation_keys);
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
    device_free(d_beam_expanded);
    device_free(d_beam_distances);
    device_free(d_beam_ids);
    device_free(d_beam_handles);
    device_free(d_query_luts);
    device_free(d_rotated_queries);
    if (query_staging_host != nullptr) cudaFreeHost(query_staging_host);
    device_free(d_queries);
    device_free(d_delta_bucket_heads);
    device_free(d_anchor_distances);
    device_free(d_anchor_vectors);
    device_free(d_entry_points);
    device_free(d_centroid);
    device_free(d_shards);
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
    {
      std::lock_guard<std::mutex> lock(admission_mutex);
      admission_queue.push_back({.descriptor = descriptor, .enqueued_at = submitted_at});
    }
    admission_cv.notify_one();
    engine.telemetry_.queries_submitted.fetch_add(1, std::memory_order_relaxed);
    return future.get();
  }

  void admission_loop() {
    bind_cuda_device("cudaSetDevice(GPU V4 admission)");
    std::vector<PendingSubmission> batch;
    batch.reserve(config.query_batch_target);
    while (!shutdown.load(std::memory_order_acquire)) {
      batch.clear();
      {
        std::unique_lock<std::mutex> lock(admission_mutex);
        admission_cv.wait(lock, [&] {
          return (!admission_queue.empty() && !admission_paused) || shutdown.load();
        });
        if (admission_queue.empty() || admission_paused) continue;
        const auto deadline = admission_queue.front().enqueued_at +
          std::chrono::microseconds(config.query_batch_wait_us);
        admission_cv.wait_until(lock, deadline, [&] {
          return admission_queue.size() >= config.query_batch_min ||
                 admission_paused || shutdown.load();
        });
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
          cudaMemcpyHostToDevice, transfer_stream), "cudaMemcpyAsync(GPU V4 query)");
      }
      check_cuda(cudaStreamSynchronize(transfer_stream), "cudaStreamSynchronize(GPU V4 query)");
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
      throw std::invalid_argument("GPU V4 delta mutation vector has an invalid size");
    }
    VamanaNode::RabitqCode code;
    f32 norm = 0.0f;
    f32 error = 0.0f;
    VamanaNode::compute_rabitq_entry(
      reinterpret_cast<const byte_t*>(decoded.data()), VectorDType::float32,
      code, norm, error);
    std::memcpy(entry.data(), code.data(), code.size());
    std::memcpy(entry.data() + format::rabitq_norm_offset(code_bits), &norm, sizeof(norm));
    std::memcpy(entry.data() + format::rabitq_error_offset(code_bits), &error, sizeof(error));
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
          (relative / region.node_stride) * index.header.graph_entry_bytes;
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

  void invalidate_graph_cache(std::span<const u64> raw_nodes) {
    if (raw_nodes.empty()) return;
    std::vector<u64> keys;
    keys.reserve(raw_nodes.size());
    for (u64 raw : raw_nodes) keys.push_back(graph_cache_key(raw));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() > graph_invalidation_capacity) {
      throw std::runtime_error("GPU V4 graph invalidation batch exceeds capacity");
    }
    bind_cuda_device("cudaSetDevice(GPU V4 graph invalidation)");
    check_cuda(cudaMemcpyAsync(d_graph_invalidation_keys, keys.data(),
                               keys.size() * sizeof(u64), cudaMemcpyHostToDevice,
                               delta_stream),
               "cudaMemcpyAsync(V4 graph invalidation keys)");
    launch_invalidate_graph_cache(
      delta_stream, d_graph_invalidation_keys, static_cast<u32>(keys.size()),
      d_graph_cache_keys, d_graph_cache_states, d_graph_cache_readers,
      graph_cache_sets, config.gpu_adjacency_cache_ways);
    check_cuda(cudaGetLastError(), "launch GPU V4 graph invalidation");
    check_cuda(cudaStreamSynchronize(delta_stream),
               "cudaStreamSynchronize(V4 graph invalidation)");
  }

  void upload_records_locked(std::vector<DeltaMutation>& mutations, bool rebuilding) {
    bind_cuda_device("cudaSetDevice(GPU V4 delta publication)");
    (void)cudaGetLastError();
    const u32 first_slot = static_cast<u32>(delta_records_host.size());
    if (first_slot + mutations.size() > delta_capacity) {
      throw std::runtime_error("GPU V4 delta live set exceeds its configured capacity");
    }
    std::vector<DeviceDeltaRecord> records;
    std::vector<f32> vectors(static_cast<size_t>(mutations.size()) * config.dim);
    std::vector<byte_t> entries(
      static_cast<size_t>(mutations.size()) * index.header.rabitq_entry_bytes);
    records.reserve(mutations.size());
    std::vector<u32> superseded_slots;
    std::vector<std::pair<u32, u64>> override_updates;
    std::vector<f32> decoded(config.dim);
    std::vector<byte_t> entry(index.header.rabitq_entry_bytes);
    for (size_t mutation_index = 0; mutation_index < mutations.size(); ++mutation_index) {
      DeltaMutation& mutation = mutations[mutation_index];
      encode_mutation_payload(mutation, decoded, entry);
      const u32 slot = static_cast<u32>(delta_records_host.size());
      if (!rebuilding) {
        const auto previous = latest_delta_slot.find(mutation.id);
        if (previous != latest_delta_slot.end()) {
          delta_records_host[previous->second].superseded_epoch = mutation.epoch;
          if (previous->second >= first_slot) {
            records[previous->second - first_slot].superseded_epoch = mutation.epoch;
          } else {
            superseded_slots.push_back(previous->second);
          }
        }
      }
      const bool deleted = mutation.kind == service::storage_owner::MutationKind::erase;
      const u64 record_remote = mutation.remote_node != 0
        ? mutation.remote_node : mutation.old_remote_node;
      const u32 bucket = deleted ? 0 : nearest_anchor(decoded, record_remote);
      DeviceDeltaRecord record{
        .id = mutation.id,
        .generation = std::max<u32>(1, mutation.generation),
        .flags = deleted ? kDeltaDeleted : 0u,
        .signature = deleted ? 0u : delta_signature(entry),
        .epoch = mutation.epoch,
        .remote_node = record_remote,
        .anchor_bucket = bucket,
      };
      delta_records_host.push_back(record);
      records.push_back(record);
      latest_delta_slot[mutation.id] = slot;
      std::copy(decoded.begin(), decoded.end(),
                vectors.begin() + mutation_index * config.dim);
      std::copy(entry.begin(), entry.end(),
                entries.begin() + mutation_index * index.header.rabitq_entry_bytes);
      u32 ordinal = 0;
      if (format::remote_to_ordinal(index, RemotePtr{mutation.old_remote_node}, ordinal)) {
        const auto [it, inserted] = base_override_epochs.emplace(ordinal, mutation.epoch);
        if (inserted) {
          override_updates.emplace_back(ordinal, mutation.epoch);
        } else if (mutation.epoch < it->second) {
          it->second = mutation.epoch;
          override_updates.emplace_back(ordinal, mutation.epoch);
        }
      }
    }

    if (!records.empty()) {
      check_cuda(cudaMemcpyAsync(d_delta_records + first_slot, records.data(),
                                 records.size() * sizeof(DeviceDeltaRecord),
                                 cudaMemcpyHostToDevice, delta_stream),
                 "cudaMemcpyAsync(V4 delta records)");
      check_cuda(cudaMemcpyAsync(
        d_delta_vectors + static_cast<size_t>(first_slot) * config.dim,
        vectors.data(), vectors.size() * sizeof(f32), cudaMemcpyHostToDevice, delta_stream),
        "cudaMemcpyAsync(V4 delta vectors)");
      check_cuda(cudaMemcpyAsync(
        d_delta_rabitq + static_cast<size_t>(first_slot) * index.header.rabitq_entry_bytes,
        entries.data(), entries.size(), cudaMemcpyHostToDevice, delta_stream),
        "cudaMemcpyAsync(V4 delta codes)");
    }
    for (u32 slot : superseded_slots) {
      launch_supersede_delta_record(delta_stream, d_delta_records, slot,
                                    delta_records_host[slot].superseded_epoch);
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
    if (rebuilding) {
      for (const auto& [ordinal, override_epoch] : base_override_epochs) {
        launch_insert_base_override(delta_stream, d_base_override_keys,
                                    d_base_override_epochs, delta_table_capacity,
                                    ordinal, override_epoch);
      }
    } else {
      for (const auto& [ordinal, override_epoch] : override_updates) {
        launch_insert_base_override(delta_stream, d_base_override_keys,
                                    d_base_override_epochs, delta_table_capacity,
                                    ordinal, override_epoch);
      }
    }
    const u32 count = static_cast<u32>(delta_records_host.size());
    launch_publish_delta_count(delta_stream, d_delta_count, count);
    check_cuda(cudaGetLastError(), "launch GPU V4 delta publication");
    check_cuda(cudaStreamSynchronize(delta_stream), "cudaStreamSynchronize(V4 delta publish)");
    engine.telemetry_.delta_live_entries.store(count, std::memory_order_relaxed);
  }

  void upload_mutations(std::vector<DeltaMutation>& mutations, u64 epoch,
                        std::span<const u64> invalidated_graph_nodes) {
    if (mutations.empty()) return;
    for (u64 raw : invalidated_graph_nodes) (void)graph_cache_key(raw);
    if (delta_records_host.size() + mutations.size() >
        static_cast<size_t>(delta_capacity) * 4 / 5) {
      compact_delta();
    }
    std::lock_guard<std::mutex> lock(delta_mutex);
    for (DeltaMutation& mutation : mutations) {
      mutation.epoch = epoch;
      const auto version = engine.delta_.version(mutation.id);
      if (version) mutation.generation = std::max(
        mutation.generation, static_cast<u32>(version->generation + 1));
    }
    upload_records_locked(mutations, false);
    invalidate_graph_cache(invalidated_graph_nodes);
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
        throw std::runtime_error("GPU V4 live delta exceeds capacity");
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
    bind_cuda_device("cudaSetDevice(GPU V4 maintenance)");
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
        std::cerr << "[gpu-search] V4 delta compaction failed: " << error.what() << '\n';
      }
    }
  }

  void report_direct_path_failure() {
    if (backend_kind != RemoteBackendKind::gpunetio || direct_disabled_host == nullptr ||
        std::atomic_ref<u32>(*direct_disabled_host).load(std::memory_order_acquire) == 0) {
      return;
    }
    bool expected = false;
    if (!direct_failure_logged.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) return;
    const i32 direct_error = direct_error_host == nullptr
      ? 0 : std::atomic_ref<i32>(*direct_error_host).load(std::memory_order_acquire);
    std::cerr << "[gpu-search] GPUNetIO direct read failed with status=" << direct_error
              << "; strict GPUNetIO mode rejects the query instead of reporting fallback performance\n";
    engine.telemetry_.direct_path_failures.fetch_add(1, std::memory_order_relaxed);
  }

  void fetch_loop() {
    std::vector<FetchDescriptor> batch;
    std::vector<i32> statuses;
    batch.reserve(std::max<u32>(config.query_batch_target *
                                config.gpu_graph_prefetch_depth, 64));
    while (!shutdown.load(std::memory_order_acquire)) {
      batch.clear();
      FetchDescriptor request;
      if (!fetches.try_pop(request)) {
        std::this_thread::yield();
        continue;
      }
      batch.push_back(request);
      const size_t target = std::max<size_t>(
        64, config.query_batch_target * config.gpu_graph_prefetch_depth);
      while (batch.size() < target && fetches.try_pop(request)) batch.push_back(request);
      statuses.assign(batch.size(), -EIO);
      report_direct_path_failure();
      try {
        backend->fetch(batch, statuses);
      } catch (...) {
        std::fill(statuses.begin(), statuses.end(), -EIO);
      }
      u64 bytes = 0;
      u64 graph_records = 0;
      u64 exact_records = 0;
      for (size_t index = 0; index < batch.size(); ++index) {
        bytes += batch[index].bytes;
        if (batch[index].kind == static_cast<u8>(FetchKind::graph_record)) ++graph_records;
        else if (batch[index].kind == static_cast<u8>(FetchKind::node_record)) ++exact_records;
        std::atomic_ref<i32>(fetch_status_host[batch[index].sequence]).store(
          statuses[index], std::memory_order_release);
      }
      if (backend_kind != RemoteBackendKind::gpunetio) {
        engine.telemetry_.rdma_read_ops.fetch_add(batch.size(), std::memory_order_relaxed);
        engine.telemetry_.rdma_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
        engine.telemetry_.graph_page_requests.fetch_add(graph_records, std::memory_order_relaxed);
        engine.telemetry_.exact_vector_reads.fetch_add(exact_records, std::memory_order_relaxed);
      }
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
      try {
        if (completion.status != 0) {
          throw std::runtime_error("persistent GPU V4 query failed with status " +
                                   std::to_string(completion.status));
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
          static_cast<u64>(completion.exact_vectors) * node_record_bytes +
          static_cast<u64>(completion.remote_pages) * index.header.graph_entry_bytes,
          std::memory_order_relaxed);
        engine.telemetry_.exact_vector_reads.fetch_add(completion.exact_vectors,
                                                       std::memory_order_relaxed);
        engine.telemetry_.graph_page_requests.fetch_add(completion.remote_pages,
                                                        std::memory_order_relaxed);
      }
      engine.telemetry_.graph_page_cache_hits.fetch_add(completion.cache_hits,
                                                        std::memory_order_relaxed);
      engine.telemetry_.exact_vector_cache_hits.fetch_add(completion.exact_cache_hits,
                                                          std::memory_order_relaxed);
    }
  }

  PersistentSearchEngine& engine;
  configuration::IndexConfiguration& config;
  format::View index;
  AnchorTable anchor_table;
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
  u32 node_record_bytes{};
  u32 delta_capacity{};
  u32 delta_table_capacity{};
  u32 graph_cache_sets{};
  u32 graph_cache_slots{};
  u32 exact_cache_sets{};
  u32 exact_cache_slots{};
  u32 exact_cache_stride{};
  u32 graph_invalidation_capacity{};
  size_t graph_cache_bytes{};
  size_t exact_cache_bytes{};
  size_t exact_region_offset{};
  size_t exact_cache_offset{};
  size_t graph_cache_offset{};
  u64 explicit_gpu_bytes{};
  u64 gpu_clock_khz{1};
  DeviceShardRegion* d_shards{};
  byte_t* d_rabitq_entries{};
  f32* d_centroid{};
  u32* d_entry_points{};
  f32* d_anchor_vectors{};
  f32* d_anchor_distances{};
  u32* d_delta_bucket_heads{};
  f32* d_queries{};
  f32* query_staging_host{};
  f32* d_rotated_queries{};
  f32* d_query_luts{};
  u32* d_beam_handles{};
  u32* d_beam_ids{};
  f32* d_beam_distances{};
  u8* d_beam_expanded{};
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
  f32* d_delta_vectors{};
  byte_t* d_delta_rabitq{};
  u32* d_delta_next{};
  u32* d_base_override_keys{};
  u64* d_base_override_epochs{};
  u64* d_delta_remote_keys{};
  u32* d_delta_remote_slots{};
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
  i32* fetch_status_host{};
  i32* fetch_status_device{};
  u32 fetch_status_stride{};
  cudaStream_t kernel_stream{};
  cudaStream_t transfer_stream{};
  cudaStream_t delta_stream{};
  PersistentKernelParams kernel_params{};
  u32 kernel_blocks{};
  bool kernel_running{};
  std::atomic<bool> direct_failure_logged{false};
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
  check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)),
             "cudaSetDevice(GPU V4 engine)");
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
  const size_t mutation_count = mutations.size();
  auto oldest = std::chrono::steady_clock::now();
  for (const DeltaMutation& mutation : mutations) {
    if (mutation.enqueued_at != std::chrono::steady_clock::time_point{}) {
      oldest = std::min(oldest, mutation.enqueued_at);
    }
  }
  impl_->upload_mutations(mutations, epoch, invalidated_graph_nodes);
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
