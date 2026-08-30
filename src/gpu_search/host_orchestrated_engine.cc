#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "common/index_path.hh"
#include "gpu_search/centroid_route_poll_policy.hh"
#include "gpu_search/graph_record_validation.hh"
#include "gpu_search/host_distance_kernel.hh"
#include "gpu_search/host_orchestrated_engine/impl.hh"
#include "gpu_search/host_orchestrated_policy.hh"
#include "gpu_search/maintenance_fence.hh"
#include "gpu_search/navigation_bootstrapper.hh"
#include "nlohmann/json.hh"
#include "vamana/storage_layout_resolver.hh"
#include "vamana/vamana_node.hh"

namespace gpu_search {
namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

size_t aligned(size_t value, size_t alignment = 64) {
  if (alignment == 0 ||
      value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    throw std::overflow_error("host query scratch alignment overflow");
  }
  return (value + alignment - 1) / alignment * alignment;
}

size_t checked_product(size_t left, size_t right, const char* what) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::overflow_error(std::string(what) + " size overflow");
  }
  return left * right;
}

size_t checked_sum(std::initializer_list<size_t> terms, const char* what) {
  size_t result = 0;
  for (size_t term : terms) {
    if (term > std::numeric_limits<size_t>::max() - result) {
      throw std::overflow_error(std::string(what) + " size overflow");
    }
    result += term;
  }
  return result;
}

u64 read_build_fingerprint(const std::filesystem::path& prefix) {
  std::ifstream input(prefix.string() + ".meta.json");
  if (!input.good()) {
    throw std::runtime_error(
      "missing index metadata while validating graph extent sidecar");
  }
  nlohmann::json metadata;
  input >> metadata;
  const u64 fingerprint = metadata.value("index_build_fingerprint", u64{0});
  if (fingerprint == 0) {
    throw std::runtime_error("index metadata has no build fingerprint");
  }
  return fingerprint;
}

}  // namespace

HostOrchestratedSearchEngine::Impl::Lane::Lane(Impl& owner, u32 lane_id)
    : engine(owner), id(lane_id) {}

void HostOrchestratedSearchEngine::Impl::Lane::initialize() {
  if (initialization_complete) {
    throw std::logic_error("host query lane was initialized more than once");
  }
  check_cuda(cudaSetDevice(static_cast<int>(engine.config.gpu_device)),
             "cudaSetDevice(host query lane)");
  scratch_bytes = engine.lane_scratch_bytes;
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&scratch), scratch_bytes,
                           cudaHostAllocPortable),
             "cudaHostAlloc(host query RDMA scratch)");
  std::memset(scratch, 0, scratch_bytes);
  scratch_region = std::make_unique<LocalMemoryRegion>(engine.data_context,
                                                       scratch, scratch_bytes);
  qps.reserve(engine.index.shards.size());
  for (size_t shard = 0; shard < engine.index.shards.size(); ++shard) {
    auto qp = std::make_unique<DetachedQP>(engine.data_context);
    qp->connect(engine.channel_context, engine.data_context.get_lid(),
                engine.connection_manager.server_qps[shard]);
    qps.push_back(std::move(qp));
  }

  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "cudaStreamCreate(host query lane)");
  check_cuda(
    cudaMalloc(reinterpret_cast<void**>(&d_query),
               checked_product(engine.config.dim, sizeof(f32), "host query")),
    "cudaMalloc(host query)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_lut),
                        checked_product(engine.code_bytes,
                                        256u * sizeof(f32), "host PQ LUT")),
             "cudaMalloc(host PQ LUT)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_ordinals),
                        checked_product(engine.score_capacity, sizeof(u32),
                                        "host candidate ordinal")),
             "cudaMalloc(host candidate ordinals)");
  check_cuda(
    cudaMalloc(reinterpret_cast<void**>(&d_dynamic_codes),
               checked_product(engine.score_capacity, engine.code_bytes,
                               "host dynamic codes")),
    "cudaMalloc(host dynamic codes)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_distances),
                        checked_product(engine.score_capacity, sizeof(f32),
                                        "host candidate distance")),
             "cudaMalloc(host candidate distances)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_exact_records),
                        checked_product(engine.exact_capacity,
                                        engine.exact_record_stride,
                                        "host exact records")),
             "cudaMalloc(host exact records)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_exact_distances),
                        checked_product(engine.exact_capacity, sizeof(f32),
                                        "host exact distances")),
             "cudaMalloc(host exact distances)");

  query.resize(engine.config.dim);
  transformed.resize(engine.config.dim);
  lut.resize(checked_product(engine.code_bytes, 256u, "host lane PQ LUT"));
  ordinals.resize(engine.score_capacity);
  packed_dynamic_codes.resize(checked_product(
    engine.score_capacity, engine.code_bytes, "host lane dynamic codes"));
  distances.resize(std::max(engine.score_capacity, engine.exact_capacity));
  beam.reserve(checked_sum(
    {engine.config.gpu_traversal_beam_width, engine.score_capacity},
    "host lane beam"));
  pending.reserve(engine.score_capacity);
  visited.reserve(checked_sum(
    {checked_product(engine.config.gpu_max_expansions,
                     engine.graph_entry_capacity, "host lane visited"),
     16u},
    "host lane visited"));
  initialization_complete = true;
}

HostOrchestratedSearchEngine::Impl::Lane::~Lane() {
  if (stream != nullptr) {
    (void)cudaSetDevice(static_cast<int>(engine.config.gpu_device));
    (void)cudaStreamSynchronize(stream);
  }
  if (d_exact_distances != nullptr) cudaFree(d_exact_distances);
  if (d_exact_records != nullptr) cudaFree(d_exact_records);
  if (d_distances != nullptr) cudaFree(d_distances);
  if (d_dynamic_codes != nullptr) cudaFree(d_dynamic_codes);
  if (d_ordinals != nullptr) cudaFree(d_ordinals);
  if (d_lut != nullptr) cudaFree(d_lut);
  if (d_query != nullptr) cudaFree(d_query);
  if (stream != nullptr) cudaStreamDestroy(stream);
  qps.clear();
  scratch_region.reset();
  if (scratch != nullptr) cudaFreeHost(scratch);
}

HostOrchestratedSearchEngine::Impl::LaneGuard::~LaneGuard() {
  if (engine != nullptr) engine->release_lane(lane);
}

HostOrchestratedSearchEngine::Impl::Impl(
  HostOrchestratedSearchEngine& owner,
  configuration::IndexConfiguration& config_in, Context& channel_context_in,
  ClientConnectionManager& connection_manager_in,
  const MemoryRegionTokens& remote_regions_in)
    : engine(owner),
      config(config_in),
      channel_context(channel_context_in),
      connection_manager(connection_manager_in),
      remote_regions(remote_regions_in),
      data_context(config_in) {}

void HostOrchestratedSearchEngine::Impl::initialize() {
  if (initialization_complete) {
    throw std::logic_error(
      "host-orchestrated engine was initialized more than once");
  }
  check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)),
             "cudaSetDevice(host-orchestrated construction)");
  route_poll_salt = connection_manager.client_id;
  if (connection_manager.num_total_clients == 0 ||
      route_poll_salt >= connection_manager.num_total_clients) {
    throw std::runtime_error("invalid compute client identity");
  }
  if (config.gpu_rdma_search_progression_mode != "coupled") {
    throw std::invalid_argument(
      "host-orchestrated engine requires coupled search progression");
  }

  std::string error;
  if (!format::synthesize_distributed_view(config.resolved_index_prefix(),
                                           index, &error)) {
    throw std::runtime_error(error);
  }
  if (!pq::read_model(
        index_path::navigation_model_file(config.resolved_index_prefix(),
                                          index.layout.pq_subquantizers),
        pq_model, &error)) {
    throw std::runtime_error(error);
  }
  if (index.layout.dim != config.dim || index.layout.graph_degree != config.R ||
      index.layout.num_shards != remote_regions.size() ||
      index.layout.pq_subquantizers != pq_model.subquantizers ||
      pq_model.dim != config.dim ||
      index.layout.pq_bits != pq_model.bits_per_code ||
      index.layout.code_bytes != pq_model.code_bytes() ||
      index.layout.model_checksum != pq_model.checksum() ||
      index.layout.graph_entry_bytes != VamanaNode::hot_graph_entry_size() ||
      index.layout.graph_shard_bits != VamanaNode::HOT_GRAPH_SHARD_BITS ||
      index.layout.vector_dtype !=
        static_cast<u32>(config.resolved_vector_dtype()) ||
      index.layout.num_shards != connection_manager.server_qps.size()) {
    throw std::runtime_error(
      "host query manifest does not match schema-v16 runtime metadata");
  }

  code_bytes = index.layout.code_bytes;
  graph_entry_bytes = index.layout.graph_entry_bytes;
  graph_entry_capacity = VamanaNode::graph_entry_capacity();
  const size_t dynamic_code_bytes = checked_sum(
    {VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES, code_bytes,
     VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES},
    "host dynamic-code record");
  const size_t dynamic_code_stride =
    aligned(dynamic_code_bytes, alignof(u32));
  const size_t exact_bytes = VamanaNode::size_until_vector_end();
  const size_t exact_stride = aligned(exact_bytes);
  const size_t score_width = checked_product(
    config.gpu_graph_commit_width, graph_entry_capacity,
    "host score capacity");
  if (dynamic_code_bytes > std::numeric_limits<u32>::max() ||
      dynamic_code_stride > std::numeric_limits<u32>::max() ||
      exact_bytes > std::numeric_limits<u32>::max() ||
      exact_stride > std::numeric_limits<u32>::max() ||
      score_width > std::numeric_limits<u32>::max()) {
    throw std::overflow_error("host query geometry exceeds 32-bit kernel fields");
  }
  dynamic_code_record_bytes = static_cast<u32>(dynamic_code_bytes);
  dynamic_code_record_stride = static_cast<u32>(dynamic_code_stride);
  exact_record_bytes = static_cast<u32>(exact_bytes);
  exact_record_stride = static_cast<u32>(exact_stride);
  score_capacity = std::max<u32>(
    format::kStorageCentroidRouteMaxLiveEntries,
    static_cast<u32>(score_width));
  // Fetch the complete terminal Beam. Exact headers are the visibility
  // authority for delete/upsert, so limiting the fetch to final_rerank_width
  // could let tombstoned prefix entries hide valid replacements farther down
  // the same Beam. This matches the persistent engine's exactification
  // contract while returning only the requested k results.
  exact_capacity = config.gpu_traversal_beam_width;
  if (code_bytes == 0 || graph_entry_capacity < config.R ||
      score_capacity == 0 || exact_capacity == 0) {
    throw std::runtime_error("invalid host query workspace geometry");
  }

  storage_region_bytes = static_cast<u64>(config.mn_memory_gb) << 30;
  for (size_t shard = 0; shard < remote_regions.size(); ++shard) {
    if (remote_regions[shard] == nullptr ||
        !remote_regions[shard]->address_range_valid() ||
        remote_regions[shard]->bytes < storage_region_bytes) {
      throw std::runtime_error(
        "storage node " + std::to_string(shard) +
        " registered fewer RDMA bytes than --mn-memory-gb requires; rebuild "
        "and restart every storage/compute binary from the same revision");
    }
  }
  const u64 publication_bytes =
    format::storage_centroid_route_publication_bytes(
      config.dim, format::CentroidScalarType::float32,
      format::kStorageCentroidRouteMaxLiveEntries);
  if (publication_bytes == 0 || publication_bytes > storage_region_bytes ||
      publication_bytes > std::numeric_limits<u32>::max()) {
    throw std::runtime_error("invalid storage centroid publication geometry");
  }
  dynamic_allocation_limit =
    (storage_region_bytes - publication_bytes) & ~u64{63};

  const bool consume_graph_extents =
    config.gpu_query_graph_read_policy == "live-extent";
  const bool validate_graph_extent_sidecar =
    consume_graph_extents || config.dynamic_graph_access_mode != "manual";
  if (validate_graph_extent_sidecar) {
    format::GraphExtentHeader header{};
    std::vector<u8> validated_classes;
    const auto path =
      index_path::graph_extent_file(config.resolved_index_prefix());
    if (!format::read_graph_extent_sidecar(path, header, validated_classes,
                                           &error)) {
      throw std::runtime_error(
        "formal/adaptive host graph access requires a "
        "valid extent sidecar: " +
        error);
    }
    if (header.num_nodes != index.layout.num_nodes ||
        header.num_shards != index.layout.num_shards ||
        header.graph_entry_bytes != graph_entry_bytes ||
        header.graph_entry_capacity != graph_entry_capacity ||
        header.graph_pointer_bytes != index.layout.graph_pointer_bytes ||
        header.build_fingerprint !=
          read_build_fingerprint(config.resolved_index_prefix()) ||
        header.payload_bytes != index.layout.num_nodes ||
        validated_classes.size() != index.layout.num_nodes) {
      throw std::runtime_error(
        "host graph extent sidecar does not match the active index");
    }
    if (consume_graph_extents) {
      graph_extent_classes = std::move(validated_classes);
    }
  }

  route_snapshot_stride = aligned(publication_bytes);
  graph_scratch_offset = 0;
  dynamic_code_scratch_offset = aligned(checked_product(
    config.gpu_graph_commit_width, graph_entry_bytes, "host graph scratch"));
  exact_scratch_offset = checked_sum(
    {dynamic_code_scratch_offset,
     aligned(checked_product(score_capacity, dynamic_code_record_stride,
                             "host dynamic-code scratch"))},
    "host dynamic-code scratch end");
  exact_header_scratch_offset =
    checked_sum({exact_scratch_offset,
                 aligned(checked_product(exact_capacity, exact_record_stride,
                                         "host exact scratch"))},
                "host exact scratch end");
  control_scratch_offset =
    checked_sum({exact_header_scratch_offset,
                 aligned(checked_product(exact_capacity, sizeof(u64),
                                         "host exact header scratch"))},
                "host exact header scratch end");
  const size_t control_stride =
    aligned(std::max(sizeof(format::StorageControlBlock),
                     sizeof(maintenance_telemetry::Snapshot)));
  route_scratch_offset =
    checked_sum({control_scratch_offset,
                 aligned(checked_product(index.shards.size(), control_stride,
                                         "host control scratch"))},
                "host control scratch end");
  route_sequence_scratch_offset = checked_sum(
    {route_scratch_offset,
     aligned(checked_product(index.shards.size(), route_snapshot_stride,
                             "host route scratch"))},
    "host route scratch end");
  lane_scratch_bytes =
    checked_sum({route_sequence_scratch_offset,
                 aligned(checked_product(index.shards.size(), sizeof(u64),
                                         "host route sequence scratch"))},
                "host lane scratch");

  const u32 lane_count = std::max<u32>(1, config.gpu_rdma_qps);
  if (index.layout.num_nodes > std::numeric_limits<size_t>::max()) {
    throw std::overflow_error("resident PQ node count exceeds size_t");
  }
  base_code_bytes = checked_product(
    static_cast<size_t>(index.layout.num_nodes), code_bytes,
    "resident PQ codes");
  const size_t per_lane_gpu_bytes = checked_sum(
    {checked_product(config.dim, sizeof(f32), "host lane query"),
     checked_product(code_bytes, 256u * sizeof(f32), "host lane PQ LUT"),
     checked_product(score_capacity, sizeof(u32), "host lane ordinals"),
     checked_product(score_capacity, code_bytes, "host lane dynamic codes"),
     checked_product(score_capacity, sizeof(f32), "host lane PQ distances"),
     checked_product(exact_capacity, exact_record_stride,
                     "host lane exact records"),
     checked_product(exact_capacity, sizeof(f32), "host lane exact distances")},
    "host lane GPU workspace");
  const size_t lane_gpu_bytes =
    checked_product(lane_count, per_lane_gpu_bytes, "host lane GPU workspaces");
  const size_t required_gpu_bytes =
    checked_sum({base_code_bytes, lane_gpu_bytes}, "host query GPU memory");
  size_t free_gpu_bytes = 0;
  size_t total_gpu_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes),
             "cudaMemGetInfo(host query budget)");
  if (config.gpu_memory_limit_gb < config.gpu_memory_reserve_gb) {
    throw std::invalid_argument(
      "GPU memory reserve exceeds the configured host-backend limit");
  }
  const u64 configured_budget =
    static_cast<u64>(config.gpu_memory_limit_gb -
                     config.gpu_memory_reserve_gb)
    << 30;
  const u64 runtime_reserve = static_cast<u64>(config.gpu_memory_reserve_gb)
                              << 30;
  const u64 physically_available =
    free_gpu_bytes > runtime_reserve
      ? static_cast<u64>(free_gpu_bytes) - runtime_reserve
      : 0;
  if (required_gpu_bytes > std::min(configured_budget, physically_available)) {
    throw std::runtime_error(
      "host query allocations exceed the configured/free GPU memory budget; "
      "required=" +
      std::to_string(required_gpu_bytes) +
      " free=" + std::to_string(free_gpu_bytes) +
      " total=" + std::to_string(total_gpu_bytes));
  }
  engine.telemetry_.gpu_memory_explicit_bytes.store(required_gpu_bytes,
                                                    std::memory_order_relaxed);
  engine.telemetry_.gpu_memory_base_pq_bytes.store(base_code_bytes,
                                                   std::memory_order_relaxed);
  check_cuda(
    cudaMalloc(reinterpret_cast<void**>(&d_base_codes), base_code_bytes),
    "cudaMalloc(host backend resident PQ codes)");
  stream_codes_to_gpu();
  free_lanes = std::make_unique<bounded::Queue<u32>>(lane_count);
  lanes.reserve(lane_count);
  for (u32 lane = 0; lane < lane_count; ++lane) {
    auto candidate = std::make_unique<Lane>(*this, lane);
    candidate->initialize();
    lanes.push_back(std::move(candidate));
    if (!free_lanes->try_push(lane)) {
      throw std::runtime_error("failed to initialize host RDMA lane pool");
    }
  }

  {
    LaneGuard guard = acquire_lane();
    initialize_storage_routes(guard.get());
  }
  maintenance_thread = std::thread([this] { maintenance_loop(); });
  std::cerr << "[gpu-search] backend=host-orchestrated"
            << " lanes=" << lane_count << " progression=strict-wave"
            << " commit_width=" << config.gpu_graph_commit_width
            << " graph_access=" << config.dynamic_graph_access_mode
            << " persistent_kernel=no gpunetio_query_transport=no\n";
  initialization_complete = true;
}

HostOrchestratedSearchEngine::Impl::~Impl() {
  stopping.store(true, std::memory_order_release);
  maintenance_shutdown.store(true, std::memory_order_release);
  maintenance_cv.notify_all();
  if (free_lanes != nullptr) free_lanes->notify_all();
  if (maintenance_thread.joinable()) maintenance_thread.join();
  lanes.clear();
  free_lanes.reset();
  if (d_base_codes != nullptr) {
    (void)cudaSetDevice(static_cast<int>(config.gpu_device));
    cudaFree(d_base_codes);
    d_base_codes = nullptr;
  }
}

void HostOrchestratedSearchEngine::Impl::stream_codes_to_gpu() {
  if (d_base_codes == nullptr || base_code_bytes == 0) {
    throw std::logic_error("host PQ bootstrap has no resident destination");
  }
  const u64 window_bytes = static_cast<u64>(config.gpu_bootstrap_window_mb)
                           << 20;
  if (window_bytes == 0 || config.gpu_bootstrap_windows == 0) {
    throw std::invalid_argument(
      "host PQ bootstrap window size and count must be positive");
  }
  NavigationBootstrapper source(config, channel_context, connection_manager,
                                remote_regions, d_base_codes, base_code_bytes);
  std::vector<NavigationRead> requests;
  std::vector<i32> statuses;
  requests.reserve(config.gpu_bootstrap_windows);
  size_t streamed = 0;
  size_t next_ordinal = 0;
  for (const auto& shard : index.shards) {
    if (shard.ordinal_base > std::numeric_limits<size_t>::max() ||
        shard.node_count > std::numeric_limits<size_t>::max() ||
        shard.code_bytes > std::numeric_limits<size_t>::max() ||
        shard.memory_node > std::numeric_limits<u16>::max()) {
      throw std::overflow_error("host PQ shard geometry exceeds runtime fields");
    }
    const size_t ordinal_base = static_cast<size_t>(shard.ordinal_base);
    const size_t node_count = static_cast<size_t>(shard.node_count);
    const size_t shard_base = checked_product(
      ordinal_base, code_bytes, "host PQ shard base");
    const size_t shard_bytes = static_cast<size_t>(shard.code_bytes);
    const size_t expected_shard_bytes = checked_product(
      node_count, code_bytes, "host PQ shard byte count");
    const size_t shard_end = checked_sum(
      {ordinal_base, node_count}, "host PQ shard ordinal end");
    if (ordinal_base != next_ordinal ||
        shard_end > static_cast<size_t>(index.layout.num_nodes) ||
        shard_bytes != expected_shard_bytes ||
        shard_base > base_code_bytes ||
        shard_bytes > base_code_bytes - shard_base) {
      throw std::runtime_error(
        "host PQ shard exceeds the resident code allocation");
    }
    next_ordinal = shard_end;
    for (u64 offset = 0; offset < shard.code_bytes;) {
      requests.clear();
      for (u32 window = 0;
           window < config.gpu_bootstrap_windows && offset < shard.code_bytes;
           ++window) {
        const u32 bytes = static_cast<u32>(std::min<u64>(
          std::min<u64>(window_bytes, std::numeric_limits<u32>::max()),
          shard.code_bytes - offset));
        if (bytes == 0) {
          throw std::logic_error("host PQ bootstrap made no forward progress");
        }
        if (offset > std::numeric_limits<size_t>::max() ||
            offset > std::numeric_limits<u64>::max() -
              shard.code_remote_offset) {
          throw std::overflow_error("host PQ bootstrap offset overflow");
        }
        const size_t local_offset = checked_sum(
          {shard_base, static_cast<size_t>(offset)},
          "host PQ bootstrap destination");
        if (local_offset > base_code_bytes ||
            bytes > base_code_bytes - local_offset) {
          throw std::runtime_error(
            "host PQ bootstrap write exceeds resident allocation");
        }
        requests.push_back({
          .remote_offset = shard.code_remote_offset + offset,
          .destination_address =
            reinterpret_cast<u64>(d_base_codes + local_offset),
          .bytes = bytes,
          .memory_node = static_cast<u16>(shard.memory_node),
        });
        offset += bytes;
      }
      statuses.assign(requests.size(), -EIO);
      source.read(requests, statuses);
      for (size_t request = 0; request < requests.size(); ++request) {
        if (statuses[request] <= 0) {
          throw std::runtime_error(
            "host backend PQ bootstrap RDMA read failed");
        }
        if (streamed > base_code_bytes ||
            requests[request].bytes > base_code_bytes - streamed) {
          throw std::overflow_error("host PQ bootstrap byte count overflow");
        }
        streamed += requests[request].bytes;
      }
    }
  }
  if (next_ordinal != static_cast<size_t>(index.layout.num_nodes) ||
      streamed != base_code_bytes) {
    throw std::runtime_error("host backend PQ bootstrap size mismatch");
  }
  check_cuda(cudaDeviceSynchronize(),
             "cudaDeviceSynchronize(host PQ bootstrap)");
}

HostOrchestratedSearchEngine::Impl::LaneGuard
HostOrchestratedSearchEngine::Impl::acquire_lane(bool account_query_wait) {
  if (stopping.load(std::memory_order_acquire) ||
      !healthy.load(std::memory_order_acquire)) {
    throw std::runtime_error(unhealthy_message());
  }
  const auto started = Clock::now();
  u32 lane = 0;
  if (free_lanes == nullptr || !free_lanes->pop_wait(lane, stopping)) {
    throw std::runtime_error(unhealthy_message());
  }
  // Queue::pop_wait may win a free cell concurrently with the fail-stop flag.
  // Do not hand that lane to new work or publish it again during shutdown.
  if (stopping.load(std::memory_order_acquire) ||
      !healthy.load(std::memory_order_acquire)) {
    throw std::runtime_error(unhealthy_message());
  }
  if (account_query_wait) {
    engine.telemetry_.submission_wait_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                           started)
        .count(),
      std::memory_order_relaxed);
  }
  return LaneGuard{this, lane};
}

void HostOrchestratedSearchEngine::Impl::release_lane(u32 lane) {
  if (free_lanes != nullptr && lane < lanes.size() &&
      host_orchestrated_policy::lane_reusable(
        lanes[lane]->poisoned, stopping.load(std::memory_order_acquire),
        healthy.load(std::memory_order_acquire))) {
    (void)free_lanes->push_wait(lane, stopping);
  }
}

std::string HostOrchestratedSearchEngine::Impl::unhealthy_message() const {
  std::lock_guard lock(health_mutex);
  return health_error.empty()
           ? "host-orchestrated query engine is stopping or unhealthy"
           : health_error;
}

void HostOrchestratedSearchEngine::Impl::mark_unhealthy(
  Lane* lane, const std::string& message) {
  if (lane != nullptr) lane->poisoned = true;
  bool transitioned = false;
  {
    std::lock_guard lock(health_mutex);
    if (healthy.load(std::memory_order_relaxed)) {
      health_error = message;
      healthy.store(false, std::memory_order_release);
      transitioned = true;
    }
  }
  // A timed-out read may still complete into registered lane scratch. Stop
  // every new admission, wake all waiters, and retain every checked-out lane
  // until engine destruction instead of attempting CQ recovery in place.
  stopping.store(true, std::memory_order_release);
  maintenance_shutdown.store(true, std::memory_order_release);
  if (free_lanes != nullptr) free_lanes->notify_all();
  maintenance_cv.notify_all();
  if (transitioned) {
    std::cerr << "[gpu-search] host query engine entered fail-stop mode: "
              << message << '\n';
  }
}

void HostOrchestratedSearchEngine::Impl::check_lane_cuda(
  Lane& lane, cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return;
  const std::string message =
    std::string(operation) + ": " + cudaGetErrorString(status);
  mark_unhealthy(&lane, message);
  throw std::runtime_error(message);
}

void HostOrchestratedSearchEngine::Impl::read_batch(
  Lane& lane, std::span<const ReadRequest> requests, bool account_query_io) {
  if (requests.empty()) return;
  const u32 max_inflight =
    std::max<u32>(1, std::min(data_context.max_qp_read_atomic(),
                              static_cast<u32>(config.max_send_queue_wr)));
  std::vector<ibv_wc> completions(max_inflight);
  for (u32 shard = 0; shard < index.shards.size(); ++shard) {
    std::vector<size_t> matching;
    matching.reserve(requests.size());
    for (size_t request = 0; request < requests.size(); ++request) {
      const auto& item = requests[request];
      if (item.shard != shard) continue;
      if (item.bytes == 0 || item.local_offset > lane.scratch_bytes ||
          item.bytes > lane.scratch_bytes - item.local_offset ||
          item.remote_offset > storage_region_bytes ||
          item.bytes > storage_region_bytes - item.remote_offset) {
        throw std::out_of_range("host query RDMA request is out of range");
      }
      matching.push_back(request);
    }
    DetachedQP& qp = *lane.qps[shard];
    for (size_t begin = 0; begin < matching.size(); begin += max_inflight) {
      const size_t count =
        std::min<size_t>(max_inflight, matching.size() - begin);
      const auto issue_started = Clock::now();
      for (size_t offset = 0; offset < count; ++offset) {
        const size_t request_index = matching[begin + offset];
        const auto& item = requests[request_index];
        try {
          qp.qp->post_send(
            reinterpret_cast<u64>(lane.scratch + item.local_offset), item.bytes,
            lane.scratch_region->get_lkey(), IBV_WR_RDMA_READ, true, false,
            remote_regions[shard].get(), item.remote_offset, 0,
            request_index + 1);
        } catch (...) {
          mark_unhealthy(&lane, "host query RDMA work-request posting failed");
          throw;
        }
      }
      const auto issued = Clock::now();
      size_t remaining = count;
      const auto deadline =
        issued + std::chrono::milliseconds(config.gpu_direct_timeout_ms);
      while (remaining != 0) {
        const i32 completed = qp.poll_send_cq(
          completions.data(),
          static_cast<i32>(std::min<size_t>(completions.size(), remaining)));
        if (completed < 0) {
          mark_unhealthy(&lane, "host query RDMA CQ polling failed");
          throw std::runtime_error("host query RDMA CQ polling failed");
        }
        if (completed == 0) {
          if (Clock::now() >= deadline) {
            mark_unhealthy(&lane, "host query RDMA read timed out");
            throw std::runtime_error("host query RDMA read timed out");
          }
          std::this_thread::yield();
          continue;
        }
        remaining -= static_cast<size_t>(completed);
        for (i32 completion = 0; completion < completed; ++completion) {
          if (completions[completion].status != IBV_WC_SUCCESS) {
            mark_unhealthy(
              &lane, "host query RDMA read completed with transport error");
            throw std::runtime_error(
              "host query RDMA read completed with transport error");
          }
        }
      }
      if (account_query_io) {
        const auto completed_at = Clock::now();
        engine.telemetry_.gpu_rdma_issue_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(issued -
                                                               issue_started)
            .count(),
          std::memory_order_relaxed);
        engine.telemetry_.gpu_rdma_wait_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(completed_at -
                                                               issued)
            .count(),
          std::memory_order_relaxed);
        engine.telemetry_.rdma_completion_latency_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(completed_at -
                                                               issue_started)
            .count(),
          std::memory_order_relaxed);
        engine.telemetry_.rdma_completion_groups.fetch_add(
          1, std::memory_order_relaxed);
      }
    }
  }
  if (account_query_io) {
    u64 bytes = 0;
    for (const auto& request : requests) bytes += request.bytes;
    engine.telemetry_.rdma_read_ops.fetch_add(requests.size(),
                                              std::memory_order_relaxed);
    engine.telemetry_.rdma_read_bytes.fetch_add(bytes,
                                                std::memory_order_relaxed);
  }
}

HostOrchestratedSearchEngine::HostOrchestratedSearchEngine(
  configuration::IndexConfiguration& config, Context& channel_context,
  ClientConnectionManager& connection_manager,
  const MemoryRegionTokens& remote_regions) {
  auto candidate = std::make_unique<Impl>(
    *this, config, channel_context, connection_manager, remote_regions);
  candidate->initialize();
  impl_ = std::move(candidate);
}

HostOrchestratedSearchEngine::~HostOrchestratedSearchEngine() = default;

service::QueryResult HostOrchestratedSearchEngine::search(
  VectorDType query_dtype, const byte_t* query_data, u32 k) {
  return impl_->search(query_dtype, query_data, k);
}

std::optional<u32> HostOrchestratedSearchEngine::select_centroid_home(
  std::span<const f32> vector) const {
  return impl_->select_centroid_home(vector);
}

bool HostOrchestratedSearchEngine::wait_for_maintenance(
  std::span<const u64> target_sequences, std::chrono::milliseconds timeout,
  std::vector<u64>* durable_sequences,
  std::vector<u64>* effective_target_sequences) {
  return impl_->wait_for_maintenance(
    target_sequences, timeout, durable_sequences, effective_target_sequences);
}

std::vector<std::optional<maintenance_telemetry::Snapshot>>
HostOrchestratedSearchEngine::read_maintenance_telemetry() {
  return impl_->read_maintenance_telemetry();
}

TelemetrySnapshot HostOrchestratedSearchEngine::telemetry() const {
  return telemetry_.snapshot();
}

void HostOrchestratedSearchEngine::reset_telemetry() { telemetry_.reset(); }

}  // namespace gpu_search
