#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

#include <filesystem>

#include "nlohmann/json.hh"

namespace gpu_search {

static_assert(kCentroidRouteMaxLiveEntries ==
              format::kStorageCentroidRouteMaxLiveEntries);

using namespace persistent_engine_detail;

namespace {

static_assert(kPersistentMaxGraphDegree == kMaxSupportedGraphDegree);

u64 read_index_build_fingerprint(const std::filesystem::path& index_prefix) {
  const std::filesystem::path metadata_path{
    index_prefix.string() + ".meta.json"};
  std::ifstream input(metadata_path);
  if (!input.good()) {
    throw std::runtime_error(
      "missing index metadata while validating graph extent sidecar: " +
      metadata_path.string());
  }
  nlohmann::json metadata;
  input >> metadata;
  const u64 fingerprint =
    metadata.value("index_build_fingerprint", u64{0});
  if (fingerprint == 0) {
    throw std::runtime_error(
      "index metadata has no build fingerprint for graph extent validation: " +
      metadata_path.string());
  }
  return fingerprint;
}

}  // namespace
PersistentSearchEngine::Impl::Impl(PersistentSearchEngine& owner,
     configuration::IndexConfiguration& config_in,
     Context& channel_context,
     ClientConnectionManager& connection_manager,
     const MemoryRegionTokens& remote_regions)
    : engine(owner), config(config_in),
      submissions(config.gpu_query_slots * 2,
                  MappedRing<QueryDescriptor>::Direction::host_to_device),
      completions(config.gpu_query_slots * 2,
                  MappedRing<CompletionDescriptor>::Direction::device_to_host),
      route_submissions(
        8, MappedRing<CentroidRoutePublishDescriptor>::Direction::host_to_device),
      route_completions(
        8, MappedRing<CentroidRoutePublishCompletion>::Direction::device_to_host) {
  bind_cuda_device("cudaSetDevice(GPU navigation construction)");
  route_poll_salt = connection_manager.client_id;
  if (connection_manager.num_total_clients == 0 ||
      route_poll_salt >= connection_manager.num_total_clients) {
    throw std::runtime_error("invalid compute client identity");
  }
  if (config.gpu_traversal_beam_width > kPersistentMaxBeam ||
      config.gpu_final_rerank_width > kPersistentMaxExact ||
      config.R > kPersistentMaxGraphDegree) {
    throw std::invalid_argument("GPU navigation beam/exact/degree limit exceeded");
  }

  std::string load_error;
  if (!format::synthesize_distributed_view(
        config.resolved_index_prefix(), index,
        &load_error)) {
    throw std::runtime_error(load_error);
  }
  std::cerr << "[gpu-search] synthesized navigation manifest in memory from metadata\n";
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
      index.layout.vector_dtype != static_cast<u32>(config.resolved_vector_dtype())) {
    throw std::runtime_error("GPU navigation manifest does not match runtime metadata");
  }
  const u32 graph_entry_capacity = VamanaNode::graph_entry_capacity();
  if (graph_entry_capacity < config.R ||
      index.layout.graph_entry_bytes <
        vamana::hot_graph::kTaggedNeighborBaseOffset +
          static_cast<u64>(graph_entry_capacity) *
          vamana::hot_graph::kCompactPointerBytes) {
    throw std::runtime_error(
      "GPU hot graph cannot contain stable and provisional backlink slots");
  }
  std::vector<u8> graph_extent_classes;
  format::GraphExtentHeader graph_extent_header{};
  const bool live_extent_graph_reads =
    config.gpu_query_graph_read_policy == "live-extent";
  const bool header_neighbor_graph_reads =
    config.gpu_query_graph_read_policy == "header-neighbor";
  const bool variable_graph_reads =
    live_extent_graph_reads || header_neighbor_graph_reads;
  const std::filesystem::path graph_extent_path =
    index_path::graph_extent_file(config.resolved_index_prefix());
  if (live_extent_graph_reads) {
    if (!format::read_graph_extent_sidecar(
          graph_extent_path, graph_extent_header, graph_extent_classes,
          &load_error)) {
      throw std::runtime_error(
        "live-extent graph reads require a valid extent sidecar: " +
        load_error);
    }
    const u64 expected_build_fingerprint =
      read_index_build_fingerprint(config.resolved_index_prefix());
    if (graph_extent_header.num_nodes != index.layout.num_nodes ||
        graph_extent_header.num_shards != index.layout.num_shards ||
        graph_extent_header.graph_entry_bytes !=
          index.layout.graph_entry_bytes ||
        graph_extent_header.graph_entry_capacity != graph_entry_capacity ||
        graph_extent_header.graph_pointer_bytes !=
          index.layout.graph_pointer_bytes ||
        graph_extent_header.build_fingerprint !=
          expected_build_fingerprint ||
        graph_extent_header.payload_bytes != index.layout.num_nodes ||
        graph_extent_classes.size() != index.layout.num_nodes) {
      throw std::runtime_error(
        "graph extent sidecar does not match the active index: " +
        graph_extent_path.string());
    }
    graph_extent_sidecar_bytes = graph_extent_classes.size();
    std::cerr << "[gpu-search] graph-read-policy=live-extent"
              << " extent_quantum=" << format::kGraphExtentQuantum
              << " extent_source=" << graph_extent_path
              << " extent_classes=" << graph_extent_classes.size()
              << " extent_payload_bytes=" << graph_extent_sidecar_bytes
              << '\n';
  } else if (header_neighbor_graph_reads) {
    std::cerr << "[gpu-search] graph-read-policy=header-neighbor"
              << " header_bytes="
              << vamana::hot_graph::kTaggedNeighborBaseOffset
              << " second_stage=exact_neighbor_body"
              << " graph_record_bytes=" << index.layout.graph_entry_bytes
              << '\n';
  } else {
    std::cerr << "[gpu-search] graph-read-policy=fixed"
              << " graph_record_bytes=" << index.layout.graph_entry_bytes
              << '\n';
  }
  const u32 score_chunk_capacity = persistent_score_chunk_capacity(
    graph_entry_capacity, config.gpu_traversal_beam_width);
  const u64 max_merge_candidates =
    static_cast<u64>(config.gpu_traversal_beam_width) +
    static_cast<u64>(std::min(config.gpu_graph_commit_width,
                              score_chunk_capacity)) * graph_entry_capacity;
  if (score_chunk_capacity == 0 ||
      max_merge_candidates > kPersistentMaxMergeCandidates) {
    throw std::invalid_argument("GPU navigation prefetch/degree exceeds parallel top-k capacity");
  }

  centroid_route_shard_capacity = static_cast<u32>(index.shards.size());
  centroid_route_versions.assign(centroid_route_shard_capacity, 0);
  centroid_route_snapshots.resize(centroid_route_shard_capacity);
  std::cerr << "[gpu-search] query routing=versioned centroid routes"
            << " centroid_shards=" << index.shards.size()
            << " live_entries_per_shard=" << centroid_route_entry_capacity
            << " start_shards=1\n";
  query_slots = config.gpu_query_slots;
  query_dispatch_capacity = memory_budget::next_power_of_two(query_slots * 2);
  result_capacity = std::max<u32>(config.k, config.gpu_final_rerank_width);
  exact_width = kPersistentMaxExact;
  code_bytes = index.layout.code_bytes;
  query_slot_states = std::make_unique<QuerySlotState[]>(query_slots);
  free_slots = std::make_unique<bounded::Queue<u32>>(query_slots);
  admission_queue =
    std::make_unique<bounded::Queue<PendingSubmission>>(query_slots);
  for (u32 slot = 0; slot < query_slots; ++slot) {
    if (!free_slots->try_push(slot)) {
      throw std::runtime_error("failed to initialize bounded GPU query slots");
    }
  }

  node_record_bytes = static_cast<u32>(VamanaNode::size_until_vector_end());
  node_record_stride = static_cast<u32>(align_up(
    static_cast<u64>(node_record_bytes) + sizeof(u64), alignof(u64)));
  dynamic_code_record_bytes =
    VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES + code_bytes +
      VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES;
  const u64 storage_region_bytes =
    static_cast<u64>(config.mn_memory_gb) << 30;
  const u64 centroid_publication_bytes =
    format::storage_centroid_route_publication_bytes(
      config.dim, format::CentroidScalarType::float32,
      format::kStorageCentroidRouteMaxLiveEntries);
  if (centroid_publication_bytes == 0 ||
      centroid_publication_bytes > storage_region_bytes) {
    throw std::runtime_error("invalid storage tail reservation for dynamic PQ arena");
  }
  const u64 dynamic_allocation_limit =
    (storage_region_bytes - centroid_publication_bytes) & ~u64{63};
  std::vector<DeviceShardRegion> device_shards;
  device_shards.reserve(index.shards.size());
  for (const format::ShardRegion& shard : index.shards) {
    if (shard.dynamic_record_bytes == 0 ||
        shard.dynamic_base_offset >= dynamic_allocation_limit) {
      throw std::runtime_error(
        "dynamic storage range cannot be represented by the GPU PQ arena");
    }
    const u64 slot_count =
      (dynamic_allocation_limit - shard.dynamic_base_offset) /
      shard.dynamic_record_bytes;
    if (slot_count == 0 ||
        dynamic_code_arena_capacity >
          std::numeric_limits<u64>::max() - slot_count) {
      throw std::runtime_error("dynamic GPU PQ arena slot count overflows");
    }
    device_shards.push_back(DeviceShardRegion{
      .ordinal_base = shard.ordinal_base,
      .node_count = shard.node_count,
      .node_base_offset = shard.node_base_offset,
      .node_stride = shard.node_stride,
      .graph_base_offset = shard.graph_base_offset,
      .dynamic_base_offset = shard.dynamic_base_offset,
      .control_remote_offset = shard.control_remote_offset,
      .code_remote_offset = shard.code_remote_offset,
      .code_bytes = shard.code_bytes,
      .memory_node = shard.memory_node,
      .dynamic_record_bytes = shard.dynamic_record_bytes,
      .dynamic_hot_offset = shard.dynamic_hot_offset,
      .dynamic_code_offset = shard.dynamic_code_offset,
      .dynamic_arena_base_slot = dynamic_code_arena_capacity,
      .dynamic_arena_slot_count = slot_count,
    });
    dynamic_code_arena_capacity += slot_count;
  }
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
    .usable_bytes = usable_budget,
    .dim = config.dim,
    .pq_subquantizers = pq_model.subquantizers,
    .code_bytes = code_bytes,
    .query_slots = query_slots,
    .beam_width = config.gpu_traversal_beam_width,
    .graph_degree = config.R,
    .exact_width = exact_width,
    .exact_record_bytes = node_record_stride,
    .shard_count = static_cast<u32>(index.shards.size()),
  });
  if (!budget.fits) {
    throw std::runtime_error(
      "GPU navigation allocations exceed the configured memory budget; codes=" +
      std::to_string(budget.code_bytes) + " fixed=" +
      std::to_string(budget.fixed_bytes));
  }
  visited_capacity = budget.visited_capacity;
  const u64 dynamic_code_scratch_bytes =
    static_cast<u64>(query_slots) * kPersistentMaxMergeCandidates *
      dynamic_code_record_bytes;
  if (dynamic_code_arena_capacity >
      std::numeric_limits<u64>::max() / (sizeof(u32) + code_bytes)) {
    throw std::runtime_error("dynamic GPU PQ arena byte count overflows");
  }
  const u64 dynamic_code_arena_bytes =
    dynamic_code_arena_capacity * (sizeof(u32) + code_bytes);
  const u64 dynamic_request_scratch_bytes =
    static_cast<u64>(query_slots) * kPersistentMaxMergeCandidates *
    (sizeof(u32) + 2 * sizeof(u64));
  const u64 navigation_candidate_bytes =
    static_cast<u64>(query_slots) * kPersistentMaxMergeCandidates *
    (sizeof(u64) + sizeof(f32));
  const u64 estimated_direct_queue_count =
    static_cast<u64>(config.gpu_rdma_qps) * index.shards.size();
  const u64 query_dispatch_bytes = 2 * sizeof(u64) +
    static_cast<u64>(query_dispatch_capacity) *
      (sizeof(u64) + sizeof(QueryDescriptor));
  const bool rdma_trace_enabled = config.query_rdma_trace_mode != "off";
  const u64 direct_queue_bytes = estimated_direct_queue_count *
    (2 * (2 * sizeof(u64) +
          sizeof(DeviceRingView<DirectBatchDescriptor>) +
          static_cast<u64>(kDirectBatchQueueCapacity) *
            (sizeof(u64) + sizeof(DirectBatchDescriptor))) +
     sizeof(DirectOwnerProgress)) +
    static_cast<u64>(query_slots) * index.shards.size() *
      (3 * sizeof(i32) + 2 * sizeof(u64) +
       (rdma_trace_enabled ? sizeof(u64) : 0));
  const u64 rdma_trace_bytes = rdma_trace_enabled
    ? static_cast<u64>(query_slots) *
        (sizeof(QueryRdmaTraceHeader) +
         static_cast<u64>(config.query_rdma_trace_events_per_query) *
           sizeof(QueryRdmaTraceEvent))
    : 0;
  const u64 graph_scratch_bytes = static_cast<u64>(query_slots) *
    kPersistentGraphScratchSlots * kPersistentGraphReadBytes;
  const u64 graph_request_metadata_bytes = variable_graph_reads
    ? static_cast<u64>(query_slots) * kPersistentMaxPrefetch * sizeof(u32)
    : 0;
  const u64 speculative_graph_request_metadata_bytes =
    static_cast<u64>(query_slots) * kPersistentFrontierRobCapacity *
      (sizeof(u32) + 3 * sizeof(u64) + sizeof(u8) +
       (variable_graph_reads ? sizeof(u32) : 0));
  // The sidecar remains exactly one byte per base node on disk. Round only
  // the device allocation to a u32 word so the last real byte can be repaired
  // with an in-bounds packed CAS.
  const u64 graph_extent_device_bytes = live_extent_graph_reads
    ? align_up(graph_extent_sidecar_bytes, sizeof(u32))
    : 0;
  const u64 centroid_route_bytes =
    static_cast<u64>(centroid_route_shard_capacity) *
      sizeof(DeviceCentroidRouteShard) +
    static_cast<u64>(centroid_route_shard_capacity) *
      centroid_route_entry_capacity * sizeof(DeviceCentroidRouteEntry) +
    sizeof(u64);
  const u64 shard_centroid_bytes = static_cast<u64>(index.shards.size()) *
    config.dim * sizeof(f32);
  route_graph_bytes = centroid_route_bytes +
    shard_centroid_bytes;
  const u64 additional_scratch_bytes =
    dynamic_code_scratch_bytes + dynamic_code_arena_bytes +
    dynamic_request_scratch_bytes +
    navigation_candidate_bytes +
    query_dispatch_bytes + direct_queue_bytes + graph_scratch_bytes +
    route_graph_bytes + rdma_trace_bytes +
    graph_request_metadata_bytes +
    speculative_graph_request_metadata_bytes + graph_extent_device_bytes;
  if (additional_scratch_bytes > usable_budget - budget.explicit_bytes) {
    throw std::runtime_error(
      "GPU navigation dynamic-code scratch exceeds the configured memory budget");
  }
  explicit_gpu_bytes = budget.explicit_bytes + additional_scratch_bytes;
  engine.telemetry_.gpu_memory_explicit_bytes.store(
    explicit_gpu_bytes, std::memory_order_relaxed);
  engine.telemetry_.gpu_memory_base_pq_bytes.store(
    budget.code_bytes, std::memory_order_relaxed);
  engine.telemetry_.dynamic_code_cache_capacity.store(
    dynamic_code_arena_capacity, std::memory_order_relaxed);
  engine.telemetry_.gpu_memory_route_graph_bytes.store(
    route_graph_bytes, std::memory_order_relaxed);
  const u64 base_code_region_bytes = budget.code_bytes;
  const u64 exact_bytes = budget.exact_bytes;
  std::cerr << "[gpu-search] navigation budget codes=" << budget.code_bytes
            << " dynamic_code_scratch=" << dynamic_code_scratch_bytes
            << " dynamic_code_arena=" << dynamic_code_arena_bytes
            << " dynamic_code_arena_slots=" << dynamic_code_arena_capacity
            << " dynamic_request_scratch=" << dynamic_request_scratch_bytes
            << " navigation_candidates=" << navigation_candidate_bytes
            << " direct_queue_scratch=" << direct_queue_bytes
            << " graph_scratch=" << graph_scratch_bytes
            << " graph_request_metadata=" << graph_request_metadata_bytes
            << " speculative_graph_request_metadata="
            << speculative_graph_request_metadata_bytes
            << " graph_extent_classes=" << graph_extent_device_bytes
            << " query_rdma_trace=" << rdma_trace_bytes
            << " centroid_route=" << centroid_route_bytes
            << " shard_centroids=" << shard_centroid_bytes
            << " explicit=" << explicit_gpu_bytes
            << " limit=" << engine_budget << " bytes\n";

  const size_t code_region_bytes = static_cast<size_t>(base_code_region_bytes);
  dynamic_code_region_offset = static_cast<size_t>(align_up(
    code_region_bytes, 256));
  exact_region_offset = static_cast<size_t>(align_up(
    dynamic_code_region_offset + dynamic_code_scratch_bytes, 256));
  graph_scratch_offset = static_cast<size_t>(align_up(
    exact_region_offset + exact_bytes, 512));
  control_region_offset = static_cast<size_t>(
    align_up(graph_scratch_offset + graph_scratch_bytes, 256));
  const size_t control_snapshot_bytes =
    index.shards.size() * sizeof(format::StorageControlBlock);
  const size_t maintenance_snapshot_offset = static_cast<size_t>(align_up(
    control_snapshot_bytes, alignof(maintenance_telemetry::Snapshot)));
  const size_t maintenance_snapshot_bytes =
    index.shards.size() * sizeof(maintenance_telemetry::Snapshot);
  const size_t maintenance_sequence_after_offset = static_cast<size_t>(align_up(
    maintenance_snapshot_offset + maintenance_snapshot_bytes, alignof(u64)));
  const size_t maintenance_sequence_after_bytes =
    index.shards.size() * sizeof(u64);
  storage_route_snapshot_stride = static_cast<size_t>(
    format::storage_centroid_route_publication_bytes(
      config.dim, format::CentroidScalarType::float32,
      centroid_route_entry_capacity));
  if (storage_route_snapshot_stride == 0) {
    throw std::runtime_error("invalid centroid route snapshot dimensions");
  }
  if (index.shards.size() >
      std::numeric_limits<size_t>::max() / storage_route_snapshot_stride) {
    throw std::runtime_error("centroid route snapshot allocation overflows size_t");
  }
  const size_t route_snapshot_offset = static_cast<size_t>(align_up(
    maintenance_sequence_after_offset + maintenance_sequence_after_bytes, 64));
  const size_t route_snapshot_bytes =
    index.shards.size() * storage_route_snapshot_stride;
  const size_t route_sequence_after_offset = static_cast<size_t>(align_up(
    route_snapshot_offset + route_snapshot_bytes, alignof(u64)));
  const size_t control_region_bytes = route_sequence_after_offset +
    index.shards.size() * sizeof(u64);
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
  d_dynamic_code_records = d_remote_buffer + dynamic_code_region_offset;
  d_exact_records = d_remote_buffer + exact_region_offset;
  d_graph_scratch = d_remote_buffer + graph_scratch_offset;
  d_control_snapshots = reinterpret_cast<format::StorageControlBlock*>(
    d_remote_buffer + control_region_offset);
  d_maintenance_snapshots =
    reinterpret_cast<maintenance_telemetry::Snapshot*>(
      d_remote_buffer + control_region_offset + maintenance_snapshot_offset);
  d_maintenance_sequence_after = reinterpret_cast<u64*>(
    d_remote_buffer + control_region_offset +
      maintenance_sequence_after_offset);
  d_storage_route_snapshots =
    d_remote_buffer + control_region_offset + route_snapshot_offset;
  d_storage_route_sequence_after = reinterpret_cast<u64*>(
    d_remote_buffer + control_region_offset + route_sequence_after_offset);

  control_bootstrapper = std::make_unique<NavigationBootstrapper>(
    config, channel_context, connection_manager, remote_regions,
    d_remote_buffer, remote_buffer_bytes);
  std::cerr << "[gpu-search] bootstrap=CPU-posted GPUDirect RDMA; "
               "queries=strict GPU-initiated GPUNetIO\n";
  initialize_storage_route_descriptors();
  // Fail before accepting queries when storage nodes do not expose the
  // canonical fixed-route extension. A concurrent publication may produce a
  // transient empty result and will simply be retried by maintenance.
  (void)read_storage_centroid_route_publications();
  stream_codes_to_gpu(*control_bootstrapper);

  device_allocate(d_shards, index.shards.size(), "cudaMalloc(GPU navigation shards)");
  if (live_extent_graph_reads) {
    device_allocate(
      d_graph_extent_class_words,
      static_cast<size_t>(graph_extent_device_bytes / sizeof(u32)),
      "cudaMalloc(static graph extent class words)");
    check_cuda(cudaMemset(
                 d_graph_extent_class_words, 0,
                 static_cast<size_t>(graph_extent_device_bytes)),
               "cudaMemset(static graph extent class words)");
    check_cuda(cudaMemcpy(
                 d_graph_extent_class_words, graph_extent_classes.data(),
                 graph_extent_classes.size() * sizeof(u8),
                 cudaMemcpyHostToDevice),
               "cudaMemcpy(static graph extent class words)");
  }
  device_allocate(d_opq_matrix, pq_model.rotation.size(), "cudaMalloc(OPQ matrix)");
  device_allocate(d_pq_centroids, pq_model.centroids.size(), "cudaMalloc(PQ centroids)");
  device_allocate(d_shard_centroids,
                  static_cast<size_t>(index.shards.size()) * config.dim,
                  "cudaMalloc(shard route centroids)");
  check_cuda(cudaMemcpy(d_shards, device_shards.data(),
                        device_shards.size() * sizeof(DeviceShardRegion),
                        cudaMemcpyHostToDevice), "cudaMemcpy(GPU navigation shards)");
  if (!pq_model.rotation.empty()) {
    check_cuda(cudaMemcpy(d_opq_matrix, pq_model.rotation.data(),
                          pq_model.rotation.size() * sizeof(f32),
                          cudaMemcpyHostToDevice), "cudaMemcpy(OPQ matrix)");
  }
  check_cuda(cudaMemcpy(d_pq_centroids, pq_model.centroids.data(),
                        pq_model.centroids.size() * sizeof(f32),
                        cudaMemcpyHostToDevice), "cudaMemcpy(PQ centroids)");
  check_cuda(cudaMemset(
               d_shard_centroids, 0,
               static_cast<size_t>(index.shards.size()) * config.dim *
                 sizeof(f32)),
             "cudaMemset(shard route centroids)");
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
  const size_t speculative_graph_request_elements =
    static_cast<size_t>(query_slots) * kPersistentFrontierRobCapacity;
  device_allocate(d_speculative_graph_request_shards,
                  speculative_graph_request_elements,
                  "cudaMalloc(speculative graph request shards)");
  device_allocate(d_speculative_graph_request_offsets,
                  speculative_graph_request_elements,
                  "cudaMalloc(speculative graph request offsets)");
  device_allocate(d_speculative_graph_request_local_iovas,
                  speculative_graph_request_elements,
                  "cudaMalloc(speculative graph request local IOVAs)");
  device_allocate(d_speculative_graph_request_handles,
                  speculative_graph_request_elements,
                  "cudaMalloc(speculative graph request handles)");
  device_allocate(d_speculative_graph_validation_states,
                  speculative_graph_request_elements,
                  "cudaMalloc(speculative graph validation states)");
  check_cuda(cudaMemset(
               d_speculative_graph_validation_states, 0,
               speculative_graph_request_elements * sizeof(u8)),
             "cudaMemset(speculative graph validation states)");
  if (variable_graph_reads) {
    const size_t graph_request_elements =
      static_cast<size_t>(query_slots) * kPersistentMaxPrefetch;
    device_allocate(d_graph_request_bytes, graph_request_elements,
                    "cudaMalloc(graph request byte lengths)");
    check_cuda(cudaMemset(
                 d_graph_request_bytes, 0,
                 graph_request_elements * sizeof(u32)),
               "cudaMemset(graph request byte lengths)");
    device_allocate(d_speculative_graph_request_bytes,
                    speculative_graph_request_elements,
                    "cudaMalloc(speculative graph request byte lengths)");
    check_cuda(cudaMemset(
                 d_speculative_graph_request_bytes, 0,
                 speculative_graph_request_elements * sizeof(u32)),
               "cudaMemset(speculative graph request byte lengths)");
  }
  if (dynamic_code_arena_capacity >
      std::numeric_limits<size_t>::max() / code_bytes) {
    throw std::runtime_error("dynamic GPU PQ arena allocation exceeds size_t");
  }
  const size_t dynamic_arena_elements =
    static_cast<size_t>(dynamic_code_arena_capacity);
  device_allocate(d_dynamic_code_arena_states, dynamic_arena_elements,
                  "cudaMalloc(dynamic PQ arena incarnation states)");
  device_allocate(d_dynamic_code_arena_records,
                  dynamic_arena_elements * code_bytes,
                  "cudaMalloc(dynamic PQ arena records)");
  check_cuda(cudaMemset(d_dynamic_code_arena_states, 0,
                        dynamic_arena_elements * sizeof(u32)),
             "cudaMemset(dynamic PQ arena incarnation states)");

  device_allocate(d_query_dispatch_enqueue, 1,
                  "cudaMalloc(GPU query dispatch enqueue)");
  device_allocate(d_query_dispatch_dequeue, 1,
                  "cudaMalloc(GPU query dispatch dequeue)");
  device_allocate(d_query_dispatch_sequences, query_dispatch_capacity,
                  "cudaMalloc(GPU query dispatch sequences)");
  device_allocate(d_query_dispatch_entries, query_dispatch_capacity,
                  "cudaMalloc(GPU query dispatch entries)");
  check_cuda(cudaMemset(d_query_dispatch_enqueue, 0, sizeof(u64)),
             "cudaMemset(GPU query dispatch enqueue)");
  check_cuda(cudaMemset(d_query_dispatch_dequeue, 0, sizeof(u64)),
             "cudaMemset(GPU query dispatch dequeue)");
  std::vector<u64> query_dispatch_sequences(query_dispatch_capacity);
  for (u32 slot = 0; slot < query_dispatch_capacity; ++slot) {
    query_dispatch_sequences[slot] = slot;
  }
  check_cuda(cudaMemcpy(d_query_dispatch_sequences,
                        query_dispatch_sequences.data(),
                        query_dispatch_sequences.size() * sizeof(u64),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(GPU query dispatch sequences)");

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
  device_allocate(d_direct_speculative_batch_enqueue,
                  direct_batch_queue_count,
                  "cudaMalloc(GPUNetIO speculative enqueue positions)");
  device_allocate(d_direct_speculative_batch_dequeue,
                  direct_batch_queue_count,
                  "cudaMalloc(GPUNetIO speculative dequeue positions)");
  device_allocate(d_direct_speculative_batch_sequences, direct_queue_slots,
                  "cudaMalloc(GPUNetIO speculative queue sequences)");
  device_allocate(d_direct_speculative_batch_entries, direct_queue_slots,
                  "cudaMalloc(GPUNetIO speculative queue entries)");
  device_allocate(d_direct_speculative_batch_queues,
                  direct_batch_queue_count,
                  "cudaMalloc(GPUNetIO speculative queue views)");
  device_allocate(d_direct_batch_statuses,
                  static_cast<size_t>(query_slots) * index.shards.size(),
                  "cudaMalloc(GPUNetIO owner completion statuses)");
  device_allocate(d_core_batch_statuses,
                  static_cast<size_t>(query_slots) * index.shards.size(),
                  "cudaMalloc(GPUNetIO core completion statuses)");
  device_allocate(d_core_batch_completion_timestamps_ns,
                  static_cast<size_t>(query_slots) * index.shards.size(),
                  "cudaMalloc(GPUNetIO core completion timestamps)");
  device_allocate(d_tail_batch_statuses,
                  static_cast<size_t>(query_slots) * index.shards.size(),
                  "cudaMalloc(GPUNetIO tail completion statuses)");
  device_allocate(d_tail_batch_completion_timestamps_ns,
                  static_cast<size_t>(query_slots) * index.shards.size(),
                  "cudaMalloc(GPUNetIO tail completion timestamps)");
  if (rdma_trace_enabled) {
    device_allocate(d_direct_batch_completion_timestamps_ns,
                    static_cast<size_t>(query_slots) * index.shards.size(),
                    "cudaMalloc(GPUNetIO owner completion timestamps)");
    device_allocate(d_query_rdma_trace_headers, query_slots,
                    "cudaMalloc(query RDMA trace headers)");
    device_allocate(
      d_query_rdma_trace_events,
      static_cast<size_t>(query_slots) *
        config.query_rdma_trace_events_per_query,
      "cudaMalloc(query RDMA trace events)");
    check_cuda(cudaMemset(
                 d_query_rdma_trace_headers, 0,
                 static_cast<size_t>(query_slots) *
                   sizeof(QueryRdmaTraceHeader)),
               "cudaMemset(query RDMA trace headers)");
    const std::filesystem::path output(config.query_rdma_trace_output);
    if (output.has_parent_path()) {
      std::filesystem::create_directories(output.parent_path());
    }
    query_rdma_trace_stream.open(output, std::ios::out | std::ios::trunc);
    if (!query_rdma_trace_stream) {
      throw std::runtime_error(
        "failed to open query RDMA trace output: " + output.string());
    }
  }
  mapped_host_allocate(direct_owner_phases_host, d_direct_owner_phases,
                       direct_batch_queue_count,
                       "cudaHostAlloc(GPUNetIO owner runtime phases)");
  check_cuda(cudaHostAlloc(
               reinterpret_cast<void**>(&direct_owner_progress_host),
               static_cast<size_t>(direct_batch_queue_count) *
                 sizeof(DirectOwnerProgress),
               cudaHostAllocPortable),
             "cudaHostAlloc(GPUNetIO owner watchdog staging)");
  device_allocate(d_direct_owner_progress, direct_batch_queue_count,
                  "cudaMalloc(GPUNetIO owner watchdog progress)");
  check_cuda(cudaMemset(d_direct_batch_enqueue, 0,
                        static_cast<size_t>(direct_batch_queue_count) * sizeof(u64)),
             "cudaMemset(GPUNetIO owner enqueue positions)");
  check_cuda(cudaMemset(d_direct_batch_dequeue, 0,
                        static_cast<size_t>(direct_batch_queue_count) * sizeof(u64)),
             "cudaMemset(GPUNetIO owner dequeue positions)");
  check_cuda(cudaMemset(
               d_direct_speculative_batch_enqueue, 0,
               static_cast<size_t>(direct_batch_queue_count) * sizeof(u64)),
             "cudaMemset(GPUNetIO speculative enqueue positions)");
  check_cuda(cudaMemset(
               d_direct_speculative_batch_dequeue, 0,
               static_cast<size_t>(direct_batch_queue_count) * sizeof(u64)),
             "cudaMemset(GPUNetIO speculative dequeue positions)");
  std::vector<u64> direct_sequences(direct_queue_slots);
  std::vector<DeviceRingView<DirectBatchDescriptor>> direct_queues(
    direct_batch_queue_count);
  std::vector<DeviceRingView<DirectBatchDescriptor>> speculative_queues(
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
    speculative_queues[queue] = {
      .enqueue_position = reinterpret_cast<unsigned long long*>(
        d_direct_speculative_batch_enqueue + queue),
      .dequeue_position = reinterpret_cast<unsigned long long*>(
        d_direct_speculative_batch_dequeue + queue),
      .sequences = reinterpret_cast<unsigned long long*>(
        d_direct_speculative_batch_sequences + queue_base),
      .entries = d_direct_speculative_batch_entries + queue_base,
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
  check_cuda(cudaMemcpy(
               d_direct_speculative_batch_sequences,
               direct_sequences.data(),
               direct_sequences.size() * sizeof(u64),
               cudaMemcpyHostToDevice),
             "cudaMemcpy(GPUNetIO speculative queue sequences)");
  check_cuda(cudaMemcpy(
               d_direct_speculative_batch_queues,
               speculative_queues.data(),
               speculative_queues.size() *
                 sizeof(DeviceRingView<DirectBatchDescriptor>),
               cudaMemcpyHostToDevice),
             "cudaMemcpy(GPUNetIO speculative queue views)");

  mapped_host_allocate(centroid_route_updates_host,
                       d_centroid_route_updates,
                       centroid_route_shard_capacity,
                       "cudaHostAlloc(centroid route metadata staging)");
  mapped_host_allocate(centroid_route_centroid_updates_host,
                       d_centroid_route_centroid_updates,
                       static_cast<size_t>(centroid_route_shard_capacity) *
                         config.dim,
                       "cudaHostAlloc(centroid route vector staging)");
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

  device_allocate(d_centroid_route_shards, centroid_route_shard_capacity,
                  "cudaMalloc(centroid route shard headers)");
  device_allocate(d_centroid_route_entries,
                  static_cast<size_t>(centroid_route_shard_capacity) *
                    centroid_route_entry_capacity,
                  "cudaMalloc(centroid route entries)");
  device_allocate(d_centroid_route_epoch, 1,
                  "cudaMalloc(centroid route publication epoch)");
  check_cuda(cudaMemset(d_centroid_route_shards, 0,
                        static_cast<size_t>(centroid_route_shard_capacity) *
                          sizeof(DeviceCentroidRouteShard)),
             "cudaMemset(centroid route shard headers)");
  check_cuda(cudaMemset(d_centroid_route_entries, 0,
                        static_cast<size_t>(centroid_route_shard_capacity) *
                          centroid_route_entry_capacity *
                          sizeof(DeviceCentroidRouteEntry)),
             "cudaMemset(centroid route entries)");
  check_cuda(cudaMemset(d_centroid_route_epoch, 0, sizeof(u64)),
             "cudaMemset(centroid route publication epoch)");
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&stop_host), sizeof(u32),
                           cudaHostAllocPortable),
             "cudaHostAlloc(GPU navigation stop staging)");
  *stop_host = 0;
  device_allocate(stop_device, 1, "cudaMalloc(GPU navigation stop)");
  check_cuda(cudaMemset(stop_device, 0, sizeof(u32)),
             "cudaMemset(GPU navigation stop)");
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&direct_disabled_host), sizeof(u32),
                           cudaHostAllocPortable),
             "cudaHostAlloc(GPU navigation direct failure staging)");
  *direct_disabled_host = 0;
  device_allocate(direct_disabled_device, 1,
                  "cudaMalloc(GPU navigation direct failure flag)");
  check_cuda(cudaMemset(direct_disabled_device, 0, sizeof(u32)),
             "cudaMemset(GPU navigation direct failure flag)");
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&direct_error_host), sizeof(i32),
                           cudaHostAllocPortable),
             "cudaHostAlloc(GPU navigation direct error staging)");
  *direct_error_host = 0;
  device_allocate(direct_error_device, 1,
                  "cudaMalloc(GPU navigation direct error)");
  check_cuda(cudaMemset(direct_error_device, 0, sizeof(i32)),
             "cudaMemset(GPU navigation direct error)");
  mapped_host_allocate(query_kernel_ready_host, d_query_kernel_ready, 1,
                       "cudaHostAlloc(GPU query kernel readiness)");
  mapped_host_allocate(dispatcher_kernel_ready_host,
                       d_dispatcher_kernel_ready, 1,
                       "cudaHostAlloc(GPU dispatcher kernel readiness)");
  mapped_host_allocate(control_kernel_ready_host, d_control_kernel_ready, 1,
                       "cudaHostAlloc(GPU control kernel readiness)");
  check_cuda(cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking),
             "cudaStreamCreate(GPU navigation kernel)");
  check_cuda(cudaStreamCreateWithFlags(&route_stream, cudaStreamNonBlocking),
             "cudaStreamCreate(GPU centroid route control)");
  check_cuda(cudaStreamCreateWithFlags(&rdma_stream, cudaStreamNonBlocking),
             "cudaStreamCreate(GPU navigation RDMA owners)");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, static_cast<int>(config.gpu_device)),
             "cudaGetDeviceProperties(GPU navigation)");
  const bool decoupled_search_progression =
    config.decoupled_gpu_rdma_search_progression_enabled();
  if (decoupled_search_progression !=
      (config.gpu_graph_issue_width > config.gpu_graph_commit_width)) {
    throw std::logic_error(
      "resolved GPU-RDMA search progression mode disagrees with graph widths");
  }
  gpu_clock_khz = static_cast<u64>(std::max(1, properties.clockRate));
  std::array<PersistentKernelOccupancy, 2> occupancies{};
  std::array<u32, 2> hardware_blocks_per_sm{};
  for (size_t index = 0; index < occupancies.size(); ++index) {
    occupancies[index] =
      inspect_persistent_search_kernel(
        kPersistentThreadCandidates[index],
        decoupled_search_progression);
    hardware_blocks_per_sm[index] =
      occupancies[index].active_blocks_per_sm;
  }
  persistent_grid_plan = plan_persistent_grid(
    hardware_blocks_per_sm, config.gpu_persistent_blocks_per_sm,
    static_cast<u32>(std::max(1, properties.multiProcessorCount)),
    query_slots, direct_batch_queue_count);
  kernel_threads = persistent_grid_plan.selected.threads;
  owner_kernel_blocks = persistent_grid_plan.selected.owner_blocks;
  kernel_blocks = persistent_grid_plan.selected.query_blocks;
  const size_t selected_index =
    kernel_threads == kPersistentThreadCandidates[0] ? 0 : 1;
  persistent_kernel_occupancy = occupancies[selected_index];
  if (query_rdma_trace_stream.is_open()) {
    query_rdma_trace_stream
      << "{\"type\":\"metadata\",\"schema\":3,"
         "\"completion_granularity\":\"shard_batch_owner_completion_boundary\","
         "\"timestamp_clock\":\"GPU globaltimer nanoseconds\","
         "\"completion_semantics\":\"owner CTA timestamp after the descriptor's "
         "priority fence CQE and before status publication; split descriptors "
         "expose independent critical-prefix and speculative-tail fences; "
         "not per-parent, per-WQE, per-descriptor physical, or NIC-internal "
         "completion\","
      << "\"num_shards\":" << index.shards.size()
      << ",\"direct_qps_per_node\":" << direct_view.qps_per_node
      << ",\"kernel_threads\":" << kernel_threads
      << ",\"natural_parent_tile\":" << std::max(1u, kernel_threads / 32u)
      << ",\"commit_width\":" << config.gpu_graph_commit_width
      << ",\"issue_width_cap\":" << config.gpu_graph_issue_width
      << ",\"traversal_beam_width\":" << config.gpu_traversal_beam_width
      << ",\"max_expansions\":" << config.gpu_max_expansions
      << ",\"beam_merge_policy\":\""
      << config.gpu_query_beam_merge_policy
      << "\",\"graph_read_policy\":\""
      << config.gpu_query_graph_read_policy
      << "\",\"graph_extent_quantum\":"
      << format::kGraphExtentQuantum
      << ",\"dynamic_graph_extent_enabled\":"
      << (live_extent_graph_reads && config.gpu_dynamic_graph_extent
            ? "true" : "false")
      << ",\"graph_extent_source\":\""
      << (live_extent_graph_reads
            ? (config.gpu_dynamic_graph_extent
                 ? "static_gextent8_plus_dynamic_incarnation_tag"
                 : "offline_global_ordinal_gextent8")
            : header_neighbor_graph_reads
              ? "dependent_header_then_exact_neighbor_body"
              : "fixed_physical_record")
      << "\"}\n";
  }

  kernel_params = PersistentKernelParams{
    .submissions = submissions.device_view(),
    .device_submissions = {
      .enqueue_position = reinterpret_cast<unsigned long long*>(
        d_query_dispatch_enqueue),
      .dequeue_position = reinterpret_cast<unsigned long long*>(
        d_query_dispatch_dequeue),
      .sequences = reinterpret_cast<unsigned long long*>(
        d_query_dispatch_sequences),
      .entries = d_query_dispatch_entries,
      .capacity = query_dispatch_capacity,
      .mask = query_dispatch_capacity - 1,
    },
    .completions = completions.device_view(),
    .route_submissions = route_submissions.device_view(),
    .route_completions = route_completions.device_view(),
    .shards = d_shards,
    .num_shards = static_cast<u32>(index.shards.size()),
    .pq_codes = d_pq_codes,
    .opq_matrix = d_opq_matrix,
    .pq_centroids = d_pq_centroids,
    .num_nodes = static_cast<u32>(index.layout.num_nodes),
    .dim = config.dim,
    .pq_subquantizers = pq_model.subquantizers,
    .pq_subvector_dim = pq_model.subvector_dim(),
    .pq_code_bytes = pq_model.code_bytes(),
    .dynamic_code_record_bytes = dynamic_code_record_bytes,
    .graph_entry_bytes = index.layout.graph_entry_bytes,
    .graph_degree = index.layout.graph_degree,
    .graph_entry_capacity = graph_entry_capacity,
    .graph_shard_bits = index.layout.graph_shard_bits,
    .node_meta_offset = 0,
    .node_record_bytes = node_record_bytes,
    .node_record_stride = node_record_stride,
    .node_vector_offset = static_cast<u32>(VamanaNode::offset_vector()),
    .node_incarnation_offset =
      static_cast<u32>(VamanaNode::offset_slot_incarnation()),
    .vector_bytes = static_cast<u32>(VamanaNode::vector_bytes()),
    .vector_dtype = static_cast<u32>(config.resolved_vector_dtype()),
    .traversal_beam_width = config.gpu_traversal_beam_width,
    .final_rerank_width = config.gpu_final_rerank_width,
    .exact_width = exact_width,
    .max_expansions = config.gpu_max_expansions,
    .commit_width = config.gpu_graph_commit_width,
    .issue_width = config.gpu_graph_issue_width,
    .beam_merge_policy =
      config.gpu_query_beam_merge_policy == "stable-run"
        ? static_cast<u32>(BeamMergePolicy::stable_run)
        : static_cast<u32>(BeamMergePolicy::legacy),
    .visited_capacity = visited_capacity,
    .query_slots = query_slots,
    .direct_region_count = direct_view.remote_region_count,
    .direct_qps_per_node = direct_view.qps_per_node,
    .direct_local_mkey = direct_view.local_mkey,
    .direct_local_iova_base = direct_view.local_iova_base,
    .direct_timeout_ns =
      static_cast<u64>(config.gpu_direct_timeout_ms) * 1'000'000ULL,
    .route_snapshot_timeout_ns = 100000000ULL,
    .direct_regions = reinterpret_cast<const DirectRemoteRegion*>(direct_view.remote_regions),
    .direct_qps = direct_view.qp_array,
    .direct_qp_locks = direct_view.qp_locks,
    .direct_batch_queues = d_direct_batch_queues,
    .direct_speculative_batch_queues =
      d_direct_speculative_batch_queues,
    .direct_batch_statuses = d_direct_batch_statuses,
    .direct_batch_completion_timestamps_ns =
      d_direct_batch_completion_timestamps_ns,
    .core_batch_statuses = d_core_batch_statuses,
    .core_batch_completion_timestamps_ns =
      d_core_batch_completion_timestamps_ns,
    .tail_batch_statuses = d_tail_batch_statuses,
    .tail_batch_completion_timestamps_ns =
      d_tail_batch_completion_timestamps_ns,
    .direct_batch_queue_count = direct_batch_queue_count,
    .direct_owner_phases = d_direct_owner_phases,
    .direct_owner_progress = d_direct_owner_progress,
    .direct_dump = direct_view.dump,
    .direct_disabled = direct_disabled_device,
    .direct_error = direct_error_device,
    .centroid_route_updates = d_centroid_route_updates,
    .centroid_route_centroid_updates = d_centroid_route_centroid_updates,
    .centroid_route_shards = d_centroid_route_shards,
    .centroid_route_entries = d_centroid_route_entries,
    .shard_centroids = d_shard_centroids,
    .centroid_route_epoch = d_centroid_route_epoch,
    .centroid_route_shard_capacity = centroid_route_shard_capacity,
    .centroid_route_entry_capacity = centroid_route_entry_capacity,
    .stop = stop_device,
    .graph_scratch = d_graph_scratch,
    .graph_read_policy = header_neighbor_graph_reads
      ? static_cast<u32>(GraphReadPolicy::header_neighbor)
      : live_extent_graph_reads
        ? static_cast<u32>(GraphReadPolicy::live_extent)
        : static_cast<u32>(GraphReadPolicy::fixed),
    .graph_extent_class_words = d_graph_extent_class_words,
    .dynamic_graph_extent_enabled =
      live_extent_graph_reads && config.gpu_dynamic_graph_extent ? 1u : 0u,
    .graph_request_bytes = d_graph_request_bytes,
    .speculative_graph_request_shards =
      d_speculative_graph_request_shards,
    .speculative_graph_request_offsets =
      d_speculative_graph_request_offsets,
    .speculative_graph_request_local_iovas =
      d_speculative_graph_request_local_iovas,
    .speculative_graph_request_bytes =
      d_speculative_graph_request_bytes,
    .speculative_graph_request_handles =
      d_speculative_graph_request_handles,
    .speculative_graph_validation_states =
      d_speculative_graph_validation_states,
    .query_rdma_trace_headers = d_query_rdma_trace_headers,
    .query_rdma_trace_events = d_query_rdma_trace_events,
    .query_rdma_trace_mode = config.query_rdma_trace_mode == "full"
      ? static_cast<u32>(QueryRdmaTraceMode::full)
      : config.query_rdma_trace_mode == "sampled"
        ? static_cast<u32>(QueryRdmaTraceMode::sampled)
        : static_cast<u32>(QueryRdmaTraceMode::off),
    .query_rdma_trace_sample_rate = config.query_rdma_trace_sample_rate,
    .query_rdma_trace_events_per_query =
      config.query_rdma_trace_events_per_query,
    .decoded_queries = d_queries,
    .transformed_queries = d_transformed_queries,
    .query_luts = d_query_luts,
    .navigation_candidate_handles = d_navigation_candidate_handles,
    .navigation_candidate_distances = d_navigation_candidate_distances,
    .visited_hash = d_visited,
    .exact_records = d_exact_records,
    .dynamic_code_records = d_dynamic_code_records,
    .dynamic_code_arena_states = d_dynamic_code_arena_states,
    .dynamic_code_arena_records = d_dynamic_code_arena_records,
    .dynamic_code_arena_capacity = dynamic_code_arena_capacity,
    .dynamic_code_request_shards = d_dynamic_code_request_shards,
    .dynamic_code_request_offsets = d_dynamic_code_request_offsets,
    .dynamic_code_request_local_iovas = d_dynamic_code_request_local_iovas,
    .result_ids = d_result_ids,
    .result_distances = d_result_distances,
  };
  start_persistent_kernel();
  // Query admission has no immutable offline fallback. Install the first
  // complete versioned centroid-entry snapshot before any descriptor can reach
  // a query CTA.
  if (synchronize_storage_routes() != StorageRouteSyncResult::changed) {
    throw std::runtime_error(
      "initial versioned centroid route snapshot was not stable");
  }
  admission_thread = std::thread([this] { admission_loop(); });
  completion_thread = std::thread([this] { completion_loop(); });
  maintenance_thread = std::thread([this] { maintenance_loop(); });
}

}  // namespace gpu_search
