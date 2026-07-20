#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

namespace gpu_search {

static_assert(kDynamicRouteSlotsPerShard == format::kStorageRouteSlots);
static_assert(kPersistentMaxSubquantizers ==
              format::kStorageRouteMaxCodeBytes);

using namespace persistent_engine_detail;

namespace {

static_assert(sizeof(DeviceShardRegion) == sizeof(format::ShardRegion));

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
  dynamic_route_capacity = static_cast<u32>(index.shards.size()) *
    kDynamicRouteSlotsPerShard;
  dynamic_route_diff =
    std::make_unique<DynamicRouteOverlayDiff>(
      static_cast<u32>(index.shards.size()));
  if (dynamic_route_diff->capacity() != dynamic_route_capacity) {
    throw std::logic_error("GPU dynamic route capacity mismatch");
  }
  dynamic_route_snapshot.resize(dynamic_route_capacity);
  dynamic_route_update_scratch.reserve(dynamic_route_capacity);
  for (u32 anchor = 0; anchor < anchor_table.raw_pointers.size(); ++anchor) {
    anchor_buckets_by_raw.emplace(anchor_table.raw_pointers[anchor], anchor);
    anchor_graph_keys_host.push_back(
      graph_record_key(anchor_table.raw_pointers[anchor]));
  }
  std::sort(anchor_graph_keys_host.begin(), anchor_graph_keys_host.end());
  anchor_graph_keys_host.erase(
    std::unique(anchor_graph_keys_host.begin(), anchor_graph_keys_host.end()),
    anchor_graph_keys_host.end());
  if (anchor_graph_keys_host.size() > std::numeric_limits<u32>::max()) {
    throw std::runtime_error("GPU anchor route table exceeds uint32 capacity");
  }
  entry_handles = index.entry_points;
  std::cerr << "[gpu-search] query routing=storage-canonical adaptive routes"
            << "+static recall fallback"
            << " static_fallback_entries=" << anchor_table.count()
            << " adaptive_slots_per_shard=" << kDynamicRouteSlotsPerShard
            << " seeds=" << config.gpu_entry_seed_count << '\n';
  query_slots = config.gpu_query_slots;
  query_dispatch_capacity = memory_budget::next_power_of_two(query_slots * 2);
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
  const u64 query_dispatch_bytes = 2 * sizeof(u64) +
    static_cast<u64>(query_dispatch_capacity) *
      (sizeof(u64) + sizeof(QueryDescriptor));
  const u64 direct_queue_bytes = estimated_direct_queue_count *
    (2 * sizeof(u64) + sizeof(DeviceRingView<DirectBatchDescriptor>) +
     static_cast<u64>(kDirectBatchQueueCapacity) *
       (sizeof(u64) + sizeof(DirectBatchDescriptor))) +
    static_cast<u64>(query_slots) * index.shards.size() * sizeof(i32);
  const u64 graph_scratch_bytes = static_cast<u64>(query_slots) *
    kPersistentMaxPrefetch * kPersistentGraphReadBytes;
  const u64 route_graph_record_bytes =
    static_cast<u64>(anchor_graph_keys_host.size()) *
    index.layout.graph_entry_bytes;
  const u64 route_graph_metadata_bytes =
    static_cast<u64>(anchor_graph_keys_host.size()) *
    (sizeof(u64) + 2 * sizeof(u32));
  const u64 dynamic_route_bytes =
    static_cast<u64>(dynamic_route_capacity) *
    sizeof(DeviceDynamicRouteSlot);
  const u64 dynamic_route_code_bytes =
    static_cast<u64>(dynamic_route_capacity) * index.layout.code_bytes;
  const u64 anchor_route_bytes =
    route_graph_record_bytes + route_graph_metadata_bytes;
  route_graph_bytes = anchor_route_bytes + dynamic_route_bytes +
    dynamic_route_code_bytes;
  const u64 additional_scratch_bytes =
    dynamic_code_scratch_bytes + dynamic_request_scratch_bytes +
    navigation_candidate_bytes + query_dispatch_bytes + direct_queue_bytes +
    graph_scratch_bytes + route_graph_bytes;
  if (additional_scratch_bytes > usable_budget - budget.explicit_bytes) {
    throw std::runtime_error(
      "GPU navigation dynamic-code scratch exceeds the configured memory budget");
  }
  const u64 available_resident_pq_bytes =
    usable_budget - budget.explicit_bytes - additional_scratch_bytes;
  const u64 requested_resident_pq_bytes =
    static_cast<u64>(config.gpu_resident_pq_budget_mb) << 20;
  const u64 resident_pq_budget_bytes = std::min(
    requested_resident_pq_bytes, available_resident_pq_bytes);
  resident_pq_capacity = memory_budget::choose_resident_pq_capacity(
    resident_pq_budget_bytes, kDeltaHandleMask, code_bytes);
  if (resident_pq_capacity < delta_capacity) {
    throw std::runtime_error(
      "GPU resident dynamic-PQ budget is too small for the bounded update tier; "
      "increase --gpu-resident-pq-budget-mb or reduce --delta-budget-mb");
  }
  resident_pq_table_capacity = memory_budget::next_power_of_two(
    static_cast<u64>(resident_pq_capacity) * 2);
  resident_pq_bytes = memory_budget::resident_pq_footprint(
    resident_pq_capacity, code_bytes);
  explicit_gpu_bytes = budget.explicit_bytes + additional_scratch_bytes +
    resident_pq_bytes;
  engine.telemetry_.gpu_memory_explicit_bytes.store(
    explicit_gpu_bytes, std::memory_order_relaxed);
  engine.telemetry_.gpu_memory_base_pq_bytes.store(
    budget.code_bytes, std::memory_order_relaxed);
  engine.telemetry_.gpu_memory_resident_pq_bytes.store(
    resident_pq_bytes, std::memory_order_relaxed);
  engine.telemetry_.resident_pq_capacity.store(
    resident_pq_capacity, std::memory_order_relaxed);
  engine.telemetry_.gpu_memory_route_graph_bytes.store(
    route_graph_bytes, std::memory_order_relaxed);
  engine.telemetry_.gpu_memory_delta_reserved_bytes.store(
    budget.delta_bytes, std::memory_order_relaxed);
  const u64 base_code_region_bytes = budget.code_bytes;
  const u64 exact_bytes = budget.exact_bytes;
  std::cerr << "[gpu-search] navigation budget codes=" << budget.code_bytes
            << " delta=" << budget.delta_bytes
            << " delta_capacity=" << budget.delta_capacity
            << " delta_codes=" << budget.delta_code_bytes
            << " resident_pq=" << resident_pq_bytes
            << " resident_pq_capacity=" << resident_pq_capacity
            << " permanent_overrides=" << budget.permanent_override_bytes
            << " dynamic_code_scratch=" << dynamic_code_scratch_bytes
            << " dynamic_request_scratch=" << dynamic_request_scratch_bytes
            << " navigation_candidates=" << navigation_candidate_bytes
            << " direct_queue_scratch=" << direct_queue_bytes
            << " graph_scratch=" << graph_scratch_bytes
            << " anchor_route=" << anchor_route_bytes
            << " dynamic_route=" << dynamic_route_bytes
            << " dynamic_route_codes=" << dynamic_route_code_bytes
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
  control_region_offset = static_cast<size_t>(
    align_up(graph_scratch_offset + graph_scratch_bytes, 256));
  const size_t control_snapshot_bytes =
    index.shards.size() * sizeof(format::StorageControlBlock);
  const size_t route_snapshot_offset = static_cast<size_t>(align_up(
    control_snapshot_bytes, alignof(format::StorageRoutePublication)));
  const size_t route_snapshot_bytes =
    index.shards.size() * sizeof(format::StorageRoutePublication);
  const size_t route_sequence_before_offset = static_cast<size_t>(align_up(
    route_snapshot_offset + route_snapshot_bytes, alignof(u64)));
  const size_t route_sequence_after_offset = route_sequence_before_offset +
    index.shards.size() * sizeof(u64);
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
  d_anchor_graph_records = d_remote_buffer + anchor_graph_region_offset;
  d_dynamic_code_records = d_remote_buffer + dynamic_code_region_offset;
  d_exact_records = d_remote_buffer + exact_region_offset;
  d_graph_scratch = d_remote_buffer + graph_scratch_offset;
  d_control_snapshots = reinterpret_cast<format::StorageControlBlock*>(
    d_remote_buffer + control_region_offset);
  d_storage_route_snapshots = reinterpret_cast<
    format::StorageRoutePublication*>(
      d_remote_buffer + control_region_offset + route_snapshot_offset);
  d_storage_route_sequence_before = reinterpret_cast<u64*>(
    d_remote_buffer + control_region_offset + route_sequence_before_offset);
  d_storage_route_sequence_after = reinterpret_cast<u64*>(
    d_remote_buffer + control_region_offset + route_sequence_after_offset);

  control_bootstrapper = std::make_unique<NavigationBootstrapper>(
    config, channel_context, connection_manager, remote_regions,
    d_remote_buffer, remote_buffer_bytes);
  std::cerr << "[gpu-search] bootstrap=CPU-posted GPUDirect RDMA; "
               "queries=strict GPU-initiated GPUNetIO\n";
  initialize_storage_reclaim_ack();
  // Fail before accepting queries when storage nodes do not expose the
  // canonical fixed-route extension. A concurrent publication may produce a
  // transient empty result and will simply be retried by maintenance.
  (void)read_storage_route_publications();
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
  device_allocate(d_direct_batch_statuses,
                  static_cast<size_t>(query_slots) * index.shards.size(),
                  "cudaMalloc(GPUNetIO owner completion statuses)");
  mapped_host_allocate(direct_owner_phases_host, d_direct_owner_phases,
                       direct_batch_queue_count,
                       "cudaHostAlloc(GPUNetIO owner runtime phases)");
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
  mapped_host_allocate(resident_pq_erase_updates_host,
                       d_resident_pq_erase_updates,
                       delta_command_capacity,
                       "cudaHostAlloc(resident dynamic PQ erase staging)");
  mapped_host_allocate(dynamic_route_updates_host,
                       d_dynamic_route_updates,
                       dynamic_route_capacity,
                       "cudaHostAlloc(dynamic query route staging)");
  mapped_host_allocate(dynamic_route_code_updates_host,
                       d_dynamic_route_code_updates,
                       static_cast<size_t>(dynamic_route_capacity) *
                         index.layout.code_bytes,
                       "cudaHostAlloc(dynamic query route code staging)");
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
  device_allocate(d_resident_pq_codes,
                  static_cast<size_t>(resident_pq_capacity) * code_bytes,
                  "cudaMalloc(resident dynamic PQ codes)");
  device_allocate(d_resident_pq_keys, resident_pq_table_capacity,
                  "cudaMalloc(resident dynamic PQ keys)");
  device_allocate(d_resident_pq_slots, resident_pq_table_capacity,
                  "cudaMalloc(resident dynamic PQ slots)");
  device_allocate(d_resident_pq_positions, resident_pq_capacity,
                  "cudaMalloc(resident dynamic PQ positions)");
  check_cuda(cudaMemset(d_resident_pq_keys, 0,
                        static_cast<size_t>(resident_pq_table_capacity) *
                          sizeof(u64)),
             "cudaMemset(resident dynamic PQ keys)");
  check_cuda(cudaMemset(d_resident_pq_slots, 0xff,
                        static_cast<size_t>(resident_pq_table_capacity) *
                          sizeof(u32)),
             "cudaMemset(resident dynamic PQ slots)");
  check_cuda(cudaMemset(d_resident_pq_positions, 0xff,
                        static_cast<size_t>(resident_pq_capacity) * sizeof(u32)),
             "cudaMemset(resident dynamic PQ positions)");
  device_allocate(d_delta_count, 1, "cudaMalloc(navigation delta count)");
  device_allocate(d_dynamic_route_slots, dynamic_route_capacity,
                  "cudaMalloc(dynamic query route slots)");
  device_allocate(d_dynamic_route_pq_codes,
                  static_cast<size_t>(dynamic_route_capacity) * code_bytes,
                  "cudaMalloc(dynamic query route PQ codes)");
  check_cuda(cudaMemset(d_dynamic_route_slots, 0,
                        static_cast<size_t>(dynamic_route_capacity) *
                          sizeof(DeviceDynamicRouteSlot)),
             "cudaMemset(dynamic query route slots)");
  check_cuda(cudaMemset(d_dynamic_route_pq_codes, 0,
                        static_cast<size_t>(dynamic_route_capacity) *
                          code_bytes),
             "cudaMemset(dynamic query route PQ codes)");
  clear_delta_device_state();

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
  constexpr u32 warp_width = 32;
  const u32 owner_warps_per_block = kPersistentQueryThreads / warp_width;
  owner_kernel_blocks =
    (direct_batch_queue_count + owner_warps_per_block - 1) /
    owner_warps_per_block;
  const u32 resident_blocks = static_cast<u32>(
    std::max(1, properties.multiProcessorCount));
  constexpr u32 control_blocks = 2;
  if (owner_kernel_blocks + control_blocks >= resident_blocks) {
    throw std::runtime_error(
      "GPU has too few SMs to keep GPUNetIO owners and control resident");
  }
  const u64 requested_blocks = static_cast<u64>(
    std::max(1, properties.multiProcessorCount)) * config.gpu_persistent_blocks_per_sm;
  const u64 useful_blocks = std::max<u64>(1, config.num_threads);
  const u64 resident_query_blocks =
    resident_blocks - owner_kernel_blocks - control_blocks;
  kernel_blocks = static_cast<u32>(std::min({
    static_cast<u64>(query_slots), requested_blocks, useful_blocks,
    resident_query_blocks}));

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
    .direct_owner_phases = d_direct_owner_phases,
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
    .resident_pq_codes = d_resident_pq_codes,
    .resident_pq_keys = d_resident_pq_keys,
    .resident_pq_slots = d_resident_pq_slots,
    .resident_pq_positions = d_resident_pq_positions,
    .resident_pq_capacity = resident_pq_capacity,
    .resident_pq_table_capacity = resident_pq_table_capacity,
    .delta_supersede_updates = d_delta_supersede_updates,
    .delta_override_updates = d_delta_override_updates,
    .delta_durable_updates = d_delta_durable_updates,
    .resident_pq_erase_updates = d_resident_pq_erase_updates,
    .dynamic_route_updates = d_dynamic_route_updates,
    .dynamic_route_code_updates = d_dynamic_route_code_updates,
    .dynamic_route_slots = d_dynamic_route_slots,
    .dynamic_route_pq_codes = d_dynamic_route_pq_codes,
    .dynamic_route_capacity = dynamic_route_capacity,
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
    .graph_scratch = d_graph_scratch,
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
    .result_ids = d_result_ids,
    .result_distances = d_result_distances,
  };
  start_persistent_kernel();
  admission_thread = std::thread([this] { admission_loop(); });
  completion_thread = std::thread([this] { completion_loop(); });
  maintenance_thread = std::thread([this] { maintenance_loop(); });
}

}  // namespace gpu_search
