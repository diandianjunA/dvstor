#include "memory_node/memory_node.hh"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "vamana/storage_layout_resolver.hh"

MemoryNode::MemoryNode(Configuration& config)
    : context_(config), cm_(context_, config), num_clients_(config.num_clients),
      storage_id_(config.storage_id),
      num_storage_nodes_(config.storage_peers.empty() ? config.num_server_nodes()
                                                      : static_cast<u32>(config.storage_peers.size())),
      storage_owner_peer_rdma_tokens_(std::max<u32>(1, config.storage_owner_peer_rdma_tokens)),
      index_region_(context_),
      peer_rdma_read_outstanding_(num_storage_nodes_),
      mn_memory_bytes_(static_cast<u64>(config.mn_memory_gb) * 1073741824ul) {
  for (auto& credit : peer_rdma_read_outstanding_) {
    credit.store(0, std::memory_order_relaxed);
  }
  cm_.connect_to_clients();

  // Keep the lifecycle thread unpinned. Storage workers are created later and
  // must not inherit a one-CPU affinity mask.

  // receive runtimes parameters from initiator
  configuration::Parameters p{};
  LocalMemoryRegion region{context_, &p, sizeof(configuration::Parameters)};

  cm_.initiator_qp->post_receive(region);
  context_.receive();

  num_compute_threads_ = p.num_threads;
  const u32 gpu_rdma_qps = p.gpu_rdma_qps;
  const filepath_t index_prefix = config.resolved_index_prefix();
  index_prefix_ = index_prefix;
  VectorDType startup_dtype = config.resolved_vector_dtype();
  const filepath_t meta_file = filepath_t(index_prefix.string() + ".meta.json");
  lib_assert(!index_prefix.empty(), "GPU storage node requires --index-prefix");
  lib_assert(std::filesystem::exists(meta_file),
             "GPU index metadata does not exist: " + meta_file.string());
  {
    service::index_metadata::Metadata metadata;
    str metadata_error;
    lib_assert(service::index_metadata::load_metadata(index_prefix, metadata, &metadata_error), metadata_error);
    lib_assert(metadata.dim == config.dim, "index metadata dim mismatch on storage node");
    lib_assert(metadata.R == config.R, "index metadata R mismatch on storage node");
    lib_assert(metadata.num_memory_nodes == num_storage_nodes_, "index metadata storage-node count mismatch");
    const bool compatible_quantizer = metadata.navigation_quantizer == "opq_pq" ||
      metadata.navigation_quantizer == "opq_pq16";
    const bool compatible_navigation = metadata.navigation_format == "opq_pq_graph_v1" ||
      metadata.navigation_format == "opq_pq16_graph_v1";
    gpu_stream_layout_ = metadata.schema_version == gpu_search::format::kMetadataSchemaVersion &&
      metadata.node_layout == "plain" &&
      metadata.storage_format == "vamana_compact_v1" &&
      compatible_quantizer && compatible_navigation;
    lib_assert(gpu_stream_layout_,
               "storage node requires a schema-15 compact OPQ/PQ index");
    lib_assert(storage_id_ < num_storage_nodes_, "invalid GPU storage shard id");
    lib_assert(metadata.hot_graph_entry_counts.size() == num_storage_nodes_,
               "GPU storage metadata has invalid static shard counts");
    lib_assert(metadata.hot_graph_dynamic_base_offsets.size() == num_storage_nodes_,
               "GPU storage metadata has invalid dynamic shard offsets");
    lib_assert(metadata.storage_control_remote_offsets.size() == num_storage_nodes_ &&
                 metadata.dynamic_node_base_offsets.size() == num_storage_nodes_,
               "GPU storage metadata has invalid control/dynamic-node offsets");
    gpu_static_node_count_ = metadata.hot_graph_entry_counts[storage_id_];
    gpu_static_dynamic_base_ = metadata.hot_graph_dynamic_base_offsets[storage_id_];
    gpu_storage_control_offset_ = metadata.storage_control_remote_offsets[storage_id_];
    gpu_dynamic_node_base_ = metadata.dynamic_node_base_offsets[storage_id_];
    gpu_navigation_code_bytes_ = metadata.navigation_code_bytes;
    lib_assert(gpu_navigation_code_bytes_ > 0 &&
                 gpu_navigation_code_bytes_ <=
                   gpu_search::format::kStorageRouteMaxCodeBytes,
               "navigation PQ width exceeds the fixed storage route publication");
    gpu_navigation_model_checksum_ = metadata.navigation_model_checksum;
    lib_assert(std::isfinite(metadata.partition_cross_shard_ratio) &&
                 metadata.partition_cross_shard_ratio >= 0.0 &&
                 metadata.partition_cross_shard_ratio <= 1.0,
               "index metadata partition_cross_shard_ratio is invalid");
    if (config.vector_data_type != "auto" && config.resolved_vector_dtype() != metadata.vector_dtype) {
      lib_failure("configured vector-data-type=" + config.vector_data_type +
                  " does not match index metadata vector_data_type=" + vector_dtype_name(metadata.vector_dtype));
    }
    startup_dtype = metadata.vector_dtype;
    config.vector_data_type = vector_dtype_name(startup_dtype);
    VamanaNode::disable_hot_graph();
    VamanaNode::init_static_storage(config.dim, config.R, startup_dtype);
    lib_assert(metadata.schema_version == gpu_search::format::kMetadataSchemaVersion &&
                 metadata.node_layout == "plain",
               "index storage format is obsolete; rebuild with the current offline builder");
    lib_assert(metadata.vector_component_size == VamanaNode::vector_component_size(),
               "index metadata vector component size mismatch on storage node");
    lib_assert(metadata.vector_bytes == VamanaNode::vector_bytes(),
               "index metadata vector byte size mismatch on storage node");
    lib_assert(metadata.node_size == VamanaNode::total_size(), "index metadata node size mismatch on storage node");
    lib_assert(metadata.graph_hot_bytes == VamanaNode::graph_hot_bytes() &&
               metadata.vector_offset == VamanaNode::offset_vector(),
               "index metadata storage offsets mismatch on storage node");
    lib_assert(metadata.hot_graph_pointer_bytes == vamana::hot_graph::kCompactPointerBytes &&
               metadata.hot_graph_entry_size == VamanaNode::hot_graph_entry_size() &&
               metadata.hot_graph_offsets.size() == num_storage_nodes_ &&
               metadata.hot_graph_entry_counts.size() == num_storage_nodes_,
               "index hot graph metadata mismatch on storage node");
    lib_assert(metadata.hot_graph_dynamic_base_offsets.size() == num_storage_nodes_ &&
               metadata.dynamic_node_base_offsets.size() == num_storage_nodes_ &&
               metadata.hot_graph_dynamic_record_bytes >=
                 metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size &&
               metadata.hot_graph_dynamic_hot_offset >= VamanaNode::total_size() &&
               metadata.dynamic_navigation_code_offset >=
                 metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size &&
               metadata.hot_graph_dynamic_record_bytes >=
                 metadata.dynamic_navigation_code_offset + metadata.navigation_code_bytes,
               "index dynamic hot graph metadata mismatch on storage node");
    VamanaNode::configure_hot_graph(metadata.hot_graph_offsets,
                                    metadata.hot_graph_entry_counts,
                                    metadata.hot_graph_entry_size,
                                    metadata.hot_graph_shard_bits,
                                    metadata.dynamic_node_base_offsets,
                                    metadata.hot_graph_dynamic_record_bytes,
                                    metadata.hot_graph_dynamic_hot_offset,
                                    metadata.dynamic_navigation_code_offset,
                                    metadata.navigation_code_bytes);
    lib_assert(VamanaNode::HAS_HOT_GRAPH, "failed to enable compact hot graph on storage node");
    lib_assert(gpu_search::pq::read_model(
                 index_path::navigation_model_file(index_prefix,
                                                   metadata.pq_subquantizers),
                 gpu_navigation_model_, &metadata_error),
               metadata_error);
    lib_assert(gpu_navigation_model_.checksum() == metadata.navigation_model_checksum &&
                 gpu_navigation_model_.code_bytes() == metadata.navigation_code_bytes &&
                 gpu_navigation_model_.dim == metadata.dim,
               "storage-node PQ model does not match index metadata");
    owner_idmap_required_ = metadata.idmap_format == "owner_sharded_v1";
    print_status("loaded index metadata from " + index_prefix.string() +
                 " (layout=" + VamanaNode::layout_name() +
                 ", vector_data_type=" + VamanaNode::vector_dtype_name() + ")");
    print_status(
      "storage-owner stage2 candidate width L=" +
      std::to_string(config.resolved_storage_owner_construction_width()) +
      " per shard");
  }
  allocate_memory();

  // free-ptr is initialized to 16 (points to first free address in the buffer)
  *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = 16;

  lib_assert(!config.server_index_file.empty(),
             "GPU storage node requires --server-index-file");
  const auto [success, message] = load_index_file(config.server_index_file.string());
  lib_assert(success, message);
  if (owner_idmap_required_) {
    lib_assert(load_owner_idmap(index_prefix_), "failed to load owner-sharded idmap");
  }

  if (config.storage_owner_update_mode == "local_stitch") {
    initialize_storage_owner_route_table();
  }

  print_status("register memory and distribute access token");
  index_region_.register_memory(index_buffer_.get_full_buffer(), index_buffer_.buffer_size, true);
  MemoryRegionToken token = index_region_.createToken();

  // send access token to all compute nodes
  for (QP& qp : cm_.client_qps) {
    qp->post_send_inlined(std::addressof(token), sizeof(token), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();
  }

  // connect for each compute thread a new QP
  print_status("connect QPs of compute threads");
  vec<u_ptr<DetachedQP>> qps;

  // note: no need for QP sharing on the memory server side
  const u32 qps_per_node = gpu_rdma_qps;
  if (gpu_rdma_qps > 0) {
    print_status("reserving " + std::to_string(gpu_rdma_qps) +
                 " GPU/bootstrap QPs per compute node");
  }
  qps.reserve(num_clients_ * qps_per_node);

  for (QP& client_qp : cm_.client_qps) {
    for (u32 thread_id = 0; thread_id < qps_per_node; ++thread_id) {
      auto& qp = qps.emplace_back(std::make_unique<DetachedQP>(context_));
      qp->connect(context_, context_.get_lid(), client_qp);
    }
  }

  // notify compute nodes that we are ready
  cm_.synchronize();

  wait_for_start_signal();
  setup_storage_peers(config);
  setup_insert_runtime(config);
  storage_worker_config_ = std::make_unique<Configuration>(config);
  start_peer_reverse_update_runtime(config);
  start_storage_owner_maintenance_runtime(config);
  start_storage_owner_insert_workers(config);
  service_storage_runtime(config);

  storage_insert_shutdown_.store(true, std::memory_order_release);
  if (storage_insert_tasks_) storage_insert_tasks_->notify_all();
  for (auto& worker : storage_insert_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  stop_storage_owner_maintenance_runtime();
  stop_peer_reverse_update_runtime();

  print_status("memory node shutting down");
  std::cout << timing_ << std::endl;
}

u64 MemoryNode::elapsed_ns_since(const std::chrono::steady_clock::time_point start) {
  return static_cast<u64>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

u64 MemoryNode::scale_ns(const u64 value, const u32 part, const u32 total) {
  if (value == 0 || part == 0 || total == 0) {
    return 0;
  }
  const u64 quotient = value / total;
  const u64 remainder = value % total;
  return quotient * part + (remainder * part) / total;
}

MemoryNode::InsertBreakdownCounters MemoryNode::scale_breakdown(const InsertBreakdownCounters& counters,
                                               const u32 part,
                                               const u32 total) {
  InsertBreakdownCounters out{};
  out.storage_owner_queue_wait_ns = scale_ns(counters.storage_owner_queue_wait_ns, part, total);
  out.storage_owner_medoid_ns = scale_ns(counters.storage_owner_medoid_ns, part, total);
  out.storage_owner_search_ns = scale_ns(counters.storage_owner_search_ns, part, total);
  out.storage_owner_prune_ns = scale_ns(counters.storage_owner_prune_ns, part, total);
  out.storage_owner_write_node_ns = scale_ns(counters.storage_owner_write_node_ns, part, total);
  out.storage_owner_local_reverse_ns = scale_ns(counters.storage_owner_local_reverse_ns, part, total);
  out.storage_owner_remote_reverse_ns = scale_ns(counters.storage_owner_remote_reverse_ns, part, total);
  out.storage_owner_peer_reverse_apply_ns =
    scale_ns(counters.storage_owner_peer_reverse_apply_ns, part, total);
  out.storage_owner_response_send_ns = scale_ns(counters.storage_owner_response_send_ns, part, total);
  out.storage_owner_prepare_mutation_ns =
    scale_ns(counters.storage_owner_prepare_mutation_ns, part, total);
  out.storage_owner_allocate_node_ns =
    scale_ns(counters.storage_owner_allocate_node_ns, part, total);
  out.storage_owner_publish_mutation_ns =
    scale_ns(counters.storage_owner_publish_mutation_ns, part, total);
  out.storage_owner_schedule_maintenance_ns =
    scale_ns(counters.storage_owner_schedule_maintenance_ns, part, total);
  out.storage_owner_response_build_ns =
    scale_ns(counters.storage_owner_response_build_ns, part, total);
  out.storage_owner_search_select_ns = scale_ns(counters.storage_owner_search_select_ns, part, total);
  out.storage_owner_search_neighbor_read_ns =
    scale_ns(counters.storage_owner_search_neighbor_read_ns, part, total);
  out.storage_owner_search_snapshot_read_ns =
    scale_ns(counters.storage_owner_search_snapshot_read_ns, part, total);
  out.storage_owner_search_distance_ns = scale_ns(counters.storage_owner_search_distance_ns, part, total);
  out.storage_owner_search_beam_update_ns =
    scale_ns(counters.storage_owner_search_beam_update_ns, part, total);
  out.storage_owner_search_result_sort_ns =
    scale_ns(counters.storage_owner_search_result_sort_ns, part, total);
  out.storage_owner_prune_snapshot_read_ns =
    scale_ns(counters.storage_owner_prune_snapshot_read_ns, part, total);
  out.storage_owner_prune_distance_ns =
    scale_ns(counters.storage_owner_prune_distance_ns, part, total);
  out.storage_owner_prune_sort_ns = scale_ns(counters.storage_owner_prune_sort_ns, part, total);
  out.storage_owner_prune_pair_distance_ns =
    scale_ns(counters.storage_owner_prune_pair_distance_ns, part, total);
  return out;
}

void MemoryNode::allocate_memory() {
  const auto t_allocate = timing_.create_enroll("allocate_index_buffer");
  std::cerr << "allocation size: " << mn_memory_bytes_ << std::endl;

  t_allocate->start();
  const size_t available_memory = index_buffer_.get_memory_size();
  lib_assert(mn_memory_bytes_ <= available_memory, "allocation failed");

  index_buffer_.allocate(mn_memory_bytes_);
  index_buffer_.touch_memory();
  t_allocate->stop();
}

void MemoryNode::wait_for_start_signal() {
  print_status("waiting for compute-node startup barrier");
  storage_startup::Request request{};
  LocalMemoryRegion region{context_, &request, sizeof(request)};
  cm_.initiator_qp->post_receive(region);
  context_.receive();
  const storage_startup::Response response{
    .ready = request.magic == storage_startup::kMagic,
  };
  cm_.initiator_qp->post_send_inlined(
    &response, sizeof(response), IBV_WR_SEND);
  context_.poll_send_cq_until_completion();
  lib_assert(response.ready, "invalid compute-node startup request");
}

std::pair<bool, str> MemoryNode::load_index_file(const str& path) {
  std::ifstream file{path, std::ios::binary};
  if (!file.good()) {
    return {false, "file \"" + path + "\" does not exist"};
  }

  file.unsetf(std::ios::skipws);
  file.seekg(0, std::ios::end);
  const size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  if (file_size > index_buffer_.buffer_size) {
    return {false, "buffer too small for index file"};
  }

  print_status("loading index (" + std::to_string(file_size) + " Bytes) from " + path);
  auto t_read = timing_.create_enroll("read_index_buffer");
  t_read->start();
  file.read(reinterpret_cast<char*>(index_buffer_.get_full_buffer()), file_size);
  t_read->stop();

  if (!file) {
    return {false, "read failed for " + path};
  }
  if (!gpu_stream_layout_ || gpu_static_node_count_ == 0 ||
      gpu_static_dynamic_base_ == 0) {
    return {false, "GPU storage metadata cannot materialize the PQ stream"};
  }
  const u64 persisted_free_pointer =
    *reinterpret_cast<const u64*>(index_buffer_.get_full_buffer());
  if (persisted_free_pointer != gpu_static_dynamic_base_) {
    return {false, "GPU navigation requires a compacted static shard before startup"};
  }
  const u64 fixed_nodes_end = gpu_search::format::kNodeBaseOffset +
    gpu_static_node_count_ * VamanaNode::total_size();
  if (fixed_nodes_end > gpu_static_dynamic_base_ ||
      gpu_static_dynamic_base_ > file_size) {
    return {false, "GPU storage shard is truncated or has inconsistent static metadata"};
  }

  const u64 remote_offset = gpu_storage_control_offset_ +
    gpu_search::format::kStorageControlBytes;
  const u64 payload_bytes = gpu_static_node_count_ * gpu_navigation_code_bytes_;
  if (remote_offset == 0 || remote_offset > index_buffer_.buffer_size ||
      payload_bytes > index_buffer_.buffer_size - remote_offset) {
    return {false, "buffer too small for GPU PQ stream"};
  }

  const filepath_t code_path = index_path::navigation_code_for_shard(
    path, gpu_navigation_code_bytes_);
  gpu_search::format::CodeHeader header;
  str error;
  if (!gpu_search::format::read_code_header(code_path, header, &error) ||
      header.memory_node != storage_id_ ||
      header.node_size != VamanaNode::total_size() ||
      header.code_bytes != gpu_navigation_code_bytes_ ||
      header.model_checksum != gpu_navigation_model_checksum_ ||
      header.entry_count != gpu_static_node_count_ ||
      header.remote_offset != remote_offset ||
      header.payload_bytes != payload_bytes) {
    return {false, error.empty() ? "incompatible GPU PQ sidecar " + code_path.string()
                                 : error};
  }
  std::ifstream codes{code_path, std::ios::binary};
  codes.seekg(static_cast<std::streamoff>(sizeof(header)));
  constexpr size_t chunk_bytes = 64ull << 20;
  u64 checksum = gpu_search::format::checksum64_initial();
  for (u64 offset = 0; offset < header.payload_bytes; offset += chunk_bytes) {
    const size_t bytes = static_cast<size_t>(
      std::min<u64>(chunk_bytes, header.payload_bytes - offset));
    byte_t* destination = index_buffer_.get_full_buffer() + header.remote_offset + offset;
    codes.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (static_cast<size_t>(codes.gcount()) != bytes) {
      return {false, "short read from " + code_path.string()};
    }
    checksum = gpu_search::format::checksum64_update(checksum, destination, bytes);
  }
  if (checksum != header.payload_checksum) {
    return {false, "GPU PQ code sidecar payload checksum mismatch: " + code_path.string()};
  }
  print_status("loaded GPU PQ codes (" + std::to_string(header.payload_bytes) +
               " Bytes) at remote offset " + std::to_string(header.remote_offset));

  const u64 region_end = remote_offset + payload_bytes;
  if (gpu_storage_control_offset_ !=
        gpu_search::format::align_up(gpu_static_dynamic_base_, 64) ||
      gpu_dynamic_node_base_ < region_end ||
      (gpu_dynamic_node_base_ - gpu_static_dynamic_base_) %
        VamanaNode::allocation_size() != 0 ||
      gpu_dynamic_node_base_ > index_buffer_.buffer_size) {
    return {false, "GPU storage control/dynamic-node layout is inconsistent"};
  }
  std::memset(index_buffer_.get_full_buffer() + gpu_storage_control_offset_, 0,
              gpu_search::format::kStorageControlBytes);
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  *control = gpu_search::format::StorageControlBlock{
    .shard_id = storage_id_,
    .dynamic_record_bytes = static_cast<u32>(VamanaNode::allocation_size()),
    .dynamic_hot_offset = VamanaNode::HOT_GRAPH_DYNAMIC_HOT_OFFSET,
    .dynamic_code_offset = VamanaNode::HOT_GRAPH_DYNAMIC_CODE_OFFSET,
    .code_bytes = VamanaNode::HOT_GRAPH_DYNAMIC_CODE_BYTES,
    .compute_client_count = num_clients_,
    .dynamic_high_watermark = gpu_dynamic_node_base_,
  };
  auto* route_publication = reinterpret_cast<
    gpu_search::format::StorageRoutePublication*>(
      index_buffer_.get_full_buffer() + gpu_storage_control_offset_ +
      gpu_search::format::kStorageRoutePublicationOffset);
  *route_publication = gpu_search::format::StorageRoutePublication{
    .sequence_begin = 2,
    .shard_id = storage_id_,
    .code_bytes = gpu_navigation_code_bytes_,
    .sequence_end = 2,
  };
  route_publication->body_checksum =
    gpu_search::format::storage_route_body_checksum(*route_publication);
  if (num_clients_ == 0 || num_clients_ > gpu_search::format::kMaxComputeClients) {
    return {false, "compute client count exceeds the storage reclaim control capacity"};
  }
  *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = gpu_dynamic_node_base_;

  return {true, ""};
}

void MemoryNode::initialize_storage_owner_route_table() {
  storage_owner_route_table_ =
    std::make_unique<vamana::routing::AdaptiveRouteTable>(
      VamanaNode::DIM, num_storage_nodes_);
  storage_owner_route_snapshot_.resize(
    storage_owner_route_table_->capacity());

  // Static graph nodes are bootstrap representatives only. Every committed
  // mutation below updates the fixed-capacity centers/representatives, so the
  // route is not frozen to the offline sample. Sampling the midpoint of each
  // equal shard interval avoids another sidecar and gives every slot a live
  // graph entry before the first online mutation.
  const u32 bootstrap_count = static_cast<u32>(std::min<u64>(
    gpu_static_node_count_,
    vamana::routing::AdaptiveRouteTable::kSlotsPerShard));
  vec<element_t> decoded(VamanaNode::DIM);
  u32 installed = 0;
  for (u32 rank = 0; rank < bootstrap_count; ++rank) {
    const u64 slot =
      ((static_cast<u64>(rank) * 2 + 1) * gpu_static_node_count_) /
      (static_cast<u64>(bootstrap_count) * 2);
    const RemotePtr pointer{
      storage_id_,
      gpu_search::format::kNodeBaseOffset + slot * VamanaNode::total_size()};
    const byte_t* vector = local_live_vector(pointer);
    if (vector == nullptr) continue;
    const byte_t* node = local_node_ptr(pointer);
    const node_t id = *reinterpret_cast<const node_t*>(
      node + VamanaNode::offset_id());
    const u32 generation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_generation());
    decode_storage_vector_to_float(
      vector, VamanaNode::vector_dtype(), VamanaNode::DIM, decoded.data());
    installed += storage_owner_route_table_->observe(
      storage_id_, id, generation, pointer,
      span<const element_t>{decoded.data(), decoded.size()}) ? 1u : 0u;
  }
  lib_assert(installed != 0,
             "adaptive storage-owner route has no live bootstrap entry");
  print_status(
    "storage-owner adaptive route initialized on shard " +
    std::to_string(storage_id_) + ": live_entries=" +
    std::to_string(installed) + " fixed_capacity=" +
    std::to_string(vamana::routing::AdaptiveRouteTable::kSlotsPerShard));
  publish_storage_owner_route_table();
}

void MemoryNode::publish_storage_owner_route_table() {
  if (storage_owner_route_table_ == nullptr ||
      storage_owner_route_snapshot_.size() !=
        storage_owner_route_table_->capacity()) {
    return;
  }
  std::lock_guard<std::mutex> publication_lock(
    storage_owner_route_publication_mutex_);
  storage_owner_route_table_->snapshot_route_slots(
    span<vamana::routing::AdaptiveRouteTable::RouteSlotSnapshot>{
      storage_owner_route_snapshot_.data(),
      storage_owner_route_snapshot_.size()});

  gpu_search::format::StorageRoutePublication next{
    .shard_id = storage_id_,
    .code_bytes = gpu_navigation_code_bytes_,
  };
  const size_t begin = static_cast<size_t>(storage_id_) *
    vamana::routing::AdaptiveRouteTable::kSlotsPerShard;
  for (u32 slot = 0;
       slot < vamana::routing::AdaptiveRouteTable::kSlotsPerShard;
       ++slot) {
    const auto& source = storage_owner_route_snapshot_[begin + slot];
    auto& destination_slot = next.slots[slot];
    if (!source.initialized) continue;
    destination_slot = gpu_search::format::StorageRouteSlot{
      .remote_node = source.live ? source.entry.raw_address : 0,
      .id = source.id,
      .generation = source.generation,
    };
    if (!source.live) continue;
    const u64 node_offset = source.entry.byte_offset();
    const byte_t* navigation_code = nullptr;
    if (node_offset >= gpu_dynamic_node_base_) {
      navigation_code = index_buffer_.get_full_buffer() + node_offset +
        VamanaNode::HOT_GRAPH_DYNAMIC_CODE_OFFSET;
    } else if (node_offset >= gpu_search::format::kNodeBaseOffset) {
      const u64 relative = node_offset - gpu_search::format::kNodeBaseOffset;
      if (relative % VamanaNode::total_size() == 0) {
        const u64 ordinal = relative / VamanaNode::total_size();
        if (ordinal < gpu_static_node_count_) {
          navigation_code = index_buffer_.get_full_buffer() +
            gpu_storage_control_offset_ +
            gpu_search::format::kStorageControlBytes +
            ordinal * gpu_navigation_code_bytes_;
        }
      }
    }
    lib_assert(navigation_code != nullptr &&
                 gpu_navigation_code_bytes_ <=
                   gpu_search::format::kStorageRouteMaxCodeBytes,
               "adaptive route entry has no publishable navigation code");
    std::memcpy(destination_slot.navigation_code.data(), navigation_code,
                gpu_navigation_code_bytes_);
  }
  next.body_checksum =
    gpu_search::format::storage_route_body_checksum(next);

  auto* destination = reinterpret_cast<
    gpu_search::format::StorageRoutePublication*>(
      index_buffer_.get_full_buffer() + gpu_storage_control_offset_ +
      gpu_search::format::kStorageRoutePublicationOffset);
  std::atomic_ref<u64> begin_sequence(destination->sequence_begin);
  std::atomic_ref<u64> end_sequence(destination->sequence_end);
  const u64 current = begin_sequence.load(std::memory_order_relaxed);
  const u64 odd = (current & ~u64{1}) + 1;
  const u64 even = odd + 1;
  begin_sequence.store(odd, std::memory_order_release);
  end_sequence.store(odd, std::memory_order_release);
  std::memcpy(
    reinterpret_cast<byte_t*>(destination) +
      offsetof(gpu_search::format::StorageRoutePublication, magic),
    reinterpret_cast<const byte_t*>(&next) +
      offsetof(gpu_search::format::StorageRoutePublication, magic),
    offsetof(gpu_search::format::StorageRoutePublication, sequence_end) -
      offsetof(gpu_search::format::StorageRoutePublication, magic));
  std::atomic_thread_fence(std::memory_order_release);
  end_sequence.store(even, std::memory_order_release);
  begin_sequence.store(even, std::memory_order_release);
}

vec<RemotePtr> MemoryNode::storage_owner_route_entries(
    const span<const element_t> query) {
  vec<RemotePtr> entries;
  if (storage_owner_route_table_ == nullptr) return entries;
  const auto routes = storage_owner_route_table_->routes_in_shard(
    query, storage_id_);
  entries.reserve(routes.size());
  for (const auto& route : routes) entries.push_back(route.entry);
  if (!entries.empty()) return entries;

  // Exceptional slow path only: long delete/upsert churn can retire every
  // current representative. Search the authoritative live set rather than a
  // fixed sample so a non-empty shard can never become unreachable. A newly
  // found entry is installed back into an empty adaptive slot before return.
  entries.reserve(vamana::routing::AdaptiveRouteTable::kSlotsPerShard);
  const auto append_live = [&](RemotePtr pointer) {
    if (entries.size() >=
          vamana::routing::AdaptiveRouteTable::kSlotsPerShard ||
        pointer.is_null() || pointer.memory_node() != storage_id_ ||
        local_live_vector(pointer) == nullptr ||
        std::find(entries.begin(), entries.end(), pointer) != entries.end()) {
      return;
    }
    entries.push_back(pointer);
  };

  for (DynamicFreshnessShard& freshness : dynamic_freshness_shards_) {
    vec<RemotePtr> live;
    {
      std::lock_guard<std::mutex> lock(freshness.mutex);
      live.reserve(std::min<size_t>(
        freshness.entries.size(),
        vamana::routing::AdaptiveRouteTable::kSlotsPerShard));
      for (const auto& [id, entry] : freshness.entries) {
        (void)id;
        if (!entry.deleted && !entry.current.is_null()) {
          live.push_back(entry.current);
          if (live.size() ==
              vamana::routing::AdaptiveRouteTable::kSlotsPerShard) break;
        }
      }
    }
    for (const RemotePtr pointer : live) append_live(pointer);
    if (entries.size() ==
        vamana::routing::AdaptiveRouteTable::kSlotsPerShard) break;
  }

  for (u64 slot = 0; slot < gpu_static_node_count_ &&
       entries.size() < vamana::routing::AdaptiveRouteTable::kSlotsPerShard;
       ++slot) {
    append_live(RemotePtr{
      storage_id_,
      gpu_search::format::kNodeBaseOffset + slot * VamanaNode::total_size()});
  }

  vec<element_t> decoded(VamanaNode::DIM);
  for (const RemotePtr pointer : entries) {
    const byte_t* vector = local_live_vector(pointer);
    if (vector == nullptr) continue;
    const byte_t* node = local_node_ptr(pointer);
    const node_t id = *reinterpret_cast<const node_t*>(
      node + VamanaNode::offset_id());
    const u32 generation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_generation());
    decode_storage_vector_to_float(
      vector, VamanaNode::vector_dtype(), VamanaNode::DIM, decoded.data());
    observe_storage_owner_route(
      id, generation, pointer,
      span<const element_t>{decoded.data(), decoded.size()});
  }

  const auto refreshed = storage_owner_route_table_->routes_in_shard(
    query, storage_id_);
  if (!refreshed.empty()) {
    entries.clear();
    entries.reserve(refreshed.size());
    for (const auto& route : refreshed) entries.push_back(route.entry);
  }
  return entries;
}

void MemoryNode::observe_storage_owner_route(
    node_t id,
    u32 generation,
    RemotePtr entry,
    const span<const element_t> vector) {
  if (storage_owner_route_table_ == nullptr) return;
  // publish_mutation releases the per-ID in-flight claim before this route
  // update. A newer mutation can therefore win in the small intervening
  // window; never let the older completion move a center or revive an entry.
  DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
  std::lock_guard<std::mutex> lock(shard.mutex);
  if (shard.mutations_inflight.contains(id)) return;
  const auto current = shard.entries.find(id);
  if (current != shard.entries.end()) {
    if (current->second.deleted ||
        current->second.generation != generation ||
        current->second.current != entry) {
      return;
    }
  } else {
    const auto base = base_idmap_.find(id);
    if (base == base_idmap_.end() || base->second.deleted ||
        base->second.generation != generation ||
        base->second.current != entry) {
      return;
    }
  }
  lib_assert(entry.memory_node() == storage_id_,
             "adaptive storage-owner route observed a non-local entry");
  if (storage_owner_route_table_->observe(
        storage_id_, id, generation, entry, vector)) {
    publish_storage_owner_route_table();
  }
}

void MemoryNode::invalidate_storage_owner_route(node_t id, u32 generation) {
  if (storage_owner_route_table_ == nullptr) return;
  DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto current = shard.entries.find(id);
  if (current == shard.entries.end() ||
      current->second.generation != generation ||
      !current->second.deleted) {
    return;
  }
  if (storage_owner_route_table_->invalidate(id, generation)) {
    publish_storage_owner_route_table();
  }
}

size_t MemoryNode::align_up(size_t value, size_t alignment) {
  while (value % alignment != 0) {
    ++value;
  }
  return value;
}

distance_t MemoryNode::distance_to_stored_vector(const span<const element_t> query,
                                                const byte_t* stored,
                                                const Configuration& config) const {
  return typed_l2_distance_float_query(
    query, stored, VamanaNode::vector_dtype(), config.dim);
}

distance_t MemoryNode::distance_between_vectors(const byte_t* lhs,
                                                VectorDType lhs_dtype,
                                                const byte_t* rhs,
                                                VectorDType rhs_dtype,
                                                const Configuration& config) const {
  return typed_l2_distance(lhs, lhs_dtype, rhs, rhs_dtype, config.dim);
}

u64 MemoryNode::load_local_node_header_acquire(RemotePtr rptr) const {
  lib_assert(local_shard(rptr.memory_node()),
             "local header lookup received a remote pointer");
  const auto header = vamana::StorageLayoutResolver::header(rptr);
  lib_assert(header.offset <= mn_memory_bytes_ &&
               sizeof(u64) <= mn_memory_bytes_ - header.offset,
             "local header lookup exceeds shard bounds");
  auto* storage = reinterpret_cast<u64*>(
    const_cast<byte_t*>(index_buffer_.get_full_buffer()) + header.offset);
  return std::atomic_ref<u64>(*storage).load(std::memory_order_acquire);
}

const byte_t* MemoryNode::local_live_vector(RemotePtr rptr) const {
  lib_assert(local_shard(rptr.memory_node()),
             "local vector lookup received a remote pointer");
  const auto vector = vamana::StorageLayoutResolver::vector(rptr);
  lib_assert(vector.offset + vector.size <= mn_memory_bytes_,
             "local vector lookup exceeds shard bounds");
  const u64 header = load_local_node_header_acquire(rptr);
  if ((header & VamanaNode::HEADER_DELETED) != 0) {
    return nullptr;
  }
  return index_buffer_.get_full_buffer() + vector.offset;
}

bool MemoryNode::local_shard(u32 shard_id) const { return shard_id == storage_id_; }

byte_t* MemoryNode::local_node_ptr(const RemotePtr& rptr) {
  return index_buffer_.get_full_buffer() + rptr.byte_offset();
}

const byte_t* MemoryNode::local_node_ptr(const RemotePtr& rptr) const {
  return index_buffer_.get_full_buffer() + rptr.byte_offset();
}

void MemoryNode::insert_into_beam(vec<BeamEntry>& beam, const RemotePtr& rptr, distance_t dist, u32 max_beam_width) {
  auto it = std::lower_bound(
    beam.begin(), beam.end(), dist, [](const BeamEntry& entry, distance_t value) { return entry.distance < value; });
  beam.insert(it, {rptr, dist, false});
  if (beam.size() > max_beam_width) {
    beam.resize(max_beam_width);
  }
}
