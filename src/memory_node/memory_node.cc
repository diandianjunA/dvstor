#include "memory_node/memory_node.hh"

#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_set>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "memory_node/storage_owner_maintenance/centroid_lifecycle_policy.hh"
#include "vamana/centroid_seed_policy.hh"
#include "vamana/centroid_state.hh"
#include "vamana/storage_layout_resolver.hh"

MemoryNode::MemoryNode(Configuration& config)
    : context_(config), cm_(context_, config), num_clients_(config.num_clients),
      storage_id_(config.storage_id),
      num_storage_nodes_(config.storage_peers.empty() ? config.num_server_nodes()
                                                      : static_cast<u32>(config.storage_peers.size())),
      vector_id_namespace_size_(config.vector_id_namespace_size),
      storage_owner_peer_rdma_tokens_(std::max<u32>(1, config.storage_owner_peer_rdma_tokens)),
      index_region_(context_),
      peer_rdma_read_outstanding_(num_storage_nodes_),
      mn_memory_bytes_(static_cast<u64>(config.mn_memory_gb) * 1073741824ul) {
  lib_assert(num_storage_nodes_ > 0 &&
               num_storage_nodes_ <= RemotePtr::MEMORY_NODE_MASK + 1,
             "tagged RemotePtr supports between 1 and 64 storage shards");
  lib_assert(mn_memory_bytes_ > 0 &&
               mn_memory_bytes_ <= RemotePtr::BYTE_OFFSET_CAPACITY,
             "storage region exceeds the 256 GiB tagged RemotePtr capacity");
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
    lib_assert(metadata.num_vectors == config.max_vectors,
               "index metadata base-vector count mismatch on storage node");
    lib_assert(metadata.num_memory_nodes == num_storage_nodes_, "index metadata storage-node count mismatch");
    gpu_stream_layout_ = metadata.schema_version == gpu_search::format::kMetadataSchemaVersion &&
      metadata.node_layout == "plain" &&
      metadata.storage_format == "vamana_tagged_v2" &&
      metadata.navigation_quantizer == "opq_pq" &&
      metadata.navigation_format == "opq_pq_graph_v1";
    lib_assert(gpu_stream_layout_,
               "storage node requires a schema-16 tagged OPQ/PQ index");
    lib_assert(storage_id_ < num_storage_nodes_, "invalid GPU storage shard id");
    lib_assert(metadata.hot_graph_entry_counts.size() == num_storage_nodes_,
               "GPU storage metadata has invalid static shard counts");
    lib_assert(metadata.hot_graph_dynamic_base_offsets.size() == num_storage_nodes_,
               "GPU storage metadata has invalid dynamic shard offsets");
    lib_assert(metadata.index_build_fingerprint != 0 &&
                 metadata.shard_build_fingerprints.size() ==
                   num_storage_nodes_ &&
                 metadata.shard_build_fingerprints[storage_id_] != 0,
               "GPU storage metadata has no bound build identity");
    lib_assert(metadata.storage_control_remote_offsets.size() == num_storage_nodes_ &&
                 metadata.dynamic_node_base_offsets.size() == num_storage_nodes_,
               "GPU storage metadata has invalid control/dynamic-node offsets");
    gpu_static_node_count_ = metadata.hot_graph_entry_counts[storage_id_];
    gpu_static_dynamic_base_ = metadata.hot_graph_dynamic_base_offsets[storage_id_];
    gpu_storage_control_offset_ = metadata.storage_control_remote_offsets[storage_id_];
    gpu_dynamic_node_base_ = metadata.dynamic_node_base_offsets[storage_id_];
    gpu_navigation_code_bytes_ = metadata.navigation_code_bytes;
    lib_assert(gpu_navigation_code_bytes_ > 0,
               "navigation PQ width must be positive");
    gpu_navigation_model_checksum_ = metadata.navigation_model_checksum;
    gpu_index_build_fingerprint_ = metadata.index_build_fingerprint;
    gpu_shard_build_fingerprint_ =
      metadata.shard_build_fingerprints[storage_id_];
    lib_assert(floating_value_is_finite(metadata.partition_cross_shard_ratio) &&
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
               metadata.vector_offset == VamanaNode::offset_vector() &&
               metadata.slot_incarnation_offset ==
                 VamanaNode::offset_slot_incarnation() &&
               metadata.remote_ptr_format ==
                 "tagged_inc24_shard6_off34x16_v1",
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
               metadata.dynamic_navigation_code_validation_bytes ==
                 VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES &&
               metadata.hot_graph_dynamic_record_bytes >=
                 metadata.dynamic_navigation_code_offset +
                   metadata.dynamic_navigation_code_validation_bytes +
                   metadata.navigation_code_bytes,
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
    lib_assert(metadata.idmap_format == "owner_sharded_v2_bound",
               "index has no build-bound owner idmap v2; rebuild it");
    owner_idmap_required_ = true;
    lib_assert(metadata.centroid_state_format ==
                 "physical_shard_centroid_v2_bound",
               "index has no exact physical-shard centroid state; rebuild it");
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

  initialize_storage_centroid_route();

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
  if (!config.disable_thread_pinning) {
    pin_main_thread(core_assignment_.get_available_core());
  }
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
    .vector_id_namespace_size = vector_id_namespace_size_,
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
  const u64 persisted_shard_fingerprint = *reinterpret_cast<const u64*>(
    index_buffer_.get_full_buffer() +
      vamana::centroid_state::kShardFingerprintOffset);
  if (persisted_shard_fingerprint == 0 ||
      persisted_shard_fingerprint != gpu_shard_build_fingerprint_) {
    return {false, "index shard does not belong to the loaded metadata build"};
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
      header.vector_dtype != static_cast<u32>(VamanaNode::vector_dtype()) ||
      header.model_checksum != gpu_navigation_model_checksum_ ||
      header.build_fingerprint != gpu_index_build_fingerprint_ ||
      header.shard_fingerprint != gpu_shard_build_fingerprint_ ||
      header.shard_fingerprint != persisted_shard_fingerprint ||
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

  // CentroidRouter retains compensated FP64 sums for stable maintenance. The
  // published route is canonical FP32 because GPU queries consume FP32 and
  // update-home selection must make the identical routing decision.
  const auto centroid_scalar =
    gpu_search::format::CentroidScalarType::float32;
  const u64 centroid_publication_bytes =
    gpu_search::format::storage_centroid_route_publication_bytes(
      VamanaNode::DIM, centroid_scalar,
      gpu_search::format::kStorageCentroidRouteMaxLiveEntries);
  if (centroid_publication_bytes == 0 ||
      centroid_publication_bytes > index_buffer_.buffer_size) {
    return {false, "invalid variable-length centroid publication layout"};
  }
  const u64 centroid_publication_offset =
    (index_buffer_.buffer_size - centroid_publication_bytes) & ~u64{63};
  const u64 dynamic_stride =
    (VamanaNode::allocation_size() + alignof(u64) - 1) &
    ~u64{alignof(u64) - 1};
  if (centroid_publication_offset < gpu_dynamic_node_base_ ||
      dynamic_stride > centroid_publication_offset - gpu_dynamic_node_base_) {
    return {false,
            "storage memory has no dynamic-node headroom before the centroid publication"};
  }
  dynamic_allocation_limit_ = centroid_publication_offset;

  std::memset(index_buffer_.get_full_buffer() + gpu_storage_control_offset_, 0,
              gpu_search::format::kStorageControlBytes);
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const u32 dynamic_navigation_code_payload_bytes =
    VamanaNode::dynamic_navigation_code_payload_bytes();
  lib_assert(dynamic_navigation_code_payload_bytes ==
               gpu_navigation_code_bytes_,
             "dynamic navigation PQ payload width does not match the index model");
  *control = gpu_search::format::StorageControlBlock{
    .shard_id = storage_id_,
    .dynamic_record_bytes = static_cast<u32>(VamanaNode::allocation_size()),
    .dynamic_hot_offset = VamanaNode::HOT_GRAPH_DYNAMIC_HOT_OFFSET,
    .dynamic_code_offset = VamanaNode::HOT_GRAPH_DYNAMIC_CODE_OFFSET,
    .code_bytes = dynamic_navigation_code_payload_bytes,
    .dynamic_high_watermark = gpu_dynamic_node_base_,
    .centroid_route = {
      .remote_offset = centroid_publication_offset,
      .publication_bytes = centroid_publication_bytes,
      .dim = VamanaNode::DIM,
      .centroid_scalar_type = static_cast<u32>(centroid_scalar),
      .shard_count = num_storage_nodes_,
    },
  };
  str centroid_error;
  const bool initialized_centroid_publication =
    gpu_search::format::prepare_storage_centroid_route_publication(
      span<byte_t>{index_buffer_.get_full_buffer() +
                     centroid_publication_offset,
                   static_cast<size_t>(centroid_publication_bytes)},
      storage_id_, VamanaNode::DIM, centroid_scalar,
      gpu_search::format::kStorageCentroidRouteMaxLiveEntries,
      1, 0, nullptr,
      span<const gpu_search::format::StorageCentroidRouteEntry>{},
      &centroid_error);
  if (!initialized_centroid_publication) {
    return {false, centroid_error};
  }
  *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = gpu_dynamic_node_base_;

  return {true, ""};
}

void MemoryNode::initialize_storage_centroid_route() {
  const filepath_t path = index_path::centroid_state_file(
    index_prefix_, storage_id_ + 1, num_storage_nodes_);
  std::ifstream input(path, std::ios::binary);
  lib_assert(input.good(), "missing physical centroid sidecar: " +
                                path.string());

  vamana::centroid_state::Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  lib_assert(input.gcount() == static_cast<std::streamsize>(sizeof(header)) &&
               header.magic == vamana::centroid_state::kMagic &&
               header.version == vamana::centroid_state::kVersion &&
               header.header_bytes == sizeof(header) &&
               vamana::centroid_state::valid_header_checksum(header),
             "invalid physical centroid sidecar envelope: " +
               path.string());

  const u64 shard_fingerprint = *reinterpret_cast<const u64*>(
    index_buffer_.get_full_buffer() +
      vamana::centroid_state::kShardFingerprintOffset);
  const u64 expected_payload_bytes =
    vamana::centroid_state::payload_bytes(
      header.dim, header.entry_count);
  const u64 file_bytes = std::filesystem::file_size(path);
  lib_assert(header.build_fingerprint == gpu_index_build_fingerprint_ &&
               header.shard_fingerprint == gpu_shard_build_fingerprint_ &&
               header.shard_fingerprint == shard_fingerprint &&
               header.shard == storage_id_ &&
               header.shard_count == num_storage_nodes_ &&
               header.dim == VamanaNode::DIM &&
               header.max_degree == VamanaNode::R &&
               header.vector_count == gpu_static_node_count_ &&
               header.entry_count >= 1 &&
               header.entry_count <=
                 vamana::centroid_state::kMaxLiveEntries &&
               header.entry_count <= header.vector_count &&
               header.node_base_offset ==
                 gpu_search::format::kNodeBaseOffset &&
               header.vector_dtype ==
                 static_cast<u32>(VamanaNode::vector_dtype()) &&
               header.vector_component_size ==
                 VamanaNode::vector_component_size() &&
               header.metadata_schema_version ==
                 gpu_search::format::kMetadataSchemaVersion &&
               header.node_size == VamanaNode::total_size() &&
               header.vector_offset == VamanaNode::offset_vector() &&
               header.vector_bytes == VamanaNode::vector_bytes() &&
               header.slot_incarnation_offset ==
                 VamanaNode::offset_slot_incarnation() &&
               header.hot_graph_version == vamana::hot_graph::kVersion3 &&
               header.hot_graph_entry_size ==
                 VamanaNode::hot_graph_entry_size() &&
               header.hot_graph_pointer_bytes ==
                 vamana::hot_graph::kCompactPointerBytes &&
               header.hot_graph_shard_bits ==
                 VamanaNode::HOT_GRAPH_SHARD_BITS &&
               header.remote_ptr_format_version ==
                 vamana::centroid_state::kRemotePtrFormatVersion &&
               header.remote_ptr_alignment_log2 ==
                 RemotePtr::OFFSET_ALIGNMENT_LOG2 &&
               header.remote_ptr_offset_bits == RemotePtr::OFFSET_UNIT_BITS &&
               header.remote_ptr_shard_bits == RemotePtr::MEMORY_NODE_BITS &&
               header.remote_ptr_incarnation_bits ==
                 RemotePtr::INCARNATION_BITS &&
               header.static_incarnation == 0 &&
               header.payload_bytes == expected_payload_bytes &&
               header.payload_bytes <=
                 std::numeric_limits<size_t>::max() &&
               header.payload_bytes <= static_cast<u64>(
                 std::numeric_limits<std::streamsize>::max()) &&
               header.payload_bytes <=
                 std::numeric_limits<u64>::max() - sizeof(header) &&
               file_bytes == sizeof(header) + header.payload_bytes,
             "physical centroid sidecar does not match the loaded index "
             "build/layout: " + path.string());
  vec<byte_t> payload(static_cast<size_t>(header.payload_bytes));
  input.read(reinterpret_cast<char*>(payload.data()),
             static_cast<std::streamsize>(payload.size()));
  lib_assert(input.gcount() == static_cast<std::streamsize>(payload.size()) &&
               vamana::centroid_state::checksum(payload) ==
                 header.payload_checksum,
             "physical centroid sidecar checksum mismatch: " +
               path.string());

  const auto* sums = reinterpret_cast<const f64*>(payload.data());
  for (u32 dimension = 0; dimension < header.dim; ++dimension) {
    lib_assert(floating_value_is_finite(sums[dimension]),
               "centroid sidecar contains a non-finite sum");
  }
  const auto* stored_entries = reinterpret_cast<
    const vamana::centroid_state::Entry*>(
      payload.data() + static_cast<size_t>(header.dim) * sizeof(f64));
  vec<vamana::routing::CentroidRouter::LiveEntry> entries;
  entries.reserve(header.entry_count);
  std::unordered_set<u64> unique_entries;
  unique_entries.reserve(header.entry_count);
  vec<byte_t> decoded_graph(VamanaNode::neighbor_read_size());
  for (u32 index = 0; index < header.entry_count; ++index) {
    const RemotePtr pointer{stored_entries[index].remote_node};
    const bool valid_static_slot = !pointer.is_null() &&
      pointer.is_well_formed() && pointer.incarnation() == 0 &&
      pointer.memory_node() == storage_id_ &&
      pointer.byte_offset() >= header.node_base_offset &&
      (pointer.byte_offset() - header.node_base_offset) %
          header.node_size == 0 &&
      (pointer.byte_offset() - header.node_base_offset) /
          header.node_size < gpu_static_node_count_;
    lib_assert(valid_static_slot && stored_entries[index].generation == 0 &&
                 stored_entries[index].reserved == 0 &&
                 unique_entries.insert(pointer.raw_address).second &&
                 pointer.memory_node() == storage_id_ &&
                 local_live_vector(pointer) != nullptr,
               "centroid sidecar contains a stale graph entry");

    const byte_t* node = local_node_ptr(pointer);
    const u64 node_header = load_local_node_header_acquire(pointer);
    const u32 node_generation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_generation());
    const u64 disallowed_flags = VamanaNode::HEADER_NODE_LOCK |
      VamanaNode::HEADER_DELETED | VamanaNode::HEADER_PROVISIONAL |
      VamanaNode::HEADER_RETIRING;
    const u64 compact_offset = VamanaNode::hot_graph_entry_offset(pointer);
    lib_assert((node_header & disallowed_flags) == 0 &&
                 (node_header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0 &&
                 VamanaNode::header_incarnation(node_header) == 0 &&
                 node_generation == stored_entries[index].generation &&
                 VamanaNode::hot_graph_entry_available(pointer) &&
                 compact_offset <= mn_memory_bytes_ &&
                 VamanaNode::hot_graph_entry_size() <=
                   mn_memory_bytes_ - compact_offset &&
                 (index_buffer_.get_full_buffer()[compact_offset + 1] &
                    VamanaNode::HOT_GRAPH_DELETED) == 0 &&
                 VamanaNode::decode_hot_graph_entry(
                   index_buffer_.get_full_buffer() + compact_offset,
                   decoded_graph.data(), 0),
               "centroid sidecar entry has an invalid static graph record");
    entries.push_back({pointer, stored_entries[index].generation});
  }

  storage_centroid_router_ =
    std::make_unique<vamana::routing::CentroidRouter>(
      VamanaNode::DIM, num_storage_nodes_);
  storage_centroid_static_live_bitmap_.assign(
    static_cast<size_t>((gpu_static_node_count_ + 63) / 64), ~u64{0});
  if (!storage_centroid_static_live_bitmap_.empty() &&
      gpu_static_node_count_ % 64 != 0) {
    storage_centroid_static_live_bitmap_.back() =
      (u64{1} << (gpu_static_node_count_ % 64)) - 1;
  }
  storage_centroid_dynamic_live_bitmap_.clear();
  storage_centroid_static_cursor_ = 0;
  storage_centroid_dynamic_cursor_ = 0;
  lib_assert(storage_centroid_router_->restore_shard_state(
               storage_id_, header.vector_count,
               span<const f64>{sums, header.dim}, entries, 1),
             "failed to restore compensated physical centroid state");
  lib_assert(storage_centroid_router_->publish(),
             "failed to publish restored physical centroid state");
  publish_storage_centroid_route();
  print_status(
    "compensated physical centroid restored on shard " +
    std::to_string(storage_id_) + ": vectors=" +
    std::to_string(header.vector_count) + " entries=" +
    std::to_string(header.entry_count));
}

void MemoryNode::publish_storage_centroid_route() {
  if (storage_centroid_router_ == nullptr) return;
  const auto snapshot = storage_centroid_router_->snapshot();
  lib_assert(snapshot != nullptr && storage_id_ < snapshot->shards.size(),
             "physical centroid snapshot is unavailable");
  const auto& shard = snapshot->shards[storage_id_];

  thread_local vec<gpu_search::format::StorageCentroidRouteEntry> entries;
  entries.clear();
  entries.reserve(shard.live_entry_count);
  for (const auto& source : shard.entries()) {
    entries.push_back({
      .remote_node = source.pointer.raw_address,
      .generation = source.generation,
    });
  }

  const auto* control = reinterpret_cast<
    const gpu_search::format::StorageControlBlock*>(
      index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const auto descriptor = control->centroid_route;
  lib_assert(descriptor.centroid_scalar_type == static_cast<u32>(
               gpu_search::format::CentroidScalarType::float32) &&
               (shard.count == 0 || shard.centroid.size() == VamanaNode::DIM),
             "centroid route publication is not canonical FP32");
  thread_local vec<byte_t> next;
  next.resize(static_cast<size_t>(descriptor.publication_bytes));
  thread_local vec<f32> route_centroid;
  route_centroid.clear();
  if (shard.count != 0) {
    route_centroid.reserve(shard.centroid.size());
    for (f64 coordinate : shard.centroid) {
      route_centroid.push_back(static_cast<f32>(coordinate));
    }
  }
  thread_local str error;
  error.clear();
  lib_assert(gpu_search::format::prepare_storage_centroid_route_publication(
               next, storage_id_, VamanaNode::DIM,
               static_cast<gpu_search::format::CentroidScalarType>(
                 descriptor.centroid_scalar_type),
               descriptor.live_entry_capacity,
               std::max<u64>(1, shard.version), shard.count,
               shard.count == 0 ? nullptr : route_centroid.data(), entries,
               &error),
             error);

  std::lock_guard<std::mutex> lock(storage_centroid_publication_mutex_);
  auto* destination = index_buffer_.get_full_buffer() +
    descriptor.remote_offset;
  auto* destination_header = reinterpret_cast<
    gpu_search::format::StorageCentroidRoutePublicationHeader*>(destination);
  std::atomic_ref<u64> sequence(destination_header->sequence);
  const u64 current = sequence.load(std::memory_order_relaxed);
  const u64 odd = (current & ~u64{1}) + 1;
  const u64 even = odd + 1;
  sequence.store(odd, std::memory_order_release);
  std::memcpy(
    destination + sizeof(u64), next.data() + sizeof(u64),
    next.size() - sizeof(u64));
  std::atomic_thread_fence(std::memory_order_release);
  sequence.store(even, std::memory_order_release);
}

vec<RemotePtr> MemoryNode::local_centroid_route_entries() const {
  vec<RemotePtr> entries;
  if (storage_centroid_router_ == nullptr) return entries;
  const auto snapshot = storage_centroid_router_->snapshot();
  if (snapshot == nullptr || storage_id_ >= snapshot->shards.size()) {
    return entries;
  }
  const auto& shard = snapshot->shards[storage_id_];
  entries.reserve(shard.live_entry_count);
  for (const auto& entry : shard.entries()) {
    if (local_live_vector(entry.pointer) != nullptr) {
      entries.push_back(entry.pointer);
    }
  }
  return entries;
}

vec<vamana::routing::CentroidRouter::LiveEntry>
MemoryNode::select_local_centroid_live_entries(
    span<const RemotePtr> preferred) {
  using LiveEntry = vamana::routing::CentroidRouter::LiveEntry;
  thread_local vec<LiveEntry> candidates;
  candidates.clear();
  candidates.reserve(
    preferred.size() + vamana::routing::CentroidRouter::kMaxLiveEntries);
  thread_local hashset_t<RemotePtr> seen;
  seen.clear();
  seen.reserve(candidates.capacity());
  const auto append = [&](RemotePtr pointer) {
    if (pointer.is_null() || pointer.memory_node() != storage_id_ ||
        !valid_local_storage_node_pointer(pointer)) {
      return false;
    }
    if (seen.contains(pointer)) return true;
    const u64 header = load_local_node_header_acquire(pointer);
    if ((header & (VamanaNode::HEADER_DELETED |
                   VamanaNode::HEADER_PROVISIONAL |
                   VamanaNode::HEADER_RETIRING)) != 0 ||
        VamanaNode::header_incarnation(header) != pointer.incarnation() ||
        (header & VamanaNode::HEADER_CENTROID_ACCOUNTED) == 0) {
      return false;
    }
    const byte_t* node = local_node_ptr(pointer);
    if (*reinterpret_cast<const u32*>(
          node + VamanaNode::offset_slot_incarnation()) !=
        pointer.incarnation()) {
      return false;
    }
    seen.insert(pointer);
    candidates.push_back(LiveEntry{
      pointer,
      *reinterpret_cast<const u32*>(
        node + VamanaNode::offset_generation()),
    });
    return true;
  };

  for (const RemotePtr pointer : preferred) append(pointer);
  if (storage_centroid_router_ != nullptr) {
    const auto snapshot = storage_centroid_router_->snapshot();
    if (snapshot != nullptr && storage_id_ < snapshot->shards.size()) {
      for (const LiveEntry& entry :
           snapshot->shards[storage_id_].entries()) {
        append(entry.pointer);
      }
    }
  }

  const u64 authoritative_count =
    storage_centroid_router_->authoritative_count(storage_id_);
  const size_t desired_entries = static_cast<size_t>(std::min<u64>(
    authoritative_count, vamana::routing::CentroidRouter::kMaxLiveEntries));
  if (desired_entries == 0) return {};

  const auto sample_bitmap = [&](vec<u64>& bitmap,
                                 u64 valid_bits,
                                 u64 base,
                                 u64 stride,
                                 bool dynamic,
                                 u64& cursor,
                                 size_t sample_budget) {
    if (bitmap.empty() || valid_bits == 0 || sample_budget == 0) {
      return size_t{0};
    }
    const size_t before = candidates.size();
    // Up to four existing roots can be encountered first. Probe past those
    // duplicates, but retain a fixed per-plane word and set-bit budget.
    const size_t candidate_probe_budget = sample_budget +
      vamana::routing::CentroidRouter::kMaxLiveEntries;
    thread_local vec<u64> ordinals;
    vamana::routing::bounded_rotating_live_samples_into(
      span<const u64>{bitmap}, valid_bits, cursor,
      candidate_probe_budget, ordinals);
    for (const u64 ordinal : ordinals) {
      if (candidates.size() - before >= sample_budget) break;
      const u64 mask = u64{1} << (ordinal % 64);
      const u64 offset = base + ordinal * stride;
      const u32 incarnation = dynamic
        ? *reinterpret_cast<const u32*>(
            index_buffer_.get_full_buffer() + offset +
            VamanaNode::offset_slot_incarnation())
        : 0;
      if (!append(RemotePtr{storage_id_, offset, incarnation})) {
        // Membership metadata is authoritative. Repair a stale bit lazily;
        // the fixed probe budget still bounds this batch under sparse reuse.
        bitmap[ordinal / 64] &= ~mask;
      }
    }
    return candidates.size() - before;
  };

  const u64 dynamic_bits = static_cast<u64>(
    storage_centroid_dynamic_live_bitmap_.size()) * 64;
  const u64 dynamic_stride =
    (VamanaNode::allocation_size() + alignof(u64) - 1) &
    ~u64{alignof(u64) - 1};
  // Explore four additional live identities on every membership batch even
  // when all four published roots remain valid. Split the first probes across
  // immutable and dynamic planes so neither population can starve, then spend
  // any unused quota on whichever plane can supply it. All scans are bounded
  // by constants and advance independent cursors through sparse regions.
  constexpr size_t kInitialPlaneQuota =
    vamana::routing::kLiveSeedExplorationSamples / 2;
  size_t sampled = sample_bitmap(
    storage_centroid_static_live_bitmap_, gpu_static_node_count_,
    gpu_search::format::kNodeBaseOffset, VamanaNode::total_size(), false,
    storage_centroid_static_cursor_, kInitialPlaneQuota);
  sampled += sample_bitmap(
    storage_centroid_dynamic_live_bitmap_, dynamic_bits,
    gpu_dynamic_node_base_, dynamic_stride, true,
    storage_centroid_dynamic_cursor_, kInitialPlaneQuota);
  size_t remaining = vamana::routing::kLiveSeedExplorationSamples - sampled;
  if (remaining != 0) {
    const size_t extra_static = sample_bitmap(
      storage_centroid_static_live_bitmap_, gpu_static_node_count_,
      gpu_search::format::kNodeBaseOffset, VamanaNode::total_size(), false,
      storage_centroid_static_cursor_, remaining);
    remaining -= extra_static;
  }
  if (remaining != 0) {
    (void)sample_bitmap(
      storage_centroid_dynamic_live_bitmap_, dynamic_bits,
      gpu_dynamic_node_base_, dynamic_stride, true,
      storage_centroid_dynamic_cursor_, remaining);
  }

  thread_local vec<f64> centroid;
  centroid.resize(VamanaNode::DIM);
  const bool have_centroid =
    storage_centroid_router_->copy_authoritative_centroid(
      storage_id_, span<f64>{centroid});
  lib_assert(have_centroid &&
               candidates.size() >= desired_entries,
             "non-empty centroid membership lost every live route seed");
  struct RankedEntry {
    LiveEntry entry;
    vamana::routing::CentroidSeedRank rank;
  };
  thread_local vec<RankedEntry> ranked;
  ranked.clear();
  ranked.reserve(candidates.size());
  thread_local vec<element_t> decoded;
  decoded.resize(VamanaNode::DIM);
  for (const LiveEntry& candidate : candidates) {
    const byte_t* node = local_node_ptr(candidate.pointer);
    decode_storage_vector_to_float(
      node + VamanaNode::offset_vector(), VamanaNode::vector_dtype(),
      VamanaNode::DIM, decoded.data());
    ranked.push_back({
      .entry = candidate,
      .rank = {
        .squared_l2 = vamana::routing::centroid_seed_squared_l2(
          span<const f32>{decoded}, span<const f64>{centroid}),
        .pointer_raw = candidate.pointer.raw_address,
      },
    });
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const RankedEntry& lhs, const RankedEntry& rhs) {
              return vamana::routing::centroid_seed_rank_less(
                lhs.rank, rhs.rank);
            });
  vec<LiveEntry> entries;
  entries.reserve(desired_entries);
  for (size_t index = 0; index < desired_entries; ++index) {
    entries.push_back(ranked[index].entry);
  }
  return entries;
}

bool MemoryNode::apply_local_centroid_membership_ops(
    span<const service::storage_owner::CentroidMembershipOp> ops) {
  using Kind = service::storage_owner::CentroidMembershipKind;
  if (ops.empty()) return true;
  std::lock_guard<std::mutex> update_lock(
    storage_centroid_update_mutex_);
  thread_local vec<RemotePtr> preferred;
  preferred.clear();
  preferred.reserve(ops.size());
  bool changed = false;
  bool success = true;
  thread_local vec<element_t> decoded;
  decoded.resize(VamanaNode::DIM);
  const auto valid_centroid_slot_address = [&](RemotePtr pointer) {
    if (pointer.is_null() || !pointer.is_well_formed() ||
        pointer.memory_node() != storage_id_ ||
        !VamanaNode::hot_graph_entry_available(pointer)) {
      return false;
    }
    const auto header_address =
      vamana::StorageLayoutResolver::header(pointer);
    if (header_address.offset > mn_memory_bytes_ ||
        sizeof(u64) > mn_memory_bytes_ - header_address.offset) {
      return false;
    }
    const u64 offset = pointer.byte_offset();
    if (offset < gpu_dynamic_node_base_) {
      return pointer.incarnation() == 0 &&
        offset >= gpu_search::format::kNodeBaseOffset &&
        (offset - gpu_search::format::kNodeBaseOffset) %
            VamanaNode::total_size() == 0 &&
        (offset - gpu_search::format::kNodeBaseOffset) /
            VamanaNode::total_size() < gpu_static_node_count_;
    }
    if (pointer.incarnation() == 0) return false;
    const u64 stride =
      (VamanaNode::allocation_size() + alignof(u64) - 1) &
      ~u64{alignof(u64) - 1};
    if ((offset - gpu_dynamic_node_base_) % stride != 0) return false;
    const auto* control = reinterpret_cast<const
      gpu_search::format::StorageControlBlock*>(
        index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
    const u64 high_watermark = std::atomic_ref<const u64>(
      control->dynamic_high_watermark).load(std::memory_order_acquire);
    return offset <= high_watermark &&
      VamanaNode::allocation_size() <= high_watermark - offset;
  };
  const auto update_live_bitmap = [&](RemotePtr pointer, bool live) {
    const u64 offset = pointer.byte_offset();
    u64 ordinal = 0;
    vec<u64>* bitmap = nullptr;
    if (offset >= gpu_dynamic_node_base_) {
      const u64 stride =
        (VamanaNode::allocation_size() + alignof(u64) - 1) &
        ~u64{alignof(u64) - 1};
      lib_assert((offset - gpu_dynamic_node_base_) % stride == 0,
                 "centroid membership references a misaligned dynamic node");
      ordinal = (offset - gpu_dynamic_node_base_) / stride;
      bitmap = &storage_centroid_dynamic_live_bitmap_;
      const size_t required_words = static_cast<size_t>(ordinal / 64 + 1);
      if (bitmap->size() < required_words) bitmap->resize(required_words, 0);
    } else {
      lib_assert(offset >= gpu_search::format::kNodeBaseOffset &&
                   (offset - gpu_search::format::kNodeBaseOffset) %
                     VamanaNode::total_size() == 0,
                 "centroid membership references a misaligned static node");
      ordinal = (offset - gpu_search::format::kNodeBaseOffset) /
        VamanaNode::total_size();
      lib_assert(ordinal < gpu_static_node_count_,
                 "centroid membership static ordinal exceeds the shard");
      bitmap = &storage_centroid_static_live_bitmap_;
    }
    const u64 mask = u64{1} << (ordinal % 64);
    if (live) {
      (*bitmap)[ordinal / 64] |= mask;
    } else {
      (*bitmap)[ordinal / 64] &= ~mask;
    }
  };

  // Reject malformed envelopes before touching any node. Identity races are
  // handled independently below: valid operations in the same wire batch are
  // still published, while the caller retries the stale item idempotently.
  for (const auto& op : ops) {
    const RemotePtr pointer{op.node_raw};
    const auto kind = static_cast<Kind>(op.kind);
    if (op.maintenance_sequence == 0 || op.reserved != 0 ||
        (kind != Kind::add && kind != Kind::remove) ||
        !valid_centroid_slot_address(pointer)) {
      return false;
    }
  }

  for (const auto& op : ops) {
    const RemotePtr pointer{op.node_raw};
    const auto kind = static_cast<Kind>(op.kind);

    const auto pointer_lock = try_lock_node(pointer);
    if (pointer_lock != memory_node_storage_owner_index_detail::
                          IncarnationLockResult::locked) {
      // A structurally valid remove whose tagged incarnation is already gone
      // has reached its postcondition.  It must not retry forever and, most
      // importantly, must never debit the slot's new occupant.
      bool old_incarnation_absent = false;
      if (kind == Kind::remove &&
          pointer_lock == memory_node_storage_owner_index_detail::
                            IncarnationLockResult::stale) {
        const u64 before = load_local_node_header_acquire(pointer);
        const byte_t* observed = local_node_ptr(pointer);
        const node_t observed_id = *reinterpret_cast<const node_t*>(
          observed + VamanaNode::offset_id());
        const u32 observed_generation = *reinterpret_cast<const u32*>(
          observed + VamanaNode::offset_generation());
        const u32 observed_slot_incarnation =
          *reinterpret_cast<const u32*>(
            observed + VamanaNode::offset_slot_incarnation());
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 after = load_local_node_header_acquire(pointer);
        if (before == after &&
            (after & VamanaNode::HEADER_NODE_LOCK) == 0) {
          old_incarnation_absent =
            memory_node_storage_owner_maintenance_detail::
              classify_centroid_remove_identity(
                pointer.incarnation(),
                VamanaNode::header_incarnation(after),
                observed_slot_incarnation,
                observed_id == op.id &&
                  observed_generation == op.generation) ==
              memory_node_storage_owner_maintenance_detail::
                CentroidRemoveIdentityDecision::already_absent;
        }
      }
      if (!old_incarnation_absent) success = false;
      continue;
    }
    auto* node = local_node_ptr(pointer);
    auto* header_storage = reinterpret_cast<u64*>(node);
    std::atomic_ref<u64> header_ref(*header_storage);
    const u64 header = header_ref.load(std::memory_order_acquire);
    const node_t id = *reinterpret_cast<const node_t*>(
      node + VamanaNode::offset_id());
    const u32 generation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_generation());
    const u32 slot_incarnation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_slot_incarnation());
    const bool accounted =
      (header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0;
    const bool id_and_generation_match =
      id == op.id && generation == op.generation;
    const bool identity_matches = id_and_generation_match &&
      VamanaNode::header_incarnation(header) == pointer.incarnation() &&
      slot_incarnation == pointer.incarnation();
    const bool addable =
      (header & (VamanaNode::HEADER_DELETED |
                 VamanaNode::HEADER_PROVISIONAL |
                 VamanaNode::HEADER_RETIRING)) == 0;
    if (kind == Kind::remove) {
      const auto decision =
        memory_node_storage_owner_maintenance_detail::
          classify_centroid_remove_identity(
            pointer.incarnation(), VamanaNode::header_incarnation(header),
            slot_incarnation, id_and_generation_match);
      if (decision == memory_node_storage_owner_maintenance_detail::
                        CentroidRemoveIdentityDecision::already_absent) {
        unlock_node(pointer);
        continue;
      }
      if (decision != memory_node_storage_owner_maintenance_detail::
                        CentroidRemoveIdentityDecision::apply_exact) {
        unlock_node(pointer);
        success = false;
        continue;
      }
    } else if (!identity_matches || !addable) {
      unlock_node(pointer);
      success = false;
      continue;
    }
    if ((kind == Kind::add && accounted) ||
        (kind == Kind::remove && !accounted)) {
      unlock_node(pointer);
      continue;
    }

    decode_storage_vector_to_float(
      node + VamanaNode::offset_vector(), VamanaNode::vector_dtype(),
      VamanaNode::DIM, decoded.data());
    const bool applied = kind == Kind::add
      ? storage_centroid_router_->insert(
          storage_id_, span<const element_t>{decoded})
      : storage_centroid_router_->erase(
          storage_id_, span<const element_t>{decoded});
    if (!applied) {
      unlock_node(pointer);
      success = false;
      continue;
    }
    if (kind == Kind::add) {
      header_ref.fetch_or(
        static_cast<u64>(VamanaNode::HEADER_CENTROID_ACCOUNTED),
        std::memory_order_release);
      update_live_bitmap(pointer, true);
      preferred.push_back(pointer);
    } else {
      header_ref.fetch_and(
        ~static_cast<u64>(VamanaNode::HEADER_CENTROID_ACCOUNTED),
        std::memory_order_release);
      update_live_bitmap(pointer, false);
    }
    unlock_node(pointer);
    changed = true;
  }

  if (!changed) return success;
  const auto entries = select_local_centroid_live_entries(
    span<const RemotePtr>{preferred});
  // Empty membership is legal only when the authoritative count became zero;
  // erase() canonicalizes that state and clears its entries itself.
  if (!entries.empty()) {
    (void)storage_centroid_router_->replace_live_entries(
      storage_id_, entries);
  }
  const u64 authoritative_count =
    storage_centroid_router_->authoritative_count(storage_id_);
  lib_assert(!entries.empty() || authoritative_count == 0,
             "non-empty shard lost every dynamic centroid route entry");
  lib_assert(storage_centroid_router_->publish(),
             "centroid membership mutation was not publishable");
  publish_storage_centroid_route();
  return success;
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
  if ((header & (VamanaNode::HEADER_NODE_LOCK |
                 VamanaNode::HEADER_DELETED)) != 0 ||
      VamanaNode::header_incarnation(header) != rptr.incarnation()) {
    return nullptr;
  }
  const byte_t* node = index_buffer_.get_full_buffer() + rptr.byte_offset();
  if (*reinterpret_cast<const u32*>(
        node + VamanaNode::offset_slot_incarnation()) !=
      rptr.incarnation()) {
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
