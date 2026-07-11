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

  if (!config.disable_thread_pinning) {
    const u32 core = core_assignment_.get_available_core();
    pin_main_thread(core);
    print_status("pinned main thread to core " + std::to_string(core));
  }

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
  str startup_anchor_format;
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
    gpu_stream_layout_ = metadata.schema_version == 14 &&
      metadata.node_layout == "plain" &&
      metadata.storage_format == "vamana_compact_v1" &&
      metadata.navigation_quantizer == "opq_pq16" &&
      metadata.navigation_format == "opq_pq16_graph_v1";
    lib_assert(gpu_stream_layout_,
               "storage node requires a schema-14 compact OPQ/PQ16 index");
    lib_assert(storage_id_ < num_storage_nodes_, "invalid GPU storage shard id");
    lib_assert(metadata.hot_graph_entry_counts.size() == num_storage_nodes_,
               "GPU storage metadata has invalid static shard counts");
    lib_assert(metadata.hot_graph_dynamic_base_offsets.size() == num_storage_nodes_,
               "GPU storage metadata has invalid dynamic shard offsets");
    gpu_static_node_count_ = metadata.hot_graph_entry_counts[storage_id_];
    gpu_static_dynamic_base_ = metadata.hot_graph_dynamic_base_offsets[storage_id_];
    gpu_navigation_code_bytes_ = metadata.navigation_code_bytes;
    gpu_navigation_model_checksum_ = metadata.navigation_model_checksum;
    if (config.vector_data_type != "auto" && config.resolved_vector_dtype() != metadata.vector_dtype) {
      lib_failure("configured vector-data-type=" + config.vector_data_type +
                  " does not match index metadata vector_data_type=" + vector_dtype_name(metadata.vector_dtype));
    }
    startup_dtype = metadata.vector_dtype;
    config.vector_data_type = vector_dtype_name(startup_dtype);
    VamanaNode::disable_hot_graph();
    VamanaNode::init_static_storage(config.dim, config.R, startup_dtype);
    lib_assert(metadata.schema_version == 14 && metadata.node_layout == "plain",
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
               metadata.hot_graph_dynamic_record_bytes >=
                 metadata.hot_graph_dynamic_hot_offset + metadata.hot_graph_entry_size &&
               metadata.hot_graph_dynamic_hot_offset >= VamanaNode::total_size(),
               "index dynamic hot graph metadata mismatch on storage node");
    VamanaNode::configure_hot_graph(metadata.hot_graph_offsets,
                                    metadata.hot_graph_entry_counts,
                                    metadata.hot_graph_entry_size,
                                    metadata.hot_graph_shard_bits,
                                    metadata.hot_graph_dynamic_base_offsets,
                                    metadata.hot_graph_dynamic_record_bytes,
                                    metadata.hot_graph_dynamic_hot_offset);
    lib_assert(VamanaNode::HAS_HOT_GRAPH, "failed to enable compact hot graph on storage node");
    owner_idmap_required_ = metadata.idmap_format == "owner_sharded_v1";
    startup_anchor_format = metadata.anchor_format;
    print_status("loaded index metadata from " + index_prefix.string() +
                 " (layout=" + VamanaNode::layout_name() +
                 ", vector_data_type=" + VamanaNode::vector_dtype_name() + ")");
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
    storage_owner_anchor_index_ = std::make_unique<vamana::anchor::Index>();
    str anchor_error;
    lib_assert(startup_anchor_format == "owner_anchor_v1" &&
                 storage_owner_anchor_index_->load(index_prefix_, config.dim,
                                                   num_storage_nodes_, &anchor_error),
               "storage-owner anchor sidecar unavailable on storage node: " + anchor_error);
    print_status("storage-owner anchors loaded on shard " + std::to_string(storage_id_) +
                 ": entries=" +
                 std::to_string(storage_owner_anchor_index_->anchor_count()) +
                 " memory=" +
                 std::to_string(storage_owner_anchor_index_->memory_bytes()) + " bytes");
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
  storage_insert_tasks_cv_.notify_all();
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
  out.storage_owner_anchor_hints = scale_ns(counters.storage_owner_anchor_hints, part, total);
  out.storage_owner_anchor_valid_hints = scale_ns(counters.storage_owner_anchor_valid_hints, part, total);
  out.storage_owner_anchor_expansions = scale_ns(counters.storage_owner_anchor_expansions, part, total);
  out.storage_owner_anchor_remote_expansions =
    scale_ns(counters.storage_owner_anchor_remote_expansions, part, total);
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

  const u64 remote_offset = gpu_search::format::align_up(gpu_static_dynamic_base_, 64);
  const u64 payload_bytes = gpu_static_node_count_ * gpu_navigation_code_bytes_;
  if (remote_offset == 0 || remote_offset > index_buffer_.buffer_size ||
      payload_bytes > index_buffer_.buffer_size - remote_offset) {
    return {false, "buffer too small for GPU PQ stream"};
  }

  const filepath_t code_path = index_path::navigation_code_for_shard(path);
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

  const u64 allocation_stride = static_cast<u64>(VamanaNode::allocation_size());
  const u64 region_end = remote_offset + payload_bytes;
  *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = gpu_static_dynamic_base_ +
    gpu_search::format::align_up(region_end - gpu_static_dynamic_base_, allocation_stride);

  return {true, ""};
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
