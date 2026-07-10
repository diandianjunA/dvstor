#include "memory_node/memory_node.hh"

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
      use_storage_owner_insert_(config.use_storage_owner_insert()),
      storage_owner_peer_rdma_tokens_(std::max<u32>(1, config.storage_owner_peer_rdma_tokens)),
      ip_distance_(config.ip_distance),
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
  qp_pool_size_ = p.qp_pool_size;
  const u32 gpu_rdma_qps = p.gpu_persistent ? p.gpu_rdma_qps : 0;
  const filepath_t index_prefix = config.resolved_index_prefix();
  index_prefix_ = index_prefix;
  VectorDType startup_dtype = config.resolved_vector_dtype();
  str startup_anchor_format;
  const filepath_t meta_file = filepath_t(index_prefix.string() + ".meta.json");
  if (!index_prefix.empty() && std::filesystem::exists(meta_file)) {
    service::index_metadata::Metadata metadata;
    str metadata_error;
    lib_assert(service::index_metadata::load_metadata(index_prefix, metadata, &metadata_error), metadata_error);
    lib_assert(metadata.dim == config.dim, "index metadata dim mismatch on storage node");
    lib_assert(metadata.R == config.R, "index metadata R mismatch on storage node");
    lib_assert(metadata.num_memory_nodes == num_storage_nodes_, "index metadata storage-node count mismatch");
    if (config.vector_data_type != "auto" && config.resolved_vector_dtype() != metadata.vector_dtype) {
      lib_failure("configured vector-data-type=" + config.vector_data_type +
                  " does not match index metadata vector_data_type=" + vector_dtype_name(metadata.vector_dtype));
    }
    startup_dtype = metadata.vector_dtype;
    config.vector_data_type = vector_dtype_name(startup_dtype);
    VamanaNode::disable_rabitq();
    VamanaNode::disable_hot_graph();
    const auto storage_format = vamana::parse_storage_format(metadata.storage_format);
    lib_assert(storage_format.has_value(),
               "index storage format is obsolete; rebuild with the current offline builder");
    VamanaNode::set_storage_format(*storage_format);
    VamanaNode::init_static_storage(config.dim, config.R, startup_dtype);
    lib_assert(metadata.schema_version == 13,
               "index storage format is obsolete; rebuild with the current offline builder");
    if (metadata.node_layout == "rabitq") {
        lib_assert(metadata.rabitq_centroid.size() == metadata.dim,
                   "RaBitQ index metadata has a missing or invalid centroid");
        VamanaNode::enable_rabitq();
        VamanaNode::set_rabitq_centroid(metadata.rabitq_centroid);
        lib_assert(metadata.rabitq_code_bits == VamanaNode::rabitq_code_bits() &&
                   metadata.rabitq_entry_size == VamanaNode::rabitq_entry_size(),
                   "RaBitQ index code layout does not match the runtime dimension");
    }
    lib_assert(metadata.vector_component_size == VamanaNode::vector_component_size(),
               "index metadata vector component size mismatch on storage node");
    lib_assert(metadata.vector_bytes == VamanaNode::vector_bytes(),
               "index metadata vector byte size mismatch on storage node");
    lib_assert(metadata.node_size == VamanaNode::total_size(), "index metadata node size mismatch on storage node");
    lib_assert(metadata.graph_hot_bytes == VamanaNode::graph_hot_bytes() &&
               metadata.vector_offset == VamanaNode::offset_vector() &&
               metadata.neighbors_offset == VamanaNode::offset_neighbors() &&
               metadata.rabitq_offset == (VamanaNode::HAS_RABITQ_CODE ? VamanaNode::offset_rabitq_code() : 0),
               "index metadata storage offsets mismatch on storage node");
    if (*storage_format == vamana::StorageFormat::compact_v1) {
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
                                      2u,
                                      metadata.hot_graph_dynamic_base_offsets,
                                      metadata.hot_graph_dynamic_record_bytes,
                                      metadata.hot_graph_dynamic_hot_offset);
      lib_assert(VamanaNode::HAS_HOT_GRAPH, "failed to enable compact hot graph on storage node");
    }
    owner_idmap_required_ = metadata.idmap_format == "owner_sharded_v1";
    startup_anchor_format = metadata.anchor_format;
    print_status("loaded index metadata from " + index_prefix.string() +
                 " (layout=" + VamanaNode::layout_name() +
                 ", vector_data_type=" + VamanaNode::vector_dtype_name() + ")");
  } else {
    VamanaNode::disable_rabitq();
    VamanaNode::disable_hot_graph();
    VamanaNode::set_storage_format(vamana::StorageFormat::aos_v1);
    VamanaNode::init_static_storage(config.dim, config.R, startup_dtype);
    if (config.use_rabitq) VamanaNode::enable_rabitq();
  }
  allocate_memory();

  // free-ptr is initialized to 16 (points to first free address in the buffer)
  *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = 16;

  if (!config.server_index_file.empty()) {
    const auto [success, message] = load_index_file(config.server_index_file.string());
    lib_assert(success, message);
    if (owner_idmap_required_) {
      lib_assert(load_owner_idmap(index_prefix_), "failed to load owner-sharded idmap");
    }
  }

  if (use_storage_owner_insert_ &&
      config.storage_owner_update_mode == "local_stitch") {
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
  const u32 worker_qps_per_node =
    (p.gpu_persistent ? 0 : std::min<u32>(num_compute_threads_, MAX_QPS)) * qp_pool_size_;
  const u32 qps_per_node = worker_qps_per_node + gpu_rdma_qps;
  if (gpu_rdma_qps > 0) {
    print_status("reserving " + std::to_string(gpu_rdma_qps) +
                 " GPU data-path QPs per compute node");
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

  // handle startup command (load/store/noop from CN init)
  print_status("waiting for commands from compute node...");
  bool running = handle_command();

  if (running && use_storage_owner_insert_) {
    setup_storage_peers(config);
    setup_insert_runtime(config);
    storage_worker_config_ = std::make_unique<Configuration>(config);
    start_peer_reverse_update_runtime(config);
    start_storage_owner_maintenance_runtime(config);
    start_storage_owner_insert_workers(config);
    service_storage_runtime(config);

  } else {
    // service mode: listen for runtime commands
    while (running) {
      running = handle_command();
    }
  }

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

bool MemoryNode::handle_command() {
  mn_command::Request req{};
  LocalMemoryRegion region{context_, &req, sizeof(mn_command::Request)};
  cm_.initiator_qp->post_receive(region);
  context_.receive();

  // receive path if present
  str path;
  if (req.path_length > 0) {
    path.resize(req.path_length);
    LocalMemoryRegion path_region{context_, path.data(), req.path_length};
    cm_.initiator_qp->post_receive(path_region);
    context_.receive();
  }

  const auto send_response = [&](bool success, const str& message = "") {
    mn_command::Response resp{success, message.size()};
    cm_.initiator_qp->post_send_inlined(&resp, sizeof(mn_command::Response), IBV_WR_SEND);
    context_.poll_send_cq_until_completion();
    if (!message.empty()) {
      cm_.initiator_qp->post_send_inlined(message.data(), message.size(), IBV_WR_SEND);
      context_.poll_send_cq_until_completion();
    }
  };

  switch (req.cmd) {
    case mn_command::NOOP:
      send_response(true);
      return true;

    case mn_command::LOAD: {
      const auto [success, message] = load_index_file(path);
      send_response(success, message);
      return true;
    }

    case mn_command::STORE: {
      const auto [success, message] = store_index_file(path);
      send_response(success, message);
      return true;
    }

    case mn_command::SHUTDOWN:
      print_status("received SHUTDOWN command");
      send_response(true);
      return false;

    default:
      send_response(false, "unknown command");
      return true;
  }
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

  const filepath_t pages_path = index_path::gpu_graph_pages_for_shard(path);
  if (std::filesystem::exists(pages_path)) {
    std::ifstream pages{pages_path, std::ios::binary};
    gpu_search::format::ShardPageFileHeader header;
    pages.read(reinterpret_cast<char*>(&header), sizeof(header));
    const u64 expected_checksum = gpu_search::format::checksum64(
      reinterpret_cast<const byte_t*>(&header),
      offsetof(gpu_search::format::ShardPageFileHeader, checksum));
    if (!pages.good() || header.magic != gpu_search::format::kShardPagesMagic ||
        header.version != gpu_search::format::kVersion ||
        header.memory_node != storage_id_ || header.checksum != expected_checksum ||
        header.remote_offset != gpu_search::format::align_up(file_size, header.page_bytes) ||
        header.data_bytes % header.page_bytes != 0) {
      return {false, "invalid GPU graph-page sidecar " + pages_path.string()};
    }
    if (header.remote_offset > index_buffer_.buffer_size ||
        header.data_bytes > index_buffer_.buffer_size - header.remote_offset) {
      return {false, "buffer too small for GPU graph-page sidecar"};
    }
    if (file_size < header.remote_offset + header.data_bytes) {
      pages.read(reinterpret_cast<char*>(index_buffer_.get_full_buffer() + header.remote_offset),
                 static_cast<std::streamsize>(header.data_bytes));
      if (!pages.good()) {
        return {false, "read failed for " + pages_path.string()};
      }
    }
    const u64 allocation_base = *reinterpret_cast<u64*>(index_buffer_.get_full_buffer());
    const u64 region_end = header.remote_offset + header.data_bytes;
    const u64 allocation_stride = VamanaNode::HAS_HOT_GRAPH
      ? static_cast<u64>(VamanaNode::allocation_size()) : 8ULL;
    *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = allocation_base +
      gpu_search::format::align_up(region_end - allocation_base, allocation_stride);
    print_status("loaded GPU graph pages (" + std::to_string(header.data_bytes) +
                 " Bytes) at remote offset " + std::to_string(header.remote_offset));
  }

  return {true, ""};
}

std::pair<bool, str> MemoryNode::store_index_file(const str& path) {
  const size_t index_size = *reinterpret_cast<u64*>(index_buffer_.get_full_buffer());
  print_status("storing index (" + std::to_string(index_size) + " Bytes) to " + path);

  create_directory(filepath_t{path}.parent_path());
  std::ofstream output_s{path, std::ios::out | std::ios::binary};

  auto t_store = timing_.create_enroll("store_index_buffer");
  t_store->start();
  if (!output_s.write(reinterpret_cast<char*>(index_buffer_.get_full_buffer()), index_size)) {
    t_store->stop();
    return {false, "write failed for " + path};
  }
  t_store->stop();
  output_s.close();

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
  return typed_distance_float_query(query, stored, VamanaNode::vector_dtype(), config.dim, ip_distance_);
}

distance_t MemoryNode::distance_between_vectors(const byte_t* lhs,
                                                VectorDType lhs_dtype,
                                                const byte_t* rhs,
                                                VectorDType rhs_dtype,
                                                const Configuration& config) const {
  return ip_distance_ ? typed_ip_distance(lhs, lhs_dtype, rhs, rhs_dtype, config.dim)
                      : typed_l2_distance(lhs, lhs_dtype, rhs, rhs_dtype, config.dim);
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

void MemoryNode::route_queries(i32 max_cqes) {
  print_status("route queries");
  size_t num_routings = 0;

  // receive routing message size
  size_t message_size;
  {
    LocalMemoryRegion region{context_, &message_size, sizeof(message_size)};
    cm_.initiator_qp->post_receive(region);
    context_.receive();

    std::cerr << "routing message size: " << message_size << " B\n";
  }

  const size_t buffer_entries = num_clients_ * query_router::LIMIT_PER_CN * (num_clients_ - 1) * 2;

  HugePage<byte_t> routing_buffer(buffer_entries * message_size);
  routing_buffer.touch_memory();

  LocalMemoryRegion lmr{
    context_, routing_buffer.get_full_buffer(), routing_buffer.buffer_size};  // register memory region

  vec<idx_t> freelist;  // offsets
  freelist.reserve(buffer_entries);

  for (idx_t i = 0; i < buffer_entries; ++i) {
    freelist.push_back(i * message_size);
  }

  constexpr u32 termination_signal_mn = static_cast<u32>(-1);
  constexpr u32 termination_signal_cn = static_cast<u32>(-2);
  u32 received_termination_signals = 0;

  vec<ibv_wc> recv_wcs(max_cqes);
  vec<ibv_wc> send_wcs(max_cqes);

  i32 posted_sends = 0;
  i32 posted_recvs = 0;

  cm_.synchronize();  // synchronize with CNs

  const auto post_receive = [&](u32 client) {
    lib_assert(!freelist.empty(), "empty freelist");
    const idx_t offset = freelist.back();
    freelist.pop_back();

    lib_assert(posted_recvs < max_cqes, "?-?-?(3)");

    const u64 wr_id = encode_64bit(client, offset);
    cm_.client_qps[client]->post_receive(lmr, message_size, wr_id, offset);
    ++posted_recvs;
  };

  const auto poll_send_cq = [&]() {
    Context::poll_send_cq(send_wcs.data(), max_cqes, context_.get_send_cq(), [&](u64 wr_id) {
      const auto [_, offset] = decode_64bit(wr_id);
      freelist.push_back(offset);
      --posted_sends;
    });
  };

  // post initial receives
  for (u32 client = 0; client < num_clients_; ++client) {
    post_receive(client);
  }

  while (received_termination_signals < num_clients_) {
    // poll for receive completion events: route query
    const u32 num_received = context_.poll_recv_cq(recv_wcs.data(), max_cqes);
    posted_recvs -= static_cast<i32>(num_received);

    for (u32 i = 0; i < num_received; ++i) {
      const auto [client, offset] = decode_64bit(recv_wcs[i].wr_id);
      const u32 destination = *reinterpret_cast<u32*>(routing_buffer.get_full_buffer() + offset);

      if (destination == termination_signal_mn) {
        std::cerr << "received termination signal from CN" << client << std::endl;
        ++received_termination_signals;

        for (idx_t cn_id = 0; cn_id < num_clients_; ++cn_id) {
          if (client != cn_id) {
            lib_assert(!freelist.empty(), "empty freelist");
            const idx_t offset_term = freelist.back();
            freelist.pop_back();
            *reinterpret_cast<u32*>(routing_buffer.get_full_buffer() + offset_term) = termination_signal_cn;

            lib_assert(posted_sends < max_cqes, "?-?-?(1)");

            cm_.client_qps[cn_id]->post_send_with_id(
              lmr, message_size, IBV_WR_SEND, encode_64bit(cn_id, offset_term), true, nullptr, 0, offset_term);
            ++posted_sends;
            std::cerr << " send termination message to CN" << cn_id << std::endl;
          }
        }

        freelist.push_back(offset);

      } else {
        // std::cerr << "route query " << *reinterpret_cast<node_t*>(routing_buffer.data() + offset + sizeof(u32))
        //           << " from CN" << client << " to CN" << destination << std::endl;
        lib_assert(destination < num_clients_, "invalid route " + std::to_string(destination));
        lib_assert(client != destination, "invalid route (client == destination)");

        // possibly unable to send, because receiver side hasn't taken the request yet
        do {
          poll_send_cq();
        } while (posted_sends >= max_cqes);

        lib_assert(posted_sends < max_cqes, "too many posts...");  // TODO: remove
        cm_.client_qps[destination]->post_send_with_id(
          lmr, message_size, IBV_WR_SEND, encode_64bit(destination, offset), true, nullptr, 0, offset);
        ++posted_sends;
        ++num_routings;

        lib_assert(posted_recvs < max_cqes, "too many recv posts...");  // TODO: remove
        post_receive(client);
      }
    }

    poll_send_cq();  // poll for send completion events and push offset(s) back to freelist
  }

  // poll remaining send completion events
  while (posted_sends > 0) {
    poll_send_cq();
  }

  lib_assert(posted_recvs == 0, "uncompleted posted receives");
  lib_assert(posted_sends == 0, "uncompleted posted sends");
  print_status("received all termination messages");

  // finally, send termination message to all CNs
  {
    const idx_t offset = freelist.back();
    freelist.pop_back();
    *reinterpret_cast<u32*>(routing_buffer.get_full_buffer() + offset) = termination_signal_mn;

    for (idx_t cn_id = 0; cn_id < num_clients_; ++cn_id) {
      std::cerr << "send final termination messages to CN" << cn_id << std::endl;
      for (idx_t b = 0; b < query_router::INITIAL_RECVS; ++b) {
        lib_assert(posted_sends < max_cqes, "?-?-?(2)");
        cm_.client_qps[cn_id]->post_send(lmr, message_size, IBV_WR_SEND, true, nullptr, 0, offset);
      }
    }

    context_.poll_send_cq_until_completion(num_clients_ * query_router::INITIAL_RECVS);
    freelist.push_back(offset);
  }

  print_status("done with routing (num routings: " + std::to_string(num_routings) + ')');
  lib_assert(freelist.size() == buffer_entries, "unfreed messages in buffer");
}

void MemoryNode::idle() {
  print_status("idle: queries");

  // dummy region
  bool done;
  LocalMemoryRegion region{context_, &done, sizeof(bool)};

  for (const QP& qp : cm_.client_qps) {
    qp->post_receive(region);
  }

  // wait
  context_.receive(num_clients_);
}
