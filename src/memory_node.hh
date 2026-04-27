#pragma once

#include <filesystem>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>
#include <library/connection_manager.hh>
#include <library/detached_qp.hh>
#include <library/hugepage.hh>
#include <library/memory_region.hh>
#include <library/utils.hh>

#include "common/configuration.hh"
#include "common/constants.hh"
#include "common/core_assignment.hh"
#include "common/distance.hh"
#include "common/timing.hh"
#include "service/rabitq_artifacts.hh"
#include "service/storage_owner_protocol.hh"
#include "vamana/vamana_node.hh"

namespace mn_command {

enum Command : u32 { NOOP = 0, LOAD = 1, STORE = 2, SHUTDOWN = 3 };

struct Request {
  Command cmd;
  size_t path_length;
};

struct Response {
  bool success;
  size_t message_length;
};

}  // namespace mn_command

/**
 *  Memory layout:
 *  -----------------------------
 *    buffer: [ free-ptr(8) | entry-ptr(8) | node_a | node_b | ... ]
 *  -----------------------------
 *  Node layout: [
 *     header: 8B                           | ... | ... | is_entry_node(1b) | ... | new_lvl_lock(1b) | ... | lock(1b) |
 *                                                  ^--------- 1B ---------^ ^--------- 1B ---------^ ^----- 1B -----^
 *     meta: 2 * 4B                         | uid(4) | level(4) |
 *     components: d * 4B                   | d_1(4) | ... | d_d(4) |
 *     base-layer: 4B + M_max_0 * 8B        | #neighbors(4) | l_0_1(8) | ... | l_0_M(8) |
 *     upper layer(s) l * (4B + M_max * 8B) | ... |                                        <- only if node's level > 0
 *   ]
 */

/**
 * @brief Establishes a connection to all involved compute nodes.
 *        Allocates a huge memory block and forwards access tokens.
 *        Creates a QP per compute thread and connects them.
 *        Waits until a termination signal is received.
 */
class MemoryNode {
  using Configuration = configuration::IndexConfiguration;
  using Assignment = CoreAssignment<interleaved>;

public:
  explicit MemoryNode(Configuration& config)
      : context_(config), cm_(context_, config), num_clients_(config.num_clients),
        storage_id_(config.storage_id),
        num_storage_nodes_(config.storage_peers.empty() ? config.num_server_nodes()
                                                        : static_cast<u32>(config.storage_peers.size())),
        use_storage_owner_insert_(config.use_storage_owner_insert()),
        ip_distance_(config.ip_distance),
        index_region_(context_),
        mn_memory_bytes_(static_cast<u64>(config.mn_memory_gb) * 1073741824ul) {
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
    VamanaNode::init_static_storage(config.dim, config.R, config.rabitq_bits);
    allocate_memory();

    // free-ptr is initialized to 16 (points to first free address in the buffer)
    *reinterpret_cast<u64*>(index_buffer_.get_full_buffer()) = 16;

    if (!config.server_index_file.empty()) {
      const auto [success, message] = load_index_file(config.server_index_file.string());
      lib_assert(success, message);
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
    const u32 qps_per_node = std::min<u32>(num_compute_threads_, MAX_QPS);
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
      if (config.use_rabitq_search()) {
        load_rabitq_artifacts(config);
      }
      setup_storage_peers(config);
      setup_insert_runtime();
      service_storage_owner_inserts(config);

    } else {
      // service mode: listen for runtime commands
      while (running) {
        running = handle_command();
      }
    }

    print_status("memory node shutting down");
    std::cout << timing_ << std::endl;
  }

private:
  using DistFn = f32 (*)(const span<const f32>&, const span<const f32>&, size_t);

  struct BeamEntry {
    RemotePtr rptr;
    distance_t distance{};
    bool expanded{false};
  };

  struct NodeSnapshot {
    RemotePtr rptr;
    u64 header{};
    node_t id{};
    u8 edge_count{};
    vec<element_t> components;
  };

  struct InsertRuntimeState {
    HugePage<byte_t> buffer;
    std::unique_ptr<LocalMemoryRegion> region;
    size_t request_bytes{};
    size_t response_offset{};
  };

  void allocate_memory() {
    const auto t_allocate = timing_.create_enroll("allocate_index_buffer");
    std::cerr << "allocation size: " << mn_memory_bytes_ << std::endl;

    t_allocate->start();
    const size_t available_memory = index_buffer_.get_memory_size();
    lib_assert(mn_memory_bytes_ <= available_memory, "allocation failed");

    index_buffer_.allocate(mn_memory_bytes_);
    index_buffer_.touch_memory();
    t_allocate->stop();
  }

  /**
   * @brief Handle a single command from the initiator (CN).
   * Blocks on context_.receive() waiting for the next command.
   * @return true if the node should continue running, false on SHUTDOWN.
   */
  bool handle_command() {
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

  std::pair<bool, str> load_index_file(const str& path) {
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

    return {true, ""};
  }

  std::pair<bool, str> store_index_file(const str& path) {
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

  void load_rabitq_artifacts(const Configuration& config) {
    if (!config.use_rabitq_search()) {
      rabitq_artifacts_ready_ = false;
      return;
    }

    str error_message;
    lib_assert(service::rabitq::load_artifacts(config.resolved_index_prefix(), rabitq_artifacts_, &error_message),
               error_message);
    lib_assert(rabitq_artifacts_.dim == config.dim, "RaBitQ artifact dim mismatch on storage node");
    lib_assert(rabitq_artifacts_.rabitq_bits == config.rabitq_bits, "RaBitQ artifact bits mismatch on storage node");
    rabitq_artifacts_ready_ = true;
  }

  void setup_storage_peers(Configuration& config) {
    if (!use_storage_owner_insert_ || num_storage_nodes_ <= 1) {
      return;
    }

    lib_assert(config.storage_peers.size() == num_storage_nodes_,
               "storage_owner mode requires one storage peer endpoint per storage node");
    const auto self_endpoint = parse_endpoint(config.storage_peers[storage_id_], config.port);

    peer_config_ = std::make_unique<configuration::Configuration>(config);
    peer_config_->port = self_endpoint.port;
    peer_config_->is_server = true;
    peer_context_ = std::make_unique<Context>(*peer_config_);
    peer_context_->bind_to_port(self_endpoint.port);

    peer_qps_.resize(num_storage_nodes_);
    peer_remote_tokens_.resize(num_storage_nodes_);
    for (u32 i = 0; i < num_storage_nodes_; ++i) {
      if (i != storage_id_) {
        peer_remote_tokens_[i] = std::make_unique<MemoryRegionToken>();
      }
    }

    for (u32 peer_id = 0; peer_id < storage_id_; ++peer_id) {
      const auto endpoint = parse_endpoint(config.storage_peers[peer_id], config.port);
      peer_qps_[peer_id] = peer_context_->connect_to_server(endpoint.address, endpoint.port, storage_id_);
    }
    for (u32 expected = storage_id_ + 1; expected < num_storage_nodes_; ++expected) {
      auto [qp, peer_id] = peer_context_->wait_for_connection();
      lib_assert(peer_id < num_storage_nodes_, "invalid peer storage id");
      peer_qps_[peer_id] = std::move(qp);
    }
    peer_context_->close_server_socket();

    const MemoryRegionToken local_token = index_region_.createToken();
    for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
      if (peer_id == storage_id_) continue;
      LocalMemoryRegion peer_token_region{*peer_context_, peer_remote_tokens_[peer_id].get(), sizeof(MemoryRegionToken)};
      peer_qps_[peer_id]->post_receive(peer_token_region);
      peer_qps_[peer_id]->post_send_inlined(&local_token, sizeof(local_token), IBV_WR_SEND);
      peer_context_->poll_send_cq_until_completion();
      peer_context_->receive();
    }

    const size_t scratch_bytes = std::max<size_t>(64ull * 1024ull * 1024ull, align_up(VamanaNode::total_size() * 4));
    peer_scratch_buffer_.allocate(scratch_bytes);
    peer_scratch_buffer_.touch_memory();
    peer_scratch_region_ =
      std::make_unique<LocalMemoryRegion>(*peer_context_, peer_scratch_buffer_.get_full_buffer(), scratch_bytes);
  }

  void setup_insert_runtime() {
    insert_runtime_.request_bytes = align_up(service::storage_owner::request_bytes(VamanaNode::DIM));
    const size_t response_bytes = align_up(sizeof(service::storage_owner::InsertResponse));
    insert_runtime_.response_offset = insert_runtime_.request_bytes * num_clients_;
    insert_runtime_.buffer.allocate(insert_runtime_.response_offset + response_bytes * num_clients_);
    insert_runtime_.buffer.touch_memory();
    insert_runtime_.region = std::make_unique<LocalMemoryRegion>(
      context_, insert_runtime_.buffer.get_full_buffer(), insert_runtime_.buffer.buffer_size);
  }

  void service_storage_owner_inserts(const Configuration& config) {
    print_status("storage-owner insert runtime enabled on shard " + std::to_string(storage_id_));
    vec<ibv_wc> recv_wcs(std::max<i32>(1, config.max_recv_queue_wr));

    for (u32 client_id = 0; client_id < num_clients_; ++client_id) {
      cm_.client_qps[client_id]->post_receive(
        *insert_runtime_.region,
        static_cast<u32>(insert_runtime_.request_bytes),
        client_id,
        static_cast<u64>(client_id) * insert_runtime_.request_bytes);
    }

    for (;;) {
      const i32 num_received = context_.poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
      if (num_received == 0) {
        std::this_thread::yield();
        continue;
      }

      for (i32 i = 0; i < num_received; ++i) {
        const u32 client_id = static_cast<u32>(recv_wcs[i].wr_id);
        const size_t offset = static_cast<size_t>(client_id) * insert_runtime_.request_bytes;
        const byte_t* payload = insert_runtime_.buffer.get_full_buffer() + offset;
        const size_t bytes = recv_wcs[i].byte_len;

        const bool success = handle_storage_insert_request(client_id, payload, bytes, config);

        auto* response_ptr = reinterpret_cast<service::storage_owner::InsertResponse*>(
          insert_runtime_.buffer.get_full_buffer() + insert_runtime_.response_offset +
          static_cast<size_t>(client_id) * align_up(sizeof(service::storage_owner::InsertResponse)));
        response_ptr->magic = service::storage_owner::kInsertMagic;
        response_ptr->status = static_cast<u32>(success ? service::storage_owner::InsertStatus::ok
                                                        : service::storage_owner::InsertStatus::failed);

        cm_.client_qps[client_id]->post_send(
          *insert_runtime_.region,
          static_cast<u32>(sizeof(service::storage_owner::InsertResponse)),
          IBV_WR_SEND,
          true,
          nullptr,
          0,
          insert_runtime_.response_offset +
            static_cast<size_t>(client_id) * align_up(sizeof(service::storage_owner::InsertResponse)));
        context_.poll_send_cq_until_completion();

        cm_.client_qps[client_id]->post_receive(
          *insert_runtime_.region,
          static_cast<u32>(insert_runtime_.request_bytes),
          client_id,
          static_cast<u64>(client_id) * insert_runtime_.request_bytes);
      }
    }
  }

  bool handle_storage_insert_request(u32 client_id, const byte_t* payload, size_t bytes, const Configuration& config) {
    (void)client_id;
    if (bytes < sizeof(service::storage_owner::InsertRequest)) {
      return false;
    }

    service::storage_owner::InsertRequest request{};
    std::memcpy(&request, payload, sizeof(request));
    if (request.magic != service::storage_owner::kInsertMagic ||
        request.dim != config.dim ||
        request.owner_storage != storage_id_ ||
        bytes < service::storage_owner::request_bytes(config.dim)) {
      return false;
    }

    auto* response_ptr = reinterpret_cast<service::storage_owner::InsertResponse*>(
      insert_runtime_.buffer.get_full_buffer() + insert_runtime_.response_offset +
      static_cast<size_t>(client_id) * align_up(sizeof(service::storage_owner::InsertResponse)));
    response_ptr->id = request.id;
    response_ptr->owner_storage = storage_id_;
    response_ptr->request_id = request.request_id;

    const auto* components = reinterpret_cast<const element_t*>(payload + sizeof(request));
    return execute_storage_owner_insert(request.id, span<const element_t>{components, config.dim}, config);
  }

  RemotePtr allocate_local_node() {
    size_t node_size = VamanaNode::total_size();
    while (node_size % 8 != 0) {
      node_size += 4;
    }

    auto* free_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer());
    std::atomic_ref<u64> alloc_ref(*free_ptr);
    const u64 offset = alloc_ref.fetch_add(node_size, std::memory_order_acq_rel);
    lib_assert(offset + node_size <= mn_memory_bytes_, "storage node out of memory");
    return RemotePtr{storage_id_, offset};
  }

  RemotePtr read_global_medoid() {
    if (storage_id_ == 0) {
      return RemotePtr{*reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8)};
    }

    u64 raw = 0;
    remote_read_bytes(0, 8, &raw, sizeof(raw), 0);
    return RemotePtr{raw};
  }

  void write_global_medoid(const RemotePtr& medoid) {
    if (storage_id_ == 0) {
      *reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8) = medoid.raw_address;
      return;
    }
    remote_write_bytes(0, 8, &medoid.raw_address, sizeof(medoid.raw_address), 0);
  }

  bool try_set_global_medoid(const RemotePtr& expected, const RemotePtr& desired, RemotePtr& observed) {
    if (storage_id_ == 0) {
      auto* slot = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8);
      std::atomic_ref<u64> ref(*slot);
      u64 current = expected.raw_address;
      const bool ok =
        ref.compare_exchange_strong(current, desired.raw_address, std::memory_order_acq_rel, std::memory_order_acquire);
      observed = RemotePtr{current};
      return ok;
    }

    const u64 original = remote_compare_and_swap(0, 8, expected.raw_address, desired.raw_address, 0);
    observed = RemotePtr{original};
    return original == expected.raw_address;
  }

  bool read_node_snapshot(RemotePtr rptr, NodeSnapshot& snapshot) {
    lib_assert(rptr.memory_node() < num_storage_nodes_,
               "invalid remote shard id in read_node_snapshot: " + std::to_string(rptr.memory_node()));
    lib_assert(rptr.byte_offset() + VamanaNode::size_until_vector_end() <= mn_memory_bytes_,
               "node snapshot read exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
                 " offset=" + std::to_string(rptr.byte_offset()) +
                 " size=" + std::to_string(VamanaNode::size_until_vector_end()) +
                 " capacity=" + std::to_string(mn_memory_bytes_));
    snapshot = NodeSnapshot{};
    snapshot.rptr = rptr;
    snapshot.components.resize(VamanaNode::DIM);

    const size_t read_size = VamanaNode::size_until_vector_end();
    if (local_shard(rptr.memory_node())) {
      const byte_t* ptr = local_node_ptr(rptr);
      snapshot.header = *reinterpret_cast<const u64*>(ptr);
      snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
      snapshot.edge_count = *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
      std::memcpy(snapshot.components.data(), ptr + VamanaNode::offset_vector(), VamanaNode::DIM * sizeof(element_t));
      return true;
    }

    remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), peer_scratch_buffer_.get_full_buffer(), read_size, 0);
    const byte_t* ptr = peer_scratch_buffer_.get_full_buffer();
    snapshot.header = *reinterpret_cast<const u64*>(ptr);
    snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
    snapshot.edge_count = *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
    std::memcpy(snapshot.components.data(), ptr + VamanaNode::offset_vector(), VamanaNode::DIM * sizeof(element_t));
    return true;
  }

  vec<RemotePtr> read_neighbor_list(RemotePtr rptr) {
    lib_assert(rptr.memory_node() < num_storage_nodes_,
               "invalid remote shard id in read_neighbor_list: " + std::to_string(rptr.memory_node()));
    lib_assert(rptr.byte_offset() + VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE <= mn_memory_bytes_,
               "neighbor-list read exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
                 " offset=" + std::to_string(rptr.byte_offset()) +
                 " size=" + std::to_string(VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE) +
                 " capacity=" + std::to_string(mn_memory_bytes_));
    vec<RemotePtr> neighbors;
    if (local_shard(rptr.memory_node())) {
      const byte_t* ptr = local_node_ptr(rptr);
      const u8 edge_count = *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
      const auto* slots = reinterpret_cast<const RemotePtr*>(ptr + VamanaNode::offset_neighbors());
      neighbors.reserve(edge_count);
      for (u32 i = 0; i < edge_count; ++i) {
        if (!slots[i].is_null()) {
          neighbors.push_back(slots[i]);
        }
      }
      return neighbors;
    }

    u8 edge_count = 0;
    remote_read_bytes(rptr.memory_node(), rptr.byte_offset() + VamanaNode::offset_edge_count(), &edge_count, sizeof(edge_count), 0);
    vec<RemotePtr> slots(VamanaNode::R);
    remote_read_bytes(rptr.memory_node(),
                      rptr.byte_offset() + VamanaNode::offset_neighbors(),
                      slots.data(),
                      VamanaNode::NEIGHBORS_SIZE,
                      align_up(sizeof(u64)));
    neighbors.reserve(edge_count);
    for (u32 i = 0; i < edge_count && i < slots.size(); ++i) {
      if (!slots[i].is_null()) {
        neighbors.push_back(slots[i]);
      }
    }
    return neighbors;
  }

  void write_neighbor_list(RemotePtr rptr, const vec<RemotePtr>& neighbors) {
    lib_assert(rptr.memory_node() < num_storage_nodes_,
               "invalid remote shard id in write_neighbor_list: " + std::to_string(rptr.memory_node()));
    lib_assert(rptr.byte_offset() + VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE <= mn_memory_bytes_,
               "neighbor-list write exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
                 " offset=" + std::to_string(rptr.byte_offset()) +
                 " size=" + std::to_string(VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE) +
                 " capacity=" + std::to_string(mn_memory_bytes_));
    const u8 edge_count = static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
    if (local_shard(rptr.memory_node())) {
      byte_t* ptr = local_node_ptr(rptr);
      *reinterpret_cast<u8*>(ptr + VamanaNode::offset_edge_count()) = edge_count;
      std::memset(ptr + VamanaNode::offset_edge_count() + sizeof(u8), 0, VamanaNode::PADDING_SIZE);
      auto* slots = reinterpret_cast<RemotePtr*>(ptr + VamanaNode::offset_neighbors());
      for (u32 i = 0; i < edge_count; ++i) {
        slots[i] = neighbors[i];
      }
      for (u32 i = edge_count; i < VamanaNode::R; ++i) {
        slots[i].reset();
      }
      return;
    }

    byte_t meta[sizeof(u8) + VamanaNode::PADDING_SIZE]{};
    meta[0] = edge_count;
    remote_write_bytes(rptr.memory_node(), rptr.byte_offset() + VamanaNode::offset_edge_count(), meta, sizeof(meta), 0);

    vec<RemotePtr> slots(VamanaNode::R);
    for (u32 i = 0; i < edge_count; ++i) {
      slots[i] = neighbors[i];
    }
    remote_write_bytes(rptr.memory_node(),
                       rptr.byte_offset() + VamanaNode::offset_neighbors(),
                       slots.data(),
                       VamanaNode::NEIGHBORS_SIZE,
                       align_up(sizeof(meta)));
  }

  void write_new_node(RemotePtr rptr,
                      node_t id,
                      const span<const element_t> components,
                      const vec<byte_t>& rabitq_data,
                      const vec<RemotePtr>& neighbors) {
    byte_t* ptr = local_node_ptr(rptr);
    std::memset(ptr, 0, VamanaNode::total_size());
    *reinterpret_cast<u64*>(ptr) = 0;
    *reinterpret_cast<u32*>(ptr + VamanaNode::offset_id()) = id;
    *reinterpret_cast<u8*>(ptr + VamanaNode::offset_edge_count()) = static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
    std::memcpy(ptr + VamanaNode::offset_vector(), components.data(), VamanaNode::DIM * sizeof(element_t));
    if (!rabitq_data.empty()) {
      std::memcpy(ptr + VamanaNode::offset_rabitq(), rabitq_data.data(), std::min<size_t>(rabitq_data.size(), VamanaNode::RABITQ_SIZE));
    }
    auto* slots = reinterpret_cast<RemotePtr*>(ptr + VamanaNode::offset_neighbors());
    for (u32 i = 0; i < neighbors.size() && i < VamanaNode::R; ++i) {
      slots[i] = neighbors[i];
    }
  }

  void lock_node(RemotePtr rptr) {
    if (local_shard(rptr.memory_node())) {
      auto* header_ptr = reinterpret_cast<u64*>(local_node_ptr(rptr));
      std::atomic_ref<u64> ref(*header_ptr);
      for (;;) {
        u64 header = ref.load(std::memory_order_acquire);
        if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
          std::this_thread::yield();
          continue;
        }
        const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
        if (ref.compare_exchange_weak(header, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
          return;
        }
      }
    }

    for (;;) {
      auto [success, header] = try_lock_remote_header(rptr);
      if (success) {
        return;
      }
      if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
        std::this_thread::yield();
      }
    }
  }

  void unlock_node(RemotePtr rptr) {
    if (local_shard(rptr.memory_node())) {
      auto* header_ptr = reinterpret_cast<u64*>(local_node_ptr(rptr));
      std::atomic_ref<u64> ref(*header_ptr);
      ref.fetch_and(~static_cast<u64>(VamanaNode::HEADER_NODE_LOCK), std::memory_order_acq_rel);
      return;
    }

    const byte_t unlock = 0;
    remote_write_bytes(rptr.memory_node(), rptr.byte_offset() + VamanaNode::HEADER_UNTIL_LOCK, &unlock, 1, 0);
  }

  vec<RemotePtr> beam_search_candidates(const span<const element_t> query, RemotePtr medoid, const Configuration& config) {
    hashset_t<RemotePtr> visited;
    vec<BeamEntry> beam;

    NodeSnapshot medoid_snapshot;
    read_node_snapshot(medoid, medoid_snapshot);
    const distance_t medoid_dist = distance_fn()(query, medoid_snapshot.components, config.dim);
    beam.push_back({medoid, medoid_dist, false});
    visited.insert(medoid);

    for (;;) {
      i32 best_idx = -1;
      distance_t best_dist = std::numeric_limits<distance_t>::max();
      for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
        if (!beam[i].expanded && beam[i].distance < best_dist) {
          best_dist = beam[i].distance;
          best_idx = i;
        }
      }
      if (best_idx < 0) {
        break;
      }

      beam[best_idx].expanded = true;
      const vec<RemotePtr> neighbors = read_neighbor_list(beam[best_idx].rptr);
      for (const RemotePtr& neighbor : neighbors) {
        if (neighbor.is_null() || visited.contains(neighbor)) {
          continue;
        }
        visited.insert(neighbor);
        NodeSnapshot snapshot;
        read_node_snapshot(neighbor, snapshot);
        const distance_t dist = distance_fn()(query, snapshot.components, config.dim);
        insert_into_beam(beam, neighbor, dist, config.beam_width_construction);
      }
    }

    vec<RemotePtr> candidates;
    candidates.reserve(beam.size());
    std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) { return lhs.distance < rhs.distance; });
    for (const auto& entry : beam) {
      candidates.push_back(entry.rptr);
    }
    return candidates;
  }

  vec<RemotePtr> robust_prune_cpu(const span<const element_t> source,
                                  const vec<RemotePtr>& candidates,
                                  const hashset_t<RemotePtr>& skip,
                                  const Configuration& config) {
    struct CandidateInfo {
      RemotePtr rptr;
      distance_t dist{};
      vec<element_t> components;
    };

    vec<CandidateInfo> infos;
    infos.reserve(candidates.size());
    for (const RemotePtr& candidate : candidates) {
      if (candidate.is_null() || skip.contains(candidate)) {
        continue;
      }
      NodeSnapshot snapshot;
      read_node_snapshot(candidate, snapshot);
      infos.push_back({candidate, distance_fn()(source, snapshot.components, config.dim), std::move(snapshot.components)});
    }

    std::sort(infos.begin(), infos.end(), [](const CandidateInfo& lhs, const CandidateInfo& rhs) {
      return lhs.dist < rhs.dist;
    });

    vec<RemotePtr> selected;
    selected.reserve(config.R);
    vec<span<const element_t>> selected_components;
    selected_components.reserve(config.R);

    for (const auto& candidate : infos) {
      if (selected.size() >= config.R) {
        break;
      }

      bool pruned = false;
      for (idx_t i = 0; i < selected_components.size(); ++i) {
        const distance_t pair_dist = distance_fn()(candidate.components, selected_components[i], config.dim);
        if (config.alpha * pair_dist <= candidate.dist) {
          pruned = true;
          break;
        }
      }

      if (!pruned) {
        selected.push_back(candidate.rptr);
        selected_components.push_back(candidate.components);
      }
    }

    return selected;
  }

  vec<byte_t> quantize_rabitq_cpu(const span<const element_t> components, const Configuration& config) const {
    vec<byte_t> output(VamanaNode::RABITQ_SIZE, 0);
    if (!config.use_rabitq_search() || !rabitq_artifacts_ready_) {
      return output;
    }

    const u32 dim = config.dim;
    const u32 bits = config.rabitq_bits;
    const u32 packed_bytes = (bits * dim + 7) / 8;
    vec<float> rotated(dim, 0.0f);
    for (u32 col = 0; col < dim; ++col) {
      float sum = 0.0f;
      for (u32 row = 0; row < dim; ++row) {
        sum += rabitq_artifacts_.rotation_matrix[row + static_cast<size_t>(col) * dim] * components[row];
      }
      rotated[col] = sum;
    }

    constexpr double kEps = 1e-5;
    vec<uint8_t> uncompressed(dim, 0);
    float l2_sqr = 0.0f;
    for (u32 j = 0; j < dim; ++j) {
      const float diff = rotated[j] - rabitq_artifacts_.rotated_centroid[j];
      l2_sqr += diff * diff;
    }
    const float l2_norm = std::sqrt(std::max(l2_sqr, 1e-12f));

    float ip_norm = 0.0f;
    for (u32 j = 0; j < dim; ++j) {
      const float abs_o = std::fabs((rotated[j] - rabitq_artifacts_.rotated_centroid[j]) / l2_norm);
      int val = static_cast<int>((rabitq_artifacts_.t_const * abs_o) + kEps);
      if (val >= (1 << (bits - 1))) {
        val = (1 << (bits - 1)) - 1;
      }
      uncompressed[j] = static_cast<uint8_t>(val);
      ip_norm += (static_cast<float>(val) + 0.5f) * abs_o;
    }
    const float ip_norm_inv = ip_norm == 0.0f ? 1.0f : (1.0f / ip_norm);

    const uint32_t mask = (1u << (bits - 1)) - 1u;
    for (u32 j = 0; j < dim; ++j) {
      const float residual = rotated[j] - rabitq_artifacts_.rotated_centroid[j];
      if (residual >= 0.0f) {
        uncompressed[j] = static_cast<uint8_t>(uncompressed[j] + (1u << (bits - 1)));
      } else {
        uncompressed[j] = static_cast<uint8_t>((~uncompressed[j]) & mask);
      }
    }

    const float cb = -(static_cast<float>(1 << (bits - 1)) - 0.5f);
    float ip_resi_xucb = 0.0f;
    float ip_cent_xucb = 0.0f;
    for (u32 j = 0; j < dim; ++j) {
      const float residual = rotated[j] - rabitq_artifacts_.rotated_centroid[j];
      const float xu_cb = static_cast<float>(uncompressed[j]) + cb;
      ip_resi_xucb += residual * xu_cb;
      ip_cent_xucb += rabitq_artifacts_.rotated_centroid[j] * xu_cb;
    }
    if (ip_resi_xucb == 0.0f) {
      ip_resi_xucb = FLT_MAX;
    }

    const float add = l2_sqr + 2.0f * l2_sqr * ip_cent_xucb / ip_resi_xucb;
    const float rescale = ip_norm_inv * -2.0f * l2_norm;

    const u32 values_per_byte = 8 / bits;
    for (u32 byte = 0; byte < packed_bytes; ++byte) {
      uint8_t packed = 0;
      for (u32 k = 0; k < values_per_byte; ++k) {
        const u32 dim_idx = byte * values_per_byte + k;
        if (dim_idx < dim) {
          packed |= static_cast<uint8_t>(uncompressed[dim_idx] << (k * bits));
        }
      }
      output[byte] = packed;
    }
    std::memcpy(output.data() + packed_bytes, &add, sizeof(float));
    std::memcpy(output.data() + packed_bytes + sizeof(float), &rescale, sizeof(float));
    return output;
  }

  void remote_read_bytes(u32 shard_id, u64 remote_offset, void* dst, size_t bytes, size_t scratch_offset) {
    if (bytes == 0) return;
    lib_assert(peer_context_ != nullptr, "storage peer context is not initialized");
    lib_assert(shard_id < num_storage_nodes_, "invalid peer shard id: " + std::to_string(shard_id));
    lib_assert(peer_qps_[shard_id] != nullptr, "peer QP is not initialized for shard " + std::to_string(shard_id));
    lib_assert(peer_remote_tokens_[shard_id] != nullptr,
               "peer token is not initialized for shard " + std::to_string(shard_id));
    lib_assert(peer_remote_tokens_[shard_id]->address != 0 && peer_remote_tokens_[shard_id]->rkey != 0,
               "peer token is invalid for shard " + std::to_string(shard_id));
    lib_assert(remote_offset + bytes <= mn_memory_bytes_,
               "peer RDMA read exceeds shard bounds: shard=" + std::to_string(shard_id) +
                 " offset=" + std::to_string(remote_offset) +
                 " bytes=" + std::to_string(bytes) +
                 " capacity=" + std::to_string(mn_memory_bytes_));
    lib_assert(scratch_offset + bytes <= peer_scratch_buffer_.buffer_size, "peer scratch buffer exhausted");
    byte_t* scratch = peer_scratch_buffer_.get_full_buffer() + scratch_offset;
    peer_qps_[shard_id]->post_send(reinterpret_cast<u64>(scratch),
                                   static_cast<u32>(bytes),
                                   peer_scratch_region_->get_lkey(),
                                   IBV_WR_RDMA_READ,
                                   true,
                                   false,
                                   peer_remote_tokens_[shard_id].get(),
                                   remote_offset,
                                   0,
                                   0);
    peer_context_->poll_send_cq_until_completion();
    std::memcpy(dst, scratch, bytes);
  }

  void remote_write_bytes(u32 shard_id, u64 remote_offset, const void* src, size_t bytes, size_t scratch_offset) {
    if (bytes == 0) return;
    lib_assert(peer_context_ != nullptr, "storage peer context is not initialized");
    lib_assert(shard_id < num_storage_nodes_, "invalid peer shard id: " + std::to_string(shard_id));
    lib_assert(peer_qps_[shard_id] != nullptr, "peer QP is not initialized for shard " + std::to_string(shard_id));
    lib_assert(peer_remote_tokens_[shard_id] != nullptr,
               "peer token is not initialized for shard " + std::to_string(shard_id));
    lib_assert(peer_remote_tokens_[shard_id]->address != 0 && peer_remote_tokens_[shard_id]->rkey != 0,
               "peer token is invalid for shard " + std::to_string(shard_id));
    lib_assert(remote_offset + bytes <= mn_memory_bytes_,
               "peer RDMA write exceeds shard bounds: shard=" + std::to_string(shard_id) +
                 " offset=" + std::to_string(remote_offset) +
                 " bytes=" + std::to_string(bytes) +
                 " capacity=" + std::to_string(mn_memory_bytes_));
    lib_assert(scratch_offset + bytes <= peer_scratch_buffer_.buffer_size, "peer scratch buffer exhausted");
    byte_t* scratch = peer_scratch_buffer_.get_full_buffer() + scratch_offset;
    std::memcpy(scratch, src, bytes);
    peer_qps_[shard_id]->post_send(reinterpret_cast<u64>(scratch),
                                   static_cast<u32>(bytes),
                                   peer_scratch_region_->get_lkey(),
                                   IBV_WR_RDMA_WRITE,
                                   true,
                                   false,
                                   peer_remote_tokens_[shard_id].get(),
                                   remote_offset,
                                   0,
                                   0);
    peer_context_->poll_send_cq_until_completion();
  }

  u64 remote_compare_and_swap(u32 shard_id, u64 remote_offset, u64 expected, u64 desired, size_t scratch_offset) {
    lib_assert(peer_context_ != nullptr, "storage peer context is not initialized");
    lib_assert(shard_id < num_storage_nodes_, "invalid peer shard id: " + std::to_string(shard_id));
    lib_assert(peer_qps_[shard_id] != nullptr, "peer QP is not initialized for shard " + std::to_string(shard_id));
    lib_assert(peer_remote_tokens_[shard_id] != nullptr,
               "peer token is not initialized for shard " + std::to_string(shard_id));
    lib_assert(peer_remote_tokens_[shard_id]->address != 0 && peer_remote_tokens_[shard_id]->rkey != 0,
               "peer token is invalid for shard " + std::to_string(shard_id));
    lib_assert(remote_offset + sizeof(u64) <= mn_memory_bytes_,
               "peer CAS exceeds shard bounds: shard=" + std::to_string(shard_id) +
                 " offset=" + std::to_string(remote_offset) +
                 " capacity=" + std::to_string(mn_memory_bytes_));
    lib_assert(scratch_offset + sizeof(u64) <= peer_scratch_buffer_.buffer_size, "peer scratch buffer exhausted");
    auto* scratch = reinterpret_cast<u64*>(peer_scratch_buffer_.get_full_buffer() + scratch_offset);
    *scratch = 0;
    peer_qps_[shard_id]->post_CAS(reinterpret_cast<u64>(scratch),
                                  peer_scratch_region_->get_lkey(),
                                  peer_remote_tokens_[shard_id].get(),
                                  remote_offset,
                                  expected,
                                  desired,
                                  true,
                                  0);
    peer_context_->poll_send_cq_until_completion();
    return *scratch;
  }

  std::pair<bool, u64> try_lock_remote_header(RemotePtr rptr) {
    u64 header = 0;
    remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), &header, sizeof(header), 0);
    if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
      return {false, header};
    }
    const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
    const u64 original = remote_compare_and_swap(rptr.memory_node(), rptr.byte_offset(), header, desired, align_up(sizeof(header)));
    return {original == header, original};
  }

  bool execute_storage_owner_insert(node_t id, const span<const element_t> components, const Configuration& config) {
    RemotePtr medoid_ptr = read_global_medoid();
    const vec<byte_t> rabitq_data = quantize_rabitq_cpu(components, config);

    if (medoid_ptr.is_null()) {
      const RemotePtr new_ptr = allocate_local_node();
      write_new_node(new_ptr, id, components, rabitq_data, {});
      RemotePtr observed;
      if (try_set_global_medoid(RemotePtr{}, new_ptr, observed) || observed.is_null()) {
        return true;
      }
      medoid_ptr = observed;
    }

    const vec<RemotePtr> candidates = beam_search_candidates(components, medoid_ptr, config);
    hashset_t<RemotePtr> empty_skip;
    vec<RemotePtr> selected_neighbors = robust_prune_cpu(components, candidates, empty_skip, config);

    const RemotePtr new_ptr = allocate_local_node();
    write_new_node(new_ptr, id, components, rabitq_data, selected_neighbors);

    for (const RemotePtr& neighbor_ptr : selected_neighbors) {
      lock_node(neighbor_ptr);
      vec<RemotePtr> updated_neighbors;
      {
        NodeSnapshot neighbor_snapshot;
        read_node_snapshot(neighbor_ptr, neighbor_snapshot);
        vec<RemotePtr> current_neighbors = read_neighbor_list(neighbor_ptr);
        bool already_present = false;
        for (const RemotePtr& existing : current_neighbors) {
          if (existing == new_ptr) {
            already_present = true;
            break;
          }
        }
        if (already_present) {
          unlock_node(neighbor_ptr);
          continue;
        }

        if (current_neighbors.size() < config.R) {
          current_neighbors.push_back(new_ptr);
          updated_neighbors = std::move(current_neighbors);
        } else {
          vec<RemotePtr> prune_candidates = current_neighbors;
          prune_candidates.push_back(new_ptr);
          hashset_t<RemotePtr> skip{neighbor_ptr};
          updated_neighbors = robust_prune_cpu(neighbor_snapshot.components, prune_candidates, skip, config);
        }
      }

      write_neighbor_list(neighbor_ptr, updated_neighbors);
      unlock_node(neighbor_ptr);
    }

    return true;
  }

  static size_t align_up(size_t value, size_t alignment = CACHELINE_SIZE) {
    while (value % alignment != 0) {
      ++value;
    }
    return value;
  }

  DistFn distance_fn() const {
    return ip_distance_ ? &ip_distance : &l2;
  }

  bool local_shard(u32 shard_id) const { return shard_id == storage_id_; }

  byte_t* local_node_ptr(const RemotePtr& rptr) {
    return index_buffer_.get_full_buffer() + rptr.byte_offset();
  }

  const byte_t* local_node_ptr(const RemotePtr& rptr) const {
    return index_buffer_.get_full_buffer() + rptr.byte_offset();
  }

  static void insert_into_beam(vec<BeamEntry>& beam, const RemotePtr& rptr, distance_t dist, u32 max_beam_width) {
    auto it = std::lower_bound(
      beam.begin(), beam.end(), dist, [](const BeamEntry& entry, distance_t value) { return entry.distance < value; });
    beam.insert(it, {rptr, dist, false});
    if (beam.size() > max_beam_width) {
      beam.resize(max_beam_width);
    }
  }

  void route_queries(i32 max_cqes) {
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

  void idle() {
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

private:
  Context context_;
  ServerConnectionManager cm_;
  Assignment core_assignment_;

  const u32 num_clients_;
  u32 num_compute_threads_{};
  const u32 storage_id_;
  const u32 num_storage_nodes_;
  const bool use_storage_owner_insert_;
  const bool ip_distance_;

  HugePage<byte_t> index_buffer_;
  MemoryRegion index_region_;
  std::unique_ptr<configuration::Configuration> peer_config_;
  std::unique_ptr<Context> peer_context_;
  QPs peer_qps_;
  MemoryRegionTokens peer_remote_tokens_;
  HugePage<byte_t> peer_scratch_buffer_;
  std::unique_ptr<LocalMemoryRegion> peer_scratch_region_;
  InsertRuntimeState insert_runtime_;
  service::rabitq::Artifacts rabitq_artifacts_;
  bool rabitq_artifacts_ready_{false};
  const u64 mn_memory_bytes_;
  timing::Timing timing_;
};
