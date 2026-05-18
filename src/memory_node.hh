#pragma once

#include <filesystem>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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
#include "coroutine.hh"
#include "http/service_types.hh"
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
      setup_insert_runtime(config);
      storage_worker_config_ = std::make_unique<Configuration>(config);
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
    u32 request_slot_count{1};
  };

  struct PeerRpcRuntimeState {
    HugePage<byte_t> buffer;
    std::unique_ptr<LocalMemoryRegion> region;
    size_t message_bytes{};
    size_t recv_region_bytes{};
  };

  struct PeerPendingSend {
    u32 target_shard{};
    u32 thread_id{};
    u32 coroutine_id{};
    bool async{};
    bool rdma_read_credit{};
  };

  struct StorageOwnerInsertTask {
    u32 client_id{};
    u32 item_count{};
    u64 batch_id{};
    std::chrono::steady_clock::time_point received_at{};
    vec<byte_t> payload;
  };

  class StorageOwnerLocalCache {
  public:
    void init(size_t bytes) {
      if (bytes == 0) {
        return;
      }
      enabled_ = true;
      const size_t snapshot_bytes = VamanaNode::size_until_vector_end() + sizeof(NodeSnapshot) + 64;
      const size_t neighbor_bytes = VamanaNode::NEIGHBORS_SIZE + sizeof(RemotePtr) + 64;
      snapshot_capacity_ = std::max<size_t>(1, bytes / 2 / std::max<size_t>(1, snapshot_bytes));
      neighbor_capacity_ = std::max<size_t>(1, bytes / 2 / std::max<size_t>(1, neighbor_bytes));
    }

    bool enabled() const { return enabled_; }

    bool lookup_snapshot(RemotePtr key, NodeSnapshot& snapshot) {
      if (!enabled_) {
        return false;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = snapshots_.find(key.raw_address);
      if (it == snapshots_.end()) {
        return false;
      }
      snapshot = it->second;
      return true;
    }

    void insert_snapshot(const NodeSnapshot& snapshot) {
      if (!enabled_ || snapshot.rptr.is_null()) {
        return;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      if (snapshots_.contains(snapshot.rptr.raw_address)) {
        snapshots_[snapshot.rptr.raw_address] = snapshot;
        return;
      }
      evict_fifo(snapshot_order_, snapshots_, snapshot_capacity_);
      snapshot_order_.push_back(snapshot.rptr.raw_address);
      snapshots_[snapshot.rptr.raw_address] = snapshot;
    }

    bool lookup_neighbors(RemotePtr key, vec<RemotePtr>& neighbors) {
      if (!enabled_) {
        return false;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = neighbors_.find(key.raw_address);
      if (it == neighbors_.end()) {
        return false;
      }
      neighbors = it->second;
      return true;
    }

    void insert_neighbors(RemotePtr key, const vec<RemotePtr>& values) {
      if (!enabled_ || key.is_null()) {
        return;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      if (neighbors_.contains(key.raw_address)) {
        neighbors_[key.raw_address] = values;
        return;
      }
      evict_fifo(neighbor_order_, neighbors_, neighbor_capacity_);
      neighbor_order_.push_back(key.raw_address);
      neighbors_[key.raw_address] = values;
    }

    void invalidate(RemotePtr key) {
      if (!enabled_ || key.is_null()) {
        return;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      snapshots_.erase(key.raw_address);
      neighbors_.erase(key.raw_address);
    }

  private:
    template <class Value>
    static void evict_fifo(std::deque<u64>& order, std::unordered_map<u64, Value>& map, size_t capacity) {
      while (map.size() >= capacity && !order.empty()) {
        const u64 victim = order.front();
        order.pop_front();
        map.erase(victim);
      }
    }

    bool enabled_{false};
    size_t snapshot_capacity_{0};
    size_t neighbor_capacity_{0};
    std::mutex mutex_;
    std::deque<u64> snapshot_order_;
    std::deque<u64> neighbor_order_;
    std::unordered_map<u64, NodeSnapshot> snapshots_;
    std::unordered_map<u64, vec<RemotePtr>> neighbors_;
  };

  struct StorageOwnerThread {
    explicit StorageOwnerThread(u32 id, u32 num_coroutines, i32 max_send_queue_wr)
        : id(id), send_wcs(std::max<i32>(1, max_send_queue_wr)), post_balances(num_coroutines) {
      for (auto& balance : post_balances) {
        balance.store(0, std::memory_order_relaxed);
      }
    }

    void init_peer_scratch(Context& peer_context, size_t bytes) {
      scratch_buffer.allocate(bytes);
      scratch_buffer.touch_memory();
      scratch_region = std::make_unique<LocalMemoryRegion>(
        peer_context, scratch_buffer.get_full_buffer(), scratch_buffer.buffer_size);
      scratch_stride = align_up(VamanaNode::total_size());
    }

    bool has_peer_scratch() const { return scratch_region != nullptr; }

    void set_current_coroutine(u32 coroutine_id) { running_coroutine = coroutine_id; }
    void track_post() { ++post_balances[running_coroutine]; }
    bool is_ready(u32 coroutine_id) const { return post_balances[coroutine_id] == 0; }
    byte_t* coroutine_scratch(size_t extra_offset = 0) {
      const size_t offset = static_cast<size_t>(running_coroutine) * scratch_stride + extra_offset;
      lib_assert(offset < scratch_buffer.buffer_size, "storage-owner coroutine scratch buffer exhausted");
      return scratch_buffer.get_full_buffer() + offset;
    }
    size_t coroutine_scratch_offset(size_t extra_offset = 0) const {
      return static_cast<size_t>(running_coroutine) * scratch_stride + extra_offset;
    }

    u32 id{};
    vec<ibv_wc> send_wcs;
    vec<std::atomic<i32>> post_balances;
    vec<u_ptr<StorageOwnerInsertCoroutine>> coroutines;
    HugePage<byte_t> scratch_buffer;
    std::unique_ptr<LocalMemoryRegion> scratch_region;
    StorageOwnerLocalCache cache;
    u32 running_coroutine{};
    size_t scratch_stride{};
  };

  struct StorageOwnerInsertJob {
    node_t id{};
    vec<element_t> components;
    bool ok{false};
  };

  using InsertBreakdownCounters = service::storage_owner::InsertBreakdownCounters;

  static u64 elapsed_ns_since(const std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
  }

  static u64 scale_ns(const u64 value, const u32 part, const u32 total) {
    if (value == 0 || part == 0 || total == 0) {
      return 0;
    }
    const u64 quotient = value / total;
    const u64 remainder = value % total;
    return quotient * part + (remainder * part) / total;
  }

  static InsertBreakdownCounters scale_breakdown(const InsertBreakdownCounters& counters,
                                                 const u32 part,
                                                 const u32 total) {
    InsertBreakdownCounters out{};
    out.storage_owner_queue_wait_ns = scale_ns(counters.storage_owner_queue_wait_ns, part, total);
    out.storage_owner_quantize_ns = scale_ns(counters.storage_owner_quantize_ns, part, total);
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
    return out;
  }

  static constexpr u32 kPeerSyncWrOwner = std::numeric_limits<u32>::max();
  static constexpr u32 kPeerAsyncWrOwner = std::numeric_limits<u32>::max() - 1;
  static constexpr u32 kPeerSafeRdAtomic = 8;

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

    peer_index_region_ = std::make_unique<MemoryRegion>(*peer_context_);
    peer_index_region_->register_memory(index_buffer_.get_full_buffer(), index_buffer_.buffer_size, true);

    const MemoryRegionToken local_token = peer_index_region_->createToken();
    std::cerr << "[storage-peer][token] self_shard=" << storage_id_
              << " local_base=" << local_token.address
              << " local_rkey=" << local_token.rkey
              << " local_bytes=" << index_buffer_.buffer_size << std::endl;
    for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
      if (peer_id == storage_id_) continue;
      LocalMemoryRegion peer_token_region{*peer_context_, peer_remote_tokens_[peer_id].get(), sizeof(MemoryRegionToken)};
      peer_qps_[peer_id]->post_receive(peer_token_region);
      peer_qps_[peer_id]->post_send_inlined(&local_token, sizeof(local_token), IBV_WR_SEND);
      peer_context_->poll_send_cq_until_completion();
      peer_context_->receive();
      std::cerr << "[storage-peer][token] self_shard=" << storage_id_
                << " peer_shard=" << peer_id
                << " remote_base=" << peer_remote_tokens_[peer_id]->address
                << " remote_rkey=" << peer_remote_tokens_[peer_id]->rkey << std::endl;
    }

    const size_t scratch_bytes = std::max<size_t>(64ull * 1024ull * 1024ull, align_up(VamanaNode::total_size() * 4));
    peer_scratch_buffer_.allocate(scratch_bytes);
    peer_scratch_buffer_.touch_memory();
    peer_scratch_region_ =
      std::make_unique<LocalMemoryRegion>(*peer_context_, peer_scratch_buffer_.get_full_buffer(), scratch_bytes);
    peer_send_wcs_.resize(std::max<i32>(1, peer_context_->get_config().max_send_queue_wr));

    setup_peer_rpc_runtime(config);

    for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
      if (peer_id == storage_id_) continue;
      u64 header_words[2]{};
      remote_read_bytes(peer_id, 0, header_words, sizeof(header_words), 0);
      std::cerr << "[storage-peer][probe] self_shard=" << storage_id_
                << " peer_shard=" << peer_id
                << " free_ptr=" << header_words[0]
                << " medoid_raw=" << header_words[1] << std::endl;
    }
  }

  void setup_peer_rpc_runtime(const Configuration& config) {
    if (!peer_context_ || num_storage_nodes_ <= 1) {
      return;
    }

    peer_rpc_runtime_.message_bytes = align_up(
      std::max(service::storage_owner::reverse_update_request_bytes(config.R * config.storage_owner_batch_max),
               service::storage_owner::reverse_update_response_bytes()));
    peer_rpc_runtime_.recv_region_bytes = peer_rpc_runtime_.message_bytes * num_storage_nodes_;
    const size_t send_region_bytes = peer_rpc_runtime_.message_bytes * num_storage_nodes_;
    peer_rpc_runtime_.buffer.allocate(peer_rpc_runtime_.recv_region_bytes + send_region_bytes);
    peer_rpc_runtime_.buffer.touch_memory();
    peer_rpc_runtime_.region = std::make_unique<LocalMemoryRegion>(
      *peer_context_, peer_rpc_runtime_.buffer.get_full_buffer(), peer_rpc_runtime_.buffer.buffer_size);

    for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
      if (peer_id == storage_id_) continue;
      peer_qps_[peer_id]->post_receive(
        *peer_rpc_runtime_.region,
        static_cast<u32>(peer_rpc_runtime_.message_bytes),
        peer_id,
        static_cast<u64>(peer_id) * peer_rpc_runtime_.message_bytes);
    }
  }

  void setup_insert_runtime(const Configuration& config) {
    const size_t insert_request_bytes =
      align_up(service::storage_owner::insert_batch_request_bytes(config.storage_owner_batch_max, VamanaNode::DIM));
    insert_runtime_.request_bytes = insert_request_bytes;
    insert_runtime_.request_slot_count = std::max<u32>(1, config.storage_owner_rpc_depth);
    const size_t insert_response_bytes =
      align_up(service::storage_owner::insert_batch_response_bytes(config.storage_owner_batch_max));
    const size_t slot_count =
      static_cast<size_t>(num_clients_) * insert_runtime_.request_slot_count;
    lib_assert(slot_count <= static_cast<size_t>(config.max_recv_queue_wr),
               "storage_owner RPC receive slots exceed memory-node receive CQ capacity");
    insert_runtime_.response_offset = insert_runtime_.request_bytes * slot_count;
    insert_runtime_.buffer.allocate(insert_runtime_.response_offset + insert_response_bytes * slot_count);
    insert_runtime_.buffer.touch_memory();
    insert_runtime_.region = std::make_unique<LocalMemoryRegion>(
      context_, insert_runtime_.buffer.get_full_buffer(), insert_runtime_.buffer.buffer_size);
  }

  void start_storage_owner_insert_workers(const Configuration& config) {
    if (!use_storage_owner_insert_) {
      return;
    }
    print_status("storage-owner peer RDMA read credits per peer: " +
                 std::to_string(peer_rdma_read_credit_limit()) +
                 " (requested=" + std::to_string(storage_owner_peer_rdma_tokens_) + ")");
    const u32 worker_count = std::max<u32>(1, std::min<u32>(8, std::max<u32>(1, num_compute_threads_ / 2)));
    const u32 coroutines_per_worker = std::max<u32>(1, config.insert_coroutines == 0 ? config.num_coroutines
                                                                                      : config.insert_coroutines);
    const size_t scratch_bytes = std::max<size_t>(64ull * 1024ull * 1024ull, align_up(VamanaNode::total_size() * 4));
    const size_t cache_bytes_per_worker =
      worker_count == 0 ? 0 : static_cast<size_t>(config.storage_owner_cache_mb) * 1024ull * 1024ull / worker_count;
    storage_owner_threads_.reserve(worker_count);
    for (u32 i = 0; i < worker_count; ++i) {
      auto thread = std::make_unique<StorageOwnerThread>(i, coroutines_per_worker, config.max_send_queue_wr);
      if (peer_context_) {
        thread->init_peer_scratch(*peer_context_, scratch_bytes);
      }
      thread->cache.init(cache_bytes_per_worker);
      storage_owner_threads_.push_back(std::move(thread));
    }
    storage_owner_async_candidates_.clear();
    storage_owner_async_candidates_.resize(worker_count);
    for (auto& worker_candidates : storage_owner_async_candidates_) {
      worker_candidates.resize(coroutines_per_worker);
    }
    for (u32 i = 0; i < worker_count; ++i) {
      storage_insert_workers_.emplace_back([this, i]() { storage_owner_insert_worker_loop(i); });
    }
  }

  void storage_owner_insert_worker_loop(u32 worker_id) {
    current_storage_owner_thread_ = storage_owner_threads_[worker_id].get();
    const Configuration& config = *storage_worker_config_;
    for (;;) {
      vec<StorageOwnerInsertTask> tasks;
      u32 total_items = 0;
      {
        std::unique_lock<std::mutex> lock(storage_insert_tasks_mutex_);
        storage_insert_tasks_cv_.wait(lock, [&]() {
          return storage_insert_shutdown_.load(std::memory_order_acquire) || !storage_insert_tasks_.empty();
        });
        if (storage_insert_shutdown_.load(std::memory_order_acquire) && storage_insert_tasks_.empty()) {
          current_storage_owner_thread_ = nullptr;
          return;
        }

        while (!storage_insert_tasks_.empty()) {
          const u32 next_items = storage_insert_tasks_.front().item_count;
          if (!tasks.empty() && total_items + next_items > std::max<u32>(config.storage_owner_batch_max, 64)) {
            break;
          }
          total_items += next_items;
          tasks.push_back(std::move(storage_insert_tasks_.front()));
          storage_insert_tasks_.pop_front();
        }
      }

      process_storage_owner_insert_tasks(tasks);
    }
  }

  void repost_peer_rpc_receive(u32 peer_id) {
    if (!peer_context_ || peer_id == storage_id_) {
      return;
    }
    peer_qps_[peer_id]->post_receive(
      *peer_rpc_runtime_.region,
      static_cast<u32>(peer_rpc_runtime_.message_bytes),
      peer_id,
      static_cast<u64>(peer_id) * peer_rpc_runtime_.message_bytes);
  }

  static u64 peer_coroutine_wr_id(u32 thread_id, u32 coroutine_id) {
    return encode_64bit(thread_id, coroutine_id);
  }

  u32 peer_rdma_read_credit_limit() const {
    return std::max<u32>(1, std::min<u32>(storage_owner_peer_rdma_tokens_, kPeerSafeRdAtomic));
  }

  bool try_acquire_peer_rdma_read_credit(u32 shard_id) {
    const u32 limit = peer_rdma_read_credit_limit();
    u32 current = peer_rdma_read_outstanding_[shard_id].load(std::memory_order_acquire);
    while (current < limit) {
      if (peer_rdma_read_outstanding_[shard_id].compare_exchange_weak(current,
                                                                       current + 1,
                                                                       std::memory_order_acq_rel,
                                                                       std::memory_order_acquire)) {
        return true;
      }
    }
    return false;
  }

  void acquire_peer_rdma_read_credit(u32 shard_id) {
    while (!try_acquire_peer_rdma_read_credit(shard_id)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }
  }

  u64 next_peer_sync_wr_id() {
    const u32 id = peer_sync_wr_id_counter_.fetch_add(1, std::memory_order_relaxed);
    return encode_64bit(kPeerSyncWrOwner, id);
  }

  u64 next_peer_async_wr_id() {
    const u32 id = peer_async_wr_id_counter_.fetch_add(1, std::memory_order_relaxed);
    return encode_64bit(kPeerAsyncWrOwner, id);
  }

  void register_peer_pending_send_locked(u64 wr_id, PeerPendingSend pending) {
    peer_pending_sends_[wr_id] = pending;
  }

  void handle_peer_send_completion(u64 wr_id) {
    const auto pending_it = peer_pending_sends_.find(wr_id);
    if (pending_it != peer_pending_sends_.end()) {
      const PeerPendingSend pending = pending_it->second;
      peer_pending_sends_.erase(pending_it);
      if (pending.rdma_read_credit) {
        peer_rdma_read_outstanding_[pending.target_shard].fetch_sub(1, std::memory_order_acq_rel);
      }
      if (pending.async) {
        if (pending.thread_id < storage_owner_threads_.size() && storage_owner_threads_[pending.thread_id]) {
          auto& balance = storage_owner_threads_[pending.thread_id]->post_balances[pending.coroutine_id];
          --balance;
          peer_async_rdma_outstanding_.fetch_sub(1, std::memory_order_acq_rel);
        }
        return;
      }
    }

    const auto [owner, id] = decode_64bit(wr_id);
    if (owner == kPeerSyncWrOwner) {
      peer_sync_completions_.insert(wr_id);
      return;
    }
    if (owner < storage_owner_threads_.size() && storage_owner_threads_[owner]) {
      auto& balance = storage_owner_threads_[owner]->post_balances[id];
      --balance;
      peer_async_rdma_outstanding_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  void poll_peer_send_cq() {
    if (!peer_context_) {
      return;
    }
    std::lock_guard<std::mutex> lock(peer_send_mutex_);
    Context::poll_send_cq(peer_send_wcs_.data(),
                          static_cast<i32>(peer_send_wcs_.size()),
                          peer_context_->get_send_cq(),
                          [&](u64 wr_id) { handle_peer_send_completion(wr_id); });
  }

  bool consume_peer_sync_completion(u64 wr_id) {
    std::lock_guard<std::mutex> lock(peer_send_mutex_);
    const auto it = peer_sync_completions_.find(wr_id);
    if (it == peer_sync_completions_.end()) {
      return false;
    }
    peer_sync_completions_.erase(it);
    return true;
  }

  void wait_peer_sync_completion(u64 wr_id) {
    while (!consume_peer_sync_completion(wr_id)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }
  }

  void send_peer_rpc_message(u32 peer_id, const void* payload, size_t bytes) {
    lib_assert(peer_context_ != nullptr, "peer context not initialized");
    lib_assert(bytes <= peer_rpc_runtime_.message_bytes, "peer rpc message too large");
    const u64 wr_id = next_peer_sync_wr_id();
    const size_t offset = peer_rpc_runtime_.recv_region_bytes + static_cast<size_t>(peer_id) * peer_rpc_runtime_.message_bytes;
    std::lock_guard<std::mutex> rpc_send_lock(peer_rpc_send_mutex_);
    std::memcpy(peer_rpc_runtime_.buffer.get_full_buffer() + offset, payload, bytes);
    {
      std::lock_guard<std::mutex> send_lock(peer_send_mutex_);
      peer_qps_[peer_id]->post_send_with_id(
        *peer_rpc_runtime_.region,
        static_cast<u32>(bytes),
        IBV_WR_SEND,
        wr_id,
        true,
        nullptr,
        0,
        offset);
    }
    wait_peer_sync_completion(wr_id);
  }

  bool handle_peer_reverse_update_request(u32 source_shard,
                                          const service::storage_owner::PeerRpcHeader& header,
                                          const service::storage_owner::ReverseUpdateOp* ops,
                                          const Configuration& config) {
    std::unordered_map<u64, vec<RemotePtr>> grouped;
    grouped.reserve(header.item_count);
    for (u32 i = 0; i < header.item_count; ++i) {
      const RemotePtr target{ops[i].target_raw};
      const RemotePtr candidate{ops[i].candidate_raw};
      lib_assert(local_shard(target.memory_node()), "reverse-update target routed to wrong shard");
      grouped[target.raw_address].push_back(candidate);
    }

    bool success = true;
    for (const auto& [target_raw, candidates] : grouped) {
      success &= apply_local_reverse_update(RemotePtr{target_raw}, candidates, config);
    }

    service::storage_owner::PeerRpcHeader response{};
    response.type = static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_response);
    response.source_shard = storage_id_;
    response.item_count = header.item_count;
    response.request_id = header.request_id;
    response.status = static_cast<u32>(success ? service::storage_owner::InsertStatus::ok
                                               : service::storage_owner::InsertStatus::failed);
    send_peer_rpc_message(source_shard, &response, sizeof(response));
    return success;
  }

  bool pump_peer_rpcs_locked(const Configuration& config, bool wait_for_event = false) {
    if (!peer_context_) {
      return false;
    }

    bool progressed = false;
    vec<ibv_wc> recv_wcs(std::max<i32>(1, peer_context_->get_config().max_recv_queue_wr));
    do {
      const i32 num_received =
        peer_context_->poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
      if (num_received <= 0) {
        break;
      }
      progressed = true;
      for (i32 i = 0; i < num_received; ++i) {
        const u32 peer_id = static_cast<u32>(recv_wcs[i].wr_id);
        const size_t offset = static_cast<size_t>(peer_id) * peer_rpc_runtime_.message_bytes;
        const byte_t* payload = peer_rpc_runtime_.buffer.get_full_buffer() + offset;
        const auto* header = reinterpret_cast<const service::storage_owner::PeerRpcHeader*>(payload);
        if (header->magic != service::storage_owner::kPeerRpcMagic) {
          repost_peer_rpc_receive(peer_id);
          continue;
        }

        if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_request)) {
          const auto* ops = service::storage_owner::reverse_update_ops(payload);
          handle_peer_reverse_update_request(peer_id, *header, ops, config);
        } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_response)) {
          peer_rpc_responses_[header->request_id] = *header;
        }

        repost_peer_rpc_receive(peer_id);
      }
    } while (wait_for_event);

    return progressed;
  }

  bool pump_peer_rpcs(const Configuration& config, bool wait_for_event = false) {
    std::unique_lock<std::mutex> rpc_lock(peer_rpc_mutex_, std::defer_lock);
    if (wait_for_event) {
      rpc_lock.lock();
    } else if (!rpc_lock.try_lock()) {
      return false;
    }
    return pump_peer_rpcs_locked(config, wait_for_event);
  }

  bool wait_for_peer_reverse_update_response_locked(u64 request_id, const Configuration& config) {
    for (;;) {
      const auto it = peer_rpc_responses_.find(request_id);
      if (it != peer_rpc_responses_.end()) {
        const bool success = it->second.status == static_cast<u32>(service::storage_owner::InsertStatus::ok);
        peer_rpc_responses_.erase(it);
        return success;
      }
      if (!pump_peer_rpcs_locked(config, false)) {
        std::this_thread::yield();
      }
    }
  }

  bool send_reverse_update_batch(u32 target_shard,
                                 const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                 const Configuration& config) {
    if (ops.empty()) {
      return true;
    }

    std::lock_guard<std::mutex> rpc_lock(peer_rpc_mutex_);
    const u32 max_items = std::max<u32>(1, config.R * config.storage_owner_batch_max);
    for (size_t begin = 0; begin < ops.size(); begin += max_items) {
      const u32 item_count = static_cast<u32>(std::min<size_t>(ops.size() - begin, max_items));
      const size_t bytes = service::storage_owner::reverse_update_request_bytes(item_count);
      vec<byte_t> message(bytes);
      auto* header = reinterpret_cast<service::storage_owner::PeerRpcHeader*>(message.data());
      header->magic = service::storage_owner::kPeerRpcMagic;
      header->type = static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_request);
      header->source_shard = storage_id_;
      header->item_count = item_count;
      header->request_id = next_peer_request_id_++;
      auto* payload_ops = service::storage_owner::reverse_update_ops(message.data());
      std::memcpy(payload_ops, ops.data() + begin, static_cast<size_t>(item_count) * sizeof(service::storage_owner::ReverseUpdateOp));
      send_peer_rpc_message(target_shard, message.data(), bytes);
      if (!wait_for_peer_reverse_update_response_locked(header->request_id, config)) {
        return false;
      }
    }
    return true;
  }

  void process_storage_owner_insert_tasks(const vec<StorageOwnerInsertTask>& tasks) {
    if (tasks.empty()) {
      return;
    }

    const Configuration& config = *storage_worker_config_;
    vec<node_t> batch_ids;
    vec<element_t> batch_vectors;
    vec<u32> item_counts;
    batch_ids.reserve(std::max<u32>(config.storage_owner_batch_max, 64));
    batch_vectors.reserve(static_cast<size_t>(std::max<u32>(config.storage_owner_batch_max, 64)) * config.dim);
    item_counts.reserve(tasks.size());

    for (const auto& task : tasks) {
      const auto* request = reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(task.payload.data());
      const node_t* ids = service::storage_owner::request_ids(task.payload.data());
      const element_t* vectors = service::storage_owner::request_vectors(task.payload.data(), request->item_count);
      item_counts.push_back(request->item_count);
      batch_ids.insert(batch_ids.end(), ids, ids + request->item_count);
      batch_vectors.insert(batch_vectors.end(),
                           vectors,
                           vectors + static_cast<size_t>(request->item_count) * config.dim);
    }

    InsertBreakdownCounters breakdown{};
    const auto process_started = std::chrono::steady_clock::now();
    for (const auto& task : tasks) {
      breakdown.storage_owner_queue_wait_ns += static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(process_started - task.received_at).count());
    }

    const bool ok = current_storage_owner_thread_ != nullptr
                      ? execute_storage_owner_batch_items_async(batch_ids.data(),
                                                                 batch_vectors.data(),
                                                                 batch_ids.size(),
                                                                 *current_storage_owner_thread_,
                                                                 breakdown,
                                                                 config)
                      : execute_storage_owner_batch_items(batch_ids.data(),
                                                          batch_vectors.data(),
                                                          batch_ids.size(),
                                                          breakdown,
                                                          config);
    for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
      const auto& task = tasks[task_idx];
      const auto* request = reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(task.payload.data());
      const u32 item_count = item_counts[task_idx];
      const size_t response_size = service::storage_owner::insert_batch_response_bytes(item_count);
      vec<byte_t> response_buffer(response_size);
      auto* response = reinterpret_cast<service::storage_owner::InsertBatchResponseHeader*>(response_buffer.data());
      response->magic = service::storage_owner::kInsertMagic;
      response->owner_storage = storage_id_;
      response->item_count = item_count;
      response->batch_id = request->batch_id;
      u32* statuses = service::storage_owner::response_statuses(response_buffer.data());
      for (u32 i = 0; i < item_count; ++i) {
        statuses[i] = static_cast<u32>(ok ? service::storage_owner::InsertStatus::ok
                                          : service::storage_owner::InsertStatus::failed);
      }
      *service::storage_owner::response_breakdown(response_buffer.data(), item_count) =
        scale_breakdown(breakdown, item_count, static_cast<u32>(std::max<size_t>(1, batch_ids.size())));

      LocalMemoryRegion response_region{context_, response_buffer.data(), response_buffer.size()};
      {
        std::lock_guard<std::mutex> lock(storage_send_mutex_);
        cm_.client_qps[task.client_id]->post_send(
          response_region, static_cast<u32>(response_size), IBV_WR_SEND, true, nullptr, 0, 0);
        context_.poll_send_cq_until_completion();
      }
    }
  }

  bool execute_storage_owner_batch_items_async(const node_t* ids,
                                               const element_t* vectors,
                                               size_t item_count,
                                               StorageOwnerThread& thread,
                                               InsertBreakdownCounters& breakdown,
                                               const Configuration& config) {
    if (item_count == 0) {
      return true;
    }

    vec<StorageOwnerInsertJob> jobs;
    jobs.reserve(item_count);
    for (size_t idx = 0; idx < item_count; ++idx) {
      StorageOwnerInsertJob job;
      job.id = ids[idx];
      job.components.assign(vectors + idx * VamanaNode::DIM, vectors + (idx + 1) * VamanaNode::DIM);
      jobs.push_back(std::move(job));
    }

    std::unordered_map<u64, vec<RemotePtr>> local_updates;
    std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_updates;

    const u32 coroutine_count = static_cast<u32>(std::max<size_t>(1, thread.post_balances.size()));
    lib_assert(thread.id < storage_owner_async_candidates_.size(),
               "storage_owner async candidate slots not initialized for worker");
    lib_assert(storage_owner_async_candidates_[thread.id].size() >= coroutine_count,
               "storage_owner async candidate slots not initialized for coroutines");

    thread.coroutines.clear();
    thread.coroutines.reserve(coroutine_count);
    for (u32 i = 0; i < coroutine_count; ++i) {
      thread.coroutines.emplace_back(std::make_unique<StorageOwnerInsertCoroutine>(dummy_storage_owner_insert_coroutine()));
    }

    size_t next_job = 0;
    for (;;) {
      bool all_done = true;
      poll_peer_send_cq();

      for (u32 coroutine_id = 0; coroutine_id < coroutine_count; ++coroutine_id) {
        auto& coroutine = *thread.coroutines[coroutine_id];
        if (coroutine.handle.done()) {
          if (next_job < jobs.size()) {
            coroutine.handle.destroy();
            thread.set_current_coroutine(coroutine_id);
            coroutine.handle = execute_storage_owner_insert_job_async(
              thread, jobs[next_job++], local_updates, remote_updates, breakdown, config).handle;
            all_done = false;
          }
        } else if (thread.is_ready(coroutine_id)) {
          thread.set_current_coroutine(coroutine_id);
          coroutine.handle.resume();
          all_done = false;
        } else {
          all_done = false;
        }
      }

      if (all_done) {
        break;
      }
    }

    for (const auto& coroutine : thread.coroutines) {
      lib_assert(coroutine->handle.done(), "storage-owner insert coroutine not done yet");
      coroutine->handle.destroy();
    }
    thread.coroutines.clear();

    bool ok = true;
    for (const auto& job : jobs) {
      ok &= job.ok;
    }
    auto t_local_reverse = std::chrono::steady_clock::now();
    for (auto& [target_raw, candidates] : local_updates) {
      ok &= apply_local_reverse_update(RemotePtr{target_raw}, candidates, config);
    }
    breakdown.storage_owner_local_reverse_ns += elapsed_ns_since(t_local_reverse);
    auto t_remote_reverse = std::chrono::steady_clock::now();
    for (auto& [target_shard, ops] : remote_updates) {
      ok &= send_reverse_update_batch(target_shard, ops, config);
    }
    breakdown.storage_owner_remote_reverse_ns += elapsed_ns_since(t_remote_reverse);
    return ok;
  }

  static StorageOwnerInsertCoroutine dummy_storage_owner_insert_coroutine() {
    co_return;
  }

  size_t insert_request_slot_offset(u32 client_id, u32 slot_id) const {
    const size_t slot_index =
      static_cast<size_t>(client_id) * insert_runtime_.request_slot_count + slot_id;
    return slot_index * insert_runtime_.request_bytes;
  }

  size_t insert_response_slot_offset(const Configuration& config, u32 client_id, u32 slot_id) const {
    const size_t slot_index =
      static_cast<size_t>(client_id) * insert_runtime_.request_slot_count + slot_id;
    return insert_runtime_.response_offset + slot_index * response_slot_bytes(config);
  }

  void service_storage_runtime(const Configuration& config) {
    print_status("storage-owner insert runtime enabled on shard " + std::to_string(storage_id_));
    vec<ibv_wc> recv_wcs(std::max<i32>(1, config.max_recv_queue_wr));

    for (u32 client_id = 0; client_id < num_clients_; ++client_id) {
      for (u32 slot_id = 0; slot_id < insert_runtime_.request_slot_count; ++slot_id) {
        cm_.client_qps[client_id]->post_receive(
          *insert_runtime_.region,
          static_cast<u32>(insert_runtime_.request_bytes),
          encode_64bit(client_id, slot_id),
          insert_request_slot_offset(client_id, slot_id));
      }
    }

    for (;;) {
      bool progressed = pump_peer_rpcs(config, false);
      const i32 num_received = context_.poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
      progressed = progressed || num_received > 0;
      if (num_received == 0) {
        if (!progressed) {
          std::this_thread::yield();
        }
        continue;
      }

      for (i32 i = 0; i < num_received; ++i) {
        const auto [client_id, slot_id] = decode_64bit(recv_wcs[i].wr_id);
        if (client_id >= num_clients_ || slot_id >= insert_runtime_.request_slot_count) {
          continue;
        }
        const size_t offset = insert_request_slot_offset(client_id, slot_id);
        const byte_t* payload = insert_runtime_.buffer.get_full_buffer() + offset;
        const size_t bytes = recv_wcs[i].byte_len;

        bool handled_async = false;
        if (bytes >= sizeof(service::storage_owner::InsertBatchRequestHeader)) {
          const auto* request = reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(payload);
          if (request->magic == service::storage_owner::kInsertMagic &&
              request->dim == config.dim &&
              request->owner_storage == storage_id_ &&
              request->item_count > 0 &&
              request->item_count <= config.storage_owner_batch_max &&
              bytes >= service::storage_owner::insert_batch_request_bytes(request->item_count, config.dim)) {
            StorageOwnerInsertTask task;
            task.client_id = client_id;
            task.item_count = request->item_count;
            task.batch_id = request->batch_id;
            task.received_at = std::chrono::steady_clock::now();
            task.payload.assign(payload, payload + bytes);
            {
              std::lock_guard<std::mutex> lock(storage_insert_tasks_mutex_);
              storage_insert_tasks_.push_back(std::move(task));
            }
            storage_insert_tasks_cv_.notify_one();
            handled_async = true;
          }
        }

        cm_.client_qps[client_id]->post_receive(
          *insert_runtime_.region,
          static_cast<u32>(insert_runtime_.request_bytes),
          encode_64bit(client_id, slot_id),
          insert_request_slot_offset(client_id, slot_id));

        if (handled_async) {
          continue;
        }

        const size_t response_bytes = handle_storage_insert_request(client_id, payload, bytes, config);
        lib_assert(response_bytes > 0, "invalid storage-owner insert request");

        cm_.client_qps[client_id]->post_send(
          *insert_runtime_.region,
          static_cast<u32>(response_bytes),
          IBV_WR_SEND,
          true,
          nullptr,
          0,
          insert_response_slot_offset(config, client_id, slot_id));
        context_.poll_send_cq_until_completion();
      }
    }
  }

  size_t response_slot_bytes(const Configuration& config) const {
    return align_up(service::storage_owner::insert_batch_response_bytes(config.storage_owner_batch_max));
  }

  size_t handle_storage_insert_request(u32 client_id, const byte_t* payload, size_t bytes, const Configuration& config) {
    if (bytes < sizeof(service::storage_owner::InsertBatchRequestHeader)) {
      return 0;
    }

    const auto* request = reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(payload);
    if (request->magic != service::storage_owner::kInsertMagic ||
        request->dim != config.dim ||
        request->owner_storage != storage_id_ ||
        request->item_count == 0 ||
        request->item_count > config.storage_owner_batch_max ||
        bytes < service::storage_owner::insert_batch_request_bytes(request->item_count, config.dim)) {
      return 0;
    }

    auto* response_ptr = reinterpret_cast<service::storage_owner::InsertBatchResponseHeader*>(
      insert_runtime_.buffer.get_full_buffer() + insert_runtime_.response_offset +
      static_cast<size_t>(client_id) *
        response_slot_bytes(config));
    response_ptr->magic = service::storage_owner::kInsertMagic;
    response_ptr->owner_storage = storage_id_;
    response_ptr->item_count = request->item_count;
    response_ptr->batch_id = request->batch_id;
    u32* statuses = service::storage_owner::response_statuses(response_ptr);

    const node_t* ids = service::storage_owner::request_ids(payload);
    const element_t* vectors = service::storage_owner::request_vectors(payload, request->item_count);
    InsertBreakdownCounters breakdown{};
    const bool ok = execute_storage_owner_batch_items(ids, vectors, request->item_count, breakdown, config);
    for (u32 i = 0; i < request->item_count; ++i) {
      statuses[i] = static_cast<u32>(ok ? service::storage_owner::InsertStatus::ok
                                        : service::storage_owner::InsertStatus::failed);
    }
    *service::storage_owner::response_breakdown(response_ptr, request->item_count) = breakdown;
    return service::storage_owner::insert_batch_response_bytes(request->item_count);
  }

  bool execute_storage_owner_batch_items(const node_t* ids,
                                         const element_t* vectors,
                                         size_t item_count,
                                         InsertBreakdownCounters& breakdown,
                                         const Configuration& config) {
    if (item_count == 0) {
      return true;
    }

    auto t_medoid = std::chrono::steady_clock::now();
    RemotePtr medoid_ptr = read_global_medoid();
    breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
    std::unordered_map<u64, vec<RemotePtr>> local_updates;
    std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_updates;

    for (size_t idx = 0; idx < item_count; ++idx) {
      const element_t* vec_ptr = vectors + idx * VamanaNode::DIM;
      const auto components = span<const element_t>{vec_ptr, VamanaNode::DIM};
      auto t_quantize = std::chrono::steady_clock::now();
      const vec<byte_t> rabitq_data = quantize_rabitq_cpu(components, config);
      breakdown.storage_owner_quantize_ns += elapsed_ns_since(t_quantize);

      if (medoid_ptr.is_null()) {
        const RemotePtr new_ptr = allocate_local_node();
        auto t_write = std::chrono::steady_clock::now();
        write_new_node(new_ptr, ids[idx], components, rabitq_data, {});
        breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
        RemotePtr observed;
        if (try_set_global_medoid(RemotePtr{}, new_ptr, observed) || observed.is_null()) {
          medoid_ptr = new_ptr;
          continue;
        }
        medoid_ptr = observed;
      }

      auto t_search = std::chrono::steady_clock::now();
      const vec<RemotePtr> candidates = beam_search_candidates(components, medoid_ptr, config, &breakdown);
      breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);
      hashset_t<RemotePtr> empty_skip;
      auto t_prune = std::chrono::steady_clock::now();
      vec<RemotePtr> selected_neighbors = robust_prune_cpu(components, candidates, empty_skip, config, &breakdown);
      breakdown.storage_owner_prune_ns += elapsed_ns_since(t_prune);
      const RemotePtr new_ptr = allocate_local_node();
      auto t_write = std::chrono::steady_clock::now();
      write_new_node(new_ptr, ids[idx], components, rabitq_data, selected_neighbors);
      breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);

      for (const RemotePtr& neighbor_ptr : selected_neighbors) {
        if (local_shard(neighbor_ptr.memory_node())) {
          local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
        } else {
          remote_updates[neighbor_ptr.memory_node()].push_back(
            service::storage_owner::ReverseUpdateOp{neighbor_ptr.raw_address, new_ptr.raw_address});
        }
      }
    }

    auto t_local_reverse = std::chrono::steady_clock::now();
    for (auto& [target_raw, candidates] : local_updates) {
      if (!apply_local_reverse_update(RemotePtr{target_raw}, candidates, config)) {
        return false;
      }
    }
    breakdown.storage_owner_local_reverse_ns += elapsed_ns_since(t_local_reverse);
    auto t_remote_reverse = std::chrono::steady_clock::now();
    for (auto& [target_shard, ops] : remote_updates) {
      if (!send_reverse_update_batch(target_shard, ops, config)) {
        return false;
      }
    }
    breakdown.storage_owner_remote_reverse_ns += elapsed_ns_since(t_remote_reverse);
    return true;
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

  void post_peer_read_async(StorageOwnerThread& thread,
                            u32 shard_id,
                            u64 remote_offset,
                            byte_t* dst,
                            size_t bytes,
                            size_t local_offset = 0) {
    if (bytes == 0) {
      return;
    }
    lib_assert(peer_context_ != nullptr, "storage peer context is not initialized");
    lib_assert(thread.has_peer_scratch(), "storage-owner thread scratch is not initialized");
    lib_assert(shard_id < num_storage_nodes_, "invalid peer shard id: " + std::to_string(shard_id));
    lib_assert(peer_qps_[shard_id] != nullptr, "peer QP is not initialized for shard " + std::to_string(shard_id));
    lib_assert(peer_remote_tokens_[shard_id] != nullptr,
               "peer token is not initialized for shard " + std::to_string(shard_id));
    lib_assert(remote_offset + bytes <= mn_memory_bytes_, "peer RDMA read exceeds shard bounds");
    acquire_peer_rdma_read_credit(shard_id);
    while (peer_async_rdma_outstanding_.load(std::memory_order_acquire) >= storage_owner_peer_rdma_tokens_) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }
    peer_async_rdma_outstanding_.fetch_add(1, std::memory_order_acq_rel);
    thread.track_post();
    const u64 wr_id = next_peer_async_wr_id();
    std::lock_guard<std::mutex> send_lock(peer_send_mutex_);
    register_peer_pending_send_locked(
      wr_id,
      PeerPendingSend{shard_id, thread.id, thread.running_coroutine, true, true});
    peer_qps_[shard_id]->post_send(reinterpret_cast<u64>(dst),
                                   static_cast<u32>(bytes),
                                   thread.scratch_region->get_lkey(),
                                   IBV_WR_RDMA_READ,
                                   true,
                                   false,
                                   peer_remote_tokens_[shard_id].get(),
                                   remote_offset,
                                   local_offset,
                                   wr_id);
  }

  auto async_read_global_medoid(StorageOwnerThread& thread) {
    struct Awaitable {
      bool ready{};
      byte_t* buffer{};
      MemoryNode* node{};

      bool await_ready() const { return ready; }
      static void await_suspend(std::coroutine_handle<>) {}
      RemotePtr await_resume() const {
        if (node->storage_id_ == 0) {
          return RemotePtr{*reinterpret_cast<u64*>(node->index_buffer_.get_full_buffer() + 8)};
        }
        return RemotePtr{*reinterpret_cast<const u64*>(buffer)};
      }
    };

    if (storage_id_ == 0) {
      return Awaitable{true, nullptr, this};
    }
    byte_t* buffer = thread.coroutine_scratch();
    post_peer_read_async(thread, 0, 8, buffer, sizeof(u64));
    return Awaitable{false, buffer, this};
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

    if (current_storage_owner_thread_ != nullptr &&
        current_storage_owner_thread_->cache.lookup_snapshot(rptr, snapshot)) {
      return true;
    }

    const size_t read_size = VamanaNode::size_until_vector_end();
    if (local_shard(rptr.memory_node())) {
      const byte_t* ptr = local_node_ptr(rptr);
      snapshot.header = *reinterpret_cast<const u64*>(ptr);
      snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
      snapshot.edge_count = *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
      std::memcpy(snapshot.components.data(), ptr + VamanaNode::offset_vector(), VamanaNode::DIM * sizeof(element_t));
      if (current_storage_owner_thread_ != nullptr) {
        current_storage_owner_thread_->cache.insert_snapshot(snapshot);
      }
      return true;
    }

    StorageOwnerThread* owner_thread = current_storage_owner_thread_;
    byte_t* read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                            ? owner_thread->scratch_buffer.get_full_buffer()
                            : peer_scratch_buffer_.get_full_buffer();
    remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), read_buffer, read_size, 0);
    const byte_t* ptr = read_buffer;
    snapshot.header = *reinterpret_cast<const u64*>(ptr);
    snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
    snapshot.edge_count = *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
    std::memcpy(snapshot.components.data(), ptr + VamanaNode::offset_vector(), VamanaNode::DIM * sizeof(element_t));
    if (current_storage_owner_thread_ != nullptr) {
      current_storage_owner_thread_->cache.insert_snapshot(snapshot);
    }
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
    if (current_storage_owner_thread_ != nullptr &&
        current_storage_owner_thread_->cache.lookup_neighbors(rptr, neighbors)) {
      return neighbors;
    }

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
      if (current_storage_owner_thread_ != nullptr) {
        current_storage_owner_thread_->cache.insert_neighbors(rptr, neighbors);
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
    if (current_storage_owner_thread_ != nullptr) {
      current_storage_owner_thread_->cache.insert_neighbors(rptr, neighbors);
    }
    return neighbors;
  }

  auto async_read_node_snapshot(RemotePtr rptr, StorageOwnerThread& thread) {
    struct Awaitable {
      bool ready{};
      RemotePtr rptr;
      byte_t* buffer{};
      NodeSnapshot snapshot;
      MemoryNode* node{};
      StorageOwnerThread* thread{};

      bool await_ready() const { return ready; }
      static void await_suspend(std::coroutine_handle<>) {}
      NodeSnapshot await_resume() {
        if (ready) {
          return std::move(snapshot);
        }
        snapshot = NodeSnapshot{};
        snapshot.rptr = rptr;
        snapshot.components.resize(VamanaNode::DIM);
        snapshot.header = *reinterpret_cast<const u64*>(buffer);
        snapshot.id = *reinterpret_cast<const u32*>(buffer + VamanaNode::offset_id());
        snapshot.edge_count = *reinterpret_cast<const u8*>(buffer + VamanaNode::offset_edge_count());
        std::memcpy(snapshot.components.data(), buffer + VamanaNode::offset_vector(), VamanaNode::DIM * sizeof(element_t));
        thread->cache.insert_snapshot(snapshot);
        return std::move(snapshot);
      }
    };

    NodeSnapshot cached;
    if (thread.cache.lookup_snapshot(rptr, cached)) {
      return Awaitable{true, rptr, nullptr, std::move(cached), this, &thread};
    }

    if (local_shard(rptr.memory_node())) {
      NodeSnapshot snapshot;
      read_node_snapshot(rptr, snapshot);
      return Awaitable{true, rptr, nullptr, std::move(snapshot), this, &thread};
    }

    byte_t* buffer = thread.coroutine_scratch();
    post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer, VamanaNode::size_until_vector_end());
    return Awaitable{false, rptr, buffer, {}, this, &thread};
  }

  auto async_read_neighbor_list(RemotePtr rptr, StorageOwnerThread& thread) {
    struct Awaitable {
      bool ready{};
      RemotePtr rptr;
      byte_t* buffer{};
      vec<RemotePtr> neighbors;
      MemoryNode* node{};
      StorageOwnerThread* thread{};

      bool await_ready() const { return ready; }
      static void await_suspend(std::coroutine_handle<>) {}
      vec<RemotePtr> await_resume() {
        if (ready) {
          return std::move(neighbors);
        }
        const u8 edge_count = *reinterpret_cast<const u8*>(buffer);
        const auto* slots = reinterpret_cast<const RemotePtr*>(buffer + align_up(sizeof(u8)));
        neighbors.reserve(edge_count);
        for (u32 i = 0; i < edge_count && i < VamanaNode::R; ++i) {
          if (!slots[i].is_null()) {
            neighbors.push_back(slots[i]);
          }
        }
        thread->cache.insert_neighbors(rptr, neighbors);
        return std::move(neighbors);
      }
    };

    vec<RemotePtr> cached;
    if (thread.cache.lookup_neighbors(rptr, cached)) {
      return Awaitable{true, rptr, nullptr, std::move(cached), this, &thread};
    }

    if (local_shard(rptr.memory_node())) {
      vec<RemotePtr> neighbors = read_neighbor_list(rptr);
      return Awaitable{true, rptr, nullptr, std::move(neighbors), this, &thread};
    }

    byte_t* buffer = thread.coroutine_scratch();
    post_peer_read_async(thread,
                         rptr.memory_node(),
                         rptr.byte_offset() + VamanaNode::offset_edge_count(),
                         buffer,
                         sizeof(u8));
    post_peer_read_async(thread,
                         rptr.memory_node(),
                         rptr.byte_offset() + VamanaNode::offset_neighbors(),
                         buffer,
                         VamanaNode::NEIGHBORS_SIZE,
                         align_up(sizeof(u8)));
    return Awaitable{false, rptr, buffer, {}, this, &thread};
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
      invalidate_storage_owner_cache(rptr);
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
    invalidate_storage_owner_cache(rptr);
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
    invalidate_storage_owner_cache(rptr);
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

  vec<RemotePtr> beam_search_candidates(const span<const element_t> query,
                                        RemotePtr medoid,
                                        const Configuration& config,
                                        InsertBreakdownCounters* breakdown = nullptr) {
    hashset_t<RemotePtr> visited;
    vec<BeamEntry> beam;

    NodeSnapshot medoid_snapshot;
    auto t_snapshot = std::chrono::steady_clock::now();
    read_node_snapshot(medoid, medoid_snapshot);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
    }
    auto t_distance = std::chrono::steady_clock::now();
    const distance_t medoid_dist = distance_fn()(query, medoid_snapshot.components, config.dim);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
    }
    beam.push_back({medoid, medoid_dist, false});
    visited.insert(medoid);

    for (;;) {
      i32 best_idx = -1;
      distance_t best_dist = std::numeric_limits<distance_t>::max();
      auto t_select = std::chrono::steady_clock::now();
      for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
        if (!beam[i].expanded && beam[i].distance < best_dist) {
          best_dist = beam[i].distance;
          best_idx = i;
        }
      }
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_select_ns += elapsed_ns_since(t_select);
      }
      if (best_idx < 0) {
        break;
      }

      beam[best_idx].expanded = true;
      auto t_neighbor_read = std::chrono::steady_clock::now();
      const vec<RemotePtr> neighbors = read_neighbor_list(beam[best_idx].rptr);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(t_neighbor_read);
      }
      for (const RemotePtr& neighbor : neighbors) {
        if (neighbor.is_null() || visited.contains(neighbor)) {
          continue;
        }
        visited.insert(neighbor);
        NodeSnapshot snapshot;
        t_snapshot = std::chrono::steady_clock::now();
        read_node_snapshot(neighbor, snapshot);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
        }
        t_distance = std::chrono::steady_clock::now();
        const distance_t dist = distance_fn()(query, snapshot.components, config.dim);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
        }
        auto t_beam_update = std::chrono::steady_clock::now();
        insert_into_beam(beam, neighbor, dist, config.beam_width_construction);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
        }
      }
    }

    vec<RemotePtr> candidates;
    candidates.reserve(beam.size());
    auto t_sort = std::chrono::steady_clock::now();
    std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) { return lhs.distance < rhs.distance; });
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
    }
    for (const auto& entry : beam) {
      candidates.push_back(entry.rptr);
    }
    return candidates;
  }

  auto beam_search_candidates_async(const span<const element_t> query,
                                    RemotePtr medoid,
                                    const Configuration& config,
                                    StorageOwnerThread& thread,
                                    InsertBreakdownCounters* breakdown = nullptr) -> StorageOwnerInsertCoroutine {
    hashset_t<RemotePtr> visited;
    vec<BeamEntry> beam;

    auto t_snapshot = std::chrono::steady_clock::now();
    NodeSnapshot medoid_snapshot = co_await async_read_node_snapshot(medoid, thread);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
    }
    auto t_distance = std::chrono::steady_clock::now();
    const distance_t medoid_dist = distance_fn()(query, medoid_snapshot.components, config.dim);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
    }
    beam.push_back({medoid, medoid_dist, false});
    visited.insert(medoid);

    for (;;) {
      i32 best_idx = -1;
      distance_t best_dist = std::numeric_limits<distance_t>::max();
      auto t_select = std::chrono::steady_clock::now();
      for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
        if (!beam[i].expanded && beam[i].distance < best_dist) {
          best_dist = beam[i].distance;
          best_idx = i;
        }
      }
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_select_ns += elapsed_ns_since(t_select);
      }
      if (best_idx < 0) {
        break;
      }

      beam[best_idx].expanded = true;
      auto t_neighbor_read = std::chrono::steady_clock::now();
      const vec<RemotePtr> neighbors = co_await async_read_neighbor_list(beam[best_idx].rptr, thread);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(t_neighbor_read);
      }
      for (const RemotePtr& neighbor : neighbors) {
        if (neighbor.is_null() || visited.contains(neighbor)) {
          continue;
        }
        visited.insert(neighbor);
        t_snapshot = std::chrono::steady_clock::now();
        NodeSnapshot snapshot = co_await async_read_node_snapshot(neighbor, thread);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
        }
        t_distance = std::chrono::steady_clock::now();
        const distance_t dist = distance_fn()(query, snapshot.components, config.dim);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
        }
        auto t_beam_update = std::chrono::steady_clock::now();
        insert_into_beam(beam, neighbor, dist, config.beam_width_construction);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
        }
      }
    }

    auto& out = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
    out.clear();
    out.reserve(beam.size());
    auto t_sort = std::chrono::steady_clock::now();
    std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) { return lhs.distance < rhs.distance; });
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
    }
    for (const auto& entry : beam) {
      out.push_back(entry.rptr);
    }
  }

  vec<RemotePtr> robust_prune_cpu(const span<const element_t> source,
                                  const vec<RemotePtr>& candidates,
                                  const hashset_t<RemotePtr>& skip,
                                  const Configuration& config,
                                  InsertBreakdownCounters* breakdown = nullptr) {
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
      auto t_snapshot = std::chrono::steady_clock::now();
      read_node_snapshot(candidate, snapshot);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_snapshot);
      }
      auto t_distance = std::chrono::steady_clock::now();
      const distance_t dist = distance_fn()(source, snapshot.components, config.dim);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_distance_ns += elapsed_ns_since(t_distance);
      }
      infos.push_back({candidate, dist, std::move(snapshot.components)});
    }

    auto t_sort = std::chrono::steady_clock::now();
    std::sort(infos.begin(), infos.end(), [](const CandidateInfo& lhs, const CandidateInfo& rhs) {
      return lhs.dist < rhs.dist;
    });
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_sort_ns += elapsed_ns_since(t_sort);
    }

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
        auto t_pair_distance = std::chrono::steady_clock::now();
        const distance_t pair_dist = distance_fn()(candidate.components, selected_components[i], config.dim);
        if (breakdown != nullptr) {
          breakdown->storage_owner_prune_pair_distance_ns += elapsed_ns_since(t_pair_distance);
        }
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

  auto execute_storage_owner_insert_job_async(StorageOwnerThread& thread,
                                              StorageOwnerInsertJob& job,
                                              std::unordered_map<u64, vec<RemotePtr>>& local_updates,
                                              std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>>& remote_updates,
                                              InsertBreakdownCounters& breakdown,
                                              const Configuration& config) -> StorageOwnerInsertCoroutine {
    const auto components = span<const element_t>{job.components.data(), VamanaNode::DIM};
    auto t_quantize = std::chrono::steady_clock::now();
    const vec<byte_t> rabitq_data = quantize_rabitq_cpu(components, config);
    breakdown.storage_owner_quantize_ns += elapsed_ns_since(t_quantize);

    auto t_medoid = std::chrono::steady_clock::now();
    RemotePtr medoid_ptr = co_await async_read_global_medoid(thread);
    breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
    if (medoid_ptr.is_null()) {
      const RemotePtr new_ptr = allocate_local_node();
      auto t_write = std::chrono::steady_clock::now();
      write_new_node(new_ptr, job.id, components, rabitq_data, {});
      breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
      RemotePtr observed;
      if (try_set_global_medoid(RemotePtr{}, new_ptr, observed) || observed.is_null()) {
        job.ok = true;
        co_return;
      }
      medoid_ptr = observed;
    }

    auto t_search = std::chrono::steady_clock::now();
    auto search = beam_search_candidates_async(components, medoid_ptr, config, thread, &breakdown);
    co_await std::suspend_always{};
    while (!search.handle.done()) {
      if (thread.is_ready(thread.running_coroutine)) {
        search.handle.resume();
      } else {
        co_await std::suspend_always{};
      }
    }
    search.handle.destroy();
    breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);

    const vec<RemotePtr>& candidates = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
    hashset_t<RemotePtr> empty_skip;
    auto t_prune = std::chrono::steady_clock::now();
    vec<RemotePtr> selected_neighbors = robust_prune_cpu(components, candidates, empty_skip, config, &breakdown);
    breakdown.storage_owner_prune_ns += elapsed_ns_since(t_prune);
    const RemotePtr new_ptr = allocate_local_node();
    auto t_write = std::chrono::steady_clock::now();
    write_new_node(new_ptr, job.id, components, rabitq_data, selected_neighbors);
    breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);

    for (const RemotePtr& neighbor_ptr : selected_neighbors) {
      if (local_shard(neighbor_ptr.memory_node())) {
        local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
      } else {
        remote_updates[neighbor_ptr.memory_node()].push_back(
          service::storage_owner::ReverseUpdateOp{neighbor_ptr.raw_address, new_ptr.raw_address});
      }
    }
    job.ok = true;
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
    static std::atomic<u32> debug_reads{0};
    const u32 debug_idx = debug_reads.fetch_add(1, std::memory_order_relaxed);
    if (debug_idx < 16) {
      std::cerr << "[storage-peer][read] self_shard=" << storage_id_
                << " target_shard=" << shard_id
                << " remote_base=" << peer_remote_tokens_[shard_id]->address
                << " rkey=" << peer_remote_tokens_[shard_id]->rkey
                << " offset=" << remote_offset
                << " bytes=" << bytes << std::endl;
    }
    StorageOwnerThread* owner_thread = current_storage_owner_thread_;
    HugePage<byte_t>& scratch_buffer =
      owner_thread != nullptr && owner_thread->has_peer_scratch() ? owner_thread->scratch_buffer : peer_scratch_buffer_;
    LocalMemoryRegion& scratch_region =
      owner_thread != nullptr && owner_thread->has_peer_scratch() ? *owner_thread->scratch_region : *peer_scratch_region_;
    lib_assert(scratch_offset + bytes <= scratch_buffer.buffer_size, "peer scratch buffer exhausted");
    byte_t* scratch = scratch_buffer.get_full_buffer() + scratch_offset;
    acquire_peer_rdma_read_credit(shard_id);
    const u64 wr_id = next_peer_sync_wr_id();
    {
      std::lock_guard<std::mutex> send_lock(peer_send_mutex_);
      register_peer_pending_send_locked(
        wr_id,
        PeerPendingSend{shard_id, 0, 0, false, true});
      peer_qps_[shard_id]->post_send(reinterpret_cast<u64>(scratch),
                                     static_cast<u32>(bytes),
                                     scratch_region.get_lkey(),
                                     IBV_WR_RDMA_READ,
                                     true,
                                     false,
                                     peer_remote_tokens_[shard_id].get(),
                                     remote_offset,
                                     0,
                                     wr_id);
    }
    wait_peer_sync_completion(wr_id);
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
    static std::atomic<u32> debug_writes{0};
    const u32 debug_idx = debug_writes.fetch_add(1, std::memory_order_relaxed);
    if (debug_idx < 16) {
      std::cerr << "[storage-peer][write] self_shard=" << storage_id_
                << " target_shard=" << shard_id
                << " remote_base=" << peer_remote_tokens_[shard_id]->address
                << " rkey=" << peer_remote_tokens_[shard_id]->rkey
                << " offset=" << remote_offset
                << " bytes=" << bytes << std::endl;
    }
    StorageOwnerThread* owner_thread = current_storage_owner_thread_;
    HugePage<byte_t>& scratch_buffer =
      owner_thread != nullptr && owner_thread->has_peer_scratch() ? owner_thread->scratch_buffer : peer_scratch_buffer_;
    LocalMemoryRegion& scratch_region =
      owner_thread != nullptr && owner_thread->has_peer_scratch() ? *owner_thread->scratch_region : *peer_scratch_region_;
    lib_assert(scratch_offset + bytes <= scratch_buffer.buffer_size, "peer scratch buffer exhausted");
    byte_t* scratch = scratch_buffer.get_full_buffer() + scratch_offset;
    std::memcpy(scratch, src, bytes);
    const u64 wr_id = next_peer_sync_wr_id();
    {
      std::lock_guard<std::mutex> send_lock(peer_send_mutex_);
      peer_qps_[shard_id]->post_send(reinterpret_cast<u64>(scratch),
                                     static_cast<u32>(bytes),
                                     scratch_region.get_lkey(),
                                     IBV_WR_RDMA_WRITE,
                                     true,
                                     false,
                                     peer_remote_tokens_[shard_id].get(),
                                     remote_offset,
                                     0,
                                     wr_id);
    }
    wait_peer_sync_completion(wr_id);
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
    static std::atomic<u32> debug_cas{0};
    const u32 debug_idx = debug_cas.fetch_add(1, std::memory_order_relaxed);
    if (debug_idx < 16) {
      std::cerr << "[storage-peer][cas] self_shard=" << storage_id_
                << " target_shard=" << shard_id
                << " remote_base=" << peer_remote_tokens_[shard_id]->address
                << " rkey=" << peer_remote_tokens_[shard_id]->rkey
                << " offset=" << remote_offset
                << " expected=" << expected
                << " desired=" << desired << std::endl;
    }
    StorageOwnerThread* owner_thread = current_storage_owner_thread_;
    HugePage<byte_t>& scratch_buffer =
      owner_thread != nullptr && owner_thread->has_peer_scratch() ? owner_thread->scratch_buffer : peer_scratch_buffer_;
    LocalMemoryRegion& scratch_region =
      owner_thread != nullptr && owner_thread->has_peer_scratch() ? *owner_thread->scratch_region : *peer_scratch_region_;
    lib_assert(scratch_offset + sizeof(u64) <= scratch_buffer.buffer_size, "peer scratch buffer exhausted");
    auto* scratch = reinterpret_cast<u64*>(scratch_buffer.get_full_buffer() + scratch_offset);
    *scratch = 0;
    acquire_peer_rdma_read_credit(shard_id);
    const u64 wr_id = next_peer_sync_wr_id();
    {
      std::lock_guard<std::mutex> send_lock(peer_send_mutex_);
      register_peer_pending_send_locked(
        wr_id,
        PeerPendingSend{shard_id, 0, 0, false, true});
      peer_qps_[shard_id]->post_CAS(reinterpret_cast<u64>(scratch),
                                    scratch_region.get_lkey(),
                                    peer_remote_tokens_[shard_id].get(),
                                    remote_offset,
                                    expected,
                                    desired,
                                    true,
                                    wr_id);
    }
    wait_peer_sync_completion(wr_id);
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

  bool apply_local_reverse_update(RemotePtr target_ptr,
                                  const vec<RemotePtr>& candidate_ptrs,
                                  const Configuration& config) {
    lib_assert(local_shard(target_ptr.memory_node()), "target reverse update must be local");
    if (candidate_ptrs.empty()) {
      return true;
    }

    lock_node(target_ptr);
    vec<RemotePtr> updated_neighbors;
    {
      NodeSnapshot target_snapshot;
      read_node_snapshot(target_ptr, target_snapshot);
      vec<RemotePtr> current_neighbors = read_neighbor_list(target_ptr);
      vec<RemotePtr> filtered_candidates;
      filtered_candidates.reserve(candidate_ptrs.size());
      for (const RemotePtr& candidate_ptr : candidate_ptrs) {
        if (candidate_ptr.is_null()) {
          continue;
        }
        bool already_present = false;
        for (const RemotePtr& existing : current_neighbors) {
          if (existing == candidate_ptr) {
            already_present = true;
            break;
          }
        }
        if (!already_present &&
            std::find(filtered_candidates.begin(), filtered_candidates.end(), candidate_ptr) == filtered_candidates.end()) {
          filtered_candidates.push_back(candidate_ptr);
        }
      }

      if (filtered_candidates.empty()) {
        unlock_node(target_ptr);
        return true;
      }

      if (current_neighbors.size() + filtered_candidates.size() <= config.R) {
        current_neighbors.insert(current_neighbors.end(), filtered_candidates.begin(), filtered_candidates.end());
        updated_neighbors = std::move(current_neighbors);
      } else {
        vec<RemotePtr> prune_candidates = current_neighbors;
        prune_candidates.insert(prune_candidates.end(), filtered_candidates.begin(), filtered_candidates.end());
        hashset_t<RemotePtr> skip{target_ptr};
        updated_neighbors = robust_prune_cpu(target_snapshot.components, prune_candidates, skip, config);
      }
    }

    write_neighbor_list(target_ptr, updated_neighbors);
    unlock_node(target_ptr);
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

  void invalidate_storage_owner_cache(RemotePtr rptr) {
    for (auto& thread : storage_owner_threads_) {
      if (thread) {
        thread->cache.invalidate(rptr);
      }
    }
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
  const u32 storage_owner_peer_rdma_tokens_;
  const bool ip_distance_;

  HugePage<byte_t> index_buffer_;
  MemoryRegion index_region_;
  std::unique_ptr<configuration::Configuration> peer_config_;
  std::unique_ptr<Context> peer_context_;
  QPs peer_qps_;
  MemoryRegionTokens peer_remote_tokens_;
  std::unique_ptr<MemoryRegion> peer_index_region_;
  HugePage<byte_t> peer_scratch_buffer_;
  std::unique_ptr<LocalMemoryRegion> peer_scratch_region_;
  PeerRpcRuntimeState peer_rpc_runtime_;
  std::unordered_map<u64, service::storage_owner::PeerRpcHeader> peer_rpc_responses_;
  std::mutex peer_rpc_mutex_;
  std::mutex peer_rpc_send_mutex_;
  std::mutex peer_send_mutex_;
  vec<ibv_wc> peer_send_wcs_;
  std::unordered_set<u64> peer_sync_completions_;
  std::unordered_map<u64, PeerPendingSend> peer_pending_sends_;
  vec<std::atomic<u32>> peer_rdma_read_outstanding_;
  std::atomic<u32> peer_sync_wr_id_counter_{1};
  std::atomic<u32> peer_async_wr_id_counter_{1};
  std::atomic<u32> peer_async_rdma_outstanding_{0};
  u64 next_peer_request_id_{1};
  InsertRuntimeState insert_runtime_;
  std::unique_ptr<Configuration> storage_worker_config_;
  std::mutex storage_send_mutex_;
  std::mutex storage_insert_tasks_mutex_;
  std::condition_variable storage_insert_tasks_cv_;
  std::deque<StorageOwnerInsertTask> storage_insert_tasks_;
  vec<u_ptr<StorageOwnerThread>> storage_owner_threads_;
  vec<vec<vec<RemotePtr>>> storage_owner_async_candidates_;
  vec<std::thread> storage_insert_workers_;
  std::atomic<bool> storage_insert_shutdown_{false};
  service::rabitq::Artifacts rabitq_artifacts_;
  bool rabitq_artifacts_ready_{false};
  const u64 mn_memory_bytes_;
  timing::Timing timing_;

  inline static thread_local StorageOwnerThread* current_storage_owner_thread_{nullptr};
};
