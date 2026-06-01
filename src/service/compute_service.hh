#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <library/connection_manager.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "common/core_assignment.hh"
#include "http/vamana_service_scheduler.hh"
#include "memory_node/command_protocol.hh"
#include "service/breakdown.hh"
#include "service/compute_service/state.hh"
#include "service/storage_owner_protocol.hh"
#include "service/rabitq_artifacts.hh"
#include "vamana/vamana.hh"
#include "worker_pool.hh"

template <class Distance>
class ComputeService {
private:
  using Configuration = configuration::IndexConfiguration;
  using Assignment = CoreAssignment<interleaved>;

  struct CommandResult {
    bool success;
    str message;
  };

  struct RpcHeader {
    u32 magic{};
    u32 type{};
    u32 source_client{};
    u32 origin_client{};
    u64 request_id{};
    u32 top_k{};
    u32 payload_count{};
  };

  enum RpcType : u32 {
    rpc_register_centroid = 1,
    rpc_register_ack = 2,
    rpc_search_proxy = 3,
    rpc_search_request = 4,
    rpc_search_response = 5,
  };

  struct RpcOutbound {
    u32 destination_client{};
    RpcType type{};
    u64 request_id{};
    u32 origin_client{};
    u32 top_k{};
    vec<element_t> float_payload;
    vec<node_t> id_payload;
  };

public:
  struct InsertItem {
    node_t id;
    vec<element_t> values;
  };

  struct Status {
    str state{"running"};
    size_t vectors_inserted{};
    u32 dimension{};
    u32 threads{};
  };

  struct ServiceProfile {
    u32 insert_workers{};
    u32 query_workers{};
    u32 insert_coroutines{};
    u32 query_coroutines{};
  };

  struct LocalMainSearchOutput {
    service::QueryResult results;
    std::shared_ptr<service::breakdown::Sample> sample;
  };

public:
  explicit ComputeService(const Configuration& config, bool shutdown_remote_on_stop = false);
  ~ComputeService();

  ComputeService(const ComputeService&) = delete;
  ComputeService& operator=(const ComputeService&) = delete;

  size_t insert(const vec<InsertItem>& batch);
  vec<node_t> search(const vec<element_t>& query, u32 k);
  bool load_index(const std::string& path, str* error_message = nullptr);
  bool store_index(const std::string& path, str* error_message = nullptr);
  Status status() const;
  void reset_breakdown_state();
  void clear_thread_statistics();
  service::breakdown::Report collect_breakdown_report() const;

  const Configuration& config() const { return config_; }

private:
  struct StorageInsertTask {
    InsertItem item;
    std::shared_ptr<service::breakdown::Sample> sample;
    std::promise<bool> result;
    std::chrono::steady_clock::time_point enqueued_at{};
    std::chrono::steady_clock::time_point sender_dequeued_at{};
  };

  struct StorageOwnerRpcSlot {
    u32 owner_storage{};
    u32 slot_id{};
    bool in_use{false};
    bool send_done{false};
    bool response_done{false};
    bool results_completed{false};
    u32 item_count{};
    u64 batch_id{};
    u64 batch_wait_ns{};
    u64 request_prepare_ns{};
    size_t request_size{};
    size_t response_size{};
    std::chrono::steady_clock::time_point send_posted_at{};
    std::chrono::steady_clock::time_point send_completed_at{};
    std::chrono::steady_clock::time_point response_completed_at{};
    vec<byte_t> request_buffer;
    vec<byte_t> response_buffer;
    std::unique_ptr<LocalMemoryRegion> request_region;
    std::unique_ptr<LocalMemoryRegion> response_region;
    vec<std::unique_ptr<StorageInsertTask>> tasks;
    vec<std::shared_ptr<service::breakdown::Sample>> samples;
  };

  struct StorageOwnerResponseSlot {
    u32 owner_storage{};
    u32 slot_id{};
    vec<byte_t> buffer;
    std::unique_ptr<LocalMemoryRegion> region;
  };

  struct StorageOwnerSenderState {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::unique_ptr<StorageInsertTask>> queue;
    vec<StorageOwnerRpcSlot> slots;
    vec<StorageOwnerResponseSlot> response_slots;
    std::deque<u32> free_slots;
    std::unordered_map<u64, u32> batch_to_slot;
    std::thread thread;
  };

  void init_remote_tokens();
  void receive_remote_access_tokens();
  void wait_for_load_or_store();
  ServiceProfile resolve_service_profile() const;
  bool maybe_load_rabitq_artifacts(const filepath_t& index_prefix, str* error_message = nullptr);
  void upload_rabitq_artifacts(const service::rabitq::Artifacts& artifacts);
  void synchronize_clients_after_startup();
  vec<CommandResult> send_index_command(mn_command::Command cmd, const std::string& path);
  void start_workers();
  void stop_workers();
  void pause_workers();
  void resume_workers();
  LocalMainSearchOutput search_local_result(const vec<element_t>& query, u32 k);
  vec<node_t> search_local(const vec<element_t>& query, u32 k);
  void start_storage_insert_runtime();
  void stop_storage_insert_runtime();
  void run_storage_insert_sender(u32 owner_storage);
  void run_storage_insert_completion_loop();
  void post_storage_owner_batch(u32 owner_storage,
                                u32 slot_id,
                                vec<std::unique_ptr<StorageInsertTask>>&& tasks,
                                u64 batch_wait_ns);
  void handle_storage_owner_send_completion(u32 owner_storage, u32 slot_id);
  void handle_storage_owner_response(u32 owner_storage, u32 response_slot_id);
  void post_storage_owner_response_receive(u32 owner_storage, u32 response_slot_id);
  void maybe_release_storage_owner_slot_locked(StorageOwnerSenderState& state,
                                               StorageOwnerRpcSlot& slot);
  void fail_storage_owner_tasks(vec<std::unique_ptr<StorageInsertTask>>& tasks);
  bool routing_enabled() const;
  size_t rpc_message_size() const;
  vec<element_t> compute_local_routing_centroid() const;
  void start_rpc();
  void stop_rpc();
  void pause_rpc();
  void resume_rpc();
  void run_rpc_loop();
  void handle_rpc_receive(const RpcHeader& header, const byte_t* payload);
  void handle_search_proxy(const RpcHeader& header, const byte_t* payload);
  void handle_search_request(const RpcHeader& header, const byte_t* payload);
  void handle_search_response(const RpcHeader& header, const byte_t* payload);
  void handle_register_centroid(const RpcHeader& header, const byte_t* payload);
  void handle_register_ack(const RpcHeader& header);
  void enqueue_rpc(RpcOutbound* outbound);
  void flush_outbound_rpc();
  void post_initial_rpc_receives();
  void post_rpc_receive(u32 peer_client);
  QueuePair& qp_for_client(u32 client_id);
  u32 choose_destination(const vec<element_t>& query) const;
  void refresh_routing_state(bool wait_for_remote_registration);
  void shutdown_remote_if_requested();
  void force_publish_graph_epoch();
  void note_graph_insertions(size_t inserted);

  auto& compute_threads() { return worker_pool_->get_compute_threads(); }
  const auto& compute_threads() const { return worker_pool_->get_compute_threads(); }

private:
  Configuration config_;
  Context context_;
  ClientConnectionManager cm_;
  const u32 num_servers_;
  const bool shutdown_remote_on_stop_;

  MemoryRegionTokens remote_access_tokens_;
  Assignment core_assignment_;

  std::atomic<bool> shutdown_{false};
  std::atomic<size_t> vectors_inserted_{0};
  std::atomic<u64> graph_epoch_{1};
  std::mutex neighbor_cache_epoch_mutex_;
  size_t pending_neighbor_cache_inserts_{0};
  std::chrono::steady_clock::time_point last_neighbor_cache_epoch_publish_{std::chrono::steady_clock::now()};

  std::mutex mn_command_mutex_;
  std::atomic<bool> workers_paused_{false};
  std::atomic<u32> workers_idle_count_{0};
  std::atomic<bool> stopped_{false};
  std::atomic<bool> rpc_shutdown_{false};
  std::atomic<bool> rpc_paused_{false};
  std::atomic<bool> rpc_idle_{false};

  std::unique_ptr<vamana::Vamana<Distance>> vamana_;
  std::unique_ptr<WorkerPool> worker_pool_;
  ServiceProfile service_profile_{};
  bool rabitq_artifacts_ready_{false};
  service::InsertQueue insert_queue_;
  service::QueryQueue query_queue_;
  vec<std::thread> workers_;
  std::thread rpc_thread_;
  std::thread storage_insert_completion_thread_;
  std::atomic<bool> storage_insert_shutdown_{false};
  std::atomic<bool> storage_insert_senders_done_{false};
  std::atomic<u32> storage_insert_inflight_{0};
  std::atomic<u32> storage_insert_timeout_logs_{0};
  vec<std::unique_ptr<StorageOwnerSenderState>> storage_insert_owners_;

  std::unique_ptr<byte_t[]> rpc_buffer_;
  std::unique_ptr<LocalMemoryRegion> rpc_region_;
  vec<idx_t> rpc_freelist_;
  concurrent_queue<RpcOutbound*> outbound_rpc_queue_;

  std::mutex pending_mutex_;
  std::unordered_map<u64, std::shared_ptr<std::promise<vec<node_t>>>> pending_queries_;
  std::unordered_map<u64, std::shared_ptr<std::promise<void>>> pending_registration_acks_;
  std::atomic<u64> next_request_id_{1};

  mutable std::mutex routing_mutex_;
  vec<vec<element_t>> routing_centroids_;
  vec<u32> routing_inflight_;
  std::atomic<u32> registered_remote_clients_{0};
  std::condition_variable routing_cv_;

  mutable std::mutex breakdown_mutex_;
  bool breakdown_enabled_{false};
  std::vector<service::breakdown::Sample> completed_query_samples_;
  std::vector<service::breakdown::Sample> completed_insert_samples_;
};

extern template class ComputeService<L2Distance>;
extern template class ComputeService<IPDistance>;
