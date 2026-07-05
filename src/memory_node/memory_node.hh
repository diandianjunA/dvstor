#pragma once

#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
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
#include "memory_node/command_protocol.hh"
#include "memory_node/storage_owner_state.hh"
#include "service/index_metadata.hh"
#include "service/storage_owner_protocol.hh"
#include "vamana/vamana_node.hh"

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

  struct PeerReverseUpdateTask {
    u32 source_shard{};
    service::storage_owner::PeerRpcHeader header{};
    vec<service::storage_owner::ReverseUpdateOp> ops;
    std::chrono::steady_clock::time_point received_at{};
  };

  struct PeerReverseUpdateResponse {
    u32 destination_shard{};
    service::storage_owner::PeerRpcHeader header{};
    std::chrono::steady_clock::time_point queued_at{};
  };

  struct PeerReverseOutgoingTask {
    u32 target_shard{};
    service::storage_owner::PeerRpcType rpc_type{
      service::storage_owner::PeerRpcType::reverse_update_request};
    vec<service::storage_owner::ReverseUpdateOp> ops;
    std::chrono::steady_clock::time_point queued_at{};
  };

  enum class StorageOwnerMaintenanceKind : u8 {
    finalize_insert,
    cleanup_deleted_node,
  };

  struct StorageOwnerMaintenanceTask {
    StorageOwnerMaintenanceKind kind{StorageOwnerMaintenanceKind::finalize_insert};
    node_t id{};
    u32 generation{};
    RemotePtr target;
    std::chrono::steady_clock::time_point queued_at{};
  };

public:
  explicit MemoryNode(Configuration& config);

private:
  using InsertBreakdownCounters = service::storage_owner::InsertBreakdownCounters;
  using BeamEntry = memory_node_detail::BeamEntry;
  using NodeSnapshot = memory_node_detail::NodeSnapshot;
  using InsertRuntimeState = memory_node_detail::InsertRuntimeState;
  using PeerRpcRuntimeState = memory_node_detail::PeerRpcRuntimeState;
  using PeerPendingSend = memory_node_detail::PeerPendingSend;
  using PeerRpcMessage = memory_node_detail::PeerRpcMessage;
  using StorageOwnerInsertTask = memory_node_detail::StorageOwnerInsertTask;
  using StorageOwnerThread = memory_node_detail::StorageOwnerThread;
  using StorageOwnerInsertJob = memory_node_detail::StorageOwnerInsertJob;
  using FreshnessEntry = memory_node_detail::FreshnessEntry;

  static constexpr u32 kPeerSyncWrOwner = std::numeric_limits<u32>::max();
  static constexpr u32 kPeerAsyncWrOwner = std::numeric_limits<u32>::max() - 1;
  static constexpr u32 kPeerSafeRdAtomic = 8;
  static constexpr u32 kPeerRpcFlagNoResponse = 1u;

  // Lifecycle and commands
  static u64 elapsed_ns_since(const std::chrono::steady_clock::time_point start);
  static u64 scale_ns(const u64 value, const u32 part, const u32 total);
  static double storage_owner_candidate_overlap(const vec<RemotePtr>& lhs,
                                                const vec<RemotePtr>& rhs,
                                                u32 limit);
  static InsertBreakdownCounters scale_breakdown(const InsertBreakdownCounters& counters,
                                                 const u32 part,
                                                 const u32 total);
  void allocate_memory();
  bool handle_command();
  std::pair<bool, str> load_index_file(const str& path);
  std::pair<bool, str> store_index_file(const str& path);

  // Peer RDMA transport
  void setup_storage_peers(Configuration& config);
  QP& peer_control_qp(u32 shard_id);
  u32 peer_data_qp_index(u32 worker_id) const;
  QP& peer_data_qp(u32 shard_id, u32 qp_idx);
  static u64 peer_coroutine_wr_id(u32 thread_id, u32 coroutine_id);
  u32 peer_rdma_read_credit_limit_per_qp() const;
  u32 peer_rdma_read_credit_limit() const;
  u32 peer_rdma_read_global_credit_limit() const;
  bool try_acquire_counter(std::atomic<u32>& counter, u32 limit);
  bool try_acquire_peer_rdma_read_credit(u32 shard_id, u32 qp_idx);
  void acquire_peer_rdma_read_credit(u32 shard_id, u32 qp_idx);
  u64 next_peer_sync_wr_id();
  u64 next_peer_async_wr_id();
  void register_peer_pending_send_locked(u64 wr_id, PeerPendingSend pending);
  void handle_peer_send_completion(u64 wr_id);
  void poll_peer_send_cq();
  bool consume_peer_sync_completion(u64 wr_id);
  void wait_peer_sync_completion(u64 wr_id);
  void post_peer_read_async(StorageOwnerThread& thread,
                            u32 shard_id,
                            u64 remote_offset,
                            byte_t* dst,
                            size_t bytes,
                            size_t local_offset = 0);
  void remote_read_bytes(u32 shard_id, u64 remote_offset, void* dst, size_t bytes, size_t scratch_offset);
  void remote_write_bytes(u32 shard_id, u64 remote_offset, const void* src, size_t bytes, size_t scratch_offset);
  u64 remote_compare_and_swap(u32 shard_id, u64 remote_offset, u64 expected, u64 desired, size_t scratch_offset);
  std::pair<bool, u64> try_lock_remote_header(RemotePtr rptr);

  // Peer reverse-update RPC
  void setup_peer_rpc_runtime(const Configuration& config);
  void start_peer_reverse_update_runtime(const Configuration& config);
  void stop_peer_reverse_update_runtime();
  size_t peer_rpc_sync_send_offset(u32 peer_id) const;
  size_t peer_rpc_async_send_offset(u32 peer_id, u32 slot_id) const;
  size_t peer_rpc_receive_offset(u32 peer_id, u32 slot_id) const;
  void repost_peer_rpc_receive(u32 peer_id, u32 slot_id);
  void send_peer_rpc_message(u32 peer_id, const void* payload, size_t bytes);
  service::storage_owner::PeerRpcHeader make_peer_reverse_update_response(
      const service::storage_owner::PeerRpcHeader& request,
      bool success) const;
  bool apply_peer_reverse_update_task(const PeerReverseUpdateTask& task, const Configuration& config);
  bool apply_peer_reverse_update_tasks(const vec<PeerReverseUpdateTask>& tasks, const Configuration& config);
  void send_peer_reverse_update_response(const PeerReverseUpdateResponse& response);
  bool handle_peer_reverse_update_request(u32 source_shard,
                                          const service::storage_owner::PeerRpcHeader& header,
                                          const service::storage_owner::ReverseUpdateOp* ops,
                                          const Configuration& config);
  bool handle_peer_cleanup_deleted_request(u32 source_shard,
                                           const service::storage_owner::PeerRpcHeader& header,
                                           const service::storage_owner::ReverseUpdateOp* ops,
                                           const Configuration& config);
  bool handle_peer_rpc_request(const PeerRpcMessage& message, const Configuration& config);
  bool enqueue_peer_reverse_update_task(PeerReverseUpdateTask&& task);
  void enqueue_peer_reverse_update_response(u32 destination_shard,
                                            const service::storage_owner::PeerRpcHeader& request,
                                            bool success);
  void peer_rpc_progress_loop();
  void peer_reverse_update_worker_loop(u32 worker_id);
  void peer_reverse_response_loop();
  void peer_reverse_outgoing_loop();
  bool handle_peer_rpc_requests(vec<PeerRpcMessage>& requests, const Configuration& config);
  bool pump_peer_rpcs_locked(const Configuration&,
                             vec<PeerRpcMessage>& requests,
                             bool wait_for_event = false);
  bool pump_peer_rpcs(const Configuration& config, bool wait_for_event = false);
  bool wait_for_peer_reverse_update_response(u64 request_id,
                                             u32 target_shard,
                                             u32 item_count,
                                             const Configuration& config);
  bool enqueue_reverse_update_batch(u32 target_shard,
                                    const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                    const Configuration& config);
  bool enqueue_cleanup_deleted_batch(u32 target_shard,
                                     const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                     const Configuration& config);
  bool send_peer_op_batch_direct(u32 target_shard,
                                 const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                 service::storage_owner::PeerRpcType rpc_type,
                                 bool wait_for_response,
                                 const Configuration& config);
  bool send_reverse_update_batch_direct(u32 target_shard,
                                        const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                        bool wait_for_response,
                                        const Configuration& config);
  bool send_reverse_update_batch(u32 target_shard,
                                 const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                 const Configuration& config);
  bool send_cleanup_deleted_batch(u32 target_shard,
                                  const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                  const Configuration& config);
  void log_slow_peer_reverse_update_response(std::chrono::steady_clock::time_point wait_started,
                                             u64 request_id,
                                             u32 target_shard,
                                             u32 item_count,
                                             bool success) const;

  // Storage-owner background graph maintenance
  static bool storage_owner_maintenance_enabled(const Configuration& config);
  void start_storage_owner_maintenance_runtime(const Configuration& config);
  void stop_storage_owner_maintenance_runtime();
  bool enqueue_storage_owner_maintenance(StorageOwnerMaintenanceTask&& task, const Configuration& config);
  bool enqueue_insert_finalization(node_t id,
                                   u32 generation,
                                   RemotePtr target,
                                   const Configuration& config);
  bool enqueue_deleted_node_cleanup(RemotePtr deleted_ptr, const Configuration& config);
  void storage_owner_maintenance_worker_loop(u32 worker_id);
  bool storage_owner_maintenance_foreground_busy(const Configuration& config);
  bool try_acquire_storage_owner_maintenance_slot(const Configuration& config);
  bool try_lock_node(RemotePtr rptr);
  bool storage_owner_task_current(node_t id, u32 generation, RemotePtr target);
  vec<RemotePtr> read_preserved_neighbor_list(RemotePtr rptr);
  bool remove_local_neighbor(RemotePtr target_ptr, RemotePtr deleted_ptr, const Configuration& config);
  bool finalize_inserted_storage_owner_node(const StorageOwnerMaintenanceTask& task,
                                            const Configuration& config);
  bool cleanup_deleted_storage_owner_node(const StorageOwnerMaintenanceTask& task,
                                          const Configuration& config);

  // Storage-owner RPC runtime
  void setup_insert_runtime(const Configuration& config);
  void start_storage_owner_insert_workers(const Configuration& config);
  void storage_owner_insert_worker_loop(u32 worker_id);
  void process_storage_owner_insert_tasks(const vec<StorageOwnerInsertTask>& tasks);
  bool execute_storage_owner_batch_items_async(const node_t* ids,
                                               const service::storage_owner::MutationKind* kinds,
                                               const element_t* vectors,
                                               const u64* anchor_hints,
                                               u32 anchor_hint_count,
                                               size_t item_count,
                                               StorageOwnerThread& thread,
                                               InsertBreakdownCounters& breakdown,
                                               const Configuration& config,
                                               vec<u64>* invalidated_neighbors = nullptr,
                                               vec<u32>* statuses = nullptr,
                                               vec<service::storage_owner::MutationResult>* results = nullptr);
  static StorageOwnerInsertCoroutine dummy_storage_owner_insert_coroutine();
  size_t insert_request_slot_offset(u32 client_id, u32 slot_id) const;
  size_t insert_response_slot_offset(const Configuration& config, u32 client_id, u32 slot_id) const;
  void service_storage_runtime(const Configuration& config);
  size_t response_slot_bytes(const Configuration& config) const;
  size_t handle_storage_insert_request(u32 client_id, const byte_t* payload, size_t bytes, const Configuration& config);
  bool execute_storage_owner_batch_items(const node_t* ids,
                                         const service::storage_owner::MutationKind* kinds,
                                         const element_t* vectors,
                                         const u64* anchor_hints,
                                         u32 anchor_hint_count,
                                         size_t item_count,
                                         InsertBreakdownCounters& breakdown,
                                         const Configuration& config,
                                         vec<u64>* invalidated_neighbors = nullptr,
                                         vec<u32>* statuses = nullptr,
                                         vec<service::storage_owner::MutationResult>* results = nullptr);

  // Storage-owner index operations
  RemotePtr allocate_local_node();
  bool load_owner_idmap(const filepath_t& index_prefix);
  bool mark_node_deleted(RemotePtr rptr, u32 generation);
  service::storage_owner::MutationStatus prepare_mutation(node_t id,
                                                          service::storage_owner::MutationKind kind,
                                                          FreshnessEntry* old_entry,
                                                          u32* new_generation);
  void publish_mutation(node_t id, RemotePtr ptr, u32 generation, bool deleted);
  RemotePtr read_global_medoid();
  auto async_read_global_medoid(StorageOwnerThread& thread);
  void write_global_medoid(const RemotePtr& medoid);
  bool try_set_global_medoid(const RemotePtr& expected, const RemotePtr& desired, RemotePtr& observed);
  bool read_node_snapshot(RemotePtr rptr, NodeSnapshot& snapshot);
  vec<RemotePtr> read_neighbor_list_aos(RemotePtr rptr);
  vec<RemotePtr> read_neighbor_list(RemotePtr rptr);
  auto async_read_node_snapshot(RemotePtr rptr, StorageOwnerThread& thread);
  auto async_read_node_snapshots(const vec<RemotePtr>& rptrs,
                                 const Configuration& config,
                                 StorageOwnerThread& thread);
  vec<NodeSnapshot> read_node_snapshots_batched(const vec<RemotePtr>& rptrs, const Configuration& config);
  auto async_read_neighbor_list(RemotePtr rptr, StorageOwnerThread& thread);
  void write_hot_graph_entry(RemotePtr rptr, u32 id, const vec<RemotePtr>& neighbors);
  void write_neighbor_list(RemotePtr rptr, const vec<RemotePtr>& neighbors);
  void write_new_node(RemotePtr rptr,
                      node_t id,
                      const span<const element_t> components,
                      const vec<RemotePtr>& neighbors,
                      u32 generation = 0);
  void lock_node(RemotePtr rptr);
  void unlock_node(RemotePtr rptr);
  vec<RemotePtr> beam_search_candidates(const span<const element_t> query,
                                        RemotePtr medoid,
                                        const Configuration& config,
                                        InsertBreakdownCounters* breakdown = nullptr);

  auto beam_search_candidates_async(const span<const element_t> query,
                                    RemotePtr medoid,
                                    const Configuration& config,
                                    StorageOwnerThread& thread,
                                    InsertBreakdownCounters* breakdown = nullptr) -> StorageOwnerInsertCoroutine;
  vec<RemotePtr> anchor_search_candidates(const span<const element_t> query,
                                          const vec<RemotePtr>& anchor_hints,
                                          const Configuration& config,
                                          InsertBreakdownCounters* breakdown = nullptr);
  auto anchor_search_candidates_async(const span<const element_t> query,
                                      const vec<RemotePtr>& anchor_hints,
                                      const Configuration& config,
                                      StorageOwnerThread& thread,
                                      InsertBreakdownCounters* breakdown = nullptr)
    -> StorageOwnerInsertCoroutine;
  vec<RemotePtr> robust_prune_cpu(const byte_t* source,
                                  VectorDType source_dtype,
                                  const vec<RemotePtr>& candidates,
                                  const hashset_t<RemotePtr>& skip,
                                  const Configuration& config,
                                  InsertBreakdownCounters* breakdown = nullptr,
                                  u32 candidate_limit_override = 0);
  auto execute_storage_owner_insert_job_async(StorageOwnerThread& thread,
                                              StorageOwnerInsertJob& job,
                                              std::unordered_map<u64, vec<RemotePtr>>& local_updates,
                                              std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>>& remote_updates,
                                              InsertBreakdownCounters& breakdown,
                                              const Configuration& config) -> StorageOwnerInsertCoroutine;
  bool apply_local_reverse_update(RemotePtr target_ptr,
                                  const vec<RemotePtr>& candidate_ptrs,
                                  const Configuration& config,
                                  bool enqueue_maintenance = true);

  // Misc helpers
  static size_t align_up(size_t value, size_t alignment = CACHELINE_SIZE);
  distance_t distance_to_stored_vector(const span<const element_t> query,
                                        const byte_t* stored,
                                        const Configuration& config) const;
  distance_t distance_between_vectors(const byte_t* lhs,
                                      VectorDType lhs_dtype,
                                      const byte_t* rhs,
                                      VectorDType rhs_dtype,
                                      const Configuration& config) const;
  bool local_shard(u32 shard_id) const;
  byte_t* local_node_ptr(const RemotePtr& rptr);
  const byte_t* local_node_ptr(const RemotePtr& rptr) const;
  static void insert_into_beam(vec<BeamEntry>& beam, const RemotePtr& rptr, distance_t dist, u32 max_beam_width);
  void route_queries(i32 max_cqes);
  void idle();
private:
  Context context_;
  ServerConnectionManager cm_;
  Assignment core_assignment_;

  const u32 num_clients_;
  u32 num_compute_threads_{};
  u32 qp_pool_size_{1};
  const u32 storage_id_;
  const u32 num_storage_nodes_;
  const bool use_storage_owner_insert_;
  const u32 storage_owner_peer_rdma_tokens_;
  const bool ip_distance_;

  HugePage<byte_t> index_buffer_;
  MemoryRegion index_region_;
  std::unique_ptr<configuration::Configuration> peer_config_;
  std::unique_ptr<Context> peer_context_;
  vec<QPs> peer_qps_;
  u32 peer_qps_per_peer_{1};
  MemoryRegionTokens peer_remote_tokens_;
  std::unique_ptr<MemoryRegion> peer_index_region_;
  HugePage<byte_t> peer_scratch_buffer_;
  std::unique_ptr<LocalMemoryRegion> peer_scratch_region_;
  PeerRpcRuntimeState peer_rpc_runtime_;
  std::unordered_map<u64, service::storage_owner::PeerRpcHeader> peer_rpc_responses_;
  std::unordered_map<u64, vec<byte_t>> peer_rpc_response_payloads_;
  std::mutex peer_rpc_mutex_;
  std::condition_variable peer_rpc_responses_cv_;
  std::mutex peer_rpc_send_mutex_;
  std::mutex peer_send_cq_mutex_;
  std::mutex peer_completion_mutex_;
  vec<ibv_wc> peer_send_wcs_;
  std::unordered_set<u64> peer_sync_completions_;
  std::unordered_map<u64, PeerPendingSend> peer_pending_sends_;
  vec<std::atomic<u32>> peer_rdma_read_outstanding_;
  std::atomic<u64> storage_owner_anchor_insert_sequence_{0};
  vec<vec<std::atomic<u32>>> peer_rdma_read_qp_outstanding_;
  vec<vec<std::unique_ptr<std::mutex>>> peer_qp_send_mutexes_;
  std::atomic<u32> peer_sync_wr_id_counter_{1};
  std::atomic<u32> peer_async_wr_id_counter_{1};
  std::atomic<u32> peer_async_rdma_outstanding_{0};
  std::atomic<u64> next_peer_request_id_{1};
  std::thread peer_rpc_progress_thread_;
  vec<std::thread> peer_reverse_workers_;
  std::thread peer_reverse_response_thread_;
  std::thread peer_reverse_outgoing_thread_;
  vec<u_ptr<StorageOwnerThread>> peer_reverse_worker_states_;
  std::mutex peer_reverse_tasks_mutex_;
  std::condition_variable peer_reverse_tasks_cv_;
  std::deque<PeerReverseUpdateTask> peer_reverse_tasks_;
  std::mutex peer_reverse_responses_mutex_;
  std::condition_variable peer_reverse_responses_cv_;
  std::deque<PeerReverseUpdateResponse> peer_reverse_responses_;
  std::mutex peer_reverse_outgoing_mutex_;
  std::condition_variable peer_reverse_outgoing_cv_;
  std::deque<PeerReverseOutgoingTask> peer_reverse_outgoing_;
  std::atomic<bool> peer_reverse_shutdown_{false};
  std::atomic<bool> peer_reverse_workers_done_{false};
  size_t peer_reverse_task_queue_limit_{1024};
  size_t peer_reverse_outgoing_queue_limit_{1024};
  vec<std::thread> storage_owner_maintenance_workers_;
  vec<u_ptr<StorageOwnerThread>> storage_owner_maintenance_worker_states_;
  std::mutex storage_owner_maintenance_mutex_;
  std::condition_variable storage_owner_maintenance_cv_;
  std::deque<StorageOwnerMaintenanceTask> storage_owner_maintenance_tasks_;
  std::atomic<bool> storage_owner_maintenance_shutdown_{false};
  std::atomic<u64> storage_owner_maintenance_enqueued_{0};
  std::atomic<u64> storage_owner_maintenance_processed_{0};
  std::atomic<u64> storage_owner_maintenance_failed_{0};
  std::atomic<u64> storage_owner_maintenance_stale_{0};
  std::atomic<u64> storage_owner_maintenance_cleanup_processed_{0};
  std::atomic<u64> storage_owner_maintenance_max_backlog_{0};
  std::atomic<u64> storage_owner_maintenance_pressure_yields_{0};
  std::atomic<u32> storage_owner_maintenance_active_workers_{0};
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
  std::atomic<u32> storage_owner_insert_active_workers_{0};
  const u64 mn_memory_bytes_;
  timing::Timing timing_;
  filepath_t index_prefix_;
  bool owner_idmap_required_{false};
  std::mutex idmap_mutex_;
  std::unordered_map<node_t, FreshnessEntry> idmap_;
  std::unordered_set<node_t> mutations_inflight_;

  inline static thread_local StorageOwnerThread* current_storage_owner_thread_{nullptr};
};
