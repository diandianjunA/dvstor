#pragma once

#include <atomic>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
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
#include "common/bounded_queue.hh"
#include "common/sliding_completion_ring.hh"
#include "common/constants.hh"
#include "common/core_assignment.hh"
#include "common/distance.hh"
#include "common/timing.hh"
#include "gpu_search/pq_index.hh"
#include "memory_node/peer_rpc/async_response.hh"
#include "memory_node/startup_protocol.hh"
#include "memory_node/storage_reclaim.hh"
#include "memory_node/storage_owner_cpu_plan.hh"
#include "memory_node/storage_owner_index/dynamic_allocation_receipt_policy.hh"
#include "memory_node/storage_owner_index/incarnation_lock.hh"
#include "memory_node/storage_owner_maintenance/home_rpc_outbox.hh"
#include "memory_node/storage_owner_maintenance/ready_context_queue.hh"
#include "memory_node/storage_owner_maintenance/reverse_outbox.hh"
#include "memory_node/storage_owner_maintenance/stage2_batch_policy.hh"
#include "memory_node/storage_owner_state.hh"
#include "memory_node/peer_rdma_credit_policy.hh"
#include "service/index_metadata.hh"
#include "service/storage_owner_protocol.hh"
#include "vamana/centroid_router.hh"
#include "vamana/vamana_node.hh"

namespace memory_node_storage_owner_maintenance_detail {
enum class Stage2SearchAdvanceResult : std::uint8_t;
struct Stage2SearchIoState;
}  // namespace memory_node_storage_owner_maintenance_detail

namespace memory_node_storage_owner_index_detail {
enum class StableNodeSnapshotState : u8;
}  // namespace memory_node_storage_owner_index_detail

/**
 * Owns one static vector/compact-graph shard, its PQ navigation-code stream,
 * and the mutable storage-owner region used by online updates. Compute nodes
 * access these regions directly; storage workers only execute update and
 * maintenance protocols.
 */
class MemoryNode {
  using Configuration = configuration::IndexConfiguration;
  using Assignment = CoreAssignment<interleaved>;

  struct PeerReverseUpdateTask {
    u32 source_shard{};
    service::storage_owner::PeerRpcHeader header{};
    memory_node_detail::PeerRequestLease dedup_lease{};
    vec<service::storage_owner::ReverseUpdateOp> ops;
    vec<service::storage_owner::ReconcileReverseOp> reconcile_ops;
    vec<service::storage_owner::CentroidMembershipOp> centroid_ops;
    std::chrono::steady_clock::time_point received_at{};

    size_t item_count() const {
      if (!centroid_ops.empty()) return centroid_ops.size();
      return reconcile_ops.empty() ? ops.size() : reconcile_ops.size();
    }
  };

  struct PeerReverseUpdateResponse {
    u32 destination_shard{};
    service::storage_owner::PeerRpcHeader header{};
    // Empty for the fixed-header reverse-update ACK. Variable-size home
    // dependency and reconciliation responses retain their immutable bytes
    // here until an asynchronous registered send slot is available.
    vec<byte_t> payload;
    bool graph_response{};
    std::chrono::steady_clock::time_point queued_at{};
  };

  struct PeerStage1Task {
    u32 source_shard{};
    // Monotonic within source_shard and assigned in RC receive order. It is
    // retained for completion telemetry; receipt safety is keyed by the
    // semantic operation token, so an unrelated slow request cannot block a
    // release.
    u64 source_sequence{};
    service::storage_owner::PeerRpcHeader header{};
    memory_node_detail::PeerRequestLease dedup_lease{};
    vec<byte_t> payload;
    // Execute/arm/abort tokens tracked by the bounded per-operation in-flight
    // table. Release tokens are deliberately excluded: a release probes this
    // table and returns an explicit retry while an older same-token operation
    // remains, rather than occupying a worker while it waits.
    vec<service::storage_owner::AuthorityOperationToken> operation_tokens;
    std::chrono::steady_clock::time_point received_at{};
    // A credit-parked request must not leave a permanent diagnostic ordering
    // hole while unrelated same-source requests complete. Semantic ordering
    // remains fenced by operation_tokens/inflight; this bit prevents the
    // source completion sequence from being recorded twice after wakeup.
    bool source_sequence_completed{};
    // True from the first capacity park until this request either arms or
    // returns an explicit retry. It keeps the semantic continuation item
    // bound intact while the task moves waiter -> runnable -> active -> waiter.
    bool admission_waiter_owned{};
    // Scheduler-only coverage for completion/queue credit that caused a
    // parked request to become runnable. This is not semantic debt and is
    // released after the arm attempt; it merely prevents a completion burst
    // from waking many whole RPCs for the same small visible credit window.
    u32 admission_wake_coverage{};
  };

  struct PeerOrderedCompletionState {
    std::mutex mutex;
    std::condition_variable changed;
    u64 completed_prefix{};
    std::unordered_set<u64> completed_out_of_order;
  };

  struct PeerPhysicalControlTask {
    u32 source_shard{};
    // Cleanup activate/release requests share the same per-source ordering
    // discipline as Stage1. Placement requests leave this field zero because
    // they do not erase a semantic receipt owned by an earlier request.
    u64 source_sequence{};
    service::storage_owner::PeerRpcHeader header{};
    memory_node_detail::PeerRequestLease dedup_lease{};
    vec<byte_t> payload;
    std::chrono::steady_clock::time_point received_at{};
  };

  struct Stage1OperationKey {
    u32 authority_shard{};
    u32 source_client{};
    u32 item_index{};
    u64 client_batch_id{};

    bool operator==(const Stage1OperationKey&) const = default;
  };

  struct Stage1OperationKeyHash {
    size_t operator()(const Stage1OperationKey& key) const {
      size_t value = std::hash<u64>{}(key.client_batch_id);
      value ^= std::hash<u64>{}(
        (static_cast<u64>(key.authority_shard) << 32) |
        key.source_client) + 0x9e3779b97f4a7c15ull +
        (value << 6) + (value >> 2);
      value ^= std::hash<u32>{}(key.item_index) +
        0x9e3779b97f4a7c15ull + (value << 6) + (value >> 2);
      return value;
    }
  };

  struct Stage1PreparedResult {
    service::storage_owner::Stage1ExecuteResult result{};
    u64 maintenance_sequence{};
    node_t id{};
    u32 generation{};
    service::storage_owner::MutationKind kind{
      service::storage_owner::MutationKind::insert};
    RemotePtr old_ptr;
    vec<byte_t> vector_data;
    vec<RemotePtr> neighbors;
    vec<memory_node_detail::BeamEntry> beam;
    vec<RemotePtr> remote_frontier;
    vec<RemotePtr> backlink_targets;
    u64 execute_initial_placement_version{};
    u64 initial_placement_version{};
    bool prepared{};
    bool arming{};
    bool armed{};
    bool aborted{};
  };

  static constexpr size_t kStage1PreparedShardCount = 64;
  static_assert((kStage1PreparedShardCount &
                 (kStage1PreparedShardCount - 1)) == 0);

  struct Stage1PreparedResultShard {
    std::mutex mutex;
    std::unordered_map<Stage1OperationKey, Stage1PreparedResult,
                       Stage1OperationKeyHash> records;
  };

  struct Stage1InflightRequestShard {
    std::mutex mutex;
    std::condition_variable changed;
    std::unordered_map<Stage1OperationKey, u32,
                       Stage1OperationKeyHash> counts;
  };

  struct CleanupActivationRecord {
    service::storage_owner::CleanupActivateItem item{};
    service::storage_owner::CleanupActivateResult result{};
    bool in_progress{};
  };

  static constexpr size_t kCleanupActivationShardCount = 64;
  static_assert((kCleanupActivationShardCount &
                 (kCleanupActivationShardCount - 1)) == 0);

  struct CleanupActivationDedupeShard {
    std::mutex mutex;
    std::condition_variable changed;
    std::unordered_map<Stage1OperationKey, CleanupActivationRecord,
                       Stage1OperationKeyHash> records;
  };

  enum class StorageOwnerMaintenanceKind : u8 {
    finalize_insert,
    cleanup_deleted_node,
  };

  struct StorageOwnerMaintenanceTask {
    StorageOwnerMaintenanceKind kind{StorageOwnerMaintenanceKind::finalize_insert};
    node_t id{};
    u32 generation{};
    u64 maintenance_sequence{};
    RemotePtr target;
    RemotePtr final_target;
    u32 final_home{};
    u32 stage2_revalidated_home{};
    u32 authority_shard{};
    u32 source_client{};
    u32 operation_item_index{};
    u64 operation_batch_id{};
    u64 initial_placement_version{};
    bool outgoing_committed{};
    bool reverse_reconciled{};
    bool placement_committed{};
    bool allocation_settled{};
    bool centroid_committed{};
    // The authority hands receipt lifetime off with an ordered same-QP release
    // fence after commit. Stage2 records whether that responsibility has been
    // resolved; it never infers a remote transport watermark locally.
    bool stage1_receipt_released{};
    bool stage2_prepared{};
    bool stage2_source_frozen{};
    // The physical home becomes immutable once its complete outgoing record
    // is published. Parent churn may re-prune that record in place, but never
    // creates a second ambiguous migration receipt for the same operation.
    bool stage2_plan_sealed{};
    // Persist the last ACKed incoming stable certificate across ordinary-edge
    // failures. Every retry revalidates it before any temporary bridge removal.
    bool stage2_promotion_committed{};
    RemotePtr stage2_promotion_parent;
    // Exact fixed-width Stage1 beam and every unique cross-partition pointer
    // observed while expanding it. Stage2 resumes from these structures and
    // never restarts a search from another shard's representative.
    vec<memory_node_detail::BeamEntry> stage1_beam;
    vec<RemotePtr> stage1_remote_frontier;
    // Targets that actually acknowledged a provisional backlink. Stage2 must
    // remove or replace exactly this set; attempted-but-full targets are not
    // part of the handoff transaction.
    vec<RemotePtr> stage1_backlink_targets;
    // Exact temporary adjacency written by stage1. Final commit compares the
    // then-current adjacency against this captured baseline, so reverse edges
    // added while stage2 was queued/in flight are rebased instead of lost.
    vec<RemotePtr> stage1_base_neighbors;
    vec<RemotePtr> stage2_neighbors;
    // Protected children captured at the same locked boundary that freezes
    // the source graph. They are preserved at an in-place destination and
    // transferred verbatim to a migrated destination before it is published.
    vec<RemotePtr> stage2_protected_children;
    // A Stage2 finalization that becomes stale after applying reverse edges
    // transfers its maintenance sequence to a cleanup task. These supplemental neighbors
    // must all be removed; pruning them would leave dangling reverse edges.
    bool cleanup_repair_only{};
    // Cleanup is activated before the successor authority commit. No graph or
    // centroid mutation may touch the old generation until the authority
    // reports that a strictly newer logical generation has retired it.
    bool cleanup_authority_retired{};
    // Ordinary deletion first quiesces the still query-visible parent, then
    // hands every live protected child to an ACKed replacement parent before
    // publishing DELETED. The parallel replacement vector makes partial RPC
    // success idempotent across cleanup-worker retries.
    bool cleanup_retiring{};
    bool cleanup_protected_reparented{};
    vec<RemotePtr> cleanup_protected_children;
    vec<RemotePtr> cleanup_replacement_parents;
    vec<RemotePtr> cleanup_neighbors;
    std::chrono::steady_clock::time_point queued_at{};
    std::chrono::steady_clock::time_point retry_not_before{};
  };

  struct StorageOwnerMaintenanceIntent {
    std::atomic<u64> sequence{0};
    node_t id{};
    u32 generation{};
    service::storage_owner::MutationKind kind{
      service::storage_owner::MutationKind::insert};
    RemotePtr new_ptr;
    RemotePtr old_ptr;
    std::chrono::steady_clock::time_point published_at{};
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
  using PeerAsyncResponseRegistry =
    memory_node_detail::PeerAsyncResponseRegistry;
  using PeerRequestDeduplicator =
    memory_node_detail::PeerRequestDeduplicator;
  using TryPeerResponse = memory_node_detail::TryPeerResponse;
  using PeerResponseLease = memory_node_detail::PeerResponseLease;
  using PeerResponseCompletionTarget =
    memory_node_detail::PeerResponseCompletionTarget;
  using Stage2HomeRpcOutbox =
    memory_node_storage_owner_maintenance_detail::Stage2HomeRpcOutbox;
  using Stage2HomeRpcEnqueueResult =
    memory_node_storage_owner_maintenance_detail::
      Stage2HomeRpcEnqueueResult;
  using Stage2ReverseOutbox =
    memory_node_storage_owner_maintenance_detail::Stage2ReverseOutbox;
  using Stage2ReverseCompletion =
    memory_node_storage_owner_maintenance_detail::Stage2ReverseCompletion;
  using StorageOwnerInsertTask = memory_node_detail::StorageOwnerInsertTask;
  using StorageOwnerResponseReady = memory_node_detail::StorageOwnerResponseReady;
  using StorageOwnerThread = memory_node_detail::StorageOwnerThread;
  using FreshnessEntry = memory_node_detail::FreshnessEntry;
  struct PeerReadRequest {
    u32 shard_id{};
    u64 remote_offset{};
    byte_t* destination{};
    size_t bytes{};
    size_t local_offset{};
  };
  struct PeerReadPairRequest {
    PeerReadRequest full_snapshot;
    PeerReadRequest after_header;
  };
  struct PeerReadSnapshotRequest {
    PeerReadRequest full_snapshot;
    // Immutable base records leave this empty. Dynamic records require the
    // ordered after-header read that closes their reuse/torn-body window.
    std::optional<PeerReadRequest> after_header;
  };
  using AuthorityOperationToken =
    memory_node_storage_owner_index_detail::AuthorityOperationToken;
  using AuthorityMutationLease =
    memory_node_storage_owner_index_detail::AuthorityMutationLease;
  using AuthorityDirectoryState =
    memory_node_storage_owner_index_detail::AuthorityDirectoryState;
  using AuthorityBeginResult =
    memory_node_storage_owner_index_detail::AuthorityBeginResult;
  using AuthorityCommitState =
    memory_node_storage_owner_index_detail::AuthorityCommitState;
  using AuthorityAbortState =
    memory_node_storage_owner_index_detail::AuthorityAbortState;
  using AuthorityCheckState =
    memory_node_storage_owner_index_detail::AuthorityCheckState;
  using AuthorityRelocateState =
    memory_node_storage_owner_index_detail::AuthorityRelocateState;

  // Stable Vamana edges and query-visible Stage1 backlinks share one compact
  // RDMA record, but they have deliberately separate capacities and mutation
  // rules.  Keeping the split explicit prevents an ordinary graph rewrite
  // from accidentally promoting transient backlinks into the durable graph.
  struct GraphAdjacency {
    vec<RemotePtr> stable;
    vec<RemotePtr> provisional;
    u32 generation{};
    bool deleted{};
  };

  static constexpr size_t kDynamicFreshnessShardCount = 256;
  static_assert((kDynamicFreshnessShardCount &
                 (kDynamicFreshnessShardCount - 1)) == 0);

  struct DynamicFreshnessShard {
    std::mutex mutex;
    std::condition_variable changed;
    dense_hashmap_t<node_t, FreshnessEntry> entries;
    dense_hashmap_t<node_t, AuthorityMutationLease> mutation_leases;
  };

  DynamicFreshnessShard& dynamic_freshness_shard(node_t id) {
    return dynamic_freshness_shards_[
      std::hash<node_t>{}(id) & (kDynamicFreshnessShardCount - 1)];
  }

  static constexpr u32 kPeerSyncWrOwner = std::numeric_limits<u32>::max();
  static constexpr u32 kPeerAsyncWrOwner = std::numeric_limits<u32>::max() - 1;
  static constexpr u32 kPeerSafeRdAtomic = 8;

  // A returned transport/scratch credit is useful only to executors that
  // actually failed on that resource.  Keep a bounded owner mask beside each
  // shared resource so its release can wake the blocked owners without either
  // losing the edge or selecting an unrelated sleeping worker.  The mask is
  // stable for the MemoryNode lifetime because peer CQ progress intentionally
  // outlives the maintenance worker states during shutdown.
  static constexpr size_t kStorageOwnerMaintenanceWaiterWords =
    (CPU_SETSIZE + 63) / 64;
  using StorageOwnerMaintenanceWaiterMask =
    std::array<u64, kStorageOwnerMaintenanceWaiterWords>;
  struct alignas(64) StorageOwnerMaintenanceWaiterSet {
    StorageOwnerMaintenanceWaiterSet() {
      for (auto& word : words) word.store(0, std::memory_order_relaxed);
      for (auto& ref : refs) ref.store(0, std::memory_order_relaxed);
    }
    StorageOwnerMaintenanceWaiterSet(
      const StorageOwnerMaintenanceWaiterSet&) = delete;
    StorageOwnerMaintenanceWaiterSet& operator=(
      const StorageOwnerMaintenanceWaiterSet&) = delete;

    std::array<std::atomic<u64>, kStorageOwnerMaintenanceWaiterWords> words;
    // A worker bit is only the release-side runnable hint. Multiple contexts
    // owned by that worker can independently wait on the same resource, so
    // retain an exact per-worker subscription count behind the hint. A
    // release may consume the bit; the next failed retry republishes it while
    // unregistering one context preserves it for the remaining subscribers.
    std::array<std::atomic<u32>, CPU_SETSIZE> refs;
    std::atomic<u32> cursor{0};
  };

  using StorageOwnerMaintenanceWaiterRegistrations =
    vec<StorageOwnerMaintenanceWaiterSet*>;

  enum class PeerRpcSendClass : u8 {
    stage1,
    graph_update,
    control,
  };

  // Lifecycle and commands
  static u64 elapsed_ns_since(const std::chrono::steady_clock::time_point start);
  static u64 scale_ns(const u64 value, const u32 part, const u32 total);
  static InsertBreakdownCounters scale_breakdown(const InsertBreakdownCounters& counters,
                                                 const u32 part,
                                                 const u32 total);
  // Publish an event epoch before waking maintenance workers. The executor
  // uses a separate waiter-registered condition variable; general foreground
  // capacity waits retain storage_owner_maintenance_cv_.
  void notify_storage_owner_maintenance();
  void notify_storage_owner_maintenance_capacity();
  void notify_storage_owner_maintenance_executor(u32 worker_id);
  void notify_storage_owner_maintenance_executor_scan(u32 worker_id);
  void notify_storage_owner_maintenance_executors();
  void notify_one_storage_owner_maintenance_executor();
  memory_node_storage_owner_maintenance_detail::Stage2ContextOwnerKey
  current_storage_owner_maintenance_context_owner() const;
  bool enqueue_storage_owner_maintenance_context_ready(
    const memory_node_storage_owner_maintenance_detail::
      Stage2ContextOwnerKey& owner,
    memory_node_storage_owner_maintenance_detail::
      Stage2ContextReadyReason reason);
  bool notify_storage_owner_maintenance_context_ready(
    const memory_node_storage_owner_maintenance_detail::
      Stage2ContextOwnerKey& owner,
    memory_node_storage_owner_maintenance_detail::
      Stage2ContextReadyReason reason);
  bool mark_current_storage_owner_maintenance_waiter(
    StorageOwnerMaintenanceWaiterSet& waiters);
  void clear_current_storage_owner_maintenance_waiter(
    StorageOwnerMaintenanceWaiterSet& waiters);
  void clear_all_current_storage_owner_maintenance_waiters();
  void take_storage_owner_maintenance_waiters(
    StorageOwnerMaintenanceWaiterSet& waiters,
    StorageOwnerMaintenanceWaiterMask& owners);
  std::optional<u32> take_one_storage_owner_maintenance_waiter(
    StorageOwnerMaintenanceWaiterSet& waiters,
    u32 avoid_worker = std::numeric_limits<u32>::max());
  void clear_storage_owner_maintenance_waiter(
    StorageOwnerMaintenanceWaiterSet& waiters, u32 worker_id);
  void notify_storage_owner_maintenance_waiters(
    const StorageOwnerMaintenanceWaiterMask& owners);
  void notify_storage_owner_maintenance_waiters_scan(
    const StorageOwnerMaintenanceWaiterMask& owners);
  void reset_storage_owner_maintenance_waiters(
    StorageOwnerMaintenanceWaiterSet& waiters);
  bool try_acquire_storage_owner_search_lane_lease(
    bool& waiter_registered);
  void cancel_storage_owner_search_lane_waiter(bool& waiter_registered);
  void return_storage_owner_search_lane_grants(u32 worker_id);
  void handoff_or_release_storage_owner_search_lane_lease(u32 avoid_worker);
  void retire_storage_owner_search_lane_grants(u32 worker_id);
  void release_storage_owner_search_lane_lease();
  void allocate_memory();
  void wait_for_start_signal(const Configuration& config);
  std::pair<bool, str> load_index_file(const str& path);

  // Peer RDMA transport
  void setup_storage_peers(Configuration& config);
  QP& peer_control_qp(u32 shard_id);
  u32 peer_data_qp_index(u32 worker_id) const;
  QP& peer_data_qp(u32 shard_id, u32 qp_idx);
  static u64 peer_coroutine_wr_id(u32 thread_id, u32 coroutine_id);
  memory_node_detail::PeerRdmaReadCreditPlan
  derive_peer_rdma_read_credit_plan() const;
  const memory_node_detail::PeerRdmaReadCreditPlan&
  peer_rdma_read_credit_plan() const;
  u32 peer_rdma_read_credit_limit_per_qp() const;
  u32 peer_rdma_read_credit_limit() const;
  u32 peer_rdma_read_global_credit_limit() const;
  bool try_acquire_peer_rdma_read_credit(u32 shard_id, u32 qp_idx);
  void acquire_peer_rdma_read_credit(u32 shard_id, u32 qp_idx);
  bool try_acquire_peer_rdma_read_group(u32 shard_id, u32 qp_idx,
                                        u32 read_count);
  void acquire_peer_rdma_read_group(u32 shard_id, u32 qp_idx,
                                    u32 read_count);
  void mark_current_peer_rdma_read_waiter(u32 shard_id, u32 qp_idx);
  void clear_current_peer_rdma_read_waiter(u32 shard_id, u32 qp_idx);
  void take_peer_rdma_read_waiters(
    u32 shard_id, u32 qp_idx,
    StorageOwnerMaintenanceWaiterMask& owners, u32 wake_budget);
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
  void post_peer_reads_async(StorageOwnerThread& thread,
                             span<const PeerReadRequest> requests);
  bool try_post_peer_reads_async(
    StorageOwnerThread& thread,
    span<const PeerReadRequest> requests);
  bool post_peer_reads_async_impl(
    StorageOwnerThread& thread,
    span<const PeerReadRequest> requests,
    bool try_only);
  void post_peer_read_pairs_async(
    StorageOwnerThread& thread,
    span<const PeerReadPairRequest> requests);
  bool try_post_peer_read_pairs_async(
    StorageOwnerThread& thread,
    span<const PeerReadPairRequest> requests);
  bool post_peer_read_pairs_async_impl(
    StorageOwnerThread& thread,
    span<const PeerReadPairRequest> requests,
    bool try_only);
  bool try_post_peer_snapshot_reads_async(
    StorageOwnerThread& thread,
    span<const PeerReadSnapshotRequest> requests);
  void remote_read_bytes(u32 shard_id, u64 remote_offset, void* dst, size_t bytes, size_t scratch_offset);
  void remote_write_bytes(u32 shard_id, u64 remote_offset, const void* src, size_t bytes, size_t scratch_offset);
  u64 remote_compare_and_swap(u32 shard_id, u64 remote_offset, u64 expected, u64 desired, size_t scratch_offset);
  u64 remote_fetch_add(u32 shard_id,
                       u64 remote_offset,
                       u64 increment,
                       size_t scratch_offset);
  std::pair<bool, u64> try_lock_remote_header(RemotePtr rptr);

  // Peer reverse-update RPC
  void setup_peer_rpc_runtime(const Configuration& config);
  void start_peer_reverse_update_runtime(const Configuration& config);
  void stop_peer_reverse_update_runtime();
  size_t peer_rpc_sync_send_offset(u32 peer_id) const;
  size_t peer_rpc_async_send_offset(u32 peer_id, u32 slot_id) const;
  size_t peer_rpc_receive_offset(u32 peer_id, u32 slot_id) const;
  bool try_acquire_peer_rpc_send_slot(u32 peer_id,
                                      PeerRpcSendClass send_class,
                                      u32& slot_id);
  PeerRpcSendClass peer_rpc_send_slot_class(u32 slot_id) const;
  void release_peer_rpc_send_slot(u32 peer_id, u32 slot_id);
  void repost_peer_rpc_receive(u32 peer_id, u32 slot_id);
  void post_peer_rpc_send_slot(u32 peer_id, u32 slot_id, size_t bytes);
  void send_peer_rpc_message(u32 peer_id, const void* payload, size_t bytes);
  service::storage_owner::PeerRpcHeader make_peer_reverse_update_response(
      const service::storage_owner::PeerRpcHeader& request,
      bool success) const;
  bool apply_peer_reverse_update_tasks(const vec<PeerReverseUpdateTask>& tasks, const Configuration& config);
  void send_peer_reverse_update_response(PeerReverseUpdateResponse& response);
  vec<byte_t> acquire_peer_graph_response_buffer(size_t bytes);
  void recycle_peer_graph_response_buffer(vec<byte_t>&& buffer);
  bool try_enqueue_peer_reverse_update_response(
    PeerReverseUpdateResponse&& response);
  bool handle_peer_stage1_execute_request(
      u32 source_shard,
      const service::storage_owner::PeerRpcHeader& header,
      const byte_t* payload,
      const Configuration& config,
      bool* admission_deferred = nullptr);
  bool handle_peer_stage2_expand_score_request(
      u32 source_shard,
      const service::storage_owner::PeerRpcHeader& header,
      const byte_t* payload,
      const Configuration& config);
  bool handle_peer_stage2_score_many_request(
      u32 source_shard,
      const service::storage_owner::PeerRpcHeader& header,
      const byte_t* payload,
      const Configuration& config);
  bool try_send_peer_stage1_retry_response(
      u32 destination_shard,
      const service::storage_owner::PeerRpcHeader& header,
      span<const byte_t> request);
  service::storage_owner::Stage1ExecuteResult prepare_local_stage1_item(
      u32 authority_shard,
      const service::storage_owner::Stage1ExecuteItem& item,
      const byte_t* raw_vector,
      const Configuration& config,
      InsertBreakdownCounters* breakdown = nullptr);
  service::storage_owner::Stage1ExecuteResult
  prepare_and_maybe_arm_local_stage1_item(
      u32 authority_shard,
      const service::storage_owner::Stage1ExecuteItem& item,
      const byte_t* raw_vector,
      const Configuration& config,
      InsertBreakdownCounters* breakdown = nullptr);
  bool try_track_stage1_inflight_request(const Stage1OperationKey& key);
  void finish_stage1_inflight_request(const Stage1OperationKey& key);
  bool stage1_inflight_quiescent(const Stage1OperationKey& key);
  bool wait_for_stage1_inflight_quiescence(const Stage1OperationKey& key);
  bool release_resolved_local_stage1_receipt(
      const StorageOwnerMaintenanceTask& task,
      const Configuration& config);
  bool handle_peer_stage1_arm_request(
      u32 source_shard,
      const service::storage_owner::PeerRpcHeader& header,
      const service::storage_owner::Stage1ArmItem* items,
      bool release_quiesced,
      const Configuration& config);
  bool arm_local_stage1_items(
      u32 authority_shard,
      span<const service::storage_owner::Stage1ArmItem> items,
      vec<service::storage_owner::Stage1ArmResult>& results,
      const Configuration& config,
      bool* admission_blocked = nullptr);
  void wake_peer_stage1_admission_waiters();
  bool handle_peer_cleanup_activate_request(
      u32 source_shard,
      const service::storage_owner::PeerRpcHeader& header,
      const service::storage_owner::CleanupActivateItem* items,
      const Configuration& config);
  bool handle_peer_authority_placement_request(
      u32 source_shard,
      const service::storage_owner::PeerRpcHeader& header,
      const service::storage_owner::AuthorityPlacementItem* items,
      const Configuration& config);
  bool handle_peer_dynamic_node_control_request(
      u32 source_shard,
      const service::storage_owner::PeerRpcHeader& header,
      const service::storage_owner::DynamicNodeControlItem* items,
      const Configuration& config);
  bool activate_local_cleanup_items(
      u32 authority_shard,
      span<const service::storage_owner::CleanupActivateItem> items,
      vec<service::storage_owner::CleanupActivateResult>& results,
      const Configuration& config);
  bool apply_local_authority_placement_items(
      span<const service::storage_owner::AuthorityPlacementItem> items,
      vec<service::storage_owner::AuthorityPlacementResult>& results);
  bool apply_local_dynamic_node_control_items(
      u32 source_shard,
      span<const service::storage_owner::DynamicNodeControlItem> items,
      vec<service::storage_owner::DynamicNodeControlResult>& results,
      const Configuration& config);
  bool enqueue_peer_reverse_update_task(PeerReverseUpdateTask&& task);
  bool enqueue_peer_stage1_task(PeerStage1Task&& task);
  bool enqueue_peer_physical_control_task(PeerPhysicalControlTask&& task);
  void enqueue_peer_reverse_update_response(u32 destination_shard,
                                            const service::storage_owner::PeerRpcHeader& request,
                                            bool success);
  void peer_rpc_progress_loop();
  void peer_reverse_update_worker_loop(u32 worker_id);
  void peer_stage1_worker_loop(u32 worker_id);
  void peer_stage2_home_worker_loop(u32 worker_id);
  void peer_cleanup_control_worker_loop();
  void peer_placement_control_worker_loop();
  void peer_reverse_response_loop();
  bool try_post_peer_rpc_request_attempt(
    u32 target_shard,
    service::storage_owner::PeerRpcType request_type,
    service::storage_owner::PeerRpcType response_type,
    u64 request_id,
    u32 item_count,
    const void* items,
    size_t item_bytes,
    size_t request_bytes,
    PeerRpcSendClass send_class);
  Stage2HomeRpcEnqueueResult try_enqueue_stage2_home_rpc_request(
    u32 target_shard,
    service::storage_owner::PeerRpcType request_type,
    u64 logical_request_id,
    u32 item_count,
    span<const byte_t> request,
    PeerResponseCompletionTarget completion_target);
  bool try_drive_stage2_home_rpc_requests(
    u32 target_shard,
    service::storage_owner::PeerRpcType request_type);
  bool cancel_stage2_home_rpc_request(u64 logical_request_id);
  bool try_handle_stage2_home_rpc_aggregate_response(
    u32 peer_id,
    span<const byte_t> response,
    std::vector<PeerResponseCompletionTarget>& completion_targets);
  bool post_peer_op_batch_async(
    u32 target_shard,
    const vec<service::storage_owner::ReverseUpdateOp>& ops,
    service::storage_owner::PeerRpcType request_type,
    u64 request_id,
    u32& item_count,
    const Configuration& config);
  TryPeerResponse try_consume_peer_rpc_response(
    u64 request_id,
    u32 expected_shard,
    service::storage_owner::PeerRpcType expected_type,
    u32 expected_item_count,
    service::storage_owner::PeerRpcHeader& header,
    vec<byte_t>& payload,
    PeerResponseLease& lease);
  bool acknowledge_peer_rpc_response(PeerResponseLease lease);
  bool rearm_peer_rpc_response(PeerResponseLease lease);
  bool await_late_peer_rpc_response(PeerResponseLease lease);
  void cancel_peer_rpc_response(u64 request_id);
  u64 allocate_peer_request_id();
  bool send_reconcile_reverse_fanout_and_wait(
      const dense_hashmap_t<
        u32, vec<service::storage_owner::ReconcileReverseOp>>& updates,
      vec<service::storage_owner::ReconcileReverseResult>& results,
      const Configuration& config);
  bool apply_centroid_membership_fanout_and_wait(
      span<const service::storage_owner::CentroidMembershipOp> ops,
      const Configuration& config);
  bool execute_remote_stage1_fanout_and_wait(
      const dense_hashmap_t<
        u32, vec<service::storage_owner::Stage1ExecuteItem>>& items_by_home,
      const dense_hashmap_t<u32, vec<byte_t>>& vectors_by_home,
      dense_hashmap_t<
        u32, vec<service::storage_owner::Stage1ExecuteResult>>& results_by_home,
      const std::function<void(
        u32,
        span<const service::storage_owner::Stage1ExecuteItem>,
        span<const service::storage_owner::Stage1ExecuteResult>)>&
        on_home_resolved,
      const std::function<void(
        u32,
        span<const service::storage_owner::Stage1ArmItem>)>&
        on_home_release_resolved,
      const std::function<bool()>& overlap_work,
      const Configuration& config);
  bool arm_remote_stage1_batch(
      u32 stage1_home,
      u32 source_client,
      span<const service::storage_owner::Stage1ArmItem> items,
      vec<service::storage_owner::Stage1ArmResult>& results,
      const Configuration& config);
  bool control_stage1_fanout_and_wait(
      const dense_hashmap_t<
        u32, vec<service::storage_owner::Stage1ArmItem>>& items_by_home,
      u32 source_client,
      const std::function<void(
        u32,
        span<const service::storage_owner::Stage1ArmItem>,
        span<const service::storage_owner::Stage1ArmResult>)>&
        on_home_resolved,
      const Configuration& config);
  bool activate_cleanup_fanout_and_wait(
      span<const service::storage_owner::CleanupActivateItem> items,
      vec<service::storage_owner::CleanupActivateResult>& results,
      const Configuration& config);
  bool relocate_via_authority(
      u32 authority_shard,
      const service::storage_owner::AuthorityPlacementItem& item,
      service::storage_owner::AuthorityPlacementResult& result,
      const Configuration& config);
  bool relocate_batch_via_authority(
      span<const u32> authority_shards,
      span<const service::storage_owner::AuthorityPlacementItem> items,
      vec<service::storage_owner::AuthorityPlacementResult>& results,
      const Configuration& config);
  bool control_dynamic_node_on_shard(
      u32 physical_shard,
      const service::storage_owner::DynamicNodeControlItem& item,
      service::storage_owner::DynamicNodeControlResult& result,
      const Configuration& config);
  bool post_peer_control_request_attempt(
      u32 target_shard,
      service::storage_owner::PeerRpcType request_type,
      service::storage_owner::PeerRpcType response_type,
      u64 request_id,
      u32 item_count,
      const void* items,
      size_t item_bytes,
      size_t request_bytes,
      const Configuration& config);
  TryPeerResponse wait_peer_control_response(
      u64 request_id,
      u32 target_shard,
      service::storage_owner::PeerRpcType response_type,
      u32 item_count,
      service::storage_owner::PeerRpcHeader& header,
      vec<byte_t>& payload,
      PeerResponseLease& lease,
      const Configuration& config);
  // Storage-owner background graph maintenance
  static bool storage_owner_maintenance_enabled(const Configuration& config);
  void start_storage_owner_maintenance_runtime(const Configuration& config);
  void stop_storage_owner_maintenance_runtime();
  u64 arm_storage_owner_maintenance(
      StorageOwnerMaintenanceTask&& task, const Configuration& config);
  u64 arm_storage_owner_maintenance_batch(
      vec<StorageOwnerMaintenanceTask>& tasks,
      const Configuration& config,
      bool* capacity_blocked = nullptr);
  u64 activate_storage_owner_cleanup(
      StorageOwnerMaintenanceTask&& task, const Configuration& config);
  u64 activate_storage_owner_cleanup_batch(
      vec<StorageOwnerMaintenanceTask>& tasks,
      const Configuration& config);
  u64 begin_storage_owner_maintenance_sequence(u32 work_items);
  u64 begin_storage_owner_maintenance_batch(span<const u32> work_items);
  u64 try_begin_storage_owner_maintenance_batch(
    span<const u32> work_items);
  void complete_storage_owner_maintenance_sequence(u64 sequence);
  void complete_storage_owner_maintenance_sequence(u64 sequence,
                                                   u32 work_items);
  bool storage_owner_cleanup_ready(u64 sequence) const;
  void publish_storage_owner_maintenance_watermarks();
  void log_storage_owner_maintenance_observation(size_t stage2_remaining,
                                                 size_t cleanup_remaining,
                                                 bool final);
  void maybe_log_storage_owner_maintenance_observation();
  void storage_owner_maintenance_worker_loop(u32 worker_id);
  bool try_acquire_storage_owner_maintenance_slot(
      const Configuration& config);
  memory_node_storage_owner_index_detail::IncarnationLockResult
    try_lock_node(RemotePtr rptr);
  bool storage_owner_task_current(node_t id, u32 generation, RemotePtr target);
  memory_node_storage_owner_index_detail::StableNodeSnapshotState
    storage_owner_physical_node_state(node_t id,
                                      u32 generation,
                                      RemotePtr target,
                                      NodeSnapshot* stable_snapshot = nullptr);
  vec<RemotePtr> read_preserved_neighbor_list(RemotePtr rptr);
  bool remove_local_neighbor(RemotePtr target_ptr, RemotePtr deleted_ptr, const Configuration& config);
  bool remove_local_neighbors_batched(
      const dense_hashmap_t<u64, vec<RemotePtr>>& removals,
      const Configuration& config);
  bool remove_local_neighbors_identity_fenced(
      span<const service::storage_owner::ReverseUpdateOp> ops,
      const Configuration& config);
  // Storage-owner RPC runtime
  void setup_insert_runtime(const Configuration& config);
  void start_storage_owner_insert_workers(const Configuration& config);
  void storage_owner_insert_worker_loop(u32 worker_id);
  void process_storage_owner_insert_task(const StorageOwnerInsertTask& task);
  void post_storage_owner_response(StorageOwnerResponseReady response);
  void post_storage_owner_token_completion(
    u32 client_id,
    u32 completion_slot_id,
    const service::storage_owner::MutationCompletionV2& completion);
  size_t insert_request_slot_offset(u32 client_id, u32 slot_id) const;
  size_t insert_response_slot_offset(const Configuration& config, u32 client_id, u32 slot_id) const;
  size_t insert_completion_slot_offset(u32 client_id, u32 slot_id) const;
  void service_storage_runtime(const Configuration& config);
  size_t response_slot_bytes(const Configuration& config) const;
  size_t handle_storage_insert_request(u32 client_id,
                                       u32 slot_id,
                                       const byte_t* payload,
                                       size_t bytes,
                                       const Configuration& config);
  bool execute_storage_owner_batch_items(const node_t* ids,
                                         const service::storage_owner::MutationKind* kinds,
                                         const byte_t* raw_vectors,
                                         const u32* stage1_homes,
                                         const u64* operation_ids,
                                         u32 source_client,
                                         size_t item_count,
                                         InsertBreakdownCounters& breakdown,
                                         const Configuration& config,
                                         vec<vec<u64>>* invalidated_neighbors = nullptr,
                                         vec<u32>* statuses = nullptr,
                                         vec<service::storage_owner::MutationResult>* results = nullptr,
                                         const std::function<void(size_t)>&
                                           on_terminal = {});
  bool execute_storage_owner_batch_items_exact(
      const node_t* ids,
      const service::storage_owner::MutationKind* kinds,
      const byte_t* raw_vectors,
      const u64* operation_ids,
      u32 source_client,
      size_t item_count,
      InsertBreakdownCounters& breakdown,
      const Configuration& config,
      vec<vec<u64>>* invalidated_neighbors = nullptr,
      vec<u32>* statuses = nullptr,
      vec<service::storage_owner::MutationResult>* results = nullptr,
      const std::function<void(size_t)>& on_terminal = {});

  // Storage-owner index operations
  RemotePtr allocate_local_node();
  void retire_local_dynamic_node(RemotePtr pointer, u64 maintenance_sequence);
  void retire_local_dynamic_node_ready(RemotePtr pointer);
  bool load_owner_idmap(const filepath_t& index_prefix);
  bool mark_node_deleted(RemotePtr rptr, u32 generation);
  AuthorityBeginResult begin_authority_mutation(
    node_t id,
    service::storage_owner::MutationKind kind,
    AuthorityOperationToken operation,
    u32 stage1_home);
  AuthorityCommitState commit_authority_mutation(
    node_t id,
    AuthorityOperationToken operation,
    RemotePtr desired,
    u32 generation,
    bool deleted,
    u64 maintenance_sequence);
  AuthorityAbortState abort_authority_mutation(
    node_t id,
    AuthorityOperationToken operation);
  AuthorityCheckState check_authority_current(
    node_t id,
    AuthorityOperationToken operation,
    u32 generation,
    RemotePtr expected,
    u64 expected_placement_version);
  AuthorityRelocateState relocate_authority_if_current(
    node_t id,
    AuthorityOperationToken operation,
    u32 generation,
    RemotePtr expected,
    RemotePtr desired,
    u64 expected_placement_version,
    u64* resulting_placement_version = nullptr);
  AuthorityDirectoryState load_authority_directory_state_locked(
    const DynamicFreshnessShard& shard,
    node_t id) const;
  void store_authority_directory_state_locked(
    DynamicFreshnessShard& shard,
    node_t id,
    const AuthorityDirectoryState& state);
  u64 load_local_node_header_acquire(RemotePtr rptr) const;
  bool read_locked_node_identity(RemotePtr rptr,
                                 u64& header,
                                 node_t& id,
                                 u32& generation);
  bool publish_locked_node_header(RemotePtr rptr,
                                  u64 observed_header,
                                  u64 set_flags,
                                  u64 clear_flags);
  void report_rejected_graph_pointer(
    const char* boundary,
    RemotePtr pointer,
    RemotePtr parent = RemotePtr{},
    u64 context = std::numeric_limits<u64>::max()) const;
  bool storage_node_pointer_addressable(RemotePtr rptr) const;
  bool valid_local_storage_node_pointer(RemotePtr rptr) const;
  bool read_node_snapshot(RemotePtr rptr, NodeSnapshot& snapshot);
  bool storage_owner_node_live(RemotePtr rptr);
  bool storage_owner_node_stable(RemotePtr rptr);
  bool read_stable_node_identity(RemotePtr rptr);
  size_t read_node_identity_headers_batched_into(
      span<const RemotePtr> rptrs,
      const configuration::IndexConfiguration& config,
      vec<std::pair<RemotePtr, u64>>& identities,
      vec<memory_node_storage_owner_index_detail::StableNodeSnapshotState>*
        states = nullptr);
  bool read_graph_adjacency(RemotePtr rptr,
                            GraphAdjacency& adjacency);
  vec<std::pair<RemotePtr, GraphAdjacency>>
    read_graph_adjacencies_batched(
      span<const RemotePtr> rptrs,
      const Configuration& config);
  size_t read_graph_adjacencies_batched_into(
      span<const RemotePtr> rptrs,
      const Configuration& config,
      vec<std::pair<RemotePtr, GraphAdjacency>>& results);
  vec<RemotePtr> read_neighbor_list(RemotePtr rptr);
  vec<RemotePtr> read_stable_neighbor_list(RemotePtr rptr);
  bool read_local_neighbor_list(RemotePtr rptr,
                                vec<RemotePtr>& neighbors,
                                vec<byte_t>& entry,
                                vec<byte_t>& decoded) const;
  vec<NodeSnapshot> read_node_snapshots_batched(
      const vec<RemotePtr>& rptrs,
      const Configuration& config,
      const char* boundary = "read_node_snapshots_batched",
      vec<memory_node_storage_owner_index_detail::StableNodeSnapshotState>*
        states = nullptr);
  size_t read_node_snapshots_batched_into(
      span<const RemotePtr> rptrs,
      const Configuration& config,
      vec<NodeSnapshot>& snapshots,
      const char* boundary = "read_node_snapshots_batched_into",
      vec<memory_node_storage_owner_index_detail::StableNodeSnapshotState>*
        states = nullptr);
  const vec<BeamEntry>& score_stable_node_vectors_batched(
      span<const RemotePtr> rptrs,
      const byte_t* stored_query,
      span<const element_t> decoded_query,
      const Configuration& config);
  void write_hot_graph_entry(
    RemotePtr rptr,
    const vec<RemotePtr>& neighbors,
    std::optional<u32> generation_override = std::nullopt,
    std::optional<bool> deleted_override = std::nullopt);
  void write_graph_adjacency(
    RemotePtr rptr,
    const vec<RemotePtr>& stable,
    const vec<RemotePtr>& provisional,
    std::optional<u32> generation_override = std::nullopt,
    std::optional<bool> deleted_override = std::nullopt);
  void write_neighbor_list(RemotePtr rptr, const vec<RemotePtr>& neighbors);
  void write_dynamic_navigation_code(RemotePtr rptr,
                                     const span<const element_t> components);
  void write_new_node(RemotePtr rptr,
                      node_t id,
                      const span<const element_t> components,
                      const vec<RemotePtr>& neighbors,
                      u32 generation = 0,
                      bool provisional = false);
  void write_new_node_on_shard(RemotePtr rptr,
                               node_t id,
                               const span<const element_t> components,
                               const vec<RemotePtr>& neighbors,
                               u32 generation,
                               bool provisional);
  bool set_node_provisional(RemotePtr rptr, bool provisional);
  void lock_node(RemotePtr rptr);
  void unlock_node(RemotePtr rptr);
  // Synchronous CPU construction search. It deliberately has no "async"
  // facade: callers must place it on an executor where a complete local graph
  // walk may run without blocking CQ/RPC progress.
  vec<RemotePtr> partition_local_search_candidates(
      const span<const element_t> query,
      const vec<RemotePtr>& entry_points,
      const Configuration& config,
      InsertBreakdownCounters* breakdown = nullptr,
      const byte_t* integral_raw_query = nullptr,
      vec<BeamEntry>* stage1_beam = nullptr,
      vec<RemotePtr>* remote_frontier = nullptr,
      bool record_pipeline_telemetry = true);
  // The returned worker-local buffer is valid until the next Stage2 search
  // on the same OS thread. Callers must consume it synchronously.
  const vec<RemotePtr>& continue_stage2_search_candidates(
      const StorageOwnerMaintenanceTask& task,
      const NodeSnapshot& target,
      const Configuration& config,
      bool record_stage2_telemetry = true);
  void continue_stage2_search_candidates_batched(
      span<const StorageOwnerMaintenanceTask> tasks,
      span<const NodeSnapshot> targets,
      vec<vec<RemotePtr>>& candidates_by_task,
      const Configuration& config);
  memory_node_storage_owner_maintenance_detail::Stage2SearchAdvanceResult
  advance_stage2_search_candidates_batched(
      span<const StorageOwnerMaintenanceTask> tasks,
      span<const NodeSnapshot> targets,
      vec<vec<RemotePtr>>& candidates_by_task,
      memory_node_storage_owner_maintenance_detail::Stage2SearchIoState& state,
      const Configuration& config);
  vec<RemotePtr> local_centroid_route_entries() const;
  void initialize_storage_centroid_route();
  void publish_storage_centroid_route();
  bool apply_local_centroid_membership_ops(
      span<const service::storage_owner::CentroidMembershipOp> ops);
  vec<vamana::routing::CentroidRouter::LiveEntry>
    select_local_centroid_live_entries(
      span<const RemotePtr> preferred = {});
  vec<RemotePtr> robust_prune_cpu(const byte_t* source,
                                  VectorDType source_dtype,
                                  const vec<RemotePtr>& candidates,
                                  const hashset_t<RemotePtr>& skip,
                                  const Configuration& config,
                                  InsertBreakdownCounters* breakdown = nullptr,
                                  u32 result_limit_override = 0);
  vec<RemotePtr> robust_prune_snapshots_cpu(
      const byte_t* source,
      VectorDType source_dtype,
      span<const NodeSnapshot> candidates,
      const hashset_t<RemotePtr>& skip,
      const Configuration& config,
      u32 result_limit_override = 0);
  vec<RemotePtr> robust_prune_snapshot_refs_cpu(
      const byte_t* source,
      VectorDType source_dtype,
      span<const NodeSnapshot* const> candidates,
      const hashset_t<RemotePtr>& skip,
      const Configuration& config,
      u32 result_limit_override = 0);
  bool apply_local_reverse_update(RemotePtr target_ptr,
                                  const vec<RemotePtr>& candidate_ptrs,
                                  const Configuration& config,
                                  bool enqueue_maintenance = true);
  bool apply_local_reverse_updates_batched(
      const dense_hashmap_t<u64, vec<RemotePtr>>& updates,
      const Configuration& config);
  bool reconcile_local_reverse_ops(
      span<const service::storage_owner::ReconcileReverseOp> ops,
      const Configuration& config,
      vec<service::storage_owner::ReconcileReverseResult>& results);
  // Synchronous-exact coordination executes both local and remote reverse
  // reconciliation from the authority owner. Remote targets use the same
  // tagged lock/read/publish protocol through one-sided RDMA and never require
  // a target-shard worker.
  bool reconcile_reverse_ops_one_sided(
      span<const service::storage_owner::ReconcileReverseOp> ops,
      const Configuration& config,
      vec<service::storage_owner::ReconcileReverseResult>& results);
  bool apply_partition_local_reverse_update(RemotePtr target_ptr,
                                            const vec<RemotePtr>& candidate_ptrs,
                                            const Configuration& config,
                                            bool* graph_changed = nullptr);
  vec<RemotePtr> install_local_provisional_backlinks(
      RemotePtr candidate,
      span<const RemotePtr> targets);
  bool remove_local_provisional_backlinks(
      RemotePtr candidate,
      span<const RemotePtr> targets);

  // Misc helpers
  static size_t align_up(size_t value, size_t alignment = kCacheLineBytes);
  distance_t distance_to_stored_vector(const span<const element_t> query,
                                        const byte_t* stored,
                                        const Configuration& config) const;
  distance_t distance_between_vectors(const byte_t* lhs,
                                      VectorDType lhs_dtype,
                                      const byte_t* rhs,
                                      VectorDType rhs_dtype,
                                      const Configuration& config) const;
  const byte_t* local_live_vector(RemotePtr rptr) const;
  bool local_shard(u32 shard_id) const;
  byte_t* local_node_ptr(const RemotePtr& rptr);
  const byte_t* local_node_ptr(const RemotePtr& rptr) const;
  static void insert_into_beam(vec<BeamEntry>& beam, const RemotePtr& rptr, distance_t dist, u32 max_beam_width);
private:
  Context context_;
  ServerConnectionManager cm_;
  Assignment core_assignment_;

  const u32 num_clients_;
  u32 num_compute_threads_{};
  bool gpu_stream_layout_{};
  u64 gpu_static_node_count_{};
  u64 gpu_static_dynamic_base_{};
  u64 gpu_storage_control_offset_{};
  u64 gpu_dynamic_node_base_{};
  u64 dynamic_allocation_limit_{};
  u32 gpu_navigation_code_bytes_{};
  u64 gpu_navigation_model_checksum_{};
  u64 gpu_index_build_fingerprint_{};
  u64 gpu_shard_build_fingerprint_{};
  gpu_search::pq::Model gpu_navigation_model_;
  const u32 storage_id_;
  const u32 num_storage_nodes_;
  // Authority metadata is retained per logical ID to preserve generation and
  // idempotent replay semantics. Binding IDs to this configured namespace
  // makes that state capacity-bounded under adversarial update streams.
  const u32 vector_id_namespace_size_;
  const u32 storage_owner_peer_rdma_tokens_;
  // Fixed and adaptive graph access share the same dynamic record layout.
  // Fixed mode publishes UNKNOWN in the existing advisory tag byte.
  const bool dynamic_graph_extent_publication_enabled_;

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
  std::unique_ptr<PeerAsyncResponseRegistry> peer_async_responses_;
  std::unique_ptr<Stage2HomeRpcOutbox> stage2_home_rpc_outbox_;
  bool stage2_home_rpc_combining_{true};
  std::unique_ptr<PeerRequestDeduplicator> peer_request_deduplicator_;
  std::mutex peer_send_cq_mutex_;
  std::mutex peer_completion_mutex_;
  std::condition_variable peer_completion_cv_;
  vec<ibv_wc> peer_send_wcs_;
  std::unordered_set<u64> peer_sync_completions_;
  // IDs returned by next_peer_*_wr_id() are reserved before the producer
  // drops peer_completion_mutex_ to populate/post its WR.  This closes the
  // allocation-to-registration race when a 32-bit sequence wraps.
  std::unordered_set<u64> peer_reserved_wr_ids_;
  std::unordered_map<u64, PeerPendingSend> peer_pending_sends_;
  vec<std::atomic<u32>> peer_rdma_read_outstanding_;
  vec<vec<std::atomic<u32>>> peer_rdma_read_qp_outstanding_;
  std::unique_ptr<StorageOwnerMaintenanceWaiterSet[]>
    peer_rdma_read_peer_waiters_;
  std::unique_ptr<StorageOwnerMaintenanceWaiterSet[]>
    peer_rdma_read_qp_waiters_;
  size_t peer_rdma_read_qp_waiter_count_{};
  StorageOwnerMaintenanceWaiterSet peer_rdma_read_global_waiters_;
  memory_node_detail::PeerRdmaReadCreditPlan peer_rdma_read_credits_{
    1, 1, 1, 1, 1};
  vec<vec<std::unique_ptr<std::mutex>>> peer_qp_send_mutexes_;
  vec<std::unique_ptr<std::mutex>> peer_rpc_sync_send_mutexes_;
  std::mutex peer_rpc_send_slots_mutex_;
  vec<std::array<std::deque<u32>, 3>> peer_rpc_free_send_slots_;
  std::unique_ptr<StorageOwnerMaintenanceWaiterSet[]>
    peer_rpc_send_slot_waiters_;
  size_t peer_rpc_send_slot_waiter_count_{};
  // Accessed only while peer_completion_mutex_ is held.  They deliberately
  // wrap; allocation probes past every still-live ID in that namespace.
  u32 peer_sync_wr_id_counter_{1};
  u32 peer_async_wr_id_counter_{1};
  std::atomic<u32> peer_async_rdma_outstanding_{0};
  std::atomic<u64> next_peer_request_id_{1};
  std::atomic<bool> peer_rpc_progress_running_{false};
  std::thread peer_rpc_progress_thread_;
  vec<std::thread> peer_reverse_workers_;
  vec<std::thread> peer_stage1_workers_;
  vec<std::thread> peer_stage2_home_workers_;
  std::thread peer_reverse_response_thread_;
  vec<std::thread> peer_cleanup_control_workers_;
  std::thread peer_placement_control_thread_;
  vec<u_ptr<StorageOwnerThread>> peer_reverse_worker_states_;
  vec<u_ptr<StorageOwnerThread>> peer_stage1_worker_states_;
  vec<u_ptr<StorageOwnerThread>> peer_stage2_home_worker_states_;
  std::mutex peer_reverse_tasks_mutex_;
  std::condition_variable peer_reverse_tasks_cv_;
  std::deque<PeerReverseUpdateTask> peer_reverse_tasks_;
  std::mutex peer_stage1_tasks_mutex_;
  std::condition_variable peer_stage1_tasks_cv_;
  std::deque<PeerStage1Task> peer_stage1_tasks_;
  // Read-only Stage2 home work shares the physical-home CPU pool, but not the
  // Stage1 admission queue.  Stage1 is always dequeued first and this queue
  // has its own bound, so an expansion burst cannot reject or starve new
  // mutations before they publish their Stage1 graph.
  std::deque<PeerStage1Task> peer_stage2_home_tasks_;
  // Execute requests whose ANN artifact is already prepared but whose fused
  // Stage2 admission window is full.  They retain their request-dedup lease
  // and per-token in-flight ownership, so a duplicate wire request can be
  // coalesced into the eventual late response instead of re-executing arm.
  // The combined runnable+waiting population is bounded by
  // peer_stage1_task_queue_limit_.
  std::deque<PeerStage1Task> peer_stage1_admission_waiters_;
  // Protected by peer_stage1_tasks_mutex_. This is the admission unit: the
  // waiter bound is expressed in semantic tokens/sequences, not RPC objects.
  size_t peer_stage1_admission_waiter_items_{};
  // Lock-free empty hint for the completion and queue-pop hot paths. Writers
  // update it while holding peer_stage1_tasks_mutex_; a newly parked task also
  // performs a self-wake recheck, so a stale zero cannot lose an edge.
  std::atomic<size_t> peer_stage1_admission_waiter_items_hint_{0};
  // Includes waiter-deque, runnable, and active continuations. This is the
  // actual bounded semantic state; moving a waiter to the runnable queue must
  // not temporarily free room for another full wire batch.
  size_t peer_stage1_admission_owned_items_{};
  // Credits already represented by waiter tasks moved to the runnable queue.
  // Protected by peer_stage1_tasks_mutex_; never included in durable Stage2
  // debt and always released after the corresponding arm attempt.
  size_t peer_stage1_admission_wake_coverage_{};
  // next sequence is protected by peer_stage1_tasks_mutex_. Completion state
  // is diagnostic only; semantic receipt lifetime uses the bounded per-token
  // table below rather than a global per-authority prefix.
  vec<u64> peer_stage1_next_source_sequences_;
  vec<u_ptr<PeerOrderedCompletionState>> peer_stage1_completion_states_;
  std::mutex peer_cleanup_control_tasks_mutex_;
  std::condition_variable peer_cleanup_control_tasks_cv_;
  std::deque<PeerPhysicalControlTask> peer_cleanup_control_tasks_;
  // Cleanup uses its own sequence namespace because Stage1 and cleanup are
  // carried by independent worker pools. Both namespaces are assigned while
  // holding their receive queues' mutexes.
  vec<u64> peer_cleanup_next_source_sequences_;
  vec<u_ptr<PeerOrderedCompletionState>> peer_cleanup_completion_states_;
  std::mutex peer_placement_control_tasks_mutex_;
  std::condition_variable peer_placement_control_tasks_cv_;
  std::deque<PeerPhysicalControlTask> peer_placement_control_tasks_;
  std::unique_ptr<bounded::Queue<PeerReverseUpdateResponse>>
    peer_reverse_responses_;
  // Producers take this mutex while publishing responses, closing the
  // predicate/notify race for the dispatcher.
  std::mutex peer_response_wait_mutex_;
  std::condition_variable peer_response_wait_cv_;
  std::mutex peer_graph_response_buffers_mutex_;
  std::deque<vec<byte_t>> peer_graph_response_buffers_;
  size_t peer_graph_response_buffer_limit_{1};
  std::atomic<bool> peer_reverse_shutdown_{false};
  std::atomic<bool> peer_reverse_workers_done_{false};
  std::atomic<bool> peer_reverse_response_done_{false};
  size_t peer_reverse_task_queue_limit_{1024};
  size_t peer_stage1_task_queue_limit_{1024};
  size_t peer_physical_control_task_queue_limit_{1024};
  std::atomic<u64> peer_reverse_update_enqueued_{0};
  std::atomic<u64> peer_reverse_update_processed_{0};
  std::atomic<u64> peer_reverse_update_items_enqueued_{0};
  std::atomic<u64> peer_reverse_update_items_processed_{0};
  std::atomic<u64> peer_reverse_update_failed_{0};
  std::atomic<u64> peer_reverse_update_max_queue_{0};
  std::atomic<u64> peer_stage1_enqueued_{0};
  std::atomic<u64> peer_stage1_processed_{0};
  std::atomic<u64> peer_stage1_items_{0};
  std::atomic<u64> peer_stage1_max_queue_{0};
  // First-execution service demand at the selected physical home. These
  // include locally coordinated and remotely coordinated inserts alike.
  std::atomic<u64> physical_stage1_items_{0};
  std::atomic<u64> physical_stage1_total_ns_{0};
  std::atomic<u64> physical_stage1_search_ns_{0};
  std::atomic<u64> physical_stage1_prune_ns_{0};
  std::atomic<u64> physical_stage1_allocate_write_ns_{0};
  std::atomic<u64> physical_stage1_backlink_ns_{0};
  std::atomic<u64> physical_stage1_candidates_{0};
  std::atomic<u64> physical_stage1_remote_frontier_items_{0};
  std::atomic<u64> physical_stage1_neighbors_{0};
  std::atomic<u64> peer_stage2_home_enqueued_{0};
  std::atomic<u64> peer_stage2_home_processed_{0};
  std::atomic<u64> peer_stage2_home_items_{0};
  std::atomic<u64> peer_stage2_home_max_queue_{0};
  std::atomic<u64> peer_stage2_home_response_queue_drops_{0};
  std::atomic<u64> peer_stage2_home_response_send_wait_ns_{0};
  std::atomic<u64> peer_stage2_home_queue_wait_ns_{0};
  std::atomic<u64> peer_stage2_home_execution_ns_{0};
  std::atomic<u64> peer_stage1_release_deferred_batches_{0};
  std::atomic<u64> peer_stage1_release_deferred_items_{0};
  std::atomic<u64> peer_stage1_duplicate_retry_responses_{0};
  std::atomic<u64> peer_stage1_admission_retry_responses_{0};
  std::atomic<u64> peer_stage1_retry_response_drops_{0};
  std::atomic<u64> peer_stage1_admission_parked_{0};
  std::atomic<u64> peer_stage1_admission_woken_{0};
  std::atomic<u64> peer_stage1_admission_reparked_{0};
  std::atomic<u64> peer_stage1_duplicate_coalesced_{0};
  std::atomic<u64> peer_stage1_max_admission_waiters_{0};
  std::atomic<u32> peer_stage1_active_workers_{0};
  std::atomic<u32> peer_stage2_home_active_workers_{0};
  std::array<Stage1PreparedResultShard,
             kStage1PreparedShardCount> stage1_prepared_results_;
  std::array<Stage1InflightRequestShard,
             kStage1PreparedShardCount> stage1_inflight_requests_;
  size_t stage1_prepared_results_limit_{1024};
  size_t stage1_prepared_results_limit_per_shard_{16};
  std::array<CleanupActivationDedupeShard,
             kCleanupActivationShardCount> cleanup_activation_dedupe_;
  size_t cleanup_activation_dedupe_limit_per_shard_{16};
  memory_node_storage_owner_index_detail::DynamicAllocationReceiptLedger
    dynamic_allocation_receipts_;
  size_t dynamic_allocation_dedupe_limit_{1024};
  vec<std::thread> storage_owner_maintenance_workers_;
  vec<u_ptr<StorageOwnerThread>> storage_owner_maintenance_worker_states_;
  std::mutex storage_owner_maintenance_mutex_;
  std::condition_variable storage_owner_maintenance_cv_;
  // Peer CQ progress intentionally outlives the maintenance runtime during
  // shutdown.  Keep wake channels in stable MemoryNode storage instead of in
  // worker_states_, which is cleared while peer progress can still release a
  // send slot or deliver a late response.  CPU assignment cannot create more
  // workers than Linux's CPU set can represent.
  struct alignas(64) StorageOwnerMaintenanceWakeChannel {
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<u64> epoch{0};
    std::atomic<u32> waiters{0};
    std::atomic<bool> context_scan_requested{false};
  };
  std::array<StorageOwnerMaintenanceWakeChannel, CPU_SETSIZE>
    storage_owner_maintenance_wake_channels_;
  using StorageOwnerReadyContextQueue =
    memory_node_storage_owner_maintenance_detail::Stage2ReadyContextQueue;
  // Active pointers are published independently of the owning vector. Old
  // runtime queues remain allocated until MemoryNode destruction so a peer CQ
  // producer that loaded a pointer just before stop can finish its epoch-
  // fenced notify without racing queue destruction.
  std::array<std::atomic<StorageOwnerReadyContextQueue*>, CPU_SETSIZE>
    storage_owner_maintenance_ready_queue_active_{};
  vec<std::unique_ptr<StorageOwnerReadyContextQueue>>
    storage_owner_maintenance_ready_queue_storage_;
  std::atomic<u64> storage_owner_maintenance_runtime_epoch_counter_{0};
  std::atomic<u32> storage_owner_maintenance_wake_worker_count_{0};
  std::atomic<u32> storage_owner_maintenance_generic_wake_cursor_{0};
  std::atomic<u64> storage_owner_maintenance_targeted_wakes_{0};
  std::atomic<u64> storage_owner_maintenance_broadcast_wakes_{0};
  std::atomic<u64> storage_owner_maintenance_generic_wakes_{0};
  std::atomic<u64> storage_owner_maintenance_context_slots_scanned_{0};
  std::atomic<u64> storage_owner_maintenance_lost_wake_avoided_{0};
  std::atomic<u64> storage_owner_maintenance_ready_notifications_{0};
  std::atomic<u64> storage_owner_maintenance_ready_stale_notifications_{0};
  std::atomic<u64> storage_owner_maintenance_ready_tickets_drained_{0};
  std::atomic<u64> storage_owner_maintenance_ready_overflow_scans_{0};
  std::atomic<u64> storage_owner_maintenance_ready_fallback_scans_{0};
  // Every worker owns enough registered scratch for its bounded local context
  // pool, while this node-wide lease is the authoritative active-lane bound.
  // This turns a fixed per-worker 2-lane partition into a work-conserving
  // 0..N allocation without moving context/continuation ownership.
  std::atomic<u32> storage_owner_search_lane_leases_{0};
  std::atomic<u32> storage_owner_search_lane_lease_limit_{0};
  std::atomic<u32> storage_owner_search_lane_lease_peak_{0};
  std::atomic<u64> storage_owner_search_lane_lease_blocked_{0};
  StorageOwnerMaintenanceWaiterSet storage_owner_search_lane_waiters_;
  std::array<std::atomic<u32>, CPU_SETSIZE>
    storage_owner_search_lane_blocked_contexts_{};
  std::array<std::atomic<u32>, CPU_SETSIZE>
    storage_owner_search_lane_grants_{};
  memory_node_storage_owner_maintenance_detail::
    Stage2PackingController storage_owner_stage2_packing_;
  std::deque<StorageOwnerMaintenanceTask> storage_owner_stage2_tasks_;
  // Min-heap ordered by maintenance_sequence then retry_not_before. The
  // predecessor durability rule makes its front the only cleanup that can be
  // runnable, eliminating admission-time scans under the global mutex.
  std::deque<StorageOwnerMaintenanceTask> storage_owner_cleanup_tasks_;
  // Arm owns a queue permit while it tries to reserve a completion batch.
  // Generic producers include these permits in their capacity check, so a
  // successfully reserved sequence can become runnable immediately.
  size_t storage_owner_maintenance_reserved_slots_{};
  std::unique_ptr<bounded::Queue<StorageOwnerMaintenanceTask>>
    storage_owner_repair_tasks_;
  std::unique_ptr<Stage2ReverseOutbox> storage_owner_reverse_outbox_;
  vec<std::unique_ptr<bounded::Queue<Stage2ReverseCompletion>>>
    storage_owner_reverse_completions_;
  std::atomic<bool> storage_owner_maintenance_shutdown_{false};
  std::atomic<u64> storage_owner_maintenance_enqueued_{0};
  std::atomic<u64> storage_owner_maintenance_finalize_enqueued_{0};
  std::atomic<u64> storage_owner_maintenance_cleanup_enqueued_{0};
  std::atomic<u64> storage_owner_maintenance_processed_{0};
  std::atomic<u64> storage_owner_maintenance_finalized_live_{0};
  std::atomic<u64> storage_owner_maintenance_failed_{0};
  std::atomic<u64> storage_owner_maintenance_rpc_timeouts_{0};
  std::atomic<u64> storage_owner_reverse_aggregate_batches_{0};
  std::atomic<u64> storage_owner_reverse_aggregate_logical_requests_{0};
  std::atomic<u64> storage_owner_reverse_aggregate_ops_{0};
  std::atomic<u64> storage_owner_maintenance_stale_{0};
  std::atomic<u64> storage_owner_maintenance_cleanup_processed_{0};
  std::atomic<u64> storage_owner_maintenance_max_backlog_{0};
  std::atomic<u64> storage_owner_maintenance_pressure_yields_{0};
  std::atomic<u64> storage_owner_stage2_batches_{0};
  std::atomic<u64> storage_owner_stage2_batched_items_{0};
  std::atomic<u64> storage_owner_stage1_search_budget_exhausted_{0};
  std::atomic<u64> storage_owner_stage2_search_budget_exhausted_{0};
  // Cumulative, attempt-level work counters make the locality mechanism
  // observable without sampling the hot path.  Continuation counters include
  // searches that later become stale; placement counters are recorded only at
  // the exactly-once Stage2 finalization boundary.
  std::atomic<u64> storage_owner_stage2_continuations_{0};
  std::atomic<u64> storage_owner_stage2_remote_frontier_items_{0};
  std::atomic<u64> storage_owner_stage2_remote_expansions_{0};
  std::atomic<u64> storage_owner_stage2_scored_candidates_{0};
  std::atomic<u64> storage_owner_stage2_graph_read_waves_{0};
  std::atomic<u64> storage_owner_stage2_graph_unique_reads_{0};
  std::atomic<u64> storage_owner_stage2_graph_prefetch_issued_{0};
  std::atomic<u64> storage_owner_stage2_graph_prefetch_hits_{0};
  std::atomic<u64> storage_owner_stage2_graph_prefetch_wasted_{0};
  // A 512-outcome rolling window bounds adaptation time after workload drift;
  // the cumulative counters above remain the externally reported telemetry.
  std::atomic<u64> storage_owner_stage2_graph_feedback_base_hits_{0};
  std::atomic<u64> storage_owner_stage2_graph_feedback_base_wasted_{0};
  std::atomic<u64> storage_owner_stage2_graph_feedback_next_outcome_{512};
  std::atomic<u32> storage_owner_stage2_graph_issue_width_current_{1};
  std::atomic<u64> storage_owner_stage2_vector_read_waves_{0};
  std::atomic<u64> storage_owner_stage2_vector_unique_reads_{0};
  std::atomic<u64> storage_owner_stage2_home_rpc_batches_{0};
  std::atomic<u64> storage_owner_stage2_home_rpc_items_{0};
  std::atomic<u64> storage_owner_stage2_home_score_rpc_batches_{0};
  std::atomic<u64> storage_owner_stage2_home_score_rpc_items_{0};
  std::atomic<u64> storage_owner_stage2_home_score_rpc_queries_{0};
  std::atomic<u64> storage_owner_stage2_home_score_rpc_request_bytes_{0};
  std::atomic<u64> storage_owner_stage2_home_score_rpc_response_bytes_{0};
  std::atomic<u64> storage_owner_stage2_home_scored_neighbors_{0};
  std::atomic<u64> storage_owner_stage2_migrations_{0};
  std::atomic<u64> storage_owner_stage2_final_edges_{0};
  std::atomic<u64> storage_owner_stage2_cross_edges_stage1_home_{0};
  std::atomic<u64> storage_owner_stage2_cross_edges_final_home_{0};
  // Stage2 timings are aggregated once per context phase/attempt.  Keeping
  // these counters at the batch boundary makes the diagnostic cost
  // independent of L, R, vector dimension, and the number of RDMA reads.
  enum class StorageOwnerStage2TimingPhase : size_t {
    continuation_search,
    freeze_prune,
    reverse_prepare,
    placement_authority,
    completion_handoff,
    finalize,
    count,
  };
  static constexpr size_t kStorageOwnerStage2TimingPhaseCount =
    static_cast<size_t>(StorageOwnerStage2TimingPhase::count);
  struct StorageOwnerStage2TimingCounters {
    std::atomic<u64> attempts{0};
    std::atomic<u64> task_attempts{0};
    std::atomic<u64> elapsed_ns{0};
  };
  std::array<StorageOwnerStage2TimingCounters,
             kStorageOwnerStage2TimingPhaseCount>
    storage_owner_stage2_phase_timing_{};
  std::atomic<u64> storage_owner_maintenance_worker_idle_waits_{0};
  std::atomic<u64> storage_owner_maintenance_worker_idle_ns_{0};
  std::atomic<u32> storage_owner_maintenance_active_workers_{0};
  std::atomic<u64> storage_owner_maintenance_started_ns_{0};
  std::atomic<u64> storage_owner_maintenance_last_observation_ns_{0};
  std::atomic<u64> storage_owner_maintenance_finalize_latency_ns_{0};
  std::atomic<u64> storage_owner_maintenance_finalize_max_latency_ns_{0};
  std::array<std::atomic<u64>, 18> storage_owner_maintenance_finalize_latency_buckets_{};
  // Synchronous-exact motivation telemetry. The atomics aggregate concurrent
  // foreground workers; the mutex serializes control-page publication only.
  std::atomic<u64> exact_insert_items_{0};
  std::atomic<u64> exact_insert_total_ns_{0};
  std::atomic<u64> exact_insert_remote_read_ns_{0};
  std::atomic<u64> exact_insert_remote_reverse_ns_{0};
  std::atomic<u64> exact_insert_search_ns_{0};
  std::atomic<u64> exact_insert_prune_ns_{0};
  std::atomic<u64> exact_insert_allocate_write_ns_{0};
  std::atomic<u64> exact_insert_local_reverse_ns_{0};
  std::mutex exact_insert_telemetry_mutex_;
  std::unique_ptr<bounded::SlidingCompletionRing>
    storage_owner_maintenance_completion_ring_;
  // Maximum accepted-but-not-yet-complete Stage2 descriptors. This is the
  // bounded foreground/queue debt window, not an active context or RPC limit;
  // execution resources are claimed independently by maintenance workers.
  size_t storage_owner_maintenance_admission_limit_{};
  size_t storage_owner_maintenance_intent_capacity_{};
  std::unique_ptr<StorageOwnerMaintenanceIntent[]>
    storage_owner_maintenance_intents_;
  mutable std::mutex storage_owner_reclaim_mutex_;
  memory_node_detail::StorageReclaimQueue storage_owner_reclaim_queue_;
  std::atomic<u64> storage_owner_reclaim_candidates_{0};
  InsertRuntimeState insert_runtime_;
  std::unique_ptr<Configuration> storage_worker_config_;
  std::unique_ptr<bounded::Queue<StorageOwnerInsertTask>> storage_insert_tasks_;
  vec<u_ptr<std::mutex>> storage_client_send_mutexes_;
  vec<u_ptr<bounded::Queue<u32>>> storage_client_completion_free_slots_;
  // An accepted batch reserves enough completion capacity for every item in
  // that batch.  Credits are per RC connection so one compute client cannot
  // consume another client's completion window.
  std::unique_ptr<std::atomic<u32>[]>
    storage_client_batch_context_credits_;
  memory_node_detail::StorageOwnerCpuPlan storage_owner_exact_cpu_plan_;
  bool storage_owner_exact_cpu_plan_initialized_{};
  vec<u_ptr<StorageOwnerThread>> storage_owner_threads_;
  vec<std::thread> storage_insert_workers_;
  std::atomic<bool> storage_insert_shutdown_{false};
  const u64 mn_memory_bytes_;
  timing::Timing timing_;
  filepath_t index_prefix_;
  std::unique_ptr<vamana::routing::CentroidRouter>
    storage_centroid_router_;
  std::mutex storage_centroid_publication_mutex_;
  std::mutex storage_centroid_update_mutex_;
  // One bit per physical node gives O(1) membership changes and word-wise
  // replacement-entry selection. It avoids rescanning an ever-growing
  // dynamic region when a published route entry is deleted.
  vec<u64> storage_centroid_static_live_bitmap_;
  vec<u64> storage_centroid_dynamic_live_bitmap_;
  u64 storage_centroid_static_cursor_{};
  u64 storage_centroid_dynamic_cursor_{};
  bool owner_idmap_required_{false};
  // Immutable base IDs need only their physical handle. Generation zero,
  // live state, placement version zero, and empty replay receipts are
  // materialized on lookup. Full transaction state exists only in sparse
  // per-shard mutation state after an ID participates in a mutation.
  dense_hashmap_t<node_t, RemotePtr> base_idmap_;
  std::array<DynamicFreshnessShard, kDynamicFreshnessShardCount>
    dynamic_freshness_shards_;

  inline static thread_local StorageOwnerThread* current_storage_owner_thread_{nullptr};
  inline static thread_local bool current_storage_owner_maintenance_worker_{false};
  inline static thread_local StorageOwnerMaintenanceWaiterRegistrations*
    current_storage_owner_maintenance_waiter_registrations_{nullptr};
  inline static thread_local memory_node_storage_owner_maintenance_detail::
    Stage2ContextOwnerKey current_storage_owner_maintenance_context_owner_{};
  inline static thread_local bool current_peer_rpc_progress_thread_{false};
};
