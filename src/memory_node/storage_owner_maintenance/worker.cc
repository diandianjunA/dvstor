#include "memory_node/storage_owner_maintenance/detail.hh"
#include "memory_node/storage_owner_maintenance/admission_policy.hh"
#include "memory_node/storage_owner_maintenance/centroid_lifecycle_policy.hh"
#include "memory_node/storage_owner_maintenance/cleanup_scheduler.hh"
#include "memory_node/storage_owner_maintenance/cleanup_policy.hh"
#include "memory_node/storage_owner_maintenance/fallback_audit_policy.hh"
#include "memory_node/storage_owner_maintenance/reconcile_batch_state.hh"
#include "memory_node/storage_owner_maintenance/search_io_state.hh"
#include "memory_node/storage_owner_maintenance/search_lane_pool.hh"
#include "memory_node/storage_owner_maintenance/stage2_tracker.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"
#include "memory_node/storage_owner_index/stage1_prune_handoff_policy.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

#include <algorithm>
#include <iterator>
#include <limits>

using namespace memory_node_storage_owner_maintenance_detail;
using memory_node_storage_owner_index_detail::IncarnationLockResult;
using memory_node_storage_owner_index_detail::StableNodeSnapshotState;
using memory_node_storage_owner_index_detail::classify_stable_node_snapshot;
namespace protocol = service::storage_owner;

void MemoryNode::storage_owner_maintenance_worker_loop(u32 worker_id) {
  lib_assert(worker_id < storage_owner_maintenance_worker_states_.size(),
             "storage-owner maintenance worker state missing");
  StorageOwnerThread& thread = *storage_owner_maintenance_worker_states_[worker_id];
  lib_assert(worker_id < storage_owner_maintenance_wake_channels_.size(),
             "storage-owner maintenance wake channel missing");
  auto& wake_channel = storage_owner_maintenance_wake_channels_[worker_id];
  current_storage_owner_thread_ = &thread;
  current_storage_owner_maintenance_worker_ = true;
  current_storage_owner_maintenance_context_owner_ = {};
  StorageOwnerReadyContextQueue* const ready_context_queue =
    storage_owner_maintenance_ready_queue_active_[worker_id].load(
      std::memory_order_acquire);
  lib_assert(ready_context_queue != nullptr,
             "storage-owner maintenance ready queue missing");
  StorageOwnerMaintenanceWaiterRegistrations fallback_waiter_registrations;
  fallback_waiter_registrations.reserve(
    5 + static_cast<size_t>(num_storage_nodes_) *
          (static_cast<size_t>(peer_qps_per_peer_) + 1));
  current_storage_owner_maintenance_waiter_registrations_ =
    &fallback_waiter_registrations;
  const Configuration& config = *storage_worker_config_;
  const bool independent_score_experiment_enabled =
    storage_owner_independent_score_experiment_enabled_;
  lib_assert(num_storage_nodes_ > 0 && num_storage_nodes_ <= 64,
             "asynchronous stage2 supports at most 64 storage shards");

  using ReverseUpdateOp = service::storage_owner::ReverseUpdateOp;
  using PeerRpcType = service::storage_owner::PeerRpcType;

  struct Stage2Context {
    bool active{};
    Stage2ContextHandle handle{};
    Stage2ContextOwnerKey ready_owner{};
    StorageOwnerMaintenanceKind kind{
      StorageOwnerMaintenanceKind::finalize_insert};
    vec<StorageOwnerMaintenanceTask> tasks;
    vec<NodeSnapshot> targets;
    // Continuation is consumed after the context has crossed a fallible
    // authority/RPC gate.  It therefore belongs to this resumable context,
    // not to worker-wide scratch that another active context may overwrite.
    vec<vec<RemotePtr>> continued_candidates_by_task;
    // Parent liveness is revalidated immediately before reconciliation.  The
    // resulting exact planner input must survive scheduler yields while the
    // three ordered remote barriers are in flight.
    vec<vec<RemotePtr>> live_stage2_neighbors_by_task;
    Stage2FinalizeSubphase finalize_subphase{
      Stage2FinalizeSubphase::prepare};
    Stage2ReconcileBatchState reconcile_batch;
    vec<vec<ReverseUpdateOp>> remote_ops_by_peer;
    vec<u64> reverse_request_ids;
    // Timestamped only after all asynchronous reconciliation barriers and the
    // synchronous placement/membership suffix have completed. The outer state
    // tracker has no reverse ACK mask at this point; this measures only the
    // state-machine/context handoff into finalization. Remote reconciliation
    // latency is charged to the context-spanning reverse_prepare timer.
    u64 completion_handoff_started_ns{};
    // Logical search state belongs to the resumable context. Registered RDMA
    // scratch belongs to search_lane only while a posted one-sided operation
    // can still touch it. This lets a home-RPC wait release and later rebind a
    // lane without restarting the continuation.
    Stage2SearchIoState search_io;
    std::optional<u32> search_lane;
    bool search_lane_wait_registered{};
    // Exact context-owned subscriptions for SEND/RDMA/lane capacity. Release
    // consumes only a worker runnable bit; success/reset removes this
    // context's references without hiding sibling contexts on the same worker.
    StorageOwnerMaintenanceWaiterRegistrations waiter_registrations;
    bool search_input_prepared{};
    bool search_timing_recorded{};
    u64 search_started_ns{};
    u64 reverse_prepare_started_ns{};
    bool reverse_prepare_timing_active{};
    // Admission-only scheduling metadata. It never participates in graph
    // semantics and exists solely to close the adaptive packing feedback loop.
    u64 packing_admitted_ns{};
    u64 packing_wait_ns{};
    size_t packing_debt_at_admission{};
    u32 packing_target_batch{1};
    bool packing_high_pressure{};
    // Exact node-wide execution-budget claim made before descriptors leave
    // storage_owner_stage2_tasks_. It deliberately survives task filtering
    // inside the context and is returned only when the context is retired.
    u32 active_task_reservation{};
    IndependentScoreSample independent_score_sample{};
  };

  // One timer update represents a whole context phase attempt.  In
  // particular, no timing atomic appears in a candidate, edge, or RDMA-read
  // loop.  The destructor records failed/deferred attempts as well, which is
  // essential when diagnosing a phase that never reaches finalization.
  struct Stage2PhaseAttemptTimer {
    StorageOwnerStage2TimingCounters* counters{};
    u64 started_ns{};
    u64 task_attempts{};

    Stage2PhaseAttemptTimer(StorageOwnerStage2TimingCounters& value,
                            size_t tasks)
        : counters(&value),
          started_ns(steady_now_ns()),
          task_attempts(static_cast<u64>(tasks)) {}

    Stage2PhaseAttemptTimer(const Stage2PhaseAttemptTimer&) = delete;
    Stage2PhaseAttemptTimer& operator=(
      const Stage2PhaseAttemptTimer&) = delete;

    ~Stage2PhaseAttemptTimer() { finish(); }

    void transition(StorageOwnerStage2TimingCounters& next,
                    size_t tasks) {
      finish();
      counters = &next;
      started_ns = steady_now_ns();
      task_attempts = static_cast<u64>(tasks);
    }

    void finish() {
      if (counters == nullptr) return;
      const u64 elapsed_ns = steady_now_ns() - started_ns;
      counters->attempts.fetch_add(1, std::memory_order_relaxed);
      counters->task_attempts.fetch_add(
        task_attempts, std::memory_order_relaxed);
      counters->elapsed_ns.fetch_add(elapsed_ns,
                                     std::memory_order_relaxed);
      counters = nullptr;
    }
  };

  // A worker owns its contexts and both trackers, so the response path needs
  // no lock beyond the bounded peer-response registry. Global admission below
  // limits all workers together to the dedicated peer RPC depth.
  const size_t context_capacity =
    std::max<size_t>(1, config.storage_owner_rpc_depth);
  const size_t remote_peer_count = num_storage_nodes_ - 1;
  const u32 construction_width =
    config.resolved_storage_owner_construction_width();
  const size_t request_capacity =
    context_capacity * std::max<size_t>(1, remote_peer_count);
  Stage2StateTracker states(context_capacity, num_storage_nodes_);
  Stage2RequestTracker requests(request_capacity);
  vec<Stage2Context> contexts(context_capacity);
  Stage2SearchLanePool search_lanes(thread.post_balances.size());
  // Stage2 resumes one global beam; it does not collect an independent L-set
  // from every shard. Its memory footprint is therefore O(batch * L), not
  // O(batch * shard_count * L).
  const size_t candidate_capacity_per_item = construction_width;
  const size_t reconcile_op_capacity =
    (static_cast<size_t>(config.R) + 1) *
      config.storage_owner_batch_max;
  lib_assert(peer_rpc_runtime_.message_bytes >
               sizeof(service::storage_owner::PeerRpcHeader),
             "peer RPC slot has no reconciliation payload capacity");
  const size_t reconcile_payload_bytes = peer_rpc_runtime_.message_bytes -
    sizeof(service::storage_owner::PeerRpcHeader);
  const size_t reconcile_wire_capacity = std::max<size_t>(
    1, reconcile_payload_bytes /
         sizeof(service::storage_owner::ReconcileReverseOp));
  const size_t reconcile_chunk_capacity = num_storage_nodes_ +
    (reconcile_op_capacity + reconcile_wire_capacity - 1) /
      reconcile_wire_capacity;
  for (Stage2Context& context : contexts) {
    context.waiter_registrations.reserve(
      5 + static_cast<size_t>(num_storage_nodes_) *
            (static_cast<size_t>(peer_qps_per_peer_) + 1));
    context.tasks.reserve(config.storage_owner_batch_max);
    context.targets.reserve(config.storage_owner_batch_max);
    context.continued_candidates_by_task.reserve(
      config.storage_owner_batch_max);
    context.live_stage2_neighbors_by_task.resize(
      config.storage_owner_batch_max);
    context.reconcile_batch.reserve(
      reconcile_op_capacity, reconcile_chunk_capacity);
    context.remote_ops_by_peer.resize(num_storage_nodes_);
    for (auto& ops : context.remote_ops_by_peer) {
      ops.reserve(static_cast<size_t>(config.R) *
                  config.storage_owner_batch_max);
    }
    context.reverse_request_ids.resize(num_storage_nodes_);
  }
  // Continuation pointers and all later snapshot waves are consumed
  // synchronously by this worker. Reusing their O(batch*L) capacity here avoids
  // multiplying full-vector scratch by every in-flight state-machine context.
  const size_t continuation_capacity =
    static_cast<size_t>(config.storage_owner_batch_max) *
    candidate_capacity_per_item;
  // Only the authoritative source-freeze boundary materializes continuation
  // vectors. Shared physical candidates are read once across the whole batch;
  // ordered pointer references are then scattered into per-task prune views.
  vec<vec<RemotePtr>> snapshot_candidates_by_task(
    config.storage_owner_batch_max);
  vec<vec<const NodeSnapshot*>> snapshots_by_task(
    config.storage_owner_batch_max);
  vec<bool> snapshot_task_active(config.storage_owner_batch_max);
  Stage2SnapshotWavePlan snapshot_plan;
  vec<NodeSnapshot> snapshot_storage;
  vec<StableNodeSnapshotState> snapshot_states;
  snapshot_storage.reserve(
    continuation_capacity +
    static_cast<size_t>(config.storage_owner_batch_max) * config.R);
  dense_hashmap_t<u64, const NodeSnapshot*> snapshot_by_raw;
  snapshot_by_raw.reserve(
    continuation_capacity +
    static_cast<size_t>(config.storage_owner_batch_max) * config.R);
  vec<vec<RemotePtr>> live_stage2_neighbors_by_task(
    config.storage_owner_batch_max);
  vec<std::pair<RemotePtr, u64>> identity_storage;
  vec<StableNodeSnapshotState> identity_states;
  identity_storage.reserve(
    static_cast<size_t>(config.storage_owner_batch_max) * config.R);
  hashset_t<RemotePtr> stable_identity_targets;
  stable_identity_targets.reserve(
    static_cast<size_t>(config.storage_owner_batch_max) * config.R);

  const u64 rpc_timeout_ns =
    static_cast<u64>(config.storage_owner_rpc_timeout_ms) * 1000ull * 1000ull;
  const u64 retry_backoff_ns = std::min<u64>(
    rpc_timeout_ns, 1000ull * 1000ull);
  lib_assert(storage_owner_reverse_outbox_ != nullptr &&
               worker_id < storage_owner_reverse_completions_.size() &&
               storage_owner_reverse_completions_[worker_id] != nullptr,
             "stage2 reverse aggregation runtime is not initialized");
  const u32 reverse_wire_max_ops =
    storage_owner_reverse_outbox_->wire_max_ops();
  vec<ReverseUpdateOp> reverse_wire_ops(reverse_wire_max_ops);
  vec<Stage2ReverseCompletion> reverse_completion_scratch(
    reverse_wire_max_ops);
  vec<u8> reverse_completion_worker_marked(
    storage_owner_reverse_completions_.size(), u8{0});
  vec<u32> reverse_completion_wake_owners;
  reverse_completion_wake_owners.reserve(
    storage_owner_reverse_completions_.size());
  vec<byte_t> reverse_response_payload;
  reverse_response_payload.reserve(peer_rpc_runtime_.message_bytes);

  const auto release_context_lane = [&](Stage2Context& context) {
    if (!context.search_lane.has_value()) return;
    const u32 lane = *context.search_lane;
    lib_assert(search_lanes.owns(lane, context.handle),
               "stage2 context lost ownership of its search lane");
    const bool rdma_ready = thread.is_ready(lane);
    lib_assert(rdma_ready,
               "stage2 context released scratch with RDMA still in flight");
    lib_assert(search_lanes.release(lane, context.handle, rdma_ready),
               "stage2 search lane release violated context generation");
    context.search_lane.reset();
    release_storage_owner_search_lane_lease();
  };

  const auto bind_context_lane = [&](Stage2Context& context) {
    if (!context.search_lane.has_value()) {
      if (!try_acquire_storage_owner_search_lane_lease(
            context.search_lane_wait_registered)) {
        return false;
      }
      const auto lane = search_lanes.try_acquire(context.handle);
      if (!lane.has_value()) {
        // With one physical lane per bounded local context this is reachable
        // only if ownership bookkeeping is inconsistent. Keep the release
        // path defensive so an unexpected retry cannot leak the node-wide
        // lease and starve every other worker.
        release_storage_owner_search_lane_lease();
        return false;
      }
      context.search_lane = *lane;
    }
    lib_assert(search_lanes.owns(*context.search_lane, context.handle),
               "stage2 search lane belongs to another context generation");
    thread.set_current_coroutine(*context.search_lane);
    return true;
  };

  const auto release_rebindable_context_lane = [&](Stage2Context& context) {
    if (!context.search_lane.has_value()) return true;
    const u32 lane = *context.search_lane;
    lib_assert(search_lanes.owns(lane, context.handle),
               "stage2 rebind check observed a foreign search lane");
    if (!stage2_search_lane_rebindable(
          thread.is_ready(lane), context.search_io.scratch_rebindable())) {
      return false;
    }
    release_context_lane(context);
    return true;
  };

  const auto release_active_task_reservation = [&](Stage2Context& context) {
    if (context.active_task_reservation == 0) return;
    const u32 released = context.active_task_reservation;
    context.active_task_reservation = 0;
    lib_assert(try_release_stage2_active_tasks(
                 storage_owner_maintenance_active_tasks_, released),
               "Stage2 active-task reservation underflow");
  };

  const auto reset_context = [&](Stage2Context& context) {
    auto* const previous_waiter_registrations =
      current_storage_owner_maintenance_waiter_registrations_;
    current_storage_owner_maintenance_waiter_registrations_ =
      &context.waiter_registrations;
    lib_assert(!context.search_lane.has_value(),
               "stage2 context reset before releasing its search lane");
    cancel_storage_owner_search_lane_waiter(
      context.search_lane_wait_registered);
    clear_all_current_storage_owner_maintenance_waiters();
    for (const Stage2ReconcileChunk& chunk :
         context.reconcile_batch.chunks()) {
      if (!chunk.complete && chunk.request_id != 0) {
        cancel_peer_rpc_response(chunk.request_id);
      }
    }
    for (size_t rpc_index = 0;
         rpc_index < context.search_io.home_expand_rpc_count; ++rpc_index) {
      const Stage2HomeExpandRpc& rpc =
        context.search_io.home_expand_rpcs[rpc_index];
      if (rpc.posted && !rpc.complete && rpc.request_id != 0) {
        cancel_peer_rpc_response(rpc.request_id);
      }
    }
    for (size_t rpc_index = 0;
         rpc_index < context.search_io.score_home_rpc_count; ++rpc_index) {
      const Stage2HomeExpandRpc& rpc =
        context.search_io.score_home_rpcs[rpc_index];
      if (rpc.posted && !rpc.complete && rpc.request_id != 0) {
        cancel_peer_rpc_response(rpc.request_id);
      }
    }
    for (Stage2SpeculativeScoreRpc& rpc :
         context.search_io.speculative_score_rpcs) {
      if (rpc.posted && rpc.request_id != 0) {
        cancel_peer_rpc_response(rpc.request_id);
      }
      if (rpc.process_credit_held) {
        if (rpc.posted) {
          // The local registry cancellation does not cancel remote execution.
          // Disable further lookahead to this peer rather than reusing a
          // request-lifetime credit whose remote work is still unaccounted.
          fail_closed_peer_rpc_speculative_credit(
            rpc.target_shard, rpc.request_id);
        } else {
          release_peer_rpc_speculative_credit(
            rpc.target_shard, rpc.request_id);
        }
        rpc.process_credit_held = false;
      }
    }
    // Cancel every transport member while its exact OwnerKey is still live.
    // A concurrent aggregate completion can then either publish to this
    // context or observe the cancellation; it can never target a reused slot.
    if (context.ready_owner.token != 0) {
      lib_assert(ready_context_queue->deactivate(context.ready_owner),
                 "stage2 context ready owner was stale at reset");
      context.ready_owner = {};
    }
    context.search_io.reset();
    context.reconcile_batch.clear();
    release_active_task_reservation(context);
    context.active = false;
    context.tasks.clear();
    context.targets.clear();
    context.targets.resize(context.tasks.size());
    for (vec<RemotePtr>& candidates :
         context.continued_candidates_by_task) {
      candidates.clear();
    }
    for (vec<RemotePtr>& neighbors :
         context.live_stage2_neighbors_by_task) {
      neighbors.clear();
    }
    for (auto& ops : context.remote_ops_by_peer) {
      ops.clear();
    }
    std::fill(context.reverse_request_ids.begin(),
              context.reverse_request_ids.end(), 0);
    context.completion_handoff_started_ns = 0;
    context.search_input_prepared = false;
    context.search_timing_recorded = false;
    context.search_started_ns = 0;
    context.reverse_prepare_started_ns = 0;
    context.reverse_prepare_timing_active = false;
    context.packing_admitted_ns = 0;
    context.packing_wait_ns = 0;
    context.packing_debt_at_admission = 0;
    context.packing_target_batch = 1;
    context.packing_high_pressure = false;
    context.active_task_reservation = 0;
    context.independent_score_sample = {};
    context.finalize_subphase = Stage2FinalizeSubphase::prepare;
    current_storage_owner_maintenance_waiter_registrations_ =
      previous_waiter_registrations;
  };

  const auto materialize_stable_snapshot_wave = [&](size_t task_count) {
    lib_assert(task_count <= snapshot_candidates_by_task.size(),
               "Stage2 snapshot wave exceeded worker batch capacity");
    snapshot_plan.build(span<const vec<RemotePtr>>{
      snapshot_candidates_by_task.data(), task_count});
    const size_t snapshot_count = read_node_snapshots_batched_into(
      span<const RemotePtr>{snapshot_plan.targets}, config,
      snapshot_storage, "stage2_freeze_prune_wave", &snapshot_states);
    lib_assert(snapshot_states.size() == snapshot_plan.targets.size(),
               "Stage2 snapshot wave lost per-target read state");
    if (std::find(snapshot_states.begin(), snapshot_states.end(),
                  StableNodeSnapshotState::retryable) !=
        snapshot_states.end()) {
      return false;
    }
    snapshot_by_raw.clear();
    snapshot_by_raw.reserve(snapshot_count);
    for (size_t snapshot_index = 0; snapshot_index < snapshot_count;
         ++snapshot_index) {
      const NodeSnapshot& snapshot = snapshot_storage[snapshot_index];
      if (stage2_parent_is_stable(snapshot.header, snapshot.deleted)) {
        snapshot_by_raw.emplace(snapshot.rptr.raw_address, &snapshot);
      }
    }

    if (snapshots_by_task.size() < task_count) {
      snapshots_by_task.resize(task_count);
    }
    for (size_t task = 0; task < task_count; ++task) {
      vec<const NodeSnapshot*>& snapshots = snapshots_by_task[task];
      snapshots.clear();
      snapshots.reserve(snapshot_plan.task_target_indices[task].size());
      for (const u32 target_index :
           snapshot_plan.task_target_indices[task]) {
        lib_assert(target_index < snapshot_plan.targets.size(),
                   "Stage2 snapshot wave task index is invalid");
        const RemotePtr target = snapshot_plan.targets[target_index];
        const auto found = snapshot_by_raw.find(target.raw_address);
        if (found != snapshot_by_raw.end()) snapshots.push_back(found->second);
      }
    }
    return true;
  };

  const auto materialize_stable_identity_wave = [&](size_t task_count) {
    lib_assert(task_count <= snapshot_candidates_by_task.size(),
               "Stage2 identity wave exceeded worker batch capacity");
    snapshot_plan.build(span<const vec<RemotePtr>>{
      snapshot_candidates_by_task.data(), task_count});
    const size_t identity_count = read_node_identity_headers_batched_into(
      span<const RemotePtr>{snapshot_plan.targets}, config,
      identity_storage, &identity_states);
    lib_assert(identity_states.size() == snapshot_plan.targets.size(),
               "Stage2 identity wave lost per-target read state");
    if (std::find(identity_states.begin(), identity_states.end(),
                  StableNodeSnapshotState::retryable) !=
        identity_states.end()) {
      return false;
    }
    stable_identity_targets.clear();
    stable_identity_targets.reserve(identity_count);
    for (size_t index = 0; index < identity_count; ++index) {
      const auto& [pointer, header] = identity_storage[index];
      if (stage2_parent_is_stable(
            header, (header & VamanaNode::HEADER_DELETED) != 0)) {
        stable_identity_targets.insert(pointer);
      }
    }
    for (size_t task = 0; task < task_count; ++task) {
      vec<RemotePtr>& live = live_stage2_neighbors_by_task[task];
      live.clear();
      live.reserve(snapshot_plan.task_target_indices[task].size());
      for (const u32 target_index :
           snapshot_plan.task_target_indices[task]) {
        lib_assert(target_index < snapshot_plan.targets.size(),
                   "Stage2 identity wave task index is invalid");
        const RemotePtr target = snapshot_plan.targets[target_index];
        if (stable_identity_targets.contains(target)) {
          live.push_back(target);
        }
      }
    }
    return true;
  };

  const auto record_finalized_live = [this](
      std::chrono::steady_clock::time_point queued_at) {
    const u64 latency_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - queued_at).count());
    storage_owner_maintenance_finalize_latency_ns_.fetch_add(
      latency_ns, std::memory_order_relaxed);
    storage_owner_maintenance_finalize_latency_buckets_[
      finalize_latency_bucket(latency_ns)].fetch_add(
        1, std::memory_order_relaxed);
    atomic_utils::update_max_relaxed(
      storage_owner_maintenance_finalize_max_latency_ns_, latency_ns);
    storage_owner_maintenance_finalized_live_.fetch_add(
      1, std::memory_order_relaxed);
  };

  using ReconcileReverseOp = service::storage_owner::ReconcileReverseOp;
  using ReconcileReverseOpKind =
    service::storage_owner::ReconcileReverseOpKind;
  using CentroidMembershipOp =
    service::storage_owner::CentroidMembershipOp;
  using CentroidMembershipKind =
    service::storage_owner::CentroidMembershipKind;

  const auto make_reconcile_op = [](
      const StorageOwnerMaintenanceTask& task,
      RemotePtr graph_target,
      RemotePtr old_candidate,
      RemotePtr new_candidate,
      ReconcileReverseOpKind kind) {
    return ReconcileReverseOp{
      .target_raw = graph_target.raw_address,
      .old_candidate_raw = old_candidate.raw_address,
      .new_candidate_raw = new_candidate.raw_address,
      .placement_sequence = task.maintenance_sequence,
      .id = task.id,
      .generation = task.generation,
      .kind = static_cast<u32>(kind),
    };
  };

  const auto make_centroid_membership_op = [](
      const StorageOwnerMaintenanceTask& task,
      RemotePtr pointer,
      CentroidMembershipKind kind) {
    return CentroidMembershipOp{
      .node_raw = pointer.raw_address,
      .maintenance_sequence = task.maintenance_sequence,
      .id = task.id,
      .generation = task.generation,
      .kind = static_cast<u32>(kind),
    };
  };

  // One maintenance context emits at most one RPC stream per destination,
  // independent of the number of inserted nodes.  The peer helper posts all
  // chunks before waiting for ACKs, preserving cross-shard fanout parallelism.
  const auto apply_reconcile_ops = [&, this](
      span<const ReconcileReverseOp> ops) {
    vec<ReconcileReverseOp> local_ops;
    dense_hashmap_t<u32, vec<ReconcileReverseOp>> remote_ops;
    local_ops.reserve(ops.size());
    for (const ReconcileReverseOp& op : ops) {
      const RemotePtr target{op.target_raw};
      if (target.is_null() || target.memory_node() >= num_storage_nodes_) {
        return false;
      }
      if (local_shard(target.memory_node())) {
        local_ops.push_back(op);
      } else {
        remote_ops[target.memory_node()].push_back(op);
      }
    }

    vec<service::storage_owner::ReconcileReverseResult> local_results;
    if (!local_ops.empty() &&
        !reconcile_local_reverse_ops(
          span<const ReconcileReverseOp>{local_ops}, config,
          local_results)) {
      return false;
    }
    lib_assert(local_results.size() == local_ops.size(),
               "local reconciliation lost per-operation results");
    for (size_t index = 0; index < local_ops.size(); ++index) {
      if (!memory_node_storage_owner_index_detail::
            reconcile_reverse_postcondition_holds(
              local_ops[index], local_results[index])) {
        return false;
      }
    }
    if (remote_ops.empty()) return true;
    vec<service::storage_owner::ReconcileReverseResult> remote_results;
    return send_reconcile_reverse_fanout_and_wait(
      remote_ops, remote_results, config);
  };

  enum class CleanupParentQuiesceResult : u8 {
    ready,
    busy,
    stale,
  };

  const auto quiesce_cleanup_parent = [&, this](
      StorageOwnerMaintenanceTask& task) -> CleanupParentQuiesceResult {
    if (task.cleanup_repair_only || task.cleanup_retiring) {
      return CleanupParentQuiesceResult::ready;
    }
    lib_assert(!task.target.is_null() &&
                 local_shard(task.target.memory_node()) &&
                 storage_node_pointer_addressable(task.target),
               "cleanup parent is not an addressable local physical node");

    const IncarnationLockResult target_lock = try_lock_node(task.target);
    if (target_lock == IncarnationLockResult::busy) {
      return CleanupParentQuiesceResult::busy;
    }
    if (target_lock == IncarnationLockResult::stale) {
      // Reuse is fenced behind completion of the original cleanup sequence.
      // Observing a different incarnation therefore means a duplicate/late
      // descriptor whose old-incarnation postcondition is already durable.
      return CleanupParentQuiesceResult::stale;
    }
    const u64 header = load_local_node_header_acquire(task.target);
    const byte_t* record = index_buffer_.get_full_buffer() +
      task.target.byte_offset();
    const node_t observed_id = *reinterpret_cast<const node_t*>(
      record + VamanaNode::offset_id());
    const u32 observed_generation = *reinterpret_cast<const u32*>(
      record + VamanaNode::offset_generation());
    if (VamanaNode::header_incarnation(header) !=
          task.target.incarnation() ||
        observed_id != task.id || observed_generation != task.generation) {
      unlock_node(task.target);
      return CleanupParentQuiesceResult::stale;
    }

    GraphAdjacency adjacency;
    if (!read_graph_adjacency(task.target, adjacency)) {
      unlock_node(task.target);
      return CleanupParentQuiesceResult::busy;
    }
    if ((header & VamanaNode::HEADER_DELETED) != 0 ||
        adjacency.deleted) {
      // An idempotent duplicate can observe the postcondition produced by the
      // original cleanup. It must not attempt to reparent from a tombstone.
      task.cleanup_retiring = true;
      task.cleanup_protected_reparented = true;
      unlock_node(task.target);
      return CleanupParentQuiesceResult::ready;
    }

    auto* header_ptr = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() +
      vamana::StorageLayoutResolver::header(task.target).offset);
    std::atomic_ref<u64>(*header_ptr).fetch_or(
      static_cast<u64>(VamanaNode::HEADER_RETIRING),
      std::memory_order_acq_rel);
    task.cleanup_protected_children = adjacency.provisional;
    task.cleanup_replacement_parents.assign(
      task.cleanup_protected_children.size(), RemotePtr{});
    task.cleanup_retiring = true;
    unlock_node(task.target);
    return CleanupParentQuiesceResult::ready;
  };

  const auto reparent_cleanup_children = [&, this](
      StorageOwnerMaintenanceTask& task) {
    if (task.cleanup_repair_only || task.cleanup_protected_reparented) {
      return true;
    }
    lib_assert(task.cleanup_retiring,
               "protected-child reparent ran before parent quiescence");
    lib_assert(task.cleanup_protected_children.size() ==
                 task.cleanup_replacement_parents.size(),
               "protected-child replacement state lost correlation");

    vec<ReconcileReverseOp> reservations;
    reservations.reserve(task.cleanup_protected_children.size());
    for (size_t child_index = 0;
         child_index < task.cleanup_protected_children.size();
         ++child_index) {
      const RemotePtr child = task.cleanup_protected_children[child_index];
      NodeSnapshot child_snapshot;
      if (!read_node_snapshot(child, child_snapshot)) {
        // A reused tagged slot is no longer this protected child. A
        // same-incarnation lock is only transient, so retain it and retry.
        u64 child_header = 0;
        if (local_shard(child.memory_node())) {
          if (!valid_local_storage_node_pointer(child)) continue;
          child_header = load_local_node_header_acquire(child);
        } else if (child.memory_node() < num_storage_nodes_) {
          const auto address =
            vamana::StorageLayoutResolver::header(child);
          if (address.offset > mn_memory_bytes_ ||
              sizeof(child_header) > mn_memory_bytes_ - address.offset) {
            continue;
          }
          remote_read_bytes(child.memory_node(), address.offset,
                            &child_header, sizeof(child_header), 0);
        } else {
          continue;
        }
        if (VamanaNode::header_incarnation(child_header) ==
            child.incarnation()) {
          return false;
        }
        continue;
      }
      if (child_snapshot.deleted) continue;

      GraphAdjacency child_adjacency;
      if (!read_graph_adjacency(child, child_adjacency) ||
          child_adjacency.deleted) {
        return false;
      }
      const vec<RemotePtr> candidates =
        order_protected_reparent_candidates(
          child, task.target,
          span<const RemotePtr>{child_adjacency.stable});

      RemotePtr& replacement =
        task.cleanup_replacement_parents[child_index];
      const auto usable_replacement = [&](RemotePtr candidate) {
        if (candidate.is_null() || candidate == task.target ||
            candidate == child) {
          return false;
        }
        NodeSnapshot candidate_snapshot;
        if (!read_node_snapshot(candidate, candidate_snapshot) ||
            candidate_snapshot.deleted ||
            !VamanaNode::stable_graph_mutation_allowed(
              candidate_snapshot.header)) {
          return false;
        }
        GraphAdjacency candidate_adjacency;
        if (!read_graph_adjacency(candidate, candidate_adjacency) ||
            candidate_adjacency.deleted) {
          return false;
        }
        if (protected_reparent_target_has_capacity(
              child,
              span<const RemotePtr>{candidate_adjacency.provisional},
              VamanaNode::provisional_slots())) {
          return true;
        }
        // A full-looking target may contain a deleted/reused tagged child;
        // ensure_reachable will reclaim that exact stale slot atomically
        // under the target lock. Never treat a transiently locked live child
        // as reclaimable.
        for (const RemotePtr protected_child :
             candidate_adjacency.provisional) {
          NodeSnapshot protected_snapshot;
          if (read_node_snapshot(protected_child, protected_snapshot)) {
            if (protected_snapshot.deleted) return true;
            continue;
          }
          u64 protected_header = 0;
          if (local_shard(protected_child.memory_node())) {
            if (!valid_local_storage_node_pointer(protected_child)) {
              return true;
            }
            protected_header =
              load_local_node_header_acquire(protected_child);
          } else if (protected_child.memory_node() < num_storage_nodes_) {
            const auto protected_address =
              vamana::StorageLayoutResolver::header(protected_child);
            if (protected_address.offset > mn_memory_bytes_ ||
                sizeof(protected_header) >
                  mn_memory_bytes_ - protected_address.offset) {
              return true;
            }
            remote_read_bytes(
              protected_child.memory_node(), protected_address.offset,
              &protected_header, sizeof(protected_header), 0);
          } else {
            return true;
          }
          if (VamanaNode::header_incarnation(protected_header) !=
              protected_child.incarnation()) {
            return true;
          }
        }
        return false;
      };

      if (!usable_replacement(replacement)) {
        replacement.reset();
        for (const RemotePtr candidate : candidates) {
          if (usable_replacement(candidate)) {
            replacement = candidate;
            break;
          }
        }
      }
      if (replacement.is_null()) {
        // Capacity pressure is correctness backpressure: keep the retiring
        // parent query-visible and retry after another Stage2/delete frees a
        // protected slot.
        return false;
      }
      reservations.push_back(ReconcileReverseOp{
        .target_raw = replacement.raw_address,
        .old_candidate_raw = 0,
        .new_candidate_raw = child.raw_address,
        .placement_sequence = task.maintenance_sequence,
        .id = child_snapshot.id,
        .generation = child_snapshot.generation,
        .kind = static_cast<u32>(
          ReconcileReverseOpKind::ensure_reachable),
      });
    }

    if (!apply_reconcile_ops(span<const ReconcileReverseOp>{reservations})) {
      return false;
    }
    task.cleanup_protected_reparented = true;
    return true;
  };

  const auto append_stage2_reconcile_ops = [&](
      auto& task,
      span<const RemotePtr> live_final_neighbors,
      vec<ReconcileReverseOp>& promotion_ops,
      vec<ReconcileReverseOp>& stable_ops,
      vec<ReconcileReverseOp>& removal_ops) {
    const auto plan = plan_stage2_backlink_reconciliation(
      span<const RemotePtr>{task.stage1_backlink_targets},
      live_final_neighbors,
      task.stage2_promotion_committed
        ? task.stage2_promotion_parent : RemotePtr{});
    if (plan.promotion_target.is_null()) return false;

    task.stage2_promotion_parent = plan.promotion_target;

    promotion_ops.push_back(make_reconcile_op(
      task, plan.promotion_target,
      plan.promotion_consumes_stage1_bridge ? task.target : RemotePtr{},
      task.final_target, ReconcileReverseOpKind::promote_stable_bridge));

    stable_ops.reserve(
      stable_ops.size() + plan.ordinary_stable_targets.size());
    for (const auto& target : plan.ordinary_stable_targets) {
      const bool replace = target.had_stage1_bridge &&
        task.final_target != task.target;
      stable_ops.push_back(make_reconcile_op(
        task, target.target, replace ? task.target : RemotePtr{},
        task.final_target,
        replace ? ReconcileReverseOpKind::replace_or_add
                : ReconcileReverseOpKind::add));
    }
    removal_ops.reserve(
      removal_ops.size() + plan.obsolete_stage1_bridges.size());
    for (const RemotePtr target : plan.obsolete_stage1_bridges) {
      removal_ops.push_back(make_reconcile_op(
        task, target, task.target, RemotePtr{},
        ReconcileReverseOpKind::remove_if_present));
    }
    return true;
  };

  const auto remove_candidate_backlinks = [&](
      const StorageOwnerMaintenanceTask& task,
      RemotePtr candidate,
      span<const RemotePtr> targets) {
    if (candidate.is_null() || targets.empty()) return true;
    hashset_t<RemotePtr> unique_targets;
    unique_targets.reserve(targets.size());
    vec<ReconcileReverseOp> ops;
    ops.reserve(targets.size());
    for (const RemotePtr target : targets) {
      if (target.is_null() || !unique_targets.insert(target).second) continue;
      ops.push_back(make_reconcile_op(
        task, target, candidate, RemotePtr{},
        ReconcileReverseOpKind::remove_if_present));
    }
    return apply_reconcile_ops(span<const ReconcileReverseOp>{ops});
  };

  const auto settle_dynamic_allocation = [&, this](
      const StorageOwnerMaintenanceTask& task) {
    if (task.final_target.is_null() ||
        task.final_target == task.target) {
      return true;
    }
    const protocol::DynamicNodeControlItem settlement{
      .token = {
        .source_client = task.source_client,
        .item_index = task.operation_item_index,
        .client_batch_id = task.operation_batch_id,
      },
      .node_raw = task.target.raw_address,
      .allocated_raw = task.final_target.raw_address,
      .id = task.id,
      .generation = task.generation,
      .authority_shard = task.authority_shard,
      .action = static_cast<u32>(
        protocol::DynamicNodeControlAction::settle_allocation),
    };
    protocol::DynamicNodeControlResult result;
    if (!control_dynamic_node_on_shard(
          task.final_target.memory_node(), settlement, result, config)) {
      return false;
    }
    return static_cast<protocol::DynamicNodeControlStatus>(result.status) ==
      protocol::DynamicNodeControlStatus::ok;
  };

  const auto complete_stale_stage2 = [&](StorageOwnerMaintenanceTask& task) {
    if (!task.stage1_receipt_released) {
      const RemotePtr gate_target = task.placement_committed
        ? task.final_target : task.target;
      const u64 gate_version = task.initial_placement_version +
        static_cast<u64>(task.placement_committed &&
                         task.final_target != task.target);
      const protocol::AuthorityPlacementItem gate{
        .token = {
          .source_client = task.source_client,
          .item_index = task.operation_item_index,
          .client_batch_id = task.operation_batch_id,
        },
        .id = task.id,
        .generation = task.generation,
        .expected_raw = gate_target.raw_address,
        .desired_raw = gate_target.raw_address,
        .expected_placement_version = gate_version,
      };
      protocol::AuthorityPlacementResult gate_result;
      if (!relocate_via_authority(
            task.authority_shard, gate, gate_result, config)) {
        return false;
      }
      const auto status = static_cast<
        protocol::AuthorityPlacementStatus>(gate_result.status);
      if (status == protocol::AuthorityPlacementStatus::busy) return false;
      if (status != protocol::AuthorityPlacementStatus::committed &&
          status != protocol::AuthorityPlacementStatus::replay &&
          status != protocol::AuthorityPlacementStatus::stale) {
        return false;
      }
      if (!release_resolved_local_stage1_receipt(task, config)) {
        return false;
      }
      task.stage1_receipt_released = true;
    }
    // A successor can win while Stage2 is searching or after part of an
    // idempotent reconciliation retry. Remove both physical incarnations
    // before retiring an uncommitted migrated record. Unlike the legacy
    // cleanup path, this explicitly addresses the provisional backlink plane.
    vec<RemotePtr> original_targets = task.stage1_backlink_targets;
    if (task.final_target.is_null() || task.final_target == task.target) {
      original_targets.insert(original_targets.end(),
                              task.stage2_neighbors.begin(),
                              task.stage2_neighbors.end());
    }
    bool cleaned = remove_candidate_backlinks(
      task, task.target, span<const RemotePtr>{original_targets});
    if (cleaned && !task.final_target.is_null() &&
        task.final_target != task.target) {
      cleaned = remove_candidate_backlinks(
        task, task.final_target,
        span<const RemotePtr>{task.stage2_neighbors});
    }
    if (!cleaned) return false;

    // A timeout can occur after final membership was published but before the
    // worker observed its ACK. Withdraw that exact tagged generation while it
    // is still readable; only then may the uncommitted destination be
    // tombstoned. The accounted bit makes this a no-op when publication never
    // happened or a successor cleanup already removed it.
    const RemotePtr accounted_candidate = task.final_target.is_null()
      ? task.target : task.final_target;
    const CentroidMembershipOp centroid_remove =
      make_centroid_membership_op(
        task, accounted_candidate, CentroidMembershipKind::remove);
    if (!apply_centroid_membership_fanout_and_wait(
          span<const CentroidMembershipOp>{&centroid_remove, 1}, config)) {
      return false;
    }
    if (!task.final_target.is_null() && task.final_target != task.target) {
      if (stale_stage2_owns_source_retirement(
            task.placement_committed, task.target, task.final_target)) {
        // Placement made final_target the authority predecessor seen by every
        // successor. If that successor wins before this worker tombstones the
        // Stage1 source, nobody else owns source cleanup. The source graph was
        // frozen before rebase, both backlink incarnations are gone above,
        // and it was never centroid-accounted, so retire it here before
        // settling the destination allocation receipt.
        NodeSnapshot source_snapshot;
        if (!read_node_snapshot(task.target, source_snapshot) ||
            source_snapshot.id != task.id ||
            source_snapshot.generation != task.generation) {
          return false;
        }
        if (!source_snapshot.deleted &&
            !mark_node_deleted(task.target, task.generation)) {
          return false;
        }
        retire_local_dynamic_node(
          task.target, task.maintenance_sequence);
      }
      const protocol::DynamicNodeControlItem retirement{
        .token = {
          .source_client = task.source_client,
          .item_index = task.operation_item_index,
          .client_batch_id = task.operation_batch_id,
        },
        .node_raw = task.final_target.raw_address,
        .id = task.id,
        .generation = task.generation,
        .authority_shard = task.authority_shard,
        .action = static_cast<u32>(
          protocol::DynamicNodeControlAction::retire),
      };
      protocol::DynamicNodeControlResult retirement_result;
      if (!control_dynamic_node_on_shard(
            task.final_target.memory_node(), retirement,
            retirement_result, config)) {
        return false;
      }
      const auto retirement_status = static_cast<
        protocol::DynamicNodeControlStatus>(retirement_result.status);
      if (retirement_status != protocol::DynamicNodeControlStatus::ok &&
          retirement_status != protocol::DynamicNodeControlStatus::stale) {
        return false;
      }
      if (!settle_dynamic_allocation(task)) return false;
      task.allocation_settled = true;
    }
    storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
    storage_owner_maintenance_processed_.fetch_add(1, std::memory_order_relaxed);
    complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
    return true;
  };

  const auto complete_stale_cleanup = [&](const StorageOwnerMaintenanceTask& task) {
    storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
    storage_owner_maintenance_cleanup_processed_.fetch_add(
      1, std::memory_order_relaxed);
    complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
  };

  const auto reverse_request_type = [&](const Stage2Context& context) {
    return context.kind == StorageOwnerMaintenanceKind::cleanup_deleted_node
      ? PeerRpcType::cleanup_deleted_request
      : PeerRpcType::reverse_update_request;
  };

  const auto enqueue_reverse_dispatch = [&](Stage2Context& context,
                                             u32 shard,
                                             u64 ready_at_ns) {
    lib_assert(storage_owner_reverse_outbox_ != nullptr,
               "stage2 reverse outbox is not initialized");
    lib_assert(shard < num_storage_nodes_ && shard != storage_id_,
               "stage2 reverse dispatch targets an invalid peer");
    lib_assert(context.remote_ops_by_peer[shard].size() <=
                 std::numeric_limits<u32>::max(),
               "stage2 reverse dispatch item count exceeds wire bound");
    const u32 item_count = static_cast<u32>(
      context.remote_ops_by_peer[shard].size());
    const Stage2ReverseDispatch dispatch{
      .logical_request_id = context.reverse_request_ids[shard],
      .context = context.handle,
      .worker_id = worker_id,
      .peer_index = shard,
      .request_type = reverse_request_type(context),
      .item_count = item_count,
      .ops = context.remote_ops_by_peer[shard].data(),
      .ready_at_ns = ready_at_ns,
    };
    const Stage2ReverseEnqueueResult result =
      storage_owner_reverse_outbox_->try_enqueue(dispatch);
    lib_assert(result == Stage2ReverseEnqueueResult::enqueued ||
                 result == Stage2ReverseEnqueueResult::duplicate,
               "bounded stage2 reverse outbox capacity/correlation invariant failed");
    if (result == Stage2ReverseEnqueueResult::enqueued) {
      // This worker immediately drives the shared outbox below; no executor
      // wake is needed for its own publication.
    }
    return result;
  };

  const auto register_reverse_requests = [&](Stage2Context& context,
                                             u64 expected_mask) {
    const u64 now = steady_now_ns();
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      const u64 bit = u64{1} << shard;
      if ((expected_mask & bit) == 0) {
        continue;
      }
      const u64 request_id = allocate_peer_request_id();
      context.reverse_request_ids[shard] = request_id;
      // Reverse retry/deadline state belongs to the aggregate wire request;
      // this logical record exists only for ACK fan-out into the context mask.
      const auto result = requests.try_register(
        request_id, context.handle, Stage2RequestKind::reverse_update, shard,
        now, std::numeric_limits<u64>::max(), states);
      lib_assert(result == Stage2RequestRegisterResult::registered,
                 "stage2 reverse request tracker capacity invariant failed");
      const auto queued = enqueue_reverse_dispatch(context, shard, now);
      lib_assert(queued == Stage2ReverseEnqueueResult::enqueued,
                 "new stage2 reverse request was already present in outbox");
    }
  };

  const auto aggregate_response_type = [](PeerRpcType request_type) {
    return request_type == PeerRpcType::cleanup_deleted_request
      ? PeerRpcType::cleanup_deleted_response
      : PeerRpcType::reverse_update_response;
  };

  const auto poll_owned_reverse_aggregates = [&]() {
    bool progressed = false;
    size_t cursor = 0;
    for (;;) {
      const auto aggregate =
        storage_owner_reverse_outbox_->claim_awaiting_response(
          worker_id, cursor);
      if (!aggregate.has_value()) break;

      service::storage_owner::PeerRpcHeader header{};
      reverse_response_payload.clear();
      const PeerRpcType response_type =
        aggregate_response_type(aggregate->request_type);
      PeerResponseLease response_lease{};
      TryPeerResponse response = try_consume_peer_rpc_response(
        aggregate->wire_request_id, aggregate->peer_index, response_type,
        aggregate->item_count, header, reverse_response_payload,
        response_lease);
      const u64 now = steady_now_ns();
      if (response == TryPeerResponse::success) {
        if (!acknowledge_peer_rpc_response(response_lease)) {
          response = TryPeerResponse::stale;
        }
      }
      if (response == TryPeerResponse::success) {
        const auto completion_count =
          storage_owner_reverse_outbox_->copy_completions(
            worker_id, aggregate->wire_request_id,
            std::span<Stage2ReverseCompletion>{
              reverse_completion_scratch.data(),
              reverse_completion_scratch.size()});
        lib_assert(completion_count.has_value() &&
                     *completion_count == aggregate->logical_count,
                   "stage2 reverse aggregate lost ACK fan-out metadata");
        // The copied completions are value snapshots. Release every logical
        // outbox entry before making an ACK visible to a destination worker:
        // that worker may consume its final ACK, reuse the context slot, and
        // enqueue replacement work immediately. Keeping the old entries until
        // after fan-out would create a transient false-full at exact capacity.
        lib_assert(storage_owner_reverse_outbox_->finish_success(
                     worker_id, aggregate->wire_request_id),
                   "stage2 reverse aggregate ACK release failed");
        reverse_completion_wake_owners.clear();
        for (size_t index = 0; index < *completion_count; ++index) {
          const Stage2ReverseCompletion& completion =
            reverse_completion_scratch[index];
          lib_assert(completion.worker_id <
                       storage_owner_reverse_completions_.size() &&
                       storage_owner_reverse_completions_[
                         completion.worker_id] != nullptr,
                     "stage2 reverse completion targets an invalid worker");
          lib_assert(storage_owner_reverse_completions_[
                       completion.worker_id]->try_push(completion),
                     "bounded stage2 reverse completion capacity invariant failed");
          if (reverse_completion_worker_marked[completion.worker_id] == 0) {
            reverse_completion_worker_marked[completion.worker_id] = 1;
            reverse_completion_wake_owners.push_back(completion.worker_id);
          }
        }
        // One aggregate may contain logical requests owned by several
        // executors. Wake exactly the distinct owners that received a queue
        // entry instead of making unrelated executors rescan every context.
        for (const u32 wake_owner : reverse_completion_wake_owners) {
          reverse_completion_worker_marked[wake_owner] = 0;
          notify_storage_owner_maintenance_executor(wake_owner);
        }
        progressed = true;
        continue;
      }

      if (response == TryPeerResponse::failure ||
          response == TryPeerResponse::stale) {
        if (response == TryPeerResponse::failure && response_lease.valid()) {
          (void)rearm_peer_rpc_response(response_lease);
        }
        if (response == TryPeerResponse::stale) {
          cancel_peer_rpc_response(aggregate->wire_request_id);
        }
        lib_assert(storage_owner_reverse_outbox_->release_poll(
                     worker_id, aggregate->wire_request_id, true,
                     now + retry_backoff_ns),
                   "stage2 reverse aggregate failure retry release failed");
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        progressed = true;
        continue;
      }

      const bool timed_out = now >= aggregate->deadline_ns;
      if (timed_out) {
        storage_owner_maintenance_rpc_timeouts_.fetch_add(
          1, std::memory_order_relaxed);
      }
      lib_assert(storage_owner_reverse_outbox_->release_poll(
                   worker_id, aggregate->wire_request_id, timed_out, now),
                 "stage2 reverse aggregate poll release failed");
      progressed = timed_out || progressed;
    }
    return progressed;
  };

  const auto form_reverse_aggregates = [&]() {
    bool progressed = false;
    const u64 now = steady_now_ns();
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      if (shard == storage_id_) continue;
      for (;;) {
        if (!storage_owner_reverse_outbox_->can_form_aggregate(shard, now)) {
          break;
        }
        const u64 wire_request_id = allocate_peer_request_id();
        const auto aggregate = storage_owner_reverse_outbox_->form_aggregate(
          shard, worker_id, wire_request_id, now);
        if (!aggregate.has_value()) break;
        storage_owner_reverse_aggregate_batches_.fetch_add(
          1, std::memory_order_relaxed);
        storage_owner_reverse_aggregate_logical_requests_.fetch_add(
          aggregate->logical_count, std::memory_order_relaxed);
        storage_owner_reverse_aggregate_ops_.fetch_add(
          aggregate->item_count, std::memory_order_relaxed);
        progressed = true;
      }
    }
    return progressed;
  };

  const auto post_owned_reverse_aggregates = [&]() {
    bool progressed = false;
    size_t cursor = 0;
    for (;;) {
      const u64 now = steady_now_ns();
      const auto aggregate =
        storage_owner_reverse_outbox_->claim_ready_to_post(
          worker_id, now, cursor);
      if (!aggregate.has_value()) break;

      reverse_wire_ops.resize(aggregate->item_count);
      lib_assert(storage_owner_reverse_outbox_->copy_ops(
                   worker_id, aggregate->wire_request_id,
                   std::span<ReverseUpdateOp>{reverse_wire_ops.data(),
                                              reverse_wire_ops.size()}),
                 "stage2 reverse aggregate payload copy failed");
      u32 item_count = 0;
      const bool sent = post_peer_op_batch_async(
        aggregate->peer_index, reverse_wire_ops, aggregate->request_type,
        aggregate->wire_request_id, item_count, config);
      lib_assert(item_count == aggregate->item_count,
                 "stage2 reverse aggregate post changed item_count");
      const u64 posted_at = steady_now_ns();
      lib_assert(storage_owner_reverse_outbox_->finish_post(
                   worker_id, aggregate->wire_request_id, sent,
                   posted_at + (sent ? rpc_timeout_ns : retry_backoff_ns)),
                 "stage2 reverse aggregate post release failed");
      if (sent) {
        progressed = true;
      } else {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
      }
    }
    reverse_wire_ops.resize(reverse_wire_max_ops);
    return progressed;
  };

  const auto drive_reverse_outbox = [&]() {
    bool progressed = poll_owned_reverse_aggregates();
    progressed = form_reverse_aggregates() || progressed;
    progressed = post_owned_reverse_aggregates() || progressed;
    return progressed;
  };

  const auto drain_reverse_completions = [&]() {
    bool progressed = false;
    Stage2ReverseCompletion completion;
    auto& completion_queue = *storage_owner_reverse_completions_[worker_id];
    while (completion_queue.try_pop(completion)) {
      lib_assert(completion.worker_id == worker_id,
                 "stage2 reverse completion reached the wrong worker");
      const auto metadata = requests.find(completion.logical_request_id);
      lib_assert(metadata.has_value() &&
                   metadata->context == completion.context &&
                   metadata->kind == Stage2RequestKind::reverse_update &&
                   metadata->peer_index == completion.peer_index,
                 "stage2 reverse completion lost logical correlation");
      const Stage2EventResult result = requests.record_response(
        completion.logical_request_id, states);
      lib_assert(result == Stage2EventResult::accepted ||
                   result == Stage2EventResult::ready_to_finalize ||
                   result == Stage2EventResult::duplicate,
                 "stage2 rejected an aggregate reverse-update ACK");
      lib_assert(completion.context.slot < contexts.size(),
                 "stage2 reverse completion context slot is out of range");
      Stage2Context& ready_context = contexts[completion.context.slot];
      lib_assert(ready_context.active &&
                   ready_context.handle == completion.context,
                 "stage2 reverse completion reached a stale context");
      lib_assert(enqueue_storage_owner_maintenance_context_ready(
                   ready_context.ready_owner,
                   Stage2ContextReadyReason::reverse_completion),
                 "stage2 reverse completion failed ready-queue handoff");
      lib_assert(requests.erase(completion.logical_request_id),
                 "stage2 reverse completion request release failed");
      completion = {};
      progressed = true;
    }
    return progressed;
  };

  const auto prepare_local = [&](Stage2Context& context)
      -> Stage2SearchAdvanceResult {
    if (context.kind == StorageOwnerMaintenanceKind::cleanup_deleted_node) {
      const auto transition = states.begin_remote_search(context.handle, 0);
      lib_assert(transition == Stage2EventResult::phase_advanced,
                 "cleanup stage2 failed to enter prune_ready");
      return Stage2SearchAdvanceResult::complete;
    }

    if (!context.search_input_prepared) {
      context.search_started_ns = steady_now_ns();
      context.targets.clear();
      context.targets.resize(context.tasks.size());

      // Retire only a stably stale Stage1 record before issuing continuation
      // reads. This preparation is run exactly once: a context suspended on
      // a CQ or a transient local NODE_LOCK resumes its private continuation,
      // rather than rebuilding from a newer graph snapshot or being mistaken
      // for a deleted insertion.
      for (size_t item = 0; item < context.tasks.size(); ++item) {
        StorageOwnerMaintenanceTask& task = context.tasks[item];
        if (task.maintenance_sequence == 0) continue;
        lib_assert(local_shard(task.target.memory_node()),
                   "Stage2 must execute on the Stage1 physical shard");

        NodeSnapshot target_snapshot;
        const StableNodeSnapshotState target_state =
          storage_owner_physical_node_state(
            task.id, task.generation, task.target, &target_snapshot);
        if (target_state == StableNodeSnapshotState::retryable) {
          // Reverse-edge publication may briefly own NODE_LOCK on this same
          // physical target. Suspend only this context so other search lanes
          // keep hiding RDMA latency; no bounded retry count changes graph
          // semantics.
          return Stage2SearchAdvanceResult::waiting_rdma;
        }
        if (target_state == StableNodeSnapshotState::terminal) {
          if (!complete_stale_stage2(task)) {
            return Stage2SearchAdvanceResult::waiting_rdma;
          }
          task.maintenance_sequence = 0;
          continue;
        }
        if (task.stage1_prune_deferred &&
            !task.stage1_prune_materialized) {
          // Reconstruct the exact local RobustPrune from the converged Stage1
          // Beam before global continuation. The foreground published only a
          // nearest-first provisional adjacency; durable finalization still
          // consumes the same diversity-pruned local seed as the legacy path.
          vec<RemotePtr> local_candidates;
          local_candidates.reserve(task.stage1_beam.size());
          for (const memory_node_detail::BeamEntry& entry : task.stage1_beam) {
            local_candidates.push_back(entry.rptr);
          }
          const hashset_t<RemotePtr> skip;
          task.stage1_pruned_neighbors = robust_prune_cpu(
            target_snapshot.vector_data.data(), VamanaNode::vector_dtype(),
            local_candidates, skip, config, nullptr, config.R);
          task.stage1_prune_materialized = true;
        }
        context.targets[item] = std::move(target_snapshot);
      }

      size_t ready = 0;
      for (size_t item = 0; item < context.tasks.size(); ++item) {
        if (context.tasks[item].maintenance_sequence == 0) continue;
        if (ready != item) {
          context.tasks[ready] = std::move(context.tasks[item]);
          context.targets[ready] = std::move(context.targets[item]);
        }
        ++ready;
      }
      context.tasks.resize(ready);
      context.targets.resize(ready);
      context.search_input_prepared = true;
    }

    // Advance every independent continuation by one dependency step per
    // wave. Graph and vector reads that are ready at the same time are issued
    // across the complete context batch, eliminating the per-task RDMA RTT
    // chain while preserving each task's private beam and convergence state.
    Stage2SearchIoState& search_io = context.search_io;
    search_io.independent_score_allowed =
      independent_score_experiment_enabled &&
      context.independent_score_sample.allows_speculation();
    const Stage2SearchAdvanceResult search_result =
      advance_stage2_search_candidates_batched(
      span<const StorageOwnerMaintenanceTask>{context.tasks},
      span<const NodeSnapshot>{context.targets},
      context.continued_candidates_by_task, search_io, config);
    if (search_result != Stage2SearchAdvanceResult::complete) {
      return search_result;
    }
    // Record the continuation at its semantic completion boundary.  The
    // prune phase can retry authority, lock, placement, or reverse work many
    // times; measuring there both double-counted one search and charged those
    // unrelated waits to continuation_search.
    lib_assert(!context.search_timing_recorded &&
                 context.search_started_ns != 0,
               "Stage2 continuation completion timing was recorded twice");
    auto& search_timing =
      storage_owner_stage2_phase_timing_[static_cast<size_t>(
        StorageOwnerStage2TimingPhase::continuation_search)];
    search_timing.attempts.fetch_add(1, std::memory_order_relaxed);
    search_timing.task_attempts.fetch_add(
      context.tasks.size(), std::memory_order_relaxed);
    search_timing.elapsed_ns.fetch_add(
      steady_now_ns() - context.search_started_ns,
      std::memory_order_relaxed);
    context.search_timing_recorded = true;
    lib_assert(context.continued_candidates_by_task.size() ==
                 context.tasks.size(),
               "Stage2 continuation lost context/task correlation");
    for (const vec<RemotePtr>& candidates :
         context.continued_candidates_by_task) {
      lib_assert(candidates.size() <= construction_width,
                 "stage2 continuation exceeded construction width L");
    }

    // The Stage1 owner remains the sole owner of one logical beam. Remote
    // homes execute only the selected expansion plus same-home scoring; they
    // never restart or recursively advance a shard-local search. There is
    // therefore no remote-search ACK mask to wait for.
    constexpr u64 expected_mask = 0;
    const auto transition =
      states.begin_remote_search(context.handle, expected_mask);
    lib_assert(transition == Stage2EventResult::phase_advanced,
               "stage2 failed to enter remote_search_pending");
    return Stage2SearchAdvanceResult::complete;
  };

  const auto defer_stage2_retry = [](Stage2Context& context) {
    const auto retry_at = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(1);
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      task.retry_not_before = retry_at;
    }
  };

  enum class ReconcilePollResult : u8 {
    waiting,
    complete,
    retry,
  };

  const auto cancel_context_reconcile = [&, this](Stage2Context& context) {
    for (const Stage2ReconcileChunk& chunk :
         context.reconcile_batch.chunks()) {
      if (!chunk.complete && chunk.request_id != 0) {
        cancel_peer_rpc_response(chunk.request_id);
      }
    }
    context.reconcile_batch.clear();
  };

  const auto finish_reverse_prepare_timing = [&](Stage2Context& context) {
    if (!context.reverse_prepare_timing_active) return;
    auto& timing = storage_owner_stage2_phase_timing_[
      static_cast<size_t>(StorageOwnerStage2TimingPhase::reverse_prepare)];
    timing.attempts.fetch_add(1, std::memory_order_relaxed);
    timing.task_attempts.fetch_add(
      context.tasks.size(), std::memory_order_relaxed);
    timing.elapsed_ns.fetch_add(
      steady_now_ns() - context.reverse_prepare_started_ns,
      std::memory_order_relaxed);
    context.reverse_prepare_timing_active = false;
    context.reverse_prepare_started_ns = 0;
  };

  const auto build_reconcile_barrier_ops = [&] (
      Stage2Context& context, Stage2ReconcileBarrier barrier,
      vec<ReconcileReverseOp>& selected_ops) {
    vec<ReconcileReverseOp> promotion_ops;
    vec<ReconcileReverseOp> stable_ops;
    vec<ReconcileReverseOp> removal_ops;
    const size_t reserve_hint =
      static_cast<size_t>(config.R) * context.tasks.size();
    promotion_ops.reserve(context.tasks.size());
    stable_ops.reserve(reserve_hint);
    removal_ops.reserve(reserve_hint);
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.reverse_reconciled) continue;
      lib_assert(item < context.live_stage2_neighbors_by_task.size(),
                 "Stage2 reconcile lost its persisted planner input");
      if (!append_stage2_reconcile_ops(
            task,
            span<const RemotePtr>{
              context.live_stage2_neighbors_by_task[item]},
            promotion_ops, stable_ops, removal_ops)) {
        return false;
      }
    }
    switch (barrier) {
      case Stage2ReconcileBarrier::install:
        // This ordering is semantic, not merely a packing preference. The
        // target-run packer below groups a target without reordering its ops,
        // so every receiver executes all ordinary stable proposals before any
        // mandatory promotion for that target and audits the mandatory set on
        // the resulting final adjacency before ACKing the one install barrier.
        stable_ops.insert(stable_ops.end(),
                          std::make_move_iterator(promotion_ops.begin()),
                          std::make_move_iterator(promotion_ops.end()));
        selected_ops = std::move(stable_ops);
        return true;
      case Stage2ReconcileBarrier::removal:
        selected_ops = std::move(removal_ops);
        return true;
      case Stage2ReconcileBarrier::none:
        return false;
    }
    return false;
  };

  // Apply the local part immediately, then retain one immutable copy of every
  // remote chunk in the context.  Posting is intentionally separate so one
  // peer with no send credit cannot prevent all other peers from being posted.
  const auto begin_reconcile_barrier = [&, this](
      Stage2Context& context, Stage2ReconcileBarrier barrier,
      span<const ReconcileReverseOp> ops) {
    vec<ReconcileReverseOp> local_ops;
    dense_hashmap_t<u32, vec<ReconcileReverseOp>> remote_ops;
    local_ops.reserve(ops.size());
    for (const ReconcileReverseOp& op : ops) {
      const RemotePtr target{op.target_raw};
      if (target.is_null() || target.memory_node() >= num_storage_nodes_) {
        return false;
      }
      if (local_shard(target.memory_node())) {
        local_ops.push_back(op);
      } else {
        remote_ops[target.memory_node()].push_back(op);
      }
    }

    vec<protocol::ReconcileReverseResult> local_results;
    if (!local_ops.empty() &&
        !reconcile_local_reverse_ops(
          span<const ReconcileReverseOp>{local_ops}, config,
          local_results)) {
      return false;
    }
    if (local_results.size() != local_ops.size()) return false;
    for (size_t index = 0; index < local_ops.size(); ++index) {
      if (!memory_node_storage_owner_index_detail::
            reconcile_reverse_postcondition_holds(
              local_ops[index], local_results[index])) {
        return false;
      }
    }

    const size_t payload_bytes = peer_rpc_runtime_.message_bytes -
      sizeof(protocol::PeerRpcHeader);
    const u32 wire_capacity = static_cast<u32>(
      payload_bytes / sizeof(ReconcileReverseOp));
    lib_assert(wire_capacity != 0,
               "peer RPC slot cannot hold a reverse reconciliation op");
    context.reconcile_batch.begin(context.handle, barrier);
    for (const auto& [target_shard, peer_ops] : remote_ops) {
      const auto packed = pack_stage2_reconcile_target_runs(
        span<const ReconcileReverseOp>{peer_ops}, wire_capacity);
      // One task can contribute an ordinary install and its mandatory
      // promotion to the same target, so a target run is bounded by twice the
      // storage_owner_batch_max. Treat an impossible oversize run as a semantic
      // retry instead of splitting it into RPCs whose completion order could
      // expose promotion before ordinary work or ACK only a subset of the
      // mandatory certificate set.
      if (!packed.has_value()) return false;
      for (const auto& chunk_ops : *packed) {
        const size_t count = chunk_ops.size();
        const bool appended = context.reconcile_batch.append_chunk(
          allocate_peer_request_id(), target_shard,
          std::span<const ReconcileReverseOp>{chunk_ops.data(), count});
        lib_assert(appended,
                   "bounded Stage2 reconcile chunk could not be persisted");
      }
    }
    context.finalize_subphase = stage2_reconcile_wait_subphase(barrier);
    return true;
  };

  // One nonblocking transport pass over every peer/chunk.  Retries preserve
  // request_id and the exact byte payload.  An ACK is accepted only when both
  // the context generation/barrier epoch and every per-op postcondition match.
  const auto poll_reconcile_barrier = [&, this](
      Stage2Context& context) -> ReconcilePollResult {
    if (!context.reconcile_batch.active() ||
        context.reconcile_batch.context() != context.handle) {
      return ReconcilePollResult::retry;
    }
    constexpr u32 kTransportAttempts = 3;
    const u32 epoch = context.reconcile_batch.epoch();
    vec<Stage2ReconcileChunk>& chunks =
      context.reconcile_batch.chunks();
    for (size_t chunk_index = 0; chunk_index < chunks.size();
         ++chunk_index) {
      Stage2ReconcileChunk& chunk = chunks[chunk_index];
      if (chunk.complete) continue;
      if (!chunk.correlates(context.handle, epoch)) {
        return ReconcilePollResult::retry;
      }

      u64 now_ns = steady_now_ns();
      if (!chunk.attempt_active) {
        if (chunk.attempts_started == kTransportAttempts) {
          return ReconcilePollResult::retry;
        }
        ++chunk.attempts_started;
        chunk.attempt_active = true;
        chunk.posted = false;
        chunk.deadline_ns = now_ns + rpc_timeout_ns;
      }

      const std::span<const ReconcileReverseOp> payload =
        context.reconcile_batch.payload(chunk);
      if (payload.size() != chunk.item_count) {
        return ReconcilePollResult::retry;
      }
      if (!chunk.posted) {
        const size_t request_bytes =
          protocol::reconcile_reverse_request_bytes(chunk.item_count);
        chunk.posted = try_post_peer_rpc_request_attempt(
          chunk.target_shard, PeerRpcType::reconcile_reverse_request,
          PeerRpcType::reconcile_reverse_response, chunk.request_id,
          chunk.item_count, payload.data(),
          payload.size_bytes(), request_bytes,
          PeerRpcSendClass::graph_update);
        now_ns = steady_now_ns();
        if (!chunk.posted && now_ns >= chunk.deadline_ns) {
          cancel_peer_rpc_response(chunk.request_id);
          chunk.attempt_active = false;
        }
        if (!chunk.posted) continue;
      }

      protocol::PeerRpcHeader response_header{};
      PeerResponseLease response_lease{};
      const TryPeerResponse response = try_consume_peer_rpc_response(
        chunk.request_id, chunk.target_shard,
        PeerRpcType::reconcile_reverse_response, chunk.item_count,
        response_header, reverse_response_payload, response_lease);
      now_ns = steady_now_ns();
      if (response == TryPeerResponse::pending) {
        if (now_ns >= chunk.deadline_ns) {
          cancel_peer_rpc_response(chunk.request_id);
          chunk.attempt_active = false;
          chunk.posted = false;
        }
        continue;
      }
      if (response == TryPeerResponse::stale) {
        chunk.attempt_active = false;
        chunk.posted = false;
        continue;
      }

      const size_t expected_bytes =
        protocol::reconcile_reverse_response_bytes(chunk.item_count);
      bool valid = response == TryPeerResponse::success &&
        reverse_response_payload.size() == expected_bytes &&
        response_header.magic == protocol::kPeerRpcMagic &&
        response_header.version == protocol::kPeerRpcVersion &&
        response_header.type == static_cast<u32>(
          PeerRpcType::reconcile_reverse_response) &&
        response_header.source_shard == chunk.target_shard &&
        response_header.item_count == chunk.item_count &&
        response_header.request_id == chunk.request_id &&
        response_header.status == static_cast<u32>(protocol::InsertStatus::ok) &&
        response_header.reserved == 0;
      if (valid) {
        const auto* results = protocol::reconcile_reverse_results(
          reverse_response_payload.data());
        for (u32 index = 0; index < chunk.item_count; ++index) {
          const protocol::ReconcileReverseResult& result = results[index];
          valid = result.accepted <= 1 && result.replaced <= 1 &&
            result.removed <= 1 && result.stale <= 1 &&
            result.reserved == 0 &&
            memory_node_storage_owner_index_detail::
              reconcile_reverse_postcondition_holds(payload[index], result);
          if (!valid) break;
        }
      }

      if (valid) {
        lib_assert(acknowledge_peer_rpc_response(response_lease),
                   "validated async reconcile response lost its lease");
        lib_assert(context.reconcile_batch.mark_complete(
                     chunk_index, context.handle, epoch),
                   "late reconcile ACK crossed a context/barrier fence");
      } else {
        if (response_lease.valid()) {
          lib_assert(rearm_peer_rpc_response(response_lease),
                     "invalid async reconcile response lost its lease");
        }
        chunk.attempt_active = false;
        chunk.posted = false;
      }
    }
    return context.reconcile_batch.complete()
      ? ReconcilePollResult::complete
      : ReconcilePollResult::waiting;
  };

  // Placement authority and centroid membership remain synchronous in this
  // patch.  They execute only after both reconciliation barriers have
  // completed, and may safely reacquire any idle search lane for their local
  // snapshot/control scratch.
  const auto finish_stage2_after_reconcile = [&, this](
      Stage2Context& context) {
    Stage2PhaseAttemptTimer placement_timer(
      storage_owner_stage2_phase_timing_[static_cast<size_t>(
        StorageOwnerStage2TimingPhase::placement_authority)],
      context.tasks.size());
    vec<u32> placement_authorities;
    vec<protocol::AuthorityPlacementItem> placements;
    vec<size_t> placement_task_indices;
    placement_authorities.reserve(context.tasks.size());
    placements.reserve(context.tasks.size());
    placement_task_indices.reserve(context.tasks.size());
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      const StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.placement_committed) continue;
      placement_authorities.push_back(task.authority_shard);
      placement_task_indices.push_back(item);
      placements.push_back(protocol::AuthorityPlacementItem{
        .token = {
          .source_client = task.source_client,
          .item_index = task.operation_item_index,
          .client_batch_id = task.operation_batch_id,
        },
        .id = task.id,
        .generation = task.generation,
        .expected_raw = task.target.raw_address,
        .desired_raw = task.final_target.raw_address,
        .expected_placement_version = task.initial_placement_version,
      });
    }
    vec<protocol::AuthorityPlacementResult> placement_results;
    if (!placements.empty() && !relocate_batch_via_authority(
          span<const u32>{placement_authorities},
          span<const protocol::AuthorityPlacementItem>{placements},
          placement_results, config)) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      defer_stage2_retry(context);
      return false;
    }
    lib_assert(placement_results.size() == placement_task_indices.size(),
               "batched Stage2 placement lost a task result");
    for (size_t slot = 0; slot < placement_task_indices.size(); ++slot) {
      StorageOwnerMaintenanceTask& task =
        context.tasks[placement_task_indices[slot]];
      const protocol::AuthorityPlacementResult& result =
        placement_results[slot];
      const auto status = static_cast<protocol::AuthorityPlacementStatus>(
        result.status);
      if (status == protocol::AuthorityPlacementStatus::busy) {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        defer_stage2_retry(context);
        return false;
      }
      if (status == protocol::AuthorityPlacementStatus::stale) {
        if (!complete_stale_stage2(task)) return false;
        task.maintenance_sequence = 0;
        continue;
      }
      lib_assert(
        status == protocol::AuthorityPlacementStatus::committed ||
          status == protocol::AuthorityPlacementStatus::replay,
        "authority rejected a structurally valid Stage2 placement token");
      const u64 expected_resulting_version =
        task.final_target == task.target
          ? task.initial_placement_version
          : task.initial_placement_version + 1;
      lib_assert(result.resulting_placement_version ==
                   expected_resulting_version,
                 "authority placement returned an unexpected version");
      task.placement_committed = true;
    }

    size_t ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      if (context.tasks[item].maintenance_sequence == 0) continue;
      if (ready != item) {
        context.tasks[ready] = std::move(context.tasks[item]);
        context.targets[ready] = std::move(context.targets[item]);
      }
      ++ready;
    }
    context.tasks.resize(ready);
    context.targets.resize(ready);

    vec<CentroidMembershipOp> centroid_adds;
    centroid_adds.reserve(context.tasks.size());
    for (const StorageOwnerMaintenanceTask& task : context.tasks) {
      if (!task.centroid_committed) {
        centroid_adds.push_back(make_centroid_membership_op(
          task, task.final_target, CentroidMembershipKind::add));
      }
    }
    if (!centroid_adds.empty() &&
        !apply_centroid_membership_fanout_and_wait(
          span<const CentroidMembershipOp>{centroid_adds}, config)) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      defer_stage2_retry(context);
      return false;
    }
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      task.centroid_committed = true;
    }

    // Publish final membership before retiring a migrated Stage1 source.
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (task.final_target == task.target || task.allocation_settled) {
        continue;
      }
      lib_assert(migrated_source_tombstone_allowed(
                   task.placement_committed, task.centroid_committed),
                 "migrated source tombstoned before final centroid publication");
      const u64 source_header = load_local_node_header_acquire(task.target);
      lib_assert((source_header & VamanaNode::HEADER_CENTROID_ACCOUNTED) == 0,
                 "Stage1 source was counted before final placement");
      (void)mark_node_deleted(task.target, task.generation);
      if (!settle_dynamic_allocation(task)) {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        defer_stage2_retry(context);
        return false;
      }
      task.allocation_settled = true;
    }

    context.completion_handoff_started_ns = steady_now_ns();
    context.finalize_subphase = Stage2FinalizeSubphase::prepare;
    constexpr u64 expected_mask = 0;
    const Stage2EventResult transition =
      states.begin_reverse(context.handle, expected_mask);
    lib_assert(transition == Stage2EventResult::phase_advanced ||
                 transition == Stage2EventResult::ready_to_finalize,
               "Stage2 finalization failed to enter reverse_pending");
    return true;
  };

  const auto reset_reconcile_for_semantic_retry = [&](Stage2Context& context) {
    cancel_context_reconcile(context);
    context.finalize_subphase = Stage2FinalizeSubphase::prepare;
    storage_owner_maintenance_pressure_yields_.fetch_add(
      1, std::memory_order_relaxed);
    defer_stage2_retry(context);
  };

  const auto start_reconcile_barrier = [&] (
      Stage2Context& context, Stage2ReconcileBarrier barrier) {
    vec<ReconcileReverseOp> ops;
    if (!build_reconcile_barrier_ops(context, barrier, ops) ||
        !begin_reconcile_barrier(
          context, barrier, span<const ReconcileReverseOp>{ops})) {
      reset_reconcile_for_semantic_retry(context);
      return false;
    }
    return true;
  };

  // Drives as many zero-remote barriers as possible, but never waits. The
  // install barrier is one receiver-side per-target transaction: ordinary
  // stable additions run first, mandatory promotions run last, and the final
  // adjacency is audited before ACK. Only then may obsolete Stage1 bridges be
  // removed by the second barrier.
  const auto advance_reconcile_pipeline = [&] (Stage2Context& context) {
    for (;;) {
      const ReconcilePollResult poll = poll_reconcile_barrier(context);
      if (poll == ReconcilePollResult::waiting) return false;
      if (poll == ReconcilePollResult::retry) {
        reset_reconcile_for_semantic_retry(context);
        return false;
      }

      const Stage2ReconcileBarrier completed =
        context.reconcile_batch.barrier();
      cancel_context_reconcile(context);
      switch (completed) {
        case Stage2ReconcileBarrier::install:
          for (StorageOwnerMaintenanceTask& task : context.tasks) {
            if (!task.reverse_reconciled) {
              task.stage2_promotion_committed = true;
            }
          }
          if (!start_reconcile_barrier(
                context, Stage2ReconcileBarrier::removal)) {
            return false;
          }
          break;
        case Stage2ReconcileBarrier::removal: {
          for (StorageOwnerMaintenanceTask& task : context.tasks) {
            task.reverse_reconciled = true;
          }
          context.finalize_subphase =
            Stage2FinalizeSubphase::placement_ready;
          finish_reverse_prepare_timing(context);
          return true;
        }
        case Stage2ReconcileBarrier::none:
          reset_reconcile_for_semantic_retry(context);
          return false;
      }
    }
  };

  const auto prepare_stage2_reverse = [&](Stage2Context& context) {
    if (context.finalize_subphase ==
        Stage2FinalizeSubphase::placement_ready) {
      return finish_stage2_after_reconcile(context);
    }
    if (context.finalize_subphase != Stage2FinalizeSubphase::prepare) {
      const Stage2FinalizeSubphase expected =
        stage2_reconcile_wait_subphase(context.reconcile_batch.barrier());
      if (expected != context.finalize_subphase) {
        reset_reconcile_for_semantic_retry(context);
        return false;
      }
      (void)advance_reconcile_pipeline(context);
      // Even when the removal ACK arrived in this pass, yield at the semantic
      // boundary.  The scheduler releases the lane; placement reacquires one
      // on the next pass without monopolizing search scratch during ACK waits.
      return false;
    }
    Stage2PhaseAttemptTimer phase_timer(
      storage_owner_stage2_phase_timing_[static_cast<size_t>(
        StorageOwnerStage2TimingPhase::freeze_prune)],
      context.tasks.size());
    // First finish every node record. No reverse edge or directory entry is
    // allowed to expose a destination whose vector/graph/PQ record is partial.
    vec<u32> gate_authorities;
    vec<protocol::AuthorityPlacementItem> gates;
    vec<size_t> gate_task_indices;
    gate_authorities.reserve(context.tasks.size());
    gates.reserve(context.tasks.size());
    gate_task_indices.reserve(context.tasks.size());
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      const StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.maintenance_sequence == 0) continue;
      const RemotePtr gate_target = task.placement_committed
        ? task.final_target : task.target;
      const u64 gate_version = task.initial_placement_version +
        static_cast<u64>(task.placement_committed &&
                         task.final_target != task.target);
      gate_authorities.push_back(task.authority_shard);
      gate_task_indices.push_back(item);
      gates.push_back(protocol::AuthorityPlacementItem{
        .token = {
          .source_client = task.source_client,
          .item_index = task.operation_item_index,
          .client_batch_id = task.operation_batch_id,
        },
        .id = task.id,
        .generation = task.generation,
        .expected_raw = gate_target.raw_address,
        .desired_raw = gate_target.raw_address,
        .expected_placement_version = gate_version,
      });
    }
    vec<protocol::AuthorityPlacementResult> gate_results;
    if (!gates.empty() && !relocate_batch_via_authority(
          span<const u32>{gate_authorities},
          span<const protocol::AuthorityPlacementItem>{gates},
          gate_results, config)) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      defer_stage2_retry(context);
      return false;
    }
    lib_assert(gate_results.size() == gate_task_indices.size(),
               "batched Stage2 gate lost a task result");
    // Do not freeze the first items in a batch if a later authority lease is
    // still busy.  A retry must cross this batch boundary as one unit.
    for (const protocol::AuthorityPlacementResult& gate_result :
         gate_results) {
      const auto status = static_cast<
        protocol::AuthorityPlacementStatus>(gate_result.status);
      if (status == protocol::AuthorityPlacementStatus::busy) {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        defer_stage2_retry(context);
        return false;
      }
    }

    const vec<RemotePtr> durable_route_entries =
      local_centroid_route_entries();
    if (snapshots_by_task.size() < context.tasks.size()) {
      snapshots_by_task.resize(context.tasks.size());
    }
    if (snapshot_task_active.size() < context.tasks.size()) {
      snapshot_task_active.resize(context.tasks.size());
    }
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      snapshot_candidates_by_task[item].clear();
      snapshots_by_task[item].clear();
      snapshot_task_active[item] = false;
    }

    for (size_t gate_slot = 0;
         gate_slot < gate_task_indices.size(); ++gate_slot) {
      const size_t item = gate_task_indices[gate_slot];
      StorageOwnerMaintenanceTask& task = context.tasks[item];

      // Arm publishes a runnable descriptor before the foreground authority
      // commit. Use a no-op placement CAS as the gate: while that mutation's
      // lease is pending it returns busy; after abort it returns stale; only a
      // committed current generation may perform any Stage2-visible graph
      // mutation. On retries after a real relocation, validate the resulting
      // physical pointer/version instead of the original Stage1 address.
      const u64 gate_version = task.initial_placement_version +
        static_cast<u64>(task.placement_committed &&
                         task.final_target != task.target);
      const protocol::AuthorityPlacementResult& gate_result =
        gate_results[gate_slot];
      const auto gate_status = static_cast<
        service::storage_owner::AuthorityPlacementStatus>(
          gate_result.status);
      lib_assert(gate_status !=
                   service::storage_owner::AuthorityPlacementStatus::busy,
                 "Stage2 authority preflight lost a busy result");
      if (gate_status ==
          service::storage_owner::AuthorityPlacementStatus::stale) {
        if (!task.stage1_receipt_released) {
          if (!release_resolved_local_stage1_receipt(task, config)) {
            return false;
          }
          task.stage1_receipt_released = true;
        }
        if (!complete_stale_stage2(task)) return false;
        task.maintenance_sequence = 0;
        continue;
      }
      if (gate_status !=
            service::storage_owner::AuthorityPlacementStatus::committed &&
          gate_status !=
            service::storage_owner::AuthorityPlacementStatus::replay) {
        storage_owner_maintenance_failed_.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
      lib_assert(gate_result.resulting_placement_version == gate_version,
                 "authority Stage2 gate returned an unexpected version");
      if (!task.stage1_receipt_released) {
        if (!release_resolved_local_stage1_receipt(task, config)) {
          return false;
        }
        task.stage1_receipt_released = true;
      }

      const RemotePtr current_physical = task.placement_committed
        ? task.final_target : task.target;
      const StableNodeSnapshotState current_physical_state =
        storage_owner_physical_node_state(
          task.id, task.generation, current_physical);
      if (current_physical_state == StableNodeSnapshotState::retryable) {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        defer_stage2_retry(context);
        return false;
      }
      if (current_physical_state == StableNodeSnapshotState::terminal) {
        if (!complete_stale_stage2(task)) return false;
        task.maintenance_sequence = 0;
        continue;
      }

      if (!task.stage2_prepared) {
        lib_assert(item < context.continued_candidates_by_task.size(),
                   "Stage2 context lost its continuation candidates");
        // Freeze the graph mutation plane at the same locked boundary as the
        // rebase snapshot. Queries still traverse this record, but every
        // ordinary reverse mutation retries after observing FROZEN. Thus an
        // edge ACKed before this boundary is in observed_adjacency and no edge
        // can be ACKed in the snapshot-to-publication window.
        GraphAdjacency observed_adjacency;
        bool target_current = false;
        bool adjacency_retryable = false;
        if (task.stage2_source_frozen) {
          // A different item in this batched context may have hit a fallible
          // authority/RPC boundary after this source was frozen.  Re-entry is
          // an ordinary state-machine retry, not a stale-node condition.
          const u64 frozen_header =
            load_local_node_header_acquire(task.target);
          const byte_t* record = index_buffer_.get_full_buffer() +
            task.target.byte_offset();
          const bool frozen_identity_current =
            VamanaNode::header_incarnation(frozen_header) ==
              task.target.incarnation() &&
            *reinterpret_cast<const node_t*>(
              record + VamanaNode::offset_id()) == task.id &&
            *reinterpret_cast<const u32*>(
              record + VamanaNode::offset_generation()) == task.generation &&
            (frozen_header & VamanaNode::HEADER_PROVISIONAL) != 0 &&
            (frozen_header & VamanaNode::HEADER_STAGE2_FROZEN) != 0 &&
            (frozen_header & (VamanaNode::HEADER_DELETED |
                              VamanaNode::HEADER_RETIRING)) == 0;
          if (frozen_identity_current) {
            if (!read_graph_adjacency(task.target, observed_adjacency)) {
              adjacency_retryable = true;
            } else {
              target_current = !observed_adjacency.deleted &&
                observed_adjacency.generation == task.generation;
            }
          }
        } else {
          lib_assert(storage_node_pointer_addressable(task.target),
                     "Stage2 freeze target is structurally invalid");
          const IncarnationLockResult target_lock =
            try_lock_node(task.target);
          if (target_lock == IncarnationLockResult::busy) {
            storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            defer_stage2_retry(context);
            return false;
          }
          if (target_lock == IncarnationLockResult::stale) {
            if (!complete_stale_stage2(task)) return false;
            task.maintenance_sequence = 0;
            continue;
          }
          const u64 locked_header =
            load_local_node_header_acquire(task.target);
          const byte_t* locked_record = index_buffer_.get_full_buffer() +
            task.target.byte_offset();
          const node_t locked_id = *reinterpret_cast<const node_t*>(
            locked_record + VamanaNode::offset_id());
          const u32 locked_generation = *reinterpret_cast<const u32*>(
            locked_record + VamanaNode::offset_generation());
          const bool locked_identity_matches =
            VamanaNode::header_incarnation(locked_header) ==
              task.target.incarnation() &&
            locked_id == task.id &&
            locked_generation == task.generation;
          if (locked_identity_matches &&
              (locked_header & (VamanaNode::HEADER_DELETED |
                                VamanaNode::HEADER_RETIRING |
                                VamanaNode::HEADER_STAGE2_FROZEN)) == 0) {
            lib_assert((locked_header & VamanaNode::HEADER_PROVISIONAL) != 0,
                       "Stage1 source lost PROVISIONAL before Stage2 freeze");
          }
          const bool locked_target_current =
            locked_identity_matches &&
            (locked_header & VamanaNode::HEADER_PROVISIONAL) != 0 &&
            (locked_header & (VamanaNode::HEADER_DELETED |
                              VamanaNode::HEADER_RETIRING |
                              VamanaNode::HEADER_STAGE2_FROZEN)) == 0;
          if (locked_target_current) {
            if (!read_graph_adjacency(task.target, observed_adjacency)) {
              adjacency_retryable = true;
            } else {
              target_current = !observed_adjacency.deleted &&
                observed_adjacency.generation == task.generation;
            }
          }
          if (target_current) {
            auto* header_ptr = reinterpret_cast<u64*>(
              index_buffer_.get_full_buffer() +
              vamana::StorageLayoutResolver::header(task.target).offset);
            std::atomic_ref<u64>(*header_ptr).fetch_or(
              static_cast<u64>(VamanaNode::HEADER_STAGE2_FROZEN),
              std::memory_order_acq_rel);
            task.stage2_source_frozen = true;
          }
          unlock_node(task.target);
        }
        if (adjacency_retryable) {
          storage_owner_maintenance_pressure_yields_.fetch_add(
            1, std::memory_order_relaxed);
          defer_stage2_retry(context);
          return false;
        }
        if (!target_current) {
          if (!complete_stale_stage2(task)) return false;
          task.maintenance_sequence = 0;
          continue;
        }
        task.stage2_protected_children = observed_adjacency.provisional;
        lib_assert(task.stage2_protected_children.empty(),
                   "query-ineligible Stage1 source accepted a protected child");

        // Build this task's exact ordered candidate set while its source is
        // frozen. The batch-wide wave below reads every shared physical
        // record once, but RobustPrune still receives this task's own order.
        lib_assert(!task.stage1_prune_deferred ||
                     task.stage1_prune_materialized,
                   "deferred Stage1 prune was not materialized");
        const span<const RemotePtr> local_prune_seed =
          task.stage1_prune_deferred
            ? span<const RemotePtr>{task.stage1_pruned_neighbors}
            : span<const RemotePtr>{task.stage1_base_neighbors};
        vec<RemotePtr> observed_reverse_delta;
        const span<const RemotePtr> observed_rebase = [&]() {
          if (!task.stage1_prune_deferred) {
            return span<const RemotePtr>{observed_adjacency.stable};
          }
          observed_reverse_delta =
            memory_node_storage_owner_index_detail::
              stage2_observed_reverse_delta(
                span<const RemotePtr>{observed_adjacency.stable},
                span<const RemotePtr>{task.stage1_base_neighbors});
          return span<const RemotePtr>{observed_reverse_delta};
        }();
        snapshot_candidates_by_task[item] = merge_stage2_rebase_candidates(
          span<const RemotePtr>{
            context.continued_candidates_by_task[item]},
          local_prune_seed, observed_rebase);
        // Preserve one small, current routing backbone in the final prune
        // candidate set.  This is not a static anchor plane: entries are the
        // same versioned live centroid representatives already maintained by
        // dynamic membership, and RobustPrune remains free to reject them.
        for (const RemotePtr route_entry : durable_route_entries) {
          if (route_entry == task.target ||
              std::find(snapshot_candidates_by_task[item].begin(),
                        snapshot_candidates_by_task[item].end(),
                        route_entry) !=
                snapshot_candidates_by_task[item].end()) {
            continue;
          }
          snapshot_candidates_by_task[item].push_back(route_entry);
        }
        snapshot_task_active[item] = true;
      }

    }

    if (!materialize_stable_snapshot_wave(context.tasks.size())) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      defer_stage2_retry(context);
      return false;
    }
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      if (!snapshot_task_active[item]) continue;
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.maintenance_sequence == 0) continue;
      hashset_t<RemotePtr> skip;
      skip.insert(task.target);
      task.stage2_neighbors = robust_prune_snapshot_refs_cpu(
        context.targets[item].vector_data.data(),
        VamanaNode::vector_dtype(),
        span<const NodeSnapshot* const>{snapshots_by_task[item]}, skip,
        config, config.R);
      lib_assert(task.stage2_neighbors.size() <= config.R,
                 "online Stage2 finalization exceeded graph degree");
      task.final_home =
        memory_node_storage_owner_index_detail::choose_min_cross_shard_home(
          span<const RemotePtr>{task.stage2_neighbors},
          num_storage_nodes_, task.target.memory_node());
      task.stage2_revalidated_home = task.final_home;
      task.stage2_prepared = true;
    }

    // Allocation and outgoing publication run only after every newly frozen
    // task has consumed the shared authoritative snapshot wave. A migrated
    // destination is still unreachable through the authority directory here;
    // an in-place destination clears FROZEN only after its adjacency write.
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.maintenance_sequence == 0) continue;
      lib_assert(task.stage2_prepared,
                 "Stage2 outgoing publication bypassed frozen prune");
      if (task.final_target.is_null()) {
        if (task.final_home == task.target.memory_node()) {
          task.final_target = task.target;
        } else {
          const protocol::DynamicNodeControlItem allocation{
            .token = {
              .source_client = task.source_client,
              .item_index = task.operation_item_index,
              .client_batch_id = task.operation_batch_id,
            },
            .node_raw = task.target.raw_address,
            .id = task.id,
            .generation = task.generation,
            .authority_shard = task.authority_shard,
            .action = static_cast<u32>(
              protocol::DynamicNodeControlAction::allocate),
          };
          protocol::DynamicNodeControlResult allocation_result;
          if (!control_dynamic_node_on_shard(
                task.final_home, allocation, allocation_result, config)) {
            storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            defer_stage2_retry(context);
            return false;
          }
          if (static_cast<protocol::DynamicNodeControlStatus>(
                allocation_result.status) !=
                protocol::DynamicNodeControlStatus::ok) {
            storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            defer_stage2_retry(context);
            return false;
          }
          const RemotePtr allocated{allocation_result.node_raw};
          if (allocated.is_null() ||
              allocated.memory_node() != task.final_home) {
            storage_owner_maintenance_failed_.fetch_add(
              1, std::memory_order_relaxed);
            return false;
          }
          task.final_target = allocated;
        }
      }

      if (!task.outgoing_committed) {
        if (task.final_target != task.target) {
          vec<element_t> components(VamanaNode::DIM);
          decode_storage_vector_to_float(
            context.targets[item].vector_data.data(),
            VamanaNode::vector_dtype(), VamanaNode::DIM,
            components.data());
          // The allocation header remains the publication lock until vector,
          // graph and PQ code are complete. The authority directory still
          // names the frozen source, so ordinary graph updates cannot discover
          // this destination during materialization.
          write_new_node_on_shard(
            task.final_target, task.id,
            span<const element_t>{components}, task.stage2_neighbors,
            task.generation, false);
          lib_assert(task.stage2_protected_children.empty(),
                     "migration cannot bypass protected Stage1 children");
        } else {
          lib_assert(storage_node_pointer_addressable(task.target),
                     "Stage2 publish target is structurally invalid");
          const IncarnationLockResult target_lock =
            try_lock_node(task.target);
          if (target_lock == IncarnationLockResult::busy) {
            storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            defer_stage2_retry(context);
            return false;
          }
          if (target_lock == IncarnationLockResult::stale) {
            if (!complete_stale_stage2(task)) return false;
            task.maintenance_sequence = 0;
            continue;
          }
          const u64 locked_header =
            load_local_node_header_acquire(task.target);
          const byte_t* locked_record = index_buffer_.get_full_buffer() +
            task.target.byte_offset();
          const bool current =
            VamanaNode::header_incarnation(locked_header) ==
              task.target.incarnation() &&
            *reinterpret_cast<const node_t*>(
              locked_record + VamanaNode::offset_id()) == task.id &&
            *reinterpret_cast<const u32*>(
              locked_record + VamanaNode::offset_generation()) ==
                task.generation &&
            (locked_header & VamanaNode::HEADER_STAGE2_FROZEN) != 0 &&
            (locked_header & (VamanaNode::HEADER_DELETED |
                              VamanaNode::HEADER_RETIRING)) == 0;
          if (!current) {
            unlock_node(task.target);
            if (!complete_stale_stage2(task)) return false;
            task.maintenance_sequence = 0;
            continue;
          }
          GraphAdjacency adjacency;
          lib_assert(read_graph_adjacency(task.target, adjacency),
                     "Stage2 target adjacency became unreadable");
          write_graph_adjacency(
            task.target, task.stage2_neighbors,
            task.stage2_protected_children,
            task.generation, false);
          auto* header_ptr = reinterpret_cast<u64*>(
            index_buffer_.get_full_buffer() +
            vamana::StorageLayoutResolver::header(task.target).offset);
          std::atomic_ref<u64>(*header_ptr).fetch_and(
            ~static_cast<u64>(VamanaNode::HEADER_PROVISIONAL |
                              VamanaNode::HEADER_STAGE2_FROZEN),
            std::memory_order_acq_rel);
          unlock_node(task.target);
          task.stage2_source_frozen = false;
        }
        task.outgoing_committed = true;
        task.stage2_plan_sealed = true;
      }
    }

    size_t ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      if (context.tasks[item].maintenance_sequence == 0) continue;
      if (ready != item) {
        context.tasks[ready] = std::move(context.tasks[item]);
        context.targets[ready] = std::move(context.targets[item]);
      }
      ++ready;
    }
    context.tasks.resize(ready);
    context.targets.resize(ready);

    phase_timer.finish();
    if (!context.reverse_prepare_timing_active) {
      context.reverse_prepare_started_ns = steady_now_ns();
      context.reverse_prepare_timing_active = true;
    }

    // There is exactly one outgoing-graph mutation boundary: the source lock
    // above freezes concurrent additions, captures them, prunes the complete
    // continuation union, and publishes that result. A second freeze/prune of
    // the same sealed plan cannot make parent liveness persistent; it only
    // repeats O((L+R)*R*D) work. Reachability churn is instead handled by the
    // independently ACKed promotion certificate below.
    for (const StorageOwnerMaintenanceTask& task : context.tasks) {
      lib_assert(task.stage2_plan_sealed && task.outgoing_committed &&
                   !task.final_target.is_null(),
                 "Stage2 reached parent validation with an open placement plan");
      lib_assert(task.final_target != task.target ||
                   !task.stage2_source_frozen,
                 "in-place Stage2 publication left its source frozen");
    }

    // A sealed outgoing edge may retire before reverse reconciliation. It
    // remains part of the already-published adjacency (normal cleanup owns
    // that stale edge), but it cannot be selected as the mandatory promotion
    // parent. Revalidate all sealed neighbors in one shared wave and retain a
    // worker-local, order-preserving planner view. This also guarantees that a
    // retry advances past a dead first choice instead of livelocking on it.
    if (snapshots_by_task.size() < context.tasks.size()) {
      snapshots_by_task.resize(context.tasks.size());
    }
    if (snapshot_task_active.size() < context.tasks.size()) {
      snapshot_task_active.resize(context.tasks.size());
    }
    if (live_stage2_neighbors_by_task.size() < context.tasks.size()) {
      live_stage2_neighbors_by_task.resize(context.tasks.size());
    }
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      snapshot_candidates_by_task[item].clear();
      snapshots_by_task[item].clear();
      live_stage2_neighbors_by_task[item].clear();
      snapshot_task_active[item] = !context.tasks[item].reverse_reconciled;
      if (snapshot_task_active[item]) {
        snapshot_candidates_by_task[item] =
          context.tasks[item].stage2_neighbors;
      }
    }
    // Liveness revalidation needs only the coherent header/incarnation pair,
    // not another copy of every D-byte vector. The compact identity wave also
    // fits several times more requests in the same registered scratch plane.
    if (!materialize_stable_identity_wave(context.tasks.size())) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      defer_stage2_retry(context);
      return false;
    }

    const auto publish_recovery_outgoing = [&](
        StorageOwnerMaintenanceTask& task,
        RemotePtr recovery_parent,
        vec<RemotePtr>& live_neighbors) {
      if (recovery_parent.is_null() ||
          recovery_parent == task.final_target) {
        return false;
      }
      lib_assert(storage_node_pointer_addressable(task.final_target),
                 "Stage2 recovery target is structurally invalid");
      const IncarnationLockResult final_lock =
        try_lock_node(task.final_target);
      if (final_lock != IncarnationLockResult::locked) {
        return false;
      }

      u64 final_header = 0;
      node_t final_id = 0;
      u32 final_generation = 0;
      GraphAdjacency adjacency;
      const bool final_current =
        read_locked_node_identity(
          task.final_target, final_header, final_id, final_generation) &&
        final_id == task.id &&
        final_generation == task.generation &&
        (final_header & (VamanaNode::HEADER_DELETED |
                         VamanaNode::HEADER_PROVISIONAL |
                         VamanaNode::HEADER_RETIRING)) == 0 &&
        read_graph_adjacency(task.final_target, adjacency) &&
        !adjacency.deleted &&
        adjacency.generation == task.generation;
      if (!final_current) {
        lib_assert(publish_locked_node_header(
                     task.final_target, final_header, 0, 0),
                   "failed to release invalid Stage2 recovery target");
        return false;
      }

      vec<RemotePtr> candidates = adjacency.stable;
      candidates.erase(
        std::remove_if(
          candidates.begin(), candidates.end(),
          [&](RemotePtr candidate) {
            return candidate.is_null() ||
              candidate == task.final_target ||
              !storage_node_pointer_addressable(candidate);
          }),
        candidates.end());
      if (std::find(candidates.begin(), candidates.end(), recovery_parent) ==
          candidates.end()) {
        candidates.push_back(recovery_parent);
      }
      vec<StableNodeSnapshotState> recovery_outgoing_states;
      const vec<NodeSnapshot> snapshots = read_node_snapshots_batched(
        candidates, config, "stage2_recovery_outgoing",
        &recovery_outgoing_states);
      if (std::find(recovery_outgoing_states.begin(),
                    recovery_outgoing_states.end(),
                    StableNodeSnapshotState::retryable) !=
          recovery_outgoing_states.end()) {
        lib_assert(publish_locked_node_header(
                     task.final_target, final_header, 0, 0),
                   "failed to release retryable Stage2 recovery target");
        return false;
      }
      hashset_t<RemotePtr> eligible;
      eligible.reserve(snapshots.size());
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.rptr != task.final_target &&
            stage2_parent_is_stable(snapshot.header, snapshot.deleted)) {
          eligible.insert(snapshot.rptr);
        }
      }

      live_neighbors.clear();
      live_neighbors.reserve(adjacency.stable.size());
      for (const RemotePtr neighbor : adjacency.stable) {
        if (eligible.contains(neighbor)) {
          live_neighbors.push_back(neighbor);
        }
      }

      bool changed = false;
      if (live_neighbors.empty() && eligible.contains(recovery_parent)) {
        auto recovery_position = std::find(
          adjacency.stable.begin(), adjacency.stable.end(), recovery_parent);
        if (recovery_position == adjacency.stable.end()) {
          if (adjacency.stable.size() < config.R) {
            adjacency.stable.push_back(recovery_parent);
          } else {
            const auto stale_position = std::find_if(
              adjacency.stable.begin(), adjacency.stable.end(),
              [&](RemotePtr neighbor) {
                return !eligible.contains(neighbor);
              });
            if (stale_position == adjacency.stable.end()) {
              lib_assert(publish_locked_node_header(
                           task.final_target, final_header, 0, 0),
                         "failed to release saturated Stage2 recovery target");
              return false;
            }
            *stale_position = recovery_parent;
          }
          changed = true;
        }
        live_neighbors.push_back(recovery_parent);
      }

      if (changed) {
        write_graph_adjacency(
          task.final_target, adjacency.stable, adjacency.provisional,
          task.generation, false);
      }
      task.stage2_neighbors = adjacency.stable;
      lib_assert(publish_locked_node_header(
                   task.final_target, final_header, 0, 0),
                 "failed to publish Stage2 recovery outgoing graph");
      return !live_neighbors.empty();
    };

    // A Stage1 parent can be tombstoned while Stage2 is queued. Revalidate the
    // acknowledged protected slots before atomically promoting one final
    // stable bridge. No protected slot survives finalization, and Stage2 must
    // never silently commit an unreachable node.
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.reverse_reconciled) continue;
      auto expected_plan = plan_stage2_backlink_reconciliation(
        span<const RemotePtr>{task.stage1_backlink_targets},
        span<const RemotePtr>{live_stage2_neighbors_by_task[item]},
        task.stage2_promotion_committed
          ? task.stage2_promotion_parent : RemotePtr{});
      if (expected_plan.promotion_target.is_null()) {
        // All parents selected by the sealed plan can legitimately retire
        // before reverse reconciliation.  Revalidate a bounded recovery set
        // from this task's Stage1 locality first, then the current versioned
        // centroid route. Under the child lock, preserve any concurrent live
        // additions or replace one proven-stale outgoing slot with that
        // parent. Placement stays sealed, while future deletion cleanup can
        // still discover and remove the resulting incoming certificate.
        vec<RemotePtr> recovery_candidates = task.stage1_base_neighbors;
        recovery_candidates.reserve(recovery_candidates.size() +
                                    durable_route_entries.size());
        for (const RemotePtr candidate : durable_route_entries) {
          if (std::find(recovery_candidates.begin(),
                        recovery_candidates.end(), candidate) ==
              recovery_candidates.end()) {
            recovery_candidates.push_back(candidate);
          }
        }
        recovery_candidates.erase(
          std::remove_if(
            recovery_candidates.begin(), recovery_candidates.end(),
            [&](RemotePtr candidate) {
              return candidate.is_null() ||
                candidate == task.target ||
                candidate == task.final_target ||
                !storage_node_pointer_addressable(candidate);
            }),
          recovery_candidates.end());
        vec<StableNodeSnapshotState> recovery_states;
        const vec<NodeSnapshot> recovery_snapshots =
          read_node_snapshots_batched(
            recovery_candidates, config,
            "stage2_reachability_recovery", &recovery_states);
        if (std::find(recovery_states.begin(), recovery_states.end(),
                      StableNodeSnapshotState::retryable) !=
            recovery_states.end()) {
          defer_stage2_retry(context);
          return false;
        }
        RemotePtr recovery_parent;
        for (const NodeSnapshot& candidate : recovery_snapshots) {
          if (!stage2_parent_is_stable(
                candidate.header, candidate.deleted)) {
            continue;
          }
          recovery_parent = candidate.rptr;
          break;
        }
        if (!recovery_parent.is_null() &&
            !publish_recovery_outgoing(
              task, recovery_parent,
              live_stage2_neighbors_by_task[item])) {
          defer_stage2_retry(context);
          return false;
        }
        expected_plan = plan_stage2_backlink_reconciliation(
          span<const RemotePtr>{task.stage1_backlink_targets},
          span<const RemotePtr>{live_stage2_neighbors_by_task[item]},
          task.stage2_promotion_committed
            ? task.stage2_promotion_parent : RemotePtr{});
      }
      const vec<RemotePtr> candidate_parents =
        stage2_revalidation_parents(
          span<const RemotePtr>{task.stage1_backlink_targets},
          task.stage2_promotion_committed
            ? task.stage2_promotion_parent : RemotePtr{},
          expected_plan.promotion_target);
      // Only Stage1 backlink targets can own a provisional reachability slot.
      // On a retry, the one selected promotion target may instead already own
      // the final stable certificate.  Ordinary base/final neighbors cannot
      // satisfy either postcondition, so scanning O(2R+L) of them (including a
      // full vector snapshot and a synchronous adjacency read apiece) was
      // pure work and dominated Stage2 latency.
      vec<StableNodeSnapshotState> parent_states;
      const vec<NodeSnapshot> parent_snapshots =
        read_node_snapshots_batched(
          candidate_parents, config, "stage2_parent_revalidation",
          &parent_states);
      if (std::find(parent_states.begin(), parent_states.end(),
                    StableNodeSnapshotState::retryable) !=
          parent_states.end()) {
        defer_stage2_retry(context);
        return false;
      }
      vec<RemotePtr> protected_parents;
      protected_parents.reserve(parent_snapshots.size());
      bool promotion_postcondition_holds = false;
      bool parent_adjacency_retryable = false;
      RemotePtr observed_promotion_parent;
      for (const NodeSnapshot& parent : parent_snapshots) {
        if (stage2_parent_is_stable(parent.header, parent.deleted)) {
          GraphAdjacency parent_adjacency;
          if (!read_graph_adjacency(parent.rptr, parent_adjacency)) {
            // The vector/header observation proved this parent current. A
            // checksum/torn adjacency miss is therefore contention, not
            // evidence that its Stage1 bridge disappeared. Preserve every
            // persisted reconciliation field and retry the whole boundary.
            parent_adjacency_retryable = true;
            break;
          }
          if (parent_adjacency.deleted) {
            continue;
          }
          const bool protects_old = std::find(
            parent_adjacency.provisional.begin(),
            parent_adjacency.provisional.end(), task.target) !=
              parent_adjacency.provisional.end();
          const bool protects_final = !task.final_target.is_null() &&
            std::find(parent_adjacency.provisional.begin(),
                      parent_adjacency.provisional.end(),
                      task.final_target) !=
              parent_adjacency.provisional.end();
          const bool has_promoted_final =
            !task.final_target.is_null() &&
            std::find(parent_adjacency.stable.begin(),
                      parent_adjacency.stable.end(),
                      task.final_target) != parent_adjacency.stable.end();
          if (has_promoted_final &&
              (!promotion_postcondition_holds ||
               parent.rptr == expected_plan.promotion_target)) {
            promotion_postcondition_holds = true;
            observed_promotion_parent = parent.rptr;
          }
          if (protects_old || protects_final) {
            protected_parents.push_back(parent.rptr);
          }
        }
      }
      if (parent_adjacency_retryable) {
        defer_stage2_retry(context);
        return false;
      }
      // Retired/reused parents already satisfy the remove postcondition and
      // must not poison a fixed retry payload. Keep only exact live protected
      // slots, even after promotion; the separately persisted stable
      // certificate carries the idempotent retry anchor.
      task.stage1_backlink_targets = std::move(protected_parents);
      task.stage2_promotion_committed = promotion_postcondition_holds;
      task.stage2_promotion_parent = promotion_postcondition_holds
        ? observed_promotion_parent : RemotePtr{};
      // Every original protected parent may retire while Stage2 is queued.
      // If any live protected parent remains, the planner consumes it even
      // when final pruning produced no outgoing edge. Otherwise it establishes
      // one bounded stable bridge at a durable final parent. The promotion ACK
      // remains the gate for all later cleanup.
    }

    // Promotion is the durable reachability certificate. Persist the exact
    // liveness view before yielding, then install ordinary stable additions
    // followed by promotion in one audited per-target transaction. Obsolete
    // Stage1 removals remain a second barrier. This retains promotion-last
    // semantics without paying a separate promotion RTT.
    lib_assert(context.live_stage2_neighbors_by_task.size() >=
                 context.tasks.size(),
               "Stage2 context cannot persist its reconciliation plan");
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      context.live_stage2_neighbors_by_task[item] =
        live_stage2_neighbors_by_task[item];
    }
    if (!start_reconcile_barrier(
          context, stage2_reconcile_first_barrier())) {
      return false;
    }
    (void)advance_reconcile_pipeline(context);
    return false;
  };

  const auto prepare_cleanup_reverse = [&](Stage2Context& context) {
    vec<vec<RemotePtr>> cleanup_neighbors_by_task;
    cleanup_neighbors_by_task.reserve(context.tasks.size());
    for (auto& ops : context.remote_ops_by_peer) {
      ops.clear();
    }

    // Cleanup descriptors are deliberately runnable before the foreground
    // authority commit. First cross that token-fenced authority barrier; an
    // aborted mutation simply cancels its descriptor without ever quiescing
    // the still-current old generation.
    size_t ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.target.is_null()) {
        storage_owner_maintenance_cleanup_processed_.fetch_add(
          1, std::memory_order_relaxed);
        complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
        task.maintenance_sequence = 0;
        continue;
      }
      if (!task.cleanup_repair_only) {
        if (!task.cleanup_authority_retired) {
          const protocol::AuthorityPlacementItem retirement_gate{
            .token = {
              .source_client = task.source_client,
              .item_index = task.operation_item_index,
              .client_batch_id = task.operation_batch_id,
            },
            .id = task.id,
            // desired_raw == 0 interprets this as a read-only proof that a
            // strictly newer authority generation retired target.
            .generation = task.generation,
            .expected_raw = task.target.raw_address,
            .desired_raw = 0,
            .expected_placement_version = 0,
          };
          protocol::AuthorityPlacementResult gate_result;
          if (!relocate_via_authority(
                task.authority_shard, retirement_gate, gate_result, config)) {
            storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            return false;
          }
          const auto gate_status = static_cast<
            protocol::AuthorityPlacementStatus>(gate_result.status);
          if (gate_status == protocol::AuthorityPlacementStatus::busy) {
            storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
            return false;
          }
          if (gate_status == protocol::AuthorityPlacementStatus::stale) {
            complete_stale_cleanup(task);
            task.maintenance_sequence = 0;
            continue;
          }
          if (gate_status != protocol::AuthorityPlacementStatus::committed &&
              gate_status != protocol::AuthorityPlacementStatus::replay) {
            storage_owner_maintenance_failed_.fetch_add(
              1, std::memory_order_relaxed);
            return false;
          }
          task.cleanup_authority_retired = true;
        }
      } else {
        task.centroid_committed = true;
      }
      if (ready != item) {
        context.tasks[ready] = std::move(task);
      }
      ++ready;
    }
    context.tasks.resize(ready);

    // Do not quiesce an earlier item in the batch while a later item's
    // authority lease is still pending. Once every item has crossed its
    // barrier, protected-child reparenting may proceed independently and be
    // retried from the task's bounded progress vectors.
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (task.cleanup_repair_only) continue;
      const CleanupParentQuiesceResult quiesce_result =
        quiesce_cleanup_parent(task);
      if (quiesce_result == CleanupParentQuiesceResult::stale) {
        complete_stale_cleanup(task);
        task.maintenance_sequence = 0;
        continue;
      }
      if (quiesce_result == CleanupParentQuiesceResult::busy ||
          !reparent_cleanup_children(task)) {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
    }

    // Stale incarnations above are complete, not retryable. Remove them
    // before centroid/removal vectors are built so task-index correlation is
    // preserved for the remainder of this cleanup context.
    ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      if (context.tasks[item].maintenance_sequence == 0) continue;
      if (ready != item) context.tasks[ready] = std::move(context.tasks[item]);
      ++ready;
    }
    context.tasks.resize(ready);

    // Withdraw the exact tagged identity, update the FP64 sum/count, elect
    // replacement entries and publish the complete route before any matching
    // node can become DELETED. A lost response is harmless: the accounted bit
    // turns retry into a no-op, while the node remains RETIRING and readable.
    vec<CentroidMembershipOp> centroid_removes;
    centroid_removes.reserve(context.tasks.size());
    for (const StorageOwnerMaintenanceTask& task : context.tasks) {
      if (!task.centroid_committed) {
        centroid_removes.push_back(make_centroid_membership_op(
          task, task.target, CentroidMembershipKind::remove));
      }
    }
    if (!centroid_removes.empty() &&
        !apply_centroid_membership_fanout_and_wait(
          span<const CentroidMembershipOp>{centroid_removes}, config)) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      task.centroid_committed = true;
    }

    ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (!task.cleanup_repair_only) {
        NodeSnapshot before_delete;
        if (!read_node_snapshot(task.target, before_delete)) {
          return false;
        }
        if (!before_delete.deleted) {
          if (before_delete.id != task.id ||
              before_delete.generation != task.generation ||
              (before_delete.header & VamanaNode::HEADER_RETIRING) == 0 ||
              !cleanup_tombstone_allowed(
                task.cleanup_authority_retired, task.cleanup_retiring,
                task.centroid_committed) ||
              !mark_node_deleted(task.target, task.generation)) {
            return false;
          }
        }
      }
      NodeSnapshot deleted_snapshot;
      const bool readable = read_node_snapshot(task.target, deleted_snapshot);
      lib_assert(readable, "local cleanup snapshot was unreadable");
      if (!deleted_snapshot.deleted && !task.cleanup_repair_only) {
        complete_stale_cleanup(task);
        task.maintenance_sequence = 0;
        continue;
      }
      task.id = deleted_snapshot.id;
      task.generation = deleted_snapshot.generation;
      // A prepared Stage2 finalization can become stale only after a later erase/upsert
      // tombstones the same physical node. That later mutation owns an
      // ordinary cleanup intent for the preserved adjacency. This repair must
      // therefore undo only the backlinks attempted by the stale Stage2 finalization; a
      // preserved+supplemental union can contain 2R operations per item and
      // cannot fit one bounded R*batch peer message.
      lib_assert(!task.cleanup_repair_only || deleted_snapshot.deleted,
                 "stale Stage2 finalization repair requires a successor tombstone cleanup");
      lib_assert(task.cleanup_repair_only || task.cleanup_neighbors.empty(),
                 "ordinary tombstone cleanup unexpectedly carried repair neighbors");
      vec<RemotePtr> preserved_neighbors;
      if (deleted_snapshot.deleted && !task.cleanup_repair_only) {
        preserved_neighbors = read_preserved_neighbor_list(task.target);
      }
      vec<RemotePtr> old_neighbors = select_cleanup_neighbors(
        task.cleanup_repair_only,
        span<const RemotePtr>{preserved_neighbors.data(),
                              preserved_neighbors.size()},
        task.cleanup_repair_only
          ? span<const RemotePtr>{task.cleanup_neighbors.data(),
                                  task.cleanup_neighbors.size()}
          : span<const RemotePtr>{task.cleanup_protected_children.data(),
                                  task.cleanup_protected_children.size()});
      lib_assert(old_neighbors.size() <= VamanaNode::graph_entry_capacity(),
                 "stage2 cleanup exceeded the per-item wire bound");
      if (ready != item) {
        context.tasks[ready] = std::move(task);
      }
      cleanup_neighbors_by_task.push_back(std::move(old_neighbors));
      ++ready;
    }
    context.tasks.resize(ready);

    // Snapshot every reverse target in one batch before constructing the
    // cleanup wire payload. Both ends carry ID/generation so a delayed retry
    // cannot mutate an unrelated node after either physical slot is reused.
    vec<RemotePtr> cleanup_targets;
    for (const vec<RemotePtr>& neighbors : cleanup_neighbors_by_task) {
      for (RemotePtr neighbor : neighbors) {
        if (!neighbor.is_null() &&
            neighbor.memory_node() < num_storage_nodes_) {
          cleanup_targets.push_back(neighbor);
        }
      }
    }
    std::sort(cleanup_targets.begin(), cleanup_targets.end(),
              [](RemotePtr lhs, RemotePtr rhs) {
                return lhs.raw_address < rhs.raw_address;
              });
    cleanup_targets.erase(
      std::unique(cleanup_targets.begin(), cleanup_targets.end()),
      cleanup_targets.end());
    vec<StableNodeSnapshotState> cleanup_target_states;
    const vec<NodeSnapshot> cleanup_target_snapshots =
      read_node_snapshots_batched(
        cleanup_targets, config, "cleanup_target_snapshot",
        &cleanup_target_states);
    if (std::find(cleanup_target_states.begin(),
                  cleanup_target_states.end(),
                  StableNodeSnapshotState::retryable) !=
        cleanup_target_states.end()) {
      return false;
    }
    dense_hashmap_t<u64, const NodeSnapshot*> cleanup_target_by_raw;
    cleanup_target_by_raw.reserve(cleanup_target_snapshots.size());
    for (const NodeSnapshot& snapshot : cleanup_target_snapshots) {
      cleanup_target_by_raw.emplace(snapshot.rptr.raw_address, &snapshot);
    }

    vec<ReverseUpdateOp> local_cleanup_ops;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      const StorageOwnerMaintenanceTask& task = context.tasks[item];
      for (RemotePtr neighbor : cleanup_neighbors_by_task[item]) {
        const auto found = cleanup_target_by_raw.find(neighbor.raw_address);
        if (found == cleanup_target_by_raw.end()) continue;
        const NodeSnapshot& target = *found->second;
        const ReverseUpdateOp op{
          .target_raw = neighbor.raw_address,
          .candidate_raw = task.target.raw_address,
          .target_id = target.id,
          .target_generation = target.generation,
          .candidate_id = task.id,
          .candidate_generation = task.generation,
        };
        if (local_shard(neighbor.memory_node())) {
          local_cleanup_ops.push_back(op);
        } else {
          context.remote_ops_by_peer[neighbor.memory_node()].push_back(op);
        }
      }
    }

    if (!remove_local_neighbors_identity_fenced(
          span<const ReverseUpdateOp>{local_cleanup_ops}, config)) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }

    u64 expected_mask = 0;
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      if (shard != storage_id_ &&
          !context.remote_ops_by_peer[shard].empty()) {
        expected_mask |= u64{1} << shard;
      }
    }
    const Stage2EventResult transition =
      states.begin_reverse(context.handle, expected_mask);
    lib_assert(transition == Stage2EventResult::phase_advanced ||
                 transition == Stage2EventResult::ready_to_finalize,
               "cleanup stage2 failed to enter reverse_pending");
    register_reverse_requests(context, expected_mask);
    return true;
  };

  const auto finalize_context = [&](Stage2Context& context) {
    const Stage2EventResult transition = states.finalize(context.handle);
    lib_assert(transition == Stage2EventResult::phase_advanced,
               "stage2 finalized before all reverse ACKs");

    if (context.kind == StorageOwnerMaintenanceKind::finalize_insert) {
      lib_assert(context.completion_handoff_started_ns != 0,
                 "Stage2 finalization lost its completion-handoff timestamp");
      auto& completion_handoff_timing =
        storage_owner_stage2_phase_timing_[static_cast<size_t>(
          StorageOwnerStage2TimingPhase::completion_handoff)];
      completion_handoff_timing.attempts.fetch_add(
        1, std::memory_order_relaxed);
      completion_handoff_timing.task_attempts.fetch_add(
        context.tasks.size(), std::memory_order_relaxed);
      completion_handoff_timing.elapsed_ns.fetch_add(
        steady_now_ns() - context.completion_handoff_started_ns,
        std::memory_order_relaxed);
      context.completion_handoff_started_ns = 0;
      Stage2PhaseAttemptTimer finalize_timer(
        storage_owner_stage2_phase_timing_[static_cast<size_t>(
          StorageOwnerStage2TimingPhase::finalize)],
        context.tasks.size());
      for (StorageOwnerMaintenanceTask& task : context.tasks) {
        lib_assert(task.outgoing_committed && task.reverse_reconciled &&
                     task.placement_committed && task.centroid_committed &&
                     (task.final_target == task.target ||
                      task.allocation_settled),
                   "Stage2 finalized before physical placement commit");
        record_finalized_live(task.queued_at);
        u64 final_edges = 0;
        u64 cross_edges_from_stage1_home = 0;
        u64 cross_edges_from_final_home = 0;
        for (const RemotePtr neighbor : task.stage2_neighbors) {
          if (neighbor.is_null() ||
              neighbor.memory_node() >= num_storage_nodes_) {
            continue;
          }
          ++final_edges;
          cross_edges_from_stage1_home +=
            neighbor.memory_node() != task.target.memory_node();
          cross_edges_from_final_home +=
            neighbor.memory_node() != task.final_target.memory_node();
        }
        storage_owner_stage2_final_edges_.fetch_add(
          final_edges, std::memory_order_relaxed);
        storage_owner_stage2_cross_edges_stage1_home_.fetch_add(
          cross_edges_from_stage1_home, std::memory_order_relaxed);
        storage_owner_stage2_cross_edges_final_home_.fetch_add(
          cross_edges_from_final_home, std::memory_order_relaxed);
        if (task.final_target != task.target) {
          storage_owner_stage2_migrations_.fetch_add(
            1, std::memory_order_relaxed);
        }
        storage_owner_maintenance_processed_.fetch_add(
          1, std::memory_order_relaxed);
        if (task.final_target != task.target) {
          // Reverse reconciliation and the authority placement ACK are both
          // complete. Retire the old Stage1 incarnation under this Stage2
          // sequence; reuse still waits for local durability and every
          // compute client's RCU acknowledgement.
          retire_local_dynamic_node(
            task.target, task.maintenance_sequence);
        }
        complete_storage_owner_maintenance_sequence(
          task.maintenance_sequence);
      }
    } else {
      for (StorageOwnerMaintenanceTask& task : context.tasks) {
        lib_assert(task.centroid_committed,
                   "cleanup finalized before centroid membership removal");
        if (!task.cleanup_repair_only) {
          retire_local_dynamic_node(task.target, task.maintenance_sequence);
        }
        storage_owner_maintenance_cleanup_processed_.fetch_add(
          1, std::memory_order_relaxed);
        complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
      }
    }

    if (context.kind == StorageOwnerMaintenanceKind::finalize_insert &&
        context.packing_admitted_ns != 0) {
      const u64 service_ns = steady_now_ns() - context.packing_admitted_ns;
      const u64 effective_cost_ns = service_ns >
          std::numeric_limits<u64>::max() - context.packing_wait_ns
        ? std::numeric_limits<u64>::max()
        : service_ns + context.packing_wait_ns;
      const size_t debt_at_completion =
        storage_owner_maintenance_completion_ring_ == nullptr ? 0 :
          storage_owner_maintenance_completion_ring_->incomplete();
      if (independent_score_experiment_enabled) {
        storage_owner_independent_score_.observe_completion(
          context.independent_score_sample,
          context.tasks.size(),
          effective_cost_ns,
          context.packing_debt_at_admission,
          debt_at_completion,
          context.search_io.independent_score_rpcs_posted,
          context.search_io.independent_score_useful);
      }
      storage_owner_stage2_packing_.observe_completion(
        context.packing_target_batch,
        context.packing_high_pressure,
        context.tasks.size(),
        effective_cost_ns,
        context.packing_debt_at_admission,
        debt_at_completion);
    }

    const Stage2ContextHandle handle = context.handle;
    release_context_lane(context);
    reset_context(context);
    lib_assert(states.release(handle),
               "stage2 context release violated finalized generation");
    storage_owner_maintenance_active_workers_.fetch_sub(
      1, std::memory_order_acq_rel);
    notify_storage_owner_maintenance_capacity();
  };

  const auto defer_cleanup_context = [&](Stage2Context& context) {
    lib_assert(context.kind ==
                 StorageOwnerMaintenanceKind::cleanup_deleted_node,
               "only cleanup work can use protected-capacity deferral");
    const auto retry_at = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(1);
    {
      std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
      for (StorageOwnerMaintenanceTask& task : context.tasks) {
        if (task.maintenance_sequence == 0) continue;
        task.retry_not_before = retry_at;
        cleanup_schedule_push(
          storage_owner_cleanup_tasks_, std::move(task));
      }
    }
    const Stage2ContextHandle handle = context.handle;
    release_context_lane(context);
    reset_context(context);
    lib_assert(states.release_retryable(handle),
               "cleanup deferral retained an asynchronous request");
    storage_owner_maintenance_active_workers_.fetch_sub(
      1, std::memory_order_acq_rel);
    notify_storage_owner_maintenance_capacity();
  };

  const auto drive_context = [&](Stage2Context& context) {
    bool progressed = false;
    for (;;) {
      const auto snapshot = states.snapshot(context.handle);
      lib_assert(snapshot.has_value(), "active stage2 context became stale");
      if (context.kind == StorageOwnerMaintenanceKind::finalize_insert) {
        const auto now = std::chrono::steady_clock::now();
        const bool deferred = std::any_of(
          context.tasks.begin(), context.tasks.end(),
          [&](const StorageOwnerMaintenanceTask& task) {
            return task.retry_not_before > now;
          });
        if (deferred) {
          // A prune retry persists all semantic progress in the context/task.
          // Do not let its bounded backoff monopolize this worker's scarce
          // registered scratch lane.  The dual readiness predicate prevents
          // an active continuation or an in-flight CQE from being discarded.
          if (snapshot->phase == Stage2Phase::prune_ready) {
            (void)release_rebindable_context_lane(context);
          }
          return progressed;
        }
      }
      const bool prune_needs_lane =
        snapshot->phase == Stage2Phase::prune_ready &&
        (context.kind == StorageOwnerMaintenanceKind::cleanup_deleted_node ||
         stage2_finalize_subphase_needs_lane(context.finalize_subphase));
      if (snapshot->phase == Stage2Phase::local_ready ||
          snapshot->phase == Stage2Phase::remote_search_pending ||
          prune_needs_lane) {
        if (!bind_context_lane(context)) return progressed;
      }
      switch (snapshot->phase) {
        case Stage2Phase::local_ready: {
          const Stage2SearchAdvanceResult local_result =
            prepare_local(context);
          if (local_result != Stage2SearchAdvanceResult::complete) {
            // Before search initialization, a transient target lock can return
            // waiting_rdma with neither live continuation state nor a posted
            // WR.  Reuse that otherwise-idle lane; an initialized search fails
            // the search_state_idle half of the predicate and remains pinned.
            (void)release_rebindable_context_lane(context);
            return progressed ||
              local_result == Stage2SearchAdvanceResult::posted_rdma;
          }
          progressed = true;
          // candidate_search copied every result into the context and reset its
          // lane state before reporting complete.  Yield at this semantic
          // boundary so another ready context can hide this context's upcoming
          // synchronous prune/control work.
          lib_assert(release_rebindable_context_lane(context),
                     "completed Stage2 search retained live lane state");
          return true;
        }
        case Stage2Phase::remote_search_pending:
          lib_failure(
            "home-executed Stage2 continuation cannot await legacy shard-search ACKs");
          return progressed;
        case Stage2Phase::prune_ready: {
          const bool prepared =
            context.kind == StorageOwnerMaintenanceKind::finalize_insert
              ? prepare_stage2_reverse(context)
              : prepare_cleanup_reverse(context);
          if (!prepared) {
            if (context.kind ==
                StorageOwnerMaintenanceKind::cleanup_deleted_node) {
              defer_cleanup_context(context);
              return true;
            }
            (void)release_rebindable_context_lane(context);
            return progressed;
          }
          progressed = true;
          continue;
        }
        case Stage2Phase::reverse_pending:
          release_context_lane(context);
          if (states.snapshot(context.handle)->phase ==
              Stage2Phase::reverse_pending &&
              states.snapshot(context.handle)->completed_reverse_mask !=
                states.snapshot(context.handle)->expected_reverse_mask) {
            return progressed;
          }
          finalize_context(context);
          return true;
        case Stage2Phase::finalized:
          lib_failure("stage2 context remained active after finalization");
      }
    }
  };

  const auto drive_owned_context = [&](Stage2Context& context) {
    auto* const previous_waiter_registrations =
      current_storage_owner_maintenance_waiter_registrations_;
    current_storage_owner_maintenance_waiter_registrations_ =
      &context.waiter_registrations;
    const Stage2ContextOwnerKey previous_context_owner =
      current_storage_owner_maintenance_context_owner_;
    current_storage_owner_maintenance_context_owner_ = context.ready_owner;
    const bool progressed = drive_context(context);
    current_storage_owner_maintenance_context_owner_ =
      previous_context_owner;
    current_storage_owner_maintenance_waiter_registrations_ =
      previous_waiter_registrations;
    return progressed;
  };

  vec<Stage2ReadyContextEvent> overflow_ready_events;
  overflow_ready_events.reserve(context_capacity);
  const auto drive_ready_event = [&](const Stage2ReadyContextEvent& event) {
    lib_assert(event.owner.worker_id == worker_id &&
                 event.owner.slot < contexts.size(),
               "stage2 ready event targets the wrong executor");
    Stage2Context& context = contexts[event.owner.slot];
    if (!context.active || context.ready_owner != event.owner) {
      return false;
    }
    return drive_owned_context(context);
  };
  const auto drain_ready_contexts = [&]() {
    bool progressed = false;
    size_t drained = 0;
    Stage2ReadyContextEvent event;
    // One active generation contributes at most one ordinary queued ticket;
    // the 2x bound also covers stale prior-generation tickets during reuse.
    const size_t ticket_budget = context_capacity * 2;
    while (drained < ticket_budget && ready_context_queue->try_pop(event)) {
      ++drained;
      progressed = drive_ready_event(event) || progressed;
    }
    if (drained != 0) {
      storage_owner_maintenance_ready_tickets_drained_.fetch_add(
        drained, std::memory_order_relaxed);
    }

    if (ready_context_queue->overflowed()) {
      overflow_ready_events.clear();
      const size_t recovered =
        ready_context_queue->recover_overflow(overflow_ready_events);
      storage_owner_maintenance_ready_overflow_scans_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_maintenance_context_slots_scanned_.fetch_add(
        context_capacity, std::memory_order_relaxed);
      storage_owner_maintenance_ready_tickets_drained_.fetch_add(
        recovered, std::memory_order_relaxed);
      drained += recovered;
      for (const Stage2ReadyContextEvent& recovered_event :
           overflow_ready_events) {
        progressed = drive_ready_event(recovered_event) || progressed;
      }
    }
    return std::pair<bool, size_t>{progressed, drained};
  };

  const auto try_admit_context = [&]() -> Stage2Context* {
    // Evaluate the shared adaptive budget before the local-limit fast path.
    // At the C32 floor every executor can already own four contexts, so
    // waiting until after that check would prevent the controller from ever
    // observing enough debt to promote.
    maybe_adjust_storage_owner_stage2_execution_budget();
    const Stage2AdmissionDecision admission = decide_stage2_admission(
      states.full(),
      storage_owner_maintenance_shutdown_.load(std::memory_order_acquire),
      [&]() { return storage_owner_maintenance_foreground_busy(config); });
    if (admission == Stage2AdmissionDecision::unavailable) {
      return nullptr;
    }
    const bool foreground_pressure =
      admission == Stage2AdmissionDecision::foreground_pressure;
    const size_t contexts_per_worker_limit =
      storage_owner_maintenance_contexts_per_worker_limit_.load(
        std::memory_order_acquire);
    const size_t local_context_limit =
      stage2_worker_context_admission_limit(
        config.storage_owner_rpc_depth, foreground_pressure,
        contexts_per_worker_limit);
    if (states.size() >= local_context_limit) {
      // Do not let this worker monopolize the process-wide context allowance.
      // Its contexts can use only this worker's registered search lanes;
      // leaving the remaining global permits for sibling workers is what
      // exposes the already-allocated RDMA concurrency.
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      return nullptr;
    }
    if (!try_acquire_storage_owner_maintenance_slot(
          config, foreground_pressure)) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      return nullptr;
    }

    const size_t batch_limit =
      std::max<size_t>(1, config.storage_owner_batch_max);
    const auto acquire_context = [&]() -> Stage2Context* {
      const auto handle = states.try_acquire();
      lib_assert(handle.has_value(),
                 "stage2 context tracker unexpectedly exhausted");
      Stage2Context& context = contexts[handle->slot];
      reset_context(context);
      context.handle = *handle;
      context.ready_owner = ready_context_queue->activate(handle->slot);
      context.active = true;
      return &context;
    };

    // Repair continuations own an already-admitted maintenance sequence and
    // therefore take priority over new Stage2 work. This removes the stale
    // Stage2 finalization's attempted backlinks before advancing the watermark and proves
    // the dedicated queue cannot grow across successive admission waves.
    if (storage_owner_repair_tasks_ != nullptr) {
      StorageOwnerMaintenanceTask repair;
      if (storage_owner_repair_tasks_->try_pop(repair)) {
        if (storage_owner_cleanup_ready(repair.maintenance_sequence)) {
          Stage2Context& context = *acquire_context();
          context.kind = StorageOwnerMaintenanceKind::cleanup_deleted_node;
          context.tasks.push_back(std::move(repair));
          repair = StorageOwnerMaintenanceTask{};
          while (context.tasks.size() < batch_limit &&
                 storage_owner_repair_tasks_->try_pop(repair)) {
            if (!storage_owner_cleanup_ready(repair.maintenance_sequence)) {
              lib_assert(storage_owner_repair_tasks_->try_push(
                           std::move(repair)),
                         "failed to return a not-yet-ready repair descriptor");
              break;
            }
            context.tasks.push_back(std::move(repair));
            repair = StorageOwnerMaintenanceTask{};
          }
          notify_storage_owner_maintenance_capacity();
          return &context;
        }
        lib_assert(storage_owner_repair_tasks_->try_push(std::move(repair)),
                   "failed to return a not-yet-ready repair descriptor");
      }
    }

    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    const auto admission_now = std::chrono::steady_clock::now();
    const bool cleanup_ready = !storage_owner_cleanup_tasks_.empty() &&
      storage_owner_cleanup_ready(
        storage_owner_cleanup_tasks_.front().maintenance_sequence) &&
      admission_now >=
        storage_owner_cleanup_tasks_.front().retry_not_before;
    const auto packing_parameters =
      storage_owner_stage2_packing_.parameters();
    const size_t completion_incomplete =
      storage_owner_maintenance_completion_ring_ == nullptr ? 0 :
        storage_owner_maintenance_completion_ring_->incomplete();
    // The accepted window is intentionally much larger than active Stage2
    // execution. Do not use half of that window as a pressure threshold: with
    // a 65K accepted backlog it would label every useful 8/16/32 batch as low
    // pressure. Visible queue depth is authoritative, with two configured
    // batches of unfinished debt as the bounded tail-coalescing signal once
    // all descriptors have already moved into active contexts.
    const bool bulk_packing = stage2_bulk_packing_enabled(batch_limit);
    const size_t completion_pressure_threshold = bulk_packing
      ? std::max<size_t>(
          kStage2BulkMinimumBatch,
          batch_limit > std::numeric_limits<size_t>::max() / 2
            ? std::numeric_limits<size_t>::max() : batch_limit * 2)
      : (storage_owner_maintenance_admission_limit_ + 1) / 2;
    const bool completion_pressure =
      completion_pressure_threshold != 0 &&
      completion_incomplete >= completion_pressure_threshold;
    const size_t queue_pressure_threshold = bulk_packing
      ? kStage2BulkMinimumBatch
      : std::max<size_t>(2, packing_parameters.target_batch * 2);
    const bool queue_pressure =
      storage_owner_stage2_tasks_.size() >= queue_pressure_threshold;
    const bool packing_high_pressure =
      completion_pressure || queue_pressure;
    Stage2PackingDecision packing_decision;
    if (!storage_owner_stage2_tasks_.empty()) {
      packing_decision = decide_stage2_packing(
        storage_owner_stage2_tasks_.size(), batch_limit,
        packing_parameters.target_batch,
        storage_owner_stage2_tasks_.front().queued_at, admission_now,
        config.storage_owner_stage2_batch_max_wait_us,
        packing_parameters.estimated_arrival_interval_us,
        packing_high_pressure);
    }
    const bool stage2_ready = packing_decision.ready;
    const bool choose_stage2 =
      stage2_ready &&
      (!cleanup_ready || storage_owner_stage2_tasks_.front().queued_at <=
                            storage_owner_cleanup_tasks_.front().queued_at);
    if (!choose_stage2 && !cleanup_ready) {
      storage_owner_maintenance_active_workers_.fetch_sub(
        1, std::memory_order_acq_rel);
      return nullptr;
    }

    Stage2Context* admitted_context = nullptr;
    if (choose_stage2) {
      const size_t queued_at_admission =
        storage_owner_stage2_tasks_.size();
      const auto oldest_queued_at =
        storage_owner_stage2_tasks_.front().queued_at;
      const u64 oldest_wait_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::max<std::chrono::steady_clock::duration>(
            std::chrono::steady_clock::duration::zero(),
            admission_now - oldest_queued_at)).count());
      const size_t selected_pop_limit = std::min(
        batch_limit, std::max<size_t>(1, packing_decision.pop_limit));
      const size_t pop_limit = stage2_execution_slice_limit(
        selected_pop_limit, batch_limit);
      const size_t actual_pop_count = std::min(
        pop_limit, storage_owner_stage2_tasks_.size());
      lib_assert(actual_pop_count != 0 &&
                   actual_pop_count <= std::numeric_limits<u32>::max(),
                 "Stage2 execution slice does not fit its task budget");
      const u32 task_reservation = static_cast<u32>(actual_pop_count);
      // Claim the complete semantic slice before removing its descriptors.
      // A failed claim leaves every task visible and batchable in the queue;
      // this execution budget never participates in Stage1 ACK admission.
      if (!try_reserve_stage2_active_tasks(
            storage_owner_maintenance_active_tasks_, task_reservation,
            storage_owner_maintenance_active_task_limit_.load(
              std::memory_order_acquire))) {
        storage_owner_maintenance_active_workers_.fetch_sub(
          1, std::memory_order_acq_rel);
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        return nullptr;
      }
      admitted_context = acquire_context();
      Stage2Context& context = *admitted_context;
      context.kind = StorageOwnerMaintenanceKind::finalize_insert;
      context.active_task_reservation = task_reservation;
      while (!storage_owner_stage2_tasks_.empty() &&
             context.tasks.size() < pop_limit) {
        context.tasks.push_back(
          std::move(storage_owner_stage2_tasks_.front()));
        storage_owner_stage2_tasks_.pop_front();
      }
      lib_assert(context.tasks.size() == task_reservation,
                 "Stage2 queue pop diverged from its active-task reservation");
      context.packing_admitted_ns = steady_now_ns();
      context.packing_wait_ns = packing_decision.wait_budget_us == 0 ? 0 :
        std::min<u64>(
          oldest_wait_ns,
          static_cast<u64>(packing_decision.wait_budget_us) * 1'000);
      context.packing_debt_at_admission = completion_incomplete;
      context.packing_target_batch = static_cast<u32>(
        packing_decision.target_batch);
      context.packing_high_pressure = packing_high_pressure;
      if (independent_score_experiment_enabled) {
        const bool stable_legacy_packing =
          packing_parameters.larger_batch_trials_disabled ||
          !storage_owner_stage2_larger_batch_trials_possible_;
        context.independent_score_sample =
          storage_owner_independent_score_.sample(
            packing_high_pressure && stable_legacy_packing &&
            packing_decision.target_batch <= 2);
      }
      storage_owner_stage2_packing_.observe_admission(
        packing_decision.reason, context.tasks.size(), oldest_wait_ns,
        packing_decision.wait_budget_us,
        packing_decision.target_batch, queued_at_admission);
      storage_owner_stage2_batches_.fetch_add(1, std::memory_order_relaxed);
      storage_owner_stage2_batched_items_.fetch_add(
        context.tasks.size(), std::memory_order_relaxed);
    } else {
      admitted_context = acquire_context();
      Stage2Context& context = *admitted_context;
      context.kind = StorageOwnerMaintenanceKind::cleanup_deleted_node;
      storage_owner_stage2_packing_.observe_admission(
        Stage2PackingFlushReason::cleanup, 0, 0, 0);
      while (!storage_owner_cleanup_tasks_.empty() &&
             context.tasks.size() < batch_limit) {
        const StorageOwnerMaintenanceTask& next =
          storage_owner_cleanup_tasks_.front();
        if (!storage_owner_cleanup_ready(next.maintenance_sequence) ||
            std::chrono::steady_clock::now() < next.retry_not_before) {
          break;
        }
        context.tasks.push_back(
          cleanup_schedule_pop(storage_owner_cleanup_tasks_));
      }
    }
    lib_assert(admitted_context != nullptr,
               "Stage2 admission lost its context");
    Stage2Context& context = *admitted_context;
    lock.unlock();
    // Removing descriptors from the bounded runnable queue is an admission
    // edge just like completing a Stage2 sequence. Wake a parked Stage1 arm
    // even when queue capacity, rather than completion credit, was the last
    // exhausted resource.
    wake_peer_stage1_admission_waiters();
    notify_storage_owner_maintenance_capacity();
    lib_assert(!context.tasks.empty(),
               "stage2 admitted an empty maintenance context");
    return &context;
  };

  u64 unpublished_idle_wait_ns = 0;
  u64 unpublished_idle_waits = 0;
  u64 unpublished_context_slots_scanned = 0;
  size_t active_context_cursor = 0;
  bool fallback_context_scan_requested = false;
  bool fallback_audit_active = false;
  auto next_fallback_context_scan = stage2_fallback_audit_deadline(
    std::chrono::steady_clock::now(), states.size());
  const auto refresh_fallback_audit_deadline = [&](const auto now) {
    next_fallback_context_scan =
      refresh_stage2_fallback_audit_deadline(
        next_fallback_context_scan, fallback_audit_active,
        states.size(), now);
    fallback_audit_active = states.size() != 0;
  };
  const auto flush_idle_timing = [&]() {
    if (unpublished_idle_waits != 0) {
      storage_owner_maintenance_worker_idle_ns_.fetch_add(
        unpublished_idle_wait_ns, std::memory_order_relaxed);
      storage_owner_maintenance_worker_idle_waits_.fetch_add(
        unpublished_idle_waits, std::memory_order_relaxed);
      unpublished_idle_wait_ns = 0;
      unpublished_idle_waits = 0;
    }
    if (unpublished_context_slots_scanned != 0) {
      storage_owner_maintenance_context_slots_scanned_.fetch_add(
        unpublished_context_slots_scanned, std::memory_order_relaxed);
      unpublished_context_slots_scanned = 0;
    }
  };

  for (;;) {
    // Worker-wide producers (currently reverse-outbox posting) rebuild their
    // resource dependencies on every scheduler pass. Remove subscriptions
    // whose operation either succeeded or was canceled before retrying; the
    // failure path registers again under the same slot mutex/counter recheck.
    current_storage_owner_maintenance_waiter_registrations_ =
      &fallback_waiter_registrations;
    clear_all_current_storage_owner_maintenance_waiters();
    // Snapshot before scanning contexts. Any completion/event published after
    // this load either becomes visible to the scan or changes the predicate
    // below. This closes the classic notify-before-wait window without busy
    // polling and without shortening the batching deadline.
    const u64 observed_wake_epoch =
      wake_channel.epoch.load(std::memory_order_acquire);
    const bool resource_context_scan_requested =
      wake_channel.context_scan_requested.exchange(
        false, std::memory_order_acq_rel);
    if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
      flush_idle_timing();
      if (storage_owner_reverse_outbox_ != nullptr) {
        (void)storage_owner_reverse_outbox_->erase_queued_worker(worker_id);
        for (;;) {
          const auto wire_request_id =
            storage_owner_reverse_outbox_->discard_owned_aggregate(worker_id);
          if (!wire_request_id.has_value()) break;
          cancel_peer_rpc_response(*wire_request_id);
        }
      }
      // Registered lane scratch cannot be reset while the HCA may still be
      // writing into it. Shutdown therefore drains every outstanding lane
      // before releasing context ownership. This path is bounded by the
      // transport's already-posted work only; it never submits new Stage2
      // reads after shutdown has been observed.
      for (;;) {
        bool lane_rdma_pending = false;
        for (const Stage2Context& context : contexts) {
          if (!context.active || !context.search_lane.has_value()) continue;
          lane_rdma_pending = lane_rdma_pending ||
            !thread.is_ready(*context.search_lane);
        }
        if (!lane_rdma_pending) break;
        poll_peer_send_cq();
        std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
        storage_owner_maintenance_cv_.wait_for(
          lock, std::chrono::microseconds(100));
      }
      for (Stage2Context& context : contexts) {
        if (!context.active) {
          continue;
        }
        current_storage_owner_maintenance_waiter_registrations_ =
          &context.waiter_registrations;
        finish_reverse_prepare_timing(context);
        cancel_context_reconcile(context);
        // Shutdown abandons active contexts instead of entering the normal
        // finalization feedback path. Consume any admission token with a
        // zero-task sample so an in-process stop/start cannot retain a false
        // speculative-generation outstanding count.
        if (independent_score_experiment_enabled) {
          storage_owner_independent_score_.observe_completion(
            context.independent_score_sample, 0, 0, 0, 0, 0, 0);
        }
        cancel_storage_owner_search_lane_waiter(
          context.search_lane_wait_registered);
        clear_all_current_storage_owner_maintenance_waiters();
        release_context_lane(context);
        // Shutdown abandons the context rather than passing through the
        // normal reset/finalize path, so return its exact execution claim
        // here before retiring the context owner.
        release_active_task_reservation(context);
        // Invalidate every logical home-RPC member before invalidating its
        // OwnerKey. A late aggregate response then suppresses this member
        // instead of publishing an owned payload to a retired context slot.
        for (size_t rpc_index = 0;
             rpc_index < context.search_io.home_expand_rpc_count;
             ++rpc_index) {
          const Stage2HomeExpandRpc& rpc =
            context.search_io.home_expand_rpcs[rpc_index];
          if (rpc.posted && !rpc.complete && rpc.request_id != 0) {
            cancel_peer_rpc_response(rpc.request_id);
          }
        }
        for (size_t rpc_index = 0;
             rpc_index < context.search_io.score_home_rpc_count;
             ++rpc_index) {
          const Stage2HomeExpandRpc& rpc =
            context.search_io.score_home_rpcs[rpc_index];
          if (rpc.posted && !rpc.complete && rpc.request_id != 0) {
            cancel_peer_rpc_response(rpc.request_id);
          }
        }
        for (Stage2SpeculativeScoreRpc& rpc :
             context.search_io.speculative_score_rpcs) {
          if (rpc.posted && rpc.request_id != 0) {
            cancel_peer_rpc_response(rpc.request_id);
          }
          if (rpc.process_credit_held) {
            if (rpc.posted) {
              fail_closed_peer_rpc_speculative_credit(
                rpc.target_shard, rpc.request_id);
            } else {
              release_peer_rpc_speculative_credit(
                rpc.target_shard, rpc.request_id);
            }
            rpc.process_credit_held = false;
          }
        }
        lib_assert(ready_context_queue->deactivate(context.ready_owner),
                   "stage2 shutdown deactivated a stale ready owner");
        context.ready_owner = {};
        storage_owner_maintenance_active_workers_.fetch_sub(
          1, std::memory_order_acq_rel);
      }
      retire_storage_owner_search_lane_grants(worker_id);
      current_storage_owner_maintenance_waiter_registrations_ =
        &fallback_waiter_registrations;
      clear_all_current_storage_owner_maintenance_waiters();
      current_storage_owner_maintenance_waiter_registrations_ = nullptr;
      current_storage_owner_maintenance_context_owner_ = {};
      current_storage_owner_maintenance_worker_ = false;
      current_storage_owner_thread_ = nullptr;
      return;
    }

    // Completion producers route directly to a generation-fenced context
    // ticket. Drain those O(1) events before worker-wide producers so a ready
    // continuation does not wait behind unrelated slot scans or aggregation.
    auto [ready_progressed, ready_count] = drain_ready_contexts();
    bool progressed = ready_progressed;

    // Drain every currently sendable per-peer descriptor. A second pass below
    // catches work produced by pruning in this iteration; neither pass waits
    // for a timer to form a batch.
    progressed = drive_reverse_outbox() || progressed;
    progressed = drain_reverse_completions() || progressed;
    auto [late_ready_progressed, late_ready_count] = drain_ready_contexts();
    progressed = late_ready_progressed || progressed;
    ready_count += late_ready_count;

    // Resource-credit wakes do not yet carry a context key, and local lock /
    // retry timers have no external producer. Recover them with a bounded
    // scan only when an unmatched worker wake requests one or once per 1 ms.
    // Ordinary RDMA/RPC/reverse completions stay on the ticket path.
    const size_t context_count = contexts.size();
    lib_assert(context_count != 0,
               "maintenance worker has no Stage2 context slots");
    const auto scan_now = std::chrono::steady_clock::now();
    // A context admitted on the preceding scheduler pass must not inherit the
    // 10 ms idle horizon. Conversely, once the final local context retires,
    // restore the idle cadence instead of scanning empty slots at 1 kHz.
    refresh_fallback_audit_deadline(scan_now);
    const bool periodic_scan_due = scan_now >= next_fallback_context_scan;
    const bool scan_unmatched_wake =
      resource_context_scan_requested ||
      (fallback_context_scan_requested && ready_count == 0);
    fallback_context_scan_requested = false;
    if (periodic_scan_due || scan_unmatched_wake) {
      const size_t scan_begin = active_context_cursor;
      bool fallback_scan_progressed = false;
      unpublished_context_slots_scanned += context_count;
      storage_owner_maintenance_ready_fallback_scans_.fetch_add(
        1, std::memory_order_relaxed);
      if (periodic_scan_due) {
        storage_owner_maintenance_periodic_fallback_audits_.fetch_add(
          1, std::memory_order_relaxed);
      }
      if (unpublished_context_slots_scanned >= 4096) {
        storage_owner_maintenance_context_slots_scanned_.fetch_add(
          unpublished_context_slots_scanned, std::memory_order_relaxed);
        unpublished_context_slots_scanned = 0;
      }
      for (size_t offset = 0; offset < context_count; ++offset) {
        const size_t context_index = stage2_round_robin_context_index(
          scan_begin, offset, context_count);
        Stage2Context& context = contexts[context_index];
        if (context.active) {
          const bool context_progressed = drive_owned_context(context);
          fallback_scan_progressed =
            context_progressed || fallback_scan_progressed;
          progressed = context_progressed || progressed;
        }
      }
      if (periodic_scan_due && fallback_scan_progressed) {
        storage_owner_maintenance_periodic_fallback_recoveries_.fetch_add(
          1, std::memory_order_relaxed);
      }
      active_context_cursor = scan_begin + 1 == context_count
        ? 0 : scan_begin + 1;
      fallback_audit_active = states.size() != 0;
      next_fallback_context_scan = stage2_fallback_audit_deadline(
        scan_now, states.size());
    }

    // Admission is intentionally a bounded scheduler action.  With a hot
    // shard, an unbounded admit-and-immediately-finish stream can keep every
    // executor inside this loop indefinitely, starving ready completions,
    // reverse ACKs, timeout progress, and even the observation heartbeat.
    // Eight workers still admit up to eight contexts per scheduler pass; each
    // pass must first service already-owned dependency chains.
    constexpr size_t kAdmissionBurstPerSchedulerPass = 1;
    size_t admission_burst = 0;
    while (admission_burst < kAdmissionBurstPerSchedulerPass) {
      Stage2Context* context = try_admit_context();
      if (context == nullptr) break;
      ++admission_burst;
      progressed = true;
      (void)drive_owned_context(*context);
    }
    // Admission occurs after the periodic-scan decision. Clamp here as well
    // so a newly suspended local-lock/timer retry reaches the 1 ms audit even
    // when this worker was completely idle at the start of the pass.
    refresh_fallback_audit_deadline(std::chrono::steady_clock::now());

    progressed = drive_reverse_outbox() || progressed;
    progressed = drain_reverse_completions() || progressed;

    maybe_log_storage_owner_maintenance_observation();
    if (!progressed) {
      const auto idle_started = std::chrono::steady_clock::now();
      const auto fallback_audit_interval =
        stage2_fallback_audit_interval(states.size());
      auto wake_at = std::min(
        next_fallback_context_scan,
        idle_started + fallback_audit_interval);
      {
        std::lock_guard<std::mutex> queue_lock(
          storage_owner_maintenance_mutex_);
        if (!storage_owner_stage2_tasks_.empty()) {
          const auto parameters = storage_owner_stage2_packing_.parameters();
          const size_t incomplete =
            storage_owner_maintenance_completion_ring_ == nullptr ? 0 :
              storage_owner_maintenance_completion_ring_->incomplete();
          const size_t batch_limit =
            std::max<size_t>(1, config.storage_owner_batch_max);
          const bool bulk_packing =
            stage2_bulk_packing_enabled(batch_limit);
          const size_t completion_pressure_threshold = bulk_packing
            ? std::max<size_t>(
                kStage2BulkMinimumBatch,
                batch_limit > std::numeric_limits<size_t>::max() / 2
                  ? std::numeric_limits<size_t>::max() : batch_limit * 2)
            : (storage_owner_maintenance_admission_limit_ + 1) / 2;
          const bool completion_pressure =
            completion_pressure_threshold != 0 &&
            incomplete >= completion_pressure_threshold;
          const size_t queue_pressure_threshold = bulk_packing
            ? kStage2BulkMinimumBatch
            : std::max<size_t>(2, parameters.target_batch * 2);
          const bool queue_pressure =
            storage_owner_stage2_tasks_.size() >=
              queue_pressure_threshold;
          const Stage2PackingDecision decision = decide_stage2_packing(
            storage_owner_stage2_tasks_.size(),
            batch_limit,
            parameters.target_batch,
            storage_owner_stage2_tasks_.front().queued_at,
            idle_started,
            config.storage_owner_stage2_batch_max_wait_us,
            parameters.estimated_arrival_interval_us,
            completion_pressure || queue_pressure);
          if (decision.deadline.has_value() &&
              *decision.deadline < wake_at) {
            wake_at = *decision.deadline;
          }
        }
      }
      const u64 idle_started_ns = steady_now_ns();
      std::unique_lock<std::mutex> wake_lock(wake_channel.mutex);
      wake_channel.waiters.fetch_add(
        1, std::memory_order_acq_rel);
      const auto unregister_waiter = [&]() {
        const u32 previous =
          wake_channel.waiters.fetch_sub(
            1, std::memory_order_acq_rel);
        lib_assert(previous != 0,
                   "maintenance wake waiter registration underflow");
      };
      const bool raced_before_wait =
        wake_channel.epoch.load(
          std::memory_order_acquire) != observed_wake_epoch;
      if (raced_before_wait) {
        storage_owner_maintenance_lost_wake_avoided_.fetch_add(
          1, std::memory_order_relaxed);
      } else {
        (void)wake_channel.cv.wait_until(
          wake_lock, wake_at, [&]() {
            return storage_owner_maintenance_shutdown_.load(
                     std::memory_order_acquire) ||
              wake_channel.epoch.load(
                std::memory_order_acquire) != observed_wake_epoch;
          });
      }
      fallback_context_scan_requested =
        wake_channel.epoch.load(std::memory_order_acquire) !=
          observed_wake_epoch;
      unregister_waiter();
      wake_lock.unlock();
      unpublished_idle_wait_ns += steady_now_ns() - idle_started_ns;
      ++unpublished_idle_waits;
      // Publishing in small batches avoids an atomic operation on every idle
      // loop while keeping the five-second observation error below one worker
      // times 64 ms.
      if (unpublished_idle_waits >= 64) flush_idle_timing();
    }
  }
}
