#include "memory_node/storage_owner_maintenance/detail.hh"
#include "memory_node/storage_owner_maintenance/admission_policy.hh"
#include "memory_node/storage_owner_maintenance/centroid_lifecycle_policy.hh"
#include "memory_node/storage_owner_maintenance/cleanup_scheduler.hh"
#include "memory_node/storage_owner_maintenance/cleanup_policy.hh"
#include "memory_node/storage_owner_maintenance/stage2_tracker.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"

#include <algorithm>
#include <limits>

using namespace memory_node_storage_owner_maintenance_detail;
using memory_node_storage_owner_index_detail::IncarnationLockResult;
namespace protocol = service::storage_owner;

void MemoryNode::storage_owner_maintenance_worker_loop(u32 worker_id) {
  lib_assert(worker_id < storage_owner_maintenance_worker_states_.size(),
             "storage-owner maintenance worker state missing");
  StorageOwnerThread& thread = *storage_owner_maintenance_worker_states_[worker_id];
  current_storage_owner_thread_ = &thread;
  const Configuration& config = *storage_worker_config_;
  lib_assert(num_storage_nodes_ > 0 && num_storage_nodes_ <= 64,
             "asynchronous stage2 supports at most 64 storage shards");

  using ReverseUpdateOp = service::storage_owner::ReverseUpdateOp;
  using PeerRpcType = service::storage_owner::PeerRpcType;

  struct Stage2Context {
    bool active{};
    Stage2ContextHandle handle{};
    StorageOwnerMaintenanceKind kind{
      StorageOwnerMaintenanceKind::finalize_insert};
    vec<StorageOwnerMaintenanceTask> tasks;
    vec<NodeSnapshot> targets;
    vec<NodeSnapshot> candidate_storage;
    vec<u32> candidate_counts;
    vec<vec<ReverseUpdateOp>> remote_ops_by_peer;
    vec<u64> reverse_request_ids;
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
  // Stage2 resumes one global beam; it does not collect an independent L-set
  // from every shard. Its memory footprint is therefore O(batch * L), not
  // O(batch * shard_count * L).
  const size_t candidate_capacity_per_item = construction_width;
  for (Stage2Context& context : contexts) {
    context.tasks.reserve(config.storage_owner_batch_max);
    context.targets.reserve(config.storage_owner_batch_max);
    context.candidate_storage.resize(
      static_cast<size_t>(config.storage_owner_batch_max) *
      candidate_capacity_per_item);
    for (NodeSnapshot& candidate : context.candidate_storage) {
      candidate.vector_data.resize(VamanaNode::vector_bytes());
    }
    context.candidate_counts.resize(config.storage_owner_batch_max);
    context.remote_ops_by_peer.resize(num_storage_nodes_);
    for (auto& ops : context.remote_ops_by_peer) {
      ops.reserve(static_cast<size_t>(config.R) *
                  config.storage_owner_batch_max);
    }
    context.reverse_request_ids.resize(num_storage_nodes_);
  }

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
  vec<byte_t> reverse_response_payload;
  reverse_response_payload.reserve(peer_rpc_runtime_.message_bytes);

  const auto reset_context = [&](Stage2Context& context) {
    context.active = false;
    context.tasks.clear();
    context.targets.clear();
    context.targets.resize(context.tasks.size());
    std::fill(context.candidate_counts.begin(),
              context.candidate_counts.end(), 0);
    for (auto& ops : context.remote_ops_by_peer) {
      ops.clear();
    }
    std::fill(context.reverse_request_ids.begin(),
              context.reverse_request_ids.end(), 0);
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

  const auto quiesce_cleanup_parent = [&, this](
      StorageOwnerMaintenanceTask& task) {
    if (task.cleanup_repair_only || task.cleanup_retiring) return true;
    if (task.target.is_null() ||
        !local_shard(task.target.memory_node())) {
      return false;
    }

    lock_node(task.target);
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
      return false;
    }

    GraphAdjacency adjacency;
    if (!read_graph_adjacency(task.target, adjacency)) {
      unlock_node(task.target);
      return false;
    }
    if ((header & VamanaNode::HEADER_DELETED) != 0 ||
        adjacency.deleted) {
      // An idempotent duplicate can observe the postcondition produced by the
      // original cleanup. It must not attempt to reparent from a tombstone.
      task.cleanup_retiring = true;
      task.cleanup_protected_reparented = true;
      unlock_node(task.target);
      return true;
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
    return true;
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
      vec<ReconcileReverseOp>& promotion_ops,
      vec<ReconcileReverseOp>& stable_ops,
      vec<ReconcileReverseOp>& removal_ops) {
    const auto plan = plan_stage2_backlink_reconciliation(
      span<const RemotePtr>{task.stage1_backlink_targets},
      span<const RemotePtr>{task.stage2_neighbors},
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
      storage_owner_maintenance_cv_.notify_all();
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
        }
        storage_owner_maintenance_cv_.notify_all();
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
        storage_owner_maintenance_failed_.fetch_add(
          1, std::memory_order_relaxed);
        progressed = true;
        continue;
      }

      const bool timed_out = now >= aggregate->deadline_ns;
      if (timed_out) {
        storage_owner_maintenance_rpc_timeouts_.fetch_add(
          1, std::memory_order_relaxed);
        storage_owner_maintenance_failed_.fetch_add(
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
      lib_assert(requests.erase(completion.logical_request_id),
                 "stage2 reverse completion request release failed");
      completion = {};
      progressed = true;
    }
    return progressed;
  };

  const auto prepare_local = [&](Stage2Context& context) {
    if (context.kind == StorageOwnerMaintenanceKind::cleanup_deleted_node) {
      const auto transition = states.begin_remote_search(context.handle, 0);
      lib_assert(transition == Stage2EventResult::phase_advanced,
                 "cleanup stage2 failed to enter prune_ready");
      return true;
    }

    context.targets.clear();
    context.targets.resize(context.tasks.size());
    std::fill(context.candidate_counts.begin(),
              context.candidate_counts.end(), 0);

    // Retire stale Stage1 records before issuing any remote continuation
    // reads. Failed cleanup remains in this context and is retried
    // idempotently; its maintenance sequence is never acknowledged early.
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (task.maintenance_sequence == 0) continue;
      lib_assert(local_shard(task.target.memory_node()),
                 "Stage2 must execute on the Stage1 physical shard");
      if (!storage_owner_physical_node_matches(
            task.id, task.generation, task.target)) {
        if (!complete_stale_stage2(task)) return false;
        task.maintenance_sequence = 0;
      }
    }
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.maintenance_sequence == 0) continue;

      NodeSnapshot target_snapshot;
      const bool readable = read_node_snapshot(task.target, target_snapshot);
      lib_assert(readable, "local stage2 target snapshot was unreadable");
      if (target_snapshot.deleted) {
        if (!complete_stale_stage2(task)) return false;
        task.maintenance_sequence = 0;
        continue;
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

    for (size_t item = 0; item < context.tasks.size(); ++item) {
      lib_assert(!context.tasks[item].stage1_beam.empty(),
                 "stage2 task lost its Stage1 continuation beam");
      const vec<RemotePtr>& continued_candidates =
        continue_stage2_search_candidates(
          context.tasks[item], context.targets[item], config);
      lib_assert(continued_candidates.size() <= construction_width,
                 "stage2 continuation exceeded construction width L");
      vec<NodeSnapshot> continued_snapshots = read_node_snapshots_batched(
        continued_candidates, config);
      lib_assert(continued_snapshots.size() <= candidate_capacity_per_item,
                 "stage2 continuation candidate capacity invariant failed");
      for (const NodeSnapshot& source : continued_snapshots) {
        if (source.deleted ||
            (source.header & VamanaNode::HEADER_PROVISIONAL) != 0) {
          continue;
        }
        const size_t slot = item * candidate_capacity_per_item +
                            context.candidate_counts[item]++;
        NodeSnapshot& destination = context.candidate_storage[slot];
        destination.rptr = source.rptr;
        destination.header = source.header;
        destination.id = source.id;
        destination.generation = source.generation;
        destination.deleted = source.deleted;
        lib_assert(source.vector_data.size() >= VamanaNode::vector_bytes(),
                   "stage2 local candidate vector is incomplete");
        std::memcpy(destination.vector_data.data(),
                    source.vector_data.data(), VamanaNode::vector_bytes());
      }
    }

    // The Stage1 owner continues one logical beam through one-sided RDMA.
    // There is no per-shard restart RPC and therefore no remote-search ACK
    // mask to wait for.
    constexpr u64 expected_mask = 0;
    const auto transition =
      states.begin_remote_search(context.handle, expected_mask);
    lib_assert(transition == Stage2EventResult::phase_advanced,
               "stage2 failed to enter remote_search_pending");
    return true;
  };

  const auto prepare_stage2_reverse = [&](Stage2Context& context) {
    // First finish every node record. No reverse edge or directory entry is
    // allowed to expose a destination whose vector/graph/PQ record is partial.
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.maintenance_sequence == 0) continue;

      // Arm publishes a runnable descriptor before the foreground authority
      // commit. Use a no-op placement CAS as the gate: while that mutation's
      // lease is pending it returns busy; after abort it returns stale; only a
      // committed current generation may perform any Stage2-visible graph
      // mutation. On retries after a real relocation, validate the resulting
      // physical pointer/version instead of the original Stage1 address.
      const RemotePtr gate_target = task.placement_committed
        ? task.final_target : task.target;
      const u64 gate_version = task.initial_placement_version +
        static_cast<u64>(task.placement_committed &&
                         task.final_target != task.target);
      const service::storage_owner::AuthorityPlacementItem gate{
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
      service::storage_owner::AuthorityPlacementResult gate_result;
      if (!relocate_via_authority(
            task.authority_shard, gate, gate_result, config)) {
        storage_owner_maintenance_failed_.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
      const auto gate_status = static_cast<
        service::storage_owner::AuthorityPlacementStatus>(
          gate_result.status);
      if (gate_status ==
          service::storage_owner::AuthorityPlacementStatus::busy) {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
      if (gate_status ==
          service::storage_owner::AuthorityPlacementStatus::stale) {
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

      const RemotePtr current_physical = task.placement_committed
        ? task.final_target : task.target;
      if (current_physical.is_null() ||
          !storage_owner_physical_node_matches(
            task.id, task.generation, current_physical)) {
        if (!complete_stale_stage2(task)) return false;
        task.maintenance_sequence = 0;
        continue;
      }

      if (!task.stage2_prepared) {
        hashset_t<RemotePtr> skip;
        skip.insert(task.target);
        vec<RemotePtr> globally_pruned = robust_prune_snapshots_cpu(
          context.targets[item].vector_data.data(),
          VamanaNode::vector_dtype(),
          span<const NodeSnapshot>{
            context.candidate_storage.data() +
              item * candidate_capacity_per_item,
            context.candidate_counts[item]},
          skip,
          config, config.R);

        // Freeze the graph mutation plane at the same locked boundary as the
        // rebase snapshot. Queries still traverse this record, but every
        // ordinary reverse mutation retries after observing FROZEN. Thus an
        // edge ACKed before this boundary is in observed_adjacency and no edge
        // can be ACKed in the snapshot-to-publication window.
        lock_node(task.target);
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
        GraphAdjacency observed_adjacency;
        const bool target_current =
          locked_identity_matches &&
          (locked_header & VamanaNode::HEADER_PROVISIONAL) != 0 &&
          (locked_header & (VamanaNode::HEADER_DELETED |
                            VamanaNode::HEADER_RETIRING |
                            VamanaNode::HEADER_STAGE2_FROZEN)) == 0 &&
          read_graph_adjacency(task.target, observed_adjacency) &&
          !observed_adjacency.deleted &&
          observed_adjacency.generation == task.generation;
        if (target_current) {
          auto* header_ptr = reinterpret_cast<u64*>(
            index_buffer_.get_full_buffer() +
            vamana::StorageLayoutResolver::header(task.target).offset);
          std::atomic_ref<u64>(*header_ptr).fetch_or(
            static_cast<u64>(VamanaNode::HEADER_STAGE2_FROZEN),
            std::memory_order_acq_rel);
          task.stage2_source_frozen = true;
          task.stage2_protected_children =
            observed_adjacency.provisional;
          lib_assert(task.stage2_protected_children.empty(),
                     "query-ineligible Stage1 source accepted a protected child");
        }
        unlock_node(task.target);
        if (!target_current) {
          if (!complete_stale_stage2(task)) return false;
          task.maintenance_sequence = 0;
          continue;
        }

        const vec<RemotePtr> rebase_candidates =
          merge_stage2_rebase_candidates(
            span<const RemotePtr>{globally_pruned},
            span<const RemotePtr>{task.stage1_base_neighbors},
            span<const RemotePtr>{observed_adjacency.stable});
        vec<NodeSnapshot> rebase_snapshots =
          read_node_snapshots_batched(rebase_candidates, config);
        rebase_snapshots.erase(
          std::remove_if(
            rebase_snapshots.begin(), rebase_snapshots.end(),
            [](const NodeSnapshot& candidate) {
              return !stage2_parent_is_stable(
                candidate.header, candidate.deleted);
            }),
          rebase_snapshots.end());
        task.stage2_neighbors = robust_prune_snapshots_cpu(
          context.targets[item].vector_data.data(),
          VamanaNode::vector_dtype(),
          span<const NodeSnapshot>{rebase_snapshots}, skip,
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
            storage_owner_maintenance_failed_.fetch_add(
              1, std::memory_order_relaxed);
            return false;
          }
          if (static_cast<protocol::DynamicNodeControlStatus>(
                allocation_result.status) !=
                protocol::DynamicNodeControlStatus::ok) {
            storage_owner_maintenance_pressure_yields_.fetch_add(
              1, std::memory_order_relaxed);
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
        vec<element_t> components(VamanaNode::DIM);
        decode_storage_vector_to_float(
          context.targets[item].vector_data.data(),
          VamanaNode::vector_dtype(), VamanaNode::DIM,
          components.data());

        if (task.final_target != task.target) {
          // The destination remains provisional until its complete contiguous
          // record is globally visible. Only then may reconciliation publish
          // pointers to it from existing graph nodes.
          write_new_node_on_shard(
            task.final_target, task.id,
            span<const element_t>{components}, task.stage2_neighbors,
            task.generation, true);
          // The source may itself protect concurrent Stage1 children. The
          // freeze captured that bounded plane; transfer it before publishing
          // the migrated destination so source retirement cannot orphan them.
          write_graph_adjacency(
            task.final_target, task.stage2_neighbors,
            task.stage2_protected_children, task.generation, false);
          lib_assert(set_node_provisional(task.final_target, false),
                     "failed to publish migrated Stage2 destination");
        } else {
          lock_node(task.target);
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
          // Publish the stable in-place record and reopen graph mutations in
          // one locked header transition. Do not call set_node_provisional()
          // while owning NODE_LOCK: that helper correctly refuses to race a
          // lock holder.
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

    // Re-read the complete bounded continuation set at the final publication
    // boundary.  Cached snapshots may now name deleted, retiring, provisional,
    // or ABA-reused slots.  Freeze the already materialized child graph while
    // rebasing concurrent stable additions, then RobustPrune only exact,
    // durable parents.  This is O(L) batched validation and runs once on the
    // normal path; retries repeat it only while Stage2 is already active.
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.reverse_reconciled) continue;
      lib_assert(task.stage2_plan_sealed && task.outgoing_committed &&
                   !task.final_target.is_null(),
                 "Stage2 reached final validation with an open placement plan");

      const IncarnationLockResult final_lock =
        try_lock_node(task.final_target);
      if (final_lock == IncarnationLockResult::busy) return false;
      if (final_lock == IncarnationLockResult::stale) {
        if (!complete_stale_stage2(task)) return false;
        task.maintenance_sequence = 0;
        continue;
      }
      u64 final_header = 0;
      node_t final_id = 0;
      u32 final_generation = 0;
      const bool final_identity_matches =
        read_locked_node_identity(task.final_target, final_header,
                                  final_id, final_generation) &&
        final_id == task.id && final_generation == task.generation &&
        (final_header & (VamanaNode::HEADER_DELETED |
                         VamanaNode::HEADER_PROVISIONAL |
                         VamanaNode::HEADER_RETIRING)) == 0;
      GraphAdjacency published_adjacency;
      const bool final_current = final_identity_matches &&
        read_graph_adjacency(task.final_target, published_adjacency) &&
        !published_adjacency.deleted &&
        published_adjacency.generation == task.generation;
      if (!final_current) {
        // We own the incarnation lock acquired above, but final_target may be
        // remote.  Release it with the exact header observed while locked;
        // the legacy one-byte remote unlock could otherwise clear NODE_LOCK
        // in an ABA-reused slot if identity validation failed because the
        // record changed unexpectedly.
        lib_assert(publish_locked_node_header(
                     task.final_target, final_header, 0, 0),
                   "failed to release invalid final Stage2 owner record");
        if (!complete_stale_stage2(task)) return false;
        task.maintenance_sequence = 0;
        continue;
      }
      lib_assert(publish_locked_node_header(
                   task.final_target, final_header,
                   VamanaNode::HEADER_STAGE2_FROZEN, 0),
                 "failed to freeze final Stage2 owner record");

      vec<RemotePtr> candidates;
      candidates.reserve(
        task.stage2_neighbors.size() + task.stage1_base_neighbors.size() +
        task.stage1_backlink_targets.size() +
        task.stage1_beam.size() + task.stage1_remote_frontier.size() +
        published_adjacency.stable.size());
      const auto append_candidates = [&](span<const RemotePtr> source) {
        candidates.insert(candidates.end(), source.begin(), source.end());
      };
      append_candidates(span<const RemotePtr>{task.stage2_neighbors});
      append_candidates(span<const RemotePtr>{task.stage1_base_neighbors});
      append_candidates(span<const RemotePtr>{task.stage1_backlink_targets});
      append_candidates(span<const RemotePtr>{task.stage1_remote_frontier});
      append_candidates(span<const RemotePtr>{published_adjacency.stable});
      for (const memory_node_detail::BeamEntry& entry : task.stage1_beam) {
        candidates.push_back(entry.rptr);
      }
      std::sort(candidates.begin(), candidates.end(),
                [](RemotePtr lhs, RemotePtr rhs) {
                  return lhs.raw_address < rhs.raw_address;
                });
      candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [](RemotePtr candidate) {
                         return candidate.is_null();
                       }),
        candidates.end());
      candidates.erase(
        std::unique(candidates.begin(), candidates.end()), candidates.end());

      vec<NodeSnapshot> fresh_candidates =
        read_node_snapshots_batched(candidates, config);
      fresh_candidates.erase(
        std::remove_if(
          fresh_candidates.begin(), fresh_candidates.end(),
          [](const NodeSnapshot& candidate) {
            return !stage2_parent_is_stable(
              candidate.header, candidate.deleted);
          }),
        fresh_candidates.end());
      hashset_t<RemotePtr> skip;
      skip.insert(task.target);
      skip.insert(task.final_target);
      vec<RemotePtr> refreshed_neighbors = robust_prune_snapshots_cpu(
        context.targets[item].vector_data.data(), VamanaNode::vector_dtype(),
        span<const NodeSnapshot>{fresh_candidates}, skip, config, config.R);
      lib_assert(refreshed_neighbors.size() <= config.R,
                 "final Stage2 revalidation exceeded graph degree");

      task.stage2_revalidated_home =
        memory_node_storage_owner_index_detail::choose_min_cross_shard_home(
          span<const RemotePtr>{refreshed_neighbors}, num_storage_nodes_,
          task.target.memory_node());
      // The allocation receipt and complete outgoing record form the seal.
      // Parent churn after this point is repaired in place; opening a second
      // migration for the same authority token would create ambiguous replay
      // ownership.  The first selection was made from the same durable-parent
      // predicate immediately before materialization, so this branch is only a
      // bounded churn repair, not a static placement shortcut.
      task.final_home = task.final_target.memory_node();

      const IncarnationLockResult rebase_lock =
        try_lock_node(task.final_target);
      if (rebase_lock != IncarnationLockResult::locked) return false;
      u64 rebased_header = 0;
      node_t rebased_id = 0;
      u32 rebased_generation = 0;
      GraphAdjacency rebased_adjacency;
      const bool can_publish_rebase =
        read_locked_node_identity(task.final_target, rebased_header,
                                  rebased_id, rebased_generation) &&
        rebased_id == task.id &&
        rebased_generation == task.generation &&
        (rebased_header & VamanaNode::HEADER_STAGE2_FROZEN) != 0 &&
        (rebased_header & (VamanaNode::HEADER_DELETED |
                           VamanaNode::HEADER_PROVISIONAL |
                           VamanaNode::HEADER_RETIRING)) == 0 &&
        read_graph_adjacency(task.final_target, rebased_adjacency) &&
        !rebased_adjacency.deleted;
      if (!can_publish_rebase) {
        lib_assert(publish_locked_node_header(
                     task.final_target, rebased_header, 0, 0),
                   "failed to release invalid rebased Stage2 owner record");
        return false;
      }
      write_graph_adjacency(
        task.final_target, refreshed_neighbors,
        rebased_adjacency.provisional, task.generation, false);
      lib_assert(publish_locked_node_header(
                   task.final_target, rebased_header, 0,
                   VamanaNode::HEADER_STAGE2_FROZEN),
                 "failed to publish rebased Stage2 owner record");
      task.stage2_neighbors = std::move(refreshed_neighbors);
      if (task.final_target == task.target) {
        task.stage2_source_frozen = false;
      }
    }

    ready = 0;
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

    // A Stage1 parent can be tombstoned while Stage2 is queued. Revalidate the
    // acknowledged protected slots before atomically promoting one final
    // stable bridge. No protected slot survives finalization, and Stage2 must
    // never silently commit an unreachable node.
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (task.reverse_reconciled) continue;
      const auto expected_plan = plan_stage2_backlink_reconciliation(
        span<const RemotePtr>{task.stage1_backlink_targets},
        span<const RemotePtr>{task.stage2_neighbors},
        task.stage2_promotion_committed
          ? task.stage2_promotion_parent : RemotePtr{});
      vec<RemotePtr> candidate_parents = task.stage1_backlink_targets;
      if (task.stage2_promotion_committed &&
          !task.stage2_promotion_parent.is_null()) {
        candidate_parents.push_back(task.stage2_promotion_parent);
      }
      candidate_parents.insert(candidate_parents.end(),
                               task.stage1_base_neighbors.begin(),
                               task.stage1_base_neighbors.end());
      candidate_parents.insert(candidate_parents.end(),
                               task.stage2_neighbors.begin(),
                               task.stage2_neighbors.end());
      GraphAdjacency current_child_adjacency;
      if (read_graph_adjacency(task.final_target, current_child_adjacency) &&
          !current_child_adjacency.deleted) {
        candidate_parents.insert(
          candidate_parents.end(),
          current_child_adjacency.stable.begin(),
          current_child_adjacency.stable.end());
      }
      std::sort(candidate_parents.begin(), candidate_parents.end(),
                [](RemotePtr lhs, RemotePtr rhs) {
                  return lhs.raw_address < rhs.raw_address;
                });
      candidate_parents.erase(
        std::unique(candidate_parents.begin(), candidate_parents.end()),
        candidate_parents.end());
      const vec<NodeSnapshot> parent_snapshots =
        read_node_snapshots_batched(candidate_parents, config);
      vec<RemotePtr> protected_parents;
      protected_parents.reserve(parent_snapshots.size());
      bool promotion_postcondition_holds = false;
      RemotePtr observed_promotion_parent;
      for (const NodeSnapshot& parent : parent_snapshots) {
        if (stage2_parent_is_stable(parent.header, parent.deleted)) {
          GraphAdjacency parent_adjacency;
          if (!read_graph_adjacency(parent.rptr, parent_adjacency) ||
              parent_adjacency.deleted) {
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

    // Reconcile in three explicit barriers: atomically promote one protected
    // certificate into an R-bounded stable edge, publish all remaining final
    // backlink proposals, then remove every obsolete Stage1 protected edge.
    // A timeout replays only idempotent postconditions; the mandatory stable
    // bridge is ACKed before any operation may remove the last temporary edge.
    vec<ReconcileReverseOp> promotion_ops;
    vec<ReconcileReverseOp> stable_ops;
    vec<ReconcileReverseOp> removal_ops;
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (task.reverse_reconciled) continue;
      if (!append_stage2_reconcile_ops(
            task, promotion_ops, stable_ops, removal_ops)) {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
    }
    const auto apply_phase = [&](const vec<ReconcileReverseOp>& ops) {
      return ops.empty() || apply_reconcile_ops(
        span<const ReconcileReverseOp>{ops});
    };
    if (!apply_phase(promotion_ops)) {
      storage_owner_maintenance_failed_.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (!task.reverse_reconciled) {
        task.stage2_promotion_committed = true;
      }
    }
    // Ordinary proposals are allowed to lose RobustPrune, but a target that
    // retires between validation and RPC is retried with a newly filtered
    // payload.  Temporary bridges remain intact because this barrier precedes
    // every removal.
    if (!apply_phase(stable_ops) || !apply_phase(removal_ops)) {
      storage_owner_maintenance_failed_.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      task.reverse_reconciled = true;
    }

    // The no-op gate above confirmed the public generation after this task
    // was armed and before graph publication. This token-fenced placement CAS
    // changes only its physical home, after every final reverse-edge
    // destination has ACKed the idempotent handoff.
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (!task.placement_committed) {
        const service::storage_owner::AuthorityPlacementItem placement{
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
        };
        service::storage_owner::AuthorityPlacementResult result;
        if (!relocate_via_authority(
              task.authority_shard, placement, result, config)) {
          storage_owner_maintenance_failed_.fetch_add(
            1, std::memory_order_relaxed);
          return false;
        }
        const auto status = static_cast<
          service::storage_owner::AuthorityPlacementStatus>(result.status);
        if (status ==
              service::storage_owner::AuthorityPlacementStatus::busy) {
          // A successor owns the authority lease. It will either abort (this
          // exact CAS may then proceed) or commit (the next attempt is stale).
          storage_owner_maintenance_pressure_yields_.fetch_add(
            1, std::memory_order_relaxed);
          return false;
        }
        if (status ==
              service::storage_owner::AuthorityPlacementStatus::stale) {
          if (!complete_stale_stage2(task)) return false;
          task.maintenance_sequence = 0;
          continue;
        }
        lib_assert(
          status == service::storage_owner::AuthorityPlacementStatus::committed ||
            status == service::storage_owner::AuthorityPlacementStatus::replay,
          "authority rejected a structurally valid Stage2 placement token");
        const u64 expected_resulting_version =
          task.final_target == task.target
            ? task.initial_placement_version
            : task.initial_placement_version + 1;
        lib_assert(result.resulting_placement_version ==
                     expected_resulting_version,
                   "authority placement returned an unexpected version");
        // Remember the authority result before any fallible membership RPC.
        // The final identity is now the only authority-visible placement, but
        // the old Stage1 source remains readable until the final identity has
        // been added to and published by its physical centroid owner.
        task.placement_committed = true;
      }
    }
    ready = 0;
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
      storage_owner_maintenance_failed_.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      task.centroid_committed = true;
    }

    // Membership publication precedes source tombstoning. This order is
    // observable on a lost ACK, so keep it explicit: retries may temporarily
    // retain an unadvertised live source, but can never advertise a dead-only
    // final route or reuse a source before its final generation is counted.
    for (StorageOwnerMaintenanceTask& task : context.tasks) {
      if (task.final_target == task.target || task.allocation_settled) {
        continue;
      }
      lib_assert(migrated_source_tombstone_allowed(
                   task.placement_committed, task.centroid_committed),
                 "migrated source tombstoned before final centroid publication");
      const u64 source_header = load_local_node_header_acquire(task.target);
      lib_assert((source_header &
                  VamanaNode::HEADER_CENTROID_ACCOUNTED) == 0,
                 "Stage1 source was counted before final placement");
      (void)mark_node_deleted(task.target, task.generation);
      if (!settle_dynamic_allocation(task)) {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
      task.allocation_settled = true;
    }

    constexpr u64 expected_mask = 0;
    const Stage2EventResult transition =
      states.begin_reverse(context.handle, expected_mask);
    lib_assert(transition == Stage2EventResult::phase_advanced ||
                 transition == Stage2EventResult::ready_to_finalize,
               "Stage2 finalization failed to enter reverse_pending");
    return true;
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
            storage_owner_maintenance_failed_.fetch_add(
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
      if (!task.cleanup_repair_only &&
          (!quiesce_cleanup_parent(task) ||
           !reparent_cleanup_children(task))) {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
        return false;
      }
    }

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
      storage_owner_maintenance_failed_.fetch_add(
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
    const vec<NodeSnapshot> cleanup_target_snapshots =
      read_node_snapshots_batched(cleanup_targets, config);
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
      storage_owner_maintenance_failed_.fetch_add(
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

    const Stage2ContextHandle handle = context.handle;
    reset_context(context);
    lib_assert(states.release(handle),
               "stage2 context release violated finalized generation");
    storage_owner_maintenance_active_workers_.fetch_sub(
      1, std::memory_order_acq_rel);
    storage_owner_maintenance_cv_.notify_all();
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
    reset_context(context);
    lib_assert(states.release_retryable(handle),
               "cleanup deferral retained an asynchronous request");
    storage_owner_maintenance_active_workers_.fetch_sub(
      1, std::memory_order_acq_rel);
    storage_owner_maintenance_cv_.notify_all();
  };

  const auto drive_context = [&](Stage2Context& context) {
    bool progressed = false;
    for (;;) {
      const auto snapshot = states.snapshot(context.handle);
      lib_assert(snapshot.has_value(), "active stage2 context became stale");
      switch (snapshot->phase) {
        case Stage2Phase::local_ready:
          if (!prepare_local(context)) return progressed;
          progressed = true;
          continue;
        case Stage2Phase::remote_search_pending:
          lib_failure(
            "one-sided Stage2 continuation cannot await per-shard search RPCs");
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
            return progressed;
          }
          progressed = true;
          continue;
        }
        case Stage2Phase::reverse_pending:
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

  const auto try_admit_context = [&]() -> Stage2Context* {
    const Stage2AdmissionDecision admission = decide_stage2_admission(
      states.full(),
      storage_owner_maintenance_shutdown_.load(std::memory_order_acquire),
      [&]() { return storage_owner_maintenance_foreground_busy(config); });
    if (admission == Stage2AdmissionDecision::unavailable) {
      return nullptr;
    }
    const bool foreground_pressure =
      admission == Stage2AdmissionDecision::foreground_pressure;
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
      context.active = true;
      context.handle = *handle;
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
          storage_owner_maintenance_cv_.notify_all();
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
    const bool choose_stage2 =
      !storage_owner_stage2_tasks_.empty() &&
      (!cleanup_ready || storage_owner_stage2_tasks_.front().queued_at <=
                            storage_owner_cleanup_tasks_.front().queued_at);
    if (!choose_stage2 && !cleanup_ready) {
      storage_owner_maintenance_active_workers_.fetch_sub(
        1, std::memory_order_acq_rel);
      storage_owner_maintenance_cv_.notify_all();
      return nullptr;
    }

    Stage2Context& context = *acquire_context();
    context.kind = choose_stage2
      ? StorageOwnerMaintenanceKind::finalize_insert
      : StorageOwnerMaintenanceKind::cleanup_deleted_node;

    if (choose_stage2) {
      while (!storage_owner_stage2_tasks_.empty() &&
             context.tasks.size() < batch_limit) {
        context.tasks.push_back(
          std::move(storage_owner_stage2_tasks_.front()));
        storage_owner_stage2_tasks_.pop_front();
      }
      storage_owner_stage2_batches_.fetch_add(1, std::memory_order_relaxed);
      storage_owner_stage2_batched_items_.fetch_add(
        context.tasks.size(), std::memory_order_relaxed);
    } else {
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
    lock.unlock();
    storage_owner_maintenance_cv_.notify_all();
    lib_assert(!context.tasks.empty(),
               "stage2 admitted an empty maintenance context");
    return &context;
  };

  for (;;) {
    if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
      if (storage_owner_reverse_outbox_ != nullptr) {
        (void)storage_owner_reverse_outbox_->erase_queued_worker(worker_id);
        for (;;) {
          const auto wire_request_id =
            storage_owner_reverse_outbox_->discard_owned_aggregate(worker_id);
          if (!wire_request_id.has_value()) break;
          cancel_peer_rpc_response(*wire_request_id);
        }
      }
      for (Stage2Context& context : contexts) {
        if (!context.active) {
          continue;
        }
        storage_owner_maintenance_active_workers_.fetch_sub(
          1, std::memory_order_acq_rel);
      }
      current_storage_owner_thread_ = nullptr;
      return;
    }

    // Drain every currently sendable per-peer descriptor before polling
    // contexts. A second pass below catches work produced by pruning in this
    // iteration; neither pass waits for a timer to form a batch.
    bool progressed = drive_reverse_outbox();
    progressed = drain_reverse_completions() || progressed;
    for (Stage2Context& context : contexts) {
      if (context.active) {
        progressed = drive_context(context) || progressed;
      }
    }

    while (Stage2Context* context = try_admit_context()) {
      progressed = true;
      (void)drive_context(*context);
    }

    progressed = drive_reverse_outbox() || progressed;
    progressed = drain_reverse_completions() || progressed;

    maybe_log_storage_owner_maintenance_observation();
    if (!progressed) {
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::milliseconds(1));
    }
  }
}
