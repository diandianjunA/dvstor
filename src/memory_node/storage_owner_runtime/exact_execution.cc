#include "memory_node/storage_owner_runtime/detail.hh"

#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"
#include "memory_node/storage_owner_runtime/exact_update_contract.hh"

using namespace memory_node_storage_owner_runtime_detail;

bool MemoryNode::execute_storage_owner_batch_items_exact(
    const node_t* ids,
    const service::storage_owner::MutationKind* kinds,
    const byte_t* raw_vectors,
    const u64* operation_ids,
    u32 source_client,
    size_t item_count,
    InsertBreakdownCounters& breakdown,
    ExactInsertPhaseCounters& exact_phases,
    const Configuration& config,
    vec<vec<u64>>* invalidated_neighbors,
    vec<u32>* statuses,
    vec<service::storage_owner::MutationResult>* results,
    const std::function<void(size_t)>& on_terminal) {
  using namespace service::storage_owner;
  using BeginState =
    memory_node_storage_owner_index_detail::AuthorityBeginState;
  using SnapshotState =
    memory_node_storage_owner_index_detail::StableNodeSnapshotState;

  if (item_count == 0) return true;
  lib_assert(config.synchronous_exact_updates_enabled(),
             "synchronous exact execution used outside coupled mode");
  lib_assert(ids != nullptr && kinds != nullptr && raw_vectors != nullptr &&
               operation_ids != nullptr,
             "synchronous exact request omitted identity or vector bytes");
  lib_assert(current_storage_owner_thread_ != nullptr &&
               (num_storage_nodes_ <= 1 ||
                current_storage_owner_thread_->has_peer_scratch()) &&
               current_storage_owner_thread_->post_balances.size() == 1,
             "synchronous exact coordinator requires one private RDMA lane");

  if (statuses != nullptr) {
    statuses->assign(
      item_count, static_cast<u32>(MutationStatus::failed));
  }
  if (results != nullptr) results->assign(item_count, {});
  if (invalidated_neighbors != nullptr) {
    invalidated_neighbors->assign(item_count, {});
  }

  const auto shutting_down = [&]() {
    return storage_insert_shutdown_.load(std::memory_order_acquire);
  };
  // Shutdown may cancel an exact mutation only before its first physical
  // graph publication. Once visible state exists, peer progress remains live
  // while this worker drives every idempotent step through authority commit.
  bool physical_mutation_started = false;
  const auto cancellable_shutdown = [&]() {
    return shutting_down() && !physical_mutation_started;
  };
  const auto retry_pause = [&]() {
    poll_peer_send_cq();
    std::this_thread::yield();
  };
  const auto map_begin_failure = [](BeginState state) {
    switch (state) {
      case BeginState::already_exists:
        return MutationStatus::already_exists;
      case BeginState::not_found:
        return MutationStatus::not_found;
      case BeginState::already_deleted:
        return MutationStatus::already_deleted;
      case BeginState::prepared:
      case BeginState::replay:
      case BeginState::committed_replay:
      case BeginState::busy:
      case BeginState::conflict:
        return MutationStatus::failed;
    }
    return MutationStatus::failed;
  };

  const auto apply_reconcile_ops = [&](span<const ReconcileReverseOp> ops) {
    if (ops.empty()) return true;
    vec<ReconcileReverseResult> reconcile_results;
    if (!reconcile_reverse_ops_one_sided(
          ops, config, reconcile_results)) {
      return false;
    }
    if (reconcile_results.size() != ops.size()) return false;
    for (size_t index = 0; index < ops.size(); ++index) {
      if (!memory_node_storage_owner_index_detail::
            reconcile_reverse_postcondition_holds(
              ops[index], reconcile_results[index])) {
        return false;
      }
    }
    return true;
  };
  const auto apply_reconcile_until_complete = [&] (
      span<const ReconcileReverseOp> ops) {
    while (!apply_reconcile_ops(ops)) {
      if (cancellable_shutdown()) return false;
      retry_pause();
    }
    return true;
  };
  for (size_t index = 0; index < item_count; ++index) {
    physical_mutation_started = false;
    if (!exact_mutation_kind_allowed(kinds[index]) ||
        operation_ids[index] == 0 ||
        ids[index] >= config.vector_id_namespace_size) {
      if (on_terminal) on_terminal(index);
      continue;
    }
    const MutationKind kind = kinds[index];
    const AuthorityOperationToken operation{
      .source_client = source_client,
      .item_index = 0,
      .client_batch_id = operation_ids[index],
    };
    AuthorityBeginResult begin;
    for (;;) {
      const auto prepare_started = std::chrono::steady_clock::now();
      begin = begin_authority_mutation(
        ids[index], kind, operation, storage_id_);
      breakdown.storage_owner_prepare_mutation_ns +=
        elapsed_ns_since(prepare_started);
      if (begin.state != BeginState::busy || shutting_down()) break;
      retry_pause();
    }

    if (begin.state == BeginState::committed_replay) {
      if (statuses != nullptr) {
        (*statuses)[index] = static_cast<u32>(MutationStatus::ok);
      }
      if (results != nullptr) {
        (*results)[index] = MutationResult{
          .new_rptr_raw = begin.replay_result.new_pointer.raw_address,
          .old_rptr_raw = begin.replay_result.old_pointer.raw_address,
          .generation = begin.replay_result.generation,
          .maintenance_sequence =
            kExactUpdateContract.public_maintenance_sequence,
        };
      }
      if (on_terminal) on_terminal(index);
      continue;
    }
    if (!begin.acquired()) {
      if (statuses != nullptr) {
        (*statuses)[index] = static_cast<u32>(
          map_begin_failure(begin.state));
      }
      if (on_terminal) on_terminal(index);
      continue;
    }

    lib_assert(begin.previous.current.is_null(),
               "append-only exact insert acquired an existing authority entry");

    const u64 mutation_cookie = exact_update_mutation_cookie(
      source_client, operation_ids[index], ids[index], begin.generation);
    RemotePtr new_pointer;
    RemotePtr mandatory_parent;
    vec<RemotePtr> selected_neighbors;
    std::function<vec<RemotePtr>()> discover_live_parents;
    std::function<bool()> ensure_mandatory_bridge;
    thread_local vec<element_t> decoded;
    decoded.resize(VamanaNode::DIM);
    const byte_t* vector_bytes =
      raw_vectors + index * VamanaNode::vector_bytes();
    decode_storage_vector_to_float(
      vector_bytes, VamanaNode::vector_dtype(), VamanaNode::DIM,
      decoded.data());
    discover_live_parents = [&]() {
      for (;;) {
        const vec<RemotePtr> entries = local_centroid_route_entries();
        lib_assert(!entries.empty(),
                   "synchronous exact search has no local centroid entry");
        vec<BeamEntry> local_beam;
        vec<RemotePtr> remote_frontier;
        const auto search_started = std::chrono::steady_clock::now();
        const auto stage1_search_started = std::chrono::steady_clock::now();
        (void)partition_local_search_candidates(
          span<const element_t>{decoded}, entries, config, &breakdown,
          vector_bytes, &local_beam, &remote_frontier, false);
        exact_phases.stage1_local_search_ns +=
          elapsed_ns_since(stage1_search_started);
        StorageOwnerMaintenanceTask continuation;
        continuation.stage1_beam = std::move(local_beam);
        continuation.stage1_remote_frontier = std::move(remote_frontier);
        NodeSnapshot target;
        target.vector_data.assign(
          vector_bytes, vector_bytes + VamanaNode::vector_bytes());
        const auto continuation_started = std::chrono::steady_clock::now();
        const vec<RemotePtr> candidates =
          continue_stage2_search_candidates(
            continuation, target, config, false);
        exact_phases.stage2_global_continuation_ns +=
          elapsed_ns_since(continuation_started);
        breakdown.storage_owner_search_ns +=
          elapsed_ns_since(search_started);

        vec<NodeSnapshot> snapshots;
        vec<SnapshotState> states;
        const auto snapshot_started = std::chrono::steady_clock::now();
        snapshots = read_node_snapshots_batched(
          candidates, config, "synchronous_exact_prune", &states);
        exact_phases.final_candidate_snapshot_ns +=
          elapsed_ns_since(snapshot_started);
        if (std::find(states.begin(), states.end(),
                      SnapshotState::retryable) != states.end()) {
          if (cancellable_shutdown()) return vec<RemotePtr>{};
          retry_pause();
          continue;
        }
        snapshots.erase(
          std::remove_if(
            snapshots.begin(), snapshots.end(),
            [&](const NodeSnapshot& node) {
              return node.rptr == new_pointer || node.deleted ||
                !VamanaNode::stable_graph_mutation_allowed(node.header) ||
                (node.header & VamanaNode::HEADER_CENTROID_ACCOUNTED) == 0;
            }),
          snapshots.end());
        hashset_t<RemotePtr> skip;
        if (!new_pointer.is_null()) skip.insert(new_pointer);
        const auto prune_started = std::chrono::steady_clock::now();
        vec<RemotePtr> selected = robust_prune_snapshots_cpu(
          vector_bytes, VamanaNode::vector_dtype(), snapshots, skip,
          config, config.R);
        breakdown.storage_owner_prune_ns +=
          elapsed_ns_since(prune_started);
        if (!selected.empty()) return selected;
        if (cancellable_shutdown()) return vec<RemotePtr>{};
        retry_pause();
      }
    };

    selected_neighbors = discover_live_parents();
    if (selected_neighbors.empty()) {
      (void)abort_authority_mutation(ids[index], operation);
      return false;
    }

    const auto allocation_started = std::chrono::steady_clock::now();
    physical_mutation_started = true;
    new_pointer = allocate_local_node();
    breakdown.storage_owner_allocate_node_ns +=
      elapsed_ns_since(allocation_started);
    const auto write_started = std::chrono::steady_clock::now();
    write_new_node(
      new_pointer, ids[index], span<const element_t>{decoded},
      selected_neighbors, begin.generation, false);
    breakdown.storage_owner_write_node_ns +=
      elapsed_ns_since(write_started);

    // Ordinary reverse additions are allowed to lose RobustPrune. Install
    // one explicit stable incoming certificate first; on a retired/racing
    // parent, rotate through the validated set and then repeat global search
    // instead of retrying one stale target forever.
    const auto reverse_started = std::chrono::steady_clock::now();
    ensure_mandatory_bridge = [&]() {
      vec<RemotePtr> parent_round;
      parent_round.reserve(selected_neighbors.size() + 1);
      if (!mandatory_parent.is_null()) {
        parent_round.push_back(mandatory_parent);
      }
      for (const RemotePtr parent : selected_neighbors) {
        if (std::find(parent_round.begin(), parent_round.end(), parent) ==
            parent_round.end()) {
          parent_round.push_back(parent);
        }
      }
      for (;;) {
        for (const RemotePtr parent : parent_round) {
          const ReconcileReverseOp mandatory{
            .target_raw = parent.raw_address,
            .old_candidate_raw = 0,
            .new_candidate_raw = new_pointer.raw_address,
            .placement_sequence = mutation_cookie,
            .id = ids[index],
            .generation = begin.generation,
            .kind = static_cast<u32>(
              ReconcileReverseOpKind::promote_stable_bridge),
          };
          if (apply_reconcile_ops(
                span<const ReconcileReverseOp>{&mandatory, 1})) {
            if (invalidated_neighbors != nullptr) {
              (void)record_exact_completed_invalidation(
                &(*invalidated_neighbors)[index], parent.raw_address);
            }
            mandatory_parent = parent;
            return true;
          }
          if (cancellable_shutdown()) return false;
          retry_pause();
        }
        parent_round = discover_live_parents();
        if (parent_round.empty()) return false;
      }
    };
    if (!ensure_mandatory_bridge()) {
      return false;
    }

    vec<ReconcileReverseOp> additions;
    additions.reserve(selected_neighbors.size());
    for (const RemotePtr neighbor : selected_neighbors) {
      if (neighbor == mandatory_parent) continue;
      additions.push_back(ReconcileReverseOp{
        .target_raw = neighbor.raw_address,
        .old_candidate_raw = 0,
        .new_candidate_raw = new_pointer.raw_address,
        .placement_sequence = mutation_cookie,
        .id = ids[index],
        .generation = begin.generation,
        .kind = static_cast<u32>(ReconcileReverseOpKind::add),
      });
      if (invalidated_neighbors != nullptr) {
        (*invalidated_neighbors)[index].push_back(
          neighbor.raw_address);
      }
    }
    if (!apply_reconcile_until_complete(additions)) {
      return false;
    }
    breakdown.storage_owner_remote_reverse_ns +=
      elapsed_ns_since(reverse_started);

    const CentroidMembershipOp add_membership{
      .node_raw = new_pointer.raw_address,
      .maintenance_sequence = mutation_cookie,
      .id = ids[index],
      .generation = begin.generation,
      .kind = static_cast<u32>(CentroidMembershipKind::add),
    };
    lib_assert(RemotePtr{add_membership.node_raw}.memory_node() == storage_id_,
               "append-only exact insert attempted remote centroid work");
    while (!apply_local_centroid_membership_ops(
             span<const CentroidMembershipOp>{&add_membership, 1})) {
      if (cancellable_shutdown()) return false;
      retry_pause();
    }

    // Ordinary reverse updates and concurrent graph pruning can invalidate the
    // first selected parent. Re-establish the stable incoming certificate as
    // this operation's final physical graph publication, then record the same
    // generation in the authority directory immediately below. A later
    // independently linearized prune may remove the edge; no cross-commit
    // parent lease is implied here.
    const auto final_bridge_started = std::chrono::steady_clock::now();
    if (!ensure_mandatory_bridge()) return false;
    breakdown.storage_owner_remote_reverse_ns +=
      elapsed_ns_since(final_bridge_started);

    const auto publish_started = std::chrono::steady_clock::now();
    const AuthorityCommitState committed = commit_authority_mutation(
      ids[index], operation, new_pointer, begin.generation, false,
      kExactUpdateContract.public_maintenance_sequence);
    breakdown.storage_owner_publish_mutation_ns +=
      elapsed_ns_since(publish_started);
    lib_assert(committed == AuthorityCommitState::committed ||
                 committed == AuthorityCommitState::replay,
               "synchronous exact authority commit lost its active lease");

    if (statuses != nullptr) {
      (*statuses)[index] = static_cast<u32>(MutationStatus::ok);
    }
    if (results != nullptr) {
      (*results)[index] = MutationResult{
        .new_rptr_raw = new_pointer.raw_address,
        .old_rptr_raw = 0,
        .generation = begin.generation,
        .maintenance_sequence =
          kExactUpdateContract.public_maintenance_sequence,
      };
    }
    if (on_terminal) on_terminal(index);
  }
  return true;
}
