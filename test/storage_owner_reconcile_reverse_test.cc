#include <algorithm>
#include <cassert>

#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"
#include "memory_node/storage_owner_maintenance/cleanup_policy.hh"
#include "vamana/vamana_node.hh"

namespace {

using service::storage_owner::ReconcileReverseOp;
using service::storage_owner::ReconcileReverseOpKind;

RemotePtr pointer(u32 shard, u64 offset) {
  return RemotePtr{shard, offset};
}

ReconcileReverseOp operation(ReconcileReverseOpKind kind,
                             RemotePtr target,
                             RemotePtr old_candidate,
                             RemotePtr new_candidate,
                             u64 placement_sequence = 9) {
  return ReconcileReverseOp{
    .target_raw = target.raw_address,
    .old_candidate_raw = old_candidate.raw_address,
    .new_candidate_raw = new_candidate.raw_address,
    .placement_sequence = placement_sequence,
    .id = 42,
    .generation = 7,
    .kind = static_cast<u32>(kind),
  };
}

void test_protocol_layout_is_additive() {
  static_assert(sizeof(service::storage_owner::ReverseUpdateOp) == 32);
  static_assert(sizeof(ReconcileReverseOp) == 48);
  static_assert(sizeof(service::storage_owner::ReconcileReverseResult) == 16);
  static_assert(static_cast<u32>(ReconcileReverseOpKind::ensure_reachable) ==
                4);
  static_assert(static_cast<u32>(
                  ReconcileReverseOpKind::promote_stable_bridge) == 5);
  static_assert(!memory_node_storage_owner_index_detail::
                  reconcile_kind_needs_new_identity(
                    ReconcileReverseOpKind::remove_if_present));
  static_assert(memory_node_storage_owner_index_detail::
                  reconcile_kind_needs_new_identity(
                    ReconcileReverseOpKind::promote_stable_bridge));
}

void test_bounded_prune_rejection_is_a_terminal_ordinary_postcondition() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr old_candidate = pointer(0, 0x2000);
  const RemotePtr new_candidate = pointer(1, 0x3000);

  auto add = operation(
    ReconcileReverseOpKind::add, target, RemotePtr{}, new_candidate);
  service::storage_owner::ReconcileReverseResult rejected_add{
    .placement_sequence = add.placement_sequence,
  };
  assert(memory_node_storage_owner_index_detail::
           reconcile_reverse_postcondition_holds(add, rejected_add));

  auto replace = operation(
    ReconcileReverseOpKind::replace_or_add,
    target, old_candidate, new_candidate);
  service::storage_owner::ReconcileReverseResult pruned_replacement{
    .placement_sequence = replace.placement_sequence,
    .removed = 1,
  };
  assert(memory_node_storage_owner_index_detail::
           reconcile_reverse_postcondition_holds(
             replace, pruned_replacement));

  auto bridge = operation(
    ReconcileReverseOpKind::ensure_reachable,
    target, old_candidate, new_candidate);
  assert(!memory_node_storage_owner_index_detail::
            reconcile_reverse_postcondition_holds(
              bridge, pruned_replacement));
  pruned_replacement.accepted = 1;
  assert(memory_node_storage_owner_index_detail::
           reconcile_reverse_postcondition_holds(
             bridge, pruned_replacement));

  const auto promotion = operation(
    ReconcileReverseOpKind::promote_stable_bridge,
    target, old_candidate, new_candidate);
  service::storage_owner::ReconcileReverseResult promoted{
    .placement_sequence = promotion.placement_sequence,
    .accepted = 1,
  };
  assert(!memory_node_storage_owner_index_detail::
            reconcile_reverse_postcondition_holds(promotion, promoted));
  promoted.removed = 1;
  assert(memory_node_storage_owner_index_detail::
           reconcile_reverse_postcondition_holds(promotion, promoted));
}

void test_stage2_backlink_plan_cleans_selected_and_unselected_bridges() {
  const RemotePtr stage1_first = pointer(0, 0x1000);
  const RemotePtr stage1_second = pointer(0, 0x2000);
  const RemotePtr final_first = pointer(1, 0x3000);
  const RemotePtr final_second = pointer(1, 0x4000);

  // Prefer a protected parent that survived Stage2 pruning. The selected
  // bridge is consumed by promotion; the unselected Stage1 bridge is the only
  // member of the explicit cleanup barrier.
  const vec<RemotePtr> stage1{stage1_first, stage1_second};
  const vec<RemotePtr> partly_overlapping_final{
    final_first, stage1_second, final_second};
  auto plan = memory_node_storage_owner_maintenance_detail::
    plan_stage2_backlink_reconciliation(
      span<const RemotePtr>{stage1},
      span<const RemotePtr>{partly_overlapping_final});
  assert(plan.promotion_target == stage1_second);
  assert(plan.promotion_consumes_stage1_bridge);
  assert(plan.ordinary_stable_targets.size() == 2);
  assert(plan.obsolete_stage1_bridges ==
         vec<RemotePtr>{stage1_first});

  // Even when no protected parent survives outgoing prune, promotion consumes
  // a live Stage1 certificate first. This anchors the insertion in the old
  // published graph instead of allowing a fresh-node dependency cycle.
  const vec<RemotePtr> disjoint_final{final_first, final_second};
  plan = memory_node_storage_owner_maintenance_detail::
    plan_stage2_backlink_reconciliation(
      span<const RemotePtr>{stage1}, span<const RemotePtr>{disjoint_final});
  assert(plan.promotion_target == stage1_first);
  assert(plan.promotion_consumes_stage1_bridge);
  assert(plan.ordinary_stable_targets.size() == 2);
  assert(plan.obsolete_stage1_bridges ==
         vec<RemotePtr>{stage1_second});

  // If all temporary parents retired before Stage2 ran, the same mandatory
  // stable operation is created at the first final parent without claiming a
  // nonexistent protected handoff.
  plan = memory_node_storage_owner_maintenance_detail::
    plan_stage2_backlink_reconciliation(
      span<const RemotePtr>{}, span<const RemotePtr>{disjoint_final});
  assert(plan.promotion_target == final_first);
  assert(!plan.promotion_consumes_stage1_bridge);
  assert(plan.obsolete_stage1_bridges.empty());

  // An empty final outgoing set is legal under heavy churn as long as a live
  // Stage1 protected certificate can be promoted. It must not leave the
  // maintenance sequence waiting forever for a non-existent final proposal.
  plan = memory_node_storage_owner_maintenance_detail::
    plan_stage2_backlink_reconciliation(
      span<const RemotePtr>{stage1}, span<const RemotePtr>{});
  assert(plan.promotion_target == stage1_first);
  assert(plan.promotion_consumes_stage1_bridge);
  assert(plan.ordinary_stable_targets.empty());
  assert(plan.obsolete_stage1_bridges ==
         vec<RemotePtr>{stage1_second});

  // A retry keeps an already ACKed stable certificate even after its protected
  // slot has been consumed and disappeared from stage1_bridges.
  plan = memory_node_storage_owner_maintenance_detail::
    plan_stage2_backlink_reconciliation(
      span<const RemotePtr>{}, span<const RemotePtr>{disjoint_final},
      stage1_first);
  assert(plan.promotion_target == stage1_first);
  assert(!plan.promotion_consumes_stage1_bridge);
  assert(plan.ordinary_stable_targets.size() == 2);

  // If both Stage1 parents remain final, one is promoted and the other is
  // reconciled as ordinary stable work; neither appears in obsolete cleanup.
  const vec<RemotePtr> fully_overlapping_final{
    stage1_first, stage1_second};
  plan = memory_node_storage_owner_maintenance_detail::
    plan_stage2_backlink_reconciliation(
      span<const RemotePtr>{stage1},
      span<const RemotePtr>{fully_overlapping_final});
  assert(plan.promotion_target == stage1_first);
  assert(plan.promotion_consumes_stage1_bridge);
  assert(plan.ordinary_stable_targets.size() == 1);
  assert(plan.ordinary_stable_targets[0].target == stage1_second);
  assert(plan.ordinary_stable_targets[0].had_stage1_bridge);
  assert(plan.obsolete_stage1_bridges.empty());
}

void test_final_parent_eligibility_rejects_churn_and_dependency_cycles() {
  using memory_node_storage_owner_maintenance_detail::
    stage2_parent_is_stable;
  const u64 stable = VamanaNode::make_header(
    4, VamanaNode::HEADER_CENTROID_ACCOUNTED);
  assert(stage2_parent_is_stable(stable, false));
  assert(!stage2_parent_is_stable(
    stable | VamanaNode::HEADER_RETIRING, false));
  assert(!stage2_parent_is_stable(
    stable | VamanaNode::HEADER_PROVISIONAL, false));
  assert(!stage2_parent_is_stable(
    stable | VamanaNode::HEADER_STAGE2_FROZEN, false));
  assert(!stage2_parent_is_stable(stable, true));

  // A freshly materialized but not yet centroid-accounted destination cannot
  // parent another fresh destination. Two concurrent inserts therefore cannot
  // satisfy reachability solely by pointing at one another.
  const u64 fresh_uncommitted = VamanaNode::make_header(5);
  assert(!stage2_parent_is_stable(fresh_uncommitted, false));
}

void test_stage2_freeze_closes_the_ack_to_publish_window() {
  const RemotePtr baseline = pointer(0, 0x1000);
  const RemotePtr globally_selected = pointer(0, 0x2000);
  const RemotePtr acknowledged_before_freeze = pointer(1, 0x3000);

  u64 header = VamanaNode::make_header(
    7, VamanaNode::HEADER_PROVISIONAL);
  assert(!VamanaNode::graph_mutation_quiesced(header));

  // Model an ordinary reverse update that completed before Stage2 acquired
  // the source lock. The locked freeze snapshot must include that ACKed edge.
  const vec<RemotePtr> observed{baseline, acknowledged_before_freeze};
  header |= VamanaNode::HEADER_STAGE2_FROZEN;
  assert(VamanaNode::graph_mutation_quiesced(header));
  const vec<RemotePtr> rebased =
    memory_node_storage_owner_maintenance_detail::
      merge_stage2_rebase_candidates(
        span<const RemotePtr>{&globally_selected, 1},
        span<const RemotePtr>{&baseline, 1},
        span<const RemotePtr>{observed});
  assert(std::find(rebased.begin(), rebased.end(),
                   acknowledged_before_freeze) != rebased.end());

  // While FROZEN, a later mutation cannot truthfully ACK, so publication of
  // this rebased set cannot overwrite any completed graph update. In-place
  // publication reopens mutations only after its final adjacency write.
  header = VamanaNode::complete_in_place_stage2_header(header);
  assert(!VamanaNode::graph_mutation_quiesced(header));
  assert((header & VamanaNode::HEADER_PROVISIONAL) == 0);
  assert(VamanaNode::stable_graph_mutation_allowed(header));
}

void test_stage2_freeze_lifecycle_has_one_mutation_boundary() {
  const u64 stable = VamanaNode::make_header(11);
  assert(VamanaNode::stable_graph_mutation_allowed(stable));

  // An ordinary add that reaches a source after Stage2's lock boundary must
  // return retry/stale instead of ACKing a write that final publication could
  // overwrite. NODE_LOCK itself is ignored because callers evaluate this
  // predicate while holding it.
  const u64 frozen_stable =
    stable | VamanaNode::HEADER_NODE_LOCK |
    VamanaNode::HEADER_STAGE2_FROZEN;
  assert(VamanaNode::graph_mutation_quiesced(frozen_stable));
  assert(!VamanaNode::stable_graph_mutation_allowed(frozen_stable));

  // FROZEN is not a query tombstone: snapshot/exact-query admission rejects
  // NODE_LOCK transiently, then accepts the same record after unlock because
  // neither DELETED nor incarnation changed.
  const u64 frozen_unlocked =
    frozen_stable & ~static_cast<u64>(VamanaNode::HEADER_NODE_LOCK);
  assert((frozen_unlocked & (VamanaNode::HEADER_NODE_LOCK |
                             VamanaNode::HEADER_DELETED)) == 0);

  // In-place publication clears PROVISIONAL and FROZEN together after the
  // final graph write, reopening ordinary mutation admission.
  const u64 in_place = VamanaNode::complete_in_place_stage2_header(
    frozen_unlocked | VamanaNode::HEADER_PROVISIONAL);
  assert((in_place & (VamanaNode::HEADER_PROVISIONAL |
                      VamanaNode::HEADER_STAGE2_FROZEN)) == 0);
  assert(VamanaNode::stable_graph_mutation_allowed(in_place));

  // Migration publishes another record, so its old source must retain the
  // freeze until the source tombstone is visible. It must never reopen as a
  // second mutable physical incarnation of the same logical generation.
  u64 migrated_source = frozen_unlocked | VamanaNode::HEADER_PROVISIONAL;
  assert(VamanaNode::graph_mutation_quiesced(migrated_source));
  assert(!VamanaNode::stable_graph_mutation_allowed(migrated_source));
  migrated_source |= VamanaNode::HEADER_DELETED;
  assert((migrated_source & VamanaNode::HEADER_STAGE2_FROZEN) != 0);
  assert(!VamanaNode::stable_graph_mutation_allowed(migrated_source));
}

void test_stale_stage2_retires_only_an_authority_detached_source() {
  const RemotePtr source = pointer(0, 0x1000);
  const RemotePtr destination = pointer(1, 0x2000);
  using memory_node_storage_owner_maintenance_detail::
    stale_stage2_owns_source_retirement;

  // Before the placement CAS, a successor captures source and its ordinary
  // cleanup owns retirement. Deleting it here could bypass child reparenting.
  assert(!stale_stage2_owns_source_retirement(
    false, source, destination));
  assert(!stale_stage2_owns_source_retirement(true, source, source));

  // After S->D commits, a successor cleans D only. A stale old continuation
  // must retire S itself or allocation settlement and its ticket never finish.
  assert(stale_stage2_owns_source_retirement(
    true, source, destination));
}

void test_equivalent_replace_is_in_place_and_idempotent() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr left = pointer(0, 0x2000);
  const RemotePtr old_candidate = pointer(0, 0x3000);
  const RemotePtr right = pointer(0, 0x4000);
  const RemotePtr new_candidate = pointer(1, 0x5000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::replace_or_add,
    target, old_candidate, new_candidate);
  vec<RemotePtr> neighbors{left, old_candidate, right};
  u32 prune_calls = 0;
  const auto prune = [&](const vec<RemotePtr>& candidates) {
    ++prune_calls;
    return candidates;
  };

  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
      op, true, true, true, true, 3, neighbors, prune);
  assert(result.accepted && result.replaced && result.removed && !result.stale);
  assert(result.placement_sequence == op.placement_sequence);
  assert((neighbors == vec<RemotePtr>{left, new_candidate, right}));
  assert(prune_calls == 0);

  result = memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
    op, true, true, true, false, 3, neighbors, prune);
  assert(result.accepted && !result.replaced && result.removed && !result.stale);
  assert((neighbors == vec<RemotePtr>{left, new_candidate, right}));
  assert(prune_calls == 0);
}

void test_remove_if_present_has_an_idempotent_postcondition() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr old_candidate = pointer(0, 0x2000);
  const RemotePtr survivor = pointer(0, 0x3000);
  const RemotePtr canonical_new = pointer(1, 0x4000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::remove_if_present,
    target, old_candidate, canonical_new);
  vec<RemotePtr> neighbors{old_candidate, survivor};
  const auto never_prune = [](const vec<RemotePtr>&) -> vec<RemotePtr> {
    assert(false);
    return {};
  };

  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
      op, true, true, true, false, 2, neighbors, never_prune);
  assert(result.removed && !result.accepted && !result.replaced && !result.stale);
  assert((neighbors == vec<RemotePtr>{survivor}));

  result = memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
    op, true, true, true, false, 2, neighbors, never_prune);
  assert(result.removed && !result.stale);
  assert((neighbors == vec<RemotePtr>{survivor}));
}

void test_add_uses_robust_prune_only_at_the_degree_bound() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr first = pointer(0, 0x2000);
  const RemotePtr second = pointer(0, 0x3000);
  const RemotePtr new_candidate = pointer(1, 0x4000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::add,
    target, RemotePtr{}, new_candidate);
  vec<RemotePtr> neighbors{first, second};
  u32 prune_calls = 0;
  const auto prune = [&](const vec<RemotePtr>& candidates) {
    ++prune_calls;
    assert((candidates ==
            vec<RemotePtr>{first, second, new_candidate}));
    return vec<RemotePtr>{first, new_candidate};
  };

  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
      op, true, true, true, false, 2, neighbors, prune);
  assert(result.accepted && !result.stale);
  assert((neighbors == vec<RemotePtr>{first, new_candidate}));
  assert(prune_calls == 1);

  result = memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
    op, true, true, true, false, 2, neighbors, prune);
  assert(result.accepted && !result.stale);
  assert((neighbors == vec<RemotePtr>{first, new_candidate}));
  assert(prune_calls == 1);
}

void test_replace_conflict_reenters_robust_prune() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr old_candidate = pointer(0, 0x2000);
  const RemotePtr survivor = pointer(0, 0x3000);
  const RemotePtr new_candidate = pointer(1, 0x4000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::replace_or_add,
    target, old_candidate, new_candidate);
  vec<RemotePtr> neighbors{old_candidate, survivor};
  u32 prune_calls = 0;
  const auto prune = [&](const vec<RemotePtr>& candidates) {
    ++prune_calls;
    assert((candidates == vec<RemotePtr>{survivor, new_candidate}));
    // Model alpha RobustPrune rejecting the relocated candidate.
    return vec<RemotePtr>{survivor};
  };

  const auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
      op, true, true, true, false, 2, neighbors, prune);
  assert(!result.accepted && !result.replaced && result.removed && !result.stale);
  assert((neighbors == vec<RemotePtr>{survivor}));
  assert(prune_calls == 1);
}

void test_identity_or_liveness_mismatch_is_stale_and_non_mutating() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr old_candidate = pointer(0, 0x2000);
  const RemotePtr new_candidate = pointer(1, 0x3000);
  const ReconcileReverseOp replace = operation(
    ReconcileReverseOpKind::replace_or_add,
    target, old_candidate, new_candidate);
  const vec<RemotePtr> original{old_candidate};
  const auto never_prune = [](const vec<RemotePtr>&) -> vec<RemotePtr> {
    assert(false);
    return {};
  };

  vec<RemotePtr> neighbors = original;
  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
      replace, true, false, true, true, 2, neighbors, never_prune);
  assert(result.stale && neighbors == original);

  neighbors = original;
  result = memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
    replace, true, true, false, true, 2, neighbors, never_prune);
  assert(result.stale && neighbors == original);

  neighbors = original;
  result = memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
    replace, false, true, true, true, 2, neighbors, never_prune);
  assert(result.stale && neighbors == original);

  const ReconcileReverseOp zero_sequence = operation(
    ReconcileReverseOpKind::replace_or_add,
    target, old_candidate, new_candidate, 0);
  neighbors = original;
  result = memory_node_storage_owner_index_detail::reconcile_reverse_neighbors(
    zero_sequence, true, true, true, true, 2, neighbors, never_prune);
  assert(result.stale && neighbors == original);
}

void test_stage1_provisional_reconciles_to_final_stable_with_free_slot() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr stable_neighbor = pointer(0, 0x2000);
  const RemotePtr old_provisional = pointer(0, 0x3000);
  const RemotePtr unrelated_first = pointer(0, 0x4000);
  const RemotePtr unrelated_second = pointer(0, 0x5000);
  const RemotePtr final_candidate = pointer(1, 0x6000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::replace_or_add,
    target, old_provisional, final_candidate);
  vec<RemotePtr> stable{stable_neighbor};
  vec<RemotePtr> provisional{
    unrelated_first, old_provisional, unrelated_second};
  u32 prune_calls = 0;
  const auto never_prune = [&](const vec<RemotePtr>&) -> vec<RemotePtr> {
    ++prune_calls;
    assert(false);
    return {};
  };

  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, true, true, true, 2, 3, stable, provisional, never_prune);
  assert(result.accepted && result.replaced && result.removed && !result.stale);
  assert(result.placement_sequence == op.placement_sequence);
  assert((stable == vec<RemotePtr>{stable_neighbor, final_candidate}));
  assert((provisional ==
          vec<RemotePtr>{unrelated_first, unrelated_second}));
  assert(prune_calls == 0);

  // Retrying after old_provisional disappeared must not duplicate the final
  // edge, invoke prune, or disturb unrelated Stage1 backlinks.
  result = memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
    op, true, false, true, false, 2, 3, stable, provisional, never_prune);
  assert(result.accepted && !result.replaced && result.removed && !result.stale);
  assert((stable == vec<RemotePtr>{stable_neighbor, final_candidate}));
  assert((provisional ==
          vec<RemotePtr>{unrelated_first, unrelated_second}));
  assert(prune_calls == 0);
}

void test_stage1_provisional_reconciles_through_prune_when_stable_is_full() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr stable_first = pointer(0, 0x2000);
  const RemotePtr stable_second = pointer(0, 0x3000);
  const RemotePtr old_provisional = pointer(0, 0x4000);
  const RemotePtr unrelated_provisional = pointer(0, 0x5000);
  const RemotePtr final_candidate = pointer(1, 0x6000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::replace_or_add,
    target, old_provisional, final_candidate);
  vec<RemotePtr> stable{stable_first, stable_second};
  vec<RemotePtr> provisional{unrelated_provisional, old_provisional};
  u32 prune_calls = 0;
  const auto prune = [&](const vec<RemotePtr>& candidates) {
    ++prune_calls;
    assert((candidates == vec<RemotePtr>{
                            stable_first, stable_second, final_candidate}));
    assert((provisional == vec<RemotePtr>{unrelated_provisional}));
    return vec<RemotePtr>{final_candidate, stable_second};
  };

  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, true, true, true, 2, 3, stable, provisional, prune);
  assert(result.accepted && result.replaced && result.removed && !result.stale);
  assert((stable == vec<RemotePtr>{final_candidate, stable_second}));
  assert((provisional == vec<RemotePtr>{unrelated_provisional}));
  assert(prune_calls == 1);

  // An already-stable final pointer is the idempotent postcondition.  In
  // particular, a retry must not prune the stable set a second time.
  result = memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
    op, true, false, true, false, 2, 3, stable, provisional, prune);
  assert(result.accepted && !result.replaced && result.removed && !result.stale);
  assert((stable == vec<RemotePtr>{final_candidate, stable_second}));
  assert((provisional == vec<RemotePtr>{unrelated_provisional}));
  assert(prune_calls == 1);
}

void test_final_no_longer_adjacent_removes_only_the_stage1_backlink() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr stable_neighbor = pointer(0, 0x2000);
  const RemotePtr old_provisional = pointer(0, 0x3000);
  const RemotePtr unrelated_first = pointer(0, 0x4000);
  const RemotePtr unrelated_second = pointer(0, 0x5000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::remove_if_present,
    target, old_provisional, RemotePtr{});
  vec<RemotePtr> stable{stable_neighbor};
  vec<RemotePtr> provisional{
    unrelated_first, old_provisional, unrelated_second};
  const auto never_prune = [](const vec<RemotePtr>&) -> vec<RemotePtr> {
    assert(false);
    return {};
  };

  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, true, false, false, 2, 3, stable, provisional, never_prune);
  assert(result.removed && !result.accepted && !result.replaced && !result.stale);
  assert((stable == vec<RemotePtr>{stable_neighbor}));
  assert((provisional ==
          vec<RemotePtr>{unrelated_first, unrelated_second}));

  // Absence is also a successful remove postcondition, even if the old
  // physical address can no longer be identity-validated on a retry.
  result = memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
    op, true, false, false, false, 2, 3, stable, provisional, never_prune);
  assert(result.removed && !result.stale);
  assert((stable == vec<RemotePtr>{stable_neighbor}));
  assert((provisional ==
          vec<RemotePtr>{unrelated_first, unrelated_second}));
}

void test_equivalent_stable_old_is_replaced_in_place() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr stable_left = pointer(0, 0x2000);
  const RemotePtr stable_old = pointer(0, 0x3000);
  const RemotePtr stable_right = pointer(0, 0x4000);
  const RemotePtr unrelated_first = pointer(0, 0x5000);
  const RemotePtr unrelated_second = pointer(0, 0x6000);
  const RemotePtr final_candidate = pointer(1, 0x7000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::replace_or_add,
    target, stable_old, final_candidate);
  vec<RemotePtr> stable{stable_left, stable_old, stable_right};
  vec<RemotePtr> provisional{unrelated_first, unrelated_second};
  u32 prune_calls = 0;
  const auto never_prune = [&](const vec<RemotePtr>&) -> vec<RemotePtr> {
    ++prune_calls;
    assert(false);
    return {};
  };

  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, true, true, true, 3, 3, stable, provisional, never_prune);
  assert(result.accepted && result.replaced && result.removed && !result.stale);
  assert((stable ==
          vec<RemotePtr>{stable_left, final_candidate, stable_right}));
  assert((provisional ==
          vec<RemotePtr>{unrelated_first, unrelated_second}));
  assert(prune_calls == 0);

  result = memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
    op, true, false, true, false, 3, 3, stable, provisional, never_prune);
  assert(result.accepted && !result.replaced && result.removed && !result.stale);
  assert((stable ==
          vec<RemotePtr>{stable_left, final_candidate, stable_right}));
  assert((provisional ==
          vec<RemotePtr>{unrelated_first, unrelated_second}));
  assert(prune_calls == 0);
}

void test_ensure_reachable_reserves_an_empty_protected_slot() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr highest_priority = pointer(0, 0x2000);
  const RemotePtr lowest_priority = pointer(0, 0x3000);
  const RemotePtr new_candidate = pointer(1, 0x4000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::ensure_reachable,
    target, RemotePtr{}, new_candidate);
  vec<RemotePtr> stable{highest_priority, lowest_priority};
  vec<RemotePtr> provisional;
  u32 prune_calls = 0;
  const auto never_prune = [&](const vec<RemotePtr>& candidates) {
    ++prune_calls;
    (void)candidates;
    assert(false);
    return vec<RemotePtr>{};
  };

  const auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, true, true, false, 2, 2, stable, provisional, never_prune);
  assert(result.accepted && !result.replaced && !result.removed &&
         !result.stale);
  assert((stable == vec<RemotePtr>{highest_priority, lowest_priority}));
  assert((provisional == vec<RemotePtr>{new_candidate}));
  assert(stable.size() == 2);
  assert(std::count(stable.begin(), stable.end(), new_candidate) == 0);
  assert(prune_calls == 0);
}

void test_ensure_reachable_never_steals_a_protected_slot() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr protected_first = pointer(0, 0x2000);
  const RemotePtr protected_second = pointer(0, 0x3000);
  const RemotePtr new_candidate = pointer(1, 0x4000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::ensure_reachable,
    target, RemotePtr{}, new_candidate);
  vec<RemotePtr> stable;
  vec<RemotePtr> provisional{protected_first, protected_second};
  const auto never_prune = [](const vec<RemotePtr>&) -> vec<RemotePtr> {
    assert(false);
    return {};
  };

  const auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, true, true, false, 2, 2, stable, provisional, never_prune);
  assert(!result.accepted && !result.replaced && !result.removed &&
         !result.stale);
  assert(stable.empty());
  assert((provisional ==
          vec<RemotePtr>{protected_first, protected_second}));
}

void test_ensure_reachable_moves_the_same_stable_edge_without_eviction() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr survivor = pointer(0, 0x2000);
  const RemotePtr child = pointer(1, 0x3000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::ensure_reachable,
    target, RemotePtr{}, child);
  vec<RemotePtr> stable{survivor, child};
  vec<RemotePtr> provisional;
  const auto never_prune = [](const vec<RemotePtr>&) -> vec<RemotePtr> {
    assert(false);
    return {};
  };

  const auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, true, true, false, 2, 2, stable, provisional, never_prune);
  assert(result.accepted && !result.replaced && !result.removed &&
         !result.stale);
  assert((stable == vec<RemotePtr>{survivor}));
  assert((provisional == vec<RemotePtr>{child}));
}

void test_ensure_reachable_promotes_in_place_stage1_provisional() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr stable_neighbor = pointer(0, 0x2000);
  const RemotePtr in_place_candidate = pointer(0, 0x3000);
  const RemotePtr unrelated_provisional = pointer(0, 0x4000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::ensure_reachable,
    target, in_place_candidate, in_place_candidate);
  vec<RemotePtr> stable{stable_neighbor};
  vec<RemotePtr> provisional{unrelated_provisional, in_place_candidate};
  u32 prune_calls = 0;
  const auto never_prune = [&](const vec<RemotePtr>&) -> vec<RemotePtr> {
    ++prune_calls;
    assert(false);
    return {};
  };

  const auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, true, true, true, 2, 3, stable, provisional, never_prune);
  assert(result.accepted && !result.replaced && !result.removed && !result.stale);
  assert((stable == vec<RemotePtr>{stable_neighbor}));
  assert((provisional ==
          vec<RemotePtr>{unrelated_provisional, in_place_candidate}));
  assert(std::count(stable.begin(), stable.end(), in_place_candidate) == 0);
  assert(std::count(provisional.begin(), provisional.end(),
                    in_place_candidate) == 1);
  assert(prune_calls == 0);
}

void test_ensure_reachable_migration_removes_old_when_prune_rejects_new() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr highest_priority = pointer(0, 0x2000);
  const RemotePtr lowest_priority = pointer(0, 0x3000);
  const RemotePtr old_provisional = pointer(0, 0x4000);
  const RemotePtr unrelated_provisional = pointer(0, 0x5000);
  const RemotePtr migrated_candidate = pointer(1, 0x6000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::ensure_reachable,
    target, old_provisional, migrated_candidate);
  vec<RemotePtr> stable{highest_priority, lowest_priority};
  vec<RemotePtr> provisional{unrelated_provisional, old_provisional};
  u32 prune_calls = 0;
  const auto never_prune = [&](const vec<RemotePtr>&) -> vec<RemotePtr> {
    ++prune_calls;
    assert(false);
    return {};
  };

  const auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, true, true, false, 2, 3, stable, provisional, never_prune);
  assert(result.accepted && result.replaced && result.removed && !result.stale);
  assert((stable == vec<RemotePtr>{highest_priority, lowest_priority}));
  assert((provisional ==
          vec<RemotePtr>{unrelated_provisional, migrated_candidate}));
  assert(std::find(stable.begin(), stable.end(), old_provisional) ==
         stable.end());
  assert(std::find(provisional.begin(), provisional.end(), old_provisional) ==
         provisional.end());
  assert(std::count(provisional.begin(), provisional.end(),
                    migrated_candidate) == 1);
  assert(prune_calls == 0);
}

void test_ensure_reachable_retry_is_idempotent_and_degree_bounded() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr survivor = pointer(0, 0x2000);
  const RemotePtr old_provisional = pointer(0, 0x3000);
  const RemotePtr unrelated_provisional = pointer(0, 0x4000);
  const RemotePtr final_candidate = pointer(1, 0x5000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::ensure_reachable,
    target, old_provisional, final_candidate);
  const vec<RemotePtr> committed_stable{survivor};
  vec<RemotePtr> stable = committed_stable;
  const vec<RemotePtr> committed_provisional{
    unrelated_provisional, final_candidate};
  vec<RemotePtr> provisional = committed_provisional;
  u32 prune_calls = 0;
  const auto never_prune = [&](const vec<RemotePtr>&) -> vec<RemotePtr> {
    ++prune_calls;
    assert(false);
    return {};
  };

  for (u32 retry = 0; retry != 3; ++retry) {
    const auto result =
      memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
        op, true, false, true, false, 2, 3, stable, provisional, never_prune);
    assert(result.accepted && !result.replaced && result.removed &&
           !result.stale);
    assert(stable == committed_stable);
    assert(provisional == committed_provisional);
    assert(stable.size() <= 2);
    assert(std::count(provisional.begin(), provisional.end(),
                      final_candidate) == 1);
  }
  assert(prune_calls == 0);
}

void test_ensure_reachable_stale_generation_or_identity_is_non_mutating() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr stable_neighbor = pointer(0, 0x2000);
  const RemotePtr old_provisional = pointer(0, 0x3000);
  const RemotePtr new_candidate = pointer(1, 0x4000);
  const ReconcileReverseOp op = operation(
    ReconcileReverseOpKind::ensure_reachable,
    target, old_provisional, new_candidate);
  const vec<RemotePtr> original_stable{stable_neighbor};
  const vec<RemotePtr> original_provisional{old_provisional};
  u32 prune_calls = 0;
  const auto never_prune = [&](const vec<RemotePtr>&) -> vec<RemotePtr> {
    ++prune_calls;
    assert(false);
    return {};
  };

  // A live pointer carrying a different id/generation is represented by a
  // failed old identity check. It must not remove the observed Stage1 edge.
  vec<RemotePtr> stable = original_stable;
  vec<RemotePtr> provisional = original_provisional;
  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, false, true, false, 2, 3, stable, provisional, never_prune);
  assert(result.stale && !result.accepted && !result.replaced &&
         !result.removed);
  assert(stable == original_stable);
  assert(provisional == original_provisional);

  // A new physical pointer that is deleted or carries a stale generation is
  // rejected before either adjacency plane is changed.
  stable = original_stable;
  provisional = original_provisional;
  result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, true, true, false, false, 2, 3, stable, provisional, never_prune);
  assert(result.stale && !result.accepted && !result.replaced &&
         !result.removed);
  assert(stable == original_stable);
  assert(provisional == original_provisional);

  // A stale/deleted target likewise fences the complete operation.
  stable = original_stable;
  provisional = original_provisional;
  result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      op, false, true, true, false, 2, 3, stable, provisional, never_prune);
  assert(result.stale && !result.accepted && !result.replaced &&
         !result.removed);
  assert(stable == original_stable);
  assert(provisional == original_provisional);
  assert(prune_calls == 0);
}

void test_stage2_promotion_forces_one_bounded_stable_bridge() {
  const RemotePtr parent = pointer(0, 0x1000);
  const RemotePtr high_priority = pointer(0, 0x2000);
  const RemotePtr low_priority = pointer(0, 0x3000);
  const RemotePtr unrelated_protected = pointer(0, 0x4000);
  const RemotePtr final_candidate = pointer(1, 0x5000);
  const ReconcileReverseOp promote = operation(
    ReconcileReverseOpKind::promote_stable_bridge,
    parent, final_candidate, final_candidate);
  vec<RemotePtr> stable{high_priority, low_priority};
  vec<RemotePtr> provisional{unrelated_protected, final_candidate};
  u32 prune_calls = 0;
  const auto reject_new = [&](const vec<RemotePtr>& candidates) {
    ++prune_calls;
    assert((candidates == vec<RemotePtr>{
                            high_priority, low_priority, final_candidate}));
    // Model the worst case: every ordinary final add would lose pruning.
    return vec<RemotePtr>{high_priority, low_priority};
  };

  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      promote, true, true, true, true, 2, 3,
      stable, provisional, reject_new);
  assert(result.accepted && result.removed && !result.stale);
  assert((stable == vec<RemotePtr>{high_priority, final_candidate}));
  assert((provisional == vec<RemotePtr>{unrelated_protected}));
  assert(stable.size() == 2);
  assert(prune_calls == 1);

  // Retry observes the stable certificate, does not prune again, and keeps
  // unrelated insertions' protected state intact.
  result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      promote, true, true, true, false, 2, 3,
      stable, provisional, reject_new);
  assert(result.accepted && result.removed && !result.stale);
  assert((stable == vec<RemotePtr>{high_priority, final_candidate}));
  assert((provisional == vec<RemotePtr>{unrelated_protected}));
  assert(prune_calls == 1);
}

void test_ordinary_rejection_cannot_substitute_for_promotion_ack() {
  const RemotePtr parent = pointer(0, 0x1000);
  const RemotePtr stable_neighbor = pointer(0, 0x2000);
  const RemotePtr final_candidate = pointer(1, 0x3000);
  const ReconcileReverseOp ordinary = operation(
    ReconcileReverseOpKind::add,
    parent, RemotePtr{}, final_candidate);
  vec<RemotePtr> stable{stable_neighbor};
  vec<RemotePtr> provisional;
  const auto reject_new = [&](const vec<RemotePtr>& candidates) {
    assert((candidates == vec<RemotePtr>{stable_neighbor, final_candidate}));
    return vec<RemotePtr>{stable_neighbor};
  };
  const auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      ordinary, true, true, true, false, 1, 2,
      stable, provisional, reject_new);
  assert(!result.accepted && !result.stale);
  assert((stable == vec<RemotePtr>{stable_neighbor}));

  // Ordinary bounded rejection is a valid proposal result, but it does not
  // satisfy the mandatory promotion postcondition that gates bridge cleanup.
  assert(memory_node_storage_owner_index_detail::
           reconcile_reverse_postcondition_holds(ordinary, result));
  const ReconcileReverseOp promotion = operation(
    ReconcileReverseOpKind::promote_stable_bridge,
    parent, RemotePtr{}, final_candidate);
  assert(!memory_node_storage_owner_index_detail::
            reconcile_reverse_postcondition_holds(promotion, result));
}

}  // namespace

int main() {
  test_protocol_layout_is_additive();
  test_bounded_prune_rejection_is_a_terminal_ordinary_postcondition();
  test_stage2_backlink_plan_cleans_selected_and_unselected_bridges();
  test_final_parent_eligibility_rejects_churn_and_dependency_cycles();
  test_stage2_freeze_closes_the_ack_to_publish_window();
  test_stage2_freeze_lifecycle_has_one_mutation_boundary();
  test_stale_stage2_retires_only_an_authority_detached_source();
  test_equivalent_replace_is_in_place_and_idempotent();
  test_remove_if_present_has_an_idempotent_postcondition();
  test_add_uses_robust_prune_only_at_the_degree_bound();
  test_replace_conflict_reenters_robust_prune();
  test_identity_or_liveness_mismatch_is_stale_and_non_mutating();
  test_stage1_provisional_reconciles_to_final_stable_with_free_slot();
  test_stage1_provisional_reconciles_through_prune_when_stable_is_full();
  test_final_no_longer_adjacent_removes_only_the_stage1_backlink();
  test_equivalent_stable_old_is_replaced_in_place();
  test_ensure_reachable_reserves_an_empty_protected_slot();
  test_ensure_reachable_never_steals_a_protected_slot();
  test_ensure_reachable_moves_the_same_stable_edge_without_eviction();
  test_ensure_reachable_promotes_in_place_stage1_provisional();
  test_ensure_reachable_migration_removes_old_when_prune_rejects_new();
  test_ensure_reachable_retry_is_idempotent_and_degree_bounded();
  test_ensure_reachable_stale_generation_or_identity_is_non_mutating();
  test_stage2_promotion_forces_one_bounded_stable_bridge();
  test_ordinary_rejection_cannot_substitute_for_promotion_ack();
  return 0;
}
