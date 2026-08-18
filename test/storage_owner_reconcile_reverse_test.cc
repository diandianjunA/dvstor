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

void test_retired_target_completes_only_non_reachability_work() {
  using memory_node_storage_owner_index_detail::
    reconcile_retired_target_result;
  using memory_node_storage_owner_index_detail::
    reconcile_reverse_postcondition_holds;
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr old_candidate = pointer(0, 0x2000);
  const RemotePtr new_candidate = pointer(1, 0x3000);

  for (const ReconcileReverseOpKind kind : {
         ReconcileReverseOpKind::add,
         ReconcileReverseOpKind::remove_if_present,
         ReconcileReverseOpKind::replace_or_add}) {
    const ReconcileReverseOp op = operation(
      kind, target,
      kind == ReconcileReverseOpKind::add ? RemotePtr{} : old_candidate,
      kind == ReconcileReverseOpKind::remove_if_present
        ? RemotePtr{} : new_candidate);
    const auto result = reconcile_retired_target_result(op);
    assert(!result.stale);
    assert(reconcile_reverse_postcondition_holds(op, result));
  }

  for (const ReconcileReverseOpKind kind : {
         ReconcileReverseOpKind::ensure_reachable,
         ReconcileReverseOpKind::promote_stable_bridge}) {
    const ReconcileReverseOp op = operation(
      kind, target, old_candidate, new_candidate);
    const auto result = reconcile_retired_target_result(op);
    assert(result.stale);
    assert(!reconcile_reverse_postcondition_holds(op, result));
  }
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

void test_stage2_snapshot_wave_deduplicates_without_losing_task_order() {
  using memory_node_storage_owner_maintenance_detail::
    Stage2SnapshotWavePlan;
  const RemotePtr a = pointer(0, 0x1000);
  const RemotePtr b = pointer(1, 0x2000);
  const RemotePtr c = pointer(2, 0x3000);
  const RemotePtr d = pointer(3, 0x4000);
  const vec<vec<RemotePtr>> candidates{
    {a, b, a, RemotePtr{}},
    {b, c},
    {d, a},
  };
  Stage2SnapshotWavePlan plan;
  plan.build(span<const vec<RemotePtr>>{candidates});

  // Physical reads follow first appearance and are shared by all tasks.
  assert((plan.targets == vec<RemotePtr>{a, b, c, d}));
  // Per-task indices retain exact logical order and repeated candidates. A
  // missing snapshot can therefore be omitted during scatter without moving
  // any other candidate across task boundaries.
  assert((plan.task_target_indices[0] == vec<u32>{0, 1, 0}));
  assert((plan.task_target_indices[1] == vec<u32>{1, 2}));
  assert((plan.task_target_indices[2] == vec<u32>{3, 0}));

  const vec<vec<RemotePtr>> retry{{c, a}};
  plan.build(span<const vec<RemotePtr>>{retry});
  assert((plan.targets == vec<RemotePtr>{c, a}));
  assert(plan.task_count == 1);
  assert((plan.task_target_indices[0] == vec<u32>{0, 1}));
  assert(plan.task_target_indices[1].empty());
}

void test_promotion_retry_skips_dead_first_sealed_neighbor() {
  using memory_node_storage_owner_maintenance_detail::
    plan_stage2_backlink_reconciliation;
  const RemotePtr dead_first = pointer(0, 0x1000);
  const RemotePtr live_second = pointer(1, 0x2000);

  // The outgoing adjacency is sealed and is not rewritten merely because its
  // first edge retired. Promotion planning consumes the independently
  // revalidated view, so a retry advances to the next durable parent.
  const vec<RemotePtr> live_planner_view{live_second};
  const auto plan = plan_stage2_backlink_reconciliation(
    span<const RemotePtr>{}, span<const RemotePtr>{live_planner_view});
  assert(plan.promotion_target == live_second);
  assert(plan.promotion_target != dead_first);

  // A lost ACK is not trusted blindly: after its certificate fails parent
  // validation the retry clears it and makes the same live fallback choice.
  const auto after_dead_certificate = plan_stage2_backlink_reconciliation(
    span<const RemotePtr>{}, span<const RemotePtr>{live_planner_view},
    RemotePtr{});
  assert(after_dead_certificate.promotion_target == live_second);
}

void test_stage2_revalidation_scans_only_reachability_holders() {
  using memory_node_storage_owner_maintenance_detail::
    stage2_revalidation_parents;
  const RemotePtr bridge_a = pointer(0, 0x1000);
  const RemotePtr bridge_b = pointer(1, 0x2000);
  const RemotePtr acknowledged = pointer(2, 0x3000);
  const RemotePtr planned = pointer(3, 0x4000);
  const vec<RemotePtr> bridges{bridge_b, bridge_a, bridge_a, RemotePtr{}};
  const vec<RemotePtr> parents = stage2_revalidation_parents(
    span<const RemotePtr>{bridges}, acknowledged, planned);
  assert(parents ==
         (vec<RemotePtr>{bridge_a, bridge_b, acknowledged, planned}));

  // Hundreds of ordinary final neighbors are intentionally absent: they
  // cannot own either the provisional bridge or the promoted certificate.
  const RemotePtr unrelated = pointer(4, 0x5000);
  assert(std::find(parents.begin(), parents.end(), unrelated) ==
         parents.end());
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

void test_add_batch_prunes_the_hot_target_union_once() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr stable_first = pointer(0, 0x2000);
  const RemotePtr stable_second = pointer(0, 0x3000);
  const RemotePtr rejected = pointer(1, 0x4000);
  const RemotePtr accepted_first = pointer(1, 0x5000);
  const RemotePtr accepted_second = pointer(1, 0x6000);
  const RemotePtr unrelated_provisional = pointer(1, 0x7000);
  const RemotePtr invalid_prune_output = pointer(1, 0x8000);
  const vec<ReconcileReverseOp> ops{
    operation(ReconcileReverseOpKind::add, target, RemotePtr{}, rejected,
              11),
    operation(ReconcileReverseOpKind::add, target, RemotePtr{},
              accepted_first, 12),
    operation(ReconcileReverseOpKind::add, target, RemotePtr{},
              accepted_second, 13),
  };
  const vec<u8> identity_live{1, 1, 1};
  vec<RemotePtr> stable{stable_first, stable_second};
  vec<RemotePtr> provisional{unrelated_provisional, rejected};
  u32 prune_calls = 0;
  const auto prune = [&](const vec<RemotePtr>& candidates) {
    ++prune_calls;
    assert((candidates == vec<RemotePtr>{
                            stable_first, stable_second, rejected,
                            accepted_first, accepted_second}));
    // The batch policy retains the existing defensive filtering contract:
    // candidates outside the input and duplicate outputs cannot enter the
    // bounded graph.
    return vec<RemotePtr>{accepted_first, invalid_prune_output,
                          accepted_first, stable_first, accepted_second};
  };

  vec<service::storage_owner::ReconcileReverseResult> results;
  memory_node_storage_owner_index_detail::reconcile_reverse_add_batch(
    span<const ReconcileReverseOp>{ops}, span<const u8>{identity_live}, true,
    3, 2, stable, provisional, results, prune);

  assert(prune_calls == 1);
  assert((stable == vec<RemotePtr>{
                    accepted_first, stable_first, accepted_second}));
  assert((provisional == vec<RemotePtr>{unrelated_provisional}));
  assert(results.size() == ops.size());
  assert(!results[0].accepted && !results[0].stale &&
         results[0].placement_sequence == 11);
  assert(results[1].accepted && !results[1].stale &&
         results[1].placement_sequence == 12);
  assert(results[2].accepted && !results[2].stale &&
         results[2].placement_sequence == 13);
  for (size_t index = 0; index < ops.size(); ++index) {
    assert(memory_node_storage_owner_index_detail::
             reconcile_reverse_postcondition_holds(ops[index],
                                                    results[index]));
  }
}

void test_add_batch_keeps_per_operation_identity_and_sequence_fences() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr live = pointer(1, 0x2000);
  const RemotePtr wrong_identity = pointer(1, 0x3000);
  const RemotePtr zero_sequence = pointer(1, 0x4000);
  const RemotePtr wrong_target_candidate = pointer(1, 0x5000);
  const vec<ReconcileReverseOp> ops{
    operation(ReconcileReverseOpKind::add, target, RemotePtr{}, live, 21),
    operation(ReconcileReverseOpKind::add, target, RemotePtr{},
              wrong_identity, 22),
    operation(ReconcileReverseOpKind::add, target, RemotePtr{},
              zero_sequence, 0),
    operation(ReconcileReverseOpKind::add, pointer(0, 0x9000), RemotePtr{},
              wrong_target_candidate, 24),
  };
  const vec<u8> identity_live{1, 0, 1, 1};
  vec<RemotePtr> stable;
  vec<RemotePtr> provisional{
    live, wrong_identity, zero_sequence, wrong_target_candidate};
  const auto never_prune = [](const vec<RemotePtr>&) -> vec<RemotePtr> {
    assert(false);
    return {};
  };

  vec<service::storage_owner::ReconcileReverseResult> results;
  memory_node_storage_owner_index_detail::reconcile_reverse_add_batch(
    span<const ReconcileReverseOp>{ops}, span<const u8>{identity_live}, true,
    4, 2, stable, provisional, results, never_prune);

  assert((stable == vec<RemotePtr>{live}));
  assert((provisional == vec<RemotePtr>{
                         wrong_identity, zero_sequence,
                         wrong_target_candidate}));
  assert(results[0].accepted && !results[0].stale &&
         results[0].placement_sequence == 21);
  assert(results[1].stale && results[1].placement_sequence == 22);
  assert(results[2].stale && results[2].placement_sequence == 0);
  assert(results[3].stale && results[3].placement_sequence == 24);
}

void test_promotion_is_a_pruning_boundary_for_same_target_adds() {
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr stable_first = pointer(0, 0x2000);
  const RemotePtr stable_second = pointer(0, 0x3000);
  const RemotePtr promoted = pointer(1, 0x4000);
  const RemotePtr ordinary_first = pointer(1, 0x5000);
  const RemotePtr ordinary_second = pointer(1, 0x6000);
  const vec<ReconcileReverseOp> ops{
    operation(ReconcileReverseOpKind::promote_stable_bridge, target,
              promoted, promoted, 31),
    operation(ReconcileReverseOpKind::add, target, RemotePtr{},
              ordinary_first, 32),
    operation(ReconcileReverseOpKind::add, target, RemotePtr{},
              ordinary_second, 33),
  };
  const vec<size_t> op_indices{0, 1, 2};
  assert(memory_node_storage_owner_index_detail::
           reconcile_reverse_add_run_end(
             span<const ReconcileReverseOp>{ops},
             span<const size_t>{op_indices}, 0) == 0);
  assert(memory_node_storage_owner_index_detail::
           reconcile_reverse_add_run_end(
             span<const ReconcileReverseOp>{ops},
             span<const size_t>{op_indices}, 1) == 3);

  vec<RemotePtr> stable{stable_first, stable_second};
  vec<RemotePtr> provisional{promoted};
  u32 prune_calls = 0;
  const auto prune = [&](const vec<RemotePtr>& candidates) {
    ++prune_calls;
    if (prune_calls == 1) {
      // Promotion is scored alone. Ordinary proposals are not allowed to
      // enter the mandatory-certificate invocation.
      assert((candidates == vec<RemotePtr>{
                              stable_first, stable_second, promoted}));
      return vec<RemotePtr>{stable_first, stable_second};
    }
    assert(prune_calls == 2);
    // Only the following compatible add run is union-pruned.
    assert((candidates == vec<RemotePtr>{
                            stable_first, promoted,
                            ordinary_first, ordinary_second}));
    return vec<RemotePtr>{stable_first, ordinary_first};
  };

  const auto promotion_result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      ops[0], true, true, true, true, 2, 2,
      stable, provisional, prune);
  assert(promotion_result.accepted && promotion_result.removed &&
         !promotion_result.stale);
  assert((stable == vec<RemotePtr>{stable_first, promoted}));
  assert(provisional.empty());

  const vec<ReconcileReverseOp> add_ops{ops[1], ops[2]};
  const vec<u8> identities{1, 1};
  vec<service::storage_owner::ReconcileReverseResult> add_results;
  memory_node_storage_owner_index_detail::reconcile_reverse_add_batch(
    span<const ReconcileReverseOp>{add_ops}, span<const u8>{identities},
    true, 2, 2, stable, provisional, add_results, prune);
  assert(prune_calls == 2);
  assert(add_results.size() == 2);
  assert(add_results[0].accepted && !add_results[0].stale);
  assert(!add_results[1].accepted && !add_results[1].stale);
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

void test_stable_then_promotion_closes_hot_parent_eviction_window() {
  const RemotePtr parent = pointer(0, 0x1000);
  const RemotePtr mandatory_child = pointer(1, 0x2000);
  const RemotePtr ordinary_child = pointer(2, 0x3000);
  vec<RemotePtr> stable;
  vec<RemotePtr> provisional{mandatory_child};

  const ReconcileReverseOp ordinary = operation(
    ReconcileReverseOpKind::add,
    parent, RemotePtr{}, ordinary_child, 71);
  const auto prefer_ordinary = [&](const vec<RemotePtr>& candidates) {
    assert(std::find(candidates.begin(), candidates.end(), ordinary_child) !=
           candidates.end());
    return vec<RemotePtr>{ordinary_child};
  };
  auto result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      ordinary, true, false, true, false, 1, 2,
      stable, provisional, prefer_ordinary);
  assert(result.accepted && !result.stale);
  assert((stable == vec<RemotePtr>{ordinary_child}));
  assert((provisional == vec<RemotePtr>{mandatory_child}));

  // The mandatory promotion is deliberately the final stable-plane barrier.
  // Even when RobustPrune prefers the ordinary edge, promotion spends the one
  // bounded reachability exception on the protected child before removal.
  const ReconcileReverseOp promotion = operation(
    ReconcileReverseOpKind::promote_stable_bridge,
    parent, mandatory_child, mandatory_child, 72);
  result =
    memory_node_storage_owner_index_detail::reconcile_reverse_adjacency(
      promotion, true, true, true, true, 1, 2,
      stable, provisional, prefer_ordinary);
  assert(result.accepted && result.removed && !result.stale);
  assert((stable == vec<RemotePtr>{mandatory_child}));
  assert(provisional.empty());
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

void test_final_target_audit_rejects_evicted_mandatory_certificate() {
  using memory_node_storage_owner_index_detail::
    reconcile_reverse_final_reachability_holds;
  const RemotePtr target = pointer(0, 0x1000);
  const RemotePtr first = pointer(1, 0x2000);
  const RemotePtr second = pointer(1, 0x3000);
  const vec<ReconcileReverseOp> ops{
    operation(ReconcileReverseOpKind::promote_stable_bridge,
              target, first, first, 41),
    operation(ReconcileReverseOpKind::promote_stable_bridge,
              target, second, second, 42),
  };
  vec<service::storage_owner::ReconcileReverseResult> results(2);
  results[0].placement_sequence = 41;
  results[0].accepted = 1;
  results[0].removed = 1;
  results[1].placement_sequence = 42;
  results[1].accepted = 1;
  results[1].removed = 1;
  const vec<size_t> indexes{0, 1};
  const vec<RemotePtr> only_second{second};
  const vec<RemotePtr> both{first, second};
  const vec<RemotePtr> empty;

  assert(!reconcile_reverse_final_reachability_holds(
    span<const ReconcileReverseOp>{ops}, span<const size_t>{indexes},
    span<const service::storage_owner::ReconcileReverseResult>{results},
    span<const RemotePtr>{only_second}, span<const RemotePtr>{empty}));
  assert(reconcile_reverse_final_reachability_holds(
    span<const ReconcileReverseOp>{ops}, span<const size_t>{indexes},
    span<const service::storage_owner::ReconcileReverseResult>{results},
    span<const RemotePtr>{both}, span<const RemotePtr>{empty}));

  vec<ReconcileReverseOp> ensure_ops{
    operation(ReconcileReverseOpKind::ensure_reachable,
              target, RemotePtr{}, first, 43),
  };
  vec<service::storage_owner::ReconcileReverseResult> ensure_results(1);
  ensure_results[0].placement_sequence = 43;
  ensure_results[0].accepted = 1;
  const vec<size_t> ensure_index{0};
  const vec<RemotePtr> protected_first{first};
  assert(reconcile_reverse_final_reachability_holds(
    span<const ReconcileReverseOp>{ensure_ops},
    span<const size_t>{ensure_index},
    span<const service::storage_owner::ReconcileReverseResult>{
      ensure_results},
    span<const RemotePtr>{empty}, span<const RemotePtr>{protected_first}));
}

}  // namespace

int main() {
  test_protocol_layout_is_additive();
  test_bounded_prune_rejection_is_a_terminal_ordinary_postcondition();
  test_retired_target_completes_only_non_reachability_work();
  test_stage2_backlink_plan_cleans_selected_and_unselected_bridges();
  test_final_parent_eligibility_rejects_churn_and_dependency_cycles();
  test_stage2_snapshot_wave_deduplicates_without_losing_task_order();
  test_promotion_retry_skips_dead_first_sealed_neighbor();
  test_stage2_revalidation_scans_only_reachability_holders();
  test_stage2_freeze_closes_the_ack_to_publish_window();
  test_stage2_freeze_lifecycle_has_one_mutation_boundary();
  test_stale_stage2_retires_only_an_authority_detached_source();
  test_equivalent_replace_is_in_place_and_idempotent();
  test_remove_if_present_has_an_idempotent_postcondition();
  test_add_uses_robust_prune_only_at_the_degree_bound();
  test_add_batch_prunes_the_hot_target_union_once();
  test_add_batch_keeps_per_operation_identity_and_sequence_fences();
  test_promotion_is_a_pruning_boundary_for_same_target_adds();
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
  test_stable_then_promotion_closes_hot_parent_eviction_window();
  test_ordinary_rejection_cannot_substitute_for_promotion_ack();
  test_final_target_audit_rejects_evicted_mandatory_certificate();
  return 0;
}
