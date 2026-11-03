#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "common/types.hh"
#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"
#include "remote_pointer.hh"
#include "service/storage_owner_protocol.hh"

namespace memory_node_storage_owner_index_detail {

struct AtomicCutPlan {
  vec<RemotePtr> stable;
  vec<RemotePtr> provisional;
  vec<service::storage_owner::ReconcileReverseResult> results;
  vec<std::pair<RemotePtr, u64>> acquired;
  vec<std::pair<RemotePtr, u64>> released;
  bool valid{};
};

// Builds one target-local graph transition. Identity/liveness checks and the
// actual lease-table update are deliberately outside this pure planner: the
// storage owner commits the returned acquire/release sets only after publishing
// the exact adjacency in this plan.
template <class RobustPrune>
AtomicCutPlan plan_atomic_cut(
    span<const service::storage_owner::ReconcileReverseOp> ops,
    span<const RemotePtr> current_stable,
    span<const RemotePtr> current_provisional,
    span<const std::pair<RemotePtr, u64>> active_leases,
    u32 degree_limit, RobustPrune&& robust_prune) {
  using service::storage_owner::ReconcileReverseOpKind;
  using service::storage_owner::ReconcileReverseResult;
  AtomicCutPlan plan;
  plan.stable.assign(current_stable.begin(), current_stable.end());
  plan.provisional.assign(current_provisional.begin(),
                          current_provisional.end());
  plan.results.resize(ops.size());

  const auto contains = [](span<const RemotePtr> values, RemotePtr candidate) {
    return std::find(values.begin(), values.end(), candidate) != values.end();
  };
  const auto erase_all = [](vec<RemotePtr>& values, RemotePtr candidate) {
    values.erase(std::remove(values.begin(), values.end(), candidate),
                 values.end());
  };
  const auto released_here = [&](RemotePtr child, u64 sequence) {
    return std::any_of(ops.begin(), ops.end(), [&](const auto& op) {
      return static_cast<ReconcileReverseOpKind>(op.kind) ==
               ReconcileReverseOpKind::release_certificate_lease &&
        RemotePtr{op.new_candidate_raw} == child &&
        op.placement_sequence == sequence;
    });
  };

  for (size_t index = 0; index < ops.size(); ++index) {
    const auto& op = ops[index];
    ReconcileReverseResult& result = plan.results[index];
    result.placement_sequence = op.placement_sequence;
    const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
    switch (kind) {
      case ReconcileReverseOpKind::remove_if_present:
        erase_all(plan.stable, RemotePtr{op.old_candidate_raw});
        erase_all(plan.provisional, RemotePtr{op.old_candidate_raw});
        result.removed = 1;
        break;
      case ReconcileReverseOpKind::remove_provisional_if_present:
        erase_all(plan.provisional, RemotePtr{op.old_candidate_raw});
        result.removed = 1;
        break;
      case ReconcileReverseOpKind::release_certificate_lease:
        plan.released.emplace_back(RemotePtr{op.new_candidate_raw},
                                   op.placement_sequence);
        result.accepted = 1;
        break;
      case ReconcileReverseOpKind::add:
      case ReconcileReverseOpKind::acquire_certificate_lease:
        break;
      default:
        result.stale = 1;
        return plan;
    }
  }

  vec<RemotePtr> candidates = plan.stable;
  const auto append_unique = [&](RemotePtr candidate) {
    if (!candidate.is_null() && !contains(span<const RemotePtr>{candidates},
                                          candidate)) {
      candidates.push_back(candidate);
    }
  };
  vec<RemotePtr> mandatory;
  for (const auto& [child, sequence] : active_leases) {
    if (!released_here(child, sequence)) {
      append_unique(child);
      if (!contains(span<const RemotePtr>{mandatory}, child)) {
        mandatory.push_back(child);
      }
    }
  }
  for (const auto& op : ops) {
    const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
    const RemotePtr candidate{op.new_candidate_raw};
    if (kind == ReconcileReverseOpKind::add ||
        kind == ReconcileReverseOpKind::acquire_certificate_lease) {
      append_unique(candidate);
    }
    if (kind == ReconcileReverseOpKind::acquire_certificate_lease) {
      if (!contains(span<const RemotePtr>{mandatory}, candidate)) {
        mandatory.push_back(candidate);
      }
      plan.acquired.emplace_back(candidate, op.placement_sequence);
    }
  }
  if (mandatory.size() > degree_limit) return plan;

  if (candidates.size() <= degree_limit) {
    plan.stable = std::move(candidates);
  } else {
    plan.stable = robust_prune(candidates);
    u64 forced = 0;
    if (!preserve_reconcile_mandatory_candidates(
          span<const RemotePtr>{candidates},
          span<const RemotePtr>{mandatory}, degree_limit,
          plan.stable, &forced)) {
      return plan;
    }
  }
  if (plan.stable.size() > degree_limit) return plan;

  for (size_t index = 0; index < ops.size(); ++index) {
    const auto& op = ops[index];
    ReconcileReverseResult& result = plan.results[index];
    const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
    const RemotePtr candidate{op.new_candidate_raw};
    if (kind == ReconcileReverseOpKind::add) {
      result.accepted = contains(span<const RemotePtr>{plan.stable}, candidate);
    } else if (kind == ReconcileReverseOpKind::acquire_certificate_lease) {
      result.accepted = contains(span<const RemotePtr>{plan.stable}, candidate);
      if (!result.accepted) {
        result.stale = 1;
        return plan;
      }
      erase_all(plan.provisional, candidate);
    }
  }
  plan.valid = true;
  return plan;
}

}  // namespace memory_node_storage_owner_index_detail
