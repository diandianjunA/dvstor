#pragma once

#include <algorithm>

#include "common/types.hh"
#include "remote_pointer.hh"
#include "service/storage_owner_protocol.hh"

namespace memory_node_storage_owner_index_detail {

inline constexpr bool reconcile_kind_needs_new_identity(
    service::storage_owner::ReconcileReverseOpKind kind) {
  using service::storage_owner::ReconcileReverseOpKind;
  switch (kind) {
    case ReconcileReverseOpKind::remove_if_present:
      return false;
    case ReconcileReverseOpKind::replace_or_add:
    case ReconcileReverseOpKind::add:
    case ReconcileReverseOpKind::ensure_reachable:
    case ReconcileReverseOpKind::promote_stable_bridge:
      return true;
  }
  return false;
}

inline bool reconcile_contains(const vec<RemotePtr>& neighbors,
                               const RemotePtr candidate) {
  return std::find(neighbors.begin(), neighbors.end(), candidate) !=
         neighbors.end();
}

// An ordinary reverse edge is a bounded-degree proposal. RobustPrune may
// legitimately reject it, which is a completed postcondition rather than a
// transport failure. Stage2's one promoted stable bridge is mandatory for
// reachability; removals must prove absence, and replacement must prove
// old-pointer removal even when a new ordinary edge loses pruning.
inline bool reconcile_reverse_postcondition_holds(
    const service::storage_owner::ReconcileReverseOp& op,
    const service::storage_owner::ReconcileReverseResult& result) {
  using service::storage_owner::ReconcileReverseOpKind;
  if (result.placement_sequence != op.placement_sequence ||
      result.stale != 0) {
    return false;
  }
  switch (static_cast<ReconcileReverseOpKind>(op.kind)) {
    case ReconcileReverseOpKind::remove_if_present:
    case ReconcileReverseOpKind::replace_or_add:
      return result.removed != 0;
    case ReconcileReverseOpKind::add:
      return true;
    case ReconcileReverseOpKind::ensure_reachable:
      return result.accepted != 0;
    case ReconcileReverseOpKind::promote_stable_bridge:
      return result.accepted != 0 &&
        (op.old_candidate_raw == 0 || result.removed != 0);
  }
  return false;
}

// Per-operation results describe the adjacency immediately after that
// operation. A later operation on the same hot target may RobustPrune a
// mandatory certificate that an earlier result already reported as accepted.
// The sender retires its Stage1 bridge only after the complete target
// transaction is acknowledged, so reachability must be checked against the
// final adjacency as well.
inline bool reconcile_reverse_final_reachability_holds(
    span<const service::storage_owner::ReconcileReverseOp> ops,
    span<const size_t> op_indices,
    span<const service::storage_owner::ReconcileReverseResult> results,
    span<const RemotePtr> stable,
    span<const RemotePtr> provisional) {
  using service::storage_owner::ReconcileReverseOpKind;
  const auto contains = [](span<const RemotePtr> values,
                           RemotePtr candidate) {
    return std::find(values.begin(), values.end(), candidate) != values.end();
  };
  for (const size_t op_index : op_indices) {
    if (op_index >= ops.size() || op_index >= results.size()) return false;
    const auto& op = ops[op_index];
    const auto& result = results[op_index];
    const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
    if (kind != ReconcileReverseOpKind::ensure_reachable &&
        kind != ReconcileReverseOpKind::promote_stable_bridge) {
      continue;
    }
    if (!reconcile_reverse_postcondition_holds(op, result)) return false;
    const RemotePtr candidate{op.new_candidate_raw};
    const bool present = contains(stable, candidate) ||
      (kind == ReconcileReverseOpKind::ensure_reachable &&
       contains(provisional, candidate));
    if (!present) return false;
  }
  return true;
}

// A tagged target whose physical incarnation has already retired cannot
// contain any edge owned by that incarnation.  Ordinary reverse additions
// are only bounded-degree proposals, so losing their target is a completed
// rejection; removals and replacements have also reached their required
// old-edge-absent postcondition.  The two reachability operations are
// deliberately different: they must force the caller to choose another live
// parent before any provisional bridge may be removed.
inline service::storage_owner::ReconcileReverseResult
reconcile_retired_target_result(
    const service::storage_owner::ReconcileReverseOp& op) {
  using service::storage_owner::ReconcileReverseOpKind;
  service::storage_owner::ReconcileReverseResult result{
    .placement_sequence = op.placement_sequence,
  };
  switch (static_cast<ReconcileReverseOpKind>(op.kind)) {
    case ReconcileReverseOpKind::add:
      break;
    case ReconcileReverseOpKind::remove_if_present:
    case ReconcileReverseOpKind::replace_or_add:
      result.removed = 1;
      break;
    case ReconcileReverseOpKind::ensure_reachable:
    case ReconcileReverseOpKind::promote_stable_bridge:
      result.stale = 1;
      break;
  }
  return result;
}

// Applies one already-validated reconciliation operation to a target's
// in-memory neighbor list. The caller owns the target lock for the complete
// invocation. Identity/liveness booleans are supplied by the storage-backed
// implementation so this policy remains independently testable.
//
// RobustPrune receives the complete current U new candidate set only when an
// add would exceed degree_limit. Its output is defensively restricted to that
// set, deduplicated, and capped at degree_limit before it becomes adjacency.
template <class RobustPrune>
service::storage_owner::ReconcileReverseResult reconcile_reverse_neighbors(
    const service::storage_owner::ReconcileReverseOp& op,
    const bool target_live,
    const bool old_identity_matches,
    const bool new_identity_live,
    const bool replacement_equivalent,
    const u32 degree_limit,
    vec<RemotePtr>& neighbors,
    RobustPrune&& robust_prune) {
  using service::storage_owner::ReconcileReverseOpKind;
  using service::storage_owner::ReconcileReverseResult;

  ReconcileReverseResult result{
    .placement_sequence = op.placement_sequence,
  };
  const auto stale = [&]() {
    result.stale = 1;
    return result;
  };

  if (!target_live || op.placement_sequence == 0 || degree_limit == 0) {
    return stale();
  }

  const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
  const RemotePtr old_candidate{op.old_candidate_raw};
  const RemotePtr new_candidate{op.new_candidate_raw};
  const bool old_present = !old_candidate.is_null() &&
    reconcile_contains(neighbors, old_candidate);
  const bool new_present = !new_candidate.is_null() &&
    reconcile_contains(neighbors, new_candidate);

  const auto erase_all = [&](const RemotePtr candidate) {
    neighbors.erase(
      std::remove(neighbors.begin(), neighbors.end(), candidate),
      neighbors.end());
  };

  const auto add_new = [&](const bool force_prune = false) {
    if (reconcile_contains(neighbors, new_candidate)) {
      result.accepted = 1;
      return;
    }
    if (!force_prune && neighbors.size() < degree_limit) {
      neighbors.push_back(new_candidate);
      result.accepted = 1;
      return;
    }

    vec<RemotePtr> candidates = neighbors;
    candidates.push_back(new_candidate);
    vec<RemotePtr> selected = robust_prune(candidates);
    vec<RemotePtr> bounded;
    bounded.reserve(std::min<size_t>(degree_limit, selected.size()));
    for (const RemotePtr candidate : selected) {
      if (candidate.is_null() ||
          std::find(candidates.begin(), candidates.end(), candidate) ==
            candidates.end() ||
          reconcile_contains(bounded, candidate)) {
        continue;
      }
      bounded.push_back(candidate);
      if (bounded.size() == degree_limit) break;
    }
    neighbors = std::move(bounded);
    result.accepted = reconcile_contains(neighbors, new_candidate) ? 1 : 0;
  };
  const auto ensure_new = [&]() {
    add_new();
    if (result.accepted) return;
    // RobustPrune order is priority order. Preserve its decision except for
    // the last (lowest-priority) survivor, which becomes the single incoming
    // connectivity bridge for this newly inserted node.
    if (neighbors.size() < degree_limit) {
      neighbors.push_back(new_candidate);
    } else {
      lib_assert(!neighbors.empty(),
                 "positive reverse degree has no replacement slot");
      neighbors.back() = new_candidate;
    }
    result.accepted = 1;
  };

  switch (kind) {
    case ReconcileReverseOpKind::remove_if_present:
      if (old_candidate.is_null()) {
        return stale();
      }
      if (old_present && !old_identity_matches) return stale();
      erase_all(old_candidate);
      // This is a postcondition: retries report removed even when the exact
      // old pointer was already absent.
      result.removed = 1;
      return result;

    case ReconcileReverseOpKind::add:
      if (!old_candidate.is_null() || new_candidate.is_null() ||
          !new_identity_live || new_candidate == RemotePtr{op.target_raw}) {
        return stale();
      }
      if (new_present) {
        result.accepted = 1;
        return result;
      }
      add_new();
      return result;

    case ReconcileReverseOpKind::replace_or_add: {
      if (old_candidate.is_null() || new_candidate.is_null() ||
          old_candidate == new_candidate || !new_identity_live ||
          new_candidate == RemotePtr{op.target_raw}) {
        return stale();
      }
      if (old_present && !old_identity_matches) return stale();

      if (new_present) {
        if (old_present) {
          erase_all(old_candidate);
          result.replaced = 1;
        }
        result.accepted = 1;
        result.removed = 1;
        return result;
      }

      if (old_present && replacement_equivalent) {
        bool wrote_replacement = false;
        for (RemotePtr& neighbor : neighbors) {
          if (neighbor != old_candidate) continue;
          if (!wrote_replacement) {
            neighbor = new_candidate;
            wrote_replacement = true;
          } else {
            neighbor.reset();
          }
        }
        neighbors.erase(
          std::remove_if(neighbors.begin(), neighbors.end(),
                         [](const RemotePtr candidate) {
                           return candidate.is_null();
                         }),
          neighbors.end());
        result.accepted = 1;
        result.replaced = 1;
        result.removed = 1;
        return result;
      }

      if (old_present) erase_all(old_candidate);
      result.removed = 1;
      // A same-ID/generation relocation should be byte-identical. If it is
      // not, treating it as an in-place substitution would bypass the
      // bounded-degree selection rule, so reconcile through RobustPrune even
      // when removing old_candidate_raw temporarily freed one slot.
      add_new(old_present && !replacement_equivalent);
      return result;
    }

    case ReconcileReverseOpKind::ensure_reachable:
      if (new_candidate.is_null() || new_candidate == RemotePtr{op.target_raw} ||
          !new_identity_live ||
          (!old_candidate.is_null() && old_present &&
           !old_identity_matches)) {
        return stale();
      }
      if (!old_candidate.is_null()) {
        erase_all(old_candidate);
        result.removed = 1;
      }
      ensure_new();
      result.replaced = old_present && result.accepted ? 1 : 0;
      return result;

    case ReconcileReverseOpKind::promote_stable_bridge:
      if (new_candidate.is_null() ||
          new_candidate == RemotePtr{op.target_raw} ||
          !new_identity_live ||
          (old_present && !old_identity_matches)) {
        return stale();
      }
      if (new_present) {
        if (!old_candidate.is_null() && old_candidate != new_candidate) {
          erase_all(old_candidate);
        }
        result.accepted = 1;
        result.removed = old_candidate.is_null() ? 0 : 1;
        result.replaced = old_present && old_candidate != new_candidate;
        return result;
      }
      if (!old_candidate.is_null()) {
        erase_all(old_candidate);
        result.removed = 1;
      }
      ensure_new();
      result.replaced = old_present && result.accepted ? 1 : 0;
      return result;
  }

  return stale();
}

// Reconciles the ordinary Vamana plane and the bounded protected-backlink
// plane. Stage1 first reserves a small protected certificate to make a new
// node reachable. Stage2 publishes the ordinary proposals, promotes exactly
// one final backlink into `stable`, and then removes every temporary protected
// edge. This makes the certificate bounded in both space and lifetime while
// leaving only the ordinary R-bounded graph after Stage2.
template <class RobustPrune>
service::storage_owner::ReconcileReverseResult reconcile_reverse_adjacency(
    const service::storage_owner::ReconcileReverseOp& op,
    const bool target_stable,
    const bool old_identity_matches,
    const bool new_identity_stable,
    const bool replacement_equivalent,
    const u32 degree_limit,
    const u32 protected_limit,
    vec<RemotePtr>& stable,
    vec<RemotePtr>& provisional,
    RobustPrune&& robust_prune) {
  using service::storage_owner::ReconcileReverseOpKind;
  using service::storage_owner::ReconcileReverseResult;

  ReconcileReverseResult result{.placement_sequence = op.placement_sequence};
  const auto stale_result = [&]() {
    result.stale = 1;
    return result;
  };
  if (!target_stable || op.placement_sequence == 0 || degree_limit == 0 ||
      protected_limit == 0) {
    return stale_result();
  }

  const RemotePtr target{op.target_raw};
  const RemotePtr old_candidate{op.old_candidate_raw};
  const RemotePtr new_candidate{op.new_candidate_raw};
  const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
  const auto contains = [](const vec<RemotePtr>& values, RemotePtr value) {
    return std::find(values.begin(), values.end(), value) != values.end();
  };
  const auto erase_all = [](vec<RemotePtr>& values, RemotePtr value) {
    values.erase(std::remove(values.begin(), values.end(), value),
                 values.end());
  };
  const auto old_present = [&]() {
    return !old_candidate.is_null() &&
      (contains(stable, old_candidate) ||
       contains(provisional, old_candidate));
  };
  const auto new_stable = [&]() {
    return !new_candidate.is_null() && contains(stable, new_candidate);
  };

  const bool observed_old = old_present();
  const auto add_stable = [&](bool force_prune) {
    erase_all(provisional, new_candidate);
    if (new_stable()) {
      result.accepted = 1;
      return;
    }
    if (!force_prune && stable.size() < degree_limit) {
      stable.push_back(new_candidate);
      result.accepted = 1;
      return;
    }
    vec<RemotePtr> candidates = stable;
    candidates.push_back(new_candidate);
    vec<RemotePtr> selected = robust_prune(candidates);
    vec<RemotePtr> bounded;
    bounded.reserve(std::min<size_t>(degree_limit, selected.size()));
    for (const RemotePtr candidate : selected) {
      if (candidate.is_null() ||
          std::find(candidates.begin(), candidates.end(), candidate) ==
            candidates.end() ||
          contains(bounded, candidate)) {
        continue;
      }
      bounded.push_back(candidate);
      if (bounded.size() == degree_limit) break;
    }
    stable = std::move(bounded);
    result.accepted = new_stable() ? 1 : 0;
  };
  const auto force_stable = [&]() {
    add_stable(false);
    if (result.accepted) return;
    // RobustPrune output is priority ordered. Preserve all but its last
    // survivor and spend the one Stage2 connectivity exception on the final
    // candidate. The stable degree remains bounded, and later ordinary graph
    // maintenance is free to prune this edge.
    if (stable.size() < degree_limit) {
      stable.push_back(new_candidate);
    } else {
      lib_assert(!stable.empty(),
                 "positive stable degree has no bridge replacement slot");
      stable.back() = new_candidate;
    }
    result.accepted = 1;
  };
  switch (kind) {
    case ReconcileReverseOpKind::remove_if_present:
      if (old_candidate.is_null() ||
          (observed_old && !old_identity_matches)) {
        return stale_result();
      }
      erase_all(stable, old_candidate);
      erase_all(provisional, old_candidate);
      result.removed = 1;
      return result;

    case ReconcileReverseOpKind::add:
      if (!old_candidate.is_null() || new_candidate.is_null() ||
          new_candidate == target || !new_identity_stable) {
        return stale_result();
      }
      add_stable(false);
      return result;

    case ReconcileReverseOpKind::replace_or_add: {
      if (old_candidate.is_null() || new_candidate.is_null() ||
          old_candidate == new_candidate || new_candidate == target ||
          !new_identity_stable ||
          (observed_old && !old_identity_matches)) {
        return stale_result();
      }
      if (new_stable()) {
        erase_all(stable, old_candidate);
        erase_all(provisional, old_candidate);
        erase_all(provisional, new_candidate);
        result.accepted = 1;
        result.replaced = observed_old ? 1 : 0;
        result.removed = 1;
        return result;
      }

      if (contains(stable, old_candidate) && replacement_equivalent) {
        for (RemotePtr& candidate : stable) {
          if (candidate == old_candidate) candidate = new_candidate;
        }
        erase_all(provisional, old_candidate);
        erase_all(provisional, new_candidate);
        result.accepted = 1;
        result.replaced = 1;
        result.removed = 1;
        return result;
      }

      const bool old_was_stable = contains(stable, old_candidate);
      erase_all(stable, old_candidate);
      erase_all(provisional, old_candidate);
      result.removed = 1;
      add_stable(old_was_stable && !replacement_equivalent);
      result.replaced = observed_old && result.accepted ? 1 : 0;
      return result;
    }

    case ReconcileReverseOpKind::ensure_reachable:
      if (new_candidate.is_null() || new_candidate == target ||
          !new_identity_stable ||
          (observed_old && !old_identity_matches)) {
        return stale_result();
      }

      // ensure_reachable is a reservation handoff, never a forced eviction
      // from the R-bounded stable graph.  The new pointer must either already
      // own the Stage1 protected slot (in-place finalization/retry), or replace
      // the exact old physical incarnation in that same slot (migration).
      // If neither postcondition is true, report rejected without mutating;
      // the caller must choose another acknowledged Stage1 parent.
      if (contains(provisional, new_candidate)) {
        if (!old_candidate.is_null() && old_candidate != new_candidate) {
          erase_all(stable, old_candidate);
          erase_all(provisional, old_candidate);
          result.removed = 1;
          result.replaced = observed_old ? 1 : 0;
        }
        erase_all(stable, new_candidate);
        result.accepted = 1;
        return result;
      }

      // Deletion repair has no old reservation to transfer. It may reserve
      // only an actually free protected slot; neither a stable neighbor nor
      // another insertion's protected edge may be displaced. If this target
      // already has an ordinary edge to the child, move that exact edge into
      // the protected plane so later RobustPrune cannot invalidate the ACK.
      if (old_candidate.is_null()) {
        if (provisional.size() >= protected_limit) {
          return result;
        }
        erase_all(stable, new_candidate);
        provisional.push_back(new_candidate);
        result.accepted = 1;
        return result;
      }

      if (old_candidate == new_candidate ||
          !contains(provisional, old_candidate)) {
        return result;
      }
      for (RemotePtr& candidate : provisional) {
        if (candidate == old_candidate) candidate = new_candidate;
      }
      erase_all(stable, old_candidate);
      erase_all(stable, new_candidate);
      result.accepted = 1;
      result.removed = 1;
      result.replaced = 1;
      return result;

    case ReconcileReverseOpKind::promote_stable_bridge: {
      if (new_candidate.is_null() || new_candidate == target ||
          !new_identity_stable ||
          (observed_old && !old_identity_matches)) {
        return stale_result();
      }

      // An ACKed stable pointer is the idempotent terminal state. Consume any
      // leftover old protected pointer, taking care not to erase the same
      // in-place final pointer from the stable plane.
      if (new_stable()) {
        if (!old_candidate.is_null()) {
          if (old_candidate != new_candidate) {
            erase_all(stable, old_candidate);
          }
          erase_all(provisional, old_candidate);
          result.removed = 1;
          result.replaced = observed_old && old_candidate != new_candidate;
        }
        erase_all(provisional, new_candidate);
        result.accepted = 1;
        return result;
      }

      if (!old_candidate.is_null()) {
        // A first promotion must own the exact Stage1 protected certificate.
        // If it vanished and the final stable pointer is also absent, retrying
        // cannot prove a safe bridge handoff.
        if (!contains(provisional, old_candidate)) return result;
        erase_all(stable, old_candidate);
        erase_all(provisional, old_candidate);
        result.removed = 1;
      }
      erase_all(provisional, new_candidate);
      force_stable();
      result.replaced = observed_old && result.accepted ? 1 : 0;
      return result;
    }
  }
  return stale_result();
}

// Returns the maximal per-target run of ordinary proposals beginning at
// `begin`. Stronger operations return an empty run and therefore remain an
// explicit execution/pruning boundary. `op_indices` preserves the original
// order of operations for one target while allowing unrelated targets to be
// grouped independently.
inline size_t reconcile_reverse_add_run_end(
    const span<const service::storage_owner::ReconcileReverseOp> ops,
    const span<const size_t> op_indices,
    const size_t begin) {
  using service::storage_owner::ReconcileReverseOpKind;
  if (begin >= op_indices.size()) return begin;
  lib_assert(op_indices[begin] < ops.size(),
             "reverse add run index exceeds operation batch");
  if (static_cast<ReconcileReverseOpKind>(
        ops[op_indices[begin]].kind) != ReconcileReverseOpKind::add) {
    return begin;
  }
  size_t end = begin + 1;
  while (end < op_indices.size()) {
    lib_assert(op_indices[end] < ops.size(),
               "reverse add run index exceeds operation batch");
    if (static_cast<ReconcileReverseOpKind>(
          ops[op_indices[end]].kind) != ReconcileReverseOpKind::add) {
      break;
    }
    ++end;
  }
  return end;
}

// Produces the receiver-side execution order for one locked target. An
// install transaction can arrive in any wire order (and future senders may be
// buggy), but ordinary stable-plane work must never run after a mandatory
// promotion: doing so could evict an ACKed reachability certificate before the
// provisional-removal barrier. If a promotion is present, accept only the
// install vocabulary and deterministically partition every ordinary
// add/replace before every promotion while retaining order inside each class.
// Transactions without a promotion (cleanup, removal, and Stage1 repair)
// retain their exact legacy order.
inline bool reconcile_reverse_target_execution_order(
    const span<const service::storage_owner::ReconcileReverseOp> ops,
    const span<const size_t> op_indices,
    vec<size_t>& execution_order) {
  using service::storage_owner::ReconcileReverseOpKind;
  execution_order.clear();
  execution_order.reserve(op_indices.size());
  bool has_promotion = false;
  for (const size_t op_index : op_indices) {
    if (op_index >= ops.size()) return false;
    has_promotion = has_promotion ||
      static_cast<ReconcileReverseOpKind>(ops[op_index].kind) ==
        ReconcileReverseOpKind::promote_stable_bridge;
  }
  if (!has_promotion) {
    execution_order.insert(
      execution_order.end(), op_indices.begin(), op_indices.end());
    return true;
  }

  for (const size_t op_index : op_indices) {
    const auto kind = static_cast<ReconcileReverseOpKind>(ops[op_index].kind);
    if (kind == ReconcileReverseOpKind::add ||
        kind == ReconcileReverseOpKind::replace_or_add) {
      execution_order.push_back(op_index);
    } else if (kind != ReconcileReverseOpKind::promote_stable_bridge) {
      return false;
    }
  }
  for (const size_t op_index : op_indices) {
    if (static_cast<ReconcileReverseOpKind>(ops[op_index].kind) ==
        ReconcileReverseOpKind::promote_stable_bridge) {
      execution_order.push_back(op_index);
    }
  }
  return execution_order.size() == op_indices.size();
}

// Applies one compatible run of ordinary reverse-edge proposals to the same
// target. Ordinary `add` is deliberately a proposal: RobustPrune may reject
// it and the RPC postcondition is still complete. This lets a receiver score
// the union of a hot target's proposals once instead of repeatedly pruning
// after every arrival in the same RPC.
//
// The caller has already read and locked the target adjacency. Per-operation
// identity/liveness decisions remain explicit inputs so batching cannot blur
// incarnation or placement-sequence fences. Non-add operations are excluded:
// promotion/replace/remove have stronger ordered postconditions and therefore
// form barriers between compatible add runs.
template <class RobustPrune>
void reconcile_reverse_add_batch(
    const span<const service::storage_owner::ReconcileReverseOp> ops,
    const span<const u8> new_identity_stable,
    const bool target_stable,
    const u32 degree_limit,
    const u32 protected_limit,
    vec<RemotePtr>& stable,
    vec<RemotePtr>& provisional,
    vec<service::storage_owner::ReconcileReverseResult>& results,
    RobustPrune&& robust_prune) {
  using service::storage_owner::ReconcileReverseOpKind;
  using service::storage_owner::ReconcileReverseResult;

  lib_assert(ops.size() == new_identity_stable.size(),
             "reverse add batch identity cardinality mismatch");
  results.assign(ops.size(), {});
  if (ops.empty()) return;

  const RemotePtr target{ops[0].target_raw};
  vec<RemotePtr> proposals;
  proposals.reserve(ops.size());
  vec<u8> valid(ops.size(), 0);
  const auto contains = [](const vec<RemotePtr>& values, RemotePtr value) {
    return std::find(values.begin(), values.end(), value) != values.end();
  };
  const auto erase_all = [](vec<RemotePtr>& values, RemotePtr value) {
    values.erase(std::remove(values.begin(), values.end(), value),
                 values.end());
  };

  for (size_t index = 0; index < ops.size(); ++index) {
    const auto& op = ops[index];
    ReconcileReverseResult& result = results[index];
    result.placement_sequence = op.placement_sequence;
    const RemotePtr candidate{op.new_candidate_raw};
    if (!target_stable || degree_limit == 0 || protected_limit == 0 ||
        op.placement_sequence == 0 || op.target_raw != target.raw_address ||
        static_cast<ReconcileReverseOpKind>(op.kind) !=
          ReconcileReverseOpKind::add ||
        op.old_candidate_raw != 0 || candidate.is_null() ||
        candidate == target || new_identity_stable[index] == 0) {
      result.stale = 1;
      continue;
    }

    valid[index] = 1;
    // A final stable proposal consumes an obsolete provisional copy even if
    // the bounded stable plane ultimately rejects that ordinary proposal.
    erase_all(provisional, candidate);
    if (!contains(stable, candidate) && !contains(proposals, candidate)) {
      proposals.push_back(candidate);
    }
  }

  if (!proposals.empty()) {
    vec<RemotePtr> candidates = stable;
    candidates.insert(candidates.end(), proposals.begin(), proposals.end());
    if (candidates.size() <= degree_limit) {
      stable = std::move(candidates);
    } else {
      vec<RemotePtr> selected = robust_prune(candidates);
      vec<RemotePtr> bounded;
      bounded.reserve(std::min<size_t>(degree_limit, selected.size()));
      for (const RemotePtr candidate : selected) {
        if (candidate.is_null() ||
            std::find(candidates.begin(), candidates.end(), candidate) ==
              candidates.end() ||
            contains(bounded, candidate)) {
          continue;
        }
        bounded.push_back(candidate);
        if (bounded.size() == degree_limit) break;
      }
      stable = std::move(bounded);
    }
  }

  for (size_t index = 0; index < ops.size(); ++index) {
    if (valid[index] == 0) continue;
    results[index].accepted = contains(
      stable, RemotePtr{ops[index].new_candidate_raw}) ? 1 : 0;
  }
}

}  // namespace memory_node_storage_owner_index_detail
