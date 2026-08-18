#pragma once

#include <algorithm>
#include <limits>

#include "common/types.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace memory_node_storage_owner_maintenance_detail {

// One maintenance worker builds a single snapshot wave for every Stage2 task
// that has crossed the same frozen graph boundary. Physical records shared by
// several tasks are read once, while task_target_indices retains each task's
// exact candidate order (including repeated candidates). Keeping the plan in
// worker-owned scratch makes its capacity O(batch * L), independent of the
// number of in-flight Stage2 contexts.
struct Stage2SnapshotWavePlan {
  static constexpr u32 missing = std::numeric_limits<u32>::max();

  vec<RemotePtr> targets;
  vec<vec<u32>> task_target_indices;
  dense_hashmap_t<RemotePtr, u32> target_indices;
  size_t task_count{};

  void build(span<const vec<RemotePtr>> candidates_by_task) {
    size_t candidate_count = 0;
    for (const vec<RemotePtr>& candidates : candidates_by_task) {
      candidate_count += candidates.size();
    }
    targets.clear();
    targets.reserve(candidate_count);
    target_indices.clear();
    target_indices.reserve(candidate_count);
    task_count = candidates_by_task.size();
    if (task_target_indices.size() < task_count) {
      task_target_indices.resize(task_count);
    }
    for (vec<u32>& indices : task_target_indices) indices.clear();

    for (size_t task = 0; task < task_count; ++task) {
      vec<u32>& indices = task_target_indices[task];
      indices.reserve(candidates_by_task[task].size());
      for (const RemotePtr candidate : candidates_by_task[task]) {
        if (candidate.is_null()) continue;
        lib_assert(targets.size() < missing,
                   "Stage2 snapshot wave target index overflow");
        const auto [position, inserted] = target_indices.emplace(
          candidate, static_cast<u32>(targets.size()));
        if (inserted) {
          targets.push_back(candidate);
        }
        indices.push_back(position->second);
      }
    }
  }
};

struct Stage2StableBacklinkTarget {
  RemotePtr target;
  bool had_stage1_bridge{};
};

struct Stage2BacklinkPlan {
  RemotePtr promotion_target;
  bool promotion_consumes_stage1_bridge{};
  vec<Stage2StableBacklinkTarget> ordinary_stable_targets;
  vec<RemotePtr> obsolete_stage1_bridges;
};

// Finalization only needs to revalidate records that can carry one of its
// reachability postconditions: an original provisional Stage1 bridge, the
// already-ACKed promotion certificate, or the deterministic promotion target
// whose response may have been lost. Ordinary final neighbors cannot contain
// either edge as a consequence of this insertion.
inline vec<RemotePtr> stage2_revalidation_parents(
    span<const RemotePtr> stage1_bridges,
    RemotePtr acknowledged_certificate,
    RemotePtr planned_promotion_target) {
  vec<RemotePtr> parents(stage1_bridges.begin(), stage1_bridges.end());
  if (!acknowledged_certificate.is_null()) {
    parents.push_back(acknowledged_certificate);
  }
  if (!planned_promotion_target.is_null()) {
    parents.push_back(planned_promotion_target);
  }
  std::sort(parents.begin(), parents.end(),
            [](RemotePtr lhs, RemotePtr rhs) {
              return lhs.raw_address < rhs.raw_address;
            });
  parents.erase(std::remove_if(parents.begin(), parents.end(),
                               [](RemotePtr value) {
                                 return value.is_null();
                               }),
                parents.end());
  parents.erase(std::unique(parents.begin(), parents.end()), parents.end());
  return parents;
}

// Final Stage2 parents must be durable members of the already-published graph,
// not merely readable records.  In particular, excluding an unaccounted
// destination prevents a wave of concurrent Stage2 insertions from using only
// one another as parents and forming a disconnected dependency cycle.
inline bool stage2_parent_is_stable(const u64 header, const bool deleted) {
  return !deleted &&
    VamanaNode::stable_graph_mutation_allowed(header) &&
    (header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0;
}

// Produces the two Stage2 backlink barriers without retaining any long-lived
// protected edge. Prefer promoting a Stage1 parent that survived final prune;
// otherwise establish the mandatory stable bridge at the first final parent
// while an existing Stage1 bridge remains query-visible. Input order is
// preserved, making the choice deterministic for an identical final beam.
inline Stage2BacklinkPlan plan_stage2_backlink_reconciliation(
    span<const RemotePtr> stage1_bridges,
    span<const RemotePtr> final_targets,
    RemotePtr acknowledged_certificate = {}) {
  Stage2BacklinkPlan plan;
  hashset_t<RemotePtr> stage1_set;
  hashset_t<RemotePtr> final_set;
  stage1_set.reserve(stage1_bridges.size());
  final_set.reserve(final_targets.size());
  for (const RemotePtr target : stage1_bridges) {
    if (!target.is_null()) stage1_set.insert(target);
  }

  vec<RemotePtr> unique_final;
  unique_final.reserve(final_targets.size());
  for (const RemotePtr target : final_targets) {
    if (!target.is_null() && final_set.insert(target).second) {
      unique_final.push_back(target);
    }
  }
  // A previously ACKed stable certificate is the strongest retry anchor.
  // Otherwise consume a surviving Stage1 protected edge before considering a
  // newly discovered final parent.  This both handles an empty final set and
  // prevents small batches of fresh nodes from depending only on one another.
  if (!acknowledged_certificate.is_null()) {
    plan.promotion_target = acknowledged_certificate;
    plan.promotion_consumes_stage1_bridge =
      stage1_set.contains(acknowledged_certificate);
  } else {
    for (const RemotePtr target : unique_final) {
      if (stage1_set.contains(target)) {
        plan.promotion_target = target;
        plan.promotion_consumes_stage1_bridge = true;
        break;
      }
    }
    if (plan.promotion_target.is_null() && !stage1_bridges.empty()) {
      for (const RemotePtr target : stage1_bridges) {
        if (target.is_null()) continue;
        plan.promotion_target = target;
        plan.promotion_consumes_stage1_bridge = true;
        break;
      }
    }
    if (plan.promotion_target.is_null() && !unique_final.empty()) {
      plan.promotion_target = unique_final.front();
    }
  }

  plan.ordinary_stable_targets.reserve(
    unique_final.empty() ? 0 : unique_final.size() - 1);
  for (const RemotePtr target : unique_final) {
    if (target == plan.promotion_target) continue;
    plan.ordinary_stable_targets.push_back(Stage2StableBacklinkTarget{
      .target = target,
      .had_stage1_bridge = stage1_set.contains(target),
    });
  }

  hashset_t<RemotePtr> seen_stage1;
  seen_stage1.reserve(stage1_set.size());
  plan.obsolete_stage1_bridges.reserve(stage1_set.size());
  for (const RemotePtr target : stage1_bridges) {
    if (target.is_null() || !seen_stage1.insert(target).second ||
        final_set.contains(target) || target == plan.promotion_target) {
      continue;
    }
    plan.obsolete_stage1_bridges.push_back(target);
  }
  return plan;
}

inline bool cleanup_deleted_candidate_matches(
    node_t expected_id, u32 expected_generation,
    node_t observed_id, u32 observed_generation,
    bool observed_deleted) {
  return observed_deleted && observed_id == expected_id &&
    observed_generation == expected_generation;
}

inline bool cleanup_reverse_target_matches(
    node_t expected_id, u32 expected_generation,
    node_t observed_id, u32 observed_generation,
    bool observed_deleted) {
  return !observed_deleted && observed_id == expected_id &&
    observed_generation == expected_generation;
}

// A stale Stage2 finalization repair owns only the backlinks attempted by
// that finalization.
// The mutation that made the Stage2 finalization stale has its own ordinary cleanup intent
// and removes the tombstone's preserved adjacency after the earlier repair
// sequence advances. Keeping the two sets separate also preserves the bounded
// R-operations-per-item peer RPC bound.
inline vec<RemotePtr> select_cleanup_neighbors(
    bool repair_only,
    span<const RemotePtr> preserved_neighbors,
    span<const RemotePtr> supplemental_neighbors) {
  vec<RemotePtr> selected;
  selected.reserve((repair_only ? 0 : preserved_neighbors.size()) +
                   supplemental_neighbors.size());

  const auto append_unique = [&](span<const RemotePtr> neighbors) {
    for (const RemotePtr neighbor : neighbors) {
      if (!neighbor.is_null() &&
          std::find(selected.begin(), selected.end(), neighbor) ==
            selected.end()) {
        selected.push_back(neighbor);
      }
    }
  };

  if (!repair_only) {
    append_unique(preserved_neighbors);
  }
  append_unique(supplemental_neighbors);
  return selected;
}

// A protected child can be reparented only through one of its own durable
// outgoing neighbors. Besides preserving graph locality, this makes the new
// protected parent discoverable by the child's later tombstone cleanup; an
// unrelated fallback parent would otherwise become untracked protected
// state. The order is deterministic and dataset-independent: prefer the
// child's physical shard, then the encoded handle order.
inline vec<RemotePtr> order_protected_reparent_candidates(
    RemotePtr child,
    RemotePtr retiring_parent,
    span<const RemotePtr> child_stable_neighbors) {
  vec<RemotePtr> candidates;
  candidates.reserve(child_stable_neighbors.size());
  for (const RemotePtr candidate : child_stable_neighbors) {
    if (candidate.is_null() || candidate == child ||
        candidate == retiring_parent ||
        std::find(candidates.begin(), candidates.end(), candidate) !=
          candidates.end()) {
      continue;
    }
    candidates.push_back(candidate);
  }
  std::sort(candidates.begin(), candidates.end(),
            [child](RemotePtr lhs, RemotePtr rhs) {
              const bool lhs_local =
                lhs.memory_node() == child.memory_node();
              const bool rhs_local =
                rhs.memory_node() == child.memory_node();
              if (lhs_local != rhs_local) return lhs_local;
              return lhs.raw_address < rhs.raw_address;
            });
  return candidates;
}

inline bool protected_reparent_target_has_capacity(
    RemotePtr child,
    span<const RemotePtr> provisional,
    u32 protected_limit) {
  if (child.is_null() || protected_limit == 0) return false;
  if (std::find(provisional.begin(), provisional.end(), child) !=
      provisional.end()) {
    return true;
  }
  // A stable edge to the same child can be moved, not duplicated, into a free
  // protected slot. No unrelated stable/protected edge is ever displaced.
  return provisional.size() < protected_limit;
}

// Once the authority CAS has moved a generation from source to destination,
// a successor captures only destination as its cleanup predecessor. If the
// old Stage2 then becomes stale before retiring source, no later cleanup owns
// source. The stale continuation must therefore finish that retirement
// itself; before the placement CAS, the successor cleanup still owns source.
inline bool stale_stage2_owns_source_retirement(
    bool placement_committed,
    RemotePtr source,
    RemotePtr destination) {
  return placement_committed && !source.is_null() &&
    !destination.is_null() && source != destination;
}

// Stage2 starts from the globally pruned outgoing set, then preserves only
// neighbors that appeared after stage1 published its temporary adjacency.
// Those later neighbors are acknowledged concurrent reverse-edge additions;
// dropping them at final commit would lose already completed graph work.
inline vec<RemotePtr> merge_stage2_rebase_candidates(
    span<const RemotePtr> globally_pruned,
    span<const RemotePtr> stage1_neighbors,
    span<const RemotePtr> observed_neighbors) {
  vec<RemotePtr> rebased;
  rebased.reserve(globally_pruned.size() + observed_neighbors.size());
  hashset_t<RemotePtr> stage1_set;
  hashset_t<RemotePtr> rebased_set;
  stage1_set.reserve(stage1_neighbors.size());
  rebased_set.reserve(globally_pruned.size() + observed_neighbors.size());
  for (const RemotePtr neighbor : stage1_neighbors) {
    if (!neighbor.is_null()) stage1_set.insert(neighbor);
  }
  for (const RemotePtr neighbor : globally_pruned) {
    if (!neighbor.is_null() && rebased_set.insert(neighbor).second) {
      rebased.push_back(neighbor);
    }
  }
  for (const RemotePtr neighbor : observed_neighbors) {
    if (neighbor.is_null() || stage1_set.contains(neighbor) ||
        !rebased_set.insert(neighbor).second) {
      continue;
    }
    rebased.push_back(neighbor);
  }
  return rebased;
}

}  // namespace memory_node_storage_owner_maintenance_detail
