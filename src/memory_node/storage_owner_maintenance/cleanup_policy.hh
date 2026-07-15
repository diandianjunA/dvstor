#pragma once

#include <algorithm>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_maintenance_detail {

// A stale stitch repair owns only the backlinks attempted by that stitch.
// The mutation that made the stitch stale has its own ordinary cleanup intent
// and removes the tombstone's preserved adjacency after the earlier repair
// sequence advances. Keeping the two sets separate also preserves the schema-15
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
  for (const RemotePtr neighbor : globally_pruned) {
    if (!neighbor.is_null() &&
        std::find(rebased.begin(), rebased.end(), neighbor) == rebased.end()) {
      rebased.push_back(neighbor);
    }
  }
  for (const RemotePtr neighbor : observed_neighbors) {
    if (neighbor.is_null() ||
        std::find(stage1_neighbors.begin(), stage1_neighbors.end(), neighbor) !=
          stage1_neighbors.end() ||
        std::find(rebased.begin(), rebased.end(), neighbor) != rebased.end()) {
      continue;
    }
    rebased.push_back(neighbor);
  }
  return rebased;
}

}  // namespace memory_node_storage_owner_maintenance_detail
