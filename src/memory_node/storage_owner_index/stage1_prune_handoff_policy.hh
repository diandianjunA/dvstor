#pragma once

#include <algorithm>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_index_detail {

// The converged Stage1 Beam is already ordered by query distance and contains
// no duplicate pointers. Publishing its nearest R entries is O(R), supplies a
// complete fixed-record provisional adjacency, and leaves the expensive
// diversity prune available for exact reconstruction in Stage2.
inline vec<RemotePtr> deferred_stage1_provisional_neighbors(
    span<const RemotePtr> ordered_candidates,
    u32 degree_limit) {
  const size_t count = std::min<size_t>(
    ordered_candidates.size(), degree_limit);
  return vec<RemotePtr>(
    ordered_candidates.begin(), ordered_candidates.begin() + count);
}

// When Stage1 publishes a nearest-first provisional adjacency, the frozen
// source later contains both that baseline and reverse edges installed by
// concurrent insertions. Only the latter must augment the exact deferred
// local-prune seed. Feeding the whole provisional baseline into the final
// prune would silently change the durable candidate set.
inline vec<RemotePtr> stage2_observed_reverse_delta(
    span<const RemotePtr> observed,
    span<const RemotePtr> provisional_baseline) {
  vec<RemotePtr> delta;
  delta.reserve(observed.size());
  for (const RemotePtr pointer : observed) {
    if (std::find(provisional_baseline.begin(),
                  provisional_baseline.end(), pointer) ==
        provisional_baseline.end()) {
      delta.push_back(pointer);
    }
  }
  return delta;
}

}  // namespace memory_node_storage_owner_index_detail
