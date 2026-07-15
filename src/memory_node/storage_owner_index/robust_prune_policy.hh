#pragma once

#include <cstddef>
#include <utility>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_index_detail {

// Applies the common Vamana alpha RobustPrune policy to candidates that are
// already ordered by their distance to the source node.  Keeping this small
// policy independent of snapshot storage lets new-node pruning and reverse
// edge overflow pruning share exactly the same selection rule.
template <class Candidate, class PointerOf, class SourceDistanceOf,
          class PairDistance>
void select_alpha_robust_pruned_sorted(
    const span<const Candidate> sorted_candidates,
    const u32 result_limit,
    const f64 alpha,
    PointerOf&& pointer_of,
    SourceDistanceOf&& source_distance_of,
    PairDistance&& pair_distance,
    vec<RemotePtr>& selected,
    vec<size_t>& selected_indices) {
  selected.clear();
  selected_indices.clear();
  selected.reserve(result_limit);
  selected_indices.reserve(result_limit);

  for (size_t candidate_index = 0;
       candidate_index < sorted_candidates.size() &&
       selected.size() < result_limit;
       ++candidate_index) {
    const Candidate& candidate = sorted_candidates[candidate_index];
    bool pruned = false;
    for (const size_t selected_index : selected_indices) {
      if (alpha * pair_distance(candidate,
                                sorted_candidates[selected_index]) <=
          source_distance_of(candidate)) {
        pruned = true;
        break;
      }
    }
    if (!pruned) {
      selected.push_back(pointer_of(candidate));
      selected_indices.push_back(candidate_index);
    }
  }
}

}  // namespace memory_node_storage_owner_index_detail
