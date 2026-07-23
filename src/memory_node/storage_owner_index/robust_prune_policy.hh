#pragma once

#include <cstddef>
#include <utility>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_index_detail {

// Predicate form of the common selection loop.  The predicate receives the
// already-computed source distance, allowing an implementation to decide the
// alpha threshold without materializing a complete pair distance.
template <class Candidate, class PointerOf, class SourceDistanceOf,
          class PairPrunes>
void select_alpha_robust_pruned_sorted_by_pair_predicate(
    const span<const Candidate> sorted_candidates,
    const u32 result_limit,
    PointerOf&& pointer_of,
    SourceDistanceOf&& source_distance_of,
    PairPrunes&& pair_prunes,
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
    const auto source_distance = source_distance_of(candidate);
    bool pruned = false;
    for (const size_t selected_index : selected_indices) {
      if (pair_prunes(candidate,
                      sorted_candidates[selected_index],
                      source_distance)) {
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
  select_alpha_robust_pruned_sorted_by_pair_predicate(
    sorted_candidates,
    result_limit,
    std::forward<PointerOf>(pointer_of),
    std::forward<SourceDistanceOf>(source_distance_of),
    [&](const Candidate& candidate,
        const Candidate& retained,
        const auto source_distance) {
      return alpha * pair_distance(candidate, retained) <= source_distance;
    },
    selected,
    selected_indices);
}

}  // namespace memory_node_storage_owner_index_detail
