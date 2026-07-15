#pragma once

#include <algorithm>
#include <utility>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace memory_node_storage_owner_index_detail {

// The caller must hold the reverse target's node lock until the resulting
// candidates have been written.  Liveness is deliberately checked here, at
// the final write boundary: a delete cleanup that finished before the lock was
// acquired is observed as dead, while one that starts afterwards must wait for
// the target lock and will remove the newly written backlink.
template <class IsLive>
void select_fresh_reverse_candidates_locked(
    const vec<RemotePtr>& current_neighbors,
    const vec<RemotePtr>& candidates,
    IsLive&& is_live,
    vec<RemotePtr>& selected) {
  selected.clear();
  selected.reserve(candidates.size());
  for (const RemotePtr& candidate : candidates) {
    if (candidate.is_null() ||
        std::find(current_neighbors.begin(), current_neighbors.end(),
                  candidate) != current_neighbors.end() ||
        std::find(selected.begin(), selected.end(), candidate) !=
          selected.end() ||
        !is_live(candidate)) {
      continue;
    }
    selected.push_back(candidate);
  }
}

}  // namespace memory_node_storage_owner_index_detail
