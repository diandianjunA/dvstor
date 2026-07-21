#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/reverse_batch_policy.hh"
#include "memory_node/storage_owner_index/robust_prune_policy.hh"

using memory_node_storage_owner_index_detail::
  select_fresh_reverse_candidates_locked;
using memory_node_storage_owner_index_detail::
  select_alpha_robust_pruned_sorted;
using memory_node_storage_owner_index_detail::IncarnationLockResult;

namespace {

struct PendingReverseUpdate {
  RemotePtr target;
  vec<RemotePtr> candidates;
  vec<RemotePtr> current_neighbors;
  vec<RemotePtr> selected_neighbors;
};

struct ScoredReverseCandidate {
  RemotePtr rptr;
  const byte_t* vector{};
  distance_t distance{};
};

bool same_neighbors(const vec<RemotePtr>& lhs, const vec<RemotePtr>& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

}  // namespace

bool MemoryNode::apply_local_reverse_updates_batched(
    const dense_hashmap_t<u64, vec<RemotePtr>>& updates,
    const Configuration& config) {
  if (updates.empty()) {
    return true;
  }

  // A stage2 batch commonly carries the same newly inserted candidate to
  // many targets. This cache is only an early rejection for pointers already
  // known to be dead. A positive result is always revalidated while holding
  // each reverse target lock at the final write boundary below.
  dense_hashmap_t<u64, bool> candidate_liveness;
  const auto candidate_live = [&](const RemotePtr& candidate) {
    const auto found = candidate_liveness.find(candidate.raw_address);
    if (found != candidate_liveness.end()) {
      return found->second;
    }
    const bool live = storage_owner_node_stable(candidate);
    candidate_liveness.emplace(candidate.raw_address, live);
    return live;
  };

  vec<u64> target_raws;
  target_raws.reserve(updates.size());
  for (const auto& [target_raw, candidates] : updates) {
    (void)candidates;
    target_raws.push_back(target_raw);
  }
  std::sort(target_raws.begin(), target_raws.end());

  vec<PendingReverseUpdate> pending;
  vec<RemotePtr> snapshots_needed;
  dense_hashmap_t<u64, vec<RemotePtr>> conflicted;
  pending.reserve(target_raws.size());

  for (const u64 target_raw : target_raws) {
    const RemotePtr target{target_raw};
    lib_assert(local_shard(target.memory_node()),
               "batched reverse-update target must be local");

    vec<RemotePtr> unique_candidates;
    const auto& candidates = updates.at(target_raw);
    unique_candidates.reserve(candidates.size());
    for (const RemotePtr& candidate : candidates) {
      if (!candidate.is_null() && candidate_live(candidate) &&
          std::find(unique_candidates.begin(), unique_candidates.end(), candidate) ==
            unique_candidates.end()) {
        unique_candidates.push_back(candidate);
      }
    }
    if (unique_candidates.empty()) {
      continue;
    }

    const IncarnationLockResult target_lock = try_lock_node(target);
    if (target_lock == IncarnationLockResult::stale) {
      // The old target incarnation no longer exists; adding an edge to it is
      // already an idempotent no-op and must not touch the replacement slot.
      continue;
    }
    if (target_lock == IncarnationLockResult::busy) return false;
    const u64 target_header = load_local_node_header_acquire(target);
    const bool target_deleted =
      (target_header & VamanaNode::HEADER_DELETED) != 0;
    const bool target_unavailable =
      !VamanaNode::stable_graph_mutation_allowed(target_header);
    if (target_deleted) {
      unlock_node(target);
      continue;
    }
    if (target_unavailable) {
      unlock_node(target);
      return false;
    }

    vec<RemotePtr> current_neighbors = read_stable_neighbor_list(target);
    vec<RemotePtr> filtered_candidates;
    select_fresh_reverse_candidates_locked(
      current_neighbors, unique_candidates,
      [this](const RemotePtr& candidate) {
        return storage_owner_node_stable(candidate);
      },
      filtered_candidates);
    if (filtered_candidates.empty()) {
      unlock_node(target);
      continue;
    }
    if (current_neighbors.size() + filtered_candidates.size() <= config.R) {
      current_neighbors.insert(current_neighbors.end(),
                               filtered_candidates.begin(),
                               filtered_candidates.end());
      write_neighbor_list(target, current_neighbors);
      unlock_node(target);
      continue;
    }
    unlock_node(target);

    PendingReverseUpdate update;
    update.target = target;
    update.candidates = std::move(filtered_candidates);
    update.current_neighbors = std::move(current_neighbors);
    for (const RemotePtr& neighbor : update.current_neighbors) {
      if (!neighbor.is_null()) {
        snapshots_needed.push_back(neighbor);
      }
    }
    for (const RemotePtr& candidate : update.candidates) {
      if (!candidate.is_null()) {
        snapshots_needed.push_back(candidate);
      }
    }
    pending.push_back(std::move(update));
  }

  if (pending.empty()) {
    return true;
  }

  std::sort(snapshots_needed.begin(), snapshots_needed.end(),
            [](const RemotePtr& lhs, const RemotePtr& rhs) {
              return lhs.raw_address < rhs.raw_address;
            });
  snapshots_needed.erase(
    std::unique(snapshots_needed.begin(), snapshots_needed.end()),
    snapshots_needed.end());
  // Snapshot all vectors before reacquiring any target lock.  The final
  // locked pass only needs its narrow liveness-boundary header checks; bulk
  // remote vector reads and alpha pruning stay outside the critical section.
  vec<NodeSnapshot> snapshots =
    read_node_snapshots_batched(
      snapshots_needed, config, "apply_local_reverse_updates_batched");
  dense_hashmap_t<u64, size_t> snapshot_index;
  snapshot_index.reserve(snapshots.size());
  for (size_t index = 0; index < snapshots.size(); ++index) {
    snapshot_index[snapshots[index].rptr.raw_address] = index;
  }

  vec<ScoredReverseCandidate> scored;
  vec<size_t> selected_indices;
  const auto robust_prune_cached = [&](const RemotePtr target,
                                       const vec<RemotePtr>& current_neighbors,
                                       const vec<RemotePtr>& fresh_candidates,
                                       vec<RemotePtr>& selected) {
    const auto target_vector_address =
      vamana::StorageLayoutResolver::vector(target);
    lib_assert(target_vector_address.offset + target_vector_address.size <=
                 mn_memory_bytes_,
               "batched reverse-update target vector exceeds shard bounds");
    const byte_t* target_vector =
      index_buffer_.get_full_buffer() + target_vector_address.offset;

    scored.clear();
    scored.reserve(current_neighbors.size() + fresh_candidates.size());
    const auto score = [&](const RemotePtr pointer) {
      if (pointer.is_null() || pointer == target ||
          std::find_if(scored.begin(), scored.end(),
                       [&](const ScoredReverseCandidate& candidate) {
                         return candidate.rptr == pointer;
                       }) != scored.end()) {
        return;
      }
      const auto iterator = snapshot_index.find(pointer.raw_address);
      if (iterator == snapshot_index.end()) {
        return;
      }
      const NodeSnapshot& snapshot = snapshots[iterator->second];
      if (snapshot.deleted ||
          (snapshot.header & VamanaNode::HEADER_PROVISIONAL) != 0 ||
          snapshot.vector_data.size() < VamanaNode::vector_bytes()) {
        return;
      }
      scored.push_back({
        pointer,
        snapshot.vector_data.data(),
        distance_between_vectors(
          target_vector, VamanaNode::vector_dtype(),
          snapshot.vector_data.data(), VamanaNode::vector_dtype(), config)});
    };
    for (const RemotePtr neighbor : current_neighbors) {
      score(neighbor);
    }
    for (const RemotePtr candidate : fresh_candidates) {
      score(candidate);
    }
    std::sort(scored.begin(), scored.end(),
              [](const ScoredReverseCandidate& lhs,
                 const ScoredReverseCandidate& rhs) {
                return lhs.distance < rhs.distance;
              });
    select_alpha_robust_pruned_sorted(
      span<const ScoredReverseCandidate>{scored.data(), scored.size()},
      config.R,
      config.alpha,
      [](const ScoredReverseCandidate& candidate) {
        return candidate.rptr;
      },
      [](const ScoredReverseCandidate& candidate) {
        return candidate.distance;
      },
      [&](const ScoredReverseCandidate& candidate,
          const ScoredReverseCandidate& retained) {
        return distance_between_vectors(
          candidate.vector, VamanaNode::vector_dtype(),
          retained.vector, VamanaNode::vector_dtype(), config);
      },
      selected,
      selected_indices);
  };

  // Compute the common-case result without holding target locks.  If a
  // candidate dies before the final write boundary, the locked pass below
  // recomputes from the cached snapshots after removing it.
  for (PendingReverseUpdate& update : pending) {
    robust_prune_cached(update.target,
                        update.current_neighbors,
                        update.candidates,
                        update.selected_neighbors);
  }

  for (PendingReverseUpdate& update : pending) {
    const IncarnationLockResult target_lock = try_lock_node(update.target);
    if (target_lock == IncarnationLockResult::stale) continue;
    if (target_lock == IncarnationLockResult::busy) return false;
    const u64 target_header = load_local_node_header_acquire(update.target);
    const bool target_deleted =
      (target_header & VamanaNode::HEADER_DELETED) != 0;
    const bool target_unavailable =
      !VamanaNode::stable_graph_mutation_allowed(target_header);
    if (target_deleted) {
      unlock_node(update.target);
      continue;
    }
    if (target_unavailable) {
      unlock_node(update.target);
      return false;
    }
    const vec<RemotePtr> observed_neighbors =
      read_stable_neighbor_list(update.target);
    if (!same_neighbors(observed_neighbors, update.current_neighbors)) {
      unlock_node(update.target);
      conflicted[update.target.raw_address] = std::move(update.candidates);
      continue;
    }

    vec<RemotePtr> fresh_candidates;
    select_fresh_reverse_candidates_locked(
      observed_neighbors, update.candidates,
      [this](const RemotePtr& candidate) {
        return storage_owner_node_stable(candidate);
      },
      fresh_candidates);
    if (fresh_candidates.empty()) {
      unlock_node(update.target);
      continue;
    }
    if (!same_neighbors(fresh_candidates, update.candidates)) {
      robust_prune_cached(update.target,
                          observed_neighbors,
                          fresh_candidates,
                          update.selected_neighbors);
    }
    write_neighbor_list(update.target, update.selected_neighbors);
    unlock_node(update.target);
  }

  bool success = true;
  for (const auto& [target_raw, candidates] : conflicted) {
    success &= apply_local_reverse_update(RemotePtr{target_raw}, candidates, config);
  }
  return success;
}
