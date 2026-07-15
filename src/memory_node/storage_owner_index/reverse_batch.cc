#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/reverse_batch_policy.hh"

using memory_node_storage_owner_index_detail::
  select_fresh_reverse_candidates_locked;

namespace {

struct PendingReverseUpdate {
  RemotePtr target;
  vec<RemotePtr> candidates;
  vec<RemotePtr> current_neighbors;
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
    const bool live = storage_owner_node_live(candidate);
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
  vec<RemotePtr> remote_snapshots_needed;
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

    lock_node(target);
    const bool target_deleted =
      (load_local_node_header_acquire(target) &
       VamanaNode::HEADER_DELETED) != 0;
    if (target_deleted) {
      unlock_node(target);
      continue;
    }

    vec<RemotePtr> current_neighbors = read_neighbor_list(target);
    vec<RemotePtr> filtered_candidates;
    select_fresh_reverse_candidates_locked(
      current_neighbors, unique_candidates,
      [this](const RemotePtr& candidate) {
        return storage_owner_node_live(candidate);
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
      if (!neighbor.is_null() && !local_shard(neighbor.memory_node())) {
        remote_snapshots_needed.push_back(neighbor);
      }
    }
    for (const RemotePtr& candidate : update.candidates) {
      if (!candidate.is_null() && !local_shard(candidate.memory_node())) {
        remote_snapshots_needed.push_back(candidate);
      }
    }
    pending.push_back(std::move(update));
  }

  if (pending.empty()) {
    return true;
  }

  std::sort(remote_snapshots_needed.begin(), remote_snapshots_needed.end(),
            [](const RemotePtr& lhs, const RemotePtr& rhs) {
              return lhs.raw_address < rhs.raw_address;
            });
  remote_snapshots_needed.erase(
    std::unique(remote_snapshots_needed.begin(), remote_snapshots_needed.end()),
    remote_snapshots_needed.end());
  vec<NodeSnapshot> remote_snapshots =
    read_node_snapshots_batched(remote_snapshots_needed, config);
  dense_hashmap_t<u64, size_t> snapshot_index;
  snapshot_index.reserve(remote_snapshots.size());
  for (size_t index = 0; index < remote_snapshots.size(); ++index) {
    snapshot_index[remote_snapshots[index].rptr.raw_address] = index;
  }

  auto vector_for = [&](const RemotePtr& pointer) -> const byte_t* {
    if (local_shard(pointer.memory_node())) {
      if (!storage_owner_node_live(pointer)) {
        return nullptr;
      }
      const auto address = vamana::StorageLayoutResolver::vector(pointer);
      lib_assert(address.offset + address.size <= mn_memory_bytes_,
                 "batched reverse-update local vector exceeds shard bounds");
      return index_buffer_.get_full_buffer() + address.offset;
    }
    const auto iterator = snapshot_index.find(pointer.raw_address);
    if (iterator == snapshot_index.end()) {
      return nullptr;
    }
    const NodeSnapshot& snapshot = remote_snapshots[iterator->second];
    return snapshot.deleted ? nullptr : snapshot.vector_data.data();
  };

  for (PendingReverseUpdate& update : pending) {
    lock_node(update.target);
    const bool target_deleted =
      (load_local_node_header_acquire(update.target) &
       VamanaNode::HEADER_DELETED) != 0;
    if (target_deleted) {
      unlock_node(update.target);
      continue;
    }
    const vec<RemotePtr> observed_neighbors = read_neighbor_list(update.target);
    if (!same_neighbors(observed_neighbors, update.current_neighbors)) {
      unlock_node(update.target);
      conflicted[update.target.raw_address] = std::move(update.candidates);
      continue;
    }

    vec<RemotePtr> fresh_candidates;
    select_fresh_reverse_candidates_locked(
      observed_neighbors, update.candidates,
      [this](const RemotePtr& candidate) {
        return storage_owner_node_live(candidate);
      },
      fresh_candidates);
    if (fresh_candidates.empty()) {
      unlock_node(update.target);
      continue;
    }

    const auto target_vector_address =
      vamana::StorageLayoutResolver::vector(update.target);
    lib_assert(target_vector_address.offset + target_vector_address.size <= mn_memory_bytes_,
               "batched reverse-update target vector exceeds shard bounds");
    const byte_t* target_vector =
      index_buffer_.get_full_buffer() + target_vector_address.offset;
    vec<RemotePtr> selected;
    vec<distance_t> selected_distances;
    selected.reserve(config.R);
    selected_distances.reserve(config.R);

    auto retain_nearest = [&](const RemotePtr& pointer) {
      const byte_t* vector = vector_for(pointer);
      if (pointer.is_null() || vector == nullptr) {
        return;
      }
      const distance_t distance = distance_between_vectors(
        target_vector,
        VamanaNode::vector_dtype(),
        vector,
        VamanaNode::vector_dtype(),
        config);
      if (selected.size() < config.R) {
        selected.push_back(pointer);
        selected_distances.push_back(distance);
        return;
      }
      size_t farthest_index = 0;
      for (size_t index = 1; index < selected_distances.size(); ++index) {
        if (selected_distances[index] > selected_distances[farthest_index]) {
          farthest_index = index;
        }
      }
      if (distance < selected_distances[farthest_index]) {
        selected[farthest_index] = pointer;
        selected_distances[farthest_index] = distance;
      }
    };

    for (const RemotePtr& neighbor : update.current_neighbors) {
      retain_nearest(neighbor);
    }
    for (const RemotePtr& candidate : fresh_candidates) {
      retain_nearest(candidate);
    }
    write_neighbor_list(update.target, selected);
    unlock_node(update.target);
  }

  bool success = true;
  for (const auto& [target_raw, candidates] : conflicted) {
    success &= apply_local_reverse_update(RemotePtr{target_raw}, candidates, config);
  }
  return success;
}
