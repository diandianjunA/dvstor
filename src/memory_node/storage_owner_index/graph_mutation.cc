#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/reverse_batch_policy.hh"
#include "memory_node/storage_owner_index/robust_prune_policy.hh"
#include "memory_node/storage_owner_index/stage1_reachability_policy.hh"

using namespace memory_node_storage_owner_index_detail;

vec<RemotePtr> MemoryNode::robust_prune_cpu(const byte_t* source,
                                            VectorDType source_dtype,
                                            const vec<RemotePtr>& candidates,
                                            const hashset_t<RemotePtr>& skip,
                                            const Configuration& config,
                                            InsertBreakdownCounters* breakdown,
                                            u32 result_limit_override) {
  StorageOwnerCoroutineScratch* scratch = current_storage_owner_thread_ != nullptr
                                            ? &current_storage_owner_thread_->coroutine_scratch_state()
                                            : nullptr;
  vec<StorageOwnerPruneCandidateInfo> local_infos;
  vec<RemotePtr> local_filtered;
  vec<RemotePtr> local_batch;
  vec<RemotePtr> local_selected;
  vec<size_t> local_selected_indices;
  if (scratch != nullptr) {
    scratch->clear_prune();
  }
  vec<StorageOwnerPruneCandidateInfo>& infos = scratch != nullptr ? scratch->prune_infos : local_infos;
  vec<RemotePtr>& filtered = scratch != nullptr ? scratch->filtered : local_filtered;
  vec<RemotePtr>& batch = scratch != nullptr ? scratch->batch : local_batch;
  vec<RemotePtr>& selected = scratch != nullptr ? scratch->selected : local_selected;
  vec<size_t>& selected_indices = scratch != nullptr
                                    ? scratch->prune_selected_indices
                                    : local_selected_indices;
  const u32 result_limit = result_limit_override == 0
                             ? config.R
                             : std::min(config.R, result_limit_override);
  infos.reserve(candidates.size());
  filtered.reserve(candidates.size());
  batch.reserve(storage_owner_snapshot_batch_size(config, current_storage_owner_thread_));
  selected.reserve(result_limit);

  for (const RemotePtr& candidate : candidates) {
    if (candidate.is_null() || skip.contains(candidate)) {
      continue;
    }
    filtered.push_back(candidate);
  }

  const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);
  for (size_t begin = 0; begin < filtered.size(); begin += snapshot_batch) {
    const size_t end = std::min(filtered.size(), begin + snapshot_batch);
    batch.clear();
    batch.insert(batch.end(), filtered.begin() + begin, filtered.begin() + end);
    auto t_snapshot = std::chrono::steady_clock::now();
    vec<NodeSnapshot> snapshots = read_node_snapshots_batched(
      batch, config, "robust_prune_cpu");
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_snapshot);
    }
    for (NodeSnapshot& snapshot : snapshots) {
      if (snapshot.deleted ||
          (snapshot.header & VamanaNode::HEADER_PROVISIONAL) != 0) {
        continue;
      }
      auto t_distance = std::chrono::steady_clock::now();
      const distance_t dist = distance_between_vectors(source, source_dtype,
                                                       snapshot.vector_data.data(), VamanaNode::vector_dtype(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_distance_ns += elapsed_ns_since(t_distance);
      }
      infos.push_back({snapshot.rptr, dist, std::move(snapshot.vector_data)});
    }
  }

  auto t_sort = std::chrono::steady_clock::now();
  std::sort(infos.begin(), infos.end(), [](const StorageOwnerPruneCandidateInfo& lhs,
                                           const StorageOwnerPruneCandidateInfo& rhs) {
    return lhs.dist < rhs.dist;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_prune_sort_ns += elapsed_ns_since(t_sort);
  }

  select_alpha_robust_pruned_sorted_by_pair_predicate(
    span<const StorageOwnerPruneCandidateInfo>{infos.data(), infos.size()},
    result_limit,
    [](const StorageOwnerPruneCandidateInfo& candidate) {
      return candidate.rptr;
    },
    [](const StorageOwnerPruneCandidateInfo& candidate) {
      return candidate.dist;
    },
    [&](const StorageOwnerPruneCandidateInfo& candidate,
        const StorageOwnerPruneCandidateInfo& retained,
        const distance_t source_distance) {
      auto t_pair_distance = std::chrono::steady_clock::now();
      const bool pruned = typed_l2_distance_alpha_leq_source(
        candidate.vector_data.data(), VamanaNode::vector_dtype(),
        retained.vector_data.data(), VamanaNode::vector_dtype(),
        config.dim, config.alpha, source_distance);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_pair_distance_ns += elapsed_ns_since(t_pair_distance);
      }
      return pruned;
    },
    selected,
    selected_indices);

  return selected;
}

vec<RemotePtr> MemoryNode::robust_prune_snapshots_cpu(
    const byte_t* source,
    VectorDType source_dtype,
    span<const NodeSnapshot> candidates,
    const hashset_t<RemotePtr>& skip,
    const Configuration& config,
    u32 result_limit_override) {
  thread_local vec<const NodeSnapshot*> references;
  references.clear();
  references.reserve(candidates.size());
  for (const NodeSnapshot& candidate : candidates) {
    references.push_back(&candidate);
  }
  return robust_prune_snapshot_refs_cpu(
    source, source_dtype, span<const NodeSnapshot* const>{references}, skip,
    config, result_limit_override);
}

vec<RemotePtr> MemoryNode::robust_prune_snapshot_refs_cpu(
    const byte_t* source,
    VectorDType source_dtype,
    span<const NodeSnapshot* const> candidates,
    const hashset_t<RemotePtr>& skip,
    const Configuration& config,
    u32 result_limit_override) {
  const u32 result_limit = result_limit_override == 0
                             ? config.R
                             : std::min(config.R, result_limit_override);
  StorageOwnerCoroutineScratch* scratch = current_storage_owner_thread_ != nullptr
                                            ? &current_storage_owner_thread_->coroutine_scratch_state()
                                            : nullptr;
  vec<StorageOwnerScoredSnapshot> local_scored;
  hashset_t<RemotePtr> local_seen;
  vec<RemotePtr> local_selected;
  vec<size_t> local_selected_indices;
  if (scratch != nullptr) {
    scratch->clear_prune();
  }
  vec<StorageOwnerScoredSnapshot>& scored = scratch != nullptr
                                              ? scratch->scored_snapshots
                                              : local_scored;
  hashset_t<RemotePtr>& seen = scratch != nullptr
                                ? scratch->prune_seen
                                : local_seen;
  vec<RemotePtr>& selected = scratch != nullptr
                               ? scratch->selected
                               : local_selected;
  vec<size_t>& selected_indices = scratch != nullptr
                                    ? scratch->prune_selected_indices
                                    : local_selected_indices;
  scored.reserve(candidates.size());
  seen.reserve(candidates.size());
  selected.reserve(result_limit);
  for (const NodeSnapshot* candidate_pointer : candidates) {
    if (candidate_pointer == nullptr) continue;
    const NodeSnapshot& candidate = *candidate_pointer;
    if (candidate.rptr.is_null() || candidate.deleted ||
        (candidate.header & VamanaNode::HEADER_PROVISIONAL) != 0 ||
        candidate.vector_data.size() < VamanaNode::vector_bytes() ||
        skip.contains(candidate.rptr) || !seen.insert(candidate.rptr).second) {
      continue;
    }
    scored.push_back({
      &candidate,
      distance_between_vectors(source,
                               source_dtype,
                               candidate.vector_data.data(),
                               VamanaNode::vector_dtype(),
                               config)});
  }
  std::sort(scored.begin(), scored.end(),
            [](const StorageOwnerScoredSnapshot& lhs,
               const StorageOwnerScoredSnapshot& rhs) {
              return lhs.distance < rhs.distance;
            });
  select_alpha_robust_pruned_sorted_by_pair_predicate(
    span<const StorageOwnerScoredSnapshot>{scored.data(), scored.size()},
    result_limit,
    [](const StorageOwnerScoredSnapshot& candidate) {
      return candidate.snapshot->rptr;
    },
    [](const StorageOwnerScoredSnapshot& candidate) {
      return candidate.distance;
    },
    [&](const StorageOwnerScoredSnapshot& candidate,
        const StorageOwnerScoredSnapshot& retained,
        const distance_t source_distance) {
      return typed_l2_distance_alpha_leq_source(
        candidate.snapshot->vector_data.data(),
        VamanaNode::vector_dtype(),
        retained.snapshot->vector_data.data(),
        VamanaNode::vector_dtype(),
        config.dim,
        config.alpha,
        source_distance);
    },
    selected,
    selected_indices);
  return selected;
}

bool MemoryNode::apply_partition_local_reverse_update(
    RemotePtr target_ptr,
    const vec<RemotePtr>& candidate_ptrs,
    const Configuration& config,
    bool* graph_changed) {
  if (graph_changed != nullptr) {
    *graph_changed = false;
  }
  lib_assert(local_shard(target_ptr.memory_node()),
             "partition-local reverse-update target must be local");
  if (candidate_ptrs.empty()) {
    return true;
  }

  vec<RemotePtr> unique_candidates;
  unique_candidates.reserve(candidate_ptrs.size());
  for (const RemotePtr& candidate : candidate_ptrs) {
    if (candidate.is_null()) {
      continue;
    }
    lib_assert(local_shard(candidate.memory_node()),
               "foreground reverse-update candidate must be partition-local");
    if (std::find(unique_candidates.begin(), unique_candidates.end(), candidate) ==
        unique_candidates.end()) {
      unique_candidates.push_back(candidate);
    }
  }
  if (unique_candidates.empty()) {
    return true;
  }

  const IncarnationLockResult target_lock = try_lock_node(target_ptr);
  if (target_lock == IncarnationLockResult::stale) {
    // A reverse edge into an incarnation that no longer exists is already an
    // idempotent no-op. Never redirect it into the replacement occupant.
    return true;
  }
  if (target_lock == IncarnationLockResult::busy) return false;
  if (!VamanaNode::stable_graph_mutation_allowed(
        load_local_node_header_acquire(target_ptr))) {
    unlock_node(target_ptr);
    return false;
  }

  vec<RemotePtr> current_neighbors =
    read_stable_neighbor_list(target_ptr);
  vec<RemotePtr> preserved_external;
  vec<RemotePtr> local_candidates;
  preserved_external.reserve(current_neighbors.size());
  local_candidates.reserve(current_neighbors.size() + unique_candidates.size());
  bool changed = false;
  for (const RemotePtr& neighbor : current_neighbors) {
    if (neighbor.is_null()) {
      changed = true;
      continue;
    }
    if (!storage_owner_node_stable(neighbor)) {
      changed = true;
      continue;
    }
    if (local_shard(neighbor.memory_node())) {
      local_candidates.push_back(neighbor);
    } else {
      preserved_external.push_back(neighbor);
    }
  }

  for (const RemotePtr& candidate : unique_candidates) {
    if (!storage_owner_node_stable(candidate)) {
      continue;
    }
    if (std::find(local_candidates.begin(), local_candidates.end(), candidate) ==
        local_candidates.end()) {
      local_candidates.push_back(candidate);
      changed = true;
    }
  }
  if (!changed) {
    unlock_node(target_ptr);
    return true;
  }

  const u32 local_capacity = preserved_external.size() >= config.R
                               ? 0
                               : config.R - static_cast<u32>(preserved_external.size());
  vec<RemotePtr> selected_local;
  if (local_candidates.size() <= local_capacity) {
    selected_local = std::move(local_candidates);
  } else if (local_capacity > 0) {
    const auto target_vector_addr = vamana::StorageLayoutResolver::vector(target_ptr);
    lib_assert(target_vector_addr.offset + target_vector_addr.size <= mn_memory_bytes_,
               "partition-local reverse-update target vector exceeds shard bounds");
    const byte_t* target_vector =
      index_buffer_.get_full_buffer() + target_vector_addr.offset;
    hashset_t<RemotePtr> skip;
    selected_local = robust_prune_cpu(target_vector,
                                      VamanaNode::vector_dtype(),
                                      local_candidates,
                                      skip,
                                      config,
                                      nullptr,
                                      local_capacity);
  }

  vec<RemotePtr> updated_neighbors;
  updated_neighbors.reserve(preserved_external.size() + selected_local.size());
  updated_neighbors.insert(updated_neighbors.end(),
                           preserved_external.begin(),
                           preserved_external.end());
  updated_neighbors.insert(updated_neighbors.end(),
                           selected_local.begin(),
                           selected_local.end());
  lib_assert(updated_neighbors.size() <= config.R,
             "partition-local reverse-update exceeded graph degree");
  const bool changed_neighbors =
    updated_neighbors.size() != current_neighbors.size() ||
    !std::equal(updated_neighbors.begin(), updated_neighbors.end(),
                current_neighbors.begin());
  if (changed_neighbors) {
    write_neighbor_list(target_ptr, updated_neighbors);
  }
  unlock_node(target_ptr);
  if (graph_changed != nullptr) {
    *graph_changed = changed_neighbors;
  }
  return true;
}

vec<RemotePtr> MemoryNode::install_local_provisional_backlinks(
    RemotePtr candidate,
    span<const RemotePtr> targets) {
  if (candidate.is_null() || !local_shard(candidate.memory_node()) ||
      !storage_owner_node_live(candidate)) {
    return {};
  }

  // A fresh insert cannot be reported as a permanent mutation failure merely
  // because maintenance held every eligible parent lock or all eligible
  // parents temporarily used their bounded provisional slots.  Stage2 frees
  // those slots when it promotes/removes the corresponding reachability
  // bridges, so both conditions are ordinary backpressure.  Keep retrying
  // while at least one live target is busy or capacity-blocked.  A sweep with
  // neither remains terminal (all candidates are stale, remote, deleted, or
  // otherwise permanently ineligible), so this cannot spin on an impossible
  // graph.
  const auto try_install = [&](const RemotePtr target) {
      using InstallDisposition =
        memory_node_storage_owner_index_detail::
          Stage1BridgeInstallDisposition;
      if (!local_shard(target.memory_node())) {
        return InstallDisposition::rejected;
      }
      const IncarnationLockResult lock = try_lock_node(target);
      if (lock == IncarnationLockResult::busy) {
        return InstallDisposition::busy;
      }
      if (lock != IncarnationLockResult::locked) {
        // A stale target is not eligible for an ACK under its old physical
        // identity.  Another target in the same sweep may still be usable.
        return InstallDisposition::rejected;
      }
      const u64 header = load_local_node_header_acquire(target);
      if (!VamanaNode::stable_graph_mutation_allowed(header)) {
        unlock_node(target);
        return InstallDisposition::rejected;
      }

      GraphAdjacency adjacency;
      if (!read_graph_adjacency(target, adjacency) || adjacency.deleted) {
        unlock_node(target);
        return InstallDisposition::rejected;
      }
      if (std::find(adjacency.stable.begin(), adjacency.stable.end(),
                    candidate) != adjacency.stable.end() ||
          std::find(adjacency.provisional.begin(),
                    adjacency.provisional.end(), candidate) !=
            adjacency.provisional.end()) {
        unlock_node(target);
        return InstallDisposition::installed;
      }
      if (adjacency.provisional.size() >=
          VamanaNode::provisional_slots()) {
        unlock_node(target);
        return InstallDisposition::busy;
      }

      adjacency.provisional.push_back(candidate);
      write_graph_adjacency(target, adjacency.stable,
                            adjacency.provisional,
                            adjacency.generation, false);
      unlock_node(target);
      return InstallDisposition::installed;
  };
  return select_stage1_reachability_bridges_retry_busy(
    candidate, targets, VamanaNode::allocation_size(), try_install,
    [&]() {
      if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
        return false;
      }
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::microseconds(100));
      return !storage_insert_shutdown_.load(std::memory_order_acquire) &&
        !storage_owner_maintenance_shutdown_.load(
          std::memory_order_acquire);
    });
}

bool MemoryNode::remove_local_provisional_backlinks(
    RemotePtr candidate,
    span<const RemotePtr> targets) {
  if (candidate.is_null() || candidate.memory_node() != storage_id_) {
    return false;
  }
  hashset_t<RemotePtr> visited;
  visited.reserve(targets.size());
  for (const RemotePtr target : targets) {
    if (target.is_null() || target.memory_node() != storage_id_ ||
        !visited.insert(target).second) {
      continue;
    }
    const IncarnationLockResult target_lock = try_lock_node(target);
    if (target_lock == IncarnationLockResult::stale) {
      // The old parent is gone, so its old-incarnation provisional edge is
      // gone as well. This is the required idempotent removal postcondition.
      continue;
    }
    if (target_lock == IncarnationLockResult::busy) return false;
    const u64 target_header = load_local_node_header_acquire(target);
    if ((target_header & VamanaNode::HEADER_RETIRING) != 0) {
      // The retiring parent's cleanup snapshot already owns this protected
      // plane. Its eventual tombstone makes removal here an idempotent no-op.
      unlock_node(target);
      continue;
    }
    if (VamanaNode::graph_mutation_quiesced(target_header)) {
      unlock_node(target);
      return false;
    }
    GraphAdjacency adjacency;
    if (!read_graph_adjacency(target, adjacency)) {
      unlock_node(target);
      return false;
    }
    const size_t old_size = adjacency.provisional.size();
    adjacency.provisional.erase(
      std::remove(adjacency.provisional.begin(),
                  adjacency.provisional.end(), candidate),
      adjacency.provisional.end());
    if (adjacency.provisional.size() != old_size) {
      write_graph_adjacency(target, adjacency.stable,
                            adjacency.provisional,
                            adjacency.generation, adjacency.deleted);
    }
    unlock_node(target);
  }
  return true;
}

bool MemoryNode::apply_local_reverse_update(RemotePtr target_ptr,
                                const vec<RemotePtr>& candidate_ptrs,
                                const Configuration& config,
                                bool enqueue_maintenance) {
  lib_assert(local_shard(target_ptr.memory_node()), "target reverse update must be local");
  if (candidate_ptrs.empty()) {
    return true;
  }

  const auto update_started = std::chrono::steady_clock::now();
  const auto target_vector_addr = vamana::StorageLayoutResolver::vector(target_ptr);
  lib_assert(target_vector_addr.offset + target_vector_addr.size <= mn_memory_bytes_,
             "local reverse-update target vector exceeds shard bounds");
  const byte_t* target_vector = index_buffer_.get_full_buffer() + target_vector_addr.offset;

  vec<RemotePtr> unique_candidates;
  unique_candidates.reserve(candidate_ptrs.size());
  for (const RemotePtr& candidate_ptr : candidate_ptrs) {
    if (!candidate_ptr.is_null() &&
        std::find(unique_candidates.begin(), unique_candidates.end(), candidate_ptr) == unique_candidates.end()) {
      unique_candidates.push_back(candidate_ptr);
    }
  }
  if (unique_candidates.empty()) {
    return true;
  }

  const auto target_unavailable = [&]() {
    return !VamanaNode::stable_graph_mutation_allowed(
      load_local_node_header_acquire(target_ptr));
  };
  (void)enqueue_maintenance;

  vec<RemotePtr> current_neighbors;
  vec<RemotePtr> fresh_candidates;
  vec<RemotePtr> revalidated_candidates;
  vec<RemotePtr> prune_candidates;
  u32 conflicts = 0;

  for (;;) {
    const IncarnationLockResult first_lock = try_lock_node(target_ptr);
    if (first_lock == IncarnationLockResult::stale) return true;
    if (first_lock == IncarnationLockResult::busy) return false;
    if (target_unavailable()) {
      unlock_node(target_ptr);
      return false;
    }
    current_neighbors = read_stable_neighbor_list(target_ptr);
    select_fresh_reverse_candidates_locked(
      current_neighbors,
      unique_candidates,
      [this](const RemotePtr& candidate) {
        return storage_owner_node_stable(candidate);
      },
      fresh_candidates);
    if (fresh_candidates.empty()) {
      unlock_node(target_ptr);
      return true;
    }
    if (current_neighbors.size() + fresh_candidates.size() <= config.R) {
      current_neighbors.insert(current_neighbors.end(),
                               fresh_candidates.begin(),
                               fresh_candidates.end());
      write_neighbor_list(target_ptr, current_neighbors);
      unlock_node(target_ptr);
      return true;
    }
    unlock_node(target_ptr);

    // The overflow path snapshots the complete current U fresh set outside
    // the target lock, then applies the same alpha RobustPrune used for a new
    // node.  The final locked compare/revalidation below makes this optimistic
    // calculation safe under concurrent reverse updates and deletes.
    prune_candidates = current_neighbors;
    for (const RemotePtr candidate : fresh_candidates) {
      if (std::find(prune_candidates.begin(), prune_candidates.end(),
                    candidate) == prune_candidates.end()) {
        prune_candidates.push_back(candidate);
      }
    }
    vec<NodeSnapshot> snapshots =
      read_node_snapshots_batched(
        prune_candidates, config, "apply_local_reverse_update");
    hashset_t<RemotePtr> skip;
    skip.insert(target_ptr);
    vec<RemotePtr> selected_neighbors = robust_prune_snapshots_cpu(
      target_vector,
      VamanaNode::vector_dtype(),
      span<const NodeSnapshot>{snapshots.data(), snapshots.size()},
      skip,
      config,
      config.R);

    const IncarnationLockResult final_lock = try_lock_node(target_ptr);
    if (final_lock == IncarnationLockResult::stale) return true;
    if (final_lock == IncarnationLockResult::busy) return false;
    if (target_unavailable()) {
      unlock_node(target_ptr);
      return false;
    }
    const vec<RemotePtr> observed_neighbors =
      read_stable_neighbor_list(target_ptr);
    const bool unchanged =
      observed_neighbors.size() == current_neighbors.size() &&
      std::equal(observed_neighbors.begin(), observed_neighbors.end(),
                 current_neighbors.begin());
    if (!unchanged) {
      ++conflicts;
      unlock_node(target_ptr);
      continue;
    }
    select_fresh_reverse_candidates_locked(
      observed_neighbors,
      fresh_candidates,
      [this](const RemotePtr& candidate) {
        return storage_owner_node_stable(candidate);
      },
      revalidated_candidates);
    const bool candidates_unchanged =
      revalidated_candidates.size() == fresh_candidates.size() &&
      std::equal(revalidated_candidates.begin(),
                 revalidated_candidates.end(),
                 fresh_candidates.begin());
    if (!candidates_unchanged) {
      ++conflicts;
      unlock_node(target_ptr);
      continue;
    }
    write_neighbor_list(target_ptr, selected_neighbors);
    unlock_node(target_ptr);

    const u64 update_ns = elapsed_ns_since(update_started);
    if (update_ns > 1000ull * 1000ull * 1000ull) {
      static std::atomic<u32> slow_update_logs{0};
      const u32 log_index =
        slow_update_logs.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 16) {
        std::cerr << "[storage-owner] slow reverse-update target"
                  << " self_shard=" << storage_id_
                  << " target_raw=" << target_ptr.raw_address
                  << " candidates=" << candidate_ptrs.size()
                  << " current_neighbors=" << current_neighbors.size()
                  << " filtered_candidates=" << fresh_candidates.size()
                  << " conflicts=" << conflicts
                  << " elapsed_ms=" << (update_ns / 1000000.0)
                  << std::endl;
      }
    }
    return true;
  }
}
