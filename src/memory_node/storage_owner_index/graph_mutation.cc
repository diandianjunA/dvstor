#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/reverse_batch_policy.hh"
#include "memory_node/storage_owner_index/robust_prune_policy.hh"

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
    vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_snapshot);
    }
    for (NodeSnapshot& snapshot : snapshots) {
      if (snapshot.deleted) {
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

  select_alpha_robust_pruned_sorted(
    span<const StorageOwnerPruneCandidateInfo>{infos.data(), infos.size()},
    result_limit,
    config.alpha,
    [](const StorageOwnerPruneCandidateInfo& candidate) {
      return candidate.rptr;
    },
    [](const StorageOwnerPruneCandidateInfo& candidate) {
      return candidate.dist;
    },
    [&](const StorageOwnerPruneCandidateInfo& candidate,
        const StorageOwnerPruneCandidateInfo& retained) {
      auto t_pair_distance = std::chrono::steady_clock::now();
      const distance_t pair_dist = distance_between_vectors(
        candidate.vector_data.data(), VamanaNode::vector_dtype(),
        retained.vector_data.data(), VamanaNode::vector_dtype(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_pair_distance_ns += elapsed_ns_since(t_pair_distance);
      }
      return pair_dist;
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
  for (const NodeSnapshot& candidate : candidates) {
    if (candidate.rptr.is_null() || candidate.deleted ||
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
  select_alpha_robust_pruned_sorted(
    span<const StorageOwnerScoredSnapshot>{scored.data(), scored.size()},
    result_limit,
    config.alpha,
    [](const StorageOwnerScoredSnapshot& candidate) {
      return candidate.snapshot->rptr;
    },
    [](const StorageOwnerScoredSnapshot& candidate) {
      return candidate.distance;
    },
    [&](const StorageOwnerScoredSnapshot& candidate,
        const StorageOwnerScoredSnapshot& retained) {
      return distance_between_vectors(
        candidate.snapshot->vector_data.data(),
        VamanaNode::vector_dtype(),
        retained.snapshot->vector_data.data(),
        VamanaNode::vector_dtype(),
        config);
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

  lock_node(target_ptr);
  if ((load_local_node_header_acquire(target_ptr) &
       VamanaNode::HEADER_DELETED) != 0) {
    unlock_node(target_ptr);
    return true;
  }

  vec<RemotePtr> current_neighbors = read_neighbor_list(target_ptr);
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
    if (!storage_owner_node_live(neighbor)) {
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
    if (!storage_owner_node_live(candidate)) {
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

auto MemoryNode::execute_storage_owner_insert_job_async(StorageOwnerThread& thread,
                                            StorageOwnerInsertJob& job,
                                            dense_hashmap_t<u64, vec<RemotePtr>>& local_updates,
                                            dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>>& remote_updates,
                                            InsertBreakdownCounters& breakdown,
                                            const Configuration& config) -> StorageOwnerInsertCoroutine {
  const auto components = span<const element_t>{reinterpret_cast<const element_t*>(job.vector_data.data()),
                                                 VamanaNode::DIM};
  FreshnessEntry old_entry{};
  u32 generation = 0;
  const auto status = prepare_mutation(job.id, job.kind, &old_entry, &generation);
  job.old_ptr = old_entry.current;
  job.generation = generation;
  const bool maintenance_enabled = storage_owner_maintenance_enabled(config);
  if (status != service::storage_owner::MutationStatus::ok) {
    complete_storage_owner_maintenance_sequence(
      job.maintenance_sequence, job.reserved_maintenance_work);
    job.status = status;
    job.ok = false;
    co_return;
  }
  if (job.kind == service::storage_owner::MutationKind::erase) {
    job.ok = mark_node_deleted(old_entry.current, generation);
    job.status = job.ok ? service::storage_owner::MutationStatus::ok
                        : service::storage_owner::MutationStatus::failed;
    if (job.ok) {
      publish_mutation(job.id, old_entry.current, generation, true);
      job.maintenance_sequence = schedule_storage_owner_maintenance(
        job.id, generation, job.kind, RemotePtr{}, old_entry.current,
        job.maintenance_sequence, job.reserved_maintenance_work, config);
    } else {
      complete_storage_owner_maintenance_sequence(
        job.maintenance_sequence, job.reserved_maintenance_work);
    }
    co_return;
  }
  lib_assert(!local_stitch_enabled(config),
             "local stage1 must run on its dedicated CPU executor");
  RemotePtr medoid_ptr{};
  const vec<RemotePtr>* candidates = nullptr;

  auto t_medoid = std::chrono::steady_clock::now();
  medoid_ptr = co_await async_read_global_medoid(thread);
  breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
  if (medoid_ptr.is_null()) {
    const RemotePtr new_ptr = allocate_local_node();
    job.new_ptr = new_ptr;
    auto t_write = std::chrono::steady_clock::now();
    write_new_node(new_ptr, job.id, components, {}, generation);
    breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
    RemotePtr observed;
    if (try_set_global_medoid(RemotePtr{}, new_ptr, observed) || observed.is_null()) {
      job.ok = true;
      job.status = service::storage_owner::MutationStatus::ok;
      if (job.kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
        mark_node_deleted(old_entry.current, old_entry.generation);
      }
      publish_mutation(job.id, new_ptr, generation, false);
      job.maintenance_sequence = schedule_storage_owner_maintenance(
        job.id, generation, job.kind, new_ptr, old_entry.current,
        job.maintenance_sequence, job.reserved_maintenance_work, config);
      co_return;
    }
    medoid_ptr = observed;
  }

  auto t_search = std::chrono::steady_clock::now();
  auto search = beam_search_candidates_async(
    components, medoid_ptr, config, thread, &breakdown);
  co_await std::suspend_always{};
  while (!search.handle.done()) {
    if (thread.is_ready(thread.running_coroutine)) {
      search.handle.resume();
    } else {
      co_await std::suspend_always{};
    }
  }
  search.handle.destroy();
  breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);
  candidates = &storage_owner_async_candidates_[thread.id][thread.running_coroutine];

  lib_assert(candidates != nullptr, "storage-owner insert search produced no candidate set");
  StorageOwnerCoroutineScratch& scratch = thread.coroutine_scratch_state();
  scratch.empty_skip.clear();
  auto t_prune = std::chrono::steady_clock::now();
  vec<RemotePtr> selected_neighbors = robust_prune_cpu(reinterpret_cast<const byte_t*>(components.data()),
                                                       VectorDType::float32, *candidates, scratch.empty_skip, config, &breakdown);
  breakdown.storage_owner_prune_ns += elapsed_ns_since(t_prune);
  const RemotePtr new_ptr = allocate_local_node();
  job.new_ptr = new_ptr;
  auto t_write = std::chrono::steady_clock::now();
  write_new_node(new_ptr, job.id, components, selected_neighbors, generation);
  breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
  if (job.kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
    mark_node_deleted(old_entry.current, old_entry.generation);
  }
  publish_mutation(job.id, new_ptr, generation, false);
  job.maintenance_sequence = schedule_storage_owner_maintenance(
    job.id, generation, job.kind, new_ptr, old_entry.current,
    job.maintenance_sequence, job.reserved_maintenance_work, config);

  if (!maintenance_enabled) {
    for (const RemotePtr& neighbor_ptr : selected_neighbors) {
      if (local_shard(neighbor_ptr.memory_node())) {
        local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
        job.invalidated_neighbors.push_back(neighbor_ptr.raw_address);
      } else {
        remote_updates[neighbor_ptr.memory_node()].push_back(
          service::storage_owner::ReverseUpdateOp{neighbor_ptr.raw_address, new_ptr.raw_address});
        job.invalidated_neighbors.push_back(neighbor_ptr.raw_address);
      }
    }
  }
  job.ok = true;
  job.status = service::storage_owner::MutationStatus::ok;
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

  const auto target_deleted = [&]() {
    return (load_local_node_header_acquire(target_ptr) &
            VamanaNode::HEADER_DELETED) != 0;
  };
  (void)enqueue_maintenance;

  vec<RemotePtr> current_neighbors;
  vec<RemotePtr> fresh_candidates;
  vec<RemotePtr> revalidated_candidates;
  vec<RemotePtr> prune_candidates;
  u32 conflicts = 0;

  for (;;) {
    lock_node(target_ptr);
    if (target_deleted()) {
      unlock_node(target_ptr);
      return true;
    }
    current_neighbors = read_neighbor_list(target_ptr);
    select_fresh_reverse_candidates_locked(
      current_neighbors,
      unique_candidates,
      [this](const RemotePtr& candidate) {
        return storage_owner_node_live(candidate);
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
      read_node_snapshots_batched(prune_candidates, config);
    hashset_t<RemotePtr> skip;
    skip.insert(target_ptr);
    vec<RemotePtr> selected_neighbors = robust_prune_snapshots_cpu(
      target_vector,
      VamanaNode::vector_dtype(),
      span<const NodeSnapshot>{snapshots.data(), snapshots.size()},
      skip,
      config,
      config.R);

    lock_node(target_ptr);
    if (target_deleted()) {
      unlock_node(target_ptr);
      return true;
    }
    const vec<RemotePtr> observed_neighbors = read_neighbor_list(target_ptr);
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
        return storage_owner_node_live(candidate);
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
