#include "memory_node/storage_owner_index/detail.hh"

using namespace memory_node_storage_owner_index_detail;

vec<RemotePtr> MemoryNode::robust_prune_cpu(const byte_t* source,
                                            VectorDType source_dtype,
                                            const vec<RemotePtr>& candidates,
                                            const hashset_t<RemotePtr>& skip,
                                            const Configuration& config,
                                            InsertBreakdownCounters* breakdown,
                                            u32 candidate_limit_override,
                                            u32 result_limit_override) {
  StorageOwnerCoroutineScratch* scratch = current_storage_owner_thread_ != nullptr
                                            ? &current_storage_owner_thread_->coroutine_scratch_state()
                                            : nullptr;
  vec<StorageOwnerPruneCandidateInfo> local_infos;
  vec<RemotePtr> local_filtered;
  vec<RemotePtr> local_batch;
  vec<RemotePtr> local_selected;
  vec<const byte_t*> local_selected_vectors;
  if (scratch != nullptr) {
    scratch->clear_prune();
  }
  vec<StorageOwnerPruneCandidateInfo>& infos = scratch != nullptr ? scratch->prune_infos : local_infos;
  vec<RemotePtr>& filtered = scratch != nullptr ? scratch->filtered : local_filtered;
  vec<RemotePtr>& batch = scratch != nullptr ? scratch->batch : local_batch;
  vec<RemotePtr>& selected = scratch != nullptr ? scratch->selected : local_selected;
  vec<const byte_t*>& selected_vectors = scratch != nullptr ? scratch->selected_vectors : local_selected_vectors;
  const u32 prune_candidate_limit = candidate_limit_override == 0
                                      ? storage_owner_prune_candidate_limit(config)
                                      : std::max(config.R, candidate_limit_override);
  const u32 result_limit = result_limit_override == 0
                             ? config.R
                             : std::min(config.R, result_limit_override);
  infos.reserve(candidates.size());
  filtered.reserve(std::min<size_t>(candidates.size(), prune_candidate_limit));
  batch.reserve(storage_owner_snapshot_batch_size(config, current_storage_owner_thread_));
  selected.reserve(result_limit);
  selected_vectors.reserve(result_limit);

  for (const RemotePtr& candidate : candidates) {
    if (candidate.is_null() || skip.contains(candidate)) {
      continue;
    }
    filtered.push_back(candidate);
    if (filtered.size() >= prune_candidate_limit) {
      break;
    }
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

  for (const auto& candidate : infos) {
    if (selected.size() >= result_limit) {
      break;
    }

    bool pruned = false;
    for (idx_t i = 0; i < selected_vectors.size(); ++i) {
      auto t_pair_distance = std::chrono::steady_clock::now();
      const distance_t pair_dist = distance_between_vectors(candidate.vector_data.data(), VamanaNode::vector_dtype(),
                                                           selected_vectors[i], VamanaNode::vector_dtype(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_pair_distance_ns += elapsed_ns_since(t_pair_distance);
      }
      if (config.alpha * pair_dist <= candidate.dist) {
        pruned = true;
        break;
      }
    }

    if (!pruned) {
      selected.push_back(candidate.rptr);
      selected_vectors.push_back(candidate.vector_data.data());
    }
  }

  return selected;
}

vec<RemotePtr> MemoryNode::robust_prune_snapshots_cpu(
    const byte_t* source,
    VectorDType source_dtype,
    const vec<NodeSnapshot>& candidates,
    const hashset_t<RemotePtr>& skip,
    const Configuration& config,
    u32 result_limit_override) {
  struct ScoredSnapshot {
    const NodeSnapshot* snapshot{};
    distance_t distance{};
  };

  const u32 result_limit = result_limit_override == 0
                             ? config.R
                             : std::min(config.R, result_limit_override);
  vec<ScoredSnapshot> scored;
  scored.reserve(candidates.size());
  hashset_t<RemotePtr> seen;
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
            [](const ScoredSnapshot& lhs, const ScoredSnapshot& rhs) {
              return lhs.distance < rhs.distance;
            });

  vec<RemotePtr> selected;
  vec<const byte_t*> selected_vectors;
  selected.reserve(result_limit);
  selected_vectors.reserve(result_limit);
  for (const ScoredSnapshot& candidate : scored) {
    if (selected.size() >= result_limit) {
      break;
    }
    bool pruned = false;
    for (const byte_t* selected_vector : selected_vectors) {
      const distance_t pair_distance = distance_between_vectors(
        candidate.snapshot->vector_data.data(),
        VamanaNode::vector_dtype(),
        selected_vector,
        VamanaNode::vector_dtype(),
        config);
      if (config.alpha * pair_distance <= candidate.distance) {
        pruned = true;
        break;
      }
    }
    if (!pruned) {
      selected.push_back(candidate.snapshot->rptr);
      selected_vectors.push_back(candidate.snapshot->vector_data.data());
    }
  }
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
  const byte_t* target_node = local_node_ptr(target_ptr);
  if ((*reinterpret_cast<const u64*>(target_node) & VamanaNode::HEADER_DELETED) != 0) {
    unlock_node(target_ptr);
    return true;
  }

  vec<RemotePtr> current_neighbors = read_neighbor_list(target_ptr);
  vec<RemotePtr> preserved_external;
  vec<RemotePtr> local_candidates;
  preserved_external.reserve(current_neighbors.size());
  local_candidates.reserve(current_neighbors.size() + unique_candidates.size());
  for (const RemotePtr& neighbor : current_neighbors) {
    if (neighbor.is_null()) {
      continue;
    }
    if (local_shard(neighbor.memory_node())) {
      local_candidates.push_back(neighbor);
    } else {
      preserved_external.push_back(neighbor);
    }
  }

  bool changed = false;
  for (const RemotePtr& candidate : unique_candidates) {
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
                                      static_cast<u32>(local_candidates.size()),
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
    job.status = status;
    job.ok = false;
    co_return;
  }
  if (job.kind == service::storage_owner::MutationKind::erase) {
    job.ok = mark_node_deleted(old_entry.current, old_entry.generation);
    job.status = job.ok ? service::storage_owner::MutationStatus::ok
                        : service::storage_owner::MutationStatus::failed;
    if (job.ok) {
      publish_mutation(job.id, old_entry.current, old_entry.generation, true);
      job.maintenance_sequence = schedule_storage_owner_maintenance(
        job.id, old_entry.generation, job.kind, RemotePtr{}, old_entry.current, config);
    }
    co_return;
  }
  const bool local_stitch = local_stitch_enabled(config);
  const bool use_anchors = anchor_update_enabled(config, job.anchor_hints);
  RemotePtr medoid_ptr{};
  bool medoid_loaded = false;
  const vec<RemotePtr>* candidates = nullptr;

  if (use_anchors) {
    auto t_search = std::chrono::steady_clock::now();
    auto search = anchor_search_candidates_async(components, job.anchor_hints, config, thread,
                                                 &breakdown, local_stitch);
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
  }

  if (!use_anchors) {
    auto t_medoid = std::chrono::steady_clock::now();
    medoid_ptr = co_await async_read_global_medoid(thread);
    medoid_loaded = true;
    breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
  }
  if (medoid_loaded && medoid_ptr.is_null()) {
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
        job.id, generation, job.kind, new_ptr, old_entry.current, config);
      co_return;
    }
    medoid_ptr = observed;
  }

  if (!use_anchors) {
    auto t_search = std::chrono::steady_clock::now();
    auto search = beam_search_candidates_async(components, medoid_ptr, config, thread, &breakdown);
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
  }

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
    job.id, generation, job.kind, new_ptr, old_entry.current, config);

  if (!maintenance_enabled || local_stitch) {
    for (const RemotePtr& neighbor_ptr : selected_neighbors) {
      if (local_shard(neighbor_ptr.memory_node())) {
        local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
        job.invalidated_neighbors.push_back(neighbor_ptr.raw_address);
      } else if (!local_stitch) {
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
  StorageOwnerCoroutineScratch* scratch = current_storage_owner_thread_ != nullptr
                                            ? &current_storage_owner_thread_->coroutine_scratch_state()
                                            : nullptr;
  vec<RemotePtr> local_unique_candidates;
  vec<RemotePtr> local_current_neighbors;
  vec<RemotePtr> local_filtered_candidates;
  vec<RemotePtr> local_updated_neighbors;
  vec<RemotePtr> local_remote_neighbors;
  vec<RemotePtr> local_remote_candidates;
  vec<distance_t> local_neighbor_dists;
  if (scratch != nullptr) {
    scratch->clear_reverse_update();
  }
  vec<RemotePtr>& unique_candidates = scratch != nullptr ? scratch->reverse_unique_candidates : local_unique_candidates;
  vec<RemotePtr>& current_neighbors = scratch != nullptr ? scratch->reverse_current_neighbors : local_current_neighbors;
  vec<RemotePtr>& filtered_candidates =
    scratch != nullptr ? scratch->reverse_filtered_candidates : local_filtered_candidates;
  vec<RemotePtr>& updated_neighbors = scratch != nullptr ? scratch->reverse_updated_neighbors : local_updated_neighbors;
  vec<RemotePtr>& remote_neighbors = scratch != nullptr ? scratch->reverse_remote_neighbors : local_remote_neighbors;
  vec<RemotePtr>& remote_candidates = scratch != nullptr ? scratch->reverse_remote_candidates : local_remote_candidates;
  vec<distance_t>& neighbor_dists = scratch != nullptr ? scratch->reverse_neighbor_dists : local_neighbor_dists;

  bool changed = false;
  bool pruned = false;
  size_t current_count = 0;
  size_t filtered_count = 0;
  u64 lock_wait_ns = 0;
  u64 snapshot_ns = 0;
  u64 neighbor_read_ns = 0;
  u64 filter_ns = 0;
  u64 prune_ns = 0;
  u64 write_ns = 0;

  auto step_started = std::chrono::steady_clock::now();
  const auto target_vector_addr = vamana::StorageLayoutResolver::vector(target_ptr);
  lib_assert(target_vector_addr.offset + target_vector_addr.size <= mn_memory_bytes_,
             "local reverse-update target vector exceeds shard bounds");
  const byte_t* target_node = local_node_ptr(target_ptr);
  const byte_t* target_vector = index_buffer_.get_full_buffer() + target_vector_addr.offset;
  if ((*reinterpret_cast<const u64*>(target_node) & VamanaNode::HEADER_DELETED) != 0) {
    return true;
  }
  snapshot_ns = elapsed_ns_since(step_started);

  for (const RemotePtr& candidate_ptr : candidate_ptrs) {
    if (!candidate_ptr.is_null() &&
        std::find(unique_candidates.begin(), unique_candidates.end(), candidate_ptr) == unique_candidates.end()) {
      unique_candidates.push_back(candidate_ptr);
    }
  }
  if (unique_candidates.empty()) {
    return true;
  }

  auto target_deleted = [&]() {
    return (*reinterpret_cast<const u64*>(local_node_ptr(target_ptr)) &
            VamanaNode::HEADER_DELETED) != 0;
  };

  auto vector_ptr = [&](const RemotePtr& rptr) {
    const auto addr = vamana::StorageLayoutResolver::vector(rptr);
    lib_assert(addr.offset + addr.size <= mn_memory_bytes_,
               "local reverse-update vector read exceeds shard bounds");
    return index_buffer_.get_full_buffer() + addr.offset;
  };

  auto push_candidate = [&](const RemotePtr& candidate, distance_t candidate_dist) {
    if (updated_neighbors.size() < config.R) {
      updated_neighbors.push_back(candidate);
      neighbor_dists.push_back(candidate_dist);
      return;
    }
    lib_assert(!neighbor_dists.empty(), "reverse-update neighbor distances are unexpectedly empty");
    size_t farthest_idx = 0;
    distance_t farthest_dist = neighbor_dists[0];
    for (size_t i = 1; i < neighbor_dists.size(); ++i) {
      if (neighbor_dists[i] > farthest_dist) {
        farthest_dist = neighbor_dists[i];
        farthest_idx = i;
      }
    }
    if (candidate_dist < farthest_dist) {
      updated_neighbors[farthest_idx] = candidate;
      neighbor_dists[farthest_idx] = candidate_dist;
    }
  };

  auto build_pruned_neighbors = [&](const vec<RemotePtr>& source_neighbors,
                                    const vec<RemotePtr>& source_candidates) {
    updated_neighbors.clear();
    neighbor_dists.clear();
    remote_neighbors.clear();
    remote_candidates.clear();
    updated_neighbors.reserve(config.R);
    neighbor_dists.reserve(config.R);
    remote_neighbors.reserve(source_neighbors.size());
    remote_candidates.reserve(source_candidates.size());

    for (const RemotePtr& neighbor : source_neighbors) {
      if (neighbor.is_null()) {
        continue;
      }
      if (local_shard(neighbor.memory_node())) {
        updated_neighbors.push_back(neighbor);
        neighbor_dists.push_back(distance_between_vectors(target_vector, VamanaNode::vector_dtype(),
                                                          vector_ptr(neighbor), VamanaNode::vector_dtype(), config));
      } else {
        remote_neighbors.push_back(neighbor);
      }
    }
    if (!remote_neighbors.empty()) {
      vec<NodeSnapshot> snapshots = read_node_snapshots_batched(remote_neighbors, config);
      for (const NodeSnapshot& snapshot : snapshots) {
        updated_neighbors.push_back(snapshot.rptr);
        neighbor_dists.push_back(distance_between_vectors(target_vector, VamanaNode::vector_dtype(),
                                                          snapshot.vector_data.data(), VamanaNode::vector_dtype(),
                                                          config));
      }
    }

    for (const RemotePtr& candidate : source_candidates) {
      if (candidate.is_null()) {
        continue;
      }
      if (local_shard(candidate.memory_node())) {
        const distance_t candidate_dist = distance_between_vectors(target_vector, VamanaNode::vector_dtype(),
                                                                   vector_ptr(candidate), VamanaNode::vector_dtype(),
                                                                   config);
        push_candidate(candidate, candidate_dist);
      } else {
        remote_candidates.push_back(candidate);
      }
    }
    if (!remote_candidates.empty()) {
      vec<NodeSnapshot> candidate_snapshots = read_node_snapshots_batched(remote_candidates, config);
      for (const NodeSnapshot& snapshot : candidate_snapshots) {
        const distance_t candidate_dist = distance_between_vectors(target_vector, VamanaNode::vector_dtype(),
                                                                   snapshot.vector_data.data(),
                                                                   VamanaNode::vector_dtype(), config);
        push_candidate(snapshot.rptr, candidate_dist);
      }
    }
  };

  const auto lock_started = std::chrono::steady_clock::now();
  lock_node(target_ptr);
  lock_wait_ns += elapsed_ns_since(lock_started);
  if (target_deleted()) {
    unlock_node(target_ptr);
    return true;
  }

  step_started = std::chrono::steady_clock::now();
  current_neighbors = read_neighbor_list(target_ptr);
  neighbor_read_ns += elapsed_ns_since(step_started);
  current_count = current_neighbors.size();

  step_started = std::chrono::steady_clock::now();
  filtered_candidates.clear();
  filtered_candidates.reserve(unique_candidates.size());
  for (const RemotePtr& candidate_ptr : unique_candidates) {
    bool already_present = false;
    for (const RemotePtr& current : current_neighbors) {
      if (current == candidate_ptr) {
        already_present = true;
        break;
      }
    }
    if (!already_present) {
      filtered_candidates.push_back(candidate_ptr);
    }
  }
  filter_ns += elapsed_ns_since(step_started);
  filtered_count = filtered_candidates.size();
  if (filtered_candidates.empty()) {
    unlock_node(target_ptr);
    return true;
  }

  changed = true;
  if (current_neighbors.size() + filtered_candidates.size() <= config.R) {
    updated_neighbors = current_neighbors;
    updated_neighbors.insert(updated_neighbors.end(), filtered_candidates.begin(), filtered_candidates.end());
  } else {
    pruned = true;
    step_started = std::chrono::steady_clock::now();
    build_pruned_neighbors(current_neighbors, filtered_candidates);
    prune_ns += elapsed_ns_since(step_started);
  }

  step_started = std::chrono::steady_clock::now();
  write_neighbor_list(target_ptr, updated_neighbors);
  write_ns += elapsed_ns_since(step_started);
  unlock_node(target_ptr);

  (void)enqueue_maintenance;

  const u64 update_ns = elapsed_ns_since(update_started);
  if (update_ns > 1000ull * 1000ull * 1000ull) {
    static std::atomic<u32> slow_update_logs{0};
    const u32 log_index = slow_update_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] slow reverse-update target"
                << " self_shard=" << storage_id_
                << " target_raw=" << target_ptr.raw_address
                << " candidates=" << candidate_ptrs.size()
                << " current_neighbors=" << current_count
                << " filtered_candidates=" << filtered_count
                << " changed=" << (changed ? 1 : 0)
                << " pruned=" << (pruned ? 1 : 0)
                << " elapsed_ms=" << (update_ns / 1000000.0)
                << " lock_wait_ms=" << (lock_wait_ns / 1000000.0)
                << " snapshot_ms=" << (snapshot_ns / 1000000.0)
                << " neighbor_read_ms=" << (neighbor_read_ns / 1000000.0)
                << " filter_ms=" << (filter_ns / 1000000.0)
                << " prune_ms=" << (prune_ns / 1000000.0)
                << " write_ms=" << (write_ns / 1000000.0)
                << std::endl;
    }
  }
  return true;
}
