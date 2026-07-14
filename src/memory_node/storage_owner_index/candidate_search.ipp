vec<RemotePtr> MemoryNode::beam_search_candidates(const span<const element_t> query,
                                                  RemotePtr medoid,
                                                  const Configuration& config,
                                                  InsertBreakdownCounters* breakdown) {
  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;

  auto t_snapshot = std::chrono::steady_clock::now();
  NodeSnapshot medoid_snapshot;
  read_node_snapshot(medoid, medoid_snapshot);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
  }
  auto t_distance = std::chrono::steady_clock::now();
  const distance_t medoid_dist = distance_to_stored_vector(query, medoid_snapshot.vector_data.data(), config);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
  }

  beam.push_back({medoid, medoid_dist, false});
  visited.insert(medoid);

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert shard locality summary
  static std::atomic<u32> insert_seq{0};
  u32 this_insert = insert_seq.fetch_add(1, std::memory_order_relaxed);
  bool should_log = (this_insert < 5) || (this_insert % 500 == 0);
  u32 iter_count = 0;
  u32 local_unvisited_sum = 0, remote_unvisited_sum = 0;
#endif

  for (;;) {
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    ++iter_count;
#endif
    i32 best_idx = -1;
    distance_t best_dist = std::numeric_limits<distance_t>::max();
    auto t_select = std::chrono::steady_clock::now();
    for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
      if (!beam[i].expanded && beam[i].distance < best_dist) {
        best_dist = beam[i].distance;
        best_idx = i;
      }
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_select_ns += elapsed_ns_since(t_select);
    }
    if (best_idx < 0) {
      break;
    }

    beam[best_idx].expanded = true;
    auto t_neighbor_read = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = read_neighbor_list(beam[best_idx].rptr);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(t_neighbor_read);
    }
    vec<RemotePtr> unvisited_neighbors;
    unvisited_neighbors.reserve(neighbors.size());
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    u32 iter_local = 0, iter_remote = 0;
#endif
    for (const RemotePtr& neighbor : neighbors) {
      if (neighbor.is_null() || visited.contains(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      unvisited_neighbors.push_back(neighbor);
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
      if (local_shard(neighbor.memory_node())) ++iter_local; else ++iter_remote;
#endif
    }
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    local_unvisited_sum += iter_local;
    remote_unvisited_sum += iter_remote;
#endif

    const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);
    const u32 construction_width = storage_owner_construction_width(config);
    for (size_t begin = 0; begin < unvisited_neighbors.size(); begin += snapshot_batch) {
      const size_t end = std::min(unvisited_neighbors.size(), begin + snapshot_batch);
      vec<RemotePtr> batch;
      batch.reserve(end - begin);
      batch.insert(batch.end(), unvisited_neighbors.begin() + begin, unvisited_neighbors.begin() + end);
      t_snapshot = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted) {
          continue;
        }
        t_distance = std::chrono::steady_clock::now();
        const distance_t dist = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
        }
        auto t_beam_update = std::chrono::steady_clock::now();
        insert_into_beam(beam, snapshot.rptr, dist, construction_width);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
        }
      }
    }
  }

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert summary
  if (should_log) {
    u32 total = local_unvisited_sum + remote_unvisited_sum;
    float local_pct = total > 0 ? 100.0f * local_unvisited_sum / total : 0;
    std::cerr << "[beam_search] insert=" << this_insert
              << " shard=" << storage_id_
              << " iters=" << iter_count
              << " local=" << local_unvisited_sum
              << " remote=" << remote_unvisited_sum
              << " local_pct=" << local_pct << "%"
              << std::endl;
  }
#endif

  vec<RemotePtr> candidates;
  candidates.reserve(beam.size());
  auto t_sort = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
  }
  for (const auto& entry : beam) {
    candidates.push_back(entry.rptr);
  }
  return candidates;
}

auto MemoryNode::beam_search_candidates_async(const span<const element_t> query,
                                              RemotePtr medoid,
                                              const Configuration& config,
                                              StorageOwnerThread& thread,
                                              InsertBreakdownCounters* breakdown) -> StorageOwnerInsertCoroutine {
  StorageOwnerCoroutineScratch& scratch = thread.coroutine_scratch_state();
  scratch.clear_search();
  hashset_t<RemotePtr>& visited = scratch.visited;
  vec<BeamEntry>& beam = scratch.beam;
  vec<RemotePtr>& unvisited_neighbors = scratch.unvisited;
  vec<RemotePtr>& batch = scratch.batch;
  const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, &thread);
  const u32 construction_width = storage_owner_construction_width(config);
  visited.reserve(static_cast<size_t>(construction_width) * std::max<u32>(1, config.R));
  beam.reserve(construction_width);
  unvisited_neighbors.reserve(config.R);
  batch.reserve(snapshot_batch);

  auto t_snapshot = std::chrono::steady_clock::now();
  NodeSnapshot medoid_snapshot = co_await async_read_node_snapshot(medoid, thread);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
  }
  auto t_distance = std::chrono::steady_clock::now();
  const distance_t medoid_dist = distance_to_stored_vector(query, medoid_snapshot.vector_data.data(), config);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
  }

  beam.push_back({medoid, medoid_dist, false});
  visited.insert(medoid);

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert per-shard distribution
  static std::atomic<u32> async_insert_seq{0};
  u32 this_insert_a = async_insert_seq.fetch_add(1, std::memory_order_relaxed);
  bool should_log_a = (this_insert_a < 5) || (this_insert_a % 500 == 0);
  u32 iter_count_a = 0;
  u32 shard_hist[6] = {0};  // [0..3]=remote by shard, [4]=local(self), [5]=total expanded
#endif

  for (;;) {
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    ++iter_count_a;
#endif
    i32 best_idx = -1;
    distance_t best_dist = std::numeric_limits<distance_t>::max();
    auto t_select = std::chrono::steady_clock::now();
    for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
      if (!beam[i].expanded && beam[i].distance < best_dist) {
        best_dist = beam[i].distance;
        best_idx = i;
      }
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_select_ns += elapsed_ns_since(t_select);
    }
    if (best_idx < 0) {
      break;
    }

    beam[best_idx].expanded = true;
    auto t_neighbor_read = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = co_await async_read_neighbor_list(beam[best_idx].rptr, thread);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(t_neighbor_read);
    }
    unvisited_neighbors.clear();
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    u32 expanded_shard = beam[best_idx].rptr.memory_node();
#endif
    for (const RemotePtr& neighbor : neighbors) {
      if (neighbor.is_null() || visited.contains(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      unvisited_neighbors.push_back(neighbor);
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
      u32 ns = neighbor.memory_node();
      if (ns == storage_id_) ++shard_hist[4]; else ++shard_hist[ns];
#endif
    }
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    ++shard_hist[5];
#endif

    for (size_t begin = 0; begin < unvisited_neighbors.size(); begin += snapshot_batch) {
      const size_t end = std::min(unvisited_neighbors.size(), begin + snapshot_batch);
      batch.clear();
      batch.insert(batch.end(), unvisited_neighbors.begin() + begin, unvisited_neighbors.begin() + end);
      t_snapshot = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = co_await async_read_node_snapshots(batch, config, thread);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted) {
          continue;
        }
        t_distance = std::chrono::steady_clock::now();
        const distance_t dist = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
        }
        auto t_beam_update = std::chrono::steady_clock::now();
        insert_into_beam(beam, snapshot.rptr, dist, construction_width);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
        }
      }
    }
  }

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert per-shard summary
  if (should_log_a) {
    u32 total_neighbors = 0;
    for (u32 s = 0; s < 5; ++s) total_neighbors += shard_hist[s];
    std::cerr << "[beam_search_async] insert=" << this_insert_a
              << " self=" << storage_id_
              << " iters=" << iter_count_a
              << " expanded=" << shard_hist[5]
              << " local=" << shard_hist[4];
    for (u32 s = 0; s < 5; ++s) {
      if (s == storage_id_) continue;
      float pct = total_neighbors > 0 ? 100.0f * shard_hist[s] / total_neighbors : 0;
      std::cerr << " sh" << s << "=" << shard_hist[s] << "(" << int(pct) << "%)";
    }
    std::cerr << std::endl;
  }
#endif

  auto& out = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  out.clear();
  out.reserve(beam.size());
  auto t_sort = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
  }
  for (const auto& entry : beam) {
    out.push_back(entry.rptr);
  }
}

auto MemoryNode::anchor_search_candidates_async(const span<const element_t> query,
                                                const vec<RemotePtr>& anchor_hints,
                                                const Configuration& config,
                                                StorageOwnerThread& thread,
                                                InsertBreakdownCounters* breakdown,
                                                bool local_only)
  -> StorageOwnerInsertCoroutine {
  StorageOwnerCoroutineScratch& scratch = thread.coroutine_scratch_state();
  scratch.clear_search();
  hashset_t<RemotePtr>& visited = scratch.visited;
  vec<BeamEntry>& beam = scratch.beam;
  vec<RemotePtr>& batch = scratch.batch;
  vec<RemotePtr>& unvisited = scratch.unvisited;
  const u32 beam_width = std::max<u32>(config.R, config.storage_owner_anchor_beam_width);
  const u32 batch_limit = storage_owner_snapshot_batch_size(config, &thread);
  visited.reserve(anchor_hints.size() +
                  static_cast<size_t>(config.storage_owner_anchor_expand_cap) *
                    std::max<u32>(1, config.R));
  beam.reserve(beam_width);
  batch.reserve(batch_limit);
  unvisited.reserve(config.R);

  if (breakdown != nullptr) {
    breakdown->storage_owner_anchor_hints += anchor_hints.size();
  }
  for (size_t begin = 0; begin < anchor_hints.size(); begin += batch_limit) {
    const size_t end = std::min(anchor_hints.size(), begin + batch_limit);
    batch.clear();
    for (size_t i = begin; i < end; ++i) {
      const RemotePtr hint = anchor_hints[i];
      if (!hint.is_null() && hint.memory_node() < num_storage_nodes_ &&
          (!local_only || local_shard(hint.memory_node())) &&
          visited.insert(hint).second) {
        batch.push_back(hint);
      }
    }
    auto started = std::chrono::steady_clock::now();
    vec<NodeSnapshot> snapshots = co_await async_read_node_snapshots(batch, config, thread);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(started);
    }
    for (const NodeSnapshot& snapshot : snapshots) {
      if (snapshot.deleted) continue;
      started = std::chrono::steady_clock::now();
      const distance_t distance = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_distance_ns += elapsed_ns_since(started);
        ++breakdown->storage_owner_anchor_valid_hints;
      }
      insert_into_beam(beam, snapshot.rptr, distance, beam_width);
    }
  }

  u32 expansions = 0;
  u32 remote_expansions = 0;
  while (expansions < config.storage_owner_anchor_expand_cap) {
    auto started = std::chrono::steady_clock::now();
    i32 best = -1;
    distance_t best_distance = std::numeric_limits<distance_t>::max();
    for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
      if (!beam[i].expanded && beam[i].distance < best_distance) {
        best = i;
        best_distance = beam[i].distance;
      }
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_select_ns += elapsed_ns_since(started);
    }
    if (best < 0) break;

    BeamEntry& entry = beam[best];
    entry.expanded = true;
    const bool remote = !local_shard(entry.rptr.memory_node());
    if (local_only && remote) {
      continue;
    }
    if (remote && remote_expansions >= config.storage_owner_anchor_remote_rescue_cap) {
      continue;
    }
    ++expansions;
    if (remote) ++remote_expansions;

    started = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = co_await async_read_neighbor_list(entry.rptr, thread);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(started);
    }
    unvisited.clear();
    for (const RemotePtr neighbor : neighbors) {
      if (!neighbor.is_null() && neighbor.memory_node() < num_storage_nodes_ &&
          (!local_only || local_shard(neighbor.memory_node())) &&
          visited.insert(neighbor).second) {
        unvisited.push_back(neighbor);
      }
    }

    for (size_t begin = 0; begin < unvisited.size(); begin += batch_limit) {
      const size_t end = std::min(unvisited.size(), begin + batch_limit);
      batch.clear();
      batch.insert(batch.end(), unvisited.begin() + begin, unvisited.begin() + end);
      started = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = co_await async_read_node_snapshots(batch, config, thread);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(started);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted) continue;
        started = std::chrono::steady_clock::now();
        const distance_t distance = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(started);
        }
        insert_into_beam(beam, snapshot.rptr, distance, beam_width);
      }
    }
  }

  if (breakdown != nullptr) {
    breakdown->storage_owner_anchor_expansions += expansions;
    breakdown->storage_owner_anchor_remote_expansions += remote_expansions;
  }
  auto& out = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  out.clear();
  out.reserve(beam.size());
  auto started = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(started);
  }
  for (const BeamEntry& entry : beam) out.push_back(entry.rptr);
}

