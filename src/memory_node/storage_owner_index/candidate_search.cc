#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"

using namespace memory_node_storage_owner_index_detail;

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

vec<RemotePtr> MemoryNode::partition_local_search_candidates(
    const span<const element_t> query,
    const vec<RemotePtr>& entry_points,
    const Configuration& config,
    InsertBreakdownCounters* breakdown,
    const byte_t* integral_raw_query) {
  StorageOwnerCoroutineScratch* scratch = current_storage_owner_thread_ != nullptr
                                            ? &current_storage_owner_thread_->coroutine_scratch_state()
                                            : nullptr;
  vec<RemotePtr> local_neighbors;
  vec<byte_t> local_neighbor_entry;
  vec<byte_t> local_neighbor_decoded;
  if (scratch != nullptr) {
    scratch->neighbors.clear();
  }
  vec<RemotePtr>& neighbors = scratch != nullptr ? scratch->neighbors : local_neighbors;
  vec<byte_t>& neighbor_entry = scratch != nullptr
                                  ? scratch->neighbor_entry
                                  : local_neighbor_entry;
  vec<byte_t>& neighbor_decoded = scratch != nullptr
                                    ? scratch->neighbor_decoded
                                    : local_neighbor_decoded;
  neighbors.reserve(config.R);

  const u32 construction_width = storage_owner_construction_width(config);
  const VectorDType dtype = VamanaNode::vector_dtype();
  const bool exact_integral_query = integral_raw_query != nullptr &&
    integral_byte_l2_sum_exact_in_float(config.dim) &&
    (dtype == VectorDType::uint8 || dtype == VectorDType::int8);
  auto score = [&](RemotePtr candidate) -> std::optional<distance_t> {
    auto started = std::chrono::steady_clock::now();
    const byte_t* vector = local_live_vector(candidate);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(started);
    }
    if (vector == nullptr) {
      return std::nullopt;
    }

    started = std::chrono::steady_clock::now();
    // Peer stage2 already carries the exact uint8/int8 bytes. Integer squared
    // L2 through DIM=258 is exactly representable in float even at the maximum
    // byte difference, so the same-dtype AVX2 path preserves candidate ordering
    // while avoiding repeated byte-to-float conversion in every expansion.
    // Wider and float indexes retain the established decoded-query reduction.
    const distance_t distance = exact_integral_query
      ? typed_l2_distance(
          integral_raw_query, dtype, vector, dtype, config.dim)
      : distance_to_stored_vector(query, vector, config);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_distance_ns += elapsed_ns_since(started);
    }
    return distance;
  };
  auto expand = [&](RemotePtr candidate, auto&& visit) {
    const auto started = std::chrono::steady_clock::now();
    bool decoded = read_local_neighbor_list(
      candidate, neighbors, neighbor_entry, neighbor_decoded);
    if (!decoded) {
      // Concurrent adjacency publication can invalidate all optimistic
      // checksum attempts. Falling back to the node lock is rare, but avoids
      // silently treating a hot node as a leaf and permanently reducing the
      // construction candidate set.
      lock_node(candidate);
      decoded = read_local_neighbor_list(
        candidate, neighbors, neighbor_entry, neighbor_decoded);
      unlock_node(candidate);
      lib_assert(decoded,
                 "partition-local construction search could not decode a "
                 "locked adjacency snapshot");
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(started);
    }
    for (const RemotePtr neighbor : neighbors) {
      visit(neighbor);
    }
  };

  // This wrapper never suspends, so one reusable state per OS thread cannot
  // be observed by another coroutine while a search is in progress.
  thread_local PartitionLocalSearchBeam reusable_search(0, 1);
  vec<PartitionLocalSearchEntry>& final_beam =
    partition_local_construction_search_into(
      reusable_search, span<const RemotePtr>{entry_points}, storage_id_,
      construction_width, score, expand);

  filter_final_partition_local_beam(
    final_beam, [&](RemotePtr candidate) {
      const auto validation_started = std::chrono::steady_clock::now();
      const bool live = storage_owner_node_live(candidate);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns +=
          elapsed_ns_since(validation_started);
      }
      return live;
    });

  const auto started = std::chrono::steady_clock::now();
  vec<RemotePtr> candidates;
  candidates.reserve(final_beam.size());
  for (const PartitionLocalSearchEntry& entry : final_beam) {
    candidates.push_back(entry.rptr);
  }
  if (breakdown != nullptr) {
    // Beam insertion keeps results ordered, so this field now covers only the
    // final result materialization rather than a redundant sort.
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(started);
  }
  return candidates;
}
