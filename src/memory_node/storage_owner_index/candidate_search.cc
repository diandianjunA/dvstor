#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"

using namespace memory_node_storage_owner_index_detail;

vec<RemotePtr> MemoryNode::partition_local_search_candidates(
    const span<const element_t> query,
    const vec<RemotePtr>& entry_points,
    const Configuration& config,
    InsertBreakdownCounters* breakdown,
    const byte_t* integral_raw_query,
    vec<BeamEntry>* stage1_beam,
    vec<RemotePtr>* remote_frontier) {
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
    // The canonical insert bytes are available on both stages. Use the same
    // chunked integer SIMD reduction for every uint8/int8 dimension so the
    // local beam, remote continuation and final prune compare one distance
    // semantics. A normal query still uses the float-query path because it
    // may contain genuinely fractional coordinates.
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
      IncarnationLockResult lock_result;
      do {
        lock_result = try_lock_node(candidate);
      } while (lock_result == IncarnationLockResult::busy);
      if (lock_result == IncarnationLockResult::stale) {
        // The beam retained an old physical handle while cleanup recycled its
        // slot.  The new incarnation is a different node and must not be
        // expanded through this candidate.
        neighbors.clear();
        return;
      }
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
  const PartitionSearchBudget search_budget =
    stage1_partition_search_budget(
      construction_width, entry_points.size(),
      VamanaNode::graph_entry_capacity());
  vec<PartitionLocalSearchEntry>& final_beam =
    partition_local_construction_search_into(
      reusable_search, span<const RemotePtr>{entry_points}, storage_id_,
      construction_width, search_budget, score, expand);
  if (reusable_search.budget_exhausted()) {
    storage_owner_stage1_search_budget_exhausted_.fetch_add(
      1, std::memory_order_relaxed);
  }

  filter_final_partition_local_beam(
    final_beam, [&](RemotePtr candidate) {
      const auto validation_started = std::chrono::steady_clock::now();
      const bool live = storage_owner_node_stable(candidate);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns +=
          elapsed_ns_since(validation_started);
      }
      return live;
    });

  if (stage1_beam != nullptr) {
    stage1_beam->clear();
    stage1_beam->reserve(final_beam.size());
    for (const PartitionLocalSearchEntry& entry : final_beam) {
      stage1_beam->push_back(
        BeamEntry{entry.rptr, entry.distance, true});
    }
  }
  if (remote_frontier != nullptr) {
    remote_frontier->assign(reusable_search.remote_frontier().begin(),
                            reusable_search.remote_frontier().end());
  }

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

const vec<RemotePtr>& MemoryNode::continue_stage2_search_candidates(
    const StorageOwnerMaintenanceTask& task,
    const NodeSnapshot& target,
    const Configuration& config) {
  lib_assert(!task.stage1_beam.empty(),
             "stage2 continuation requires the exact Stage1 beam");
  lib_assert(target.vector_data.size() >= VamanaNode::vector_bytes(),
             "stage2 continuation target vector is incomplete");

  thread_local vec<PartitionLocalSearchEntry> local_beam;
  local_beam.clear();
  local_beam.reserve(task.stage1_beam.size());
  for (const BeamEntry& entry : task.stage1_beam) {
    // Stage1 already computed the exact distance. Revalidation needs only the
    // stable physical identity, not another D-byte vector materialization.
    if (!read_stable_node_identity(entry.rptr)) continue;
    local_beam.push_back(
      PartitionLocalSearchEntry{entry.rptr, entry.distance, true});
  }

  thread_local vec<element_t> query;
  const VectorDType dtype = VamanaNode::vector_dtype();
  if (dtype == VectorDType::float32) {
    query.resize(VamanaNode::DIM);
    decode_storage_vector_to_float(
      target.vector_data.data(), dtype, VamanaNode::DIM, query.data());
  } else {
    // Integer Stage2 uses the canonical raw bytes on both sides; decoding an
    // unused float query would add O(D) work to every insertion.
    query.clear();
  }

  const auto score_batch = [&](span<const RemotePtr> pointers, auto&& emit) {
    // Count requested records rather than only live emissions: this is the
    // actual remote snapshot traffic Stage2 paid for under churn.
    storage_owner_stage2_scored_candidates_.fetch_add(
      pointers.size(), std::memory_order_relaxed);
    const vec<BeamEntry>& scores = score_stable_node_vectors_batched(
      pointers, target.vector_data.data(), span<const element_t>{query},
      config);
    for (const BeamEntry& score : scores) {
      emit(score.rptr, score.distance);
    }
  };
  const auto expand = [&](RemotePtr pointer, auto&& visit) {
    // Reuse adjacency storage across the O(L*R) continuation loop. Returning
    // a fresh neighbor vector here would otherwise allocate once per remote
    // expansion even though the consumer only streams its elements.
    thread_local GraphAdjacency adjacency;
    if (!read_graph_adjacency(pointer, adjacency) || adjacency.deleted) {
      return;
    }
    for (const RemotePtr neighbor : adjacency.stable) visit(neighbor);
    for (const RemotePtr neighbor : adjacency.provisional) visit(neighbor);
  };

  bool budget_exhausted = false;
  u64 remote_expansions = 0;
  const u32 construction_width =
    storage_owner_construction_width(config);
  storage_owner_stage2_continuations_.fetch_add(
    1, std::memory_order_relaxed);
  storage_owner_stage2_remote_frontier_items_.fetch_add(
    task.stage1_remote_frontier.size(), std::memory_order_relaxed);
  const vec<PartitionLocalSearchEntry>& final_beam =
    continue_partition_construction_search_into(
      span<const PartitionLocalSearchEntry>{local_beam},
      span<const RemotePtr>{task.stage1_remote_frontier}, storage_id_,
      construction_width,
      stage2_partition_search_budget(
        construction_width, VamanaNode::graph_entry_capacity()),
      score_batch, expand, &budget_exhausted, &remote_expansions);
  storage_owner_stage2_remote_expansions_.fetch_add(
    remote_expansions, std::memory_order_relaxed);
  if (budget_exhausted) {
    storage_owner_stage2_search_budget_exhausted_.fetch_add(
      1, std::memory_order_relaxed);
  }
  thread_local vec<RemotePtr> candidates;
  candidates.clear();
  candidates.reserve(final_beam.size());
  for (const PartitionLocalSearchEntry& entry : final_beam) {
    candidates.push_back(entry.rptr);
  }
  return candidates;
}
