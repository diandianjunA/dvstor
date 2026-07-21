#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

#include <numeric>

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

void MemoryNode::continue_stage2_search_candidates_batched(
    span<const StorageOwnerMaintenanceTask> tasks,
    span<const NodeSnapshot> targets,
    vec<vec<RemotePtr>>& candidates_by_task,
    const Configuration& config) {
  lib_assert(tasks.size() == targets.size(),
             "batched Stage2 continuation lost task/target correlation");
  // Each resumable Stage2 context owns this output. Preserve active inner
  // vector capacity across context reuse instead of destroying every row on
  // each batch; the final-beam scatter below clears the rows it writes.
  candidates_by_task.resize(tasks.size());
  if (tasks.empty()) return;

  const u32 construction_width = storage_owner_construction_width(config);
  const PartitionSearchBudget budget = stage2_partition_search_budget(
    construction_width, VamanaNode::graph_entry_capacity());

  // Keep independent beams/visited sets but retain their capacity across
  // high-frequency batches on this maintenance OS worker.
  thread_local vec<vec<PartitionLocalSearchEntry>> local_beams;
  thread_local vec<PartitionContinuationSeed> seeds;
  thread_local PartitionContinuationBatch continuation_batch;
  thread_local vec<bool> budget_exhausted;
  thread_local vec<u64> expansion_counts;
  thread_local vec<RemotePtr> score_unique;
  thread_local vec<size_t> score_order;
  thread_local vec<size_t> score_group_offsets;
  thread_local vec<RemotePtr> graph_unique;
  thread_local vec<std::pair<RemotePtr, GraphAdjacency>> graph_snapshots;
  thread_local dense_hashmap_t<u64, const GraphAdjacency*> graph_by_pointer;
  local_beams.resize(tasks.size());
  seeds.resize(tasks.size());

  u64 frontier_items = 0;
  for (size_t item = 0; item < tasks.size(); ++item) {
    const StorageOwnerMaintenanceTask& task = tasks[item];
    const NodeSnapshot& target = targets[item];
    lib_assert(!task.stage1_beam.empty(),
               "stage2 task lost its Stage1 continuation beam");
    lib_assert(target.vector_data.size() >= VamanaNode::vector_bytes(),
               "stage2 continuation target vector is incomplete");
    vec<PartitionLocalSearchEntry>& local = local_beams[item];
    local.clear();
    local.reserve(task.stage1_beam.size());
    for (const BeamEntry& entry : task.stage1_beam) {
      // Stage1 distances remain authoritative for its local beam. Identity
      // validation is local to the Stage1 owner and therefore adds no RDMA
      // serialization to the cross-task wavefront.
      if (!read_stable_node_identity(entry.rptr)) continue;
      local.push_back({entry.rptr, entry.distance, true});
    }
    seeds[item] = {
      .local_beam = span<const PartitionLocalSearchEntry>{local},
      .remote_frontier = span<const RemotePtr>{task.stage1_remote_frontier},
    };
    frontier_items += task.stage1_remote_frontier.size();
  }
  storage_owner_stage2_continuations_.fetch_add(
    tasks.size(), std::memory_order_relaxed);
  storage_owner_stage2_remote_frontier_items_.fetch_add(
    frontier_items, std::memory_order_relaxed);

  const auto score_wave = [&](
      span<const PartitionContinuationScoreRequest> requests,
      auto&& emit) {
    storage_owner_stage2_scored_candidates_.fetch_add(
      requests.size(), std::memory_order_relaxed);
    score_order.resize(requests.size());
    std::iota(score_order.begin(), score_order.end(), size_t{0});
    std::sort(score_order.begin(), score_order.end(),
              [&](size_t lhs, size_t rhs) {
                const auto& left = requests[lhs];
                const auto& right = requests[rhs];
                if (left.pointer.raw_address != right.pointer.raw_address) {
                  return left.pointer.raw_address < right.pointer.raw_address;
                }
                return left.search_index < right.search_index;
              });
    score_unique.clear();
    score_group_offsets.clear();
    for (size_t position = 0; position < score_order.size(); ++position) {
      const RemotePtr pointer = requests[score_order[position]].pointer;
      if (score_unique.empty() || score_unique.back() != pointer) {
        score_unique.push_back(pointer);
        score_group_offsets.push_back(position);
      }
    }
    score_group_offsets.push_back(score_order.size());
    storage_owner_stage2_vector_read_waves_.fetch_add(
      1, std::memory_order_relaxed);
    storage_owner_stage2_vector_unique_reads_.fetch_add(
      score_unique.size(), std::memory_order_relaxed);

    const VectorDType dtype = VamanaNode::vector_dtype();
    const auto emit_group = [&](size_t group_index,
                                const byte_t* candidate_vector) {
      for (size_t position = score_group_offsets[group_index];
           position < score_group_offsets[group_index + 1]; ++position) {
        const auto& request = requests[score_order[position]];
        const distance_t distance = distance_between_vectors(
          targets[request.search_index].vector_data.data(), dtype,
          candidate_vector, dtype, config);
        emit(request.search_index, request.pointer, distance);
      }
    };

    StorageOwnerThread* thread = current_storage_owner_thread_;
    if (thread == nullptr || !thread->has_peer_scratch()) {
      for (size_t group = 0; group < score_unique.size(); ++group) {
        NodeSnapshot snapshot;
        if (read_node_snapshot(score_unique[group], snapshot) &&
            !snapshot.deleted &&
            (snapshot.header & VamanaNode::HEADER_PROVISIONAL) == 0 &&
            snapshot.vector_data.size() >= VamanaNode::vector_bytes()) {
          emit_group(group, snapshot.vector_data.data());
        }
      }
      return;
    }

    struct PendingVectorRead {
      size_t group_index{};
      RemotePtr pointer;
      byte_t* buffer{};
      u64 before{};
      u32 slot_incarnation{};
    };
    thread_local vec<PendingVectorRead> pending;
    const size_t snapshot_size = snapshot_buffer_bytes();
    const size_t snapshot_stride = aligned_snapshot_bytes();
    const size_t max_batch =
      storage_owner_snapshot_batch_size(config, thread);
    pending.reserve(max_batch);
    for (size_t begin = 0; begin < score_unique.size();
         begin += max_batch) {
      const size_t end = std::min(score_unique.size(), begin + max_batch);
      pending.clear();
      u32 remote_slot = 0;
      for (size_t group = begin; group < end; ++group) {
        const RemotePtr pointer = score_unique[group];
        lib_assert(!pointer.is_null() && pointer.is_well_formed() &&
                     pointer.memory_node() < num_storage_nodes_,
                   "batched Stage2 score received an invalid pointer");
        if (local_shard(pointer.memory_node())) {
          NodeSnapshot snapshot;
          if (read_node_snapshot(pointer, snapshot) && !snapshot.deleted &&
              (snapshot.header & VamanaNode::HEADER_PROVISIONAL) == 0 &&
              snapshot.vector_data.size() >= VamanaNode::vector_bytes()) {
            emit_group(group, snapshot.vector_data.data());
          }
          continue;
        }
        const auto vector_address =
          vamana::StorageLayoutResolver::vector(pointer);
        lib_assert(vector_address.offset <= mn_memory_bytes_ &&
                     vector_address.size <=
                       mn_memory_bytes_ - vector_address.offset,
                   "batched Stage2 vector read exceeds shard bounds");
        const size_t scratch_offset =
          static_cast<size_t>(remote_slot++) * snapshot_stride;
        lib_assert(scratch_offset + snapshot_size <= thread->scratch_stride,
                   "storage-owner scratch cannot hold Stage2 score wave");
        byte_t* buffer = thread->coroutine_scratch(scratch_offset);
        post_peer_read_async(
          *thread, pointer.memory_node(), pointer.byte_offset(), buffer,
          VamanaNode::size_until_vector_end());
        pending.push_back({group, pointer, buffer});
      }
      while (!thread->is_ready(thread->running_coroutine)) {
        poll_peer_send_cq();
        std::this_thread::yield();
      }

      size_t valid_count = 0;
      for (PendingVectorRead& read : pending) {
        read.before = *reinterpret_cast<const u64*>(read.buffer);
        read.slot_incarnation = *reinterpret_cast<const u32*>(
          read.buffer + VamanaNode::offset_slot_incarnation());
        if (!stable_vector_snapshot_valid(
              read.pointer, read.before, read.before,
              read.slot_incarnation)) {
          continue;
        }
        if (valid_count != static_cast<size_t>(&read - pending.data())) {
          pending[valid_count] = read;
        }
        PendingVectorRead& accepted = pending[valid_count++];
        post_peer_read_async(
          *thread, accepted.pointer.memory_node(),
          accepted.pointer.byte_offset(), accepted.buffer,
          VamanaNode::HEADER_SIZE);
      }
      pending.resize(valid_count);
      while (!thread->is_ready(thread->running_coroutine)) {
        poll_peer_send_cq();
        std::this_thread::yield();
      }
      for (const PendingVectorRead& read : pending) {
        const u64 after = *reinterpret_cast<const u64*>(read.buffer);
        if (stable_vector_snapshot_valid(
              read.pointer, read.before, after,
              read.slot_incarnation)) {
          emit_group(read.group_index,
                     read.buffer + VamanaNode::offset_vector());
        }
      }
    }
  };

  const auto expand_wave = [&](
      span<const PartitionContinuationExpandRequest> requests,
      auto&& emit) {
    graph_unique.clear();
    graph_unique.reserve(requests.size());
    for (const auto& request : requests) {
      graph_unique.push_back(request.pointer);
    }
    std::sort(graph_unique.begin(), graph_unique.end(),
              [](RemotePtr lhs, RemotePtr rhs) {
                return lhs.raw_address < rhs.raw_address;
              });
    graph_unique.erase(
      std::unique(graph_unique.begin(), graph_unique.end()),
      graph_unique.end());
    storage_owner_stage2_graph_read_waves_.fetch_add(
      1, std::memory_order_relaxed);
    storage_owner_stage2_graph_unique_reads_.fetch_add(
      graph_unique.size(), std::memory_order_relaxed);
    const size_t graph_snapshot_count = read_graph_adjacencies_batched_into(
      span<const RemotePtr>{graph_unique}, config, graph_snapshots);
    graph_by_pointer.clear();
    graph_by_pointer.reserve(graph_snapshot_count);
    for (size_t snapshot_index = 0;
         snapshot_index < graph_snapshot_count; ++snapshot_index) {
      const auto& [pointer, adjacency] = graph_snapshots[snapshot_index];
      if (!adjacency.deleted) {
        graph_by_pointer.emplace(pointer.raw_address, &adjacency);
      }
    }
    for (const auto& request : requests) {
      const auto found = graph_by_pointer.find(request.pointer.raw_address);
      if (found == graph_by_pointer.end()) continue;
      const GraphAdjacency& adjacency = *found->second;
      for (const RemotePtr neighbor : adjacency.stable) {
        emit(request.search_index, neighbor);
      }
      for (const RemotePtr neighbor : adjacency.provisional) {
        emit(request.search_index, neighbor);
      }
    }
  };

  const auto& final_beams = continuation_batch.run(
    span<const PartitionContinuationSeed>{seeds}, storage_id_,
    construction_width, budget, score_wave, expand_wave,
    &budget_exhausted, &expansion_counts);
  u64 total_expansions = 0;
  u64 exhausted = 0;
  for (size_t item = 0; item < final_beams.size(); ++item) {
    total_expansions += expansion_counts[item];
    exhausted += budget_exhausted[item];
    vec<RemotePtr>& candidates = candidates_by_task[item];
    candidates.clear();
    candidates.reserve(final_beams[item].size());
    for (const PartitionLocalSearchEntry& entry : final_beams[item]) {
      candidates.push_back(entry.rptr);
    }
  }
  storage_owner_stage2_remote_expansions_.fetch_add(
    total_expansions, std::memory_order_relaxed);
  storage_owner_stage2_search_budget_exhausted_.fetch_add(
    exhausted, std::memory_order_relaxed);
}
