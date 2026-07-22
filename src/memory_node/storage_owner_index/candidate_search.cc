#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"
#include "memory_node/storage_owner_maintenance/search_io_state.hh"
#include "memory_node/storage_owner_maintenance/detail.hh"

#include <numeric>

using namespace memory_node_storage_owner_index_detail;
using namespace memory_node_storage_owner_maintenance_detail;

namespace {

// Stage1 may traverse a query-visible insertion while its Stage2 maintenance
// is still pending.  PROVISIONAL is therefore a traversal state, not a dead
// physical identity.  The final Stage1 handoff is filtered separately with
// classify_stable_node_snapshot(), so only stable nodes become final
// candidates or inherited Stage2 beam entries.
StableNodeSnapshotState classify_stage1_traversal_snapshot(
    RemotePtr pointer,
    u64 before,
    u64 after,
    u32 slot_incarnation) {
  const StableNodeSnapshotState physical = classify_physical_node_snapshot(
    pointer, before, after, slot_incarnation);
  if (physical != StableNodeSnapshotState::stable) return physical;
  return (after & VamanaNode::HEADER_DELETED) != 0
    ? StableNodeSnapshotState::terminal
    : StableNodeSnapshotState::stable;
}

}  // namespace

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
    if (!storage_node_pointer_addressable(candidate)) {
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns +=
          elapsed_ns_since(started);
      }
      return std::nullopt;
    }

    const byte_t* node = local_node_ptr(candidate);
    const byte_t* vector = node + VamanaNode::offset_vector();
    for (;;) {
      const u64 before = load_local_node_header_acquire(candidate);
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0) {
        std::this_thread::yield();
        continue;
      }
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        node + VamanaNode::offset_slot_incarnation());

      // Avoid an O(D) distance calculation for a conclusively dead physical
      // identity.  It is conclusive only when the header remains unchanged
      // and unlocked around the slot-incarnation observation.
      if (VamanaNode::header_incarnation(before) != candidate.incarnation() ||
          slot_incarnation != candidate.incarnation() ||
          (before & VamanaNode::HEADER_DELETED) != 0) {
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 after = load_local_node_header_acquire(candidate);
        const StableNodeSnapshotState state =
          classify_stage1_traversal_snapshot(
            candidate, before, after, slot_incarnation);
        if (state == StableNodeSnapshotState::terminal) {
          if (breakdown != nullptr) {
            breakdown->storage_owner_search_snapshot_read_ns +=
              elapsed_ns_since(started);
          }
          return std::nullopt;
        }
        std::this_thread::yield();
        continue;
      }

      const auto distance_started = std::chrono::steady_clock::now();
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
        breakdown->storage_owner_search_distance_ns +=
          elapsed_ns_since(distance_started);
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(candidate);
      const StableNodeSnapshotState state =
        classify_stage1_traversal_snapshot(
          candidate, before, after, slot_incarnation);
      if (state == StableNodeSnapshotState::stable) {
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_snapshot_read_ns +=
            elapsed_ns_since(started);
        }
        return distance;
      }
      if (state == StableNodeSnapshotState::terminal) {
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_snapshot_read_ns +=
            elapsed_ns_since(started);
        }
        return std::nullopt;
      }
      // NODE_LOCK or any optimistic before/after mismatch is contention, not
      // evidence that the candidate is absent.  There is intentionally no
      // retry count that converts this state into successful convergence.
      std::this_thread::yield();
    }
  };
  auto expand = [&](RemotePtr candidate, auto&& visit) {
    const auto started = std::chrono::steady_clock::now();
    const byte_t* node = local_node_ptr(candidate);
    const auto validate_decoded_neighbors = [&] {
      const u32 edge_count =
        VamanaNode::decoded_neighbor_count(neighbor_decoded.data());
      const auto* slots = reinterpret_cast<const RemotePtr*>(
        neighbor_decoded.data() +
          VamanaNode::neighbor_payload_offset_in_read());
      for (u32 index = 0;
           index < edge_count &&
             index < VamanaNode::graph_entry_capacity();
           ++index) {
        if (!slots[index].is_null() &&
            !storage_node_pointer_addressable(slots[index])) {
          lib_failure(
            "stable Stage1 graph contains a malformed remote pointer");
        }
      }
    };
    u32 stable_optimistic_misses = 0;
    constexpr u32 kStableOptimisticMissesBeforeLock = 3;
    for (;;) {
      const u64 before = load_local_node_header_acquire(candidate);
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        node + VamanaNode::offset_slot_incarnation());
      const bool decoded = read_local_neighbor_list(
        candidate, neighbors, neighbor_entry, neighbor_decoded);
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(candidate);
      const StableNodeSnapshotState state =
        classify_stage1_traversal_snapshot(
          candidate, before, after, slot_incarnation);
      if (state == StableNodeSnapshotState::terminal) {
        neighbors.clear();
        return;
      }
      if (state == StableNodeSnapshotState::stable && decoded) {
        validate_decoded_neighbors();
        break;
      }

      if (state != StableNodeSnapshotState::stable) {
        // A changing/locked header explains this miss; it is ordinary
        // contention rather than evidence of persistent graph corruption.
        stable_optimistic_misses = 0;
        std::this_thread::yield();
        continue;
      }

      ++stable_optimistic_misses;
      if (stable_optimistic_misses <
          kStableOptimisticMissesBeforeLock) {
        std::this_thread::yield();
        continue;
      }

      // Repeated checksum failure under one coherent physical identity needs
      // a race-closing diagnosis. The short count is not a search budget: it
      // only distinguishes optimistic publication tearing from a durable
      // malformed adjacency. Once the incarnation lock is held, a second
      // decode cannot race a graph writer and therefore must succeed.
      const IncarnationLockResult lock = try_lock_node(candidate);
      if (lock == IncarnationLockResult::busy) {
        std::this_thread::yield();
        continue;
      }
      if (lock == IncarnationLockResult::stale) {
        neighbors.clear();
        return;
      }
      const u64 locked_header = load_local_node_header_acquire(candidate);
      if ((locked_header & VamanaNode::HEADER_DELETED) != 0) {
        neighbors.clear();
        unlock_node(candidate);
        return;
      }
      const bool locked_decoded = read_local_neighbor_list(
        candidate, neighbors, neighbor_entry, neighbor_decoded);
      unlock_node(candidate);
      if (!locked_decoded) {
        lib_failure(
          "stable Stage1 graph remains malformed while incarnation-locked");
      }
      validate_decoded_neighbors();
      break;
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
      const byte_t* node = local_node_ptr(candidate);
      for (;;) {
        const u64 before = load_local_node_header_acquire(candidate);
        const u32 slot_incarnation = *reinterpret_cast<const u32*>(
          node + VamanaNode::offset_slot_incarnation());
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 after = load_local_node_header_acquire(candidate);
        const StableNodeSnapshotState state = classify_stable_node_snapshot(
          candidate, before, after, slot_incarnation);
        if (state != StableNodeSnapshotState::retryable) {
          if (breakdown != nullptr) {
            breakdown->storage_owner_search_snapshot_read_ns +=
              elapsed_ns_since(validation_started);
          }
          return state == StableNodeSnapshotState::stable;
        }
        std::this_thread::yield();
      }
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
  // All three externally visible Stage1 products have now been copied.
  // Release only exceptional thread-local high-water capacity; this is a
  // completed-search retention policy and never constrains an active search.
  reusable_search.trim_oversized_capacity(
    std::max<size_t>(1024, static_cast<size_t>(construction_width) * 8));
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

  const auto classify_inherited_stage1_entry = [&](RemotePtr pointer) {
    if (!storage_node_pointer_addressable(pointer) ||
        !local_shard(pointer.memory_node())) {
      return StableNodeSnapshotState::terminal;
    }
    const byte_t* node = local_node_ptr(pointer);
    const u64 before = load_local_node_header_acquire(pointer);
    const u32 slot_incarnation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_slot_incarnation());
    std::atomic_thread_fence(std::memory_order_acquire);
    const u64 after = load_local_node_header_acquire(pointer);
    return classify_stable_node_snapshot(
      pointer, before, after, slot_incarnation);
  };

  thread_local vec<PartitionLocalSearchEntry> local_beam;
  local_beam.clear();
  local_beam.reserve(task.stage1_beam.size());
  for (const BeamEntry& entry : task.stage1_beam) {
    // Stage1 already computed the exact distance. Revalidation needs only the
    // stable physical identity, not another D-byte vector materialization.
    // A transient lock/torn observation is retried without a fixed count;
    // only a coherent stale/deleted/provisional identity is omitted.
    for (;;) {
      const StableNodeSnapshotState state =
        classify_inherited_stage1_entry(entry.rptr);
      if (state == StableNodeSnapshotState::stable) {
        local_beam.push_back(
          PartitionLocalSearchEntry{entry.rptr, entry.distance, true});
        break;
      }
      if (state == StableNodeSnapshotState::terminal) break;
      std::this_thread::yield();
    }
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

  const auto classify_inherited_stage1_entry = [&](RemotePtr pointer) {
    if (!storage_node_pointer_addressable(pointer) ||
        !local_shard(pointer.memory_node())) {
      return StableNodeSnapshotState::terminal;
    }
    const byte_t* node = local_node_ptr(pointer);
    const u64 before = load_local_node_header_acquire(pointer);
    const u32 slot_incarnation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_slot_incarnation());
    std::atomic_thread_fence(std::memory_order_acquire);
    const u64 after = load_local_node_header_acquire(pointer);
    return classify_stable_node_snapshot(
      pointer, before, after, slot_incarnation);
  };

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
      for (;;) {
        const StableNodeSnapshotState state =
          classify_inherited_stage1_entry(entry.rptr);
        if (state == StableNodeSnapshotState::stable) {
          local.push_back({entry.rptr, entry.distance, true});
          break;
        }
        if (state == StableNodeSnapshotState::terminal) break;
        std::this_thread::yield();
      }
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
      byte_t* after_header{};
      u64 before{};
      u32 slot_incarnation{};
    };
    thread_local vec<PendingVectorRead> pending;
    thread_local vec<PeerReadRequest> read_requests;
    thread_local vec<PeerReadPairRequest> read_pairs;
    const size_t snapshot_stride = aligned_snapshot_bytes();
    const size_t validation_offset =
      memory_node_detail::storage_owner_snapshot_validation_offset();
    const size_t max_batch =
      storage_owner_snapshot_batch_size(config, thread);
    const bool ordered_pair_supported =
      memory_node_detail::peer_rdma_read_pair_group_limit(
        peer_rdma_read_credit_plan()) != 0;
    lib_assert(validation_offset + VamanaNode::HEADER_SIZE <=
                 snapshot_stride,
               "storage-owner snapshot slot lost after-header scratch");
    pending.reserve(max_batch);
    for (size_t begin = 0; begin < score_unique.size();
         begin += max_batch) {
      const size_t end = std::min(score_unique.size(), begin + max_batch);
      pending.clear();
      read_requests.clear();
      read_pairs.clear();
      u32 remote_slot = 0;
      for (size_t group = begin; group < end; ++group) {
        const RemotePtr pointer = score_unique[group];
        if (!storage_node_pointer_addressable(pointer)) {
          if (!pointer.is_null()) {
            report_rejected_graph_pointer(
              "stage2_continuation_score", pointer, RemotePtr{}, group);
          }
          continue;
        }
        if (local_shard(pointer.memory_node())) {
          NodeSnapshot snapshot;
          if (read_node_snapshot(pointer, snapshot) && !snapshot.deleted &&
              (snapshot.header & VamanaNode::HEADER_PROVISIONAL) == 0 &&
              snapshot.vector_data.size() >= VamanaNode::vector_bytes()) {
            emit_group(group, snapshot.vector_data.data());
          }
          continue;
        }
        const size_t scratch_offset =
          static_cast<size_t>(remote_slot++) * snapshot_stride;
        lib_assert(scratch_offset + validation_offset +
                     VamanaNode::HEADER_SIZE <= thread->scratch_stride,
                   "storage-owner scratch cannot hold Stage2 score wave");
        byte_t* buffer = thread->coroutine_scratch(scratch_offset);
        const PeerReadRequest full_snapshot{
          .shard_id = pointer.memory_node(),
          .remote_offset = pointer.byte_offset(),
          .destination = buffer,
          .bytes = VamanaNode::size_until_vector_end(),
        };
        byte_t* after_header = buffer + validation_offset;
        if (ordered_pair_supported) {
          read_pairs.push_back(PeerReadPairRequest{
            .full_snapshot = full_snapshot,
            .after_header = PeerReadRequest{
              .shard_id = pointer.memory_node(),
              .remote_offset = pointer.byte_offset(),
              .destination = after_header,
              .bytes = VamanaNode::HEADER_SIZE,
            },
          });
        } else {
          read_requests.push_back(full_snapshot);
        }
        pending.push_back({group, pointer, buffer, after_header});
      }
      if (ordered_pair_supported) {
        post_peer_read_pairs_async(
          *thread, span<const PeerReadPairRequest>{read_pairs});
      } else {
        post_peer_reads_async(
          *thread, span<const PeerReadRequest>{read_requests});
      }
      while (!thread->is_ready(thread->running_coroutine)) {
        poll_peer_send_cq();
        std::this_thread::yield();
      }

      size_t valid_count = 0;
      read_requests.clear();
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
        if (!ordered_pair_supported) {
          read_requests.push_back(PeerReadRequest{
            .shard_id = accepted.pointer.memory_node(),
            .remote_offset = accepted.pointer.byte_offset(),
            .destination = accepted.buffer,
            .bytes = VamanaNode::HEADER_SIZE,
          });
        }
      }
      pending.resize(valid_count);
      if (ordered_pair_supported) {
        for (const PendingVectorRead& read : pending) {
          const u64 after = *reinterpret_cast<const u64*>(
            read.after_header);
          if (stable_vector_snapshot_valid(
                read.pointer, read.before, after,
                read.slot_incarnation)) {
            emit_group(read.group_index,
                       read.buffer + VamanaNode::offset_vector());
          }
        }
        continue;
      }

      // A transport with only one atomic/WQE credit cannot reserve an
      // ordered two-read chain. Preserve the original two-wave validation;
      // never weaken a stable snapshot to a single read.
      post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
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


Stage2SearchAdvanceResult
MemoryNode::advance_stage2_search_candidates_batched(
    span<const StorageOwnerMaintenanceTask> tasks,
    span<const NodeSnapshot> targets,
    vec<vec<RemotePtr>>& candidates_by_task,
    Stage2SearchIoState& state,
    const Configuration& config) {
  lib_assert(tasks.size() == targets.size(),
             "asynchronous Stage2 continuation lost task/target correlation");
  candidates_by_task.resize(tasks.size());
  if (tasks.empty()) {
    state.reset();
    return Stage2SearchAdvanceResult::complete;
  }

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    // Unit/single-node callers have no registered lane scratch.  Production
    // maintenance always enters the resumable path below.
    continue_stage2_search_candidates_batched(
      tasks, targets, candidates_by_task, config);
    state.reset();
    return Stage2SearchAdvanceResult::complete;
  }

  const u32 construction_width = storage_owner_construction_width(config);
  const PartitionSearchBudget budget = stage2_partition_search_budget(
    construction_width, VamanaNode::graph_entry_capacity());
  const auto& credit_plan = peer_rdma_read_credit_plan();
  const auto ordinary_dispatch_limits =
    memory_node_detail::peer_rdma_read_dispatch_limits(credit_plan);
  const auto pair_dispatch_limits =
    memory_node_detail::peer_rdma_read_pair_dispatch_limits(credit_plan);
  const auto snapshot_dispatch_limits =
    memory_node_detail::peer_rdma_snapshot_dispatch_limits(credit_plan);
  lib_assert(ordinary_dispatch_limits.global_items != 0 &&
               ordinary_dispatch_limits.per_peer_items != 0,
             "asynchronous Stage2 has no RDMA READ credit");
  const bool ordered_pairs = pair_dispatch_limits.global_items != 0 &&
    pair_dispatch_limits.per_peer_items != 0;
  const bool mixed_snapshots = ordered_pairs &&
    snapshot_dispatch_limits.per_peer_pairs != 0;
  // Scratch bounds distinct remote physical records, not logical consumers.
  // Local/terminal work and multiple searches sharing one pointer consume no
  // additional registered bytes and can retire in this dispatcher turn. RDMA
  // quota is enforced independently below: global_items caps the whole
  // physical wave and per_peer_items caps each destination shard.
  const size_t score_dispatch_limit =
    storage_owner_snapshot_batch_size(config, thread);
  const size_t graph_dispatch_limit =
    storage_owner_graph_batch_size(config, thread);
  lib_assert(score_dispatch_limit != 0 && graph_dispatch_limit != 0,
             "asynchronous Stage2 scratch cannot hold one RDMA record");

  const auto classify_inherited_stage1_entry = [&](RemotePtr pointer) {
    if (!storage_node_pointer_addressable(pointer) ||
        !local_shard(pointer.memory_node())) {
      return StableNodeSnapshotState::terminal;
    }
    const byte_t* node = local_node_ptr(pointer);
    const u64 before = load_local_node_header_acquire(pointer);
    const u32 slot_incarnation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_slot_incarnation());
    std::atomic_thread_fence(std::memory_order_acquire);
    const u64 after = load_local_node_header_acquire(pointer);
    return classify_stable_node_snapshot(
      pointer, before, after, slot_incarnation);
  };

  if (!state.initialized) {
    lib_assert(state.phase == Stage2SearchIoPhase::idle,
               "new Stage2 continuation inherited pending lane I/O");
    state.local_beams.resize(tasks.size());
    state.seeds.resize(tasks.size());
    state.search_seeded.assign(tasks.size(), 0);
    state.score_collect_cursors.assign(tasks.size(), {});
    state.continuation.initialize(
      tasks.size(), storage_id_, construction_width, budget);
    state.round_robin_search = 0;
    state.prefer_graph = false;
    state.initialized = true;

    u64 frontier_items = 0;
    for (const StorageOwnerMaintenanceTask& task : tasks) {
      frontier_items += task.stage1_remote_frontier.size();
    }
    storage_owner_stage2_continuations_.fetch_add(
      tasks.size(), std::memory_order_relaxed);
    storage_owner_stage2_remote_frontier_items_.fetch_add(
      frontier_items, std::memory_order_relaxed);
  }

  // Activate every stable Stage1 handoff independently.  A transient lock in
  // one inherited local beam leaves only that search uninitialized; searches
  // already activated below continue to issue and retire RDMA dispatches.
  for (size_t item = 0; item < tasks.size(); ++item) {
    if (state.search_seeded[item] != 0) continue;
    const StorageOwnerMaintenanceTask& task = tasks[item];
    const NodeSnapshot& target = targets[item];
    lib_assert(!task.stage1_beam.empty(),
               "stage2 task lost its Stage1 continuation beam");
    lib_assert(target.vector_data.size() >= VamanaNode::vector_bytes(),
               "stage2 continuation target vector is incomplete");

    vec<PartitionLocalSearchEntry>& local = state.local_beams[item];
    local.clear();
    local.reserve(task.stage1_beam.size());
    bool retryable = false;
    for (const BeamEntry& entry : task.stage1_beam) {
      const StableNodeSnapshotState entry_state =
        classify_inherited_stage1_entry(entry.rptr);
      if (entry_state == StableNodeSnapshotState::retryable) {
        retryable = true;
        break;
      }
      if (entry_state == StableNodeSnapshotState::stable) {
        local.push_back({entry.rptr, entry.distance, true});
      }
    }
    if (retryable) continue;

    state.seeds[item] = {
      .local_beam = span<const PartitionLocalSearchEntry>{local},
      .remote_frontier = span<const RemotePtr>{task.stage1_remote_frontier},
    };
    state.continuation.initialize_search(item, state.seeds[item]);
    state.search_seeded[item] = 1;
  }

  const auto classify_vector_snapshot = [](
      RemotePtr pointer, u64 before, u64 after, u32 slot_incarnation) {
    return classify_stable_node_snapshot(
      pointer, before, after, slot_incarnation);
  };

  const auto clear_score_dispatch = [&] {
    state.score_consumers.clear();
    state.score_order.clear();
    state.score_group_offsets.clear();
    state.score_unique.clear();
    state.pending_vectors.clear();
    state.ordered_snapshot_pairs = false;
    state.phase = Stage2SearchIoPhase::idle;
  };
  const auto clear_graph_dispatch = [&] {
    state.graph_consumers.clear();
    state.graph_order.clear();
    state.graph_group_offsets.clear();
    state.graph_unique.clear();
    state.pending_graph.clear();
    for (vec<RemotePtr>& neighbors : state.graph_neighbors) {
      neighbors.clear();
    }
    state.home_expand_rpc_count = 0;
    state.phase = Stage2SearchIoPhase::idle;
  };

  const auto resolve_score_group = [&] (
      size_t group_index,
      const byte_t* candidate_vector) -> size_t {
    lib_assert(group_index + 1 < state.score_group_offsets.size(),
               "Stage2 score dispatch group is out of range");
    const VectorDType dtype = VamanaNode::vector_dtype();
    size_t resolved = 0;
    for (size_t position = state.score_group_offsets[group_index];
         position < state.score_group_offsets[group_index + 1]; ++position) {
      const Stage2ScoreConsumer& consumer =
        state.score_consumers[state.score_order[position]];
      const distance_t distance = distance_between_vectors(
        targets[consumer.search_index].vector_data.data(), dtype,
        candidate_vector, dtype, config);
      resolved += state.continuation.resolve_score_request(
        consumer.search_index, consumer.generation, consumer.pointer,
        std::optional<distance_t>{distance});
    }
    if (resolved != 0) {
      storage_owner_stage2_scored_candidates_.fetch_add(
        resolved, std::memory_order_relaxed);
    }
    return resolved;
  };

  const auto resolve_terminal_score_group = [&] (
      size_t group_index) -> size_t {
    lib_assert(group_index + 1 < state.score_group_offsets.size(),
               "Stage2 terminal score group is out of range");
    size_t resolved = 0;
    for (size_t position = state.score_group_offsets[group_index];
         position < state.score_group_offsets[group_index + 1]; ++position) {
      const Stage2ScoreConsumer& consumer =
        state.score_consumers[state.score_order[position]];
      resolved += state.continuation.resolve_score_request(
        consumer.search_index, consumer.generation, consumer.pointer,
        std::nullopt);
    }
    if (resolved != 0) {
      storage_owner_stage2_scored_candidates_.fetch_add(
        resolved, std::memory_order_relaxed);
    }
    return resolved;
  };

  const auto resolve_graph_group = [&] (
      size_t group_index, span<const RemotePtr> neighbors) -> size_t {
    lib_assert(group_index + 1 < state.graph_group_offsets.size(),
               "Stage2 graph dispatch group is out of range");
    size_t resolved = 0;
    for (size_t position = state.graph_group_offsets[group_index];
         position < state.graph_group_offsets[group_index + 1]; ++position) {
      const Stage2GraphConsumer& consumer =
        state.graph_consumers[state.graph_order[position]];
      resolved += state.continuation.resolve_expand_request(
        consumer.search_index, consumer.generation, neighbors);
    }
    return resolved;
  };

  enum class GraphDecodeResult : u8 {
    invalid_snapshot,
    terminal_snapshot,
    valid,
    malformed_pointer,
  };
  const auto decode_graph = [&](size_t unique_index, RemotePtr pointer,
                                const byte_t* entry) {
    vec<RemotePtr>& neighbors = state.graph_neighbors[unique_index];
    neighbors.clear();
    const u8 stable_count = entry[0];
    const u8 provisional_count =
      vamana::hot_graph::provisional_count(entry);
    const u16 expected = vamana::hot_graph::load_u16_le(entry + 2);
    const u16 actual = vamana::hot_graph::checksum16(
      entry, VamanaNode::hot_graph_entry_size());
    if (stable_count > VamanaNode::R ||
        provisional_count > VamanaNode::provisional_slots() ||
        static_cast<u32>(stable_count) + provisional_count >
          VamanaNode::graph_entry_capacity() ||
        (entry[1] & 0x0e) != 0 ||
        vamana::hot_graph::load_u32_le(entry + 12) != 0 ||
        expected != actual) {
      return GraphDecodeResult::invalid_snapshot;
    }
    if (vamana::hot_graph::load_u32_le(entry + 8) !=
        pointer.incarnation()) {
      return GraphDecodeResult::terminal_snapshot;
    }
    if ((entry[1] & VamanaNode::HOT_GRAPH_DELETED) != 0) {
      return GraphDecodeResult::valid;
    }
    neighbors.reserve(
      static_cast<size_t>(stable_count) + provisional_count);
    bool malformed = false;
    for (u32 index = 0;
         index < static_cast<u32>(stable_count) + provisional_count;
         ++index) {
      const RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
        entry + vamana::hot_graph::neighbor_offset(index),
        VamanaNode::HOT_GRAPH_SHARD_BITS);
      // A counted prefix slot is part of the authoritative adjacency. Null
      // here is malformed; only slots after the counted prefix are padding.
      if (neighbor.is_null()) {
        malformed = true;
        continue;
      }
      if (!storage_node_pointer_addressable(neighbor)) {
        malformed = true;
        report_rejected_graph_pointer(
          index < stable_count
            ? "stage2_async_graph/stable"
            : "stage2_async_graph/provisional",
          neighbor, pointer, index);
        continue;
      }
      neighbors.push_back(neighbor);
    }
    return malformed ? GraphDecodeResult::malformed_pointer
                     : GraphDecodeResult::valid;
  };

  const auto remember_graph_retry = [&](RemotePtr pointer, u32 attempt) {
    for (Stage2GraphRetryState& retry : state.graph_retry_state) {
      if (retry.pointer == pointer) {
        retry.attempt = attempt;
        return;
      }
    }
    state.graph_retry_state.push_back({pointer, attempt});
  };
  const auto forget_graph_retry = [&](RemotePtr pointer) {
    const auto found = std::find_if(
      state.graph_retry_state.begin(), state.graph_retry_state.end(),
      [&](const Stage2GraphRetryState& retry) {
        return retry.pointer == pointer;
      });
    if (found != state.graph_retry_state.end()) {
      *found = state.graph_retry_state.back();
      state.graph_retry_state.pop_back();
    }
  };

  const auto collect_score_dispatch = [&] {
    state.score_consumers.clear();
    state.score_selected_remote.clear();
    state.score_selected_remote.reserve(score_dispatch_limit);
    for (Stage2ScoreRoundRobinCursor& cursor :
         state.score_collect_cursors) {
      cursor.begin_dispatch();
    }
    thread_local vec<u32> remote_wrs_by_peer;
    thread_local vec<u32> remote_pairs_by_peer;
    remote_wrs_by_peer.resize(num_storage_nodes_);
    remote_pairs_by_peer.resize(num_storage_nodes_);
    memory_node_detail::PeerRdmaVectorSnapshotDispatchQuota quota;
    // This collector prepares a new dispatch, so choose its accounting from
    // the transport capability, not from state.ordered_snapshot_pairs (which
    // describes the previously prepared/in-flight dispatch and is cleared at
    // retirement). Dynamic records must consume both WR credits before this
    // wave is materialized.
    quota.reset(
      mixed_snapshots, snapshot_dispatch_limits,
      ordinary_dispatch_limits,
      std::span<u32>{
        remote_wrs_by_peer.data(), remote_wrs_by_peer.size()},
      std::span<u32>{
        remote_pairs_by_peer.data(), remote_pairs_by_peer.size()});
    const size_t search_count = tasks.size();
    size_t last_search = state.round_robin_search % search_count;
    size_t physical_reads = 0;
    bool selected_any = false;
    // Visit every logical request at most once. The continuation bounds this
    // set naturally; only distinct remote records are bounded by lane scratch
    // and transport credit. This lets local work and duplicate consumers pass
    // a full remote wave instead of waiting behind its CQ.
    for (;;) {
      bool examined_this_round = false;
      for (size_t offset = 0; offset < search_count; ++offset) {
        const size_t search_index =
          (state.round_robin_search + offset) % search_count;
        if (state.search_seeded[search_index] == 0) continue;
        const auto requests =
          state.continuation.pending_score_requests(search_index);
        const std::optional<size_t> position =
          state.score_collect_cursors[search_index].take(requests.size());
        if (!position.has_value()) continue;
        examined_this_round = true;
        const PartitionContinuationScoreRequest& request =
          requests[*position];
        const bool remote = storage_node_pointer_addressable(
          request.pointer) && !local_shard(request.pointer.memory_node());
        const bool duplicate_remote = remote &&
          state.score_selected_remote.contains(request.pointer);
        const bool distinct_remote = remote && !duplicate_remote;
        const bool requires_after_header = distinct_remote &&
          !VamanaNode::immutable_base_record(request.pointer);
        const bool accepted =
          stage2_consumer_fits_physical_scratch(
            distinct_remote, physical_reads, score_dispatch_limit) &&
          quota.try_accept(request.pointer.memory_node(), distinct_remote,
                           requires_after_header);
        if (!accepted) {
          // The cursor has advanced, but the continuation request remains
          // unresolved.  This prevents a hot peer at its quota from hiding a
          // later request for another peer, and revisits the skipped request
          // in a subsequent finite dispatch without duplicating it here.
          continue;
        }
        state.score_consumers.push_back({
          request.search_index, request.generation, request.pointer});
        if (distinct_remote) {
          state.score_selected_remote.insert(request.pointer);
          ++physical_reads;
        }
        last_search = search_index;
        selected_any = true;
      }
      if (!examined_this_round) break;
    }
    if (selected_any) {
      state.round_robin_search = (last_search + 1) % search_count;
    }
    return selected_any;
  };

  const auto collect_graph_dispatch = [&] {
    state.graph_consumers.clear();
    state.graph_selected_remote.clear();
    state.graph_selected_remote.reserve(graph_dispatch_limit);
    thread_local vec<u32> remote_items_by_peer;
    remote_items_by_peer.resize(num_storage_nodes_);
    memory_node_detail::PeerRdmaReadDispatchQuota quota;
    quota.reset(
      ordinary_dispatch_limits,
      std::span<u32>{
        remote_items_by_peer.data(), remote_items_by_peer.size()});
    const size_t search_count = tasks.size();
    size_t last_search = state.round_robin_search % search_count;
    size_t physical_reads = 0;
    bool selected_any = false;
    for (size_t offset = 0; offset < search_count; ++offset) {
      const size_t search_index =
        (state.round_robin_search + offset) % search_count;
      if (state.search_seeded[search_index] == 0) continue;
      const auto request =
        state.continuation.pending_expand_request(search_index);
      if (!request.has_value()) continue;
      const bool remote = storage_node_pointer_addressable(
        request->pointer) && !local_shard(request->pointer.memory_node());
      const bool duplicate_remote = remote &&
        state.graph_selected_remote.contains(request->pointer);
      const bool distinct_remote = remote && !duplicate_remote;
      if (!stage2_consumer_fits_physical_scratch(
            distinct_remote, physical_reads, graph_dispatch_limit) ||
          !quota.try_accept(request->pointer.memory_node(), distinct_remote)) {
        continue;
      }
      state.graph_consumers.push_back({
        request->search_index, request->generation, request->pointer});
      if (distinct_remote) {
        state.graph_selected_remote.insert(request->pointer);
        ++physical_reads;
      }
      last_search = search_index;
      selected_any = true;
    }
    if (selected_any) {
      state.round_robin_search = (last_search + 1) % search_count;
    }
    return selected_any;
  };

  const auto prepare_score_dispatch = [&] {
    if (!collect_score_dispatch()) return std::pair{false, false};
    state.score_order.resize(state.score_consumers.size());
    std::iota(state.score_order.begin(), state.score_order.end(), size_t{0});
    std::sort(state.score_order.begin(), state.score_order.end(),
              [&](size_t lhs, size_t rhs) {
                const Stage2ScoreConsumer& left =
                  state.score_consumers[lhs];
                const Stage2ScoreConsumer& right =
                  state.score_consumers[rhs];
                if (left.pointer.raw_address != right.pointer.raw_address) {
                  return left.pointer.raw_address < right.pointer.raw_address;
                }
                if (left.search_index != right.search_index) {
                  return left.search_index < right.search_index;
                }
                return left.generation < right.generation;
              });
    state.score_unique.clear();
    state.score_group_offsets.clear();
    for (size_t position = 0; position < state.score_order.size();
         ++position) {
      const RemotePtr pointer =
        state.score_consumers[state.score_order[position]].pointer;
      if (state.score_unique.empty() ||
          state.score_unique.back() != pointer) {
        state.score_unique.push_back(pointer);
        state.score_group_offsets.push_back(position);
      }
    }
    state.score_group_offsets.push_back(state.score_order.size());
    state.pending_vectors.clear();
    const size_t snapshot_stride = aligned_snapshot_bytes();
    const size_t validation_offset =
      memory_node_detail::storage_owner_snapshot_validation_offset();
    u32 remote_slot = 0;
    bool progressed = false;
    for (size_t group = 0; group < state.score_unique.size(); ++group) {
      const RemotePtr pointer = state.score_unique[group];
      if (!storage_node_pointer_addressable(pointer)) {
        if (!pointer.is_null()) {
          report_rejected_graph_pointer(
            "stage2_async_score/input", pointer, RemotePtr{}, group);
        }
        progressed |= resolve_terminal_score_group(group) != 0;
        continue;
      }
      if (local_shard(pointer.memory_node())) {
        NodeSnapshot snapshot;
        if (read_node_snapshot(pointer, snapshot)) {
          const StableNodeSnapshotState disposition =
            classify_vector_snapshot(
              pointer, snapshot.header, snapshot.header,
              snapshot.slot_incarnation);
          if (disposition == StableNodeSnapshotState::stable &&
              snapshot.vector_data.size() >= VamanaNode::vector_bytes()) {
            progressed |= resolve_score_group(
              group, snapshot.vector_data.data()) != 0;
          } else if (disposition == StableNodeSnapshotState::terminal) {
            progressed |= resolve_terminal_score_group(group) != 0;
          }
          continue;
        }
        const u64 before = load_local_node_header_acquire(pointer);
        const u32 slot_incarnation = *reinterpret_cast<const u32*>(
          local_node_ptr(pointer) + VamanaNode::offset_slot_incarnation());
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 after = load_local_node_header_acquire(pointer);
        if (classify_stable_node_snapshot(
              pointer, before, after, slot_incarnation) ==
            StableNodeSnapshotState::terminal) {
          progressed |= resolve_terminal_score_group(group) != 0;
        }
        continue;
      }

      const size_t scratch_offset =
        static_cast<size_t>(remote_slot++) * snapshot_stride;
      lib_assert(scratch_offset + validation_offset +
                   VamanaNode::HEADER_SIZE <= thread->scratch_stride,
                 "Stage2 vector dispatch exceeded lane scratch");
      byte_t* buffer = thread->coroutine_scratch(scratch_offset);
      state.pending_vectors.push_back(Stage2PendingVectorRead{
        .group_index = group,
        .pointer = pointer,
        .buffer = buffer,
        .after_header = buffer + validation_offset,
        .requires_after_header =
          !VamanaNode::immutable_base_record(pointer),
      });
    }
    if (!state.pending_vectors.empty()) {
      storage_owner_stage2_vector_read_waves_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_stage2_vector_unique_reads_.fetch_add(
        state.pending_vectors.size(), std::memory_order_relaxed);
      state.ordered_snapshot_pairs = mixed_snapshots;
      state.phase = Stage2SearchIoPhase::score_body_ready;
    }
    return std::pair{progressed, !state.pending_vectors.empty()};
  };

  const auto prepare_graph_dispatch = [&] {
    if (!collect_graph_dispatch()) return std::pair{false, false};
    state.graph_order.resize(state.graph_consumers.size());
    std::iota(state.graph_order.begin(), state.graph_order.end(), size_t{0});
    std::sort(state.graph_order.begin(), state.graph_order.end(),
              [&](size_t lhs, size_t rhs) {
                const Stage2GraphConsumer& left =
                  state.graph_consumers[lhs];
                const Stage2GraphConsumer& right =
                  state.graph_consumers[rhs];
                if (left.pointer.raw_address != right.pointer.raw_address) {
                  return left.pointer.raw_address < right.pointer.raw_address;
                }
                if (left.search_index != right.search_index) {
                  return left.search_index < right.search_index;
                }
                return left.generation < right.generation;
              });
    state.graph_unique.clear();
    state.graph_group_offsets.clear();
    for (size_t position = 0; position < state.graph_order.size();
         ++position) {
      const RemotePtr pointer =
        state.graph_consumers[state.graph_order[position]].pointer;
      if (state.graph_unique.empty() ||
          state.graph_unique.back() != pointer) {
        state.graph_unique.push_back(pointer);
        state.graph_group_offsets.push_back(position);
      }
    }
    state.graph_group_offsets.push_back(state.graph_order.size());
    if (state.graph_neighbors.size() < state.graph_unique.size()) {
      state.graph_neighbors.resize(state.graph_unique.size());
    }
    for (size_t group = 0; group < state.graph_unique.size(); ++group) {
      state.graph_neighbors[group].clear();
    }
    state.pending_graph.clear();
    bool progressed = false;
    for (size_t group = 0; group < state.graph_unique.size(); ++group) {
      const RemotePtr pointer = state.graph_unique[group];
      if (!storage_node_pointer_addressable(pointer)) {
        if (!pointer.is_null()) {
          report_rejected_graph_pointer(
            "stage2_async_graph/input", pointer, RemotePtr{}, group);
        }
        progressed |= resolve_graph_group(
          group, span<const RemotePtr>{}) != 0;
        continue;
      }
      if (local_shard(pointer.memory_node())) {
        GraphAdjacency adjacency;
        if (read_graph_adjacency(pointer, adjacency)) {
          vec<RemotePtr>& neighbors = state.graph_neighbors[group];
          if (!adjacency.deleted) {
            neighbors.reserve(adjacency.stable.size() +
                              adjacency.provisional.size());
            neighbors.insert(neighbors.end(), adjacency.stable.begin(),
                             adjacency.stable.end());
            neighbors.insert(neighbors.end(), adjacency.provisional.begin(),
                             adjacency.provisional.end());
          }
          progressed |= resolve_graph_group(
            group, span<const RemotePtr>{neighbors}) != 0;
          forget_graph_retry(pointer);
          continue;
        }
        const u64 before = load_local_node_header_acquire(pointer);
        const u32 slot_incarnation = *reinterpret_cast<const u32*>(
          local_node_ptr(pointer) + VamanaNode::offset_slot_incarnation());
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 after = load_local_node_header_acquire(pointer);
        if (classify_stable_node_snapshot(
              pointer, before, after, slot_incarnation) ==
            StableNodeSnapshotState::terminal) {
          progressed |= resolve_graph_group(
            group, span<const RemotePtr>{}) != 0;
          forget_graph_retry(pointer);
        }
        continue;
      }

      // Remote expansion is executed at the pointer's physical home below.
      // Do not consume one-sided graph scratch here: the home response also
      // carries scores for same-home neighbors and removes the following
      // vector-read round trips without changing the beam's chosen pointer.
    }
    state.home_expand_rpc_count = 0;
    for (Stage2HomeExpandRpc& rpc : state.home_expand_rpcs) {
      rpc.posted = false;
      rpc.complete = false;
      rpc.deadline_ns = 0;
      rpc.request.clear();
    }
    thread_local vec<u32> home_counts;
    thread_local vec<size_t> rpc_by_shard;
    home_counts.assign(num_storage_nodes_, 0);
    rpc_by_shard.assign(num_storage_nodes_, std::numeric_limits<size_t>::max());
    for (const Stage2GraphConsumer& consumer : state.graph_consumers) {
      if (storage_node_pointer_addressable(consumer.pointer) &&
          !local_shard(consumer.pointer.memory_node())) {
        ++home_counts[consumer.pointer.memory_node()];
      }
    }
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      if (home_counts[shard] == 0) continue;
      if (state.home_expand_rpc_count == state.home_expand_rpcs.size()) {
        state.home_expand_rpcs.emplace_back();
      }
      Stage2HomeExpandRpc& rpc =
        state.home_expand_rpcs[state.home_expand_rpc_count];
      rpc.target_shard = shard;
      rpc.item_count = home_counts[shard];
      rpc.request_id = allocate_peer_request_id();
      rpc.request.resize(
        service::storage_owner::stage2_expand_score_request_bytes(
          rpc.item_count));
      std::fill(rpc.request.begin(), rpc.request.end(), byte_t{0});
      rpc_by_shard[shard] = state.home_expand_rpc_count++;
    }
    home_counts.assign(num_storage_nodes_, 0);
    for (const Stage2GraphConsumer& consumer : state.graph_consumers) {
      if (!storage_node_pointer_addressable(consumer.pointer) ||
          local_shard(consumer.pointer.memory_node())) {
        continue;
      }
      const u32 shard = consumer.pointer.memory_node();
      Stage2HomeExpandRpc& rpc = state.home_expand_rpcs[rpc_by_shard[shard]];
      const u32 item_index = home_counts[shard]++;
      auto* items = service::storage_owner::stage2_expand_score_items(
        rpc.request.data());
      items[item_index] = service::storage_owner::Stage2ExpandScoreItem{
        .pointer_raw = consumer.pointer.raw_address,
        .generation = consumer.generation,
        .search_index = static_cast<u32>(consumer.search_index),
      };
      byte_t* queries = service::storage_owner::stage2_expand_score_queries(
        rpc.request.data(), rpc.item_count);
      std::memcpy(
        queries + static_cast<size_t>(item_index) *
          VamanaNode::vector_bytes(),
        targets[consumer.search_index].vector_data.data(),
        VamanaNode::vector_bytes());
    }
    if (state.home_expand_rpc_count != 0) {
      storage_owner_stage2_home_rpc_batches_.fetch_add(
        state.home_expand_rpc_count, std::memory_order_relaxed);
      u64 home_rpc_items = 0;
      for (size_t rpc_index = 0;
           rpc_index < state.home_expand_rpc_count; ++rpc_index) {
        home_rpc_items += state.home_expand_rpcs[rpc_index].item_count;
      }
      storage_owner_stage2_home_rpc_items_.fetch_add(
        home_rpc_items, std::memory_order_relaxed);
      storage_owner_stage2_graph_read_waves_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_stage2_graph_unique_reads_.fetch_add(
        state.graph_consumers.size(), std::memory_order_relaxed);
      state.phase = Stage2SearchIoPhase::graph_home_pending;
    }
    return std::pair{progressed, state.home_expand_rpc_count != 0};
  };

  thread_local vec<PeerReadRequest> read_requests;
  thread_local vec<PeerReadSnapshotRequest> snapshot_requests;
  thread_local vec<byte_t> home_response_payload;
  thread_local vec<RemotePtr> home_neighbors;
  u8 idle_attempt_mask = 0;

  // Only a bounded transport dispatch is synchronized on one CQ.  Once its
  // WRs retire, every stable/terminal consumer is resolved immediately and
  // retryable consumers simply remain in their own search generation.
  for (;;) {
    if (state.continuation.all_complete()) {
      const auto& final_beams = state.continuation.results();
      const auto& exhausted =
        state.continuation.budget_exhausted_results();
      const auto& expansions =
        state.continuation.expansion_count_results();
      u64 total_expansions = 0;
      u64 exhausted_count = 0;
      lib_assert(final_beams.size() == tasks.size(),
                 "asynchronous Stage2 result count changed");
      for (size_t item = 0; item < final_beams.size(); ++item) {
        total_expansions += expansions[item];
        exhausted_count += exhausted[item];
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
        exhausted_count, std::memory_order_relaxed);
      const size_t retained_capacity = std::max<size_t>(
        1024, static_cast<size_t>(construction_width) * 8);
      state.reset_completed(retained_capacity);
      return Stage2SearchAdvanceResult::complete;
    }

    if (state.phase == Stage2SearchIoPhase::score_body_pending) {
      if (!thread->is_ready(thread->running_coroutine)) {
        return Stage2SearchAdvanceResult::waiting_rdma;
      }
      if (state.ordered_snapshot_pairs) {
        for (Stage2PendingVectorRead& read : state.pending_vectors) {
          const u64 before = *reinterpret_cast<const u64*>(read.buffer);
          const u32 slot_incarnation = *reinterpret_cast<const u32*>(
            read.buffer + VamanaNode::offset_slot_incarnation());
          const u64 after = read.requires_after_header
            ? *reinterpret_cast<const u64*>(read.after_header)
            : before;
          const StableNodeSnapshotState disposition =
            classify_vector_snapshot(
              read.pointer, before, after, slot_incarnation);
          if (disposition == StableNodeSnapshotState::stable) {
            resolve_score_group(
              read.group_index,
              read.buffer + VamanaNode::offset_vector());
          } else if (disposition == StableNodeSnapshotState::terminal) {
            resolve_terminal_score_group(read.group_index);
          } else {
            lib_assert(read.attempt != std::numeric_limits<u32>::max(),
                       "Stage2 vector retry counter overflow");
          }
        }
        clear_score_dispatch();
        state.prefer_graph = true;
        idle_attempt_mask = 0;
        continue;
      }
      size_t dynamic_count = 0;
      for (Stage2PendingVectorRead& read : state.pending_vectors) {
        read.before = *reinterpret_cast<const u64*>(read.buffer);
        read.slot_incarnation = *reinterpret_cast<const u32*>(
          read.buffer + VamanaNode::offset_slot_incarnation());
        if (!read.requires_after_header) {
          const StableNodeSnapshotState disposition =
            classify_vector_snapshot(
              read.pointer, read.before, read.before,
              read.slot_incarnation);
          if (disposition == StableNodeSnapshotState::stable) {
            resolve_score_group(
              read.group_index,
              read.buffer + VamanaNode::offset_vector());
          } else if (disposition == StableNodeSnapshotState::terminal) {
            resolve_terminal_score_group(read.group_index);
          }
          continue;
        }
        if (dynamic_count !=
            static_cast<size_t>(&read - state.pending_vectors.data())) {
          state.pending_vectors[dynamic_count] = read;
        }
        ++dynamic_count;
      }
      state.pending_vectors.resize(dynamic_count);
      if (state.pending_vectors.empty()) {
        clear_score_dispatch();
        state.prefer_graph = true;
        idle_attempt_mask = 0;
        continue;
      }
      state.phase = Stage2SearchIoPhase::score_header_ready;
      continue;
    }

    if (state.phase == Stage2SearchIoPhase::score_header_pending) {
      if (!thread->is_ready(thread->running_coroutine)) {
        return Stage2SearchAdvanceResult::waiting_rdma;
      }
      for (Stage2PendingVectorRead& read : state.pending_vectors) {
        const u64 after = *reinterpret_cast<const u64*>(read.buffer);
        const StableNodeSnapshotState disposition = classify_vector_snapshot(
          read.pointer, read.before, after, read.slot_incarnation);
        if (disposition == StableNodeSnapshotState::stable) {
          resolve_score_group(
            read.group_index,
            read.buffer + VamanaNode::offset_vector());
        } else if (disposition == StableNodeSnapshotState::terminal) {
          resolve_terminal_score_group(read.group_index);
        } else {
          lib_assert(read.attempt != std::numeric_limits<u32>::max(),
                     "Stage2 vector retry counter overflow");
        }
      }
      clear_score_dispatch();
      state.prefer_graph = true;
      idle_attempt_mask = 0;
      continue;
    }

    if (state.phase == Stage2SearchIoPhase::score_header_ready) {
      read_requests.clear();
      for (const Stage2PendingVectorRead& read : state.pending_vectors) {
        read_requests.push_back(PeerReadRequest{
          .shard_id = read.pointer.memory_node(),
          .remote_offset = read.pointer.byte_offset(),
          .destination = read.buffer,
          .bytes = VamanaNode::HEADER_SIZE,
        });
      }
      lib_assert(read_requests.size() <=
                   ordinary_dispatch_limits.global_items,
                 "Stage2 header dispatch exceeds RDMA credit");
      if (!try_post_peer_reads_async(
            *thread, span<const PeerReadRequest>{read_requests})) {
        return Stage2SearchAdvanceResult::waiting_rdma;
      }
      state.phase = Stage2SearchIoPhase::score_header_pending;
      return Stage2SearchAdvanceResult::posted_rdma;
    }

    if (state.phase == Stage2SearchIoPhase::score_body_ready) {
      if (state.ordered_snapshot_pairs) {
        snapshot_requests.clear();
        for (const Stage2PendingVectorRead& read : state.pending_vectors) {
          const PeerReadRequest body{
            .shard_id = read.pointer.memory_node(),
            .remote_offset = read.pointer.byte_offset(),
            .destination = read.buffer,
            .bytes = VamanaNode::size_until_vector_end(),
          };
          snapshot_requests.push_back(PeerReadSnapshotRequest{
            .full_snapshot = body,
            .after_header = read.requires_after_header
              ? std::optional<PeerReadRequest>{PeerReadRequest{
                  .shard_id = read.pointer.memory_node(),
                  .remote_offset = read.pointer.byte_offset(),
                  .destination = read.after_header,
                  .bytes = VamanaNode::HEADER_SIZE,
                }}
              : std::nullopt,
          });
        }
        if (!try_post_peer_snapshot_reads_async(
              *thread,
              span<const PeerReadSnapshotRequest>{snapshot_requests})) {
          return Stage2SearchAdvanceResult::waiting_rdma;
        }
      } else {
        read_requests.clear();
        for (const Stage2PendingVectorRead& read : state.pending_vectors) {
          read_requests.push_back(PeerReadRequest{
            .shard_id = read.pointer.memory_node(),
            .remote_offset = read.pointer.byte_offset(),
            .destination = read.buffer,
            .bytes = VamanaNode::size_until_vector_end(),
          });
        }
        lib_assert(read_requests.size() <=
                     ordinary_dispatch_limits.global_items,
                   "Stage2 vector dispatch exceeds RDMA credit");
        if (!try_post_peer_reads_async(
              *thread, span<const PeerReadRequest>{read_requests})) {
          return Stage2SearchAdvanceResult::waiting_rdma;
        }
      }
      state.phase = Stage2SearchIoPhase::score_body_pending;
      return Stage2SearchAdvanceResult::posted_rdma;
    }

    if (state.phase == Stage2SearchIoPhase::graph_home_pending) {
      bool all_complete = true;
      for (size_t rpc_index = 0;
           rpc_index < state.home_expand_rpc_count; ++rpc_index) {
        Stage2HomeExpandRpc& rpc = state.home_expand_rpcs[rpc_index];
        if (rpc.complete) continue;
        all_complete = false;
        if (!rpc.posted) {
          const size_t request_bytes = rpc.request.size();
          rpc.posted = try_post_peer_rpc_request_attempt(
            rpc.target_shard,
            service::storage_owner::PeerRpcType::stage2_expand_score_request,
            service::storage_owner::PeerRpcType::stage2_expand_score_response,
            rpc.request_id, rpc.item_count,
            rpc.request.data() + sizeof(service::storage_owner::PeerRpcHeader),
            request_bytes - sizeof(service::storage_owner::PeerRpcHeader),
            request_bytes, PeerRpcSendClass::graph_update);
          if (rpc.posted) {
            rpc.deadline_ns = steady_now_ns() +
              static_cast<u64>(config.storage_owner_rpc_timeout_ms) *
                1000ull * 1000ull;
          }
          if (!rpc.posted) continue;
        }

        service::storage_owner::PeerRpcHeader response_header{};
        PeerResponseLease response_lease{};
        home_response_payload.clear();
        const TryPeerResponse response = try_consume_peer_rpc_response(
          rpc.request_id, rpc.target_shard,
          service::storage_owner::PeerRpcType::stage2_expand_score_response,
          rpc.item_count, response_header, home_response_payload,
          response_lease);
        if (response == TryPeerResponse::pending) {
          if (steady_now_ns() >= rpc.deadline_ns) {
            cancel_peer_rpc_response(rpc.request_id);
            rpc.posted = false;
            storage_owner_maintenance_rpc_timeouts_.fetch_add(
              1, std::memory_order_relaxed);
          }
          continue;
        }
        const size_t expected_bytes =
          service::storage_owner::stage2_expand_score_response_bytes(
            rpc.item_count);
        bool valid = response == TryPeerResponse::success &&
          home_response_payload.size() == expected_bytes &&
          response_header.magic == service::storage_owner::kPeerRpcMagic &&
          response_header.version == service::storage_owner::kPeerRpcVersion &&
          response_header.type == static_cast<u32>(
            service::storage_owner::PeerRpcType::stage2_expand_score_response) &&
          response_header.source_shard == rpc.target_shard &&
          response_header.item_count == rpc.item_count &&
          response_header.request_id == rpc.request_id &&
          response_header.status == static_cast<u32>(
            service::storage_owner::InsertStatus::ok) &&
          response_header.reserved == 0;
        const auto* request_items =
          service::storage_owner::stage2_expand_score_items(
            rpc.request.data());
        const auto* results = valid
          ? service::storage_owner::stage2_expand_score_results(
              home_response_payload.data())
          : nullptr;
        const auto* neighbors = valid
          ? service::storage_owner::stage2_expand_score_neighbors(
              home_response_payload.data(), rpc.item_count)
          : nullptr;
        const size_t neighbor_stride = VamanaNode::graph_entry_capacity();
        for (u32 item_index = 0; valid && item_index < rpc.item_count;
             ++item_index) {
          const auto& request = request_items[item_index];
          const auto& result = results[item_index];
          valid = result.pointer_raw == request.pointer_raw &&
            result.generation == request.generation &&
            result.search_index == request.search_index &&
            result.search_index < tasks.size() &&
            result.neighbor_count <= neighbor_stride &&
            result.reserved == 0 &&
            result.disposition <= static_cast<u32>(
              service::storage_owner::Stage2HomeDisposition::terminal);
          for (u32 neighbor_index = 0;
               valid && neighbor_index < result.neighbor_count;
               ++neighbor_index) {
            const auto& neighbor = neighbors[
              static_cast<size_t>(item_index) * neighbor_stride +
              neighbor_index];
            valid = neighbor.disposition <= static_cast<u32>(
              service::storage_owner::Stage2HomeDisposition::unscored);
          }
        }
        if (!valid) {
          if (response_lease.valid()) {
            lib_assert(rearm_peer_rpc_response(response_lease),
                       "invalid Stage2 home response lost its lease");
          }
          rpc.posted = false;
          continue;
        }

        lib_assert(acknowledge_peer_rpc_response(response_lease),
                   "validated Stage2 home response lost its lease");
        for (u32 item_index = 0; item_index < rpc.item_count; ++item_index) {
          const auto& result = results[item_index];
          const auto disposition = static_cast<
            service::storage_owner::Stage2HomeDisposition>(
              result.disposition);
          if (disposition ==
              service::storage_owner::Stage2HomeDisposition::retryable) {
            continue;
          }
          home_neighbors.clear();
          if (disposition ==
              service::storage_owner::Stage2HomeDisposition::stable) {
            home_neighbors.reserve(result.neighbor_count);
            for (u32 neighbor_index = 0;
                 neighbor_index < result.neighbor_count; ++neighbor_index) {
              home_neighbors.push_back(RemotePtr{neighbors[
                static_cast<size_t>(item_index) * neighbor_stride +
                neighbor_index].pointer_raw});
            }
          }
          if (!state.continuation.resolve_expand_request(
                result.search_index, result.generation,
                span<const RemotePtr>{home_neighbors})) {
            continue;
          }
          forget_graph_retry(RemotePtr{result.pointer_raw});
          const u64 score_generation =
            state.continuation.generation(result.search_index);
          for (u32 neighbor_index = 0;
               neighbor_index < result.neighbor_count; ++neighbor_index) {
            const auto& neighbor = neighbors[
              static_cast<size_t>(item_index) * neighbor_stride +
              neighbor_index];
            const RemotePtr pointer{neighbor.pointer_raw};
            const auto neighbor_disposition = static_cast<
              service::storage_owner::Stage2HomeDisposition>(
                neighbor.disposition);
            bool resolved = false;
            if (neighbor_disposition ==
                service::storage_owner::Stage2HomeDisposition::stable) {
              resolved = state.continuation.resolve_score_request(
                result.search_index, score_generation, pointer,
                std::optional<distance_t>{neighbor.distance});
              if (resolved) {
                storage_owner_stage2_home_scored_neighbors_.fetch_add(
                  1, std::memory_order_relaxed);
              }
            } else if (neighbor_disposition ==
                       service::storage_owner::Stage2HomeDisposition::terminal) {
              resolved = state.continuation.resolve_score_request(
                result.search_index, score_generation, pointer,
                std::nullopt);
            }
            if (resolved) {
              storage_owner_stage2_scored_candidates_.fetch_add(
                1, std::memory_order_relaxed);
            }
          }
        }
        rpc.complete = true;
      }
      all_complete = std::all_of(
        state.home_expand_rpcs.begin(),
        state.home_expand_rpcs.begin() + state.home_expand_rpc_count,
        [](const Stage2HomeExpandRpc& rpc) { return rpc.complete; });
      if (!all_complete) return Stage2SearchAdvanceResult::waiting_rdma;
      clear_graph_dispatch();
      state.prefer_graph = false;
      idle_attempt_mask = 0;
      continue;
    }

    if (state.phase == Stage2SearchIoPhase::graph_pending) {
      if (!thread->is_ready(thread->running_coroutine)) {
        return Stage2SearchAdvanceResult::waiting_rdma;
      }
      constexpr u32 kMalformedRetryAttempts = 3;
      for (Stage2PendingGraphRead& read : state.pending_graph) {
        const GraphDecodeResult decoded = decode_graph(
          read.unique_index, read.pointer, read.buffer);
        if (decoded == GraphDecodeResult::valid) {
          resolve_graph_group(
            read.unique_index,
            span<const RemotePtr>{state.graph_neighbors[read.unique_index]});
          forget_graph_retry(read.pointer);
          continue;
        }
        if (decoded == GraphDecodeResult::terminal_snapshot) {
          resolve_graph_group(
            read.unique_index, span<const RemotePtr>{});
          forget_graph_retry(read.pointer);
          continue;
        }
        if (decoded == GraphDecodeResult::malformed_pointer) {
          if (read.attempt >= kMalformedRetryAttempts - 1) {
            lib_failure(
              "stable Stage2 graph contains a malformed remote pointer");
          }
          lib_assert(read.attempt != std::numeric_limits<u32>::max(),
                     "Stage2 graph retry counter overflow");
          remember_graph_retry(read.pointer, read.attempt + 1);
        }
        // An invalid checksum/header is an optimistic torn observation, not
        // one of the three stable malformed records required for fail-stop.
        // Leave the consumer unresolved and retain its prior diagnostic count.
      }
      clear_graph_dispatch();
      state.prefer_graph = false;
      idle_attempt_mask = 0;
      continue;
    }

    if (state.phase == Stage2SearchIoPhase::graph_ready) {
      read_requests.clear();
      for (const Stage2PendingGraphRead& read : state.pending_graph) {
        read_requests.push_back(PeerReadRequest{
          .shard_id = read.pointer.memory_node(),
          .remote_offset = VamanaNode::hot_graph_entry_offset(read.pointer),
          .destination = read.buffer,
          .bytes = VamanaNode::hot_graph_entry_size(),
        });
      }
      lib_assert(read_requests.size() <=
                   ordinary_dispatch_limits.global_items,
                 "Stage2 graph dispatch exceeds RDMA credit");
      if (!try_post_peer_reads_async(
            *thread, span<const PeerReadRequest>{read_requests})) {
        return Stage2SearchAdvanceResult::waiting_rdma;
      }
      state.phase = Stage2SearchIoPhase::graph_pending;
      return Stage2SearchAdvanceResult::posted_rdma;
    }

    lib_assert(state.phase == Stage2SearchIoPhase::idle,
               "Stage2 dispatcher entered an invalid I/O phase");
    const bool has_score =
      !state.continuation.pending_score_requests().empty();
    const bool has_graph =
      !state.continuation.pending_expand_requests().empty();
    const bool score_available = has_score && (idle_attempt_mask & 1u) == 0;
    const bool graph_available = has_graph && (idle_attempt_mask & 2u) == 0;
    if (!score_available && !graph_available) {
      // No ready work means either every active search is complete while a
      // handoff is retryable, or both retry-only kinds were attempted once in
      // this scheduler turn.  Yield instead of spinning on NODE_LOCK/torn
      // snapshots.
      return Stage2SearchAdvanceResult::waiting_rdma;
    }

    const bool choose_graph = graph_available &&
      (!score_available || state.prefer_graph);
    std::pair<bool, bool> prepared;
    if (choose_graph) {
      idle_attempt_mask |= 2u;
      prepared = prepare_graph_dispatch();
      state.prefer_graph = false;
      if (!prepared.second) clear_graph_dispatch();
    } else {
      idle_attempt_mask |= 1u;
      prepared = prepare_score_dispatch();
      state.prefer_graph = true;
      if (!prepared.second) clear_score_dispatch();
    }
    if (prepared.second) {
      // The next loop iteration posts the prepared WRs.  No scratch address
      // is reused until the corresponding *_pending phase observes CQ ready.
      continue;
    }
    if (prepared.first) {
      // Stable/terminal local consumers may have exposed a new generation.
      idle_attempt_mask = 0;
    }
  }
}
