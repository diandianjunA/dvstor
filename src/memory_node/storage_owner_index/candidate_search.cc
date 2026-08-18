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
    state.graph_prefetch_cache.resize(tasks.size());
    state.speculative_score_rpcs.resize(num_storage_nodes_);
    for (u32 peer = 0; peer < num_storage_nodes_; ++peer) {
      state.speculative_score_rpcs[peer].target_shard = peer;
    }
    state.speculative_peer_cursor = static_cast<u32>(
      tasks[0].maintenance_sequence % num_storage_nodes_);
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
    state.score_home_rpc_count = 0;
    state.ordered_snapshot_pairs = false;
    state.score_many_dispatch = false;
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
      if (consumer.speculative) {
        state.resolve_graph_prefetch_score(
          consumer.search_index, consumer.expansion_pointer,
          consumer.pointer, std::optional<distance_t>{distance},
          static_cast<u32>(
            service::storage_owner::Stage2HomeDisposition::stable));
      } else {
        resolved += state.continuation.resolve_score_request(
          consumer.search_index, consumer.generation, consumer.pointer,
          std::optional<distance_t>{distance});
      }
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
      if (consumer.speculative) {
        state.resolve_graph_prefetch_score(
          consumer.search_index, consumer.expansion_pointer,
          consumer.pointer, std::nullopt,
          static_cast<u32>(
            service::storage_owner::Stage2HomeDisposition::terminal));
      } else {
        resolved += state.continuation.resolve_score_request(
          consumer.search_index, consumer.generation, consumer.pointer,
          std::nullopt);
      }
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
      if (consumer.speculative) continue;
      resolved += state.continuation.resolve_expand_request(
        consumer.search_index, consumer.generation, consumer.pointer,
        neighbors);
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
    thread_local vec<u32> score_many_items_by_peer;
    thread_local vec<size_t> score_many_pending_by_peer;
    thread_local vec<size_t> authoritative_pending_by_peer;
    thread_local vec<u8> score_many_peer_eligible;
    thread_local vec<u8> authoritative_peer_selected;
    remote_wrs_by_peer.resize(num_storage_nodes_);
    remote_pairs_by_peer.resize(num_storage_nodes_);
    score_many_items_by_peer.resize(num_storage_nodes_);
    score_many_pending_by_peer.assign(num_storage_nodes_, 0);
    authoritative_pending_by_peer.assign(num_storage_nodes_, 0);
    score_many_peer_eligible.assign(num_storage_nodes_, 0);
    authoritative_peer_selected.assign(num_storage_nodes_, 0);
    const u32 score_many_item_limit =
      std::max<u32>(1, config.storage_owner_search_snapshot_batch);
    state.score_many_dispatch = false;
    if (config.storage_owner_stage2_score_many) {
      for (size_t search_index = 0; search_index < tasks.size();
           ++search_index) {
        if (state.search_seeded[search_index] == 0) continue;
        for (const PartitionContinuationScoreRequest& request :
             state.continuation.pending_score_requests(search_index)) {
          if (storage_node_pointer_addressable(request.pointer) &&
              !local_shard(request.pointer.memory_node())) {
            ++score_many_pending_by_peer[request.pointer.memory_node()];
          }
        }
      }
      authoritative_pending_by_peer = score_many_pending_by_peer;
      for (size_t search_index = 0;
           search_index < state.graph_prefetch_cache.size(); ++search_index) {
        for (const Stage2PrefetchedGraphExpansion& expansion :
             state.graph_prefetch_cache[search_index]) {
          for (const Stage2PrefetchedGraphNeighbor& neighbor :
               expansion.neighbors) {
            if (neighbor.score_prefetched ||
                neighbor.independent_score_prefetched ||
                neighbor.score_prefetch_issues != 0 ||
                neighbor.independent_score_issues != 0 ||
                !storage_node_pointer_addressable(neighbor.pointer) ||
                local_shard(neighbor.pointer.memory_node())) {
              continue;
            }
            const auto disposition = static_cast<
              service::storage_owner::Stage2HomeDisposition>(
                neighbor.disposition);
            if (disposition ==
                  service::storage_owner::Stage2HomeDisposition::stable ||
                disposition ==
                  service::storage_owner::Stage2HomeDisposition::terminal) {
              continue;
            }
            const u32 peer = neighbor.pointer.memory_node();
            if (authoritative_pending_by_peer[peer] != 0) {
              ++score_many_pending_by_peer[peer];
            }
          }
        }
      }
      for (u32 peer = 0; peer < num_storage_nodes_; ++peer) {
        score_many_peer_eligible[peer] =
          stage2_score_many_peer_eligible(
            score_many_pending_by_peer[peer], score_many_item_limit);
        state.score_many_dispatch |= score_many_peer_eligible[peer] != 0;
      }
    }
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
    Stage2ScoreManyDispatchQuota score_many_quota;
    score_many_quota.reset(
      std::span<u32>{score_many_items_by_peer.data(),
                     score_many_items_by_peer.size()},
      score_many_item_limit);
    const size_t search_count = tasks.size();
    size_t last_search = state.round_robin_search % search_count;
    size_t physical_reads = 0;
    bool selected_any = false;
    // Visit every logical request at most once. In the one-sided path only
    // distinct remote records consume lane scratch/read credit; score-many
    // instead counts logical wire items per destination RPC. In either mode a
    // full peer cannot hide local work or a different peer in this pass.
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
        // Independent lookahead scoring is first-completion-wins rather than
        // an authoritative dependency owner. The ordinary exact request is
        // always eligible here; whichever response resolves this
        // generation/pointer first wins and the other is rejected
        // idempotently by resolve_score_request(). Thus a slow speculative
        // peer can never add latency to the correctness path.
        const bool remote = storage_node_pointer_addressable(
          request.pointer) && !local_shard(request.pointer.memory_node());
        const bool duplicate_remote = remote &&
          !state.score_many_dispatch &&
          state.score_selected_remote.contains(request.pointer);
        const bool distinct_remote = remote && !duplicate_remote;
        const bool requires_after_header = distinct_remote &&
          !VamanaNode::immutable_base_record(request.pointer);
        const bool accepted = state.score_many_dispatch
          ? (!remote ||
             (score_many_peer_eligible[request.pointer.memory_node()] != 0 &&
              score_many_quota.try_accept(
                request.pointer.memory_node(), true)))
          : stage2_consumer_fits_physical_scratch(
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
          request.search_index, request.generation, request.pointer,
          false, RemotePtr{}});
        if (remote) {
          authoritative_peer_selected[request.pointer.memory_node()] = 1;
        }
        if (!state.score_many_dispatch && distinct_remote) {
          state.score_selected_remote.insert(request.pointer);
          ++physical_reads;
        }
        last_search = search_index;
        selected_any = true;
      }
      if (!examined_this_round) break;
    }
    // Score prefetch consumes only unused capacity in an already-required
    // score wave.  One-sided READs may use any peer with spare per-peer/global
    // credit because all of them retire on the same CQ dependency; this is
    // the key overlap that turns the high-hit ordered graph lookahead into a
    // score pipeline.  Score-many remains restricted to an authoritative
    // destination so speculation never creates another two-sided RPC tail.
    // Cached exact distances are committed only if their expansion later
    // becomes authoritative.
    const u64 score_prefetch_hits =
      storage_owner_stage2_score_prefetch_hits_.load(
        std::memory_order_relaxed);
    const u64 score_prefetch_wasted =
      storage_owner_stage2_score_prefetch_wasted_.load(
        std::memory_order_relaxed);
    const u64 score_prefetch_outcomes =
      score_prefetch_hits + score_prefetch_wasted;
    u64 next_score_feedback =
      storage_owner_stage2_score_feedback_next_outcome_.load(
        std::memory_order_relaxed);
    if (score_prefetch_outcomes >= next_score_feedback &&
        storage_owner_stage2_score_feedback_next_outcome_
          .compare_exchange_strong(
            next_score_feedback, score_prefetch_outcomes + 512,
            std::memory_order_relaxed, std::memory_order_relaxed)) {
      const u64 base_hits =
        storage_owner_stage2_score_feedback_base_hits_.exchange(
          score_prefetch_hits, std::memory_order_relaxed);
      const u64 base_wasted =
        storage_owner_stage2_score_feedback_base_wasted_.exchange(
          score_prefetch_wasted, std::memory_order_relaxed);
      storage_owner_stage2_score_prefetch_enabled_.store(
        stage2_score_prefetch_enabled(
          score_prefetch_hits >= base_hits
            ? score_prefetch_hits - base_hits : 0,
          score_prefetch_wasted >= base_wasted
            ? score_prefetch_wasted - base_wasted : 0),
        std::memory_order_relaxed);
    }
    u64 speculative_scores = 0;
    if (selected_any &&
        storage_owner_stage2_score_prefetch_enabled_.load(
          std::memory_order_relaxed)) {
      for (size_t search_index = 0;
           search_index < state.graph_prefetch_cache.size(); ++search_index) {
        if (state.search_seeded[search_index] == 0) continue;
        for (Stage2PrefetchedGraphExpansion& expansion :
             state.graph_prefetch_cache[search_index]) {
          for (Stage2PrefetchedGraphNeighbor& neighbor :
               expansion.neighbors) {
            const auto disposition = static_cast<
              service::storage_owner::Stage2HomeDisposition>(
                neighbor.disposition);
            if (neighbor.score_prefetched ||
                neighbor.independent_score_prefetched ||
                neighbor.score_prefetch_issues != 0 ||
                neighbor.independent_score_issues != 0 ||
                disposition ==
                  service::storage_owner::Stage2HomeDisposition::stable ||
                disposition ==
                  service::storage_owner::Stage2HomeDisposition::terminal ||
                !storage_node_pointer_addressable(neighbor.pointer) ||
                local_shard(neighbor.pointer.memory_node())) {
              continue;
            }
            const u32 peer = neighbor.pointer.memory_node();
            if (!stage2_score_prefetch_peer_eligible(
                  state.score_many_dispatch,
                  authoritative_peer_selected[peer] != 0)) {
              continue;
            }
            const bool duplicate_remote = !state.score_many_dispatch &&
              state.score_selected_remote.contains(neighbor.pointer);
            const bool distinct_remote = !duplicate_remote;
            const bool requires_after_header = distinct_remote &&
              !VamanaNode::immutable_base_record(neighbor.pointer);
            if (!state.score_many_dispatch && requires_after_header) {
              // The one-sided dispatcher deliberately retires immutable and
              // mutable records in separate waves. A mutable speculative item
              // would be dropped at that boundary, so leave it authoritative.
              continue;
            }
            const bool accepted = state.score_many_dispatch
              ? (score_many_peer_eligible[peer] != 0 &&
                 score_many_quota.try_accept(peer, true))
              : stage2_consumer_fits_physical_scratch(
                  distinct_remote, physical_reads, score_dispatch_limit) &&
                quota.try_accept(peer, distinct_remote, false);
            if (!accepted) continue;
            state.score_consumers.push_back({
              search_index,
              state.continuation.generation(search_index),
              neighbor.pointer,
              true,
              expansion.pointer,
            });
            if (!state.score_many_dispatch && distinct_remote) {
              state.score_selected_remote.insert(neighbor.pointer);
              ++physical_reads;
            }
            lib_assert(
              neighbor.score_prefetch_issues !=
                std::numeric_limits<u32>::max(),
              "Stage2 score prefetch issue counter overflow");
            ++neighbor.score_prefetch_issues;
            ++speculative_scores;
          }
        }
      }
    }
    if (speculative_scores != 0) {
      storage_owner_stage2_score_prefetch_issued_.fetch_add(
        speculative_scores, std::memory_order_relaxed);
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
    thread_local vec<u32> wire_items_by_peer;
    remote_items_by_peer.resize(num_storage_nodes_);
    wire_items_by_peer.assign(num_storage_nodes_, 0);
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
        request->search_index, request->generation, request->pointer,
        false});
      if (remote) ++wire_items_by_peer[request->pointer.memory_node()];
      if (distinct_remote) {
        state.graph_selected_remote.insert(request->pointer);
        ++physical_reads;
      }
      last_search = search_index;
      selected_any = true;
    }
    // Ordered issue is piggyback-only: it may occupy unused items in any peer
    // RPC already selected by this context wave, but it can neither create a
    // new destination RPC nor split an existing one.  Ranking candidates
    // globally avoids wasting a high-confidence preview slot merely because
    // the next nearest candidate lives on a different already-active peer.
    const u64 prefetch_hits =
      storage_owner_stage2_graph_prefetch_hits_.load(
        std::memory_order_relaxed);
    const u64 prefetch_wasted =
      storage_owner_stage2_graph_prefetch_wasted_.load(
        std::memory_order_relaxed);
    const u64 prefetch_outcomes = prefetch_hits + prefetch_wasted;
    u64 next_feedback_outcome =
      storage_owner_stage2_graph_feedback_next_outcome_.load(
        std::memory_order_relaxed);
    if (prefetch_outcomes >= next_feedback_outcome &&
        storage_owner_stage2_graph_feedback_next_outcome_
          .compare_exchange_strong(
            next_feedback_outcome, prefetch_outcomes + 512,
            std::memory_order_relaxed, std::memory_order_relaxed)) {
      const u64 base_hits =
        storage_owner_stage2_graph_feedback_base_hits_.exchange(
          prefetch_hits, std::memory_order_relaxed);
      const u64 base_wasted =
        storage_owner_stage2_graph_feedback_base_wasted_.exchange(
          prefetch_wasted, std::memory_order_relaxed);
      storage_owner_stage2_graph_issue_width_current_.store(
        stage2_ordered_issue_width(
          prefetch_hits >= base_hits ? prefetch_hits - base_hits : 0,
          prefetch_wasted >= base_wasted
            ? prefetch_wasted - base_wasted : 0,
          config.storage_owner_stage2_graph_issue_width),
        std::memory_order_relaxed);
    }
    const u32 issue_width = std::min(
      config.storage_owner_stage2_graph_issue_width,
      storage_owner_stage2_graph_issue_width_current_.load(
        std::memory_order_relaxed));
    const size_t cache_capacity =
      static_cast<size_t>(config.storage_owner_stage2_graph_issue_width);
    u64 speculative_added = 0;
    if (issue_width > 1) {
      thread_local vec<RemotePtr> prefetch_candidates;
      const size_t authoritative_count = state.graph_consumers.size();
      for (size_t consumer_index = 0;
           consumer_index < authoritative_count; ++consumer_index) {
        const Stage2GraphConsumer authoritative =
          state.graph_consumers[consumer_index];
        if (!storage_node_pointer_addressable(authoritative.pointer) ||
            local_shard(authoritative.pointer.memory_node())) {
          continue;
        }
        const size_t cached =
          state.graph_prefetch_size(authoritative.search_index);
        if (cached >= cache_capacity) continue;
        const size_t per_search_limit = std::min<size_t>(
          static_cast<size_t>(issue_width - 1),
          cache_capacity - cached);
        if (per_search_limit == 0) continue;
        prefetch_candidates.clear();
        state.continuation.append_expand_prefetch_candidates(
          authoritative.search_index, construction_width,
          prefetch_candidates);
        size_t admitted = 0;
        for (const RemotePtr pointer : prefetch_candidates) {
          if (admitted == per_search_limit) break;
          if (!storage_node_pointer_addressable(pointer) ||
              local_shard(pointer.memory_node())) continue;
          const u32 peer = pointer.memory_node();
          // Do not turn previewing into a new transport dependency. The
          // per-peer exact combiner can relax this restriction later; today
          // an authoritative item in this wave must already own the RPC.
          if (wire_items_by_peer[peer] == 0 ||
              wire_items_by_peer[peer] >= config.storage_owner_batch_max) {
            continue;
          }
          if (state.graph_prefetch_contains(
                authoritative.search_index, pointer)) {
            continue;
          }
          state.graph_consumers.push_back({
            authoritative.search_index, authoritative.generation, pointer,
            true});
          ++wire_items_by_peer[peer];
          ++admitted;
          ++speculative_added;
        }
      }
    }
    if (speculative_added != 0) {
      storage_owner_stage2_graph_prefetch_issued_.fetch_add(
        speculative_added, std::memory_order_relaxed);
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

      if (!state.score_many_dispatch &&
          VamanaNode::immutable_base_record(pointer)) {
        const size_t scratch_offset =
          static_cast<size_t>(remote_slot++) * snapshot_stride;
        lib_assert(scratch_offset + validation_offset +
                     VamanaNode::HEADER_SIZE <= thread->scratch_stride,
                   "Stage2 base-vector dispatch exceeded lane scratch");
        byte_t* buffer = thread->coroutine_scratch(scratch_offset);
        state.pending_vectors.push_back(Stage2PendingVectorRead{
          .group_index = group,
          .pointer = pointer,
          .buffer = buffer,
          .after_header = buffer + validation_offset,
          .requires_after_header = false,
        });
      }
      // Mutable/recyclable records are scored at their physical home below.
      // Their home RPC retains the incarnation validation, while immutable
      // base records above need only one authoritative one-sided READ.
    }

    // Resolve the cheap immutable wave first.  Dynamic requests remain in the
    // continuation and are selected by the next finite dispatch; this avoids
    // imposing an all-RPC barrier on base candidates and keeps one CQ wave
    // bounded by the existing credit calculation.
    if (!state.pending_vectors.empty()) {
      storage_owner_stage2_vector_read_waves_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_stage2_vector_unique_reads_.fetch_add(
        state.pending_vectors.size(), std::memory_order_relaxed);
      state.ordered_snapshot_pairs = mixed_snapshots;
      state.phase = Stage2SearchIoPhase::score_body_ready;
      return std::pair{progressed, true};
    }

    state.score_home_rpc_count = 0;
    for (Stage2HomeExpandRpc& rpc : state.score_home_rpcs) {
      rpc.posted = false;
      rpc.complete = false;
      rpc.deadline_ns = 0;
      rpc.request.clear();
      rpc.score_consumer_indexes.clear();
    }
    thread_local vec<vec<size_t>> consumers_by_shard;
    consumers_by_shard.clear();
    consumers_by_shard.resize(num_storage_nodes_);
    for (size_t consumer_index = 0;
         consumer_index < state.score_consumers.size(); ++consumer_index) {
      const Stage2ScoreConsumer& consumer =
        state.score_consumers[consumer_index];
      if (storage_node_pointer_addressable(consumer.pointer) &&
          !local_shard(consumer.pointer.memory_node()) &&
          (state.score_many_dispatch ||
           !VamanaNode::immutable_base_record(consumer.pointer))) {
        consumers_by_shard[consumer.pointer.memory_node()].push_back(
          consumer_index);
      }
    }
    const size_t rpc_item_limit = state.score_many_dispatch
      ? std::max<size_t>(1, config.storage_owner_search_snapshot_batch)
      : std::max<size_t>(1, config.storage_owner_batch_max);
    thread_local vec<size_t> score_many_query_searches;
    thread_local vec<u32> score_many_query_indexes;
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      const vec<size_t>& shard_consumers = consumers_by_shard[shard];
      for (size_t begin = 0; begin < shard_consumers.size();
           begin += rpc_item_limit) {
        const u32 item_count = static_cast<u32>(std::min(
          rpc_item_limit, shard_consumers.size() - begin));
        if (state.score_home_rpc_count == state.score_home_rpcs.size()) {
          state.score_home_rpcs.emplace_back();
        }
        Stage2HomeExpandRpc& rpc =
          state.score_home_rpcs[state.score_home_rpc_count++];
        rpc.target_shard = shard;
        rpc.item_count = item_count;
        rpc.request_id = allocate_peer_request_id();
        rpc.score_consumer_indexes.resize(item_count);
        for (u32 item = 0; item < item_count; ++item) {
          rpc.score_consumer_indexes[item] =
            shard_consumers[begin + item];
        }
        if (state.score_many_dispatch) {
          score_many_query_searches.clear();
          score_many_query_indexes.assign(
            tasks.size(), std::numeric_limits<u32>::max());
          for (u32 item = 0; item < item_count; ++item) {
            const Stage2ScoreConsumer& consumer = state.score_consumers[
              shard_consumers[begin + item]];
            lib_assert(consumer.search_index < tasks.size(),
                       "Stage2 score-many consumer escaped its context");
            u32& query_index =
              score_many_query_indexes[consumer.search_index];
            if (query_index == std::numeric_limits<u32>::max()) {
              query_index = static_cast<u32>(
                score_many_query_searches.size());
              score_many_query_searches.push_back(consumer.search_index);
            }
          }
          const u32 query_count = static_cast<u32>(
            score_many_query_searches.size());
          rpc.request.resize(
            service::storage_owner::stage2_score_many_request_bytes(
              item_count, query_count));
          std::fill(rpc.request.begin(), rpc.request.end(), byte_t{0});
          auto* own_header =
            service::storage_owner::stage2_score_many_header(
              rpc.request.data());
          own_header->query_count = query_count;
          auto* items = service::storage_owner::stage2_score_many_items(
            rpc.request.data());
          for (u32 item = 0; item < item_count; ++item) {
            const Stage2ScoreConsumer& consumer = state.score_consumers[
              shard_consumers[begin + item]];
            items[item] = service::storage_owner::Stage2ScoreManyItem{
              .pointer_raw = consumer.pointer.raw_address,
              .generation = consumer.generation,
              .search_index = static_cast<u32>(consumer.search_index),
              .query_index =
                score_many_query_indexes[consumer.search_index],
            };
          }
          byte_t* queries =
            service::storage_owner::stage2_score_many_queries(
              rpc.request.data(), item_count);
          for (u32 query_index = 0; query_index < query_count;
               ++query_index) {
            const size_t search_index =
              score_many_query_searches[query_index];
            std::memcpy(
              queries + static_cast<size_t>(query_index) *
                VamanaNode::vector_bytes(),
              targets[search_index].vector_data.data(),
              VamanaNode::vector_bytes());
          }
        } else {
          rpc.request.resize(
            service::storage_owner::stage2_expand_score_request_bytes(
              item_count));
          std::fill(rpc.request.begin(), rpc.request.end(), byte_t{0});
          auto* items = service::storage_owner::stage2_expand_score_items(
            rpc.request.data());
          byte_t* queries =
            service::storage_owner::stage2_expand_score_queries(
              rpc.request.data(), item_count);
          for (u32 item = 0; item < item_count; ++item) {
            const Stage2ScoreConsumer& consumer = state.score_consumers[
              shard_consumers[begin + item]];
            items[item] = service::storage_owner::Stage2ExpandScoreItem{
              .pointer_raw = consumer.pointer.raw_address,
              .generation = consumer.generation,
              .search_index = static_cast<u32>(consumer.search_index),
              .operation = static_cast<u32>(
                service::storage_owner::Stage2HomeOperation::score_only),
            };
            std::memcpy(
              queries + static_cast<size_t>(item) *
                VamanaNode::vector_bytes(),
              targets[consumer.search_index].vector_data.data(),
              VamanaNode::vector_bytes());
          }
        }
      }
    }
    if (state.score_home_rpc_count != 0) {
      u64 rpc_items = 0;
      for (size_t rpc_index = 0;
           rpc_index < state.score_home_rpc_count; ++rpc_index) {
        const Stage2HomeExpandRpc& rpc = state.score_home_rpcs[rpc_index];
        rpc_items += rpc.item_count;
      }
      storage_owner_stage2_vector_read_waves_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_stage2_vector_unique_reads_.fetch_add(
        rpc_items, std::memory_order_relaxed);
      state.phase = Stage2SearchIoPhase::score_home_pending;
    }
    return std::pair{progressed, state.score_home_rpc_count != 0};
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
      rpc.graph_consumer_indexes.clear();
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
      rpc.graph_consumer_indexes.resize(rpc.item_count);
      std::fill(rpc.request.begin(), rpc.request.end(), byte_t{0});
      rpc_by_shard[shard] = state.home_expand_rpc_count++;
    }
    home_counts.assign(num_storage_nodes_, 0);
    for (size_t consumer_index = 0;
         consumer_index < state.graph_consumers.size(); ++consumer_index) {
      const Stage2GraphConsumer& consumer =
        state.graph_consumers[consumer_index];
      if (!storage_node_pointer_addressable(consumer.pointer) ||
          local_shard(consumer.pointer.memory_node())) {
        continue;
      }
      const u32 shard = consumer.pointer.memory_node();
      Stage2HomeExpandRpc& rpc = state.home_expand_rpcs[rpc_by_shard[shard]];
      const u32 item_index = home_counts[shard]++;
      rpc.graph_consumer_indexes[item_index] = consumer_index;
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
  thread_local vec<byte_t> speculative_score_response_payload;
  thread_local vec<RemotePtr> home_neighbors;
  thread_local vec<u32> speculative_query_indexes;
  thread_local vec<size_t> speculative_query_searches;

  // Authoritative Stage2-home work is transported through the node-wide
  // combiner. The context keeps its original logical request ID and consumes
  // an ordinary logical response; only the wire transaction is shared across
  // contexts. No timer is introduced here and no search work is added.
  const auto enqueue_authoritative_home_rpc = [&] (
      const Stage2HomeExpandRpc& rpc,
      service::storage_owner::PeerRpcType request_type) {
    const auto owner = current_storage_owner_maintenance_context_owner();
    lib_assert(owner.runtime_epoch != 0 && owner.token != 0,
               "Stage2 home RPC has no generation-fenced context owner");
    const auto result = try_enqueue_stage2_home_rpc_request(
      rpc.target_shard, request_type, rpc.request_id, rpc.item_count,
      span<const byte_t>{rpc.request.data(), rpc.request.size()}, false,
      PeerResponseCompletionTarget{.context_owner = owner});
    return result == Stage2HomeRpcEnqueueResult::enqueued ||
      result == Stage2HomeRpcEnqueueResult::duplicate;
  };

  const auto clear_speculative_score_rpc = [&] (
      Stage2SpeculativeScoreRpc& rpc) {
    if (rpc.process_credit_held) {
      release_peer_rpc_speculative_credit(
        rpc.target_shard, rpc.request_id);
    }
    rpc.posted = false;
    rpc.process_credit_held = false;
    rpc.item_count = 0;
    rpc.request_id = 0;
    rpc.deadline_ns = 0;
    rpc.request.clear();
    rpc.consumers.clear();
  };

  // Roll back cache ownership for a transport attempt that did not produce
  // an exact result. If the expansion was already promoted, the cache entry
  // is intentionally absent and its ordinary pending-score request becomes
  // selectable as soon as this RPC is cleared.
  const auto abandon_speculative_score_rpc = [&] (
      Stage2SpeculativeScoreRpc& rpc, bool cancel_registered,
      bool count_wasted, bool remote_completion_observed = false) {
    if (cancel_registered && rpc.request_id != 0) {
      cancel_peer_rpc_response(rpc.request_id);
    }
    for (const Stage2SpeculativeScoreConsumer& consumer : rpc.consumers) {
      (void)state.cancel_graph_prefetch_independent_score_issue(
        consumer.search_index, consumer.expansion_pointer,
        consumer.pointer);
    }
    if (count_wasted && !rpc.consumers.empty()) {
      storage_owner_stage2_independent_score_wasted_.fetch_add(
        rpc.consumers.size(), std::memory_order_relaxed);
    }
    if (rpc.process_credit_held && rpc.posted &&
        !remote_completion_observed) {
      fail_closed_peer_rpc_speculative_credit(
        rpc.target_shard, rpc.request_id);
      rpc.process_credit_held = false;
    }
    clear_speculative_score_rpc(rpc);
  };

  // Poll independently from state.phase. A completed lookahead score either
  // fills its still-bounded graph cache entry, or (after that expansion has
  // been promoted) races the ordinary exact RPC to resolve the same
  // generation/pointer dependency. The continuation fence makes this
  // first-completion-wins; every other outcome is a performance miss and
  // leaves the authoritative path unchanged.
  const auto poll_speculative_score_rpcs = [&] {
    bool progressed = false;
    for (Stage2SpeculativeScoreRpc& rpc :
         state.speculative_score_rpcs) {
      if (!rpc.posted) continue;
      service::storage_owner::PeerRpcHeader response_header{};
      PeerResponseLease response_lease{};
      speculative_score_response_payload.clear();
      const TryPeerResponse response = try_consume_peer_rpc_response(
        rpc.request_id, rpc.target_shard,
        service::storage_owner::PeerRpcType::stage2_score_many_response,
        rpc.item_count, response_header, speculative_score_response_payload,
        response_lease);
      if (response == TryPeerResponse::pending) {
        if (steady_now_ns() >= rpc.deadline_ns) {
          abandon_speculative_score_rpc(rpc, true, true);
          progressed = true;
        }
        continue;
      }

      const size_t expected_bytes =
        service::storage_owner::stage2_score_many_response_bytes(
          rpc.item_count);
      bool valid = response == TryPeerResponse::success &&
        speculative_score_response_payload.size() == expected_bytes &&
        response_header.magic == service::storage_owner::kPeerRpcMagic &&
        response_header.version == service::storage_owner::kPeerRpcVersion &&
        response_header.type == static_cast<u32>(
          service::storage_owner::PeerRpcType::stage2_score_many_response) &&
        response_header.source_shard == rpc.target_shard &&
        response_header.item_count == rpc.item_count &&
        response_header.request_id == rpc.request_id &&
        response_header.status == static_cast<u32>(
          service::storage_owner::InsertStatus::ok) &&
        response_header.reserved == 0;
      const auto* request_items = valid
        ? service::storage_owner::stage2_score_many_items(
            rpc.request.data())
        : nullptr;
      const auto* results = valid
        ? service::storage_owner::stage2_score_many_results(
            speculative_score_response_payload.data())
        : nullptr;
      for (u32 item = 0; valid && item < rpc.item_count; ++item) {
        const auto& request = request_items[item];
        const auto& result = results[item];
        valid = result.pointer_raw == request.pointer_raw &&
          result.generation == request.generation &&
          result.search_index == request.search_index &&
          result.search_index < tasks.size() && result.reserved == 0 &&
          result.disposition <= static_cast<u32>(
            service::storage_owner::Stage2HomeDisposition::terminal);
      }
      if (!valid) {
        const bool remote_completion_observed = response_lease.valid();
        if (response_lease.valid()) {
          lib_assert(rearm_peer_rpc_response(response_lease),
                     "invalid speculative score response lost its lease");
        }
        abandon_speculative_score_rpc(
          rpc, true, true, remote_completion_observed);
        progressed = true;
        continue;
      }
      lib_assert(acknowledge_peer_rpc_response(response_lease),
                 "validated speculative score response lost its lease");

      u64 direct_hits = 0;
      u64 direct_scored = 0;
      u64 direct_home_scored = 0;
      u64 wasted = 0;
      for (u32 item = 0; item < rpc.item_count; ++item) {
        const Stage2SpeculativeScoreConsumer& consumer =
          rpc.consumers[item];
        const auto& result = results[item];
        const auto disposition = static_cast<
          service::storage_owner::Stage2HomeDisposition>(
            result.disposition);
        const bool stable = disposition ==
          service::storage_owner::Stage2HomeDisposition::stable;
        const bool terminal = disposition ==
          service::storage_owner::Stage2HomeDisposition::terminal;
        std::optional<distance_t> exact_distance;
        if (stable) exact_distance.emplace(result.distance);

        bool consumed = false;
        if (stable || terminal) {
          consumed = state.resolve_graph_prefetch_score(
            consumer.search_index, consumer.expansion_pointer,
            consumer.pointer, exact_distance, result.disposition, true);
          if (!consumed && consumer.authoritative_generation != 0) {
            consumed = state.continuation.resolve_score_request(
              consumer.search_index, consumer.authoritative_generation,
              consumer.pointer, exact_distance);
            if (consumed) {
              ++direct_hits;
              ++direct_scored;
              direct_home_scored += stable;
            }
          }
        }
        if (!consumed) {
          (void)state.cancel_graph_prefetch_independent_score_issue(
            consumer.search_index, consumer.expansion_pointer,
            consumer.pointer);
          ++wasted;
        }
      }
      if (direct_hits != 0) {
        // Promotion already created the authoritative dependency. A direct
        // race win is exact but is not proof that an RPC/wave was avoided, so
        // classify it conservatively as non-useful independent work.
        storage_owner_stage2_independent_score_wasted_.fetch_add(
          direct_hits, std::memory_order_relaxed);
      }
      if (wasted != 0) {
        storage_owner_stage2_independent_score_wasted_.fetch_add(
          wasted, std::memory_order_relaxed);
      }
      if (direct_scored != 0) {
        storage_owner_stage2_scored_candidates_.fetch_add(
          direct_scored, std::memory_order_relaxed);
      }
      if (direct_home_scored != 0) {
        storage_owner_stage2_home_scored_neighbors_.fetch_add(
          direct_home_scored, std::memory_order_relaxed);
      }
      clear_speculative_score_rpc(rpc);
      progressed = true;
    }
    return progressed;
  };

  // Fill one exact score-many message per peer from already-bounded graph
  // lookahead. The minimum payload, one-per-context budget, and per-peer
  // request-lifetime credit prevent sparse or slow deployments from turning
  // this into RPC amplification. Its independent no-spec/spec controller
  // retains the path only after an actual posted-RPC cohort reduces complete
  // context cost without increasing completion debt.
  const auto post_speculative_score_rpcs = [&] {
    if (!peer_stage2_home_speculation_enabled_ ||
        !state.independent_score_allowed ||
        state.independent_score_rpcs_started != 0 ||
        !config.storage_owner_stage2_score_many) {
      return false;
    }
    // One independent RPC per context is sufficient to overlap useful work
    // and prevents a large lookahead cache from multiplying transport debt.
    if (std::any_of(
          state.speculative_score_rpcs.begin(),
          state.speculative_score_rpcs.end(),
          [](const Stage2SpeculativeScoreRpc& rpc) {
            return rpc.posted;
          })) {
      return false;
    }
    // Low-priority work is launched only after the correctness path already
    // owns an in-flight home RPC. Together with the transport's reserved last
    // graph slot this guarantees speculation cannot get ahead of or block the
    // authoritative producer.
    const auto authoritative_rpc_in_flight = [&](u32 peer) {
      return std::any_of(
        state.score_home_rpcs.begin(),
        state.score_home_rpcs.begin() + state.score_home_rpc_count,
        [&](const Stage2HomeExpandRpc& rpc) {
          return rpc.target_shard == peer && rpc.posted && !rpc.complete;
        }) ||
      std::any_of(
        state.home_expand_rpcs.begin(),
        state.home_expand_rpcs.begin() + state.home_expand_rpc_count,
        [&](const Stage2HomeExpandRpc& rpc) {
          return rpc.target_shard == peer && rpc.posted && !rpc.complete;
        });
    };
    bool posted_any = false;
    const u32 item_limit = std::max<u32>(
      1, config.storage_owner_search_snapshot_batch);
    const u32 minimum_items = stage2_independent_score_min_items(
      item_limit, std::max<u32>(1, config.storage_owner_batch_max));
    speculative_query_indexes.resize(tasks.size());
    for (u32 peer_offset = 0; peer_offset < num_storage_nodes_;
         ++peer_offset) {
      const u32 peer = (state.speculative_peer_cursor + peer_offset) %
        num_storage_nodes_;
      if (peer == storage_id_) continue;
      if (!authoritative_rpc_in_flight(peer)) continue;
      Stage2SpeculativeScoreRpc& rpc =
        state.speculative_score_rpcs[peer];
      if (rpc.posted) continue;
      lib_assert(rpc.consumers.empty(),
                 "inactive speculative score RPC retained consumers");

      rpc.target_shard = peer;
      rpc.request_id = allocate_peer_request_id();
      if (!try_reserve_peer_rpc_speculative_credit(
            peer, rpc.request_id)) {
        rpc.request_id = 0;
        continue;
      }
      rpc.process_credit_held = true;
      // Count the complete build/post opportunity, not only a successful
      // wire request. A sparse cache or unavailable SEND slot must not cause
      // this context to rescan and rebuild on every scheduler pass.
      if (state.independent_score_rpcs_started == 0) {
        ++state.independent_score_rpcs_started;
      }
      speculative_query_searches.clear();
      std::fill(speculative_query_indexes.begin(),
                speculative_query_indexes.end(),
                std::numeric_limits<u32>::max());
      for (size_t search_index = 0;
           search_index < state.graph_prefetch_cache.size() &&
             rpc.consumers.size() < item_limit;
           ++search_index) {
        if (state.search_seeded[search_index] == 0) continue;
        for (Stage2PrefetchedGraphExpansion& expansion :
             state.graph_prefetch_cache[search_index]) {
          for (Stage2PrefetchedGraphNeighbor& neighbor :
               expansion.neighbors) {
            if (rpc.consumers.size() == item_limit) break;
            const auto disposition = static_cast<
              service::storage_owner::Stage2HomeDisposition>(
                neighbor.disposition);
            if (neighbor.score_prefetched ||
                neighbor.independent_score_prefetched ||
                neighbor.score_prefetch_issues != 0 ||
                neighbor.independent_score_issues != 0 ||
                disposition !=
                  service::storage_owner::Stage2HomeDisposition::unscored ||
                !storage_node_pointer_addressable(neighbor.pointer) ||
                neighbor.pointer.memory_node() != peer) {
              continue;
            }
            u32& query_index = speculative_query_indexes[search_index];
            if (query_index == std::numeric_limits<u32>::max()) {
              query_index = static_cast<u32>(
                speculative_query_searches.size());
              speculative_query_searches.push_back(search_index);
            }
            rpc.consumers.push_back(Stage2SpeculativeScoreConsumer{
              .search_index = search_index,
              .expansion_pointer = expansion.pointer,
              .pointer = neighbor.pointer,
              .authoritative_generation = 0,
            });
            ++neighbor.independent_score_issues;
          }
          if (rpc.consumers.size() == item_limit) break;
        }
      }
      if (rpc.consumers.size() < minimum_items) {
        abandon_speculative_score_rpc(rpc, false, false);
        // A context may have authoritative work at several peers. Sparse
        // first-peer placement must not hide a dense later peer, especially
        // because this bounded build opportunity is attempted only once per
        // context. Credits are released before moving on and at most one wire
        // request is still posted below.
        continue;
      }

      rpc.item_count = static_cast<u32>(rpc.consumers.size());
      const u32 query_count = static_cast<u32>(
        speculative_query_searches.size());
      rpc.request.resize(
        service::storage_owner::stage2_score_many_request_bytes(
          rpc.item_count, query_count));
      std::fill(rpc.request.begin(), rpc.request.end(), byte_t{0});
      auto* own_header = service::storage_owner::stage2_score_many_header(
        rpc.request.data());
      own_header->query_count = query_count;
      own_header->flags =
        service::storage_owner::kStage2ScoreManyFlagSpeculative;
      auto* items = service::storage_owner::stage2_score_many_items(
        rpc.request.data());
      for (u32 item = 0; item < rpc.item_count; ++item) {
        const Stage2SpeculativeScoreConsumer& consumer =
          rpc.consumers[item];
        items[item] = service::storage_owner::Stage2ScoreManyItem{
          .pointer_raw = consumer.pointer.raw_address,
          // This is a wire correlation token. The authoritative generation
          // is installed separately only if the expansion is promoted.
          .generation = state.continuation.generation(
            consumer.search_index),
          .search_index = static_cast<u32>(consumer.search_index),
          .query_index = speculative_query_indexes[
            consumer.search_index],
        };
      }
      byte_t* queries = service::storage_owner::stage2_score_many_queries(
        rpc.request.data(), rpc.item_count);
      for (u32 query_index = 0; query_index < query_count; ++query_index) {
        const size_t search_index =
          speculative_query_searches[query_index];
        std::memcpy(
          queries + static_cast<size_t>(query_index) *
            VamanaNode::vector_bytes(),
          targets[search_index].vector_data.data(),
          VamanaNode::vector_bytes());
      }

      const size_t request_bytes = rpc.request.size();
      rpc.posted = try_post_peer_rpc_request_attempt(
        rpc.target_shard,
        service::storage_owner::PeerRpcType::stage2_score_many_request,
        service::storage_owner::PeerRpcType::stage2_score_many_response,
        rpc.request_id, rpc.item_count,
        rpc.request.data() + sizeof(service::storage_owner::PeerRpcHeader),
        request_bytes - sizeof(service::storage_owner::PeerRpcHeader),
        request_bytes, PeerRpcSendClass::speculative);
      if (!rpc.posted) {
        // register_send_attempt() precedes send-slot acquisition. A failed
        // try-post therefore still owns a response-registry cell even though
        // no wire request exists; cancel it before abandoning this request id.
        abandon_speculative_score_rpc(rpc, true, false);
        return false;
      }
      // Speculation is never allowed to inherit the 30-second correctness
      // timeout. Two milliseconds is far above the measured microsecond RPC
      // service time yet bounds low-priority transport and registry debt in
      // a slower environment; the authoritative path never waits for it.
      constexpr u64 kSpeculativeTimeoutNs = 2ull * 1000ull * 1000ull;
      const u64 configured_timeout_ns =
        static_cast<u64>(config.storage_owner_rpc_timeout_ms) *
          1000ull * 1000ull;
      rpc.deadline_ns = steady_now_ns() +
        std::min(kSpeculativeTimeoutNs, configured_timeout_ns);

      storage_owner_stage2_independent_score_rpc_batches_.fetch_add(
        1, std::memory_order_relaxed);
      ++state.independent_score_rpcs_posted;
      storage_owner_stage2_independent_score_issued_.fetch_add(
        rpc.item_count, std::memory_order_relaxed);
      storage_owner_stage2_home_rpc_batches_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_stage2_home_rpc_items_.fetch_add(
        rpc.item_count, std::memory_order_relaxed);
      storage_owner_stage2_home_score_rpc_batches_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_stage2_home_score_rpc_items_.fetch_add(
        rpc.item_count, std::memory_order_relaxed);
      storage_owner_stage2_home_score_rpc_queries_.fetch_add(
        query_count, std::memory_order_relaxed);
      storage_owner_stage2_home_score_rpc_request_bytes_.fetch_add(
        request_bytes, std::memory_order_relaxed);
      storage_owner_stage2_home_score_rpc_response_bytes_.fetch_add(
        service::storage_owner::stage2_score_many_response_bytes(
          rpc.item_count),
        std::memory_order_relaxed);
      state.speculative_peer_cursor = (peer + 1) % num_storage_nodes_;
      posted_any = true;
      // One request for the complete context is enough for the guarded first
      // trial. A later context begins from a different peer cursor, avoiding
      // a low-shard hotspot without multiplying receiver work.
      break;
    }
    return posted_any;
  };

  const auto commit_prefetched_graph = [&] {
    const auto score_issue_count = [](const auto& neighbors) {
      u64 count = 0;
      for (const auto& neighbor : neighbors) {
        count += neighbor.score_prefetch_issues;
      }
      return count;
    };
    const auto independent_issue_count = [](const auto& neighbors) {
      u64 count = 0;
      for (const auto& neighbor : neighbors) {
        if (neighbor.independent_score_prefetched) {
          count += neighbor.independent_score_issues;
        }
      }
      return count;
    };
    u64 hits = 0;
    u64 wasted = 0;
    u64 scored = 0;
    u64 home_scored = 0;
    u64 score_prefetch_hits = 0;
    u64 score_prefetch_wasted = 0;
    u64 independent_score_useful = 0;
    u64 independent_score_wasted = 0;
    bool progressed = false;
    for (size_t search_index = 0; search_index < tasks.size();
         ++search_index) {
      if (state.search_seeded[search_index] == 0) continue;
      const auto request =
        state.continuation.pending_expand_request(search_index);
      if (!request.has_value()) continue;
      auto cached = state.take_graph_prefetch(
        search_index, request->pointer);
      if (!cached.has_value()) continue;
      const auto disposition = static_cast<
        service::storage_owner::Stage2HomeDisposition>(
          cached->disposition);
      if (disposition !=
            service::storage_owner::Stage2HomeDisposition::stable &&
          disposition !=
            service::storage_owner::Stage2HomeDisposition::terminal) {
        ++wasted;
        score_prefetch_wasted += score_issue_count(cached->neighbors);
        independent_score_wasted +=
          independent_issue_count(cached->neighbors);
        continue;
      }
      home_neighbors.clear();
      if (disposition ==
          service::storage_owner::Stage2HomeDisposition::stable) {
        home_neighbors.reserve(cached->neighbors.size());
        for (const Stage2PrefetchedGraphNeighbor& neighbor :
             cached->neighbors) {
          home_neighbors.push_back(neighbor.pointer);
        }
      }
      if (!state.continuation.resolve_expand_request(
            search_index, request->generation, request->pointer,
            span<const RemotePtr>{home_neighbors})) {
        ++wasted;
        score_prefetch_wasted += score_issue_count(cached->neighbors);
        independent_score_wasted +=
          independent_issue_count(cached->neighbors);
        continue;
      }
      ++hits;
      progressed = true;
      const u64 score_generation =
        state.continuation.generation(search_index);
      (void)state.promote_speculative_scores(
        search_index, cached->pointer, score_generation);
      for (const Stage2PrefetchedGraphNeighbor& neighbor :
           cached->neighbors) {
        const auto neighbor_disposition = static_cast<
          service::storage_owner::Stage2HomeDisposition>(
            neighbor.disposition);
        bool resolved = false;
        if (neighbor_disposition ==
            service::storage_owner::Stage2HomeDisposition::stable) {
          resolved = state.continuation.resolve_score_request(
            search_index, score_generation, neighbor.pointer,
            std::optional<distance_t>{neighbor.distance});
          home_scored += resolved;
        } else if (neighbor_disposition ==
                   service::storage_owner::Stage2HomeDisposition::terminal) {
          resolved = state.continuation.resolve_score_request(
            search_index, score_generation, neighbor.pointer,
            std::nullopt);
        }
        scored += resolved;
        // An independent RPC that survived promotion now owns this exact
        // pending-score dependency. Its completion (or bounded timeout)
        // accounts the outcome, so do not prematurely classify it as wasted
        // merely because the cache entry has been consumed.
        const bool independent_in_flight =
          state.speculative_score_covers(
            search_index, score_generation, neighbor.pointer);
        if (neighbor.score_prefetch_issues != 0 &&
            !independent_in_flight) {
          if (resolved) {
            ++score_prefetch_hits;
            score_prefetch_wasted +=
              neighbor.score_prefetch_issues - 1;
          } else {
            score_prefetch_wasted += neighbor.score_prefetch_issues;
          }
        }
        if (neighbor.independent_score_issues != 0 &&
            !independent_in_flight) {
          if (resolved && neighbor.independent_score_prefetched) {
            ++independent_score_useful;
            independent_score_wasted +=
              neighbor.independent_score_issues - 1;
          } else {
            independent_score_wasted +=
              neighbor.independent_score_issues;
          }
        }
      }
    }
    if (hits != 0) {
      storage_owner_stage2_graph_prefetch_hits_.fetch_add(
        hits, std::memory_order_relaxed);
    }
    if (wasted != 0) {
      storage_owner_stage2_graph_prefetch_wasted_.fetch_add(
        wasted, std::memory_order_relaxed);
    }
    if (scored != 0) {
      storage_owner_stage2_scored_candidates_.fetch_add(
        scored, std::memory_order_relaxed);
    }
    if (home_scored != 0) {
      storage_owner_stage2_home_scored_neighbors_.fetch_add(
        home_scored, std::memory_order_relaxed);
    }
    if (score_prefetch_hits != 0) {
      storage_owner_stage2_score_prefetch_hits_.fetch_add(
        score_prefetch_hits, std::memory_order_relaxed);
    }
    if (score_prefetch_wasted != 0) {
      storage_owner_stage2_score_prefetch_wasted_.fetch_add(
        score_prefetch_wasted, std::memory_order_relaxed);
    }
    if (independent_score_useful != 0) {
      state.independent_score_useful += independent_score_useful;
      storage_owner_stage2_independent_score_useful_.fetch_add(
        independent_score_useful, std::memory_order_relaxed);
    }
    if (independent_score_wasted != 0) {
      storage_owner_stage2_independent_score_wasted_.fetch_add(
        independent_score_wasted, std::memory_order_relaxed);
    }
    return progressed;
  };
  u8 idle_attempt_mask = 0;

  // Only a bounded transport dispatch is synchronized on one CQ.  Once its
  // WRs retire, every stable/terminal consumer is resolved immediately and
  // retryable consumers simply remain in their own search generation.
  for (;;) {
    const bool speculative_progress = poll_speculative_score_rpcs();
    if (speculative_progress) idle_attempt_mask = 0;
    if (state.continuation.all_complete()) {
      // Speculation is never part of durable search completion. Cancel any
      // late request and let the ordinary unused-cache accounting below
      // classify completed-but-unpromoted exact results.
      for (Stage2SpeculativeScoreRpc& rpc :
           state.speculative_score_rpcs) {
        if (rpc.posted) {
          abandon_speculative_score_rpc(rpc, true, true);
        }
      }
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
      const u64 unused_prefetch = state.graph_prefetch_entry_count();
      const u64 unused_score_prefetch =
        state.graph_prefetched_score_count();
      const u64 unused_independent_score =
        state.graph_independent_score_count();
      if (unused_prefetch != 0) {
        storage_owner_stage2_graph_prefetch_wasted_.fetch_add(
          unused_prefetch, std::memory_order_relaxed);
      }
      if (unused_score_prefetch != 0) {
        storage_owner_stage2_score_prefetch_wasted_.fetch_add(
          unused_score_prefetch, std::memory_order_relaxed);
      }
      if (unused_independent_score != 0) {
        storage_owner_stage2_independent_score_wasted_.fetch_add(
          unused_independent_score, std::memory_order_relaxed);
      }
      const size_t retained_capacity = std::max<size_t>(
        1024, static_cast<size_t>(construction_width) * 8);
      state.reset_completed(retained_capacity);
      return Stage2SearchAdvanceResult::complete;
    }

    (void)post_speculative_score_rpcs();

    // pending_expand_request() is a per-search dependency fence: a search is
    // in either score or expand phase, never both. Do not gate cache commit on
    // every other search in this batched context having no pending scores.
    // That global gate caused 130809 to refetch an authoritative graph for A
    // while unrelated B was scoring, wasting every ordered cache entry and
    // disabling the controller during warm-up.
    if (state.phase == Stage2SearchIoPhase::idle &&
        commit_prefetched_graph()) {
      idle_attempt_mask = 0;
      continue;
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

    if (state.phase == Stage2SearchIoPhase::score_home_pending) {
      bool all_complete = true;
      for (size_t rpc_index = 0;
           rpc_index < state.score_home_rpc_count; ++rpc_index) {
        Stage2HomeExpandRpc& rpc = state.score_home_rpcs[rpc_index];
        if (rpc.complete) continue;
        all_complete = false;
        if (!rpc.posted) {
          const auto request_type = state.score_many_dispatch
            ? service::storage_owner::PeerRpcType::stage2_score_many_request
            : service::storage_owner::PeerRpcType::stage2_expand_score_request;
          rpc.posted = enqueue_authoritative_home_rpc(rpc, request_type);
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
          state.score_many_dispatch
            ? service::storage_owner::PeerRpcType::stage2_score_many_response
            : service::storage_owner::PeerRpcType::stage2_expand_score_response,
          rpc.item_count, response_header, home_response_payload,
          response_lease);
        if (response == TryPeerResponse::pending) {
          // The outer aggregate owns same-wire-ID timeout/retry. Cancelling a
          // logical member here would separate its response-registry cell
          // from the retained aggregate and defeat exact fan-out.
          continue;
        }
        const size_t expected_bytes = state.score_many_dispatch
          ? service::storage_owner::stage2_score_many_response_bytes(
              rpc.item_count)
          : service::storage_owner::stage2_expand_score_response_bytes(
              rpc.item_count, 0);
        const auto expected_response_type =
          state.score_many_dispatch
            ? service::storage_owner::PeerRpcType::stage2_score_many_response
            : service::storage_owner::PeerRpcType::stage2_expand_score_response;
        bool valid = response == TryPeerResponse::success &&
          home_response_payload.size() == expected_bytes &&
          response_header.magic == service::storage_owner::kPeerRpcMagic &&
          response_header.version == service::storage_owner::kPeerRpcVersion &&
          response_header.type == static_cast<u32>(
            expected_response_type) &&
          response_header.source_shard == rpc.target_shard &&
          response_header.item_count == rpc.item_count &&
          response_header.request_id == rpc.request_id &&
          response_header.status == static_cast<u32>(
            service::storage_owner::InsertStatus::ok) &&
          response_header.reserved == 0;
        const auto* legacy_results = valid &&
            !state.score_many_dispatch
          ? service::storage_owner::stage2_expand_score_results(
              home_response_payload.data())
          : nullptr;
        const auto* score_many_results = valid &&
            state.score_many_dispatch
          ? service::storage_owner::stage2_score_many_results(
              home_response_payload.data())
          : nullptr;
        if (state.score_many_dispatch) {
          const auto* request_items =
            service::storage_owner::stage2_score_many_items(
              rpc.request.data());
          for (u32 item = 0; valid && item < rpc.item_count; ++item) {
            const auto& request = request_items[item];
            const auto& result = score_many_results[item];
            valid = result.pointer_raw == request.pointer_raw &&
              result.generation == request.generation &&
              result.search_index == request.search_index &&
              result.search_index < tasks.size() && result.reserved == 0 &&
              result.disposition <= static_cast<u32>(
                service::storage_owner::Stage2HomeDisposition::terminal);
          }
        } else {
          const auto* request_items =
            service::storage_owner::stage2_expand_score_items(
              rpc.request.data());
          for (u32 item = 0; valid && item < rpc.item_count; ++item) {
            const auto& request = request_items[item];
            const auto& result = legacy_results[item];
            valid = request.operation == static_cast<u32>(
                      service::storage_owner::Stage2HomeOperation::score_only) &&
              result.operation == request.operation &&
              result.pointer_raw == request.pointer_raw &&
              result.generation == request.generation &&
              result.search_index == request.search_index &&
              result.search_index < tasks.size() &&
              result.neighbor_count == 0 && result.neighbor_offset == 0 &&
              result.disposition <= static_cast<u32>(
                service::storage_owner::Stage2HomeDisposition::terminal);
          }
        }
        if (!valid) {
          if (response_lease.valid()) {
            lib_assert(rearm_peer_rpc_response(response_lease),
                       "invalid Stage2 score response lost its lease");
          }
          rpc.posted = false;
          continue;
        }
        lib_assert(acknowledge_peer_rpc_response(response_lease),
                   "validated Stage2 score response lost its lease");
        for (u32 item = 0; item < rpc.item_count; ++item) {
          lib_assert(item < rpc.score_consumer_indexes.size() &&
                       rpc.score_consumer_indexes[item] <
                         state.score_consumers.size(),
                     "Stage2 score response lost consumer metadata");
          const Stage2ScoreConsumer& consumer = state.score_consumers[
            rpc.score_consumer_indexes[item]];
          const u32 result_search_index =
            state.score_many_dispatch
              ? score_many_results[item].search_index
              : legacy_results[item].search_index;
          const u64 result_generation =
            state.score_many_dispatch
              ? score_many_results[item].generation
              : legacy_results[item].generation;
          const u64 result_pointer_raw =
            state.score_many_dispatch
              ? score_many_results[item].pointer_raw
              : legacy_results[item].pointer_raw;
          const u32 result_disposition =
            state.score_many_dispatch
              ? score_many_results[item].disposition
              : legacy_results[item].disposition;
          const distance_t result_distance =
            state.score_many_dispatch
              ? score_many_results[item].distance
              : legacy_results[item].distance;
          const auto disposition = static_cast<
            service::storage_owner::Stage2HomeDisposition>(
              result_disposition);
          if (consumer.speculative) {
            if (disposition ==
                service::storage_owner::Stage2HomeDisposition::stable) {
              state.resolve_graph_prefetch_score(
                consumer.search_index, consumer.expansion_pointer,
                consumer.pointer,
                std::optional<distance_t>{result_distance},
                result_disposition);
            } else if (disposition ==
                       service::storage_owner::Stage2HomeDisposition::terminal) {
              state.resolve_graph_prefetch_score(
                consumer.search_index, consumer.expansion_pointer,
                consumer.pointer, std::nullopt, result_disposition);
            }
            continue;
          }
          bool resolved = false;
          if (disposition ==
              service::storage_owner::Stage2HomeDisposition::stable) {
            resolved = state.continuation.resolve_score_request(
              result_search_index, result_generation,
              RemotePtr{result_pointer_raw},
              std::optional<distance_t>{result_distance});
          } else if (disposition ==
                     service::storage_owner::Stage2HomeDisposition::terminal) {
            resolved = state.continuation.resolve_score_request(
              result_search_index, result_generation,
              RemotePtr{result_pointer_raw}, std::nullopt);
          }
          if (resolved) {
            storage_owner_stage2_scored_candidates_.fetch_add(
              1, std::memory_order_relaxed);
            storage_owner_stage2_home_scored_neighbors_.fetch_add(
              disposition ==
                  service::storage_owner::Stage2HomeDisposition::stable,
              std::memory_order_relaxed);
          }
        }
        rpc.complete = true;
      }
      all_complete = std::all_of(
        state.score_home_rpcs.begin(),
        state.score_home_rpcs.begin() + state.score_home_rpc_count,
        [](const Stage2HomeExpandRpc& rpc) { return rpc.complete; });
      if (!all_complete) return Stage2SearchAdvanceResult::waiting_rdma;
      clear_score_dispatch();
      state.prefer_graph = true;
      idle_attempt_mask = 0;
      continue;
    }

    if (state.phase == Stage2SearchIoPhase::graph_home_pending) {
      bool all_complete = true;
      for (size_t rpc_index = 0;
           rpc_index < state.home_expand_rpc_count; ++rpc_index) {
        Stage2HomeExpandRpc& rpc = state.home_expand_rpcs[rpc_index];
        if (rpc.complete) continue;
        all_complete = false;
        if (!rpc.posted) {
          rpc.posted = enqueue_authoritative_home_rpc(
            rpc,
            service::storage_owner::PeerRpcType::stage2_expand_score_request);
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
          // The combined outer request, not an individual logical member,
          // owns transport retry and its immutable wire image.
          continue;
        }
        const size_t minimum_bytes =
          service::storage_owner::stage2_expand_score_response_bytes(
            rpc.item_count, 0);
        const size_t maximum_bytes =
          service::storage_owner::stage2_expand_score_response_bytes(
            rpc.item_count);
        const size_t response_bytes = home_response_payload.size();
        const bool compact_size_valid = response_bytes >= minimum_bytes &&
          response_bytes <= maximum_bytes &&
          (response_bytes - minimum_bytes) %
              sizeof(service::storage_owner::Stage2ExpandScoreNeighbor) == 0;
        const size_t compact_neighbor_count = compact_size_valid
          ? (response_bytes - minimum_bytes) /
              sizeof(service::storage_owner::Stage2ExpandScoreNeighbor)
          : 0;
        bool valid = response == TryPeerResponse::success &&
          compact_size_valid &&
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
        size_t expected_neighbor_offset = 0;
        for (u32 item_index = 0; valid && item_index < rpc.item_count;
             ++item_index) {
          const auto& request = request_items[item_index];
          const auto& result = results[item_index];
          valid = result.pointer_raw == request.pointer_raw &&
            result.generation == request.generation &&
            result.search_index == request.search_index &&
            result.search_index < tasks.size() &&
            request.operation == static_cast<u32>(
              service::storage_owner::Stage2HomeOperation::expand_score) &&
            result.operation == request.operation &&
            result.neighbor_count <= neighbor_stride &&
            result.neighbor_offset == expected_neighbor_offset &&
            result.neighbor_offset <= compact_neighbor_count &&
            result.neighbor_count <=
              compact_neighbor_count - result.neighbor_offset &&
            result.disposition <= static_cast<u32>(
              service::storage_owner::Stage2HomeDisposition::terminal);
          for (u32 neighbor_index = 0;
               valid && neighbor_index < result.neighbor_count;
               ++neighbor_index) {
            const auto& neighbor = neighbors[
              static_cast<size_t>(result.neighbor_offset) + neighbor_index];
            valid = neighbor.disposition <= static_cast<u32>(
              service::storage_owner::Stage2HomeDisposition::unscored);
          }
          expected_neighbor_offset += result.neighbor_count;
        }
        valid = valid && expected_neighbor_offset == compact_neighbor_count;
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
          lib_assert(item_index < rpc.graph_consumer_indexes.size() &&
                       rpc.graph_consumer_indexes[item_index] <
                         state.graph_consumers.size(),
                     "Stage2 graph response lost consumer metadata");
          const Stage2GraphConsumer& consumer = state.graph_consumers[
            rpc.graph_consumer_indexes[item_index]];
          const auto disposition = static_cast<
            service::storage_owner::Stage2HomeDisposition>(
              result.disposition);
          if (consumer.speculative) {
            bool retained = false;
            if (disposition !=
                service::storage_owner::Stage2HomeDisposition::retryable) {
              Stage2PrefetchedGraphExpansion cached{
                .pointer = RemotePtr{result.pointer_raw},
                .disposition = result.disposition,
                .neighbors = {},
              };
              cached.neighbors.reserve(result.neighbor_count);
              for (u32 neighbor_index = 0;
                   neighbor_index < result.neighbor_count;
                   ++neighbor_index) {
                const auto& neighbor = neighbors[
                  static_cast<size_t>(result.neighbor_offset) +
                  neighbor_index];
                cached.neighbors.push_back({
                  .pointer = RemotePtr{neighbor.pointer_raw},
                  .distance = neighbor.distance,
                  .disposition = neighbor.disposition,
                  .score_prefetched = false,
                  .score_prefetch_issues = 0,
                });
              }
              retained = state.insert_graph_prefetch(
                consumer.search_index, std::move(cached),
                static_cast<size_t>(
                  config.storage_owner_stage2_graph_issue_width));
            }
            if (!retained) {
              storage_owner_stage2_graph_prefetch_wasted_.fetch_add(
                1, std::memory_order_relaxed);
            }
            continue;
          }
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
                static_cast<size_t>(result.neighbor_offset) +
                neighbor_index].pointer_raw});
            }
          }
          if (!state.continuation.resolve_expand_request(
                result.search_index, result.generation,
                RemotePtr{result.pointer_raw},
                span<const RemotePtr>{home_neighbors})) {
            continue;
          }
          forget_graph_retry(RemotePtr{result.pointer_raw});
          const u64 score_generation =
            state.continuation.generation(result.search_index);
          for (u32 neighbor_index = 0;
               neighbor_index < result.neighbor_count; ++neighbor_index) {
            const auto& neighbor = neighbors[
              static_cast<size_t>(result.neighbor_offset) + neighbor_index];
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
