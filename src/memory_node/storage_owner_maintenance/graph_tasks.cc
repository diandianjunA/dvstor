#include "memory_node/storage_owner_maintenance/detail.hh"
#include "memory_node/storage_owner_maintenance/cleanup_policy.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

using namespace memory_node_storage_owner_maintenance_detail;
using memory_node_storage_owner_index_detail::IncarnationLockResult;
using memory_node_storage_owner_index_detail::StableNodeSnapshotState;
using memory_node_storage_owner_index_detail::classify_stage2_target_snapshot;

bool MemoryNode::storage_owner_task_current(node_t id, u32 generation, RemotePtr target) {
  DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto dynamic = shard.entries.find(id);
  if (dynamic != shard.entries.end()) {
    return !dynamic->second.deleted &&
           dynamic->second.generation == generation &&
           dynamic->second.current == target;
  }
  const auto& immutable_base = base_idmap_;
  const auto base = immutable_base.find(id);
  return base != immutable_base.end() &&
         generation == 0 && base->second == target;
}

StableNodeSnapshotState MemoryNode::storage_owner_physical_node_state(
    node_t id,
    u32 generation,
    RemotePtr target,
    NodeSnapshot* stable_snapshot) {
  if (!storage_node_pointer_addressable(target)) {
    return StableNodeSnapshotState::terminal;
  }

  // When the caller also needs the vector, first try the existing complete
  // optimistic snapshot.  A failed bounded read is not terminal: it is
  // followed by a small identity observation below so a stable replacement
  // can still be distinguished from transient lock contention.
  if (stable_snapshot != nullptr) {
    NodeSnapshot candidate;
    if (read_node_snapshot(target, candidate)) {
      const StableNodeSnapshotState state = classify_stage2_target_snapshot(
        target, candidate.header, candidate.header,
        candidate.slot_incarnation, candidate.id, candidate.generation,
        id, generation);
      if (state == StableNodeSnapshotState::stable) {
        *stable_snapshot = std::move(candidate);
      }
      return state;
    }
  }

  constexpr size_t kIdentityBytes =
    VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;
  std::array<byte_t, kIdentityBytes> identity{};
  u64 after = 0;
  if (local_shard(target.memory_node())) {
    const u64 before = load_local_node_header_acquire(target);
    std::memcpy(identity.data(), &before, sizeof(before));
    std::memcpy(identity.data() + VamanaNode::HEADER_SIZE,
                index_buffer_.get_full_buffer() + target.byte_offset() +
                  VamanaNode::HEADER_SIZE,
                VamanaNode::COMPACT_META_SIZE);
    std::atomic_thread_fence(std::memory_order_acquire);
    after = load_local_node_header_acquire(target);
  } else {
    remote_read_bytes(target.memory_node(), target.byte_offset(),
                      identity.data(), identity.size(), 0);
    remote_read_bytes(target.memory_node(), target.byte_offset(),
                      &after, sizeof(after), 0);
  }

  u64 before = 0;
  node_t observed_id = 0;
  u32 observed_generation = 0;
  u32 slot_incarnation = 0;
  std::memcpy(&before, identity.data(), sizeof(before));
  std::memcpy(&observed_id, identity.data() + VamanaNode::offset_id(),
              sizeof(observed_id));
  std::memcpy(&observed_generation,
              identity.data() + VamanaNode::offset_generation(),
              sizeof(observed_generation));
  std::memcpy(&slot_incarnation,
              identity.data() + VamanaNode::offset_slot_incarnation(),
              sizeof(slot_incarnation));
  const StableNodeSnapshotState identity_state =
    classify_stage2_target_snapshot(
      target, before, after, slot_incarnation, observed_id,
      observed_generation, id, generation);
  if (stable_snapshot != nullptr &&
      identity_state == StableNodeSnapshotState::stable) {
    // The identity is current, but the preceding full vector observation did
    // not stabilize.  Suspend this context and retry it after other lanes have
    // advanced; never fabricate a vector or discard the task.
    return StableNodeSnapshotState::retryable;
  }
  return identity_state;
}

vec<RemotePtr> MemoryNode::read_preserved_neighbor_list(RemotePtr rptr) {
  // This is not an optimistic query read.  The tombstone's preserved
  // adjacency is the sole authority for removing old backlinks before its
  // physical slot can be reclaimed.  Treating corruption as an empty list
  // would acknowledge cleanup while silently leaving dangling graph edges.
  lib_assert(storage_node_pointer_addressable(rptr),
             "preserved adjacency target is not addressable: raw=" +
               std::to_string(rptr.raw_address));
  vec<byte_t> entry(VamanaNode::hot_graph_entry_size());
  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
  if (local_shard(rptr.memory_node())) {
    std::memcpy(entry.data(),
                index_buffer_.get_full_buffer() + hot_offset,
                entry.size());
  } else {
    remote_read_bytes(rptr.memory_node(), hot_offset, entry.data(), entry.size(), 0);
  }

  const u8 edge_count = entry[0];
  lib_assert(edge_count <= VamanaNode::R,
             "preserved adjacency edge count exceeds R: raw=" +
               std::to_string(rptr.raw_address) + " count=" +
               std::to_string(edge_count));
  const u16 expected = vamana::hot_graph::load_u16_le(entry.data() + 2);
  const u16 actual = vamana::hot_graph::checksum16(entry.data(), entry.size());
  lib_assert(expected == actual,
             "preserved adjacency checksum mismatch: raw=" +
               std::to_string(rptr.raw_address));
  lib_assert(vamana::hot_graph::load_u32_le(entry.data() + 8) ==
               rptr.incarnation(),
             "preserved adjacency incarnation mismatch: raw=" +
               std::to_string(rptr.raw_address));
  lib_assert(vamana::hot_graph::load_u32_le(entry.data() + 12) == 0,
             "preserved adjacency reserved field is nonzero: raw=" +
               std::to_string(rptr.raw_address));
  vec<RemotePtr> neighbors;
  neighbors.reserve(edge_count);
  for (u32 i = 0; i < edge_count; ++i) {
    RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
      entry.data() + vamana::hot_graph::neighbor_offset(i),
      VamanaNode::HOT_GRAPH_SHARD_BITS);
    lib_assert(!neighbor.is_null(),
               "preserved adjacency contains a null counted edge: parent_raw=" +
                 std::to_string(rptr.raw_address) + " index=" +
                 std::to_string(i));
    lib_assert(storage_node_pointer_addressable(neighbor),
               "preserved adjacency contains malformed neighbor: parent_raw=" +
                 std::to_string(rptr.raw_address) + " neighbor_raw=" +
                 std::to_string(neighbor.raw_address) + " index=" +
                 std::to_string(i));
    neighbors.push_back(neighbor);
  }
  return neighbors;
}

bool MemoryNode::remove_local_neighbor(RemotePtr target_ptr,
                                       RemotePtr deleted_ptr,
                                       const Configuration&) {
  if (target_ptr.is_null() || deleted_ptr.is_null() || !local_shard(target_ptr.memory_node())) {
    return false;
  }

  const IncarnationLockResult target_lock = try_lock_node(target_ptr);
  if (target_lock == IncarnationLockResult::stale) return true;
  if (target_lock == IncarnationLockResult::busy) return false;
  const u64 target_header = load_local_node_header_acquire(target_ptr);
  if ((target_header & (VamanaNode::HEADER_DELETED |
                        VamanaNode::HEADER_RETIRING)) != 0) {
    unlock_node(target_ptr);
    return true;
  }
  if (VamanaNode::graph_mutation_quiesced(target_header)) {
    unlock_node(target_ptr);
    return false;
  }

  vec<RemotePtr> neighbors = read_stable_neighbor_list(target_ptr);
  const auto old_size = neighbors.size();
  neighbors.erase(
    std::remove(neighbors.begin(), neighbors.end(), deleted_ptr),
    neighbors.end());
  if (neighbors.size() != old_size) {
    write_neighbor_list(target_ptr, neighbors);
  }
  unlock_node(target_ptr);
  return true;
}

bool MemoryNode::remove_local_neighbors_batched(
    const dense_hashmap_t<u64, vec<RemotePtr>>& removals,
    const Configuration&) {
  bool success = true;
  for (const auto& [target_raw, deleted_ptrs] : removals) {
    const RemotePtr target_ptr{target_raw};
    if (target_ptr.is_null() || !local_shard(target_ptr.memory_node())) {
      success = false;
      continue;
    }

    const IncarnationLockResult target_lock = try_lock_node(target_ptr);
    if (target_lock == IncarnationLockResult::stale) continue;
    if (target_lock == IncarnationLockResult::busy) {
      success = false;
      continue;
    }
    const u64 target_header = load_local_node_header_acquire(target_ptr);
    if ((target_header & (VamanaNode::HEADER_DELETED |
                          VamanaNode::HEADER_RETIRING)) != 0) {
      unlock_node(target_ptr);
      continue;
    }
    if (VamanaNode::graph_mutation_quiesced(target_header)) {
      unlock_node(target_ptr);
      success = false;
      continue;
    }

    vec<RemotePtr> neighbors = read_stable_neighbor_list(target_ptr);
    const auto old_size = neighbors.size();
    neighbors.erase(
      std::remove_if(neighbors.begin(), neighbors.end(), [&](const RemotePtr& neighbor) {
        return std::find(deleted_ptrs.begin(), deleted_ptrs.end(), neighbor) !=
               deleted_ptrs.end();
      }),
      neighbors.end());
    if (neighbors.size() != old_size) {
      write_neighbor_list(target_ptr, neighbors);
    }
    unlock_node(target_ptr);
  }
  return success;
}

bool MemoryNode::remove_local_neighbors_identity_fenced(
    span<const service::storage_owner::ReverseUpdateOp> ops,
    const Configuration& config) {
  using service::storage_owner::ReverseUpdateOp;
  if (ops.empty()) return true;

  // Candidate identity is not encoded in graph edges, so validate every
  // physical candidate record before touching a target. The batched snapshot
  // path coalesces repeated deleted candidates and remote reads across the
  // whole cleanup RPC instead of issuing one RDMA read per edge.
  vec<RemotePtr> candidates;
  candidates.reserve(ops.size());
  for (const ReverseUpdateOp& op : ops) {
    const RemotePtr target{op.target_raw};
    const RemotePtr candidate{op.candidate_raw};
    if (!valid_local_storage_node_pointer(target) || candidate.is_null() ||
        candidate.memory_node() >= num_storage_nodes_) {
      // Malformed traffic is a protocol failure. A well-formed but stale
      // identity below is instead an idempotent no-op/ACK.
      return false;
    }
    candidates.push_back(candidate);
  }
  std::sort(candidates.begin(), candidates.end(),
            [](RemotePtr lhs, RemotePtr rhs) {
              return lhs.raw_address < rhs.raw_address;
            });
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  const vec<NodeSnapshot> candidate_snapshots =
    read_node_snapshots_batched(
      candidates, config, "remove_local_neighbors_identity_fenced");
  dense_hashmap_t<u64, NodeSnapshot> candidate_by_raw;
  candidate_by_raw.reserve(candidate_snapshots.size());
  for (const NodeSnapshot& snapshot : candidate_snapshots) {
    candidate_by_raw.emplace(snapshot.rptr.raw_address, snapshot);
  }

  dense_hashmap_t<u64, vec<const ReverseUpdateOp*>> grouped;
  grouped.reserve(ops.size());
  for (const ReverseUpdateOp& op : ops) {
    const auto found = candidate_by_raw.find(op.candidate_raw);
    // Delayed/replayed cleanup for a reclaimed candidate must succeed as an
    // idempotent no-op. It must never remove a new edge that happens to reuse
    // the same physical address.
    if (found == candidate_by_raw.end() ||
        !cleanup_deleted_candidate_matches(
          op.candidate_id, op.candidate_generation,
          found->second.id, found->second.generation,
          found->second.deleted)) {
      continue;
    }
    grouped[op.target_raw].push_back(&op);
  }

  for (const auto& [target_raw, removals] : grouped) {
    const RemotePtr target{target_raw};
    const IncarnationLockResult target_lock = try_lock_node(target);
    if (target_lock == IncarnationLockResult::stale) {
      // A reclaimed cleanup parent has no old-incarnation adjacency left to
      // edit.  Treat absence as the idempotent cleanup postcondition.
      continue;
    }
    if (target_lock == IncarnationLockResult::busy) return false;
    const byte_t* record = index_buffer_.get_full_buffer() +
      target.byte_offset();
    const u64 header = load_local_node_header_acquire(target);
    const node_t observed_id = *reinterpret_cast<const node_t*>(
      record + VamanaNode::offset_id());
    const u32 observed_generation = *reinterpret_cast<const u32*>(
      record + VamanaNode::offset_generation());
    const ReverseUpdateOp& identity = *removals.front();
    if ((header & VamanaNode::HEADER_RETIRING) != 0) {
      // This target has already stopped accepting graph mutations and will
      // itself be tombstoned only after its protected children are handed
      // off. Treat removal as complete; mutating its preserved adjacency
      // would race that cleanup snapshot without improving live reachability.
      unlock_node(target);
      continue;
    }
    if (VamanaNode::graph_mutation_quiesced(header)) {
      unlock_node(target);
      return false;
    }
    if (!cleanup_reverse_target_matches(
          identity.target_id, identity.target_generation,
          observed_id, observed_generation,
          (header & VamanaNode::HEADER_DELETED) != 0)) {
      unlock_node(target);
      continue;
    }

    // All operations grouped under one raw target must describe the same
    // target generation. Conflicting stale identities are ignored rather
    // than allowed to widen the deletion set.
    vec<RemotePtr> deleted;
    deleted.reserve(removals.size());
    for (const ReverseUpdateOp* removal : removals) {
      if (removal->target_id == observed_id &&
          removal->target_generation == observed_generation) {
        deleted.emplace_back(removal->candidate_raw);
      }
    }
    GraphAdjacency adjacency;
    if (!read_graph_adjacency(target, adjacency)) {
      unlock_node(target);
      return false;
    }
    const size_t old_stable_size = adjacency.stable.size();
    const size_t old_protected_size = adjacency.provisional.size();
    const auto remove_deleted =
      [&](vec<RemotePtr>& neighbors) {
        neighbors.erase(
          std::remove_if(neighbors.begin(), neighbors.end(),
                     [&](RemotePtr neighbor) {
                       return std::find(deleted.begin(), deleted.end(),
                                        neighbor) != deleted.end();
                     }),
          neighbors.end());
      };
    remove_deleted(adjacency.stable);
    remove_deleted(adjacency.provisional);
    if (adjacency.stable.size() != old_stable_size ||
        adjacency.provisional.size() != old_protected_size) {
      write_graph_adjacency(target, adjacency.stable,
                            adjacency.provisional,
                            adjacency.generation, false);
    }
    unlock_node(target);
  }
  return true;
}
