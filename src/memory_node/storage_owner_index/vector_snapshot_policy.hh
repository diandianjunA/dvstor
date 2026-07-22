#pragma once

#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace memory_node_storage_owner_index_detail {

// A failed optimistic read is not synonymous with a dead graph node.  Keep
// the three outcomes explicit so construction search cannot accidentally
// turn lock contention (or a header change around the payload read) into a
// missing candidate/empty adjacency.
enum class StableNodeSnapshotState : u8 {
  stable,
  terminal,
  retryable,
};

// Classifies only whether one physical record was observed coherently.  It
// intentionally ignores lifecycle policy bits: callers that successfully
// captured a coherent deleted/provisional snapshot may still need its logical
// metadata before deciding eligibility.
inline StableNodeSnapshotState classify_physical_node_snapshot(
    RemotePtr pointer,
    u64 before,
    u64 after,
    u32 slot_incarnation) {
  const bool same_unlocked_header = before == after &&
    (after & VamanaNode::HEADER_NODE_LOCK) == 0;
  if (!same_unlocked_header) {
    return StableNodeSnapshotState::retryable;
  }
  if (VamanaNode::header_incarnation(after) != pointer.incarnation() ||
      slot_incarnation != pointer.incarnation()) {
    return StableNodeSnapshotState::terminal;
  }
  return StableNodeSnapshotState::stable;
}

inline StableNodeSnapshotState classify_stable_node_snapshot(
    RemotePtr pointer,
    u64 before,
    u64 after,
    u32 slot_incarnation) {
  const StableNodeSnapshotState physical = classify_physical_node_snapshot(
    pointer, before, after, slot_incarnation);
  if (physical != StableNodeSnapshotState::stable) return physical;
  if ((after & (VamanaNode::HEADER_DELETED |
                VamanaNode::HEADER_PROVISIONAL)) != 0) {
    return StableNodeSnapshotState::terminal;
  }
  return StableNodeSnapshotState::stable;
}

// A Stage2 source is deliberately PROVISIONAL until the continuation has
// converged and the final graph has been published.  Its physical-identity
// gate therefore has different lifecycle semantics from an ordinary search
// candidate: PROVISIONAL (and a previously frozen source on retry) is valid.
// What matters here is that one unlocked observation proves the same slot,
// logical id, and generation.  Lock ownership or a changing header always
// wins over apparent terminal bits/fields; those observations must be retried
// instead of retiring a live maintenance task.
inline StableNodeSnapshotState classify_stage2_target_snapshot(
    RemotePtr pointer,
    u64 before,
    u64 after,
    u32 slot_incarnation,
    node_t observed_id,
    u32 observed_generation,
    node_t expected_id,
    u32 expected_generation) {
  const StableNodeSnapshotState physical = classify_physical_node_snapshot(
    pointer, before, after, slot_incarnation);
  if (physical != StableNodeSnapshotState::stable) return physical;
  if ((after & VamanaNode::HEADER_DELETED) != 0 ||
      observed_id != expected_id ||
      observed_generation != expected_generation) {
    return StableNodeSnapshotState::terminal;
  }
  return StableNodeSnapshotState::stable;
}

// Stage2 may consume a vector directly from local storage or registered RDMA
// scratch only when one immutable physical incarnation spans the complete
// observation. Deleted and provisional records are not members of the stable
// continuation graph. RETIRING/FROZEN are intentionally not rejected here:
// the existing snapshot path also admits them for search while excluding only
// mutation/query-invalid flags.
inline bool stable_vector_snapshot_valid(RemotePtr pointer,
                                         u64 before,
                                         u64 after,
                                         u32 slot_incarnation) {
  return classify_stable_node_snapshot(
           pointer, before, after, slot_incarnation) ==
    StableNodeSnapshotState::stable;
}

}  // namespace memory_node_storage_owner_index_detail
