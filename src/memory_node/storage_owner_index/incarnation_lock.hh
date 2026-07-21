#pragma once

#include <atomic>

#include "common/types.hh"
#include "vamana/vamana_node.hh"

namespace memory_node_storage_owner_index_detail {

// A data-plane handle can outlive the physical occupant of its slot.  Keep
// contention separate from a conclusive incarnation mismatch so callers can
// retry the former and treat the latter as a stale/idempotent postcondition.
enum class IncarnationLockResult : u8 {
  locked,
  busy,
  stale,
};

// Performs exactly one identity-fenced CAS.  The incarnation is part of the
// value compared by the CAS, so a slot recycled after an optimistic validity
// check cannot be locked through the old handle.  Incarnations never wrap;
// consequently the physical-address ABA is converted into a tag mismatch.
inline IncarnationLockResult try_lock_header_once(
    u64& header_storage, u32 expected_incarnation) {
  std::atomic_ref<u64> header_ref(header_storage);
  u64 observed = header_ref.load(std::memory_order_acquire);
  if (VamanaNode::header_incarnation(observed) != expected_incarnation) {
    return IncarnationLockResult::stale;
  }
  if ((observed & VamanaNode::HEADER_NODE_LOCK) != 0) {
    return IncarnationLockResult::busy;
  }

  const u64 desired = observed | VamanaNode::HEADER_NODE_LOCK;
  if (header_ref.compare_exchange_strong(
        observed, desired, std::memory_order_acq_rel,
        std::memory_order_acquire)) {
    return IncarnationLockResult::locked;
  }
  return VamanaNode::header_incarnation(observed) == expected_incarnation
    ? IncarnationLockResult::busy
    : IncarnationLockResult::stale;
}

}  // namespace memory_node_storage_owner_index_detail
