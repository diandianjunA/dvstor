#pragma once

#include <cstdint>
#include <utility>

#include "common/types.hh"

namespace compute_service_detail {

// The storage-owner sender normally observes a one-item queue because the
// public mutation API is synchronous and its dedicated CQ thread can dequeue
// faster than the other application threads finish centroid routing.  Waiting
// for a fixed time would penalize an isolated mutation and introduce a
// workload-specific tuning knob.  Instead, admit a few scheduler handoff
// rounds only when another producer has explicitly announced work for this
// same logical owner.
inline constexpr u32 kConcurrentProducerBatchRounds = 4;

struct ConcurrentBatchDrainResult {
  u32 item_count{};
  u32 wait_rounds{};
};

template <class TryPop, class HasPendingProducer,
          class Append, class Relax>
ConcurrentBatchDrainResult drain_concurrent_storage_owner_batch(
    u32 initial_items,
    u32 max_items,
    TryPop&& try_pop,
    HasPendingProducer&& has_pending_producer,
    Append&& append,
    Relax&& relax) {
  ConcurrentBatchDrainResult result{.item_count = initial_items};
  while (result.item_count < max_items) {
    u32 task_id = 0;
    if (try_pop(task_id)) {
      append(task_id);
      ++result.item_count;
      continue;
    }
    if (result.wait_rounds >= kConcurrentProducerBatchRounds ||
        !has_pending_producer()) {
      // Closing an announcement is release-ordered after queue publication.
      // Re-probe once after observing that closure so the producer cannot
      // strand a just-published task in the next singleton RPC.
      if (try_pop(task_id)) {
        append(task_id);
        ++result.item_count;
        continue;
      }
      break;
    }
    ++result.wait_rounds;
    relax();
  }
  return result;
}

}  // namespace compute_service_detail
