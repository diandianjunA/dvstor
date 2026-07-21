#pragma once

#include <algorithm>
#include <cstdint>
#include "common/types.hh"

namespace compute_service_detail {

struct StorageOwnerBatchDecision {
  bool saturated{};
  bool tail_escape{};
  u32 take{};
};

// Queue::push_wait() is linearizable, but a Vyukov MPMC producer reserves its
// FIFO position before publishing that cell.  A later producer may therefore
// publish position N+1 (and increment the external published-task count) while
// a preempted producer still owns the invisible position N.  The single
// consumer must treat the counter as an admission/batching hint, not as proof
// that the whole requested FIFO prefix is immediately dequeue-visible.
//
// Return the exact visible prefix.  In particular, an invisible first cell
// consumes neither an RPC slot nor published-task credit; a partial prefix is
// safe to send and is charged by its actual size.
template <class Queue, class Output>
inline u32 dequeue_storage_owner_visible_prefix(
    Queue& queue, u32 requested, Output& output) {
  u32 popped = 0;
  while (popped < requested) {
    u32 task_id = 0;
    if (!queue.try_pop(task_id)) break;
    output.push_back(task_id);
    ++popped;
  }
  return popped;
}

// A synchronous caller population forms a closed queueing loop. If every RPC
// slot consumes the first item it sees, N callers are permanently fragmented
// across rpc_depth tiny requests (N / rpc_depth items each). This policy keeps
// isolated writes immediate, then latches a saturated epoch once every lane is
// busy or a full batch is already ready. During that epoch only full batches
// open additional lanes; the last active lane is the progress escape for a
// finite tail. No timer, scheduler yield, or dataset-specific threshold is
// involved, and the CQ progress thread never waits for a producer.
inline StorageOwnerBatchDecision decide_storage_owner_batch(
    bool saturated,
    u32 ready_tasks,
    u32 active_rpcs,
    u32 free_rpc_slots,
    u32 pending_producers,
    u32 batch_max) {
  if (batch_max == 0) return {};
  if (!saturated &&
      (ready_tasks >= batch_max ||
       (free_rpc_slots == 0 && ready_tasks != 0))) {
    saturated = true;
  }
  if (saturated && active_rpcs == 0 && ready_tasks == 0 &&
      pending_producers == 0) {
    saturated = false;
  }
  if (free_rpc_slots == 0 || ready_tasks == 0) {
    return {.saturated = saturated};
  }
  // A producer increments pending_producers before publishing its queue
  // entry.  When no RPC is active, wait for those already-announced entries
  // to become visible instead of prematurely sending the currently visible
  // tail.  Once every producer closes its announcement, the tail escape below
  // guarantees progress without a timer.
  if (saturated && ready_tasks < batch_max &&
      (active_rpcs != 0 || pending_producers != 0)) {
    return {.saturated = true};
  }
  const bool tail_escape = saturated && active_rpcs == 0 &&
    ready_tasks < batch_max;
  return {
    .saturated = saturated,
    .tail_escape = tail_escape,
    .take = std::min(ready_tasks, batch_max),
  };
}

}  // namespace compute_service_detail
