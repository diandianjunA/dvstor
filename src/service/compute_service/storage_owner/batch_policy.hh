#pragma once

#include <algorithm>
#include <cstdint>
#include "common/types.hh"

namespace compute_service_detail {

struct StorageOwnerBatchDecision {
  bool idle_flush{};
  bool max_wait_flush{};
  u32 take{};
};

inline constexpr u32 kStorageOwnerAmortizedBatchFloor = 4;

// Spread one already-admitted dispatch epoch over the transport credits that
// are available now. Greedily putting batch_max items into the first free RPC
// serializes their storage-side Stage1 work and can leave every other lane
// idle. Conversely, spreading a small epoch over every free credit destroys
// the storage-side execute/arm/release batching. Keep an amortized four-item
// floor whenever enough work exists; fewer than four items are sent only for a
// finite tail. ceil(ready/free) then grows naturally above that floor when the
// current credit window could not otherwise carry the epoch.
inline u32 balanced_storage_owner_batch_take(
    u32 ready_tasks, u32 free_rpc_slots, u32 batch_max) {
  if (ready_tasks == 0 || free_rpc_slots == 0 || batch_max == 0) return 0;
  const u32 fair_share =
    ready_tasks / free_rpc_slots + (ready_tasks % free_rpc_slots != 0);
  return std::min({
    ready_tasks,
    batch_max,
    std::max(kStorageOwnerAmortizedBatchFloor, fair_share),
  });
}

// An idle partial tail has no latency to hide behind another request and must
// retain all available batching: send it once rather than fragmenting it over
// empty lanes. Full or non-idle epochs use the amortized balanced policy above.
inline u32 storage_owner_dispatch_epoch_take(
    u32 ready_tasks,
    u32 free_rpc_slots,
    u32 batch_max,
    bool idle_flush) {
  if (ready_tasks == 0 || free_rpc_slots == 0 || batch_max == 0) return 0;
  if (idle_flush) return std::min(ready_tasks, batch_max);
  return balanced_storage_owner_batch_take(
    ready_tasks, free_rpc_slots, batch_max);
}

// A successful dequeue closes the current batching epoch. Remaining published
// credit belongs to a new epoch; carrying the old timestamp forward would make
// every later partial batch permanently expired under continuous load.
inline u64 rearm_storage_owner_batch_wait(
    u32 remaining_published_tasks, u64 dequeued_at_ns) {
  return remaining_published_tasks == 0 ? 0 : dequeued_at_ns;
}

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
// across rpc_depth tiny requests (N / rpc_depth items each). Conversely, a
// policy that waits for a full batch can strand a finite tail forever while
// unrelated RPCs or producers remain active.
//
// Form a bounded microbatch instead. A full batch is always ready immediately.
// An actually idle, isolated tail also progresses immediately. Otherwise the
// oldest observed published task supplies a hard deadline; a zero maximum wait
// flushes every visible tail immediately. This function only
// decides whether the single CQ/progress thread should attempt a dequeue; the
// external published count remains a hint and dequeue must still consume only
// the queue-visible FIFO prefix.
inline StorageOwnerBatchDecision decide_storage_owner_batch(
    u32 ready_tasks,
    u32 active_rpcs,
    u32 free_rpc_slots,
    u32 pending_producers,
    u32 batch_max,
    u64 oldest_ready_since_ns,
    u64 now_ns,
    u64 max_wait_ns) {
  if (batch_max == 0) return {};
  if (free_rpc_slots == 0 || ready_tasks == 0) {
    return {};
  }

  if (ready_tasks >= batch_max) {
    return {.take = storage_owner_dispatch_epoch_take(
              ready_tasks, free_rpc_slots, batch_max, false)};
  }

  const bool idle_flush = active_rpcs == 0 && pending_producers == 0;
  const bool max_wait_flush = !idle_flush &&
    (max_wait_ns == 0 ||
     (oldest_ready_since_ns != 0 &&
      now_ns >= oldest_ready_since_ns &&
      now_ns - oldest_ready_since_ns >= max_wait_ns));
  if (!idle_flush && !max_wait_flush) return {};

  return {
    .idle_flush = idle_flush,
    .max_wait_flush = max_wait_flush,
    .take = storage_owner_dispatch_epoch_take(
      ready_tasks, free_rpc_slots, batch_max, idle_flush),
  };
}

}  // namespace compute_service_detail
