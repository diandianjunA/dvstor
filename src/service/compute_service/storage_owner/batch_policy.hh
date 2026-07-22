#pragma once

#include <algorithm>
#include <cstdint>
#include "common/types.hh"

namespace compute_service_detail {

struct StorageOwnerBatchDecision {
  bool tail_escape{};
  bool max_wait_flush{};
  u32 take{};
};

// A partial visible dequeue stopped at an MPMC publication hole.  The credit
// behind that hole belongs to the same (possibly already expired) batch, so it
// must inherit the old timestamp.  Only a completely consumed prefix starts a
// new tail's coalescing interval.
inline u64 next_storage_owner_batch_observed_ns(
    u64 previous_observed_ns,
    u32 remaining_tasks,
    u32 dequeued_tasks,
    u32 requested_tasks,
    u64 dequeued_at_ns) {
  if (remaining_tasks == 0) return 0;
  return dequeued_tasks < requested_tasks
    ? previous_observed_ns : dequeued_at_ns;
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
// Keep the policy deliberately conventional: a full batch is sent immediately;
// an isolated tail is sent immediately; a concurrent partial batch waits only
// until a hard maximum latency.  Crucially, an admitted tail is kept intact in
// one RPC.  Dividing it by the number of free slots creates a closed-loop
// collapse in which synchronous callers return one by one and every later RPC
// is another singleton.  Multiple full/expired batches may still occupy
// independent RPC slots, preserving storage-side request parallelism.
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
    return {.take = batch_max};
  }

  const bool isolated_tail = active_rpcs == 0 && pending_producers == 0;
  const bool max_wait_flush = !isolated_tail &&
    (max_wait_ns == 0 ||
     (oldest_ready_since_ns != 0 &&
      now_ns >= oldest_ready_since_ns &&
      now_ns - oldest_ready_since_ns >= max_wait_ns));
  if (!isolated_tail && !max_wait_flush) return {};

  return {
    .tail_escape = isolated_tail,
    .max_wait_flush = max_wait_flush,
    .take = std::min(ready_tasks, batch_max),
  };
}

}  // namespace compute_service_detail
