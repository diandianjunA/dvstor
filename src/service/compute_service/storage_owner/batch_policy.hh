#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include "common/types.hh"

namespace compute_service_detail {

struct StorageOwnerBatchDecision {
  bool tail_escape{};
  bool max_wait_flush{};
  bool occupancy_flush{};
  bool adaptive_wait_flush{};
  u32 take{};
};

struct StorageOwnerBatchPressure {
  u32 rpc_capacity{};
  u32 concurrency_target{};
  u32 low_water{};
  u32 efficient_quantum{};
  u32 ramp_lanes{};
  u32 launch_quantum{};
  u64 adaptive_wait_ns{};
};

// Keep one transport slot outside the normal partial-batch ramp.  A full batch
// or a batch at its hard deadline may still consume that slot, so this is not
// unused capacity: it prevents a burst of small partials from blocking an
// already-coalesced or latency-bound batch behind them.
inline StorageOwnerBatchPressure storage_owner_batch_pressure(
    u32 ready_tasks,
    u32 active_rpcs,
    u32 free_rpc_slots,
    u32 pending_producers,
    u32 batch_max,
    u64 max_wait_ns) {
  const u64 capacity64 = static_cast<u64>(active_rpcs) + free_rpc_slots;
  const u32 capacity = static_cast<u32>(std::min<u64>(
    capacity64, std::numeric_limits<u32>::max()));
  if (capacity == 0 || batch_max == 0) return {};

  const u32 target = capacity > 1 ? capacity - 1 : 1;
  const u32 low_water = std::max<u32>(1, (target + 1) / 2);
  const u64 offered = static_cast<u64>(ready_tasks) + pending_producers;
  // One full batch distributed over the low-water window defines the
  // smallest efficient ramp quantum.  More offered load increases, rather
  // than decreases, the per-RPC quantum.  This avoids both extremes of one
  // giant partial RPC and a permanent singleton closed loop.
  const u32 efficient_quantum = static_cast<u32>(std::max<u64>(
    1, (static_cast<u64>(batch_max) + low_water - 1) / low_water));
  u32 ramp_lanes = 1;
  u64 per_lane = std::max<u64>(1, offered);
  if (active_rpcs < low_water) {
    const u32 vacant_low_water_lanes = low_water - active_rpcs;
    const u64 demand_lanes = std::max<u64>(
      1, (offered + efficient_quantum - 1) / efficient_quantum);
    ramp_lanes = static_cast<u32>(std::min<u64>(
      vacant_low_water_lanes, demand_lanes));
    per_lane = (offered + ramp_lanes - 1) / ramp_lanes;
  }
  const u32 launch_quantum = static_cast<u32>(std::max<u64>(
    1, std::min<u64>(batch_max, per_lane)));

  u64 adaptive_wait_ns = max_wait_ns;
  if (max_wait_ns != 0 && active_rpcs < target) {
    // Scale the coalescing interval from roughly max_wait / capacity at an
    // empty pipeline to max_wait near the target.  This is capacity-derived,
    // not a dataset/QPS threshold.  The subtraction form cannot overflow and
    // always remains within the configured hard maximum.
    const u64 denominator = static_cast<u64>(target) + 1;
    const u64 numerator = static_cast<u64>(active_rpcs) + 1;
    const u64 unit = max_wait_ns / denominator;
    adaptive_wait_ns = max_wait_ns - unit * (denominator - numerator);
    adaptive_wait_ns = std::max<u64>(1, adaptive_wait_ns);
  }

  return {
    .rpc_capacity = capacity,
    .concurrency_target = target,
    .low_water = low_water,
    .efficient_quantum = efficient_quantum,
    .ramp_lanes = ramp_lanes,
    .launch_quantum = launch_quantum,
    .adaptive_wait_ns = adaptive_wait_ns,
  };
}

inline bool storage_owner_batch_wait_elapsed(
    u64 oldest_ready_since_ns,
    u64 now_ns,
    u64 wait_ns) {
  return wait_ns == 0 ||
    (oldest_ready_since_ns != 0 &&
     now_ns >= oldest_ready_since_ns &&
     now_ns - oldest_ready_since_ns >= wait_ns);
}

// Any published work remaining after a dequeue may belong to the same oldest
// batch (including an MPMC prefix hidden behind a publication hole), so it
// inherits the old timestamp. Only observing the published queue empty ends
// that coalescing interval.
inline u64 next_storage_owner_batch_observed_ns(
    u64 previous_observed_ns,
    u32 remaining_tasks,
    u32 dequeued_tasks,
    u32 requested_tasks,
    u64 dequeued_at_ns) {
  if (remaining_tasks == 0) return 0;
  (void)dequeued_tasks;
  (void)requested_tasks;
  // Every task left behind by this dequeue may already have shared the
  // previous oldest timestamp. Resetting its age merely because the requested
  // prefix was fully visible lets an old tail exceed max_wait repeatedly
  // (e.g. dequeue 32 from 44, then restart the remaining 12 at age zero).
  // Keep the conservative old timestamp until the published queue is observed
  // empty. If only newly-arrived work remains this can flush it slightly early,
  // but never violates the configured latency bound.
  return previous_observed_ns != 0
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
// A full batch and an isolated tail are sent immediately.  Under concurrent
// load, however, waiting every partial for max_wait can leave most of the RPC
// window idle: synchronous callers cannot publish their next task until a
// previous RPC returns.  Ramp toward the existing concurrency window by
// partitioning only the *offered demand* (published + announced producers)
// across its vacant lanes.  This produces a load-derived launch quantum rather
// than forcing singleton RPCs.  The coalescing interval grows with occupancy;
// near the target it becomes max_wait again.
//
// Below low water, a visible prefix is divided only by the load-derived launch
// quantum so a closed loop can actually populate several lanes.  It is never
// divided into a fixed singleton policy.  An adaptive/deadline flush keeps the
// remaining visible tail intact, and batch_max/max_wait remain hard bounds.
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

  const auto pressure = storage_owner_batch_pressure(
    ready_tasks, active_rpcs, free_rpc_slots, pending_producers,
    batch_max, max_wait_ns);
  if (pressure.rpc_capacity == 0) return {};

  const bool max_wait_flush = storage_owner_batch_wait_elapsed(
    oldest_ready_since_ns, now_ns, max_wait_ns);
  if (max_wait_flush) {
    return {
      .max_wait_flush = true,
      .take = std::min(ready_tasks, batch_max),
    };
  }

  const bool isolated_tail = active_rpcs == 0 &&
    pending_producers == 0 &&
    ready_tasks <= pressure.efficient_quantum;
  if (isolated_tail) {
    return {
      .tail_escape = true,
      .take = std::min(ready_tasks, batch_max),
    };
  }

  if (active_rpcs >= pressure.concurrency_target) {
    return {};
  }

  const bool adaptive_wait_flush = storage_owner_batch_wait_elapsed(
    oldest_ready_since_ns, now_ns, pressure.adaptive_wait_ns);
  if (adaptive_wait_flush) {
    return {
      .occupancy_flush = true,
      .adaptive_wait_flush = true,
      .take = std::min(ready_tasks, batch_max),
    };
  }

  const bool below_low_water = active_rpcs < pressure.low_water;
  // Do not let the last low-water lane enter a permanent singleton loop.
  // With synchronous producers, a one-item RPC can complete and publish its
  // replacement before the sender observes any `pending_producers`; using the
  // instantaneous launch_quantum alone would then keep active_rpcs at
  // low_water - 1 forever.  Outside the true isolated-tail and deadline paths,
  // require at least the capacity-derived efficient quantum.  The adaptive
  // deadline still bounds latency when a partial tail cannot reach it.
  const u32 occupancy_quantum = std::max(
    pressure.efficient_quantum, pressure.launch_quantum);
  const bool launch_quantum_ready = ready_tasks >= occupancy_quantum;
  if (!below_low_water || !launch_quantum_ready) return {};

  return {
    .occupancy_flush = true,
    .take = std::min(ready_tasks, occupancy_quantum),
  };
}

}  // namespace compute_service_detail
