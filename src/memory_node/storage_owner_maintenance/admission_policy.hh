#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace memory_node_storage_owner_maintenance_detail {

enum class Stage2AdmissionDecision : std::uint8_t {
  admit,
  unavailable,
  foreground_pressure,
};

inline std::size_t saturating_admission_multiply(
    std::size_t lhs, std::size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return std::numeric_limits<std::size_t>::max();
  }
  return lhs * rhs;
}

// Bound accepted Stage2 descriptors independently of active execution
// resources. Stage1 may acknowledge every descriptor in this window after it
// is assigned a completion sequence and published to the maintenance queue;
// workers claim their separately bounded context/lane resources only later.
//
// The queue bound protects descriptor memory, while the completion-ring bound
// prevents a modulo cell from being reused before the contiguous durable
// watermark has crossed it. The accepted window must respect both. Neither
// maintenance worker count nor RPC depth belongs here: tying acceptance to
// active contexts recreates a completion-clocked foreground pipeline and
// prevents the queue from accumulating a useful Stage2 batch.
inline std::size_t stage2_accepted_sequence_limit(
    std::size_t maintenance_queue_depth,
    std::size_t completion_capacity) {
  return std::min(maintenance_queue_depth, completion_capacity);
}

// Stage2 contexts are execution resources, not foreground acceptance credit.
// Keep their bound stable across foreground and drain: expanding from four to
// the full RPC depth when the foreground becomes idle moved thousands of
// tasks out of the visible queue into whole-context barrier chains, precisely
// when drain needed bounded independent work. The existing global/per-peer
// RDMA credits still bound posted work, while a depth-one configuration
// naturally retains the original single-context floor.
inline std::size_t stage2_context_admission_limit(
    std::size_t maintenance_workers,
    std::size_t rpc_depth,
    bool foreground_pressure) {
  (void)foreground_pressure;
  const std::size_t workers = std::max<std::size_t>(1, maintenance_workers);
  const std::size_t depth = std::max<std::size_t>(1, rpc_depth);
  const std::size_t contexts_per_worker = std::min<std::size_t>(depth, 4);
  return saturating_admission_multiply(workers, contexts_per_worker);
}

// The global context counter above is a debt/scratch bound, not a fair-share
// scheduler.  Without this local bound, the first OS worker to run can claim
// the entire global allowance even though it owns only its own small search
// lane pool; the other workers and their RDMA lanes then remain idle.  Apply
// the same per-worker share before touching the global counter.  This changes
// only which executor owns an admitted context, never the search, completion
// window, or amount of acknowledged work.
inline std::size_t stage2_worker_context_admission_limit(
    std::size_t rpc_depth,
    bool foreground_pressure) {
  (void)foreground_pressure;
  const std::size_t depth = std::max<std::size_t>(1, rpc_depth);
  return std::min<std::size_t>(depth, 4);
}

// Active Stage2 tasks are a second, node-wide execution bound. Context count
// alone was insufficient while one context could consume a 16/32-item cohort:
// even 32 contexts could hide 1024 coupled search/prune chains from the queue.
// Budget four eight-task semantic slices per executor (256 tasks for the
// tested 8-worker production geometry) and leave later accepted descriptors
// visible. This is deliberately unrelated to Stage1 ACK admission; the exact
// task account also remains correct if later scheduling changes make context
// sizes non-uniform.
inline constexpr std::size_t kStage2ActiveTaskContextsPerWorker = 4;

inline std::size_t stage2_active_task_limit(
    std::size_t maintenance_workers,
    std::size_t semantic_execution_batch,
    std::size_t accepted_descriptor_limit) {
  const std::size_t workers = std::max<std::size_t>(1, maintenance_workers);
  const std::size_t batch = std::max<std::size_t>(1,
                                                   semantic_execution_batch);
  const std::size_t contexts = saturating_admission_multiply(
    workers, kStage2ActiveTaskContextsPerWorker);
  return std::min(
    accepted_descriptor_limit,
    saturating_admission_multiply(contexts, batch));
}

inline bool stage2_active_task_reservation_available(
    std::uint32_t active_tasks,
    std::uint32_t requested_tasks,
    std::uint32_t limit) {
  return requested_tasks != 0 && active_tasks <= limit &&
    requested_tasks <= limit - active_tasks;
}

inline bool try_reserve_stage2_active_tasks(
    std::atomic<std::uint32_t>& active_tasks,
    std::uint32_t requested_tasks,
    std::uint32_t limit) {
  std::uint32_t active = active_tasks.load(std::memory_order_acquire);
  while (stage2_active_task_reservation_available(
      active, requested_tasks, limit)) {
    if (active_tasks.compare_exchange_weak(
          active, active + requested_tasks,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

inline bool try_release_stage2_active_tasks(
    std::atomic<std::uint32_t>& active_tasks,
    std::uint32_t released_tasks) {
  if (released_tasks == 0) return false;
  std::uint32_t active = active_tasks.load(std::memory_order_acquire);
  while (released_tasks <= active) {
    if (active_tasks.compare_exchange_weak(
          active, active - released_tasks,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

// Credit-return callbacks can race and observe the same completion-ring
// availability. Runnable waiter coverage is a scheduler-only claim that makes
// those snapshots idempotent without reserving a maintenance sequence.
inline std::size_t stage1_waiter_uncovered_wake_capacity(
    std::size_t resource_available,
    std::size_t runnable_coverage) {
  return resource_available > runnable_coverage
    ? resource_available - runnable_coverage : 0;
}

// Wake an oversized FIFO head even when only one semantic credit is visible;
// its existing per-token fallback makes partial progress. Cover its *whole*
// demand as a scheduling baton: if another 31 credits arrive before this
// 32-item request runs, a second whole request must not be woken for the same
// eventual capacity. This is not durable debt or a sequence reservation.
inline std::size_t stage1_waiter_head_wake_coverage(
    std::size_t item_count,
    std::size_t uncovered_capacity) {
  if (item_count == 0 || uncovered_capacity == 0) return 0;
  return item_count;
}

// A Stage1 arm permit bridges the queue-capacity check and the try-only
// completion-ring transaction. Other producers must include those permits in
// their queue-capacity test or they could steal a runnable slot after arm has
// reserved a sequence.
inline bool maintenance_queue_permit_available(
    std::size_t runnable_tasks,
    std::size_t reserved_slots,
    std::size_t capacity) {
  return runnable_tasks < capacity &&
    reserved_slots < capacity - runnable_tasks;
}

// A control RPC is one admission transaction.  Claiming its queue permits
// item-by-item can leave a partial batch runnable while the caller is still
// waiting for the remaining items.  Those runnable tasks may in turn wait for
// the caller's authority commit, creating a closed wait cycle.  Admit all
// items together or leave the queue unchanged.
inline bool maintenance_queue_batch_permit_available(
    std::size_t runnable_tasks,
    std::size_t reserved_slots,
    std::size_t requested_slots,
    std::size_t capacity) {
  if (requested_slots == 0 || runnable_tasks > capacity ||
      reserved_slots > capacity - runnable_tasks) {
    return false;
  }
  return requested_slots <=
    capacity - runnable_tasks - reserved_slots;
}

// Update the shared permit account only when the entire batch fits. Keeping
// this arithmetic in one helper makes both the no-partial-admission property
// and the transient-failure no-op directly testable.
inline bool try_acquire_maintenance_queue_batch_permit(
    std::size_t runnable_tasks,
    std::size_t requested_slots,
    std::size_t capacity,
    std::size_t& reserved_slots) {
  if (!maintenance_queue_batch_permit_available(
        runnable_tasks, reserved_slots, requested_slots, capacity)) {
    return false;
  }
  reserved_slots += requested_slots;
  return true;
}

// Release is similarly checked so a failed completion-ring try cannot wrap
// the reservation account and silently expose more queue capacity than exists.
inline bool release_maintenance_queue_batch_permit(
    std::size_t released_slots,
    std::size_t& reserved_slots) {
  if (released_slots == 0 || released_slots > reserved_slots) return false;
  reserved_slots -= released_slots;
  return true;
}

// Avoid calling the pressure probe when this executor cannot admit work at
// all. The production probe may poll a shared peer CQ, so shutdown/full paths
// must remain side-effect free.
template <class ForegroundPressureProbe>
Stage2AdmissionDecision decide_stage2_admission(
    bool local_contexts_full,
    bool shutting_down,
    ForegroundPressureProbe&& foreground_pressure) {
  if (local_contexts_full || shutting_down) {
    return Stage2AdmissionDecision::unavailable;
  }
  return std::forward<ForegroundPressureProbe>(foreground_pressure)()
           ? Stage2AdmissionDecision::foreground_pressure
           : Stage2AdmissionDecision::admit;
}

}  // namespace memory_node_storage_owner_maintenance_detail
