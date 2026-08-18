#pragma once

#include <algorithm>
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

// Keep the acknowledged-but-unfinished task window aligned with the actual
// runtime pipeline rather than a historical worker count.  One search lane
// needs one runnable context plus one context suspended on an RPC/resource
// edge.  Production contexts contain at most logical_batch_limit tasks; wire
// aggregation is independently bounded by the home/reverse outboxes.
//
// The resulting window is still exact incomplete-task credit, not queue
// capacity: the completion ring and every transport retain their existing
// independent bounds.
inline std::size_t stage2_sequence_admission_limit(
    std::size_t maintenance_workers,
    std::size_t rpc_depth,
    std::size_t active_search_lanes,
    std::size_t logical_batch_limit,
    std::size_t wire_batch_max) {
  const std::size_t workers = std::max<std::size_t>(1, maintenance_workers);
  const std::size_t depth = std::max<std::size_t>(1, rpc_depth);
  const std::size_t physical_contexts =
    saturating_admission_multiply(workers, depth);
  const std::size_t pipeline_contexts = std::min(
    physical_contexts,
    saturating_admission_multiply(
      std::max<std::size_t>(1, active_search_lanes), 2));
  const std::size_t pipeline_tasks = saturating_admission_multiply(
    pipeline_contexts, std::max<std::size_t>(1, logical_batch_limit));
  // Stage1 reserves and publishes one wire request atomically. Even a tiny
  // executor must therefore be able to arm one complete legal batch; pipeline
  // geometry controls additional debt above that correctness floor.
  return std::max(
    std::max<std::size_t>(1, wire_batch_max), pipeline_tasks);
}

// Foreground pressure may reduce asynchronous Stage2 concurrency, but it must
// never collapse the pipeline to one context per executor. Logical search
// state is context-owned, and a context waiting on a home RPC releases its
// registered RDMA lane. Keep two bounded contexts per active lane under
// foreground pressure: one can own the lane while the other independently
// waits for a home response/resource edge. Restore the full per-worker RPC
// depth otherwise.
// The existing global/per-peer RDMA credits still bound posted work, while a
// depth-one configuration naturally retains the original single-context
// floor.
inline std::size_t stage2_context_admission_limit(
    std::size_t maintenance_workers,
    std::size_t rpc_depth,
    std::size_t active_search_lanes,
    bool foreground_pressure) {
  const std::size_t workers = std::max<std::size_t>(1, maintenance_workers);
  const std::size_t depth = std::max<std::size_t>(1, rpc_depth);
  const std::size_t physical_contexts =
    saturating_admission_multiply(workers, depth);
  if (!foreground_pressure) return physical_contexts;
  return std::min(
    physical_contexts,
    saturating_admission_multiply(
      std::max<std::size_t>(1, active_search_lanes), 2));
}

// The global context counter above is a debt/scratch bound, not a fair-share
// scheduler.  Without this local bound, the first OS worker to run can claim
// the entire global allowance even though it owns only its own small search
// lane pool; the other workers and their RDMA lanes then remain idle.  Apply
// the same per-worker share before touching the global counter.  This changes
// only which executor owns an admitted context, never the search, completion
// window, or amount of acknowledged work.
inline std::size_t stage2_worker_context_admission_limit(
    std::size_t worker_id,
    std::size_t maintenance_workers,
    std::size_t rpc_depth,
    std::size_t active_search_lanes,
    bool foreground_pressure) {
  const std::size_t depth = std::max<std::size_t>(1, rpc_depth);
  if (!foreground_pressure) return depth;
  const std::size_t workers =
    std::max<std::size_t>(1, maintenance_workers);
  const std::size_t lanes = std::max<std::size_t>(1, active_search_lanes);
  const std::size_t lane_share = lanes / workers +
    (worker_id < lanes % workers ? 1 : 0);
  return std::min(
    depth,
    std::max<std::size_t>(
      1, saturating_admission_multiply(lane_share, 2)));
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
