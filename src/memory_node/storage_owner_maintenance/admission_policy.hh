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

// Bound acknowledged-but-unfinished Stage2 debt.  The caller supplies the
// pre-rebalance Stage2 lane count, so moving CPUs from reverse processing to
// Stage2 cannot alter this window for any wire batch size.  Four wire batches
// are enough to absorb burst and batch-formation jitter; at larger dedicated
// deployments, one task per available worker/RPC context keeps every executor
// feedable.  The legacy four-tasks-per-context bound remains a ceiling unless
// one complete wire batch itself is larger.
//
// In particular, moving two CPUs from an idle reverse pool to Stage2 on the
// five-way colocated deployment must improve the rate at which debt is paid,
// not double the amount of debt hidden before backpressure starts.
inline std::size_t stage2_sequence_admission_limit(
    std::size_t maintenance_workers,
    std::size_t rpc_depth,
    std::size_t batch_max) {
  const std::size_t workers = std::max<std::size_t>(1, maintenance_workers);
  const std::size_t depth = std::max<std::size_t>(1, rpc_depth);
  const std::size_t batch = std::max<std::size_t>(1, batch_max);
  const std::size_t contexts = saturating_admission_multiply(workers, depth);
  const std::size_t legacy_limit =
    saturating_admission_multiply(contexts, 4);
  const std::size_t batch_burst = saturating_admission_multiply(batch, 4);
  const std::size_t service_demand = std::max(contexts, batch_burst);
  return std::max(batch, std::min(legacy_limit, service_demand));
}

// Foreground pressure may reduce asynchronous Stage2 concurrency, but it must
// never collapse the pipeline to one context per executor.  One context per
// worker exposed no latency-hiding headroom on the colocated deployment even
// though bounded RDMA credits and scratch lanes were still available. Retain
// at most two contexts per worker under pressure and restore the full
// per-worker RPC depth otherwise.
// The existing global/per-peer RDMA credits still bound posted work, while a
// depth-one configuration naturally retains the original single-context
// floor.
inline std::size_t stage2_context_admission_limit(
    std::size_t maintenance_workers,
    std::size_t rpc_depth,
    bool foreground_pressure) {
  const std::size_t workers = std::max<std::size_t>(1, maintenance_workers);
  const std::size_t depth = std::max<std::size_t>(1, rpc_depth);
  const std::size_t contexts_per_worker = foreground_pressure
    ? std::min<std::size_t>(depth, 2) : depth;
  if (workers >
      std::numeric_limits<std::size_t>::max() / contexts_per_worker) {
    return std::numeric_limits<std::size_t>::max();
  }
  return workers * contexts_per_worker;
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
