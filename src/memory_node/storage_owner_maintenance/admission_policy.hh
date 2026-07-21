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

// Foreground pressure may reduce asynchronous Stage2 concurrency, but it must
// never suppress every dedicated maintenance executor. Stage1 publication
// reserves an ordered completion ticket before replying to the compute node;
// if all Stage2 contexts yield while foreground workers wait for that window,
// neither side can make progress. Keep one context per physical maintenance
// worker under pressure and restore the full per-worker RPC depth otherwise.
inline std::size_t stage2_context_admission_limit(
    std::size_t maintenance_workers,
    std::size_t rpc_depth,
    bool foreground_pressure) {
  const std::size_t workers = std::max<std::size_t>(1, maintenance_workers);
  if (foreground_pressure) return workers;
  const std::size_t depth = std::max<std::size_t>(1, rpc_depth);
  if (workers > std::numeric_limits<std::size_t>::max() / depth) {
    return std::numeric_limits<std::size_t>::max();
  }
  return workers * depth;
}

// A Stage1 arm permit is counted before it waits for completion-ring credit.
// Other producers must include those permits in their queue-capacity test or
// they could steal the slot after arm has reserved a sequence.
inline bool maintenance_queue_permit_available(
    std::size_t runnable_tasks,
    std::size_t reserved_slots,
    std::size_t capacity) {
  return runnable_tasks < capacity &&
    reserved_slots < capacity - runnable_tasks;
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
