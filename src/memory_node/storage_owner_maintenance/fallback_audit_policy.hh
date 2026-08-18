#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>

namespace memory_node_storage_owner_maintenance_detail {

inline constexpr auto kStage2ActiveFallbackAuditInterval =
  std::chrono::milliseconds(1);
inline constexpr auto kStage2IdleFallbackAuditInterval =
  std::chrono::milliseconds(10);

inline constexpr std::chrono::milliseconds stage2_fallback_audit_interval(
    std::size_t active_contexts) {
  return active_contexts == 0
    ? kStage2IdleFallbackAuditInterval
    : kStage2ActiveFallbackAuditInterval;
}

template <typename Clock, typename Duration>
inline std::chrono::time_point<Clock, Duration>
stage2_fallback_audit_deadline(
    std::chrono::time_point<Clock, Duration> now,
    std::size_t active_contexts) {
  return now + std::chrono::duration_cast<Duration>(
    stage2_fallback_audit_interval(active_contexts));
}

// Preserve an existing periodic deadline while activity is unchanged.  The
// first live context must not inherit the idle 10 ms horizon, while the last
// release may immediately restore the low-overhead idle cadence.
template <typename Clock, typename Duration>
inline std::chrono::time_point<Clock, Duration>
refresh_stage2_fallback_audit_deadline(
    std::chrono::time_point<Clock, Duration> current_deadline,
    bool previously_active,
    std::size_t active_contexts,
    std::chrono::time_point<Clock, Duration> now) {
  const bool active = active_contexts != 0;
  if (active == previously_active) return current_deadline;
  const auto transition_deadline = stage2_fallback_audit_deadline(
    now, active_contexts);
  return active
    ? std::min(current_deadline, transition_deadline)
    : transition_deadline;
}

}  // namespace memory_node_storage_owner_maintenance_detail
