#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tools::breakdown_benchmark {

// Keep this in sync with kFinalizeLatencyBucketUpperNs. The explicit 5000 ms
// bucket makes the acceptance predicate p99 <= 5 s exact instead of rounding
// every latency in (4 s, 8 s] up to 8 s.
inline constexpr size_t kMaintenanceLatencyBucketCount = 18;

struct MaintenanceObservation {
  uint64_t stitch_enqueued{};
  uint64_t stitched_live{};
  uint64_t stale{};
  uint64_t remaining{};
  uint64_t peer_reverse_remaining{};
  uint64_t failed{};
  uint64_t peer_reverse_failed{};
  double p99_stitch_delay_upper_ms{};
  bool p99_stitch_delay_over_30s{};
  std::array<uint64_t, kMaintenanceLatencyBucketCount>
    stitch_delay_histogram{};
  bool failure_counters_available{};
  bool stitch_delay_histogram_available{};

  uint64_t backlog() const;
};

struct MaintenanceLogCursor {
  std::string path;
  uint64_t offset{};
  MaintenanceObservation baseline;
  bool baseline_available{};
};

struct MaintenanceLogSummary {
  size_t requested_logs{};
  size_t readable_logs{};
  size_t logs_with_observations{};
  size_t logs_with_slope_observations{};
  size_t observations{};
  uint64_t remaining{};
  uint64_t max_backlog_observed{};
  uint64_t failures{};
  double p99_stitch_delay_upper_ms{};
  bool p99_stitch_delay_over_30s{};
  uint64_t p99_stitch_delay_samples{};
  bool p99_stitch_delay_available{};
  size_t logs_with_failure_deltas{};
  size_t logs_with_histogram_deltas{};
  bool failure_delta_available{};
  double backlog_slope_per_sec{};
  bool backlog_slope_available{};
  std::vector<std::string> unreadable_logs;
};

std::vector<MaintenanceLogCursor> snapshot_maintenance_logs(
  const std::vector<std::string>& paths);

MaintenanceLogSummary summarize_maintenance_logs(
  const std::vector<MaintenanceLogCursor>& cursors,
  double observation_period_seconds = 5.0);

MaintenanceLogSummary summarize_maintenance_log_window(
  const std::vector<MaintenanceLogCursor>& begin_cursors,
  const std::vector<MaintenanceLogCursor>& end_cursors,
  double observation_period_seconds = 5.0);

}  // namespace tools::breakdown_benchmark
