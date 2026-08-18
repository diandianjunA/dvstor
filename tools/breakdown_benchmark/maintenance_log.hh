#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gpu_search/maintenance_telemetry.hh"

namespace tools::breakdown_benchmark {

// Keep this in sync with kFinalizeLatencyBucketUpperNs. The explicit 5000 ms
// bucket keeps p99 reporting precise without rounding
// every latency in (4 s, 8 s] up to 8 s.
inline constexpr size_t kMaintenanceLatencyBucketCount = 18;
inline constexpr size_t kStage2TimingPhaseCount =
  gpu_search::maintenance_telemetry::kStage2PhaseCount;

struct MaintenanceObservation {
  uint64_t stage2_enqueued{};
  uint64_t stage2_finalized_live{};
  uint64_t stale{};
  uint64_t remaining{};
  uint64_t peer_reverse_remaining{};
  uint64_t failed{};
  // The storage log/control-page field is historically named
  // peer_reverse_failed, but it counts retryable apply attempts. A failed
  // response is not cached by receiver deduplication and the sender retries
  // the same idempotent request until it receives a successful ACK.
  uint64_t peer_reverse_retry_attempts{};
  uint64_t admission_window{};
  uint64_t completion_outstanding{};
  uint64_t stage2_continuations{};
  uint64_t stage2_remote_frontier_items{};
  uint64_t stage2_remote_expansions{};
  uint64_t stage2_scored_candidates{};
  uint64_t stage2_migrations{};
  uint64_t stage2_final_edges{};
  uint64_t stage2_cross_edges_stage1_home{};
  uint64_t stage2_cross_edges_final_home{};
  uint64_t stage1_search_budget_exhausted{};
  uint64_t stage2_search_budget_exhausted{};
  uint64_t pressure_yields{};
  uint64_t stage2_batches{};
  uint64_t stage2_batched_items{};
  uint64_t stage2_graph_read_waves{};
  uint64_t stage2_graph_unique_reads{};
  uint64_t stage2_graph_prefetch_predictions{};
  uint64_t stage2_graph_prefetch_top1_hits{};
  uint64_t stage2_graph_prefetch_top2_hits{};
  uint64_t stage2_vector_read_waves{};
  uint64_t stage2_vector_unique_reads{};
  uint64_t stage2_home_score_rpc_batches{};
  uint64_t stage2_home_score_rpc_items{};
  uint64_t stage2_home_score_rpc_queries{};
  uint64_t stage2_home_score_rpc_request_bytes{};
  uint64_t stage2_home_score_rpc_response_bytes{};
  std::array<uint64_t, kStage2TimingPhaseCount> stage2_phase_attempts{};
  std::array<uint64_t, kStage2TimingPhaseCount>
    stage2_phase_task_attempts{};
  std::array<uint64_t, kStage2TimingPhaseCount> stage2_phase_elapsed_ns{};
  uint64_t maintenance_worker_idle_waits{};
  uint64_t maintenance_worker_idle_ns{};
  uint64_t physical_stage1_items{};
  uint64_t physical_stage1_total_ns{};
  uint64_t physical_stage1_search_ns{};
  uint64_t physical_stage1_prune_ns{};
  uint64_t physical_stage1_allocate_write_ns{};
  uint64_t physical_stage1_backlink_ns{};
  uint64_t physical_stage1_candidates{};
  uint64_t physical_stage1_remote_frontier_items{};
  uint64_t physical_stage1_neighbors{};
  double p99_stage2_delay_upper_ms{};
  bool p99_stage2_delay_over_30s{};
  std::array<uint64_t, kMaintenanceLatencyBucketCount>
    stage2_delay_histogram{};
  bool failure_counters_available{};
  bool peer_reverse_retry_counter_available{};
  bool stage2_delay_histogram_available{};
  bool completion_window_available{};
  bool locality_counters_available{};
  bool search_budget_counters_available{};
  bool timing_counters_available{};

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
  // Terminal maintenance failures only. Retryable peer reverse attempts are
  // reported separately and must not make a successfully drained run look
  // inconsistent.
  uint64_t failures{};
  uint64_t peer_reverse_retry_attempts{};
  uint64_t admission_window{};
  uint64_t completion_outstanding{};
  uint64_t max_completion_outstanding_per_shard{};
  uint64_t stage2_finalized_live{};
  uint64_t stage2_continuations{};
  uint64_t stage2_remote_frontier_items{};
  uint64_t stage2_remote_expansions{};
  uint64_t stage2_scored_candidates{};
  uint64_t stage2_migrations{};
  uint64_t stage2_final_edges{};
  uint64_t stage2_cross_edges_stage1_home{};
  uint64_t stage2_cross_edges_final_home{};
  uint64_t stage1_search_budget_exhausted{};
  uint64_t stage2_search_budget_exhausted{};
  uint64_t pressure_yields{};
  uint64_t stage2_batches{};
  uint64_t stage2_batched_items{};
  uint64_t stage2_graph_read_waves{};
  uint64_t stage2_graph_unique_reads{};
  uint64_t stage2_graph_prefetch_predictions{};
  uint64_t stage2_graph_prefetch_top1_hits{};
  uint64_t stage2_graph_prefetch_top2_hits{};
  uint64_t stage2_vector_read_waves{};
  uint64_t stage2_vector_unique_reads{};
  uint64_t stage2_home_score_rpc_batches{};
  uint64_t stage2_home_score_rpc_items{};
  uint64_t stage2_home_score_rpc_queries{};
  uint64_t stage2_home_score_rpc_request_bytes{};
  uint64_t stage2_home_score_rpc_response_bytes{};
  std::array<uint64_t, kStage2TimingPhaseCount> stage2_phase_attempts{};
  std::array<uint64_t, kStage2TimingPhaseCount>
    stage2_phase_task_attempts{};
  std::array<uint64_t, kStage2TimingPhaseCount> stage2_phase_elapsed_ns{};
  uint64_t maintenance_worker_idle_waits{};
  uint64_t maintenance_worker_idle_ns{};
  uint64_t physical_stage1_items{};
  uint64_t physical_stage1_total_ns{};
  uint64_t physical_stage1_search_ns{};
  uint64_t physical_stage1_prune_ns{};
  uint64_t physical_stage1_allocate_write_ns{};
  uint64_t physical_stage1_backlink_ns{};
  uint64_t physical_stage1_candidates{};
  uint64_t physical_stage1_remote_frontier_items{};
  uint64_t physical_stage1_neighbors{};
  double p99_stage2_delay_upper_ms{};
  bool p99_stage2_delay_over_30s{};
  uint64_t p99_stage2_delay_samples{};
  bool p99_stage2_delay_available{};
  size_t logs_with_failure_deltas{};
  size_t logs_with_peer_reverse_retry_deltas{};
  size_t logs_with_histogram_deltas{};
  size_t logs_with_completion_window{};
  size_t logs_with_locality_deltas{};
  size_t logs_with_search_budget_deltas{};
  size_t logs_with_execution_counter_deltas{};
  size_t logs_with_score_rpc_wire_counter_deltas{};
  size_t logs_with_timing_counter_deltas{};
  bool failure_delta_available{};
  bool peer_reverse_retry_delta_available{};
  bool completion_window_available{};
  bool locality_delta_available{};
  bool search_budget_delta_available{};
  bool execution_counter_delta_available{};
  bool score_rpc_wire_counter_delta_available{};
  bool timing_counter_delta_available{};
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

MaintenanceLogSummary summarize_maintenance_snapshot_window(
  const std::vector<std::optional<
    gpu_search::maintenance_telemetry::Snapshot>>& begin,
  const std::vector<std::optional<
    gpu_search::maintenance_telemetry::Snapshot>>& end);

}  // namespace tools::breakdown_benchmark
