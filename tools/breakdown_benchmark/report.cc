#include "tools/breakdown_benchmark/report.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace tools::breakdown_benchmark {

std::string normalize_path(const std::string& path) {
  if (path.empty()) return {};
  std::error_code error;
  const auto absolute = std::filesystem::absolute(path, error);
  if (error) {
    throw std::runtime_error(
      "failed to make path absolute: " + path);
  }
  const auto canonical = std::filesystem::weakly_canonical(absolute, error);
  if (error) return absolute.lexically_normal().string();
  return canonical.string();
}

std::vector<uint32_t> filter_base_only_recall_ids(
    const std::vector<node_t>& results,
    uint32_t base_id_limit,
    size_t result_limit) {
  std::vector<uint32_t> filtered;
  filtered.reserve(std::min(results.size(), result_limit));
  for (const node_t id : results) {
    if (id >= base_id_limit) continue;
    filtered.push_back(static_cast<uint32_t>(id));
    if (filtered.size() == result_limit) break;
  }
  return filtered;
}

nlohmann::json telemetry_to_json(
    const gpu_search::TelemetrySnapshot& telemetry) {
  const uint64_t explicit_phase_ns =
    telemetry.gpu_prepare_ns + telemetry.gpu_beam_selection_ns +
    telemetry.gpu_frontier_preview_ns + telemetry.gpu_rdma_issue_ns +
    telemetry.gpu_rdma_wait_ns +
    telemetry.gpu_graph_validation_ns + telemetry.gpu_neighbor_decode_ns +
    telemetry.gpu_pq_score_ns + telemetry.gpu_visited_ns +
    telemetry.gpu_beam_merge_ns + telemetry.gpu_exact_ns;
  const uint64_t gpu_other_ns = telemetry.gpu_active_ns > explicit_phase_ns
    ? telemetry.gpu_active_ns - explicit_phase_ns : 0;
  const uint64_t terminal_exact_cache_unpromoted_records =
    telemetry.terminal_exact_cache_issued_records >
        telemetry.terminal_exact_cache_promoted_records
      ? telemetry.terminal_exact_cache_issued_records -
          telemetry.terminal_exact_cache_promoted_records
      : 0;
  const uint64_t dynamic_graph_nonfallback_full_attempts =
    telemetry.dynamic_graph_full_reads >
        telemetry.dynamic_graph_fallback_reads
      ? telemetry.dynamic_graph_full_reads -
          telemetry.dynamic_graph_fallback_reads
      : 0;
  const uint64_t dynamic_graph_snapshot_attempts =
    telemetry.dynamic_graph_short_reads + telemetry.dynamic_graph_full_reads;
  nlohmann::json expanded_degree_histogram = nlohmann::json::array();
  nlohmann::json dynamic_expanded_degree_histogram = nlohmann::json::array();
  for (const auto count : telemetry.expanded_degree_histogram) {
    expanded_degree_histogram.push_back(count);
  }
  for (const auto count : telemetry.dynamic_expanded_degree_histogram) {
    dynamic_expanded_degree_histogram.push_back(count);
  }
  return {
    {"gpu_memory_explicit_bytes", telemetry.gpu_memory_explicit_bytes},
    {"gpu_memory_base_pq_bytes", telemetry.gpu_memory_base_pq_bytes},
    {"gpu_memory_route_graph_bytes", telemetry.gpu_memory_route_graph_bytes},
    {"queries_submitted", telemetry.queries_submitted},
    {"queries_completed", telemetry.queries_completed},
    {"batches", telemetry.batches},
    {"batch_queries", telemetry.batch_queries},
    {"average_batch_size", telemetry.batches == 0 ? 0.0
      : static_cast<double>(telemetry.batch_queries) /
          static_cast<double>(telemetry.batches)},
    {"submission_wait_ns", telemetry.submission_wait_ns},
    {"average_submission_wait_us", telemetry.queries_submitted == 0 ? 0.0
      : static_cast<double>(telemetry.submission_wait_ns) /
          static_cast<double>(telemetry.queries_submitted) / 1000.0},
    {"completion_wait_ns", telemetry.completion_wait_ns},
    {"gpu_query_residence_ns", telemetry.gpu_active_ns},
    {"gpu_other_ns", gpu_other_ns},
    {"average_gpu_query_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_active_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_prepare_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_prepare_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_graph_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_graph_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_score_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_score_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_beam_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_beam_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_exact_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_exact_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_beam_selection_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_beam_selection_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_rdma_issue_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_rdma_issue_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_frontier_preview_us",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_frontier_preview_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_frontier_prepare_us",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_frontier_prepare_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_frontier_enqueue_us",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_frontier_enqueue_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_rdma_wait_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_rdma_wait_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_graph_validation_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_graph_validation_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_neighbor_decode_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_neighbor_decode_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_pq_score_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_pq_score_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_visited_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_visited_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_beam_merge_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_beam_merge_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"gpu_beam_merge_prepare_ns", telemetry.gpu_beam_merge_prepare_ns},
    {"gpu_beam_merge_sort_ns", telemetry.gpu_beam_merge_sort_ns},
    {"gpu_beam_merge_materialize_ns",
      telemetry.gpu_beam_merge_materialize_ns},
    {"average_gpu_beam_merge_prepare_us",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_beam_merge_prepare_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_beam_merge_sort_us",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_beam_merge_sort_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_beam_merge_materialize_us",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.gpu_beam_merge_materialize_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"average_gpu_other_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(gpu_other_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"rdma_read_ops", telemetry.rdma_read_ops},
    {"rdma_read_bytes", telemetry.rdma_read_bytes},
    {"rdma_merged_requests", telemetry.rdma_merged_requests},
    {"owner_submitted_wqes", telemetry.owner_submitted_wqes},
    {"owner_submission_wqe_capacity",
      telemetry.owner_submission_wqe_capacity},
    {"owner_wqe_submission_utilization",
      telemetry.owner_submission_wqe_capacity == 0 ? 0.0
      : static_cast<double>(telemetry.owner_submitted_wqes) /
          static_cast<double>(telemetry.owner_submission_wqe_capacity)},
    {"owner_critical_batches", telemetry.owner_critical_batches},
    {"owner_speculative_batches", telemetry.owner_speculative_batches},
    {"direct_path_failures", telemetry.direct_path_failures},
    {"graph_page_requests", telemetry.graph_page_requests},
    {"logical_graph_reads",
      telemetry.critical_graph_reads + telemetry.speculative_graph_reads},
    {"average_logical_graph_reads_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(
          telemetry.critical_graph_reads + telemetry.speculative_graph_reads) /
          static_cast<double>(telemetry.queries_completed)},
    {"graph_shard_batches", telemetry.graph_shard_batches},
    {"average_graph_shard_batches_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.graph_shard_batches) /
          static_cast<double>(telemetry.queries_completed)},
    {"graph_read_retries", telemetry.graph_read_retries},
    {"average_graph_read_retries", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.graph_read_retries) /
          static_cast<double>(telemetry.queries_completed)},
    {"graph_read_bytes", telemetry.graph_read_bytes},
    {"average_graph_read_bytes_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.graph_read_bytes) /
          static_cast<double>(telemetry.queries_completed)},
    {"average_graph_read_bytes_per_logical_parent",
      telemetry.graph_page_requests == 0 ? 0.0
      : static_cast<double>(telemetry.graph_read_bytes) /
          static_cast<double>(telemetry.graph_page_requests)},
    {"graph_live_extent_reads", telemetry.graph_live_extent_reads},
    {"graph_full_record_reads", telemetry.graph_full_record_reads},
    {"graph_extent_fallback_reads", telemetry.graph_extent_fallback_reads},
    {"graph_extent_underhint_reads",
      telemetry.graph_extent_underhint_reads},
    {"graph_extent_hint_promotions",
      telemetry.graph_extent_hint_promotions},
    {"expanded_parent_count", telemetry.expanded_parent_count},
    {"expanded_neighbor_count_sum",
      telemetry.expanded_neighbor_count_sum},
    {"average_expanded_parent_degree",
      telemetry.expanded_parent_count == 0 ? 0.0
      : static_cast<double>(telemetry.expanded_neighbor_count_sum) /
          static_cast<double>(telemetry.expanded_parent_count)},
    {"expanded_degree_histogram_quantum", 8},
    {"expanded_degree_histogram", expanded_degree_histogram},
    {"dynamic_expanded_parent_count",
      telemetry.dynamic_expanded_parent_count},
    {"dynamic_expanded_neighbor_count_sum",
      telemetry.dynamic_expanded_neighbor_count_sum},
    {"dynamic_expanded_degree_histogram_quantum", 8},
    {"dynamic_expanded_degree_histogram",
      dynamic_expanded_degree_histogram},
    {"average_dynamic_expanded_parent_degree",
      telemetry.dynamic_expanded_parent_count == 0 ? 0.0
      : static_cast<double>(telemetry.dynamic_expanded_neighbor_count_sum) /
          static_cast<double>(telemetry.dynamic_expanded_parent_count)},
    {"dynamic_expanded_parent_ratio",
      telemetry.expanded_parent_count == 0 ? 0.0
      : static_cast<double>(telemetry.dynamic_expanded_parent_count) /
          static_cast<double>(telemetry.expanded_parent_count)},
    {"dynamic_graph_short_reads", telemetry.dynamic_graph_short_reads},
    {"dynamic_graph_full_reads", telemetry.dynamic_graph_full_reads},
    {"dynamic_graph_read_bytes", telemetry.dynamic_graph_read_bytes},
    {"dynamic_graph_fallback_reads",
      telemetry.dynamic_graph_fallback_reads},
    {"dynamic_graph_hint_promotions",
      telemetry.dynamic_graph_hint_promotions},
    {"dynamic_graph_hint_demotions",
      telemetry.dynamic_graph_hint_demotions},
    {"dynamic_graph_snapshot_attempts", dynamic_graph_snapshot_attempts},
    {"dynamic_graph_nonfallback_full_attempts",
      dynamic_graph_nonfallback_full_attempts},
    {"dynamic_graph_short_physical_ratio",
      dynamic_graph_snapshot_attempts == 0 ? 0.0
      : static_cast<double>(telemetry.dynamic_graph_short_reads) /
          static_cast<double>(dynamic_graph_snapshot_attempts)},
    {"dynamic_graph_fallback_ratio",
      telemetry.dynamic_graph_short_reads == 0 ? 0.0
      : static_cast<double>(telemetry.dynamic_graph_fallback_reads) /
          static_cast<double>(telemetry.dynamic_graph_short_reads)},
    {"average_dynamic_graph_read_bytes_per_physical_read",
      dynamic_graph_snapshot_attempts == 0 ? 0.0
      : static_cast<double>(telemetry.dynamic_graph_read_bytes) /
          static_cast<double>(dynamic_graph_snapshot_attempts)},
    {"average_dynamic_graph_read_bytes_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.dynamic_graph_read_bytes) /
          static_cast<double>(telemetry.queries_completed)},
    {"graph_live_extent_read_ratio",
      telemetry.graph_live_extent_reads + telemetry.graph_full_record_reads == 0
        ? 0.0
        : static_cast<double>(telemetry.graph_live_extent_reads) /
            static_cast<double>(
              telemetry.graph_live_extent_reads +
              telemetry.graph_full_record_reads)},
    {"graph_extent_fallback_ratio",
      telemetry.graph_live_extent_reads == 0 ? 0.0
      : static_cast<double>(telemetry.graph_extent_fallback_reads) /
          static_cast<double>(telemetry.graph_live_extent_reads)},
    {"logical_expansions", telemetry.logical_expansions},
    {"average_logical_expansions_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.logical_expansions) /
          static_cast<double>(telemetry.queries_completed)},
    {"critical_graph_reads", telemetry.critical_graph_reads},
    {"critical_graph_bytes", telemetry.critical_graph_bytes},
    {"average_critical_graph_reads_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.critical_graph_reads) /
          static_cast<double>(telemetry.queries_completed)},
    {"average_critical_graph_bytes_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.critical_graph_bytes) /
          static_cast<double>(telemetry.queries_completed)},
    {"speculative_graph_reads", telemetry.speculative_graph_reads},
    {"speculative_graph_bytes", telemetry.speculative_graph_bytes},
    {"average_speculative_graph_reads_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.speculative_graph_reads) /
          static_cast<double>(telemetry.queries_completed)},
    {"average_speculative_graph_bytes_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.speculative_graph_bytes) /
          static_cast<double>(telemetry.queries_completed)},
    {"speculative_wasted_bytes", telemetry.speculative_wasted_bytes},
    {"average_speculative_wasted_bytes_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.speculative_wasted_bytes) /
          static_cast<double>(telemetry.queries_completed)},
    {"speculative_arrived", telemetry.speculative_arrived},
    {"speculative_promoted", telemetry.speculative_promoted},
    {"speculative_stale", telemetry.speculative_stale},
    {"speculative_queue_rejects", telemetry.speculative_queue_rejects},
    {"core_prefetch_reads", telemetry.core_prefetch_reads},
    {"core_prefetch_bytes", telemetry.core_prefetch_bytes},
    {"core_prefetch_arrived", telemetry.core_prefetch_arrived},
    {"core_prefetch_promoted", telemetry.core_prefetch_promoted},
    {"core_prefetch_stale", telemetry.core_prefetch_stale},
    {"core_prefetch_queue_rejects",
      telemetry.core_prefetch_queue_rejects},
    {"core_prefetch_waves", telemetry.core_prefetch_waves},
    {"core_ready_waves", telemetry.core_ready_waves},
    {"core_ready_wave_ratio",
      telemetry.core_prefetch_waves == 0 ? 0.0
      : static_cast<double>(telemetry.core_ready_waves) /
          static_cast<double>(telemetry.core_prefetch_waves)},
    {"terminal_exact_cache_attempted_queries",
      telemetry.terminal_exact_cache_attempted_queries},
    {"terminal_exact_cache_attempt_rate",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(
          telemetry.terminal_exact_cache_attempted_queries) /
          static_cast<double>(telemetry.queries_completed)},
    {"terminal_exact_cache_issued_records",
      telemetry.terminal_exact_cache_issued_records},
    {"average_terminal_exact_cache_issued_records_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.terminal_exact_cache_issued_records) /
          static_cast<double>(telemetry.queries_completed)},
    {"terminal_exact_cache_promoted_records",
      telemetry.terminal_exact_cache_promoted_records},
    {"average_terminal_exact_cache_promoted_records_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(
          telemetry.terminal_exact_cache_promoted_records) /
          static_cast<double>(telemetry.queries_completed)},
    {"terminal_exact_cache_wasted_bytes",
      telemetry.terminal_exact_cache_wasted_bytes},
    {"average_terminal_exact_cache_wasted_bytes_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.terminal_exact_cache_wasted_bytes) /
          static_cast<double>(telemetry.queries_completed)},
    {"terminal_exact_cache_queue_rejects",
      telemetry.terminal_exact_cache_queue_rejects},
    {"average_terminal_exact_cache_queue_rejects_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(
          telemetry.terminal_exact_cache_queue_rejects) /
          static_cast<double>(telemetry.queries_completed)},
    {"terminal_exact_cache_miss_records",
      telemetry.terminal_exact_cache_miss_records},
    {"average_terminal_exact_cache_miss_records_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.terminal_exact_cache_miss_records) /
          static_cast<double>(telemetry.queries_completed)},
    {"terminal_exact_cache_promotion_ratio",
      telemetry.terminal_exact_cache_issued_records == 0 ? 0.0
      : static_cast<double>(
          telemetry.terminal_exact_cache_promoted_records) /
          static_cast<double>(telemetry.terminal_exact_cache_issued_records)},
    {"terminal_exact_cache_waste_ratio",
      telemetry.terminal_exact_cache_issued_records == 0 ? 0.0
      : static_cast<double>(terminal_exact_cache_unpromoted_records) /
          static_cast<double>(telemetry.terminal_exact_cache_issued_records)},
    {"completion_score_batches", telemetry.completion_score_batches},
    {"average_completion_score_batches_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.completion_score_batches) /
          static_cast<double>(telemetry.queries_completed)},
    {"completion_score_candidates", telemetry.completion_score_candidates},
    {"average_completion_score_candidates_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.completion_score_candidates) /
          static_cast<double>(telemetry.queries_completed)},
    {"average_completion_score_candidates_per_batch",
      telemetry.completion_score_batches == 0 ? 0.0
      : static_cast<double>(telemetry.completion_score_candidates) /
          static_cast<double>(telemetry.completion_score_batches)},
    {"frontier_reusable_certificates",
      telemetry.frontier_reusable_certificates},
    {"average_frontier_reusable_certificates_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.frontier_reusable_certificates) /
          static_cast<double>(telemetry.queries_completed)},
    {"frontier_streamed_candidate_runs",
      telemetry.frontier_streamed_candidate_runs},
    {"average_frontier_streamed_candidate_runs_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.frontier_streamed_candidate_runs) /
          static_cast<double>(telemetry.queries_completed)},
    {"ordered_score_batches", telemetry.ordered_score_batches},
    {"average_ordered_score_batches_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.ordered_score_batches) /
          static_cast<double>(telemetry.queries_completed)},
    {"ordered_score_candidates", telemetry.ordered_score_candidates},
    {"average_ordered_score_candidates_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.ordered_score_candidates) /
          static_cast<double>(telemetry.queries_completed)},
    {"average_ordered_score_candidates_per_batch",
      telemetry.ordered_score_batches == 0 ? 0.0
      : static_cast<double>(telemetry.ordered_score_candidates) /
          static_cast<double>(telemetry.ordered_score_batches)},
    {"ooo_bypassed_parents", telemetry.ooo_bypassed_parents},
    {"average_ooo_bypassed_parents_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.ooo_bypassed_parents) /
          static_cast<double>(telemetry.queries_completed)},
    {"frontier_reusable_prefix_ranks",
      telemetry.frontier_reusable_prefix_ranks},
    {"average_frontier_reusable_prefix_ranks_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.frontier_reusable_prefix_ranks) /
          static_cast<double>(telemetry.queries_completed)},
    {"frontier_reusable_full_prefix_certificates",
      telemetry.frontier_reusable_full_prefix_certificates},
    {"average_frontier_reusable_full_prefix_certificates_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(
          telemetry.frontier_reusable_full_prefix_certificates) /
          static_cast<double>(telemetry.queries_completed)},
    {"frontier_reusable_issued_certificates",
      telemetry.frontier_reusable_issued_certificates},
    {"average_frontier_reusable_issued_certificates_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(
          telemetry.frontier_reusable_issued_certificates) /
          static_cast<double>(telemetry.queries_completed)},
    {"frontier_certificate_rejects",
      telemetry.frontier_certificate_rejects},
    {"average_frontier_certificate_rejects_per_query",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.frontier_certificate_rejects) /
          static_cast<double>(telemetry.queries_completed)},
    {"speculative_promotion_ratio",
      telemetry.speculative_graph_reads == 0 ? 0.0
      : static_cast<double>(telemetry.speculative_promoted) /
          static_cast<double>(telemetry.speculative_graph_reads)},
    {"speculative_waste_ratio",
      telemetry.speculative_graph_reads == 0 ? 0.0
      : static_cast<double>(telemetry.speculative_stale) /
          static_cast<double>(telemetry.speculative_graph_reads)},
    {"speculative_wasted_byte_ratio",
      telemetry.speculative_graph_bytes == 0 ? 0.0
      : static_cast<double>(telemetry.speculative_wasted_bytes) /
          static_cast<double>(telemetry.speculative_graph_bytes)},
    {"tail_promotion_ratio",
      telemetry.speculative_graph_reads == 0 ? 0.0
      : static_cast<double>(telemetry.speculative_promoted) /
          static_cast<double>(telemetry.speculative_graph_reads)},
    {"rdma_completion_latency_ns", telemetry.rdma_completion_latency_ns},
    {"speculative_completion_latency_ns",
      telemetry.speculative_completion_latency_ns},
    {"rdma_completion_groups", telemetry.rdma_completion_groups},
    {"speculative_completion_groups",
      telemetry.speculative_completion_groups},
    {"average_rdma_completion_latency_us",
      telemetry.rdma_completion_groups == 0 ? 0.0
      : static_cast<double>(telemetry.rdma_completion_latency_ns) /
          static_cast<double>(telemetry.rdma_completion_groups) / 1000.0},
    {"average_speculative_completion_latency_us",
      telemetry.speculative_completion_groups == 0 ? 0.0
      : static_cast<double>(telemetry.speculative_completion_latency_ns) /
          static_cast<double>(telemetry.speculative_completion_groups) /
          1000.0},
    {"issue_epochs", telemetry.issue_epochs},
    {"commit_epochs", telemetry.commit_epochs},
    {"issue_width_sum", telemetry.issue_width_sum},
    {"issue_width_capacity_sum", telemetry.issue_width_capacity_sum},
    {"commit_width_sum", telemetry.commit_width_sum},
    {"average_issue_width", telemetry.issue_epochs == 0 ? 0.0
      : static_cast<double>(telemetry.issue_width_sum) /
          static_cast<double>(telemetry.issue_epochs)},
    {"average_issue_width_capacity", telemetry.issue_epochs == 0 ? 0.0
      : static_cast<double>(telemetry.issue_width_capacity_sum) /
          static_cast<double>(telemetry.issue_epochs)},
    {"issue_frontier_utilization",
      telemetry.issue_width_capacity_sum == 0 ? 0.0
      : static_cast<double>(telemetry.issue_width_sum) /
          static_cast<double>(telemetry.issue_width_capacity_sum)},
    {"average_commit_width", telemetry.commit_epochs == 0 ? 0.0
      : static_cast<double>(telemetry.commit_width_sum) /
          static_cast<double>(telemetry.commit_epochs)},
    {"max_issue_width", telemetry.max_issue_width},
    {"max_commit_width", telemetry.max_commit_width},
    {"critical_rob_hits", telemetry.critical_rob_hits},
    {"critical_misses", telemetry.critical_misses},
    {"critical_rob_hit_ratio",
      telemetry.critical_rob_hits + telemetry.critical_misses == 0 ? 0.0
      : static_cast<double>(telemetry.critical_rob_hits) /
          static_cast<double>(
            telemetry.critical_rob_hits + telemetry.critical_misses)},
    {"speculative_wait_ns", telemetry.speculative_wait_ns},
    {"average_speculative_wait_us",
      telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.speculative_wait_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
    {"gpu_kernel_threads", telemetry.gpu_kernel_threads},
    {"gpu_registers_per_thread", telemetry.gpu_registers_per_thread},
    {"gpu_static_shared_bytes", telemetry.gpu_static_shared_bytes},
    {"gpu_active_blocks_per_sm", telemetry.gpu_active_blocks_per_sm},
    {"gpu_effective_blocks_per_sm", telemetry.gpu_effective_blocks_per_sm},
    {"gpu_query_blocks", telemetry.gpu_query_blocks},
    {"gpu_owner_blocks", telemetry.gpu_owner_blocks},
    {"gpu_total_persistent_blocks", telemetry.gpu_total_persistent_blocks},
    {"graph_dependency_rounds", telemetry.graph_dependency_rounds},
    {"average_graph_rounds_per_query", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.graph_dependency_rounds) /
          static_cast<double>(telemetry.queries_completed)},
    {"average_parent_batch_size", telemetry.graph_dependency_rounds == 0 ? 0.0
      : static_cast<double>(telemetry.graph_page_requests) /
          static_cast<double>(telemetry.graph_dependency_rounds)},
    {"graph_route_hits", telemetry.graph_route_hits},
    {"graph_route_refreshes", telemetry.graph_route_refreshes},
    {"centroid_route_publications", telemetry.centroid_route_publications},
    {"centroid_route_shard_updates", telemetry.centroid_route_shard_updates},
    {"centroid_route_live_entries", telemetry.centroid_route_live_entries},
    {"centroid_route_snapshot_skips", telemetry.centroid_route_snapshot_skips},
    {"centroid_route_probe_reads", telemetry.centroid_route_probe_reads},
    {"centroid_route_body_reads", telemetry.centroid_route_body_reads},
    {"centroid_route_unchanged_polls",
      telemetry.centroid_route_unchanged_polls},
    {"centroid_route_poll_delay_us",
      telemetry.centroid_route_poll_delay_us},
    {"centroid_route_query_retries",
      telemetry.centroid_route_query_retries},
    {"centroid_route_query_timeouts",
      telemetry.centroid_route_query_timeouts},
    {"graph_route_hit_ratio",
      telemetry.graph_page_requests + telemetry.graph_route_hits == 0 ? 0.0
      : static_cast<double>(telemetry.graph_route_hits) /
          static_cast<double>(
            telemetry.graph_page_requests + telemetry.graph_route_hits)},
    {"exact_vector_reads", telemetry.exact_vector_reads},
    {"exact_snapshot_train_batches",
      telemetry.exact_snapshot_train_batches},
    {"exact_snapshot_train_fallbacks",
      telemetry.exact_snapshot_train_fallbacks},
    {"exact_snapshot_train_success_ratio",
      telemetry.exact_snapshot_train_batches == 0 ? 0.0
      : static_cast<double>(
          telemetry.exact_snapshot_train_batches >
              telemetry.exact_snapshot_train_fallbacks
            ? telemetry.exact_snapshot_train_batches -
                telemetry.exact_snapshot_train_fallbacks
            : 0) /
          static_cast<double>(telemetry.exact_snapshot_train_batches)},
    {"dynamic_code_candidates", telemetry.dynamic_code_candidates},
    {"dynamic_code_reads", telemetry.dynamic_code_reads},
    {"dynamic_code_read_bytes", telemetry.dynamic_code_read_bytes},
    {"dynamic_code_incarnation_rejects",
      telemetry.dynamic_code_incarnation_rejects},
    {"dynamic_code_wait_ns", telemetry.dynamic_code_wait_ns},
    {"dynamic_code_cache_hits", telemetry.dynamic_code_cache_hits},
    {"dynamic_code_arena_hits", telemetry.dynamic_code_cache_hits},
    {"dynamic_code_batch_deduplicated",
      telemetry.dynamic_code_batch_deduplicated},
    {"dynamic_code_cache_publish_successes",
      telemetry.dynamic_code_cache_publish_successes},
    {"dynamic_code_arena_publish_successes",
      telemetry.dynamic_code_cache_publish_successes},
    {"dynamic_code_cache_publish_races",
      telemetry.dynamic_code_cache_publish_races},
    {"dynamic_code_arena_publish_races",
      telemetry.dynamic_code_cache_publish_races},
    {"dynamic_code_cache_lookup_probe_exhaustions",
      telemetry.dynamic_code_cache_lookup_probe_exhaustions},
    {"dynamic_code_cache_publish_probe_exhaustions",
      telemetry.dynamic_code_cache_publish_probe_exhaustions},
    {"dynamic_code_cache_lookup_probes",
      telemetry.dynamic_code_cache_lookup_probes},
    {"dynamic_code_cache_max_lookup_probes",
      telemetry.dynamic_code_cache_max_lookup_probes},
    {"dynamic_code_cache_occupied", telemetry.dynamic_code_cache_occupied},
    {"dynamic_code_cache_capacity", telemetry.dynamic_code_cache_capacity},
    {"dynamic_code_arena_capacity", telemetry.dynamic_code_cache_capacity},
    {"dynamic_code_cache_hit_ratio", telemetry.dynamic_code_candidates == 0
      ? 0.0 : static_cast<double>(telemetry.dynamic_code_cache_hits) /
          static_cast<double>(telemetry.dynamic_code_candidates)},
    {"dynamic_code_authoritative_avoidance_ratio",
      telemetry.dynamic_code_candidates == 0 ? 0.0
      : static_cast<double>(telemetry.dynamic_code_candidates -
          std::min(telemetry.dynamic_code_candidates,
                   telemetry.dynamic_code_reads)) /
          static_cast<double>(telemetry.dynamic_code_candidates)},
    {"dynamic_code_cache_load_factor", telemetry.dynamic_code_cache_capacity == 0
      ? 0.0 : static_cast<double>(telemetry.dynamic_code_cache_occupied) /
          static_cast<double>(telemetry.dynamic_code_cache_capacity)},
    {"average_dynamic_code_cache_lookup_probes",
      telemetry.dynamic_code_candidates == 0 ? 0.0
      : static_cast<double>(telemetry.dynamic_code_cache_lookup_probes) /
          static_cast<double>(telemetry.dynamic_code_candidates)},
    {"average_dynamic_code_reads", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.dynamic_code_reads) /
          static_cast<double>(telemetry.queries_completed)},
    {"average_dynamic_code_wait_us", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.dynamic_code_wait_ns) /
          static_cast<double>(telemetry.queries_completed) / 1000.0},
  };
}

FormattedReport format_report(const nlohmann::json& root,
                              const service::breakdown::Report& report) {
  nlohmann::json summaries = nlohmann::json::object();
  std::ostringstream output;
  const auto system_variant = root["meta"].value(
    "system_variant", nlohmann::json::object());
  const auto resolved_modes = system_variant.value(
    "resolved_modes", nlohmann::json::object());
  const auto variant_index = system_variant.value(
    "index", nlohmann::json::object());
  output << "system_variant\n";
  output << "  profile_name: "
         << system_variant.value("profile_name", "unspecified") << '\n';
  output << "  label: "
         << system_variant.value("label", "unspecified") << '\n';
  output << "  update_mutation_api: "
         << system_variant.value("update_mutation_api", "unspecified")
         << '\n';
  output << "  storage_owner_update_completion_mode: "
         << resolved_modes.value(
              "storage_owner_update_completion_mode", "unspecified")
         << '\n';
  output << "  gpu_dynamic_graph_access_mode: "
         << resolved_modes.value("gpu_dynamic_graph_access_mode", "unspecified")
         << '\n';
  output << "  gpu_rdma_search_progression_mode: "
         << resolved_modes.value(
              "gpu_rdma_search_progression_mode", "unspecified")
         << '\n';
  output << "  index_prefix: "
         << variant_index.value("prefix", "") << '\n';
  output << "  index_schema_version: "
         << variant_index.value("schema_version", 0U) << '\n';
  output << "  index_build_fingerprint: "
         << variant_index.value("build_fingerprint", 0ULL) << '\n';

  const auto& recall_query_meta = root["meta"]["recall_query"];
  const auto& performance_query_meta = root["meta"]["performance_query"];
  output << "query_inputs\n";
  output << "  recall_source: " << recall_query_meta.value("source", "") << '\n';
  output << "  recall_rows: " << recall_query_meta.value("rows", 0ULL) << '\n';
  output << "  performance_source: "
         << performance_query_meta.value("source", "") << '\n';
  output << "  performance_rows: "
         << performance_query_meta.value("rows", 0ULL) << '\n';
  output << "  performance_row_reuse_policy: "
         << performance_query_meta.value("row_reuse_policy", "") << '\n';
  output << "  performance_warmup/measure/total_rows_consumed: "
         << performance_query_meta.value("warmup_rows_consumed", 0ULL) << "/"
         << performance_query_meta.value("measure_rows_consumed", 0ULL) << "/"
         << performance_query_meta.value("total_rows_consumed", 0ULL) << '\n';

  const auto& throughput = root["throughput"];
  const double throughput_duration = throughput.value("duration_seconds", 0.0);
  if (throughput_duration > 0.0) {
    const auto query_ops = throughput.value("query_ops", 0ULL);
    const auto write_ops = throughput.value("write_ops", 0ULL);
    output << "throughput\n";
    output << "  duration_seconds: " << throughput_duration << '\n';
    output << "  total_ops_per_sec: " << throughput.value("total_ops_per_sec", 0.0)
           << " (ops=" << (query_ops + write_ops) << ")\n";
    output << "  query_ops_per_sec: " << throughput.value("query_ops_per_sec", 0.0)
           << " (ops=" << query_ops << ")\n";
    output << "  effective_query_ops_per_sec: "
           << throughput.value("effective_query_ops_per_sec", 0.0) << '\n';
    output << "  write_ops_per_sec: " << throughput.value("write_ops_per_sec", 0.0)
           << " (ops=" << write_ops << ")\n";
    output << "  effective_write_ops_per_sec: "
           << throughput.value("effective_write_ops_per_sec", 0.0) << '\n';
    output << "  insert_ops_per_sec: " << throughput.value("insert_ops_per_sec", 0.0)
           << " (ops=" << throughput.value("insert_ops", 0ULL) << ")\n";
    output << "  client_drain_seconds: "
           << throughput.value("client_drain_seconds", 0.0) << '\n';
    output << "  scheduled_query/write_ops: "
           << throughput.value("scheduled_query_ops", 0ULL) << "/"
           << throughput.value("scheduled_write_ops", 0ULL) << '\n';
    output << "  query/write_rate_attainment_ratio: "
           << throughput.value("query_rate_attainment_ratio", 1.0) << "/"
           << throughput.value("write_rate_attainment_ratio", 1.0) << '\n';
    output << "  nominal/effective_rate_basis: "
           << throughput.value("nominal_rate_basis", "") << "/"
           << throughput.value("effective_rate_basis", "") << '\n';
    if (root["meta"].value("workload", "") == "mixed") {
      output << "  write_mix_completed: insert=" << throughput.value("insert_ops", 0ULL)
             << " upsert=" << throughput.value("upsert_ops", 0ULL)
             << " delete=" << throughput.value("delete_ops", 0ULL) << '\n';
    }
    const auto& stability = root["stability"];
    output << "  query_head/tail_qps: "
           << stability.value("query_head_ops_per_sec", 0.0) << "/"
           << stability.value("query_tail_ops_per_sec", 0.0) << '\n';
    output << "  query_tail_to_head_ratio: "
           << stability.value("query_tail_to_head_ratio", 0.0) << '\n';
    output << "  query_min_5s_qps: "
           << stability.value("query_min_window_ops_per_sec", 0.0) << '\n';
    output << "  write_head/tail_qps: "
           << stability.value("write_head_ops_per_sec", 0.0) << "/"
           << stability.value("write_tail_ops_per_sec", 0.0) << '\n';
    output << "  write_tail_to_head_ratio: "
           << stability.value("write_tail_to_head_ratio", 0.0) << '\n';
    output << "  write_min_5s_qps: "
           << stability.value("write_min_window_ops_per_sec", 0.0) << '\n';
    output << "  zero_completion_windows: "
           << stability.value("zero_completion_windows", 0ULL) << '\n';
    output << "  zero_query/write_windows: "
           << stability.value("zero_query_windows", 0ULL) << "/"
           << stability.value("zero_write_windows", 0ULL) << '\n';
  }

  if (root.contains("stage2") &&
      root["stage2"].value("requested_logs", 0ULL) != 0) {
    const auto& stage2 = root["stage2"];
    output << "stage2\n";
    output << "  source: "
           << stage2.value("source", "storage_logs") << '\n';
    output << "  shards observed/requested: "
           << stage2.value("logs_with_observations", 0ULL) << "/"
           << stage2.value("requested_logs", 0ULL) << '\n';
    output << "  p99_stage2_delay_upper_ms: ";
    if (stage2.value("p99_stage2_delay_available", false)) {
      output << stage2.value("p99_stage2_delay_upper_ms", 0.0)
             << " (samples="
             << stage2.value("p99_stage2_delay_samples", 0ULL) << ")\n";
    } else {
      output << "unavailable\n";
    }
    output << "  remaining/max_backlog: "
           << stage2.value("remaining", 0ULL) << "/"
           << stage2.value("max_backlog_observed", 0ULL) << '\n';
    output << "  completion_outstanding/admission_window: ";
    if (stage2.value("completion_window_available", false)) {
      output << stage2.value("completion_outstanding", 0ULL) << "/"
             << stage2.value("admission_window", 0ULL)
             << " (max_per_shard="
             << stage2.value(
                  "max_completion_outstanding_per_shard", 0ULL)
             << ")\n";
    } else {
      output << "unavailable\n";
    }
    output << "  completion_incomplete/admission_window: ";
    if (stage2.value("exact_completion_credit_available", false)) {
      output << stage2.value("completion_incomplete", 0ULL) << "/"
             << stage2.value("admission_window", 0ULL)
             << " (max_per_shard="
             << stage2.value(
                  "max_completion_incomplete_per_shard", 0ULL)
             << ", completed_behind_hole="
             << stage2.value("completed_behind_hole", 0ULL)
             << ")\n";
    } else {
      output << "unavailable\n";
    }
    output << "  completion admission stalls logical/physical: ";
    if (stage2.value(
          "completion_admission_failure_delta_available", false)) {
      output << stage2.value("completion_logical_full_failures", 0ULL)
             << "/"
             << stage2.value("completion_physical_full_failures", 0ULL)
             << "\n";
    } else {
      output << "unavailable\n";
    }
    output << "  backlog_slope_per_sec: "
           << stage2.value("backlog_slope_per_sec", 0.0) << '\n';
    output << "  failures (hard): "
           << stage2.value("failures", 0ULL) << '\n';
    output << "  peer_reverse_retry_attempts: ";
    if (stage2.value("peer_reverse_retry_delta_available", false)) {
      output << stage2.value("peer_reverse_retry_attempts", 0ULL) << '\n';
    } else {
      output << "unavailable\n";
    }
    output << "  locality telemetry: ";
    if (stage2.value("locality_delta_available", false)) {
      output << "home_match_rate="
             << stage2.value("home_match_rate", 0.0)
             << " cross_edge_reduction="
             << stage2.value("cross_edge_reduction_ratio", 0.0)
             << " avg_frontier/expansions/scored="
             << stage2.value("avg_stage2_remote_frontier", 0.0) << "/"
             << stage2.value("avg_stage2_remote_expansions", 0.0) << "/"
             << stage2.value("avg_stage2_scored_candidates", 0.0) << '\n';
    } else {
      output << "unavailable\n";
    }
    output << "  search budget exhausted (stage1/stage2): ";
    if (stage2.value("search_budget_delta_available", false)) {
      output << stage2.value("stage1_search_budget_exhausted", 0ULL)
             << "/"
             << stage2.value("stage2_search_budget_exhausted", 0ULL)
             << '\n';
    } else {
      output << "unavailable\n";
    }
    output << "  execution batching: ";
    if (stage2.value("execution_counter_delta_available", false)) {
      output << "contexts=" << stage2.value("stage2_batches", 0ULL)
             << " avg_items=" << stage2.value("avg_stage2_batch_size", 0.0)
             << " graph_reads/wave="
             << stage2.value("avg_stage2_graph_reads_per_wave", 0.0)
             << " vector_reads/wave="
             << stage2.value("avg_stage2_vector_reads_per_wave", 0.0)
             << " pressure_yields="
             << stage2.value("pressure_yields", 0ULL) << '\n';
      if (stage2.contains("ordered_graph_issue")) {
        const auto& ordered = stage2["ordered_graph_issue"];
        output << "  ordered graph issue: issued/hit/wasted="
               << ordered.value("issued", 0ULL) << "/"
               << ordered.value("hits", 0ULL) << "/"
               << ordered.value("wasted", 0ULL)
               << " promotion_ratio="
               << ordered.value("promotion_ratio", 0.0) << '\n';
      }
      if (stage2.contains("home_rpc_wire")) {
        const auto& wire = stage2["home_rpc_wire"];
        output << "  graph home RPC: batches/items/avg_items="
               << wire.value("graph_batches", 0ULL) << "/"
               << wire.value("graph_items", 0ULL) << "/"
               << wire.value("avg_graph_items_per_rpc", 0.0) << '\n';
      }
    } else {
      output << "unavailable\n";
    }
    output << "  executor service demand: ";
    if (stage2.value("timing_counter_delta_available", false)) {
      const auto& stage1 = stage2["physical_stage1"];
      const auto& timing = stage2["phase_timing"];
      output << "stage1_total/search/prune_us="
             << stage1.value("avg_total_us", 0.0) << "/"
             << stage1.value("avg_search_us", 0.0) << "/"
             << stage1.value("avg_prune_us", 0.0)
             << " stage1_candidates/frontier/neighbors="
             << stage1.value("avg_candidates", 0.0) << "/"
             << stage1.value("avg_remote_frontier", 0.0) << "/"
             << stage1.value("avg_neighbors", 0.0)
             << " stage2_search/freeze/reverse_us="
             << timing["search"].value("avg_us_per_task", 0.0) << "/"
             << timing["freeze_prune"].value("avg_us_per_task", 0.0)
             << "/"
             << timing["reverse_prepare"].value(
                  "avg_us_per_task", 0.0)
             << '\n';
    } else {
      output << "unavailable\n";
    }
  }

  if (root.contains("recall")) {
    const auto& recall = root["recall"];
    output << "recall\n";
    output << "  recall@" << recall.value("k", 0) << ": "
           << recall.value("recall", 0.0) << '\n';
    output << "  queries: " << recall.value("queries", 0) << '\n';
    output << "  mode/base_id_limit/search_width/insufficient_queries: "
           << recall.value("mode", "all") << "/"
           << recall.value("base_id_limit", 0ULL) << "/"
           << recall.value("search_result_width", 0ULL) << "/"
           << recall.value(
                "queries_with_insufficient_base_results", 0ULL) << '\n';
    output << "  query_file: " << recall.value("query_file", "") << '\n';
    output << "  groundtruth_file: " << recall.value("groundtruth_file", "") << '\n';
  }
  if (root.contains("static_gt_post_recall")) {
    const auto& recall = root["static_gt_post_recall"];
    output << "static_gt_post_recall\n";
    output << "  recall@" << recall.value("k", 0) << ": "
           << recall.value("recall", 0.0) << '\n';
    output << "  queries: " << recall.value("queries", 0) << '\n';
    output << "  mode/base_id_limit/search_width/insufficient_queries: "
           << recall.value("mode", "all") << "/"
           << recall.value("base_id_limit", 0ULL) << "/"
           << recall.value("search_result_width", 0ULL) << "/"
           << recall.value(
                "queries_with_insufficient_base_results", 0ULL) << '\n';
    output << "  query_file: " << recall.value("query_file", "") << '\n';
    output << "  groundtruth_file: " << recall.value("groundtruth_file", "") << '\n';
  }
  if (root.contains("gpu_persistent")) {
    const auto& gpu = root["gpu_persistent"];
    constexpr double bytes_per_gib = 1024.0 * 1024.0 * 1024.0;
    output << "gpu_persistent\n";
    output << "  GPU memory explicit/base_pq/route GiB: "
           << static_cast<double>(gpu.value("gpu_memory_explicit_bytes", 0ULL)) / bytes_per_gib << "/"
           << static_cast<double>(gpu.value("gpu_memory_base_pq_bytes", 0ULL)) / bytes_per_gib << "/"
           << static_cast<double>(gpu.value("gpu_memory_route_graph_bytes", 0ULL)) / bytes_per_gib << '\n';
    output << "  average_batch_size: " << gpu.value("average_batch_size", 0.0) << '\n';
    output << "  average_submission_wait_us: "
           << gpu.value("average_submission_wait_us", 0.0) << '\n';
    output << "  rdma_read_bytes: " << gpu.value("rdma_read_bytes", 0ULL) << '\n';
    output << "  owner WQE submitted/capacity/utilization: "
           << gpu.value("owner_submitted_wqes", 0ULL) << "/"
           << gpu.value("owner_submission_wqe_capacity", 0ULL) << "/"
           << gpu.value("owner_wqe_submission_utilization", 0.0) << '\n';
    output << "  owner critical/speculative batches: "
           << gpu.value("owner_critical_batches", 0ULL) << "/"
           << gpu.value("owner_speculative_batches", 0ULL) << '\n';
    output << "  graph_route_hit_ratio/refreshes: "
           << gpu.value("graph_route_hit_ratio", 0.0) << "/"
           << gpu.value("graph_route_refreshes", 0ULL) << '\n';
    output << "  graph snapshot reread records total/per_query: "
           << gpu.value("graph_read_retries", 0ULL) << "/"
           << gpu.value("average_graph_read_retries", 0.0) << '\n';
    output << "  graph RDMA bytes total/per_query/per_parent: "
           << gpu.value("graph_read_bytes", 0ULL) << "/"
           << gpu.value("average_graph_read_bytes_per_query", 0.0) << "/"
           << gpu.value(
                "average_graph_read_bytes_per_logical_parent", 0.0) << '\n';
    output << "  graph live/full/fallback/underhint/promotions reads: "
           << gpu.value("graph_live_extent_reads", 0ULL) << "/"
           << gpu.value("graph_full_record_reads", 0ULL) << "/"
           << gpu.value("graph_extent_fallback_reads", 0ULL) << "/"
           << gpu.value("graph_extent_underhint_reads", 0ULL) << "/"
           << gpu.value("graph_extent_hint_promotions", 0ULL) << '\n';
    output << "  expanded parents/neighbor sum/average degree: "
           << gpu.value("expanded_parent_count", 0ULL) << "/"
           << gpu.value("expanded_neighbor_count_sum", 0ULL) << "/"
           << gpu.value("average_expanded_parent_degree", 0.0) << '\n';
    output << "  expanded degree histogram (ceil(degree/8), overflow=13): "
           << gpu.value(
                "expanded_degree_histogram", nlohmann::json::array()).dump()
           << '\n';
    output << "  dynamic expanded parents/neighbor sum/average degree/share: "
           << gpu.value("dynamic_expanded_parent_count", 0ULL) << "/"
           << gpu.value("dynamic_expanded_neighbor_count_sum", 0ULL) << "/"
           << gpu.value("average_dynamic_expanded_parent_degree", 0.0) << "/"
           << gpu.value("dynamic_expanded_parent_ratio", 0.0) << '\n';
    output << "  dynamic expanded degree histogram "
              "(ceil(degree/8), overflow=13): "
           << gpu.value(
                "dynamic_expanded_degree_histogram",
                nlohmann::json::array()).dump()
           << '\n';
    output << "  DynaExtent short/full/bytes/fallback/promotions/demotions: "
           << gpu.value("dynamic_graph_short_reads", 0ULL) << "/"
           << gpu.value("dynamic_graph_full_reads", 0ULL) << "/"
           << gpu.value("dynamic_graph_read_bytes", 0ULL) << "/"
           << gpu.value("dynamic_graph_fallback_reads", 0ULL) << "/"
           << gpu.value("dynamic_graph_hint_promotions", 0ULL) << "/"
           << gpu.value("dynamic_graph_hint_demotions", 0ULL) << '\n';
    output << "  DynaExtent snapshot-attempts/nonfallback-full-attempts/"
              "short-physical-ratio/fallback-ratio/bytes-per-physical-read: "
           << gpu.value("dynamic_graph_snapshot_attempts", 0ULL) << "/"
           << gpu.value(
                "dynamic_graph_nonfallback_full_attempts", 0ULL) << "/"
           << gpu.value("dynamic_graph_short_physical_ratio", 0.0) << "/"
           << gpu.value("dynamic_graph_fallback_ratio", 0.0) << "/"
           << gpu.value(
                "average_dynamic_graph_read_bytes_per_physical_read", 0.0)
           << '\n';
    output << "  ASFE logical expansions total/per_query: "
           << gpu.value("logical_expansions", 0ULL) << "/"
           << gpu.value("average_logical_expansions_per_query", 0.0) << '\n';
    output << "  ASFE critical/speculative graph reads: "
           << gpu.value("critical_graph_reads", 0ULL) << "/"
           << gpu.value("speculative_graph_reads", 0ULL) << '\n';
    output << "  ASFE critical/speculative graph bytes: "
           << gpu.value("critical_graph_bytes", 0ULL) << "/"
           << gpu.value("speculative_graph_bytes", 0ULL) << '\n';
    output << "  ASFE core reads/bytes/arrived/promoted/stale: "
           << gpu.value("core_prefetch_reads", 0ULL) << "/"
           << gpu.value("core_prefetch_bytes", 0ULL) << "/"
           << gpu.value("core_prefetch_arrived", 0ULL) << "/"
           << gpu.value("core_prefetch_promoted", 0ULL) << "/"
           << gpu.value("core_prefetch_stale", 0ULL) << '\n';
    output << "  ASFE core waves/ready waves/ready ratio: "
           << gpu.value("core_prefetch_waves", 0ULL) << "/"
           << gpu.value("core_ready_waves", 0ULL) << "/"
           << gpu.value("core_ready_wave_ratio", 0.0) << '\n';
    output << "  terminal exact cache attempts/rate: "
           << gpu.value("terminal_exact_cache_attempted_queries", 0ULL)
           << "/"
           << gpu.value("terminal_exact_cache_attempt_rate", 0.0) << '\n';
    output << "  terminal exact cache issued/promoted/misses/rejects "
              "(total/per_query): "
           << gpu.value("terminal_exact_cache_issued_records", 0ULL) << "/"
           << gpu.value(
                "average_terminal_exact_cache_issued_records_per_query", 0.0)
           << ", "
           << gpu.value("terminal_exact_cache_promoted_records", 0ULL)
           << "/"
           << gpu.value(
                "average_terminal_exact_cache_promoted_records_per_query", 0.0)
           << ", "
           << gpu.value("terminal_exact_cache_miss_records", 0ULL) << "/"
           << gpu.value(
                "average_terminal_exact_cache_miss_records_per_query", 0.0)
           << ", "
           << gpu.value("terminal_exact_cache_queue_rejects", 0ULL) << "/"
           << gpu.value(
                "average_terminal_exact_cache_queue_rejects_per_query", 0.0)
           << '\n';
    output << "  terminal exact cache promotion/waste ratio: "
           << gpu.value("terminal_exact_cache_promotion_ratio", 0.0)
           << "/"
           << gpu.value("terminal_exact_cache_waste_ratio", 0.0) << '\n';
    output << "  terminal exact cache wasted bytes total/per_query: "
           << gpu.value("terminal_exact_cache_wasted_bytes", 0ULL) << "/"
           << gpu.value(
                "average_terminal_exact_cache_wasted_bytes_per_query", 0.0)
           << '\n';
    output << "  ASFE completion scoring batches/candidates "
              "(total/per_query/candidates_per_batch): "
           << gpu.value("completion_score_batches", 0ULL) << "/"
           << gpu.value(
                "average_completion_score_batches_per_query", 0.0)
           << ", "
           << gpu.value("completion_score_candidates", 0ULL) << "/"
           << gpu.value(
                "average_completion_score_candidates_per_query", 0.0)
           << "/"
           << gpu.value(
                "average_completion_score_candidates_per_batch", 0.0)
           << '\n';
    output << "  ASFE reusable exact certificates total/per_query: "
           << gpu.value("frontier_reusable_certificates", 0ULL) << "/"
           << gpu.value(
                "average_frontier_reusable_certificates_per_query", 0.0)
           << '\n';
    output << "  ASFE streamed runs/ordered scoring batches/candidates "
              "(total/per_query/candidates_per_batch): "
           << gpu.value("frontier_streamed_candidate_runs", 0ULL) << "/"
           << gpu.value(
                "average_frontier_streamed_candidate_runs_per_query", 0.0)
           << ", "
           << gpu.value("ordered_score_batches", 0ULL) << "/"
           << gpu.value(
                "average_ordered_score_batches_per_query", 0.0)
           << ", "
           << gpu.value("ordered_score_candidates", 0ULL) << "/"
           << gpu.value(
                "average_ordered_score_candidates_per_query", 0.0)
           << "/"
           << gpu.value(
                "average_ordered_score_candidates_per_batch", 0.0)
           << '\n';
    output << "  OOO ROB parents retired beyond an unresolved rank hole "
              "(total/per_query): "
           << gpu.value("ooo_bypassed_parents", 0ULL) << "/"
           << gpu.value(
                "average_ooo_bypassed_parents_per_query", 0.0)
           << '\n';
    output << "  ASFE reusable prefix ranks/full/issued certificates "
              "(total/per_query): "
           << gpu.value("frontier_reusable_prefix_ranks", 0ULL) << "/"
           << gpu.value(
                "average_frontier_reusable_prefix_ranks_per_query", 0.0)
           << ", "
           << gpu.value(
                "frontier_reusable_full_prefix_certificates", 0ULL)
           << "/"
           << gpu.value(
                "average_frontier_reusable_full_prefix_certificates_per_query",
                0.0)
           << ", "
           << gpu.value("frontier_reusable_issued_certificates", 0ULL)
           << "/"
           << gpu.value(
                "average_frontier_reusable_issued_certificates_per_query",
                0.0)
           << '\n';
    output << "  ASFE certificate rejects total/per_query: "
           << gpu.value("frontier_certificate_rejects", 0ULL) << "/"
           << gpu.value(
                "average_frontier_certificate_rejects_per_query", 0.0)
           << '\n';
    output << "  ASFE speculative wasted bytes/ratio: "
           << gpu.value("speculative_wasted_bytes", 0ULL) << "/"
           << gpu.value("speculative_wasted_byte_ratio", 0.0) << '\n';
    output << "  ASFE speculative arrived/promoted/stale/queue_rejects: "
           << gpu.value("speculative_arrived", 0ULL) << "/"
           << gpu.value("speculative_promoted", 0ULL) << "/"
           << gpu.value("speculative_stale", 0ULL) << "/"
           << gpu.value("speculative_queue_rejects", 0ULL) << '\n';
    output << "  ASFE promotion/waste ratio: "
           << gpu.value("speculative_promotion_ratio", 0.0) << "/"
           << gpu.value("speculative_waste_ratio", 0.0) << '\n';
    output << "  ASFE issue/commit epochs: "
           << gpu.value("issue_epochs", 0ULL) << "/"
           << gpu.value("commit_epochs", 0ULL) << '\n';
    output << "  ASFE average issue/commit width: "
           << gpu.value("average_issue_width", 0.0) << "/"
           << gpu.value("average_commit_width", 0.0) << '\n';
    output << "  ASFE issue capacity/utilization: "
           << gpu.value("average_issue_width_capacity", 0.0) << "/"
           << gpu.value("issue_frontier_utilization", 0.0) << '\n';
    output << "  ASFE max issue/commit width: "
           << gpu.value("max_issue_width", 0ULL) << "/"
           << gpu.value("max_commit_width", 0ULL) << '\n';
    output << "  ASFE critical ROB hits/speculative wait ns/us_per_query: "
           << gpu.value("critical_rob_hits", 0ULL) << "/"
           << gpu.value("speculative_wait_ns", 0ULL) << "/"
           << gpu.value("average_speculative_wait_us", 0.0) << '\n';
    output << "  ASFE critical misses/ROB hit ratio: "
           << gpu.value("critical_misses", 0ULL) << "/"
           << gpu.value("critical_rob_hit_ratio", 0.0) << '\n';
    output << "  ASFE RDMA completion groups/avg us (critical/speculative): "
           << gpu.value("rdma_completion_groups", 0ULL) << "/"
           << gpu.value("average_rdma_completion_latency_us", 0.0) << " ("
           << gpu.value("speculative_completion_groups", 0ULL) << "/"
           << gpu.value("average_speculative_completion_latency_us", 0.0)
           << ")\n";
    output << "  GPU occupancy threads/registers/static_shared: "
           << gpu.value("gpu_kernel_threads", 0ULL) << "/"
           << gpu.value("gpu_registers_per_thread", 0ULL) << "/"
           << gpu.value("gpu_static_shared_bytes", 0ULL) << '\n';
    output << "  GPU occupancy active/effective blocks per SM: "
           << gpu.value("gpu_active_blocks_per_sm", 0ULL) << "/"
           << gpu.value("gpu_effective_blocks_per_sm", 0ULL) << '\n';
    output << "  GPU persistent query/owner/total blocks: "
           << gpu.value("gpu_query_blocks", 0ULL) << "/"
           << gpu.value("gpu_owner_blocks", 0ULL) << "/"
           << gpu.value("gpu_total_persistent_blocks", 0ULL) << '\n';
    output << "  centroid route publications/shard_updates/live/snapshot_skips: "
           << gpu.value("centroid_route_publications", 0ULL) << "/"
           << gpu.value("centroid_route_shard_updates", 0ULL) << "/"
           << gpu.value("centroid_route_live_entries", 0ULL) << "/"
           << gpu.value("centroid_route_snapshot_skips", 0ULL) << '\n';
    output << "  centroid route probe_reads/body_reads/unchanged_polls/poll_us: "
           << gpu.value("centroid_route_probe_reads", 0ULL) << "/"
           << gpu.value("centroid_route_body_reads", 0ULL) << "/"
           << gpu.value("centroid_route_unchanged_polls", 0ULL) << "/"
           << gpu.value("centroid_route_poll_delay_us", 0ULL) << '\n';
    output << "  centroid route query snapshot retries/timeouts: "
           << gpu.value("centroid_route_query_retries", 0ULL) << "/"
           << gpu.value("centroid_route_query_timeouts", 0ULL) << '\n';
    output << "  GPU query/prepare/graph/score/beam/exact us: "
           << gpu.value("average_gpu_query_us", 0.0) << "/"
           << gpu.value("average_gpu_prepare_us", 0.0) << "/"
           << gpu.value("average_gpu_graph_us", 0.0) << "/"
           << gpu.value("average_gpu_score_us", 0.0) << "/"
           << gpu.value("average_gpu_beam_us", 0.0) << "/"
           << gpu.value("average_gpu_exact_us", 0.0) << '\n';
    output << "  exact snapshot train batches/fallbacks/success ratio: "
           << gpu.value("exact_snapshot_train_batches", 0ULL) << "/"
           << gpu.value("exact_snapshot_train_fallbacks", 0ULL) << "/"
           << gpu.value("exact_snapshot_train_success_ratio", 0.0)
           << '\n';
    output << "  GPU graph commit/issue width: "
           << root["meta"].value("gpu_graph_commit_width", 0U) << "/"
           << root["meta"].value("gpu_graph_issue_width", 0U) << '\n';
    output << "  GPU graph read/dynamic extent: "
           << root["meta"].value(
                "gpu_query_graph_read_policy", "fixed") << "/"
           << root["meta"].value("gpu_dynamic_graph_extent", false) << '\n';
    output << "  GPU Beam merge policy: "
           << root["meta"].value(
                "gpu_query_beam_merge_policy", "legacy") << '\n';
    output << "  Stage2 score-many/graph-issue/home-combining: "
           << root["meta"].value(
                "storage_owner_stage2_score_many", false) << "/"
           << root["meta"].value(
                "storage_owner_stage2_graph_issue_width", 1U) << "/"
           << root["meta"].value(
                "storage_owner_stage2_home_rpc_combining", false) << '\n';
    output << "  GPU Beam merge total/prepare/sort/materialize us: "
           << gpu.value("average_gpu_beam_merge_us", 0.0) << "/"
           << gpu.value("average_gpu_beam_merge_prepare_us", 0.0) << "/"
           << gpu.value("average_gpu_beam_merge_sort_us", 0.0) << "/"
           << gpu.value(
                "average_gpu_beam_merge_materialize_us", 0.0) << '\n';
    output << "  dynamic PQ candidates/reads/bytes/incarnation_rejects: "
           << gpu.value("dynamic_code_candidates", 0ULL) << "/"
           << gpu.value("dynamic_code_reads", 0ULL) << "/"
           << gpu.value("dynamic_code_read_bytes", 0ULL) << "/"
           << gpu.value("dynamic_code_incarnation_rejects", 0ULL) << '\n';
    output << "  dynamic PQ reads/query and wait_us/query: "
           << gpu.value("average_dynamic_code_reads", 0.0) << "/"
           << gpu.value("average_dynamic_code_wait_us", 0.0) << '\n';
    output << "  dynamic PQ arena hits/batch_dedup/authoritative_avoidance: "
           << gpu.value("dynamic_code_arena_hits", 0ULL) << "/"
           << gpu.value("dynamic_code_batch_deduplicated", 0ULL) << "/"
           << gpu.value("dynamic_code_authoritative_avoidance_ratio", 0.0)
           << '\n';
    output << "  dynamic PQ arena publish success/race/capacity: "
           << gpu.value("dynamic_code_arena_publish_successes", 0ULL) << "/"
           << gpu.value("dynamic_code_arena_publish_races", 0ULL) << "/"
           << gpu.value("dynamic_code_arena_capacity", 0ULL) << '\n';
  }
  if (root.contains("storage_owner_runtime")) {
    const auto& runtime = root["storage_owner_runtime"];
    output << "storage_owner_runtime\n";
    output << "  submitted/completed batches: "
           << runtime.value("submitted_batches", 0ULL) << "/"
           << runtime.value("completed_batches", 0ULL) << '\n';
    output << "  submitted/completed items: "
           << runtime.value("submitted_items", 0ULL) << "/"
           << runtime.value("completed_items", 0ULL) << '\n';
    output << "  average batch / average RPC wall us / max RPC wall us: "
           << runtime.value("average_submitted_batch_size", 0.0) << "/"
           << runtime.value("average_completed_rpc_wall_us", 0.0) << "/"
           << runtime.value("max_rpc_wall_ns", 0ULL) / 1000.0 << '\n';
  }
  if (report.has_insert()) {
    const auto summary = service::breakdown::aggregate_text_summary(report.insert);
    summaries["insert"] = summary;
    output << summary;
  }
  if (report.has_query()) {
    const auto summary = service::breakdown::aggregate_text_summary(report.query);
    summaries["query"] = summary;
    output << summary;
  }
  return {.bottleneck_summary = std::move(summaries), .text = output.str()};
}

}  // namespace tools::breakdown_benchmark
