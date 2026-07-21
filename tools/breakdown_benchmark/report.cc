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
    {"rdma_read_ops", telemetry.rdma_read_ops},
    {"rdma_read_bytes", telemetry.rdma_read_bytes},
    {"rdma_merged_requests", telemetry.rdma_merged_requests},
    {"direct_path_failures", telemetry.direct_path_failures},
    {"graph_page_requests", telemetry.graph_page_requests},
    {"graph_read_retries", telemetry.graph_read_retries},
    {"average_graph_read_retries", telemetry.queries_completed == 0 ? 0.0
      : static_cast<double>(telemetry.graph_read_retries) /
          static_cast<double>(telemetry.queries_completed)},
    {"graph_dependency_rounds", telemetry.graph_dependency_rounds},
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
    {"graph_route_hit_ratio",
      telemetry.graph_page_requests + telemetry.graph_route_hits == 0 ? 0.0
      : static_cast<double>(telemetry.graph_route_hits) /
          static_cast<double>(
            telemetry.graph_page_requests + telemetry.graph_route_hits)},
    {"exact_vector_reads", telemetry.exact_vector_reads},
  };
}

FormattedReport format_report(const nlohmann::json& root,
                              const service::breakdown::Report& report) {
  nlohmann::json summaries = nlohmann::json::object();
  std::ostringstream output;
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
    output << "  write_head/tail_qps: "
           << stability.value("write_head_ops_per_sec", 0.0) << "/"
           << stability.value("write_tail_ops_per_sec", 0.0) << '\n';
    output << "  write_tail_to_head_ratio: "
           << stability.value("write_tail_to_head_ratio", 0.0) << '\n';
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
    output << "  logs observed/requested: "
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
    output << "  backlog_slope_per_sec: "
           << stage2.value("backlog_slope_per_sec", 0.0) << '\n';
    output << "  failures: " << stage2.value("failures", 0ULL) << '\n';
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
    output << "  graph_route_hit_ratio/refreshes: "
           << gpu.value("graph_route_hit_ratio", 0.0) << "/"
           << gpu.value("graph_route_refreshes", 0ULL) << '\n';
    output << "  graph snapshot reread records total/per_query: "
           << gpu.value("graph_read_retries", 0ULL) << "/"
           << gpu.value("average_graph_read_retries", 0.0) << '\n';
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
    output << "  GPU query/prepare/graph/score/beam/exact us: "
           << gpu.value("average_gpu_query_us", 0.0) << "/"
           << gpu.value("average_gpu_prepare_us", 0.0) << "/"
           << gpu.value("average_gpu_graph_us", 0.0) << "/"
           << gpu.value("average_gpu_score_us", 0.0) << "/"
           << gpu.value("average_gpu_beam_us", 0.0) << "/"
           << gpu.value("average_gpu_exact_us", 0.0) << '\n';
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
