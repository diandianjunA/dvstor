#include "tools/breakdown_benchmark/report.hh"

#include <sstream>

namespace tools::breakdown_benchmark {

nlohmann::json telemetry_to_json(const gpu_search::TelemetrySnapshot& telemetry) {
  return {
      {"gpu_memory_explicit_bytes", telemetry.gpu_memory_explicit_bytes},
      {"gpu_memory_base_pq_bytes", telemetry.gpu_memory_base_pq_bytes},
      {"gpu_memory_resident_pq_bytes", telemetry.gpu_memory_resident_pq_bytes},
      {"gpu_memory_route_graph_bytes", telemetry.gpu_memory_route_graph_bytes},
      {"gpu_memory_delta_reserved_bytes", telemetry.gpu_memory_delta_reserved_bytes},
      {"gpu_memory_graph_cache_bytes", telemetry.gpu_memory_graph_cache_bytes},
      {"gpu_memory_exact_cache_bytes", telemetry.gpu_memory_exact_cache_bytes},
      {"queries_submitted", telemetry.queries_submitted},
      {"queries_completed", telemetry.queries_completed},
      {"batches", telemetry.batches},
      {"batch_queries", telemetry.batch_queries},
      {"average_batch_size", telemetry.batches == 0 ? 0.0
        : static_cast<double>(telemetry.batch_queries) / static_cast<double>(telemetry.batches)},
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
      {"graph_dependency_rounds", telemetry.graph_dependency_rounds},
      {"graph_page_cache_hits", telemetry.graph_page_cache_hits},
      {"graph_route_hits", telemetry.graph_route_hits},
      {"graph_route_refreshes", telemetry.graph_route_refreshes},
      {"graph_cache_invalidations", telemetry.graph_cache_invalidations},
      {"graph_page_cache_hit_ratio",
        telemetry.graph_page_requests + telemetry.graph_page_cache_hits == 0 ? 0.0
        : static_cast<double>(telemetry.graph_page_cache_hits) /
            static_cast<double>(telemetry.graph_page_requests +
                                telemetry.graph_page_cache_hits)},
      {"graph_route_hit_ratio",
        telemetry.graph_page_requests + telemetry.graph_page_cache_hits +
            telemetry.graph_route_hits == 0 ? 0.0
        : static_cast<double>(telemetry.graph_route_hits) /
            static_cast<double>(telemetry.graph_page_requests +
                                telemetry.graph_page_cache_hits +
                                telemetry.graph_route_hits)},
      {"exact_vector_reads", telemetry.exact_vector_reads},
      {"exact_vector_cache_hits", telemetry.exact_vector_cache_hits},
      {"exact_vector_cache_hit_ratio",
       telemetry.exact_vector_reads + telemetry.exact_vector_cache_hits == 0
         ? 0.0
         : static_cast<double>(telemetry.exact_vector_cache_hits) /
             static_cast<double>(telemetry.exact_vector_reads +
                                 telemetry.exact_vector_cache_hits)},
      {"delta_queries", telemetry.delta_queries},
      {"mutations_published", telemetry.mutations_published},
      {"delta_publications", telemetry.delta_publications},
      {"average_mutations_per_publication", telemetry.delta_publications == 0 ? 0.0
        : static_cast<double>(telemetry.mutations_published) /
            static_cast<double>(telemetry.delta_publications)},
      {"delta_reclaim_batches", telemetry.delta_reclaim_batches},
      {"delta_entries_retired", telemetry.delta_entries_retired},
      {"storage_reclaim_ack_writes", telemetry.storage_reclaim_ack_writes},
      {"storage_reclaim_ack_sequence", telemetry.storage_reclaim_ack_sequence},
      {"delta_live_entries", telemetry.delta_live_entries},
      {"delta_physical_entries", telemetry.delta_physical_entries},
      {"delta_mutable_entries", telemetry.delta_mutable_entries},
      {"delta_durable_entries", telemetry.delta_durable_entries},
      {"resident_pq_capacity", telemetry.resident_pq_capacity},
      {"resident_pq_entries", telemetry.resident_pq_entries},
      {"resident_pq_peak_entries", telemetry.resident_pq_peak_entries},
      {"resident_pq_reclaimed", telemetry.resident_pq_reclaimed},
      {"mutation_capacity_rejections", telemetry.mutation_capacity_rejections},
      {"mutation_capacity_reserved", telemetry.mutation_capacity_reserved},
      {"mutation_capacity_reserved_max", telemetry.mutation_capacity_reserved_max},
      {"average_visibility_us", telemetry.mutations_published == 0 ? 0.0
        : static_cast<double>(telemetry.visibility_ns_total) /
            static_cast<double>(telemetry.mutations_published) / 1000.0},
      {"max_visibility_us", static_cast<double>(telemetry.visibility_ns_max) / 1000.0},
      {"average_publication_queue_us", telemetry.mutations_published == 0 ? 0.0
        : static_cast<double>(telemetry.publication_queue_ns_total) /
            static_cast<double>(telemetry.mutations_published) / 1000.0},
      {"average_publication_prepare_us", telemetry.delta_publications == 0 ? 0.0
        : static_cast<double>(telemetry.publication_prepare_ns_total) /
            static_cast<double>(telemetry.delta_publications) / 1000.0},
      {"average_publication_command_us", telemetry.delta_publications == 0 ? 0.0
        : static_cast<double>(telemetry.publication_command_ns_total) /
            static_cast<double>(telemetry.delta_publications) / 1000.0},
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
    output << "  write_ops_per_sec: " << throughput.value("write_ops_per_sec", 0.0)
           << " (ops=" << write_ops << ")\n";
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
    output << "  zero_completion_windows: "
           << stability.value("zero_completion_windows", 0ULL) << '\n';
    output << "  acceptance_passed: "
           << (root["acceptance"].value("passed", false) ? "true" : "false") << '\n';
  }

  if (root.contains("recall")) {
    const auto& recall = root["recall"];
    output << "recall\n";
    output << "  recall@" << recall.value("k", 0) << ": "
           << recall.value("recall", 0.0) << '\n';
    output << "  queries: " << recall.value("queries", 0) << '\n';
    output << "  passed: " << (recall.value("passed", false) ? "true" : "false") << '\n';
    output << "  query_file: " << recall.value("query_file", "") << '\n';
    output << "  groundtruth_file: " << recall.value("groundtruth_file", "") << '\n';
  }
  if (root.contains("static_gt_post_recall")) {
    const auto& recall = root["static_gt_post_recall"];
    output << "static_gt_post_recall\n";
    output << "  recall@" << recall.value("k", 0) << ": "
           << recall.value("recall", 0.0) << '\n';
    output << "  queries: " << recall.value("queries", 0) << '\n';
    output << "  query_file: " << recall.value("query_file", "") << '\n';
    output << "  groundtruth_file: " << recall.value("groundtruth_file", "") << '\n';
  }
  if (root.contains("gpu_persistent")) {
    const auto& gpu = root["gpu_persistent"];
    constexpr double bytes_per_gib = 1024.0 * 1024.0 * 1024.0;
    output << "gpu_persistent\n";
    output << "  GPU memory explicit/base_pq/resident_pq/route/delta/graph_cache/exact_cache GiB: "
           << static_cast<double>(gpu.value("gpu_memory_explicit_bytes", 0ULL)) / bytes_per_gib << "/"
           << static_cast<double>(gpu.value("gpu_memory_base_pq_bytes", 0ULL)) / bytes_per_gib << "/"
           << static_cast<double>(gpu.value("gpu_memory_resident_pq_bytes", 0ULL)) / bytes_per_gib << "/"
           << static_cast<double>(gpu.value("gpu_memory_route_graph_bytes", 0ULL)) / bytes_per_gib << "/"
           << static_cast<double>(gpu.value("gpu_memory_delta_reserved_bytes", 0ULL)) / bytes_per_gib << "/"
           << static_cast<double>(gpu.value("gpu_memory_graph_cache_bytes", 0ULL)) / bytes_per_gib << "/"
           << static_cast<double>(gpu.value("gpu_memory_exact_cache_bytes", 0ULL)) / bytes_per_gib << '\n';
    output << "  average_batch_size: " << gpu.value("average_batch_size", 0.0) << '\n';
    output << "  average_submission_wait_us: "
           << gpu.value("average_submission_wait_us", 0.0) << '\n';
    output << "  rdma_read_bytes: " << gpu.value("rdma_read_bytes", 0ULL) << '\n';
    output << "  graph_page_cache_hit_ratio: "
           << gpu.value("graph_page_cache_hit_ratio", 0.0) << '\n';
    output << "  graph_route_hit_ratio/refreshes: "
           << gpu.value("graph_route_hit_ratio", 0.0) << "/"
           << gpu.value("graph_route_refreshes", 0ULL) << '\n';
    output << "  graph_cache_invalidations: "
           << gpu.value("graph_cache_invalidations", 0ULL) << '\n';
    output << "  GPU query/prepare/graph/score/beam/exact us: "
           << gpu.value("average_gpu_query_us", 0.0) << "/"
           << gpu.value("average_gpu_prepare_us", 0.0) << "/"
           << gpu.value("average_gpu_graph_us", 0.0) << "/"
           << gpu.value("average_gpu_score_us", 0.0) << "/"
           << gpu.value("average_gpu_beam_us", 0.0) << "/"
           << gpu.value("average_gpu_exact_us", 0.0) << '\n';
    output << "  average_visibility_us: "
           << gpu.value("average_visibility_us", 0.0) << '\n';
    output << "  publication queue/prepare/command us: "
           << gpu.value("average_publication_queue_us", 0.0) << "/"
           << gpu.value("average_publication_prepare_us", 0.0) << "/"
           << gpu.value("average_publication_command_us", 0.0) << '\n';
    output << "  average_mutations_per_publication: "
           << gpu.value("average_mutations_per_publication", 0.0) << '\n';
    output << "  delta logical/physical/L0/L1 entries: "
           << gpu.value("delta_live_entries", 0ULL) << "/"
           << gpu.value("delta_physical_entries", 0ULL) << "/"
           << gpu.value("delta_mutable_entries", 0ULL) << "/"
           << gpu.value("delta_durable_entries", 0ULL) << '\n';
    output << "  resident PQ current/peak/capacity/reclaimed: "
           << gpu.value("resident_pq_entries", 0ULL) << "/"
           << gpu.value("resident_pq_peak_entries", 0ULL) << "/"
           << gpu.value("resident_pq_capacity", 0ULL) << "/"
           << gpu.value("resident_pq_reclaimed", 0ULL) << '\n';
    output << "  mutation capacity rejected/current/peak: "
           << gpu.value("mutation_capacity_rejections", 0ULL) << "/"
           << gpu.value("mutation_capacity_reserved", 0ULL) << "/"
           << gpu.value("mutation_capacity_reserved_max", 0ULL) << '\n';
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
