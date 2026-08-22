#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "tools/breakdown_benchmark/dataset.hh"
#include "tools/breakdown_benchmark/maintenance_log.hh"
#include "tools/breakdown_benchmark/report.hh"

namespace {

void test_deterministic_dataset() {
  const auto first = tools::breakdown_benchmark::make_deterministic_vector(17, 32);
  const auto second = tools::breakdown_benchmark::make_deterministic_vector(17, 32);
  const auto different = tools::breakdown_benchmark::make_deterministic_vector(18, 32);
  assert(first == second);
  assert(first != different);

  const std::vector<uint32_t> ids{17, 18};
  const auto dataset = tools::breakdown_benchmark::make_dataset(ids, 32);
  assert(dataset.size() == 64);
  assert(std::equal(first.begin(), first.end(), dataset.begin()));
  assert(std::equal(different.begin(), different.end(), dataset.begin() + 32));
}

void test_vector_file_reader() {
  const auto path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_benchmark_support_test.u8bin";
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const uint32_t rows = 2;
    const uint32_t dim = 3;
    const std::array<uint8_t, 6> values{1, 2, 3, 4, 5, 6};
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    output.write(reinterpret_cast<const char*>(values.data()), values.size());
  }

  const auto rows = tools::breakdown_benchmark::read_vector_rows(path.string(), true);
  std::filesystem::remove(path);
  assert(rows.dtype == VectorDType::uint8);
  assert(rows.count == 2);
  assert(rows.dim == 3);
  assert(rows.vector_bytes == 3);
  assert(rows.decoded == std::vector<float>({1, 2, 3, 4, 5, 6}));
}

void test_single_pass_stream() {
  constexpr size_t row_count = 257;
  tools::breakdown_benchmark::SinglePassRowStream stream(row_count);
  std::array<std::atomic_uint32_t, row_count> claims{};
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&] {
      while (const auto row = stream.try_claim()) {
        claims[*row].fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& worker : workers) worker.join();

  assert(stream.exhausted());
  assert(stream.consumed() == row_count);
  assert(stream.capacity() == row_count);
  assert(!stream.try_claim().has_value());
  for (const auto& count : claims) {
    assert(count.load(std::memory_order_relaxed) == 1);
  }
}

void test_recall_and_report_formatting() {
  const std::array<uint32_t, 4> truth{1, 2, 3, 4};
  const std::vector<uint32_t> results{4, 8, 2, 9};
  assert(std::abs(tools::breakdown_benchmark::recall_at(
    results, truth.data(), truth.size()) - 0.5) < 1e-9);

  gpu_search::TelemetrySnapshot telemetry;
  telemetry.queries_completed = 2;
  telemetry.gpu_beam_merge_prepare_ns = 4'000;
  telemetry.gpu_beam_merge_sort_ns = 6'000;
  telemetry.gpu_beam_merge_materialize_ns = 8'000;
  telemetry.graph_read_retries = 11;
  telemetry.graph_page_requests = 10;
  telemetry.graph_read_bytes = 4'000;
  telemetry.graph_live_extent_reads = 8;
  telemetry.graph_full_record_reads = 3;
  telemetry.graph_extent_fallback_reads = 1;
  telemetry.graph_extent_underhint_reads = 1;
  telemetry.graph_extent_hint_promotions = 1;
  telemetry.dynamic_graph_short_reads = 8;
  telemetry.dynamic_graph_full_reads = 3;
  telemetry.dynamic_graph_read_bytes = 2'000;
  telemetry.dynamic_graph_fallback_reads = 1;
  telemetry.dynamic_graph_hint_promotions = 2;
  telemetry.dynamic_graph_hint_demotions = 4;
  telemetry.centroid_route_publications = 7;
  telemetry.centroid_route_shard_updates = 9;
  telemetry.centroid_route_live_entries = 13;
  telemetry.centroid_route_probe_reads = 101;
  telemetry.centroid_route_body_reads = 3;
  telemetry.centroid_route_unchanged_polls = 47;
  telemetry.centroid_route_poll_delay_us = 8000;
  telemetry.centroid_route_query_retries = 23;
  telemetry.centroid_route_query_timeouts = 0;
  telemetry.dynamic_code_candidates = 100;
  telemetry.dynamic_code_reads = 7;
  telemetry.dynamic_code_cache_hits = 89;
  telemetry.dynamic_code_batch_deduplicated = 4;
  telemetry.dynamic_code_cache_lookup_probes = 125;
  telemetry.dynamic_code_cache_max_lookup_probes = 3;
  telemetry.dynamic_code_cache_occupied = 64;
  telemetry.dynamic_code_cache_capacity = 256;
  telemetry.ooo_bypassed_parents = 6;
  const auto telemetry_json = tools::breakdown_benchmark::telemetry_to_json(telemetry);
  assert(telemetry_json.at("graph_read_retries") == 11);
  assert(telemetry_json.at("graph_read_bytes") == 4'000);
  assert(telemetry_json.at("average_graph_read_bytes_per_query") == 2'000.0);
  assert(
    telemetry_json.at("average_graph_read_bytes_per_logical_parent") == 400.0);
  assert(telemetry_json.at("graph_live_extent_reads") == 8);
  assert(telemetry_json.at("graph_full_record_reads") == 3);
  assert(telemetry_json.at("graph_extent_fallback_reads") == 1);
  assert(telemetry_json.at("graph_extent_underhint_reads") == 1);
  assert(telemetry_json.at("graph_extent_hint_promotions") == 1);
  assert(telemetry_json.at("dynamic_graph_short_reads") == 8);
  assert(telemetry_json.at("dynamic_graph_full_reads") == 3);
  assert(telemetry_json.at("dynamic_graph_read_bytes") == 2'000);
  assert(telemetry_json.at("dynamic_graph_fallback_reads") == 1);
  assert(telemetry_json.at("dynamic_graph_hint_promotions") == 2);
  assert(telemetry_json.at("dynamic_graph_hint_demotions") == 4);
  assert(telemetry_json.at("dynamic_graph_snapshot_attempts") == 11);
  assert(telemetry_json.at(
           "dynamic_graph_nonfallback_full_attempts") == 2);
  assert(std::abs(
    telemetry_json.at("dynamic_graph_short_physical_ratio").get<double>() -
    8.0 / 11.0) < 1e-9);
  assert(std::abs(
    telemetry_json.at("dynamic_graph_fallback_ratio").get<double>() -
    0.125) < 1e-9);
  assert(std::abs(telemetry_json.at(
                    "average_dynamic_graph_read_bytes_per_physical_read")
                    .get<double>() -
                  2'000.0 / 11.0) < 1e-9);
  assert(telemetry_json.at(
           "average_dynamic_graph_read_bytes_per_query") == 1'000.0);
  assert(std::abs(
    telemetry_json.at("graph_live_extent_read_ratio").get<double>() -
    8.0 / 11.0) < 1e-9);
  assert(telemetry_json.at("graph_extent_fallback_ratio") == 0.125);
  assert(telemetry_json.at("gpu_beam_merge_prepare_ns") == 4'000);
  assert(telemetry_json.at("gpu_beam_merge_sort_ns") == 6'000);
  assert(telemetry_json.at("gpu_beam_merge_materialize_ns") == 8'000);
  assert(telemetry_json.at("average_gpu_beam_merge_prepare_us") == 2.0);
  assert(telemetry_json.at("average_gpu_beam_merge_sort_us") == 3.0);
  assert(telemetry_json.at("average_gpu_beam_merge_materialize_us") == 4.0);
  assert(telemetry_json.at("centroid_route_publications") == 7);
  assert(telemetry_json.at("centroid_route_shard_updates") == 9);
  assert(telemetry_json.at("centroid_route_live_entries") == 13);
  assert(telemetry_json.at("centroid_route_probe_reads") == 101);
  assert(telemetry_json.at("centroid_route_body_reads") == 3);
  assert(telemetry_json.at("centroid_route_unchanged_polls") == 47);
  assert(telemetry_json.at("centroid_route_poll_delay_us") == 8000);
  assert(telemetry_json.at("centroid_route_query_retries") == 23);
  assert(telemetry_json.at("centroid_route_query_timeouts") == 0);
  assert(telemetry_json.at("ooo_bypassed_parents") == 6);
  assert(telemetry_json.at("average_ooo_bypassed_parents_per_query") == 3.0);
  assert(telemetry_json.at("dynamic_code_cache_hits") == 89);
  assert(telemetry_json.at("dynamic_code_batch_deduplicated") == 4);
  assert(std::abs(telemetry_json.at("dynamic_code_cache_hit_ratio").get<double>() -
                  0.89) < 1e-9);
  assert(std::abs(telemetry_json.at(
                    "dynamic_code_authoritative_avoidance_ratio").get<double>() -
                  0.93) < 1e-9);
  assert(std::abs(telemetry_json.at("dynamic_code_cache_load_factor").get<double>() -
                  0.25) < 1e-9);
  assert(std::abs(telemetry_json.at(
                    "average_dynamic_code_cache_lookup_probes").get<double>() -
                  1.25) < 1e-9);

  nlohmann::json root;
  root["meta"] = {
    {"system_variant", {
      {"profile_name", "04_gpu_persistent_gpunetio_baseline"},
      {"label", "baseline"},
      {"update_mutation_api", "append_only"},
      {"resolved_modes", {
        {"storage_owner_update_completion_mode", "coupled"},
        {"gpu_dynamic_graph_access_mode", "fixed"},
        {"gpu_rdma_search_progression_mode", "coupled"},
      }},
      {"index", {
        {"prefix", "/data/index/shared"},
        {"schema_version", 16},
        {"build_fingerprint", 0x123456789abcdef0ULL},
      }},
    }},
    {"workload", "query"},
    {"gpu_graph_commit_width", 16},
    {"gpu_graph_issue_width", 32},
    {"gpu_query_graph_read_policy", "live-extent"},
    {"gpu_dynamic_graph_extent", true},
    {"gpu_query_beam_merge_policy", "stable-run"},
    {"recall_query", {{"source", "recall.u8bin"}, {"rows", 1000}}},
    {"performance_query", {
      {"source", "performance.u8bin"},
      {"rows", 3000},
      {"row_reuse_policy", "single_pass_no_reuse"},
      {"warmup_rows_consumed", 100},
      {"measure_rows_consumed", 200},
      {"total_rows_consumed", 300},
    }},
  };
  root["throughput"] = {{"duration_seconds", 0.0}};
  root["gpu_persistent"] = telemetry_json;
  assert(root["meta"]["system_variant"]["update_mutation_api"] ==
         "append_only");
  root["stage2"] = {
    {"requested_logs", 1},
    {"failures", 0},
    {"peer_reverse_retry_attempts", 7},
    {"peer_reverse_retry_delta_available", true},
    {"execution_counter_delta_available", true},
    {"ordered_graph_issue", {
      {"issued", 100}, {"hits", 75}, {"wasted", 20},
      {"promotion_ratio", 75.0 / 95.0},
    }},
    {"home_rpc_wire", {
      {"graph_batches", 10}, {"graph_items", 80},
      {"avg_graph_items_per_rpc", 8.0},
    }},
  };
  service::breakdown::Report report;
  report.query.operation = service::breakdown::Operation::query;
  report.query.count = 1;
  report.query.end_to_end_latencies_ns.push_back(1'000'000);
  const auto formatted = tools::breakdown_benchmark::format_report(root, report);
  assert(formatted.bottleneck_summary.contains("query"));
  assert(formatted.text.find(
           "profile_name: 04_gpu_persistent_gpunetio_baseline") !=
         std::string::npos);
  assert(formatted.text.find("update_mutation_api: append_only") !=
         std::string::npos);
  assert(formatted.text.find(
           "storage_owner_update_completion_mode: coupled") !=
         std::string::npos);
  assert(formatted.text.find(
           "gpu_dynamic_graph_access_mode: fixed") !=
         std::string::npos);
  assert(formatted.text.find(
           "gpu_rdma_search_progression_mode: coupled") !=
         std::string::npos);
  assert(formatted.text.find("index_prefix: /data/index/shared") !=
         std::string::npos);
  assert(formatted.text.find("index_build_fingerprint: 1311768467463790320") !=
         std::string::npos);
  assert(formatted.text.find("single_pass_no_reuse") != std::string::npos);
  assert(formatted.text.find("query breakdown") != std::string::npos);
  assert(formatted.text.find("failures (hard): 0") != std::string::npos);
  assert(formatted.text.find("peer_reverse_retry_attempts: 7") !=
         std::string::npos);
  assert(formatted.text.find(
           "ordered graph issue: issued/hit/wasted=100/75/20") !=
         std::string::npos);
  assert(formatted.text.find(
           "graph home RPC: batches/items/avg_items=10/80/8") !=
         std::string::npos);
  assert(formatted.text.find("GPU Beam merge policy: stable-run") !=
         std::string::npos);
  assert(formatted.text.find("GPU graph commit/issue width: 16/32") !=
         std::string::npos);
  assert(formatted.text.find(
           "GPU graph read/dynamic extent: live-extent/1") !=
         std::string::npos);
  assert(formatted.text.find(
           "GPU Beam merge total/prepare/sort/materialize us:") !=
         std::string::npos);
  assert(formatted.text.find(
           "DynaExtent short/full/bytes/fallback/promotions/demotions: "
           "8/3/2000/1/2/4") != std::string::npos);
  assert(formatted.text.find(
           "DynaExtent snapshot-attempts/nonfallback-full-attempts/"
           "short-physical-ratio/fallback-ratio/"
           "bytes-per-physical-read: 11/2/") != std::string::npos);
}

void test_dynamic_cache_telemetry_reset_preserves_lifetime_gauges() {
  gpu_search::Telemetry telemetry;
  telemetry.dynamic_code_cache_hits.store(17, std::memory_order_relaxed);
  telemetry.dynamic_code_cache_publish_successes.store(
    5, std::memory_order_relaxed);
  telemetry.dynamic_code_cache_occupied.store(123, std::memory_order_relaxed);
  telemetry.dynamic_code_cache_capacity.store(256, std::memory_order_relaxed);
  telemetry.dynamic_graph_short_reads.store(11, std::memory_order_relaxed);
  telemetry.dynamic_graph_full_reads.store(12, std::memory_order_relaxed);
  telemetry.dynamic_graph_read_bytes.store(13, std::memory_order_relaxed);
  telemetry.dynamic_graph_fallback_reads.store(14, std::memory_order_relaxed);
  telemetry.dynamic_graph_hint_promotions.store(15, std::memory_order_relaxed);
  telemetry.dynamic_graph_hint_demotions.store(16, std::memory_order_relaxed);
  const auto before_reset = telemetry.snapshot();
  assert(before_reset.dynamic_graph_short_reads == 11);
  assert(before_reset.dynamic_graph_full_reads == 12);
  assert(before_reset.dynamic_graph_read_bytes == 13);
  assert(before_reset.dynamic_graph_fallback_reads == 14);
  assert(before_reset.dynamic_graph_hint_promotions == 15);
  assert(before_reset.dynamic_graph_hint_demotions == 16);
  telemetry.reset();
  const auto snapshot = telemetry.snapshot();
  assert(snapshot.dynamic_code_cache_hits == 0);
  assert(snapshot.dynamic_code_cache_publish_successes == 0);
  assert(snapshot.dynamic_code_cache_occupied == 123);
  assert(snapshot.dynamic_code_cache_capacity == 256);
  assert(snapshot.dynamic_graph_short_reads == 0);
  assert(snapshot.dynamic_graph_full_reads == 0);
  assert(snapshot.dynamic_graph_read_bytes == 0);
  assert(snapshot.dynamic_graph_fallback_reads == 0);
  assert(snapshot.dynamic_graph_hint_promotions == 0);
  assert(snapshot.dynamic_graph_hint_demotions == 0);
}

void test_base_only_recall_filter() {
  const std::vector<node_t> results{
    100'000'001u, 7u, 100'000'002u, 9u, 11u, 13u};
  const auto filtered =
    tools::breakdown_benchmark::filter_base_only_recall_ids(
      results, 100'000'000u, 3);
  assert((filtered == std::vector<uint32_t>{7u, 9u, 11u}));
}

void test_maintenance_log_window() {
  using tools::breakdown_benchmark::kMaintenanceLatencyBucketCount;
  using Histogram = std::array<uint64_t, kMaintenanceLatencyBucketCount>;
  auto histogram_text = [](const Histogram& histogram) {
    std::string text;
    for (size_t index = 0; index < histogram.size(); ++index) {
      if (index != 0) text.push_back(',');
      text += std::to_string(histogram[index]);
    }
    return text;
  };
  auto write_observation = [&](std::ostream& output,
                               uint64_t enqueued,
                               uint64_t completed,
                               uint64_t remaining,
                               uint64_t failed,
                               uint64_t peer_failed,
                               const Histogram& histogram,
                               bool include_packing = true) {
    output << "[STATUS]: storage-owner maintenance observation: "
           << "stage2_enqueued=" << enqueued << ' '
           << "stage2_finalized_live=" << completed << " stale=0 "
           << "remaining=" << remaining << ' '
           << "peer_reverse_remaining=0 "
           << "failed=" << failed << ' '
           << "peer_reverse_failed=" << peer_failed << ' '
           << "admission_window=512 "
           << "completion_outstanding=" << remaining << ' '
           << "completion_incomplete=" << remaining / 2 << ' '
           << "completion_logical_full_failures=" << completed * 5 << ' '
           << "completion_physical_full_failures=" << completed << ' ';
    output
           << "stage2_continuations=" << completed << ' '
           << "stage2_remote_frontier_items=" << completed * 2 << ' '
           << "stage2_remote_expansions=" << completed * 3 << ' '
           << "stage2_scored_candidates=" << completed * 4 << ' '
           << "stage2_migrations=" << completed / 10 << ' '
           << "stage2_final_edges=" << completed * 8 << ' '
           << "stage2_cross_edges_stage1_home=" << completed * 4 << ' '
           << "stage2_cross_edges_final_home=" << completed * 2 << ' '
           << "stage1_search_budget_exhausted=" << completed / 20 << ' '
           << "stage2_search_budget_exhausted=" << completed / 10 << ' '
           << "maintenance_targeted_wakes=" << completed * 2 << ' '
           << "maintenance_generic_wakes=" << completed * 3 << ' '
           << "maintenance_broadcast_wakes=" << completed / 10 << ' '
           << "maintenance_context_slots_scanned=" << completed * 16 << ' ';
    if (include_packing) {
      output << "packing_target_batch=8 "
             << "packing_arrival_interval_us=" << completed + 100 << ' '
             << "packing_waited_batches=" << completed << ' '
             << "packing_wait_ms=" << completed / 1000.0 << ' '
             << "packing_target_flushes=" << completed * 2 << ' '
             << "packing_deadline_flushes=" << completed * 3 << ' '
             << "packing_cleanup_flushes=" << completed * 6 << ' ';
    }
    output
           // Deliberately stale cumulative p99 fields: the parser must use
           // only the histogram delta from the cursor baseline.
           << "p99_stage2_delay_upper_ms=30000 "
           << "p99_stage2_delay_over_30s=true "
           << "stage2_delay_histogram=" << histogram_text(histogram)
           << '\n';
  };

  const auto path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_maintenance_test.log";
  Histogram histogram{};
  histogram.back() = 5;
  {
    std::ofstream output(path, std::ios::trunc);
    write_observation(output, 9, 0, 9, 9, 3, histogram);
  }
  const auto load_begin =
    tools::breakdown_benchmark::snapshot_maintenance_logs({path.string()});
  {
    std::ofstream output(path, std::ios::app);
    histogram[2] = 20;
    write_observation(output, 100, 90, 10, 9, 3, histogram);
    histogram[2] = 60;
    write_observation(output, 200, 195, 5, 9, 3, histogram);
    histogram[2] = 100;
    write_observation(output, 300, 300, 0, 10, 7, histogram);
  }

  const auto post_stop =
    tools::breakdown_benchmark::snapshot_maintenance_logs({path.string()});
  const auto summary = tools::breakdown_benchmark::
    summarize_maintenance_log_window(load_begin, post_stop);
  assert(summary.requested_logs == 1);
  assert(summary.readable_logs == 1);
  assert(summary.logs_with_observations == 1);
  assert(summary.observations == 3);
  assert(summary.remaining == 0);
  assert(summary.max_backlog_observed == 10);
  assert(summary.failures == 1);
  assert(summary.peer_reverse_retry_attempts == 4);
  assert(summary.completion_window_available);
  assert(summary.locality_delta_available);
  assert(summary.stage2_finalized_live == 300);
  assert(summary.stage2_continuations == 300);
  assert(summary.stage2_remote_frontier_items == 600);
  assert(summary.stage2_remote_expansions == 900);
  assert(summary.stage2_scored_candidates == 1200);
  assert(summary.stage2_migrations == 30);
  assert(summary.stage2_final_edges == 2400);
  assert(summary.stage2_cross_edges_stage1_home == 1200);
  assert(summary.stage2_cross_edges_final_home == 600);
  assert(summary.search_budget_delta_available);
  assert(summary.stage1_search_budget_exhausted == 15);
  assert(summary.stage2_search_budget_exhausted == 30);
  assert(summary.wake_counter_delta_available);
  assert(summary.maintenance_targeted_wakes == 600);
  assert(summary.maintenance_generic_wakes == 900);
  assert(summary.maintenance_broadcast_wakes == 30);
  assert(summary.maintenance_context_slots_scanned == 4'800);
  assert(summary.packing_delta_available);
  assert(summary.packing_target_batch_max == 8);
  assert(summary.packing_arrival_interval_us_max == 400);
  assert(summary.packing_waited_batches == 300);
  assert(summary.packing_wait_ns == 300'000);
  assert(summary.packing_target_flushes == 600);
  assert(summary.packing_deadline_flushes == 900);
  assert(summary.packing_cleanup_flushes == 1'800);
  assert(summary.admission_window == 512);
  assert(summary.completion_outstanding == 0);
  assert(summary.max_completion_outstanding_per_shard == 10);
  assert(summary.exact_completion_credit_available);
  assert(summary.completion_incomplete == 0);
  assert(summary.max_completion_incomplete_per_shard == 5);
  assert(summary.completion_admission_failure_delta_available);
  assert(summary.completion_logical_full_failures == 1'500);
  assert(summary.completion_physical_full_failures == 300);
  assert(summary.failure_delta_available);
  assert(summary.peer_reverse_retry_delta_available);
  assert(summary.p99_stage2_delay_available);
  assert(summary.p99_stage2_delay_samples == 100);
  assert(summary.p99_stage2_delay_upper_ms == 4.0);
  assert(!summary.p99_stage2_delay_over_30s);
  assert(summary.backlog_slope_available);
  assert(summary.backlog_slope_per_sec < 0.0);

  // A new post-stop cursor has no observation yet. Its default remaining=0
  // must not be mistaken for a completed drain.
  const auto empty_post_stop =
    tools::breakdown_benchmark::summarize_maintenance_logs(post_stop);
  assert(empty_post_stop.logs_with_observations == 0);
  assert(!empty_post_stop.failure_delta_available);
  assert(!empty_post_stop.packing_delta_available);
  assert(!empty_post_stop.p99_stage2_delay_available);

  {
    std::ofstream output(path, std::ios::app);
    write_observation(output, 303, 300, 3, 10, 7, histogram);
  }
  const auto post_stop_without_completion =
    tools::breakdown_benchmark::summarize_maintenance_logs(post_stop);
  assert(post_stop_without_completion.logs_with_observations == 1);
  assert(post_stop_without_completion.failure_delta_available);
  assert(post_stop_without_completion.failures == 0);
  assert(post_stop_without_completion.peer_reverse_retry_attempts == 0);
  assert(post_stop_without_completion.peer_reverse_retry_delta_available);
  assert(post_stop_without_completion.packing_delta_available);
  assert(post_stop_without_completion.p99_stage2_delay_samples == 0);
  assert(!post_stop_without_completion.p99_stage2_delay_available);

  // Appending drain observations cannot change the frozen load window.
  const auto frozen_again = tools::breakdown_benchmark::
    summarize_maintenance_log_window(load_begin, post_stop);
  assert(frozen_again.observations == 3);
  assert(frozen_again.backlog_slope_per_sec == summary.backlog_slope_per_sec);

  // Keep an explicit 5-second bucket so the raw p99 report distinguishes an
  // exact 5000 ms boundary from the following 8000 ms bucket.
  {
    std::ofstream output(path, std::ios::trunc);
  }
  const auto exact_5s_begin =
    tools::breakdown_benchmark::snapshot_maintenance_logs({path.string()});
  histogram = {};
  histogram[13] = 100;
  {
    std::ofstream output(path, std::ios::app);
    write_observation(output, 100, 100, 0, 0, 0, histogram);
  }
  const auto exact_5s =
    tools::breakdown_benchmark::summarize_maintenance_logs(exact_5s_begin);
  assert(exact_5s.p99_stage2_delay_available);
  assert(exact_5s.p99_stage2_delay_upper_ms == 5000.0);
  assert(exact_5s.completion_admission_failure_delta_available);
  assert(exact_5s.completion_logical_full_failures == 500);
  assert(exact_5s.completion_physical_full_failures == 100);

  {
    std::ofstream output(path, std::ios::trunc);
  }
  const auto over_5s_begin =
    tools::breakdown_benchmark::snapshot_maintenance_logs({path.string()});
  histogram = {};
  histogram[14] = 100;
  {
    std::ofstream output(path, std::ios::app);
    write_observation(output, 100, 100, 0, 0, 0, histogram);
  }
  const auto over_5s =
    tools::breakdown_benchmark::summarize_maintenance_logs(over_5s_begin);
  assert(over_5s.p99_stage2_delay_available);
  assert(over_5s.p99_stage2_delay_upper_ms == 8000.0);

  // Older services did not emit adaptive-packing fields.  Their missing
  // counters must remain distinguishable from a valid all-zero delta.
  {
    std::ofstream output(path, std::ios::trunc);
    write_observation(output, 0, 0, 0, 0, 0, histogram, false);
  }
  const auto legacy_begin =
    tools::breakdown_benchmark::snapshot_maintenance_logs({path.string()});
  {
    std::ofstream output(path, std::ios::app);
    write_observation(output, 1, 1, 0, 0, 0, histogram, false);
  }
  const auto legacy_summary =
    tools::breakdown_benchmark::summarize_maintenance_logs(legacy_begin);
  assert(!legacy_summary.packing_delta_available);
  assert(legacy_summary.packing_target_batch_max == 0);
  assert(legacy_summary.packing_waited_batches == 0);

  std::filesystem::remove(path);
}

void test_in_band_maintenance_snapshot_window() {
  namespace telemetry = gpu_search::maintenance_telemetry;
  using Snapshot = telemetry::Snapshot;
  std::vector<std::optional<Snapshot>> begin(2);
  std::vector<std::optional<Snapshot>> end(2);
  for (uint32_t shard = 0; shard < 2; ++shard) {
    begin[shard] = Snapshot{
      .sequence = 2,
      .shard_id = shard,
      .published_steady_ns = 10'000'000'000ull,
      .stage2_enqueued = 10,
      .stage2_finalized_live = 8,
      .remaining = 2,
      .failed = 3,
      .peer_reverse_failed = 5,
      .admission_window = 128,
      .completion_outstanding = 2,
      .max_backlog = 2,
      .stage1_search_budget_exhausted = 1,
      .stage2_search_budget_exhausted = 2,
      .stage2_continuations = 8,
      .stage2_remote_frontier_items = 80,
      .stage2_remote_expansions = 24,
      .stage2_scored_candidates = 800,
      .stage2_migrations = 1,
      .stage2_final_edges = 64,
      .stage2_cross_edges_stage1_home = 20,
      .stage2_cross_edges_final_home = 18,
      .pressure_yields = 4,
      .stage2_batches = 2,
      .stage2_batched_items = 8,
      .stage2_graph_read_waves = 5,
      .stage2_graph_unique_reads = 40,
      .stage2_graph_prefetch_issued = 10,
      .stage2_graph_prefetch_hits = 6,
      .stage2_graph_prefetch_wasted = 2,
      .stage2_vector_read_waves = 6,
      .stage2_vector_unique_reads = 60,
    };
    for (size_t phase = 0;
         phase < tools::breakdown_benchmark::kStage2TimingPhaseCount;
         ++phase) {
      begin[shard]->stage2_phase_attempts[phase] = 1;
      begin[shard]->stage2_phase_task_attempts[phase] = 2;
      begin[shard]->stage2_phase_elapsed_ns[phase] =
        static_cast<uint64_t>(phase + 1) * 2'000;
    }
    begin[shard]->maintenance_worker_idle_waits = 3;
    begin[shard]->maintenance_worker_idle_ns = 4'000;
    begin[shard]->maintenance_lost_wake_avoided = 5;
    begin[shard]->maintenance_targeted_wakes = 6;
    begin[shard]->maintenance_generic_wakes = 7;
    begin[shard]->maintenance_broadcast_wakes = 8;
    begin[shard]->maintenance_context_slots_scanned = 9;
    begin[shard]->packing_target_batch = 8;
    begin[shard]->packing_arrival_interval_us = 1'000;
    begin[shard]->packing_waited_batches = 3;
    begin[shard]->packing_wait_ns = 4'000;
    begin[shard]->completion_incomplete = 1;
    begin[shard]->completion_logical_full_failures = 5;
    begin[shard]->completion_physical_full_failures = 1;
    begin[shard]->physical_stage1_items = 2;
    begin[shard]->physical_stage1_total_ns = 2'000;
    begin[shard]->physical_stage1_search_ns = 1'000;
    begin[shard]->physical_stage1_prune_ns = 500;
    begin[shard]->physical_stage1_allocate_write_ns = 300;
    begin[shard]->physical_stage1_backlink_ns = 200;
    begin[shard]->physical_stage1_candidates = 256;
    begin[shard]->physical_stage1_remote_frontier_items = 1'200;
    begin[shard]->physical_stage1_neighbors = 96;
    begin[shard]->stage2_home_rpc_batches = 5;
    begin[shard]->stage2_home_rpc_items = 24;
    begin[shard]->stage2_home_scored_neighbors = 12;
    begin[shard]->stage2_home_score_rpc_batches = 2;
    begin[shard]->stage2_home_score_rpc_items = 16;
    begin[shard]->stage2_home_score_rpc_queries = 4;
    begin[shard]->stage2_home_score_rpc_request_bytes = 2'000;
    begin[shard]->stage2_home_score_rpc_response_bytes = 1'000;
    begin[shard]->active_stage2_contexts = 30 + shard;
    begin[shard]->active_stage2_context_limit = 32;
    begin[shard]->stage2_finalize_latency_ns = 1'000;
    begin[shard]->exact_insert_items = 10;
    begin[shard]->exact_insert_total_ns = 100'000;
    begin[shard]->exact_insert_remote_read_ns = 30'000;
    begin[shard]->exact_insert_remote_reverse_ns = 20'000;
    begin[shard]->exact_insert_search_ns = 50'000;
    begin[shard]->exact_insert_prune_ns = 20'000;
    begin[shard]->exact_insert_allocate_write_ns = 10'000;
    begin[shard]->exact_insert_local_reverse_ns = 5'000;
    end[shard] = *begin[shard];
    auto& latest = *end[shard];
    latest.sequence = 4;
    latest.published_steady_ns = 20'000'000'000ull;
    latest.stage2_enqueued = 30;
    latest.stage2_finalized_live = 28;
    latest.remaining = 2;
    latest.failed = 4;
    latest.peer_reverse_failed = 8;
    latest.completion_outstanding = 2;
    latest.max_backlog = 7;
    latest.stage1_search_budget_exhausted = 3;
    latest.stage2_search_budget_exhausted = 5;
    latest.stage2_continuations = 28;
    latest.stage2_remote_frontier_items = 280;
    latest.stage2_remote_expansions = 84;
    latest.stage2_scored_candidates = 2800;
    latest.stage2_migrations = 3;
    latest.stage2_final_edges = 224;
    latest.stage2_cross_edges_stage1_home = 70;
    latest.stage2_cross_edges_final_home = 60;
    latest.pressure_yields = 14;
    latest.stage2_batches = 7;
    latest.stage2_batched_items = 28;
    latest.stage2_graph_read_waves = 15;
    latest.stage2_graph_unique_reads = 120;
    latest.stage2_graph_prefetch_issued += 100;
    latest.stage2_graph_prefetch_hits += 70;
    latest.stage2_graph_prefetch_wasted += 20;
    latest.completion_incomplete = 1;
    latest.completion_logical_full_failures += 11;
    latest.completion_physical_full_failures += 2;
    latest.stage2_vector_read_waves = 18;
    latest.stage2_vector_unique_reads = 180;
    for (size_t phase = 0;
         phase < tools::breakdown_benchmark::kStage2TimingPhaseCount;
         ++phase) {
      latest.stage2_phase_attempts[phase] += 10;
      latest.stage2_phase_task_attempts[phase] += 20;
      latest.stage2_phase_elapsed_ns[phase] +=
        static_cast<uint64_t>(phase + 1) * 20'000;
    }
    latest.maintenance_worker_idle_waits += 30;
    latest.maintenance_worker_idle_ns += 40'000;
    latest.maintenance_lost_wake_avoided += 50;
    latest.maintenance_targeted_wakes += 60;
    latest.maintenance_generic_wakes += 70;
    latest.maintenance_broadcast_wakes += 80;
    latest.maintenance_context_slots_scanned += 90;
    latest.packing_target_batch = 8;
    latest.packing_arrival_interval_us = 2'000;
    latest.packing_waited_batches += 10;
    latest.packing_wait_ns += 20'000;
    latest.packing_target_flushes += 8;
    latest.packing_deadline_flushes += 2;
    latest.physical_stage1_items += 20;
    latest.physical_stage1_total_ns += 20'000;
    latest.physical_stage1_search_ns += 10'000;
    latest.physical_stage1_prune_ns += 5'000;
    latest.physical_stage1_allocate_write_ns += 3'000;
    latest.physical_stage1_backlink_ns += 2'000;
    latest.physical_stage1_candidates += 2'560;
    latest.physical_stage1_remote_frontier_items += 12'000;
    latest.physical_stage1_neighbors += 960;
    latest.stage2_home_rpc_batches += 50;
    latest.stage2_home_rpc_items += 240;
    latest.stage2_home_scored_neighbors += 120;
    latest.stage2_home_score_rpc_batches += 20;
    latest.stage2_home_score_rpc_items += 160;
    latest.stage2_home_score_rpc_queries += 40;
    latest.stage2_home_score_rpc_request_bytes += 20'000;
    latest.stage2_home_score_rpc_response_bytes += 10'000;
    latest.stage2_delay_histogram[8] = 20;
    latest.active_stage2_contexts = 30 + shard;
    latest.active_stage2_context_limit = 32;
    latest.stage2_finalize_latency_ns += 200'000;
    latest.exact_insert_items += 20;
    latest.exact_insert_total_ns += 200'000;
    latest.exact_insert_remote_read_ns += 60'000;
    latest.exact_insert_remote_reverse_ns += 40'000;
    latest.exact_insert_search_ns += 100'000;
    latest.exact_insert_prune_ns += 40'000;
    latest.exact_insert_allocate_write_ns += 20'000;
    latest.exact_insert_local_reverse_ns += 10'000;
  }
  const auto summary = tools::breakdown_benchmark::
    summarize_maintenance_snapshot_window(begin, end);
  assert(summary.requested_logs == 2);
  assert(summary.readable_logs == 2);
  assert(summary.logs_with_observations == 2);
  assert(summary.observations == 4);
  assert(summary.remaining == 4);
  assert(summary.max_backlog_observed == 7);
  assert(summary.failures == 2);
  assert(summary.peer_reverse_retry_attempts == 6);
  assert(summary.failure_delta_available);
  assert(summary.peer_reverse_retry_delta_available);
  assert(summary.completion_window_available);
  assert(summary.admission_window == 256);
  assert(summary.completion_outstanding == 4);
  assert(summary.exact_completion_credit_available);
  assert(summary.completion_incomplete == 2);
  assert(summary.max_completion_incomplete_per_shard == 1);
  assert(summary.active_stage2_contexts_latest_sum == 61);
  assert(summary.active_stage2_context_limit_sum == 64);
  assert(summary.completion_admission_failure_delta_available);
  assert(summary.completion_logical_full_failures == 22);
  assert(summary.completion_physical_full_failures == 4);
  assert(summary.locality_delta_available);
  assert(summary.stage2_finalized_live == 40);
  assert(summary.stage2_latency_sum_delta_available);
  assert(summary.stage2_finalize_latency_ns == 400'000);
  assert(summary.exact_insert_counter_delta_available);
  assert(summary.exact_insert_items == 40);
  assert(summary.exact_insert_total_ns == 400'000);
  assert(summary.exact_insert_remote_read_ns == 120'000);
  assert(summary.exact_insert_remote_reverse_ns == 80'000);
  assert(summary.stage2_remote_expansions == 120);
  assert(summary.stage2_scored_candidates == 4000);
  assert(summary.stage2_migrations == 4);
  assert(summary.stage2_cross_edges_stage1_home == 100);
  assert(summary.stage2_cross_edges_final_home == 84);
  assert(summary.search_budget_delta_available);
  assert(summary.stage1_search_budget_exhausted == 4);
  assert(summary.stage2_search_budget_exhausted == 6);
  assert(summary.execution_counter_delta_available);
  assert(summary.pressure_yields == 20);
  assert(summary.stage2_batches == 10);
  assert(summary.stage2_batched_items == 40);
  assert(summary.stage2_graph_read_waves == 20);
  assert(summary.stage2_graph_unique_reads == 160);
  assert(summary.stage2_graph_prefetch_issued == 200);
  assert(summary.stage2_graph_prefetch_hits == 140);
  assert(summary.stage2_graph_prefetch_wasted == 40);
  assert(summary.stage2_vector_read_waves == 24);
  assert(summary.stage2_vector_unique_reads == 240);
  assert(summary.home_rpc_wire_counter_delta_available);
  assert(summary.stage2_home_rpc_batches == 100);
  assert(summary.stage2_home_rpc_items == 480);
  assert(summary.stage2_home_scored_neighbors == 240);
  assert(summary.score_rpc_wire_counter_delta_available);
  assert(summary.stage2_home_score_rpc_batches == 40);
  assert(summary.stage2_home_score_rpc_items == 320);
  assert(summary.stage2_home_score_rpc_queries == 80);
  assert(summary.stage2_home_score_rpc_request_bytes == 40'000);
  assert(summary.stage2_home_score_rpc_response_bytes == 20'000);
  assert(summary.timing_counter_delta_available);
  assert(summary.logs_with_timing_counter_deltas == 2);
  assert(summary.stage2_phase_attempts[0] == 20);
  assert(summary.stage2_phase_task_attempts[0] == 40);
  assert(summary.stage2_phase_elapsed_ns[0] == 40'000);
  assert(summary.stage2_phase_elapsed_ns[5] == 240'000);
  assert(summary.maintenance_worker_idle_waits == 60);
  assert(summary.maintenance_worker_idle_ns == 80'000);
  assert(summary.maintenance_lost_wake_avoided == 100);
  assert(summary.wake_counter_delta_available);
  assert(summary.logs_with_wake_counter_deltas == 2);
  assert(summary.maintenance_targeted_wakes == 120);
  assert(summary.maintenance_generic_wakes == 140);
  assert(summary.maintenance_broadcast_wakes == 160);
  assert(summary.maintenance_context_slots_scanned == 180);
  assert(summary.packing_delta_available);
  assert(summary.packing_target_batch_max == 8);
  assert(summary.packing_arrival_interval_us_max == 2'000);
  assert(summary.packing_waited_batches == 20);
  assert(summary.packing_wait_ns == 40'000);
  assert(summary.packing_target_flushes == 16);
  assert(summary.packing_deadline_flushes == 4);
  assert(summary.physical_stage1_items == 40);
  assert(summary.physical_stage1_total_ns == 40'000);
  assert(summary.physical_stage1_search_ns == 20'000);
  assert(summary.physical_stage1_prune_ns == 10'000);
  assert(summary.physical_stage1_allocate_write_ns == 6'000);
  assert(summary.physical_stage1_backlink_ns == 4'000);
  assert(summary.physical_stage1_candidates == 5'120);
  assert(summary.physical_stage1_remote_frontier_items == 24'000);
  assert(summary.physical_stage1_neighbors == 1'920);
  assert(summary.p99_stage2_delay_available);
  assert(summary.p99_stage2_delay_samples == 40);
  assert(summary.p99_stage2_delay_upper_ms == 256.0);
  assert(summary.backlog_slope_available);
  assert(summary.backlog_slope_per_sec == 0.0);
}

}  // namespace

int main() {
  test_deterministic_dataset();
  test_vector_file_reader();
  test_single_pass_stream();
  test_recall_and_report_formatting();
  test_dynamic_cache_telemetry_reset_preserves_lifetime_gauges();
  test_base_only_recall_filter();
  test_maintenance_log_window();
  test_in_band_maintenance_snapshot_window();
  return 0;
}
