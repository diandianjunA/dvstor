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
  telemetry.graph_read_retries = 11;
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
  const auto telemetry_json = tools::breakdown_benchmark::telemetry_to_json(telemetry);
  assert(telemetry_json.at("graph_read_retries") == 11);
  assert(telemetry_json.at("centroid_route_publications") == 7);
  assert(telemetry_json.at("centroid_route_shard_updates") == 9);
  assert(telemetry_json.at("centroid_route_live_entries") == 13);
  assert(telemetry_json.at("centroid_route_probe_reads") == 101);
  assert(telemetry_json.at("centroid_route_body_reads") == 3);
  assert(telemetry_json.at("centroid_route_unchanged_polls") == 47);
  assert(telemetry_json.at("centroid_route_poll_delay_us") == 8000);
  assert(telemetry_json.at("centroid_route_query_retries") == 23);
  assert(telemetry_json.at("centroid_route_query_timeouts") == 0);
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
    {"workload", "query"},
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
  root["stage2"] = {
    {"requested_logs", 1},
    {"failures", 0},
    {"peer_reverse_retry_attempts", 7},
    {"peer_reverse_retry_delta_available", true},
  };
  service::breakdown::Report report;
  report.query.operation = service::breakdown::Operation::query;
  report.query.count = 1;
  report.query.end_to_end_latencies_ns.push_back(1'000'000);
  const auto formatted = tools::breakdown_benchmark::format_report(root, report);
  assert(formatted.bottleneck_summary.contains("query"));
  assert(formatted.text.find("single_pass_no_reuse") != std::string::npos);
  assert(formatted.text.find("query breakdown") != std::string::npos);
  assert(formatted.text.find("failures (hard): 0") != std::string::npos);
  assert(formatted.text.find("peer_reverse_retry_attempts: 7") !=
         std::string::npos);
}

void test_dynamic_cache_telemetry_reset_preserves_lifetime_gauges() {
  gpu_search::Telemetry telemetry;
  telemetry.dynamic_code_cache_hits.store(17, std::memory_order_relaxed);
  telemetry.dynamic_code_cache_publish_successes.store(
    5, std::memory_order_relaxed);
  telemetry.dynamic_code_cache_occupied.store(123, std::memory_order_relaxed);
  telemetry.dynamic_code_cache_capacity.store(256, std::memory_order_relaxed);
  telemetry.reset();
  const auto snapshot = telemetry.snapshot();
  assert(snapshot.dynamic_code_cache_hits == 0);
  assert(snapshot.dynamic_code_cache_publish_successes == 0);
  assert(snapshot.dynamic_code_cache_occupied == 123);
  assert(snapshot.dynamic_code_cache_capacity == 256);
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
                               const Histogram& histogram) {
    output << "[STATUS]: storage-owner maintenance observation: "
           << "stage2_enqueued=" << enqueued << ' '
           << "stage2_finalized_live=" << completed << " stale=0 "
           << "remaining=" << remaining << ' '
           << "peer_reverse_remaining=0 "
           << "failed=" << failed << ' '
           << "peer_reverse_failed=" << peer_failed << ' '
           << "admission_window=512 "
           << "completion_outstanding=" << remaining << ' '
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
  assert(summary.admission_window == 512);
  assert(summary.completion_outstanding == 0);
  assert(summary.max_completion_outstanding_per_shard == 10);
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
      .stage2_vector_read_waves = 6,
      .stage2_vector_unique_reads = 60,
    };
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
    latest.stage2_vector_read_waves = 18;
    latest.stage2_vector_unique_reads = 180;
    latest.stage2_delay_histogram[8] = 20;
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
  assert(summary.locality_delta_available);
  assert(summary.stage2_finalized_live == 40);
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
  assert(summary.stage2_vector_read_waves == 24);
  assert(summary.stage2_vector_unique_reads == 240);
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
