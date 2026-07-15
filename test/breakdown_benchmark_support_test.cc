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
  telemetry.delta_reclaim_batches = 7;
  telemetry.mutation_capacity_wait_events = 3;
  telemetry.mutation_capacity_wait_ns = 2'000'000;
  const auto telemetry_json = tools::breakdown_benchmark::telemetry_to_json(telemetry);
  assert(telemetry_json.at("delta_reclaim_batches") == 7);
  assert(telemetry_json.at("mutation_capacity_wait_events") == 3);
  assert(telemetry_json.at("mutation_capacity_wait_ms") == 2.0);

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
  service::breakdown::Report report;
  report.query.operation = service::breakdown::Operation::query;
  report.query.count = 1;
  report.query.end_to_end_latencies_ns.push_back(1'000'000);
  const auto formatted = tools::breakdown_benchmark::format_report(root, report);
  assert(formatted.bottleneck_summary.contains("query"));
  assert(formatted.text.find("single_pass_no_reuse") != std::string::npos);
  assert(formatted.text.find("query breakdown") != std::string::npos);
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
           << "stitch_enqueued=" << enqueued << ' '
           << "stitched_live=" << completed << " stale=0 "
           << "remaining=" << remaining << ' '
           << "peer_reverse_remaining=0 "
           << "failed=" << failed << ' '
           << "peer_reverse_failed=" << peer_failed << ' '
           // Deliberately stale cumulative p99 fields: the parser must use
           // only the histogram delta from the cursor baseline.
           << "p99_stitch_delay_upper_ms=30000 "
           << "p99_stitch_delay_over_30s=true "
           << "stitch_delay_histogram=" << histogram_text(histogram)
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
    write_observation(output, 300, 300, 0, 10, 3, histogram);
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
  assert(summary.failure_delta_available);
  assert(summary.p99_stitch_delay_available);
  assert(summary.p99_stitch_delay_samples == 100);
  assert(summary.p99_stitch_delay_upper_ms == 4.0);
  assert(!summary.p99_stitch_delay_over_30s);
  assert(summary.backlog_slope_available);
  assert(summary.backlog_slope_per_sec < 0.0);

  // A new post-stop cursor has no observation yet. Its default remaining=0
  // must not be mistaken for a completed drain.
  const auto empty_post_stop =
    tools::breakdown_benchmark::summarize_maintenance_logs(post_stop);
  assert(empty_post_stop.logs_with_observations == 0);
  assert(!empty_post_stop.failure_delta_available);
  assert(!empty_post_stop.p99_stitch_delay_available);

  {
    std::ofstream output(path, std::ios::app);
    write_observation(output, 303, 300, 3, 10, 3, histogram);
  }
  const auto post_stop_without_completion =
    tools::breakdown_benchmark::summarize_maintenance_logs(post_stop);
  assert(post_stop_without_completion.logs_with_observations == 1);
  assert(post_stop_without_completion.failure_delta_available);
  assert(post_stop_without_completion.failures == 0);
  assert(post_stop_without_completion.p99_stitch_delay_samples == 0);
  assert(!post_stop_without_completion.p99_stitch_delay_available);

  // Appending drain observations cannot change the frozen load window.
  const auto frozen_again = tools::breakdown_benchmark::
    summarize_maintenance_log_window(load_begin, post_stop);
  assert(frozen_again.observations == 3);
  assert(frozen_again.backlog_slope_per_sec == summary.backlog_slope_per_sec);

  // The 5-second SLO needs its own bucket. A p99 at exactly this boundary
  // passes as 5000 ms, while the next bucket remains conservatively 8000 ms.
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
  assert(exact_5s.p99_stitch_delay_available);
  assert(exact_5s.p99_stitch_delay_upper_ms == 5000.0);

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
  assert(over_5s.p99_stitch_delay_available);
  assert(over_5s.p99_stitch_delay_upper_ms == 8000.0);

  std::filesystem::remove(path);
}

}  // namespace

int main() {
  test_deterministic_dataset();
  test_vector_file_reader();
  test_single_pass_stream();
  test_recall_and_report_formatting();
  test_base_only_recall_filter();
  test_maintenance_log_window();
  return 0;
}
