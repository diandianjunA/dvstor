#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tools/breakdown_benchmark/args.hh"
#include "tools/breakdown_benchmark/progress.hh"

namespace {

tools::breakdown_benchmark::Args parse(std::vector<std::string> values) {
  std::vector<char*> argv;
  argv.reserve(values.size());
  for (auto& value : values) argv.push_back(value.data());
  return tools::breakdown_benchmark::parse_args(
    static_cast<int>(argv.size()), argv.data());
}

void test_rate_limited_args(const std::string& config_path) {
  const auto args = parse({
    "benchmark",
    "--service-config", config_path,
    "--profile-name", "04_gpu_persistent_gpunetio_baseline",
    "--system-variant-label", "baseline",
    "--workload", "mixed",
    "--warmup-seconds", "60",
    "--measure-seconds", "900",
    "--client-threads", "64",
    "--mixed-mode", "rate_limited",
    "--target-query-qps", "5000",
    "--target-insert-qps", "1000",
    "--performance-query-file", "queries.u8bin",
    "--write-insert-ratio", "1",
    "--write-upsert-ratio", "0",
    "--write-delete-ratio", "0",
    "--report-json", "report.json",
  });
  assert(args.client_threads == 64);
  assert(args.profile_name == "04_gpu_persistent_gpunetio_baseline");
  assert(args.system_variant_label == "baseline");
  assert(args.mixed_mode == "rate_limited");
  assert(args.target_query_qps == 5000.0);
  assert(args.target_write_qps == 1000.0);
}

void test_write_only_mixed_args(const std::string& config_path) {
  const auto args = parse({
    "benchmark",
    "--service-config", config_path,
    "--workload", "mixed",
    "--read-ratio", "0",
    "--mixed-mode", "fixed_threads",
    "--client-threads", "24",
    "--report-json", "report.json",
  });
  assert(args.read_ratio == 0.0);
  assert(args.performance_query_file.empty());
}

void test_removed_threshold_args_are_rejected(const std::string& config_path) {
  bool threw = false;
  try {
    (void)parse({
      "benchmark",
      "--service-config", config_path,
      "--workload", "insert",
      "--min-insert-qps", "-0.5",
      "--report-json", "report.json",
    });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)parse({
      "benchmark",
      "--service-config", config_path,
      "--workload", "insert",
      "--max-stage2-p99-ms", "5000",
      "--report-json", "report.json",
    });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)parse({
      "benchmark",
      "--service-config", config_path,
      "--workload", "mixed",
      "--warmup-seconds", "1",
      "--measure-seconds", "1",
      "--performance-query-file", "queries.u8bin",
      "--query-baseline-qps", "5000",
      "--min-query-baseline-ratio", "0.9",
      "--report-json", "report.json",
    });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
}

void test_base_only_recall_args(const std::string& config_path) {
  const auto args = parse({
    "benchmark",
    "--service-config", config_path,
    "--workload", "mixed",
    "--warmup-seconds", "1",
    "--measure-seconds", "1",
    "--performance-query-file", "queries.u8bin",
    "--recall-query-file", "recall.u8bin",
    "--groundtruth-file", "truth.bin",
    "--recall-mode", "base_only",
    "--recall-base-id-limit", "100000000",
    "--report-json", "report.json",
  });
  assert(args.recall_mode == "base_only");
  assert(args.recall_base_id_limit == 100000000u);
}

void test_paced_dispatcher() {
  using namespace std::chrono_literals;
  using tools::breakdown_benchmark::PacedOperationDispatcher;
  using tools::breakdown_benchmark::PacedOperationKind;

  assert(PacedOperationDispatcher::scheduled_count(5000.0, 900) == 4'500'000);
  assert(PacedOperationDispatcher::scheduled_count(1000.0, 900) == 900'000);

  PacedOperationDispatcher dispatcher(30.0, 20.0);
  const auto start = std::chrono::steady_clock::now() + 5ms;
  dispatcher.start(start, start + 90ms);
  size_t queries = 0;
  size_t writes = 0;
  while (const auto claim = dispatcher.claim()) {
    if (claim->kind == PacedOperationKind::query) {
      ++queries;
    } else {
      ++writes;
    }
  }
  assert(queries == 3);
  assert(writes == 2);
}

void test_progress_deadline_records_zero_tail_and_excludes_drain() {
  using namespace std::chrono_literals;
  using tools::breakdown_benchmark::ProgressReporter;

  std::atomic<size_t> completed{1};
  std::atomic<size_t> writes{1};
  ProgressReporter reporter(
    "deadline-sampling", completed, 0, 1, nullptr, &writes, 250ms);

  const auto wait_deadline = std::chrono::steady_clock::now() + 2s;
  while (reporter.samples().size() < 4 &&
         std::chrono::steady_clock::now() < wait_deadline) {
    std::this_thread::sleep_for(10ms);
  }
  const auto at_deadline = reporter.samples();
  assert(at_deadline.size() == 4);
  assert(at_deadline.front().interval_ops == 1);
  assert(at_deadline.back().elapsed_seconds == 1.0);
  assert(at_deadline.back().interval_ops == 0);
  assert(at_deadline.back().interval_writes == 0);

  // A synchronous operation completing while the caller drains after the
  // deadline must affect aggregate completion counters, not stability windows.
  completed.store(2, std::memory_order_relaxed);
  writes.store(2, std::memory_order_relaxed);
  reporter.finish();
  const auto after_drain = reporter.samples();
  assert(after_drain.size() == at_deadline.size());
  assert(after_drain.back().completed_ops == 1);
}

}  // namespace

int main() {
  const auto config_path = std::filesystem::temp_directory_path() /
    "dvstor_breakdown_benchmark_control_test.ini";
  {
    std::ofstream config(config_path, std::ios::trunc);
    config << "insert-start-id = 100000000\n";
  }
  test_rate_limited_args(config_path.string());
  test_write_only_mixed_args(config_path.string());
  test_removed_threshold_args_are_rejected(config_path.string());
  test_base_only_recall_args(config_path.string());
  test_paced_dispatcher();
  test_progress_deadline_records_zero_tail_and_excludes_drain();
  std::filesystem::remove(config_path);
  return 0;
}
