#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "tools/breakdown_benchmark/dataset.hh"
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
  const auto telemetry_json = tools::breakdown_benchmark::telemetry_to_json(telemetry);
  assert(telemetry_json.at("delta_reclaim_batches") == 7);

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

}  // namespace

int main() {
  test_deterministic_dataset();
  test_vector_file_reader();
  test_single_pass_stream();
  test_recall_and_report_formatting();
  return 0;
}
