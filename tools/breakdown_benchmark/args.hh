#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.hh"

namespace tools::breakdown_benchmark {

using ConfigMap = std::unordered_map<std::string, std::string>;

struct Args {
  std::string service_config_path;
  std::string workload{"both"};
  size_t warmup_ops{100};
  size_t measure_ops{1000};
  size_t warmup_seconds{0};
  size_t measure_seconds{0};
  size_t client_threads{4};
  double read_ratio{0.5};
  std::string mixed_mode{"probability"};
  double target_query_qps{};
  double target_write_qps{};
  std::string recall_query_file;
  std::string performance_query_file;
  std::string insert_file;
  std::string groundtruth_file;
  size_t recall_queries{1000};
  uint32_t recall_k{0};
  std::string recall_mode{"all"};
  uint32_t recall_base_id_limit{0};
  double min_recall{-1.0};
  double min_query_qps{-1.0};
  double min_insert_qps{-1.0};
  double min_stability_ratio{-1.0};
  double min_write_stability_ratio{-1.0};
  double query_baseline_qps{-1.0};
  std::string query_baseline_report;
  double min_query_baseline_ratio{-1.0};
  double max_recall_drop{-1.0};
  int64_t max_zero_completion_windows{-1};
  int64_t max_zero_query_windows{-1};
  int64_t max_zero_write_windows{-1};
  double max_drain_seconds{-1.0};
  double min_rate_attainment_ratio{-1.0};
  double max_gpu_visibility_ms{-1.0};
  int64_t max_final_mutation_capacity_reserved{-1};
  int64_t max_final_delta_mutable_entries{-1};
  int64_t max_late_storage_owner_rpcs{-1};
  std::vector<std::string> storage_maintenance_logs;
  double max_stage2_p99_ms{-1.0};
  double max_stage2_backlog_slope{-1.0};
  int64_t max_stage2_remaining{-1};
  size_t stage2_drain_timeout_seconds{};
  bool recall_only{false};
  bool synthetic{false};
  std::string report_json_path;
  std::string report_text_path;
  uint32_t insert_start_id{0};
  double write_insert_ratio{0.5};
  double write_upsert_ratio{0.4};
  double write_delete_ratio{0.1};
};

ConfigMap read_config(const std::string& path);
std::vector<std::string> build_service_argv(const std::string& service_config_path);
std::vector<char*> make_argv(std::vector<std::string>& args);
Args parse_args(int argc, char** argv);

}  // namespace tools::breakdown_benchmark
