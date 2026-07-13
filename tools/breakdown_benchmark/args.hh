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
  std::string recall_query_file;
  std::string performance_query_file;
  std::string insert_file;
  std::string groundtruth_file;
  size_t recall_queries{1000};
  uint32_t recall_k{0};
  double min_recall{-1.0};
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
