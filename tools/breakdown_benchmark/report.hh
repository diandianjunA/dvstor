#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hh"

#include "common/types.hh"
#include "gpu_search/types.hh"
#include "service/breakdown.hh"

namespace tools::breakdown_benchmark {

struct FormattedReport {
  nlohmann::json bottleneck_summary;
  std::string text;
};

nlohmann::json telemetry_to_json(const gpu_search::TelemetrySnapshot& telemetry);
std::string normalize_path(const std::string& path);
std::vector<uint32_t> filter_base_only_recall_ids(
  const std::vector<node_t>& results,
  uint32_t base_id_limit,
  size_t result_limit);
FormattedReport format_report(const nlohmann::json& root,
                              const service::breakdown::Report& report);

}  // namespace tools::breakdown_benchmark
