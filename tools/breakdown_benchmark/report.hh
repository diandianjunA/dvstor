#pragma once

#include <string>

#include "nlohmann/json.hh"

#include "gpu_search/types.hh"
#include "service/breakdown.hh"

namespace tools::breakdown_benchmark {

struct FormattedReport {
  nlohmann::json bottleneck_summary;
  std::string text;
};

nlohmann::json telemetry_to_json(const gpu_search::TelemetrySnapshot& telemetry);
FormattedReport format_report(const nlohmann::json& root,
                              const service::breakdown::Report& report);

}  // namespace tools::breakdown_benchmark
