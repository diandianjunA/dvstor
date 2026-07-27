#pragma once

#include "nlohmann/json.hh"
#include "service/compute_service.hh"
#include "tools/breakdown_benchmark/args.hh"

namespace tools::breakdown_benchmark {

nlohmann::json run_benchmark(ComputeService& service, const Args& args);

}  // namespace tools::breakdown_benchmark
