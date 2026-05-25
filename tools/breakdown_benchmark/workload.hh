#pragma once

#include "nlohmann/json.hh"
#include "service/compute_service.hh"
#include "tools/breakdown_benchmark/args.hh"

namespace tools::breakdown_benchmark {

template <class Distance>
nlohmann::json run_benchmark(ComputeService<Distance>& service, const Args& args);

}  // namespace tools::breakdown_benchmark
