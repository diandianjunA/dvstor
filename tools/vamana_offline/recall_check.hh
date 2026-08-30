#pragma once

#include "tools/vamana_offline/config.hh"
#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/graph.hh"

namespace tools::vamana_offline {

void preflight_optional_recall_inputs(
  const Dataset& dataset, const VamanaBuildConfig& config);

void run_optional_recall_check(VamanaGraph& graph,
                               const Dataset& dataset,
                               const VamanaBuildConfig& config);

}  // namespace tools::vamana_offline
