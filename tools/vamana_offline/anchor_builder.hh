#pragma once

#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/partitioning.hh"

namespace tools::vamana_offline {

void write_anchor_sidecar(const VamanaGraph& graph,
                          const Dataset& dataset,
                          const VamanaBuildConfig& config,
                          const vec<NodePlacement>& placements,
                          const filepath_t& output_prefix);

}  // namespace tools::vamana_offline
