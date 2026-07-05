#pragma once

#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/partitioning.hh"

namespace tools::vamana_offline {

void write_vamana_shards(const VamanaGraph& graph,
                         const Dataset& dataset,
                         const VamanaBuildConfig& config,
                         const filepath_t& output_prefix);

}  // namespace tools::vamana_offline
