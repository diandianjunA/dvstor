#pragma once

#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/partitioning.hh"
#include "tools/vamana_offline/rabitq.hh"

namespace tools::vamana_offline {

vec<NodePlacement> assign_nodes_to_shards(size_t num_vectors, u32 num_memory_nodes);
void write_vamana_shards(const VamanaGraph& graph,
                         const Dataset& dataset,
                         const VamanaBuildConfig& config,
                         const RaBitQState& rabitq_state,
                         const vec<vec<byte_t>>& rabitq_data,
                         const filepath_t& output_prefix);

}  // namespace tools::vamana_offline
