#pragma once

#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/partitioning.hh"

namespace tools::vamana_offline {

vec<NodePlacement> assign_nodes_to_shards(size_t num_vectors, u32 num_memory_nodes);
void validate_vamana_shard_capacity(size_t num_vectors,
                                    const VamanaBuildConfig& config);
void write_vamana_shards(const VamanaGraph& graph,
                         const Dataset& dataset,
                         const VamanaBuildConfig& config,
                         const filepath_t& output_prefix);

}  // namespace tools::vamana_offline
