#pragma once

#include "tools/vamana_offline/graph.hh"

namespace tools::vamana_offline {

void repair_reverse_edges(VamanaGraph& graph, const Dataset& dataset,
                          size_t num_threads, u32 reverse_slots,
                          size_t chunk_target_bytes);

}  // namespace tools::vamana_offline
