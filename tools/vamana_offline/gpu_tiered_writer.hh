#pragma once

#include "gpu_search/index_format.hh"
#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/partitioning.hh"

namespace tools::vamana_offline {

struct GpuTieredWriteResult {
  filepath_t index_file;
  vec<u64> graph_page_offsets;
  vec<u64> graph_page_bytes;
  u32 hot_degree{};
  u32 entry_points{};
  u32 page_bytes{};
};

GpuTieredWriteResult write_gpu_tiered_index(
  const VamanaGraph& graph,
  const Dataset& dataset,
  const VamanaBuildConfig& config,
  const vec<NodePlacement>& placements,
  const vec<u64>& shard_file_bytes,
  const filepath_t& output_prefix);

}  // namespace tools::vamana_offline
