#pragma once

#include "gpu_search/index_format.hh"
#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/partitioning.hh"

namespace tools::vamana_offline {

struct GpuTieredWriteResult {
  filepath_t index_file;
  vec<filepath_t> code_files;
  vec<u64> code_remote_offsets;
  vec<u64> code_bytes;
  u32 entry_points{};
};

GpuTieredWriteResult write_gpu_tiered_index(
  const VamanaGraph& graph,
  const Dataset& dataset,
  const VamanaBuildConfig& config,
  const vec<NodePlacement>& placements,
  const vec<u64>& shard_file_bytes,
  const filepath_t& output_prefix);

}  // namespace tools::vamana_offline
