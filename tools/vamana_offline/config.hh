#pragma once

#include <limits>

#include "common/types.hh"
#include "common/vector_dtype.hh"

namespace tools::vamana_offline {

struct VamanaBuildConfig {
  filepath_t data_path{};
  filepath_t output_prefix{};
  filepath_t query_path{};
  filepath_t groundtruth_path{};
  u32 num_memory_nodes{1};
  u32 threads{0};
  u32 R{64};
  u32 beam_width{128};
  f64 alpha{1.2};
  str vector_data_type{"auto"};
  str partition_strategy{"balanced"};
  u32 partition_max_degree{16};
  double partition_imbalance{1.03};
  bool skip_sanity_check{false};
  bool use_rabitq{false};
  str storage_format{"vamana_compact_v1"};
  i32 seed{1234};
  size_t max_vectors{std::numeric_limits<u32>::max()};
  bool ip_distance{false};
  u32 anchor_count_per_shard{4096};
};

filepath_t default_vamana_prefix(const filepath_t& data_path, u32 R, u32 beam_width);
VamanaBuildConfig parse_configuration(int argc, char** argv);

}  // namespace tools::vamana_offline
