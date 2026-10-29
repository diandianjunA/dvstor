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
  // Shard export writes every shard file in parallel by default; on
  // disk-constrained builders a small cap bounds the local footprint because
  // only this many shard files grow concurrently.
  u32 export_threads{0};
  u32 R{64};
  u32 beam_width{128};
  f64 alpha{1.2};
  str vector_data_type{"auto"};
  str partition_strategy{"balanced"};
  u32 partition_max_degree{16};
  u64 partition_kmeans_samples{1000000};
  u32 partition_kmeans_iters{10};
  double partition_imbalance{1.03};
  bool skip_sanity_check{false};
  bool preflight_only{false};
  i32 seed{1234};
  // Target upper bound for one reverse-edge consolidation chunk's CSR edge
  // array in bytes; 0 = automatic (32 GiB).  More chunks bound the peak
  // memory of bulk consolidation at the cost of extra neighbor-array scans.
  size_t consolidation_chunk_bytes{0};
  size_t max_vectors{std::numeric_limits<u32>::max()};
  // Empty = checkpointing disabled. When set, SIGUSR1 requests a pause: the
  // builder commits an on-disk checkpoint at the next segment boundary and
  // exits with code 75; relaunching with the same flags resumes from it.
  filepath_t checkpoint_prefix{};
  size_t checkpoint_segment_nodes{8ull << 20};
  size_t checkpoint_interval_segments{0};
};

filepath_t default_vamana_prefix(const filepath_t& data_path, u32 R, u32 beam_width);
VamanaBuildConfig parse_configuration(int argc, char** argv);

}  // namespace tools::vamana_offline
