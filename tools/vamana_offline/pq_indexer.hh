#pragma once

#include "common/types.hh"

namespace tools::vamana_offline {

struct PqIndexOptions {
  filepath_t index_prefix;
  filepath_t reuse_model;
  // When set, training vectors are sampled from this raw bin file
  // ([u32 rows][u32 dim][data]) instead of the index shard files.  This
  // permits training on a machine that holds the original dataset but not
  // every index shard.
  filepath_t train_data_path;
  // 0-based shard ordinals to encode locally; empty encodes every shard.
  // Shard preflight and output publication are restricted to these shards so
  // re-encoding can run shard-locally on distributed storage nodes.
  vec<u32> only_shards;
  // Train and publish only the PQ model; no shard codes are encoded.
  bool skip_encode{};
  u32 subquantizers{16};
  u32 train_samples{262144};
  u32 opq_iterations{20};
  u32 pq_iterations{25};
  u32 chunk_vectors{32768};
  u32 threads{};
  u64 seed{1234};
  bool overwrite{};
  // Skip OPQ rotation training (plain PQ).  Full-dimension OPQ needs a
  // d x d covariance eigendecomposition per outer iteration, which is
  // infeasible at very high dimensions; on i.i.d. random data the learned
  // rotation converges to a near-identity map anyway.
  bool no_opq{};
};

struct PqIndexResult {
  filepath_t model_file;
  vec<filepath_t> code_files;
  u64 model_checksum{};
  u64 node_count{};
  u64 code_bytes{};
};

PqIndexResult build_pq_index(const PqIndexOptions& options);

}  // namespace tools::vamana_offline
