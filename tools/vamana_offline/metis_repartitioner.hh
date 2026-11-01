#pragma once

#include "common/types.hh"

namespace tools::vamana_offline {

struct MetisRepartitionOptions {
  filepath_t input_prefix;
  filepath_t output_prefix;
  filepath_t data_path;
  filepath_t reuse_model;
  // Source index partition strategy the input metadata must declare.
  str input_strategy{"balanced"};
  // Partition strategy used to re-place nodes in the output index.
  str output_strategy{"metis"};
  // Lloyd kmeans tuning, only consumed when output_strategy == "kmeans".
  u64 kmeans_samples{1000000};
  u32 kmeans_iters{10};
  u32 partition_max_degree{};
  f64 partition_imbalance{};
  u32 threads{16};
  u32 pq_chunk_vectors{32768};
  // When non-empty, pause after the source shards, graph, and dataset are
  // fully resident in memory and poll every 5s until this flag file exists
  // before writing any output shard.
  filepath_t pause_after_load;
  bool graph_only{};
  bool validate_only{};
};

struct MetisRepartitionResult {
  u64 node_count{};
  u64 edge_count{};
  u32 shards{};
  u64 source_build_fingerprint{};
  u64 output_build_fingerprint{};
  bool graph_written{};
  bool pq_built{};
  bool extent_built{};
  bool resumed{};
  filepath_t metadata_file;
};

// Reconstruct the immutable global graph from a complete schema-16 index,
// compute a fresh placement with the requested output strategy (METIS or
// kmeans), and publish a new independently bound schema-16 index. Input and
// output prefixes must differ. The source is never modified.
MetisRepartitionResult
repartition_schema16_index(const MetisRepartitionOptions &options);

} // namespace tools::vamana_offline
