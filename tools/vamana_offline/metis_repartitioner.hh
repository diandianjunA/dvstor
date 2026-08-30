#pragma once

#include "common/types.hh"

namespace tools::vamana_offline {

struct MetisRepartitionOptions {
  filepath_t input_prefix;
  filepath_t output_prefix;
  filepath_t data_path;
  filepath_t reuse_model;
  u32 partition_max_degree{};
  f64 partition_imbalance{};
  u32 threads{16};
  u32 pq_chunk_vectors{32768};
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

// Reconstruct the immutable global graph from a complete schema-16 balanced
// index, compute a fresh METIS placement, and publish a new independently
// bound schema-16 index. Input and output prefixes must differ. The source is
// never modified.
MetisRepartitionResult
repartition_schema16_index(const MetisRepartitionOptions &options);

} // namespace tools::vamana_offline
