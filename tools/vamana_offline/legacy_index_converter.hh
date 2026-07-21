#pragma once

#include "common/types.hh"

namespace tools::vamana_offline {

struct LegacyIndexConvertOptions {
  filepath_t input_prefix;
  filepath_t output_prefix;
  filepath_t reuse_model;
  // Zero means infer it from the legacy schema-15 metadata.
  u32 subquantizers{};
  u32 chunk_nodes{65536};
  // Compatibility spelling used by the PQ indexer CLI; zero selects
  // chunk_nodes.
  u32 chunk_vectors{};
  u32 threads{};
  bool dry_run{};
  // Leave the converted graph as the current schema-15 tagged-v2
  // intermediate instead of re-encoding PQ and committing schema 16.
  bool graph_only{};
};

struct LegacyIndexConvertResult {
  u64 node_count{};
  u64 input_bytes{};
  u64 output_bytes{};
  u32 shards{};
  u32 subquantizers{};
  filepath_t legacy_model_file;
  u64 edge_count{};
  filepath_t metadata_file;
  bool wrote_graph{};
  bool built_pq{};
};

// Converts the retired compact-v1 static base index into the tagged-v2
// schema-15 intermediate accepted by the current PQ indexer.  It preserves
// physical shard and slot order, exact vector bytes, and graph topology.  The
// function never overwrites the input or an existing output prefix.
LegacyIndexConvertResult convert_legacy_index(
  const LegacyIndexConvertOptions& options);

using LegacyIndexConverterOptions = LegacyIndexConvertOptions;
using LegacyIndexConversionResult = LegacyIndexConvertResult;

LegacyIndexConversionResult convert_legacy_schema15_index(
  const LegacyIndexConverterOptions& options);

}  // namespace tools::vamana_offline
