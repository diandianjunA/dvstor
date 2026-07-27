#pragma once

#include "common/types.hh"

namespace tools::vamana_offline {

struct GraphExtentIndexOptions {
  filepath_t index_prefix;
  filepath_t output;
  u32 chunk_records{65'536};
  bool overwrite{};
};

struct GraphExtentIndexResult {
  filepath_t output;
  u64 node_count{};
  u64 payload_bytes{};
  u64 payload_checksum{};
  u64 graph_bytes_validated{};
  u32 maximum_class{};
};

// Validate every immutable base graph record in a schema-16 index and emit one
// extent class per global physical ordinal. Shard payloads are concatenated in
// metadata shard order, exactly matching NavigationLayout::ordinal_base.
GraphExtentIndexResult build_graph_extent_index(
  const GraphExtentIndexOptions& options);

}  // namespace tools::vamana_offline
