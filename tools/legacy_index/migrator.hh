#pragma once

#include "common/types.hh"

namespace tools::legacy_index {

struct MigrationOptions {
  filepath_t source_prefix;
  filepath_t output_prefix;
  u32 io_threads{};
  u32 chunk_nodes{65536};
  bool overwrite{};
};

struct MigrationResult {
  filepath_t output_prefix;
  u64 node_count{};
  u64 source_bytes{};
  u64 output_bytes{};
};

MigrationResult migrate_schema13_index(const MigrationOptions& options);

}  // namespace tools::legacy_index
