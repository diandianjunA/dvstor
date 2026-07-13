#pragma once

#include "common/types.hh"

namespace tools::vamana_offline {

struct PqIndexOptions {
  filepath_t index_prefix;
  filepath_t reuse_model;
  u32 subquantizers{16};
  u32 train_samples{262144};
  u32 opq_iterations{20};
  u32 pq_iterations{25};
  u32 chunk_vectors{32768};
  u32 entry_points{256};
  u32 threads{};
  u64 seed{1234};
  bool overwrite{};
  bool upgrade_layout_only{};
  u32 local_shard{};
};

struct PqIndexResult {
  filepath_t model_file;
  vec<filepath_t> code_files;
  u64 model_checksum{};
  u64 node_count{};
  u64 code_bytes{};
};

PqIndexResult build_pq_index(const PqIndexOptions& options);
PqIndexResult upgrade_pq_layout(const PqIndexOptions& options);

}  // namespace tools::vamana_offline
