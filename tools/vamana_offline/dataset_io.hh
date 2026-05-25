#pragma once

#include "tools/vamana_offline/config.hh"

namespace tools::vamana_offline {

struct Dataset {
  filepath_t source_file{};
  u32 dim{0};
  size_t total_vectors{0};
  vec<element_t> vectors;
  vec<node_t> ids;

  const float* vector(size_t i) const { return vectors.data() + i * dim; }
};

filepath_t resolve_dataset_file(const filepath_t& input_path);
Dataset read_dataset(const VamanaBuildConfig& config);

}  // namespace tools::vamana_offline
