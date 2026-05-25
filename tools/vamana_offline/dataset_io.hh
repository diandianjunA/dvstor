#pragma once

#include "tools/vamana_offline/config.hh"

namespace tools::vamana_offline {

struct Dataset {
  filepath_t source_file{};

filepath_t resolve_dataset_file(const filepath_t& input_path);
Dataset read_dataset(const VamanaBuildConfig& config);

}  // namespace tools::vamana_offline
