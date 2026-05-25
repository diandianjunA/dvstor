#pragma once

#include <limits>

#include "common/types.hh"

namespace tools::vamana_offline {

struct VamanaBuildConfig {
  filepath_t data_path{};

filepath_t default_vamana_prefix(const filepath_t& data_path, u32 R, u32 beam_width);
VamanaBuildConfig parse_configuration(int argc, char** argv);

}  // namespace tools::vamana_offline
