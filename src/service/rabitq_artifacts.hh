#pragma once

#include "common/types.hh"

namespace service::rabitq {

struct Artifacts {
  filepath_t index_prefix{};
  u32 dim{};
  u32 rabitq_bits{};
  u32 num_memory_nodes{};
  u32 rabitq_size{};
  vec<float> rotation_matrix;
  vec<float> rotated_centroid;
  double t_const{};
};

bool load_artifacts(const filepath_t& index_prefix, Artifacts& artifacts, str* error_message = nullptr);

}  // namespace service::rabitq
