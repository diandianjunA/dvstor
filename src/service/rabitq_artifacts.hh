#pragma once

#include "common/types.hh"

namespace service::rabitq {

struct Artifacts {
  filepath_t index_prefix{};
  u32 dim{};
  u32 rabitq_bits{};
  u32 num_memory_nodes{};
  u32 rabitq_size{};
  u32 R{};
  u32 beam_width_construction{};
  u32 node_size{};
  str node_layout{"legacy"};
  vec<float> rotation_matrix;
  vec<float> rotated_centroid;
  double t_const{};
};

bool load_metadata(const filepath_t& index_prefix, Artifacts& artifacts, str* error_message = nullptr);
bool load_artifacts(const filepath_t& index_prefix, Artifacts& artifacts, str* error_message = nullptr);

}  // namespace service::rabitq
