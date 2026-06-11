#pragma once

#include "common/types.hh"
#include "common/vector_dtype.hh"

namespace service::index_metadata {

struct Metadata {
  u32 schema_version{1};
  u32 dim{};
  u32 R{};
  u32 beam_width_construction{};
  u32 num_memory_nodes{};
  u32 node_size{};
  str node_layout{"standard"};
  str storage_format{"legacy_aos"};
  u32 graph_hot_bytes{};
  u32 vector_offset{};
  u32 neighbors_offset{};
  u32 rabitq_offset{};
  VectorDType vector_dtype{VectorDType::float32};
  u32 vector_component_size{sizeof(element_t)};
  u32 vector_bytes{};
  u32 rabitq_code_bits{};
  u32 rabitq_entry_size{};
  vec<float> rabitq_centroid;
};

bool load_metadata(const filepath_t& index_prefix, Metadata& metadata, str* error_message = nullptr);

}  // namespace service::index_metadata
