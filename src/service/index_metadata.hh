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
  str storage_format{};
  u32 graph_hot_bytes{};
  u32 vector_offset{};
  u32 neighbors_offset{};
  u32 rabitq_offset{};
  VectorDType vector_dtype{VectorDType::float32};
  u32 vector_component_size{sizeof(element_t)};
  u32 vector_bytes{};
  u32 rabitq_code_bits{};
  u32 rabitq_entry_size{};
  u32 rabitq_cache_bits{};
  u32 rabitq_cache_entry_size{};
  f32 rabitq_cache_norm_min{};
  f32 rabitq_cache_norm_max{};
  f32 rabitq_cache_error_min{};
  f32 rabitq_cache_error_max{};
  vec<float> rabitq_centroid;
  u32 hot_graph_entry_size{};
  u32 hot_graph_pointer_bytes{};
  u32 hot_graph_shard_bits{};
  vec<u64> hot_graph_offsets;
  vec<u64> hot_graph_entry_counts;
  vec<u64> hot_graph_dynamic_base_offsets;
  u32 hot_graph_dynamic_record_bytes{};
  u32 hot_graph_dynamic_hot_offset{};
  u32 allocation_size{};
  str idmap_format{};
};

bool load_metadata(const filepath_t& index_prefix, Metadata& metadata, str* error_message = nullptr);

}  // namespace service::index_metadata
