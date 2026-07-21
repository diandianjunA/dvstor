#pragma once

#include "common/types.hh"
#include "common/vector_dtype.hh"

namespace service::index_metadata {

struct Metadata {
  u32 schema_version{16};
  u32 dim{};
  u32 R{};
  u32 beam_width_construction{};
  u32 partition_max_degree{};
  double partition_cross_shard_ratio{};
  u32 num_memory_nodes{};
  u32 node_size{};
  str node_layout{"plain"};
  str storage_format{};
  u32 graph_hot_bytes{};
  u32 vector_offset{};
  u32 slot_incarnation_offset{};
  str remote_ptr_format{};
  VectorDType vector_dtype{VectorDType::float32};
  u32 vector_component_size{sizeof(element_t)};
  u32 vector_bytes{};
  str navigation_quantizer{};
  u32 navigation_code_bytes{};
  u32 pq_subquantizers{};
  u32 pq_bits{};
  u64 navigation_model_checksum{};
  u32 hot_graph_entry_size{};
  u32 hot_graph_pointer_bytes{};
  u32 hot_graph_shard_bits{};
  vec<u64> hot_graph_offsets;
  vec<u64> hot_graph_entry_counts;
  vec<u64> hot_graph_dynamic_base_offsets;
  vec<u64> storage_control_remote_offsets;
  vec<u64> dynamic_node_base_offsets;
  u32 hot_graph_dynamic_record_bytes{};
  u32 hot_graph_dynamic_hot_offset{};
  u32 dynamic_navigation_code_offset{};
  u32 dynamic_navigation_code_validation_bytes{};
  u32 allocation_size{};
  str idmap_format{};
  str centroid_state_format{};
  u64 index_build_fingerprint{};
  vec<u64> shard_build_fingerprints;
  str navigation_format{};
  vec<u64> navigation_code_remote_offsets;
  vec<u64> navigation_code_region_bytes;
};

bool load_metadata(const filepath_t& index_prefix, Metadata& metadata, str* error_message = nullptr);

}  // namespace service::index_metadata
