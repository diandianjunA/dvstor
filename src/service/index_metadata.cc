#include "service/index_metadata.hh"

#include <filesystem>
#include <fstream>

#include "nlohmann/json.hh"

namespace service::index_metadata {

namespace {

bool fail(str* error_message, const str& message) {
  if (error_message != nullptr) {
    *error_message = message;
  }
  return false;
}

}  // namespace

bool load_metadata(const filepath_t& index_prefix, Metadata& metadata, str* error_message) {
  const filepath_t metadata_file = filepath_t(index_prefix.string() + ".meta.json");
  std::ifstream input(metadata_file);
  if (!input.good()) {
    return fail(error_message, "missing index metadata file: " + metadata_file.string());
  }

  try {
    nlohmann::json json;
    input >> json;

    metadata.schema_version = json.value("schema_version", 1u);
    metadata.dim = json.at("dim").get<u32>();
    metadata.R = json.at("R").get<u32>();
    metadata.beam_width_construction = json.value("beam_width_construction", 0u);
    metadata.num_memory_nodes = json.at("num_memory_nodes").get<u32>();
    metadata.node_size = json.at("node_size").get<u32>();
    metadata.node_layout = json.value("node_layout", str{"plain"});
    metadata.storage_format = json.at("storage_format").get<str>();
    metadata.graph_hot_bytes = json.value("graph_hot_bytes", 0u);
    metadata.vector_offset = json.value("vector_offset", 0u);
    metadata.vector_dtype = parse_vector_dtype(json.value("vector_data_type", str{"float32"}));
    metadata.vector_component_size = json.value(
      "vector_component_size", static_cast<u32>(vector_dtype_component_size(metadata.vector_dtype)));
    metadata.vector_bytes = json.value(
      "vector_bytes", static_cast<u32>(vector_dtype_bytes(metadata.vector_dtype, metadata.dim)));
    metadata.navigation_quantizer = json.value("navigation_quantizer", str{});
    metadata.navigation_code_bytes = json.value("navigation_code_bytes", 0u);
    metadata.pq_subquantizers = json.value("pq_subquantizers", 0u);
    metadata.pq_bits = json.value("pq_bits", 0u);
    metadata.navigation_model_checksum = json.value("navigation_model_checksum", 0ull);
    metadata.hot_graph_entry_size = json.value("hot_graph_entry_size", 0u);
    metadata.hot_graph_pointer_bytes = json.value("hot_graph_pointer_bytes", 0u);
    metadata.hot_graph_shard_bits = json.value("hot_graph_shard_bits", 0u);
    if (json.contains("hot_graph_offsets")) {
      metadata.hot_graph_offsets = json["hot_graph_offsets"].get<vec<u64>>();
    }
    if (json.contains("hot_graph_entry_counts")) {
      metadata.hot_graph_entry_counts = json["hot_graph_entry_counts"].get<vec<u64>>();
    }
    if (json.contains("hot_graph_dynamic_base_offsets")) {
      metadata.hot_graph_dynamic_base_offsets =
        json["hot_graph_dynamic_base_offsets"].get<vec<u64>>();
    }
    metadata.hot_graph_dynamic_record_bytes =
      json.value("hot_graph_dynamic_record_bytes", 0u);
    metadata.hot_graph_dynamic_hot_offset =
      json.value("hot_graph_dynamic_hot_offset", 0u);
    metadata.allocation_size = json.value("allocation_size", metadata.node_size);
    metadata.idmap_format = json.value("idmap_format", str{});
    metadata.anchor_format = json.value("anchor_format", str{});
    metadata.anchor_count_per_shard = json.value("anchor_count_per_shard", 0u);
    metadata.navigation_format = json.value("navigation_format", str{});
    if (json.contains("navigation_code_remote_offsets")) {
      metadata.navigation_code_remote_offsets =
        json["navigation_code_remote_offsets"].get<vec<u64>>();
    }
    if (json.contains("navigation_code_region_bytes")) {
      metadata.navigation_code_region_bytes =
        json["navigation_code_region_bytes"].get<vec<u64>>();
    }
  } catch (const std::exception& e) {
    return fail(error_message, "failed to parse index metadata " + metadata_file.string() + ": " + e.what());
  }

  return true;
}

}  // namespace service::index_metadata
