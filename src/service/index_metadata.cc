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
    metadata.node_layout = json.value("node_layout", str{"standard"});
    metadata.storage_format = json.at("storage_format").get<str>();
    metadata.graph_hot_bytes = json.value("graph_hot_bytes", 0u);
    metadata.vector_offset = json.value("vector_offset", 0u);
    metadata.neighbors_offset = json.value("neighbors_offset", 0u);
    metadata.rabitq_offset = json.value("rabitq_offset", 0u);
    metadata.vector_dtype = parse_vector_dtype(json.value("vector_data_type", str{"float32"}));
    metadata.vector_component_size = json.value(
      "vector_component_size", static_cast<u32>(vector_dtype_component_size(metadata.vector_dtype)));
    metadata.vector_bytes = json.value(
      "vector_bytes", static_cast<u32>(vector_dtype_bytes(metadata.vector_dtype, metadata.dim)));
    metadata.rabitq_code_bits = json.value("rabitq_code_bits", 0u);
    metadata.rabitq_entry_size = json.value("rabitq_entry_size", 0u);
    metadata.rabitq_cache_bits = json.value("rabitq_cache_bits", 0u);
    metadata.rabitq_cache_entry_size = json.value("rabitq_cache_entry_size", 0u);
    metadata.rabitq_cache_norm_min = json.value("rabitq_cache_norm_min", 0.0f);
    metadata.rabitq_cache_norm_max = json.value("rabitq_cache_norm_max", 0.0f);
    metadata.rabitq_cache_error_min = json.value("rabitq_cache_error_min", 0.0f);
    metadata.rabitq_cache_error_max = json.value("rabitq_cache_error_max", 0.0f);
    if (json.contains("rabitq_centroid")) {
        metadata.rabitq_centroid = json["rabitq_centroid"].get<vec<float>>();
    }
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
  } catch (const std::exception& e) {
    return fail(error_message, "failed to parse index metadata " + metadata_file.string() + ": " + e.what());
  }

  return true;
}

}  // namespace service::index_metadata
