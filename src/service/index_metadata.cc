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
    metadata.vector_dtype = parse_vector_dtype(json.value("vector_data_type", str{"float32"}));
    metadata.vector_component_size = json.value(
      "vector_component_size", static_cast<u32>(vector_dtype_component_size(metadata.vector_dtype)));
    metadata.vector_bytes = json.value(
      "vector_bytes", static_cast<u32>(vector_dtype_bytes(metadata.vector_dtype, metadata.dim)));
  } catch (const std::exception& e) {
    return fail(error_message, "failed to parse index metadata " + metadata_file.string() + ": " + e.what());
  }

  return true;
}

}  // namespace service::index_metadata
