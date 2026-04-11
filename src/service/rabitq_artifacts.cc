#include "service/rabitq_artifacts.hh"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hh"

namespace service::rabitq {

namespace {

bool fail(str* error_message, const str& message) {
  if (error_message) {
    *error_message = message;
  }
  return false;
}

}  // namespace

bool load_artifacts(const filepath_t& index_prefix, Artifacts& artifacts, str* error_message) {
  artifacts = Artifacts{};
  artifacts.index_prefix = index_prefix;

  const filepath_t meta_file = filepath_t(index_prefix.string() + ".meta.json");
  std::ifstream meta_input(meta_file);
  if (!meta_input.good()) {
    return fail(error_message, "missing metadata file: " + meta_file.string());
  }

  nlohmann::json metadata;
  try {
    meta_input >> metadata;
    artifacts.dim = metadata.at("dim").get<u32>();
    artifacts.rabitq_bits = metadata.at("rabitq_bits").get<u32>();
    artifacts.num_memory_nodes = metadata.at("num_memory_nodes").get<u32>();
    artifacts.rabitq_size = metadata.at("rabitq_size").get<u32>();
  } catch (const std::exception& e) {
    return fail(error_message, "invalid metadata file " + meta_file.string() + ": " + e.what());
  }

  const filepath_t rotation_file = filepath_t(index_prefix.string() + ".rotation.bin");
  std::ifstream rotation_input(rotation_file, std::ios::binary);
  if (!rotation_input.good()) {
    return fail(error_message, "missing RaBitQ rotation file: " + rotation_file.string());
  }

  u32 file_dim = 0;
  rotation_input.read(reinterpret_cast<char*>(&file_dim), sizeof(u32));
  if (!rotation_input.good()) {
    return fail(error_message, "failed to read rotation header: " + rotation_file.string());
  }
  if (file_dim != artifacts.dim) {
    return fail(error_message,
                "rotation dim mismatch in " + rotation_file.string() + ": expected " +
                  std::to_string(artifacts.dim) + ", got " + std::to_string(file_dim));
  }

  artifacts.rotation_matrix.resize(static_cast<size_t>(artifacts.dim) * artifacts.dim);
  rotation_input.read(reinterpret_cast<char*>(artifacts.rotation_matrix.data()),
                      static_cast<std::streamsize>(artifacts.rotation_matrix.size() * sizeof(float)));
  artifacts.rotated_centroid.resize(artifacts.dim);
  rotation_input.read(reinterpret_cast<char*>(artifacts.rotated_centroid.data()),
                      static_cast<std::streamsize>(artifacts.rotated_centroid.size() * sizeof(float)));
  rotation_input.read(reinterpret_cast<char*>(&artifacts.t_const), sizeof(double));

  if (!rotation_input.good()) {
    return fail(error_message, "failed to read complete RaBitQ payload: " + rotation_file.string());
  }

  return true;
}

}  // namespace service::rabitq
