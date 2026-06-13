#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

#include "common/index_path.hh"
#include "nlohmann/json.hh"
#include "vamana/rabitq_cache.hh"

namespace {

void usage(const char* program) {
  std::cerr << "Usage: " << program << " --index-prefix PATH\n";
}

}  // namespace

int main(int argc, char** argv) {
  filepath_t prefix;
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--index-prefix" && i + 1 < argc) {
      prefix = argv[++i];
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (prefix.empty()) {
    usage(argv[0]);
    return 2;
  }

  std::ifstream metadata_input(prefix.string() + ".meta.json");
  if (!metadata_input.good()) {
    std::cerr << "missing metadata for " << prefix << "\n";
    return 1;
  }
  nlohmann::json metadata;
  metadata_input >> metadata;
  const u32 nodes = metadata.at("num_memory_nodes").get<u32>();
  const u32 node_size = metadata.at("node_size").get<u32>();
  const u32 rabitq_offset = metadata.at("rabitq_offset").get<u32>();
  const u32 code_bits = metadata.at("rabitq_code_bits").get<u32>();
  if (metadata.value("node_layout", std::string{}) != "rabitq" ||
      code_bits != vamana::rabitq::kCodeBits) {
    std::cerr << "converter requires an index with full 128-bit RaBitQ entries\n";
    return 1;
  }
  vec<u64> counts;
  if (metadata.contains("hot_graph_entry_counts")) {
    counts = metadata.at("hot_graph_entry_counts").get<vec<u64>>();
  } else {
    counts.resize(nodes);
    for (u32 node = 0; node < nodes; ++node) {
      const auto bytes = std::filesystem::file_size(
        index_path::shard_file(prefix, node + 1, nodes));
      counts[node] = bytes >= 16 ? (bytes - 16) / node_size : 0;
    }
  }
  if (counts.size() != nodes) return 1;
  vamana::rabitq::Quantization quantization{
    std::numeric_limits<f32>::max(), std::numeric_limits<f32>::lowest(),
    std::numeric_limits<f32>::max(), std::numeric_limits<f32>::lowest()};
  for (u32 node = 0; node < nodes; ++node) {
    std::ifstream shard(index_path::shard_file(prefix, node + 1, nodes), std::ios::binary);
    for (u64 slot = 0; slot < counts[node]; ++slot) {
      f32 norm = 0.0f;
      f32 error = 0.0f;
      const u64 scalar_offset = 16 + slot * node_size + rabitq_offset +
        vamana::rabitq::kCodeBytes;
      shard.seekg(static_cast<std::streamoff>(scalar_offset));
      shard.read(reinterpret_cast<char*>(&norm), sizeof(norm));
      shard.read(reinterpret_cast<char*>(&error), sizeof(error));
      if (!shard.good()) {
        std::cerr << "truncated RaBitQ entry while scanning quantization\n";
        return 1;
      }
      quantization.norm_min = std::min(quantization.norm_min, norm);
      quantization.norm_max = std::max(quantization.norm_max, norm);
      quantization.error_min = std::min(quantization.error_min, error);
      quantization.error_max = std::max(quantization.error_max, error);
    }
  }

  for (u32 node = 0; node < nodes; ++node) {
    const filepath_t shard_path = index_path::shard_file(prefix, node + 1, nodes);
    const filepath_t sidecar_path = index_path::rabitq_cache_file(prefix, node + 1, nodes);
    std::ifstream shard(shard_path, std::ios::binary);
    std::ofstream sidecar(sidecar_path, std::ios::binary | std::ios::trunc);
    if (!shard.good() || !sidecar.good()) {
      std::cerr << "failed to open shard or sidecar for node " << node + 1 << "\n";
      return 1;
    }
    vamana::rabitq::SidecarHeader header;
    header.node_size = node_size;
    header.entry_count = counts[node];
    header.quantization = quantization;
    sidecar.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (u64 slot = 0; slot < counts[node]; ++slot) {
      std::array<byte_t, vamana::rabitq::kCodeBytes> code{};
      f32 norm = 0.0f;
      f32 error = 0.0f;
      const u64 entry_offset = 16 + slot * node_size + rabitq_offset;
      shard.seekg(static_cast<std::streamoff>(entry_offset));
      shard.read(reinterpret_cast<char*>(code.data()), code.size());
      shard.read(reinterpret_cast<char*>(&norm), sizeof(norm));
      shard.read(reinterpret_cast<char*>(&error), sizeof(error));
      if (!shard.good()) {
        std::cerr << "truncated RaBitQ entry in " << shard_path << "\n";
        return 1;
      }
      vamana::rabitq::CompactEntry entry;
      entry.code = code;
      entry.norm_q = vamana::rabitq::quantize(
        norm, quantization.norm_min, quantization.norm_max);
      entry.error_q = vamana::rabitq::quantize(
        error, quantization.error_min, quantization.error_max);
      sidecar.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
    }
    if (!sidecar.good()) {
      std::cerr << "failed to write " << sidecar_path << "\n";
      return 1;
    }
    std::cout << "wrote " << sidecar_path << " (" << counts[node] << " entries)\n";
  }
  metadata["rabitq_cache_bits"] = vamana::rabitq::kCodeBits;
  metadata["rabitq_cache_entry_size"] = vamana::rabitq::kEntryBytes;
  metadata["rabitq_cache_norm_min"] = quantization.norm_min;
  metadata["rabitq_cache_norm_max"] = quantization.norm_max;
  metadata["rabitq_cache_error_min"] = quantization.error_min;
  metadata["rabitq_cache_error_max"] = quantization.error_max;
  std::ofstream metadata_output(prefix.string() + ".meta.json", std::ios::trunc);
  metadata_output << std::setw(2) << metadata << '\n';
  if (!metadata_output.good()) {
    std::cerr << "failed to update RaBitQ metadata\n";
    return 1;
  }
  return 0;
}
