#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "vamana/rabitq_cache.hh"

namespace {

bool write_sidecar(const filepath_t& prefix, u32 ordinal, u32 nodes,
                   u32 node_size, const vamana::rabitq::Quantization& quantization,
                   const vamana::rabitq::CompactEntry& entry) {
  const filepath_t path = index_path::rabitq_cache_file(prefix, ordinal, nodes);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  vamana::rabitq::SidecarHeader header;
  header.node_size = node_size;
  header.entry_count = 1;
  header.quantization = quantization;
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
  return output.good();
}

}  // namespace

int main() {
  constexpr u32 dim = 128;
  VamanaNode::disable_rabitq();
  VamanaNode::init_static_storage(dim, 48, VectorDType::uint8);
  VamanaNode::enable_rabitq();
  VamanaNode::set_rabitq_centroid(vec<float>(dim, 64.0f));

  std::array<byte_t, dim> vector{};
  for (u32 i = 0; i < dim; ++i) vector[i] = static_cast<byte_t>(32 + (i % 64));
  vamana::rabitq::Quantization quantization{0.0f, 1024.0f, 0.0f, 1.0f};
  const auto entry = vamana::rabitq::encode(vector.data(), VectorDType::uint8, quantization);

  vec<float> rotated(VamanaNode::rabitq_code_bits());
  float norm2 = 0.0f;
  VamanaNode::compute_rotated_query(vector.data(), VectorDType::uint8, rotated.data(), &norm2);
  const float estimate = vamana::rabitq::estimate_distance(
    rotated.data(), norm2, entry, quantization);
  const auto lut = vamana::rabitq::build_query_lut(rotated.data());
  const float lut_estimate = vamana::rabitq::estimate_distance_lut(
    lut, norm2, entry, quantization);
  if (!std::isfinite(estimate) || estimate > 1e-2f * std::max(1.0f, norm2)) {
    std::cerr << "compact RaBitQ self-distance is invalid: " << estimate << "\n";
    return 1;
  }
  if (std::abs(estimate - lut_estimate) >
      1e-3f * std::max(1.0f, std::abs(estimate))) {
    std::cerr << "compact RaBitQ LUT estimate differs: " << estimate
              << " vs " << lut_estimate << "\n";
    return 1;
  }

  const filepath_t temp_dir = std::filesystem::temp_directory_path() / "dvstor_rabitq_cache_test";
  std::filesystem::create_directories(temp_dir);
  const filepath_t prefix = temp_dir / "index";
  const u32 node_size = static_cast<u32>(VamanaNode::total_size());
  if (!write_sidecar(prefix, 1, 2, node_size, quantization, entry) ||
      !write_sidecar(prefix, 2, 2, node_size, quantization, entry)) {
    return 1;
  }

  vamana::rabitq::Cache cache;
  str error;
  if (!cache.load(prefix, 2, node_size, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  if (cache.find(RemotePtr{0, 16}) == nullptr ||
      cache.find(RemotePtr{1, 16}) == nullptr ||
      cache.find(RemotePtr{0, 16 + node_size}) != nullptr ||
      cache.size_bytes() != 2 * sizeof(vamana::rabitq::CompactEntry)) {
    std::cerr << "compact RaBitQ sidecar address mapping failed\n";
    return 1;
  }
  std::filesystem::remove_all(temp_dir);
  return 0;
}
