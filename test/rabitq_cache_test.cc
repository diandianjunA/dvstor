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
  header.raw_vector_bytes = static_cast<u32>(VamanaNode::vector_bytes());
  header.entry_count = 1;
  header.cache_budget_bytes = sizeof(header) + sizeof(entry);
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
  const auto lut = vamana::rabitq::build_query_lut(rotated.data());
  const float lut_estimate = vamana::rabitq::estimate_distance_lut(
    lut, norm2, entry, quantization);
  float scalar_dot = 0.0f;
  for (u32 bit = 0; bit < vamana::rabitq::kCodeBits; ++bit) {
    const bool positive = (entry.code[bit >> 3] & (1u << (7u - (bit & 7u)))) != 0;
    scalar_dot += positive ? rotated[bit] : -rotated[bit];
  }
  const float decoded_norm = vamana::rabitq::dequantize(
    entry.norm_q, quantization.norm_min, quantization.norm_max);
  const float decoded_error = vamana::rabitq::dequantize(
    entry.error_q, quantization.error_min, quantization.error_max);
  const float scalar_estimate = std::max(
    norm2 + decoded_norm * decoded_norm -
      2.0f * decoded_norm * scalar_dot /
        (std::sqrt(static_cast<float>(vamana::rabitq::kCodeBits)) * decoded_error),
    0.0f);
  if (!std::isfinite(lut_estimate) ||
      lut_estimate > 1e-2f * std::max(1.0f, norm2)) {
    std::cerr << "compact RaBitQ self-distance is invalid: " << lut_estimate << "\n";
    return 1;
  }
  if (std::abs(scalar_estimate - lut_estimate) >
      1e-3f * std::max(1.0f, std::abs(scalar_estimate))) {
    std::cerr << "compact RaBitQ LUT estimate differs: " << scalar_estimate
              << " vs " << lut_estimate << "\n";
    return 1;
  }

  const vec<float> gate_distances{1.0f, 2.0f, 2.05f, 2.2f, 0.5f};
  const vec<u32> gate = vamana::rabitq::select_gate(
    gate_distances, vec<u32>{4}, 2, 3, 0.05f);
  if (gate != vec<u32>({4, 0, 1, 2})) {
    std::cerr << "RaBitQ gate margin or cache-miss selection failed\n";
    return 1;
  }
  if (vamana::rabitq::select_gate(vec<f32>{3.0f, 1.0f}, {}, 16, 24, 0.05f) !=
      vec<u32>({1, 0})) {
    std::cerr << "RaBitQ gate short-batch selection failed\n";
    return 1;
  }
  if (vamana::rabitq::select_gate(vec<f32>{1.0f, 1.0f, 1.0f, 1.0f}, {},
                                  2, 3, 0.05f) != vec<u32>({0, 1, 2})) {
    std::cerr << "RaBitQ gate tie cap failed\n";
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
  if (!cache.load(prefix, 2, node_size, 4096, &error)) {
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
  const RemotePtr dynamic_ptr{0, 16 + node_size * 4};
  if (!cache.upsert_dynamic(dynamic_ptr, vector.data(), VectorDType::uint8) ||
      cache.find(dynamic_ptr) == nullptr ||
      !cache.erase_dynamic(dynamic_ptr) ||
      cache.find(dynamic_ptr) != nullptr) {
    std::cerr << "compact RaBitQ dynamic overlay failed\n";
    return 1;
  }
  std::filesystem::remove_all(temp_dir);
  return 0;
}
