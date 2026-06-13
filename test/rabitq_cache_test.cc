#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "vamana/rabitq_cache.hh"

namespace {

bool write_sidecar(const filepath_t& prefix, u32 ordinal, u32 nodes,
                   u32 node_size, const vamana::rabitq::Quantization& quantization,
                   u32 code_bits, const vec<byte_t>& entry) {
  const filepath_t path = index_path::rabitq_cache_file(prefix, ordinal, nodes);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  vamana::rabitq::SidecarHeader header;
  header.entry_size = static_cast<u32>(entry.size());
  header.code_bits = code_bits;
  header.node_size = node_size;
  header.raw_vector_bytes = static_cast<u32>(VamanaNode::vector_bytes());
  header.entry_count = 1;
  header.cache_budget_bytes = sizeof(header) + entry.size();
  header.quantization = quantization;
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(reinterpret_cast<const char*>(entry.data()),
               static_cast<std::streamsize>(entry.size()));
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
  vamana::rabitq::Quantization quantization{0.0f, 4096.0f, 0.0f, 0.0f};
  const u32 entry_bytes = vamana::rabitq::choose_entry_bytes(
    static_cast<u32>(VamanaNode::vector_bytes()));
  const u32 code_bits = vamana::rabitq::entry_code_bits(entry_bytes);
  const auto entry = vamana::rabitq::encode(
    vector.data(), VectorDType::uint8, quantization, code_bits, entry_bytes);

  vec<float> rotated(VamanaNode::rabitq_code_bits());
  float norm2 = 0.0f;
  VamanaNode::compute_rotated_query(vector.data(), VectorDType::uint8, rotated.data(), &norm2);
  const auto lut = vamana::rabitq::build_query_lut(rotated.data(), code_bits);
  const float lut_estimate = vamana::rabitq::estimate_distance_lut(
    lut, norm2, entry.data(), quantization);
  const float lower_bound = vamana::rabitq::lower_bound_lut(
    lut, norm2, entry.data(), quantization);
  if (!std::isfinite(lut_estimate) || !std::isfinite(lower_bound)) {
    std::cerr << "RFQ5 estimate/lower-bound is invalid\n";
    return 1;
  }
  if (lower_bound > 1e-3f * std::max(1.0f, norm2)) {
    std::cerr << "RFQ5 exact-safe self lower-bound is too high: " << lower_bound << "\n";
    return 1;
  }
  for (u32 trial = 0; trial < 64; ++trial) {
    std::array<byte_t, dim> candidate{};
    std::array<byte_t, dim> query{};
    for (u32 i = 0; i < dim; ++i) {
      candidate[i] = static_cast<byte_t>((17u * i + 31u * trial) & 0xffu);
      query[i] = static_cast<byte_t>((29u * i + 7u * trial + 11u) & 0xffu);
    }
    const auto candidate_entry = vamana::rabitq::encode(
      candidate.data(), VectorDType::uint8, quantization, code_bits, entry_bytes);
    VamanaNode::compute_rotated_query(query.data(), VectorDType::uint8,
                                      rotated.data(), &norm2);
    const auto query_lut = vamana::rabitq::build_query_lut(rotated.data(), code_bits);
    const float lb = vamana::rabitq::lower_bound_lut(
      query_lut, norm2, candidate_entry.data(), quantization);
    float exact = 0.0f;
    for (u32 i = 0; i < dim; ++i) {
      const float delta = static_cast<float>(query[i]) - static_cast<float>(candidate[i]);
      exact += delta * delta;
    }
    if (lb > exact + 1e-3f * std::max(1.0f, exact)) {
      std::cerr << "RFQ5 lower-bound exceeds exact distance: " << lb
                << " vs " << exact << "\n";
      return 1;
    }
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
  if (!write_sidecar(prefix, 1, 2, node_size, quantization, code_bits, entry) ||
      !write_sidecar(prefix, 2, 2, node_size, quantization, code_bits, entry)) {
    return 1;
  }

  vamana::rabitq::Cache cache;
  str error;
  if (!cache.load(prefix, 2, node_size, 4096, &error, 32.0)) {
    std::cerr << error << "\n";
    return 1;
  }
  if (cache.find(RemotePtr{0, 16}) == nullptr ||
      cache.find(RemotePtr{1, 16}) == nullptr ||
      cache.find(RemotePtr{0, 16 + node_size}) != nullptr ||
      cache.size_bytes() != 2 * entry.size() ||
      cache.entry_bytes() != entry.size() ||
      cache.code_bits() != code_bits) {
    std::cerr << "RFQ5 sidecar address mapping failed\n";
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
