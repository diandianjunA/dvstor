#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "gpu_search/index_format.hh"
#include "vamana/rabitq_cache.hh"
#include "vamana/vamana_node.hh"

namespace {

float gpu_layout_estimate(const vamana::rabitq::QueryLut& lut,
                          float query_norm2,
                          const byte_t* entry) {
  float signed_dot = 0.0f;
  for (u32 byte = 0; byte < lut.code_bytes; ++byte) {
    signed_dot += lut.signed_dot[static_cast<size_t>(byte) * 256 + entry[byte]];
  }
  float norm = 0.0f;
  float error = 1.0f;
  std::memcpy(&norm, entry + gpu_search::format::rabitq_norm_offset(lut.code_bits),
              sizeof(norm));
  std::memcpy(&error, entry + gpu_search::format::rabitq_error_offset(lut.code_bits),
              sizeof(error));
  const float denominator = std::sqrt(static_cast<float>(lut.code_bits)) *
    std::max(error, 1e-6f);
  const float inner_product = norm * signed_dot / denominator;
  return std::max(query_norm2 + norm * norm - 2.0f * inner_product, 0.0f);
}

void verify_dimension(u32 dim) {
  VamanaNode::init_static_storage(dim, 8, VectorDType::uint8);
  VamanaNode::enable_rabitq();
  std::vector<float> centroid(dim);
  for (u32 d = 0; d < dim; ++d) centroid[d] = 8.0f + static_cast<float>(d % 11);
  VamanaNode::set_rabitq_centroid(centroid);

  const u32 code_bits = VamanaNode::rabitq_code_bits();
  assert(gpu_search::format::rabitq_code_storage_bytes(code_bits) ==
         VamanaNode::rabitq_code_storage_size());
  assert(gpu_search::format::rabitq_entry_bytes(code_bits) ==
         VamanaNode::rabitq_entry_size());

  std::mt19937 generator(1000 + dim);
  std::uniform_int_distribution<int> values(0, 255);
  std::vector<std::vector<byte_t>> vectors(12, std::vector<byte_t>(dim));
  for (auto& vector : vectors) {
    for (byte_t& value : vector) value = static_cast<byte_t>(values(generator));
  }

  for (const auto& vector : vectors) {
    VamanaNode::RabitqCode code;
    float norm = 0.0f;
    float error = 0.0f;
    VamanaNode::compute_rabitq_entry(vector.data(), VectorDType::uint8,
                                     code, norm, error);
    std::vector<byte_t> gpu_entry(VamanaNode::rabitq_entry_size(), 0);
    std::memcpy(gpu_entry.data(), code.data(), code.size());
    std::memcpy(gpu_entry.data() + gpu_search::format::rabitq_norm_offset(code_bits),
                &norm, sizeof(norm));
    std::memcpy(gpu_entry.data() + gpu_search::format::rabitq_error_offset(code_bits),
                &error, sizeof(error));
    const std::vector<byte_t> cpu_entry =
      vamana::rabitq::encode_full(vector.data(), VectorDType::uint8);

    for (const auto& query : vectors) {
      std::vector<float> rotated(code_bits);
      float query_norm2 = 0.0f;
      VamanaNode::compute_rotated_query(query.data(), VectorDType::uint8,
                                        rotated.data(), &query_norm2);
      const auto lut = vamana::rabitq::build_query_lut(rotated.data(), code_bits);
      const float cpu_distance =
        vamana::rabitq::estimate_distance_lut_full(lut, query_norm2, cpu_entry.data());
      const float gpu_distance = gpu_layout_estimate(lut, query_norm2, gpu_entry.data());
      const float arithmetic_scale = std::max(1.0f, query_norm2 + norm * norm);
      const float tolerance = 2e-6f * arithmetic_scale;
      if (std::abs(cpu_distance - gpu_distance) > tolerance) {
        std::cerr << "RaBitQ layout mismatch dim=" << dim
                  << " bits=" << code_bits
                  << " norm=" << norm
                  << " error=" << error
                  << " query_norm2=" << query_norm2
                  << " cpu=" << cpu_distance
                  << " gpu=" << gpu_distance << '\n';
      }
      assert(std::abs(cpu_distance - gpu_distance) <= tolerance);
      if (&query == &vector) assert(gpu_distance <= tolerance);
    }
  }
}

}  // namespace

int main() {
  verify_dimension(4);
  verify_dimension(9);
  verify_dimension(128);
  return 0;
}
