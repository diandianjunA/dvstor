#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#include "vamana/vamana_node.hh"

namespace {

float signed_dot(const float* rotated, const VamanaNode::RabitqCode& code) {
  float sum = 0.0f;
  for (u32 bit = 0; bit < VamanaNode::rabitq_code_bits(); ++bit) {
    sum += rotated[bit] * (VamanaNode::rabitq_bit(code, bit) ? 1.0f : -1.0f);
  }
  return sum;
}

bool nearly_equal(float lhs, float rhs, float tolerance) {
  return std::abs(lhs - rhs) <= tolerance * std::max(1.0f, std::max(std::abs(lhs), std::abs(rhs)));
}

template <class T>
bool run_case(u32 dim, VectorDType dtype) {
  VamanaNode::disable_rabitq();
  VamanaNode::init_static_storage(dim, 32, dtype);
  VamanaNode::enable_rabitq();

  std::vector<float> centroid(dim);
  std::vector<T> vector(dim);
  for (u32 d = 0; d < dim; ++d) {
    centroid[d] = static_cast<float>(static_cast<int>(d % 7) - 3) * 0.25f;
    const int value = static_cast<int>(d % 31) - 15;
    if constexpr (std::is_same_v<T, float>) {
      vector[d] = static_cast<float>(value) * 0.5f;
    } else if constexpr (std::is_same_v<T, u8>) {
      vector[d] = static_cast<u8>(value + 64);
    } else {
      vector[d] = static_cast<i8>(value);
    }
  }
  VamanaNode::set_rabitq_centroid(centroid);

  const auto* bytes = reinterpret_cast<const byte_t*>(vector.data());
  VamanaNode::RabitqCode code;
  float norm = 0.0f;
  float error = 0.0f;
  VamanaNode::compute_rabitq_entry(bytes, dtype, code, norm, error);

  std::vector<float> rotated(VamanaNode::rabitq_code_bits());
  float norm2 = 0.0f;
  VamanaNode::compute_rotated_query(bytes, dtype, rotated.data(), &norm2);

  if (code.size() != VamanaNode::rabitq_code_size() ||
      !nearly_equal(norm * norm, norm2, 1e-4f) ||
      !(error > 0.0f) || !std::isfinite(error)) {
    return false;
  }

  const float approximate_inner_product =
      norm * signed_dot(rotated.data(), code) /
      (std::sqrt(static_cast<float>(VamanaNode::rabitq_code_bits())) * error);
  const float approximate_self_distance = norm2 + norm2 - 2.0f * approximate_inner_product;
  return nearly_equal(approximate_self_distance, 0.0f, 1e-3f);
}

}  // namespace

int main() {
  for (u32 dim : {7u, 128u, 300u}) {
    if (!run_case<float>(dim, VectorDType::float32) ||
        !run_case<u8>(dim, VectorDType::uint8) ||
        !run_case<i8>(dim, VectorDType::int8)) {
      std::cerr << "RaBitQ encoding test failed at dimension " << dim << "\n";
      return 1;
    }
  }
  return 0;
}
