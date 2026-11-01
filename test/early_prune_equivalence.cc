#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "common/vector_dtype.hh"

int main() {
  std::mt19937 rng(1234567);
  std::uniform_real_distribution<float> component(-1.0f, 1.0f);
  std::uniform_real_distribution<float> scale_distribution(-3.0f, 3.0f);
  std::uniform_real_distribution<float> factor_distribution(0.0f, 3.0f);
  std::uniform_int_distribution<int> byte_component(-128, 127);
  size_t checked = 0;
  for (const VectorDType dtype : {VectorDType::float32, VectorDType::int8, VectorDType::uint8}) {
    const u32 dim = dtype == VectorDType::float32 ? 96 : 100;
    const size_t bytes = dim * (dtype == VectorDType::float32 ? sizeof(float) : 1);
    std::vector<byte_t> lhs(bytes), rhs(bytes);
    for (size_t iteration = 0; iteration < 500000; ++iteration) {
      const float scale = std::pow(10.0f, scale_distribution(rng));
      for (u32 index = 0; index < dim; ++index) {
        if (dtype == VectorDType::float32) {
          reinterpret_cast<float*>(lhs.data())[index] = component(rng) * scale;
          reinterpret_cast<float*>(rhs.data())[index] = component(rng) * scale;
        } else {
          lhs[index] = static_cast<byte_t>(byte_component(rng));
          rhs[index] = static_cast<byte_t>(byte_component(rng));
        }
      }
      const float exact = typed_l2_distance(lhs.data(), dtype, rhs.data(), dtype, dim);
      const float alpha = 1.2f;
      float source = 0.0f;
      switch (iteration % 8) {
        case 0: source = alpha * exact; break;
        case 1: source = alpha * exact * (1.0f + 1e-6f); break;
        case 2: source = alpha * exact * (1.0f - 1e-6f); break;
        case 3: source = alpha * exact * (1.0f + 1e-5f); break;
        case 4: source = alpha * exact * (1.0f - 1e-5f); break;
        case 5: source = exact * factor_distribution(rng); break;
        case 6: source = 0.0f; break;
        default: source = std::numeric_limits<float>::max(); break;
      }
      const bool expected = alpha * exact < source;
      const auto decision = typed_l2_prune_predicate_early(
        lhs.data(), dtype, rhs.data(), dtype, dim, 1.2, source);
      if ((decision == L2PruneEarlyDecision::pruned && !expected) ||
          (decision == L2PruneEarlyDecision::not_pruned && expected)) {
        std::cerr << "mismatch dtype=" << vector_dtype_name(dtype)
                  << " iteration=" << iteration << " exact=" << exact
                  << " source=" << source << " expected=" << expected
                  << " decision=" << static_cast<u32>(decision) << "\n";
        return 1;
      }
      ++checked;
    }
  }
  std::cout << "equivalent decisions=" << checked << "\n";
  return 0;
}
