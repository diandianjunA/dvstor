#pragma once

#include <cstdint>

namespace gpu_search {

#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr std::uint32_t initial_seed_budget(
    std::uint32_t configured,
    std::uint32_t traversal_capacity) {
  return configured < traversal_capacity ? configured : traversal_capacity;
}

}  // namespace gpu_search
