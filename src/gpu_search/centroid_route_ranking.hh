#pragma once

#include <array>
#include <cstddef>

#include "gpu_search/types.hh"

namespace gpu_search::centroid_route_ranking {

// Route publications can be absent or transiently unavailable while their
// per-shard seqlock is odd.  Keep validity in the key instead of encoding it
// as FLT_MAX: a valid, saturated distance must still sort ahead of an absent
// shard.  Shard is the final key, matching the CPU home selector exactly.
struct RankedShard {
  f32 distance{};
  u32 shard{};
  u32 valid{};
};

#if defined(__CUDACC__)
#define DVSTOR_CENTROID_ROUTE_HD __host__ __device__
#else
#define DVSTOR_CENTROID_ROUTE_HD
#endif

DVSTOR_CENTROID_ROUTE_HD inline constexpr bool less(
    const RankedShard& lhs, const RankedShard& rhs) {
  if (lhs.valid != rhs.valid) return lhs.valid > rhs.valid;
  if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
  return lhs.shard < rhs.shard;
}

DVSTOR_CENTROID_ROUTE_HD inline constexpr bool should_exchange(
    const RankedShard& lhs, const RankedShard& rhs, bool ascending) {
  return ascending ? less(rhs, lhs) : less(lhs, rhs);
}

// Ranking is meaningful only when every per-shard snapshot belongs to one
// complete table publication. An even, unchanged epoch proves that no shard
// subset was being rewritten while the CTA collected and sorted its inputs.
DVSTOR_CENTROID_ROUTE_HD inline constexpr bool stable_publication_epoch(
    u64 before, u64 after) {
  return (before & 1u) == 0 && before == after && (after & 1u) == 0;
}

#undef DVSTOR_CENTROID_ROUTE_HD

// CPU-testable mirror of the fixed-size CUDA sorting network.  Production
// routing executes each compare-exchange stage across the CTA with a barrier;
// keeping the network schedule here makes exhaustive/randomized equivalence
// tests possible on hosts without a CUDA device.
template <std::size_t N>
inline void bitonic_sort_for_test(std::array<RankedShard, N>& values) {
  static_assert(N != 0 && (N & (N - 1)) == 0,
                "bitonic route capacity must be a power of two");
  for (std::size_t sequence = 2; sequence <= N; sequence <<= 1) {
    for (std::size_t stride = sequence >> 1; stride != 0; stride >>= 1) {
      for (std::size_t index = 0; index < N; ++index) {
        const std::size_t partner = index ^ stride;
        if (partner <= index) continue;
        const bool ascending = (index & sequence) == 0;
        if (should_exchange(values[index], values[partner], ascending)) {
          const RankedShard temporary = values[index];
          values[index] = values[partner];
          values[partner] = temporary;
        }
      }
    }
  }
}

}  // namespace gpu_search::centroid_route_ranking
