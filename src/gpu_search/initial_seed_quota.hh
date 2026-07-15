#pragma once

#include <cstdint>

namespace gpu_search {

struct InitialSeedQuota {
  std::uint32_t static_count{};
  std::uint32_t dynamic_count{};
};

// Preserve the established route whenever both route tiers contend for a
// bounded traversal beam.  All requested static seeds are retained when they
// fit; otherwise one slot is reserved for the adaptive route and every other
// slot remains static.  A one-entry beam cannot represent both tiers, so the
// static fallback wins that degenerate case.
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr InitialSeedQuota choose_initial_seed_quota(
    std::uint32_t static_available,
    std::uint32_t dynamic_available,
    std::uint32_t capacity) {
  if (capacity == 0) return {};
  if (static_available == 0) {
    return InitialSeedQuota{
      .dynamic_count = dynamic_available < capacity
        ? dynamic_available : capacity,
    };
  }
  if (dynamic_available == 0 || capacity == 1) {
    return InitialSeedQuota{
      .static_count = static_available < capacity
        ? static_available : capacity,
    };
  }

  const std::uint32_t static_limit = capacity - 1;
  const std::uint32_t static_count = static_available < static_limit
    ? static_available : static_limit;
  const std::uint32_t dynamic_limit = capacity - static_count;
  const std::uint32_t dynamic_count = dynamic_available < dynamic_limit
    ? dynamic_available : dynamic_limit;
  return InitialSeedQuota{
    .static_count = static_count,
    .dynamic_count = dynamic_count,
  };
}

}  // namespace gpu_search
