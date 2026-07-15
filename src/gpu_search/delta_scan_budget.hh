#pragma once

#include <cstdint>

namespace gpu_search {

// Query-time delta injection is a short-lived visibility aid while stage2 is
// making a mutation reachable through the authoritative graph.  Its work must
// not grow with the maintenance backlog.  This is an algorithm constant (and
// deliberately not another deployment knob): the normal graph search and
// stage2 construction widths remain unchanged.
inline constexpr std::uint32_t kDeltaScanRecordBudget = 2048;

struct DeltaScanSegment {
  std::uint32_t offset{};
  std::uint32_t count{};
};

#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr DeltaScanSegment delta_scan_segment(
    std::uint32_t index,
    std::uint32_t segment_count,
    std::uint32_t budget = kDeltaScanRecordBudget) {
  if (segment_count == 0 || index >= segment_count) return {};
  const std::uint32_t base = budget / segment_count;
  const std::uint32_t remainder = budget % segment_count;
  return DeltaScanSegment{
    .offset = index * base + (index < remainder ? index : remainder),
    .count = base + (index < remainder ? 1u : 0u),
  };
}

}  // namespace gpu_search
