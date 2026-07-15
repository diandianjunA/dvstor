#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace core_assignment_detail {

// Split an already topology-ordered CPU sequence between colocated processes.
// When the second half contains the SMT siblings of the first half, keep both
// logical CPUs of a physical core in the same process partition.  This avoids
// two storage shards unknowingly competing on sibling threads while other
// physical cores remain idle.
inline std::vector<std::uint32_t> partition_ordered_cores(
    const std::vector<std::uint32_t>& ordered,
    bool paired_smt_halves,
    std::uint32_t rank,
    std::uint32_t count) {
  if (ordered.empty() || count == 0 || rank >= count) {
    throw std::invalid_argument("invalid local process CPU partition");
  }

  const std::size_t group_count = paired_smt_halves
    ? ordered.size() / 2 : ordered.size();
  if (group_count == 0 || count > group_count ||
      (paired_smt_halves && ordered.size() % 2 != 0)) {
    throw std::invalid_argument("local process CPU partition exceeds core groups");
  }

  const std::size_t base = group_count / count;
  const std::size_t extra = group_count % count;
  const std::size_t begin = static_cast<std::size_t>(rank) * base +
                            std::min<std::size_t>(rank, extra);
  const std::size_t groups = base + (rank < extra ? 1 : 0);
  const std::size_t end = begin + groups;

  std::vector<std::uint32_t> result;
  result.reserve(groups * (paired_smt_halves ? 2 : 1));
  result.insert(result.end(), ordered.begin() + begin, ordered.begin() + end);
  if (paired_smt_halves) {
    result.insert(result.end(),
                  ordered.begin() + group_count + begin,
                  ordered.begin() + group_count + end);
  }
  return result;
}

}  // namespace core_assignment_detail
