#include <array>
#include <cassert>
#include <stdexcept>

#include "gpu_search/maintenance_fence.hh"

int main() {
  using gpu_search::maintenance_fence::capture_targets;
  using gpu_search::u64;

  // Cross-shard upsert: compute knows only new-home Stage2 sequence 9 on B,
  // while the control snapshot exposes old-home cleanup sequence 6 on A.
  const std::array<u64, 3> requested{0, 9, 0};
  const std::array<u64, 3> next{7, 10, 1};
  const auto captured = capture_targets(requested, next);
  assert((captured == std::vector<u64>{6, 9, 0}));

  // A caller-specific target may already be newer than the sampled prefix.
  const std::array<u64, 3> newer_requested{8, 9, 4};
  const auto newer = capture_targets(newer_requested, next);
  assert((newer == std::vector<u64>{8, 9, 4}));

  bool rejected_zero = false;
  try {
    const std::array<u64, 1> invalid_next{0};
    (void)capture_targets(std::span<const u64>{requested}.first(1),
                          invalid_next);
  } catch (const std::runtime_error&) {
    rejected_zero = true;
  }
  assert(rejected_zero);
  return 0;
}
