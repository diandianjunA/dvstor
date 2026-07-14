#pragma once

#include "memory_node/memory_node.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <unordered_map>

#include "common/atomic_utils.hh"
#include "gpu_search/index_format.hh"
#include "vamana/hot_graph.hh"
#include "vamana/storage_layout_resolver.hh"

namespace memory_node_storage_owner_maintenance_detail {

inline constexpr u64 kMaintenanceObservationPeriodNs =
  5ull * 1000ull * 1000ull * 1000ull;
inline constexpr u64 kStitchCompactionMaxDelayNs =
  10ull * 1000ull * 1000ull;
inline constexpr u64 kStitchCompactionPaceSlotNs =
  1ull * 1000ull * 1000ull;
inline constexpr size_t kForegroundQueueYieldMultiplier = 2;

inline u64 steady_now_ns() {
  return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline bool same_neighbors(const vec<RemotePtr>& lhs,
                           const vec<RemotePtr>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t index = 0; index < lhs.size(); ++index) {
    if (lhs[index] != rhs[index]) return false;
  }
  return true;
}

inline double ratio_or_zero(u64 numerator, u64 denominator) {
  return denominator == 0
           ? 0.0
           : static_cast<double>(numerator) / static_cast<double>(denominator);
}

inline u64 stitch_compaction_round_ns(u32 shard_count, size_t backlog,
                                      size_t batch_limit, u32 worker_count) {
  const size_t saturation_backlog = std::max<size_t>(
    batch_limit, batch_limit * std::max<u32>(1, worker_count));
  if (backlog >= saturation_backlog) return 0;
  return kStitchCompactionPaceSlotNs * std::max<u32>(1, shard_count);
}

inline bool queue_near_limit(size_t size, size_t limit) {
  if (limit == 0) return size != 0;
  const size_t threshold = std::max<size_t>(1, (limit * 3) / 4);
  return size >= threshold;
}

inline bool counter_above_fraction(u32 value, u32 limit, u32 numerator,
                                   u32 denominator) {
  const u32 threshold = std::max<u32>(1, (limit * numerator) / denominator);
  return value >= threshold;
}

}  // namespace memory_node_storage_owner_maintenance_detail
