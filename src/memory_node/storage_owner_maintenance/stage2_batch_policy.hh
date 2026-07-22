#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace memory_node_storage_owner_maintenance_detail {

inline std::optional<std::chrono::steady_clock::time_point>
stage2_partial_batch_deadline(
    std::size_t queued_tasks,
    std::size_t batch_limit,
    std::chrono::steady_clock::time_point oldest_queued_at,
    std::uint32_t max_wait_us) {
  const std::size_t effective_batch_limit =
    std::max<std::size_t>(1, batch_limit);
  if (queued_tasks == 0 || queued_tasks >= effective_batch_limit ||
      max_wait_us == 0) {
    return std::nullopt;
  }
  return oldest_queued_at + std::chrono::microseconds(max_wait_us);
}

// Stage2 batches are bounded by the oldest queued descriptor, not by the
// latest arrival. A full batch runs immediately; a partial batch waits at most
// max_wait_us from that fixed enqueue timestamp. Zero explicitly disables the
// batching delay and is useful for latency-sensitive deployments.
inline bool stage2_batch_ready(
    std::size_t queued_tasks,
    std::size_t batch_limit,
    std::chrono::steady_clock::time_point oldest_queued_at,
    std::chrono::steady_clock::time_point now,
    std::uint32_t max_wait_us) {
  if (queued_tasks == 0) return false;
  const std::size_t effective_batch_limit =
    std::max<std::size_t>(1, batch_limit);
  if (queued_tasks >= effective_batch_limit || max_wait_us == 0) return true;
  return now >= oldest_queued_at + std::chrono::microseconds(max_wait_us);
}

}  // namespace memory_node_storage_owner_maintenance_detail
