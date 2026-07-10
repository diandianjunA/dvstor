#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

#include "gpu_search/persistent_kernel.hh"

namespace gpu_search::memory_budget {

struct Request {
  u64 nodes{};
  u64 max_delta_vectors{};
  u64 usable_bytes{};
  u64 requested_cache_bytes{};
  u64 delta_budget_bytes{};
  u32 dim{};
  u32 code_bits{};
  u32 code_entry_bytes{};
  u32 vector_bytes{};
  u32 query_slots{};
  u32 beam_width{};
  u32 graph_degree{};
  u32 exact_width{};
  u32 anchor_count{};
  u32 shard_count{};
  u32 entry_point_count{};
  u32 cache_ways{4};
};

struct Result {
  u64 code_bytes{};
  u64 delta_bytes{};
  u64 query_workspace_bytes{};
  u64 exact_bytes{};
  u64 metadata_bytes{};
  u64 fixed_bytes{};
  u64 cache_total_bytes{};
  u64 cache_payload_bytes{};
  u64 explicit_bytes{};
  u32 delta_capacity{};
  u32 delta_table_capacity{};
  u32 visited_capacity{};
  u32 cache_sets{};
  u32 cache_slots{};
  bool fits{};
};

inline u32 next_power_of_two(u32 value) {
  if (value >= (1u << 31)) return 1u << 31;
  return std::max<u32>(2, std::bit_ceil(value));
}

inline u64 delta_footprint(u32 capacity, u32 dim, u32 entry_bytes) {
  if (capacity == 0) return 0;
  const u32 table_capacity = next_power_of_two(capacity * 2);
  return static_cast<u64>(capacity) *
      (sizeof(DeviceDeltaRecord) + static_cast<u64>(dim) * sizeof(f32) +
       entry_bytes + sizeof(u32)) +
    static_cast<u64>(table_capacity) *
      (sizeof(u32) + sizeof(u64) + sizeof(u64) + sizeof(u32));
}

inline u32 choose_delta_capacity(u64 budget, u64 max_vectors,
                                 u32 dim, u32 entry_bytes) {
  u32 low = 1;
  u32 high = static_cast<u32>(std::min<u64>(
    std::min<u64>(max_vectors, kDeltaHandleMask),
    budget / std::max<u64>(1, static_cast<u64>(dim) * sizeof(f32))));
  if (high == 0) return 0;
  while (low < high) {
    const u32 middle = low + (high - low + 1) / 2;
    if (delta_footprint(middle, dim, entry_bytes) <= budget) low = middle;
    else high = middle - 1;
  }
  return delta_footprint(low, dim, entry_bytes) <= budget ? low : 0;
}

inline Result estimate(const Request& request) {
  Result result;
  if (request.nodes == 0 || request.nodes >= (1ull << 30) || request.dim == 0 ||
      request.code_bits == 0 || request.code_entry_bytes == 0 ||
      request.query_slots == 0 || request.beam_width == 0 ||
      request.graph_degree == 0 || request.exact_width == 0 ||
      request.cache_ways == 0) {
    return result;
  }
  result.delta_capacity = choose_delta_capacity(
    request.delta_budget_bytes, request.max_delta_vectors,
    request.dim, request.code_entry_bytes);
  if (result.delta_capacity == 0) return result;
  result.delta_table_capacity = next_power_of_two(result.delta_capacity * 2);
  result.visited_capacity = next_power_of_two(
    std::max<u32>(256, request.beam_width * request.graph_degree * 2));
  result.code_bytes = request.nodes * request.code_entry_bytes;
  result.delta_bytes = delta_footprint(
    result.delta_capacity, request.dim, request.code_entry_bytes);
  result.query_workspace_bytes =
    static_cast<u64>(request.query_slots) * request.dim * sizeof(f32) +
    static_cast<u64>(request.query_slots) * request.code_bits * sizeof(f32) +
    static_cast<u64>(request.query_slots) * (request.code_bits / 8) * 256 * sizeof(f32) +
    static_cast<u64>(request.query_slots) * request.beam_width *
      (sizeof(u32) + sizeof(f32) + sizeof(u8)) +
    static_cast<u64>(request.query_slots) * result.visited_capacity * sizeof(u32) +
    static_cast<u64>(request.query_slots) * request.anchor_count * sizeof(f32);
  result.exact_bytes = static_cast<u64>(request.query_slots) * request.exact_width *
    (8 + request.vector_bytes);
  result.metadata_bytes = static_cast<u64>(request.shard_count) *
      sizeof(DeviceShardRegion) +
    static_cast<u64>(request.dim) * sizeof(f32) +
    static_cast<u64>(request.entry_point_count) * sizeof(u32) +
    static_cast<u64>(request.anchor_count) * request.dim * sizeof(f32) +
    (64ull << 20);
  result.fixed_bytes = result.code_bytes + result.delta_bytes +
    result.query_workspace_bytes + result.exact_bytes + result.metadata_bytes;
  if (result.fixed_bytes >= request.usable_bytes) return result;
  const u64 cache_budget = std::min(
    request.requested_cache_bytes, request.usable_bytes - result.fixed_bytes);
  const u64 bytes_per_set = static_cast<u64>(request.cache_ways) *
      (kPersistentGraphCacheLineBytes + 3 * sizeof(u64) + 2 * sizeof(u32)) +
    sizeof(u32);
  result.cache_sets = static_cast<u32>(std::min<u64>(
    cache_budget / bytes_per_set, std::numeric_limits<u32>::max()));
  if (result.cache_sets < request.query_slots) return result;
  result.cache_slots = result.cache_sets * request.cache_ways;
  result.cache_total_bytes = static_cast<u64>(result.cache_sets) * bytes_per_set;
  result.cache_payload_bytes = static_cast<u64>(result.cache_slots) *
    kPersistentGraphCacheLineBytes;
  result.explicit_bytes = result.fixed_bytes + result.cache_total_bytes;
  result.fits = result.explicit_bytes <= request.usable_bytes;
  return result;
}

}  // namespace gpu_search::memory_budget
