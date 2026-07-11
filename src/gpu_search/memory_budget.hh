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
  u64 requested_exact_cache_bytes{};
  u64 delta_budget_bytes{};
  u32 dim{};
  u32 pq_subquantizers{};
  u32 code_bytes{};
  u32 vector_bytes{};
  u32 query_slots{};
  u32 beam_width{};
  u32 graph_degree{};
  u32 exact_width{};
  u32 exact_record_bytes{};
  u32 anchor_count{};
  u32 shard_count{};
  u32 entry_point_count{};
  u32 cache_ways{4};
  u32 exact_cache_ways{4};
};

struct Result {
  u64 code_bytes{};
  u64 delta_bytes{};
  u64 delta_code_bytes{};
  u64 query_workspace_bytes{};
  u64 exact_bytes{};
  u64 metadata_bytes{};
  u64 fixed_bytes{};
  u64 cache_total_bytes{};
  u64 cache_payload_bytes{};
  u64 exact_cache_total_bytes{};
  u64 exact_cache_payload_bytes{};
  u64 explicit_bytes{};
  u32 delta_capacity{};
  u32 delta_table_capacity{};
  u32 visited_capacity{};
  u32 cache_sets{};
  u32 cache_slots{};
  u32 exact_cache_sets{};
  u32 exact_cache_slots{};
  u32 exact_cache_stride{};
  bool fits{};
};

inline u32 next_power_of_two(u64 value) {
  if (value >= (1u << 31)) return 1u << 31;
  return std::max<u32>(2, std::bit_ceil(static_cast<u32>(value)));
}

inline u64 delta_footprint(u32 capacity, u32 vector_bytes, u32 code_bytes) {
  if (capacity == 0) return 0;
  const u32 table_capacity = next_power_of_two(static_cast<u64>(capacity) * 2);
  return static_cast<u64>(capacity) *
      (sizeof(DeviceDeltaRecord) + vector_bytes +
       code_bytes + sizeof(u32)) +
    static_cast<u64>(table_capacity) *
      (sizeof(u32) + sizeof(u64) + sizeof(u64) + sizeof(u32));
}

inline u32 choose_delta_capacity(u64 budget, u64 max_vectors,
                                 u32 vector_bytes, u32 code_bytes) {
  u32 low = 1;
  u32 high = static_cast<u32>(std::min<u64>(
    std::min<u64>(max_vectors, kDeltaHandleMask),
    budget / std::max<u64>(1, vector_bytes)));
  if (high == 0) return 0;
  while (low < high) {
    const u32 middle = low + (high - low + 1) / 2;
    if (delta_footprint(middle, vector_bytes, code_bytes) <= budget) low = middle;
    else high = middle - 1;
  }
  return delta_footprint(low, vector_bytes, code_bytes) <= budget ? low : 0;
}

inline Result estimate(const Request& request) {
  Result result;
  if (request.nodes == 0 || request.nodes >= (1ull << 30) || request.dim == 0 ||
      request.pq_subquantizers == 0 || request.code_bytes == 0 ||
      request.vector_bytes == 0 ||
      request.code_bytes != request.pq_subquantizers ||
      request.dim % request.pq_subquantizers != 0 ||
      request.query_slots == 0 || request.beam_width == 0 ||
      request.graph_degree == 0 || request.exact_width == 0 ||
      request.exact_record_bytes == 0 || request.cache_ways == 0 ||
      request.exact_cache_ways == 0) {
    return result;
  }
  result.delta_capacity = choose_delta_capacity(
    request.delta_budget_bytes, request.max_delta_vectors,
    request.vector_bytes, request.code_bytes);
  if (result.delta_capacity == 0) return result;
  result.delta_table_capacity = next_power_of_two(
    static_cast<u64>(result.delta_capacity) * 2);
  result.visited_capacity = next_power_of_two(
    std::max<u32>(256, request.beam_width * request.graph_degree * 8));
  result.code_bytes = request.nodes * request.code_bytes;
  result.delta_bytes = delta_footprint(
    result.delta_capacity, request.vector_bytes, request.code_bytes);
  result.delta_code_bytes = static_cast<u64>(result.delta_capacity) * request.code_bytes;
  result.query_workspace_bytes =
    static_cast<u64>(request.query_slots) * request.dim * sizeof(f32) +
    static_cast<u64>(request.query_slots) * request.dim * sizeof(f32) +
    static_cast<u64>(request.query_slots) * request.pq_subquantizers * 256 * sizeof(f32) +
    static_cast<u64>(request.query_slots) * result.visited_capacity * sizeof(u32) +
    static_cast<u64>(request.query_slots) * request.anchor_count * sizeof(f32);
  result.exact_bytes = static_cast<u64>(request.query_slots) * request.exact_width *
    (8 + request.vector_bytes);
  result.metadata_bytes = static_cast<u64>(request.shard_count) *
      sizeof(DeviceShardRegion) +
    (static_cast<u64>(request.dim) * request.dim +
     static_cast<u64>(request.dim) * 256) * sizeof(f32) +
    static_cast<u64>(request.entry_point_count) * sizeof(u32) +
    static_cast<u64>(request.anchor_count) * request.dim * sizeof(f32) +
    static_cast<u64>(request.anchor_count) * sizeof(u32) +
    (64ull << 20);
  result.fixed_bytes = result.code_bytes + result.delta_bytes +
    result.query_workspace_bytes + result.exact_bytes + result.metadata_bytes;
  if (result.fixed_bytes >= request.usable_bytes) return result;

  result.exact_cache_stride = static_cast<u32>(
    (static_cast<u64>(request.exact_record_bytes) + 15) & ~15ull);
  const u64 bytes_per_set = static_cast<u64>(request.cache_ways) *
      (kPersistentGraphCacheLineBytes + 3 * sizeof(u64) + 2 * sizeof(u32)) +
    sizeof(u32);
  const u64 exact_bytes_per_set = static_cast<u64>(request.exact_cache_ways) *
      (result.exact_cache_stride + 3 * sizeof(u32)) + sizeof(u32);
  const u64 minimum_cache_bytes = request.requested_cache_bytes == 0
    ? 0 : static_cast<u64>(request.query_slots) * bytes_per_set;
  const u64 minimum_exact_cache_bytes = request.requested_exact_cache_bytes == 0
    ? 0 : static_cast<u64>(request.query_slots) * exact_bytes_per_set;
  u64 available_cache_bytes = request.usable_bytes - result.fixed_bytes;
  if (available_cache_bytes < minimum_cache_bytes + minimum_exact_cache_bytes) {
    return result;
  }

  const u64 cache_budget = std::min(
    request.requested_cache_bytes,
    available_cache_bytes - minimum_exact_cache_bytes);
  const u64 max_cache_sets = std::numeric_limits<u32>::max() / request.cache_ways;
  result.cache_sets = static_cast<u32>(std::min<u64>(
    cache_budget / bytes_per_set, max_cache_sets));
  if (request.requested_cache_bytes != 0) {
    if (result.cache_sets < request.query_slots) return result;
    result.cache_slots = result.cache_sets * request.cache_ways;
    result.cache_total_bytes = static_cast<u64>(result.cache_sets) * bytes_per_set;
    result.cache_payload_bytes = static_cast<u64>(result.cache_slots) *
      kPersistentGraphCacheLineBytes;
  }
  available_cache_bytes -= result.cache_total_bytes;

  const u64 exact_cache_budget = std::min(
    request.requested_exact_cache_bytes, available_cache_bytes);
  const u64 max_exact_sets = std::numeric_limits<u32>::max() /
    request.exact_cache_ways;
  result.exact_cache_sets = static_cast<u32>(std::min<u64>(
    exact_cache_budget / exact_bytes_per_set, max_exact_sets));
  if (request.requested_exact_cache_bytes != 0) {
    if (result.exact_cache_sets < request.query_slots) return result;
    result.exact_cache_slots = result.exact_cache_sets * request.exact_cache_ways;
    result.exact_cache_payload_bytes = static_cast<u64>(result.exact_cache_slots) *
      result.exact_cache_stride;
    result.exact_cache_total_bytes = static_cast<u64>(result.exact_cache_sets) *
      exact_bytes_per_set;
  }
  result.explicit_bytes = result.fixed_bytes + result.cache_total_bytes +
    result.exact_cache_total_bytes;
  result.fits = result.explicit_bytes <= request.usable_bytes;
  return result;
}

}  // namespace gpu_search::memory_budget
