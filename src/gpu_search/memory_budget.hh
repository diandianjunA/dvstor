#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include "gpu_search/persistent_kernel.hh"

namespace gpu_search::memory_budget {

struct Request {
  u64 nodes{};
  u64 max_delta_vectors{};
  u64 usable_bytes{};
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
};

struct Result {
  u64 code_bytes{};
  u64 delta_bytes{};
  u64 delta_code_bytes{};
  u64 query_workspace_bytes{};
  u64 exact_bytes{};
  u64 metadata_bytes{};
  u64 permanent_override_bytes{};
  u64 fixed_bytes{};
  u64 explicit_bytes{};
  u32 delta_capacity{};
  u32 delta_table_capacity{};
  u32 visited_capacity{};
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
       code_bytes + 3 * sizeof(u32)) +
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

inline u64 resident_pq_footprint(u32 capacity, u32 code_bytes) {
  if (capacity == 0 || code_bytes == 0) return 0;
  const u32 table_capacity = next_power_of_two(static_cast<u64>(capacity) * 2);
  return static_cast<u64>(capacity) * (code_bytes + sizeof(u32)) +
    static_cast<u64>(table_capacity) * (sizeof(u64) + sizeof(u32));
}

inline u32 choose_resident_pq_capacity(u64 budget, u64 max_vectors,
                                       u32 code_bytes) {
  if (budget == 0 || max_vectors == 0 || code_bytes == 0) return 0;
  u32 low = 1;
  u32 high = static_cast<u32>(std::min<u64>(
    std::min<u64>(max_vectors, kDeltaHandleMask), budget / code_bytes));
  if (high == 0) return 0;
  while (low < high) {
    const u32 middle = low + (high - low + 1) / 2;
    if (resident_pq_footprint(middle, code_bytes) <= budget) low = middle;
    else high = middle - 1;
  }
  return resident_pq_footprint(low, code_bytes) <= budget ? low : 0;
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
      request.exact_record_bytes == 0) {
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
    static_cast<u64>(request.query_slots) * result.visited_capacity * sizeof(u32);
  result.exact_bytes = static_cast<u64>(request.query_slots) * request.exact_width *
    request.exact_record_bytes;
  result.metadata_bytes = static_cast<u64>(request.shard_count) *
      sizeof(DeviceShardRegion) +
    (static_cast<u64>(request.dim) * request.dim +
     static_cast<u64>(request.dim) * 256) * sizeof(f32) +
    static_cast<u64>(request.entry_point_count) * sizeof(u32) +
    static_cast<u64>(request.anchor_count) * request.dim * sizeof(f32) +
    static_cast<u64>(request.anchor_count) * sizeof(u32) +
    static_cast<u64>(request.anchor_count) * request.code_bytes +
    (64ull << 20);
  result.permanent_override_bytes =
    ((request.nodes + 31) / 32) * sizeof(u32);
  result.fixed_bytes = result.code_bytes + result.delta_bytes +
    result.query_workspace_bytes + result.exact_bytes + result.metadata_bytes +
    result.permanent_override_bytes;
  result.explicit_bytes = result.fixed_bytes;
  result.fits = result.explicit_bytes <= request.usable_bytes;
  return result;
}

}  // namespace gpu_search::memory_budget
