#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include "gpu_search/persistent_kernel.hh"

namespace gpu_search::memory_budget {

struct Request {
  u64 nodes{};
  u64 usable_bytes{};
  u32 dim{};
  u32 pq_subquantizers{};
  u32 code_bytes{};
  u32 query_slots{};
  u32 beam_width{};
  u32 graph_degree{};
  u32 exact_width{};
  u32 exact_record_bytes{};
  u32 shard_count{};
};

struct Result {
  u64 code_bytes{};
  u64 query_workspace_bytes{};
  u64 exact_bytes{};
  u64 metadata_bytes{};
  u64 fixed_bytes{};
  u64 explicit_bytes{};
  u32 visited_capacity{};
  bool fits{};
};

inline u32 next_power_of_two(u64 value) {
  if (value >= (1u << 31)) return 1u << 31;
  return std::max<u32>(2, std::bit_ceil(static_cast<u32>(value)));
}

inline bool checked_add(u64 lhs, u64 rhs, u64& result) {
  if (rhs > std::numeric_limits<u64>::max() - lhs) return false;
  result = lhs + rhs;
  return true;
}

inline bool checked_multiply(u64 lhs, u64 rhs, u64& result) {
  if (lhs != 0 && rhs > std::numeric_limits<u64>::max() / lhs) return false;
  result = lhs * rhs;
  return true;
}

template <typename... Factors>
inline bool checked_product(u64& result, Factors... factors) {
  result = 1;
  return (checked_multiply(result, static_cast<u64>(factors), result) && ...);
}

template <typename... Terms>
inline bool checked_sum(u64& result, Terms... terms) {
  result = 0;
  return (checked_add(result, static_cast<u64>(terms), result) && ...);
}

inline Result estimate(const Request& request) {
  Result result;
  if (request.nodes == 0 || request.nodes >= (1ull << 30) || request.dim == 0 ||
      request.pq_subquantizers == 0 || request.code_bytes == 0 ||
      request.code_bytes != request.pq_subquantizers ||
      request.dim % request.pq_subquantizers != 0 ||
      request.query_slots == 0 || request.beam_width == 0 ||
      request.graph_degree == 0 || request.exact_width == 0 ||
      request.exact_record_bytes == 0) {
    return result;
  }
  u64 visited_items = 0;
  if (!checked_product(visited_items, request.beam_width,
                       request.graph_degree, 8)) {
    return {};
  }
  result.visited_capacity = next_power_of_two(
    std::max<u64>(256, visited_items));

  u64 decoded_queries = 0;
  u64 transformed_queries = 0;
  u64 query_luts = 0;
  u64 visited_bytes = 0;
  u64 shard_metadata = 0;
  u64 opq_matrix = 0;
  u64 pq_centroids = 0;
  if (!checked_product(result.code_bytes, request.nodes,
                       request.code_bytes) ||
      !checked_product(decoded_queries, request.query_slots,
                       request.dim, sizeof(f32)) ||
      !checked_product(transformed_queries, request.query_slots,
                       request.dim, sizeof(f32)) ||
      !checked_product(query_luts, request.query_slots,
                       request.pq_subquantizers, 256, sizeof(f32)) ||
      !checked_product(visited_bytes, request.query_slots,
                       result.visited_capacity, sizeof(u64)) ||
      !checked_sum(result.query_workspace_bytes, decoded_queries,
                   transformed_queries, query_luts, visited_bytes) ||
      !checked_product(result.exact_bytes, request.query_slots,
                       request.exact_width, request.exact_record_bytes) ||
      !checked_product(shard_metadata, request.shard_count,
                       sizeof(DeviceShardRegion)) ||
      !checked_product(opq_matrix, request.dim, request.dim, sizeof(f32)) ||
      !checked_product(pq_centroids, request.dim, 256, sizeof(f32)) ||
      !checked_sum(result.metadata_bytes, shard_metadata, opq_matrix,
                   pq_centroids, 64ull << 20) ||
      !checked_sum(result.fixed_bytes, result.code_bytes,
                   result.query_workspace_bytes, result.exact_bytes,
                   result.metadata_bytes)) {
    return {};
  }
  result.explicit_bytes = result.fixed_bytes;
  result.fits = result.explicit_bytes <= request.usable_bytes;
  return result;
}

}  // namespace gpu_search::memory_budget
