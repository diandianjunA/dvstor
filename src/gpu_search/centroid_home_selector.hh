#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "common/types.hh"
#include "common/vector_dtype.hh"

namespace gpu_search::centroid_home {

struct ShardSnapshot {
  u64 vector_count{};
  // CentroidRouter keeps exact FP64 sums, but the route representation is
  // deliberately FP32: this is the same representation installed on the GPU.
  // Keeping the immutable CPU snapshot in that representation prevents an
  // insert and a query from selecting different homes after FP64->FP32
  // publication rounding.
  std::vector<f32> centroid;
  u32 live_entry_count{};
};

using Snapshot = std::vector<ShardSnapshot>;

namespace detail {

// Keep the multiversioned hot loop non-throwing.  GCC implements
// target_clones through an IFUNC dispatcher; allowing validation exceptions to
// cross that dispatcher is not reliable on all supported GCC/libstdc++
// combinations.  The public wrapper below validates the complete immutable
// snapshot before entering this loop.
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// Resolve once to an FMA-specialized implementation on capable hosts, where
// std::fma is one instruction per dimension. The default clone retains exact
// fmaf semantics through libm on unusual FMA-less x86 hosts.
__attribute__((target_clones("fma", "default")))
#endif
inline std::optional<u32> select_validated(
    std::span<const f32> query, const Snapshot& snapshot) {
  const size_t dim = query.size();
  std::optional<u32> best_shard;
  f32 best_distance = std::numeric_limits<f32>::infinity();
  for (u32 shard = 0; shard < snapshot.size(); ++shard) {
    const ShardSnapshot& candidate = snapshot[shard];
    if (candidate.vector_count == 0 || candidate.live_entry_count == 0) {
      continue;
    }
    f32 distance = 0.0f;
    for (size_t dimension = 0; dimension < dim; ++dimension) {
      const f32 centroid = candidate.centroid[dimension];
      const f32 difference = query[dimension] - centroid;
      // CUDA uses fmaf for the same left-to-right recurrence. Explicit fused
      // accumulation fixes the CPU reduction tree as part of the routing
      // contract, independent of unrelated compiler optimization settings.
      distance = std::fma(difference, difference, distance);
    }
    if (!floating_value_is_finite(distance) ||
        distance == std::numeric_limits<f32>::max()) {
      f64 wide_distance = 0.0;
      for (size_t dimension = 0; dimension < dim; ++dimension) {
        const f64 difference = static_cast<f64>(query[dimension]) -
          static_cast<f64>(candidate.centroid[dimension]);
        wide_distance = std::fma(
          difference, difference, wide_distance);
      }
      distance = saturate_squared_l2(wide_distance);
    }
    // Iteration is in physical-shard order, so exact ties retain the smaller
    // shard without an epsilon that could invert genuinely close centroids.
    if (!best_shard.has_value() || distance < best_distance ||
        (distance == best_distance && shard < *best_shard)) {
      best_shard = shard;
      best_distance = distance;
    }
  }
  return best_shard;
}

}  // namespace detail

// Fast path for a snapshot that has already crossed the storage publication
// validator.  Query validation remains at the API boundary, but immutable
// centroid dimensions/finiteness are not rescanned on every high-rate insert.
inline std::optional<u32> select_published_snapshot(
    std::span<const f32> query, const Snapshot& snapshot) {
  if (snapshot.empty()) return std::nullopt;
  const size_t dim = snapshot.front().centroid.size();
  if (dim == 0 || query.size() != dim) {
    throw std::invalid_argument("centroid home query dimension mismatch");
  }
  for (f32 value : query) {
    if (!floating_value_is_finite(value)) {
      throw std::invalid_argument("centroid home query must be finite");
    }
  }
  return detail::select_validated(query, snapshot);
}

// Pure CPU policy used by update-home selection. The caller publishes Snapshot
// objects immutably, so this function performs no I/O, locking, or waiting.
inline std::optional<u32> select(
    std::span<const f32> query, const Snapshot& snapshot) {
  if (snapshot.empty()) return std::nullopt;
  const size_t dim = snapshot.front().centroid.size();
  if (dim == 0 || query.size() != dim) {
    throw std::invalid_argument("centroid home query dimension mismatch");
  }
  for (f32 value : query) {
    if (!floating_value_is_finite(value)) {
      throw std::invalid_argument("centroid home query must be finite");
    }
  }
  for (const ShardSnapshot& candidate : snapshot) {
    if (candidate.centroid.size() != dim) {
      throw std::logic_error("centroid home snapshot dimension mismatch");
    }
    if (candidate.vector_count == 0 || candidate.live_entry_count == 0) {
      continue;
    }
    for (f32 value : candidate.centroid) {
      if (!floating_value_is_finite(value)) {
        throw std::logic_error("centroid home snapshot must be finite");
      }
    }
  }
  return detail::select_validated(query, snapshot);
}

}  // namespace gpu_search::centroid_home
