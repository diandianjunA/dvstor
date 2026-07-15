#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "memory_node/storage_owner_index/robust_prune_policy.hh"

namespace memory_node_storage_owner_index_detail {

// Algorithm-only reference for the partitioned insertion semantics.  It is
// deliberately independent of RPCs, queues, retries, and graph mutation:
//
//   direct:    complete the same width-L construction search in every
//              partition, merge every final beam, then RobustPrune once;
//   two-stage: save the complete owner beam in stage 1, complete every other
//              partition in stage 2, then perform that same final prune.
//
// Stage-1 temporary outgoing edges are useful for immediate visibility, but
// they are not authoritative and do not participate in the equality claim.
// Direct-vs-staged equality assumes both observe the same logical graph
// snapshot. The production runtime protects the inserted target's generation
// and revalidates final neighbors, but it deliberately does not freeze every
// shard for the duration of stage2. Under concurrent graph changes the claim
// is therefore quiescent/reference equivalence plus eventual cleanup, not a
// byte-for-byte linearizable construction history.
struct PartitionedInsertOracleResult {
  size_t candidate_capacity{};
  vec<PartitionLocalSearchEntry> merged_candidates;
  vec<RemotePtr> final_neighbors;
};

struct PartitionedInsertStage1 {
  u32 owner_partition{};
  u32 partition_count{};
  u32 beam_width{};
  u32 result_limit{};
  f64 alpha{};
  size_t candidate_capacity{};
  vec<PartitionLocalSearchEntry> owner_beam;
  vec<RemotePtr> temporary_neighbors;
};

inline size_t partitioned_insert_candidate_capacity(
    const u32 partition_count, const u32 beam_width) {
  if (partition_count == 0) {
    throw std::invalid_argument(
      "partitioned insertion requires at least one partition");
  }
  if (beam_width == 0) {
    throw std::invalid_argument(
      "partitioned insertion construction width must be positive");
  }
  if (static_cast<size_t>(partition_count) >
      std::numeric_limits<size_t>::max() /
        static_cast<size_t>(beam_width)) {
    throw std::overflow_error(
      "partitioned insertion candidate capacity overflows size_t");
  }
  return static_cast<size_t>(partition_count) * beam_width;
}

inline bool partitioned_insert_candidate_less(
    const PartitionLocalSearchEntry& lhs,
    const PartitionLocalSearchEntry& rhs) {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.rptr.raw_address < rhs.rptr.raw_address;
}

inline void append_partitioned_insert_beam(
    vec<PartitionLocalSearchEntry>& merged,
    const span<const PartitionLocalSearchEntry> beam,
    const u32 beam_width,
    const size_t candidate_capacity) {
  if (beam.size() > beam_width || beam.size() > candidate_capacity ||
      merged.size() > candidate_capacity - beam.size()) {
    throw std::length_error(
      "partitioned insertion exceeded partition_count * construction_width");
  }
  merged.insert(merged.end(), beam.begin(), beam.end());
}

template <typename PairDistance>
vec<RemotePtr> robust_prune_partitioned_insert_candidates(
    vec<PartitionLocalSearchEntry>& merged_candidates,
    const u32 result_limit,
    const f64 alpha,
    PairDistance&& pair_distance) {
  std::sort(merged_candidates.begin(), merged_candidates.end(),
            partitioned_insert_candidate_less);

  vec<RemotePtr> selected;
  vec<size_t> selected_indices;
  select_alpha_robust_pruned_sorted(
    span<const PartitionLocalSearchEntry>{merged_candidates.data(),
                                          merged_candidates.size()},
    result_limit,
    alpha,
    [](const PartitionLocalSearchEntry& candidate) {
      return candidate.rptr;
    },
    [](const PartitionLocalSearchEntry& candidate) {
      return candidate.distance;
    },
    [&](const PartitionLocalSearchEntry& candidate,
        const PartitionLocalSearchEntry& retained) {
      return std::invoke(pair_distance, candidate.rptr, retained.rptr);
    },
    selected,
    selected_indices);
  return selected;
}

template <typename Score, typename Expand, typename PairDistance>
PartitionedInsertOracleResult partitioned_direct_insert_reference(
    const span<const vec<RemotePtr>> entry_points_by_partition,
    const u32 beam_width,
    const u32 result_limit,
    const f64 alpha,
    Score&& score,
    Expand&& expand,
    PairDistance&& pair_distance) {
  const u32 partition_count =
    static_cast<u32>(entry_points_by_partition.size());
  PartitionedInsertOracleResult result;
  result.candidate_capacity =
    partitioned_insert_candidate_capacity(partition_count, beam_width);
  result.merged_candidates.reserve(result.candidate_capacity);

  for (u32 partition = 0; partition < partition_count; ++partition) {
    vec<PartitionLocalSearchEntry> beam =
      partition_local_construction_search(
        span<const RemotePtr>{entry_points_by_partition[partition]},
        partition,
        beam_width,
        score,
        expand);
    append_partitioned_insert_beam(
      result.merged_candidates,
      span<const PartitionLocalSearchEntry>{beam},
      beam_width,
      result.candidate_capacity);
  }

  result.final_neighbors = robust_prune_partitioned_insert_candidates(
    result.merged_candidates, result_limit, alpha, pair_distance);
  return result;
}

template <typename Score, typename Expand, typename PairDistance>
PartitionedInsertStage1 partitioned_two_stage_insert_begin(
    const span<const vec<RemotePtr>> entry_points_by_partition,
    const u32 owner_partition,
    const u32 beam_width,
    const u32 result_limit,
    const f64 alpha,
    Score&& score,
    Expand&& expand,
    PairDistance&& pair_distance) {
  const u32 partition_count =
    static_cast<u32>(entry_points_by_partition.size());
  if (owner_partition >= partition_count) {
    throw std::out_of_range("two-stage insertion owner is not a partition");
  }

  PartitionedInsertStage1 stage1;
  stage1.owner_partition = owner_partition;
  stage1.partition_count = partition_count;
  stage1.beam_width = beam_width;
  stage1.result_limit = result_limit;
  stage1.alpha = alpha;
  stage1.candidate_capacity =
    partitioned_insert_candidate_capacity(partition_count, beam_width);
  stage1.owner_beam = partition_local_construction_search(
    span<const RemotePtr>{entry_points_by_partition[owner_partition]},
    owner_partition,
    beam_width,
    score,
    expand);

  // Prune a copy: the complete owner beam is the stage boundary and must not
  // be reduced to the temporary outgoing edge set before stage 2.
  vec<PartitionLocalSearchEntry> temporary_candidates = stage1.owner_beam;
  stage1.temporary_neighbors = robust_prune_partitioned_insert_candidates(
    temporary_candidates, result_limit, alpha, pair_distance);
  return stage1;
}

template <typename Score, typename Expand, typename PairDistance>
PartitionedInsertOracleResult partitioned_two_stage_insert_finalize(
    const PartitionedInsertStage1& stage1,
    const span<const vec<RemotePtr>> entry_points_by_partition,
    Score&& score,
    Expand&& expand,
    PairDistance&& pair_distance) {
  if (entry_points_by_partition.size() != stage1.partition_count) {
    throw std::invalid_argument(
      "two-stage insertion partition set changed between stages");
  }

  PartitionedInsertOracleResult result;
  result.candidate_capacity = stage1.candidate_capacity;
  result.merged_candidates.reserve(result.candidate_capacity);
  append_partitioned_insert_beam(
    result.merged_candidates,
    span<const PartitionLocalSearchEntry>{stage1.owner_beam},
    stage1.beam_width,
    result.candidate_capacity);

  for (u32 partition = 0; partition < stage1.partition_count; ++partition) {
    if (partition == stage1.owner_partition) {
      continue;
    }
    vec<PartitionLocalSearchEntry> beam =
      partition_local_construction_search(
        span<const RemotePtr>{entry_points_by_partition[partition]},
        partition,
        stage1.beam_width,
        score,
        expand);
    append_partitioned_insert_beam(
      result.merged_candidates,
      span<const PartitionLocalSearchEntry>{beam},
      stage1.beam_width,
      result.candidate_capacity);
  }

  result.final_neighbors = robust_prune_partitioned_insert_candidates(
    result.merged_candidates,
    stage1.result_limit,
    stage1.alpha,
    pair_distance);
  return result;
}

}  // namespace memory_node_storage_owner_index_detail
