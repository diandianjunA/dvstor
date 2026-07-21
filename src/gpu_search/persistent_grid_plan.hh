#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "gpu_search/types.hh"

namespace gpu_search {

inline constexpr std::array<u32, 2> kPersistentThreadCandidates{128, 256};
inline constexpr u32 kPersistentControlBlocks = 2;
inline constexpr u32 kPersistentWarpWidth = 32;

// All CTAs in the unified persistent grid are non-terminating and cooperate
// through bounded queues.  The complete grid therefore has to fit in the
// device at once; this plan records the proof for one supported CTA size.
struct PersistentGridCandidate {
  u32 threads{};
  u32 hardware_blocks_per_sm{};
  u32 effective_blocks_per_sm{};
  u32 grid_capacity{};
  u32 owner_blocks{};
  u32 query_blocks{};
  u32 total_blocks{};
  u32 resident_query_warps{};

  [[nodiscard]] bool viable() const { return query_blocks != 0; }
};

struct PersistentGridPlan {
  std::array<PersistentGridCandidate, 2> candidates{};
  PersistentGridCandidate selected{};
};

inline u32 checked_persistent_grid_product(u32 lhs, u32 rhs) {
  const u64 result = static_cast<u64>(lhs) * rhs;
  if (result > std::numeric_limits<u32>::max()) {
    throw std::overflow_error("persistent GPU grid size exceeds u32");
  }
  return static_cast<u32>(result);
}

inline PersistentGridCandidate evaluate_persistent_grid_candidate(
    u32 threads, u32 hardware_blocks_per_sm, u32 configured_blocks_per_sm,
    u32 multiprocessor_count, u32 query_slots, u32 direct_owner_warps) {
  if ((threads != kPersistentThreadCandidates[0] &&
       threads != kPersistentThreadCandidates[1]) ||
      configured_blocks_per_sm == 0 || multiprocessor_count == 0 ||
      query_slots == 0 || direct_owner_warps == 0) {
    throw std::invalid_argument("invalid persistent GPU grid input");
  }

  PersistentGridCandidate candidate{
    .threads = threads,
    .hardware_blocks_per_sm = hardware_blocks_per_sm,
    .effective_blocks_per_sm =
      std::min(hardware_blocks_per_sm, configured_blocks_per_sm),
  };
  candidate.grid_capacity = checked_persistent_grid_product(
    multiprocessor_count, candidate.effective_blocks_per_sm);

  const u32 owner_warps_per_block = threads / kPersistentWarpWidth;
  candidate.owner_blocks = direct_owner_warps / owner_warps_per_block +
    (direct_owner_warps % owner_warps_per_block != 0 ? 1u : 0u);
  const u64 fixed_blocks =
    static_cast<u64>(candidate.owner_blocks) + kPersistentControlBlocks;
  if (fixed_blocks >= candidate.grid_capacity) return candidate;

  candidate.query_blocks = std::min(
    query_slots,
    candidate.grid_capacity - static_cast<u32>(fixed_blocks));
  candidate.total_blocks = static_cast<u32>(fixed_blocks) +
    candidate.query_blocks;
  candidate.resident_query_warps = checked_persistent_grid_product(
    candidate.query_blocks, owner_warps_per_block);
  return candidate;
}

inline bool persistent_grid_candidate_better(
    const PersistentGridCandidate& candidate,
    const PersistentGridCandidate& current) {
  if (candidate.viable() != current.viable()) return candidate.viable();
  if (!candidate.viable()) return false;
  if (candidate.resident_query_warps != current.resident_query_warps) {
    return candidate.resident_query_warps > current.resident_query_warps;
  }
  // With equal aggregate query parallelism, the wider CTA reduces the number
  // of strided passes and uses the one-pass 256-thread candidate merge.
  if (candidate.threads != current.threads) {
    return candidate.threads > current.threads;
  }
  return candidate.query_blocks > current.query_blocks;
}

inline PersistentGridPlan plan_persistent_grid(
    const std::array<u32, 2>& hardware_blocks_per_sm,
    u32 configured_blocks_per_sm, u32 multiprocessor_count,
    u32 query_slots, u32 direct_owner_warps) {
  PersistentGridPlan plan;
  for (size_t index = 0; index < plan.candidates.size(); ++index) {
    plan.candidates[index] = evaluate_persistent_grid_candidate(
      kPersistentThreadCandidates[index], hardware_blocks_per_sm[index],
      configured_blocks_per_sm, multiprocessor_count, query_slots,
      direct_owner_warps);
    if (persistent_grid_candidate_better(
          plan.candidates[index], plan.selected)) {
      plan.selected = plan.candidates[index];
    }
  }
  if (!plan.selected.viable()) {
    throw std::runtime_error(
      "GPU cannot keep persistent owner, query, dispatcher, and control CTAs resident");
  }
  return plan;
}

}  // namespace gpu_search
