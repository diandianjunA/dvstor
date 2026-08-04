#pragma once

#include "gpu_search/persistent_kernel.hh"

#include <climits>

namespace gpu_search::persistent_kernel_detail {

inline constexpr u32 kFrontierStagingScratchBits = 8;
inline constexpr u32 kFrontierStagingScratchMask =
  (u32{1} << kFrontierStagingScratchBits) - 1u;
inline constexpr u32 kFrontierStagingMaxIssueEpoch =
  UINT32_MAX >> kFrontierStagingScratchBits;

#ifdef __CUDACC__
#define DVSTOR_FRONTIER_STAGING_INLINE __host__ __device__ __forceinline__
#else
#define DVSTOR_FRONTIER_STAGING_INLINE inline constexpr
#endif

DVSTOR_FRONTIER_STAGING_INLINE bool frontier_staging_capacity_valid(
    u32 parent_count, u32 graph_entry_capacity,
    u32 workspace_capacity = kPersistentMaxMergeCandidates) {
  return graph_entry_capacity != 0 &&
    parent_count <= workspace_capacity / graph_entry_capacity;
}

DVSTOR_FRONTIER_STAGING_INLINE u32 frontier_staging_source_index(
    u32 parent, u32 neighbor, u32 graph_entry_capacity) {
  return parent * graph_entry_capacity + neighbor;
}

DVSTOR_FRONTIER_STAGING_INLINE bool frontier_staging_token_encodable(
    u32 issue_epoch, u32 scratch_slot) {
  return issue_epoch <= kFrontierStagingMaxIssueEpoch &&
    scratch_slot <= kFrontierStagingScratchMask;
}

DVSTOR_FRONTIER_STAGING_INLINE u32 make_frontier_staging_token(
    u32 issue_epoch, u32 scratch_slot) {
  return (issue_epoch << kFrontierStagingScratchBits) | scratch_slot;
}

DVSTOR_FRONTIER_STAGING_INLINE u32 frontier_staging_token_epoch(u32 token) {
  return token >> kFrontierStagingScratchBits;
}

DVSTOR_FRONTIER_STAGING_INLINE u32 frontier_staging_token_scratch_slot(
    u32 token) {
  return token & kFrontierStagingScratchMask;
}

DVSTOR_FRONTIER_STAGING_INLINE bool frontier_staging_payload_reusable(
    u32 staged_parent_mask, u32 position, u32 staged_token,
    u64 selected_handle, const FrontierRobEntry& entry,
    u32 graph_record_slot, u32 graph_scratch_bit) {
  if (position >= sizeof(staged_parent_mask) * CHAR_BIT ||
      (staged_parent_mask & (u32{1} << position)) == 0 ||
      !frontier_staging_token_encodable(
        entry.issue_epoch, entry.scratch_slot)) {
    return false;
  }
  return entry.state == static_cast<u8>(FrontierRequestState::committed) &&
    entry.node_handle == selected_handle &&
    staged_token ==
      make_frontier_staging_token(entry.issue_epoch, entry.scratch_slot) &&
    graph_record_slot ==
      (graph_scratch_bit | static_cast<u32>(entry.scratch_slot));
}

#undef DVSTOR_FRONTIER_STAGING_INLINE

}  // namespace gpu_search::persistent_kernel_detail
