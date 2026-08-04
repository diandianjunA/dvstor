#pragma once

#include "gpu_search/persistent_kernel.hh"

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <algorithm>
#include <cfloat>
#include <cerrno>
#include <cmath>
#include <cstdint>

#ifdef DVSTOR_HAVE_GPUNETIO
#ifndef IBV_WC_DRIVER1
#define IBV_WC_DRIVER1 135
#define IBV_WC_DRIVER2 136
#define IBV_WC_DRIVER3 137
#endif
#include <doca_gpunetio_dev_verbs_onesided.cuh>
#endif

namespace gpu_search::persistent_kernel_detail {

inline constexpr u32 kApproximateSortThreadsWide = 256;
inline constexpr u32 kApproximateSortItemsWide =
  kPersistentMaxMergeCandidates / kApproximateSortThreadsWide;
inline constexpr u32 kApproximateSortThreadsCompact = 128;
inline constexpr u32 kApproximateSortItemsWideRun = 4;
inline constexpr u32 kApproximateSortItemsCompactPass = 8;
inline constexpr u32 kApproximateSortItemsCompactFinal = 2;
inline constexpr u32 kApproximateSortItemsCompactFinal256 = 4;
inline constexpr u32 kApproximateSortCapacityWide =
  kApproximateSortThreadsWide * kApproximateSortItemsWide;
inline constexpr u32 kApproximateSortCapacityCompactPass =
  kApproximateSortThreadsCompact * kApproximateSortItemsCompactPass;

using ApproximateBlockSortWide = cub::BlockRadixSort<
  f32, kApproximateSortThreadsWide, kApproximateSortItemsWide, u64>;
using ApproximateBlockSortWideRun = cub::BlockRadixSort<
  f32, kApproximateSortThreadsWide, kApproximateSortItemsWideRun, u64>;
using ApproximateBlockSortCompactPass = cub::BlockRadixSort<
  f32, kApproximateSortThreadsCompact, kApproximateSortItemsCompactPass, u64>;
using ApproximateBlockSortCompactFinal = cub::BlockRadixSort<
  f32, kApproximateSortThreadsCompact, kApproximateSortItemsCompactFinal, u64>;
using ApproximateBlockSortCompactFinal256 = cub::BlockRadixSort<
  f32, kApproximateSortThreadsCompact,
  kApproximateSortItemsCompactFinal256, u64>;
// Four physical warps own the four immutable 512-item Stable-Run leaves.
// Sorting an ordered (distance, raw ordinal) key and resolving the handle
// afterwards avoids CUB's key/value merge path while retaining exact stable
// input order, including equal-distance candidates.
inline constexpr u32 kWarpLeafSortWarps = 4;
inline constexpr u32 kWarpLeafSortThreads = 32;
inline constexpr u32 kWarpLeafSortItemsPerThread = 16;
inline constexpr u32 kWarpLeafSortCapacity =
  kWarpLeafSortThreads * kWarpLeafSortItemsPerThread;
using ApproximateWarpLeafSort = cub::WarpMergeSort<
  u64, kWarpLeafSortItemsPerThread, kWarpLeafSortThreads>;

struct ApproximateWarpLeafSortStorage {
  ApproximateWarpLeafSort::TempStorage leaves[kWarpLeafSortWarps];
};

struct OrderedU64Less {
  __device__ __forceinline__ bool operator()(u64 lhs, u64 rhs) const {
    return lhs < rhs;
  }
};

// The 1024-item pass remains the largest compact-sort temporary.  Supporting
// the 512-item final merge therefore does not increase per-query shared memory
// (and cannot reduce the resident beam-128 CTA count).
static_assert(sizeof(ApproximateBlockSortCompactFinal256::TempStorage) <=
              sizeof(ApproximateBlockSortCompactPass::TempStorage));
static_assert(sizeof(ApproximateBlockSortWideRun::TempStorage) <=
              sizeof(ApproximateBlockSortWide::TempStorage));
static_assert(kApproximateSortThreadsWide *
                kApproximateSortItemsWideRun ==
              kApproximateSortCapacityCompactPass);
static_assert(kApproximateSortCapacityWide ==
              kPersistentMaxMergeCandidates);
static_assert(kApproximateSortCapacityCompactPass * 2 ==
              kPersistentMaxMergeCandidates);
static_assert(kWarpLeafSortWarps * kWarpLeafSortCapacity ==
              kPersistentMaxMergeCandidates);

}  // namespace gpu_search::persistent_kernel_detail
