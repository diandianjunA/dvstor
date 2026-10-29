#pragma once

#include "gpu_search/persistent_kernel.hh"
#include "gpu_search/persistent_kernel/warp_bitonic_sort.cuh"

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>

#include <algorithm>
#include <cfloat>
#include <cerrno>
#include <cmath>
#include <cstdint>

#ifdef DVSTOR_HAVE_GPUNETIO
#if !defined(IBV_WC_DRIVER1) && !defined(DVSTOR_HAVE_IBV_WC_DRIVER1)
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

#if DVSTOR_PERSISTENT_WIDE_SORT
using ApproximateBlockSortWide = cub::BlockRadixSort<
  f32, kApproximateSortThreadsWide, kApproximateSortItemsWide, u64>;
using ApproximateBlockSortWideRun = cub::BlockRadixSort<
  f32, kApproximateSortThreadsWide, kApproximateSortItemsWideRun, u64>;
#endif
#if DVSTOR_PERSISTENT_LEGACY_MERGE
using ApproximateBlockSortCompactPass = cub::BlockRadixSort<
  f32, kApproximateSortThreadsCompact, kApproximateSortItemsCompactPass, u64>;
using ApproximateBlockSortCompactFinal = cub::BlockRadixSort<
  f32, kApproximateSortThreadsCompact, kApproximateSortItemsCompactFinal, u64>;
using ApproximateBlockSortCompactFinal256 = cub::BlockRadixSort<
  f32, kApproximateSortThreadsCompact,
  kApproximateSortItemsCompactFinal256, u64>;
#endif
// Four physical warps own the four immutable 512-item Stable-Run leaves.
// Sorting an ordered (distance, raw ordinal) key and resolving the handle
// afterwards reproduces the exact stable input order, including
// equal-distance candidates.  The register bitonic network in
// warp_bitonic_sort.cuh performs this sort entirely in registers, so the
// leaves no longer carry a per-warp CUB TempStorage (4,104 B each) in the
// CTA's shared-memory budget.
inline constexpr u32 kWarpLeafSortWarps = 4;
inline constexpr u32 kWarpLeafSortThreads = 32;
inline constexpr u32 kWarpLeafSortItemsPerThread = 16;
inline constexpr u32 kWarpLeafSortCapacity =
  kWarpLeafSortThreads * kWarpLeafSortItemsPerThread;

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
