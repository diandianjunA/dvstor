#pragma once

#include "gpu_search/persistent_kernel.hh"

#include <cuda_runtime.h>

namespace gpu_search::persistent_kernel_detail {

// Register-only warp bitonic sort for u64 keys in blocked arrangement: with
// 32 lanes and ItemsPerThread registers per lane, lane L holds positions
// [L*ItemsPerThread, (L+1)*ItemsPerThread) of a virtual array of
// 32*ItemsPerThread keys.  The network is the canonical bitonic sorter, so
// the result is a plain ascending total order; unique keys make the output
// permutation deterministic and exactly equal to a stable sort by the same
// key.  Cross-lane compare-exchanges use __shfl_xor_sync, so no shared memory
// and no block-level synchronization are needed.  Every lane of the warp must
// execute the call convergently.
template <u32 ItemsPerThread>
__device__ __forceinline__ void warp_bitonic_sort_blocked(
    u64 (&keys)[ItemsPerThread]) {
  constexpr u32 warp_width = 32;
  constexpr u32 capacity = warp_width * ItemsPerThread;
  static_assert((capacity & (capacity - 1u)) == 0,
                "warp bitonic sort capacity must be a power of two");
  static_assert((ItemsPerThread & (ItemsPerThread - 1u)) == 0,
                "items per thread must be a power of two");
  const u32 lane = threadIdx.x % warp_width;
  for (u32 size = 2; size <= capacity; size <<= 1) {
    for (u32 stride = size >> 1; stride != 0; stride >>= 1) {
      if (stride >= ItemsPerThread) {
        // Partner is the same register slot in lane ^ lane_mask.  For these
        // stages size >= 2*ItemsPerThread, so the direction bit and the
        // lower/upper bit are uniform across the lane's registers.
        const u32 lane_mask = stride / ItemsPerThread;
        const bool lower = (lane & lane_mask) == 0;
        const bool ascending = (lane & (size / ItemsPerThread)) == 0;
#pragma unroll
        for (u32 item = 0; item < ItemsPerThread; ++item) {
          const u64 other = __shfl_xor_sync(0xffffffffu, keys[item], lane_mask);
          keys[item] = ascending == lower
                         ? (keys[item] < other ? keys[item] : other)
                         : (keys[item] < other ? other : keys[item]);
        }
      } else {
#pragma unroll
        for (u32 item = 0; item < ItemsPerThread; ++item) {
          if ((item & stride) != 0) continue;
          const u32 partner = item | stride;
          const u64 a = keys[item];
          const u64 b = keys[partner];
          const bool ascending =
            ((lane * ItemsPerThread + item) & size) == 0;
          keys[item] = ascending ? (a < b ? a : b) : (a < b ? b : a);
          keys[partner] = ascending ? (a < b ? b : a) : (a < b ? a : b);
        }
      }
    }
  }
}

// Bitonic merge (half-cleaner network) for a bitonic sequence held in blocked
// warp registers, producing a plain ascending order.  Use after forming a
// bitonic sequence from two sorted runs, e.g. C[i] = min(A[i], B[n-1-i]).
template <u32 ItemsPerThread>
__device__ __forceinline__ void warp_bitonic_merge_blocked(
    u64 (&keys)[ItemsPerThread]) {
  constexpr u32 warp_width = 32;
  constexpr u32 capacity = warp_width * ItemsPerThread;
  static_assert((capacity & (capacity - 1u)) == 0,
                "warp bitonic merge capacity must be a power of two");
  const u32 lane = threadIdx.x % warp_width;
  for (u32 stride = capacity >> 1; stride != 0; stride >>= 1) {
    if (stride >= ItemsPerThread) {
      const u32 lane_mask = stride / ItemsPerThread;
      const bool lower = (lane & lane_mask) == 0;
#pragma unroll
      for (u32 item = 0; item < ItemsPerThread; ++item) {
        const u64 other = __shfl_xor_sync(0xffffffffu, keys[item], lane_mask);
        keys[item] =
          lower ? (keys[item] < other ? keys[item] : other)
                : (keys[item] < other ? other : keys[item]);
      }
    } else {
#pragma unroll
      for (u32 item = 0; item < ItemsPerThread; ++item) {
        if ((item & stride) != 0) continue;
        const u32 partner = item | stride;
        const u64 a = keys[item];
        const u64 b = keys[partner];
        keys[item] = a < b ? a : b;
        keys[partner] = a < b ? b : a;
      }
    }
  }
}

}  // namespace gpu_search::persistent_kernel_detail
