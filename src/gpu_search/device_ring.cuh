#pragma once

#include <cstdint>

namespace gpu_search {

template <class T>
struct DeviceRingView {
  unsigned long long* enqueue_position{};
  unsigned long long* dequeue_position{};
  unsigned long long* sequences{};
  T* entries{};
  unsigned int capacity{};
  unsigned int mask{};
};

#ifdef __CUDACC__

__device__ __forceinline__ void device_ring_relax(unsigned int cycles = 64) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  __nanosleep(cycles);
#else
  for (volatile unsigned int i = 0; i < cycles; ++i) {
    asm volatile("");
  }
#endif
}

template <class T>
__device__ __forceinline__ bool device_ring_try_pop(DeviceRingView<T> ring, T& value) {
  unsigned long long position = atomicAdd(ring.dequeue_position, 0ULL);
  for (;;) {
    const unsigned long long published =
      *reinterpret_cast<volatile unsigned long long*>(ring.enqueue_position);
    if (position >= published) return false;
    const unsigned long long observed = atomicCAS(
      ring.dequeue_position, position, position + 1ULL);
    if (observed == position) break;
    position = observed;
  }

  const unsigned int slot = static_cast<unsigned int>(position) & ring.mask;
  while (*reinterpret_cast<volatile unsigned long long*>(&ring.sequences[slot]) !=
         position + 1ULL) {
    device_ring_relax();
  }
  value = ring.entries[slot];
  __threadfence_system();
  atomicExch(&ring.sequences[slot], position + ring.capacity);
  return true;
}

template <class T>
__device__ __forceinline__ void device_ring_push(DeviceRingView<T> ring, const T& value) {
  const unsigned long long position = atomicAdd(ring.enqueue_position, 1ULL);
  const unsigned int slot = static_cast<unsigned int>(position) & ring.mask;
  while (*reinterpret_cast<volatile unsigned long long*>(&ring.sequences[slot]) != position) {
    device_ring_relax();
  }
  ring.entries[slot] = value;
  __threadfence_system();
  atomicExch(&ring.sequences[slot], position + 1ULL);
}

#endif

}  // namespace gpu_search
