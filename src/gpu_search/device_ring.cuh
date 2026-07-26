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

__device__ __forceinline__ unsigned long long device_ring_load_acquire(
    const unsigned long long* address) {
  unsigned long long value = 0;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ void device_ring_store_release(
    unsigned long long* address, unsigned long long value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
}

template <class T>
__device__ __forceinline__ bool device_ring_try_pop(DeviceRingView<T> ring, T& value) {
  const unsigned long long position = __ldcg(ring.dequeue_position);
  const unsigned int slot = static_cast<unsigned int>(position) & ring.mask;
  const unsigned long long sequence = device_ring_load_acquire(ring.sequences + slot);
  bool claimed = false;
  if (sequence == position + 1ULL) {
    claimed = atomicCAS(ring.dequeue_position, position, position + 1ULL) == position;
    if (claimed) {
      value = ring.entries[slot];
      device_ring_store_release(ring.sequences + slot, position + ring.capacity);
    }
  }
  return claimed;
}

template <class T>
__device__ __forceinline__ bool device_ring_try_push(DeviceRingView<T> ring,
                                                     const T& value) {
  const unsigned long long position = atomicAdd(ring.enqueue_position, 0ULL);
  const unsigned int slot = static_cast<unsigned int>(position) & ring.mask;
  if (device_ring_load_acquire(ring.sequences + slot) != position) return false;
  if (atomicCAS(ring.enqueue_position, position, position + 1ULL) != position) {
    return false;
  }
  ring.entries[slot] = value;
  device_ring_store_release(ring.sequences + slot, position + 1ULL);
  return true;
}

template <class T>
__device__ __forceinline__ bool device_ring_is_full(DeviceRingView<T> ring) {
  const unsigned long long enqueue = atomicAdd(
    ring.enqueue_position, 0ULL);
  const unsigned long long dequeue = atomicAdd(
    ring.dequeue_position, 0ULL);
  return enqueue - dequeue >= ring.capacity;
}

template <class T>
__device__ __forceinline__ void device_ring_push(DeviceRingView<T> ring, const T& value) {
  const unsigned long long position = atomicAdd(ring.enqueue_position, 1ULL);
  const unsigned int slot = static_cast<unsigned int>(position) & ring.mask;
  while (device_ring_load_acquire(ring.sequences + slot) != position) {
    device_ring_relax();
  }
  ring.entries[slot] = value;
  device_ring_store_release(ring.sequences + slot, position + 1ULL);
}

#endif

}  // namespace gpu_search
