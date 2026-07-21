#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "common/types.hh"

namespace gpu_search::persistent_engine_detail {

inline constexpr u32 kDirectBatchQueueCapacity = 64;
inline void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

inline u64 align_up(u64 value, u64 alignment) {
  return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

template <class T>
void device_allocate(T*& pointer, size_t count, const char* operation) {
  if (count == 0) {
    pointer = nullptr;
    return;
  }
  if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(operation) + ": allocation size overflow");
  }
  const size_t bytes = count * sizeof(T);
  const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&pointer), bytes);
  if (status != cudaSuccess) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    (void)cudaMemGetInfo(&free_bytes, &total_bytes);
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status) +
      " requested=" + std::to_string(bytes) +
      " free=" + std::to_string(free_bytes) +
      " total=" + std::to_string(total_bytes));
  }
}

template <class T>
void device_free(T*& pointer) {
  if (pointer != nullptr) cudaFree(pointer);
  pointer = nullptr;
}

template <class T>
void mapped_host_allocate(T*& host_pointer, T*& device_pointer,
                          size_t count, const char* operation) {
  host_pointer = nullptr;
  device_pointer = nullptr;
  if (count == 0) return;
  if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::overflow_error(std::string(operation) + ": allocation size overflow");
  }
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&host_pointer),
                           count * sizeof(T),
                           cudaHostAllocMapped | cudaHostAllocPortable),
             operation);
  check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_pointer),
                                      host_pointer, 0),
             "cudaHostGetDevicePointer(mapped staging)");
}

}  // namespace gpu_search::persistent_engine_detail
