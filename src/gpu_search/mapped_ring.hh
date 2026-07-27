#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "common/types.hh"
#include "gpu_search/device_ring.cuh"

namespace gpu_search {

template <class T>
class MappedRing {
public:
  enum class Direction {
    host_to_device,
    device_to_host,
  };

  MappedRing(u32 requested_capacity, Direction direction)
      : capacity_(normalize_capacity(requested_capacity)) {
    try {
      allocate_mapped(&enqueue_host_, 1, "cudaHostAlloc(ring enqueue)");
      allocate_mapped(&dequeue_host_, 1, "cudaHostAlloc(ring dequeue)");
      allocate_mapped(&sequences_host_, capacity_, "cudaHostAlloc(ring sequences)");
      allocate_mapped(&entries_host_, capacity_, "cudaHostAlloc(ring entries)");

      *enqueue_host_ = 0;
      *dequeue_host_ = 0;
      for (u32 index = 0; index < capacity_; ++index) sequences_host_[index] = index;

      u64* enqueue_device = device_pointer(
        enqueue_host_, "cudaHostGetDevicePointer(ring enqueue)");
      u64* dequeue_device = device_pointer(
        dequeue_host_, "cudaHostGetDevicePointer(ring dequeue)");
      u64* sequences_device = device_pointer(
        sequences_host_, "cudaHostGetDevicePointer(ring sequences)");
      T* entries_device = device_pointer(
        entries_host_, "cudaHostGetDevicePointer(ring entries)");

      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_owned_position_), sizeof(u64)),
                 "cudaMalloc(ring device position)");
      check_cuda(cudaMemset(device_owned_position_, 0, sizeof(u64)),
                 "cudaMemset(ring device position)");
      if (direction == Direction::host_to_device) {
        dequeue_device = device_owned_position_;
      } else {
        enqueue_device = device_owned_position_;
      }
      device_view_ = {
        .enqueue_position = reinterpret_cast<unsigned long long*>(enqueue_device),
        .dequeue_position = reinterpret_cast<unsigned long long*>(dequeue_device),
        .sequences = reinterpret_cast<unsigned long long*>(sequences_device),
        .entries = entries_device,
        .capacity = capacity_,
        .mask = capacity_ - 1,
      };
    } catch (...) {
      release();
      throw;
    }
  }

  ~MappedRing() { release(); }

  MappedRing(const MappedRing&) = delete;
  MappedRing& operator=(const MappedRing&) = delete;

  bool try_push(const T& value) {
    std::atomic_ref<u64> enqueue(*enqueue_host_);
    const u64 position = enqueue.load(std::memory_order_relaxed);
    const u32 slot = static_cast<u32>(position) & (capacity_ - 1);
    std::atomic_ref<u64> sequence(sequences_host_[slot]);
    if (sequence.load(std::memory_order_acquire) != position) return false;
    entries_host_[slot] = value;
    sequence.store(position + 1, std::memory_order_release);
    enqueue.store(position + 1, std::memory_order_release);
    return true;
  }

  bool try_pop(T& value) {
    std::atomic_ref<u64> dequeue(*dequeue_host_);
    const u64 position = dequeue.load(std::memory_order_relaxed);
    const u32 slot = static_cast<u32>(position) & (capacity_ - 1);
    std::atomic_ref<u64> sequence(sequences_host_[slot]);
    if (sequence.load(std::memory_order_acquire) != position + 1) return false;
    value = entries_host_[slot];
    sequence.store(position + capacity_, std::memory_order_release);
    dequeue.store(position + 1, std::memory_order_release);
    return true;
  }

  DeviceRingView<T> device_view() const { return device_view_; }

private:
  void release() noexcept {
    if (device_owned_position_ != nullptr) cudaFree(device_owned_position_);
    if (entries_host_ != nullptr) cudaFreeHost(entries_host_);
    if (sequences_host_ != nullptr) cudaFreeHost(sequences_host_);
    if (dequeue_host_ != nullptr) cudaFreeHost(dequeue_host_);
    if (enqueue_host_ != nullptr) cudaFreeHost(enqueue_host_);
    device_owned_position_ = nullptr;
    entries_host_ = nullptr;
    sequences_host_ = nullptr;
    dequeue_host_ = nullptr;
    enqueue_host_ = nullptr;
  }

  static u32 normalize_capacity(u32 requested) {
    if (requested >= (1u << 31)) return 1u << 31;
    return std::max<u32>(2, std::bit_ceil(requested));
  }

  static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string(operation) + ": " +
                               cudaGetErrorString(status));
    }
  }

  template <class U>
  static void allocate_mapped(U** pointer, size_t count, const char* operation) {
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(pointer), count * sizeof(U),
                             cudaHostAllocMapped),
               operation);
  }

  template <class U>
  static U* device_pointer(U* host_pointer, const char* operation) {
    U* result = nullptr;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&result),
                                        host_pointer, 0),
               operation);
    return result;
  }

  u32 capacity_{};
  u64* enqueue_host_{};
  u64* dequeue_host_{};
  u64* sequences_host_{};
  T* entries_host_{};
  u64* device_owned_position_{};
  DeviceRingView<T> device_view_{};
};

}  // namespace gpu_search
