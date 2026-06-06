#include "gpu/gpu_node_cache.hh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>

#define CUDA_CHECK_RET(call)                                                   \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(err));                                   \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace gpu {

GpuNodeCache::~GpuNodeCache() {
  destroy();
}

bool GpuNodeCache::init(size_t cache_bytes, size_t vector_bytes) {
  destroy();
  if (cache_bytes == 0 || vector_bytes == 0) {
    return false;
  }

  vector_bytes_ = vector_bytes;
  slot_count_ = cache_bytes / vector_bytes_;
  slot_count_ = (slot_count_ / kWays) * kWays;
  if (slot_count_ < kWays) {
    slot_count_ = 0;
    vector_bytes_ = 0;
    return false;
  }
  set_count_ = slot_count_ / kWays;

  CUDA_CHECK_RET(cudaMalloc(reinterpret_cast<void**>(&d_vectors_), slot_count_ * vector_bytes_));
  slots_ = std::make_unique<Slot[]>(slot_count_);
  set_locks_ = std::make_unique<std::mutex[]>(set_count_);
  enabled_ = true;
  std::fprintf(stderr, "[GPU node cache] enabled: slots=%zu vector_bytes=%zu cache_bytes=%zu\n",
               slot_count_, vector_bytes_, slot_count_ * vector_bytes_);
  return true;
}

void GpuNodeCache::destroy() {
  if (d_vectors_) {
    cudaDeviceSynchronize();
    cudaFree(d_vectors_);
    d_vectors_ = nullptr;
  }
  slots_.reset();
  set_locks_.reset();
  slot_count_ = 0;
  set_count_ = 0;
  vector_bytes_ = 0;
  enabled_ = false;
}

namespace {
uint64_t hash_key(uint64_t value) {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return value;
}
}

size_t GpuNodeCache::set_index(uint64_t key) const {
  return hash_key(key) % set_count_;
}

uint8_t* GpuNodeCache::slot_ptr(size_t slot) const {
  return d_vectors_ + slot * vector_bytes_;
}

bool GpuNodeCache::lookup(uint64_t key, const void** device_ptr) {
  if (!enabled_ || key == 0) {
    return false;
  }
  const size_t set = set_index(key);
  const size_t begin = set * kWays;
  for (uint32_t way = 0; way < kWays; ++way) {
    Slot& slot = slots_[begin + way];
    if (slot.state.load(std::memory_order_acquire) != kValid) {
      continue;
    }
    if (slot.key.load(std::memory_order_acquire) == key) {
      slot.last_access.store(epoch_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
      if (device_ptr) {
        *device_ptr = slot_ptr(begin + way);
      }
      hits_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }
  misses_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

bool GpuNodeCache::reserve_slot(uint64_t key, size_t* slot_out, bool* eviction_out) {
  const size_t set = set_index(key);
  const size_t begin = set * kWays;

  for (uint32_t way = 0; way < kWays; ++way) {
    Slot& slot = slots_[begin + way];
    const uint32_t state = slot.state.load(std::memory_order_acquire);
    if (state == kValid && slot.key.load(std::memory_order_acquire) == key) {
      *slot_out = begin + way;
      *eviction_out = false;
      return true;
    }
    if (state == kEmpty) {
      uint32_t expected = kEmpty;
      if (slot.state.compare_exchange_strong(expected, kLoading, std::memory_order_acq_rel)) {
        slot.key.store(key, std::memory_order_release);
        *slot_out = begin + way;
        *eviction_out = false;
        return true;
      }
    }
  }

  std::lock_guard<std::mutex> lock(set_locks_[set]);
  for (uint32_t way = 0; way < kWays; ++way) {
    Slot& slot = slots_[begin + way];
    if (slot.state.load(std::memory_order_acquire) == kValid &&
        slot.key.load(std::memory_order_acquire) == key) {
      *slot_out = begin + way;
      *eviction_out = false;
      return true;
    }
  }

  size_t victim = begin;
  uint64_t oldest = std::numeric_limits<uint64_t>::max();
  for (uint32_t way = 0; way < kWays; ++way) {
    Slot& slot = slots_[begin + way];
    const uint32_t state = slot.state.load(std::memory_order_acquire);
    if (state == kEmpty) {
      victim = begin + way;
      oldest = 0;
      break;
    }
    if (state == kLoading) {
      continue;
    }
    const uint64_t access = slot.last_access.load(std::memory_order_relaxed);
    if (access < oldest) {
      oldest = access;
      victim = begin + way;
    }
  }

  Slot& slot = slots_[victim];
  if (slot.state.load(std::memory_order_acquire) == kLoading) {
    return false;
  }
  if (slot.state.exchange(kLoading, std::memory_order_acq_rel) == kValid) {
    *eviction_out = true;
  } else {
    *eviction_out = false;
  }
  slot.key.store(key, std::memory_order_release);
  *slot_out = victim;
  return true;
}

namespace {
struct AdmissionCallbackData {
  GpuNodeCache* cache{};
  size_t slot{};
  bool eviction{};
};

void CUDART_CB admission_complete_callback(void* user_data) {
  auto* data = static_cast<AdmissionCallbackData*>(user_data);
  data->cache->complete_admission(data->slot, data->eviction);
  delete data;
}
}

void GpuNodeCache::complete_admission(size_t slot, bool eviction) {
  slots_[slot].last_access.store(epoch_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
  slots_[slot].state.store(kValid, std::memory_order_release);
  admissions_.fetch_add(1, std::memory_order_relaxed);
  if (eviction) {
    evictions_.fetch_add(1, std::memory_order_relaxed);
  }
}

GpuNodeCache::AdmissionResult GpuNodeCache::admit_from_device(uint64_t key, const void* source_device_ptr, cudaStream_t stream) {
  if (!enabled_ || key == 0 || source_device_ptr == nullptr) {
    fill_skips_.fetch_add(1, std::memory_order_relaxed);
    return AdmissionResult{false, false, true};
  }

  size_t slot = 0;
  bool eviction = false;
  if (!reserve_slot(key, &slot, &eviction)) {
    fill_skips_.fetch_add(1, std::memory_order_relaxed);
    return AdmissionResult{false, false, true};
  }

  cudaError_t err = cudaMemcpyAsync(slot_ptr(slot), source_device_ptr, vector_bytes_, cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    slots_[slot].state.store(kEmpty, std::memory_order_release);
    fill_skips_.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    return AdmissionResult{false, false, true};
  }

  auto* callback_data = new AdmissionCallbackData{this, slot, eviction};
  err = cudaLaunchHostFunc(stream, admission_complete_callback, callback_data);
  if (err != cudaSuccess) {
    delete callback_data;
    slots_[slot].state.store(kEmpty, std::memory_order_release);
    fill_skips_.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
    return AdmissionResult{false, false, true};
  }

  return AdmissionResult{true, eviction, false};
}

GpuNodeCache::StatsSnapshot GpuNodeCache::stats() const {
  return StatsSnapshot{
    hits_.load(std::memory_order_relaxed),
    misses_.load(std::memory_order_relaxed),
    admissions_.load(std::memory_order_relaxed),
    evictions_.load(std::memory_order_relaxed),
    fill_skips_.load(std::memory_order_relaxed),
  };
}

}  // namespace gpu
