#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>


struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace gpu {

class GpuNodeCache {
public:
  struct AdmissionResult {
    bool enqueued{};
    bool evicted{};
    bool skipped{};
  };

  struct StatsSnapshot {
    uint64_t hits{};
    uint64_t misses{};
    uint64_t admissions{};
    uint64_t evictions{};
    uint64_t fill_skips{};
  };

  GpuNodeCache() = default;
  ~GpuNodeCache();

  GpuNodeCache(const GpuNodeCache&) = delete;
  GpuNodeCache& operator=(const GpuNodeCache&) = delete;

  bool init(size_t cache_bytes, size_t vector_bytes);
  void destroy();

  bool enabled() const { return enabled_; }
  size_t slot_count() const { return slot_count_; }
  size_t cache_bytes() const { return slot_count_ * vector_bytes_; }
  size_t vector_bytes() const { return vector_bytes_; }

  bool lookup(uint64_t key, const void** device_ptr);
  AdmissionResult admit_from_device(uint64_t key, const void* source_device_ptr, cudaStream_t stream);

  StatsSnapshot stats() const;
  void complete_admission(size_t slot, bool eviction);

private:
  static constexpr uint32_t kWays = 4;
  static constexpr uint32_t kEmpty = 0;
  static constexpr uint32_t kLoading = 1;
  static constexpr uint32_t kValid = 2;

  struct Slot {
    std::atomic<uint64_t> key{0};
    std::atomic<uint32_t> state{kEmpty};
    std::atomic<uint64_t> last_access{0};
  };

  size_t set_index(uint64_t key) const;
  uint8_t* slot_ptr(size_t slot) const;
  bool reserve_slot(uint64_t key, size_t* slot_out, bool* eviction_out);

  uint8_t* d_vectors_{nullptr};
  std::unique_ptr<Slot[]> slots_;
  std::unique_ptr<std::mutex[]> set_locks_;
  size_t slot_count_{0};
  size_t set_count_{0};
  size_t vector_bytes_{0};
  bool enabled_{false};

  std::atomic<uint64_t> epoch_{1};
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
  std::atomic<uint64_t> admissions_{0};
  std::atomic<uint64_t> evictions_{0};
  std::atomic<uint64_t> fill_skips_{0};
};

}  // namespace gpu
