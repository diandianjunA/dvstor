#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct ibv_mr;
struct ibv_pd;

namespace gpu {

struct GpuRabitqCacheResolve {
  bool ok{false};
  uint32_t n{0};
  uint32_t miss_count{0};
  uint32_t hit_count{0};
  uint32_t duplicate_loading_count{0};
};

class GpuRabitqCache {
public:
  GpuRabitqCache() = default;
  ~GpuRabitqCache();

  GpuRabitqCache(const GpuRabitqCache&) = delete;
  GpuRabitqCache& operator=(const GpuRabitqCache&) = delete;

  bool init(size_t bytes, uint32_t stride, ibv_pd* pd);
  void destroy();

  bool enabled() const { return enabled_; }
  uint8_t* base() const { return static_cast<uint8_t*>(pool_); }
  uint32_t stride() const { return stride_; }
  uint32_t lkey() const { return lkey_; }
  size_t slot_count() const { return slot_count_; }

  GpuRabitqCacheResolve resolve_batch(const void* remote_ptrs,
                                      uint32_t n,
                                      uint32_t* out_slot_ids,
                                      std::vector<uint32_t>& miss_indices,
                                      std::vector<uint32_t>& miss_slots,
                                      std::vector<uint64_t>& miss_addrs);
  void publish_batch(const std::vector<uint32_t>& slots);
  void acquire_slots(const uint32_t* slots, uint32_t n);
  void release_slots(const uint32_t* slots, uint32_t n);
  void rollback_loading(const std::vector<uint32_t>& slots);

private:
  enum class State : uint8_t { empty = 0, loading = 1, ready = 2 };
  static constexpr uint32_t kInvalidSlot = UINT32_MAX;

  struct Entry {
    uint64_t key{0};
    uint32_t slot{kInvalidSlot};
  };

  static size_t next_power_of_two(size_t value);
  static uint64_t hash(uint64_t key);

  bool lookup(uint64_t key, uint32_t& slot) const;
  bool insert_entry(uint64_t key, uint32_t slot);
  void remove_entry(uint64_t key);
  void rehash_cluster(size_t pos);
  bool allocate_slot(uint32_t& slot);
  void free_slot(uint32_t slot);
  uint64_t slot_addr(uint32_t slot) const;

  bool enabled_{false};
  void* pool_{nullptr};
  ibv_mr* mr_{nullptr};
  uint32_t lkey_{0};
  uint32_t stride_{0};
  size_t slot_count_{0};
  size_t table_capacity_{0};
  uint32_t next_slot_{0};
  uint32_t clock_hand_{0};

  std::vector<Entry> table_;
  std::vector<uint64_t> slot_keys_;
  std::vector<State> slot_states_;
  std::vector<uint8_t> slot_refs_;
  std::vector<uint32_t> slot_use_counts_;
};

}  // namespace gpu
