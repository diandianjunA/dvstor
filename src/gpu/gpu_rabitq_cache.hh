#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

struct ibv_mr;
struct ibv_pd;

namespace gpu {

struct GpuRabitqCacheResolve {
  bool ok{false};
  uint32_t n{0};
  uint32_t hit_count{0};
  uint32_t fill_count{0};
  uint32_t inflight_fallback_count{0};
  uint32_t duplicate_loading_count{0};
};

struct GpuRabitqTileTask {
  uint32_t tile_id{0};
  uint32_t start{0};
  uint32_t count{0};
};

struct GpuRabitqCacheTiledResolve {
  GpuRabitqCacheResolve base{};
  std::vector<GpuRabitqTileTask> hit_tasks;
  std::vector<uint32_t> task_offsets;
  std::vector<uint32_t> task_candidate_indices;
  uint32_t unique_tiles{0};
};

class GpuRabitqCache {
public:
  GpuRabitqCache() = default;
  ~GpuRabitqCache();

  GpuRabitqCache(const GpuRabitqCache&) = delete;
  GpuRabitqCache& operator=(const GpuRabitqCache&) = delete;

  bool init(size_t bytes, uint32_t stride, ibv_pd* pd,
            const char* mode = "slot_clock",
            uint32_t tile_slots = 32,
            double nursery_ratio = 0.25,
            uint32_t promotion_threshold = 2,
            bool enable_promotion = false,
            bool enable_value_bin = false,
            bool enable_hit_tile_grouping = true);
  void destroy();

  bool enabled() const { return enabled_; }
  bool gentile_enabled() const { return enabled_ && mode_ == Mode::gentile; }
  bool hit_tile_grouping_enabled() const { return gentile_enabled() && enable_hit_tile_grouping_; }
  uint8_t* base() const { return static_cast<uint8_t*>(pool_); }
  uint32_t stride() const { return stride_; }
  uint32_t tile_slots() const { return tile_slots_; }
  uint32_t lkey() const { return lkey_; }
  size_t slot_count() const { return slot_count_; }

  GpuRabitqCacheResolve resolve_batch(const void* remote_ptrs,
                                      uint32_t n,
                                      uint32_t* out_slot_ids,
                                      std::vector<uint32_t>& fill_indices,
                                      std::vector<uint32_t>& fill_slots,
                                      std::vector<uint64_t>& fill_addrs,
                                      std::vector<uint32_t>& inflight_indices);
  GpuRabitqCacheTiledResolve resolve_batch_tiled(const void* remote_ptrs,
                                                 uint32_t n,
                                                 uint32_t* out_slot_ids,
                                                 std::vector<uint32_t>& fill_indices,
                                                 std::vector<uint32_t>& fill_slots,
                                                 std::vector<uint64_t>& fill_addrs,
                                                 std::vector<uint32_t>& inflight_indices);
  void publish_batch(const std::vector<uint32_t>& slots);
  void acquire_slots(const uint32_t* slots, uint32_t n);
  void release_slots(const uint32_t* slots, uint32_t n);
  void rollback_loading(const std::vector<uint32_t>& slots);
  uint64_t slot_addr(uint32_t slot) const;

private:
  enum class Mode : uint8_t { off = 0, slot_clock = 1, gentile = 2 };
  enum class Region : uint8_t { nursery = 0, hot = 1 };
  enum class State : uint8_t { empty = 0, loading = 1, ready = 2, stale = 3 };
  static constexpr uint32_t kInvalidSlot = UINT32_MAX;

  struct Entry {
    uint64_t key{0};
    uint32_t slot{kInvalidSlot};
    uint32_t tile_id{0};
    uint32_t offset{0};
    uint32_t generation{0};
    Region region{Region::nursery};
    State state{State::empty};
    uint8_t credit{0};
  };

  static size_t next_power_of_two(size_t value);
  static uint64_t hash(uint64_t key);

  bool lookup(uint64_t key, uint32_t& slot) const;
  Entry* lookup_entry(uint64_t key);
  bool insert_entry(uint64_t key, uint32_t slot);
  bool insert_entry(const Entry& entry);
  bool allocate_gentile_nursery_slot(uint32_t& slot, uint32_t& tile_id, uint32_t& offset, uint32_t& generation);
  void append_tile_task(uint32_t tile_id, uint32_t offset, uint32_t candidate_index,
                        GpuRabitqCacheTiledResolve& result);
  void remove_entry(uint64_t key);
  void rehash_cluster(size_t pos);
  bool allocate_slot(uint32_t& slot);
  void free_slot(uint32_t slot);

  Mode mode_{Mode::slot_clock};
  bool enabled_{false};
  void* pool_{nullptr};
  ibv_mr* mr_{nullptr};
  uint32_t lkey_{0};
  uint32_t stride_{0};
  size_t slot_count_{0};
  size_t table_capacity_{0};
  uint32_t next_slot_{0};
  uint32_t clock_hand_{0};
  uint32_t tile_slots_{32};
  uint32_t tile_count_{0};
  uint32_t nursery_tile_count_{0};
  uint32_t active_nursery_tile_{0};
  uint32_t active_nursery_offset_{0};
  uint32_t promotion_threshold_{2};
  bool enable_promotion_{false};
  bool enable_value_bin_{false};
  bool enable_hit_tile_grouping_{true};

  std::vector<uint32_t> tile_generations_;
  std::vector<Entry> table_;
  std::vector<uint64_t> slot_keys_;
  std::vector<State> slot_states_;
  std::vector<uint8_t> slot_refs_;
  std::vector<uint32_t> slot_use_counts_;
  mutable std::mutex mutex_;
  size_t evictions_{0};
};

}  // namespace gpu
