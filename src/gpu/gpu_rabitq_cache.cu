#include "gpu/gpu_rabitq_cache.hh"

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

namespace gpu {

#define CUDA_CHECK_CACHE(call)                                                \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err));                                  \
      return false;                                                           \
    }                                                                         \
  } while (0)

GpuRabitqCache::~GpuRabitqCache() { destroy(); }

bool GpuRabitqCache::init(size_t bytes, uint32_t stride, ibv_pd* pd,
                           const char* mode,
                           uint32_t tile_slots,
                           double nursery_ratio,
                           uint32_t promotion_threshold,
                           bool enable_promotion,
                           bool enable_value_bin,
                           bool enable_hit_tile_grouping) {
  destroy();
  if (bytes == 0 || stride == 0 || pd == nullptr) return false;

  if (mode && std::strcmp(mode, "off") == 0) return false;
  mode_ = (mode && std::strcmp(mode, "gentile") == 0) ? Mode::gentile : Mode::slot_clock;
  stride_ = stride;
  slot_count_ = bytes / stride_;
  if (slot_count_ < 2) return false;
  if (mode_ == Mode::gentile) {
    tile_slots_ = std::max<uint32_t>(1, tile_slots);
    tile_count_ = static_cast<uint32_t>(slot_count_ / tile_slots_);
    if (tile_count_ == 0) return false;
    slot_count_ = static_cast<size_t>(tile_count_) * tile_slots_;
    nursery_tile_count_ = std::max<uint32_t>(1, static_cast<uint32_t>(tile_count_ * nursery_ratio));
    nursery_tile_count_ = std::min<uint32_t>(nursery_tile_count_, tile_count_);
    active_nursery_tile_ = 0;
    active_nursery_offset_ = 0;
    promotion_threshold_ = promotion_threshold;
    enable_promotion_ = enable_promotion;
    enable_value_bin_ = enable_value_bin;
    enable_hit_tile_grouping_ = enable_hit_tile_grouping;
  }

  const size_t pool_bytes = slot_count_ * static_cast<size_t>(stride_);
  CUDA_CHECK_CACHE(cudaMalloc(&pool_, pool_bytes));
  mr_ = ibv_reg_mr(pd, pool_, pool_bytes, IBV_ACCESS_LOCAL_WRITE);
  if (!mr_) {
    cudaFree(pool_);
    pool_ = nullptr;
    std::fprintf(stderr, "[GPU RaBitQ cache] failed to register GPU memory for RDMA\n");
    return false;
  }

  lkey_ = mr_->lkey;
  table_capacity_ = next_power_of_two(slot_count_ * 2);
  table_.assign(table_capacity_, Entry{});
  slot_keys_.assign(slot_count_, 0);
  slot_states_.assign(slot_count_, State::empty);
  slot_refs_.assign(slot_count_, 0);
  slot_use_counts_.assign(slot_count_, 0);
  tile_generations_.assign(tile_count_, 1);
  next_slot_ = 0;
  clock_hand_ = 0;
  enabled_ = true;
  if (mode_ == Mode::gentile) {
    std::fprintf(stderr, "[GPU RaBitQ cache] enabled: mode=gentile slots=%zu stride=%u bytes=%zu tile_slots=%u tiles=%u nursery_tiles=%u grouping=%s\n",
                 slot_count_, stride_, pool_bytes, tile_slots_, tile_count_, nursery_tile_count_,
                 enable_hit_tile_grouping_ ? "on" : "off");
  } else {
    std::fprintf(stderr, "[GPU RaBitQ cache] enabled: mode=slot_clock slots=%zu stride=%u bytes=%zu\n",
                 slot_count_, stride_, pool_bytes);
  }
  return true;
}

void GpuRabitqCache::destroy() {
  if (mr_) {
    ibv_dereg_mr(mr_);
    mr_ = nullptr;
  }
  if (pool_) {
    cudaFree(pool_);
    pool_ = nullptr;
  }
  enabled_ = false;
  lkey_ = 0;
  stride_ = 0;
  slot_count_ = 0;
  table_capacity_ = 0;
  next_slot_ = 0;
  clock_hand_ = 0;
  tile_count_ = 0;
  nursery_tile_count_ = 0;
  active_nursery_tile_ = 0;
  active_nursery_offset_ = 0;
  table_.clear();
  tile_generations_.clear();
  slot_keys_.clear();
  slot_states_.clear();
  slot_refs_.clear();
  slot_use_counts_.clear();
}

GpuRabitqCache::Entry* GpuRabitqCache::lookup_entry(uint64_t key) {
  if (key == 0 || table_capacity_ == 0) return nullptr;
  size_t pos = hash(key) & (table_capacity_ - 1);
  for (size_t probe = 0; probe < table_capacity_; ++probe) {
    auto& entry = table_[pos];
    if (entry.key == 0) return nullptr;
    if (entry.key == key) return &entry;
    pos = (pos + 1) & (table_capacity_ - 1);
  }
  return nullptr;
}

bool GpuRabitqCache::insert_entry(const Entry& entry) {
  if (entry.key == 0 || table_capacity_ == 0) return false;
  size_t pos = hash(entry.key) & (table_capacity_ - 1);
  for (size_t probe = 0; probe < table_capacity_; ++probe) {
    const bool empty = table_[pos].key == 0;
    const bool same_key = table_[pos].key == entry.key;
    const bool stale = mode_ == Mode::gentile && table_[pos].key != 0 &&
                       (table_[pos].tile_id >= tile_generations_.size() ||
                        table_[pos].generation != tile_generations_[table_[pos].tile_id] ||
                        table_[pos].state == State::stale);
    if (empty || same_key || stale) {
      table_[pos] = entry;
      return true;
    }
    pos = (pos + 1) & (table_capacity_ - 1);
  }
  return false;
}

bool GpuRabitqCache::allocate_gentile_nursery_slot(uint32_t& slot,
                                                   uint32_t& tile_id,
                                                   uint32_t& offset,
                                                   uint32_t& generation) {
  if (tile_count_ == 0 || nursery_tile_count_ == 0) return false;
  if (active_nursery_offset_ >= tile_slots_) {
    active_nursery_tile_ = (active_nursery_tile_ + 1) % nursery_tile_count_;
    active_nursery_offset_ = 0;
    ++tile_generations_[active_nursery_tile_];
    if (tile_generations_[active_nursery_tile_] == 0) tile_generations_[active_nursery_tile_] = 1;
    ++evictions_;
  }
  tile_id = active_nursery_tile_;
  offset = active_nursery_offset_++;
  generation = tile_generations_[tile_id];
  slot = tile_id * tile_slots_ + offset;
  if (slot >= slot_count_) return false;
  return true;
}

void GpuRabitqCache::append_tile_task(uint32_t tile_id,
                                      uint32_t offset,
                                      uint32_t candidate_index,
                                      GpuRabitqCacheTiledResolve& result) {
  for (auto& task : result.hit_tasks) {
    if (task.tile_id == tile_id) {
      const uint32_t insert_pos = task.start + task.count;
      result.task_offsets.insert(result.task_offsets.begin() + insert_pos, offset);
      result.task_candidate_indices.insert(result.task_candidate_indices.begin() + insert_pos, candidate_index);
      ++task.count;
      for (auto& later : result.hit_tasks) {
        if (&later != &task && later.start > task.start) ++later.start;
      }
      return;
    }
  }
  result.hit_tasks.push_back(GpuRabitqTileTask{tile_id, static_cast<uint32_t>(result.task_offsets.size()), 1});
  result.task_offsets.push_back(offset);
  result.task_candidate_indices.push_back(candidate_index);
}

GpuRabitqCacheResolve GpuRabitqCache::resolve_batch(const void* remote_ptrs,
                                                    uint32_t n,
                                                    uint32_t* out_slot_ids,
                                                    std::vector<uint32_t>& fill_indices,
                                                    std::vector<uint32_t>& fill_slots,
                                                    std::vector<uint64_t>& fill_addrs,
                                                    std::vector<uint32_t>& inflight_indices) {
  GpuRabitqCacheResolve result{};
  result.n = n;
  if (!enabled_) return result;

  std::lock_guard<std::mutex> lock(mutex_);

  fill_indices.clear();
  fill_slots.clear();
  fill_addrs.clear();
  inflight_indices.clear();
  fill_indices.reserve(n);
  fill_slots.reserve(n);
  fill_addrs.reserve(n);
  inflight_indices.reserve(n);

  std::vector<uint32_t> newly_reserved;
  newly_reserved.reserve(n);

  for (uint32_t i = 0; i < n; ++i) {
    uint64_t key = 0;
    std::memcpy(&key, static_cast<const uint8_t*>(remote_ptrs) + static_cast<size_t>(i) * sizeof(uint64_t),
                sizeof(uint64_t));
    uint32_t slot = kInvalidSlot;
    if (lookup(key, slot)) {
      out_slot_ids[i] = slot;
      slot_refs_[slot] = 1;
      if (slot_states_[slot] == State::ready) {
        ++result.hit_count;
      } else {
        ++result.duplicate_loading_count;
        ++result.inflight_fallback_count;
        out_slot_ids[i] = kInvalidSlot;
        inflight_indices.push_back(i);
      }
      continue;
    }

    if (!allocate_slot(slot) || !insert_entry(key, slot)) {
      for (uint32_t reserved_slot : newly_reserved) {
        if (reserved_slot >= slot_states_.size() || slot_states_[reserved_slot] != State::loading) continue;
        remove_entry(slot_keys_[reserved_slot]);
        free_slot(reserved_slot);
      }
      return result;
    }
    slot_keys_[slot] = key;
    slot_states_[slot] = State::loading;
    slot_refs_[slot] = 1;
    newly_reserved.push_back(slot);
    out_slot_ids[i] = slot;
    ++result.fill_count;
    fill_indices.push_back(i);
    fill_slots.push_back(slot);
    fill_addrs.push_back(slot_addr(slot));
  }

  result.ok = true;
  return result;
}

GpuRabitqCacheTiledResolve GpuRabitqCache::resolve_batch_tiled(const void* remote_ptrs,
                                                                      uint32_t n,
                                                                      uint32_t* out_slot_ids,
                                                                      std::vector<uint32_t>& fill_indices,
                                                                      std::vector<uint32_t>& fill_slots,
                                                                      std::vector<uint64_t>& fill_addrs,
                                                                      std::vector<uint32_t>& inflight_indices) {
  GpuRabitqCacheTiledResolve result{};
  result.base.n = n;
  if (!enabled_ || mode_ != Mode::gentile) return result;

  std::lock_guard<std::mutex> lock(mutex_);
  fill_indices.clear();
  fill_slots.clear();
  fill_addrs.clear();
  inflight_indices.clear();
  result.hit_tasks.clear();
  result.task_offsets.clear();
  result.task_candidate_indices.clear();
  fill_indices.reserve(n);
  fill_slots.reserve(n);
  fill_addrs.reserve(n);
  inflight_indices.reserve(n);
  result.task_offsets.reserve(n);
  result.task_candidate_indices.reserve(n);
  result.hit_tasks.reserve(n);

  std::vector<uint32_t> newly_reserved;
  newly_reserved.reserve(n);

  for (uint32_t i = 0; i < n; ++i) {
    uint64_t key = 0;
    std::memcpy(&key, static_cast<const uint8_t*>(remote_ptrs) + static_cast<size_t>(i) * sizeof(uint64_t),
                sizeof(uint64_t));
    Entry* entry = lookup_entry(key);
    if (entry) {
      const bool generation_ok = entry->tile_id < tile_generations_.size() &&
                                 entry->generation == tile_generations_[entry->tile_id];
      if (!generation_ok || entry->state == State::stale) {
        remove_entry(key);
        entry = nullptr;
      }
    }
    if (entry) {
      if (entry->state == State::ready) {
        const uint32_t slot = entry->tile_id * tile_slots_ + entry->offset;
        out_slot_ids[i] = slot;
        slot_refs_[slot] = 1;
        if (entry->credit < 255) ++entry->credit;
        append_tile_task(entry->tile_id, entry->offset, i, result);
        ++result.base.hit_count;
      } else {
        ++result.base.duplicate_loading_count;
        ++result.base.inflight_fallback_count;
        out_slot_ids[i] = kInvalidSlot;
        inflight_indices.push_back(i);
      }
      continue;
    }

    uint32_t slot = kInvalidSlot;
    uint32_t tile_id = 0;
    uint32_t offset = 0;
    uint32_t generation = 0;
    if (!allocate_gentile_nursery_slot(slot, tile_id, offset, generation)) {
      for (uint32_t reserved_slot : newly_reserved) {
        if (reserved_slot >= slot_states_.size() || slot_states_[reserved_slot] != State::loading) continue;
        remove_entry(slot_keys_[reserved_slot]);
        free_slot(reserved_slot);
      }
      return result;
    }
    Entry new_entry{};
    new_entry.key = key;
    new_entry.slot = slot;
    new_entry.tile_id = tile_id;
    new_entry.offset = offset;
    new_entry.generation = generation;
    new_entry.region = Region::nursery;
    new_entry.state = State::loading;
    if (!insert_entry(new_entry)) {
      return result;
    }
    slot_keys_[slot] = key;
    slot_states_[slot] = State::loading;
    slot_refs_[slot] = 1;
    newly_reserved.push_back(slot);
    out_slot_ids[i] = slot;
    ++result.base.fill_count;
    fill_indices.push_back(i);
    fill_slots.push_back(slot);
    fill_addrs.push_back(slot_addr(slot));
    append_tile_task(tile_id, offset, i, result);
  }

  result.unique_tiles = static_cast<uint32_t>(result.hit_tasks.size());
  result.base.ok = true;
  return result;
}

void GpuRabitqCache::publish_batch(const std::vector<uint32_t>& slots) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (uint32_t slot : slots) {
    if (slot < slot_states_.size()) {
      slot_states_[slot] = State::ready;
      slot_refs_[slot] = 1;
      Entry* entry = lookup_entry(slot_keys_[slot]);
      if (entry && entry->slot == slot) entry->state = State::ready;
    }
  }
}

void GpuRabitqCache::acquire_slots(const uint32_t* slots, uint32_t n) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t slot = slots[i];
    if (slot < slot_use_counts_.size()) ++slot_use_counts_[slot];
  }
}

void GpuRabitqCache::release_slots(const uint32_t* slots, uint32_t n) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t slot = slots[i];
    if (slot < slot_use_counts_.size() && slot_use_counts_[slot] > 0) {
      --slot_use_counts_[slot];
    }
  }
}

void GpuRabitqCache::rollback_loading(const std::vector<uint32_t>& slots) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (uint32_t slot : slots) {
    if (slot >= slot_states_.size() || slot_states_[slot] != State::loading) continue;
    remove_entry(slot_keys_[slot]);
    free_slot(slot);
  }
}

size_t GpuRabitqCache::next_power_of_two(size_t value) {
  size_t out = 1;
  while (out < value) out <<= 1;
  return out;
}

uint64_t GpuRabitqCache::hash(uint64_t key) {
  key ^= key >> 33;
  key *= 0xff51afd7ed558ccdULL;
  key ^= key >> 33;
  key *= 0xc4ceb9fe1a85ec53ULL;
  key ^= key >> 33;
  return key;
}

bool GpuRabitqCache::lookup(uint64_t key, uint32_t& slot) const {
  size_t pos = hash(key) & (table_capacity_ - 1);
  for (size_t probe = 0; probe < table_capacity_; ++probe) {
    const auto& entry = table_[pos];
    if (entry.key == 0) return false;
    if (entry.key == key) {
      slot = entry.slot;
      return true;
    }
    pos = (pos + 1) & (table_capacity_ - 1);
  }
  return false;
}

bool GpuRabitqCache::insert_entry(uint64_t key, uint32_t slot) {
  size_t pos = hash(key) & (table_capacity_ - 1);
  for (size_t probe = 0; probe < table_capacity_; ++probe) {
    if (table_[pos].key == 0 || table_[pos].key == key) {
      table_[pos] = Entry{key, slot};
      return true;
    }
    pos = (pos + 1) & (table_capacity_ - 1);
  }
  return false;
}

void GpuRabitqCache::remove_entry(uint64_t key) {
  if (key == 0 || table_capacity_ == 0) return;
  size_t pos = hash(key) & (table_capacity_ - 1);
  for (size_t probe = 0; probe < table_capacity_; ++probe) {
    if (table_[pos].key == 0) return;
    if (table_[pos].key == key) {
      table_[pos] = Entry{};
      rehash_cluster((pos + 1) & (table_capacity_ - 1));
      return;
    }
    pos = (pos + 1) & (table_capacity_ - 1);
  }
}

void GpuRabitqCache::rehash_cluster(size_t pos) {
  while (table_[pos].key != 0) {
    const Entry entry = table_[pos];
    table_[pos] = Entry{};
    insert_entry(entry);
    pos = (pos + 1) & (table_capacity_ - 1);
  }
}

bool GpuRabitqCache::allocate_slot(uint32_t& slot) {
  if (next_slot_ < slot_count_) {
    slot = next_slot_++;
    return true;
  }
  for (size_t scanned = 0; scanned < slot_count_ * 2; ++scanned) {
    const uint32_t candidate = clock_hand_;
    clock_hand_ = (clock_hand_ + 1) % static_cast<uint32_t>(slot_count_);
    if (slot_states_[candidate] == State::loading || slot_use_counts_[candidate] > 0) continue;
    if (slot_refs_[candidate]) {
      slot_refs_[candidate] = 0;
      continue;
    }
    remove_entry(slot_keys_[candidate]);
    free_slot(candidate);
    ++evictions_;
    slot = candidate;
    return true;
  }
  return false;
}

void GpuRabitqCache::free_slot(uint32_t slot) {
  slot_keys_[slot] = 0;
  slot_states_[slot] = State::empty;
  slot_refs_[slot] = 0;
  slot_use_counts_[slot] = 0;
}

uint64_t GpuRabitqCache::slot_addr(uint32_t slot) const {
  return reinterpret_cast<uint64_t>(base()) + static_cast<uint64_t>(slot) * stride_;
}

}  // namespace gpu
