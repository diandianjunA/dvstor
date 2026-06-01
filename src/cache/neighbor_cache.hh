#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "common/types.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace cache {

class NeighborCache {
public:
  void init(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes == 0 || VamanaNode::R == 0) {
      return;
    }

    const size_t entry_bytes = sizeof(u64) + sizeof(u8) + VamanaNode::NEIGHBORS_SIZE;
    const size_t requested_slots = std::max<size_t>(1, bytes / entry_bytes);
    slot_count_ = requested_slots;
    table_capacity_ = next_power_of_two(std::max<size_t>(2, requested_slots * 2));

    table_keys_.assign(table_capacity_, 0);
    table_slots_.assign(table_capacity_, kInvalidSlot);
    table_generations_.assign(table_capacity_, 0);
    slot_keys_.assign(slot_count_, 0);
    slot_counts_.assign(slot_count_, 0);
    slot_refs_.assign(slot_count_, 0);
    slot_hits_.assign(slot_count_, 0);
    slot_pinned_.assign(slot_count_, 0);
    neighbors_.assign(slot_count_ * VamanaNode::R, RemotePtr{});
    enabled_ = true;
  }

  bool enabled() const { return enabled_; }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    clear_locked();
  }

  bool lookup_copy(RemotePtr key, vec<RemotePtr>& out) {
    out.clear();
    if (!enabled_ || key.is_null()) {
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    u32 slot = kInvalidSlot;
    if (!find_slot_locked(key.raw_address, slot)) {
      return false;
    }

    const u8 count = slot_counts_[slot];
    const auto* src = neighbors_.data() + static_cast<size_t>(slot) * VamanaNode::R;
    out.reserve(count);
    for (u32 i = 0; i < count; ++i) {
      out.push_back(src[i]);
    }
    slot_refs_[slot] = 1;
    if (slot_hits_[slot] < UINT32_MAX) {
      ++slot_hits_[slot];
    }
    if (slot_hits_[slot] >= kPinAfterHits) {
      slot_pinned_[slot] = 1;
    }
    return true;
  }

  void insert(RemotePtr key, span<RemotePtr> values, bool pin = false) {
    if (!enabled_ || key.is_null()) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const u32 count = static_cast<u32>(std::min<size_t>(values.size(), VamanaNode::R));
    size_t pos = hash(key.raw_address) & (table_capacity_ - 1);
    for (size_t probe = 0; probe < table_capacity_; ++probe) {
      const bool occupied = table_generations_[pos] == generation_ && table_keys_[pos] != 0;
      if (occupied && table_keys_[pos] == key.raw_address) {
        const u32 slot = table_slots_[pos];
        store_slot(slot, key.raw_address, values, count, pin || slot_pinned_[slot]);
        return;
      }
      if (!occupied) {
        const u32 slot = allocate_slot_locked();
        table_keys_[pos] = key.raw_address;
        table_slots_[pos] = slot;
        table_generations_[pos] = generation_;
        store_slot(slot, key.raw_address, values, count, pin);
        return;
      }
      pos = (pos + 1) & (table_capacity_ - 1);
    }
  }

  void invalidate(RemotePtr key) {
    if (!enabled_ || key.is_null()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    u32 slot = kInvalidSlot;
    if (!find_slot_locked(key.raw_address, slot)) {
      return;
    }
    remove_from_table_locked(key.raw_address);
    free_slot_locked(slot);
  }

  size_t slot_count() const { return slot_count_; }

private:
  static constexpr u32 kInvalidSlot = UINT32_MAX;
  static constexpr u32 kPinAfterHits = 2;

  static size_t next_power_of_two(size_t value) {
    size_t out = 1;
    while (out < value) {
      out <<= 1;
    }
    return out;
  }

  static u64 hash(u64 key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
  }

  bool find_slot_locked(u64 key, u32& slot) const {
    if (key == 0 || table_capacity_ == 0) {
      return false;
    }
    size_t pos = hash(key) & (table_capacity_ - 1);
    for (size_t probe = 0; probe < table_capacity_; ++probe) {
      const u64 table_key = table_generations_[pos] == generation_ ? table_keys_[pos] : 0;
      if (table_key == 0) {
        return false;
      }
      if (table_key == key) {
        slot = table_slots_[pos];
        return slot != kInvalidSlot;
      }
      pos = (pos + 1) & (table_capacity_ - 1);
    }
    return false;
  }

  void clear_locked() {
    if (!enabled_) {
      return;
    }
    if (++generation_ == 0) {
      std::fill(table_keys_.begin(), table_keys_.end(), 0);
      std::fill(table_slots_.begin(), table_slots_.end(), kInvalidSlot);
      std::fill(table_generations_.begin(), table_generations_.end(), 0);
      std::fill(slot_keys_.begin(), slot_keys_.end(), 0);
      std::fill(slot_counts_.begin(), slot_counts_.end(), 0);
      std::fill(slot_refs_.begin(), slot_refs_.end(), 0);
      std::fill(slot_hits_.begin(), slot_hits_.end(), 0);
      std::fill(slot_pinned_.begin(), slot_pinned_.end(), 0);
      std::fill(neighbors_.begin(), neighbors_.end(), RemotePtr{});
      generation_ = 1;
    }
    next_slot_ = 0;
    clock_hand_ = 0;
  }

  void remove_from_table_locked(u64 key) {
    size_t pos = hash(key) & (table_capacity_ - 1);
    for (size_t probe = 0; probe < table_capacity_; ++probe) {
      const bool occupied = table_generations_[pos] == generation_ && table_keys_[pos] != 0;
      if (!occupied) {
        return;
      }
      if (table_keys_[pos] == key) {
        table_keys_[pos] = 0;
        table_slots_[pos] = kInvalidSlot;
        table_generations_[pos] = 0;
        rehash_cluster_locked((pos + 1) & (table_capacity_ - 1));
        return;
      }
      pos = (pos + 1) & (table_capacity_ - 1);
    }
  }

  void rehash_cluster_locked(size_t pos) {
    while (table_generations_[pos] == generation_ && table_keys_[pos] != 0) {
      const u64 key = table_keys_[pos];
      const u32 slot = table_slots_[pos];
      table_keys_[pos] = 0;
      table_slots_[pos] = kInvalidSlot;
      table_generations_[pos] = 0;

      size_t dst = hash(key) & (table_capacity_ - 1);
      while (table_generations_[dst] == generation_ && table_keys_[dst] != 0) {
        dst = (dst + 1) & (table_capacity_ - 1);
      }
      table_keys_[dst] = key;
      table_slots_[dst] = slot;
      table_generations_[dst] = generation_;
      pos = (pos + 1) & (table_capacity_ - 1);
    }
  }

  u32 allocate_slot_locked() {
    while (next_slot_ < slot_count_) {
      const u32 slot = next_slot_++;
      if (slot_keys_[slot] == 0) {
        return slot;
      }
    }

    for (size_t scanned = 0; scanned < slot_count_ * 2; ++scanned) {
      const u32 slot = clock_hand_;
      clock_hand_ = (clock_hand_ + 1) % static_cast<u32>(slot_count_);
      if (slot_pinned_[slot]) {
        continue;
      }
      if (slot_refs_[slot]) {
        slot_refs_[slot] = 0;
        continue;
      }
      remove_from_table_locked(slot_keys_[slot]);
      return slot;
    }

    for (size_t scanned = 0; scanned < slot_count_; ++scanned) {
      const u32 slot = clock_hand_;
      clock_hand_ = (clock_hand_ + 1) % static_cast<u32>(slot_count_);
      if (!slot_pinned_[slot]) {
        remove_from_table_locked(slot_keys_[slot]);
        return slot;
      }
    }

    const u32 slot = clock_hand_;
    clock_hand_ = (clock_hand_ + 1) % static_cast<u32>(slot_count_);
    remove_from_table_locked(slot_keys_[slot]);
    return slot;
  }

  void store_slot(u32 slot, u64 key, span<RemotePtr> values, u32 count, bool pin) {
    slot_keys_[slot] = key;
    slot_counts_[slot] = static_cast<u8>(count);
    slot_refs_[slot] = 1;
    slot_hits_[slot] = pin ? kPinAfterHits : 0;
    slot_pinned_[slot] = pin ? 1 : 0;
    auto* dst = neighbors_.data() + static_cast<size_t>(slot) * VamanaNode::R;
    for (u32 i = 0; i < count; ++i) {
      dst[i] = values[i];
    }
    for (u32 i = count; i < VamanaNode::R; ++i) {
      dst[i] = RemotePtr{};
    }
  }

  void free_slot_locked(u32 slot) {
    if (slot >= slot_count_) {
      return;
    }
    slot_keys_[slot] = 0;
    slot_counts_[slot] = 0;
    slot_refs_[slot] = 0;
    slot_hits_[slot] = 0;
    slot_pinned_[slot] = 0;
    auto* dst = neighbors_.data() + static_cast<size_t>(slot) * VamanaNode::R;
    for (u32 i = 0; i < VamanaNode::R; ++i) {
      dst[i] = RemotePtr{};
    }
  }

  bool enabled_{false};
  size_t slot_count_{0};
  size_t table_capacity_{0};
  u32 next_slot_{0};
  u32 clock_hand_{0};
  u32 generation_{1};

  std::vector<u64> table_keys_;
  std::vector<u32> table_slots_;
  std::vector<u32> table_generations_;
  std::vector<u64> slot_keys_;
  std::vector<u8> slot_counts_;
  std::vector<u8> slot_refs_;
  std::vector<u32> slot_hits_;
  std::vector<u8> slot_pinned_;
  std::vector<RemotePtr> neighbors_;
  mutable std::mutex mutex_;
};

}  // namespace cache
