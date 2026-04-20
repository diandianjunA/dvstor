#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/types.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace cache {

class NeighborCache {
public:
  void init(size_t bytes) {
    if (bytes == 0 || VamanaNode::R == 0) {
      return;
    }

    const size_t entry_bytes = sizeof(u64) + sizeof(u8) + VamanaNode::NEIGHBORS_SIZE;
    const size_t requested_slots = std::max<size_t>(1, bytes / entry_bytes);
    slot_count_ = requested_slots;
    table_capacity_ = next_power_of_two(std::max<size_t>(2, requested_slots * 2));

    table_keys_.assign(table_capacity_, 0);
    table_slots_.assign(table_capacity_, kInvalidSlot);
    slot_keys_.assign(slot_count_, 0);
    slot_counts_.assign(slot_count_, 0);
    slot_refs_.assign(slot_count_, 0);
    neighbors_.assign(slot_count_ * VamanaNode::R, RemotePtr{});
    enabled_ = true;
  }

  bool enabled() const { return enabled_; }

  bool lookup(RemotePtr key, u8& count, const RemotePtr*& neighbors) {
    if (!enabled_ || key.is_null()) {
      return false;
    }

    size_t pos = hash(key.raw_address) & (table_capacity_ - 1);
    for (size_t probe = 0; probe < table_capacity_; ++probe) {
      const u64 table_key = table_keys_[pos];
      if (table_key == 0) {
        return false;
      }
      if (table_key == key.raw_address) {
        const u32 slot = table_slots_[pos];
        count = slot_counts_[slot];
        neighbors = neighbors_.data() + static_cast<size_t>(slot) * VamanaNode::R;
        slot_refs_[slot] = 1;
        return true;
      }
      pos = (pos + 1) & (table_capacity_ - 1);
    }
    return false;
  }

  void insert(RemotePtr key, span<RemotePtr> values) {
    if (!enabled_ || key.is_null()) {
      return;
    }

    const u32 count = static_cast<u32>(std::min<size_t>(values.size(), VamanaNode::R));
    size_t pos = hash(key.raw_address) & (table_capacity_ - 1);
    for (size_t probe = 0; probe < table_capacity_; ++probe) {
      if (table_keys_[pos] == key.raw_address) {
        const u32 slot = table_slots_[pos];
        store_slot(slot, key.raw_address, values, count);
        return;
      }
      if (table_keys_[pos] == 0) {
        const u32 slot = allocate_slot();
        table_keys_[pos] = key.raw_address;
        table_slots_[pos] = slot;
        store_slot(slot, key.raw_address, values, count);
        return;
      }
      pos = (pos + 1) & (table_capacity_ - 1);
    }
  }

  size_t slot_count() const { return slot_count_; }

private:
  static constexpr u32 kInvalidSlot = UINT32_MAX;

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

  void remove_from_table(u64 key) {
    size_t pos = hash(key) & (table_capacity_ - 1);
    for (size_t probe = 0; probe < table_capacity_; ++probe) {
      if (table_keys_[pos] == 0) {
        return;
      }
      if (table_keys_[pos] == key) {
        table_keys_[pos] = 0;
        table_slots_[pos] = kInvalidSlot;
        rehash_cluster((pos + 1) & (table_capacity_ - 1));
        return;
      }
      pos = (pos + 1) & (table_capacity_ - 1);
    }
  }

  void rehash_cluster(size_t pos) {
    while (table_keys_[pos] != 0) {
      const u64 key = table_keys_[pos];
      const u32 slot = table_slots_[pos];
      table_keys_[pos] = 0;
      table_slots_[pos] = kInvalidSlot;

      size_t dst = hash(key) & (table_capacity_ - 1);
      while (table_keys_[dst] != 0) {
        dst = (dst + 1) & (table_capacity_ - 1);
      }
      table_keys_[dst] = key;
      table_slots_[dst] = slot;
      pos = (pos + 1) & (table_capacity_ - 1);
    }
  }

  u32 allocate_slot() {
    if (next_slot_ < slot_count_) {
      return next_slot_++;
    }

    for (size_t scanned = 0; scanned < slot_count_ * 2; ++scanned) {
      const u32 slot = clock_hand_;
      clock_hand_ = (clock_hand_ + 1) % static_cast<u32>(slot_count_);
      if (slot_refs_[slot]) {
        slot_refs_[slot] = 0;
        continue;
      }
      remove_from_table(slot_keys_[slot]);
      return slot;
    }

    const u32 slot = clock_hand_;
    clock_hand_ = (clock_hand_ + 1) % static_cast<u32>(slot_count_);
    remove_from_table(slot_keys_[slot]);
    return slot;
  }

  void store_slot(u32 slot, u64 key, span<RemotePtr> values, u32 count) {
    slot_keys_[slot] = key;
    slot_counts_[slot] = static_cast<u8>(count);
    slot_refs_[slot] = 1;
    auto* dst = neighbors_.data() + static_cast<size_t>(slot) * VamanaNode::R;
    for (u32 i = 0; i < count; ++i) {
      dst[i] = values[i];
    }
  }

  bool enabled_{false};
  size_t slot_count_{0};
  size_t table_capacity_{0};
  u32 next_slot_{0};
  u32 clock_hand_{0};

  std::vector<u64> table_keys_;
  std::vector<u32> table_slots_;
  std::vector<u64> slot_keys_;
  std::vector<u8> slot_counts_;
  std::vector<u8> slot_refs_;
  std::vector<RemotePtr> neighbors_;
};

}  // namespace cache
