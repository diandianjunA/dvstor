#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tools::vamana_offline {

// Compact set for the offline graph-search hot path. Every probe is bounded,
// and the table grows before its load factor exceeds 0.5.
class LocalIdSet {
public:
  explicit LocalIdSet(size_t expected_items) {
    rehash(initial_capacity(expected_items));
  }

  bool contains(uint32_t value) const {
    if (value == kEmpty)
      return false;
    return find_slot(table_, value).second;
  }

  bool insert(uint32_t value) {
    if (value == kEmpty) {
      throw std::invalid_argument("LocalIdSet cannot store its empty sentinel");
    }

    auto slot = find_slot(table_, value);
    if (slot.second)
      return false;
    if (slot.first == kNoSlot || size_ + 1 > table_.size() / 2) {
      grow();
      slot = find_slot(table_, value);
      if (slot.second)
        return false;
      if (slot.first == kNoSlot) {
        throw std::logic_error("LocalIdSet has no free slot after growing");
      }
    }

    table_[slot.first] = value;
    ++size_;
    return true;
  }

  size_t size() const noexcept { return size_; }
  size_t capacity() const noexcept { return table_.size(); }

private:
  static constexpr uint32_t kEmpty = std::numeric_limits<uint32_t>::max();
  static constexpr size_t kNoSlot = std::numeric_limits<size_t>::max();
  static constexpr size_t kMinimumCapacity = 8;

  static size_t hash(uint32_t value) {
    uint64_t x = value;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return static_cast<size_t>(x);
  }

  static size_t initial_capacity(size_t expected_items) {
    const size_t max_capacity = std::numeric_limits<size_t>::max() / 2 + 1;
    if (expected_items > max_capacity / 2) {
      throw std::length_error("LocalIdSet requested capacity is too large");
    }
    const size_t required =
        std::max<size_t>(kMinimumCapacity, expected_items * 2);
    size_t capacity = kMinimumCapacity;
    while (capacity < required) {
      if (capacity > max_capacity / 2) {
        throw std::length_error("LocalIdSet capacity overflow");
      }
      capacity <<= 1;
    }
    return capacity;
  }

  static std::pair<size_t, bool> find_slot(const std::vector<uint32_t> &table,
                                           uint32_t value) {
    const size_t mask = table.size() - 1;
    size_t position = hash(value) & mask;
    for (size_t probes = 0; probes < table.size(); ++probes) {
      const uint32_t current = table[position];
      if (current == kEmpty)
        return {position, false};
      if (current == value)
        return {position, true};
      position = (position + 1) & mask;
    }
    return {kNoSlot, false};
  }

  void grow() {
    if (table_.size() > std::numeric_limits<size_t>::max() / 2) {
      throw std::length_error("LocalIdSet capacity overflow");
    }
    rehash(table_.size() * 2);
  }

  void rehash(size_t new_capacity) {
    std::vector<uint32_t> replacement(new_capacity, kEmpty);
    for (const uint32_t value : table_) {
      if (value == kEmpty)
        continue;
      const auto slot = find_slot(replacement, value);
      if (slot.first == kNoSlot || slot.second) {
        throw std::logic_error("LocalIdSet rehash invariant failed");
      }
      replacement[slot.first] = value;
    }
    table_.swap(replacement);
  }

  std::vector<uint32_t> table_;
  size_t size_{0};
};

} // namespace tools::vamana_offline
