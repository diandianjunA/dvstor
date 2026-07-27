#pragma once

#include <limits>
#include <ostream>
#include <stdexcept>

#include "common/types.hh"

struct RemotePtr {
  static constexpr size_t SIZE = sizeof(u64);
  // A graph edge is a logical physical handle, not just an address. Keeping
  // the slot incarnation in every edge prevents a persistent stale edge from
  // becoming valid again when the dynamic allocator reuses the same bytes.
  //
  //   [ slot incarnation (24b) | memory node (6b) | 16-byte offset (34b) ]
  //
  // 34 offset units cover 256 GiB per shard. Incarnation zero is reserved for
  // immutable base-index nodes; dynamic slots start at one. The all-ones
  // incarnation is reserved so UINT64_MAX remains an invalid GPU sentinel.
  static constexpr u32 OFFSET_ALIGNMENT_LOG2 = 4;
  static constexpr u64 OFFSET_ALIGNMENT = u64{1} << OFFSET_ALIGNMENT_LOG2;
  static constexpr u32 OFFSET_UNIT_BITS = 34;
  static constexpr u32 MEMORY_NODE_BITS = 6;
  static constexpr u32 INCARNATION_BITS = 24;
  static constexpr u32 MEMORY_NODE_SHIFT = OFFSET_UNIT_BITS;
  static constexpr u32 INCARNATION_SHIFT =
    OFFSET_UNIT_BITS + MEMORY_NODE_BITS;
  static constexpr u64 OFFSET_UNIT_MASK =
    (u64{1} << OFFSET_UNIT_BITS) - 1;
  static constexpr u64 MEMORY_NODE_MASK =
    (u64{1} << MEMORY_NODE_BITS) - 1;
  static constexpr u32 MAX_INCARNATION =
    (u32{1} << INCARNATION_BITS) - 2;
  static constexpr u64 BYTE_OFFSET_CAPACITY =
    (u64{1} << OFFSET_UNIT_BITS) * OFFSET_ALIGNMENT;

  u64 raw_address{};

  RemotePtr() = default;
  explicit RemotePtr(u64 raw_address) : raw_address(raw_address) {}
  RemotePtr(u32 memory_node, u64 byte_offset, u32 incarnation = 0) {
    store_address(memory_node, byte_offset, incarnation);
  }

  u32 memory_node() const {
    return static_cast<u32>(
      (raw_address >> MEMORY_NODE_SHIFT) & MEMORY_NODE_MASK);
  }
  u64 byte_offset() const {
    return (raw_address & OFFSET_UNIT_MASK) << OFFSET_ALIGNMENT_LOG2;
  }
  u32 incarnation() const {
    return static_cast<u32>(raw_address >> INCARNATION_SHIFT);
  }
  bool is_static() const { return !is_null() && incarnation() == 0; }
  bool is_dynamic() const { return incarnation() != 0; }
  bool is_well_formed() const {
    return incarnation() <= MAX_INCARNATION;
  }
  u64 physical_address_raw() const {
    return raw_address & ((u64{1} << INCARNATION_SHIFT) - 1);
  }
  RemotePtr with_incarnation(u32 value) const {
    return RemotePtr{memory_node(), byte_offset(), value};
  }
  bool is_null() const { return raw_address == 0; }
  void reset() { raw_address = 0; }

  static constexpr bool representable(u32 memory_node, u64 byte_offset,
                                      u32 incarnation = 0) {
    return memory_node <= MEMORY_NODE_MASK &&
      byte_offset % OFFSET_ALIGNMENT == 0 &&
      byte_offset < BYTE_OFFSET_CAPACITY &&
      incarnation <= MAX_INCARNATION;
  }

  void store_address(u32 memory_node, u64 byte_offset,
                     u32 incarnation = 0) {
    if (!representable(memory_node, byte_offset, incarnation)) {
      throw std::out_of_range(
        "RemotePtr exceeds tagged-handle shard, offset, alignment, or "
        "incarnation capacity");
    }
    raw_address =
      (static_cast<u64>(incarnation) << INCARNATION_SHIFT) |
      (static_cast<u64>(memory_node) << MEMORY_NODE_SHIFT) |
      (byte_offset >> OFFSET_ALIGNMENT_LOG2);
  }

  bool operator==(const RemotePtr&) const = default;  // compares raw_address

  friend std::ostream& operator<<(std::ostream& os, const RemotePtr& r) {
    return os << "[node: " << r.memory_node()
              << " | offset: " << r.byte_offset()
              << " | incarnation: " << r.incarnation() << "]";
  }
};

static_assert(RemotePtr::INCARNATION_SHIFT + RemotePtr::INCARNATION_BITS == 64);
static_assert(RemotePtr::BYTE_OFFSET_CAPACITY == (u64{256} << 30));

template <>
struct std::hash<RemotePtr> {
  size_t operator()(const RemotePtr& r) const noexcept {
    u64 h = std::hash<u64>{}(r.raw_address);

    // murmur64
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;

    // murmur32
    // h ^= h >> 16;
    // h *= 0x85ebca6b;
    // h ^= h >> 13;
    // h *= 0xc2b2ae35;
    // h ^= h >> 16;

    return h;
  }
};

template <>
struct ankerl::unordered_dense::hash<RemotePtr> {
  using is_avalanching = void;

  size_t operator()(const RemotePtr& r) const noexcept {
    return ankerl::unordered_dense::hash<u64>{}(r.raw_address);
  }
};
