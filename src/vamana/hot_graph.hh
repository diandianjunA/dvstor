#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace vamana::hot_graph {

constexpr u32 kMagic = 0x31474844;  // DHG1
constexpr u16 kVersion = 1;
constexpr u16 kVersion2 = 2;
constexpr u32 kCompactPointerBytes = 5;
constexpr u64 kNodeBaseOffset = 16;
constexpr u64 kNullCompactPointer = (1ull << 40) - 1ull;
constexpr u32 kV2NeighborBaseOffset = 8;

#pragma pack(push, 1)
struct Header {
  u32 magic{kMagic};
  u16 version{kVersion};
  u16 header_bytes{64};
  u32 entry_bytes{};
  u32 max_degree{};
  u32 compact_pointer_bytes{kCompactPointerBytes};
  u32 compact_pointer_shard_bits{};
  u32 flags{};
  u64 entry_count{};
  u64 node_base_offset{kNodeBaseOffset};
  u64 reserved0{};
  u64 reserved1{};
  u32 reserved2{};
};
#pragma pack(pop)

static_assert(sizeof(Header) == 64);

inline size_t align8(size_t value) {
  return (value + 7) & ~size_t{7};
}

inline size_t entry_bytes(u32 max_degree) {
  return align8(8 + static_cast<size_t>(max_degree) * kCompactPointerBytes);
}

inline u32 shard_bits_for(u32 shard_count) {
  u32 bits = 0;
  u32 capacity = 1;
  while (capacity < std::max<u32>(shard_count, 1)) {
    capacity <<= 1;
    ++bits;
  }
  return bits;
}

inline bool encode_remote_ptr(RemotePtr ptr, u32 shard_bits, byte_t* out) {
  if (ptr.is_null() || ptr.byte_offset() % 8 != 0 || shard_bits >= 16) {
    std::memset(out, 0xff, kCompactPointerBytes);
    return ptr.is_null();
  }

  const u32 offset_bits = 40 - shard_bits;
  const u64 max_shards = 1ull << shard_bits;
  const u64 offset_units = ptr.byte_offset() / 8;
  if (ptr.memory_node() >= max_shards || offset_units >= (1ull << offset_bits)) {
    std::memset(out, 0xff, kCompactPointerBytes);
    return false;
  }

  const u64 packed = (static_cast<u64>(ptr.memory_node()) << offset_bits) | offset_units;
  for (u32 i = 0; i < kCompactPointerBytes; ++i) {
    out[i] = static_cast<byte_t>((packed >> (8 * i)) & 0xffu);
  }
  return true;
}

inline RemotePtr decode_remote_ptr(const byte_t* in, u32 shard_bits) {
  u64 packed = 0;
  for (u32 i = 0; i < kCompactPointerBytes; ++i) {
    packed |= static_cast<u64>(in[i]) << (8 * i);
  }
  if (packed == kNullCompactPointer || shard_bits >= 16) {
    return RemotePtr{};
  }
  const u32 offset_bits = 40 - shard_bits;
  const u64 offset_mask = (1ull << offset_bits) - 1ull;
  const u32 shard = static_cast<u32>(packed >> offset_bits);
  const u64 offset = (packed & offset_mask) * 8;
  return RemotePtr{shard, offset};
}

inline size_t neighbor_offset(u32 index) {
  return 8 + static_cast<size_t>(index) * kCompactPointerBytes;
}

inline u16 checksum16(const byte_t* data, size_t bytes) {
  u32 hash = 2166136261u;
  for (size_t i = 0; i < bytes; ++i) {
    if (i == 2 || i == 3) continue;
    hash ^= data[i];
    hash *= 16777619u;
  }
  hash ^= hash >> 16;
  return static_cast<u16>(hash);
}

inline u32 load_u32_le(const byte_t* data) {
  u32 value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

inline void store_u32_le(byte_t* data, u32 value) {
  std::memcpy(data, &value, sizeof(value));
}

inline u16 load_u16_le(const byte_t* data) {
  u16 value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

inline void store_u16_le(byte_t* data, u16 value) {
  std::memcpy(data, &value, sizeof(value));
}

}  // namespace vamana::hot_graph
