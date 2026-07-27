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
constexpr u16 kVersion3 = 3;
constexpr u32 kTaggedPointerBytes = sizeof(u64);
constexpr u32 kCompactPointerBytes = kTaggedPointerBytes;
constexpr u64 kNodeBaseOffset = 16;
constexpr u32 kV2NeighborBaseOffset = 8;
constexpr u32 kTaggedNeighborBaseOffset = 16;
constexpr u8 kDeletedFlag = 1u << 0;
constexpr u8 kProvisionalCountShift = 4;
constexpr u8 kProvisionalCountMask = 0xf0u;

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

inline size_t entry_bytes(u32 max_degree, u32 provisional_slots = 0) {
  return align8(kTaggedNeighborBaseOffset +
    static_cast<size_t>(max_degree + provisional_slots) *
      kCompactPointerBytes);
}

inline u8 provisional_count(const byte_t* entry) {
  return static_cast<u8>(
    (entry[1] & kProvisionalCountMask) >> kProvisionalCountShift);
}

inline void store_provisional_count(byte_t* entry, u8 count) {
  entry[1] = static_cast<u8>(
    (entry[1] & ~kProvisionalCountMask) |
    ((count << kProvisionalCountShift) & kProvisionalCountMask));
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
  (void)shard_bits;
  std::memcpy(out, &ptr.raw_address, sizeof(ptr.raw_address));
  return true;
}

inline RemotePtr decode_remote_ptr(const byte_t* in, u32 shard_bits) {
  (void)shard_bits;
  u64 raw = 0;
  std::memcpy(&raw, in, sizeof(raw));
  return RemotePtr{raw};
}

inline size_t neighbor_offset(u32 index) {
  return kTaggedNeighborBaseOffset +
    static_cast<size_t>(index) * kCompactPointerBytes;
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
