#pragma once

#include <algorithm>
#include <cstring>
#include <limits>

#include "common/types.hh"

namespace tools::vamana_offline::legacy_schema15 {

// On-disk contracts emitted by the retired compact-v1/schema-14 builder and
// consumed by its schema-15 PQ runtime.  These definitions are deliberately
// isolated from runtime headers: they exist only to perform an offline,
// fail-closed migration.
inline constexpr u32 kIdmapMagic = 0x504d4444;  // DDMP
inline constexpr u32 kIdmapVersion = 1;
inline constexpr u32 kIdmapDeleted = 1u << 0;
inline constexpr u32 kHotGraphMagic = 0x31474844;  // DHG1
inline constexpr u16 kHotGraphVersion = 2;
inline constexpr u32 kCompactPointerBytes = 5;
inline constexpr u64 kNodeBaseOffset = 16;
inline constexpr u64 kNullCompactPointer = (u64{1} << 40) - 1;
inline constexpr u64 kMedoidHeaderFlag = u64{1} << 16;
inline constexpr u32 kSchemaVersion = 15;
inline constexpr const char* kStorageFormat = "vamana_compact_v1";
inline constexpr u32 kFixedHeaderOffset = 0;
inline constexpr u32 kIdOffset = 8;
inline constexpr u32 kGenerationOffset = 12;
inline constexpr u32 kVectorOffset = 16;
inline constexpr u64 kNodeLock = 1;
inline constexpr u64 kIsMedoid = kMedoidHeaderFlag;
inline constexpr u64 kDeleted = u64{1} << 24;
inline constexpr u32 kNeighborBaseOffset = 8;
inline constexpr u8 kHotGraphDeleted = 1;

#pragma pack(push, 1)
struct IdmapHeader {
  u32 magic{kIdmapMagic};
  u32 version{kIdmapVersion};
  u32 owner_shard{};
  u32 shard_count{};
  u64 entry_count{};
};

struct IdmapEntry {
  node_t id{};
  u64 rptr_raw{};
  u32 generation{};
  u32 flags{};
};

struct HotGraphHeader {
  u32 magic{kHotGraphMagic};
  u16 version{kHotGraphVersion};
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

static_assert(sizeof(IdmapHeader) == 24);
static_assert(sizeof(IdmapEntry) == 20);
static_assert(sizeof(HotGraphHeader) == 64);

inline size_t align8(size_t value) {
  return (value + 7) & ~size_t{7};
}

inline size_t align16(size_t value) {
  return (value + 15) & ~size_t{15};
}

inline size_t fixed_node_bytes(size_t vector_bytes) {
  return align16(16 + align8(vector_bytes));
}

inline size_t node_bytes(size_t vector_bytes) {
  return fixed_node_bytes(vector_bytes);
}

inline size_t graph_entry_bytes(u32 max_degree) {
  return align8(8 + static_cast<size_t>(max_degree) *
    kCompactPointerBytes);
}

inline size_t hot_graph_entry_bytes(u32 max_degree) {
  return graph_entry_bytes(max_degree);
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

struct RemotePtr {
  u32 shard{};
  u64 byte_offset{};
  bool is_null{};
};

using DecodedPointer = RemotePtr;

inline RemotePtr decode_raw_remote_ptr(u64 raw) {
  return {
    .shard = static_cast<u32>(raw >> 48),
    .byte_offset = (raw << 16) >> 16,
    .is_null = raw == 0,
  };
}

inline u64 encode_raw_remote_ptr(u32 shard, u64 byte_offset) {
  return (static_cast<u64>(shard) << 48) | byte_offset;
}

inline RemotePtr decode_compact_remote_ptr(const byte_t* input,
                                           u32 shard_bits) {
  u64 packed = 0;
  for (u32 index = 0; index < kCompactPointerBytes; ++index) {
    packed |= static_cast<u64>(input[index]) << (8 * index);
  }
  if (packed == kNullCompactPointer || shard_bits >= 16) {
    return {.is_null = true};
  }
  const u32 offset_bits = 40 - shard_bits;
  const u64 offset_mask = (u64{1} << offset_bits) - 1;
  return {
    .shard = static_cast<u32>(packed >> offset_bits),
    .byte_offset = (packed & offset_mask) * 8,
    .is_null = false,
  };
}

inline bool decode_compact_pointer(const byte_t* input, u32 shard_bits,
                                   DecodedPointer& result) {
  if (shard_bits >= 16) return false;
  result = decode_compact_remote_ptr(input, shard_bits);
  return true;
}

inline void encode_null_compact_remote_ptr(byte_t* output) {
  std::memset(output, 0xff, kCompactPointerBytes);
}

inline size_t neighbor_offset(u32 index) {
  return 8 + static_cast<size_t>(index) * kCompactPointerBytes;
}

inline size_t hot_graph_neighbor_offset(u32 index) {
  return neighbor_offset(index);
}

inline u16 checksum16(const byte_t* data, size_t bytes) {
  u32 hash = 2166136261u;
  for (size_t index = 0; index < bytes; ++index) {
    if (index == 2 || index == 3) continue;
    hash ^= data[index];
    hash *= 16777619u;
  }
  hash ^= hash >> 16;
  return static_cast<u16>(hash);
}

inline u16 load_u16(const byte_t* data) {
  u16 value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

inline u32 load_u32(const byte_t* data) {
  u32 value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

inline u64 load_u64(const byte_t* data) {
  u64 value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

}  // namespace tools::vamana_offline::legacy_schema15
