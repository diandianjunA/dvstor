#pragma once

#include "common/types.hh"

namespace vamana::idmap {

constexpr u32 kMagic = 0x504d4444;  // DDMP
constexpr u32 kVersion = 1;
constexpr u32 kDeleted = 1u << 0;

#pragma pack(push, 1)
struct Header {
  u32 magic{kMagic};
  u32 version{kVersion};
  u32 owner_shard{};
  u32 shard_count{};
  u64 entry_count{};
};

struct Entry {
  node_t id{};
  u64 rptr_raw{};
  u32 generation{};
  u32 flags{};
};
#pragma pack(pop)

static_assert(sizeof(Header) == 24);
static_assert(sizeof(Entry) == 20);

}  // namespace vamana::idmap
