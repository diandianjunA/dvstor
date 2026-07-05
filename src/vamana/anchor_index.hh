#pragma once

#include <optional>
#include <shared_mutex>

#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "remote_pointer.hh"

namespace vamana::anchor {

constexpr u64 kMagic = 0x3148434e414c4441ull;  // "ADLANCH1"
constexpr u32 kVersion = 2;
constexpr u32 kLegacyVersion = 1;

struct Header {
  u64 magic{kMagic};
  u32 version{kVersion};
  u32 dim{};
  u32 shard_count{};
  u32 vector_dtype{};
  u32 vector_bytes{};
  u32 anchors_per_shard{};
  u32 reserved{};
  u64 total_anchors{};
};

struct ShardHeader {
  u32 shard{};
  u32 anchor_count{};
};

struct EntryHeader {
  u64 rptr_raw{};
  u32 id{};
  u16 degree{};
  u16 reserved{};
};

struct Route {
  u32 owner{};
  u32 semantic_owner{};
  u32 anchor_version{kVersion};
  f32 confidence{1.0f};
  u32 flags{};
  vec<RemotePtr> hints;
};

class Index {
public:
  bool load(const filepath_t& index_prefix,
            u32 expected_dim,
            u32 expected_shards,
            str* error_message = nullptr);

  bool empty() const { return shards_.empty(); }
  size_t anchor_count() const { return total_anchors_; }
  size_t memory_bytes() const;
  bool add_runtime_anchor(u32 shard,
                          const span<const element_t> vector,
                          RemotePtr pointer,
                          u32 max_runtime_anchors_per_shard);

  Route route(const span<const element_t> query,
              u32 hint_count,
              u32 top_owners = 1,
              std::optional<u32> owner_override = std::nullopt) const;

private:
  struct Shard {
    vec<element_t> centroid;
    vec<element_t> vectors;
    vec<RemotePtr> pointers;
    u32 static_anchor_count{};
  };

  struct ShardScore {
    u32 shard{};
    distance_t distance{};
  };

  ShardScore nearest_shard(const span<const element_t> query,
                           distance_t* second_distance = nullptr) const;
  vec<ShardScore> top_shards(const span<const element_t> query, u32 count) const;
  distance_t centroid_distance(const span<const element_t> query, u32 shard) const;
  distance_t shard_distance(const span<const element_t> query, u32 shard) const;
  vec<RemotePtr> nearest_anchors(const span<const element_t> query,
                                 u32 shard,
                                 u32 count) const;

  u32 dim_{};
  u32 version_{kVersion};
  u32 route_epoch_{kVersion};
  size_t total_anchors_{};
  vec<Shard> shards_;
  mutable std::shared_mutex mutex_;
};

}  // namespace vamana::anchor
