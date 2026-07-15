#pragma once

#include <array>
#include <optional>
#include <shared_mutex>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace vamana::routing {

// A small, live-mutation-driven routing table.  Capacity and adaptation are
// algorithm constants on purpose: routing cost and memory do not grow with the
// number of mutations, and deployment scripts cannot silently change routing
// quality.  Reads take a shared lock; observe/invalidate are serialized by an
// exclusive lock, so callers do not need an external single-writer protocol.
class AdaptiveRouteTable {
public:
  static constexpr u32 kSlotsPerShard = 8;
  static constexpr element_t kCenterEmaWeight = 0.125F;
  static_assert(kCenterEmaWeight > 0 && kCenterEmaWeight <= 1);

  struct Route {
    u32 shard{};
    node_t id{};
    u32 generation{};
    RemotePtr entry;
    distance_t shard_distance{};
    distance_t entry_distance{};
  };

  struct SlotSnapshot {
    u32 shard{};
    u32 slot{};
    bool initialized{};
    bool live{};
    node_t id{};
    u32 generation{};
    RemotePtr entry;
    u64 observations{};
    vec<element_t> center;
    vec<element_t> representative;
  };

  struct Snapshot {
    u32 dim{};
    u32 shard_count{};
    vec<SlotSnapshot> slots;
  };

  // Allocation-free metadata view for hot-path consumers such as the GPU
  // route publisher.  Centers and representative vectors deliberately stay
  // inside the table.
  struct RouteSlotSnapshot {
    u32 shard{};
    u32 slot{};
    bool initialized{};
    bool live{};
    node_t id{};
    u32 generation{};
    RemotePtr entry;
  };

  AdaptiveRouteTable(u32 dim, u32 shard_count);

  AdaptiveRouteTable(const AdaptiveRouteTable&) = delete;
  AdaptiveRouteTable& operator=(const AdaptiveRouteTable&) = delete;

  u32 dim() const { return dim_; }
  u32 shard_count() const { return shard_count_; }
  size_t capacity() const {
    return static_cast<size_t>(shard_count_) * kSlotsPerShard;
  }

  // Observe an already-committed live mutation.  New observations update the
  // nearest center in their physical shard with a non-overshooting EMA.  The
  // slot keeps whichever live observed vector is nearer to the updated center.
  // An upsert of the slot's current id always refreshes its pointer and vector.
  // entry.memory_node() must equal shard. Returns false for invalid input or
  // a stale/idempotent generation of an identity still tracked by a slot.
  // Because the table intentionally has no per-ID history, callers must admit
  // only authoritative committed-current observations for non-representatives.
  bool observe(u32 shard,
               node_t id,
               u32 generation,
               RemotePtr entry,
               const span<const element_t>& vector);

  // Invalidate a tracked representative at or below the supplied tombstone
  // generation. A delayed invalidation cannot remove a newer tracked
  // generation. Returns true only when table state changed; caller-side
  // freshness remains authoritative for identities not occupying a slot.
  bool invalidate(node_t id, u32 generation);

  // Pick the nearest live center globally, then the nearest live
  // representative in that center's shard.
  std::optional<Route> route(const span<const element_t>& query) const;

  // Pick a live entry in a prescribed shard.  This is the stage-2/peer-search
  // form of the same routing operation.
  std::optional<Route> route_in_shard(
    const span<const element_t>& query, u32 shard) const;

  // Return every live representative in one shard, ordered by distance to
  // the query. Construction search starts from this complete fixed-capacity
  // route set; graph-search convergence is governed only by construction
  // beam width L, never by an expansion/depth cap.
  vec<Route> routes_in_shard(
    const span<const element_t>& query, u32 shard) const;

  size_t live_count(u32 shard) const;
  void snapshot_route_slots(span<RouteSlotSnapshot> output) const;
  Snapshot snapshot() const;

private:
  struct Slot {
    bool initialized{};
    bool live{};
    node_t id{};
    u32 generation{};
    RemotePtr entry;
    u64 observations{};
    vec<element_t> center;
    vec<element_t> representative;
  };

  struct Shard {
    std::array<Slot, kSlotsPerShard> slots;
  };

  std::optional<Route> route_in_shard_locked(
    const span<const element_t>& query, u32 shard) const;
  void initialize_slot(Slot& slot,
                       node_t id,
                       u32 generation,
                       RemotePtr entry,
                       const span<const element_t>& vector);
  void update_slot(Slot& slot,
                   node_t id,
                   u32 generation,
                   RemotePtr entry,
                   const span<const element_t>& vector,
                   bool force_representative);

  u32 dim_{};
  u32 shard_count_{};
  vec<Shard> shards_;
  mutable std::shared_mutex mutex_;
};

}  // namespace vamana::routing
