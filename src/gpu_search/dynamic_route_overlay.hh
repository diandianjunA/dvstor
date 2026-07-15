#pragma once

#include <span>
#include <vector>

#include "gpu_search/types.hh"
#include "vamana/adaptive_route_table.hh"

namespace gpu_search {

// Converts the fixed storage-canonical route snapshot into the minimal set of
// GPU slot updates. prepare() is side-effect free; commit() advances the
// compute-side mirror only after the control CTA has acknowledged the command.
class DynamicRouteOverlayDiff {
public:
  explicit DynamicRouteOverlayDiff(u32 shard_count);

  u32 capacity() const { return static_cast<u32>(slots_.size()); }

  void prepare(
    span<const vamana::routing::AdaptiveRouteTable::RouteSlotSnapshot> snapshot,
    u64 epoch,
    std::vector<DynamicRouteUpdate>& updates) const;

  void commit(std::span<const DynamicRouteUpdate> updates);

private:
  struct Slot {
    u64 epoch{};
    u64 remote_node{};
    u32 shard{};
    u32 id{};
    u32 generation{};
    u32 flags{};
  };

  u32 shard_count_{};
  std::vector<Slot> slots_;
};

}  // namespace gpu_search
