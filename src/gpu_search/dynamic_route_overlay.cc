#include "gpu_search/dynamic_route_overlay.hh"

#include <stdexcept>

namespace gpu_search {

static_assert(kDynamicRouteSlotsPerShard ==
              vamana::routing::AdaptiveRouteTable::kSlotsPerShard);

DynamicRouteOverlayDiff::DynamicRouteOverlayDiff(u32 shard_count)
    : shard_count_(shard_count),
      slots_(static_cast<size_t>(shard_count) *
             kDynamicRouteSlotsPerShard) {
  if (shard_count == 0) {
    throw std::invalid_argument(
      "dynamic route overlay requires at least one shard");
  }
  for (u32 shard = 0; shard < shard_count_; ++shard) {
    for (u32 local = 0; local < kDynamicRouteSlotsPerShard; ++local) {
      slots_[static_cast<size_t>(shard) * kDynamicRouteSlotsPerShard + local]
        .shard = shard;
    }
  }
}

void DynamicRouteOverlayDiff::prepare(
    span<const vamana::routing::AdaptiveRouteTable::RouteSlotSnapshot> snapshot,
    u64 epoch,
    std::vector<DynamicRouteUpdate>& updates) const {
  if (epoch == 0 || snapshot.size() != slots_.size()) {
    throw std::invalid_argument("invalid adaptive route snapshot for GPU overlay");
  }

  updates.clear();
  if (updates.capacity() < slots_.size()) {
    throw std::invalid_argument(
      "GPU dynamic route update buffer was not preallocated");
  }
  for (const auto& source : snapshot) {
    if (source.shard >= shard_count_ ||
        source.slot >= kDynamicRouteSlotsPerShard) {
      throw std::invalid_argument("adaptive route snapshot contains an invalid slot");
    }
    const u32 slot = source.shard * kDynamicRouteSlotsPerShard + source.slot;
    const Slot& current = slots_[slot];
    const u32 flags = source.live ? kDynamicRouteLive : 0u;
    const u64 remote_node = source.live ? source.entry.raw_address : 0u;
    if (source.live &&
        (remote_node == 0 ||
         source.entry.memory_node() != source.shard)) {
      throw std::invalid_argument("adaptive route snapshot contains an invalid live entry");
    }
    if (current.shard == source.shard && current.id == source.id &&
        current.generation == source.generation &&
        current.remote_node == remote_node && current.flags == flags) {
      continue;
    }
    updates.push_back(DynamicRouteUpdate{
      .epoch = epoch,
      .remote_node = remote_node,
      .slot = slot,
      .shard = source.shard,
      .id = source.id,
      .generation = source.generation,
      .flags = flags,
    });
  }
}

void DynamicRouteOverlayDiff::commit(
    std::span<const DynamicRouteUpdate> updates) {
  for (const DynamicRouteUpdate& update : updates) {
    if (update.slot >= slots_.size() || update.shard >= shard_count_ ||
        update.slot / kDynamicRouteSlotsPerShard != update.shard ||
        update.epoch == 0 || (update.flags & ~kDynamicRouteLive) != 0 ||
        ((update.flags & kDynamicRouteLive) != 0 &&
         (update.remote_node == 0 ||
          static_cast<u32>(update.remote_node >> 48) != update.shard))) {
      throw std::invalid_argument("invalid committed GPU dynamic route update");
    }
    const Slot& current = slots_[update.slot];
    if ((current.epoch != 0 && update.epoch <= current.epoch) ||
        (current.id == update.id &&
         current.generation > update.generation)) {
      throw std::invalid_argument("stale committed GPU dynamic route update");
    }
    slots_[update.slot] = Slot{
      .epoch = update.epoch,
      .remote_node = update.remote_node,
      .shard = update.shard,
      .id = update.id,
      .generation = update.generation,
      .flags = update.flags,
    };
  }
}

}  // namespace gpu_search
