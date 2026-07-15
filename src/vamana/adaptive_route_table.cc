#include "vamana/adaptive_route_table.hh"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "common/distance.hh"

namespace vamana::routing {

AdaptiveRouteTable::AdaptiveRouteTable(u32 dim, u32 shard_count)
    : dim_(dim), shard_count_(shard_count), shards_(shard_count) {
  if (dim == 0 || shard_count == 0) {
    throw std::invalid_argument(
      "adaptive route table requires non-zero dimension and shard count");
  }
  for (Shard& shard : shards_) {
    for (Slot& slot : shard.slots) {
      // All mutation-time storage is allocated here.  observe() never grows
      // route state, even after an unbounded mutation stream.
      slot.center.resize(dim_);
      slot.representative.resize(dim_);
    }
  }
}

void AdaptiveRouteTable::initialize_slot(
    Slot& slot,
    node_t id,
    u32 generation,
    RemotePtr entry,
    const span<const element_t>& vector) {
  std::copy(vector.begin(), vector.end(), slot.center.begin());
  std::copy(vector.begin(), vector.end(), slot.representative.begin());
  slot.initialized = true;
  slot.live = true;
  slot.id = id;
  slot.generation = generation;
  slot.entry = entry;
  slot.observations = 1;
}

void AdaptiveRouteTable::update_slot(
    Slot& slot,
    node_t id,
    u32 generation,
    RemotePtr entry,
    const span<const element_t>& vector,
    bool force_representative) {
  for (u32 dimension = 0; dimension < dim_; ++dimension) {
    const element_t old_center = slot.center[dimension];
    slot.center[dimension] = old_center +
      kCenterEmaWeight * (vector[dimension] - old_center);
  }
  if (slot.observations != std::numeric_limits<u64>::max()) {
    ++slot.observations;
  }

  const distance_t current_distance = L2Distance::dist(
    slot.center, slot.representative, dim_);
  const distance_t candidate_distance = L2Distance::dist(
    slot.center, vector, dim_);
  if (force_representative || candidate_distance <= current_distance) {
    std::copy(vector.begin(), vector.end(), slot.representative.begin());
    slot.id = id;
    slot.generation = generation;
    slot.entry = entry;
    slot.live = true;
  }
}

bool AdaptiveRouteTable::observe(
    u32 shard,
    node_t id,
    u32 generation,
    RemotePtr entry,
    const span<const element_t>& vector) {
  if (shard >= shard_count_ || vector.size() != dim_ || entry.is_null() ||
      entry.memory_node() != shard) {
    return false;
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  bool identity_found = false;
  u32 newest_generation = 0;
  Slot* same_shard_representative = nullptr;
  for (u32 current_shard = 0; current_shard < shard_count_; ++current_shard) {
    for (Slot& slot : shards_[current_shard].slots) {
      if (!slot.initialized || slot.id != id) continue;
      if (!identity_found || slot.generation > newest_generation) {
        newest_generation = slot.generation;
        identity_found = true;
      }
      if (current_shard == shard && slot.live &&
          (same_shard_representative == nullptr ||
           slot.generation > same_shard_representative->generation)) {
        same_shard_representative = &slot;
      }
    }
  }
  if (identity_found && generation <= newest_generation) {
    return false;
  }

  if (same_shard_representative != nullptr) {
    // The representative itself was upserted.  Its old vector/pointer cannot
    // remain a route entry even if another observation would be geometrically
    // closer to the center.
    update_slot(*same_shard_representative, id, generation, entry, vector, true);
    return true;
  }

  // A representative that moved to another shard must stop routing readers to
  // its old physical location before the new observation is installed.
  if (identity_found) {
    for (Shard& current_shard : shards_) {
      for (Slot& slot : current_shard.slots) {
        if (slot.initialized && slot.live && slot.id == id) {
          slot.live = false;
          slot.entry.reset();
        }
      }
    }
  }

  Shard& destination = shards_[shard];
  for (Slot& slot : destination.slots) {
    if (!slot.live) {
      // Empty/deleted slots are reset instead of slowly dragging a stale
      // center across the vector space.
      initialize_slot(slot, id, generation, entry, vector);
      return true;
    }
  }

  Slot* nearest = &destination.slots.front();
  distance_t nearest_distance = L2Distance::dist(
    vector, nearest->center, dim_);
  for (u32 slot_index = 1; slot_index < kSlotsPerShard; ++slot_index) {
    Slot& candidate = destination.slots[slot_index];
    const distance_t distance = L2Distance::dist(
      vector, candidate.center, dim_);
    if (distance < nearest_distance) {
      nearest = &candidate;
      nearest_distance = distance;
    }
  }
  update_slot(*nearest, id, generation, entry, vector, false);
  return true;
}

bool AdaptiveRouteTable::invalidate(node_t id, u32 generation) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  bool changed = false;
  for (Shard& shard : shards_) {
    for (Slot& slot : shard.slots) {
      if (!slot.initialized || slot.id != id || generation < slot.generation) {
        continue;
      }
      if (slot.live || generation > slot.generation) {
        changed = true;
      }
      slot.live = false;
      slot.generation = generation;
      slot.entry.reset();
    }
  }
  return changed;
}

std::optional<AdaptiveRouteTable::Route>
AdaptiveRouteTable::route_in_shard_locked(
    const span<const element_t>& query, u32 shard) const {
  if (shard >= shard_count_ || query.size() != dim_) return std::nullopt;

  const Shard& selected = shards_[shard];
  distance_t shard_distance = std::numeric_limits<distance_t>::max();
  const Slot* nearest_entry = nullptr;
  distance_t entry_distance = std::numeric_limits<distance_t>::max();
  for (const Slot& slot : selected.slots) {
    if (!slot.live) continue;
    shard_distance = std::min(
      shard_distance, L2Distance::dist(query, slot.center, dim_));
    const distance_t distance = L2Distance::dist(
      query, slot.representative, dim_);
    if (nearest_entry == nullptr || distance < entry_distance) {
      nearest_entry = &slot;
      entry_distance = distance;
    }
  }
  if (nearest_entry == nullptr) return std::nullopt;
  return Route{
    .shard = shard,
    .id = nearest_entry->id,
    .generation = nearest_entry->generation,
    .entry = nearest_entry->entry,
    .shard_distance = shard_distance,
    .entry_distance = entry_distance,
  };
}

std::optional<AdaptiveRouteTable::Route> AdaptiveRouteTable::route(
    const span<const element_t>& query) const {
  if (query.size() != dim_) return std::nullopt;
  std::shared_lock<std::shared_mutex> lock(mutex_);

  std::optional<u32> nearest_shard;
  distance_t nearest_distance = std::numeric_limits<distance_t>::max();
  for (u32 shard = 0; shard < shard_count_; ++shard) {
    for (const Slot& slot : shards_[shard].slots) {
      if (!slot.live) continue;
      const distance_t distance = L2Distance::dist(query, slot.center, dim_);
      if (!nearest_shard.has_value() || distance < nearest_distance) {
        nearest_shard = shard;
        nearest_distance = distance;
      }
    }
  }
  if (!nearest_shard.has_value()) return std::nullopt;
  return route_in_shard_locked(query, *nearest_shard);
}

std::optional<AdaptiveRouteTable::Route> AdaptiveRouteTable::route_in_shard(
    const span<const element_t>& query, u32 shard) const {
  if (query.size() != dim_ || shard >= shard_count_) return std::nullopt;
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return route_in_shard_locked(query, shard);
}

vec<AdaptiveRouteTable::Route> AdaptiveRouteTable::routes_in_shard(
    const span<const element_t>& query, u32 shard) const {
  vec<Route> routes;
  if (query.size() != dim_ || shard >= shard_count_) return routes;
  std::shared_lock<std::shared_mutex> lock(mutex_);

  routes.reserve(kSlotsPerShard);
  for (const Slot& slot : shards_[shard].slots) {
    if (!slot.live) continue;
    routes.push_back(Route{
      .shard = shard,
      .id = slot.id,
      .generation = slot.generation,
      .entry = slot.entry,
      .shard_distance = L2Distance::dist(query, slot.center, dim_),
      .entry_distance = L2Distance::dist(
        query, slot.representative, dim_),
    });
  }
  std::sort(routes.begin(), routes.end(),
            [](const Route& lhs, const Route& rhs) {
              if (lhs.entry_distance != rhs.entry_distance) {
                return lhs.entry_distance < rhs.entry_distance;
              }
              return lhs.entry.raw_address < rhs.entry.raw_address;
            });
  return routes;
}

size_t AdaptiveRouteTable::live_count(u32 shard) const {
  if (shard >= shard_count_) return 0;
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return static_cast<size_t>(std::count_if(
    shards_[shard].slots.begin(), shards_[shard].slots.end(),
    [](const Slot& slot) { return slot.live; }));
}

void AdaptiveRouteTable::snapshot_route_slots(
    span<RouteSlotSnapshot> output) const {
  if (output.size() != capacity()) {
    throw std::invalid_argument(
      "adaptive route metadata snapshot has the wrong capacity");
  }
  std::shared_lock<std::shared_mutex> lock(mutex_);
  for (u32 shard = 0; shard < shard_count_; ++shard) {
    for (u32 slot_index = 0; slot_index < kSlotsPerShard; ++slot_index) {
      const Slot& slot = shards_[shard].slots[slot_index];
      output[static_cast<size_t>(shard) * kSlotsPerShard + slot_index] =
        RouteSlotSnapshot{
          .shard = shard,
          .slot = slot_index,
          .initialized = slot.initialized,
          .live = slot.live,
          .id = slot.id,
          .generation = slot.generation,
          .entry = slot.entry,
        };
    }
  }
}

AdaptiveRouteTable::Snapshot AdaptiveRouteTable::snapshot() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  Snapshot result{
    .dim = dim_,
    .shard_count = shard_count_,
    .slots = {},
  };
  result.slots.reserve(capacity());
  for (u32 shard = 0; shard < shard_count_; ++shard) {
    for (u32 slot_index = 0; slot_index < kSlotsPerShard; ++slot_index) {
      const Slot& slot = shards_[shard].slots[slot_index];
      result.slots.push_back(SlotSnapshot{
        .shard = shard,
        .slot = slot_index,
        .initialized = slot.initialized,
        .live = slot.live,
        .id = slot.id,
        .generation = slot.generation,
        .entry = slot.entry,
        .observations = slot.observations,
        .center = slot.center,
        .representative = slot.representative,
      });
    }
  }
  return result;
}

}  // namespace vamana::routing
