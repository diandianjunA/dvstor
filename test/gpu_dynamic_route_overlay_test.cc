#include <cassert>
#include <stdexcept>
#include <vector>

#include "gpu_search/delta_scan_budget.hh"
#include "gpu_search/dynamic_route_overlay.hh"
#include "gpu_search/dynamic_route_consistency.hh"
#include "gpu_search/initial_seed_budget.hh"

namespace {

RemotePtr dynamic_pointer(u32 shard, u64 slot) {
  // Only the shard bits matter to this pure policy test.  Keep the offset
  // non-zero so it is also a valid RemotePtr.
  return RemotePtr{shard, (slot + 1) * 64};
}

}  // namespace

int main() {
  using gpu_search::DynamicRouteOverlayDiff;
  using gpu_search::kDynamicRouteLive;
  using gpu_search::kDynamicRouteSlotsPerShard;
  using vamana::routing::AdaptiveRouteTable;

  // Static and dynamic routes compete inside this total. They must never turn
  // the configured 32-entry route into the old 32+40=72-entry traversal.
  static_assert(gpu_search::initial_seed_budget(32, 128) == 32);
  static_assert(gpu_search::initial_seed_budget(256, 128) == 128);

  // Fixed query work is divided without gaps or overlap. It is independent of
  // the number of live delta records behind each anchor.
  {
    u32 end = 0;
    for (u32 anchor = 0; anchor < 32; ++anchor) {
      const auto segment = gpu_search::delta_scan_segment(anchor, 32);
      assert(segment.offset == end);
      end += segment.count;
    }
    assert(end == gpu_search::kDeltaScanRecordBudget);
    const auto first = gpu_search::delta_scan_segment(0, 3, 8);
    const auto second = gpu_search::delta_scan_segment(1, 3, 8);
    const auto third = gpu_search::delta_scan_segment(2, 3, 8);
    assert(first.offset == 0 && first.count == 3);
    assert(second.offset == 3 && second.count == 3);
    assert(third.offset == 6 && third.count == 2);
    assert(gpu_search::delta_scan_segment(3, 3, 8).count == 0);
  }

  // The score itself must remain inside the seqlock window. An odd writer
  // window or a completed update to a different even generation is rejected.
  static_assert(gpu_search::dynamic_route_window_stable(2, 2));
  static_assert(!gpu_search::dynamic_route_window_stable(2, 3));
  static_assert(!gpu_search::dynamic_route_window_stable(2, 4));
  static_assert(!gpu_search::dynamic_route_window_stable(3, 3));

  // Immutable schema-15 base records use generation zero. Canonical storage
  // routing must be able to publish them as bootstrap representatives.
  {
    DynamicRouteOverlayDiff base_overlay(1);
    std::vector<AdaptiveRouteTable::RouteSlotSnapshot> base_snapshot(
      base_overlay.capacity());
    for (u32 slot = 0; slot < kDynamicRouteSlotsPerShard; ++slot) {
      base_snapshot[slot].shard = 0;
      base_snapshot[slot].slot = slot;
    }
    base_snapshot[0].initialized = true;
    base_snapshot[0].live = true;
    base_snapshot[0].id = 0;
    base_snapshot[0].generation = 0;
    base_snapshot[0].entry = dynamic_pointer(0, 0);
    std::vector<gpu_search::DynamicRouteUpdate> base_updates;
    base_updates.reserve(base_overlay.capacity());
    base_overlay.prepare(base_snapshot, 1, base_updates);
    assert(base_updates.size() == 1);
    assert(base_updates[0].generation == 0);
    base_overlay.commit(base_updates);
  }

  AdaptiveRouteTable routes(2, 2);
  DynamicRouteOverlayDiff overlay(2);
  std::vector<AdaptiveRouteTable::RouteSlotSnapshot> snapshot(
    overlay.capacity());
  std::vector<gpu_search::DynamicRouteUpdate> updates;
  updates.reserve(overlay.capacity());
  const auto prepare = [&](u64 epoch) {
    routes.snapshot_route_slots(snapshot);
    overlay.prepare(snapshot, epoch, updates);
  };
  assert(overlay.capacity() == 2 * kDynamicRouteSlotsPerShard);
  prepare(1);
  assert(updates.empty());

  const std::vector<element_t> first{1.0F, 2.0F};
  assert(routes.observe(0, 10, 4, dynamic_pointer(0, 1), first));
  prepare(7);
  assert(updates.size() == 1);
  assert(updates[0].slot < kDynamicRouteSlotsPerShard);
  assert(updates[0].shard == 0);
  assert(updates[0].id == 10);
  assert(updates[0].generation == 4);
  assert(updates[0].epoch == 7);
  assert(updates[0].flags == kDynamicRouteLive);
  overlay.commit(updates);
  prepare(8);
  assert(updates.empty());

  // Stale/idempotent observations cannot roll back a published route.
  assert(!routes.observe(0, 10, 4, dynamic_pointer(0, 2), first));
  prepare(8);
  assert(updates.empty());

  const std::vector<element_t> moved{3.0F, 4.0F};
  assert(routes.observe(0, 10, 5, dynamic_pointer(0, 2), moved));
  prepare(8);
  assert(updates.size() == 1);
  assert(updates[0].generation == 5);
  assert(updates[0].remote_node == dynamic_pointer(0, 2).raw_address);
  overlay.commit(updates);

  bool rejected = false;
  try {
    const gpu_search::DynamicRouteUpdate stale{
      .epoch = 7,
      .remote_node = dynamic_pointer(0, 1).raw_address,
      .slot = updates[0].slot,
      .shard = 0,
      .id = 10,
      .generation = 4,
      .flags = kDynamicRouteLive,
    };
    overlay.commit(span<const gpu_search::DynamicRouteUpdate>{&stale, 1});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  // A delayed delete is ignored.  The current generation publishes a
  // generation-aware tombstone, and a later different id may reuse the same
  // fixed slot even though its generation number is smaller.
  assert(!routes.invalidate(10, 4));
  prepare(9);
  assert(updates.empty());
  assert(routes.invalidate(10, 5));
  prepare(9);
  assert(updates.size() == 1);
  assert(updates[0].generation == 5);
  assert(updates[0].flags == 0);
  assert(updates[0].remote_node == 0);
  const u32 recycled_slot = updates[0].slot;
  overlay.commit(updates);

  const std::vector<element_t> replacement{5.0F, 6.0F};
  assert(routes.observe(0, 11, 1, dynamic_pointer(0, 3), replacement));
  prepare(10);
  assert(updates.size() == 1);
  assert(updates[0].slot == recycled_slot);
  assert(updates[0].id == 11);
  assert(updates[0].generation == 1);
  assert(updates[0].flags == kDynamicRouteLive);
  overlay.commit(updates);

  const std::vector<element_t> other_shard{9.0F, 9.0F};
  assert(routes.observe(1, 20, 1, dynamic_pointer(1, 0), other_shard));
  prepare(11);
  assert(updates.size() == 1);
  assert(updates[0].slot >= kDynamicRouteSlotsPerShard);
  assert(updates[0].shard == 1);
  overlay.commit(updates);

  // A route pointer may never escape its physical shard.
  assert(!routes.observe(0, 30, 1, dynamic_pointer(1, 4), first));
  prepare(12);
  assert(updates.empty());

  rejected = false;
  try {
    prepare(0);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}
