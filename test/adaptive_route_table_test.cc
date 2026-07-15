#include <cassert>
#include <stdexcept>

#include "vamana/adaptive_route_table.hh"

namespace {

using RouteTable = vamana::routing::AdaptiveRouteTable;

vec<element_t> point(element_t x) {
  return {x};
}

RemotePtr pointer(u32 shard, u64 offset) {
  return RemotePtr{shard, offset + 1};
}

void test_validation_and_basic_routing() {
  bool rejected_empty_shape = false;
  try {
    RouteTable invalid(0, 2);
  } catch (const std::invalid_argument&) {
    rejected_empty_shape = true;
  }
  assert(rejected_empty_shape);

  RouteTable routes(1, 2);
  assert(!routes.route(point(0)).has_value());
  assert(!routes.observe(2, 1, 1, pointer(0, 1), point(0)));
  assert(!routes.observe(0, 1, 1, RemotePtr{}, point(0)));
  assert(!routes.observe(0, 1, 1, pointer(1, 1), point(0)));
  assert(routes.live_count(0) == 0);
  const vec<element_t> wrong_dimension{0, 1};
  assert(!routes.observe(0, 1, 1, pointer(0, 1), wrong_dimension));

  const RemotePtr left = pointer(0, 10);
  const RemotePtr right = pointer(1, 20);
  assert(routes.observe(0, 10, 1, left, point(0)));
  assert(routes.observe(1, 20, 1, right, point(100)));

  auto route = routes.route(point(4));
  assert(route.has_value());
  assert(route->shard == 0);
  assert(route->entry == left);
  route = routes.route(point(90));
  assert(route.has_value());
  assert(route->shard == 1);
  assert(route->entry == right);

  route = routes.route_in_shard(point(4), 1);
  assert(route.has_value());
  assert(route->shard == 1);
  assert(route->entry == right);
  assert(!routes.route_in_shard(point(4), 2).has_value());
}

void test_fixed_capacity_ema_adapts_routing_and_entry() {
  RouteTable routes(1, 2);
  for (u32 slot = 0; slot < RouteTable::kSlotsPerShard; ++slot) {
    assert(routes.observe(0, 100 + slot, 1, pointer(0, slot), point(0)));
    assert(routes.observe(1, 200 + slot, 1, pointer(1, slot), point(100)));
  }
  assert(routes.capacity() == 2 * RouteTable::kSlotsPerShard);
  assert(routes.live_count(0) == RouteTable::kSlotsPerShard);
  assert(routes.live_count(1) == RouteTable::kSlotsPerShard);
  assert(routes.route(point(10))->shard == 0);

  // The table is full, so every mutation updates one of the existing centers;
  // no slot or side map is appended.  Repeated live mutations move one shard's
  // center from 100 toward 10 and eventually change the routing decision.
  for (u32 mutation = 0; mutation < 48; ++mutation) {
    assert(routes.observe(1,
                          1000 + mutation,
                          1,
                          pointer(1, 1000 + mutation),
                          point(10)));
  }
  assert(routes.capacity() == 2 * RouteTable::kSlotsPerShard);
  assert(routes.live_count(1) == RouteTable::kSlotsPerShard);
  const auto route = routes.route(point(10));
  assert(route.has_value());
  assert(route->shard == 1);
  assert(route->id >= 1000);

  const auto snapshot = routes.snapshot();
  assert(snapshot.slots.size() == routes.capacity());
  bool found_adapted_center = false;
  for (const auto& slot : snapshot.slots) {
    if (slot.shard == 1 && slot.observations > 1) {
      assert(slot.center[0] > 10);
      assert(slot.center[0] < 11);
      found_adapted_center = true;
    }
  }
  assert(found_adapted_center);
}

void test_generation_safe_upsert_and_invalidation() {
  RouteTable routes(1, 1);
  const RemotePtr generation_one = pointer(0, 1);
  const RemotePtr generation_two = pointer(0, 2);
  const RemotePtr generation_three = pointer(0, 3);
  assert(routes.observe(0, 7, 1, generation_one, point(0)));
  assert(routes.observe(0, 7, 2, generation_two, point(1)));

  auto route = routes.route(point(1));
  assert(route.has_value());
  assert(route->id == 7);
  assert(route->generation == 2);
  assert(route->entry == generation_two);

  // A delayed cleanup for generation 1 cannot invalidate generation 2, and a
  // delayed observation cannot roll the route back either.
  assert(!routes.invalidate(7, 1));
  assert(!routes.observe(0, 7, 1, generation_one, point(0)));
  assert(routes.route(point(1))->entry == generation_two);

  assert(routes.invalidate(7, 2));
  assert(!routes.route(point(1)).has_value());
  assert(!routes.invalidate(7, 2));
  assert(!routes.observe(0, 7, 2, generation_two, point(1)));
  assert(routes.observe(0, 7, 3, generation_three, point(2)));
  route = routes.route(point(2));
  assert(route.has_value());
  assert(route->generation == 3);
  assert(route->entry == generation_three);

  // A newer tombstone supersedes the current representative.
  assert(routes.invalidate(7, 4));
  assert(!routes.observe(0, 7, 4, pointer(0, 4), point(3)));
  assert(routes.observe(0, 7, 5, pointer(0, 5), point(4)));
}

void test_cross_shard_upsert_retires_old_entry() {
  RouteTable routes(1, 2);
  const RemotePtr old_entry = pointer(0, 70);
  const RemotePtr new_entry = pointer(1, 71);
  assert(routes.observe(0, 7, 1, old_entry, point(0)));
  assert(routes.observe(1, 7, 2, new_entry, point(100)));
  assert(!routes.route_in_shard(point(0), 0).has_value());

  const auto route = routes.route_in_shard(point(100), 1);
  assert(route.has_value());
  assert(route->entry == new_entry);
  assert(route->generation == 2);
  assert(!routes.invalidate(7, 1));
  assert(routes.route_in_shard(point(100), 1)->entry == new_entry);
}

void test_delete_falls_back_to_another_live_entry() {
  RouteTable routes(1, 1);
  const RemotePtr near = pointer(0, 10);
  const RemotePtr fallback = pointer(0, 20);
  assert(routes.observe(0, 10, 1, near, point(0)));
  assert(routes.observe(0, 20, 1, fallback, point(20)));
  assert(routes.route(point(1))->entry == near);
  assert(routes.invalidate(10, 1));
  const auto route = routes.route(point(1));
  assert(route.has_value());
  assert(route->entry == fallback);
}

void test_routes_in_shard_returns_all_entries_in_distance_order() {
  RouteTable routes(1, 1);
  assert(routes.observe(0, 10, 1, pointer(0, 10), point(10)));
  assert(routes.observe(0, 20, 1, pointer(0, 20), point(2)));
  assert(routes.observe(0, 30, 1, pointer(0, 30), point(6)));

  const auto entries = routes.routes_in_shard(point(0), 0);
  assert(entries.size() == 3);
  assert(entries[0].id == 20);
  assert(entries[1].id == 30);
  assert(entries[2].id == 10);
}

}  // namespace

int main() {
  test_validation_and_basic_routing();
  test_fixed_capacity_ema_adapts_routing_and_entry();
  test_generation_safe_upsert_and_invalidation();
  test_cross_shard_upsert_retires_old_entry();
  test_delete_falls_back_to_another_live_entry();
  test_routes_in_shard_returns_all_entries_in_distance_order();
  return 0;
}
