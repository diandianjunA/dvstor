#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

#include "vamana/centroid_state.hh"
#include "vamana/centroid_router.hh"
#include "vamana/hot_graph.hh"

namespace {

using Router = vamana::routing::CentroidRouter;

vec<f32> point(std::initializer_list<f32> values) {
  return values;
}

RemotePtr pointer(u32 shard, u64 offset) {
  return RemotePtr{shard, (offset + 1) * 16};
}

Router::LiveEntry entry(u32 shard, u64 offset, u32 generation) {
  return Router::LiveEntry{
    .pointer = pointer(shard, offset),
    .generation = generation,
  };
}

bool near(f64 lhs, f64 rhs) {
  return std::abs(lhs - rhs) <= 1e-12;
}

f64 sum_component(const Router::ShardSnapshot& shard, size_t dimension) {
  // A never-populated empty shard deliberately owns no O(dim) sum buffer.
  // Once a shard has held data, its canonical empty state keeps an explicit
  // zero vector.  Both representations denote the same exact empty sum.
  return shard.sum.empty() ? 0.0 : shard.sum[dimension];
}

void test_restore_success_and_publication() {
  Router router(3, 2);
  const auto initial = router.snapshot();
  const vec<f64> shard_sum{6.0, 9.0, 12.0};
  const vec<Router::LiveEntry> entries{
    entry(0, 10, 3), entry(0, 20, 5)};
  assert(router.restore_shard_state(0, 3, shard_sum, entries, 17));
  const vec<f64> authoritative = router.authoritative_centroid(0);
  assert(authoritative == vec<f64>({2.0, 3.0, 4.0}));

  const vec<f64> empty_sum{0.0, -0.0, 0.0};
  const vec<Router::LiveEntry> no_entries;
  assert(router.restore_shard_state(
    1, 0, empty_sum, no_entries, 4));
  assert(router.authoritative_centroid(1).empty());

  // Restore obeys the same explicit publication boundary as online updates.
  assert(router.snapshot() == initial);
  assert(router.publish());

  const auto restored = router.snapshot();
  assert(restored != initial);
  assert(restored->version == 17);
  assert(restored->shards[0].version == 17);
  assert(restored->shards[0].count == 3);
  assert(restored->shards[0].sum == shard_sum);
  assert(near(restored->shards[0].centroid[0], 2.0));
  assert(near(restored->shards[0].centroid[1], 3.0));
  assert(near(restored->shards[0].centroid[2], 4.0));
  assert(restored->shards[0].entries().size() == 2);
  assert(restored->shards[0].entries()[1] == entries[1]);
  assert(restored->shards[1].version == 4);
  assert(restored->shards[1].count == 0);
  assert(restored->shards[1].entries().empty());
  assert(restored->shards[1].sum == vec<f64>({0.0, -0.0, 0.0}));

}

void test_restore_rejects_invalid_state() {
  Router router(2, 2);
  const vec<f64> valid_sum{1.0, 2.0};
  const vec<Router::LiveEntry> valid_entries{entry(0, 10, 1)};
  const vec<Router::LiveEntry> no_entries;

  assert(!router.restore_shard_state(2, 1, valid_sum, valid_entries, 1));
  assert(!router.restore_shard_state(
    0, 1, vec<f64>{1.0}, valid_entries, 1));
  assert(!router.restore_shard_state(
    0, 1,
    vec<f64>{std::numeric_limits<f64>::quiet_NaN(), 2.0},
    valid_entries, 1));
  assert(!router.restore_shard_state(
    0, 1,
    vec<f64>{1.0, std::numeric_limits<f64>::infinity()},
    valid_entries, 1));
  assert(!router.restore_shard_state(
    0, 0, vec<f64>{1.0, 0.0}, no_entries, 1));
  assert(!router.restore_shard_state(
    0, 0, vec<f64>{0.0, 0.0}, valid_entries, 1));
  assert(!router.restore_shard_state(
    0, 1, valid_sum, no_entries, 1));
  assert(!router.restore_shard_state(
    0, 1, valid_sum,
    vec<Router::LiveEntry>{entry(0, 10, 1), entry(0, 20, 1)}, 1));
  assert(!router.restore_shard_state(
    0, 1, valid_sum,
    vec<Router::LiveEntry>{Router::LiveEntry{}}, 1));
  assert(!router.restore_shard_state(
    0, 1, valid_sum,
    vec<Router::LiveEntry>{entry(1, 10, 1)}, 1));
  assert(!router.restore_shard_state(
    0, 2, valid_sum,
    vec<Router::LiveEntry>{entry(0, 10, 1), entry(0, 10, 2)}, 1));
  assert(!router.restore_shard_state(
    0, 5, valid_sum,
    vec<Router::LiveEntry>{
      entry(0, 1, 1), entry(0, 2, 1), entry(0, 3, 1),
      entry(0, 4, 1), entry(0, 5, 1)}, 1));
  assert(!router.restore_shard_state(
    0, 1, valid_sum, valid_entries, 0));

  // Invalid attempts neither mutate state nor consume the one restore slot.
  assert(router.snapshot()->version == 0);
  assert(router.restore_shard_state(0, 1, valid_sum, valid_entries, 1));
  assert(!router.restore_shard_state(0, 1, valid_sum, valid_entries, 2));
  assert(router.publish());
  assert(router.snapshot()->shards[0].count == 1);
}

void test_restore_preserves_fp64_precision() {
  Router router(2, 1);
  // Both sums are exact doubles but lose information if routed through f32.
  const vec<f64> exact_sum{50331651.0, -50331657.0};
  const vec<Router::LiveEntry> entries{entry(0, 99, 8)};
  assert(router.restore_shard_state(0, 3, exact_sum, entries, 23));
  assert(router.publish());

  const auto snapshot = router.snapshot();
  assert(snapshot->shards[0].sum[0] == 50331651.0);
  assert(snapshot->shards[0].sum[1] == -50331657.0);
  assert(snapshot->shards[0].centroid[0] == 16777217.0);
  assert(snapshot->shards[0].centroid[1] == -16777219.0);
}

void test_restore_versions_and_startup_window() {
  Router router(1, 2);
  const vec<Router::LiveEntry> left{entry(0, 10, 1)};
  const vec<Router::LiveEntry> right{entry(1, 20, 2)};
  assert(router.restore_shard_state(0, 1, vec<f64>{1.0}, left, 9));
  assert(router.restore_shard_state(1, 1, vec<f64>{2.0}, right, 4));
  assert(router.publish());

  auto snapshot = router.snapshot();
  assert(snapshot->version == 9);
  assert(snapshot->shards[0].version == 9);
  assert(snapshot->shards[1].version == 4);

  assert(router.insert(1, point({3.0F})));
  assert(router.publish());
  snapshot = router.snapshot();
  assert(snapshot->version == 10);
  assert(snapshot->shards[0].version == 9);
  assert(snapshot->shards[1].version == 5);
  assert(snapshot->shards[1].count == 2);
  assert(snapshot->shards[1].sum[0] == 5.0);
  assert(!router.restore_shard_state(
    0, 1, vec<f64>{1.0}, left, 11));

  Router published_empty(1, 1);
  assert(!published_empty.publish());
  assert(!published_empty.restore_shard_state(
    0, 1, vec<f64>{1.0}, left, 1));

  Router mutated(1, 2);
  assert(mutated.insert(0, point({1.0F})));
  assert(!mutated.restore_shard_state(
    1, 1, vec<f64>{2.0}, right, 1));
}

void test_validation_and_explicit_live_entries() {
  bool rejected = false;
  try {
    Router invalid(0, 2);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  Router router(2, 2);
  assert(router.snapshot()->version == 0);
  assert(!router.insert(2, point({0, 0})));
  assert(!router.insert(0, point({0})));
  assert(!router.insert(
    0, point({std::numeric_limits<f32>::infinity(), 0})));

  assert(router.insert(0, point({1, 2})));
  // A populated shard is deliberately not routable until the caller supplies
  // at least one real live graph entry.
  assert(router.snapshot()->shards[0].live_entry_count == 0);

  vec<Router::LiveEntry> entries{entry(0, 10, 4), entry(0, 20, 7)};
  // A route cannot advertise more distinct entry records than the physical
  // shard currently contains.
  assert(!router.replace_live_entries(0, entries));
  assert(router.insert(0, point({3, 4})));
  assert(router.replace_live_entries(0, entries));
  assert(!router.replace_live_entries(0, entries));  // idempotent no-op

  vec<Router::LiveEntry> wrong_shard{entry(1, 10, 1)};
  assert(!router.replace_live_entries(0, wrong_shard));
  vec<Router::LiveEntry> duplicates{entry(0, 10, 4), entry(0, 10, 5)};
  assert(!router.replace_live_entries(0, duplicates));
  vec<Router::LiveEntry> too_many{
    entry(0, 1, 1), entry(0, 2, 1), entry(0, 3, 1),
    entry(0, 4, 1), entry(0, 5, 1)};
  assert(!router.replace_live_entries(0, too_many));

  // Neither the centroid nor entry mutations are visible until the explicit
  // maintenance publication boundary.
  assert(router.snapshot()->shards[0].live_entry_count == 0);
  assert(router.publish());
  assert(!router.publish());

  const auto snapshot = router.snapshot();
  assert(snapshot->shards[0].entries().size() == 2);
  assert(snapshot->shards[0].entries()[0].pointer == entries[0].pointer);
  assert(snapshot->shards[0].entries()[0].generation == 4);
}

void test_exact_insert_erase_and_upsert() {
  Router router(2, 1);
  assert(router.insert(0, point({1, 2})));
  assert(router.insert(0, point({3, 4})));
  assert(router.authoritative_count(0) == 2);
  // Writer-side state includes both mutations while readers still observe
  // the previous immutable publication.
  assert(router.snapshot()->shards[0].count == 0);
  vec<Router::LiveEntry> entries{entry(0, 3, 1)};
  assert(router.replace_live_entries(0, entries));
  assert(router.publish());

  auto snapshot = router.snapshot();
  assert(snapshot->version == 3);
  assert(snapshot->shards[0].version == 3);
  assert(snapshot->shards[0].count == 2);
  assert(near(snapshot->shards[0].sum[0], 4));
  assert(near(snapshot->shards[0].sum[1], 6));
  assert(near(snapshot->shards[0].centroid[0], 2));
  assert(near(snapshot->shards[0].centroid[1], 3));

  assert(router.upsert(0, point({1, 2}), point({5, 6})));
  assert(router.publish());
  snapshot = router.snapshot();
  assert(snapshot->shards[0].count == 2);
  assert(near(snapshot->shards[0].sum[0], 8));
  assert(near(snapshot->shards[0].sum[1], 10));
  assert(near(snapshot->shards[0].centroid[0], 4));
  assert(near(snapshot->shards[0].centroid[1], 5));

  assert(router.erase(0, point({3, 4})));
  assert(router.publish());
  snapshot = router.snapshot();
  assert(snapshot->shards[0].count == 1);
  assert(near(snapshot->shards[0].sum[0], 5));
  assert(near(snapshot->shards[0].sum[1], 6));

  assert(router.erase(0, point({5, 6})));
  assert(router.authoritative_count(0) == 0);
  assert(router.publish());
  snapshot = router.snapshot();
  assert(snapshot->shards[0].count == 0);
  assert(near(snapshot->shards[0].sum[0], 0));
  assert(near(snapshot->shards[0].sum[1], 0));
  assert(snapshot->shards[0].entries().empty());
  assert(!router.erase(0, point({5, 6})));
  assert(!router.upsert(0, point({5, 6}), point({7, 8})));
}

void test_compensated_high_dynamic_range_updates() {
  Router router(1, 1);
  const f32 large = std::ldexp(1.0F, 100);
  assert(std::isfinite(large));
  assert(router.insert(0, point({large})));
  assert(router.insert(0, point({1.0F})));
  assert(router.insert(0, point({-large})));
  assert(router.replace_live_entries(
    0, vec<Router::LiveEntry>{entry(0, 44, 1)}));
  assert(router.publish());

  // A naive FP64 recurrence computes (2^100 + 1) - 2^100 as zero.
  // Compensated state retains the small vector and therefore the correct
  // centroid under arbitrary finite float32 data ranges.
  auto snapshot = router.snapshot();
  assert(snapshot->shards[0].sum[0] == 1.0);
  assert(snapshot->shards[0].centroid[0] == 1.0 / 3.0);

  assert(router.erase(0, point({1.0F})));
  assert(router.publish());
  snapshot = router.snapshot();
  assert(snapshot->shards[0].count == 2);
  assert(snapshot->shards[0].sum[0] == 0.0);
  assert(snapshot->shards[0].centroid[0] == 0.0);
}

void test_move_is_one_atomic_publication() {
  Router router(2, 2);
  const vec<f32> moving = point({7, -3});
  assert(router.insert(0, moving));
  assert(router.publish());
  const auto before = router.snapshot();
  assert(before->version == 1);
  assert(before->shards[0].count == 1);
  assert(before->shards[1].count == 0);

  assert(router.move(0, 1, moving));
  // The authoritative move is complete, but readers still see the old
  // internally consistent publication until maintenance publishes it.
  assert(router.snapshot() == before);
  assert(router.publish());
  const auto after = router.snapshot();
  assert(after->version == 2);
  assert(after->shards[0].version == 2);
  assert(after->shards[1].version == 1);
  assert(after->shards[0].count == 0);
  assert(after->shards[1].count == 1);
  assert(near(after->shards[1].sum[0], 7));
  assert(near(after->shards[1].sum[1], -3));

  // Previously acquired snapshots remain immutable after publication.
  assert(before->version == 1);
  assert(before->shards[0].count == 1);
  assert(before->shards[1].count == 0);
  assert(!router.move(0, 0, moving));
  assert(!router.move(0, 1, moving));
}

void test_concurrent_readers_never_observe_a_torn_move() {
  Router router(2, 2);
  const vec<f32> moving = point({9, -4});
  assert(router.insert(0, moving));
  assert(router.publish());

  std::atomic<bool> done{false};
  std::thread writer([&] {
    for (u32 iteration = 0; iteration < 5000; ++iteration) {
      const bool left_to_right = (iteration % 2) == 0;
      assert(router.move(left_to_right ? 0 : 1,
                         left_to_right ? 1 : 0,
                         moving));
      assert(router.publish());
    }
    done.store(true, std::memory_order_release);
  });

  while (!done.load(std::memory_order_acquire)) {
    const auto snapshot = router.snapshot();
    const u64 total_count =
      snapshot->shards[0].count + snapshot->shards[1].count;
    const f64 sum_x =
      sum_component(snapshot->shards[0], 0) +
      sum_component(snapshot->shards[1], 0);
    const f64 sum_y =
      sum_component(snapshot->shards[0], 1) +
      sum_component(snapshot->shards[1], 1);
    assert(total_count == 1);
    assert(near(sum_x, 9));
    assert(near(sum_y, -4));
  }
  writer.join();
}

void test_mutation_batch_is_invisible_until_one_publish() {
  Router router(2, 2);
  const auto initial = router.snapshot();
  assert(initial->version == 0);
  assert(initial->shards[0].count == 0);
  assert(initial->shards[1].count == 0);

  assert(router.insert(0, point({1, 2})));
  assert(router.insert(0, point({3, 4})));
  assert(router.upsert(0, point({1, 2}), point({5, 6})));
  vec<Router::LiveEntry> entries{entry(0, 55, 9)};
  assert(router.replace_live_entries(0, entries));

  // All four mutations changed authoritative state, but readers retain the
  // exact same immutable publication and cannot observe a partial batch.
  assert(router.snapshot() == initial);

  assert(router.publish());
  const auto published = router.snapshot();
  assert(published != initial);
  assert(published->version == 4);
  assert(published->shards[0].version == 4);
  assert(published->shards[0].count == 2);
  assert(near(published->shards[0].sum[0], 8));
  assert(near(published->shards[0].sum[1], 10));
  assert(near(published->shards[0].centroid[0], 4));
  assert(near(published->shards[0].centroid[1], 5));
  assert(published->shards[0].entries().size() == 1);
  assert(published->shards[0].entries()[0].generation == 9);
  assert(!router.publish());
  assert(router.snapshot() == published);
}

void test_bound_centroid_sidecar_header_checksum() {
  vamana::centroid_state::Header header;
  header.build_fingerprint = 0x123456789abcdef0ULL;
  header.shard_fingerprint = 0xfedcba9876543210ULL;
  header.vector_count = 99;
  header.node_base_offset = 16;
  header.payload_bytes = vamana::centroid_state::payload_bytes(3, 2);
  header.payload_checksum = 77;
  header.shard = 1;
  header.shard_count = 4;
  header.dim = 3;
  header.max_degree = 64;
  header.entry_count = 2;
  header.vector_dtype = static_cast<u32>(VectorDType::uint8);
  header.vector_component_size = sizeof(u8);
  header.node_size = 128;
  header.vector_offset = 24;
  header.vector_bytes = 3;
  header.slot_incarnation_offset = 16;
  header.hot_graph_version = vamana::hot_graph::kVersion3;
  header.hot_graph_entry_size = 560;
  header.hot_graph_pointer_bytes = sizeof(u64);
  header.hot_graph_shard_bits = 2;
  header.header_checksum =
    vamana::centroid_state::compute_header_checksum(header);
  assert(vamana::centroid_state::valid_header_checksum(header));

  auto wrong_build = header;
  ++wrong_build.build_fingerprint;
  assert(!vamana::centroid_state::valid_header_checksum(wrong_build));
  auto wrong_dtype = header;
  wrong_dtype.vector_dtype = static_cast<u32>(VectorDType::int8);
  assert(!vamana::centroid_state::valid_header_checksum(wrong_dtype));
  auto missing_checksum = header;
  missing_checksum.header_checksum = 0;
  assert(!vamana::centroid_state::valid_header_checksum(missing_checksum));
}

}  // namespace

int main() {
  test_restore_success_and_publication();
  test_restore_rejects_invalid_state();
  test_restore_preserves_fp64_precision();
  test_restore_versions_and_startup_window();
  test_validation_and_explicit_live_entries();
  test_exact_insert_erase_and_upsert();
  test_compensated_high_dynamic_range_updates();
  test_move_is_one_atomic_publication();
  test_concurrent_readers_never_observe_a_torn_move();
  test_mutation_batch_is_invisible_until_one_publish();
  test_bound_centroid_sidecar_header_checksum();
  return 0;
}
