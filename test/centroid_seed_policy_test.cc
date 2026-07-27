#include <algorithm>
#include <array>
#include <cassert>
#include <limits>

#include "vamana/centroid_seed_policy.hh"

namespace routing = vamana::routing;

int main() {
  const std::array<f64, 2> centroid{10.0, -2.0};
  const std::array<f32, 2> central{10.0f, -2.0f};
  const std::array<f32, 2> farther{13.0f, 2.0f};
  assert(routing::centroid_seed_squared_l2(
           {central.data(), central.size()},
           {centroid.data(), centroid.size()}) == 0);
  assert(routing::centroid_seed_squared_l2(
           {farther.data(), farther.size()},
           {centroid.data(), centroid.size()}) == 25);

  std::array<routing::CentroidSeedRank, 5> candidates{{
    {9, 40}, {1, 50}, {4, 30}, {1, 20}, {16, 10},
  }};
  std::sort(candidates.begin(), candidates.end(),
            routing::centroid_seed_rank_less);
  // A newly added node competes by centroid distance; it never wins merely
  // because it was the most recent insertion. Equal distances use pointer
  // identity for deterministic CPU/offline ordering.
  assert(candidates[0].pointer_raw == 20);
  assert(candidates[1].pointer_raw == 50);
  assert(candidates[4].pointer_raw == 10);

  const f32 maximum = std::numeric_limits<f32>::max();
  const std::array<f32, 2> extreme{maximum, -maximum};
  const std::array<f64, 2> origin{};
  const long double extreme_distance =
    routing::centroid_seed_squared_l2(
      {extreme.data(), extreme.size()}, {origin.data(), origin.size()});
  assert(extreme_distance > 0 &&
         extreme_distance < std::numeric_limits<long double>::infinity());

  // Four valid roots do not stop exploration. The bounded probe advances
  // past those duplicate ordinals and exposes later live nodes, one of which
  // can replace an old root after a non-root deletion shifts the centroid.
  const std::array<u64, 1> dense_bitmap{0xffff};
  u64 dense_cursor = 0;
  const vec<u64> first_probe = routing::bounded_rotating_live_samples(
    {dense_bitmap.data(), dense_bitmap.size()}, 16, dense_cursor, 8);
  assert(first_probe.size() == 8);
  assert(first_probe.front() == 0 && first_probe.back() == 7);
  assert(dense_cursor == 8);
  std::array<routing::CentroidSeedRank, 5> full_roots_plus_sample{{
    {100, 0}, {81, 1}, {64, 2}, {49, 3}, {1, 4},
  }};
  std::sort(full_roots_plus_sample.begin(),
            full_roots_plus_sample.end(),
            routing::centroid_seed_rank_less);
  assert(full_roots_plus_sample.front().pointer_raw == 4);
  assert(full_roots_plus_sample[4].pointer_raw == 0);

  // Sparse populations also have dataset-independent per-batch work. Empty
  // probes move the cursor by exactly eight words; repeated batches
  // eventually reach the distant live word without one full-table scan.
  std::array<u64, 32> sparse_bitmap{};
  sparse_bitmap[20] = 1;
  u64 sparse_cursor = 0;
  assert(routing::bounded_rotating_live_samples(
           {sparse_bitmap.data(), sparse_bitmap.size()},
           sparse_bitmap.size() * 64, sparse_cursor, 4).empty());
  assert(sparse_cursor == 8 * 64);
  assert(routing::bounded_rotating_live_samples(
           {sparse_bitmap.data(), sparse_bitmap.size()},
           sparse_bitmap.size() * 64, sparse_cursor, 4).empty());
  assert(sparse_cursor == 16 * 64);
  const vec<u64> sparse_probe = routing::bounded_rotating_live_samples(
    {sparse_bitmap.data(), sparse_bitmap.size()},
    sparse_bitmap.size() * 64, sparse_cursor, 4);
  assert(sparse_probe.size() == 1 && sparse_probe.front() == 20 * 64);
  return 0;
}
