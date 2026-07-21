#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include "gpu_search/centroid_home_selector.hh"
#include "gpu_search/centroid_route_ranking.hh"
#include "gpu_search/persistent_kernel.hh"

namespace {

using gpu_search::f32;
using gpu_search::u32;
using gpu_search::u64;
using gpu_search::centroid_route_ranking::RankedShard;

constexpr std::size_t kCapacity = gpu_search::kPersistentMaxShards;

std::array<RankedShard, kCapacity> scalar_rank(
    std::array<RankedShard, kCapacity> values) {
  std::sort(values.begin(), values.end(),
            gpu_search::centroid_route_ranking::less);
  return values;
}

void assert_same_rank(const std::array<RankedShard, kCapacity>& actual,
                      const std::array<RankedShard, kCapacity>& expected) {
  for (std::size_t index = 0; index < kCapacity; ++index) {
    assert(actual[index].valid == expected[index].valid);
    assert(actual[index].shard == expected[index].shard);
    assert(actual[index].distance == expected[index].distance);
  }
}

std::vector<u64> select_seed_handles(
    const std::array<RankedShard, kCapacity>& rank,
    const std::array<std::array<u64, gpu_search::kCentroidRouteMaxLiveEntries>,
                     kCapacity>& entries,
    const std::array<u32, kCapacity>& entry_counts) {
  std::vector<u64> result;
  if (rank.front().valid == 0) return result;
  const u32 shard = rank.front().shard;
  for (u32 local = 0; local < entry_counts[shard]; ++local) {
    result.push_back(entries[shard][local]);
  }
  return result;
}

void test_network_matches_scalar_reference() {
  std::mt19937 generator(0x5eed1234u);
  std::uniform_real_distribution<f32> distance(-10000.0f, 10000.0f);
  for (u32 trial = 0; trial < 2000; ++trial) {
    std::array<RankedShard, kCapacity> routes{};
    for (u32 shard = 0; shard < routes.size(); ++shard) {
      // Include many exact distance ties and sparse publication snapshots.
      const f32 value = (trial + shard) % 5 == 0
        ? static_cast<f32>((trial + shard) % 7) : distance(generator);
      routes[shard] = RankedShard{
        .distance = value,
        .shard = shard,
        .valid = static_cast<u32>((generator() & 3u) != 0),
      };
    }
    // Valid saturation must rank before an invalid slot with the same value.
    routes[3] = RankedShard{
      .distance = std::numeric_limits<f32>::max(), .shard = 3, .valid = 1};
    routes[61] = RankedShard{
      .distance = std::numeric_limits<f32>::max(), .shard = 61, .valid = 0};
    const auto expected = scalar_rank(routes);
    gpu_search::centroid_route_ranking::bitonic_sort_for_test(routes);
    assert_same_rank(routes, expected);
  }
}

void test_fma_home_and_seed_parity() {
  constexpr u32 kShards = 64;
  constexpr u32 kDim = 257;
  std::mt19937 generator(0xc001d00du);
  std::uniform_real_distribution<f32> component(-16.0f, 16.0f);
  gpu_search::centroid_home::Snapshot snapshot(kShards);
  std::vector<f32> query(kDim);
  for (f32& value : query) value = component(generator);

  std::array<RankedShard, kCapacity> routes{};
  std::array<u32, kCapacity> entry_counts{};
  std::array<std::array<u64, gpu_search::kCentroidRouteMaxLiveEntries>,
             kCapacity> entries{};
  for (u32 shard = 0; shard < kShards; ++shard) {
    snapshot[shard].vector_count = 1 + shard;
    snapshot[shard].live_entry_count = 1 + shard % 4;
    snapshot[shard].centroid.resize(kDim);
    for (f32& value : snapshot[shard].centroid) {
      value = component(generator);
    }
    f32 distance = 0.0f;
    for (u32 dimension = 0; dimension < kDim; ++dimension) {
      const f32 difference =
        query[dimension] - snapshot[shard].centroid[dimension];
      distance = std::fma(difference, difference, distance);
    }
    routes[shard] = RankedShard{
      .distance = distance, .shard = shard, .valid = 1};
    entry_counts[shard] = snapshot[shard].live_entry_count;
    for (u32 local = 0; local < entry_counts[shard]; ++local) {
      entries[shard][local] =
        (static_cast<u64>(shard) << 32) | local | 1u;
    }
  }

  const auto expected = scalar_rank(routes);
  gpu_search::centroid_route_ranking::bitonic_sort_for_test(routes);
  assert_same_rank(routes, expected);
  const auto home = gpu_search::centroid_home::select(query, snapshot);
  assert(home.has_value());
  assert(*home == routes.front().shard);

  const auto scalar_seeds = select_seed_handles(
    expected, entries, entry_counts);
  const auto network_seeds = select_seed_handles(
    routes, entries, entry_counts);
  assert(!network_seeds.empty());
  assert(network_seeds.size() <= gpu_search::kCentroidRouteMaxLiveEntries);
  assert(network_seeds == scalar_seeds);
}

void test_physical_shard_tie_break() {
  std::array<RankedShard, kCapacity> routes{};
  for (u32 shard = 0; shard < routes.size(); ++shard) {
    routes[shard] = RankedShard{
      .distance = 42.0f, .shard = shard, .valid = 1};
  }
  std::reverse(routes.begin(), routes.end());
  gpu_search::centroid_route_ranking::bitonic_sort_for_test(routes);
  for (u32 shard = 0; shard < routes.size(); ++shard) {
    assert(routes[shard].shard == shard);
  }
}

void test_whole_table_publication_epoch() {
  using gpu_search::centroid_route_ranking::stable_publication_epoch;
  assert(stable_publication_epoch(0, 0));
  assert(stable_publication_epoch(42, 42));
  assert(!stable_publication_epoch(1, 1));
  assert(!stable_publication_epoch(41, 42));
  assert(!stable_publication_epoch(42, 43));
  assert(!stable_publication_epoch(42, 44));
}

}  // namespace

int main() {
  test_network_matches_scalar_reference();
  test_fma_home_and_seed_parity();
  test_physical_shard_tie_break();
  test_whole_table_publication_epoch();
  return 0;
}
