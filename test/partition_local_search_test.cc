#include <cassert>
#include <optional>
#include <stdexcept>

#include "memory_node/storage_owner_index/partition_local_search.hh"

namespace detail = memory_node_storage_owner_index_detail;

namespace {

constexpr u32 kPartition = 2;

RemotePtr local(u32 index) {
  return RemotePtr{kPartition, static_cast<u64>(index) + 1};
}

u32 index_of(RemotePtr pointer) {
  assert(pointer.memory_node() == kPartition);
  assert(pointer.byte_offset() > 0);
  return static_cast<u32>(pointer.byte_offset() - 1);
}

void test_multi_entry_fixed_beam_and_partition_boundary() {
  vec<vec<RemotePtr>> graph(6);
  graph[0] = {local(5)};
  graph[1] = {local(2), local(2), RemotePtr{3, 99}, local(3)};
  graph[2] = {local(4)};
  const vec<distance_t> distances{10.0F, 4.0F, 3.0F, 8.0F, 1.0F, 0.5F};
  const vec<RemotePtr> entries{local(0), RemotePtr{3, 77}, local(1)};

  vec<u32> score_count(graph.size(), 0);
  vec<u32> expansion_order;
  const vec<detail::PartitionLocalSearchEntry> beam =
    detail::partition_local_construction_search(
      span<const RemotePtr>{entries}, kPartition, 2,
      [&](RemotePtr pointer) -> std::optional<distance_t> {
        const u32 index = index_of(pointer);
        ++score_count[index];
        return distances[index];
      },
      [&](RemotePtr pointer, auto&& visit) {
        const u32 index = index_of(pointer);
        expansion_order.push_back(index);
        for (const RemotePtr neighbor : graph[index]) {
          visit(neighbor);
        }
      });

  assert(beam.size() == 2);
  assert(beam[0].rptr == local(4));
  assert(beam[0].distance == 1.0F);
  assert(beam[1].rptr == local(2));
  assert(beam[1].distance == 3.0F);
  assert(beam[0].expanded && beam[1].expanded);

  // The farther entry is evicted by the fixed-L beam before expansion, and a
  // duplicate neighbor is scored only once. Remote pointers never reach the
  // score/expand callbacks.
  assert((expansion_order == vec<u32>{1, 2, 4}));
  assert(score_count[2] == 1);
  assert(score_count[0] == 1);
  assert(score_count[5] == 0);
}

void test_search_converges_beyond_old_fixed_expansion_limit() {
  constexpr u32 kNodeCount = 40;
  vec<vec<RemotePtr>> graph(kNodeCount);
  for (u32 index = 0; index + 1 < kNodeCount; ++index) {
    graph[index].push_back(local(index + 1));
  }
  const vec<RemotePtr> entries{local(0)};

  u32 expansions = 0;
  const vec<detail::PartitionLocalSearchEntry> beam =
    detail::partition_local_construction_search(
      span<const RemotePtr>{entries}, kPartition, 2,
      [](RemotePtr pointer) -> std::optional<distance_t> {
        return 100.0F - static_cast<distance_t>(index_of(pointer));
      },
      [&](RemotePtr pointer, auto&& visit) {
        ++expansions;
        for (const RemotePtr neighbor : graph[index_of(pointer)]) {
          visit(neighbor);
        }
      });

  assert(expansions == kNodeCount);
  assert(expansions > 16);
  assert(beam.size() == 2);
  assert(beam[0].rptr == local(39));
  assert(beam[1].rptr == local(38));
  assert(beam[0].expanded && beam[1].expanded);
}

void test_stale_entries_do_not_block_other_roots() {
  const vec<RemotePtr> entries{local(0), local(1)};
  u32 expansions = 0;
  const vec<detail::PartitionLocalSearchEntry> beam =
    detail::partition_local_construction_search(
      span<const RemotePtr>{entries}, kPartition, 4,
      [](RemotePtr pointer) -> std::optional<distance_t> {
        if (index_of(pointer) == 0) {
          return std::nullopt;
        }
        return 2.0F;
      },
      [&](RemotePtr, auto&&) { ++expansions; });

  assert(beam.size() == 1);
  assert(beam[0].rptr == local(1));
  assert(beam[0].expanded);
  assert(expansions == 1);
}

void test_final_validation_removes_concurrently_stale_entries() {
  const vec<RemotePtr> entries{local(0), local(1), local(2)};
  vec<detail::PartitionLocalSearchEntry> beam =
    detail::partition_local_construction_search(
      span<const RemotePtr>{entries}, kPartition, 3,
      [](RemotePtr pointer) -> std::optional<distance_t> {
        return static_cast<distance_t>(index_of(pointer));
      },
      [](RemotePtr, auto&&) {});
  assert(beam.size() == 3);

  // Model an entry that was live while scored but was tombstoned before the
  // final candidate set crossed the search boundary.
  detail::filter_final_partition_local_beam(
    beam, [](RemotePtr pointer) { return index_of(pointer) != 1; });
  assert(beam.size() == 2);
  assert(beam[0].rptr == local(0));
  assert(beam[1].rptr == local(2));
}

void test_reusable_state_is_cleared_between_searches() {
  detail::PartitionLocalSearchBeam reusable(kPartition, 3);
  const vec<RemotePtr> first_entries{local(0), local(1), local(2)};
  auto no_expand = [](RemotePtr, auto&&) {};
  u32 scores = 0;
  vec<detail::PartitionLocalSearchEntry>& first =
    detail::partition_local_construction_search_into(
      reusable, span<const RemotePtr>{first_entries}, kPartition, 3,
      [&](RemotePtr pointer) -> std::optional<distance_t> {
        ++scores;
        return static_cast<distance_t>(index_of(pointer));
      },
      no_expand);
  assert(first.size() == 3);
  assert(scores == 3);

  const vec<RemotePtr> second_entries{local(0)};
  vec<detail::PartitionLocalSearchEntry>& second =
    detail::partition_local_construction_search_into(
      reusable, span<const RemotePtr>{second_entries}, kPartition, 1,
      [&](RemotePtr) -> std::optional<distance_t> {
        ++scores;
        return 7.0F;
      },
      no_expand);
  assert(second.size() == 1);
  assert(second[0].rptr == local(0));
  assert(second[0].distance == 7.0F);
  assert(scores == 4);
  assert(reusable.beam_width() == 1);
}

void test_zero_width_is_rejected() {
  bool rejected = false;
  try {
    detail::PartitionLocalSearchBeam search(kPartition, 0);
    (void)search;
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

}  // namespace

int main() {
  test_multi_entry_fixed_beam_and_partition_boundary();
  test_search_converges_beyond_old_fixed_expansion_limit();
  test_stale_entries_do_not_block_other_roots();
  test_final_validation_removes_concurrently_stale_entries();
  test_reusable_state_is_cleared_between_searches();
  test_zero_width_is_rejected();
  return 0;
}
