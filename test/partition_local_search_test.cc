#include <cassert>
#include <optional>
#include <stdexcept>

#include "memory_node/storage_owner_index/partition_local_search.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

namespace detail = memory_node_storage_owner_index_detail;

namespace {

constexpr u32 kPartition = 2;

RemotePtr local(u32 index) {
  return RemotePtr{kPartition, (static_cast<u64>(index) + 1) * 16};
}

u32 index_of(RemotePtr pointer) {
  assert(pointer.memory_node() == kPartition);
  assert(pointer.byte_offset() > 0);
  return static_cast<u32>(pointer.byte_offset() / 16 - 1);
}

void test_multi_entry_fixed_beam_and_partition_boundary() {
  vec<vec<RemotePtr>> graph(6);
  graph[0] = {local(5)};
  graph[1] = {local(2), local(2), RemotePtr{3, 160}, local(3)};
  graph[2] = {local(4)};
  const vec<distance_t> distances{10.0F, 4.0F, 3.0F, 8.0F, 1.0F, 0.5F};
  const vec<RemotePtr> entries{local(0), RemotePtr{3, 80}, local(1)};

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

void test_stage1_budget_bounds_expansion_and_reports_truncation() {
  constexpr u32 kNodeCount = 40;
  vec<vec<RemotePtr>> graph(kNodeCount);
  for (u32 index = 0; index + 1 < kNodeCount; ++index) {
    graph[index].push_back(local(index + 1));
  }
  detail::PartitionLocalSearchBeam search(kPartition, 2);
  u32 expansions = 0;
  const detail::PartitionSearchBudget budget =
    detail::stage1_partition_search_budget(2, 1, 4);
  const vec<RemotePtr> entries{local(0)};
  const vec<detail::PartitionLocalSearchEntry> beam =
    detail::partition_local_construction_search_into(
    search, span<const RemotePtr>{entries}, kPartition, 2, budget,
    [](RemotePtr pointer) -> std::optional<distance_t> {
      return 100.0F - static_cast<distance_t>(index_of(pointer));
    },
    [&](RemotePtr pointer, auto&& visit) {
      ++expansions;
      for (RemotePtr neighbor : graph[index_of(pointer)]) visit(neighbor);
    });

  assert(expansions == 4);  // Stage1 E1 = 2L.
  assert(search.expansion_count() == 4);
  assert(search.budget_exhausted());
  assert(beam.size() == 2);
  assert(beam[0].rptr == local(4));
  assert(!beam[0].expanded);
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

void test_stage2_vector_snapshot_validation_matches_stable_search() {
  const RemotePtr pointer{kPartition, 4096, 7};
  const u64 stable = VamanaNode::make_header(pointer.incarnation());
  assert(detail::stable_vector_snapshot_valid(
    pointer, stable, stable, pointer.incarnation()));

  // Query/search visibility intentionally survives maintenance-only
  // quiescence flags when the record is otherwise one stable incarnation.
  const u64 frozen = stable | VamanaNode::HEADER_STAGE2_FROZEN;
  const u64 retiring = stable | VamanaNode::HEADER_RETIRING;
  assert(detail::stable_vector_snapshot_valid(
    pointer, frozen, frozen, pointer.incarnation()));
  assert(detail::stable_vector_snapshot_valid(
    pointer, retiring, retiring, pointer.incarnation()));

  assert(!detail::stable_vector_snapshot_valid(
    pointer, stable, stable | VamanaNode::HEADER_NODE_LOCK,
    pointer.incarnation()));
  assert(!detail::stable_vector_snapshot_valid(
    pointer, stable, stable | VamanaNode::HEADER_DELETED,
    pointer.incarnation()));
  assert(!detail::stable_vector_snapshot_valid(
    pointer, stable, stable | VamanaNode::HEADER_PROVISIONAL,
    pointer.incarnation()));
  assert(!detail::stable_vector_snapshot_valid(
    pointer, stable, stable, pointer.incarnation() + 1));

  const u64 replacement = VamanaNode::make_header(
    pointer.incarnation() + 1);
  assert(!detail::stable_vector_snapshot_valid(
    pointer, stable, replacement, pointer.incarnation() + 1));
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

void test_stage1_exports_unique_remote_frontier() {
  detail::PartitionLocalSearchBeam search(kPartition, 4);
  const RemotePtr remote_a{1, 128};
  const RemotePtr remote_b{3, 256};
  const vec<RemotePtr> entries{local(0)};
  detail::partition_local_construction_search_into(
    search, span<const RemotePtr>{entries}, kPartition, 4,
    [](RemotePtr) -> std::optional<distance_t> { return 1.0F; },
    [&](RemotePtr, auto&& visit) {
      visit(remote_a);
      visit(remote_a);
      visit(remote_b);
    });
  assert((search.remote_frontier() == vec<RemotePtr>{remote_a, remote_b}));
}

void test_stage1_frontier_is_bounded_and_deterministic() {
  detail::PartitionLocalSearchBeam search(kPartition, 2);
  const vec<RemotePtr> entries{local(0)};
  detail::partition_local_construction_search_into(
    search, span<const RemotePtr>{entries}, kPartition, 2,
    detail::stage1_partition_search_budget(2, entries.size(), 8),
    [](RemotePtr) -> std::optional<distance_t> { return 1.0F; },
    [&](RemotePtr, auto&& visit) {
      // Discovery order is deliberately reversed. Equal-priority frontier
      // candidates are retained by full handle order, not timing/order.
      for (u32 index = 0; index < 6; ++index) {
        visit(RemotePtr{3, static_cast<u64>(6 - index) * 16});
      }
    });
  assert(search.remote_frontier().size() == 4);  // F = 2L.
  assert(search.budget_exhausted());
  for (size_t index = 1; index < search.remote_frontier().size(); ++index) {
    assert(search.remote_frontier()[index - 1].raw_address <
           search.remote_frontier()[index].raw_address);
  }
}

void test_stage2_continues_stage1_without_restarting_local_search() {
  const RemotePtr remote_a{1, 128};
  const RemotePtr remote_b{1, 256};
  const RemotePtr remote_c{3, 384};
  const vec<detail::PartitionLocalSearchEntry> local_beam{
    {local(0), 2.0F, true},
    {local(1), 5.0F, true},
  };
  const vec<RemotePtr> frontier{remote_a, remote_b, remote_a};
  dense_hashmap_t<u64, distance_t> distances{
    {remote_a.raw_address, 4.0F},
    {remote_b.raw_address, 8.0F},
    {remote_c.raw_address, 1.0F},
  };
  vec<RemotePtr> expanded;
  u32 scored = 0;
  const vec<detail::PartitionLocalSearchEntry> final_beam =
    detail::continue_partition_construction_search(
      span<const detail::PartitionLocalSearchEntry>{local_beam},
      span<const RemotePtr>{frontier}, kPartition, 3,
      [&](span<const RemotePtr> batch, auto&& emit) {
        for (const RemotePtr pointer : batch) {
          ++scored;
          emit(pointer, distances.at(pointer.raw_address));
        }
      },
      [&](RemotePtr pointer, auto&& visit) {
        expanded.push_back(pointer);
        if (pointer == remote_a) {
          visit(remote_c);
          visit(local(1));  // Stage1 partition is never revisited.
        }
      });

  assert(scored == 3);  // duplicate frontier and local return are skipped
  assert((expanded == vec<RemotePtr>{remote_a, remote_c}));
  assert(final_beam.size() == 3);
  assert(final_beam[0].rptr == remote_c);
  assert(final_beam[1].rptr == local(0));
  assert(final_beam[2].rptr == remote_a);
  for (const auto& entry : final_beam) assert(entry.expanded);
}

void test_stage2_budget_reserves_exactly_l_remote_expansions() {
  const RemotePtr remote_a{1, 128};
  const RemotePtr remote_b{1, 256};
  const RemotePtr remote_c{1, 384};
  const vec<detail::PartitionLocalSearchEntry> local_beam{
    {local(0), 10.0F, true},
  };
  const vec<RemotePtr> frontier{remote_a};
  dense_hashmap_t<u64, distance_t> distances{
    {remote_a.raw_address, 3.0F},
    {remote_b.raw_address, 2.0F},
    {remote_c.raw_address, 1.0F},
  };
  u32 expansions = 0;
  bool exhausted = false;
  const auto result = detail::continue_partition_construction_search(
    span<const detail::PartitionLocalSearchEntry>{local_beam},
    span<const RemotePtr>{frontier}, kPartition, 2,
    detail::stage2_partition_search_budget(2, 4),
    [&](span<const RemotePtr> batch, auto&& emit) {
      for (RemotePtr pointer : batch) {
        emit(pointer, distances.at(pointer.raw_address));
      }
    },
    [&](RemotePtr pointer, auto&& visit) {
      ++expansions;
      if (pointer == remote_a) visit(remote_b);
      if (pointer == remote_b) visit(remote_c);
    },
    &exhausted);
  assert(expansions == 2);  // Stage2 E2 = L; total is at most 3L.
  assert(exhausted);
  assert(result.size() == 2);
  assert(result[0].rptr == remote_c);
  assert(!result[0].expanded);
}

void test_stage2_batch_wavefront_preserves_independent_searches() {
  const RemotePtr a0{1, 128};
  const RemotePtr a1{1, 256};
  const RemotePtr a2{1, 384};
  const RemotePtr b0{3, 128};
  const RemotePtr b1{3, 256};
  const RemotePtr b2{3, 384};
  const vec<detail::PartitionLocalSearchEntry> local_a{
    {local(0), 20.0F, true}};
  const vec<detail::PartitionLocalSearchEntry> local_b{
    {local(1), 20.0F, true}};
  const vec<RemotePtr> frontier_a{a0};
  const vec<RemotePtr> frontier_b{b0};
  const std::array<detail::PartitionContinuationSeed, 2> seeds{{
    {span<const detail::PartitionLocalSearchEntry>{local_a},
     span<const RemotePtr>{frontier_a}},
    {span<const detail::PartitionLocalSearchEntry>{local_b},
     span<const RemotePtr>{frontier_b}},
  }};
  dense_hashmap_t<u64, distance_t> distances{
    {a0.raw_address, 6.0F}, {a1.raw_address, 4.0F},
    {a2.raw_address, 2.0F}, {b0.raw_address, 7.0F},
    {b1.raw_address, 5.0F}, {b2.raw_address, 3.0F},
  };
  dense_hashmap_t<u64, RemotePtr> next{
    {a0.raw_address, a1}, {a1.raw_address, a2},
    {b0.raw_address, b1}, {b1.raw_address, b2},
  };

  size_t score_waves = 0;
  size_t expansion_waves = 0;
  size_t max_score_wave = 0;
  size_t max_expansion_wave = 0;
  detail::PartitionContinuationBatch batch;
  vec<bool> exhausted;
  vec<u64> expansions;
  const auto& results = batch.run(
    span<const detail::PartitionContinuationSeed>{seeds.data(), seeds.size()},
    kPartition, 2,
    detail::stage2_partition_search_budget(2, 4),
    [&](span<const detail::PartitionContinuationScoreRequest> requests,
        auto&& emit) {
      ++score_waves;
      max_score_wave = std::max(max_score_wave, requests.size());
      for (const auto& request : requests) {
        emit(request.search_index, request.pointer,
             distances.at(request.pointer.raw_address));
      }
    },
    [&](span<const detail::PartitionContinuationExpandRequest> requests,
        auto&& emit) {
      ++expansion_waves;
      max_expansion_wave = std::max(max_expansion_wave, requests.size());
      for (const auto& request : requests) {
        const auto found = next.find(request.pointer.raw_address);
        if (found != next.end()) {
          emit(request.search_index, found->second);
        }
      }
    },
    &exhausted, &expansions);

  assert(results.size() == 2);
  assert(results[0][0].rptr == a2);
  assert(results[1][0].rptr == b2);
  assert(expansions[0] == 2 && expansions[1] == 2);
  assert(exhausted[0] && exhausted[1]);
  // Two independent searches share each physical I/O wave. Their expansion
  // budgets remain per-search rather than becoming one shared batch budget.
  assert(score_waves == 3);
  assert(expansion_waves == 2);
  assert(max_score_wave == 2);
  assert(max_expansion_wave == 2);
}

void test_stage2_batch_missing_snapshot_is_isolated() {
  const RemotePtr unreadable{1, 128};
  const RemotePtr healthy0{3, 128};
  const RemotePtr healthy1{3, 256};
  const vec<detail::PartitionLocalSearchEntry> local_a{
    {local(0), 20.0F, true}};
  const vec<detail::PartitionLocalSearchEntry> local_b{
    {local(1), 20.0F, true}};
  const vec<RemotePtr> frontier_a{unreadable};
  const vec<RemotePtr> frontier_b{healthy0};
  const std::array<detail::PartitionContinuationSeed, 2> seeds{{
    {span<const detail::PartitionLocalSearchEntry>{local_a},
     span<const RemotePtr>{frontier_a}},
    {span<const detail::PartitionLocalSearchEntry>{local_b},
     span<const RemotePtr>{frontier_b}},
  }};

  detail::PartitionContinuationBatch batch;
  vec<u64> expansions;
  const auto& results = batch.run(
    span<const detail::PartitionContinuationSeed>{seeds.data(), seeds.size()},
    kPartition, 2,
    detail::stage2_partition_search_budget(2, 4),
    [&](span<const detail::PartitionContinuationScoreRequest> requests,
        auto&& emit) {
      for (const auto& request : requests) {
        // Model one failed/stale vector snapshot by omitting its callback.
        if (request.pointer == unreadable) continue;
        emit(request.search_index, request.pointer,
             request.pointer == healthy0 ? 4.0F : 1.0F);
      }
    },
    [&](span<const detail::PartitionContinuationExpandRequest> requests,
        auto&& emit) {
      for (const auto& request : requests) {
        if (request.pointer == healthy0) {
          emit(request.search_index, healthy1);
        }
      }
    },
    nullptr, &expansions);

  // A partial read failure rejects only that physical candidate. It neither
  // retries/discards the healthy task nor consumes the healthy task's budget.
  assert(results[0].size() == 1);
  assert(results[0][0].rptr == local(0));
  assert(expansions[0] == 0);
  assert(results[1][0].rptr == healthy1);
  assert(expansions[1] == 2);
}

void test_final_home_strictly_reduces_cross_shard_edges() {
  const vec<RemotePtr> neighbors{
    RemotePtr{0, 64}, RemotePtr{2, 64}, RemotePtr{2, 128},
    RemotePtr{2, 192}, RemotePtr{1, 64}};
  assert(detail::choose_min_cross_shard_home(neighbors, 4, 0) == 2);

  // A tie stays at Stage1 home: migration is justified only by a strict
  // locality gain, avoiding churn that cannot reduce remote accesses.
  const vec<RemotePtr> tied{
    RemotePtr{0, 64}, RemotePtr{0, 128},
    RemotePtr{3, 64}, RemotePtr{3, 128}};
  assert(detail::choose_min_cross_shard_home(tied, 4, 3) == 3);
}

}  // namespace

int main() {
  test_multi_entry_fixed_beam_and_partition_boundary();
  test_search_converges_beyond_old_fixed_expansion_limit();
  test_stage1_budget_bounds_expansion_and_reports_truncation();
  test_stale_entries_do_not_block_other_roots();
  test_final_validation_removes_concurrently_stale_entries();
  test_stage2_vector_snapshot_validation_matches_stable_search();
  test_reusable_state_is_cleared_between_searches();
  test_zero_width_is_rejected();
  test_stage1_exports_unique_remote_frontier();
  test_stage1_frontier_is_bounded_and_deterministic();
  test_stage2_continues_stage1_without_restarting_local_search();
  test_stage2_budget_reserves_exactly_l_remote_expansions();
  test_stage2_batch_wavefront_preserves_independent_searches();
  test_stage2_batch_missing_snapshot_is_isolated();
  test_final_home_strictly_reduces_cross_shard_edges();
  return 0;
}
