#include <cassert>
#include <cmath>
#include <limits>
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

void test_stage1_policy_converges_beyond_old_2l_expansion_limit() {
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

  assert(expansions == kNodeCount);
  assert(expansions > 4);  // The old Stage1 E1 bound was 2L.
  assert(search.expansion_count() == kNodeCount);
  assert(!search.budget_exhausted());
  assert(beam.size() == 2);
  assert(beam[0].rptr == local(kNodeCount - 1));
  assert(beam[1].rptr == local(kNodeCount - 2));
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

  // A single header observation is deliberately insufficient for a dynamic
  // slot: its body could have started with the old identity and ended with a
  // replacement vector after reuse. The ordered after-header changes the
  // classification from an apparent stable old identity to retryable.
  assert(detail::classify_stable_node_snapshot(
           pointer, stable, stable, pointer.incarnation()) ==
         detail::StableNodeSnapshotState::stable);
  assert(detail::classify_stable_node_snapshot(
           pointer, stable, replacement, pointer.incarnation()) ==
         detail::StableNodeSnapshotState::retryable);

  // For an immutable incarnation-zero base payload, one unlocked live header
  // is a valid linearization point. Lifecycle flags retain the same policy as
  // the paired path; NODE_LOCK is contention and deletion is terminal.
  const RemotePtr base{kPartition, 8192, 0};
  const u64 base_live = VamanaNode::make_header(0);
  assert(detail::classify_stable_node_snapshot(
           base, base_live, base_live, 0) ==
         detail::StableNodeSnapshotState::stable);
  assert(detail::classify_stable_node_snapshot(
           base, base_live | VamanaNode::HEADER_NODE_LOCK,
           base_live | VamanaNode::HEADER_NODE_LOCK, 0) ==
         detail::StableNodeSnapshotState::retryable);
  assert(detail::classify_stable_node_snapshot(
           base, base_live | VamanaNode::HEADER_DELETED,
           base_live | VamanaNode::HEADER_DELETED, 0) ==
         detail::StableNodeSnapshotState::terminal);
}

void test_transient_node_snapshot_is_retryable_not_terminal() {
  using detail::StableNodeSnapshotState;
  const RemotePtr pointer{kPartition, 4096, 7};
  const u64 stable = VamanaNode::make_header(pointer.incarnation());
  const u64 locked = stable | VamanaNode::HEADER_NODE_LOCK;

  assert(detail::classify_stable_node_snapshot(
           pointer, locked, locked, pointer.incarnation()) ==
         StableNodeSnapshotState::retryable);
  assert(detail::classify_stable_node_snapshot(
           pointer, stable, locked, pointer.incarnation()) ==
         StableNodeSnapshotState::retryable);

  // Even a lifecycle bit cannot be accepted as a stable terminal observation
  // while another writer still owns NODE_LOCK.
  const u64 locked_deleted = locked | VamanaNode::HEADER_DELETED;
  assert(detail::classify_stable_node_snapshot(
           pointer, locked_deleted, locked_deleted,
           pointer.incarnation()) == StableNodeSnapshotState::retryable);

  assert(detail::classify_stable_node_snapshot(
           pointer, stable, stable, pointer.incarnation()) ==
         StableNodeSnapshotState::stable);
  const u64 provisional = stable | VamanaNode::HEADER_PROVISIONAL;
  // Stage1 traversal uses the coherent physical identity and may therefore
  // walk through a provisional node, while its final-beam validation uses the
  // stable lifecycle policy and excludes the same node from the handoff.
  assert(detail::classify_physical_node_snapshot(
           pointer, provisional, provisional, pointer.incarnation()) ==
         StableNodeSnapshotState::stable);
  assert(detail::classify_stable_node_snapshot(
           pointer, provisional, provisional, pointer.incarnation()) ==
         StableNodeSnapshotState::terminal);
  const u64 deleted = stable | VamanaNode::HEADER_DELETED;
  assert(detail::classify_stable_node_snapshot(
           pointer, deleted, deleted, pointer.incarnation()) ==
         StableNodeSnapshotState::terminal);
  assert(detail::classify_stable_node_snapshot(
           pointer, stable, stable, pointer.incarnation() + 1) ==
         StableNodeSnapshotState::terminal);
}

void test_stage2_target_snapshot_classification() {
  using detail::StableNodeSnapshotState;
  const RemotePtr pointer{kPartition, 4096, 7};
  const node_t expected_id = 41;
  const u32 expected_generation = 9;
  const u64 stable = VamanaNode::make_header(pointer.incarnation());
  const auto classify = [&](u64 before,
                            u64 after,
                            u32 slot_incarnation,
                            node_t id,
                            u32 generation) {
    return detail::classify_stage2_target_snapshot(
      pointer, before, after, slot_incarnation, id, generation,
      expected_id, expected_generation);
  };

  assert(classify(stable, stable, pointer.incarnation(), expected_id,
                  expected_generation) == StableNodeSnapshotState::stable);
  assert(classify(stable | VamanaNode::HEADER_PROVISIONAL,
                  stable | VamanaNode::HEADER_PROVISIONAL,
                  pointer.incarnation(), expected_id,
                  expected_generation) == StableNodeSnapshotState::stable);
  assert(classify(stable | VamanaNode::HEADER_RETIRING |
                            VamanaNode::HEADER_STAGE2_FROZEN,
                  stable | VamanaNode::HEADER_RETIRING |
                            VamanaNode::HEADER_STAGE2_FROZEN,
                  pointer.incarnation(), expected_id,
                  expected_generation) == StableNodeSnapshotState::stable);

  const u64 locked = stable | VamanaNode::HEADER_NODE_LOCK;
  assert(classify(locked, locked, pointer.incarnation(), expected_id,
                  expected_generation) ==
         StableNodeSnapshotState::retryable);
  // Lock/torn observations take precedence over fields that only appear
  // terminal while another writer owns the record.
  assert(classify(locked | VamanaNode::HEADER_DELETED,
                  locked | VamanaNode::HEADER_DELETED,
                  pointer.incarnation() + 1, expected_id + 1,
                  expected_generation + 1) ==
         StableNodeSnapshotState::retryable);
  assert(classify(stable, locked, pointer.incarnation(), expected_id,
                  expected_generation) ==
         StableNodeSnapshotState::retryable);
  const u64 replacement = VamanaNode::make_header(
    pointer.incarnation() + 1);
  assert(classify(stable, replacement, pointer.incarnation() + 1,
                  expected_id, expected_generation) ==
         StableNodeSnapshotState::retryable);

  assert(classify(stable | VamanaNode::HEADER_DELETED,
                  stable | VamanaNode::HEADER_DELETED,
                  pointer.incarnation(), expected_id,
                  expected_generation) == StableNodeSnapshotState::terminal);
  assert(classify(replacement, replacement, pointer.incarnation() + 1,
                  expected_id, expected_generation) ==
         StableNodeSnapshotState::terminal);
  assert(classify(stable, stable, pointer.incarnation() + 1, expected_id,
                  expected_generation) == StableNodeSnapshotState::terminal);
  assert(classify(stable, stable, pointer.incarnation(), expected_id + 1,
                  expected_generation) == StableNodeSnapshotState::terminal);
  assert(classify(stable, stable, pointer.incarnation(), expected_id,
                  expected_generation + 1) ==
         StableNodeSnapshotState::terminal);
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

void test_nan_distances_are_canonicalized_to_positive_infinity() {
  const distance_t nan = std::numeric_limits<distance_t>::quiet_NaN();
  detail::PartitionLocalSearchBeam local_search(kPartition, 3);
  assert(local_search.try_visit(local(2)));
  local_search.add_visited(local(2), nan);
  assert(local_search.try_visit(local(0)));
  local_search.add_visited(local(0), nan);
  assert(local_search.try_visit(local(1)));
  local_search.add_visited(local(1), 1.0F);

  const auto& local_beam = local_search.final_beam();
  assert(local_beam.size() == 3);
  assert(local_beam[0].rptr == local(1));
  assert(local_beam[0].distance == 1.0F);
  assert(local_beam[1].rptr == local(0));
  assert(local_beam[2].rptr == local(2));
  assert(std::isinf(local_beam[1].distance));
  assert(std::isinf(local_beam[2].distance));
  assert(!std::isnan(local_beam[1].distance));

  detail::PartitionContinuationBeam continuation(kPartition, 2);
  const vec<detail::PartitionLocalSearchEntry> inherited{
    {local(0), nan, true},
  };
  continuation.seed_local(
    span<const detail::PartitionLocalSearchEntry>{inherited});
  const RemotePtr remote{1, 128};
  assert(continuation.try_visit_remote(remote));
  continuation.add_remote(remote, nan);
  assert(continuation.final_beam().size() == 2);
  for (const auto& entry : continuation.final_beam()) {
    assert(std::isinf(entry.distance));
    assert(!std::isnan(entry.distance));
  }
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

void test_stage1_production_frontier_is_complete_and_deterministic() {
  detail::PartitionLocalSearchBeam search(kPartition, 2);
  const vec<RemotePtr> entries{local(0)};
  u32 expansions = 0;
  detail::partition_local_construction_search_into(
    search, span<const RemotePtr>{entries}, kPartition, 2,
    detail::stage1_partition_search_budget(2, entries.size(), 8),
    [](RemotePtr pointer) -> std::optional<distance_t> {
      return 100.0F - static_cast<distance_t>(index_of(pointer));
    },
    [&](RemotePtr pointer, auto&& visit) {
      ++expansions;
      // Discovery order is deliberately reversed. Equal-priority frontier
      // candidates are retained by full handle order, not timing/order.
      if (pointer == local(0)) {
        for (u32 index = 0; index < 6; ++index) {
          visit(RemotePtr{3, static_cast<u64>(6 - index) * 16});
        }
        visit(local(1));
      } else if (pointer == local(1)) {
        visit(local(2));
      }
    });
  // Stage2 needs every unique boundary pointer because Stage1 has not read the
  // remote vectors and cannot soundly decide which ones fit the final beam.
  assert(search.remote_frontier().size() == 6);
  assert(!search.remote_frontier_truncated());
  assert(!search.budget_exhausted());
  assert(expansions == 3);  // Frontier overflow did not stop local convergence.
  for (size_t index = 1; index < search.remote_frontier().size(); ++index) {
    assert(search.remote_frontier()[index - 1].raw_address <
           search.remote_frontier()[index].raw_address);
  }
}

void test_algorithm_only_frontier_limit_is_diagnostic() {
  detail::PartitionLocalSearchBeam search(kPartition, 2);
  detail::PartitionSearchBudget budget =
    detail::PartitionSearchBudget::unbounded();
  budget.max_remote_frontier = 2;
  const vec<RemotePtr> entries{local(0)};
  detail::partition_local_construction_search_into(
    search, span<const RemotePtr>{entries}, kPartition, 2, budget,
    [](RemotePtr) -> std::optional<distance_t> { return 1.0F; },
    [&](RemotePtr, auto&& visit) {
      visit(RemotePtr{3, 48});
      visit(RemotePtr{3, 16});
      visit(RemotePtr{3, 32});
    });
  assert((search.remote_frontier() ==
          vec<RemotePtr>{RemotePtr{3, 16}, RemotePtr{3, 32}}));
  assert(search.remote_frontier_truncated());
  assert(!search.budget_exhausted());
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

void test_stage2_rejects_home_shard_returns() {
  const RemotePtr remote_a{1, 128};
  const RemotePtr remote_b{3, 256};
  const RemotePtr unseen_home = local(2);
  const vec<detail::PartitionLocalSearchEntry> local_beam{
    {local(0), 10.0F, true},
  };
  const vec<RemotePtr> frontier{remote_a};
  dense_hashmap_t<u64, distance_t> distances{
    {remote_a.raw_address, 8.0F},
    {unseen_home.raw_address, 4.0F},
    {remote_b.raw_address, 1.0F},
  };
  vec<RemotePtr> scored;
  vec<RemotePtr> expanded;

  const auto result = detail::continue_partition_construction_search(
    span<const detail::PartitionLocalSearchEntry>{local_beam},
      span<const RemotePtr>{frontier}, kPartition, 2,
    [&](span<const RemotePtr> batch, auto&& emit) {
      for (const RemotePtr pointer : batch) {
        scored.push_back(pointer);
        emit(pointer, distances.at(pointer.raw_address));
      }
    },
    [&](RemotePtr pointer, auto&& visit) {
      expanded.push_back(pointer);
      if (pointer == remote_a) {
        visit(local(0));       // Seeded Stage1 result is not repeated.
        visit(unseen_home);    // Stage2 never restarts the home-shard walk.
      } else if (pointer == unseen_home) {
        visit(remote_b);
      }
    });

  assert(result.size() == 2);
  assert(result[0].rptr == remote_a);
  assert(result[1].rptr == local(0));
  assert((scored == vec<RemotePtr>{remote_a}));
  assert((expanded == vec<RemotePtr>{remote_a}));
}

void test_completed_search_capacity_trim_preserves_reusability() {
  detail::PartitionLocalSearchBeam local_search(kPartition, 2);
  const vec<RemotePtr> entries{local(0)};
  detail::partition_local_construction_search_into(
    local_search, span<const RemotePtr>{entries}, kPartition, 2,
    [](RemotePtr) -> std::optional<distance_t> { return 1.0F; },
    [](RemotePtr, auto&&) {});
  const vec<detail::PartitionLocalSearchEntry> copied_local =
    local_search.final_beam();
  assert(copied_local.size() == 1);
  local_search.trim_oversized_capacity(0);
  detail::partition_local_construction_search_into(
    local_search, span<const RemotePtr>{entries}, kPartition, 2,
    [](RemotePtr) -> std::optional<distance_t> { return 2.0F; },
    [](RemotePtr, auto&&) {});
  const auto& reused_local = local_search.final_beam();
  assert(reused_local.size() == 1);
  assert(reused_local[0].distance == 2.0F);

  const RemotePtr remote{1, 128};
  const vec<detail::PartitionLocalSearchEntry> inherited{
    {local(0), 2.0F, true},
  };
  const vec<RemotePtr> frontier{remote};
  const std::array<detail::PartitionContinuationSeed, 1> seeds{{
    {span<const detail::PartitionLocalSearchEntry>{inherited},
     span<const RemotePtr>{frontier}},
  }};
  detail::PartitionContinuationBatch batch;
  batch.initialize(
    span<const detail::PartitionContinuationSeed>{seeds.data(), seeds.size()},
    kPartition, 2, detail::PartitionSearchBudget::unbounded());
  const std::array<detail::PartitionContinuationScoreResult, 1> scores{{
    {0, remote, 1.0F},
  }};
  batch.consume_score_results(
    span<const detail::PartitionContinuationScoreResult>{
      scores.data(), scores.size()});
  batch.consume_expand_results(
    span<const detail::PartitionContinuationExpandResult>{});
  assert(batch.complete());
  const auto copied_results = batch.results();
  assert(copied_results[0][0].rptr == remote);
  batch.trim_oversized_capacity(0);

  batch.initialize(
    span<const detail::PartitionContinuationSeed>{seeds.data(), seeds.size()},
    kPartition, 2, detail::PartitionSearchBudget::unbounded());
  batch.consume_score_results(
    span<const detail::PartitionContinuationScoreResult>{
      scores.data(), scores.size()});
  batch.consume_expand_results(
    span<const detail::PartitionContinuationExpandResult>{});
  assert(batch.complete());
  assert(batch.results()[0][0].rptr == remote);
}

void test_stage2_policy_converges_beyond_old_l_expansion_limit() {
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
  assert(expansions == 3);
  assert(expansions > 2);  // The old Stage2 E2 bound was L.
  assert(!exhausted);
  assert(result.size() == 2);
  assert(result[0].rptr == remote_c);
  assert(result[0].expanded);
}

void test_stage2_state_machine_pauses_and_resumes_between_waves() {
  const RemotePtr remote_a{1, 128};
  const RemotePtr remote_b{1, 256};
  const RemotePtr remote_c{1, 384};
  const vec<detail::PartitionLocalSearchEntry> local_beam{
    {local(0), 10.0F, true},
  };
  const vec<RemotePtr> frontier{remote_a};
  const std::array<detail::PartitionContinuationSeed, 1> seeds{{
    {span<const detail::PartitionLocalSearchEntry>{local_beam},
     span<const RemotePtr>{frontier}},
  }};

  detail::PartitionContinuationBatch batch;
  batch.initialize(
    span<const detail::PartitionContinuationSeed>{seeds.data(), seeds.size()},
    kPartition, 2, detail::stage2_partition_search_budget(2, 4));
  assert(batch.wave() == detail::PartitionContinuationWave::score);
  assert(batch.pending_score_requests().size() == 1);
  assert(batch.pending_score_requests()[0].pointer == remote_a);
  // Merely observing a ready wave is a pause: no beam state advances until
  // the corresponding completion is explicitly consumed.
  assert(batch.pending_score_requests()[0].pointer == remote_a);

  const std::array<detail::PartitionContinuationScoreResult, 1> score_a{{
    {0, remote_a, 3.0F},
  }};
  batch.consume_score_results(
    span<const detail::PartitionContinuationScoreResult>{
      score_a.data(), score_a.size()});
  assert(batch.wave() == detail::PartitionContinuationWave::expand);
  assert(batch.pending_expand_requests()[0].pointer == remote_a);

  const std::array<detail::PartitionContinuationExpandResult, 1> expand_a{{
    {0, remote_b},
  }};
  batch.consume_expand_results(
    span<const detail::PartitionContinuationExpandResult>{
      expand_a.data(), expand_a.size()});
  assert(batch.wave() == detail::PartitionContinuationWave::score);
  assert(batch.pending_score_requests()[0].pointer == remote_b);

  const std::array<detail::PartitionContinuationScoreResult, 1> score_b{{
    {0, remote_b, 2.0F},
  }};
  batch.consume_score_results(
    span<const detail::PartitionContinuationScoreResult>{
      score_b.data(), score_b.size()});
  assert(batch.pending_expand_requests()[0].pointer == remote_b);

  const std::array<detail::PartitionContinuationExpandResult, 1> expand_b{{
    {0, remote_c},
  }};
  batch.consume_expand_results(
    span<const detail::PartitionContinuationExpandResult>{
      expand_b.data(), expand_b.size()});
  const std::array<detail::PartitionContinuationScoreResult, 1> score_c{{
    {0, remote_c, 1.0F},
  }};
  batch.consume_score_results(
    span<const detail::PartitionContinuationScoreResult>{
      score_c.data(), score_c.size()});
  assert(batch.pending_expand_requests()[0].pointer == remote_c);
  batch.consume_expand_results(
    span<const detail::PartitionContinuationExpandResult>{});

  assert(batch.complete());
  assert(batch.results().size() == 1);
  assert(batch.results()[0][0].rptr == remote_c);
  assert(batch.expansion_count_results()[0] == 3);
  assert(!batch.budget_exhausted_results()[0]);
}

void test_stage2_state_machine_interleaves_independent_searches() {
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

  detail::PartitionContinuationBatch batch;
  batch.initialize(
    span<const detail::PartitionContinuationSeed>{seeds.data(), seeds.size()},
    kPartition, 2, detail::stage2_partition_search_budget(2, 4));
  size_t score_waves = 0;
  size_t expand_waves = 0;
  while (!batch.complete()) {
    if (batch.wave() == detail::PartitionContinuationWave::score) {
      const auto requests = batch.pending_score_requests();
      assert(requests.size() == 2);
      assert(requests[0].search_index != requests[1].search_index);
      vec<detail::PartitionContinuationScoreResult> scores;
      // Reverse completion order to prove one search's transport order does
      // not determine another search's beam.
      for (size_t index = requests.size(); index-- > 0;) {
        const auto& request = requests[index];
        scores.push_back({
          request.search_index, request.pointer,
          distances.at(request.pointer.raw_address)});
      }
      batch.consume_score_results(
        span<const detail::PartitionContinuationScoreResult>{scores});
      ++score_waves;
      continue;
    }
    const auto requests = batch.pending_expand_requests();
    assert(requests.size() == 2);
    assert(requests[0].search_index != requests[1].search_index);
    vec<detail::PartitionContinuationExpandResult> neighbors;
    for (size_t index = requests.size(); index-- > 0;) {
      const auto& request = requests[index];
      const auto found = next.find(request.pointer.raw_address);
      if (found != next.end()) {
        neighbors.push_back({request.search_index, found->second});
      }
    }
    batch.consume_expand_results(
      span<const detail::PartitionContinuationExpandResult>{neighbors});
    ++expand_waves;
  }

  assert(score_waves == 3);
  assert(expand_waves == 3);
  assert(batch.results()[0][0].rptr == a2);
  assert(batch.results()[1][0].rptr == b2);
  assert(batch.expansion_count_results()[0] == 3);
  assert(batch.expansion_count_results()[1] == 3);
}

void test_stage2_state_machine_matches_serial_natural_convergence() {
  const RemotePtr a0{1, 128};
  const RemotePtr a1{1, 256};
  const RemotePtr a2{3, 384};
  const RemotePtr b0{3, 128};
  const RemotePtr b1{3, 256};
  const vec<detail::PartitionLocalSearchEntry> local_a{
    {local(0), 15.0F, true}, {local(2), 18.0F, true}};
  const vec<detail::PartitionLocalSearchEntry> local_b{
    {local(1), 14.0F, true}};
  const vec<RemotePtr> frontier_a{a0};
  const vec<RemotePtr> frontier_b{b0};
  const std::array<detail::PartitionContinuationSeed, 2> seeds{{
    {span<const detail::PartitionLocalSearchEntry>{local_a},
     span<const RemotePtr>{frontier_a}},
    {span<const detail::PartitionLocalSearchEntry>{local_b},
     span<const RemotePtr>{frontier_b}},
  }};
  const std::array<dense_hashmap_t<u64, distance_t>, 2> distances{{
    {{a0.raw_address, 8.0F}, {a1.raw_address, 4.0F},
     {a2.raw_address, 1.0F}, {local(1).raw_address, 30.0F}},
    {{b0.raw_address, 7.0F}, {b1.raw_address, 2.0F},
     {a2.raw_address, 6.0F}},
  }};
  dense_hashmap_t<u64, vec<RemotePtr>> graph{
    {a0.raw_address, {a1, a1, local(1)}},
    {a1.raw_address, {a2}},
    {b0.raw_address, {b1}},
    {b1.raw_address, {a2}},
  };
  const auto budget = detail::stage2_partition_search_budget(3, 4);

  std::array<vec<detail::PartitionLocalSearchEntry>, 2> serial;
  std::array<bool, 2> serial_exhausted{};
  std::array<u64, 2> serial_expansions{};
  for (size_t search_index = 0; search_index < seeds.size(); ++search_index) {
    serial[search_index] = detail::continue_partition_construction_search(
      seeds[search_index].local_beam,
      seeds[search_index].remote_frontier, kPartition, 3, budget,
      [&](span<const RemotePtr> requests, auto&& emit) {
        for (const RemotePtr pointer : requests) {
          emit(pointer,
               distances[search_index].at(pointer.raw_address));
        }
      },
      [&](RemotePtr pointer, auto&& visit) {
        const auto found = graph.find(pointer.raw_address);
        if (found == graph.end()) return;
        for (const RemotePtr neighbor : found->second) visit(neighbor);
      }, &serial_exhausted[search_index],
      &serial_expansions[search_index]);
  }

  detail::PartitionContinuationBatch batch;
  batch.initialize(
    span<const detail::PartitionContinuationSeed>{seeds.data(), seeds.size()},
    kPartition, 3, budget);
  while (!batch.complete()) {
    if (batch.wave() == detail::PartitionContinuationWave::score) {
      vec<detail::PartitionContinuationScoreResult> scores;
      for (const auto& request : batch.pending_score_requests()) {
        scores.push_back({
          request.search_index, request.pointer,
          distances[request.search_index].at(request.pointer.raw_address)});
      }
      batch.consume_score_results(
        span<const detail::PartitionContinuationScoreResult>{scores});
      continue;
    }
    vec<detail::PartitionContinuationExpandResult> neighbors;
    for (const auto& request : batch.pending_expand_requests()) {
      const auto found = graph.find(request.pointer.raw_address);
      if (found == graph.end()) continue;
      for (const RemotePtr neighbor : found->second) {
        neighbors.push_back({request.search_index, neighbor});
      }
    }
    batch.consume_expand_results(
      span<const detail::PartitionContinuationExpandResult>{neighbors});
  }

  const auto& resumed = batch.results();
  assert(resumed.size() == serial.size());
  for (size_t search_index = 0; search_index < serial.size();
       ++search_index) {
    assert(resumed[search_index].size() == serial[search_index].size());
    for (size_t item = 0; item < serial[search_index].size(); ++item) {
      assert(resumed[search_index][item].rptr ==
             serial[search_index][item].rptr);
      assert(resumed[search_index][item].distance ==
             serial[search_index][item].distance);
      assert(resumed[search_index][item].expanded ==
             serial[search_index][item].expanded);
    }
    assert(batch.budget_exhausted_results()[search_index] ==
           serial_exhausted[search_index]);
    assert(batch.expansion_count_results()[search_index] ==
           serial_expansions[search_index]);
  }
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
  assert(expansions[0] == 3 && expansions[1] == 3);
  assert(!exhausted[0] && !exhausted[1]);
  // Two independent searches share each physical I/O wave. Their expansion
  // state remains per-search while both beams run to natural convergence.
  assert(score_waves == 3);
  assert(expansion_waves == 3);
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

void test_stage2_per_search_progress_is_not_blocked_by_delayed_peer() {
  const RemotePtr stable_a{1, 128};
  const RemotePtr delayed{1, 256};
  const RemotePtr healthy0{3, 128};
  const RemotePtr healthy1{3, 256};
  const vec<detail::PartitionLocalSearchEntry> local_a{
    {local(0), 20.0F, true}};
  const vec<detail::PartitionLocalSearchEntry> local_b{
    {local(1), 20.0F, true}};
  const vec<RemotePtr> frontier_a{stable_a, delayed};
  const vec<RemotePtr> frontier_b{healthy0};
  const std::array<detail::PartitionContinuationSeed, 2> seeds{{
    {span<const detail::PartitionLocalSearchEntry>{local_a},
     span<const RemotePtr>{frontier_a}},
    {span<const detail::PartitionLocalSearchEntry>{local_b},
     span<const RemotePtr>{frontier_b}},
  }};

  detail::PartitionContinuationBatch batch;
  batch.initialize(
    span<const detail::PartitionContinuationSeed>{seeds.data(), seeds.size()},
    kPartition, 2, detail::PartitionSearchBudget::unbounded());
  const u64 delayed_generation = batch.generation(0);
  const u64 healthy_score0 = batch.generation(1);
  assert(batch.search_wave(0) == detail::PartitionContinuationWave::score);
  assert(batch.search_wave(1) == detail::PartitionContinuationWave::score);

  // Resolving only part of search 0's score wave must not permit it to select
  // an expansion early. Its second vector remains retryable in this model.
  assert(batch.resolve_score_request(
    0, delayed_generation, stable_a, distance_t{3.0F}));
  assert(batch.search_wave(0) == detail::PartitionContinuationWave::score);
  assert(batch.generation(0) == delayed_generation);
  assert(batch.pending_score_requests(0).size() == 1);

  // Search 0 models a retryable vector and remains unresolved. Search 1 is
  // nevertheless allowed to score, expand, score again, and finish.
  assert(batch.resolve_score_request(
    1, healthy_score0, healthy0, distance_t{4.0F}));
  assert(batch.search_wave(0) == detail::PartitionContinuationWave::score);
  assert(batch.search_wave(1) == detail::PartitionContinuationWave::expand);
  const u64 healthy_expand0 = batch.generation(1);
  const std::array<RemotePtr, 1> next{{healthy1}};
  assert(batch.resolve_expand_request(
    1, healthy_expand0, span<const RemotePtr>{next.data(), next.size()}));
  const u64 healthy_score1 = batch.generation(1);
  assert(batch.resolve_score_request(
    1, healthy_score1, healthy1, distance_t{1.0F}));
  const u64 healthy_expand1 = batch.generation(1);
  assert(batch.resolve_expand_request(
    1, healthy_expand1, span<const RemotePtr>{}));

  assert(batch.search_complete(1));
  assert(batch.result(1)[0].rptr == healthy1);
  assert(batch.expansion_count_result(1) == 2);
  assert(!batch.search_complete(0));
  assert(batch.generation(0) == delayed_generation);
  assert(batch.pending_score_requests(0).size() == 1);
  assert(batch.pending_score_requests(0)[0].pointer == delayed);
  assert(!batch.all_complete());

  // A stable terminal observation releases the complete score wave. The
  // surviving stable candidate then follows the normal expand dependency.
  assert(batch.resolve_score_request(
    0, delayed_generation, delayed, std::nullopt));
  assert(batch.search_wave(0) == detail::PartitionContinuationWave::expand);
  const u64 delayed_expand_generation = batch.generation(0);
  assert(batch.resolve_expand_request(
    0, delayed_expand_generation, span<const RemotePtr>{}));
  assert(batch.all_complete());
  assert(batch.result(0).size() == 2);
  assert(batch.result(0)[0].rptr == stable_a);
}

void test_stage2_generation_rejects_stale_completions() {
  const RemotePtr remote{1, 128};
  const vec<detail::PartitionLocalSearchEntry> local_beam{
    {local(0), 10.0F, true}};
  const vec<RemotePtr> frontier{remote};
  const std::array<detail::PartitionContinuationSeed, 1> seeds{{
    {span<const detail::PartitionLocalSearchEntry>{local_beam},
     span<const RemotePtr>{frontier}},
  }};
  detail::PartitionContinuationBatch batch;
  batch.initialize(
    span<const detail::PartitionContinuationSeed>{seeds.data(), seeds.size()},
    kPartition, 2, detail::PartitionSearchBudget::unbounded());

  const u64 score_generation = batch.generation(0);
  assert(score_generation != 0);
  assert(!batch.resolve_score_request(
    0, score_generation + 1, remote, distance_t{1.0F}));
  assert(batch.pending_score_requests(0).size() == 1);
  assert(batch.resolve_score_request(
    0, score_generation, remote, distance_t{1.0F}));
  assert(batch.search_wave(0) == detail::PartitionContinuationWave::expand);
  const u64 expand_generation = batch.generation(0);
  assert(expand_generation != score_generation);

  // A late score completion from the preceding generation cannot alter the
  // beam or satisfy the current adjacency dependency.
  assert(!batch.resolve_score_request(
    0, score_generation, remote, distance_t{0.0F}));
  assert(batch.generation(0) == expand_generation);
  assert(batch.resolve_expand_request(
    0, expand_generation, span<const RemotePtr>{}));
  assert(batch.search_complete(0));
  assert(batch.result(0)[0].distance == 1.0F);
  assert(!batch.resolve_expand_request(
    0, expand_generation, span<const RemotePtr>{}));
}

void test_stage2_searches_can_be_activated_independently() {
  const vec<detail::PartitionLocalSearchEntry> local_a{
    {local(0), 1.0F, true}};
  const vec<detail::PartitionLocalSearchEntry> local_b{
    {local(1), 2.0F, true}};
  const detail::PartitionContinuationSeed seed_a{
    span<const detail::PartitionLocalSearchEntry>{local_a},
    span<const RemotePtr>{}};
  const detail::PartitionContinuationSeed seed_b{
    span<const detail::PartitionLocalSearchEntry>{local_b},
    span<const RemotePtr>{}};

  detail::PartitionContinuationBatch batch;
  batch.initialize(
    2, kPartition, 2, detail::PartitionSearchBudget::unbounded());
  assert(!batch.search_active(0));
  assert(!batch.search_active(1));
  assert(!batch.all_complete());

  batch.initialize_search(1, seed_b);
  assert(batch.search_complete(1));
  assert(batch.result(1)[0].rptr == local(1));
  // An inactive Stage1 handoff is not mistaken for completed work.
  assert(!batch.all_complete());
  batch.initialize_search(0, seed_a);
  assert(batch.all_complete());
  assert(batch.result(0)[0].rptr == local(0));
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
  test_stage1_policy_converges_beyond_old_2l_expansion_limit();
  test_stale_entries_do_not_block_other_roots();
  test_final_validation_removes_concurrently_stale_entries();
  test_stage2_vector_snapshot_validation_matches_stable_search();
  test_transient_node_snapshot_is_retryable_not_terminal();
  test_stage2_target_snapshot_classification();
  test_reusable_state_is_cleared_between_searches();
  test_nan_distances_are_canonicalized_to_positive_infinity();
  test_zero_width_is_rejected();
  test_stage1_exports_unique_remote_frontier();
  test_stage1_production_frontier_is_complete_and_deterministic();
  test_algorithm_only_frontier_limit_is_diagnostic();
  test_stage2_continues_stage1_without_restarting_local_search();
  test_stage2_rejects_home_shard_returns();
  test_completed_search_capacity_trim_preserves_reusability();
  test_stage2_policy_converges_beyond_old_l_expansion_limit();
  test_stage2_state_machine_pauses_and_resumes_between_waves();
  test_stage2_state_machine_interleaves_independent_searches();
  test_stage2_state_machine_matches_serial_natural_convergence();
  test_stage2_batch_wavefront_preserves_independent_searches();
  test_stage2_batch_missing_snapshot_is_isolated();
  test_stage2_per_search_progress_is_not_blocked_by_delayed_peer();
  test_stage2_generation_rejects_stale_completions();
  test_stage2_searches_can_be_activated_independently();
  test_final_home_strictly_reduces_cross_shard_edges();
  return 0;
}
