#include <cassert>
#include <limits>

#include "memory_node/storage_owner_index/partition_local_search.hh"

namespace detail = memory_node_storage_owner_index_detail;

namespace {

constexpr u32 kPartition = 2;

RemotePtr local(u32 index) {
  return RemotePtr{kPartition, (static_cast<u64>(index) + 1) * 16};
}

detail::Stage2BudgetBacklogPolicy policy(
    u32 low, u32 high, u64 expansions, u64 visits) {
  return detail::Stage2BudgetBacklogPolicy{
    .backlog_low = low,
    .backlog_high = high,
    .max_expansions = expansions,
    .max_candidate_visits = visits,
  };
}

void test_disabled_policy_is_unbounded() {
  const auto unbounded = detail::PartitionSearchBudget::unbounded();
  assert(!detail::stage2_budget_backlog_policy_enabled(policy(0, 0, 8, 256)));
  // A lone low watermark is not an enabled controller.
  assert(!detail::stage2_budget_backlog_policy_enabled(policy(4, 0, 8, 256)));
  for (const auto& p : {policy(0, 0, 8, 256), policy(4, 0, 8, 256)}) {
    const auto budget =
      detail::stage2_partition_search_budget(64, 72, 1'000'000, p);
    assert(budget.max_expansions == unbounded.max_expansions);
    assert(budget.max_candidate_visits == unbounded.max_candidate_visits);
    assert(budget.max_remote_frontier == unbounded.max_remote_frontier);
  }
}

void test_low_watermark_keeps_unbounded_budget() {
  const auto p = policy(100, 200, 8, 256);
  assert(detail::stage2_budget_backlog_policy_enabled(p));
  const auto unbounded = detail::PartitionSearchBudget::unbounded();
  for (const u64 backlog : {0ul, 1ul, 99ul, 100ul}) {
    const auto budget =
      detail::stage2_partition_search_budget(64, 72, backlog, p);
    assert(budget.max_expansions == unbounded.max_expansions);
    assert(budget.max_candidate_visits == unbounded.max_candidate_visits);
    assert(budget.max_remote_frontier == unbounded.max_remote_frontier);
  }
}

void test_high_watermark_applies_tight_budget() {
  const auto p = policy(100, 200, 8, 256);
  for (const u64 backlog : {200ul, 201ul, 1'000'000ul}) {
    const auto budget =
      detail::stage2_partition_search_budget(64, 72, backlog, p);
    assert(budget.max_expansions == 8);
    assert(budget.max_candidate_visits == 256);
    // The Stage1 handoff frontier is never truncated by the backlog policy.
    assert(budget.max_remote_frontier == std::numeric_limits<size_t>::max());
  }
}

void test_linear_interpolation_between_watermarks() {
  const auto p = policy(100, 200, 8, 256);
  // Nominal anchors: expansions max(8, beam 64) = 64; visits
  // max(256, 64 * 72) = 4608.
  const auto mid = detail::stage2_partition_search_budget(64, 72, 150, p);
  assert(mid.max_expansions == 64 - (64 - 8) * 50 / 100);
  assert(mid.max_candidate_visits == 4608 - (4608 - 256) * 50 / 100);
  const auto quarter =
    detail::stage2_partition_search_budget(64, 72, 125, p);
  assert(quarter.max_expansions == 64 - (64 - 8) * 25 / 100);

  u64 previous = std::numeric_limits<u64>::max();
  for (u64 backlog = 101; backlog <= 200; ++backlog) {
    const auto budget =
      detail::stage2_partition_search_budget(64, 72, backlog, p);
    assert(budget.max_expansions <= previous);
    assert(budget.max_expansions >= 8);
    previous = budget.max_expansions;
  }
}

void test_tight_above_nominal_clamps_to_tight() {
  // A configured tight value beyond the nominal anchor applies as soon as the
  // backlog crosses the low watermark.
  const auto p = policy(100, 200, 500, 256);
  const auto budget = detail::stage2_partition_search_budget(64, 72, 101, p);
  assert(budget.max_expansions == 500);
}

void test_zero_tight_dimension_stays_unbounded() {
  const auto p = policy(100, 200, 0, 256);
  const auto budget = detail::stage2_partition_search_budget(64, 72, 500, p);
  assert(budget.max_expansions == std::numeric_limits<u64>::max());
  assert(budget.max_candidate_visits == 256);
}

void test_tight_budget_bounds_continuation_and_marks_exhaustion() {
  const RemotePtr remote_a{1, 128};
  const RemotePtr remote_b{1, 256};
  const RemotePtr remote_c{1, 384};
  const vec<detail::PartitionLocalSearchEntry> local_beam{
    {local(0), 10.0F, true},
  };
  const vec<RemotePtr> frontier{remote_a};
  const dense_hashmap_t<u64, distance_t> distances{
    {remote_a.raw_address, 3.0F},
    {remote_b.raw_address, 2.0F},
    {remote_c.raw_address, 1.0F},
  };
  const auto budget = detail::stage2_partition_search_budget(
    2, 4, 500, policy(100, 200, 1, 0));
  assert(budget.max_expansions == 1);

  u32 expansions = 0;
  bool exhausted = false;
  const auto result = detail::continue_partition_construction_search(
    span<const detail::PartitionLocalSearchEntry>{local_beam},
    span<const RemotePtr>{frontier}, kPartition, 2, budget,
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
  assert(expansions == 1);
  assert(exhausted);
  // The single expansion still contributed its candidate to the beam.
  assert(!result.empty());
  assert(result[0].rptr == remote_b);
}

}  // namespace

int main() {
  test_disabled_policy_is_unbounded();
  test_low_watermark_keeps_unbounded_budget();
  test_high_watermark_applies_tight_budget();
  test_linear_interpolation_between_watermarks();
  test_tight_above_nominal_clamps_to_tight();
  test_zero_tight_dimension_stays_unbounded();
  test_tight_budget_bounds_continuation_and_marks_exhaustion();
  return 0;
}
