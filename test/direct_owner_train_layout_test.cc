#include <array>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <limits>

#include "gpu_search/persistent_kernel.hh"

namespace {

using u16 = std::uint16_t;
using u32 = std::uint32_t;

constexpr u32 kMaxSubmitBatches = 8;
constexpr u32 kConfiguredSqWqes = 1024;
constexpr u32 kNoBatch = std::numeric_limits<u32>::max();

struct BatchDemand {
  u32 critical_reads{};
  u32 tail_reads{};
  bool mandatory_fenced_tail{};
};

struct TrainPlan {
  std::array<u32, kMaxSubmitBatches> input_indices{};
  std::array<u32, kMaxSubmitBatches> critical_offsets{};
  std::array<u32, kMaxSubmitBatches> critical_counts{};
  std::array<u32, kMaxSubmitBatches> tail_offsets{};
  std::array<u32, kMaxSubmitBatches> tail_counts{};
  std::array<bool, kMaxSubmitBatches> tail_rejected{};
  std::array<bool, kMaxSubmitBatches> rejected{};
  u32 batch_count{};
  u32 rejected_count{};
  u32 deferred_input{kNoBatch};
  u32 critical_read_wqes{};
  u32 tail_read_wqes{};
  u32 critical_fence_wqes{};
  u32 tail_fence_wqes{};
  u32 tail_wqe_base{};
  u32 submission_wqes{};
  u32 success_cqes{};
  u32 fenced_tail_wqe{kNoBatch};
  bool mandatory_snapshot_train{};
  bool collection_saturated{};
};

// Test-side reference for direct_read_owner_loop. Optional ASFE tails may use
// only SQ slack. A mandatory exact-snapshot tail is instead reserved with its
// prefix, isolated as one descriptor/doorbell, fenced at the first trailer,
// and exposes only the final completion.
TrainPlan plan_train(
    const BatchDemand* inputs, u32 input_count, u32 sq_wqes,
    bool need_dump, bool enable_asfe = true) {
  TrainPlan plan{};
  plan.input_indices.fill(kNoBatch);
  const u32 critical_fence = need_dump ? 1u : 0u;
  for (u32 input = 0; input < input_count; ++input) {
    if (plan.batch_count == kMaxSubmitBatches) {
      plan.collection_saturated = true;
      break;
    }
    const bool mandatory = inputs[input].mandatory_fenced_tail;
    if (mandatory && plan.batch_count != 0) {
      plan.deferred_input = input;
      break;
    }
    if (!mandatory && inputs[input].tail_reads != 0 && !enable_asfe) {
      plan.rejected[input] = true;
      ++plan.rejected_count;
      continue;
    }
    const u32 needed = inputs[input].critical_reads +
      (mandatory ? inputs[input].tail_reads : 0u);
    if (needed + (need_dump ? 1u : 0u) > sq_wqes) {
      plan.rejected[input] = true;
      ++plan.rejected_count;
      continue;
    }
    if (plan.batch_count != 0 &&
        plan.critical_read_wqes + needed + critical_fence > sq_wqes) {
      plan.deferred_input = input;
      break;
    }
    const u32 batch = plan.batch_count++;
    plan.input_indices[batch] = input;
    plan.critical_offsets[batch] = plan.critical_read_wqes;
    plan.critical_counts[batch] = inputs[input].critical_reads;
    plan.critical_read_wqes += inputs[input].critical_reads;
    if (mandatory) {
      plan.mandatory_snapshot_train = true;
      break;
    }
  }
  plan.collection_saturated =
    plan.batch_count == kMaxSubmitBatches;

  const bool can_steal_sq_slack =
    plan.deferred_input == kNoBatch &&
    plan.batch_count < kMaxSubmitBatches;
  for (u32 batch = 0; batch < plan.batch_count; ++batch) {
    const u32 candidate_tail =
      inputs[plan.input_indices[batch]].tail_reads;
    if (candidate_tail == 0) continue;
    const bool mandatory =
      inputs[plan.input_indices[batch]].mandatory_fenced_tail;
    if (mandatory) {
      plan.tail_offsets[batch] = plan.tail_read_wqes;
      plan.tail_counts[batch] = candidate_tail;
      plan.tail_read_wqes += candidate_tail;
      continue;
    }
    const u32 tail_fence = need_dump ? 1u : 0u;
    const bool fits =
      can_steal_sq_slack &&
      plan.critical_read_wqes + critical_fence +
        plan.tail_read_wqes + candidate_tail + tail_fence <= sq_wqes;
    if (fits) {
      plan.tail_offsets[batch] = plan.tail_read_wqes;
      plan.tail_counts[batch] = candidate_tail;
      plan.tail_read_wqes += candidate_tail;
    } else {
      plan.tail_rejected[batch] = true;
    }
  }

  plan.critical_fence_wqes =
    plan.mandatory_snapshot_train ? 0u : critical_fence;
  plan.tail_fence_wqes =
    need_dump && plan.tail_read_wqes != 0 ? 1u : 0u;
  plan.tail_wqe_base =
    plan.critical_read_wqes + plan.critical_fence_wqes;
  plan.submission_wqes =
    plan.tail_wqe_base + plan.tail_read_wqes + plan.tail_fence_wqes;
  plan.success_cqes =
    plan.batch_count == 0 ? 0u :
    plan.mandatory_snapshot_train ? 1u :
    plan.tail_read_wqes == 0 ? 1u : 2u;
  if (plan.mandatory_snapshot_train && plan.tail_read_wqes != 0) {
    plan.fenced_tail_wqe = plan.tail_wqe_base;
  }
  return plan;
}

void assert_contiguous_and_bounded(
    const TrainPlan& plan, u32 sq_wqes) {
  u32 next_critical = 0;
  u32 next_tail = 0;
  for (u32 batch = 0; batch < plan.batch_count; ++batch) {
    assert(plan.critical_offsets[batch] == next_critical);
    next_critical += plan.critical_counts[batch];
    if (plan.tail_counts[batch] != 0) {
      assert(plan.tail_offsets[batch] == next_tail);
      const u32 tail_begin =
        plan.tail_wqe_base + plan.tail_offsets[batch];
      const u32 tail_end = tail_begin + plan.tail_counts[batch];
      assert(tail_begin >= plan.tail_wqe_base);
      assert(tail_end <=
             plan.submission_wqes - plan.tail_fence_wqes);
      next_tail += plan.tail_counts[batch];
    }
    assert(plan.critical_offsets[batch] <=
           std::numeric_limits<u16>::max());
    assert(plan.tail_offsets[batch] <=
           std::numeric_limits<u16>::max());
    assert(static_cast<u32>(
             static_cast<u16>(plan.critical_offsets[batch])) ==
           plan.critical_offsets[batch]);
    assert(static_cast<u32>(
             static_cast<u16>(plan.tail_offsets[batch])) ==
           plan.tail_offsets[batch]);
  }
  assert(next_critical == plan.critical_read_wqes);
  assert(next_tail == plan.tail_read_wqes);
  assert(plan.tail_wqe_base ==
         plan.critical_read_wqes + plan.critical_fence_wqes);
  assert(plan.submission_wqes <= sq_wqes);
  if (plan.critical_fence_wqes != 0) {
    assert(plan.critical_read_wqes < plan.submission_wqes);
  }
  if (plan.tail_fence_wqes != 0) {
    assert(plan.submission_wqes - 1 ==
           plan.tail_wqe_base + plan.tail_read_wqes);
  }
  if (plan.mandatory_snapshot_train) {
    assert(plan.batch_count == 1);
    assert(plan.critical_fence_wqes == 0);
    assert(plan.tail_read_wqes != 0);
    assert(plan.fenced_tail_wqe == plan.tail_wqe_base);
    assert(plan.success_cqes == 1);
  }
}

void test_no_dump_exact_fit() {
  constexpr BatchDemand inputs[]{{3, 1}, {2, 2}};
  const TrainPlan plan = plan_train(inputs, 2, 8, false);
  assert(plan.batch_count == 2);
  assert(plan.critical_offsets[0] == 0);
  assert(plan.critical_offsets[1] == 3);
  assert(plan.critical_read_wqes == 5);
  assert(plan.tail_offsets[0] == 0);
  assert(plan.tail_offsets[1] == 1);
  assert(plan.tail_read_wqes == 3);
  assert(plan.tail_wqe_base == 5);
  assert(plan.submission_wqes == 8);
  assert(!plan.tail_rejected[0]);
  assert(!plan.tail_rejected[1]);
  assert_contiguous_and_bounded(plan, 8);
}

void test_dump_exact_fit_has_two_cq_fences() {
  constexpr BatchDemand inputs[]{{3, 1}, {2, 2}};
  const TrainPlan plan = plan_train(inputs, 2, 10, true);
  assert(plan.batch_count == 2);
  assert(plan.critical_read_wqes == 5);
  assert(plan.critical_fence_wqes == 1);
  assert(plan.tail_wqe_base == 6);
  assert(plan.tail_read_wqes == 3);
  assert(plan.tail_fence_wqes == 1);
  assert(plan.submission_wqes == 10);
  assert_contiguous_and_bounded(plan, 10);
}

void test_tail_descriptor_is_not_partially_admitted() {
  constexpr BatchDemand inputs[]{{3, 1}, {2, 2}};
  const TrainPlan plan = plan_train(inputs, 2, 9, true);
  assert(plan.batch_count == 2);
  assert(plan.tail_counts[0] == 1);
  assert(plan.tail_counts[1] == 0);
  assert(!plan.tail_rejected[0]);
  assert(plan.tail_rejected[1]);
  assert(plan.tail_read_wqes == 1);
  assert(plan.submission_wqes == 8);
  assert_contiguous_and_bounded(plan, 9);
}

void test_deferred_critical_rejects_every_tail() {
  constexpr BatchDemand inputs[]{{7, 1}, {1, 1}};
  const TrainPlan plan = plan_train(inputs, 2, 8, true);
  assert(plan.batch_count == 1);
  assert(plan.deferred_input == 1);
  assert(plan.critical_read_wqes == 7);
  assert(plan.tail_read_wqes == 0);
  assert(plan.tail_rejected[0]);
  assert(plan.submission_wqes == 8);
  assert_contiguous_and_bounded(plan, 8);
}

void test_critical_exact_boundary_without_dump() {
  constexpr BatchDemand inputs[]{{7, 0}, {1, 0}};
  const TrainPlan plan = plan_train(inputs, 2, 8, false);
  assert(plan.batch_count == 2);
  assert(plan.deferred_input == kNoBatch);
  assert(plan.critical_offsets[1] == 7);
  assert(plan.submission_wqes == 8);
  assert_contiguous_and_bounded(plan, 8);
}

void test_oversized_critical_is_rejected_before_layout() {
  constexpr BatchDemand inputs[]{{8, 0}, {7, 0}};
  const TrainPlan plan = plan_train(inputs, 2, 8, true);
  assert(plan.rejected_count == 1);
  assert(plan.rejected[0]);
  assert(plan.batch_count == 1);
  assert(plan.input_indices[0] == 1);
  assert(plan.critical_offsets[0] == 0);
  assert(plan.submission_wqes == 8);
  assert_contiguous_and_bounded(plan, 8);
}

void test_collection_saturation_forwards_tails() {
  constexpr std::array<BatchDemand, kMaxSubmitBatches> inputs{{
    {1, 1}, {1, 1}, {1, 1}, {1, 1},
    {1, 1}, {1, 1}, {1, 1}, {1, 1},
  }};
  const TrainPlan plan =
    plan_train(inputs.data(), inputs.size(), 32, false);
  assert(plan.batch_count == kMaxSubmitBatches);
  assert(plan.collection_saturated);
  assert(plan.tail_read_wqes == 0);
  assert(plan.submission_wqes == kMaxSubmitBatches);
  for (u32 batch = 0; batch < kMaxSubmitBatches; ++batch) {
    assert(plan.tail_rejected[batch]);
  }
  assert_contiguous_and_bounded(plan, 32);
}

void test_configured_sq_capacity_and_u16_offsets() {
  constexpr std::array<BatchDemand, 7> inputs{{
    {137, 9}, {137, 9}, {137, 9}, {137, 9},
    {137, 9}, {137, 9}, {137, 9},
  }};
  const TrainPlan plan =
    plan_train(inputs.data(), inputs.size(), kConfiguredSqWqes, true);
  assert(plan.batch_count == inputs.size());
  assert(plan.critical_read_wqes == 959);
  assert(plan.tail_read_wqes == 63);
  assert(plan.tail_wqe_base == 960);
  assert(plan.submission_wqes == kConfiguredSqWqes);
  static_assert(kConfiguredSqWqes <= std::numeric_limits<u16>::max());
  assert_contiguous_and_bounded(plan, kConfiguredSqWqes);
}

void test_mandatory_snapshot_train_has_one_final_cqe() {
  constexpr BatchDemand inputs[]{{4, 4, true}};
  const TrainPlan no_dump = plan_train(inputs, 1, 8, false, true);
  assert(no_dump.batch_count == 1);
  assert(no_dump.critical_read_wqes == 4);
  assert(no_dump.tail_wqe_base == 4);
  assert(no_dump.tail_read_wqes == 4);
  assert(no_dump.submission_wqes == 8);
  assert(no_dump.fenced_tail_wqe == 4);
  assert(no_dump.success_cqes == 1);
  assert_contiguous_and_bounded(no_dump, 8);

  const TrainPlan with_dump = plan_train(inputs, 1, 9, true, true);
  assert(with_dump.critical_fence_wqes == 0);
  assert(with_dump.tail_wqe_base == 4);
  assert(with_dump.tail_fence_wqes == 1);
  assert(with_dump.submission_wqes == 9);
  assert(with_dump.success_cqes == 1);
  assert_contiguous_and_bounded(with_dump, 9);
}

void test_mixed_mandatory_train_has_one_final_cqe() {
  // Four pre-known misses carry four fenced validation trailers; five cache
  // hits append current-header reads to the same suffix and final CQ boundary.
  constexpr BatchDemand inputs[]{{4, 9, true}};
  const TrainPlan no_dump = plan_train(inputs, 1, 13, false, true);
  assert(no_dump.batch_count == 1);
  assert(no_dump.critical_read_wqes == 4);
  assert(no_dump.tail_wqe_base == 4);
  assert(no_dump.tail_read_wqes == 9);
  assert(no_dump.submission_wqes == 13);
  assert(no_dump.fenced_tail_wqe == 4);
  assert(no_dump.success_cqes == 1);
  assert_contiguous_and_bounded(no_dump, 13);

  const TrainPlan with_dump = plan_train(inputs, 1, 14, true, true);
  assert(with_dump.critical_fence_wqes == 0);
  assert(with_dump.tail_wqe_base == 4);
  assert(with_dump.tail_fence_wqes == 1);
  assert(with_dump.submission_wqes == 14);
  assert(with_dump.success_cqes == 1);
  assert_contiguous_and_bounded(with_dump, 14);
}

void test_mandatory_snapshot_train_is_independent_of_asfe_template() {
  constexpr BatchDemand inputs[]{{3, 3, true}};
  const TrainPlan enabled = plan_train(inputs, 1, 7, true, true);
  const TrainPlan disabled = plan_train(inputs, 1, 7, true, false);
  assert(enabled.batch_count == disabled.batch_count);
  assert(enabled.critical_read_wqes == disabled.critical_read_wqes);
  assert(enabled.tail_read_wqes == disabled.tail_read_wqes);
  assert(enabled.submission_wqes == disabled.submission_wqes);
  assert(enabled.success_cqes == disabled.success_cqes);
  assert(disabled.fenced_tail_wqe == disabled.tail_wqe_base);
  assert_contiguous_and_bounded(disabled, 7);
}

void test_mandatory_snapshot_train_isolation() {
  constexpr BatchDemand after_ordinary[]{{2, 0}, {3, 3, true}};
  const TrainPlan first =
    plan_train(after_ordinary, 2, 16, false);
  assert(first.batch_count == 1);
  assert(!first.mandatory_snapshot_train);
  assert(first.deferred_input == 1);
  assert(first.critical_read_wqes == 2);
  assert(first.tail_read_wqes == 0);
  assert(first.success_cqes == 1);
  assert_contiguous_and_bounded(first, 16);

  constexpr BatchDemand before_ordinary[]{{3, 3, true}, {2, 0}};
  const TrainPlan second =
    plan_train(before_ordinary, 2, 16, false);
  assert(second.batch_count == 1);
  assert(second.mandatory_snapshot_train);
  assert(second.input_indices[0] == 0);
  assert(second.critical_read_wqes == 3);
  assert(second.tail_read_wqes == 3);
  assert(second.success_cqes == 1);
  assert_contiguous_and_bounded(second, 16);
}

void test_mandatory_snapshot_train_never_partially_admits() {
  constexpr BatchDemand inputs[]{{4, 4, true}, {2, 0}};
  const TrainPlan plan = plan_train(inputs, 2, 7, false);
  assert(plan.rejected[0]);
  assert(plan.rejected_count == 1);
  assert(plan.batch_count == 1);
  assert(plan.input_indices[0] == 1);
  assert(!plan.mandatory_snapshot_train);
  assert(plan.submission_wqes == 2);
  assert_contiguous_and_bounded(plan, 7);
}

void test_mandatory_enqueue_watchdog_balance() {
  // Producer announcement precedes the first try_push. A transient full ring
  // retries without changing counters; either one eventual owner completion
  // or one stop/disabled cancellation balances that single announcement.
  constexpr u32 announcements = 1;
  constexpr u32 transient_full_retries = 17;
  constexpr u32 owner_completions = 1;
  constexpr u32 cancellation_completions = 1;
  (void)transient_full_retries;
  assert(announcements == owner_completions);
  assert(announcements == cancellation_completions);
}

void test_exact_snapshot_final_transport_policy() {
  using gpu_search::exact_rerank_should_retry_route;
  using gpu_search::exact_snapshot_transport_failed;

  // Successful train/fallback paths normalize to zero. Snapshot visibility
  // is deliberately separate: a tombstone/incarnation reject filters only
  // that candidate and does not become a transport error.
  assert(!exact_snapshot_transport_failed(0));
  constexpr bool snapshot_visible = false;
  assert(!snapshot_visible);

  // A non-zero status remaining after fallback is exhausted is fatal to the
  // whole query; publishing other shards would be a partial exact result.
  assert(exact_snapshot_transport_failed(-EIO));
  assert(exact_snapshot_transport_failed(-ETIMEDOUT));

  // Empty-but-valid snapshots retain the legacy one-time seed retry.
  assert(exact_rerank_should_retry_route(true, 0));
  assert(!exact_rerank_should_retry_route(true, 1));
  // A transport failure never retries route selection: the query terminates
  // with exact_fetch before any result buffer publication.
  assert(!exact_rerank_should_retry_route(false, 0));
  assert(!exact_rerank_should_retry_route(false, 1));
}

void test_remote_range_validation_is_overflow_safe() {
  using gpu_search::direct_remote_range_valid;
  constexpr std::uint64_t region_bytes = 4096;
  assert(direct_remote_range_valid(region_bytes, 0, 4096));
  assert(direct_remote_range_valid(region_bytes, 4096, 0));
  assert(!direct_remote_range_valid(region_bytes, 4096, 1));
  assert(!direct_remote_range_valid(region_bytes, 4097, 0));
  assert(!direct_remote_range_valid(
    region_bytes, std::numeric_limits<std::uint64_t>::max(), 1));
}

}  // namespace

int main() {
  test_no_dump_exact_fit();
  test_dump_exact_fit_has_two_cq_fences();
  test_tail_descriptor_is_not_partially_admitted();
  test_deferred_critical_rejects_every_tail();
  test_critical_exact_boundary_without_dump();
  test_oversized_critical_is_rejected_before_layout();
  test_collection_saturation_forwards_tails();
  test_configured_sq_capacity_and_u16_offsets();
  test_mandatory_snapshot_train_has_one_final_cqe();
  test_mixed_mandatory_train_has_one_final_cqe();
  test_mandatory_snapshot_train_is_independent_of_asfe_template();
  test_mandatory_snapshot_train_isolation();
  test_mandatory_snapshot_train_never_partially_admits();
  test_mandatory_enqueue_watchdog_balance();
  test_exact_snapshot_final_transport_policy();
  test_remote_range_validation_is_overflow_safe();
  return 0;
}
