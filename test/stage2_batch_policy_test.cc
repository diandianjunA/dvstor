#include <cassert>
#include <chrono>

#include "memory_node/storage_owner_maintenance/stage2_batch_policy.hh"

namespace {

namespace detail = memory_node_storage_owner_maintenance_detail;
using Clock = std::chrono::steady_clock;

void test_legacy_helpers_keep_oldest_deadline() {
  const Clock::time_point oldest{};
  assert(!detail::stage2_batch_ready(0, 32, oldest, oldest, 50));
  assert(detail::stage2_batch_ready(32, 32, oldest, oldest, 50));
  const auto deadline = detail::stage2_partial_batch_deadline(
    7, 32, oldest, 50);
  assert(deadline == oldest + std::chrono::microseconds(50));
  assert(!detail::stage2_batch_ready(
    7, 32, oldest, oldest + std::chrono::microseconds(49), 50));
  assert(detail::stage2_batch_ready(
    7, 32, oldest, oldest + std::chrono::microseconds(50), 50));
}

void test_legacy_low_pressure_flushes_immediately() {
  const Clock::time_point oldest{};
  const auto flush = detail::decide_stage2_packing(
    1, 4, 4, oldest, oldest, 50, 2'000, false);
  assert(flush.ready);
  assert(flush.reason == detail::Stage2PackingFlushReason::low_pressure);
  assert(flush.target_batch == 2);
  assert(flush.pop_limit == 4);
  assert(flush.wait_budget_us == 0);
  assert(!flush.deadline.has_value());
}

void test_legacy_rollback_does_not_cap_context_at_two() {
  const Clock::time_point oldest{};
  const auto waiting = detail::decide_stage2_packing(
    3, 4, 2, oldest, oldest, 50, 1'000, true);
  assert(!waiting.ready);
  assert(waiting.pop_limit == 4);
  assert(waiting.deadline == oldest + std::chrono::microseconds(50));
  const auto flush = detail::decide_stage2_packing(
    3, 4, 2, oldest, oldest + std::chrono::microseconds(50),
    50, 1'000, true);
  assert(flush.ready);
  assert(flush.reason == detail::Stage2PackingFlushReason::deadline);
  assert(flush.pop_limit == 4);
}

void test_trial_wait_is_arrival_adaptive_and_hard_bounded() {
  const Clock::time_point oldest{};
  const auto estimated = detail::decide_stage2_packing(
    2, 4, 4, oldest, oldest, 50, 1'200, true);
  assert(!estimated.ready);
  assert(estimated.wait_budget_us == 2'400);
  assert(estimated.deadline == oldest + std::chrono::microseconds(2'400));

  const auto capped = detail::decide_stage2_packing(
    1, 4, 4, oldest, oldest, 50, 5'000, true);
  assert(capped.wait_budget_us == detail::kStage2AdaptivePackingMaxWaitUs);
  assert(capped.deadline == oldest + std::chrono::microseconds(5'000));

  const auto target = detail::decide_stage2_packing(
    4, 4, 4, oldest, oldest, 50, 5'000, true);
  assert(target.ready);
  assert(target.reason == detail::Stage2PackingFlushReason::full);
  assert(target.pop_limit == 4);
}

void test_visible_backlog_uses_8_16_32_ladder_without_wait() {
  const Clock::time_point oldest{};
  assert(detail::stage2_visible_backlog_target(1, 32) == 8);
  assert(detail::stage2_visible_backlog_target(8, 32) == 8);
  assert(detail::stage2_visible_backlog_target(15, 32) == 8);
  assert(detail::stage2_visible_backlog_target(16, 32) == 16);
  assert(detail::stage2_visible_backlog_target(31, 32) == 16);
  assert(detail::stage2_visible_backlog_target(32, 32) == 32);

  const auto eight = detail::decide_stage2_packing(
    8, 32, 2, oldest, oldest, 50, 5'000, false);
  assert(eight.ready);
  assert(eight.reason == detail::Stage2PackingFlushReason::target);
  assert(eight.target_batch == 8);
  assert(eight.pop_limit == 8);
  assert(eight.wait_budget_us == 0);
  assert(!eight.deadline.has_value());

  const auto sixteen = detail::decide_stage2_packing(
    24, 32, 2, oldest, oldest, 50, 5'000, true);
  assert(sixteen.ready);
  assert(sixteen.reason == detail::Stage2PackingFlushReason::target);
  assert(sixteen.target_batch == 16);
  assert(sixteen.pop_limit == 16);
  assert(sixteen.wait_budget_us == 0);

  const auto thirty_two = detail::decide_stage2_packing(
    32, 32, 2, oldest, oldest, 50, 5'000, true);
  assert(thirty_two.ready);
  assert(thirty_two.reason == detail::Stage2PackingFlushReason::full);
  assert(thirty_two.target_batch == 32);
  assert(thirty_two.pop_limit == 32);
  assert(thirty_two.wait_budget_us == 0);
}

void test_bulk_tail_waits_in_queue_until_eight_or_bounded_deadline() {
  const Clock::time_point oldest{};
  const auto low_pressure = detail::decide_stage2_packing(
    1, 32, 8, oldest, oldest, 50, 2'000, false);
  assert(!low_pressure.ready);
  assert(low_pressure.target_batch == 8);
  assert(low_pressure.pop_limit == 8);
  assert(low_pressure.wait_budget_us == 14'000);
  assert(low_pressure.deadline == oldest + std::chrono::microseconds(14'000));

  const auto waiting = detail::decide_stage2_packing(
    7, 32, 8, oldest, oldest, 50, 2'000, true);
  assert(!waiting.ready);
  assert(waiting.target_batch == 8);
  assert(waiting.pop_limit == 8);
  assert(waiting.wait_budget_us == 14'000);
  assert(waiting.deadline == oldest + std::chrono::microseconds(14'000));

  const auto flush = detail::decide_stage2_packing(
    7, 32, 8, oldest, oldest + std::chrono::microseconds(14'000),
    50, 2'000, true);
  assert(flush.ready);
  assert(flush.reason == detail::Stage2PackingFlushReason::deadline);
  assert(flush.wait_budget_us == 14'000);

  const auto unknown_rate = detail::decide_stage2_packing(
    1, 32, 8, oldest, oldest, 50, 0, false);
  assert(!unknown_rate.ready);
  assert(unknown_rate.wait_budget_us ==
         detail::kStage2BulkTailMaxWaitUs);

  // A steady two-millisecond stream must not double-subtract elapsed time and
  // flush at five to seven items. Every partial observation retains the same
  // oldest-anchored 14 ms deadline; the eighth item is immediately ready.
  for (std::size_t queued = 1; queued < 8; ++queued) {
    const auto now = oldest + std::chrono::milliseconds(
      static_cast<int>(queued - 1) * 2);
    const auto partial = detail::decide_stage2_packing(
      queued, 32, 8, oldest, now, 50, 2'000, true);
    assert(!partial.ready);
    assert(partial.deadline ==
           oldest + std::chrono::microseconds(14'000));
  }
  const auto complete = detail::decide_stage2_packing(
    8, 32, 8, oldest, oldest + std::chrono::milliseconds(14),
    50, 2'000, true);
  assert(complete.ready);
  assert(complete.reason == detail::Stage2PackingFlushReason::target);
}

void establish_legacy_baseline(
    detail::Stage2AdaptivePackingController& controller) {
  for (std::size_t context = 0; context < 256; ++context) {
    controller.observe_completion(
      2, true, 2, 2'000'000, 96, 96);
  }
  assert(controller.parameters().target_batch == 4);
}

void complete_evaluation_window(
    detail::Stage2AdaptivePackingController& controller,
    std::size_t target,
    std::size_t batch,
    std::uint64_t context_cost_ns,
    std::size_t debt_at_completion = 96,
    bool high_pressure = true) {
  assert(detail::kStage2PackingEvaluationTasks % batch == 0);
  for (std::size_t context = 0;
       context < detail::kStage2PackingEvaluationTasks / batch;
       ++context) {
    controller.observe_completion(
      target, high_pressure, batch, context_cost_ns, 96,
      debt_at_completion);
  }
}

void test_feedback_accepts_real_per_task_gain() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(4, 50);
  establish_legacy_baseline(controller);
  for (std::size_t context = 0; context < 128; ++context) {
    controller.observe_completion(
      4, true, 4, 2'400'000, 96, 96);
  }
  const auto telemetry = controller.telemetry();
  assert(telemetry.target_batch == 4);
  assert(telemetry.promotions == 1);
  assert(telemetry.rollbacks == 0);
  assert(telemetry.accepted_trial_windows == 1);
}

void test_validated_four_still_rolls_back_after_workload_drift() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(4, 50);
  establish_legacy_baseline(controller);
  for (std::size_t context = 0; context < 128; ++context) {
    controller.observe_completion(
      4, true, 4, 2'400'000, 96, 96);
  }
  assert(controller.parameters().target_batch == 4);
  for (std::size_t context = 0; context < 128; ++context) {
    controller.observe_completion(
      4, true, 4, 5'000'000, 96, 99);
  }
  const auto telemetry = controller.telemetry();
  assert(telemetry.target_batch == 2);
  assert(telemetry.promotions == 1);
  assert(telemetry.rollbacks == 1);
  assert(controller.parameters().larger_batch_trials_disabled);
}

void test_feedback_rolls_back_negative_trial() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(4, 50);
  establish_legacy_baseline(controller);
  for (std::size_t context = 0; context < 128; ++context) {
    controller.observe_completion(
      4, true, 4, 5'000'000, 96, 98);
  }
  const auto telemetry = controller.telemetry();
  assert(telemetry.target_batch == 2);
  assert(telemetry.promotions == 1);
  assert(telemetry.rollbacks == 1);
  assert(controller.parameters().larger_batch_trials_disabled);
}

void test_zero_wait_disables_adaptive_trial() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(4, 0);
  for (std::size_t context = 0; context < 256; ++context) {
    controller.observe_completion(2, true, 2, 1'000, 0, 0);
  }
  assert(controller.parameters().target_batch == 2);
}

void test_rollback_cooldown_is_bounded_exponential() {
  assert(detail::stage2_packing_rollback_cooldown_tasks(0) == 0);
  assert(detail::stage2_packing_rollback_cooldown_tasks(1) == 2'048);
  assert(detail::stage2_packing_rollback_cooldown_tasks(2) == 4'096);
  assert(detail::stage2_packing_rollback_cooldown_tasks(3) == 8'192);
  assert(detail::stage2_packing_rollback_cooldown_tasks(4) == 16'384);
  assert(detail::stage2_packing_rollback_cooldown_tasks(5) == 32'768);
  assert(detail::stage2_packing_rollback_cooldown_tasks(100) == 32'768);
}

void test_first_negative_trial_permanently_disables_larger_batches() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(4, 50);
  establish_legacy_baseline(controller);

  complete_evaluation_window(controller, 4, 4, 5'000'000, 98);
  auto parameters = controller.parameters();
  assert(parameters.target_batch == 2);
  assert(parameters.consecutive_trial_failures == 1);
  assert(parameters.rollback_cooldown_tasks == 0);
  assert(parameters.larger_batch_trials_disabled);
  const auto telemetry_after_failure = controller.telemetry();
  assert(telemetry_after_failure.promotions == 1);
  assert(telemetry_after_failure.rollbacks == 1);

  // Neither low pressure nor a sustained high-pressure workload may re-arm
  // target four after the first measured negative cohort. Use far more work
  // than the former maximum retry cooldown to prove this is a fuse rather
  // than another backoff interval.
  complete_evaluation_window(
    controller, 2, 2, 2'000'000, 96, false);
  for (std::size_t window = 0; window < 128; ++window) {
    complete_evaluation_window(controller, 2, 2, 2'000'000);
    assert(controller.parameters().target_batch == 2);
  }
  parameters = controller.parameters();
  assert(parameters.target_batch == 2);
  assert(parameters.rollback_cooldown_tasks == 0);
  assert(parameters.larger_batch_trials_disabled);
  const auto telemetry_after_pressure = controller.telemetry();
  assert(telemetry_after_pressure.promotions == 1);
  assert(telemetry_after_pressure.rollbacks == 1);

  // reset() can reconfigure a live controller, but must not clear a
  // process-lifetime negative-result fuse.
  controller.reset(4, 50);
  assert(controller.parameters().larger_batch_trials_disabled);
  for (std::size_t window = 0; window < 4; ++window) {
    complete_evaluation_window(controller, 2, 2, 2'000'000);
  }
  parameters = controller.parameters();
  assert(parameters.target_batch == 2);
  assert(parameters.larger_batch_trials_disabled);
  assert(controller.telemetry().promotions == 0);
}

void test_periodic_legacy_probe_refreshes_stale_baseline() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(4, 50);

  // Initial legacy=100 ns/task and target4=80 ns/task: target four is valid.
  complete_evaluation_window(controller, 2, 2, 200);
  assert(controller.parameters().target_batch == 4);
  for (std::size_t window = 0;
       window < detail::kStage2PackingInitialProbeIntervalWindows;
       ++window) {
    complete_evaluation_window(controller, 4, 4, 320);
  }
  auto parameters = controller.parameters();
  assert(parameters.target_batch == 2);
  assert(parameters.target2_probe_active);
  assert(parameters.rollback_cooldown_tasks == 0);
  assert(controller.telemetry().rollbacks == 0);

  // Environment drift made legacy twice as efficient. Refreshing the probe
  // baseline exposes target four's now-negative 80 vs 50 ns/task result.
  complete_evaluation_window(controller, 2, 2, 100);
  parameters = controller.parameters();
  assert(parameters.target_batch == 4);
  assert(!parameters.target2_probe_active);
  complete_evaluation_window(controller, 4, 4, 320);
  parameters = controller.parameters();
  assert(parameters.target_batch == 2);
  assert(parameters.consecutive_trial_failures == 1);
  assert(parameters.rollback_cooldown_tasks == 0);
  assert(parameters.larger_batch_trials_disabled);
  assert(controller.telemetry().rollbacks == 1);
}

void test_successful_periodic_probe_reduces_probe_tax() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(4, 50);

  complete_evaluation_window(controller, 2, 2, 200);
  for (std::size_t window = 0;
       window < detail::kStage2PackingInitialProbeIntervalWindows;
       ++window) {
    complete_evaluation_window(controller, 4, 4, 320);
  }
  assert(controller.parameters().target2_probe_active);

  // An unchanged refreshed baseline followed by another beneficial
  // target-four cohort doubles the next legacy-probe interval.
  complete_evaluation_window(controller, 2, 2, 200);
  complete_evaluation_window(controller, 4, 4, 320);
  auto parameters = controller.parameters();
  assert(parameters.target_batch == 4);
  assert(parameters.probe_interval_windows == 16);

  // The revalidation cohort is the first accepted window at interval 16.
  for (std::size_t window = 0; window < 14; ++window) {
    complete_evaluation_window(controller, 4, 4, 320);
    assert(!controller.parameters().target2_probe_active);
  }
  complete_evaluation_window(controller, 4, 4, 320);
  parameters = controller.parameters();
  assert(parameters.target_batch == 2);
  assert(parameters.target2_probe_active);
}

void test_bulk_controller_starts_at_eight_and_records_real_batches() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(32, 50);
  assert(controller.parameters().target_batch == 8);

  controller.observe_admission(
    detail::Stage2PackingFlushReason::target,
    8, 0, 0, 8, 12);
  controller.observe_admission(
    detail::Stage2PackingFlushReason::target,
    16, 0, 0, 16, 24);
  controller.observe_admission(
    detail::Stage2PackingFlushReason::full,
    32, 0, 0, 32, 64);
  // Bulk contexts must not be fed into the legacy target2/target4 A/B loop.
  for (std::size_t context = 0; context < 128; ++context) {
    controller.observe_completion(32, true, 32, 1'000'000, 64, 64);
  }

  const auto parameters = controller.parameters();
  const auto telemetry = controller.telemetry();
  assert(parameters.target_batch == 8);
  assert(telemetry.target_batch == 32);
  assert(telemetry.admitted_queue_depth_sum == 100);
  assert(telemetry.admitted_queue_depth_max == 64);
  assert(telemetry.batch_1_to_7 == 0);
  assert(telemetry.batch_8_to_15 == 1);
  assert(telemetry.batch_16_to_31 == 1);
  assert(telemetry.batch_32_plus == 1);
  assert(telemetry.bulk_assembly_batches == 0);
  assert(telemetry.promotions == 0);
  assert(telemetry.rollbacks == 0);
}

}  // namespace

int main() {
  test_legacy_helpers_keep_oldest_deadline();
  test_legacy_low_pressure_flushes_immediately();
  test_legacy_rollback_does_not_cap_context_at_two();
  test_trial_wait_is_arrival_adaptive_and_hard_bounded();
  test_visible_backlog_uses_8_16_32_ladder_without_wait();
  test_bulk_tail_waits_in_queue_until_eight_or_bounded_deadline();
  test_feedback_accepts_real_per_task_gain();
  test_feedback_rolls_back_negative_trial();
  test_validated_four_still_rolls_back_after_workload_drift();
  test_zero_wait_disables_adaptive_trial();
  test_rollback_cooldown_is_bounded_exponential();
  test_first_negative_trial_permanently_disables_larger_batches();
  test_periodic_legacy_probe_refreshes_stale_baseline();
  test_successful_periodic_probe_reduces_probe_tax();
  test_bulk_controller_starts_at_eight_and_records_real_batches();
  return 0;
}
