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

void test_low_pressure_never_waits() {
  const Clock::time_point oldest{};
  const auto decision = detail::decide_stage2_packing(
    1, 32, 4, oldest, oldest, 50, 2'000, false);
  assert(decision.ready);
  assert(decision.reason == detail::Stage2PackingFlushReason::low_pressure);
  assert(decision.target_batch == 2);
  assert(decision.pop_limit == 32);
  assert(!decision.deadline.has_value());
}

void test_legacy_rollback_does_not_cap_context_at_two() {
  const Clock::time_point oldest{};
  const auto waiting = detail::decide_stage2_packing(
    7, 32, 2, oldest, oldest, 50, 1'000, true);
  assert(!waiting.ready);
  assert(waiting.pop_limit == 32);
  assert(waiting.deadline == oldest + std::chrono::microseconds(50));
  const auto flush = detail::decide_stage2_packing(
    7, 32, 2, oldest, oldest + std::chrono::microseconds(50),
    50, 1'000, true);
  assert(flush.ready);
  assert(flush.reason == detail::Stage2PackingFlushReason::deadline);
  assert(flush.pop_limit == 32);
}

void test_trial_wait_is_arrival_adaptive_and_hard_bounded() {
  const Clock::time_point oldest{};
  const auto estimated = detail::decide_stage2_packing(
    2, 32, 4, oldest, oldest, 50, 1'200, true);
  assert(!estimated.ready);
  assert(estimated.wait_budget_us == 2'400);
  assert(estimated.deadline == oldest + std::chrono::microseconds(2'400));

  const auto capped = detail::decide_stage2_packing(
    1, 32, 4, oldest, oldest, 50, 5'000, true);
  assert(capped.wait_budget_us == detail::kStage2AdaptivePackingMaxWaitUs);
  assert(capped.deadline == oldest + std::chrono::microseconds(5'000));

  const auto target = detail::decide_stage2_packing(
    4, 32, 4, oldest, oldest, 50, 5'000, true);
  assert(target.ready);
  assert(target.reason == detail::Stage2PackingFlushReason::target);
  assert(target.pop_limit == 4);
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
  controller.reset(32, 50);
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
  controller.reset(32, 50);
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
}

void test_feedback_rolls_back_negative_trial() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(32, 50);
  establish_legacy_baseline(controller);
  for (std::size_t context = 0; context < 128; ++context) {
    controller.observe_completion(
      4, true, 4, 5'000'000, 96, 98);
  }
  const auto telemetry = controller.telemetry();
  assert(telemetry.target_batch == 2);
  assert(telemetry.promotions == 1);
  assert(telemetry.rollbacks == 1);
}

void test_zero_wait_disables_adaptive_trial() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(32, 0);
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

void test_repeated_failure_backs_off_and_success_resets_streak() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(32, 50);
  establish_legacy_baseline(controller);

  complete_evaluation_window(controller, 4, 4, 5'000'000, 98);
  auto parameters = controller.parameters();
  assert(parameters.target_batch == 2);
  assert(parameters.consecutive_trial_failures == 1);
  assert(parameters.rollback_cooldown_tasks == 2'048);

  // Low-pressure completions neither consume cooldown nor start another
  // experiment.
  complete_evaluation_window(
    controller, 2, 2, 2'000'000, 96, false);
  assert(controller.parameters().rollback_cooldown_tasks == 2'048);

  for (std::size_t window = 0; window < 4; ++window) {
    complete_evaluation_window(controller, 2, 2, 2'000'000);
  }
  parameters = controller.parameters();
  assert(parameters.target_batch == 2);
  assert(parameters.rollback_cooldown_tasks == 0);
  complete_evaluation_window(controller, 2, 2, 2'000'000);
  assert(controller.parameters().target_batch == 4);

  complete_evaluation_window(controller, 4, 4, 5'000'000, 98);
  parameters = controller.parameters();
  assert(parameters.consecutive_trial_failures == 2);
  assert(parameters.rollback_cooldown_tasks == 4'096);

  for (std::size_t window = 0; window < 8; ++window) {
    complete_evaluation_window(controller, 2, 2, 2'000'000);
  }
  complete_evaluation_window(controller, 2, 2, 2'000'000);
  assert(controller.parameters().target_batch == 4);

  // A genuinely beneficial trial clears the failure streak immediately.
  complete_evaluation_window(controller, 4, 4, 2'400'000);
  parameters = controller.parameters();
  assert(parameters.target_batch == 4);
  assert(parameters.consecutive_trial_failures == 0);
  assert(parameters.rollback_cooldown_tasks == 0);

  complete_evaluation_window(controller, 4, 4, 5'000'000, 98);
  parameters = controller.parameters();
  assert(parameters.target_batch == 2);
  assert(parameters.consecutive_trial_failures == 1);
  assert(parameters.rollback_cooldown_tasks == 2'048);
}

void test_periodic_legacy_probe_refreshes_stale_baseline() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(32, 50);

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
  assert(parameters.rollback_cooldown_tasks == 2'048);
  assert(controller.telemetry().rollbacks == 1);
}

void test_successful_periodic_probe_reduces_probe_tax() {
  detail::Stage2AdaptivePackingController controller;
  controller.reset(32, 50);

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

}  // namespace

int main() {
  test_legacy_helpers_keep_oldest_deadline();
  test_low_pressure_never_waits();
  test_legacy_rollback_does_not_cap_context_at_two();
  test_trial_wait_is_arrival_adaptive_and_hard_bounded();
  test_feedback_accepts_real_per_task_gain();
  test_feedback_rolls_back_negative_trial();
  test_validated_four_still_rolls_back_after_workload_drift();
  test_zero_wait_disables_adaptive_trial();
  test_rollback_cooldown_is_bounded_exponential();
  test_repeated_failure_backs_off_and_success_resets_streak();
  test_periodic_legacy_probe_refreshes_stale_baseline();
  test_successful_periodic_probe_reduces_probe_tax();
  return 0;
}
