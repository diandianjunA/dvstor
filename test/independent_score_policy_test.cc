#include <cassert>
#include <cstddef>
#include <vector>

#include "memory_node/storage_owner_maintenance/independent_score_policy.hh"

namespace {

namespace detail = memory_node_storage_owner_maintenance_detail;

void complete_window(
    detail::IndependentScoreController& controller,
    detail::IndependentScoreMode expected_mode,
    std::uint64_t context_cost_ns,
    std::size_t debt_after = 100,
    std::size_t posted_rpcs = 0,
    std::size_t useful = 0) {
  for (std::size_t context = 0; context < 256; ++context) {
    const auto sample = controller.sample(true);
    assert(sample.mode == expected_mode);
    assert(sample.eligible);
    assert(sample.registration_id != 0);
    controller.observe_completion(
      sample, 2, context_cost_ns, 100, debt_after,
      posted_rpcs, useful);
  }
}

void establish_enabled(detail::IndependentScoreController& controller) {
  controller.reset();
  complete_window(
    controller, detail::IndependentScoreMode::baseline, 200);
  complete_window(
    controller, detail::IndependentScoreMode::trial,
    180, 100, 1, 1);
  complete_window(
    controller, detail::IndependentScoreMode::confirmation, 200);
  const auto telemetry = controller.telemetry();
  assert(telemetry.mode == detail::IndependentScoreMode::enabled);
  assert(telemetry.trials_started == 1);
  assert(telemetry.confirmations == 1);
  assert(telemetry.trials_accepted == 1);
}

void test_ineligible_contexts_do_not_register_or_advance() {
  detail::IndependentScoreController controller;
  controller.reset();
  for (std::size_t context = 0; context < 256; ++context) {
    const auto sample = controller.sample(false);
    assert(!sample.eligible);
    assert(sample.registration_id == 0);
    controller.observe_completion(sample, 2, 1, 0, 0, 0, 0);
  }
  const auto telemetry = controller.telemetry();
  assert(telemetry.mode == detail::IndependentScoreMode::baseline);
  assert(telemetry.window_tasks == 0);
}

void test_bracketed_five_percent_gain_enables() {
  detail::IndependentScoreController controller;
  establish_enabled(controller);
  assert(controller.telemetry().rollbacks == 0);
  assert(controller.sample(true).allows_speculation());
}

void test_cold_baseline_natural_speedup_is_rejected_by_confirmation() {
  detail::IndependentScoreController controller;
  controller.reset();
  // Natural 150 -> 120 -> 100 ns/task drift looks favorable against only the
  // cold baseline, but fails the trailing contemporaneous control.
  complete_window(
    controller, detail::IndependentScoreMode::baseline, 300);
  complete_window(
    controller, detail::IndependentScoreMode::trial,
    240, 100, 1, 1);
  complete_window(
    controller, detail::IndependentScoreMode::confirmation, 200);
  const auto telemetry = controller.telemetry();
  assert(telemetry.mode == detail::IndependentScoreMode::disabled);
  assert(telemetry.rollbacks == 1);
  assert(telemetry.trials_accepted == 0);
}

void test_all_wasted_or_worse_debt_is_rejected() {
  {
    detail::IndependentScoreController controller;
    controller.reset();
    complete_window(
      controller, detail::IndependentScoreMode::baseline, 200);
    complete_window(
      controller, detail::IndependentScoreMode::trial,
      180, 100, 1, 0);
    assert(controller.telemetry().mode ==
           detail::IndependentScoreMode::disabled);
  }
  {
    detail::IndependentScoreController controller;
    controller.reset();
    complete_window(
      controller, detail::IndependentScoreMode::baseline, 200);
    complete_window(
      controller, detail::IndependentScoreMode::trial,
      180, 101, 1, 1);
    assert(controller.telemetry().mode ==
           detail::IndependentScoreMode::disabled);
  }
}

void advance_to_revalidation_control(
    detail::IndependentScoreController& controller) {
  for (std::size_t window = 0;
       window < detail::kIndependentScoreRevalidationIntervalWindows;
       ++window) {
    complete_window(
      controller, detail::IndependentScoreMode::enabled,
      180, 100, 1, 1);
  }
  assert(controller.telemetry().mode ==
         detail::IndependentScoreMode::revalidation_control);
}

void test_enabled_window_has_immediate_absolute_stop_loss() {
  detail::IndependentScoreController controller;
  establish_enabled(controller);
  complete_window(
    controller, detail::IndependentScoreMode::enabled,
    202, 100, 1, 1);
  const auto telemetry = controller.telemetry();
  assert(telemetry.mode == detail::IndependentScoreMode::disabled);
  assert(telemetry.rollbacks == 1);
}

void test_periodic_fresh_control_detects_drift_and_fuses() {
  detail::IndependentScoreController controller;
  establish_enabled(controller);
  advance_to_revalidation_control(controller);
  // Current no-spec is 80 ns/task; spec is 77 ns/task. Both remain below the
  // old control, but the fresh benefit is only 3.75%.
  complete_window(
    controller, detail::IndependentScoreMode::revalidation_control, 160);
  complete_window(
    controller, detail::IndependentScoreMode::revalidation_trial,
    154, 100, 1, 1);
  const auto telemetry = controller.telemetry();
  assert(telemetry.mode == detail::IndependentScoreMode::disabled);
  assert(telemetry.revalidation_controls == 1);
  assert(telemetry.revalidations_accepted == 0);
  assert(telemetry.rollbacks == 1);
}

void test_periodic_control_accepts_current_gain() {
  detail::IndependentScoreController controller;
  establish_enabled(controller);
  advance_to_revalidation_control(controller);
  complete_window(
    controller, detail::IndependentScoreMode::revalidation_control, 200);
  complete_window(
    controller, detail::IndependentScoreMode::revalidation_trial,
    180, 100, 1, 1);
  const auto telemetry = controller.telemetry();
  assert(telemetry.mode == detail::IndependentScoreMode::enabled);
  assert(telemetry.revalidations_accepted == 1);
  assert(telemetry.rollbacks == 0);
}

void test_spec_to_control_waits_for_every_old_context() {
  detail::IndependentScoreController controller;
  controller.reset();
  complete_window(
    controller, detail::IndependentScoreMode::baseline, 200);

  std::vector<detail::IndependentScoreSample> old_spec;
  for (std::size_t index = 0; index < 4; ++index) {
    old_spec.push_back(controller.sample(true));
    assert(old_spec.back().mode == detail::IndependentScoreMode::trial);
    assert(old_spec.back().allows_speculation());
  }
  complete_window(
    controller, detail::IndependentScoreMode::trial,
    180, 100, 1, 1);
  auto telemetry = controller.telemetry();
  assert(telemetry.mode ==
         detail::IndependentScoreMode::confirmation_drain);
  assert(telemetry.drain_outstanding == 4);

  // Arbitrarily many newly admitted contexts during washout execute no spec,
  // own no registration, and cannot accidentally form the control cohort.
  for (std::size_t index = 0; index < 64; ++index) {
    const auto washout = controller.sample(true);
    assert(washout.mode ==
           detail::IndependentScoreMode::confirmation_drain);
    assert(!washout.eligible);
    assert(!washout.allows_speculation());
    assert(washout.registration_id == 0);
    controller.observe_completion(washout, 2, 1, 100, 0, 0, 0);
  }
  assert(controller.telemetry().window_tasks == 0);

  for (std::size_t index = 0; index + 1 < old_spec.size(); ++index) {
    controller.observe_completion(
      old_spec[index], 2, 180, 100, 100, 1, 1);
    telemetry = controller.telemetry();
    assert(telemetry.mode ==
           detail::IndependentScoreMode::confirmation_drain);
    assert(telemetry.drain_outstanding == old_spec.size() - index - 1);
  }
  const auto last = old_spec.back();
  // A context whose every task became stale still releases its admission
  // token and must not wedge drain forever.
  controller.observe_completion(last, 0, 0, 0, 0, 0, 0);
  telemetry = controller.telemetry();
  assert(telemetry.mode == detail::IndependentScoreMode::confirmation);
  assert(telemetry.drain_outstanding == 0);
  assert(telemetry.window_tasks == 0);

  // Duplicate release cannot advance or contaminate the fresh control.
  controller.observe_completion(last, 512, 1, 100, 0, 1, 1);
  assert(controller.telemetry().window_tasks == 0);
  const auto control = controller.sample(true);
  assert(control.mode == detail::IndependentScoreMode::confirmation);
  assert(control.registration_id != 0);
  controller.observe_completion(control, 2, 200, 100, 100, 0, 0);
  assert(controller.telemetry().window_tasks == 2);
}

void test_enabled_to_revalidation_control_also_drains_exactly() {
  detail::IndependentScoreController controller;
  establish_enabled(controller);
  std::vector<detail::IndependentScoreSample> old_enabled;
  for (std::size_t index = 0; index < 3; ++index) {
    old_enabled.push_back(controller.sample(true));
    assert(old_enabled.back().mode == detail::IndependentScoreMode::enabled);
  }
  for (std::size_t window = 0;
       window < detail::kIndependentScoreRevalidationIntervalWindows;
       ++window) {
    complete_window(
      controller, detail::IndependentScoreMode::enabled,
      180, 100, 1, 1);
  }
  auto telemetry = controller.telemetry();
  assert(telemetry.mode ==
         detail::IndependentScoreMode::revalidation_control_drain);
  assert(telemetry.drain_outstanding == 3);
  for (const auto& sample : old_enabled) {
    controller.observe_completion(sample, 2, 180, 100, 100, 1, 1);
  }
  telemetry = controller.telemetry();
  assert(telemetry.mode ==
         detail::IndependentScoreMode::revalidation_control);
  assert(telemetry.drain_outstanding == 0);
  assert(telemetry.window_tasks == 0);
}

void test_late_non_spec_generation_is_excluded_from_trial() {
  detail::IndependentScoreController controller;
  controller.reset();
  const auto old_baseline = controller.sample(true);
  complete_window(
    controller, detail::IndependentScoreMode::baseline, 200);
  assert(controller.telemetry().mode == detail::IndependentScoreMode::trial);
  controller.observe_completion(
    old_baseline, 512, 1, 100, 0, 0, 0);
  assert(controller.telemetry().window_tasks == 0);
}

void test_reset_invalidates_old_tokens_and_preserves_failure_fuse() {
  {
    detail::IndependentScoreController controller;
    controller.reset();
    const auto old = controller.sample(true);
    controller.reset();
    controller.observe_completion(old, 512, 1, 100, 0, 0, 0);
    assert(controller.telemetry().window_tasks == 0);
    assert(controller.telemetry().mode ==
           detail::IndependentScoreMode::baseline);
  }
  {
    detail::IndependentScoreController controller;
    controller.reset();
    complete_window(
      controller, detail::IndependentScoreMode::baseline, 200);
    complete_window(
      controller, detail::IndependentScoreMode::trial,
      180, 100, 1, 0);
    controller.reset();
    assert(controller.telemetry().mode ==
           detail::IndependentScoreMode::disabled);
    assert(!controller.sample(true).allows_speculation());
  }
}

}  // namespace

int main() {
  test_ineligible_contexts_do_not_register_or_advance();
  test_bracketed_five_percent_gain_enables();
  test_cold_baseline_natural_speedup_is_rejected_by_confirmation();
  test_all_wasted_or_worse_debt_is_rejected();
  test_enabled_window_has_immediate_absolute_stop_loss();
  test_periodic_fresh_control_detects_drift_and_fuses();
  test_periodic_control_accepts_current_gain();
  test_spec_to_control_waits_for_every_old_context();
  test_enabled_to_revalidation_control_also_drains_exactly();
  test_late_non_spec_generation_is_excluded_from_trial();
  test_reset_invalidates_old_tokens_and_preserves_failure_fuse();
  return 0;
}
