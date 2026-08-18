#include "memory_node/storage_owner_maintenance/stage2_batch_policy.hh"

#include <cassert>
#include <chrono>

namespace detail = memory_node_storage_owner_maintenance_detail;
using namespace std::chrono_literals;

namespace {

void test_deterministic_b8_target_and_execution_slice() {
  assert(detail::stage2_visible_backlog_target(32) == 8);
  assert(detail::stage2_visible_backlog_target(8) == 8);
  assert(detail::stage2_visible_backlog_target(4) == 4);
  assert(detail::stage2_visible_backlog_target(0) == 1);
  assert(detail::stage2_execution_slice_limit(32, 32) == 8);
  assert(detail::stage2_execution_slice_limit(4, 32) == 4);
}

void test_complete_b8_is_immediately_runnable() {
  const auto now = std::chrono::steady_clock::now();
  const auto decision = detail::decide_stage2_packing(
    32, 32, now, now, 25'000, 2'000);
  assert(decision.ready);
  assert(decision.pop_limit == 8);
  assert(decision.target_batch == 8);
  assert(decision.wait_budget_us == 0);
  assert(decision.reason == detail::Stage2PackingFlushReason::target);
}

void test_sub_b8_tail_uses_arrival_prediction() {
  const auto oldest = std::chrono::steady_clock::now();
  const auto waiting = detail::decide_stage2_packing(
    3, 32, oldest, oldest + 13ms, 25'000, 2'000);
  assert(!waiting.ready);
  assert(waiting.target_batch == 8);
  assert(waiting.pop_limit == 8);
  assert(waiting.wait_budget_us == 14'000);
  assert(waiting.deadline == oldest + 14ms);

  const auto ready = detail::decide_stage2_packing(
    3, 32, oldest, oldest + 14ms, 25'000, 2'000);
  assert(ready.ready);
  assert(ready.reason == detail::Stage2PackingFlushReason::deadline);
}

void test_configured_wait_is_authoritative_upper_bound() {
  const auto oldest = std::chrono::steady_clock::now();

  const auto capped = detail::decide_stage2_packing(
    1, 32, oldest, oldest, 5'000, 10'000);
  assert(!capped.ready);
  assert(capped.wait_budget_us == 5'000);
  assert(capped.deadline == oldest + 5ms);

  const auto below_prediction_floor = detail::decide_stage2_packing(
    1, 32, oldest, oldest, 50, 1);
  assert(below_prediction_floor.wait_budget_us == 50);
  assert(below_prediction_floor.deadline == oldest + 50us);

  const auto immediate = detail::decide_stage2_packing(
    1, 32, oldest, oldest, 0, 2'000);
  assert(immediate.ready);
  assert(immediate.wait_budget_us == 0);
  assert(immediate.reason == detail::Stage2PackingFlushReason::deadline);

  const auto unknown_rate = detail::decide_stage2_packing(
    1, 32, oldest, oldest, 10'000, 0);
  assert(unknown_rate.wait_budget_us == 10'000);
}

void test_enqueue_ewma_and_telemetry() {
  detail::Stage2PackingController controller;
  controller.reset(32);
  const auto first = std::chrono::steady_clock::now();
  controller.observe_enqueue(first, 1);
  controller.observe_enqueue(first + 8ms, 4);
  assert(controller.parameters().estimated_arrival_interval_us == 2'000);

  controller.observe_admission(
    detail::Stage2PackingFlushReason::target, 8, 2'000'000, 0, 8, 16);
  controller.observe_admission(
    detail::Stage2PackingFlushReason::deadline, 3, 5'000'000, 5'000, 8, 3);
  controller.observe_admission(
    detail::Stage2PackingFlushReason::cleanup, 0, 0, 0);
  const auto telemetry = controller.telemetry();
  assert(telemetry.target_batch == 8);
  assert(telemetry.target_flushes == 1);
  assert(telemetry.deadline_flushes == 1);
  assert(telemetry.cleanup_flushes == 1);
  assert(telemetry.batch_8 == 1);
  assert(telemetry.batch_1_to_7 == 1);
  assert(telemetry.waited_batches == 1);
  assert(telemetry.wait_ns == 5'000'000);
  assert(telemetry.admitted_queue_depth_sum == 19);
  assert(telemetry.admitted_queue_depth_max == 16);
}

}  // namespace

int main() {
  test_deterministic_b8_target_and_execution_slice();
  test_complete_b8_is_immediately_runnable();
  test_sub_b8_tail_uses_arrival_prediction();
  test_configured_wait_is_authoritative_upper_bound();
  test_enqueue_ewma_and_telemetry();
  return 0;
}
