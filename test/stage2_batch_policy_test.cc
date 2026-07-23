#include <cassert>
#include <chrono>

#include "memory_node/storage_owner_maintenance/stage2_batch_policy.hh"

namespace {

using memory_node_storage_owner_maintenance_detail::stage2_batch_ready;
using memory_node_storage_owner_maintenance_detail::
  stage2_partial_batch_deadline;
using Clock = std::chrono::steady_clock;

void test_empty_queue_is_not_ready() {
  const Clock::time_point epoch{};
  assert(!stage2_batch_ready(0, 32, epoch, epoch, 100));
}

void test_full_batch_runs_without_waiting() {
  const Clock::time_point epoch{};
  assert(stage2_batch_ready(32, 32, epoch, epoch, 100));
}

void test_partial_batch_uses_oldest_task_deadline() {
  const Clock::time_point oldest{};
  const auto deadline = stage2_partial_batch_deadline(
    7, 32, oldest, 100);
  assert(deadline.has_value());
  assert(*deadline == oldest + std::chrono::microseconds(100));
  assert(!stage2_batch_ready(
    7, 32, oldest, oldest + std::chrono::microseconds(99), 100));
  assert(stage2_batch_ready(
    7, 32, oldest, oldest + std::chrono::microseconds(100), 100));

  // Later arrivals change only the queue size. They cannot re-arm or extend
  // the deadline supplied by the oldest descriptor.
  assert(stage2_batch_ready(
    15, 32, oldest, oldest + std::chrono::microseconds(100), 100));
  assert(stage2_partial_batch_deadline(15, 32, oldest, 100) == deadline);
}

void test_zero_wait_runs_partial_batch_immediately() {
  const Clock::time_point epoch{};
  assert(stage2_batch_ready(1, 32, epoch, epoch, 0));
  assert(!stage2_partial_batch_deadline(1, 32, epoch, 0).has_value());
}

}  // namespace

int main() {
  test_empty_queue_is_not_ready();
  test_full_batch_runs_without_waiting();
  test_partial_batch_uses_oldest_task_deadline();
  test_zero_wait_runs_partial_batch_immediately();
  return 0;
}
