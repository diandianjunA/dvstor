#include <cassert>

#include "service/compute_service/storage_owner/batch_policy.hh"

namespace detail = compute_service_detail;

namespace {

void test_backlog_lower_bound() {
  // An idle shard has next = durable + 1 and therefore no backlog.
  assert(detail::storage_owner_flow_control_backlog(1, 0, 0) == 0);
  assert(detail::storage_owner_flow_control_backlog(101, 0, 50) == 50);
  // A newer completion target observed by this client reduces the bound.
  assert(detail::storage_owner_flow_control_backlog(101, 80, 50) == 20);
  // A newer durable watermark dominates an older completion target.
  assert(detail::storage_owner_flow_control_backlog(101, 50, 80) == 20);
  // Stale or edge snapshots never underflow.
  assert(detail::storage_owner_flow_control_backlog(50, 80, 80) == 0);
  assert(detail::storage_owner_flow_control_backlog(81, 80, 0) == 0);
  assert(detail::storage_owner_flow_control_backlog(0, 0, 0) == 0);
}

void test_delay_disabled_or_unbounded_inputs() {
  assert(detail::storage_owner_flow_control_delay_ns(
           1'000'000, 0, 0, 20'000'000) == 0);
  assert(detail::storage_owner_flow_control_delay_ns(
           1'000'000, 1'000, 3'000, 0) == 0);
  // An inverted watermark pair is not an enabled controller.
  assert(detail::storage_owner_flow_control_delay_ns(
           1'000'000, 3'000, 1'000, 20'000'000) == 0);
}

void test_delay_watermark_edges_and_interpolation() {
  constexpr u64 kMaxDelay = 20'000'000;  // 20 ms
  assert(detail::storage_owner_flow_control_delay_ns(
           999, 1'000, 3'000, kMaxDelay) == 0);
  assert(detail::storage_owner_flow_control_delay_ns(
           1'000, 1'000, 3'000, kMaxDelay) == 0);
  assert(detail::storage_owner_flow_control_delay_ns(
           3'000, 1'000, 3'000, kMaxDelay) == kMaxDelay);
  assert(detail::storage_owner_flow_control_delay_ns(
           999'999, 1'000, 3'000, kMaxDelay) == kMaxDelay);
  assert(detail::storage_owner_flow_control_delay_ns(
           2'000, 1'000, 3'000, kMaxDelay) == kMaxDelay / 2);
  assert(detail::storage_owner_flow_control_delay_ns(
           1'500, 1'000, 3'000, kMaxDelay) == kMaxDelay / 4);

  u64 previous = 0;
  for (u64 backlog = 1'001; backlog <= 3'000; ++backlog) {
    const u64 delay = detail::storage_owner_flow_control_delay_ns(
      backlog, 1'000, 3'000, kMaxDelay);
    assert(delay >= previous);
    assert(delay <= kMaxDelay);
    previous = delay;
  }
}

}  // namespace

int main() {
  test_backlog_lower_bound();
  test_delay_disabled_or_unbounded_inputs();
  test_delay_watermark_edges_and_interpolation();
  return 0;
}
