#include <cassert>
#include <chrono>

#include "memory_node/storage_owner_maintenance/fallback_audit_policy.hh"

namespace detail = memory_node_storage_owner_maintenance_detail;

namespace {

using TestClock = std::chrono::steady_clock;
using namespace std::chrono_literals;

void test_interval_tracks_local_context_activity() {
  assert(detail::stage2_fallback_audit_interval(0) == 10ms);
  assert(detail::stage2_fallback_audit_interval(1) == 1ms);
  assert(detail::stage2_fallback_audit_interval(48) == 1ms);
}

void test_first_context_clamps_idle_deadline() {
  const TestClock::time_point now{100ms};
  const auto idle_deadline = now + 10ms;
  assert(detail::refresh_stage2_fallback_audit_deadline(
           idle_deadline, false, 1, now) == now + 1ms);

  // An already-earlier deadline remains earlier; admission cannot postpone a
  // pending audit while changing to the active cadence.
  assert(detail::refresh_stage2_fallback_audit_deadline(
           now + 500us, false, 1, now) == now + 500us);
}

void test_last_context_restores_idle_deadline() {
  const TestClock::time_point now{200ms};
  assert(detail::refresh_stage2_fallback_audit_deadline(
           now + 1ms, true, 0, now) == now + 10ms);
}

void test_unchanged_state_preserves_periodic_schedule() {
  const TestClock::time_point now{300ms};
  const auto active_deadline = now + 250us;
  const auto idle_deadline = now + 7ms;
  assert(detail::refresh_stage2_fallback_audit_deadline(
           active_deadline, true, 3, now) == active_deadline);
  assert(detail::refresh_stage2_fallback_audit_deadline(
           idle_deadline, false, 0, now) == idle_deadline);
}

}  // namespace

int main() {
  test_interval_tracks_local_context_activity();
  test_first_context_clamps_idle_deadline();
  test_last_context_restores_idle_deadline();
  test_unchanged_state_preserves_periodic_schedule();
  return 0;
}
