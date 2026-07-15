#include <cassert>
#include <cstdint>

#include "memory_node/storage_owner_maintenance/stage2_tracker.hh"

namespace detail = memory_node_storage_owner_maintenance_detail;

namespace {

constexpr std::uint64_t peer(std::uint32_t index) {
  return std::uint64_t{1} << index;
}

void test_context_capacity_and_state_transitions() {
  detail::Stage2StateTracker states(2, 5);
  const auto first = states.try_acquire();
  const auto second = states.try_acquire();
  assert(first.has_value());
  assert(second.has_value());
  assert(states.full());
  assert(!states.try_acquire().has_value());

  auto snapshot = states.snapshot(*first);
  assert(snapshot.has_value());
  assert(snapshot->phase == detail::Stage2Phase::local_ready);

  const std::uint64_t search_peers = peer(1) | peer(3) | peer(4);
  assert(states.begin_remote_search(*first, search_peers) ==
         detail::Stage2EventResult::phase_advanced);
  assert(states.record_remote_search_response(*first, 3) ==
         detail::Stage2EventResult::accepted);
  assert(states.record_remote_search_response(*first, 1) ==
         detail::Stage2EventResult::accepted);
  assert(states.record_remote_search_response(*first, 3) ==
         detail::Stage2EventResult::duplicate);
  assert(states.record_remote_search_response(*first, 2) ==
         detail::Stage2EventResult::unexpected_peer);
  assert(states.record_remote_search_response(*first, 4) ==
         detail::Stage2EventResult::phase_advanced);

  snapshot = states.snapshot(*first);
  assert(snapshot->phase == detail::Stage2Phase::prune_ready);
  assert(snapshot->completed_search_mask == search_peers);

  const std::uint64_t reverse_peers = peer(0) | peer(2) | peer(4);
  assert(states.begin_reverse(*first, reverse_peers) ==
         detail::Stage2EventResult::phase_advanced);
  assert(states.record_reverse_ack(*first, 4) ==
         detail::Stage2EventResult::accepted);
  assert(states.record_reverse_ack(*first, 0) ==
         detail::Stage2EventResult::accepted);
  assert(states.record_reverse_ack(*first, 4) ==
         detail::Stage2EventResult::duplicate);
  assert(states.finalize(*first) == detail::Stage2EventResult::incomplete);
  assert(states.record_reverse_ack(*first, 2) ==
         detail::Stage2EventResult::ready_to_finalize);

  snapshot = states.snapshot(*first);
  assert(snapshot->phase == detail::Stage2Phase::reverse_pending);
  assert(snapshot->completed_reverse_mask == reverse_peers);
  assert(states.finalize(*first) ==
         detail::Stage2EventResult::phase_advanced);
  snapshot = states.snapshot(*first);
  assert(snapshot->phase == detail::Stage2Phase::finalized);

  const detail::Stage2ContextHandle old_handle = *first;
  assert(states.release(old_handle));
  const auto reused = states.try_acquire();
  assert(reused.has_value());
  assert(reused->slot == old_handle.slot);
  assert(reused->generation != old_handle.generation);
  assert(states.record_remote_search_response(old_handle, 1) ==
         detail::Stage2EventResult::stale_context);
  assert(!states.snapshot(old_handle).has_value());
}

void test_request_retry_duplicates_and_reverse_fan_in() {
  detail::Stage2StateTracker states(1, 4);
  const auto context = states.try_acquire();
  assert(context.has_value());
  assert(states.begin_remote_search(*context, peer(0) | peer(2)) ==
         detail::Stage2EventResult::phase_advanced);

  detail::Stage2RequestTracker requests(2);
  assert(requests.try_register(100, *context,
                               detail::Stage2RequestKind::remote_search, 0,
                               5, 10, states) ==
         detail::Stage2RequestRegisterResult::registered);
  assert(requests.try_register(101, *context,
                               detail::Stage2RequestKind::remote_search, 2,
                               5, 10, states) ==
         detail::Stage2RequestRegisterResult::registered);
  assert(!requests.retry_due(100, 9));
  assert(requests.retry_due(100, 10));

  const auto retried = requests.mark_retry(100, 12, 20);
  assert(retried.has_value());
  assert(retried->request_id == 100);
  assert(retried->attempt_count == 2);
  assert(retried->last_send_time == 12);
  assert(retried->deadline == 20);
  assert(!requests.retry_due(100, 19));
  assert(requests.retry_due(100, 20));

  // Responses can arrive in any peer order, and a duplicate request response
  // must not advance the peer mask twice.
  assert(requests.record_response(101, states) ==
         detail::Stage2EventResult::accepted);
  assert(requests.record_response(101, states) ==
         detail::Stage2EventResult::duplicate);
  assert(requests.record_response(100, states) ==
         detail::Stage2EventResult::phase_advanced);
  assert(!requests.mark_retry(100, 21, 30).has_value());

  assert(requests.erase(100));
  assert(requests.erase(101));
  assert(states.begin_reverse(*context, peer(1) | peer(3)) ==
         detail::Stage2EventResult::phase_advanced);
  assert(requests.try_register(200, *context,
                               detail::Stage2RequestKind::reverse_update, 1,
                               30, 40, states) ==
         detail::Stage2RequestRegisterResult::registered);
  assert(requests.try_register(201, *context,
                               detail::Stage2RequestKind::reverse_update, 3,
                               30, 40, states) ==
         detail::Stage2RequestRegisterResult::registered);

  assert(requests.record_response(201, states) ==
         detail::Stage2EventResult::accepted);
  assert(states.finalize(*context) == detail::Stage2EventResult::incomplete);
  assert(requests.record_response(200, states) ==
         detail::Stage2EventResult::ready_to_finalize);
  assert(states.finalize(*context) ==
         detail::Stage2EventResult::phase_advanced);

  // Keep the completed request record alive across slot reuse. Generation
  // validation must win over its response_seen flag.
  const detail::Stage2ContextHandle old_context = *context;
  assert(states.release(old_context));
  const auto reused = states.try_acquire();
  assert(reused.has_value());
  assert(reused->slot == old_context.slot);
  assert(reused->generation != old_context.generation);
  assert(requests.record_response(200, states) ==
         detail::Stage2EventResult::stale_context);
}

void test_request_capacity_and_validation() {
  detail::Stage2StateTracker states(1, 2);
  const auto context = states.try_acquire();
  assert(context.has_value());
  assert(states.begin_remote_search(*context, peer(0) | peer(1)) ==
         detail::Stage2EventResult::phase_advanced);

  detail::Stage2RequestTracker requests(1);
  assert(requests.try_register(7, *context,
                               detail::Stage2RequestKind::remote_search, 0,
                               0, 1, states) ==
         detail::Stage2RequestRegisterResult::registered);
  assert(requests.full());
  assert(requests.try_register(7, *context,
                               detail::Stage2RequestKind::remote_search, 0,
                               0, 1, states) ==
         detail::Stage2RequestRegisterResult::duplicate_request_id);
  assert(requests.try_register(8, *context,
                               detail::Stage2RequestKind::remote_search, 1,
                               0, 1, states) ==
         detail::Stage2RequestRegisterResult::capacity_exhausted);
  assert(requests.record_response(999, states) ==
         detail::Stage2EventResult::unknown_request);

  assert(requests.erase(7));
  assert(requests.try_register(8, *context,
                               detail::Stage2RequestKind::remote_search, 1,
                               1, 2, states) ==
         detail::Stage2RequestRegisterResult::registered);
  assert(requests.size() == 1);

  const detail::Stage2ContextHandle fabricated{
    context->slot, static_cast<std::uint32_t>(context->generation + 1)};
  assert(requests.try_register(9, fabricated,
                               detail::Stage2RequestKind::remote_search, 0,
                               1, 2, states) ==
         detail::Stage2RequestRegisterResult::stale_context);
  assert(states.begin_remote_search(*context, peer(0)) ==
         detail::Stage2EventResult::invalid_phase);
  assert(states.begin_reverse(*context, peer(0)) ==
         detail::Stage2EventResult::invalid_phase);
}

void test_zero_peer_fast_paths_and_mask_validation() {
  detail::Stage2StateTracker states(1, 3);
  const auto context = states.try_acquire();
  assert(context.has_value());
  assert(states.begin_remote_search(*context, peer(3)) ==
         detail::Stage2EventResult::invalid_peer_mask);
  assert(states.begin_remote_search(*context, 0) ==
         detail::Stage2EventResult::phase_advanced);
  assert(states.snapshot(*context)->phase == detail::Stage2Phase::prune_ready);
  assert(states.begin_reverse(*context, 0) ==
         detail::Stage2EventResult::ready_to_finalize);
  assert(states.finalize(*context) ==
         detail::Stage2EventResult::phase_advanced);
}

}  // namespace

int main() {
  test_context_capacity_and_state_transitions();
  test_request_retry_duplicates_and_reverse_fan_in();
  test_request_capacity_and_validation();
  test_zero_peer_fast_paths_and_mask_validation();
  return 0;
}
