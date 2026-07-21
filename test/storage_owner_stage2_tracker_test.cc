#include <cassert>
#include <chrono>
#include <cstdint>
#include <vector>

#include "memory_node/storage_owner_maintenance/cleanup_scheduler.hh"
#include "memory_node/storage_owner_maintenance/stage2_tracker.hh"

namespace detail = memory_node_storage_owner_maintenance_detail;

namespace {

constexpr std::uint64_t peer(std::uint32_t index) {
  return std::uint64_t{1} << index;
}

constexpr std::uint64_t request_hash(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

struct CleanupTask {
  std::uint64_t maintenance_sequence{};
  std::chrono::steady_clock::time_point retry_not_before{};
  std::chrono::steady_clock::time_point queued_at{};
  int value{};
};

void test_cleanup_scheduler_is_ordered_and_retry_fenced() {
  const auto epoch = std::chrono::steady_clock::time_point{};
  std::vector<CleanupTask> heap;
  detail::cleanup_schedule_push(heap, CleanupTask{
    .maintenance_sequence = 9,
    .retry_not_before = epoch,
    .queued_at = epoch + std::chrono::nanoseconds(9),
    .value = 9,
  });
  detail::cleanup_schedule_push(heap, CleanupTask{
    .maintenance_sequence = 3,
    .retry_not_before = epoch + std::chrono::milliseconds(5),
    .queued_at = epoch + std::chrono::nanoseconds(3),
    .value = 3,
  });
  detail::cleanup_schedule_push(heap, CleanupTask{
    .maintenance_sequence = 7,
    .retry_not_before = epoch,
    .queued_at = epoch + std::chrono::nanoseconds(7),
    .value = 7,
  });
  assert(detail::cleanup_schedule_valid(heap));
  assert(heap.front().maintenance_sequence == 3);
  assert(!detail::cleanup_schedule_ready(
    heap.front(), 1, epoch + std::chrono::milliseconds(10)));
  assert(!detail::cleanup_schedule_ready(
    heap.front(), 2, epoch + std::chrono::milliseconds(4)));
  assert(detail::cleanup_schedule_ready(
    heap.front(), 2, epoch + std::chrono::milliseconds(5)));

  CleanupTask first = detail::cleanup_schedule_pop(heap);
  assert(first.value == 3);
  assert(detail::cleanup_schedule_valid(heap));
  assert(heap.front().maintenance_sequence == 7);
  // Completing sequence 3 does not make sequence 7 runnable; the completion
  // watermark must first cross every intervening maintenance sequence.
  assert(!detail::cleanup_schedule_ready(heap.front(), 3, epoch));
  assert(detail::cleanup_schedule_ready(heap.front(), 6, epoch));
}

void test_cleanup_scheduler_never_bypasses_delayed_predecessor() {
  const auto epoch = std::chrono::steady_clock::time_point{};
  std::vector<CleanupTask> heap;
  detail::cleanup_schedule_push(heap, CleanupTask{
    .maintenance_sequence = 5,
    .retry_not_before = epoch + std::chrono::milliseconds(10),
    .queued_at = epoch,
    .value = 5,
  });
  detail::cleanup_schedule_push(heap, CleanupTask{
    .maintenance_sequence = 6,
    .retry_not_before = epoch,
    .queued_at = epoch + std::chrono::nanoseconds(1),
    .value = 6,
  });
  assert(heap.front().value == 5);
  assert(!detail::cleanup_schedule_ready(
    heap.front(), 4, epoch + std::chrono::milliseconds(9)));
  assert(detail::cleanup_schedule_ready(
    heap.front(), 4, epoch + std::chrono::milliseconds(10)));
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

void test_retryable_cleanup_release_cannot_pin_the_only_context() {
  detail::Stage2StateTracker states(1, 2);
  const auto local = states.try_acquire();
  assert(local.has_value());
  assert(states.release_retryable(*local));
  assert(!states.snapshot(*local).has_value());

  const auto prune = states.try_acquire();
  assert(prune.has_value());
  assert(states.begin_remote_search(*prune, 0) ==
         detail::Stage2EventResult::phase_advanced);
  assert(states.release_retryable(*prune));

  const auto asynchronous = states.try_acquire();
  assert(asynchronous.has_value());
  assert(states.begin_remote_search(*asynchronous, peer(1)) ==
         detail::Stage2EventResult::phase_advanced);
  assert(!states.release_retryable(*asynchronous));
}

void test_request_tracker_true_deletion_survives_long_churn() {
  constexpr std::size_t capacity = 32;
  constexpr std::size_t iterations = capacity * 12;
  detail::Stage2StateTracker states(1, 1);
  const auto context = states.try_acquire();
  assert(context.has_value());
  assert(states.begin_remote_search(*context, peer(0)) ==
         detail::Stage2EventResult::phase_advanced);
  detail::Stage2RequestTracker requests(capacity);

  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    const std::uint64_t request_id = 1000 + iteration * 17;
    assert(requests.size() == 0);
    assert(requests.lookup_probe_count(request_id) == 1);
    assert(requests.try_register(
             request_id, *context,
             detail::Stage2RequestKind::remote_search, 0,
             iteration, iteration + 1, states) ==
           detail::Stage2RequestRegisterResult::registered);
    assert(requests.lookup_probe_count(request_id) == 1);
    const auto metadata = requests.find(request_id);
    assert(metadata.has_value());
    assert(metadata->last_send_time == iteration);
    assert(requests.erase(request_id));
    assert(!requests.find(request_id).has_value());
    assert(requests.lookup_probe_count(request_id) == 1);
  }

  // Historical traffic exceeded both 10C records and the 2C bucket array,
  // yet a fresh operation still begins at an empty home bucket.
  constexpr std::uint64_t fresh = 0xfedcba9876543210ULL;
  assert(requests.lookup_probe_count(fresh) == 1);
  assert(requests.try_register(
           fresh, *context, detail::Stage2RequestKind::remote_search, 0,
           500, 600, states) ==
         detail::Stage2RequestRegisterResult::registered);
  assert(requests.lookup_probe_count(fresh) == 1);
}

void test_backward_shift_preserves_colliding_record_indices() {
  constexpr std::size_t capacity = 8;
  constexpr std::size_t bucket_count = 16;
  constexpr std::size_t bucket_mask = bucket_count - 1;
  detail::Stage2StateTracker states(1, 1);
  const auto context = states.try_acquire();
  assert(context.has_value());
  assert(states.begin_remote_search(*context, peer(0)) ==
         detail::Stage2EventResult::phase_advanced);
  detail::Stage2RequestTracker requests(capacity);

  std::vector<std::uint64_t> colliding;
  for (std::uint64_t candidate = 1; colliding.size() != 5; ++candidate) {
    if ((request_hash(candidate) & bucket_mask) == 14) {
      colliding.push_back(candidate);
    }
  }
  for (std::size_t index = 0; index < colliding.size(); ++index) {
    assert(requests.try_register(
             colliding[index], *context,
             detail::Stage2RequestKind::remote_search, 0,
             100 + index, 200 + index, states) ==
           detail::Stage2RequestRegisterResult::registered);
    assert(requests.lookup_probe_count(colliding[index]) == index + 1);
  }

  // Delete from the middle of a wrapped/linear cluster. Every survivor must
  // still resolve to its original slab metadata after buckets shift backward.
  assert(requests.erase(colliding[1]));
  assert(!requests.find(colliding[1]).has_value());
  for (std::size_t index : {std::size_t{0}, std::size_t{2},
                            std::size_t{3}, std::size_t{4}}) {
    const auto metadata = requests.find(colliding[index]);
    assert(metadata.has_value());
    assert(metadata->request_id == colliding[index]);
    assert(metadata->last_send_time == 100 + index);
    assert(requests.lookup_probe_count(colliding[index]) <= requests.size());
  }
  assert(requests.lookup_probe_count(colliding[1]) <= requests.size() + 1);

  assert(requests.erase(colliding[0]));
  assert(requests.erase(colliding[2]));
  assert(requests.erase(colliding[3]));
  assert(requests.erase(colliding[4]));
  assert(requests.size() == 0);
  assert(requests.lookup_probe_count(colliding[4]) == 1);
}

}  // namespace

int main() {
  test_cleanup_scheduler_is_ordered_and_retry_fenced();
  test_cleanup_scheduler_never_bypasses_delayed_predecessor();
  test_context_capacity_and_state_transitions();
  test_request_retry_duplicates_and_reverse_fan_in();
  test_request_capacity_and_validation();
  test_zero_peer_fast_paths_and_mask_validation();
  test_retryable_cleanup_release_cannot_pin_the_only_context();
  test_request_tracker_true_deletion_survives_long_churn();
  test_backward_shift_preserves_colliding_record_indices();
  return 0;
}
