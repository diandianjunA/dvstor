#include <cassert>
#include <deque>
#include <vector>

#include "service/compute_service/storage_owner/batch_policy.hh"
#include "service/compute_service/storage_owner/response_validation.hh"

namespace {

using compute_service_detail::StorageOwnerResponseValidation;
using compute_service_detail::dequeue_storage_owner_visible_prefix;
using compute_service_detail::decide_storage_owner_batch;
using compute_service_detail::next_storage_owner_batch_observed_ns;
using compute_service_detail::storage_owner_batch_pressure;
using compute_service_detail::validate_storage_owner_response;

struct ScriptedPrefixQueue {
  std::deque<u32> entries;
  u32 visible{};

  bool try_pop(u32& value) {
    if (visible == 0 || entries.empty()) return false;
    value = entries.front();
    entries.pop_front();
    --visible;
    return true;
  }
};

void test_matched_malformed_response_fails() {
  constexpr u32 kOwner = 2;
  constexpr u32 kItems = 4;
  constexpr u64 kBatch = 77;
  const size_t expected_bytes =
    service::storage_owner::insert_batch_response_bytes(kItems);
  service::storage_owner::InsertBatchResponseHeader response{
    .magic = service::storage_owner::kInsertMagic,
    .owner_storage = kOwner,
    .item_count = kItems,
    .batch_id = kBatch,
  };

  const auto classify = [&](size_t received_bytes) {
    return validate_storage_owner_response(
      response,
      received_bytes,
      service::storage_owner::insert_batch_response_bytes(32),
      service::storage_owner::kInsertMagic,
      kOwner,
      kItems,
      kBatch,
      expected_bytes);
  };
  assert(classify(expected_bytes) ==
         StorageOwnerResponseValidation::matched_valid);

  response.batch_id = kBatch + 1;
  assert(classify(expected_bytes) ==
         StorageOwnerResponseValidation::unmatched);
  response.batch_id = kBatch;

  response.magic = 0;
  assert(classify(expected_bytes) ==
         StorageOwnerResponseValidation::matched_invalid);
  response.magic = service::storage_owner::kInsertMagic;
  response.owner_storage = kOwner + 1;
  assert(classify(expected_bytes) ==
         StorageOwnerResponseValidation::matched_invalid);
  response.owner_storage = kOwner;
  response.item_count = UINT32_MAX;
  assert(classify(expected_bytes) ==
         StorageOwnerResponseValidation::matched_invalid);
  response.item_count = kItems;
  assert(classify(expected_bytes - 1) ==
         StorageOwnerResponseValidation::matched_invalid);
  assert(classify(expected_bytes + 1) ==
         StorageOwnerResponseValidation::matched_invalid);
}

void test_batch_policy_sends_full_batch_immediately() {
  const auto decision = decide_storage_owner_batch(
    41, 4, 12, 7, 32, 100, 101, 0);
  assert(!decision.tail_escape);
  assert(!decision.max_wait_flush);
  assert(decision.take == 32);
}

void test_batch_pressure_is_capacity_and_load_derived() {
  // The 16 existing slots produce a 15-RPC target, leaving one lane for a
  // full or deadline-bound request.  No capacity is added by this policy.
  const auto empty = storage_owner_batch_pressure(
    4, 0, 16, 47, 32, 10'000);
  assert(empty.rpc_capacity == 16);
  assert(empty.concurrency_target == 15);
  assert(empty.low_water == 8);
  assert(empty.efficient_quantum == 4);  // ceil(batch_max / low_water)
  assert(empty.ramp_lanes == 8);
  assert(empty.launch_quantum == 7);  // ceil((4 + 47) / 8)
  assert(empty.adaptive_wait_ns == 625);

  // As the pipeline fills, the launch quantum and coalescing time grow.  This
  // naturally restores batching near the target instead of fixing batch=1.
  const auto low_water = storage_owner_batch_pressure(
    4, 7, 9, 0, 32, 10'000);
  assert(low_water.concurrency_target == 15);
  assert(low_water.low_water == 8);
  assert(low_water.ramp_lanes == 1);
  assert(low_water.launch_quantum == 4);
  assert(low_water.adaptive_wait_ns == 5'000);

  const auto near_target = storage_owner_batch_pressure(
    8, 14, 2, 8, 32, 10'000);
  assert(near_target.ramp_lanes == 1);
  assert(near_target.launch_quantum == 16);
  assert(near_target.adaptive_wait_ns == 9'375);
}

void test_low_occupancy_launches_a_load_sized_partial() {
  // Fifty-one offered writes are distributed over the eight vacant low-water
  // lanes, so this owner waits for seven visible items instead of issuing a
  // singleton or putting the whole queue in one RPC.
  const auto waiting = decide_storage_owner_batch(
    6, 0, 16, 45, 32, 100, 100, 10'000);
  assert(waiting.take == 0);

  const auto launch = decide_storage_owner_batch(
    7, 0, 16, 44, 32, 100, 100, 10'000);
  assert(launch.occupancy_flush);
  assert(!launch.adaptive_wait_flush);
  assert(launch.take == 7);

  // If publication is slow, the occupancy-scaled deadline still fills the
  // idle transport window without waiting for the 10ms hard deadline.
  const auto early_waiting = decide_storage_owner_batch(
    2, 0, 16, 49, 32, 100, 724, 10'000);
  assert(early_waiting.take == 0);
  const auto early_flush = decide_storage_owner_batch(
    2, 0, 16, 49, 32, 100, 725, 10'000);
  assert(early_flush.occupancy_flush);
  assert(early_flush.adaptive_wait_flush);
  assert(!early_flush.max_wait_flush);
  assert(early_flush.take == 2);
}

void test_closed_loop_ramps_to_low_water_without_singleton_collapse() {
  u32 ready = 30;
  u32 active = 0;
  u32 free = 16;
  u32 dispatched = 0;
  u32 batches = 0;
  while (ready >= 4) {
    const auto decision = decide_storage_owner_batch(
      ready, active, free, 0, 32, 100, 100, 10'000);
    assert(decision.occupancy_flush);
    assert(!decision.adaptive_wait_flush);
    assert(decision.take == 4);
    assert(decision.take <= ready);
    ready -= decision.take;
    dispatched += decision.take;
    ++active;
    --free;
    ++batches;
  }
  assert(ready == 2);
  assert(active == 7);
  const auto coalescing_tail = decide_storage_owner_batch(
    ready, active, free, 0, 32, 100, 5'099, 10'000);
  assert(coalescing_tail.take == 0);
  const auto tail = decide_storage_owner_batch(
    ready, active, free, 0, 32, 100, 5'100, 10'000);
  assert(tail.occupancy_flush);
  assert(tail.adaptive_wait_flush);
  assert(tail.take == 2);
  ready -= tail.take;
  dispatched += tail.take;
  ++active;
  --free;
  ++batches;
  assert(dispatched == 30);
  assert(batches == 8);
  assert(active == 8);  // capacity-derived low water for 16 slots
  assert(free == 8);
}

void test_closed_loop_single_refill_cannot_pin_a_singleton_lane() {
  // Model the dangerous steady state after seven RPCs are already active.
  // A synchronous producer publishes exactly one replacement only after its
  // previous RPC completes, so pending_producers is normally zero whenever
  // the sender sees the ready item.  Sending it immediately would preserve
  // active=7 forever as a singleton closed loop.
  constexpr u64 oldest = 100;
  const auto one_ready = decide_storage_owner_batch(
    1, 7, 9, 0, 32, oldest, oldest, 10'000);
  assert(one_ready.take == 0);

  const auto three_ready = decide_storage_owner_batch(
    3, 7, 9, 0, 32, oldest, oldest, 10'000);
  assert(three_ready.take == 0);

  // Four replacements form the capacity-derived efficient quantum and fill
  // the eighth low-water lane in one useful request.
  const auto coalesced = decide_storage_owner_batch(
    4, 7, 9, 0, 32, oldest, oldest, 10'000);
  assert(coalesced.occupancy_flush);
  assert(!coalesced.adaptive_wait_flush);
  assert(coalesced.take == 4);

  // A genuinely finite one-item tail is still bounded by the occupancy-scaled
  // deadline (5ms at active=7), rather than being starved until the 10ms hard
  // deadline.
  const auto before_deadline = decide_storage_owner_batch(
    1, 7, 9, 0, 32, oldest, oldest + 4'999, 10'000);
  assert(before_deadline.take == 0);
  const auto at_deadline = decide_storage_owner_batch(
    1, 7, 9, 0, 32, oldest, oldest + 5'000, 10'000);
  assert(at_deadline.occupancy_flush);
  assert(at_deadline.adaptive_wait_flush);
  assert(at_deadline.take == 1);
}

void test_near_target_coalesces_and_reserves_the_last_lane() {
  // At 14/15 target occupancy, a partial waits almost the full interval.
  const auto waiting = decide_storage_owner_batch(
    8, 14, 2, 8, 32, 100, 9'474, 10'000);
  assert(waiting.take == 0);
  const auto adaptive = decide_storage_owner_batch(
    8, 14, 2, 8, 32, 100, 9'475, 10'000);
  assert(adaptive.occupancy_flush);
  assert(adaptive.adaptive_wait_flush);
  assert(adaptive.take == 8);

  // Once the target is reached, an ordinary partial cannot consume the
  // reserve. A full batch or the hard deadline can still use it.
  const auto target_wait = decide_storage_owner_batch(
    8, 15, 1, 8, 32, 100, 10'099, 10'000);
  assert(target_wait.take == 0);
  const auto target_deadline = decide_storage_owner_batch(
    8, 15, 1, 8, 32, 100, 10'100, 10'000);
  assert(target_deadline.max_wait_flush);
  assert(target_deadline.take == 8);
  const auto target_full = decide_storage_owner_batch(
    32, 15, 1, 8, 32, 100, 100, 10'000);
  assert(target_full.take == 32);
}

void test_finite_tail_uses_efficiency_quantum_and_isolated_tail_is_intact() {
  const auto concurrent_work = decide_storage_owner_batch(
    9, 3, 13, 0, 32, 101, 101, 10'000);
  assert(concurrent_work.occupancy_flush);
  assert(concurrent_work.take == 4);

  const auto isolated_tail = decide_storage_owner_batch(
    4, 0, 16, 0, 32, 100, 100, 50);
  assert(isolated_tail.tail_escape);
  assert(isolated_tail.take == 4);
}

void test_announced_producer_keeps_partial_waiting_until_deadline() {
  // The producer counter is only a liveness hint (it can briefly overlap the
  // published counter), but one outstanding producer means this is not yet an
  // isolated tail and should not consume a lane as a singleton.
  const auto decision = decide_storage_owner_batch(
    1, 0, 16, 31, 32, 100, 100, 50);
  assert(decision.take == 0);
}

void test_partial_visible_dequeue_preserves_expired_batch_age() {
  assert(next_storage_owner_batch_observed_ns(
           100, 2, 1, 3, 1'000) == 100);
  assert(next_storage_owner_batch_observed_ns(
           100, 2, 3, 3, 1'000) == 100);
  assert(next_storage_owner_batch_observed_ns(
           100, 0, 1, 3, 1'000) == 0);

  // A fully visible large dequeue must not rejuvenate its old remainder.
  // At t=9ms, removing 32 of 44 leaves 12 tasks with the original t=0 age;
  // the next decision at the 10ms hard bound must flush them immediately.
  const u64 inherited = next_storage_owner_batch_observed_ns(
    100, 12, 32, 32, 9'100);
  assert(inherited == 100);
  const auto old_tail = decide_storage_owner_batch(
    12, 9, 7, 0, 32, inherited, 10'100, 10'000);
  assert(old_tail.max_wait_flush);
  assert(old_tail.take == 12);
}

void test_expired_concurrent_tail_is_sent_intact() {
  const auto expired = decide_storage_owner_batch(
    31, 5, 11, 3, 32, 100, 150, 50);
  assert(expired.max_wait_flush);
  assert(!expired.tail_escape);
  assert(expired.take == 31);
}

void test_zero_hard_wait_flushes_immediately() {
  const auto zero_wait = decide_storage_owner_batch(
    7, 0, 16, 3, 32, 0, 0, 0);
  assert(zero_wait.max_wait_flush);
  assert(zero_wait.take == 7);
}

void test_batch_policy_never_consumes_rpc_slot_without_credit() {
  const auto no_slot = decide_storage_owner_batch(
    32, 16, 0, 0, 32, 100, 1000, 50);
  assert(no_slot.take == 0);
  const auto no_ready = decide_storage_owner_batch(
    0, 0, 16, 0, 32, 0, 1000, 50);
  assert(no_ready.take == 0);

  // Queue age continues while transport credit is unavailable.  Reclaiming a
  // slot must expose the already-expired whole tail immediately.
  const auto reclaimed = decide_storage_owner_batch(
    7, 15, 1, 1, 32, 100, 1'000, 50);
  assert(reclaimed.max_wait_flush);
  assert(reclaimed.take == 7);
}

void test_sender_consumes_only_the_queue_visible_prefix() {
  ScriptedPrefixQueue queue{{10, 11, 12}, 0};
  std::vector<u32> output;

  // The external published count may describe a later cell while the FIFO
  // head is still being published. No RPC slot or counter credit is consumed.
  assert(dequeue_storage_owner_visible_prefix(queue, 3, output) == 0);
  assert(output.empty());

  // If a hole follows a visible prefix, charge and send only that prefix.
  queue.visible = 2;
  assert(dequeue_storage_owner_visible_prefix(queue, 3, output) == 2);
  assert((output == std::vector<u32>{10, 11}));
  assert(queue.entries.size() == 1 && queue.entries.front() == 12);

  // Once the reserved head is published, a later progress pass drains it.
  queue.visible = 1;
  assert(dequeue_storage_owner_visible_prefix(queue, 1, output) == 1);
  assert((output == std::vector<u32>{10, 11, 12}));
}

void test_expired_batch_still_waits_for_the_visible_fifo_head() {
  const auto expired = decide_storage_owner_batch(
    3, 0, 16, 1, 32, 100, 200, 50);
  assert(expired.max_wait_flush && expired.take == 3);

  ScriptedPrefixQueue queue{{20, 21, 22}, 0};
  std::vector<u32> output;
  // Credit can belong to positions behind a producer-reserved FIFO hole. A
  // deadline authorizes an attempt, never a pop past that hole.
  assert(dequeue_storage_owner_visible_prefix(
           queue, expired.take, output) == 0);
  assert(output.empty());
  queue.visible = 1;
  assert(dequeue_storage_owner_visible_prefix(
           queue, expired.take, output) == 1);
  assert((output == std::vector<u32>{20}));
}

}  // namespace

int main() {
  test_matched_malformed_response_fails();
  test_batch_policy_sends_full_batch_immediately();
  test_batch_pressure_is_capacity_and_load_derived();
  test_low_occupancy_launches_a_load_sized_partial();
  test_closed_loop_ramps_to_low_water_without_singleton_collapse();
  test_closed_loop_single_refill_cannot_pin_a_singleton_lane();
  test_near_target_coalesces_and_reserves_the_last_lane();
  test_finite_tail_uses_efficiency_quantum_and_isolated_tail_is_intact();
  test_announced_producer_keeps_partial_waiting_until_deadline();
  test_partial_visible_dequeue_preserves_expired_batch_age();
  test_expired_concurrent_tail_is_sent_intact();
  test_zero_hard_wait_flushes_immediately();
  test_batch_policy_never_consumes_rpc_slot_without_credit();
  test_sender_consumes_only_the_queue_visible_prefix();
  test_expired_batch_still_waits_for_the_visible_fifo_head();
  return 0;
}
