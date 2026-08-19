#include <cassert>
#include <deque>
#include <string_view>
#include <vector>

#include "service/compute_service/storage_owner/batch_policy.hh"
#include "service/compute_service/storage_owner/response_validation.hh"
#include "service/storage_owner_client_helpers.hh"

namespace {

using compute_service_detail::StorageOwnerResponseValidation;
using compute_service_detail::dequeue_storage_owner_visible_prefix;
using compute_service_detail::decide_storage_owner_batch;
using compute_service_detail::next_storage_owner_batch_observed_ns;
using compute_service_detail::validate_storage_owner_response;
using service::storage_owner_client::valid_success_maintenance_sequence;

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

void test_concurrent_partial_waits_for_the_true_deadline() {
  const auto waiting = decide_storage_owner_batch(
    9, 3, 13, 0, 32, 101, 10'100, 10'000);
  assert(waiting.take == 0);
  const auto expired = decide_storage_owner_batch(
    9, 3, 13, 0, 32, 101, 10'101, 10'000);
  assert(expired.max_wait_flush);
  assert(expired.take == 9);

  const auto isolated_tail = decide_storage_owner_batch(
    4, 0, 16, 0, 32, 100, 100, 50);
  assert(isolated_tail.tail_escape);
  assert(isolated_tail.take == 4);
}

double simulate_closed_sender(u32 rpc_depth) {
  constexpr u32 kCallers = 51;
  constexpr u32 kBatchMax = 32;
  constexpr u64 kWait = 10'000;
  std::deque<u32> active_batches;
  u32 ready = kCallers;
  u64 now = 1;
  u64 oldest = now;
  u64 dispatched_items = 0;
  u64 dispatched_batches = 0;

  for (u32 completion = 0; completion < 2'000; ++completion) {
    for (;;) {
      const u32 active = static_cast<u32>(active_batches.size());
      const u32 free = rpc_depth - active;
      auto decision = decide_storage_owner_batch(
        ready, active, free, 0, kBatchMax, oldest, now, kWait);
      if (decision.take == 0 && ready != 0 && free != 0) {
        now = oldest + kWait;
        decision = decide_storage_owner_batch(
          ready, active, free, 0, kBatchMax, oldest, now, kWait);
      }
      if (decision.take == 0) break;
      assert(decision.take <= ready);
      ready -= decision.take;
      active_batches.push_back(decision.take);
      dispatched_items += decision.take;
      ++dispatched_batches;
      if (ready == 0) oldest = 0;
    }

    assert(!active_batches.empty());
    const u32 returned_callers = active_batches.front();
    active_batches.pop_front();
    ++now;
    if (ready == 0) oldest = now;
    ready += returned_callers;
  }
  return static_cast<double>(dispatched_items) /
    static_cast<double>(dispatched_batches);
}

void test_rpc_depth_is_a_ceiling_not_a_fragmentation_target() {
  // Model the actual closed-loop workload: 51 synchronous callers per owner
  // republish only after their previous batch completes. Thousands of refill
  // cycles must retain useful batches, and doubling the safety depth must not
  // cut the average batch by spreading the same callers over more lanes.
  const double depth8_batch = simulate_closed_sender(8);
  const double depth16_batch = simulate_closed_sender(16);
  assert(depth8_batch >= 20.0);
  assert(depth16_batch >= 20.0);
  assert(depth16_batch >= depth8_batch);
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
           100, 2, 1'000) == 100);
  assert(next_storage_owner_batch_observed_ns(
           100, 2, 1'000) == 100);
  assert(next_storage_owner_batch_observed_ns(
           100, 0, 1'000) == 0);

  // A fully visible large dequeue must not rejuvenate its old remainder.
  // At t=9ms, removing 32 of 44 leaves 12 tasks with the original t=0 age;
  // the next decision at the 10ms hard bound must flush them immediately.
  const u64 inherited = next_storage_owner_batch_observed_ns(
    100, 12, 9'100);
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

void test_update_completion_mode_owns_maintenance_sequence() {
  assert(valid_success_maintenance_sequence(true, 0));
  assert(!valid_success_maintenance_sequence(true, 1));
  assert(!valid_success_maintenance_sequence(false, 0));
  assert(valid_success_maintenance_sequence(false, 1));
  assert(valid_success_maintenance_sequence(false, UINT64_MAX));
}

void test_coupled_update_mode_is_strictly_append_only() {
  using service::storage_owner::MutationKind;
  using service::storage_owner::mutation_api_name_for_completion_mode;
  using service::storage_owner::mutation_supported_by_completion_mode;
  static_assert(mutation_supported_by_completion_mode(
    true, MutationKind::insert));
  static_assert(!mutation_supported_by_completion_mode(
    true, MutationKind::upsert));
  static_assert(!mutation_supported_by_completion_mode(
    true, MutationKind::erase));
  static_assert(mutation_supported_by_completion_mode(
    false, MutationKind::insert));
  static_assert(mutation_supported_by_completion_mode(
    false, MutationKind::upsert));
  static_assert(mutation_supported_by_completion_mode(
    false, MutationKind::erase));
  static_assert(std::string_view(
    mutation_api_name_for_completion_mode(true)) == "append_only");
  static_assert(std::string_view(
    mutation_api_name_for_completion_mode(false)) ==
      "insert_upsert_erase");
}

}  // namespace

int main() {
  test_matched_malformed_response_fails();
  test_batch_policy_sends_full_batch_immediately();
  test_concurrent_partial_waits_for_the_true_deadline();
  test_rpc_depth_is_a_ceiling_not_a_fragmentation_target();
  test_announced_producer_keeps_partial_waiting_until_deadline();
  test_partial_visible_dequeue_preserves_expired_batch_age();
  test_expired_concurrent_tail_is_sent_intact();
  test_zero_hard_wait_flushes_immediately();
  test_batch_policy_never_consumes_rpc_slot_without_credit();
  test_sender_consumes_only_the_queue_visible_prefix();
  test_expired_batch_still_waits_for_the_visible_fifo_head();
  test_update_completion_mode_owns_maintenance_sequence();
  test_coupled_update_mode_is_strictly_append_only();
  return 0;
}
