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

  // The remaining finite tail stays intact.  It waits while the full RPC is
  // active, then consumes exactly one slot at its deadline rather than being
  // spread over every free slot.
  const auto waiting_tail = decide_storage_owner_batch(
    9, 1, 11, 0, 32, 101, 149, 50);
  assert(waiting_tail.take == 0);
  const auto expired_tail = decide_storage_owner_batch(
    9, 1, 11, 0, 32, 101, 151, 50);
  assert(expired_tail.max_wait_flush);
  assert(expired_tail.take == 9);
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
           100, 2, 3, 3, 1'000) == 1'000);
  assert(next_storage_owner_batch_observed_ns(
           100, 0, 1, 3, 1'000) == 0);
}

void test_expired_concurrent_tail_is_sent_intact() {
  const auto expired = decide_storage_owner_batch(
    31, 5, 11, 3, 32, 100, 150, 50);
  assert(expired.max_wait_flush);
  assert(!expired.tail_escape);
  assert(expired.take == 31);
}

void test_finite_announced_tail_has_a_bounded_initial_wait() {
  const auto waiting = decide_storage_owner_batch(
    9, 0, 16, 3, 32, 100, 149, 50);
  assert(!waiting.max_wait_flush);
  assert(waiting.take == 0);

  const auto expired = decide_storage_owner_batch(
    9, 0, 16, 3, 32, 100, 150, 50);
  assert(expired.max_wait_flush);
  assert(expired.take == 9);

  const auto zero_wait = decide_storage_owner_batch(
    7, 0, 16, 3, 32, 0, 0, 0);
  assert(zero_wait.max_wait_flush);
  assert(zero_wait.take == 7);
}

void test_isolated_write_is_immediate() {
  const auto tail = decide_storage_owner_batch(
    1, 0, 16, 0, 32, 100, 100, 50);
  assert(tail.tail_escape);
  assert(tail.take == 1);
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
  test_announced_producer_keeps_partial_waiting_until_deadline();
  test_partial_visible_dequeue_preserves_expired_batch_age();
  test_expired_concurrent_tail_is_sent_intact();
  test_finite_announced_tail_has_a_bounded_initial_wait();
  test_isolated_write_is_immediate();
  test_batch_policy_never_consumes_rpc_slot_without_credit();
  test_sender_consumes_only_the_queue_visible_prefix();
  test_expired_batch_still_waits_for_the_visible_fifo_head();
  return 0;
}
