#include <cassert>
#include <deque>
#include <vector>

#include "service/compute_service/storage_owner/batch_policy.hh"
#include "service/compute_service/storage_owner/response_validation.hh"

namespace {

using compute_service_detail::StorageOwnerResponseValidation;
using compute_service_detail::balanced_storage_owner_batch_take;
using compute_service_detail::dequeue_storage_owner_visible_prefix;
using compute_service_detail::decide_storage_owner_batch;
using compute_service_detail::rearm_storage_owner_batch_wait;
using compute_service_detail::storage_owner_dispatch_epoch_take;
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
  assert(!decision.idle_flush);
  assert(!decision.max_wait_flush);
  assert(decision.take == 4);
}

void test_batch_policy_uses_all_existing_rpc_lanes() {
  assert(balanced_storage_owner_batch_take(51, 16, 32) == 4);
  assert(balanced_storage_owner_batch_take(47, 15, 32) == 4);
  assert(balanced_storage_owner_batch_take(32, 16, 32) == 4);
  assert(balanced_storage_owner_batch_take(512, 16, 32) == 32);
  assert(balanced_storage_owner_batch_take(7, 1, 32) == 7);
  assert(balanced_storage_owner_batch_take(7, 16, 32) == 4);
  assert(balanced_storage_owner_batch_take(3, 16, 32) == 3);
  assert(balanced_storage_owner_batch_take(32, 16, 2) == 2);
  assert(balanced_storage_owner_batch_take(0, 16, 32) == 0);
}

void test_batch_policy_idle_tail_preserves_batch_amortization() {
  assert(storage_owner_dispatch_epoch_take(9, 16, 32, true) == 9);
  assert(storage_owner_dispatch_epoch_take(9, 16, 32, false) == 4);
  assert(storage_owner_dispatch_epoch_take(40, 16, 32, true) == 32);
}

void test_batch_policy_holds_concurrent_partial_batch_until_deadline() {
  const auto hold_tail = decide_storage_owner_batch(
    31, 5, 11, 3, 32, 100, 149, 50);
  assert(!hold_tail.idle_flush);
  assert(!hold_tail.max_wait_flush);
  assert(hold_tail.take == 0);

  const auto expired = decide_storage_owner_batch(
    31, 5, 11, 3, 32, 100, 150, 50);
  assert(!expired.idle_flush);
  assert(expired.max_wait_flush);
  assert(expired.take == 4);
}

void test_batch_policy_zero_wait_flushes_concurrent_tail_immediately() {
  const auto decision = decide_storage_owner_batch(
    7, 5, 11, 3, 32, 0, 0, 0);
  assert(!decision.idle_flush);
  assert(decision.max_wait_flush);
  assert(decision.take == 4);
}

void test_batch_policy_isolated_tail_is_immediate() {
  const auto tail = decide_storage_owner_batch(
    9, 0, 16, 0, 32, 100, 101, 50);
  assert(tail.idle_flush);
  assert(!tail.max_wait_flush);
  assert(tail.take == 9);

  const auto producer_gap = decide_storage_owner_batch(
    0, 0, 16, 1, 32, 0, 200, 50);
  assert(producer_gap.take == 0);

  const auto announced_tail = decide_storage_owner_batch(
    9, 0, 16, 3, 32, 100, 101, 50);
  assert(!announced_tail.idle_flush);
  assert(!announced_tail.max_wait_flush);
  assert(announced_tail.take == 0);
}

void test_batch_policy_continuous_load_cannot_strand_tail() {
  for (u64 now = 100; now < 150; ++now) {
    const auto waiting = decide_storage_owner_batch(
      7, 8, 8, 4, 32, 100, now, 50);
    assert(waiting.take == 0);
  }
  const auto deadline = decide_storage_owner_batch(
    7, 8, 8, 4, 32, 100, 150, 50);
  assert(deadline.max_wait_flush);
  assert(deadline.take == 4);

  // A real dequeue starts a new bounded epoch. Reusing the original timestamp
  // would make this second partial batch immediately eligible forever.
  const u64 rearmed_at = rearm_storage_owner_batch_wait(5, 150);
  assert(rearmed_at == 150);
  const auto second_wait = decide_storage_owner_batch(
    5, 8, 8, 4, 32, rearmed_at, 199, 50);
  assert(!second_wait.max_wait_flush && second_wait.take == 0);
  const auto second_deadline = decide_storage_owner_batch(
    5, 8, 8, 4, 32, rearmed_at, 200, 50);
  assert(second_deadline.max_wait_flush && second_deadline.take == 4);
  assert(rearm_storage_owner_batch_wait(0, 200) == 0);
}

void test_batch_policy_never_consumes_rpc_slot_without_credit() {
  const auto no_slot = decide_storage_owner_batch(
    32, 16, 0, 0, 32, 100, 1000, 50);
  assert(no_slot.take == 0);
  const auto no_ready = decide_storage_owner_batch(
    0, 0, 16, 0, 32, 0, 1000, 50);
  assert(no_ready.take == 0);
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
    3, 4, 12, 1, 32, 100, 200, 50);
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
  test_batch_policy_uses_all_existing_rpc_lanes();
  test_batch_policy_idle_tail_preserves_batch_amortization();
  test_batch_policy_holds_concurrent_partial_batch_until_deadline();
  test_batch_policy_zero_wait_flushes_concurrent_tail_immediately();
  test_batch_policy_isolated_tail_is_immediate();
  test_batch_policy_continuous_load_cannot_strand_tail();
  test_batch_policy_never_consumes_rpc_slot_without_credit();
  test_sender_consumes_only_the_queue_visible_prefix();
  test_expired_batch_still_waits_for_the_visible_fifo_head();
  return 0;
}
