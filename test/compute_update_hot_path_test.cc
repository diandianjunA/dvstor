#include <cassert>

#include "service/compute_service/storage_owner/batch_policy.hh"
#include "service/compute_service/storage_owner/response_validation.hh"

namespace {

using compute_service_detail::StorageOwnerResponseValidation;
using compute_service_detail::decide_storage_owner_batch;
using compute_service_detail::validate_storage_owner_response;

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

void test_batch_policy_is_immediate_below_saturation() {
  const auto decision = decide_storage_owner_batch(
    false, 3, 4, 12, 7, 32);
  assert(!decision.saturated);
  assert(!decision.tail_escape);
  assert(decision.take == 3);
}

void test_batch_policy_latches_and_forms_full_batches() {
  const auto latch = decide_storage_owner_batch(
    false, 7, 16, 0, 9, 32);
  assert(latch.saturated);
  assert(latch.take == 0);

  const auto hold_tail = decide_storage_owner_batch(
    true, 31, 5, 11, 3, 32);
  assert(hold_tail.saturated);
  assert(!hold_tail.tail_escape);
  assert(hold_tail.take == 0);

  const auto full = decide_storage_owner_batch(
    true, 41, 5, 11, 3, 32);
  assert(full.saturated);
  assert(!full.tail_escape);
  assert(full.take == 32);
}

void test_batch_policy_tail_is_self_clocked_and_epoch_exits() {
  const auto tail = decide_storage_owner_batch(
    true, 9, 0, 16, 0, 32);
  assert(tail.saturated);
  assert(tail.tail_escape);
  assert(tail.take == 9);

  const auto producer_gap = decide_storage_owner_batch(
    true, 0, 0, 16, 1, 32);
  assert(producer_gap.saturated);
  assert(producer_gap.take == 0);

  const auto announced_tail = decide_storage_owner_batch(
    true, 9, 0, 16, 3, 32);
  assert(announced_tail.saturated);
  assert(!announced_tail.tail_escape);
  assert(announced_tail.take == 0);

  const auto drained = decide_storage_owner_batch(
    true, 0, 0, 16, 0, 32);
  assert(!drained.saturated);
  assert(drained.take == 0);
}

}  // namespace

int main() {
  test_matched_malformed_response_fails();
  test_batch_policy_is_immediate_below_saturation();
  test_batch_policy_latches_and_forms_full_batches();
  test_batch_policy_tail_is_self_clocked_and_epoch_exits();
  return 0;
}
