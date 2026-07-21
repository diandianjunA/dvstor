#include <cassert>
#include <deque>

#include "service/compute_service/storage_owner/batch_policy.hh"
#include "service/compute_service/storage_owner/response_validation.hh"

namespace {

using compute_service_detail::StorageOwnerResponseValidation;
using compute_service_detail::drain_concurrent_storage_owner_batch;
using compute_service_detail::kConcurrentProducerBatchRounds;
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

void test_batch_policy_drains_ready_items_without_waiting() {
  std::deque<u32> ready{11, 12, 13};
  vec<u32> output{10};
  u32 relaxations = 0;
  const auto result = drain_concurrent_storage_owner_batch(
    1, 4,
    [&](u32& item) {
      if (ready.empty()) return false;
      item = ready.front();
      ready.pop_front();
      return true;
    },
    []() { return false; },
    [&](u32 item) { output.push_back(item); },
    [&]() { ++relaxations; });
  assert(result.item_count == 4);
  assert(result.wait_rounds == 0);
  assert(relaxations == 0);
  assert((output == vec<u32>{10, 11, 12, 13}));
}

void test_batch_policy_waits_only_for_announced_concurrency() {
  std::deque<u32> ready;
  vec<u32> output{20};
  bool pending = true;
  u32 relaxations = 0;
  const auto result = drain_concurrent_storage_owner_batch(
    1, 4,
    [&](u32& item) {
      if (ready.empty()) return false;
      item = ready.front();
      ready.pop_front();
      return true;
    },
    [&]() { return pending; },
    [&](u32 item) { output.push_back(item); },
    [&]() {
      ++relaxations;
      ready.push_back(21);
      ready.push_back(22);
      pending = false;
    });
  assert(result.item_count == 3);
  assert(result.wait_rounds == 1);
  assert(relaxations == 1);
  assert((output == vec<u32>{20, 21, 22}));

  ready.clear();
  output.assign(1, 30);
  relaxations = 0;
  const auto isolated = drain_concurrent_storage_owner_batch(
    1, 4,
    [&](u32&) { return false; },
    []() { return false; },
    [&](u32 item) { output.push_back(item); },
    [&]() { ++relaxations; });
  assert(isolated.item_count == 1);
  assert(isolated.wait_rounds == 0);
  assert(relaxations == 0);
}

void test_batch_policy_has_a_hard_wait_bound() {
  u32 relaxations = 0;
  const auto result = drain_concurrent_storage_owner_batch(
    1, 32,
    [&](u32&) { return false; },
    []() { return true; },
    [&](u32) { assert(false); },
    [&]() { ++relaxations; });
  assert(result.item_count == 1);
  assert(result.wait_rounds == kConcurrentProducerBatchRounds);
  assert(relaxations == kConcurrentProducerBatchRounds);
}

void test_batch_policy_reprobes_after_producer_closes() {
  vec<u32> output{40};
  u32 probes = 0;
  const auto result = drain_concurrent_storage_owner_batch(
    1, 4,
    [&](u32& item) {
      ++probes;
      if (probes != 2) return false;
      item = 41;
      return true;
    },
    []() { return false; },
    [&](u32 item) { output.push_back(item); },
    []() { assert(false); });
  assert(result.item_count == 2);
  assert(result.wait_rounds == 0);
  assert((output == vec<u32>{40, 41}));
}

}  // namespace

int main() {
  test_matched_malformed_response_fails();
  test_batch_policy_drains_ready_items_without_waiting();
  test_batch_policy_waits_only_for_announced_concurrency();
  test_batch_policy_has_a_hard_wait_bound();
  test_batch_policy_reprobes_after_producer_closes();
  return 0;
}
