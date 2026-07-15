#include <array>
#include <cassert>
#include <span>
#include <vector>

#include "gpu_search/delta_index.hh"
#include "service/compute_service/storage_owner/response_validation.hh"

namespace {

using compute_service_detail::StorageOwnerResponseValidation;
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

void test_coordinator_preserves_reusable_vector() {
  gpu_search::DeltaCoordinator delta;
  std::array<gpu_search::DeltaMutation, 1> mutations;
  auto& mutation = mutations.front();
  mutation.id = 42;
  mutation.owner_storage = 0;
  mutation.maintenance_sequence = 1;
  mutation.vector.resize(128, 7);
  const byte_t* const allocation = mutation.vector.data();

  const u64 epoch = delta.reserve_epoch();
  assert(delta.publish_metadata(
    std::span<gpu_search::DeltaMutation>{mutations}, epoch));
  assert(mutation.vector.data() == allocation);
  assert(mutation.vector.size() == 128);
  assert(mutation.vector.front() == 7);

  const auto retired = delta.retire_durable(std::vector<u64>{1});
  assert(retired.size() == 1);
  assert(retired.front().vector.empty());
}

}  // namespace

int main() {
  test_matched_malformed_response_fails();
  test_coordinator_preserves_reusable_vector();
  return 0;
}
