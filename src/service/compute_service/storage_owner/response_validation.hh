#pragma once

#include <cstddef>

#include "service/storage_owner_protocol.hh"

namespace compute_service_detail {

enum class StorageOwnerResponseValidation {
  unmatched,
  matched_invalid,
  matched_valid,
};

inline StorageOwnerResponseValidation validate_storage_owner_response(
    const service::storage_owner::InsertBatchResponseHeader& response,
    size_t received_bytes,
    size_t response_buffer_bytes,
    u32 expected_magic,
    u32 expected_owner,
    u32 expected_item_count,
    u64 expected_batch_id,
    size_t expected_response_bytes) {
  if (response.batch_id != expected_batch_id) {
    return StorageOwnerResponseValidation::unmatched;
  }
  const bool valid =
    response.magic == expected_magic &&
    response.owner_storage == expected_owner &&
    response.item_count == expected_item_count &&
    expected_response_bytes <= response_buffer_bytes &&
    received_bytes == expected_response_bytes;
  return valid
    ? StorageOwnerResponseValidation::matched_valid
    : StorageOwnerResponseValidation::matched_invalid;
}

}  // namespace compute_service_detail
