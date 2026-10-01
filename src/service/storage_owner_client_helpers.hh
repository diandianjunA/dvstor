#pragma once

#include <chrono>

#include <library/utils.hh>

#include "common/types.hh"

namespace service::storage_owner_client {

inline u64 duration_ns(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point end) {
  return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

inline u64 duration_ns_clamped(std::chrono::steady_clock::time_point start,
                               std::chrono::steady_clock::time_point end) {
  if (end <= start) {
    return 0;
  }
  return duration_ns(start, end);
}

inline u64 storage_owner_wr_id(u32 owner_storage, u32 slot_id) {
  return encode_64bit(owner_storage, slot_id);
}

inline constexpr u32 kStorageOwnerCompletionWrBit = 0x80000000u;

inline u64 storage_owner_completion_wr_id(u32 owner_storage, u32 slot_id) {
  return encode_64bit(owner_storage | kStorageOwnerCompletionWrBit, slot_id);
}

inline bool storage_owner_is_completion_wr(u32 encoded_owner) {
  return (encoded_owner & kStorageOwnerCompletionWrBit) != 0;
}

inline u32 storage_owner_wr_owner(u32 encoded_owner) {
  return encoded_owner & ~kStorageOwnerCompletionWrBit;
}

// A successful coupled mutation is already exact and publishes no background
// maintenance debt. Every successful decoupled mutation, including erase and
// replay, commits only after obtaining its non-zero runnable Stage2 fence.
inline constexpr bool valid_success_maintenance_sequence(
    bool synchronous_exact, u64 maintenance_sequence) {
  return synchronous_exact
    ? maintenance_sequence == 0 : maintenance_sequence != 0;
}

}  // namespace service::storage_owner_client
