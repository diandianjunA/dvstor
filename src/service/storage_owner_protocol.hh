#pragma once

#include "common/types.hh"

namespace service::storage_owner {

constexpr u32 kInsertMagic = 0x53494e54;  // "SINT"

enum class InsertStatus : u32 {
  ok = 0,
  failed = 1,
};

struct InsertRequest {
  u32 magic{kInsertMagic};
  u32 dim{};
  node_t id{};
  u32 owner_storage{};
  u32 source_client{};
  u32 reserved{};
  u64 request_id{};
};

struct InsertResponse {
  u32 magic{kInsertMagic};
  u32 status{static_cast<u32>(InsertStatus::failed)};
  node_t id{};
  u32 owner_storage{};
  u64 request_id{};
};

inline size_t request_bytes(u32 dim) {
  return sizeof(InsertRequest) + static_cast<size_t>(dim) * sizeof(element_t);
}

}  // namespace service::storage_owner
