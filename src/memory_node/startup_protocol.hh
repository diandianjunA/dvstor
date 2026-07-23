#pragma once

#include "common/types.hh"

namespace storage_startup {

inline constexpr u32 kMagic = 0x44565354;  // DVST

struct Request {
  u32 magic{kMagic};
};

struct Response {
  bool ready{};
  u8 reserved[3]{};
  u32 vector_id_namespace_size{};
};

static_assert(sizeof(Response) == 8);

}  // namespace storage_startup
