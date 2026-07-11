#pragma once

#include "common/types.hh"

namespace storage_startup {

inline constexpr u32 kMagic = 0x44565354;  // DVST

struct Request {
  u32 magic{kMagic};
};

struct Response {
  bool ready{};
};

}  // namespace storage_startup
