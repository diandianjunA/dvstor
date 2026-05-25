#pragma once

#include <cstdint>

#include "common/types.hh"

namespace service::compute_service_detail {

inline constexpr u32 kRpcMagic = 0x53484e57;
inline constexpr u32 kRpcVersion = 1;
inline constexpr u32 kInitialRpcRecvsPerPeer = 8;
inline constexpr u32 kMaxRpcResults = 512;
inline constexpr u32 kRabitqSearchBeamSlack = 64;

}  // namespace service::compute_service_detail
