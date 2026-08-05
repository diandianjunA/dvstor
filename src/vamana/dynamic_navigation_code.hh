#pragma once

#include <cstdint>

namespace vamana::dynamic_navigation_code {

// Dynamic PQ records are fetched with one RDMA READ as
//
//   [ incarnation/extent tag | PQ payload | incarnation-bound checksum ]
//
// The checksum deliberately ignores the advisory extent byte.  Graph updates
// may therefore co-publish a new extent in the existing graph+tag WRITE
// without touching the immutable PQ payload or its trailer.  Slot reuse
// changes the low 24-bit incarnation and the payload, so mixed/torn snapshots
// are detected subject to the explicit 32-bit checksum-collision boundary.
inline constexpr std::uint32_t kIncarnationMask = 0x00ffffffu;
inline constexpr std::uint32_t kChecksumBytes = sizeof(std::uint32_t);

#if defined(__CUDACC__)
#define DVSTOR_DYNAMIC_NAV_HD __host__ __device__ __forceinline__
#else
#define DVSTOR_DYNAMIC_NAV_HD inline
#endif

DVSTOR_DYNAMIC_NAV_HD std::uint32_t checksum(
    std::uint32_t packed_tag,
    const std::uint8_t* payload,
    std::uint32_t payload_bytes) {
  std::uint32_t hash = 2166136261u;
  const std::uint32_t incarnation = packed_tag & kIncarnationMask;
  // Hash the normalized four-byte little-endian incarnation (the high byte is
  // zero), followed by the fixed-width PQ payload.
  for (std::uint32_t byte = 0; byte < sizeof(incarnation); ++byte) {
    hash ^= static_cast<std::uint8_t>(incarnation >> (byte * 8u));
    hash *= 16777619u;
  }
  for (std::uint32_t byte = 0; byte < payload_bytes; ++byte) {
    hash ^= payload[byte];
    hash *= 16777619u;
  }
  hash ^= hash >> 16;
  return hash;
}

DVSTOR_DYNAMIC_NAV_HD std::uint32_t load_u32_le(
    const std::uint8_t* data) {
  return static_cast<std::uint32_t>(data[0]) |
    (static_cast<std::uint32_t>(data[1]) << 8) |
    (static_cast<std::uint32_t>(data[2]) << 16) |
    (static_cast<std::uint32_t>(data[3]) << 24);
}

DVSTOR_DYNAMIC_NAV_HD void store_u32_le(
    std::uint8_t* data, std::uint32_t value) {
  data[0] = static_cast<std::uint8_t>(value);
  data[1] = static_cast<std::uint8_t>(value >> 8);
  data[2] = static_cast<std::uint8_t>(value >> 16);
  data[3] = static_cast<std::uint8_t>(value >> 24);
}

DVSTOR_DYNAMIC_NAV_HD bool validate(
    std::uint32_t packed_tag,
    const std::uint8_t* payload,
    std::uint32_t payload_bytes,
    const std::uint8_t* checksum_bytes) {
  return payload != nullptr && checksum_bytes != nullptr &&
    load_u32_le(checksum_bytes) ==
      checksum(packed_tag, payload, payload_bytes);
}

#undef DVSTOR_DYNAMIC_NAV_HD

}  // namespace vamana::dynamic_navigation_code
