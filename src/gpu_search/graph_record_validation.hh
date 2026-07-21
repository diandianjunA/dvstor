#pragma once

#include <cstdint>

namespace gpu_search::graph_record_validation {

enum class SnapshotState : std::uint8_t {
  valid,
  stale_incarnation,
  invalid,
};

enum class ReadAction : std::uint8_t {
  accept,
  discard_stale,
  retry,
  fail,
};

#if defined(__CUDACC__)
#define DVSTOR_GRAPH_RECORD_HD __host__ __device__ __forceinline__
#else
#define DVSTOR_GRAPH_RECORD_HD inline
#endif

DVSTOR_GRAPH_RECORD_HD std::uint16_t checksum16(
    const std::uint8_t* data, std::uint32_t bytes) {
  std::uint32_t hash = 2166136261u;
  for (std::uint32_t index = 0; index < bytes; ++index) {
    if (index == 2 || index == 3) continue;
    hash ^= data[index];
    hash *= 16777619u;
  }
  hash ^= hash >> 16;
  return static_cast<std::uint16_t>(hash);
}

DVSTOR_GRAPH_RECORD_HD std::uint32_t load_u32(
    const std::uint8_t* data) {
  return static_cast<std::uint32_t>(data[0]) |
    (static_cast<std::uint32_t>(data[1]) << 8) |
    (static_cast<std::uint32_t>(data[2]) << 16) |
    (static_cast<std::uint32_t>(data[3]) << 24);
}

// A checksum-valid dynamic graph record with another incarnation is not a
// damaged RDMA response: cleanup may have recycled the slot while an older
// read-committed query still holds its tagged handle.  Static records are never
// recycled, so an incarnation mismatch for incarnation zero remains invalid.
DVSTOR_GRAPH_RECORD_HD SnapshotState classify_snapshot(
    const std::uint8_t* record,
    std::uint32_t record_bytes,
    std::uint32_t graph_degree,
    std::uint32_t graph_entry_capacity,
    std::uint32_t expected_incarnation) {
  if (record == nullptr || record_bytes < 16 ||
      graph_entry_capacity < graph_degree) {
    return SnapshotState::invalid;
  }
  const std::uint16_t stored_checksum =
    static_cast<std::uint16_t>(record[2]) |
    static_cast<std::uint16_t>(
      static_cast<std::uint16_t>(record[3]) << 8);
  const std::uint32_t stable_count = record[0];
  const std::uint32_t provisional_count = (record[1] >> 4) & 0xfu;
  const std::uint32_t provisional_capacity =
    graph_entry_capacity - graph_degree;
  const bool structurally_valid =
    (record[1] & 0x0eu) == 0 &&
    stable_count <= graph_degree &&
    provisional_count <= provisional_capacity &&
    stable_count + provisional_count <= graph_entry_capacity &&
    load_u32(record + 12) == 0 &&
    stored_checksum == checksum16(record, record_bytes);
  if (!structurally_valid) return SnapshotState::invalid;

  const std::uint32_t stored_incarnation = load_u32(record + 8);
  if (stored_incarnation == expected_incarnation) {
    return SnapshotState::valid;
  }
  return expected_incarnation == 0
    ? SnapshotState::invalid : SnapshotState::stale_incarnation;
}

DVSTOR_GRAPH_RECORD_HD ReadAction decide_read_action(
    bool transport_succeeded,
    SnapshotState snapshot,
    bool attempts_remain) {
  if (!transport_succeeded) return ReadAction::fail;
  if (snapshot == SnapshotState::valid) return ReadAction::accept;
  if (snapshot == SnapshotState::stale_incarnation) {
    return ReadAction::discard_stale;
  }
  return attempts_remain ? ReadAction::retry : ReadAction::fail;
}

#undef DVSTOR_GRAPH_RECORD_HD

}  // namespace gpu_search::graph_record_validation
