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

inline constexpr std::uint32_t kGraphRecordHeaderBytes = 16;
inline constexpr std::uint32_t kGraphPointerBytes = 8;
inline constexpr std::uint32_t kGraphExtentEdgesPerClass = 8;
inline constexpr std::uint8_t kGraphExtentClassUnknown = 0xffu;

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

// Inspect only the fixed header. In particular, this helper is safe before a
// live-extent reader has fetched the complete counted neighbor prefix. The
// returned byte count is exact (not class-rounded) and must fit inside the
// transferred prefix before a reconstructed record may be accepted.
DVSTOR_GRAPH_RECORD_HD bool required_live_extent_bytes(
    const std::uint8_t* record,
    std::uint32_t available_bytes,
    std::uint32_t graph_degree,
    std::uint32_t graph_entry_capacity,
    std::uint32_t& required_bytes) {
  required_bytes = 0;
  if (record == nullptr || available_bytes < kGraphRecordHeaderBytes ||
      graph_entry_capacity < graph_degree) {
    return false;
  }
  const std::uint32_t stable_count = record[0];
  const std::uint32_t provisional_count = (record[1] >> 4) & 0xfu;
  const std::uint32_t provisional_capacity =
    graph_entry_capacity - graph_degree;
  if ((record[1] & 0x0eu) != 0 ||
      stable_count > graph_degree ||
      provisional_count > provisional_capacity ||
      stable_count + provisional_count > graph_entry_capacity ||
      load_u32(record + 12) != 0) {
    return false;
  }
  required_bytes = kGraphRecordHeaderBytes +
    (stable_count + provisional_count) * kGraphPointerBytes;
  return true;
}

// Convert an immutable one-byte sidecar class into the one-shot RDMA length.
// The final class is clamped to the physical record because capacities need
// not be a multiple of eight (for example, R96 plus six provisional slots).
// Unknown or malformed classes deliberately fall back to a full record.
DVSTOR_GRAPH_RECORD_HD std::uint32_t graph_extent_bytes_for_class(
    std::uint8_t extent_class,
    std::uint32_t record_bytes,
    std::uint32_t graph_entry_capacity) {
  if (record_bytes < kGraphRecordHeaderBytes ||
      extent_class == kGraphExtentClassUnknown) {
    return record_bytes;
  }
  const std::uint32_t maximum_class =
    (graph_entry_capacity + kGraphExtentEdgesPerClass - 1) /
      kGraphExtentEdgesPerClass;
  if (extent_class > maximum_class) return record_bytes;
  const std::uint32_t covered_edges =
    static_cast<std::uint32_t>(extent_class) *
      kGraphExtentEdgesPerClass;
  const std::uint32_t bounded_edges =
    covered_edges < graph_entry_capacity
      ? covered_edges : graph_entry_capacity;
  const std::uint32_t extent_bytes =
    kGraphRecordHeaderBytes + bounded_edges * kGraphPointerBytes;
  return extent_bytes < record_bytes ? extent_bytes : record_bytes;
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
  std::uint32_t required_bytes = 0;
  const bool structurally_valid =
    required_live_extent_bytes(
      record, record_bytes, graph_degree, graph_entry_capacity,
      required_bytes) &&
    required_bytes <= record_bytes &&
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

// A live-extent hint attempt is outside the authoritative full-record
// snapshot budget. `batch_attempt` is zero based. Once an entry has upgraded
// to full, an entry that started short has consumed one fewer full attempt
// than an entry that started full in the same mixed batch.
DVSTOR_GRAPH_RECORD_HD bool snapshot_retry_available(
    std::uint32_t batch_attempt,
    bool started_with_short_extent,
    bool current_read_is_partial,
    std::uint32_t maximum_batch_attempts,
    std::uint32_t maximum_full_attempts) {
  if (batch_attempt + 1u >= maximum_batch_attempts) return false;
  if (current_read_is_partial) return maximum_full_attempts != 0;
  if (started_with_short_extent && batch_attempt == 0) return false;
  const std::uint32_t full_attempts_after_current =
    batch_attempt + 1u - (started_with_short_extent ? 1u : 0u);
  return full_attempts_after_current < maximum_full_attempts;
}

#undef DVSTOR_GRAPH_RECORD_HD

}  // namespace gpu_search::graph_record_validation
