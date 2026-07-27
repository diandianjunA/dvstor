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

// Return 16777619^exponent modulo 2^32. Unsigned overflow is the arithmetic
// required by FNV-1a, so exponentiation by squaring can append a long run of
// zero bytes without reading or materializing them.
DVSTOR_GRAPH_RECORD_HD std::uint32_t fnv1a_prime_power(
    std::uint32_t exponent) {
  std::uint32_t result = 1u;
  std::uint32_t base = 16777619u;
  while (exponent != 0) {
    if ((exponent & 1u) != 0) result *= base;
    exponent >>= 1;
    if (exponent != 0) base *= base;
  }
  return result;
}

// Compute the existing full-record checksum from an actually consumed prefix
// and a logical all-zero suffix. This is byte-for-byte equivalent to
// checksum16() over a canonical zero-padded record, but touches only
// prefix_bytes and advances the zero suffix in O(log suffix_bytes). A caller
// may therefore exclude class-rounding padding that was transferred but is
// outside the count declared by the validated header.
//
// Bytes 2 and 3 store the checksum and are excluded by the on-disk checksum
// format. The generic skipped_suffix calculation also keeps the helper exact
// for prefixes shorter than the graph header, although live-extent callers
// always transfer at least the complete 16-byte header.
DVSTOR_GRAPH_RECORD_HD std::uint16_t checksum16_zero_extended_prefix(
    const std::uint8_t* data,
    std::uint32_t prefix_bytes,
    std::uint32_t total_bytes) {
  if (data == nullptr || prefix_bytes > total_bytes) return 0;
  std::uint32_t hash = 2166136261u;
  for (std::uint32_t index = 0; index < prefix_bytes; ++index) {
    if (index == 2 || index == 3) continue;
    hash ^= data[index];
    hash *= 16777619u;
  }
  std::uint32_t zero_steps = total_bytes - prefix_bytes;
  if (prefix_bytes <= 2 && total_bytes > 2) --zero_steps;
  if (prefix_bytes <= 3 && total_bytes > 3) --zero_steps;
  hash *= fnv1a_prime_power(zero_steps);
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

// Derive the smallest eight-edge class that covers an exact counted prefix.
// The helper intentionally rejects malformed byte counts instead of rounding
// them: callers use the result to monotonically repair a performance hint,
// never as an authority for record validity.
DVSTOR_GRAPH_RECORD_HD std::uint8_t graph_extent_class_for_required_bytes(
    std::uint32_t required_bytes,
    std::uint32_t graph_entry_capacity) {
  if (required_bytes < kGraphRecordHeaderBytes) {
    return kGraphExtentClassUnknown;
  }
  const std::uint32_t payload_bytes =
    required_bytes - kGraphRecordHeaderBytes;
  if ((payload_bytes % kGraphPointerBytes) != 0) {
    return kGraphExtentClassUnknown;
  }
  const std::uint32_t live_neighbors = payload_bytes / kGraphPointerBytes;
  if (live_neighbors > graph_entry_capacity) {
    return kGraphExtentClassUnknown;
  }
  const std::uint32_t extent_class =
    (live_neighbors + kGraphExtentEdgesPerClass - 1u) /
      kGraphExtentEdgesPerClass;
  return extent_class < kGraphExtentClassUnknown
    ? static_cast<std::uint8_t>(extent_class)
    : kGraphExtentClassUnknown;
}

// Device extent hints are stored four u8 classes per aligned u32 so a class
// can be promoted with one CAS while preserving its three neighbours. These
// pure helpers keep the byte packing and monotonic rule host-testable.
DVSTOR_GRAPH_RECORD_HD std::uint8_t packed_graph_extent_class(
    std::uint32_t word, std::uint32_t byte_index) {
  if (byte_index >= sizeof(std::uint32_t)) {
    return kGraphExtentClassUnknown;
  }
  return static_cast<std::uint8_t>(
    word >> (byte_index * 8u));
}

DVSTOR_GRAPH_RECORD_HD bool promoted_graph_extent_word(
    std::uint32_t observed_word,
    std::uint32_t byte_index,
    std::uint8_t requested_class,
    std::uint32_t& promoted_word) {
  promoted_word = observed_word;
  if (byte_index >= sizeof(std::uint32_t) ||
      requested_class == kGraphExtentClassUnknown) {
    return false;
  }
  const std::uint32_t shift = byte_index * 8u;
  const std::uint32_t mask = 0xffu << shift;
  const std::uint8_t observed_class =
    static_cast<std::uint8_t>((observed_word & mask) >> shift);
  if (observed_class == kGraphExtentClassUnknown ||
      requested_class <= observed_class) {
    return false;
  }
  promoted_word =
    (observed_word & ~mask) |
      (static_cast<std::uint32_t>(requested_class) << shift);
  return true;
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

// Validate a one-sided short read without touching bytes that the NIC did not
// overwrite. Static graph records are canonically zero-padded at build time;
// therefore the stored full-record checksum must equal the checksum of the
// transferred prefix followed by logical zeros. A stale extent hint whose
// unseen suffix contains published data fails this test and is upgraded to
// the authoritative full-record path by the caller.
DVSTOR_GRAPH_RECORD_HD SnapshotState classify_zero_extended_snapshot(
    const std::uint8_t* record,
    std::uint32_t transferred_bytes,
    std::uint32_t record_bytes,
    std::uint32_t graph_degree,
    std::uint32_t graph_entry_capacity,
    std::uint32_t expected_incarnation) {
  if (record == nullptr ||
      transferred_bytes < kGraphRecordHeaderBytes ||
      transferred_bytes >= record_bytes ||
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
      record, transferred_bytes, graph_degree, graph_entry_capacity,
      required_bytes) &&
    required_bytes <= transferred_bytes &&
    stored_checksum == checksum16_zero_extended_prefix(
      record, required_bytes, record_bytes);
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
