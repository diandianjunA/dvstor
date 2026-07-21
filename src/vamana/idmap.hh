#pragma once

#include <algorithm>
#include <array>
#include <istream>
#include <limits>

#include "common/types.hh"
#include "remote_pointer.hh"

namespace vamana::idmap {

// Owner-sharded authority directory, version 2.  Version 1 deliberately has
// no compatibility path: it was not bound to an index build and could be
// paired with an identically-shaped shard from another build.
inline constexpr u64 kMagic = 0x32444e424d444944ULL;  // "DIDMBND2"
inline constexpr u32 kVersion = 2;
inline constexpr u32 kMetadataSchemaVersion = 16;
inline constexpr u32 kRemotePtrFormatVersion = 1;
inline constexpr u32 kNoFlags = 0;

#pragma pack(push, 1)
struct Entry {
  node_t id{};
  u64 rptr_raw{};
  u32 generation{};
  u32 flags{};
  u32 reserved{};
};

struct Header {
  u64 magic{kMagic};
  u64 build_fingerprint{};
  u64 owner_shard_fingerprint{};
  u64 entry_count{};
  u64 payload_bytes{};
  u64 payload_checksum{};
  u64 header_checksum{};
  u64 node_base_offset{};

  u32 version{kVersion};
  u32 header_bytes{sizeof(Header)};
  u32 owner_shard{};
  u32 shard_count{};
  u32 entry_bytes{sizeof(Entry)};
  u32 metadata_schema_version{kMetadataSchemaVersion};
  u32 node_size{};
  u32 id_offset{};
  u32 generation_offset{};
  u32 slot_incarnation_offset{};
  u32 remote_ptr_format_version{kRemotePtrFormatVersion};
  u32 remote_ptr_alignment_log2{RemotePtr::OFFSET_ALIGNMENT_LOG2};
  u32 remote_ptr_offset_bits{RemotePtr::OFFSET_UNIT_BITS};
  u32 remote_ptr_shard_bits{RemotePtr::MEMORY_NODE_BITS};
  u32 remote_ptr_incarnation_bits{RemotePtr::INCARNATION_BITS};
  u32 static_incarnation{};
};

#pragma pack(pop)

static_assert(sizeof(Header) == 128);
static_assert(sizeof(Entry) == 24);

struct ValidationContext {
  u64 build_fingerprint{};
  u64 owner_shard_fingerprint{};
  u64 node_base_offset{};
  u32 owner_shard{};
  u32 shard_count{};
  u32 node_size{};
  u32 id_offset{};
  u32 generation_offset{};
  u32 slot_incarnation_offset{};
  span<const u64> static_entry_counts{};
};

inline u64 checksum_initial() { return 1469598103934665603ULL; }

inline u64 checksum_update(u64 value, const void* data, size_t size) {
  const auto* bytes = static_cast<const byte_t*>(data);
  for (size_t index = 0; index < size; ++index) {
    value ^= static_cast<u8>(bytes[index]);
    value *= 1099511628211ULL;
  }
  return value;
}

inline u64 checksum(span<const byte_t> bytes) {
  return checksum_update(checksum_initial(), bytes.data(), bytes.size());
}

inline bool checked_payload_bytes(u64 entry_count, u64& bytes) {
  if (entry_count > std::numeric_limits<u64>::max() / sizeof(Entry)) {
    return false;
  }
  bytes = entry_count * sizeof(Entry);
  return true;
}

inline u64 compute_header_checksum(const Header& source) {
  Header copy = source;
  copy.header_checksum = 0;
  return checksum(span<const byte_t>{
    reinterpret_cast<const byte_t*>(&copy), sizeof(copy)});
}

inline bool valid_header_checksum(const Header& header) {
  return header.header_checksum != 0 &&
    header.header_checksum == compute_header_checksum(header);
}

inline bool valid_header(const Header& header, u64 exact_file_bytes,
                         const ValidationContext& expected) {
  u64 expected_payload_bytes = 0;
  if (!checked_payload_bytes(header.entry_count, expected_payload_bytes) ||
      expected_payload_bytes >
        std::numeric_limits<u64>::max() - sizeof(Header)) {
    return false;
  }
  u64 total_static_entries = 0;
  for (const u64 count : expected.static_entry_counts) {
    if (count > std::numeric_limits<u64>::max() - total_static_entries) {
      return false;
    }
    total_static_entries += count;
  }
  return header.magic == kMagic && header.version == kVersion &&
    header.header_bytes == sizeof(Header) &&
    valid_header_checksum(header) &&
    header.build_fingerprint != 0 &&
    header.build_fingerprint == expected.build_fingerprint &&
    header.owner_shard_fingerprint != 0 &&
    header.owner_shard_fingerprint == expected.owner_shard_fingerprint &&
    header.owner_shard == expected.owner_shard &&
    header.shard_count == expected.shard_count &&
    header.shard_count > 0 &&
    header.shard_count <= RemotePtr::MEMORY_NODE_MASK + 1 &&
    header.entry_count <= total_static_entries &&
    header.payload_bytes == expected_payload_bytes &&
    header.payload_checksum != 0 &&
    exact_file_bytes == sizeof(Header) + expected_payload_bytes &&
    header.node_base_offset == expected.node_base_offset &&
    header.entry_bytes == sizeof(Entry) &&
    header.metadata_schema_version == kMetadataSchemaVersion &&
    header.node_size == expected.node_size && header.node_size != 0 &&
    header.id_offset == expected.id_offset &&
    header.generation_offset == expected.generation_offset &&
    header.slot_incarnation_offset == expected.slot_incarnation_offset &&
    header.remote_ptr_format_version == kRemotePtrFormatVersion &&
    header.remote_ptr_alignment_log2 == RemotePtr::OFFSET_ALIGNMENT_LOG2 &&
    header.remote_ptr_offset_bits == RemotePtr::OFFSET_UNIT_BITS &&
    header.remote_ptr_shard_bits == RemotePtr::MEMORY_NODE_BITS &&
    header.remote_ptr_incarnation_bits == RemotePtr::INCARNATION_BITS &&
    header.static_incarnation == 0 &&
    expected.static_entry_counts.size() == expected.shard_count;
}

inline bool valid_entry(const Entry& entry,
                        const ValidationContext& expected) {
  if (entry.generation != 0 || entry.flags != kNoFlags ||
      entry.reserved != 0 || expected.shard_count == 0 ||
      expected.node_size == 0 ||
      entry.id % expected.shard_count != expected.owner_shard) {
    return false;
  }
  const RemotePtr pointer{entry.rptr_raw};
  if (pointer.is_null() || !pointer.is_well_formed() ||
      pointer.incarnation() != 0 ||
      pointer.memory_node() >= expected.shard_count ||
      pointer.memory_node() >= expected.static_entry_counts.size() ||
      pointer.byte_offset() < expected.node_base_offset) {
    return false;
  }
  const u64 relative = pointer.byte_offset() - expected.node_base_offset;
  return relative % expected.node_size == 0 &&
    relative / expected.node_size <
      expected.static_entry_counts[pointer.memory_node()];
}

// Stream in fixed-size batches so validating a multi-billion-vector idmap
// never creates a second O(N) payload. The consumer returns false for a
// duplicate or for a record-level validation failure.
template <typename EntryConsumer>
bool read_validated_payload(std::istream& input, const Header& header,
                            const ValidationContext& expected,
                            EntryConsumer&& consume) {
  constexpr size_t kChunkEntries = 4096;
  std::array<Entry, kChunkEntries> entries{};
  u64 remaining = header.entry_count;
  u64 payload_checksum = checksum_initial();
  while (remaining != 0) {
    const size_t count = static_cast<size_t>(
      std::min<u64>(remaining, entries.size()));
    const size_t bytes = count * sizeof(Entry);
    input.read(reinterpret_cast<char*>(entries.data()),
               static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes)) {
      return false;
    }
    payload_checksum = checksum_update(
      payload_checksum, entries.data(), bytes);
    for (size_t index = 0; index < count; ++index) {
      if (!valid_entry(entries[index], expected) ||
          !consume(entries[index])) {
        return false;
      }
    }
    remaining -= count;
  }
  return payload_checksum == header.payload_checksum;
}

}  // namespace vamana::idmap
