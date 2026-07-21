#pragma once

#include <cstring>

#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "remote_pointer.hh"

namespace vamana::centroid_state {

inline constexpr u64 kMagic = 0x32544e4543565344ULL;  // "DSVCENT2"
inline constexpr u32 kVersion = 2;
inline constexpr u32 kMaxLiveEntries = 4;
inline constexpr u64 kShardFingerprintOffset = sizeof(u64);
inline constexpr u32 kMetadataSchemaVersion = 16;
inline constexpr u32 kRemotePtrFormatVersion = 1;

#pragma pack(push, 1)
struct Header {
  u64 magic{kMagic};
  // A build has one identity and each physical shard derives a distinct
  // fingerprint from it. The latter is also stored in the first 16 bytes of
  // the shard file, so a sidecar can never be paired with a different build.
  u64 build_fingerprint{};
  u64 shard_fingerprint{};
  u64 vector_count{};
  u64 node_base_offset{};
  u64 payload_bytes{};
  u64 payload_checksum{};
  u64 header_checksum{};

  // Bind the checkpoint to every layout property used to interpret either
  // its entries or the corresponding shard. There is deliberately no v1
  // compatibility path: an old checkpoint must be rebuilt.
  u32 version{kVersion};
  u32 header_bytes{sizeof(Header)};
  u32 shard{};
  u32 shard_count{};
  u32 dim{};
  u32 max_degree{};
  u32 entry_count{};
  u32 vector_dtype{static_cast<u32>(VectorDType::float32)};
  u32 vector_component_size{};
  u32 metadata_schema_version{kMetadataSchemaVersion};
  u32 node_size{};
  u32 vector_offset{};
  u32 vector_bytes{};
  u32 slot_incarnation_offset{};
  u32 hot_graph_version{};
  u32 hot_graph_entry_size{};
  u32 hot_graph_pointer_bytes{};
  u32 hot_graph_shard_bits{};
  u32 remote_ptr_format_version{kRemotePtrFormatVersion};
  u32 remote_ptr_alignment_log2{RemotePtr::OFFSET_ALIGNMENT_LOG2};
  u32 remote_ptr_offset_bits{RemotePtr::OFFSET_UNIT_BITS};
  u32 remote_ptr_shard_bits{RemotePtr::MEMORY_NODE_BITS};
  u32 remote_ptr_incarnation_bits{RemotePtr::INCARNATION_BITS};
  u32 static_incarnation{};
};

struct Entry {
  u64 remote_node{};
  u32 generation{};
  u32 reserved{};
};
#pragma pack(pop)

static_assert(sizeof(Header) == 160);
static_assert(sizeof(Entry) == 16);

inline u64 checksum(span<const byte_t> bytes) {
  u64 value = 1469598103934665603ULL;
  for (const byte_t byte : bytes) {
    value ^= static_cast<u8>(byte);
    value *= 1099511628211ULL;
  }
  return value;
}

inline u64 payload_bytes(u32 dim, u32 entry_count) {
  return static_cast<u64>(dim) * sizeof(f64) +
    static_cast<u64>(entry_count) * sizeof(Entry);
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

}  // namespace vamana::centroid_state
