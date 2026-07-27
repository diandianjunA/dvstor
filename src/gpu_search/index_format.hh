#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "remote_pointer.hh"

namespace gpu_search::format {

inline constexpr std::array<char, 8> kCodeMagic{'D', 'V', 'G', 'P', 'U', 'C', '6', '\0'};
inline constexpr u32 kVersion = 6;
inline constexpr u32 kEndianMarker = 0x01020304;
inline constexpr std::array<char, 8> kGraphExtentMagic{
  'D', 'V', 'G', 'E', 'X', 'T', '8', '\0'};
inline constexpr u32 kGraphExtentVersion = 1;
inline constexpr u32 kGraphExtentQuantum = 8;
inline constexpr u32 kGraphExtentClassBytes = sizeof(u8);
inline constexpr u32 kMaxGraphEntryBytes = 2048;
inline constexpr u32 kCompactPointerBytes = sizeof(u64);
inline constexpr u64 kNodeBaseOffset = 16;
inline constexpr u32 kMetadataSchemaVersion = 16;
inline constexpr u32 kStorageControlBytes = 4096;
inline constexpr u64 kStorageControlMagic = 0x314c525443565344ULL;  // "DSVCTRL1"
inline constexpr u32 kStorageControlVersion = 4;
inline constexpr u64 kStorageCentroidRouteDescriptorMagic =
  0x31445243565344ULL;  // "DSVCRD1"
inline constexpr u32 kStorageCentroidRouteDescriptorVersion = 1;
inline constexpr u64 kStorageCentroidRoutePublicationMagic =
  0x31545243565344ULL;  // "DSVCRT1"
inline constexpr u32 kStorageCentroidRoutePublicationVersion = 1;
inline constexpr u32 kStorageCentroidRouteMaxLiveEntries = 4;
inline constexpr u32 kStorageCentroidRouteLive = 1u;

enum class CentroidScalarType : u32 {
  float32 = 1,
  float64 = 2,
};


enum class QuantizerKind : u32 {
  opq_pq = 1,
};

struct NavigationLayout {
  u32 dim{};
  u32 graph_degree{};
  u32 vector_dtype{};
  u32 quantizer_kind{static_cast<u32>(QuantizerKind::opq_pq)};
  u32 pq_subquantizers{};
  u32 pq_bits{};
  u32 code_bytes{};
  u32 num_shards{};
  u32 graph_entry_bytes{};
  u32 graph_pointer_bytes{kCompactPointerBytes};
  u32 graph_shard_bits{};
  u64 num_nodes{};
  u64 base_generation{1};
  u64 model_checksum{};
};

struct ShardRegion {
  u64 ordinal_base{};
  u64 node_count{};
  u64 node_base_offset{kNodeBaseOffset};
  u64 node_stride{};
  u64 graph_base_offset{};
  u64 dynamic_base_offset{};
  u64 control_remote_offset{};
  u64 code_remote_offset{};
  u64 code_bytes{};
  u32 memory_node{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
  u32 dynamic_code_offset{};

  bool operator==(const ShardRegion&) const = default;
};

// Located in the fixed control block; the publication itself is a separately
// reserved, registered variable-length record near the storage-region tail.
// publication_bytes is derived from dim, scalar type and entry capacity rather
// than bounded by the 4 KiB control page.
struct alignas(64) StorageCentroidRouteDescriptor {
  u64 magic{kStorageCentroidRouteDescriptorMagic};
  u32 version{kStorageCentroidRouteDescriptorVersion};
  u32 descriptor_bytes{sizeof(StorageCentroidRouteDescriptor)};
  u64 remote_offset{};
  u64 publication_bytes{};
  u64 layout_version{1};
  u32 dim{};
  u32 centroid_scalar_type{
    static_cast<u32>(CentroidScalarType::float32)};
  u32 shard_count{};
  u32 live_entry_capacity{kStorageCentroidRouteMaxLiveEntries};
  u64 reserved{};
};

struct alignas(64) StorageControlBlock {
  u64 magic{kStorageControlMagic};
  u32 version{kStorageControlVersion};
  u32 header_bytes{sizeof(StorageControlBlock)};
  u32 shard_id{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
  u32 dynamic_code_offset{};
  u32 code_bytes{};
  u32 reserved0{};
  u64 next_maintenance_sequence{1};
  u64 durable_maintenance_sequence{};
  u64 dynamic_high_watermark{};
  u64 reclaim_pending_nodes{};
  u64 reclaim_reused_nodes{};
  u64 reserved1{};
  StorageCentroidRouteDescriptor centroid_route{};
};

struct StorageCentroidRouteEntry {
  u64 remote_node{};
  u32 generation{};
  u32 flags{kStorageCentroidRouteLive};
};

// Variable-length record prefix. centroid_offset and entries_offset are from
// the beginning of this header. sequence is a cache-line seqlock: odd while a
// writer replaces metadata+centroid+entries, even when stable. Compute readers
// probe the complete header once; an unchanged identity keeps the cached
// snapshot, while a changed identity triggers a complete-record read followed
// by one sequence verification read.
struct alignas(64) StorageCentroidRoutePublicationHeader {
  u64 sequence{};
  u64 magic{kStorageCentroidRoutePublicationMagic};
  u32 version{kStorageCentroidRoutePublicationVersion};
  u32 header_bytes{sizeof(StorageCentroidRoutePublicationHeader)};
  u32 shard_id{};
  u32 dim{};
  u32 centroid_scalar_type{
    static_cast<u32>(CentroidScalarType::float32)};
  u32 live_entry_count{};
  u32 live_entry_capacity{kStorageCentroidRouteMaxLiveEntries};
  u32 reserved0{};
  u64 total_bytes{};
  u64 shard_version{};
  u64 vector_count{};
  u64 centroid_offset{};
  u64 centroid_bytes{};
  u64 entries_offset{};
  u64 entries_bytes{};
  std::array<u64, 2> reserved{};
  u64 body_checksum{};
};


struct CodeHeader {
  std::array<char, 8> magic{kCodeMagic};
  u32 version{kVersion};
  u32 header_bytes{sizeof(CodeHeader)};
  u32 endian_marker{kEndianMarker};
  u32 memory_node{};
  u32 quantizer_kind{static_cast<u32>(QuantizerKind::opq_pq)};
  u32 code_bytes{};
  u32 node_size{};
  u32 vector_dtype{};
  u64 entry_count{};
  u64 remote_offset{};
  u64 payload_bytes{};
  u64 model_checksum{};
  u64 payload_checksum{};
  u64 header_checksum{};
  u64 build_fingerprint{};
  u64 shard_fingerprint{};
  std::array<u64, 2> reserved{};
};

// One byte per immutable base-node ordinal. A class c means that a one-sided
// graph READ must cover at least c groups of kGraphExtentQuantum consecutive
// RemotePtrs after the fixed 16-byte graph header. The final class is clamped
// to graph_entry_bytes, so layouts whose capacity is not a multiple of the
// quantum remain exactly representable.
struct GraphExtentHeader {
  std::array<char, 8> magic{kGraphExtentMagic};
  u32 version{kGraphExtentVersion};
  u32 header_bytes{sizeof(GraphExtentHeader)};
  u32 endian_marker{kEndianMarker};
  u32 extent_quantum{kGraphExtentQuantum};
  u32 class_bytes{kGraphExtentClassBytes};
  u32 graph_pointer_bytes{kCompactPointerBytes};
  u32 graph_entry_bytes{};
  u32 graph_entry_capacity{};
  u32 num_shards{};
  u32 reserved0{};
  u64 num_nodes{};
  u64 payload_bytes{};
  u64 build_fingerprint{};
  u64 payload_checksum{};
  u64 header_checksum{};
  std::array<u64, 5> reserved{};
};

static_assert(sizeof(ShardRegion) == 88);
static_assert(sizeof(StorageCentroidRouteDescriptor) == 64);
static_assert(offsetof(StorageControlBlock, centroid_route) == 128);
static_assert(sizeof(StorageControlBlock) == 192);
static_assert(sizeof(StorageControlBlock) <= kStorageControlBytes);
static_assert(sizeof(StorageCentroidRouteEntry) == 16);
static_assert(sizeof(StorageCentroidRoutePublicationHeader) == 128);
static_assert(sizeof(CodeHeader) == 120);
static_assert(sizeof(GraphExtentHeader) == 128);

u32 centroid_scalar_bytes(CentroidScalarType type);
u64 storage_centroid_route_publication_bytes(
  u32 dim, CentroidScalarType scalar_type, u32 live_entry_capacity);
bool validate_storage_centroid_route_descriptor(
  const StorageCentroidRouteDescriptor& descriptor,
  u32 expected_dim, u32 expected_shards,
  std::string* error = nullptr);
u64 storage_centroid_route_body_checksum(span<const byte_t> publication);
const void* storage_centroid_route_centroid_data(
  span<const byte_t> publication);
span<const StorageCentroidRouteEntry> storage_centroid_route_entries(
  span<const byte_t> publication);
bool prepare_storage_centroid_route_publication(
  span<byte_t> publication,
  u32 shard, u32 dim, CentroidScalarType scalar_type,
  u32 live_entry_capacity, u64 shard_version, u64 vector_count,
  const void* centroid_data,
  span<const StorageCentroidRouteEntry> live_entries,
  std::string* error = nullptr);
bool validate_storage_centroid_route_publication(
  span<const byte_t> publication,
  const StorageCentroidRouteDescriptor& descriptor,
  u32 expected_shard,
  std::string* error = nullptr);


struct View {
  NavigationLayout layout{};
  std::vector<ShardRegion> shards;
};

u64 align_up(u64 value, u64 alignment);
u64 checksum64(const byte_t* data, size_t bytes);
u64 checksum64_update(u64 state, const byte_t* data, size_t bytes);
u64 checksum64_initial();

bool validate_layout(const NavigationLayout& layout, std::string* error = nullptr);
bool validate_view(const View& view, std::string* error = nullptr);
bool synthesize_distributed_view(
  const std::filesystem::path& index_prefix, View& view,
  std::string* error = nullptr);

bool validate_code_header(const CodeHeader& header, std::string* error = nullptr);
bool read_code_header(const std::filesystem::path& path, CodeHeader& header,
                      std::string* error = nullptr);
bool write_code_header(std::ostream& output, const CodeHeader& header,
                       std::string* error = nullptr);

u32 graph_extent_class(u32 live_neighbors);
u32 graph_extent_read_bytes(u32 extent_class, u32 graph_entry_bytes);
bool validate_graph_extent_header(
  const GraphExtentHeader& header, std::string* error = nullptr);
bool read_graph_extent_header(
  const std::filesystem::path& path, GraphExtentHeader& header,
  std::string* error = nullptr);
bool read_graph_extent_sidecar(
  const std::filesystem::path& path, GraphExtentHeader& header,
  std::vector<u8>& classes, std::string* error = nullptr);
bool write_graph_extent_header(
  std::ostream& output, const GraphExtentHeader& header,
  std::string* error = nullptr);

bool ordinal_to_remote(const View& view, u32 ordinal, RemotePtr& pointer);
bool remote_to_ordinal(const View& view, RemotePtr pointer, u32& ordinal);

}  // namespace gpu_search::format
