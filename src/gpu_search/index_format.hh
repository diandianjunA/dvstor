#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "common/types.hh"
#include "common/vector_dtype.hh"

namespace gpu_search::format {

inline constexpr std::array<char, 8> kMagic{'D', 'V', 'G', 'P', 'U', 'I', 'D', 'X'};
inline constexpr u32 kVersion = 3;
inline constexpr u32 kEndianMarker = 0x01020304;
inline constexpr u32 kDefaultPageBytes = 4096;
inline constexpr u32 kMaxHotDegree = 32;
inline constexpr u32 kFlagDeleted = 1u;
inline constexpr u32 kShardPagesMagic = 0x47505344;

inline constexpr u32 rabitq_code_bytes(u32 code_bits) {
  return code_bits / 8;
}

inline constexpr u32 rabitq_code_storage_bytes(u32 code_bits) {
  return (rabitq_code_bytes(code_bits) + 3u) & ~3u;
}

inline constexpr u32 rabitq_norm_offset(u32 code_bits) {
  return rabitq_code_storage_bytes(code_bits);
}

inline constexpr u32 rabitq_error_offset(u32 code_bits) {
  return rabitq_norm_offset(code_bits) + sizeof(f32);
}

inline constexpr u32 rabitq_entry_bytes(u32 code_bits) {
  return (rabitq_error_offset(code_bits) + sizeof(f32) + 7u) & ~7u;
}

enum class IdEncoding : u8 {
  u24 = 3,
  u32 = 4,
};

struct alignas(64) Header {
  std::array<char, 8> magic{kMagic};
  u32 version{kVersion};
  u32 header_bytes{sizeof(Header)};
  u32 endian_marker{kEndianMarker};
  u32 page_bytes{kDefaultPageBytes};
  u32 dim{};
  u32 graph_degree{};
  u32 hot_degree{};
  u32 vector_dtype{};
  u32 rabitq_code_bits{};
  u32 rabitq_entry_bytes{};
  u32 id_encoding_bytes{4};
  u32 num_shards{};
  u32 medoid_id{};
  u32 reserved0{};
  u64 num_nodes{};
  u64 base_generation{1};
  u64 node_records_offset{};
  u64 node_records_bytes{};
  u64 hot_neighbors_offset{};
  u64 hot_neighbors_bytes{};
  u64 rabitq_offset{};
  u64 rabitq_bytes{};
  u64 shard_regions_offset{};
  u64 shard_regions_bytes{};
  u64 centroid_offset{};
  u64 centroid_bytes{};
  u64 entry_points_offset{};
  u64 entry_points_bytes{};
  u64 file_bytes{};
  u64 checksum{};
  std::array<u64, 2> reserved{};
};

struct NodeRecord {
  u64 remote_node{};
  u64 cold_page_offset{};
  u32 cold_record_offset{};
  u32 generation{1};
  u32 hot_neighbor_begin{};
  u16 hot_neighbor_count{};
  u16 shard{};
  u32 flags{};
};

struct ShardRegion {
  u64 graph_pages_offset{};
  u64 graph_pages_bytes{};
  u64 vector_region_offset{};
  u64 vector_stride{};
  u64 node_count{};
  u32 memory_node{};
  u32 reserved{};
};

struct PageHeader {
  u32 magic{0x47504750};
  u16 version{1};
  u16 node_count{};
  u32 payload_bytes{};
  u32 generation{1};
};

struct PageNodeHeader {
  u32 node_id{};
  u16 degree{};
  u16 flags{};
};

struct ShardPageFileHeader {
  u32 magic{kShardPagesMagic};
  u32 version{kVersion};
  u32 page_bytes{kDefaultPageBytes};
  u32 memory_node{};
  u64 remote_offset{};
  u64 data_bytes{};
  u64 checksum{};
};

struct View {
  Header header{};
  std::vector<NodeRecord> nodes;
  std::vector<u32> hot_neighbors;
  std::vector<byte_t> rabitq_entries;
  std::vector<ShardRegion> shards;
  std::vector<f32> centroid;
  std::vector<u32> entry_points;
};

u64 align_up(u64 value, u64 alignment);
u64 checksum64(const byte_t* data, size_t bytes);
bool validate_header(const Header& header, std::string* error = nullptr);
bool validate_view(const View& view, std::string* error = nullptr);
bool write_file(const std::filesystem::path& path, const View& view, std::string* error = nullptr);
bool read_file(const std::filesystem::path& path, View& view, std::string* error = nullptr);

void encode_id(byte_t* destination, u32 id, IdEncoding encoding);
u32 decode_id(const byte_t* source, IdEncoding encoding);

}  // namespace gpu_search::format
