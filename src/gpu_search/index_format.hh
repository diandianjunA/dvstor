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

inline constexpr std::array<char, 8> kMagic{'D', 'V', 'G', 'P', 'U', 'V', '4', '\0'};
inline constexpr std::array<char, 8> kLegacyMagic{'D', 'V', 'G', 'P', 'U', 'I', 'D', 'X'};
inline constexpr std::array<char, 8> kCodeMagic{'D', 'V', 'G', 'P', 'U', 'C', '4', '\0'};
inline constexpr u32 kVersion = 4;
inline constexpr u32 kEndianMarker = 0x01020304;
inline constexpr u32 kMaxEntryPoints = 512;
inline constexpr u32 kGraphCacheLineBytes = 512;
inline constexpr u32 kCompactPointerBytes = 5;
inline constexpr u64 kNodeBaseOffset = 16;

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

struct Header {
  std::array<char, 8> magic{kMagic};
  u32 version{kVersion};
  u32 header_bytes{sizeof(Header)};
  u32 endian_marker{kEndianMarker};
  u32 dim{};
  u32 graph_degree{};
  u32 vector_dtype{};
  u32 rabitq_code_bits{};
  u32 rabitq_entry_bytes{};
  u32 num_shards{};
  u32 graph_entry_bytes{};
  u32 graph_pointer_bytes{kCompactPointerBytes};
  u32 graph_shard_bits{};
  u32 medoid_ordinal{};
  u32 reserved0{};
  u64 num_nodes{};
  u64 base_generation{1};
  u64 shard_regions_offset{};
  u64 shard_regions_bytes{};
  u64 centroid_offset{};
  u64 centroid_bytes{};
  u64 entry_points_offset{};
  u64 entry_points_bytes{};
  u64 file_bytes{};
  u64 checksum{};
  std::array<u64, 4> reserved{};
};

struct ShardRegion {
  u64 ordinal_base{};
  u64 node_count{};
  u64 node_base_offset{kNodeBaseOffset};
  u64 node_stride{};
  u64 graph_base_offset{};
  u64 dynamic_base_offset{};
  u64 code_remote_offset{};
  u64 code_bytes{};
  u32 memory_node{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
  u32 reserved{};

  bool operator==(const ShardRegion&) const = default;
};

struct CodeHeader {
  std::array<char, 8> magic{kCodeMagic};
  u32 version{kVersion};
  u32 header_bytes{sizeof(CodeHeader)};
  u32 endian_marker{kEndianMarker};
  u32 memory_node{};
  u32 code_bits{};
  u32 entry_bytes{};
  u32 node_size{};
  u32 reserved0{};
  u64 entry_count{};
  u64 remote_offset{};
  u64 payload_bytes{};
  u64 payload_checksum{};
  u64 header_checksum{};
  std::array<u64, 4> reserved{};
};

static_assert(sizeof(Header) == 176);
static_assert(sizeof(ShardRegion) == 80);
static_assert(sizeof(CodeHeader) == 112);

struct View {
  Header header{};
  std::vector<ShardRegion> shards;
  std::vector<f32> centroid;
  std::vector<u32> entry_points;
};

struct SynthesisOptions {
  u32 entry_points{};
  u64 seed{1234};
};

u64 align_up(u64 value, u64 alignment);
u64 checksum64(const byte_t* data, size_t bytes);
u64 checksum64_update(u64 state, const byte_t* data, size_t bytes);
u64 checksum64_initial();

bool validate_header(const Header& header, std::string* error = nullptr);
bool validate_view(const View& view, std::string* error = nullptr);
bool write_file(const std::filesystem::path& path, const View& view,
                std::string* error = nullptr);
bool read_file(const std::filesystem::path& path, View& view,
               std::string* error = nullptr);
bool synthesize_distributed_view(
  const std::filesystem::path& index_prefix, View& view,
  const SynthesisOptions& options = {},
  bool* used_anchor_entry_points = nullptr,
  std::string* error = nullptr);

bool validate_code_header(const CodeHeader& header, std::string* error = nullptr);
bool read_code_header(const std::filesystem::path& path, CodeHeader& header,
                      std::string* error = nullptr);
bool write_code_header(std::ostream& output, const CodeHeader& header,
                       std::string* error = nullptr);

bool ordinal_to_remote(const View& view, u32 ordinal, RemotePtr& pointer);
bool remote_to_ordinal(const View& view, RemotePtr pointer, u32& ordinal);

}  // namespace gpu_search::format
