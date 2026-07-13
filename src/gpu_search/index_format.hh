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

inline constexpr std::array<char, 8> kCodeMagic{'D', 'V', 'G', 'P', 'U', 'C', '5', '\0'};
inline constexpr u32 kVersion = 5;
inline constexpr u32 kEndianMarker = 0x01020304;
inline constexpr u32 kMaxEntryPoints = 512;
inline constexpr u32 kGraphCacheLineBytes = 512;
inline constexpr u32 kCompactPointerBytes = 5;
inline constexpr u64 kNodeBaseOffset = 16;
inline constexpr u32 kMetadataSchemaVersion = 15;
inline constexpr u32 kStorageControlBytes = 4096;
inline constexpr u64 kStorageControlMagic = 0x314c525443565344ULL;  // "DSVCTRL1"
inline constexpr u32 kStorageControlVersion = 2;
inline constexpr u32 kMaxComputeClients = 64;

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
  u32 medoid_ordinal{};
  u32 reserved0{};
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

struct alignas(64) StorageControlBlock {
  u64 magic{kStorageControlMagic};
  u32 version{kStorageControlVersion};
  u32 header_bytes{sizeof(StorageControlBlock)};
  u32 shard_id{};
  u32 dynamic_record_bytes{};
  u32 dynamic_hot_offset{};
  u32 dynamic_code_offset{};
  u32 code_bytes{};
  u32 compute_client_count{};
  u32 reserved0{};
  u64 next_maintenance_sequence{1};
  u64 durable_maintenance_sequence{};
  u64 dynamic_high_watermark{};
  u64 reclaim_pending_nodes{};
  u64 reclaim_reused_nodes{};
  u64 reserved1{};
  std::array<u64, kMaxComputeClients> reclaim_ack_sequences{};
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
  u32 reserved0{};
  u64 entry_count{};
  u64 remote_offset{};
  u64 payload_bytes{};
  u64 model_checksum{};
  u64 payload_checksum{};
  u64 header_checksum{};
  std::array<u64, 4> reserved{};
};

static_assert(sizeof(ShardRegion) == 88);
static_assert(sizeof(StorageControlBlock) == 640);
static_assert(sizeof(StorageControlBlock) <= kStorageControlBytes);
static_assert(sizeof(CodeHeader) == 120);

struct View {
  NavigationLayout layout{};
  std::vector<ShardRegion> shards;
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

bool validate_layout(const NavigationLayout& layout, std::string* error = nullptr);
bool validate_view(const View& view, std::string* error = nullptr);
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
