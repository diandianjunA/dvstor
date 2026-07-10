#include "gpu_search/index_format.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace gpu_search::format {
namespace {

constexpr u64 kChecksumOffset = 1469598103934665603ULL;
constexpr u64 kChecksumPrime = 1099511628211ULL;
constexpr u64 kRemoteOffsetLimit = 1ull << 48;

void set_error(std::string* error, const std::string& value) {
  if (error != nullptr) *error = value;
}

template <class T>
bool section_in_file(u64 offset, u64 bytes, u64 file_bytes) {
  return bytes % sizeof(T) == 0 && offset <= file_bytes && bytes <= file_bytes - offset;
}

template <class T>
bool write_section(std::ofstream& output, u64 offset, const std::vector<T>& values) {
  if (values.empty()) return true;
  output.seekp(static_cast<std::streamoff>(offset));
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  return output.good();
}

template <class T>
bool read_section(std::ifstream& input, u64 offset, u64 bytes, std::vector<T>& values) {
  values.resize(static_cast<size_t>(bytes / sizeof(T)));
  if (values.empty()) return true;
  input.seekg(static_cast<std::streamoff>(offset));
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  return input.good();
}

u32 shard_bits_for(u32 shard_count) {
  u32 bits = 0;
  u32 capacity = 1;
  while (capacity < shard_count && bits < 31) {
    capacity <<= 1;
    ++bits;
  }
  return bits;
}

}  // namespace

u64 align_up(u64 value, u64 alignment) {
  if (alignment == 0) return value;
  const u64 remainder = value % alignment;
  if (remainder == 0) return value;
  if (value > std::numeric_limits<u64>::max() - (alignment - remainder)) return 0;
  return value + alignment - remainder;
}

u64 checksum64_initial() {
  return kChecksumOffset;
}

u64 checksum64_update(u64 state, const byte_t* data, size_t bytes) {
  for (size_t index = 0; index < bytes; ++index) {
    state ^= static_cast<u64>(data[index]);
    state *= kChecksumPrime;
  }
  return state;
}

u64 checksum64(const byte_t* data, size_t bytes) {
  return checksum64_update(checksum64_initial(), data, bytes);
}

bool validate_header(const Header& header, std::string* error) {
  if (header.magic != kMagic) {
    set_error(error, header.magic == kLegacyMagic
      ? "GPU tiered V3 is unsupported; run vamana_gpu_sidecar_converter to create V4 files"
      : "GPU tiered V4 manifest magic mismatch");
    return false;
  }
  if (header.version != kVersion || header.header_bytes != sizeof(Header)) {
    set_error(error, "unsupported GPU tiered manifest version");
    return false;
  }
  if (header.endian_marker != kEndianMarker) {
    set_error(error, "GPU tiered manifest byte order mismatch");
    return false;
  }
  if (header.dim == 0 || header.graph_degree == 0 || header.num_shards == 0 ||
      header.num_nodes == 0 || header.num_nodes >= (1ull << 30) ||
      header.num_nodes > std::numeric_limits<u32>::max() || header.base_generation == 0) {
    set_error(error, "GPU tiered V4 manifest has invalid dimensions");
    return false;
  }
  if (header.rabitq_code_bits < header.dim || header.rabitq_code_bits < 8 ||
      (header.rabitq_code_bits & (header.rabitq_code_bits - 1)) != 0 ||
      header.rabitq_entry_bytes != rabitq_entry_bytes(header.rabitq_code_bits)) {
    set_error(error, "GPU tiered V4 manifest has an invalid RaBitQ layout");
    return false;
  }
  if (header.graph_pointer_bytes != kCompactPointerBytes ||
      header.graph_entry_bytes <
        8 + static_cast<u64>(header.graph_degree) * kCompactPointerBytes ||
      header.graph_entry_bytes > kGraphCacheLineBytes ||
      header.graph_shard_bits != shard_bits_for(header.num_shards) ||
      header.graph_shard_bits >= 16) {
    set_error(error, "GPU tiered V4 requires the compact graph plane to fit one cache line");
    return false;
  }
  if (header.medoid_ordinal >= header.num_nodes ||
      header.shard_regions_bytes != header.num_shards * sizeof(ShardRegion) ||
      header.centroid_bytes != static_cast<u64>(header.dim) * sizeof(f32) ||
      header.entry_points_bytes == 0 ||
      header.entry_points_bytes > kMaxEntryPoints * sizeof(u32)) {
    set_error(error, "GPU tiered V4 manifest section cardinality mismatch");
    return false;
  }
  if (!section_in_file<ShardRegion>(header.shard_regions_offset,
                                    header.shard_regions_bytes, header.file_bytes) ||
      !section_in_file<f32>(header.centroid_offset,
                            header.centroid_bytes, header.file_bytes) ||
      !section_in_file<u32>(header.entry_points_offset,
                            header.entry_points_bytes, header.file_bytes)) {
    set_error(error, "GPU tiered V4 manifest section exceeds file bounds");
    return false;
  }
  return true;
}

bool validate_view(const View& view, std::string* error) {
  if (view.shards.size() != view.header.num_shards ||
      view.centroid.size() != view.header.dim || view.entry_points.empty() ||
      view.entry_points.size() > kMaxEntryPoints ||
      !std::all_of(view.centroid.begin(), view.centroid.end(), [](f32 value) {
        return std::isfinite(value);
      })) {
    set_error(error, "GPU tiered V4 view cardinality mismatch");
    return false;
  }
  u64 next_ordinal = 0;
  for (size_t shard_index = 0; shard_index < view.shards.size(); ++shard_index) {
    const ShardRegion& shard = view.shards[shard_index];
    const bool node_range_overflows = shard.node_base_offset > kRemoteOffsetLimit ||
      (shard.node_stride != 0 &&
       shard.node_count >
         (kRemoteOffsetLimit - shard.node_base_offset) / shard.node_stride);
    const bool graph_range_overflows = shard.graph_base_offset > kRemoteOffsetLimit ||
      (view.header.graph_entry_bytes != 0 &&
       shard.node_count >
         (kRemoteOffsetLimit - shard.graph_base_offset) / view.header.graph_entry_bytes);
    const bool code_range_overflows =
      shard.code_remote_offset > kRemoteOffsetLimit ||
      shard.code_bytes > kRemoteOffsetLimit - shard.code_remote_offset;
    const u64 node_end = node_range_overflows ? kRemoteOffsetLimit :
      shard.node_base_offset + shard.node_count * shard.node_stride;
    const u64 graph_end = graph_range_overflows ? kRemoteOffsetLimit :
      shard.graph_base_offset + shard.node_count * view.header.graph_entry_bytes;
    if (shard.memory_node != shard_index || shard.ordinal_base != next_ordinal ||
        shard.node_count == 0 || shard.node_base_offset != kNodeBaseOffset ||
        shard.node_stride == 0 || shard.graph_base_offset == 0 ||
        shard.dynamic_base_offset == 0 || shard.dynamic_record_bytes == 0 ||
        shard.dynamic_hot_offset == 0 ||
        node_range_overflows || graph_range_overflows || code_range_overflows ||
        node_end > shard.graph_base_offset || graph_end > shard.dynamic_base_offset ||
        shard.dynamic_hot_offset < shard.node_stride ||
        shard.dynamic_hot_offset > shard.dynamic_record_bytes ||
        view.header.graph_entry_bytes >
          shard.dynamic_record_bytes - shard.dynamic_hot_offset ||
        shard.code_remote_offset != align_up(shard.dynamic_base_offset, 64) ||
        shard.code_bytes != shard.node_count * view.header.rabitq_entry_bytes) {
      set_error(error, "GPU tiered V4 contains an invalid shard region");
      return false;
    }
    next_ordinal += shard.node_count;
  }
  if (next_ordinal != view.header.num_nodes) {
    set_error(error, "GPU tiered V4 shard ranges do not cover all nodes");
    return false;
  }
  for (u32 entry : view.entry_points) {
    if (entry >= view.header.num_nodes) {
      set_error(error, "GPU tiered V4 contains an invalid entry point");
      return false;
    }
  }
  return true;
}

bool write_file(const std::filesystem::path& path, const View& view,
                std::string* error) {
  Header header = view.header;
  header.magic = kMagic;
  header.version = kVersion;
  header.header_bytes = sizeof(Header);
  header.endian_marker = kEndianMarker;
  header.num_shards = static_cast<u32>(view.shards.size());
  u64 cursor = align_up(sizeof(Header), 64);
  header.shard_regions_offset = cursor;
  header.shard_regions_bytes = view.shards.size() * sizeof(ShardRegion);
  cursor = align_up(cursor + header.shard_regions_bytes, 64);
  header.centroid_offset = cursor;
  header.centroid_bytes = view.centroid.size() * sizeof(f32);
  cursor = align_up(cursor + header.centroid_bytes, 64);
  header.entry_points_offset = cursor;
  header.entry_points_bytes = view.entry_points.size() * sizeof(u32);
  header.file_bytes = cursor + header.entry_points_bytes;
  header.checksum = 0;
  if (!validate_header(header, error)) return false;
  View normalized = view;
  normalized.header = header;
  if (!validate_view(normalized, error)) return false;
  u64 checksum = checksum64(reinterpret_cast<const byte_t*>(&header), sizeof(header));
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(view.shards.data()),
    view.shards.size() * sizeof(ShardRegion));
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(view.centroid.data()),
    view.centroid.size() * sizeof(f32));
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(view.entry_points.data()),
    view.entry_points.size() * sizeof(u32));
  header.checksum = checksum;

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.good()) {
    set_error(error, "failed to create GPU tiered V4 manifest: " + path.string());
    return false;
  }
  output.seekp(static_cast<std::streamoff>(header.file_bytes - 1));
  output.put(0);
  output.seekp(0);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  const bool ok = output.good() &&
    write_section(output, header.shard_regions_offset, view.shards) &&
    write_section(output, header.centroid_offset, view.centroid) &&
    write_section(output, header.entry_points_offset, view.entry_points);
  if (!ok) set_error(error, "failed to write GPU tiered V4 manifest: " + path.string());
  return ok;
}

bool read_file(const std::filesystem::path& path, View& view, std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    set_error(error, "GPU tiered V4 manifest does not exist: " + path.string());
    return false;
  }
  input.seekg(0, std::ios::end);
  const u64 actual_bytes = static_cast<u64>(input.tellg());
  input.seekg(0);
  Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good()) {
    set_error(error, "GPU tiered V4 manifest header is truncated");
    return false;
  }
  if (!validate_header(header, error)) return false;
  if (actual_bytes != header.file_bytes) {
    set_error(error, "GPU tiered V4 manifest file size mismatch");
    return false;
  }
  const u64 stored_checksum = header.checksum;
  header.checksum = 0;
  u64 checksum = checksum64(reinterpret_cast<const byte_t*>(&header), sizeof(header));
  header.checksum = stored_checksum;
  View loaded;
  loaded.header = header;
  if (!read_section(input, header.shard_regions_offset, header.shard_regions_bytes,
                    loaded.shards) ||
      !read_section(input, header.centroid_offset, header.centroid_bytes, loaded.centroid) ||
      !read_section(input, header.entry_points_offset, header.entry_points_bytes,
                    loaded.entry_points) ||
      !validate_view(loaded, error)) {
    if (error != nullptr && error->empty()) {
      *error = "failed to read GPU tiered V4 manifest sections";
    }
    return false;
  }
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(loaded.shards.data()),
    loaded.shards.size() * sizeof(ShardRegion));
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(loaded.centroid.data()),
    loaded.centroid.size() * sizeof(f32));
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(loaded.entry_points.data()),
    loaded.entry_points.size() * sizeof(u32));
  if (checksum != stored_checksum) {
    set_error(error, "GPU tiered V4 manifest checksum mismatch");
    return false;
  }
  view = std::move(loaded);
  return true;
}

bool validate_code_header(const CodeHeader& header, std::string* error) {
  if (header.magic != kCodeMagic || header.version != kVersion ||
      header.header_bytes != sizeof(CodeHeader) || header.endian_marker != kEndianMarker) {
    set_error(error, "invalid GPU V4 code sidecar header");
    return false;
  }
  if (header.entry_count == 0 || header.node_size == 0 || header.remote_offset == 0 ||
      header.code_bits < 8 || (header.code_bits & (header.code_bits - 1)) != 0 ||
      header.entry_bytes != rabitq_entry_bytes(header.code_bits) ||
      header.payload_bytes != header.entry_count * header.entry_bytes) {
    set_error(error, "invalid GPU V4 code sidecar dimensions");
    return false;
  }
  CodeHeader copy = header;
  const u64 stored_checksum = copy.header_checksum;
  copy.header_checksum = 0;
  if (checksum64(reinterpret_cast<const byte_t*>(&copy), sizeof(copy)) != stored_checksum) {
    set_error(error, "GPU V4 code sidecar header checksum mismatch");
    return false;
  }
  return true;
}

bool read_code_header(const std::filesystem::path& path, CodeHeader& header,
                      std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    set_error(error, "GPU V4 code sidecar does not exist: " + path.string());
    return false;
  }
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good()) {
    set_error(error, "GPU V4 code sidecar header is truncated: " + path.string());
    return false;
  }
  if (!validate_code_header(header, error)) return false;
  const u64 expected = sizeof(CodeHeader) + header.payload_bytes;
  if (std::filesystem::file_size(path) != expected) {
    set_error(error, "GPU V4 code sidecar file size mismatch: " + path.string());
    return false;
  }
  return true;
}

bool write_code_header(std::ostream& output, const CodeHeader& source, std::string* error) {
  CodeHeader header = source;
  header.magic = kCodeMagic;
  header.version = kVersion;
  header.header_bytes = sizeof(CodeHeader);
  header.endian_marker = kEndianMarker;
  header.header_checksum = 0;
  header.header_checksum = checksum64(reinterpret_cast<const byte_t*>(&header), sizeof(header));
  if (!validate_code_header(header, error)) return false;
  output.seekp(0);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  if (!output.good()) {
    set_error(error, "failed to write GPU V4 code sidecar header");
    return false;
  }
  return true;
}

bool ordinal_to_remote(const View& view, u32 ordinal, RemotePtr& pointer) {
  if (ordinal >= view.header.num_nodes) return false;
  const auto it = std::upper_bound(
    view.shards.begin(), view.shards.end(), ordinal,
    [](u32 value, const ShardRegion& shard) { return value < shard.ordinal_base; });
  if (it == view.shards.begin()) return false;
  const ShardRegion& shard = *(it - 1);
  const u64 slot = static_cast<u64>(ordinal) - shard.ordinal_base;
  if (slot >= shard.node_count) return false;
  pointer = RemotePtr{shard.memory_node, shard.node_base_offset + slot * shard.node_stride};
  return true;
}

bool remote_to_ordinal(const View& view, RemotePtr pointer, u32& ordinal) {
  if (pointer.is_null() || pointer.memory_node() >= view.shards.size()) return false;
  const ShardRegion& shard = view.shards[pointer.memory_node()];
  if (pointer.byte_offset() < shard.node_base_offset || shard.node_stride == 0) return false;
  const u64 relative = pointer.byte_offset() - shard.node_base_offset;
  if (relative % shard.node_stride != 0) return false;
  const u64 slot = relative / shard.node_stride;
  if (slot >= shard.node_count || shard.ordinal_base + slot >= (1ull << 30)) return false;
  ordinal = static_cast<u32>(shard.ordinal_base + slot);
  return true;
}

}  // namespace gpu_search::format
