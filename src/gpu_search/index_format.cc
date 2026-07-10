#include "gpu_search/index_format.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <type_traits>

namespace gpu_search::format {
namespace {

template <class T>
bool section_in_file(u64 offset, u64 bytes, u64 file_bytes) {
  if (bytes % sizeof(T) != 0) return false;
  return offset <= file_bytes && bytes <= file_bytes - offset;
}

void set_error(std::string* error, const std::string& value) {
  if (error != nullptr) *error = value;
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

}  // namespace

u64 align_up(u64 value, u64 alignment) {
  if (alignment == 0) return value;
  const u64 remainder = value % alignment;
  if (remainder == 0) return value;
  if (value > std::numeric_limits<u64>::max() - (alignment - remainder)) return 0;
  return value + alignment - remainder;
}

u64 checksum64(const byte_t* data, size_t bytes) {
  constexpr u64 kOffset = 1469598103934665603ULL;
  constexpr u64 kPrime = 1099511628211ULL;
  u64 hash = kOffset;
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= static_cast<u64>(data[i]);
    hash *= kPrime;
  }
  return hash;
}

bool validate_header(const Header& header, std::string* error) {
  if (header.magic != kMagic) {
    set_error(error, "GPU tiered index magic mismatch");
    return false;
  }
  if (header.version != kVersion || header.header_bytes != sizeof(Header)) {
    set_error(error, "unsupported GPU tiered index version");
    return false;
  }
  if (header.endian_marker != kEndianMarker) {
    set_error(error, "GPU tiered index byte order mismatch");
    return false;
  }
  if (header.page_bytes < sizeof(PageHeader) ||
      (header.page_bytes & (header.page_bytes - 1)) != 0) {
    set_error(error, "GPU graph page size must be a power of two");
    return false;
  }
  if (header.dim == 0 || header.graph_degree == 0 ||
      header.hot_degree == 0 || header.hot_degree > kMaxHotDegree ||
      header.hot_degree > header.graph_degree) {
    set_error(error, "invalid GPU tiered graph dimensions");
    return false;
  }
  if (header.id_encoding_bytes != 3 && header.id_encoding_bytes != 4) {
    set_error(error, "GPU tiered index ID width must be 3 or 4 bytes");
    return false;
  }
  const u32 code_storage_bytes = ((header.rabitq_code_bits + 7) / 8 + 3) & ~3u;
  if (header.rabitq_code_bits < header.dim ||
      (header.rabitq_code_bits & (header.rabitq_code_bits - 1)) != 0 ||
      header.rabitq_entry_bytes < code_storage_bytes + 2 * sizeof(f32)) {
    set_error(error, "GPU tiered index has invalid RaBitQ dimensions");
    return false;
  }
  if (header.num_nodes == 0 || header.num_shards == 0 || header.base_generation == 0) {
    set_error(error, "GPU tiered index has an empty topology");
    return false;
  }
  if (header.medoid_id >= header.num_nodes) {
    set_error(error, "GPU tiered index medoid is out of bounds");
    return false;
  }
  if (header.node_records_bytes != header.num_nodes * sizeof(NodeRecord) ||
      header.shard_regions_bytes != header.num_shards * sizeof(ShardRegion)) {
    set_error(error, "GPU tiered index section cardinality mismatch");
    return false;
  }
  if (!section_in_file<NodeRecord>(header.node_records_offset,
                                   header.node_records_bytes,
                                   header.file_bytes) ||
      !section_in_file<u32>(header.hot_neighbors_offset,
                            header.hot_neighbors_bytes,
                            header.file_bytes) ||
      !section_in_file<byte_t>(header.rabitq_offset,
                               header.rabitq_bytes,
                               header.file_bytes) ||
      !section_in_file<ShardRegion>(header.shard_regions_offset,
                                    header.shard_regions_bytes,
                                    header.file_bytes) ||
      !section_in_file<f32>(header.centroid_offset,
                            header.centroid_bytes,
                            header.file_bytes) ||
      !section_in_file<u32>(header.entry_points_offset,
                            header.entry_points_bytes,
                            header.file_bytes)) {
    set_error(error, "GPU tiered index section exceeds file bounds");
    return false;
  }
  return true;
}

bool validate_view(const View& view, std::string* error) {
  if (view.nodes.size() != view.header.num_nodes ||
      view.shards.size() != view.header.num_shards ||
      view.centroid.size() != view.header.dim || view.entry_points.empty() ||
      view.entry_points.size() > 512) {
    set_error(error, "GPU tiered index view cardinality mismatch");
    return false;
  }
  if (view.header.rabitq_entry_bytes == 0 ||
      view.rabitq_entries.size() !=
        view.nodes.size() * static_cast<size_t>(view.header.rabitq_entry_bytes)) {
    set_error(error, "GPU tiered index RaBitQ section cardinality mismatch");
    return false;
  }
  for (size_t shard_index = 0; shard_index < view.shards.size(); ++shard_index) {
    const ShardRegion& shard = view.shards[shard_index];
    if (shard.memory_node != shard_index || shard.graph_pages_bytes == 0 ||
        shard.graph_pages_offset % view.header.page_bytes != 0 ||
        shard.graph_pages_bytes % view.header.page_bytes != 0 ||
        shard.vector_stride == 0) {
      set_error(error, "GPU tiered index contains an invalid shard region");
      return false;
    }
  }
  for (const NodeRecord& node : view.nodes) {
    if (node.shard >= view.shards.size() || node.generation == 0 ||
        node.hot_neighbor_count > view.header.hot_degree ||
        node.hot_neighbor_begin > view.hot_neighbors.size() ||
        node.hot_neighbor_count > view.hot_neighbors.size() - node.hot_neighbor_begin ||
        node.cold_page_offset < view.shards[node.shard].graph_pages_offset ||
        node.cold_page_offset - view.shards[node.shard].graph_pages_offset >=
          view.shards[node.shard].graph_pages_bytes ||
        node.cold_page_offset % view.header.page_bytes != 0 ||
        node.cold_record_offset < sizeof(PageHeader) ||
        node.cold_record_offset + sizeof(PageNodeHeader) > view.header.page_bytes) {
      set_error(error, "GPU tiered index contains an invalid node record");
      return false;
    }
  }
  for (const u32 neighbor : view.hot_neighbors) {
    if (neighbor >= view.header.num_nodes) {
      set_error(error, "GPU tiered index contains a non-dense node ID");
      return false;
    }
  }
  for (const u32 entry : view.entry_points) {
    if (entry >= view.header.num_nodes) {
      set_error(error, "GPU tiered index contains an invalid entry point");
      return false;
    }
  }
  return true;
}

bool write_file(const std::filesystem::path& path, const View& view, std::string* error) {
  if (!validate_view(view, error)) return false;

  Header header = view.header;
  u64 cursor = align_up(sizeof(Header), 64);
  header.node_records_offset = cursor;
  header.node_records_bytes = view.nodes.size() * sizeof(NodeRecord);
  cursor = align_up(cursor + header.node_records_bytes, 64);
  header.hot_neighbors_offset = cursor;
  header.hot_neighbors_bytes = view.hot_neighbors.size() * sizeof(u32);
  cursor = align_up(cursor + header.hot_neighbors_bytes, 64);
  header.rabitq_offset = cursor;
  header.rabitq_bytes = view.rabitq_entries.size();
  cursor = align_up(cursor + header.rabitq_bytes, 64);
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
  header.checksum = checksum64(reinterpret_cast<const byte_t*>(&header), sizeof(header));

  if (!validate_header(header, error)) return false;
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.good()) {
    set_error(error, "failed to create GPU tiered index: " + path.string());
    return false;
  }
  if (header.file_bytes > 0) {
    output.seekp(static_cast<std::streamoff>(header.file_bytes - 1));
    output.put(0);
  }
  output.seekp(0);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  const bool ok = output.good() &&
    write_section(output, header.node_records_offset, view.nodes) &&
    write_section(output, header.hot_neighbors_offset, view.hot_neighbors) &&
    write_section(output, header.rabitq_offset, view.rabitq_entries) &&
    write_section(output, header.shard_regions_offset, view.shards) &&
    write_section(output, header.centroid_offset, view.centroid) &&
    write_section(output, header.entry_points_offset, view.entry_points);
  if (!ok) set_error(error, "failed to write GPU tiered index: " + path.string());
  return ok;
}

bool read_file(const std::filesystem::path& path, View& view, std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    set_error(error, "GPU tiered index does not exist: " + path.string());
    return false;
  }
  input.seekg(0, std::ios::end);
  const u64 actual_bytes = static_cast<u64>(input.tellg());
  input.seekg(0);
  Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good() || actual_bytes != header.file_bytes || !validate_header(header, error)) {
    if (error != nullptr && error->empty()) *error = "GPU tiered index file size mismatch";
    return false;
  }
  const u64 stored_checksum = header.checksum;
  header.checksum = 0;
  if (checksum64(reinterpret_cast<const byte_t*>(&header), sizeof(header)) != stored_checksum) {
    set_error(error, "GPU tiered index header checksum mismatch");
    return false;
  }
  header.checksum = stored_checksum;

  View loaded;
  loaded.header = header;
  if (!read_section(input, header.node_records_offset, header.node_records_bytes, loaded.nodes) ||
      !read_section(input, header.hot_neighbors_offset, header.hot_neighbors_bytes,
                    loaded.hot_neighbors) ||
      !read_section(input, header.rabitq_offset, header.rabitq_bytes, loaded.rabitq_entries) ||
      !read_section(input, header.shard_regions_offset, header.shard_regions_bytes, loaded.shards) ||
      !read_section(input, header.centroid_offset, header.centroid_bytes, loaded.centroid) ||
      !read_section(input, header.entry_points_offset, header.entry_points_bytes,
                    loaded.entry_points) ||
      !validate_view(loaded, error)) {
    if (error != nullptr && error->empty()) *error = "failed to read GPU tiered index sections";
    return false;
  }
  view = std::move(loaded);
  return true;
}

void encode_id(byte_t* destination, u32 id, IdEncoding encoding) {
  destination[0] = static_cast<byte_t>(id);
  destination[1] = static_cast<byte_t>(id >> 8);
  destination[2] = static_cast<byte_t>(id >> 16);
  if (encoding == IdEncoding::u32) destination[3] = static_cast<byte_t>(id >> 24);
}

u32 decode_id(const byte_t* source, IdEncoding encoding) {
  u32 id = static_cast<u32>(source[0]) |
           (static_cast<u32>(source[1]) << 8) |
           (static_cast<u32>(source[2]) << 16);
  if (encoding == IdEncoding::u32) id |= static_cast<u32>(source[3]) << 24;
  return id;
}

}  // namespace gpu_search::format
