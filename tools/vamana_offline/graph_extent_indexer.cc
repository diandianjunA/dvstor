#include "tools/vamana_offline/graph_extent_indexer.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "common/constants.hh"
#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "vamana/hot_graph.hh"

namespace tools::vamana_offline {
namespace {

struct GraphLayout {
  u32 degree{};
  u32 capacity{};
  u32 shards{};
  u32 shard_bits{};
  u32 node_bytes{};
  u32 entry_bytes{};
  u64 nodes{};
  u64 build_fingerprint{};
  vec<u64> entry_counts;
  vec<u64> header_offsets;
  vec<u64> entry_offsets;
  vec<u64> dynamic_offsets;
  vec<u64> shard_fingerprints;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

u32 shard_bits_for(u32 shard_count) {
  u32 bits = 0;
  u32 capacity = 1;
  while (capacity < shard_count) {
    capacity <<= 1;
    ++bits;
  }
  return bits;
}

u32 graph_capacity_for_degree(u32 degree) {
  const u32 provisional =
    std::min<u32>(15, std::max<u32>(2, (degree + 15) / 16));
  return degree + provisional;
}

template <typename T>
vec<T> required_array(
    const nlohmann::json& metadata, const char* name, u32 count) {
  const auto found = metadata.find(name);
  if (found == metadata.end() || !found->is_array()) {
    fail(std::string{"missing graph extent metadata array: "} + name);
  }
  vec<T> result = found->get<vec<T>>();
  if (result.size() != count) {
    fail(std::string{"invalid graph extent metadata array cardinality: "} +
         name);
  }
  return result;
}

GraphLayout read_layout(const filepath_t& prefix) {
  const filepath_t metadata_path{prefix.string() + ".meta.json"};
  std::ifstream input(metadata_path);
  if (!input.good()) {
    fail("missing schema-16 metadata: " + metadata_path.string());
  }
  nlohmann::json metadata;
  try {
    input >> metadata;
  } catch (const std::exception& error) {
    fail(
      "cannot parse schema-16 metadata " + metadata_path.string() +
      ": " + error.what());
  }
  if (metadata.value("schema_version", 0u) !=
        gpu_search::format::kMetadataSchemaVersion ||
      metadata.value("node_layout", str{}) != "plain" ||
      metadata.value("storage_format", str{}) !=
        "vamana_tagged_v2" ||
      metadata.value("remote_ptr_format", str{}) !=
        "tagged_inc24_shard6_off34x16_v1" ||
      metadata.value("navigation_format", str{}) !=
        "opq_pq_graph_v1") {
    fail(
      "graph extent indexer requires a schema-16 tagged GPU graph");
  }

  GraphLayout layout;
  layout.degree = metadata.at("R").get<u32>();
  layout.shards = metadata.at("num_memory_nodes").get<u32>();
  layout.shard_bits = metadata.at("hot_graph_shard_bits").get<u32>();
  layout.node_bytes = metadata.at("node_size").get<u32>();
  layout.entry_bytes = metadata.at("hot_graph_entry_size").get<u32>();
  layout.nodes = metadata.at("num_vectors").get<u64>();
  layout.build_fingerprint =
    metadata.at("index_build_fingerprint").get<u64>();
  if (layout.degree == 0 || layout.degree > kMaxSupportedGraphDegree ||
      layout.shards == 0 ||
      layout.shards > RemotePtr::MEMORY_NODE_MASK + 1 ||
      layout.shard_bits != shard_bits_for(layout.shards) ||
      layout.node_bytes == 0 ||
      layout.node_bytes % RemotePtr::OFFSET_ALIGNMENT != 0 ||
      layout.nodes == 0 || layout.nodes > kMaxGpuNavigationNodes ||
      layout.build_fingerprint == 0 ||
      metadata.value("hot_graph_pointer_bytes", 0u) !=
        vamana::hot_graph::kCompactPointerBytes) {
    fail("schema-16 metadata has an invalid graph extent layout");
  }
  layout.capacity = graph_capacity_for_degree(layout.degree);
  const u64 expected_entry_bytes =
    vamana::hot_graph::kTaggedNeighborBaseOffset +
    static_cast<u64>(layout.capacity) *
      vamana::hot_graph::kCompactPointerBytes;
  if (expected_entry_bytes != layout.entry_bytes ||
      layout.entry_bytes > gpu_search::format::kMaxGraphEntryBytes) {
    fail(
      "schema-16 graph entry size does not match degree plus "
      "provisional capacity");
  }

  layout.entry_counts =
    required_array<u64>(metadata, "hot_graph_entry_counts", layout.shards);
  layout.header_offsets =
    required_array<u64>(metadata, "hot_graph_header_offsets", layout.shards);
  layout.entry_offsets =
    required_array<u64>(metadata, "hot_graph_offsets", layout.shards);
  layout.dynamic_offsets = required_array<u64>(
    metadata, "hot_graph_dynamic_base_offsets", layout.shards);
  layout.shard_fingerprints = required_array<u64>(
    metadata, "shard_build_fingerprints", layout.shards);

  u64 total = 0;
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    const u64 count = layout.entry_counts[shard];
    if (count >
          (std::numeric_limits<u64>::max() -
           layout.entry_offsets[shard]) / layout.entry_bytes ||
        layout.header_offsets[shard] >
          std::numeric_limits<u64>::max() -
            sizeof(vamana::hot_graph::Header) ||
        layout.header_offsets[shard] +
            sizeof(vamana::hot_graph::Header) >
          layout.entry_offsets[shard] ||
        layout.entry_offsets[shard] + count * layout.entry_bytes >
          layout.dynamic_offsets[shard] ||
        layout.shard_fingerprints[shard] == 0 ||
        total > std::numeric_limits<u64>::max() - count) {
      fail("schema-16 metadata contains an invalid graph shard range");
    }
    total += count;
  }
  if (total != layout.nodes) {
    fail("schema-16 graph shard counts do not cover num_vectors");
  }
  return layout;
}

void read_exact(
    std::istream& input, void* destination, size_t bytes,
    const std::string& operation) {
  if (bytes >
      static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    fail(operation + " exceeds host I/O limits");
  }
  input.read(
    reinterpret_cast<char*>(destination),
    static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(input.gcount()) != bytes) {
    fail("short read while " + operation);
  }
}

u32 load_u32(const byte_t* data) {
  u32 value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

u64 load_u64(const byte_t* data) {
  u64 value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

void validate_pointer(
    u64 raw, const GraphLayout& layout, u32 source_shard,
    u64 source_slot, u32 neighbor) {
  const RemotePtr pointer{raw};
  if (pointer.is_null() || !pointer.is_well_formed() ||
      pointer.incarnation() != 0 ||
      pointer.memory_node() >= layout.shards ||
      pointer.byte_offset() < vamana::hot_graph::kNodeBaseOffset) {
    fail(
      "invalid active graph pointer at shard=" +
      std::to_string(source_shard) + " slot=" +
      std::to_string(source_slot) + " neighbor=" +
      std::to_string(neighbor));
  }
  const u64 relative =
    pointer.byte_offset() - vamana::hot_graph::kNodeBaseOffset;
  const u64 target_slot = relative / layout.node_bytes;
  if (relative % layout.node_bytes != 0 ||
      target_slot >= layout.entry_counts[pointer.memory_node()]) {
    fail(
      "out-of-range static graph pointer at shard=" +
      std::to_string(source_shard) + " slot=" +
      std::to_string(source_slot) + " neighbor=" +
      std::to_string(neighbor));
  }
}

u8 validate_record_and_class(
    const byte_t* record, const GraphLayout& layout,
    u32 shard, u64 slot) {
  const u32 stable_count = record[0];
  const u32 provisional_count =
    vamana::hot_graph::provisional_count(record);
  const u32 provisional_capacity = layout.capacity - layout.degree;
  const u16 stored_checksum =
    vamana::hot_graph::load_u16_le(record + 2);
  if ((record[1] & vamana::hot_graph::kDeletedFlag) != 0 ||
      (record[1] & 0x0eu) != 0 ||
      stable_count > layout.degree ||
      provisional_count > provisional_capacity ||
      stable_count + provisional_count > layout.capacity ||
      load_u32(record + 4) != 0 ||
      load_u32(record + 8) != 0 ||
      load_u32(record + 12) != 0 ||
      stored_checksum !=
        vamana::hot_graph::checksum16(record, layout.entry_bytes)) {
    fail(
      "invalid immutable graph record at shard=" +
      std::to_string(shard) + " slot=" + std::to_string(slot));
  }
  const u32 live_count = stable_count + provisional_count;
  for (u32 neighbor = 0; neighbor < layout.capacity; ++neighbor) {
    const u64 raw = load_u64(
      record + vamana::hot_graph::neighbor_offset(neighbor));
    if (neighbor < live_count) {
      validate_pointer(raw, layout, shard, slot, neighbor);
    } else if (raw != 0) {
      fail(
        "non-null inactive graph pointer at shard=" +
        std::to_string(shard) + " slot=" + std::to_string(slot) +
        " neighbor=" + std::to_string(neighbor));
    }
  }
  const u32 extent_class =
    gpu_search::format::graph_extent_class(live_count);
  if (extent_class > std::numeric_limits<u8>::max()) {
    fail("graph extent class exceeds the u8 sidecar format");
  }
  return static_cast<u8>(extent_class);
}

void validate_shard_envelope(
    std::ifstream& input, const filepath_t& path,
    const GraphLayout& layout, u32 shard) {
  std::error_code error;
  const u64 file_bytes = std::filesystem::file_size(path, error);
  if (error || file_bytes < layout.dynamic_offsets[shard]) {
    fail("graph shard is truncated: " + path.string());
  }
  std::array<byte_t, 16> prefix{};
  input.seekg(0);
  read_exact(input, prefix.data(), prefix.size(), "reading graph shard identity");
  if (load_u64(prefix.data()) != layout.dynamic_offsets[shard] ||
      load_u64(prefix.data() + sizeof(u64)) !=
        layout.shard_fingerprints[shard]) {
    fail("graph shard does not match schema-16 metadata: " + path.string());
  }
  vamana::hot_graph::Header header;
  input.seekg(static_cast<std::streamoff>(layout.header_offsets[shard]));
  read_exact(input, &header, sizeof(header), "reading compact graph header");
  if (header.magic != vamana::hot_graph::kMagic ||
      header.version != vamana::hot_graph::kVersion3 ||
      header.header_bytes != sizeof(vamana::hot_graph::Header) ||
      header.entry_bytes != layout.entry_bytes ||
      header.max_degree != layout.degree ||
      header.compact_pointer_bytes !=
        vamana::hot_graph::kCompactPointerBytes ||
      header.compact_pointer_shard_bits != layout.shard_bits ||
      header.flags != 0 ||
      header.entry_count != layout.entry_counts[shard] ||
      header.node_base_offset != vamana::hot_graph::kNodeBaseOffset ||
      header.reserved0 != layout.dynamic_offsets[shard] ||
      header.reserved1 == 0 ||
      header.reserved2 != layout.node_bytes) {
    fail("compact graph header does not match metadata: " + path.string());
  }
}

struct TemporaryFile {
  filepath_t path;
  bool published{};

  ~TemporaryFile() {
    if (!published && !path.empty()) {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  }
};

filepath_t make_temporary_path(const filepath_t& output) {
  const u64 nonce = static_cast<u64>(
    std::chrono::high_resolution_clock::now()
      .time_since_epoch().count());
  return filepath_t{
    output.string() + ".tmp." + std::to_string(nonce)};
}

}  // namespace

GraphExtentIndexResult build_graph_extent_index(
    const GraphExtentIndexOptions& options) {
  if (options.index_prefix.empty()) {
    throw std::invalid_argument("index prefix is required");
  }
  if (options.chunk_records == 0) {
    throw std::invalid_argument("chunk-records must be greater than zero");
  }
  const GraphLayout layout = read_layout(options.index_prefix);
  const filepath_t output = options.output.empty()
    ? index_path::graph_extent_file(options.index_prefix)
    : options.output;
  if (output.empty()) {
    throw std::invalid_argument("graph extent output path is empty");
  }
  if (std::filesystem::exists(output) && !options.overwrite) {
    throw std::runtime_error(
      "graph extent sidecar already exists; pass --overwrite to replace it: " +
      output.string());
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  TemporaryFile temporary{make_temporary_path(output)};
  if (std::filesystem::exists(temporary.path)) {
    throw std::runtime_error(
      "graph extent temporary output already exists: " +
      temporary.path.string());
  }
  std::fstream extent(
    temporary.path,
    std::ios::binary | std::ios::in | std::ios::out |
      std::ios::trunc);
  if (!extent.good()) {
    throw std::runtime_error(
      "failed to create graph extent sidecar: " +
      temporary.path.string());
  }
  const gpu_search::format::GraphExtentHeader placeholder;
  extent.write(
    reinterpret_cast<const char*>(&placeholder), sizeof(placeholder));
  if (!extent.good()) {
    fail("failed to reserve the graph extent sidecar header");
  }

  const u32 chunk_records = options.chunk_records;
  if (chunk_records >
      std::numeric_limits<size_t>::max() / layout.entry_bytes) {
    fail("graph extent chunk allocation overflows size_t");
  }
  vec<byte_t> records(
    static_cast<size_t>(chunk_records) * layout.entry_bytes);
  vec<u8> classes(chunk_records);
  u64 payload_checksum = gpu_search::format::checksum64_initial();
  u64 nodes_written = 0;
  u64 graph_bytes_validated = 0;
  u32 maximum_class = 0;
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    const filepath_t shard_path = index_path::shard_file(
      options.index_prefix, shard + 1, layout.shards);
    std::ifstream input(shard_path, std::ios::binary);
    if (!input.good()) {
      fail("missing graph shard: " + shard_path.string());
    }
    validate_shard_envelope(input, shard_path, layout, shard);
    input.seekg(
      static_cast<std::streamoff>(layout.entry_offsets[shard]));
    for (u64 base = 0; base < layout.entry_counts[shard];
         base += chunk_records) {
      const u32 count = static_cast<u32>(std::min<u64>(
        chunk_records, layout.entry_counts[shard] - base));
      const size_t bytes =
        static_cast<size_t>(count) * layout.entry_bytes;
      read_exact(
        input, records.data(), bytes,
        "reading complete compact graph records");
      for (u32 index = 0; index < count; ++index) {
        classes[index] = validate_record_and_class(
          records.data() + static_cast<size_t>(index) *
            layout.entry_bytes,
          layout, shard, base + index);
        maximum_class =
          std::max(maximum_class, static_cast<u32>(classes[index]));
      }
      extent.write(
        reinterpret_cast<const char*>(classes.data()),
        static_cast<std::streamsize>(count));
      if (!extent.good()) {
        fail("failed to write graph extent sidecar payload");
      }
      payload_checksum = gpu_search::format::checksum64_update(
        payload_checksum, classes.data(), count);
      nodes_written += count;
      graph_bytes_validated += bytes;
    }
  }
  if (nodes_written != layout.nodes ||
      graph_bytes_validated != layout.nodes * layout.entry_bytes) {
    fail("graph extent sidecar cardinality mismatch");
  }

  gpu_search::format::GraphExtentHeader header{
    .graph_entry_bytes = layout.entry_bytes,
    .graph_entry_capacity = layout.capacity,
    .num_shards = layout.shards,
    .num_nodes = layout.nodes,
    .payload_bytes = layout.nodes,
    .build_fingerprint = layout.build_fingerprint,
    .payload_checksum = payload_checksum,
  };
  std::string header_error;
  if (!gpu_search::format::write_graph_extent_header(
        extent, header, &header_error)) {
    fail(header_error);
  }
  extent.flush();
  if (!extent.good()) {
    fail("failed to flush graph extent sidecar");
  }
  extent.close();
  if (extent.fail()) {
    fail("failed to close graph extent sidecar");
  }
  std::error_code size_error;
  const u64 actual_bytes =
    std::filesystem::file_size(temporary.path, size_error);
  if (size_error ||
      actual_bytes != sizeof(header) + layout.nodes) {
    fail("temporary graph extent sidecar has an incomplete payload");
  }
  if (!options.overwrite && std::filesystem::exists(output)) {
    fail(
      "graph extent sidecar appeared during generation; refusing to "
      "overwrite it");
  }
  std::error_code rename_error;
  std::filesystem::rename(temporary.path, output, rename_error);
  if (rename_error) {
    fail(
      "failed to publish graph extent sidecar " + output.string() +
      ": " + rename_error.message());
  }
  temporary.published = true;
  return {
    .output = output,
    .node_count = layout.nodes,
    .payload_bytes = layout.nodes,
    .payload_checksum = payload_checksum,
    .graph_bytes_validated = graph_bytes_validated,
    .maximum_class = maximum_class,
  };
}

}  // namespace tools::vamana_offline
