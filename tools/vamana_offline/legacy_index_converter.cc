#include "tools/vamana_offline/legacy_index_converter.hh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

#include "common/constants.hh"
#include "common/index_path.hh"
#include "common/vector_dtype.hh"
#include "gpu_search/index_format.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "tools/vamana_offline/legacy_schema15.hh"
#include "tools/vamana_offline/pq_indexer.hh"
#include "vamana/centroid_state.hh"
#include "vamana/hot_graph.hh"
#include "vamana/idmap.hh"
#include "vamana/vamana_node.hh"

namespace tools::vamana_offline {
namespace {

namespace legacy = legacy_schema15;

struct Layout {
  u32 dim{};
  u32 degree{};
  u32 shards{};
  VectorDType dtype{VectorDType::float32};
  u32 vector_bytes{};
  u32 old_node_bytes{};
  u32 old_graph_bytes{};
  u32 shard_bits{};
  u64 node_count{};
  vec<u64> counts;
  vec<u64> old_header_offsets;
  vec<u64> old_graph_offsets;
  vec<u64> old_dynamic_offsets;
  vec<u64> new_header_offsets;
  vec<u64> new_graph_offsets;
  vec<u64> new_dynamic_offsets;
  vec<u64> shard_fingerprints;
  u64 build_fingerprint{};
  u32 subquantizers{};
  filepath_t model_file;
};

struct OutputCleanup {
  vec<filepath_t> paths;
  bool active{true};
  ~OutputCleanup() {
    if (!active) return;
    for (const auto& path : paths) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  }
};

[[noreturn]] void fail(const str& message) {
  throw std::runtime_error("legacy schema-15 conversion: " + message);
}

u64 align_up(u64 value, u64 alignment) {
  if (alignment == 0 || value >
      std::numeric_limits<u64>::max() - (alignment - 1)) {
    fail("layout alignment overflow");
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

u64 checked_add(u64 lhs, u64 rhs, const char* what) {
  if (lhs > std::numeric_limits<u64>::max() - rhs) fail(what);
  return lhs + rhs;
}

u64 checked_mul(u64 lhs, u64 rhs, const char* what) {
  if (lhs != 0 && rhs > std::numeric_limits<u64>::max() / lhs) fail(what);
  return lhs * rhs;
}

u64 file_bytes_or_zero(const filepath_t& path) {
  std::error_code error;
  const u64 bytes = std::filesystem::file_size(path, error);
  return error ? 0 : bytes;
}

u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

u64 make_build_fingerprint(const filepath_t& output_prefix,
                           const nlohmann::json& metadata) {
  const str material = output_prefix.string() + "\n" + metadata.dump();
  u64 value = vamana::centroid_state::checksum(span<const byte_t>{
    reinterpret_cast<const byte_t*>(material.data()), material.size()});
  value = mix64(value ^ static_cast<u64>(
    std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  std::random_device entropy;
  value = mix64(value ^ (static_cast<u64>(entropy()) << 32) ^ entropy());
  return value == 0 ? 0x9e3779b97f4a7c15ULL : value;
}

filepath_t metadata_path(const filepath_t& prefix) {
  return filepath_t{prefix.string() + ".meta.json"};
}

filepath_t temporary_path(const filepath_t& final_path) {
  return filepath_t{final_path.string() + ".legacy-v2.tmp"};
}

str normalized_prefix(const filepath_t& path) {
  std::error_code error;
  filepath_t absolute = std::filesystem::absolute(path, error);
  if (error) fail("cannot resolve index prefix: " + path.string());
  return absolute.lexically_normal().string();
}

template <typename T>
T required(const nlohmann::json& metadata, const char* name) {
  try {
    return metadata.at(name).get<T>();
  } catch (const std::exception&) {
    fail(str{"missing or invalid metadata field: "} + name);
  }
}

Layout parse_layout(const nlohmann::json& metadata,
                    const LegacyIndexConvertOptions& options) {
  if (metadata.value("schema_version", 0u) != legacy::kSchemaVersion ||
      metadata.value("node_layout", str{}) != "plain" ||
      metadata.value("storage_format", str{}) != legacy::kStorageFormat ||
      metadata.value("distance", str{}) != "l2") {
    fail("input must be a schema-15 plain vamana_compact_v1 L2 index");
  }

  Layout layout;
  layout.dim = required<u32>(metadata, "dim");
  layout.degree = required<u32>(metadata, "R");
  layout.shards = required<u32>(metadata, "num_memory_nodes");
  layout.node_count = required<u64>(metadata, "num_vectors");
  layout.dtype = parse_vector_dtype(required<str>(metadata, "vector_data_type"));
  const size_t vector_bytes = vector_dtype_bytes(layout.dtype, layout.dim);
  if (vector_bytes > std::numeric_limits<u32>::max()) {
    fail("vector byte width overflows the runtime layout");
  }
  layout.vector_bytes = static_cast<u32>(vector_bytes);
  layout.old_node_bytes = static_cast<u32>(legacy::node_bytes(vector_bytes));
  layout.old_graph_bytes = static_cast<u32>(
    legacy::hot_graph_entry_bytes(layout.degree));
  layout.shard_bits = legacy::shard_bits_for(layout.shards);
  layout.counts = required<vec<u64>>(metadata, "hot_graph_entry_counts");
  layout.old_header_offsets =
    required<vec<u64>>(metadata, "hot_graph_header_offsets");
  layout.old_graph_offsets = required<vec<u64>>(metadata, "hot_graph_offsets");
  layout.old_dynamic_offsets =
    required<vec<u64>>(metadata, "hot_graph_dynamic_base_offsets");

  if (layout.dim == 0 || layout.degree == 0 ||
      layout.degree > kMaxSupportedGraphDegree || layout.shards == 0 ||
      layout.shards > RemotePtr::MEMORY_NODE_MASK + 1 ||
      layout.node_count == 0 || layout.shard_bits >= 16 ||
      layout.counts.size() != layout.shards ||
      layout.old_header_offsets.size() != layout.shards ||
      layout.old_graph_offsets.size() != layout.shards ||
      layout.old_dynamic_offsets.size() != layout.shards ||
      required<u32>(metadata, "node_size") != layout.old_node_bytes ||
      required<u32>(metadata, "vector_offset") != legacy::kVectorOffset ||
      required<u32>(metadata, "vector_bytes") != layout.vector_bytes ||
      required<u32>(metadata, "hot_graph_entry_size") !=
        layout.old_graph_bytes ||
      required<u32>(metadata, "hot_graph_pointer_bytes") !=
        legacy::kCompactPointerBytes ||
      required<u32>(metadata, "hot_graph_shard_bits") != layout.shard_bits) {
    fail("legacy metadata layout is inconsistent or unsupported");
  }
  if (metadata.value("graph_hot_bytes", legacy::kVectorOffset) !=
        legacy::kVectorOffset ||
      metadata.value("vector_component_size", 0u) !=
        vector_dtype_component_size(layout.dtype)) {
    fail("legacy fixed-record metadata is inconsistent");
  }

  u64 total = 0;
  layout.new_header_offsets.resize(layout.shards);
  layout.new_graph_offsets.resize(layout.shards);
  layout.new_dynamic_offsets.resize(layout.shards);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    if (layout.counts[shard] == 0) {
      fail("legacy index contains an empty physical shard");
    }
    total = checked_add(total, layout.counts[shard], "node count overflows");
    const u64 old_nodes_end = checked_add(
      legacy::kNodeBaseOffset,
      checked_mul(layout.counts[shard], layout.old_node_bytes,
                  "legacy node plane overflows"),
      "legacy node plane overflows");
    const u64 expected_old_header = align_up(old_nodes_end, 64);
    const u64 expected_old_graph = align_up(
      checked_add(expected_old_header, sizeof(vamana::hot_graph::Header),
                  "legacy graph header overflows"), 64);
    const u64 expected_old_dynamic = align_up(
      checked_add(expected_old_graph,
                  checked_mul(layout.counts[shard], layout.old_graph_bytes,
                              "legacy graph plane overflows"),
                  "legacy graph plane overflows"), 64);
    if (layout.old_header_offsets[shard] != expected_old_header ||
        layout.old_graph_offsets[shard] != expected_old_graph ||
        layout.old_dynamic_offsets[shard] != expected_old_dynamic) {
      fail("legacy shard offsets are not the canonical compact-v1 layout");
    }

    const u64 new_nodes_end = checked_add(
      vamana::hot_graph::kNodeBaseOffset,
      checked_mul(layout.counts[shard], VamanaNode::total_size(),
                  "tagged node plane overflows"),
      "tagged node plane overflows");
    layout.new_header_offsets[shard] = align_up(new_nodes_end, 64);
    layout.new_graph_offsets[shard] = align_up(
      checked_add(layout.new_header_offsets[shard],
                  sizeof(vamana::hot_graph::Header),
                  "tagged graph header overflows"), 64);
    layout.new_dynamic_offsets[shard] = align_up(
      checked_add(layout.new_graph_offsets[shard],
                  checked_mul(layout.counts[shard],
                              VamanaNode::hot_graph_entry_size(),
                              "tagged graph plane overflows"),
                  "tagged graph plane overflows"), 64);
    if (layout.new_dynamic_offsets[shard] >=
        RemotePtr::BYTE_OFFSET_CAPACITY) {
      fail("converted shard exceeds the 256-GiB tagged RemotePtr capacity");
    }
  }
  if (total != layout.node_count) fail("per-shard node counts do not sum to num_vectors");

  layout.subquantizers = options.subquantizers == 0
    ? metadata.value("pq_subquantizers", 0u) : options.subquantizers;
  if (!options.graph_only && layout.subquantizers == 0) {
    fail("cannot infer PQ subquantizers; pass --subquantizers");
  }
  if (!options.graph_only) {
    if (!options.reuse_model.empty()) {
      layout.model_file = options.reuse_model;
    } else if (metadata.contains("navigation_model_file")) {
      layout.model_file = metadata.at("navigation_model_file").get<filepath_t>();
    } else {
      layout.model_file = index_path::navigation_model_file(
        options.input_prefix, layout.subquantizers);
    }
    if (!std::filesystem::is_regular_file(layout.model_file)) {
      const filepath_t fallback = index_path::navigation_model_file(
        options.input_prefix, layout.subquantizers);
      if (std::filesystem::is_regular_file(fallback)) layout.model_file = fallback;
    }
    if (!std::filesystem::is_regular_file(layout.model_file)) {
      fail("legacy PQ model is missing: " + layout.model_file.string());
    }
  }
  return layout;
}

void read_exact(std::istream& input, void* destination, size_t bytes,
                const str& context) {
  input.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
  if (input.gcount() != static_cast<std::streamsize>(bytes)) {
    fail("short read while " + context);
  }
}

bool valid_legacy_pointer(const legacy::DecodedPointer& pointer,
                          const Layout& layout) {
  if (pointer.is_null || pointer.shard >= layout.shards ||
      pointer.byte_offset < legacy::kNodeBaseOffset) return false;
  const u64 relative = pointer.byte_offset - legacy::kNodeBaseOffset;
  return relative % layout.old_node_bytes == 0 &&
    relative / layout.old_node_bytes < layout.counts[pointer.shard];
}

RemotePtr convert_pointer(const legacy::DecodedPointer& pointer,
                          const Layout& layout) {
  if (!valid_legacy_pointer(pointer, layout)) fail("graph contains an invalid static pointer");
  const u64 slot = (pointer.byte_offset - legacy::kNodeBaseOffset) /
    layout.old_node_bytes;
  const u64 new_offset = checked_add(
    vamana::hot_graph::kNodeBaseOffset,
    checked_mul(slot, VamanaNode::total_size(), "tagged pointer overflows"),
    "tagged pointer overflows");
  if (!RemotePtr::representable(pointer.shard, new_offset, 0)) {
    fail("converted static pointer exceeds tagged RemotePtr capacity");
  }
  return RemotePtr{pointer.shard, new_offset, 0};
}

u64 validate_shard(const filepath_t& path, const Layout& layout, u32 shard) {
  std::error_code error;
  const u64 file_bytes = std::filesystem::file_size(path, error);
  if (error) {
    fail("legacy shard is missing or unreadable: " + path.string() +
         ": " + error.message());
  }
  if (file_bytes != layout.old_dynamic_offsets[shard]) {
    fail("legacy shard has the wrong file size: " + path.string() +
         " (expected " + std::to_string(layout.old_dynamic_offsets[shard]) +
         ", actual " + std::to_string(file_bytes) + ")");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) fail("cannot open legacy shard: " + path.string());
  u64 stored_size = 0;
  u64 medoid_raw = 0;
  read_exact(input, &stored_size, sizeof(stored_size), "reading shard size");
  read_exact(input, &medoid_raw, sizeof(medoid_raw), "reading legacy medoid");
  if (stored_size != file_bytes) fail("legacy shard size word does not match the file");
  if (shard != 0 && medoid_raw != 0) fail("nonzero legacy medoid word outside shard zero");

  constexpr size_t kChunkRecords = 4096;
  vec<byte_t> nodes(kChunkRecords * layout.old_node_bytes);
  input.seekg(static_cast<std::streamoff>(legacy::kNodeBaseOffset));
  u64 remaining = layout.counts[shard];
  while (remaining != 0) {
    const size_t count = static_cast<size_t>(std::min<u64>(remaining, kChunkRecords));
    const size_t bytes = count * layout.old_node_bytes;
    read_exact(input, nodes.data(), bytes, "validating fixed records");
    for (size_t index = 0; index < count; ++index) {
      const byte_t* node = nodes.data() + index * layout.old_node_bytes;
      const u64 header = legacy::load_u64(node + legacy::kFixedHeaderOffset);
      if ((header & ~legacy::kIsMedoid) != 0 ||
          (header & legacy::kDeleted) != 0 ||
          legacy::load_u32(node + legacy::kGenerationOffset) != 0) {
        fail("legacy input is not an immutable, undeleted generation-zero snapshot");
      }
      if (layout.dtype == VectorDType::float32) {
        for (u32 dimension = 0; dimension < layout.dim; ++dimension) {
          if (!vector_component_is_finite(vector_component_as_float(
                node + legacy::kVectorOffset, layout.dtype, dimension))) {
            fail("legacy fixed record contains a non-finite vector component");
          }
        }
      }
    }
    remaining -= count;
  }

  vamana::hot_graph::Header graph_header;
  input.seekg(static_cast<std::streamoff>(layout.old_header_offsets[shard]));
  read_exact(input, &graph_header, sizeof(graph_header), "reading hot-graph header");
  const u32 old_dynamic_record = static_cast<u32>(legacy::align16(
    static_cast<size_t>(layout.old_node_bytes) + layout.old_graph_bytes));
  if (graph_header.magic != vamana::hot_graph::kMagic ||
      graph_header.version != legacy::kHotGraphVersion ||
      graph_header.header_bytes != sizeof(vamana::hot_graph::Header) ||
      graph_header.entry_bytes != layout.old_graph_bytes ||
      graph_header.max_degree != layout.degree ||
      graph_header.compact_pointer_bytes != legacy::kCompactPointerBytes ||
      graph_header.compact_pointer_shard_bits != layout.shard_bits ||
      graph_header.flags != 0 || graph_header.entry_count != layout.counts[shard] ||
      graph_header.node_base_offset != legacy::kNodeBaseOffset ||
      graph_header.reserved0 != layout.old_dynamic_offsets[shard] ||
      graph_header.reserved1 != old_dynamic_record ||
      graph_header.reserved2 != layout.old_node_bytes) {
    fail("legacy hot-graph v2 header is inconsistent: " + path.string());
  }

  vec<byte_t> entries(kChunkRecords * layout.old_graph_bytes);
  input.seekg(static_cast<std::streamoff>(layout.old_graph_offsets[shard]));
  remaining = layout.counts[shard];
  u64 edge_count = 0;
  while (remaining != 0) {
    const size_t count = static_cast<size_t>(std::min<u64>(remaining, kChunkRecords));
    const size_t bytes = count * layout.old_graph_bytes;
    read_exact(input, entries.data(), bytes, "validating compact graph entries");
    for (size_t index = 0; index < count; ++index) {
      const byte_t* entry = entries.data() + index * layout.old_graph_bytes;
      const u8 degree = static_cast<u8>(entry[0]);
      if (degree > layout.degree || entry[1] != 0 ||
          legacy::load_u16(entry + 2) != legacy::checksum16(entry, layout.old_graph_bytes) ||
          legacy::load_u32(entry + 4) != 0) {
        fail("legacy graph entry is corrupt, deleted, or not generation zero");
      }
      edge_count = checked_add(edge_count, degree, "edge count overflows");
      for (u32 neighbor = 0; neighbor < layout.degree; ++neighbor) {
        legacy::DecodedPointer pointer;
        if (!legacy::decode_compact_pointer(
              entry + legacy::hot_graph_neighbor_offset(neighbor),
              layout.shard_bits, pointer) ||
            (neighbor < degree ? !valid_legacy_pointer(pointer, layout)
                               : !pointer.is_null)) {
          fail("legacy graph contains an invalid active or non-null inactive edge");
        }
      }
    }
    remaining -= count;
  }
  return edge_count;
}

void reject_output_collision(const filepath_t& path) {
  if (std::filesystem::exists(path) ||
      std::filesystem::exists(temporary_path(path))) {
    fail("refusing to overwrite existing output: " + path.string());
  }
}

void create_sized_file(const filepath_t& path, u64 bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.good() || bytes == 0 ||
      bytes > static_cast<u64>(std::numeric_limits<std::streamoff>::max())) {
    fail("cannot create converted shard: " + path.string());
  }
  output.seekp(static_cast<std::streamoff>(bytes - 1));
  output.put(0);
  if (!output.good()) fail("cannot size converted shard: " + path.string());
}

void finalize_idmap(std::fstream& output, const filepath_t& path,
                    vamana::idmap::Header header, u64 count, u64 checksum) {
  u64 payload_bytes = 0;
  if (!vamana::idmap::checked_payload_bytes(count, payload_bytes)) {
    fail("idmap payload size overflows");
  }
  header.entry_count = count;
  header.payload_bytes = payload_bytes;
  header.payload_checksum = checksum;
  header.header_checksum = vamana::idmap::compute_header_checksum(header);
  output.flush();
  output.seekp(0);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.flush();
  if (!output.good()) fail("cannot finalize converted idmap: " + path.string());
  output.close();
}

struct RouteCandidate {
  long double distance{};
  RemotePtr pointer{};
};

bool route_less(const RouteCandidate& lhs, const RouteCandidate& rhs) {
  return lhs.distance < rhs.distance ||
    (lhs.distance == rhs.distance &&
     lhs.pointer.raw_address < rhs.pointer.raw_address);
}

void write_metadata(const filepath_t& path, const nlohmann::json& metadata) {
  std::ofstream output(path, std::ios::trunc);
  output << std::setw(2) << metadata << '\n';
  if (!output.good()) fail("cannot write converted metadata: " + path.string());
}

void publish_file(const filepath_t& temporary, const filepath_t& final) {
  std::error_code error;
  std::filesystem::rename(temporary, final, error);
  if (error) fail("cannot publish " + final.string() + ": " + error.message());
}

}  // namespace

LegacyIndexConvertResult convert_legacy_index(
    const LegacyIndexConvertOptions& options) {
  if (options.input_prefix.empty() || options.output_prefix.empty()) {
    throw std::invalid_argument("legacy conversion requires input and output prefixes");
  }
  if (normalized_prefix(options.input_prefix) ==
      normalized_prefix(options.output_prefix)) {
    fail("input and output prefixes must differ; in-place conversion is forbidden");
  }

  std::ifstream metadata_input(metadata_path(options.input_prefix));
  if (!metadata_input.good()) {
    fail("missing metadata: " + metadata_path(options.input_prefix).string());
  }
  nlohmann::json metadata;
  metadata_input >> metadata;

  // Initialize the target layout before deriving any output offsets.
  const u32 dim = required<u32>(metadata, "dim");
  const u32 degree = required<u32>(metadata, "R");
  const VectorDType dtype = parse_vector_dtype(
    required<str>(metadata, "vector_data_type"));
  if (dim == 0 || degree == 0 || degree > kMaxSupportedGraphDegree) {
    fail("dimension or graph degree is outside the current runtime limits");
  }
  VamanaNode::init_static_storage(dim, degree, dtype);
  Layout layout = parse_layout(metadata, options);

  LegacyIndexConvertResult result;
  result.node_count = layout.node_count;
  result.shards = layout.shards;
  result.subquantizers = layout.subquantizers;
  result.legacy_model_file = layout.model_file;
  result.metadata_file = metadata_path(options.output_prefix);
  result.input_bytes = file_bytes_or_zero(metadata_path(options.input_prefix));
  result.output_bytes = file_bytes_or_zero(metadata_path(options.input_prefix));
  result.output_bytes = checked_add(
    result.output_bytes,
    checked_mul(layout.shards, sizeof(vamana::idmap::Header),
                "estimated output size overflows"),
    "estimated output size overflows");
  result.output_bytes = checked_add(
    result.output_bytes,
    checked_mul(layout.node_count, sizeof(vamana::idmap::Entry),
                "estimated output size overflows"),
    "estimated output size overflows");
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    result.input_bytes = checked_add(
      result.input_bytes, layout.old_dynamic_offsets[shard],
      "input byte count overflows");
    result.input_bytes = checked_add(
      result.input_bytes,
      file_bytes_or_zero(index_path::owner_idmap_file(
        options.input_prefix, shard + 1, layout.shards)),
      "input byte count overflows");
    result.output_bytes = checked_add(
      result.output_bytes, layout.new_dynamic_offsets[shard],
      "estimated output size overflows");
    result.output_bytes = checked_add(
      result.output_bytes,
      sizeof(vamana::centroid_state::Header) +
        static_cast<u64>(layout.dim) * sizeof(f64) +
        std::min<u64>(layout.counts[shard],
                      vamana::centroid_state::kMaxLiveEntries) *
          sizeof(vamana::centroid_state::Entry),
      "estimated output size overflows");
    result.edge_count = checked_add(
      result.edge_count,
      validate_shard(index_path::shard_file(
        options.input_prefix, shard + 1, layout.shards), layout, shard),
      "edge count overflows");
  }
  if (!options.graph_only) {
    const u64 model_bytes = file_bytes_or_zero(layout.model_file);
    result.input_bytes = checked_add(
      result.input_bytes, model_bytes, "input byte count overflows");
    result.output_bytes = checked_add(
      result.output_bytes, model_bytes,
      "estimated output size overflows");
    result.output_bytes = checked_add(
      result.output_bytes,
      checked_add(
        checked_mul(layout.shards, sizeof(gpu_search::format::CodeHeader),
                    "estimated PQ output size overflows"),
        checked_mul(layout.node_count, layout.subquantizers,
                    "estimated PQ output size overflows"),
        "estimated PQ output size overflows"),
      "estimated output size overflows");
  }

  if (options.dry_run) return result;

  const filepath_t output_directory = options.output_prefix.parent_path();
  if (!output_directory.empty()) {
    std::filesystem::create_directories(output_directory);
  }
  reject_output_collision(result.metadata_file);
  vec<filepath_t> shard_paths(layout.shards);
  vec<filepath_t> idmap_paths(layout.shards);
  vec<filepath_t> centroid_paths(layout.shards);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    shard_paths[shard] = index_path::shard_file(
      options.output_prefix, shard + 1, layout.shards);
    idmap_paths[shard] = index_path::owner_idmap_file(
      options.output_prefix, shard + 1, layout.shards);
    centroid_paths[shard] = index_path::centroid_state_file(
      options.output_prefix, shard + 1, layout.shards);
    reject_output_collision(shard_paths[shard]);
    reject_output_collision(idmap_paths[shard]);
    reject_output_collision(centroid_paths[shard]);
  }
  if (!options.graph_only) {
    reject_output_collision(index_path::navigation_model_file(
      options.output_prefix, layout.subquantizers));
    for (u32 shard = 0; shard < layout.shards; ++shard) {
      reject_output_collision(index_path::navigation_code_file(
        options.output_prefix, shard + 1, layout.shards,
        layout.subquantizers));
    }
  }

  layout.build_fingerprint = make_build_fingerprint(options.output_prefix, metadata);
  layout.shard_fingerprints.resize(layout.shards);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    layout.shard_fingerprints[shard] = mix64(
      layout.build_fingerprint ^ mix64(shard) ^
      layout.new_dynamic_offsets[shard] ^ mix64(layout.counts[shard]) ^
      layout.new_graph_offsets[shard]);
    if (layout.shard_fingerprints[shard] == 0) {
      layout.shard_fingerprints[shard] = mix64(
        layout.build_fingerprint ^ shard ^ 1);
    }
  }

  OutputCleanup cleanup;
  vec<filepath_t> temporary_shards(layout.shards);
  vec<filepath_t> temporary_idmaps(layout.shards);
  vec<filepath_t> temporary_centroids(layout.shards);
  vec<std::fstream> output_shards(layout.shards);
  vec<std::fstream> idmap_outputs(layout.shards);
  vec<u64> idmap_counts(layout.shards, 0);
  vec<u64> idmap_checksums(layout.shards, vamana::idmap::checksum_initial());
  vec<vec<vamana::idmap::Entry>> idmap_buffers(layout.shards);
  vec<vec<f64>> centroid_sums(layout.shards, vec<f64>(layout.dim, 0.0));
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    idmap_buffers[shard].reserve(4096);
    temporary_shards[shard] = temporary_path(shard_paths[shard]);
    temporary_idmaps[shard] = temporary_path(idmap_paths[shard]);
    temporary_centroids[shard] = temporary_path(centroid_paths[shard]);
    cleanup.paths.push_back(temporary_shards[shard]);
    cleanup.paths.push_back(temporary_idmaps[shard]);
    cleanup.paths.push_back(temporary_centroids[shard]);
    cleanup.paths.push_back(shard_paths[shard]);
    cleanup.paths.push_back(idmap_paths[shard]);
    cleanup.paths.push_back(centroid_paths[shard]);

    create_sized_file(temporary_shards[shard], layout.new_dynamic_offsets[shard]);
    output_shards[shard].open(temporary_shards[shard],
      std::ios::binary | std::ios::in | std::ios::out);
    const u64 file_size = layout.new_dynamic_offsets[shard];
    output_shards[shard].write(reinterpret_cast<const char*>(&file_size), sizeof(file_size));
    output_shards[shard].write(
      reinterpret_cast<const char*>(&layout.shard_fingerprints[shard]),
      sizeof(layout.shard_fingerprints[shard]));
    vamana::hot_graph::Header header;
    header.version = vamana::hot_graph::kVersion3;
    header.entry_bytes = static_cast<u32>(VamanaNode::hot_graph_entry_size());
    header.max_degree = layout.degree;
    header.compact_pointer_shard_bits = layout.shard_bits;
    header.entry_count = layout.counts[shard];
    header.reserved0 = layout.new_dynamic_offsets[shard];
    header.reserved1 = VamanaNode::dynamic_record_size();
    header.reserved2 = static_cast<u32>(VamanaNode::total_size());
    output_shards[shard].seekp(
      static_cast<std::streamoff>(layout.new_header_offsets[shard]));
    output_shards[shard].write(reinterpret_cast<const char*>(&header), sizeof(header));

    idmap_outputs[shard].open(temporary_idmaps[shard],
      std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    const vamana::idmap::Header placeholder;
    idmap_outputs[shard].write(
      reinterpret_cast<const char*>(&placeholder), sizeof(placeholder));
    if (!output_shards[shard].good() || !idmap_outputs[shard].good()) {
      fail("cannot initialize converted output files");
    }
  }

  auto flush_idmap = [&](u32 owner) {
    auto& buffer = idmap_buffers[owner];
    if (buffer.empty()) return;
    idmap_outputs[owner].write(reinterpret_cast<const char*>(buffer.data()),
      static_cast<std::streamsize>(buffer.size() * sizeof(buffer.front())));
    idmap_checksums[owner] = vamana::idmap::checksum_update(
      idmap_checksums[owner], buffer.data(), buffer.size() * sizeof(buffer.front()));
    idmap_counts[owner] += buffer.size();
    buffer.clear();
  };

  constexpr size_t kChunkRecords = 4096;
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    const filepath_t source_path = index_path::shard_file(
      options.input_prefix, shard + 1, layout.shards);
    std::ifstream input(source_path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(legacy::kNodeBaseOffset));
    output_shards[shard].seekp(
      static_cast<std::streamoff>(vamana::hot_graph::kNodeBaseOffset));
    vec<byte_t> old_nodes(kChunkRecords * layout.old_node_bytes);
    vec<byte_t> new_nodes(kChunkRecords * VamanaNode::total_size());
    u64 slot_base = 0;
    while (slot_base < layout.counts[shard]) {
      const size_t count = static_cast<size_t>(std::min<u64>(
        kChunkRecords, layout.counts[shard] - slot_base));
      read_exact(input, old_nodes.data(), count * layout.old_node_bytes,
                 "converting fixed records");
      std::fill(new_nodes.begin(),
                new_nodes.begin() + count * VamanaNode::total_size(), byte_t{});
      for (size_t index = 0; index < count; ++index) {
        const byte_t* old_node = old_nodes.data() + index * layout.old_node_bytes;
        byte_t* new_node = new_nodes.data() + index * VamanaNode::total_size();
        const node_t id = legacy::load_u32(old_node + legacy::kIdOffset);
        const u64 new_header = VamanaNode::make_header(
          0, VamanaNode::HEADER_CENTROID_ACCOUNTED);
        std::memcpy(new_node, &new_header, sizeof(new_header));
        std::memcpy(new_node + VamanaNode::offset_id(), &id, sizeof(id));
        std::memcpy(new_node + VamanaNode::offset_vector(),
                    old_node + legacy::kVectorOffset, layout.vector_bytes);
        for (u32 dimension = 0; dimension < layout.dim; ++dimension) {
          centroid_sums[shard][dimension] += static_cast<f64>(
            vector_component_as_float(old_node + legacy::kVectorOffset,
                                      layout.dtype, dimension));
        }
        const u64 slot = slot_base + index;
        const u64 offset = vamana::hot_graph::kNodeBaseOffset +
          slot * VamanaNode::total_size();
        const RemotePtr pointer{shard, offset, 0};
        const u32 owner = id % layout.shards;
        idmap_buffers[owner].push_back(vamana::idmap::Entry{
          id, pointer.raw_address, 0, 0, 0});
        if (idmap_buffers[owner].size() == idmap_buffers[owner].capacity()) {
          flush_idmap(owner);
        }
      }
      output_shards[shard].write(reinterpret_cast<const char*>(new_nodes.data()),
        static_cast<std::streamsize>(count * VamanaNode::total_size()));
      slot_base += count;
    }

    input.seekg(static_cast<std::streamoff>(layout.old_graph_offsets[shard]));
    output_shards[shard].seekp(
      static_cast<std::streamoff>(layout.new_graph_offsets[shard]));
    vec<byte_t> old_entries(kChunkRecords * layout.old_graph_bytes);
    vec<byte_t> new_entries(kChunkRecords * VamanaNode::hot_graph_entry_size());
    vec<RemotePtr> neighbors(layout.degree);
    slot_base = 0;
    while (slot_base < layout.counts[shard]) {
      const size_t count = static_cast<size_t>(std::min<u64>(
        kChunkRecords, layout.counts[shard] - slot_base));
      read_exact(input, old_entries.data(), count * layout.old_graph_bytes,
                 "converting graph entries");
      for (size_t index = 0; index < count; ++index) {
        const byte_t* old_entry = old_entries.data() + index * layout.old_graph_bytes;
        const u8 edge_count = static_cast<u8>(old_entry[0]);
        for (u32 neighbor = 0; neighbor < edge_count; ++neighbor) {
          legacy::DecodedPointer decoded;
          if (!legacy::decode_compact_pointer(
                old_entry + legacy::hot_graph_neighbor_offset(neighbor),
                layout.shard_bits, decoded)) {
            fail("cannot decode a legacy compact pointer");
          }
          neighbors[neighbor] = convert_pointer(decoded, layout);
        }
        VamanaNode::encode_hot_graph_entry(
          new_entries.data() + index * VamanaNode::hot_graph_entry_size(),
          edge_count, neighbors.data(), edge_count, layout.shard_bits, 0,
          false, nullptr, 0, 0);
      }
      output_shards[shard].write(
        reinterpret_cast<const char*>(new_entries.data()),
        static_cast<std::streamsize>(count * VamanaNode::hot_graph_entry_size()));
      slot_base += count;
    }
    output_shards[shard].flush();
    if (!output_shards[shard].good()) fail("cannot flush converted shard");
  }
  for (u32 owner = 0; owner < layout.shards; ++owner) flush_idmap(owner);

  for (u32 owner = 0; owner < layout.shards; ++owner) {
    vamana::idmap::Header header;
    header.build_fingerprint = layout.build_fingerprint;
    header.owner_shard_fingerprint = layout.shard_fingerprints[owner];
    header.owner_shard = owner;
    header.shard_count = layout.shards;
    header.node_base_offset = vamana::hot_graph::kNodeBaseOffset;
    header.node_size = static_cast<u32>(VamanaNode::total_size());
    header.id_offset = static_cast<u32>(VamanaNode::offset_id());
    header.generation_offset = static_cast<u32>(VamanaNode::offset_generation());
    header.slot_incarnation_offset = static_cast<u32>(
      VamanaNode::offset_slot_incarnation());
    finalize_idmap(idmap_outputs[owner], temporary_idmaps[owner], header,
                   idmap_counts[owner], idmap_checksums[owner]);
  }

  vec<vec<RouteCandidate>> route_candidates(layout.shards);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    if (layout.counts[shard] == 0) fail("centroid routing does not support an empty shard");
    std::ifstream input(temporary_shards[shard], std::ios::binary);
    input.seekg(static_cast<std::streamoff>(vamana::hot_graph::kNodeBaseOffset));
    vec<byte_t> node(VamanaNode::total_size());
    const f64 inverse_count = 1.0 / static_cast<f64>(layout.counts[shard]);
    for (u64 slot = 0; slot < layout.counts[shard]; ++slot) {
      read_exact(input, node.data(), node.size(), "selecting centroid route entries");
      long double distance = 0;
      for (u32 dimension = 0; dimension < layout.dim; ++dimension) {
        const long double component = vector_component_as_float(
          node.data() + VamanaNode::offset_vector(), layout.dtype, dimension);
        const long double centroid = static_cast<long double>(
          centroid_sums[shard][dimension] * inverse_count);
        const long double difference = component - centroid;
        distance += difference * difference;
      }
      RouteCandidate candidate{
        distance,
        RemotePtr{shard, vamana::hot_graph::kNodeBaseOffset +
                         slot * VamanaNode::total_size(), 0},
      };
      auto& entries = route_candidates[shard];
      if (entries.size() < vamana::centroid_state::kMaxLiveEntries) {
        entries.push_back(candidate);
        std::sort(entries.begin(), entries.end(), route_less);
      } else if (route_less(candidate, entries.back())) {
        entries.back() = candidate;
        std::sort(entries.begin(), entries.end(), route_less);
      }
    }

    vec<vamana::centroid_state::Entry> entries;
    for (const auto& candidate : route_candidates[shard]) {
      entries.push_back({candidate.pointer.raw_address, 0, 0});
    }
    vamana::centroid_state::Header header;
    header.build_fingerprint = layout.build_fingerprint;
    header.shard_fingerprint = layout.shard_fingerprints[shard];
    header.vector_count = layout.counts[shard];
    header.node_base_offset = vamana::hot_graph::kNodeBaseOffset;
    header.shard = shard;
    header.shard_count = layout.shards;
    header.dim = layout.dim;
    header.max_degree = layout.degree;
    header.entry_count = static_cast<u32>(entries.size());
    header.vector_dtype = static_cast<u32>(layout.dtype);
    header.vector_component_size = static_cast<u32>(
      vector_dtype_component_size(layout.dtype));
    header.node_size = static_cast<u32>(VamanaNode::total_size());
    header.vector_offset = static_cast<u32>(VamanaNode::offset_vector());
    header.vector_bytes = layout.vector_bytes;
    header.slot_incarnation_offset = static_cast<u32>(
      VamanaNode::offset_slot_incarnation());
    header.hot_graph_version = vamana::hot_graph::kVersion3;
    header.hot_graph_entry_size = static_cast<u32>(VamanaNode::hot_graph_entry_size());
    header.hot_graph_pointer_bytes = vamana::hot_graph::kCompactPointerBytes;
    header.hot_graph_shard_bits = layout.shard_bits;
    header.payload_bytes = vamana::centroid_state::payload_bytes(
      layout.dim, header.entry_count);
    vec<byte_t> payload(static_cast<size_t>(header.payload_bytes));
    std::memcpy(payload.data(), centroid_sums[shard].data(),
                static_cast<size_t>(layout.dim) * sizeof(f64));
    std::memcpy(payload.data() + static_cast<size_t>(layout.dim) * sizeof(f64),
                entries.data(), entries.size() * sizeof(entries.front()));
    header.payload_checksum = vamana::centroid_state::checksum(payload);
    header.header_checksum = vamana::centroid_state::compute_header_checksum(header);
    std::ofstream output(temporary_centroids[shard],
                         std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    if (!output.good()) fail("cannot write converted centroid state");
  }

  metadata["output_prefix"] = options.output_prefix.string();
  metadata["schema_version"] = 15;
  metadata["storage_format"] = VamanaNode::storage_format_name();
  metadata["remote_ptr_format"] = "tagged_inc24_shard6_off34x16_v1";
  metadata["node_size"] = VamanaNode::total_size();
  metadata["graph_hot_bytes"] = VamanaNode::graph_hot_bytes();
  metadata["vector_offset"] = VamanaNode::offset_vector();
  metadata["slot_incarnation_offset"] = VamanaNode::offset_slot_incarnation();
  metadata["vector_storage_bytes"] = VamanaNode::vector_storage_bytes();
  metadata["hot_graph_neighbor_read_bytes"] = VamanaNode::hot_graph_entry_size();
  metadata["hot_graph_neighbor_update_bytes"] = VamanaNode::hot_graph_entry_size();
  metadata["hot_graph_entry_size"] = VamanaNode::hot_graph_entry_size();
  metadata["hot_graph_pointer_bytes"] = vamana::hot_graph::kCompactPointerBytes;
  metadata["hot_graph_shard_bits"] = layout.shard_bits;
  metadata["hot_graph_offsets"] = layout.new_graph_offsets;
  metadata["hot_graph_header_offsets"] = layout.new_header_offsets;
  metadata["hot_graph_entry_counts"] = layout.counts;
  metadata["hot_graph_dynamic_base_offsets"] = layout.new_dynamic_offsets;
  metadata["hot_graph_dynamic_record_bytes"] = VamanaNode::dynamic_record_size();
  metadata["hot_graph_dynamic_hot_offset"] = VamanaNode::total_size();
  metadata["allocation_size"] = VamanaNode::dynamic_record_size();
  metadata["idmap_format"] = "owner_sharded_v2_bound";
  metadata["centroid_state_format"] = "physical_shard_centroid_v2_bound";
  metadata["index_build_fingerprint"] = layout.build_fingerprint;
  metadata["shard_build_fingerprints"] = layout.shard_fingerprints;
  metadata["centroid_state_header_bytes"] = sizeof(vamana::centroid_state::Header);
  metadata["navigation_quantizer"] = "";
  metadata["navigation_code_bytes"] = 0;
  metadata["pq_subquantizers"] = layout.subquantizers;
  metadata["pq_bits"] = 8;
  metadata["navigation_model_checksum"] = 0;
  metadata["navigation_model_file"] = "";
  metadata["navigation_format"] = "";
  metadata["navigation_code_remote_offsets"] = vec<u64>{};
  metadata["navigation_code_region_bytes"] = vec<u64>{};
  metadata["navigation_code_materialization"] = "";
  metadata["navigation_graph_source"] = "storage_compact_graph";
  metadata["navigation_execution"] = "";
  for (const char* obsolete : {"medoid", "anchor_format", "anchor_count_per_shard",
                               "navigation_entry_points", "storage_control_remote_offsets",
                               "dynamic_node_base_offsets",
                               "dynamic_navigation_code_offset",
                               "dynamic_navigation_code_validation_bytes",
                               "dynamic_navigation_code_checksum_bytes"}) {
    metadata.erase(obsolete);
  }

  const filepath_t temporary_metadata = temporary_path(result.metadata_file);
  cleanup.paths.push_back(temporary_metadata);
  cleanup.paths.push_back(result.metadata_file);
  write_metadata(temporary_metadata, metadata);
  for (auto& shard : output_shards) shard.close();
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    publish_file(temporary_shards[shard], shard_paths[shard]);
    publish_file(temporary_idmaps[shard], idmap_paths[shard]);
    publish_file(temporary_centroids[shard], centroid_paths[shard]);
  }
  // Metadata is the commit marker and is always published last.
  publish_file(temporary_metadata, result.metadata_file);
  cleanup.active = false;
  result.wrote_graph = true;

  if (!options.graph_only) {
    PqIndexOptions pq;
    pq.index_prefix = options.output_prefix;
    pq.reuse_model = layout.model_file;
    pq.subquantizers = layout.subquantizers;
    pq.chunk_vectors = options.chunk_vectors == 0
      ? options.chunk_nodes : options.chunk_vectors;
    pq.threads = options.threads;
    (void)build_pq_index(pq);
    result.built_pq = true;
  }
  // Replace the dry-run estimate with the exact set of files emitted by this
  // invocation. Metadata is included once; legacy codes are never read.
  result.output_bytes = file_bytes_or_zero(result.metadata_file);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    result.output_bytes = checked_add(
      result.output_bytes, file_bytes_or_zero(shard_paths[shard]),
      "output byte count overflows");
    result.output_bytes = checked_add(
      result.output_bytes, file_bytes_or_zero(idmap_paths[shard]),
      "output byte count overflows");
    result.output_bytes = checked_add(
      result.output_bytes, file_bytes_or_zero(centroid_paths[shard]),
      "output byte count overflows");
    if (!options.graph_only) {
      result.output_bytes = checked_add(
        result.output_bytes,
        file_bytes_or_zero(index_path::navigation_code_file(
          options.output_prefix, shard + 1, layout.shards,
          layout.subquantizers)),
        "output byte count overflows");
    }
  }
  if (!options.graph_only) {
    result.output_bytes = checked_add(
      result.output_bytes,
      file_bytes_or_zero(index_path::navigation_model_file(
        options.output_prefix, layout.subquantizers)),
      "output byte count overflows");
  }
  return result;
}

LegacyIndexConversionResult convert_legacy_schema15_index(
    const LegacyIndexConverterOptions& options) {
  return convert_legacy_index(options);
}

}  // namespace tools::vamana_offline
