#include "gpu_search/index_format.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "common/index_path.hh"
#include "nlohmann/json.hh"
#include "vamana/anchor_index.hh"

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

u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

void read_exact_or_throw(std::istream& input, void* destination, size_t bytes,
                         const std::filesystem::path& path) {
  input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(input.gcount()) != bytes) {
    throw std::runtime_error("short read from " + path.string());
  }
}

bool append_anchor_entry_points(
    const std::filesystem::path& prefix, u32 dim, VectorDType dtype,
    u32 vector_bytes, const View& view, u32 target,
    std::unordered_set<u32>& selected, std::vector<u32>& entry_points) {
  const std::filesystem::path path = index_path::anchor_file(prefix);
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) return false;
  vamana::anchor::Header header;
  read_exact_or_throw(input, &header, sizeof(header), path);
  if (header.magic != vamana::anchor::kMagic ||
      header.version != vamana::anchor::kVersion || header.dim != dim ||
      header.shard_count != view.shards.size() ||
      header.vector_dtype != static_cast<u32>(dtype) ||
      header.vector_bytes != vector_bytes ||
      header.total_anchors > (1u << 24)) {
    throw std::runtime_error("invalid anchor sidecar for GPU V4 entry points: " +
                             path.string());
  }
  std::vector<std::vector<u32>> anchor_ordinals(view.shards.size());
  std::vector<f32> shard_centroid(dim);
  std::vector<byte_t> vector(vector_bytes);
  u64 loaded = 0;
  for (u32 shard = 0; shard < view.shards.size(); ++shard) {
    vamana::anchor::ShardHeader shard_header;
    read_exact_or_throw(input, &shard_header, sizeof(shard_header), path);
    if (shard_header.shard != shard ||
        shard_header.anchor_count > header.anchors_per_shard ||
        loaded + shard_header.anchor_count > header.total_anchors) {
      throw std::runtime_error("invalid anchor shard for GPU V4 entry points: " +
                               path.string());
    }
    read_exact_or_throw(input, shard_centroid.data(),
                        shard_centroid.size() * sizeof(f32), path);
    anchor_ordinals[shard].reserve(shard_header.anchor_count);
    for (u32 index = 0; index < shard_header.anchor_count; ++index) {
      vamana::anchor::EntryHeader entry;
      read_exact_or_throw(input, &entry, sizeof(entry), path);
      read_exact_or_throw(input, vector.data(), vector.size(), path);
      const RemotePtr pointer{entry.rptr_raw};
      u32 ordinal = 0;
      if (pointer.is_null() || pointer.memory_node() != shard ||
          !remote_to_ordinal(view, pointer, ordinal)) {
        throw std::runtime_error("anchor points outside its static GPU V4 shard");
      }
      anchor_ordinals[shard].push_back(ordinal);
      ++loaded;
    }
  }
  if (loaded != header.total_anchors) {
    throw std::runtime_error("anchor sidecar count mismatch for GPU V4 entry points");
  }
  bool appended = false;
  for (u32 rank = 0; entry_points.size() < target; ++rank) {
    bool have_rank = false;
    for (u32 shard = 0; shard < anchor_ordinals.size() &&
         entry_points.size() < target; ++shard) {
      if (rank >= anchor_ordinals[shard].size()) continue;
      have_rank = true;
      const u32 ordinal = anchor_ordinals[shard][rank];
      if (selected.insert(ordinal).second) {
        entry_points.push_back(ordinal);
        appended = true;
      }
    }
    if (!have_rank) break;
  }
  return appended;
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

bool synthesize_distributed_view(
    const std::filesystem::path& index_prefix, View& view,
    const SynthesisOptions& options, bool* used_anchor_entry_points,
    std::string* error) {
  if (used_anchor_entry_points != nullptr) *used_anchor_entry_points = false;
  try {
    const std::filesystem::path metadata_path{
      index_prefix.string() + ".meta.json"};
    std::ifstream metadata_input(metadata_path);
    if (!metadata_input.good()) {
      throw std::runtime_error("missing index metadata: " + metadata_path.string());
    }
    nlohmann::json metadata;
    metadata_input >> metadata;
    if (metadata.value("schema_version", 0u) != 13 ||
        metadata.value("distance", std::string{"l2"}) != "l2" ||
        metadata.value("node_layout", std::string{}) != "rabitq" ||
        metadata.value("storage_format", std::string{}) != "vamana_compact_v1") {
      throw std::runtime_error(
        "runtime GPU manifest synthesis requires schema-13 compact L2 RaBitQ metadata");
    }

    const u32 shard_count = metadata.at("num_memory_nodes").get<u32>();
    const std::vector<u64> counts =
      metadata.at("hot_graph_entry_counts").get<std::vector<u64>>();
    const std::vector<u64> graph_offsets =
      metadata.at("hot_graph_offsets").get<std::vector<u64>>();
    const std::vector<u64> dynamic_offsets =
      metadata.at("hot_graph_dynamic_base_offsets").get<std::vector<u64>>();
    if (shard_count == 0 || counts.size() != shard_count ||
        graph_offsets.size() != shard_count || dynamic_offsets.size() != shard_count) {
      throw std::runtime_error("GPU manifest metadata has invalid shard arrays");
    }

    View synthesized;
    synthesized.header.dim = metadata.at("dim").get<u32>();
    synthesized.header.graph_degree = metadata.at("R").get<u32>();
    const VectorDType dtype = parse_vector_dtype(
      metadata.value("vector_data_type", std::string{"float32"}));
    synthesized.header.vector_dtype = static_cast<u32>(dtype);
    synthesized.header.rabitq_code_bits = metadata.at("rabitq_code_bits").get<u32>();
    synthesized.header.rabitq_entry_bytes = metadata.at("rabitq_entry_size").get<u32>();
    synthesized.header.num_shards = shard_count;
    synthesized.header.graph_entry_bytes = metadata.at("hot_graph_entry_size").get<u32>();
    synthesized.header.graph_pointer_bytes =
      metadata.at("hot_graph_pointer_bytes").get<u32>();
    synthesized.header.graph_shard_bits = metadata.at("hot_graph_shard_bits").get<u32>();
    synthesized.header.base_generation = 1;
    synthesized.centroid = metadata.at("rabitq_centroid").get<std::vector<f32>>();
    synthesized.shards.resize(shard_count);

    const u64 node_stride = metadata.at("node_size").get<u64>();
    if (node_stride > std::numeric_limits<u32>::max()) {
      throw std::runtime_error("GPU manifest node stride exceeds uint32 range");
    }
    const u64 inferred_dynamic_record_bytes = align_up(
      node_stride + synthesized.header.graph_entry_bytes, 16);
    if (inferred_dynamic_record_bytes > std::numeric_limits<u32>::max()) {
      throw std::runtime_error("GPU manifest dynamic record layout exceeds uint32 range");
    }
    const u32 dynamic_record_bytes = metadata.value(
      "hot_graph_dynamic_record_bytes",
      static_cast<u32>(inferred_dynamic_record_bytes));
    const u32 dynamic_hot_offset = metadata.value(
      "hot_graph_dynamic_hot_offset", static_cast<u32>(node_stride));
    std::vector<u64> advertised_code_offsets;
    std::vector<u64> advertised_code_bytes;
    if (metadata.contains("gpu_code_remote_offsets") &&
        metadata["gpu_code_remote_offsets"].is_array()) {
      advertised_code_offsets =
        metadata["gpu_code_remote_offsets"].get<std::vector<u64>>();
    }
    if (metadata.contains("gpu_code_region_bytes") &&
        metadata["gpu_code_region_bytes"].is_array()) {
      advertised_code_bytes =
        metadata["gpu_code_region_bytes"].get<std::vector<u64>>();
    }
    if ((!advertised_code_offsets.empty() &&
         advertised_code_offsets.size() != shard_count) ||
        (!advertised_code_bytes.empty() &&
         advertised_code_bytes.size() != shard_count)) {
      throw std::runtime_error("GPU manifest metadata has invalid code-region arrays");
    }

    u64 node_count = 0;
    for (u32 shard = 0; shard < shard_count; ++shard) {
      if (counts[shard] == 0 || graph_offsets[shard] == 0 ||
          dynamic_offsets[shard] == 0 ||
          counts[shard] >= (1ull << 30) - node_count) {
        throw std::runtime_error("GPU manifest metadata contains an invalid shard");
      }
      const u64 code_offset = align_up(dynamic_offsets[shard], 64);
      const u64 code_bytes = counts[shard] *
        synthesized.header.rabitq_entry_bytes;
      if ((!advertised_code_offsets.empty() &&
           advertised_code_offsets[shard] != code_offset) ||
          (!advertised_code_bytes.empty() &&
           advertised_code_bytes[shard] != code_bytes)) {
        throw std::runtime_error("GPU code-region metadata is inconsistent");
      }
      synthesized.shards[shard] = {
        .ordinal_base = node_count,
        .node_count = counts[shard],
        .node_base_offset = kNodeBaseOffset,
        .node_stride = node_stride,
        .graph_base_offset = graph_offsets[shard],
        .dynamic_base_offset = dynamic_offsets[shard],
        .code_remote_offset = code_offset,
        .code_bytes = code_bytes,
        .memory_node = shard,
        .dynamic_record_bytes = dynamic_record_bytes,
        .dynamic_hot_offset = dynamic_hot_offset,
      };
      node_count += counts[shard];
    }
    if (node_count != metadata.at("num_vectors").get<u64>() ||
        node_count == 0 || node_count >= (1ull << 30)) {
      throw std::runtime_error("GPU manifest metadata has an invalid node count");
    }
    synthesized.header.num_nodes = node_count;

    const auto& medoid = metadata.at("medoid");
    const RemotePtr medoid_pointer{
      medoid.at("memory_node").get<u32>(), medoid.at("offset").get<u64>()};
    if (!remote_to_ordinal(
          synthesized, medoid_pointer, synthesized.header.medoid_ordinal)) {
      throw std::runtime_error("GPU manifest metadata has an invalid medoid");
    }

    const u32 requested_entry_points = options.entry_points == 0
      ? metadata.value("gpu_entry_points", 256u) : options.entry_points;
    if (requested_entry_points == 0 || requested_entry_points > kMaxEntryPoints) {
      throw std::runtime_error("GPU manifest entry-point count must be in [1, 512]");
    }
    const u32 target = static_cast<u32>(std::min<u64>(
      requested_entry_points, node_count));
    std::unordered_set<u32> selected;
    selected.insert(synthesized.header.medoid_ordinal);
    synthesized.entry_points.push_back(synthesized.header.medoid_ordinal);
    const bool used_anchors = append_anchor_entry_points(
      index_prefix, synthesized.header.dim, dtype,
      metadata.at("vector_bytes").get<u32>(), synthesized, target,
      selected, synthesized.entry_points);
    const u32 quota = (target + shard_count - 1) / shard_count;
    for (u32 shard = 0; shard < shard_count &&
         synthesized.entry_points.size() < target; ++shard) {
      for (u32 sample = 0; sample < quota * 16 &&
           synthesized.entry_points.size() < target; ++sample) {
        const u64 slot = mix64(options.seed ^
          (static_cast<u64>(shard) << 32) ^ sample) % counts[shard];
        const u32 ordinal = static_cast<u32>(
          synthesized.shards[shard].ordinal_base + slot);
        if (selected.insert(ordinal).second) {
          synthesized.entry_points.push_back(ordinal);
        }
      }
    }
    for (u32 ordinal = 0; synthesized.entry_points.size() < target &&
         ordinal < node_count; ++ordinal) {
      if (selected.insert(ordinal).second) {
        synthesized.entry_points.push_back(ordinal);
      }
    }
    if (synthesized.entry_points.size() != target) {
      throw std::runtime_error("failed to synthesize GPU manifest entry points");
    }
    std::string validation_error;
    if (!validate_view(synthesized, &validation_error)) {
      throw std::runtime_error(validation_error);
    }
    if (used_anchor_entry_points != nullptr) {
      *used_anchor_entry_points = used_anchors;
    }
    view = std::move(synthesized);
    return true;
  } catch (const std::exception& exception) {
    set_error(error, exception.what());
    return false;
  }
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
