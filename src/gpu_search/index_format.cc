#include "gpu_search/index_format.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "common/index_path.hh"
#include "nlohmann/json.hh"
#include "vamana/anchor_index.hh"

namespace gpu_search::format {

u64 storage_route_body_checksum(
    const StorageRoutePublication& publication) {
  u64 checksum = checksum64_initial();
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(&publication.magic),
    offsetof(StorageRoutePublication, body_checksum) -
      offsetof(StorageRoutePublication, magic));
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(publication.slots.data()),
    publication.slots.size() * sizeof(StorageRouteSlot));
  return checksum;
}

bool validate_storage_route_publication(
    const StorageRoutePublication& publication, u32 expected_shard,
    std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (publication.sequence_begin == 0 ||
      (publication.sequence_begin & 1u) != 0 ||
      publication.sequence_begin != publication.sequence_end) {
    return fail("storage route snapshot overlaps publication");
  }
  if (publication.magic != kStorageRoutePublicationMagic ||
      publication.version != kStorageRoutePublicationVersion ||
      publication.header_bytes != sizeof(StorageRoutePublication) ||
      publication.shard_id != expected_shard ||
      publication.slot_count != kStorageRouteSlots ||
      publication.code_bytes == 0 ||
      publication.code_bytes > kStorageRouteMaxCodeBytes) {
    return fail("storage route publication header mismatch");
  }
  if (publication.body_checksum !=
      storage_route_body_checksum(publication)) {
    return fail("storage route publication checksum mismatch");
  }
  for (const StorageRouteSlot& slot : publication.slots) {
    if (slot.remote_node == 0) {
      if (slot.generation == 0 && slot.id != 0) {
        return fail("storage route publication contains an invalid empty slot");
      }
      continue;
    }
    // Schema-15 immutable base nodes store generation zero; online versions
    // start at one. Both are valid canonical route representatives.
    if (static_cast<u32>(slot.remote_node >> 48) != expected_shard) {
      return fail("storage route publication contains an invalid live slot");
    }
  }
  if (error != nullptr) error->clear();
  return true;
}
namespace {

constexpr u64 kChecksumOffset = 1469598103934665603ULL;
constexpr u64 kChecksumPrime = 1099511628211ULL;
constexpr u64 kRemoteOffsetLimit = 1ull << 48;

void set_error(std::string* error, const std::string& value) {
  if (error != nullptr) *error = value;
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
      header.vector_bytes != vector_bytes || header.total_anchors > (1u << 24)) {
    throw std::runtime_error("invalid anchor sidecar for GPU entry points: " + path.string());
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
      throw std::runtime_error("invalid anchor shard for GPU entry points: " + path.string());
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
        throw std::runtime_error("anchor points outside its static GPU shard");
      }
      anchor_ordinals[shard].push_back(ordinal);
      ++loaded;
    }
  }
  if (loaded != header.total_anchors) {
    throw std::runtime_error("anchor sidecar count mismatch for GPU entry points");
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

bool validate_layout(const NavigationLayout& layout, std::string* error) {
  if (layout.dim == 0 || layout.graph_degree == 0 || layout.num_shards == 0 ||
      layout.num_nodes == 0 || layout.num_nodes >= (1ull << 30) ||
      layout.num_nodes > std::numeric_limits<u32>::max() || layout.base_generation == 0) {
    set_error(error, "GPU navigation layout has invalid dimensions");
    return false;
  }
  if (layout.quantizer_kind != static_cast<u32>(QuantizerKind::opq_pq) ||
      layout.pq_bits != 8 || layout.pq_subquantizers == 0 ||
      layout.dim % layout.pq_subquantizers != 0 ||
      layout.code_bytes != layout.pq_subquantizers || layout.model_checksum == 0) {
    set_error(error, "GPU navigation layout has an invalid PQ configuration");
    return false;
  }
  if (layout.graph_pointer_bytes != kCompactPointerBytes ||
      layout.graph_entry_bytes <
        8 + static_cast<u64>(layout.graph_degree) * kCompactPointerBytes ||
      layout.graph_entry_bytes > kMaxGraphEntryBytes ||
      layout.graph_shard_bits != shard_bits_for(layout.num_shards) ||
      layout.graph_shard_bits >= 16) {
    set_error(error, "GPU navigation requires compact graph records within one read block");
    return false;
  }
  if (layout.medoid_ordinal >= layout.num_nodes) {
    set_error(error, "GPU navigation layout has an invalid medoid");
    return false;
  }
  return true;
}

bool validate_view(const View& view, std::string* error) {
  if (view.shards.size() != view.layout.num_shards || view.entry_points.empty() ||
      view.entry_points.size() > kMaxEntryPoints || !validate_layout(view.layout, error)) {
    if (error != nullptr && error->empty()) *error = "GPU navigation view cardinality mismatch";
    return false;
  }
  u64 next_ordinal = 0;
  for (size_t shard_index = 0; shard_index < view.shards.size(); ++shard_index) {
    const ShardRegion& shard = view.shards[shard_index];
    const bool node_range_overflows = shard.node_base_offset > kRemoteOffsetLimit ||
      (shard.node_stride != 0 && shard.node_count >
       (kRemoteOffsetLimit - shard.node_base_offset) / shard.node_stride);
    const bool graph_range_overflows = shard.graph_base_offset > kRemoteOffsetLimit ||
      (view.layout.graph_entry_bytes != 0 && shard.node_count >
       (kRemoteOffsetLimit - shard.graph_base_offset) / view.layout.graph_entry_bytes);
    const bool code_range_overflows = shard.code_remote_offset > kRemoteOffsetLimit ||
      shard.code_bytes > kRemoteOffsetLimit - shard.code_remote_offset;
    const u64 node_end = node_range_overflows ? kRemoteOffsetLimit :
      shard.node_base_offset + shard.node_count * shard.node_stride;
    const u64 graph_end = graph_range_overflows ? kRemoteOffsetLimit :
      shard.graph_base_offset + shard.node_count * view.layout.graph_entry_bytes;
    if (shard.memory_node != shard_index || shard.ordinal_base != next_ordinal ||
        shard.node_count == 0 || shard.node_base_offset != kNodeBaseOffset ||
        shard.node_stride == 0 || shard.graph_base_offset == 0 ||
        shard.dynamic_base_offset == 0 || shard.control_remote_offset == 0 ||
        shard.dynamic_record_bytes == 0 ||
        shard.dynamic_hot_offset == 0 || node_range_overflows || graph_range_overflows ||
        code_range_overflows || node_end > shard.graph_base_offset ||
        graph_end > shard.control_remote_offset ||
        shard.dynamic_hot_offset < shard.node_stride ||
        shard.dynamic_hot_offset > shard.dynamic_record_bytes ||
        view.layout.graph_entry_bytes >
          shard.dynamic_record_bytes - shard.dynamic_hot_offset ||
        shard.dynamic_code_offset <
          shard.dynamic_hot_offset + view.layout.graph_entry_bytes ||
        shard.dynamic_code_offset > shard.dynamic_record_bytes ||
        view.layout.code_bytes >
          shard.dynamic_record_bytes - shard.dynamic_code_offset ||
        shard.code_remote_offset !=
          shard.control_remote_offset + kStorageControlBytes ||
        shard.dynamic_base_offset < shard.code_remote_offset + shard.code_bytes ||
        shard.code_bytes != shard.node_count * view.layout.code_bytes) {
      set_error(error, "GPU navigation layout contains an invalid shard region");
      return false;
    }
    next_ordinal += shard.node_count;
  }
  if (next_ordinal != view.layout.num_nodes) {
    set_error(error, "GPU navigation shard ranges do not cover all nodes");
    return false;
  }
  for (u32 entry : view.entry_points) {
    if (entry >= view.layout.num_nodes) {
      set_error(error, "GPU navigation layout contains an invalid entry point");
      return false;
    }
  }
  return true;
}

bool synthesize_distributed_view(
    const std::filesystem::path& index_prefix, View& view,
    const SynthesisOptions& options, bool* used_anchor_entry_points,
    std::string* error) {
  if (used_anchor_entry_points != nullptr) *used_anchor_entry_points = false;
  try {
    const std::filesystem::path metadata_path{index_prefix.string() + ".meta.json"};
    std::ifstream metadata_input(metadata_path);
    if (!metadata_input.good()) {
      throw std::runtime_error("missing index metadata: " + metadata_path.string());
    }
    nlohmann::json metadata;
    metadata_input >> metadata;
    const std::string quantizer = metadata.value("navigation_quantizer", std::string{});
    const std::string navigation_format = metadata.value("navigation_format", std::string{});
    if (metadata.value("schema_version", 0u) != kMetadataSchemaVersion ||
        metadata.value("distance", std::string{"l2"}) != "l2" ||
        metadata.value("node_layout", std::string{}) != "plain" ||
        metadata.value("storage_format", std::string{}) != "vamana_compact_v1" ||
        (quantizer != "opq_pq" && quantizer != "opq_pq16") ||
        (navigation_format != "opq_pq_graph_v1" &&
         navigation_format != "opq_pq16_graph_v1")) {
      throw std::runtime_error(
        "GPU navigation requires schema-15 compact L2 metadata with persistent dynamic PQ codes");
    }

    const u32 shard_count = metadata.at("num_memory_nodes").get<u32>();
    const std::vector<u64> counts =
      metadata.at("hot_graph_entry_counts").get<std::vector<u64>>();
    const std::vector<u64> graph_offsets =
      metadata.at("hot_graph_offsets").get<std::vector<u64>>();
    const std::vector<u64> dynamic_offsets =
      metadata.at("hot_graph_dynamic_base_offsets").get<std::vector<u64>>();
    const std::vector<u64> control_offsets =
      metadata.at("storage_control_remote_offsets").get<std::vector<u64>>();
    const std::vector<u64> dynamic_node_offsets =
      metadata.at("dynamic_node_base_offsets").get<std::vector<u64>>();
    const std::vector<u64> code_offsets =
      metadata.at("navigation_code_remote_offsets").get<std::vector<u64>>();
    const std::vector<u64> code_sizes =
      metadata.at("navigation_code_region_bytes").get<std::vector<u64>>();
    if (shard_count == 0 || counts.size() != shard_count ||
        graph_offsets.size() != shard_count || dynamic_offsets.size() != shard_count ||
        control_offsets.size() != shard_count || dynamic_node_offsets.size() != shard_count ||
        code_offsets.size() != shard_count || code_sizes.size() != shard_count) {
      throw std::runtime_error("GPU navigation metadata has invalid shard arrays");
    }

    View synthesized;
    synthesized.layout.dim = metadata.at("dim").get<u32>();
    synthesized.layout.graph_degree = metadata.at("R").get<u32>();
    const VectorDType dtype = parse_vector_dtype(
      metadata.value("vector_data_type", std::string{"float32"}));
    synthesized.layout.vector_dtype = static_cast<u32>(dtype);
    synthesized.layout.quantizer_kind = static_cast<u32>(QuantizerKind::opq_pq);
    synthesized.layout.pq_subquantizers = metadata.at("pq_subquantizers").get<u32>();
    synthesized.layout.pq_bits = metadata.at("pq_bits").get<u32>();
    synthesized.layout.code_bytes = metadata.at("navigation_code_bytes").get<u32>();
    synthesized.layout.model_checksum = metadata.at("navigation_model_checksum").get<u64>();
    synthesized.layout.num_shards = shard_count;
    synthesized.layout.graph_entry_bytes = metadata.at("hot_graph_entry_size").get<u32>();
    synthesized.layout.graph_pointer_bytes =
      metadata.at("hot_graph_pointer_bytes").get<u32>();
    synthesized.layout.graph_shard_bits = metadata.at("hot_graph_shard_bits").get<u32>();
    synthesized.layout.base_generation = 1;
    synthesized.shards.resize(shard_count);

    const u64 node_stride = metadata.at("node_size").get<u64>();
    const u32 dynamic_record_bytes = metadata.at("hot_graph_dynamic_record_bytes").get<u32>();
    const u32 dynamic_hot_offset = metadata.at("hot_graph_dynamic_hot_offset").get<u32>();
    const u32 dynamic_code_offset = metadata.at("dynamic_navigation_code_offset").get<u32>();
    u64 node_count = 0;
    for (u32 shard = 0; shard < shard_count; ++shard) {
      const u64 expected_control_offset = align_up(dynamic_offsets[shard], 64);
      const u64 expected_code_offset = expected_control_offset + kStorageControlBytes;
      const u64 expected_code_bytes = counts[shard] * synthesized.layout.code_bytes;
      if (counts[shard] == 0 || graph_offsets[shard] == 0 ||
          dynamic_offsets[shard] == 0 ||
          control_offsets[shard] != expected_control_offset ||
          code_offsets[shard] != expected_code_offset ||
          code_sizes[shard] != expected_code_bytes ||
          dynamic_node_offsets[shard] < expected_code_offset + expected_code_bytes ||
          counts[shard] >= (1ull << 30) - node_count) {
        throw std::runtime_error("GPU navigation metadata contains an invalid shard");
      }
      synthesized.shards[shard] = {
        .ordinal_base = node_count,
        .node_count = counts[shard],
        .node_base_offset = kNodeBaseOffset,
        .node_stride = node_stride,
        .graph_base_offset = graph_offsets[shard],
        .dynamic_base_offset = dynamic_node_offsets[shard],
        .control_remote_offset = control_offsets[shard],
        .code_remote_offset = code_offsets[shard],
        .code_bytes = code_sizes[shard],
        .memory_node = shard,
        .dynamic_record_bytes = dynamic_record_bytes,
        .dynamic_hot_offset = dynamic_hot_offset,
        .dynamic_code_offset = dynamic_code_offset,
      };
      node_count += counts[shard];
    }
    if (node_count != metadata.at("num_vectors").get<u64>() ||
        node_count == 0 || node_count >= (1ull << 30)) {
      throw std::runtime_error("GPU navigation metadata has an invalid node count");
    }
    synthesized.layout.num_nodes = node_count;

    const auto& medoid = metadata.at("medoid");
    const RemotePtr medoid_pointer{
      medoid.at("memory_node").get<u32>(), medoid.at("offset").get<u64>()};
    if (!remote_to_ordinal(synthesized, medoid_pointer,
                           synthesized.layout.medoid_ordinal)) {
      throw std::runtime_error("GPU navigation metadata has an invalid medoid");
    }

    const u32 requested_entry_points = options.entry_points == 0
      ? metadata.value("navigation_entry_points", 256u) : options.entry_points;
    if (requested_entry_points == 0 || requested_entry_points > kMaxEntryPoints) {
      throw std::runtime_error("GPU entry-point count must be in [1, 512]");
    }
    const u32 target = static_cast<u32>(std::min<u64>(requested_entry_points, node_count));
    std::unordered_set<u32> selected;
    selected.insert(synthesized.layout.medoid_ordinal);
    synthesized.entry_points.push_back(synthesized.layout.medoid_ordinal);
    const bool used_anchors = append_anchor_entry_points(
      index_prefix, synthesized.layout.dim, dtype,
      metadata.at("vector_bytes").get<u32>(), synthesized, target,
      selected, synthesized.entry_points);
    // The anchor-free fallback must not fill the table from the first shard
    // before later shards get a chance to contribute.  Walk shards at every
    // sample rank so the fixed entry set remains balanced even when the
    // requested count is smaller than one shard's sampling budget.
    const u32 quota = (target + shard_count - 1) / shard_count;
    for (u32 sample = 0; sample < quota * 16 &&
         synthesized.entry_points.size() < target; ++sample) {
      for (u32 shard = 0; shard < shard_count &&
           synthesized.entry_points.size() < target; ++shard) {
        const u64 slot = mix64(options.seed ^
          (static_cast<u64>(shard) << 32) ^ sample) % counts[shard];
        const u32 ordinal = static_cast<u32>(
          synthesized.shards[shard].ordinal_base + slot);
        if (selected.insert(ordinal).second) synthesized.entry_points.push_back(ordinal);
      }
    }
    for (u32 ordinal = 0; synthesized.entry_points.size() < target &&
         ordinal < node_count; ++ordinal) {
      if (selected.insert(ordinal).second) synthesized.entry_points.push_back(ordinal);
    }
    std::string validation_error;
    if (!validate_view(synthesized, &validation_error)) {
      throw std::runtime_error(validation_error);
    }
    if (used_anchor_entry_points != nullptr) *used_anchor_entry_points = used_anchors;
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
    set_error(error, "invalid GPU PQ code sidecar header");
    return false;
  }
  if (header.quantizer_kind != static_cast<u32>(QuantizerKind::opq_pq) ||
      header.entry_count == 0 || header.node_size == 0 || header.remote_offset == 0 ||
      header.code_bytes == 0 || header.model_checksum == 0 ||
      header.payload_bytes != header.entry_count * header.code_bytes) {
    set_error(error, "invalid GPU PQ code sidecar dimensions");
    return false;
  }
  CodeHeader copy = header;
  const u64 stored_checksum = copy.header_checksum;
  copy.header_checksum = 0;
  if (checksum64(reinterpret_cast<const byte_t*>(&copy), sizeof(copy)) != stored_checksum) {
    set_error(error, "GPU PQ code sidecar header checksum mismatch");
    return false;
  }
  return true;
}

bool read_code_header(const std::filesystem::path& path, CodeHeader& header,
                      std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    set_error(error, "GPU PQ code sidecar does not exist: " + path.string());
    return false;
  }
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good() || !validate_code_header(header, error)) return false;
  if (std::filesystem::file_size(path) != sizeof(CodeHeader) + header.payload_bytes) {
    set_error(error, "GPU PQ code sidecar file size mismatch: " + path.string());
    return false;
  }
  return true;
}

bool write_code_header(std::ostream& output, const CodeHeader& source,
                       std::string* error) {
  CodeHeader header = source;
  header.magic = kCodeMagic;
  header.version = kVersion;
  header.header_bytes = sizeof(CodeHeader);
  header.endian_marker = kEndianMarker;
  header.header_checksum = 0;
  header.header_checksum = checksum64(
    reinterpret_cast<const byte_t*>(&header), sizeof(header));
  if (!validate_code_header(header, error)) return false;
  output.seekp(0);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  if (!output.good()) {
    set_error(error, "failed to write GPU PQ code sidecar header");
    return false;
  }
  return true;
}

bool ordinal_to_remote(const View& view, u32 ordinal, RemotePtr& pointer) {
  if (ordinal >= view.layout.num_nodes) return false;
  const auto it = std::upper_bound(
    view.shards.begin(), view.shards.end(), ordinal,
    [](u32 value, const ShardRegion& shard) { return value < shard.ordinal_base; });
  if (it == view.shards.begin()) return false;
  const ShardRegion& shard = *(it - 1);
  const u64 slot = static_cast<u64>(ordinal) - shard.ordinal_base;
  if (slot >= shard.node_count) return false;
  pointer = RemotePtr{shard.memory_node,
    shard.node_base_offset + slot * shard.node_stride};
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
