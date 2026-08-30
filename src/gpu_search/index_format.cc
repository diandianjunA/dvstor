#include "gpu_search/index_format.hh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "common/constants.hh"
#include "nlohmann/json.hh"
#include "vamana/hot_graph.hh"
#include "vamana/vamana_node.hh"

namespace gpu_search::format {

u32 centroid_scalar_bytes(CentroidScalarType type) {
  switch (type) {
    case CentroidScalarType::float32: return sizeof(f32);
    case CentroidScalarType::float64: return sizeof(f64);
  }
  return 0;
}

u64 storage_centroid_route_publication_bytes(
    u32 dim, CentroidScalarType scalar_type, u32 live_entry_capacity) {
  const u64 scalar_bytes = centroid_scalar_bytes(scalar_type);
  if (dim == 0 || scalar_bytes == 0 || live_entry_capacity == 0 ||
      live_entry_capacity > kStorageCentroidRouteMaxLiveEntries ||
      dim > (std::numeric_limits<u64>::max() -
             sizeof(StorageCentroidRoutePublicationHeader)) / scalar_bytes) {
    return 0;
  }
  const u64 centroid_end =
    sizeof(StorageCentroidRoutePublicationHeader) +
    static_cast<u64>(dim) * scalar_bytes;
  const u64 entries_offset = align_up(
    centroid_end, alignof(StorageCentroidRouteEntry));
  const u64 entry_bytes = static_cast<u64>(live_entry_capacity) *
    sizeof(StorageCentroidRouteEntry);
  if (entries_offset > std::numeric_limits<u64>::max() - entry_bytes) return 0;
  return align_up(entries_offset + entry_bytes, 64);
}

bool validate_storage_centroid_route_descriptor(
    const StorageCentroidRouteDescriptor& descriptor,
    u32 expected_dim, u32 expected_shards, std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  const auto scalar_type =
    static_cast<CentroidScalarType>(descriptor.centroid_scalar_type);
  const u64 expected_bytes = storage_centroid_route_publication_bytes(
    descriptor.dim, scalar_type, descriptor.live_entry_capacity);
  if (descriptor.magic != kStorageCentroidRouteDescriptorMagic ||
      descriptor.version != kStorageCentroidRouteDescriptorVersion ||
      descriptor.descriptor_bytes != sizeof(StorageCentroidRouteDescriptor) ||
      descriptor.layout_version == 0 || descriptor.remote_offset == 0 ||
      descriptor.remote_offset % 64 != 0 || descriptor.dim != expected_dim ||
      descriptor.shard_count != expected_shards || expected_bytes == 0 ||
      descriptor.publication_bytes != expected_bytes ||
      descriptor.reserved != 0 ||
      descriptor.remote_offset > std::numeric_limits<u64>::max() -
        descriptor.publication_bytes) {
    return fail("storage centroid route descriptor mismatch");
  }
  if (error != nullptr) error->clear();
  return true;
}

u64 storage_centroid_route_body_checksum(span<const byte_t> publication) {
  if (publication.size() < sizeof(StorageCentroidRoutePublicationHeader)) {
    return 0;
  }
  const auto* header = reinterpret_cast<
    const StorageCentroidRoutePublicationHeader*>(publication.data());
  if (header->header_bytes != sizeof(StorageCentroidRoutePublicationHeader) ||
      header->total_bytes < header->header_bytes ||
      header->total_bytes > publication.size()) {
    return 0;
  }
  u64 checksum = checksum64_initial();
  checksum = checksum64_update(
    checksum, reinterpret_cast<const byte_t*>(&header->magic),
    offsetof(StorageCentroidRoutePublicationHeader, body_checksum) -
      offsetof(StorageCentroidRoutePublicationHeader, magic));
  checksum = checksum64_update(
    checksum, publication.data() + header->header_bytes,
    header->total_bytes - header->header_bytes);
  return checksum;
}

const void* storage_centroid_route_centroid_data(
    span<const byte_t> publication) {
  if (publication.size() < sizeof(StorageCentroidRoutePublicationHeader)) {
    return nullptr;
  }
  const auto* header = reinterpret_cast<
    const StorageCentroidRoutePublicationHeader*>(publication.data());
  if (header->centroid_offset > publication.size() ||
      header->centroid_bytes > publication.size() - header->centroid_offset) {
    return nullptr;
  }
  return publication.data() + header->centroid_offset;
}

span<const StorageCentroidRouteEntry> storage_centroid_route_entries(
    span<const byte_t> publication) {
  if (publication.size() < sizeof(StorageCentroidRoutePublicationHeader)) {
    return {};
  }
  const auto* header = reinterpret_cast<
    const StorageCentroidRoutePublicationHeader*>(publication.data());
  const u64 bytes = static_cast<u64>(header->live_entry_count) *
    sizeof(StorageCentroidRouteEntry);
  if (header->entries_offset % alignof(StorageCentroidRouteEntry) != 0 ||
      header->entries_offset > publication.size() ||
      bytes > publication.size() - header->entries_offset) {
    return {};
  }
  return {reinterpret_cast<const StorageCentroidRouteEntry*>(
            publication.data() + header->entries_offset),
          header->live_entry_count};
}

bool prepare_storage_centroid_route_publication(
    span<byte_t> publication, u32 shard, u32 dim,
    CentroidScalarType scalar_type, u32 live_entry_capacity,
    u64 shard_version, u64 vector_count, const void* centroid_data,
    span<const StorageCentroidRouteEntry> live_entries,
    std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  const u64 expected_bytes = storage_centroid_route_publication_bytes(
    dim, scalar_type, live_entry_capacity);
  if (expected_bytes == 0 || publication.size() != expected_bytes ||
      shard_version == 0 || live_entries.size() > live_entry_capacity ||
      (vector_count == 0) != live_entries.empty() ||
      (vector_count != 0 && centroid_data == nullptr)) {
    return fail("invalid storage centroid route publication input");
  }
  for (size_t index = 0; index < live_entries.size(); ++index) {
    const StorageCentroidRouteEntry& entry = live_entries[index];
    const RemotePtr pointer{entry.remote_node};
    if (pointer.is_null() || !pointer.is_well_formed() ||
        pointer.memory_node() != shard ||
        entry.flags != kStorageCentroidRouteLive) {
      return fail("invalid storage centroid route live entry");
    }
    for (size_t prior = 0; prior < index; ++prior) {
      if (live_entries[prior].remote_node == entry.remote_node) {
        return fail("duplicate storage centroid route live entry");
      }
    }
  }

  std::fill(publication.begin(), publication.end(), byte_t{0});
  auto* header = reinterpret_cast<
    StorageCentroidRoutePublicationHeader*>(publication.data());
  header->sequence = 2;
  header->magic = kStorageCentroidRoutePublicationMagic;
  header->version = kStorageCentroidRoutePublicationVersion;
  header->header_bytes = sizeof(StorageCentroidRoutePublicationHeader);
  header->total_bytes = expected_bytes;
  header->shard_id = shard;
  header->dim = dim;
  header->centroid_scalar_type = static_cast<u32>(scalar_type);
  header->live_entry_count = static_cast<u32>(live_entries.size());
  header->live_entry_capacity = live_entry_capacity;
  header->shard_version = shard_version;
  header->vector_count = vector_count;
  header->centroid_offset = sizeof(StorageCentroidRoutePublicationHeader);
  header->centroid_bytes =
    static_cast<u64>(dim) * centroid_scalar_bytes(scalar_type);
  header->entries_offset = align_up(
    static_cast<u64>(header->centroid_offset) + header->centroid_bytes,
    alignof(StorageCentroidRouteEntry));
  header->entries_bytes =
    static_cast<u64>(live_entry_capacity) *
      sizeof(StorageCentroidRouteEntry);
  if (centroid_data != nullptr) {
    std::memcpy(publication.data() + header->centroid_offset,
                centroid_data, header->centroid_bytes);
  }
  if (!live_entries.empty()) {
    std::memcpy(publication.data() + header->entries_offset,
                live_entries.data(), live_entries.size() *
                  sizeof(StorageCentroidRouteEntry));
  }
  header->body_checksum = storage_centroid_route_body_checksum(publication);
  if (error != nullptr) error->clear();
  return true;
}

bool validate_storage_centroid_route_publication(
    span<const byte_t> publication,
    const StorageCentroidRouteDescriptor& descriptor,
    u32 expected_shard, std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (publication.size() != descriptor.publication_bytes ||
      publication.size() < sizeof(StorageCentroidRoutePublicationHeader)) {
    return fail("storage centroid route publication size mismatch");
  }
  const auto* header = reinterpret_cast<
    const StorageCentroidRoutePublicationHeader*>(publication.data());
  const auto scalar_type =
    static_cast<CentroidScalarType>(header->centroid_scalar_type);
  const u32 scalar_bytes = centroid_scalar_bytes(scalar_type);
  const u64 expected_bytes = storage_centroid_route_publication_bytes(
    header->dim, scalar_type, header->live_entry_capacity);
  const u64 expected_entries_offset = align_up(
    sizeof(StorageCentroidRoutePublicationHeader) +
      static_cast<u64>(header->dim) * scalar_bytes,
    alignof(StorageCentroidRouteEntry));
  if (header->sequence == 0 || (header->sequence & 1u) != 0) {
    return fail("storage centroid route snapshot overlaps publication");
  }
  if (header->magic != kStorageCentroidRoutePublicationMagic ||
      header->version != kStorageCentroidRoutePublicationVersion ||
      header->header_bytes != sizeof(StorageCentroidRoutePublicationHeader) ||
      header->total_bytes != publication.size() ||
      header->shard_id != expected_shard ||
      header->dim != descriptor.dim ||
      header->centroid_scalar_type != descriptor.centroid_scalar_type ||
      header->live_entry_capacity != descriptor.live_entry_capacity ||
      header->live_entry_count > header->live_entry_capacity ||
      header->reserved0 != 0 || header->reserved[0] != 0 ||
      header->reserved[1] != 0 || header->shard_version == 0 ||
      expected_bytes != publication.size() ||
      header->centroid_offset != sizeof(StorageCentroidRoutePublicationHeader) ||
      header->centroid_bytes != header->dim * scalar_bytes ||
      header->entries_offset != expected_entries_offset ||
      header->entries_bytes != header->live_entry_capacity *
        sizeof(StorageCentroidRouteEntry) ||
      (header->vector_count == 0) != (header->live_entry_count == 0)) {
    return fail("storage centroid route publication header mismatch");
  }
  if (header->body_checksum !=
      storage_centroid_route_body_checksum(publication)) {
    return fail("storage centroid route publication checksum mismatch");
  }

  const void* centroid = storage_centroid_route_centroid_data(publication);
  if (centroid == nullptr) {
    return fail("storage centroid route centroid range mismatch");
  }
  for (u32 dimension = 0; dimension < header->dim; ++dimension) {
    const f64 value = scalar_type == CentroidScalarType::float32
      ? static_cast<const f32*>(centroid)[dimension]
      : static_cast<const f64*>(centroid)[dimension];
    if (!floating_value_is_finite(value)) {
      return fail("storage centroid route contains a non-finite centroid");
    }
  }
  const auto entries = storage_centroid_route_entries(publication);
  if (entries.size() != header->live_entry_count) {
    return fail("storage centroid route entry range mismatch");
  }
  for (size_t index = 0; index < entries.size(); ++index) {
    const RemotePtr pointer{entries[index].remote_node};
    if (pointer.is_null() || !pointer.is_well_formed() ||
        pointer.memory_node() != expected_shard ||
        entries[index].flags != kStorageCentroidRouteLive) {
      return fail("storage centroid route contains an invalid live entry");
    }
    for (size_t prior = 0; prior < index; ++prior) {
      if (entries[prior].remote_node == entries[index].remote_node) {
        return fail("storage centroid route contains duplicate live entries");
      }
    }
  }
  if (error != nullptr) error->clear();
  return true;
}
namespace {

constexpr u64 kChecksumOffset = 1469598103934665603ULL;
constexpr u64 kChecksumPrime = 1099511628211ULL;
constexpr u64 kRemoteOffsetLimit = RemotePtr::BYTE_OFFSET_CAPACITY;

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
      layout.num_nodes == 0 || layout.num_nodes > kMaxGpuNavigationNodes ||
      layout.num_nodes > std::numeric_limits<u32>::max() || layout.base_generation == 0) {
    set_error(error, "GPU navigation layout has invalid dimensions");
    return false;
  }
  if (layout.graph_degree > kMaxSupportedGraphDegree) {
    set_error(error, "GPU navigation graph degree exceeds the system-wide limit");
    return false;
  }
  if (layout.num_shards > RemotePtr::MEMORY_NODE_MASK + 1) {
    set_error(error, "GPU navigation shard count exceeds tagged RemotePtr capacity");
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
        vamana::hot_graph::kTaggedNeighborBaseOffset +
          static_cast<u64>(layout.graph_degree) * kCompactPointerBytes ||
      layout.graph_entry_bytes > kMaxGraphEntryBytes ||
      layout.graph_shard_bits != shard_bits_for(layout.num_shards) ||
      layout.graph_shard_bits > RemotePtr::MEMORY_NODE_BITS) {
    set_error(error, "GPU navigation requires tagged graph records within one read block");
    return false;
  }
  return true;
}

bool validate_view(const View& view, std::string* error) {
  if (view.shards.size() != view.layout.num_shards ||
      !validate_layout(view.layout, error)) {
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
    const bool control_range_overflows =
      shard.control_remote_offset > kRemoteOffsetLimit ||
      kStorageControlBytes >
        kRemoteOffsetLimit - shard.control_remote_offset;
    const bool dynamic_range_overflows =
      !RemotePtr::representable(
        shard.memory_node, shard.dynamic_base_offset, 1) ||
      shard.dynamic_record_bytes >
        kRemoteOffsetLimit - shard.dynamic_base_offset;
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
        code_range_overflows || control_range_overflows ||
        dynamic_range_overflows || node_end > shard.graph_base_offset ||
        graph_end > shard.control_remote_offset ||
        shard.dynamic_hot_offset < shard.node_stride ||
        shard.dynamic_hot_offset > shard.dynamic_record_bytes ||
        view.layout.graph_entry_bytes >
          shard.dynamic_record_bytes - shard.dynamic_hot_offset ||
        shard.dynamic_code_offset !=
          shard.dynamic_hot_offset + view.layout.graph_entry_bytes ||
        shard.dynamic_code_offset > shard.dynamic_record_bytes ||
        VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
          static_cast<u64>(view.layout.code_bytes) +
          VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES >
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
  return true;
}

bool synthesize_distributed_view(
    const std::filesystem::path& index_prefix, View& view,
    std::string* error) {
  try {
    const std::filesystem::path metadata_path{index_prefix.string() + ".meta.json"};
    std::ifstream metadata_input(metadata_path, std::ios::binary);
    if (!metadata_input.good()) {
      throw std::runtime_error("missing index metadata: " + metadata_path.string());
    }
    metadata_input.seekg(0, std::ios::end);
    const std::streamoff metadata_end = metadata_input.tellg();
    constexpr u64 kMaximumMetadataBytes = 4ull << 20;
    if (metadata_end <= 0 ||
        static_cast<u64>(metadata_end) > kMaximumMetadataBytes) {
      throw std::runtime_error(
        "index metadata exceeds the runtime JSON safety limit");
    }
    std::string metadata_document(
      static_cast<size_t>(metadata_end), '\0');
    metadata_input.seekg(0, std::ios::beg);
    metadata_input.read(
      metadata_document.data(),
      static_cast<std::streamsize>(metadata_document.size()));
    if (metadata_input.gcount() !=
          static_cast<std::streamsize>(metadata_document.size())) {
      throw std::runtime_error("short read from index metadata");
    }
    char trailing_byte = 0;
    if (metadata_input.get(trailing_byte)) {
      throw std::runtime_error("index metadata changed while it was read");
    }
    const nlohmann::json metadata =
      nlohmann::json::parse(metadata_document);
    if (!metadata.is_object()) {
      throw std::runtime_error("index metadata root is not an object");
    }
    const std::string quantizer = metadata.value("navigation_quantizer", std::string{});
    const std::string navigation_format = metadata.value("navigation_format", std::string{});
    if (metadata.value("schema_version", 0u) != kMetadataSchemaVersion ||
        metadata.value("distance", std::string{"l2"}) != "l2" ||
        metadata.value("node_layout", std::string{}) != "plain" ||
        metadata.value("storage_format", std::string{}) != "vamana_tagged_v2" ||
        metadata.value("remote_ptr_format", std::string{}) !=
          "tagged_inc24_shard6_off34x16_v1" ||
        metadata.value("centroid_state_format", std::string{}) !=
          "physical_shard_centroid_v2_bound" ||
        metadata.value("index_build_fingerprint", 0ull) == 0 ||
        metadata.value("slot_incarnation_offset", 0u) !=
          VamanaNode::offset_slot_incarnation() ||
        quantizer != "opq_pq" ||
        navigation_format != "opq_pq_graph_v1") {
      throw std::runtime_error(
        "GPU navigation requires schema-16 tagged L2 metadata with persistent dynamic PQ codes");
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
    const std::vector<u64> shard_fingerprints =
      metadata.at("shard_build_fingerprints").get<std::vector<u64>>();
    if (shard_count == 0 || counts.size() != shard_count ||
        graph_offsets.size() != shard_count || dynamic_offsets.size() != shard_count ||
        control_offsets.size() != shard_count || dynamic_node_offsets.size() != shard_count ||
        code_offsets.size() != shard_count || code_sizes.size() != shard_count ||
        shard_fingerprints.size() != shard_count ||
        std::find(shard_fingerprints.begin(), shard_fingerprints.end(), 0) !=
          shard_fingerprints.end()) {
      throw std::runtime_error("GPU navigation metadata has invalid shard arrays");
    }

    View synthesized;
    synthesized.layout.dim = metadata.at("dim").get<u32>();
    synthesized.layout.graph_degree = metadata.at("R").get<u32>();
    synthesized.layout.vector_dtype = static_cast<u32>(parse_vector_dtype(
      metadata.value("vector_data_type", std::string{"float32"})));
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
    if (metadata.value("dynamic_navigation_code_validation_bytes", 0u) !=
        VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES) {
      throw std::runtime_error(
        "GPU navigation metadata lacks dynamic slot-incarnation validation");
    }
    if (metadata.value("dynamic_navigation_code_checksum_bytes",
                       VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES) !=
        VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES) {
      throw std::runtime_error(
        "GPU navigation metadata has an incompatible dynamic PQ checksum");
    }
    u64 node_count = 0;
    for (u32 shard = 0; shard < shard_count; ++shard) {
      const u64 expected_control_offset = align_up(dynamic_offsets[shard], 64);
      if (expected_control_offset == 0 ||
          expected_control_offset >
            std::numeric_limits<u64>::max() - kStorageControlBytes ||
          synthesized.layout.code_bytes == 0 ||
          counts[shard] >
            std::numeric_limits<u64>::max() /
              synthesized.layout.code_bytes) {
        throw std::runtime_error(
          "GPU navigation metadata shard layout overflows");
      }
      const u64 expected_code_offset = expected_control_offset + kStorageControlBytes;
      const u64 expected_code_bytes = counts[shard] * synthesized.layout.code_bytes;
      if (expected_code_offset >
          std::numeric_limits<u64>::max() - expected_code_bytes) {
        throw std::runtime_error(
          "GPU navigation metadata code region overflows");
      }
      const u64 expected_code_end =
        expected_code_offset + expected_code_bytes;
      if (counts[shard] == 0 || graph_offsets[shard] == 0 ||
          dynamic_offsets[shard] == 0 ||
          control_offsets[shard] != expected_control_offset ||
          code_offsets[shard] != expected_code_offset ||
          code_sizes[shard] != expected_code_bytes ||
          dynamic_node_offsets[shard] < expected_code_end ||
          counts[shard] > kMaxGpuNavigationNodes - node_count) {
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
        node_count == 0 || node_count > kMaxGpuNavigationNodes) {
      throw std::runtime_error("GPU navigation metadata has an invalid node count");
    }
    synthesized.layout.num_nodes = node_count;

    std::string validation_error;
    if (!validate_view(synthesized, &validation_error)) {
      throw std::runtime_error(validation_error);
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
    set_error(error, "invalid GPU PQ code sidecar header");
    return false;
  }
  if (header.quantizer_kind != static_cast<u32>(QuantizerKind::opq_pq) ||
      header.entry_count == 0 || header.node_size == 0 || header.remote_offset == 0 ||
      header.code_bytes == 0 ||
      header.vector_dtype > static_cast<u32>(VectorDType::int8) ||
      header.model_checksum == 0 || header.build_fingerprint == 0 ||
      header.shard_fingerprint == 0 || header.reserved[0] != 0 ||
      header.reserved[1] != 0 ||
      header.entry_count >
        std::numeric_limits<u64>::max() / header.code_bytes ||
      header.payload_bytes != header.entry_count * header.code_bytes ||
      header.payload_bytes >
        std::numeric_limits<u64>::max() - sizeof(CodeHeader) ||
      header.remote_offset >= RemotePtr::BYTE_OFFSET_CAPACITY ||
      header.payload_bytes >
        RemotePtr::BYTE_OFFSET_CAPACITY - header.remote_offset) {
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
  std::error_code file_error;
  const std::uintmax_t file_bytes =
    std::filesystem::file_size(path, file_error);
  if (file_error || file_bytes > std::numeric_limits<u64>::max() ||
      static_cast<u64>(file_bytes) !=
        sizeof(CodeHeader) + header.payload_bytes) {
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

u32 graph_extent_class(u32 live_neighbors) {
  return live_neighbors / kGraphExtentQuantum +
    (live_neighbors % kGraphExtentQuantum != 0 ? 1u : 0u);
}

u32 graph_extent_read_bytes(u32 extent_class, u32 graph_entry_bytes) {
  constexpr u64 header_bytes = vamana::hot_graph::kTaggedNeighborBaseOffset;
  const u64 requested = header_bytes +
    static_cast<u64>(extent_class) * kGraphExtentQuantum *
      kCompactPointerBytes;
  return static_cast<u32>(std::min<u64>(requested, graph_entry_bytes));
}

bool validate_graph_extent_header(
    const GraphExtentHeader& header, std::string* error) {
  const auto fail = [&](const std::string& message) {
    set_error(error, message);
    return false;
  };
  if (header.magic != kGraphExtentMagic ||
      header.version != kGraphExtentVersion ||
      header.header_bytes != sizeof(GraphExtentHeader) ||
      header.endian_marker != kEndianMarker ||
      header.extent_quantum != kGraphExtentQuantum ||
      header.class_bytes != kGraphExtentClassBytes ||
      header.graph_pointer_bytes != kCompactPointerBytes ||
      header.graph_entry_bytes <
        vamana::hot_graph::kTaggedNeighborBaseOffset ||
      header.graph_entry_bytes > kMaxGraphEntryBytes ||
      header.graph_entry_capacity == 0 ||
      header.graph_entry_capacity >
        (kMaxGraphEntryBytes -
         vamana::hot_graph::kTaggedNeighborBaseOffset) /
          kCompactPointerBytes ||
      header.graph_entry_bytes !=
        vamana::hot_graph::kTaggedNeighborBaseOffset +
          static_cast<u64>(header.graph_entry_capacity) *
            kCompactPointerBytes ||
      header.num_shards == 0 ||
      header.num_shards > RemotePtr::MEMORY_NODE_MASK + 1 ||
      header.reserved0 != 0 || header.num_nodes == 0 ||
      header.num_nodes > kMaxGpuNavigationNodes ||
      header.num_nodes >
        std::numeric_limits<u64>::max() / header.class_bytes ||
      header.payload_bytes != header.num_nodes * header.class_bytes ||
      header.build_fingerprint == 0 ||
      std::any_of(
        header.reserved.begin(), header.reserved.end(),
        [](u64 value) { return value != 0; })) {
    return fail("invalid graph extent sidecar header");
  }
  GraphExtentHeader copy = header;
  const u64 stored_checksum = copy.header_checksum;
  copy.header_checksum = 0;
  if (checksum64(
        reinterpret_cast<const byte_t*>(&copy), sizeof(copy)) !=
      stored_checksum) {
    return fail("graph extent sidecar header checksum mismatch");
  }
  if (error != nullptr) error->clear();
  return true;
}

bool read_graph_extent_header(
    const std::filesystem::path& path, GraphExtentHeader& header,
    std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    set_error(
      error, "GPU graph extent sidecar does not exist: " + path.string());
    return false;
  }
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good() || !validate_graph_extent_header(header, error)) {
    return false;
  }
  std::error_code file_error;
  const u64 file_bytes = std::filesystem::file_size(path, file_error);
  if (file_error ||
      header.payload_bytes >
        std::numeric_limits<u64>::max() - sizeof(GraphExtentHeader) ||
      file_bytes != sizeof(GraphExtentHeader) + header.payload_bytes) {
    set_error(
      error, "GPU graph extent sidecar file size mismatch: " +
        path.string());
    return false;
  }
  if (error != nullptr) error->clear();
  return true;
}

bool read_graph_extent_sidecar(
    const std::filesystem::path& path, GraphExtentHeader& header,
    std::vector<u8>& classes, std::string* error) {
  classes.clear();
  if (!read_graph_extent_header(path, header, error)) return false;
  if (header.payload_bytes > std::numeric_limits<size_t>::max() ||
      header.payload_bytes >
        static_cast<u64>(std::numeric_limits<std::streamsize>::max())) {
    set_error(
      error, "GPU graph extent sidecar payload exceeds host I/O limits");
    return false;
  }
  std::ifstream input(path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(sizeof(GraphExtentHeader)));
  classes.resize(static_cast<size_t>(header.payload_bytes));
  input.read(
    reinterpret_cast<char*>(classes.data()),
    static_cast<std::streamsize>(classes.size()));
  if (!input.good() ||
      static_cast<size_t>(input.gcount()) != classes.size()) {
    classes.clear();
    set_error(
      error, "short read from GPU graph extent sidecar: " + path.string());
    return false;
  }
  const u32 maximum_class =
    graph_extent_class(header.graph_entry_capacity);
  if (std::any_of(
        classes.begin(), classes.end(),
        [maximum_class](u8 extent_class) {
          return static_cast<u32>(extent_class) > maximum_class;
        })) {
    classes.clear();
    set_error(error, "GPU graph extent sidecar contains an invalid class");
    return false;
  }
  if (checksum64(classes.data(), classes.size()) !=
      header.payload_checksum) {
    classes.clear();
    set_error(
      error, "GPU graph extent sidecar payload checksum mismatch: " +
        path.string());
    return false;
  }
  if (error != nullptr) error->clear();
  return true;
}

bool write_graph_extent_header(
    std::ostream& output, const GraphExtentHeader& source,
    std::string* error) {
  GraphExtentHeader header = source;
  header.magic = kGraphExtentMagic;
  header.version = kGraphExtentVersion;
  header.header_bytes = sizeof(GraphExtentHeader);
  header.endian_marker = kEndianMarker;
  header.extent_quantum = kGraphExtentQuantum;
  header.class_bytes = kGraphExtentClassBytes;
  header.graph_pointer_bytes = kCompactPointerBytes;
  header.header_checksum = 0;
  header.header_checksum = checksum64(
    reinterpret_cast<const byte_t*>(&header), sizeof(header));
  if (!validate_graph_extent_header(header, error)) return false;
  output.seekp(0);
  output.write(
    reinterpret_cast<const char*>(&header), sizeof(header));
  if (!output.good()) {
    set_error(error, "failed to write GPU graph extent sidecar header");
    return false;
  }
  if (error != nullptr) error->clear();
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
  if (pointer.is_null() || pointer.incarnation() != 0 ||
      pointer.memory_node() >= view.shards.size()) return false;
  const ShardRegion& shard = view.shards[pointer.memory_node()];
  if (pointer.byte_offset() < shard.node_base_offset || shard.node_stride == 0) return false;
  const u64 relative = pointer.byte_offset() - shard.node_base_offset;
  if (relative % shard.node_stride != 0) return false;
  const u64 slot = relative / shard.node_stride;
  if (slot >= shard.node_count || shard.ordinal_base + slot > kMaxGpuNavigationNodes) return false;
  ordinal = static_cast<u32>(shard.ordinal_base + slot);
  return true;
}

}  // namespace gpu_search::format
