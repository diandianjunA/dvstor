#include "service/index_metadata.hh"

#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "common/constants.hh"
#include "gpu_search/index_format.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "vamana/hot_graph.hh"

namespace service::index_metadata {

namespace {

constexpr u64 kMaxMetadataFileBytes = 4ull << 20;

bool fail(str* error_message, const str& message) {
  if (error_message != nullptr) {
    *error_message = message;
  }
  return false;
}

[[noreturn]] void invalid_field(const str& name, const str& reason) {
  throw std::invalid_argument("index metadata field '" + name + "' " +
                              reason);
}

const nlohmann::json* find_field(const nlohmann::json& root,
                                 const char* name) {
  const auto found = root.find(name);
  return found == root.end() ? nullptr : std::addressof(*found);
}

template <typename T>
T unsigned_value(const nlohmann::json& value, const str& name) {
  if (!value.is_number_unsigned()) {
    invalid_field(name, "must be an unsigned integer");
  }
  const u64 parsed = value.get<u64>();
  if (parsed > static_cast<u64>(std::numeric_limits<T>::max())) {
    invalid_field(name, "is out of range");
  }
  return static_cast<T>(parsed);
}

template <typename T>
T required_unsigned(const nlohmann::json& root, const char* name) {
  const nlohmann::json* value = find_field(root, name);
  if (value == nullptr) invalid_field(name, "is missing");
  return unsigned_value<T>(*value, name);
}

template <typename T>
T optional_unsigned(const nlohmann::json& root, const char* name,
                    T fallback) {
  const nlohmann::json* value = find_field(root, name);
  return value == nullptr ? fallback : unsigned_value<T>(*value, name);
}

str required_string(const nlohmann::json& root, const char* name) {
  const nlohmann::json* value = find_field(root, name);
  if (value == nullptr) invalid_field(name, "is missing");
  if (!value->is_string()) invalid_field(name, "must be a string");
  return value->get<str>();
}

str optional_string(const nlohmann::json& root, const char* name,
                    str fallback = {}) {
  const nlohmann::json* value = find_field(root, name);
  if (value == nullptr) return fallback;
  if (!value->is_string()) invalid_field(name, "must be a string");
  return value->get<str>();
}

f64 optional_number(const nlohmann::json& root, const char* name,
                    f64 fallback) {
  const nlohmann::json* value = find_field(root, name);
  if (value == nullptr) return fallback;
  if (!value->is_number()) invalid_field(name, "must be a number");
  return value->get<f64>();
}

vec<u64> optional_u64_array(const nlohmann::json& root, const char* name,
                            size_t maximum_elements) {
  const nlohmann::json* value = find_field(root, name);
  if (value == nullptr) return {};
  if (!value->is_array()) invalid_field(name, "must be an array");
  if (value->size() > maximum_elements) {
    invalid_field(name, "has too many elements");
  }
  vec<u64> result;
  result.reserve(value->size());
  for (size_t index = 0; index < value->size(); ++index) {
    result.push_back(unsigned_value<u64>(
      (*value)[index], str{name} + "[" + std::to_string(index) + "]"));
  }
  return result;
}

void require(bool condition, const str& message) {
  if (!condition) throw std::invalid_argument(message);
}

u64 checked_add(u64 lhs, u64 rhs, const char* description) {
  if (lhs > std::numeric_limits<u64>::max() - rhs) {
    throw std::overflow_error(str{description} + " overflows");
  }
  return lhs + rhs;
}

u64 checked_multiply(u64 lhs, u64 rhs, const char* description) {
  if (lhs != 0 && rhs > std::numeric_limits<u64>::max() / lhs) {
    throw std::overflow_error(str{description} + " overflows");
  }
  return lhs * rhs;
}

u64 checked_align(u64 value, u64 alignment, const char* description) {
  require(alignment != 0, str{description} + " has zero alignment");
  const u64 remainder = value % alignment;
  return remainder == 0
    ? value
    : checked_add(value, alignment - remainder, description);
}

u32 expected_shard_bits(u32 shards) {
  u32 bits = 0;
  u32 capacity = 1;
  while (capacity < shards) {
    capacity <<= 1;
    ++bits;
  }
  return bits;
}

void validate_schema16(const Metadata& metadata) {
  using gpu_search::format::kMetadataSchemaVersion;
  using gpu_search::format::kNodeBaseOffset;
  using gpu_search::format::kStorageControlBytes;

  require(metadata.schema_version == kMetadataSchemaVersion,
          "index metadata schema is not schema 16");
  require(metadata.distance == "l2" && metadata.node_layout == "plain" &&
            metadata.storage_format == "vamana_tagged_v2" &&
            metadata.remote_ptr_format ==
              "tagged_inc24_shard6_off34x16_v1",
          "schema-16 index metadata has an incompatible storage format");
  require(metadata.navigation_quantizer == "opq_pq" &&
            metadata.navigation_format == "opq_pq_graph_v1" &&
            metadata.pq_bits == 8 && metadata.pq_subquantizers != 0 &&
            metadata.pq_subquantizers <= kMaxPersistentSubquantizers &&
            metadata.navigation_code_bytes == metadata.pq_subquantizers &&
            metadata.dim % metadata.pq_subquantizers == 0 &&
            metadata.navigation_model_checksum != 0,
          "schema-16 index metadata has an incompatible OPQ/PQ layout");
  require(metadata.idmap_format == "owner_sharded_v2_bound" &&
            metadata.centroid_state_format ==
              "physical_shard_centroid_v2_bound" &&
            metadata.index_build_fingerprint != 0,
          "schema-16 index metadata has no build-bound sidecars");

  const size_t shards = metadata.num_memory_nodes;
  const auto exact_shard_array = [shards](const vec<u64>& values) {
    return values.size() == shards;
  };
  require(exact_shard_array(metadata.hot_graph_offsets) &&
            exact_shard_array(metadata.hot_graph_entry_counts) &&
            exact_shard_array(metadata.hot_graph_dynamic_base_offsets) &&
            exact_shard_array(metadata.storage_control_remote_offsets) &&
            exact_shard_array(metadata.dynamic_node_base_offsets) &&
            exact_shard_array(metadata.shard_build_fingerprints) &&
            exact_shard_array(metadata.navigation_code_remote_offsets) &&
            exact_shard_array(metadata.navigation_code_region_bytes),
          "schema-16 index metadata has invalid shard-array cardinality");

  const size_t component_bytes =
    vector_dtype_component_size(metadata.vector_dtype);
  const u64 expected_vector_bytes = checked_multiply(
    metadata.dim, component_bytes, "schema-16 vector byte width");
  const u64 vector_storage_bytes = checked_align(
    expected_vector_bytes, 8, "schema-16 vector storage width");
  const u64 expected_node_size = checked_align(
    checked_add(24, vector_storage_bytes, "schema-16 node size"),
    16, "schema-16 node size");
  require(metadata.vector_component_size == component_bytes &&
            metadata.vector_bytes == expected_vector_bytes &&
            metadata.graph_hot_bytes == 24 &&
            metadata.vector_offset == 24 &&
            metadata.slot_incarnation_offset == 16 &&
            metadata.node_size == expected_node_size,
          "schema-16 index metadata has an incompatible fixed-node layout");

  const u32 provisional_slots = std::min<u32>(
    15, std::max<u32>(2, (metadata.R + 15) / 16));
  const u64 expected_graph_entry_bytes = checked_align(
    checked_add(
      vamana::hot_graph::kTaggedNeighborBaseOffset,
      checked_multiply(metadata.R + provisional_slots,
                       vamana::hot_graph::kCompactPointerBytes,
                       "schema-16 graph entry width"),
      "schema-16 graph entry width"),
    8, "schema-16 graph entry width");
  require(metadata.hot_graph_pointer_bytes ==
              vamana::hot_graph::kCompactPointerBytes &&
            metadata.hot_graph_entry_size == expected_graph_entry_bytes &&
            metadata.hot_graph_shard_bits ==
              expected_shard_bits(metadata.num_memory_nodes),
          "schema-16 index metadata has an incompatible compact graph layout");

  const u64 expected_dynamic_code_offset = checked_add(
    metadata.node_size, metadata.hot_graph_entry_size,
    "schema-16 dynamic code offset");
  const u64 expected_dynamic_record_bytes = checked_align(
    checked_add(
      expected_dynamic_code_offset,
      checked_add(
        metadata.dynamic_navigation_code_validation_bytes,
        checked_add(metadata.navigation_code_bytes,
                    metadata.dynamic_navigation_code_checksum_bytes,
                    "schema-16 dynamic PQ trailer"),
        "schema-16 dynamic PQ record"),
      "schema-16 dynamic record"),
    16, "schema-16 dynamic record");
  require(metadata.hot_graph_dynamic_hot_offset == metadata.node_size &&
            metadata.dynamic_navigation_code_offset ==
              expected_dynamic_code_offset &&
            metadata.dynamic_navigation_code_validation_bytes == sizeof(u32) &&
            metadata.dynamic_navigation_code_checksum_bytes == sizeof(u32) &&
            metadata.hot_graph_dynamic_record_bytes ==
              expected_dynamic_record_bytes &&
            metadata.allocation_size == expected_dynamic_record_bytes,
          "schema-16 index metadata has an incompatible dynamic-node layout");

  u64 total_nodes = 0;
  for (u32 shard = 0; shard < metadata.num_memory_nodes; ++shard) {
    const u64 count = metadata.hot_graph_entry_counts[shard];
    require(count != 0 && metadata.shard_build_fingerprints[shard] != 0,
            "schema-16 index metadata contains an empty or unbound shard");
    total_nodes = checked_add(total_nodes, count,
                              "schema-16 aggregate node count");

    const u64 fixed_end = checked_add(
      kNodeBaseOffset,
      checked_multiply(count, metadata.node_size,
                       "schema-16 fixed-node range"),
      "schema-16 fixed-node range");
    const u64 graph_header_offset = checked_align(
      fixed_end, 64, "schema-16 graph header offset");
    const u64 expected_graph_offset = checked_align(
      checked_add(graph_header_offset, sizeof(vamana::hot_graph::Header),
                  "schema-16 graph offset"),
      64, "schema-16 graph offset");
    const u64 graph_end = checked_add(
      expected_graph_offset,
      checked_multiply(count, metadata.hot_graph_entry_size,
                       "schema-16 graph range"),
      "schema-16 graph range");
    const u64 expected_static_dynamic_base = checked_align(
      graph_end, 64, "schema-16 static dynamic base");
    const u64 expected_control_offset = checked_align(
      expected_static_dynamic_base, 64, "schema-16 control offset");
    const u64 expected_code_offset = checked_add(
      expected_control_offset, kStorageControlBytes,
      "schema-16 PQ code offset");
    const u64 expected_code_bytes = checked_multiply(
      count, metadata.navigation_code_bytes,
      "schema-16 PQ code region");
    const u64 code_end = checked_add(
      expected_code_offset, expected_code_bytes,
      "schema-16 PQ code region");
    const u64 aligned_relative_end = checked_align(
      code_end - expected_static_dynamic_base,
      metadata.hot_graph_dynamic_record_bytes,
      "schema-16 dynamic-node base");
    const u64 expected_dynamic_node_base = checked_add(
      expected_static_dynamic_base, aligned_relative_end,
      "schema-16 dynamic-node base");

    require(metadata.hot_graph_offsets[shard] == expected_graph_offset &&
              metadata.hot_graph_dynamic_base_offsets[shard] ==
                expected_static_dynamic_base &&
              metadata.storage_control_remote_offsets[shard] ==
                expected_control_offset &&
              metadata.navigation_code_remote_offsets[shard] ==
                expected_code_offset &&
              metadata.navigation_code_region_bytes[shard] ==
                expected_code_bytes &&
              metadata.dynamic_node_base_offsets[shard] ==
                expected_dynamic_node_base,
            "schema-16 index metadata contains inconsistent shard offsets");
    require(RemotePtr::representable(
              shard, expected_static_dynamic_base, 1) &&
              RemotePtr::representable(shard, expected_dynamic_node_base, 1) &&
              expected_dynamic_node_base <=
                RemotePtr::BYTE_OFFSET_CAPACITY -
                  metadata.hot_graph_dynamic_record_bytes,
            "schema-16 index metadata exceeds tagged RemotePtr capacity");
  }
  require(total_nodes == metadata.num_vectors,
          "schema-16 shard counts do not cover num_vectors");
}

str read_metadata_document(const filepath_t& path) {
  std::error_code size_error;
  const std::uintmax_t file_bytes = std::filesystem::file_size(path, size_error);
  if (size_error) {
    throw std::runtime_error("cannot inspect index metadata: " +
                             size_error.message());
  }
  if (file_bytes == 0 || file_bytes > kMaxMetadataFileBytes) {
    throw std::runtime_error("index metadata file size is outside the 1.." +
                             std::to_string(kMaxMetadataFileBytes) +
                             " byte safety limit");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    throw std::runtime_error("cannot open index metadata");
  }
  str document(static_cast<size_t>(file_bytes), '\0');
  input.read(document.data(), static_cast<std::streamsize>(document.size()));
  if (input.gcount() != static_cast<std::streamsize>(document.size())) {
    throw std::runtime_error("short read from index metadata");
  }
  char extra = 0;
  if (input.get(extra)) {
    throw std::runtime_error("index metadata changed while it was read");
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while reading index metadata");
  }
  return document;
}

}  // namespace

bool load_metadata(const filepath_t& index_prefix, Metadata& metadata, str* error_message) {
  const filepath_t metadata_file = filepath_t(index_prefix.string() + ".meta.json");
  try {
    bool duplicate_root_key = false;
    std::unordered_set<str> root_keys;
    const auto callback = [&duplicate_root_key, &root_keys](
        int depth, nlohmann::json::parse_event_t event,
        nlohmann::json& parsed_value) {
      if (depth == 1 && event == nlohmann::json::parse_event_t::key) {
        duplicate_root_key |=
          !root_keys.insert(parsed_value.get<str>()).second;
      }
      return true;
    };
    const str document = read_metadata_document(metadata_file);
    const nlohmann::json json = nlohmann::json::parse(document, callback);
    require(!duplicate_root_key,
            "index metadata contains a duplicate root key");
    require(json.is_object(), "index metadata root must be an object");

    // Parse into a temporary object. A caller never observes a half-populated
    // Metadata instance after a malformed file is rejected.
    Metadata parsed;
    parsed.schema_version = optional_unsigned<u32>(json, "schema_version", 1);
    parsed.distance = optional_string(json, "distance", "l2");
    parsed.dim = required_unsigned<u32>(json, "dim");
    parsed.R = required_unsigned<u32>(json, "R");
    parsed.beam_width_construction =
      optional_unsigned<u32>(json, "beam_width_construction", 0);
    parsed.partition_max_degree =
      optional_unsigned<u32>(json, "partition_max_degree", parsed.R);
    parsed.partition_cross_shard_ratio =
      optional_number(json, "partition_cross_shard_ratio", 0.0);
    parsed.num_vectors = required_unsigned<u64>(json, "num_vectors");
    parsed.num_memory_nodes =
      required_unsigned<u32>(json, "num_memory_nodes");
    require(parsed.num_memory_nodes != 0 &&
              parsed.num_memory_nodes <= RemotePtr::MEMORY_NODE_MASK + 1,
            "index metadata exceeds the tagged RemotePtr shard limit");
    parsed.node_size = required_unsigned<u32>(json, "node_size");
    parsed.node_layout = optional_string(json, "node_layout", "plain");
    parsed.storage_format = required_string(json, "storage_format");
    parsed.graph_hot_bytes =
      optional_unsigned<u32>(json, "graph_hot_bytes", 0);
    parsed.vector_offset = optional_unsigned<u32>(json, "vector_offset", 0);
    parsed.slot_incarnation_offset =
      optional_unsigned<u32>(json, "slot_incarnation_offset", 0);
    parsed.remote_ptr_format = optional_string(json, "remote_ptr_format");
    parsed.vector_dtype = parse_vector_dtype(
      optional_string(json, "vector_data_type", "float32"));
    parsed.vector_component_size = optional_unsigned<u32>(
      json,
      "vector_component_size",
      static_cast<u32>(vector_dtype_component_size(parsed.vector_dtype)));
    const size_t default_vector_bytes =
      vector_dtype_bytes(parsed.vector_dtype, parsed.dim);
    if (default_vector_bytes > std::numeric_limits<u32>::max()) {
      return fail(error_message, "index metadata vector byte size overflows");
    }
    parsed.vector_bytes = optional_unsigned<u32>(
      json, "vector_bytes", static_cast<u32>(default_vector_bytes));
    parsed.navigation_quantizer =
      optional_string(json, "navigation_quantizer");
    parsed.navigation_code_bytes =
      optional_unsigned<u32>(json, "navigation_code_bytes", 0);
    parsed.pq_subquantizers =
      optional_unsigned<u32>(json, "pq_subquantizers", 0);
    parsed.pq_bits = optional_unsigned<u32>(json, "pq_bits", 0);
    parsed.navigation_model_checksum =
      optional_unsigned<u64>(json, "navigation_model_checksum", 0);
    parsed.hot_graph_entry_size =
      optional_unsigned<u32>(json, "hot_graph_entry_size", 0);
    parsed.hot_graph_pointer_bytes =
      optional_unsigned<u32>(json, "hot_graph_pointer_bytes", 0);
    parsed.hot_graph_shard_bits =
      optional_unsigned<u32>(json, "hot_graph_shard_bits", 0);
    parsed.hot_graph_offsets = optional_u64_array(
      json, "hot_graph_offsets", parsed.num_memory_nodes);
    parsed.hot_graph_entry_counts = optional_u64_array(
      json, "hot_graph_entry_counts", parsed.num_memory_nodes);
    parsed.hot_graph_dynamic_base_offsets = optional_u64_array(
      json, "hot_graph_dynamic_base_offsets", parsed.num_memory_nodes);
    parsed.storage_control_remote_offsets = optional_u64_array(
      json, "storage_control_remote_offsets", parsed.num_memory_nodes);
    parsed.dynamic_node_base_offsets = optional_u64_array(
      json, "dynamic_node_base_offsets", parsed.num_memory_nodes);
    parsed.hot_graph_dynamic_record_bytes = optional_unsigned<u32>(
      json, "hot_graph_dynamic_record_bytes", 0);
    parsed.hot_graph_dynamic_hot_offset = optional_unsigned<u32>(
      json, "hot_graph_dynamic_hot_offset", 0);
    parsed.dynamic_navigation_code_offset = optional_unsigned<u32>(
      json, "dynamic_navigation_code_offset", 0);
    parsed.dynamic_navigation_code_validation_bytes = optional_unsigned<u32>(
      json, "dynamic_navigation_code_validation_bytes", 0);
    parsed.dynamic_navigation_code_checksum_bytes = optional_unsigned<u32>(
      json, "dynamic_navigation_code_checksum_bytes", sizeof(u32));
    parsed.allocation_size =
      optional_unsigned<u32>(json, "allocation_size", parsed.node_size);
    parsed.idmap_format = optional_string(json, "idmap_format");
    parsed.centroid_state_format =
      optional_string(json, "centroid_state_format");
    parsed.index_build_fingerprint =
      optional_unsigned<u64>(json, "index_build_fingerprint", 0);
    parsed.shard_build_fingerprints = optional_u64_array(
      json, "shard_build_fingerprints", parsed.num_memory_nodes);
    parsed.navigation_format = optional_string(json, "navigation_format");
    parsed.navigation_code_remote_offsets = optional_u64_array(
      json, "navigation_code_remote_offsets", parsed.num_memory_nodes);
    parsed.navigation_code_region_bytes = optional_u64_array(
      json, "navigation_code_region_bytes", parsed.num_memory_nodes);

    if (parsed.dim == 0 || parsed.R == 0 ||
        parsed.R > kMaxSupportedGraphDegree ||
        parsed.num_vectors == 0 ||
        parsed.num_vectors > kMaxGpuNavigationNodes) {
      return fail(error_message,
                  "index metadata has an unsupported dimension, graph degree, or vector count");
    }
    if (parsed.partition_max_degree == 0 ||
        !floating_value_is_finite(parsed.partition_cross_shard_ratio) ||
        parsed.partition_cross_shard_ratio < 0.0 ||
        parsed.partition_cross_shard_ratio > 1.0) {
      return fail(error_message,
                  "index metadata partition fields are invalid");
    }
    const u64 expected_vector_bytes = vector_dtype_bytes(
      parsed.vector_dtype, parsed.dim);
    if (parsed.vector_component_size !=
          vector_dtype_component_size(parsed.vector_dtype) ||
        parsed.vector_bytes != expected_vector_bytes ||
        parsed.vector_offset > parsed.node_size ||
        parsed.vector_bytes > parsed.node_size - parsed.vector_offset) {
      return fail(error_message,
                  "index metadata fixed-vector layout is invalid");
    }
    if (parsed.schema_version == gpu_search::format::kMetadataSchemaVersion) {
      validate_schema16(parsed);
    }
    metadata = std::move(parsed);
    if (error_message != nullptr) error_message->clear();
  } catch (const std::exception& e) {
    return fail(error_message, "failed to parse index metadata " + metadata_file.string() + ": " + e.what());
  }

  return true;
}

}  // namespace service::index_metadata
