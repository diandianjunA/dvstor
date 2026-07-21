#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/constants.hh"
#include "gpu_search/graph_record_validation.hh"
#include "gpu_search/index_format.hh"
#include "nlohmann/json.hh"
#include "vamana/hot_graph.hh"

namespace {

void test_supported_gpu_layout_limits() {
  namespace format = gpu_search::format;
  format::NavigationLayout layout{
    .dim = 128,
    .graph_degree = kMaxSupportedGraphDegree,
    .vector_dtype = static_cast<u32>(VectorDType::float32),
    .pq_subquantizers = 16,
    .pq_bits = 8,
    .code_bytes = 16,
    .num_shards = 64,
    .graph_entry_bytes = 1088,
    .graph_pointer_bytes = format::kCompactPointerBytes,
    .graph_shard_bits = RemotePtr::MEMORY_NODE_BITS,
    .num_nodes = 64,
    .base_generation = 1,
    .model_checksum = 1,
  };
  std::string error;
  assert(format::validate_layout(layout, &error));
  format::View view;
  view.layout = layout;
  view.shards.reserve(layout.num_shards);
  for (u32 shard = 0; shard < layout.num_shards; ++shard) {
    view.shards.push_back(format::ShardRegion{
      .ordinal_base = shard,
      .node_count = 1,
      .node_base_offset = format::kNodeBaseOffset,
      .node_stride = 576,
      .graph_base_offset = 4096,
      .dynamic_base_offset = 16384,
      .control_remote_offset = 8192,
      .code_remote_offset = 12288,
      .code_bytes = 16,
      .memory_node = shard,
      .dynamic_record_bytes = 1728,
      .dynamic_hot_offset = 576,
      .dynamic_code_offset = 1664,
    });
  }
  assert(format::validate_view(view, &error));

  layout.num_shards = 65;
  layout.graph_shard_bits = RemotePtr::MEMORY_NODE_BITS + 1;
  assert(!format::validate_layout(layout, &error));

  layout.num_shards = 64;
  layout.graph_shard_bits = RemotePtr::MEMORY_NODE_BITS;
  layout.graph_degree = kMaxSupportedGraphDegree + 1;
  assert(!format::validate_layout(layout, &error));
}

void test_tagged_remote_pointer() {
  const RemotePtr static_node{63, RemotePtr::BYTE_OFFSET_CAPACITY - 16};
  const RemotePtr first_dynamic{
    63, RemotePtr::BYTE_OFFSET_CAPACITY - 16, 1};
  const RemotePtr last_dynamic{
    63, RemotePtr::BYTE_OFFSET_CAPACITY - 16,
    RemotePtr::MAX_INCARNATION};
  assert(static_node.memory_node() == 63);
  assert(static_node.byte_offset() == RemotePtr::BYTE_OFFSET_CAPACITY - 16);
  assert(static_node.incarnation() == 0);
  assert(first_dynamic.physical_address_raw() ==
         static_node.physical_address_raw());
  assert(first_dynamic != static_node);
  assert(last_dynamic.incarnation() == RemotePtr::MAX_INCARNATION);
  assert(RemotePtr{last_dynamic.raw_address} == last_dynamic);
  std::array<byte_t, vamana::hot_graph::kTaggedPointerBytes> encoded{};
  assert(vamana::hot_graph::encode_remote_ptr(
    last_dynamic, 0, encoded.data()));
  assert(vamana::hot_graph::decode_remote_ptr(encoded.data(), 0) ==
         last_dynamic);

  bool rejected = false;
  try {
    (void)RemotePtr{0, 17};
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  assert(rejected);
  rejected = false;
  try {
    (void)RemotePtr{0, 16, RemotePtr::MAX_INCARNATION + 1};
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  assert(rejected);
}

void test_graph_record_stale_incarnation_is_not_transport_failure() {
  namespace validation = gpu_search::graph_record_validation;
  constexpr u32 graph_degree = 3;
  constexpr u32 graph_capacity = 5;
  constexpr u32 record_bytes = 56;
  std::array<byte_t, record_bytes> record{};
  record[0] = 1;
  record[1] = 0;
  const auto store_u32 = [&](size_t offset, u32 value) {
    record[offset + 0] = static_cast<byte_t>(value);
    record[offset + 1] = static_cast<byte_t>(value >> 8);
    record[offset + 2] = static_cast<byte_t>(value >> 16);
    record[offset + 3] = static_cast<byte_t>(value >> 24);
  };
  store_u32(8, 12);
  store_u32(12, 0);
  const auto seal = [&]() {
    const u16 checksum = validation::checksum16(
      record.data(), static_cast<u32>(record.size()));
    record[2] = static_cast<byte_t>(checksum);
    record[3] = static_cast<byte_t>(checksum >> 8);
  };
  seal();

  const auto current = validation::classify_snapshot(
    record.data(), record.size(), graph_degree, graph_capacity, 12);
  assert(current == validation::SnapshotState::valid);
  assert(validation::decide_read_action(true, current, false) ==
         validation::ReadAction::accept);

  // An older dynamic handle can outlive durable cleanup in a read-committed
  // query. A complete record for the replacement incarnation is stale for that
  // handle, not evidence of transport corruption.
  const auto stale = validation::classify_snapshot(
    record.data(), record.size(), graph_degree, graph_capacity, 11);
  assert(stale == validation::SnapshotState::stale_incarnation);
  assert(validation::decide_read_action(true, stale, false) ==
         validation::ReadAction::discard_stale);

  // Static slots never recycle, and a torn record must not be mistaken for a
  // benign stale dynamic handle merely because its incarnation bytes differ.
  assert(validation::classify_snapshot(
           record.data(), record.size(), graph_degree, graph_capacity, 0) ==
         validation::SnapshotState::invalid);
  record[16] ^= 1;
  assert(validation::classify_snapshot(
           record.data(), record.size(), graph_degree, graph_capacity, 11) ==
         validation::SnapshotState::invalid);
  assert(validation::decide_read_action(
           true, validation::SnapshotState::invalid, true) ==
         validation::ReadAction::retry);
  assert(validation::decide_read_action(
           true, validation::SnapshotState::invalid, false) ==
         validation::ReadAction::fail);
  assert(validation::decide_read_action(
           false, validation::SnapshotState::valid, true) ==
         validation::ReadAction::fail);
}

void test_centroid_route_publication(
    gpu_search::format::CentroidScalarType scalar_type, u32 dim) {
  namespace format = gpu_search::format;
  constexpr u32 shard = 1;
  constexpr u32 shard_count = 3;
  constexpr u32 entry_capacity = format::kStorageCentroidRouteMaxLiveEntries;
  const u64 publication_bytes =
    format::storage_centroid_route_publication_bytes(
      dim, scalar_type, entry_capacity);
  assert(publication_bytes != 0);
  assert(publication_bytes % 64 == 0);

  void* allocation = std::aligned_alloc(
    64, static_cast<size_t>(publication_bytes));
  assert(allocation != nullptr);
  span<byte_t> publication{
    static_cast<byte_t*>(allocation), static_cast<size_t>(publication_bytes)};

  format::StorageCentroidRouteDescriptor descriptor{
    .remote_offset = 64 * 1024,
    .publication_bytes = publication_bytes,
    .layout_version = 9,
    .dim = dim,
    .centroid_scalar_type = static_cast<u32>(scalar_type),
    .shard_count = shard_count,
    .live_entry_capacity = entry_capacity,
  };
  std::string error;
  assert(format::validate_storage_centroid_route_descriptor(
    descriptor, dim, shard_count, &error));

  std::vector<f32> centroid32;
  std::vector<f64> centroid64;
  const void* centroid_data = nullptr;
  if (scalar_type == format::CentroidScalarType::float32) {
    centroid32.resize(dim);
    for (u32 index = 0; index < dim; ++index) {
      centroid32[index] = static_cast<f32>(index) * 0.125f - 7.0f;
    }
    centroid_data = centroid32.data();
  } else {
    centroid64.resize(dim);
    for (u32 index = 0; index < dim; ++index) {
      centroid64[index] = static_cast<f64>(index) * 0.0625 - 11.0;
    }
    centroid_data = centroid64.data();
  }
  const std::array<format::StorageCentroidRouteEntry, entry_capacity>
    entry_storage{{
    {.remote_node = RemotePtr{shard, 64}.raw_address,
     .generation = 7,
     .flags = format::kStorageCentroidRouteLive},
    {.remote_node = RemotePtr{shard, 128}.raw_address,
     .generation = 8,
     .flags = format::kStorageCentroidRouteLive},
    // Poison immediately follows the logical span. A capacity-sized memcpy
    // would copy these records and is caught even without an ASan build.
    {.remote_node = ~u64{0}, .generation = ~u32{0}, .flags = ~u32{0}},
    {.remote_node = ~u64{0}, .generation = ~u32{0}, .flags = ~u32{0}},
  }};
  const span<const format::StorageCentroidRouteEntry> entries{
    entry_storage.data(), 2};
  assert(format::prepare_storage_centroid_route_publication(
    publication, shard, dim, scalar_type, entry_capacity,
    17, 1234, centroid_data, entries, &error));
  if (!format::validate_storage_centroid_route_publication(
        publication, descriptor, shard, &error)) {
    throw std::runtime_error(error);
  }

  auto* header = reinterpret_cast<
    format::StorageCentroidRoutePublicationHeader*>(publication.data());
  assert(header->sequence == 2);
  assert(header->total_bytes == publication_bytes);
  assert(header->shard_version == 17);
  assert(header->vector_count == 1234);
  assert(header->live_entry_count == entries.size());
  const auto decoded_entries = format::storage_centroid_route_entries(
    span<const byte_t>{publication.data(), publication.size()});
  assert(decoded_entries.size() == entries.size());
  assert(std::memcmp(decoded_entries.data(), entries.data(),
                     entries.size() * sizeof(entries[0])) == 0);
  const auto* capacity_entries = reinterpret_cast<const
    format::StorageCentroidRouteEntry*>(
      publication.data() + header->entries_offset);
  for (size_t index = entries.size(); index < entry_capacity; ++index) {
    assert(capacity_entries[index].remote_node == 0);
    assert(capacity_entries[index].generation == 0);
    assert(capacity_entries[index].flags == 0);
  }
  const void* decoded_centroid = format::storage_centroid_route_centroid_data(
    span<const byte_t>{publication.data(), publication.size()});
  const size_t centroid_bytes = static_cast<size_t>(dim) *
    format::centroid_scalar_bytes(scalar_type);
  assert(decoded_centroid != nullptr);
  assert(std::memcmp(decoded_centroid, centroid_data, centroid_bytes) == 0);

  // An odd sequence is a publication in progress and must never be consumed.
  header->sequence = 3;
  assert(!format::validate_storage_centroid_route_publication(
    publication, descriptor, shard, &error));
  header->sequence = 2;

  // The seqlock does not replace body integrity: a stable-looking torn body is
  // rejected by its checksum.
  publication[header->centroid_offset + centroid_bytes / 2] ^= 1;
  assert(!format::validate_storage_centroid_route_publication(
    publication, descriptor, shard, &error));
  publication[header->centroid_offset + centroid_bytes / 2] ^= 1;
  assert(format::validate_storage_centroid_route_publication(
    publication, descriptor, shard, &error));

  auto malformed_descriptor = descriptor;
  malformed_descriptor.publication_bytes -= 64;
  assert(!format::validate_storage_centroid_route_descriptor(
    malformed_descriptor, dim, shard_count, &error));
  std::free(allocation);
}

}  // namespace

int main() {
  namespace format = gpu_search::format;
  test_supported_gpu_layout_limits();
  test_tagged_remote_pointer();
  test_graph_record_stale_incarnation_is_not_transport_failure();
  format::View view;
  view.layout.dim = 16;
  view.layout.graph_degree = 3;
  view.layout.vector_dtype = static_cast<u32>(VectorDType::uint8);
  view.layout.pq_subquantizers = 16;
  view.layout.pq_bits = 8;
  view.layout.code_bytes = 16;
  view.layout.model_checksum = 0x12345678ULL;
  view.layout.num_shards = 2;
  view.layout.graph_entry_bytes = 40;
  view.layout.graph_pointer_bytes = 8;
  view.layout.graph_shard_bits = 1;
  view.layout.num_nodes = 4;
  view.shards = {
    {.ordinal_base = 0, .node_count = 2, .node_base_offset = 16,
     .node_stride = 64, .graph_base_offset = 4096,
     .dynamic_base_offset = 16384, .control_remote_offset = 8192,
     .code_remote_offset = 12288,
     .code_bytes = 32, .memory_node = 0, .dynamic_record_bytes = 128,
     .dynamic_hot_offset = 64, .dynamic_code_offset = 104},
    {.ordinal_base = 2, .node_count = 2, .node_base_offset = 16,
     .node_stride = 64, .graph_base_offset = 4096,
     .dynamic_base_offset = 16384, .control_remote_offset = 8192,
     .code_remote_offset = 12288,
     .code_bytes = 32, .memory_node = 1, .dynamic_record_bytes = 128,
     .dynamic_hot_offset = 64, .dynamic_code_offset = 104},
  };
  std::string error;
  assert(format::validate_view(view, &error));
  static_assert(format::kStorageControlVersion == 4);
  format::StorageControlBlock control;
  assert(control.version == format::kStorageControlVersion);
  assert(control.header_bytes == sizeof(format::StorageControlBlock));
  static_assert(sizeof(format::StorageControlBlock) == 192);
  static_assert(offsetof(format::StorageControlBlock, centroid_route) == 128);
  assert(sizeof(format::StorageControlBlock) <= format::kStorageControlBytes);
  assert(control.reserved0 == 0);
  assert(control.reserved1 == 0);
  test_centroid_route_publication(
    format::CentroidScalarType::float32, 257);
  test_centroid_route_publication(
    format::CentroidScalarType::float64, 1024);
  assert(format::storage_centroid_route_publication_bytes(
           1024, format::CentroidScalarType::float64,
           format::kStorageCentroidRouteMaxLiveEntries) >
         format::kStorageControlBytes);

  RemotePtr pointer;
  assert(format::ordinal_to_remote(view, 0, pointer));
  assert(pointer == RemotePtr(0, 16));
  assert(format::ordinal_to_remote(view, 3, pointer));
  assert(pointer == RemotePtr(1, 80));
  u32 ordinal = 0;
  assert(format::remote_to_ordinal(view, RemotePtr(1, 16), ordinal));
  assert(ordinal == 2);
  assert(!format::remote_to_ordinal(view, RemotePtr(1, 16384), ordinal));

  format::View malformed = view;
  malformed.shards[1].ordinal_base = 3;
  assert(!format::validate_view(malformed, &error));
  malformed = view;
  malformed.shards[0].code_remote_offset += 64;
  assert(!format::validate_view(malformed, &error));

  const auto code_path =
    std::filesystem::temp_directory_path() / "dvstor-pq16-codes.bin";
  std::vector<byte_t> payload(32, 0x5a);
  format::CodeHeader code_header;
  code_header.memory_node = 0;
  code_header.code_bytes = 16;
  code_header.node_size = 64;
  code_header.vector_dtype = static_cast<u32>(VectorDType::uint8);
  code_header.entry_count = 2;
  code_header.remote_offset = 12288;
  code_header.payload_bytes = payload.size();
  code_header.model_checksum = view.layout.model_checksum;
  code_header.payload_checksum = format::checksum64(payload.data(), payload.size());
  code_header.build_fingerprint = 0x123456789abcdef0ULL;
  code_header.shard_fingerprint = 0x0fedcba987654321ULL;
  {
    std::ofstream output(code_path, std::ios::binary | std::ios::trunc);
    format::CodeHeader placeholder;
    output.write(reinterpret_cast<const char*>(&placeholder), sizeof(placeholder));
    output.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    assert(format::write_code_header(output, code_header, &error));
  }
  format::CodeHeader loaded_code;
  assert(format::read_code_header(code_path, loaded_code, &error));
  assert(loaded_code.payload_checksum == code_header.payload_checksum);
  assert(loaded_code.vector_dtype == static_cast<u32>(VectorDType::uint8));
  assert(loaded_code.build_fingerprint == code_header.build_fingerprint);
  assert(loaded_code.shard_fingerprint == code_header.shard_fingerprint);

  auto invalid_code_header = loaded_code;
  invalid_code_header.vector_dtype = static_cast<u32>(VectorDType::int8) + 1;
  assert(!format::validate_code_header(invalid_code_header, &error));
  invalid_code_header = loaded_code;
  invalid_code_header.build_fingerprint = 0;
  assert(!format::validate_code_header(invalid_code_header, &error));
  invalid_code_header = loaded_code;
  invalid_code_header.shard_fingerprint = 0;
  assert(!format::validate_code_header(invalid_code_header, &error));
  invalid_code_header = loaded_code;
  invalid_code_header.reserved[0] = 1;
  assert(!format::validate_code_header(invalid_code_header, &error));

  // Header identity is checksum-covered. A sidecar renamed from another build
  // cannot have its build fingerprint edited to look local.
  {
    std::fstream codes(code_path, std::ios::binary | std::ios::in | std::ios::out);
    codes.seekg(static_cast<std::streamoff>(
      offsetof(format::CodeHeader, build_fingerprint)));
    char byte = 0;
    codes.read(&byte, 1);
    byte ^= 1;
    codes.seekp(static_cast<std::streamoff>(
      offsetof(format::CodeHeader, build_fingerprint)));
    codes.write(&byte, 1);
    assert(codes.good());
  }
  format::CodeHeader tampered_code;
  assert(!format::read_code_header(code_path, tampered_code, &error));
  {
    std::fstream codes(code_path, std::ios::binary | std::ios::in | std::ios::out);
    codes.seekg(static_cast<std::streamoff>(
      offsetof(format::CodeHeader, build_fingerprint)));
    char byte = 0;
    codes.read(&byte, 1);
    byte ^= 1;
    codes.seekp(static_cast<std::streamoff>(
      offsetof(format::CodeHeader, build_fingerprint)));
    codes.write(&byte, 1);
    assert(codes.good());
  }
  assert(format::read_code_header(code_path, tampered_code, &error));

  {
    std::fstream codes(code_path, std::ios::binary | std::ios::in | std::ios::out);
    codes.seekg(static_cast<std::streamoff>(sizeof(format::CodeHeader)));
    char byte = 0;
    codes.read(&byte, 1);
    byte ^= 1;
    codes.seekp(static_cast<std::streamoff>(sizeof(format::CodeHeader)));
    codes.write(&byte, 1);
    assert(codes.good());
  }
  format::CodeHeader unchanged_header;
  assert(format::read_code_header(code_path, unchanged_header, &error));
  assert(unchanged_header.payload_checksum == code_header.payload_checksum);

  // Runtime layout synthesis is deliberately independent of an offline
  // medoid or a sampled static entry table. Query seeds arrive only through
  // validated storage-canonical centroid publications.
  const auto metadata_prefix =
    std::filesystem::temp_directory_path() / "dvstor-layout-metadata-test";
  const auto metadata_path =
    std::filesystem::path{metadata_prefix.string() + ".meta.json"};
  const std::vector<u64> counts{100, 100, 100, 100, 100};
  const std::vector<u64> dynamic_offsets{196608, 196608, 196608, 196608, 196608};
  const std::vector<u64> control_offsets{196608, 196608, 196608, 196608, 196608};
  const std::vector<u64> code_offsets{
    196608 + format::kStorageControlBytes,
    196608 + format::kStorageControlBytes,
    196608 + format::kStorageControlBytes,
    196608 + format::kStorageControlBytes,
    196608 + format::kStorageControlBytes,
  };
  const std::vector<u64> code_sizes{3200, 3200, 3200, 3200, 3200};
  const std::vector<u64> dynamic_node_offsets{
    204960, 204960, 204960, 204960, 204960,
  };
  nlohmann::json metadata{
    {"schema_version", format::kMetadataSchemaVersion},
    {"distance", "l2"},
    {"node_layout", "plain"},
    {"storage_format", "vamana_tagged_v2"},
    {"remote_ptr_format", "tagged_inc24_shard6_off34x16_v1"},
    {"centroid_state_format", "physical_shard_centroid_v2_bound"},
    {"index_build_fingerprint", 0x123456789abcdef0ull},
    {"shard_build_fingerprints",
     std::vector<u64>{11, 12, 13, 14, 15}},
    {"slot_incarnation_offset", 16},
    {"navigation_quantizer", "opq_pq"},
    {"navigation_format", "opq_pq_graph_v1"},
    {"num_memory_nodes", 5},
    {"hot_graph_entry_counts", counts},
    {"hot_graph_offsets", std::vector<u64>{65536, 65536, 65536, 65536, 65536}},
    {"hot_graph_dynamic_base_offsets", dynamic_offsets},
    {"storage_control_remote_offsets", control_offsets},
    {"dynamic_node_base_offsets", dynamic_node_offsets},
    {"dynamic_navigation_code_validation_bytes", 4},
    {"navigation_code_remote_offsets", code_offsets},
    {"navigation_code_region_bytes", code_sizes},
    {"dim", 128},
    {"R", 96},
    {"vector_data_type", "uint8"},
    {"vector_bytes", 128},
    {"pq_subquantizers", 32},
    {"pq_bits", 8},
    {"navigation_code_bytes", 32},
    {"navigation_model_checksum", 1},
    {"hot_graph_entry_size", 832},
    {"hot_graph_pointer_bytes", 8},
    {"hot_graph_shard_bits", 3},
    {"node_size", 512},
    {"hot_graph_dynamic_record_bytes", 1392},
    {"hot_graph_dynamic_hot_offset", 512},
    {"dynamic_navigation_code_offset", 1344},
    {"num_vectors", 500},
  };
  assert(!metadata.contains("medoid"));
  assert(!metadata.contains("navigation_entry_points"));
  {
    std::ofstream output(metadata_path, std::ios::trunc);
    output << metadata;
  }
  format::View synthesized;
  if (!format::synthesize_distributed_view(
        metadata_prefix, synthesized, &error)) {
    throw std::runtime_error(error);
  }
  assert(synthesized.layout.num_nodes == 500);
  assert(synthesized.layout.num_shards == 5);
  assert(synthesized.shards.size() == 5);
  assert(format::validate_view(synthesized, &error));

  metadata["R"] = kMaxSupportedGraphDegree + 1;
  {
    std::ofstream output(metadata_path, std::ios::trunc);
    output << metadata;
  }
  assert(!format::synthesize_distributed_view(
    metadata_prefix, synthesized, &error));

  std::filesystem::remove(code_path);
  std::filesystem::remove(metadata_path);
  return 0;
}
