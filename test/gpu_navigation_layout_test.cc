#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/index_format.hh"
#include "nlohmann/json.hh"

int main() {
  namespace format = gpu_search::format;
  format::View view;
  view.layout.dim = 16;
  view.layout.graph_degree = 3;
  view.layout.vector_dtype = static_cast<u32>(VectorDType::uint8);
  view.layout.pq_subquantizers = 16;
  view.layout.pq_bits = 8;
  view.layout.code_bytes = 16;
  view.layout.model_checksum = 0x12345678ULL;
  view.layout.num_shards = 2;
  view.layout.graph_entry_bytes = 24;
  view.layout.graph_pointer_bytes = 5;
  view.layout.graph_shard_bits = 1;
  view.layout.medoid_ordinal = 2;
  view.layout.num_nodes = 4;
  view.shards = {
    {.ordinal_base = 0, .node_count = 2, .node_base_offset = 16,
     .node_stride = 64, .graph_base_offset = 4096,
     .dynamic_base_offset = 16384, .control_remote_offset = 8192,
     .code_remote_offset = 12288,
     .code_bytes = 32, .memory_node = 0, .dynamic_record_bytes = 112,
     .dynamic_hot_offset = 64, .dynamic_code_offset = 88},
    {.ordinal_base = 2, .node_count = 2, .node_base_offset = 16,
     .node_stride = 64, .graph_base_offset = 4096,
     .dynamic_base_offset = 16384, .control_remote_offset = 8192,
     .code_remote_offset = 12288,
     .code_bytes = 32, .memory_node = 1, .dynamic_record_bytes = 112,
     .dynamic_hot_offset = 64, .dynamic_code_offset = 88},
  };
  view.entry_points = {2, 0, 3};

  std::string error;
  assert(format::validate_view(view, &error));
  format::StorageControlBlock control;
  assert(control.version == format::kStorageControlVersion);
  assert(control.header_bytes == sizeof(format::StorageControlBlock));
  assert(sizeof(format::StorageControlBlock) <= format::kStorageControlBytes);
  for (u64 ack : control.reclaim_ack_sequences) assert(ack == 0);
  format::StorageRoutePublication routes{
    .sequence_begin = 2,
    .shard_id = 1,
    .code_bytes = 16,
    .sequence_end = 2,
  };
  routes.slots[0].remote_node = RemotePtr{1, 16384}.raw_address;
  routes.slots[0].id = 7;
  routes.slots[0].generation = 3;
  routes.slots[0].navigation_code[0] = 42;
  routes.body_checksum = format::storage_route_body_checksum(routes);
  assert(format::validate_storage_route_publication(routes, 1, &error));
  auto torn_routes = routes;
  torn_routes.sequence_end = 4;
  assert(!format::validate_storage_route_publication(torn_routes, 1, &error));
  torn_routes = routes;
  torn_routes.slots[0].navigation_code[0] ^= 1;
  assert(!format::validate_storage_route_publication(torn_routes, 1, &error));
  assert(format::kStorageRoutePublicationOffset + sizeof(routes) <=
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
  code_header.entry_count = 2;
  code_header.remote_offset = 12288;
  code_header.payload_bytes = payload.size();
  code_header.model_checksum = view.layout.model_checksum;
  code_header.payload_checksum = format::checksum64(payload.data(), payload.size());
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

  // Anchor-free navigation bootstraps from ordinary graph entry points.  Keep
  // those points balanced across shards; otherwise a small requested table is
  // exhausted by shard zero before any later shard contributes.
  const auto anchorless_prefix =
    std::filesystem::temp_directory_path() / "dvstor-anchorless-layout-test";
  const auto anchorless_metadata =
    std::filesystem::path{anchorless_prefix.string() + ".meta.json"};
  const std::vector<u64> counts{100, 100, 100, 100, 100};
  const std::vector<u64> dynamic_offsets{131072, 131072, 131072, 131072, 131072};
  const std::vector<u64> control_offsets{131072, 131072, 131072, 131072, 131072};
  const std::vector<u64> code_offsets{
    131072 + format::kStorageControlBytes,
    131072 + format::kStorageControlBytes,
    131072 + format::kStorageControlBytes,
    131072 + format::kStorageControlBytes,
    131072 + format::kStorageControlBytes,
  };
  const std::vector<u64> code_sizes{3200, 3200, 3200, 3200, 3200};
  const std::vector<u64> dynamic_node_offsets{
    code_offsets[0] + code_sizes[0], code_offsets[1] + code_sizes[1],
    code_offsets[2] + code_sizes[2], code_offsets[3] + code_sizes[3],
    code_offsets[4] + code_sizes[4],
  };
  nlohmann::json metadata{
    {"schema_version", format::kMetadataSchemaVersion},
    {"distance", "l2"},
    {"node_layout", "plain"},
    {"storage_format", "vamana_compact_v1"},
    {"navigation_quantizer", "opq_pq"},
    {"navigation_format", "opq_pq_graph_v1"},
    {"num_memory_nodes", 5},
    {"hot_graph_entry_counts", counts},
    {"hot_graph_offsets", std::vector<u64>{65536, 65536, 65536, 65536, 65536}},
    {"hot_graph_dynamic_base_offsets", dynamic_offsets},
    {"storage_control_remote_offsets", control_offsets},
    {"dynamic_node_base_offsets", dynamic_node_offsets},
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
    {"hot_graph_entry_size", 488},
    {"hot_graph_pointer_bytes", 5},
    {"hot_graph_shard_bits", 3},
    {"node_size", 512},
    {"hot_graph_dynamic_record_bytes", 1088},
    {"hot_graph_dynamic_hot_offset", 512},
    {"dynamic_navigation_code_offset", 1024},
    {"num_vectors", 500},
    {"navigation_entry_points", 26},
    {"medoid", {{"memory_node", 0}, {"offset", format::kNodeBaseOffset}}},
  };
  {
    std::ofstream output(anchorless_metadata, std::ios::trunc);
    output << metadata;
  }
  format::View anchorless;
  bool used_anchor_entry_points = true;
  if (!format::synthesize_distributed_view(
        anchorless_prefix, anchorless, {}, &used_anchor_entry_points, &error)) {
    throw std::runtime_error(error);
  }
  assert(!used_anchor_entry_points);
  assert(anchorless.entry_points.size() == 26);
  std::vector<u32> entries_per_shard(5);
  for (u32 entry : anchorless.entry_points) {
    assert(entry < anchorless.layout.num_nodes);
    ++entries_per_shard[entry / 100];
  }
  const auto [minimum_entries, maximum_entries] = std::minmax_element(
    entries_per_shard.begin(), entries_per_shard.end());
  assert(*minimum_entries > 0);
  assert(*maximum_entries - *minimum_entries <= 1);

  std::filesystem::remove(code_path);
  std::filesystem::remove(anchorless_metadata);
  return 0;
}
