#include <algorithm>
#include <cassert>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "tools/vamana_offline/gpu_sidecar_converter.hh"
#include "vamana/anchor_index.hh"
#include "vamana/hot_graph.hh"
#include "vamana/idmap.hh"
#include "vamana/rabitq_cache.hh"
#include "vamana/storage_format.hh"
#include "vamana/vamana_node.hh"

namespace {

constexpr u32 kShardCount = 2;
constexpr u32 kDimension = 4;
constexpr u32 kDegree = 3;
constexpr u32 kNodeCount = 4;

struct Placement {
  u32 shard{};
  u64 offset{};
};

u64 align64(u64 value) {
  return (value + 63) & ~u64{63};
}

void size_file(std::fstream& output, u64 bytes) {
  output.seekp(static_cast<std::streamoff>(bytes - 1));
  output.put(0);
}

void write_at(std::fstream& output, u64 offset, const void* data, size_t bytes) {
  output.seekp(static_cast<std::streamoff>(offset));
  output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  assert(output.good());
}

void create_fixture(const filepath_t& prefix,
                    vec<Placement>& placements,
                    vec<vec<u32>>& graph,
                    vec<std::array<byte_t, kDimension>>& vectors,
                    vec<f32>& centroid) {
  VamanaNode::disable_hot_graph();
  VamanaNode::disable_rabitq();
  VamanaNode::set_storage_format(vamana::StorageFormat::compact_v1);
  VamanaNode::init_static_storage(kDimension, kDegree, VectorDType::uint8);
  VamanaNode::enable_rabitq();

  vectors = {{{1, 2, 3, 4}}, {{2, 4, 6, 8}}, {{3, 6, 9, 12}}, {{4, 8, 12, 16}}};
  centroid.assign(kDimension, 0.0f);
  for (const auto& vector : vectors) {
    for (u32 dimension = 0; dimension < kDimension; ++dimension) {
      centroid[dimension] += vector[dimension];
    }
  }
  for (f32& value : centroid) value /= vectors.size();
  VamanaNode::set_rabitq_centroid(centroid);

  const u64 node_bytes = VamanaNode::total_size();
  placements = {
    {1, 16},
    {0, 16},
    {1, 16 + node_bytes},
    {0, 16 + node_bytes},
  };
  graph = {{1, 2, 3}, {0, 2}, {3, 1}, {2, 0}};
  const vec<vec<u32>> shard_ids{{1, 3}, {0, 2}};
  const u64 hot_entry_bytes = VamanaNode::hot_graph_entry_size();
  const u32 hot_shard_bits = vamana::hot_graph::shard_bits_for(kShardCount);
  const u64 fixed_end = 16 + 2 * node_bytes;
  const u64 hot_header_offset = align64(fixed_end);
  const u64 hot_offset = align64(hot_header_offset + sizeof(vamana::hot_graph::Header));
  const u64 dynamic_base = align64(hot_offset + 2 * hot_entry_bytes);

  for (u32 shard = 0; shard < kShardCount; ++shard) {
    const filepath_t shard_path = index_path::shard_file(prefix, shard + 1, kShardCount);
    std::fstream output(shard_path,
                        std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    assert(output.good());
    size_file(output, dynamic_base);
    write_at(output, 0, &dynamic_base, sizeof(dynamic_base));
    const u64 medoid = shard == 0
      ? RemotePtr{placements[0].shard, placements[0].offset}.raw_address
      : 0;
    write_at(output, sizeof(u64), &medoid, sizeof(medoid));

    vamana::hot_graph::Header hot_header;
    hot_header.version = vamana::hot_graph::kVersion2;
    hot_header.entry_bytes = hot_entry_bytes;
    hot_header.max_degree = kDegree;
    hot_header.compact_pointer_shard_bits = hot_shard_bits;
    hot_header.entry_count = 2;
    hot_header.reserved0 = dynamic_base;
    hot_header.reserved1 = VamanaNode::dynamic_record_size();
    hot_header.reserved2 = node_bytes;
    write_at(output, hot_header_offset, &hot_header, sizeof(hot_header));

    for (u32 slot = 0; slot < shard_ids[shard].size(); ++slot) {
      const u32 id = shard_ids[shard][slot];
      vec<byte_t> fixed(node_bytes, 0);
      u64 header = id == 0 ? VamanaNode::HEADER_IS_MEDOID : 0;
      std::memcpy(fixed.data(), &header, sizeof(header));
      std::memcpy(fixed.data() + VamanaNode::offset_id(), &id, sizeof(id));
      std::memcpy(fixed.data() + VamanaNode::offset_vector(),
                  vectors[id].data(), vectors[id].size());
      VamanaNode::RabitqCode code;
      f32 norm = 0.0f;
      f32 error = 0.0f;
      VamanaNode::compute_rabitq_entry(vectors[id].data(), VectorDType::uint8,
                                       code, norm, error);
      std::memcpy(fixed.data() + VamanaNode::offset_rabitq_code(),
                  code.data(), code.size());
      std::memcpy(fixed.data() + VamanaNode::offset_rabitq_norm(), &norm, sizeof(norm));
      std::memcpy(fixed.data() + VamanaNode::offset_rabitq_error(), &error, sizeof(error));
      write_at(output, 16 + slot * node_bytes, fixed.data(), fixed.size());

      vec<RemotePtr> neighbor_pointers;
      for (u32 neighbor : graph[id]) {
        neighbor_pointers.emplace_back(
          placements[neighbor].shard, placements[neighbor].offset);
      }
      vec<byte_t> hot(hot_entry_bytes, 0);
      VamanaNode::encode_hot_graph_entry(
        hot.data(), id, static_cast<u8>(neighbor_pointers.size()),
        neighbor_pointers.data(), neighbor_pointers.size(), hot_shard_bits,
        0, vamana::hot_graph::kVersion2, false);
      write_at(output, hot_offset + slot * hot_entry_bytes, hot.data(), hot.size());
    }
  }

  vec<vec<vamana::idmap::Entry>> owner_entries(kShardCount);
  for (u32 id = 0; id < kNodeCount; ++id) {
    owner_entries[id % kShardCount].push_back({
      id, RemotePtr{placements[id].shard, placements[id].offset}.raw_address, 0, 0});
  }
  for (u32 owner = 0; owner < kShardCount; ++owner) {
    const filepath_t path = index_path::owner_idmap_file(prefix, owner + 1, kShardCount);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    vamana::idmap::Header header;
    header.owner_shard = owner;
    header.shard_count = kShardCount;
    header.entry_count = owner_entries[owner].size();
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(owner_entries[owner].data()),
                 owner_entries[owner].size() * sizeof(vamana::idmap::Entry));
    assert(output.good());
  }

  for (u32 shard = 0; shard < kShardCount; ++shard) {
    const filepath_t path = index_path::rabitq_cache_file(prefix, shard + 1, kShardCount);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    vamana::rabitq::SidecarHeader header;
    header.entry_size = vamana::rabitq::full_entry_bytes();
    header.code_bits = VamanaNode::rabitq_code_bits();
    header.node_size = node_bytes;
    header.raw_vector_bytes = VamanaNode::vector_bytes();
    header.entry_count = shard_ids[shard].size();
    header.cache_budget_bytes = sizeof(header) + header.entry_count * header.entry_size;
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (u32 id : shard_ids[shard]) {
      const auto entry = vamana::rabitq::encode_full(
        vectors[id].data(), VectorDType::uint8);
      output.write(reinterpret_cast<const char*>(entry.data()), entry.size());
    }
    assert(output.good());
  }

  nlohmann::json metadata{
    {"schema_version", 13},
    {"distance", "l2"},
    {"output_prefix", prefix.string()},
    {"num_vectors", kNodeCount},
    {"num_memory_nodes", kShardCount},
    {"dim", kDimension},
    {"R", kDegree},
    {"medoid", {{"memory_node", placements[0].shard},
                {"offset", placements[0].offset}}},
    {"node_layout", "rabitq"},
    {"storage_format", "vamana_compact_v1"},
    {"node_size", node_bytes},
    {"graph_hot_bytes", VamanaNode::graph_hot_bytes()},
    {"vector_offset", VamanaNode::offset_vector()},
    {"vector_bytes", VamanaNode::vector_bytes()},
    {"neighbors_offset", VamanaNode::offset_neighbors()},
    {"rabitq_offset", VamanaNode::offset_rabitq_code()},
    {"rabitq_code_bits", VamanaNode::rabitq_code_bits()},
    {"rabitq_entry_size", VamanaNode::rabitq_entry_size()},
    {"rabitq_centroid", centroid},
    {"vector_data_type", "uint8"},
    {"hot_graph_entry_size", hot_entry_bytes},
    {"hot_graph_pointer_bytes", vamana::hot_graph::kCompactPointerBytes},
    {"hot_graph_shard_bits", hot_shard_bits},
    {"hot_graph_entry_counts", vec<u64>{2, 2}},
    {"hot_graph_header_offsets", vec<u64>{hot_header_offset, hot_header_offset}},
    {"hot_graph_offsets", vec<u64>{hot_offset, hot_offset}},
    {"hot_graph_dynamic_base_offsets", vec<u64>{dynamic_base, dynamic_base}},
    {"idmap_format", "owner_sharded_v1"},
  };
  std::ofstream metadata_output(prefix.string() + ".meta.json", std::ios::trunc);
  metadata_output << metadata;
  assert(metadata_output.good());
}

vec<byte_t> read_code_payload(const filepath_t& path,
                              gpu_search::format::CodeHeader& header) {
  str error;
  assert(gpu_search::format::read_code_header(path, header, &error));
  std::ifstream input(path, std::ios::binary);
  input.seekg(sizeof(header));
  vec<byte_t> payload(header.payload_bytes);
  input.read(reinterpret_cast<char*>(payload.data()), payload.size());
  assert(input.good());
  assert(gpu_search::format::checksum64(payload.data(), payload.size()) ==
         header.payload_checksum);
  return payload;
}

void create_anchor_fixture(
    const filepath_t& prefix, const vec<Placement>& placements,
    const vec<std::array<byte_t, kDimension>>& vectors) {
  std::ofstream output(index_path::anchor_file(prefix),
                       std::ios::binary | std::ios::trunc);
  vamana::anchor::Header header;
  header.dim = kDimension;
  header.shard_count = kShardCount;
  header.vector_dtype = static_cast<u32>(VectorDType::uint8);
  header.vector_bytes = kDimension;
  header.anchors_per_shard = 1;
  header.total_anchors = kShardCount;
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  const std::array<u32, kShardCount> anchor_ids{1, 0};
  const std::array<f32, kDimension> shard_centroid{};
  for (u32 shard = 0; shard < kShardCount; ++shard) {
    const u32 id = anchor_ids[shard];
    const vamana::anchor::ShardHeader shard_header{.shard = shard, .anchor_count = 1};
    const vamana::anchor::EntryHeader entry{
      .rptr_raw = RemotePtr{placements[id].shard, placements[id].offset}.raw_address,
      .id = id,
    };
    output.write(reinterpret_cast<const char*>(&shard_header), sizeof(shard_header));
    output.write(reinterpret_cast<const char*>(shard_centroid.data()),
                 sizeof(shard_centroid));
    output.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
    output.write(reinterpret_cast<const char*>(vectors[id].data()), vectors[id].size());
  }
  assert(output.good());
}

}

int main() {
  const filepath_t directory = std::filesystem::temp_directory_path() /
    "dvstor_gpu_sidecar_converter_test";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const filepath_t prefix = directory / "index";

  vec<Placement> placements;
  vec<vec<u32>> graph;
  vec<std::array<byte_t, kDimension>> vectors;
  vec<f32> centroid;
  create_fixture(prefix, placements, graph, vectors, centroid);

  tools::vamana_offline::GpuSidecarConversionOptions options;
  options.index_prefix = prefix;
  options.entry_points = 3;
  options.threads = 2;
  const auto result = tools::vamana_offline::convert_gpu_sidecars(options);
  assert(result.node_count == kNodeCount);
  assert(result.entry_point_count == 3);
  assert(result.used_rabitq_sidecars);
  assert(result.code_files.size() == kShardCount);

  gpu_search::format::View view;
  str error;
  assert(gpu_search::format::read_file(result.index_file, view, &error));
  assert(view.header.medoid_ordinal == 2);
  assert(view.header.rabitq_code_bits == 8);
  assert(view.header.rabitq_entry_bytes == 16);
  assert(view.header.graph_entry_bytes == VamanaNode::hot_graph_entry_size());
  assert(view.shards[0].ordinal_base == 0 && view.shards[0].node_count == 2);
  assert(view.shards[1].ordinal_base == 2 && view.shards[1].node_count == 2);
  const vec<vec<u32>> shard_ids{{1, 3}, {0, 2}};
  vec<vec<byte_t>> sidecar_payloads;
  for (u32 shard = 0; shard < kShardCount; ++shard) {
    gpu_search::format::CodeHeader header;
    sidecar_payloads.push_back(read_code_payload(result.code_files[shard], header));
    assert(header.memory_node == shard);
    assert(header.entry_count == 2);
    assert(header.remote_offset == view.shards[shard].code_remote_offset);
    for (u32 slot = 0; slot < 2; ++slot) {
      const u32 id = shard_ids[shard][slot];
      VamanaNode::RabitqCode code;
      f32 expected_norm = 0.0f;
      f32 expected_error = 0.0f;
      VamanaNode::compute_rabitq_entry(vectors[id].data(), VectorDType::uint8,
                                       code, expected_norm, expected_error);
      const byte_t* entry = sidecar_payloads.back().data() + slot * header.entry_bytes;
      assert(entry[0] == code[0]);
      for (u32 padding = 1; padding < 4; ++padding) assert(entry[padding] == 0);
      f32 actual_norm = 0.0f;
      f32 actual_error = 0.0f;
      std::memcpy(&actual_norm, entry + gpu_search::format::rabitq_norm_offset(8), sizeof(f32));
      std::memcpy(&actual_error, entry + gpu_search::format::rabitq_error_offset(8), sizeof(f32));
      assert(std::abs(actual_norm - expected_norm) < 1e-6f);
      assert(std::abs(actual_error - expected_error) < 1e-6f);
    }
  }

  u32 ordinal = 0;
  assert(gpu_search::format::remote_to_ordinal(
    view, RemotePtr{placements[0].shard, placements[0].offset}, ordinal));
  assert(ordinal == 2);

  std::ifstream metadata_input(prefix.string() + ".meta.json");
  nlohmann::json metadata;
  metadata_input >> metadata;
  assert(metadata.at("gpu_tiered_format") == "gpu_tiered_v4");
  assert(metadata.at("gpu_tiered_source") == "legacy_sidecar_conversion_v2");
  assert(metadata.at("gpu_graph_source") == "storage_compact_plane");
  assert(metadata.at("gpu_tiered_rabitq_source") == "full_sidecars");

  bool rejected_existing_output = false;
  try {
    (void)tools::vamana_offline::convert_gpu_sidecars(options);
  } catch (const std::exception&) {
    rejected_existing_output = true;
  }
  assert(rejected_existing_output);

  const filepath_t stale_sidecar = index_path::rabitq_cache_file(prefix, 1, kShardCount);
  {
    std::fstream sidecar(stale_sidecar, std::ios::binary | std::ios::in | std::ios::out);
    sidecar.seekg(sizeof(vamana::rabitq::SidecarHeader));
    char value = 0;
    sidecar.read(&value, 1);
    value ^= 1;
    sidecar.seekp(sizeof(vamana::rabitq::SidecarHeader));
    sidecar.write(&value, 1);
    assert(sidecar.good());
  }
  options.overwrite = true;
  options.rabitq_source = tools::vamana_offline::GpuRabitqSource::automatic;
  const auto node_result = tools::vamana_offline::convert_gpu_sidecars(options);
  assert(!node_result.used_rabitq_sidecars);
  for (u32 shard = 0; shard < kShardCount; ++shard) {
    gpu_search::format::CodeHeader header;
    const auto node_payload = read_code_payload(node_result.code_files[shard], header);
    assert(node_payload == sidecar_payloads[shard]);
  }

  create_anchor_fixture(prefix, placements, vectors);
  for (u32 shard = 0; shard < kShardCount; ++shard) {
    std::filesystem::remove(index_path::shard_file(prefix, shard + 1, kShardCount));
    std::filesystem::remove(index_path::gpu_code_file(prefix, shard + 1, kShardCount));
  }
  options.manifest_only = true;
  const auto manifest_result = tools::vamana_offline::convert_gpu_sidecars(options);
  assert(manifest_result.node_count == kNodeCount);
  assert(manifest_result.entry_point_count == 3);
  assert(manifest_result.code_files.empty());
  assert(manifest_result.code_remote_offsets.size() == kShardCount);
  assert(manifest_result.code_bytes.size() == kShardCount);
  assert(std::filesystem::file_size(manifest_result.index_file) < 4096);
  gpu_search::format::View manifest_view;
  assert(gpu_search::format::read_file(manifest_result.index_file, manifest_view, &error));
  assert(manifest_view.entry_points.front() == 2);
  assert(std::find(manifest_view.entry_points.begin(), manifest_view.entry_points.end(), 0) !=
         manifest_view.entry_points.end());
  const u64 expected_graph_header = align64(16 + 2 * VamanaNode::total_size());
  const u64 expected_graph_offset = align64(
    expected_graph_header + sizeof(vamana::hot_graph::Header));
  const u64 expected_dynamic_base = align64(
    expected_graph_offset + 2 * VamanaNode::hot_graph_entry_size());
  for (u32 shard = 0; shard < kShardCount; ++shard) {
    assert(manifest_view.shards[shard].code_remote_offset == expected_dynamic_base);
    assert(manifest_view.shards[shard].code_bytes ==
           2 * VamanaNode::rabitq_entry_size());
    assert(!std::filesystem::exists(
      index_path::gpu_code_file(prefix, shard + 1, kShardCount)));
  }
  {
    std::ifstream manifest_metadata_input(prefix.string() + ".meta.json");
    nlohmann::json manifest_metadata;
    manifest_metadata_input >> manifest_metadata;
    assert(manifest_metadata.at("gpu_tiered_source") == "distributed_manifest_v1");
    assert(manifest_metadata.at("gpu_code_materialization") == "storage_startup");
    assert(manifest_metadata.at("gpu_tiered_rabitq_source") == "authoritative_nodes");
    assert(manifest_metadata.at("gpu_entry_point_source") == "anchors_then_shard_hash");
    assert(manifest_metadata.at("gpu_code_files").empty());
  }
  std::filesystem::remove_all(directory);
  return 0;
}
