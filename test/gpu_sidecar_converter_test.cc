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

vec<u32> read_cold_neighbors(const filepath_t& prefix,
                             const gpu_search::format::View& view,
                             u32 id) {
  const auto& record = view.nodes[id];
  const filepath_t path = index_path::gpu_graph_pages_file(
    prefix, record.shard + 1, view.header.num_shards);
  std::ifstream input(path, std::ios::binary);
  gpu_search::format::ShardPageFileHeader file_header;
  input.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));
  assert(input.good());
  const u64 page_offset = sizeof(file_header) +
    record.cold_page_offset - file_header.remote_offset;
  input.seekg(static_cast<std::streamoff>(page_offset));
  gpu_search::format::PageHeader page_header;
  input.read(reinterpret_cast<char*>(&page_header), sizeof(page_header));
  assert(input.good());
  assert(page_header.magic == 0x47504750);
  assert(page_header.version == 1);
  assert(page_header.node_count > 0);
  const u64 file_offset = page_offset + record.cold_record_offset;
  input.seekg(static_cast<std::streamoff>(file_offset));
  gpu_search::format::PageNodeHeader node_header;
  input.read(reinterpret_cast<char*>(&node_header), sizeof(node_header));
  assert(input.good() && node_header.node_id == id);
  const auto encoding = static_cast<gpu_search::format::IdEncoding>(
    view.header.id_encoding_bytes);
  vec<byte_t> encoded(node_header.degree * view.header.id_encoding_bytes);
  input.read(reinterpret_cast<char*>(encoded.data()), encoded.size());
  assert(input.good());
  vec<u32> neighbors(node_header.degree);
  for (u32 index = 0; index < neighbors.size(); ++index) {
    neighbors[index] = gpu_search::format::decode_id(
      encoded.data() + index * view.header.id_encoding_bytes, encoding);
  }
  return neighbors;
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
  options.hot_degree = 2;
  options.entry_points = 3;
  options.page_bytes = 4096;
  options.threads = 2;
  const auto result = tools::vamana_offline::convert_gpu_sidecars(options);
  assert(result.node_count == kNodeCount);
  assert(result.graph_edge_count == 9);
  assert(result.hot_edge_count == 8);
  assert(result.entry_point_count == 3);
  assert(result.used_rabitq_sidecars);

  gpu_search::format::View view;
  str error;
  assert(gpu_search::format::read_file(result.index_file, view, &error));
  assert(view.header.medoid_id == 0);
  assert(view.header.rabitq_code_bits == 8);
  assert(view.header.rabitq_entry_bytes == 16);
  assert(view.hot_neighbors.size() == kNodeCount * options.hot_degree);
  for (u32 id = 0; id < kNodeCount; ++id) {
    const auto& record = view.nodes[id];
    const RemotePtr expected_pointer{
      placements[id].shard, placements[id].offset};
    assert(record.remote_node == expected_pointer.raw_address);
    assert(record.hot_neighbor_begin == id * options.hot_degree);
    assert(record.hot_neighbor_count == std::min<size_t>(graph[id].size(), 2));
    assert(record.cold_record_offset % alignof(gpu_search::format::PageNodeHeader) == 0);
    for (u32 index = 0; index < record.hot_neighbor_count; ++index) {
      assert(view.hot_neighbors[record.hot_neighbor_begin + index] == graph[id][index]);
    }
    assert(read_cold_neighbors(prefix, view, id) == graph[id]);

    VamanaNode::RabitqCode code;
    f32 expected_norm = 0.0f;
    f32 expected_error = 0.0f;
    VamanaNode::compute_rabitq_entry(vectors[id].data(), VectorDType::uint8,
                                     code, expected_norm, expected_error);
    const byte_t* entry = view.rabitq_entries.data() +
      id * view.header.rabitq_entry_bytes;
    assert(entry[0] == code[0]);
    for (u32 padding = 1; padding < 4; ++padding) assert(entry[padding] == 0);
    f32 actual_norm = 0.0f;
    f32 actual_error = 0.0f;
    std::memcpy(&actual_norm, entry + gpu_search::format::rabitq_norm_offset(8), sizeof(f32));
    std::memcpy(&actual_error, entry + gpu_search::format::rabitq_error_offset(8), sizeof(f32));
    assert(std::abs(actual_norm - expected_norm) < 1e-6f);
    assert(std::abs(actual_error - expected_error) < 1e-6f);
  }

  std::ifstream metadata_input(prefix.string() + ".meta.json");
  nlohmann::json metadata;
  metadata_input >> metadata;
  assert(metadata.at("gpu_tiered_format") == "gpu_tiered_v3");
  assert(metadata.at("gpu_tiered_source") == "legacy_sidecar_conversion_v1");
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
  gpu_search::format::View node_view;
  assert(gpu_search::format::read_file(node_result.index_file, node_view, &error));
  assert(node_view.rabitq_entries == view.rabitq_entries);
  assert(node_view.hot_neighbors == view.hot_neighbors);
  std::filesystem::remove_all(directory);
  return 0;
}
