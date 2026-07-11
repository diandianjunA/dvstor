#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "common/index_path.hh"
#include "nlohmann/json.hh"
#include "tools/legacy_index/migrator.hh"
#include "vamana/anchor_index.hh"
#include "vamana/hot_graph.hh"
#include "vamana/idmap.hh"
#include "vamana/vamana_node.hh"

namespace {

u64 align64(u64 value) {
  return (value + 63) & ~63ull;
}

struct FixtureLayout {
  static constexpr u32 dim = 8;
  static constexpr u32 degree = 2;
  static constexpr u32 shards = 2;
  static constexpr u32 old_node_bytes = 48;
  static constexpr u32 old_code_offset = 24;
  static constexpr u32 graph_entry_bytes = 24;
  static constexpr u32 old_dynamic_record_bytes = 80;
  std::vector<u64> counts{2, 1};
  std::vector<u64> graph_headers;
  std::vector<u64> graphs;
  std::vector<u64> dynamics;

  FixtureLayout() {
    for (u64 count : counts) {
      const u64 header = align64(16 + count * old_node_bytes);
      const u64 graph = align64(header + sizeof(vamana::hot_graph::Header));
      graph_headers.push_back(header);
      graphs.push_back(graph);
      dynamics.push_back(align64(graph + count * graph_entry_bytes));
    }
  }
};

void write_shards(const filepath_t& prefix, const FixtureLayout& layout) {
  const std::vector<std::vector<RemotePtr>> neighbors{
    {RemotePtr{1, 16}},
    {RemotePtr{0, 16}},
    {RemotePtr{0, 16 + FixtureLayout::old_node_bytes}},
  };
  u32 global = 0;
  for (u32 shard = 0; shard < FixtureLayout::shards; ++shard) {
    const filepath_t path = index_path::shard_file(
      prefix, shard + 1, FixtureLayout::shards);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.seekp(static_cast<std::streamoff>(layout.dynamics[shard] - 1));
    output.put(0);
    output.seekp(0);
    output.write(reinterpret_cast<const char*>(&layout.dynamics[shard]), sizeof(u64));
    const u64 medoid = shard == 0 ? RemotePtr{1, 16}.raw_address : 0;
    output.write(reinterpret_cast<const char*>(&medoid), sizeof(medoid));
    std::vector<byte_t> node(FixtureLayout::old_node_bytes);
    for (u64 slot = 0; slot < layout.counts[shard]; ++slot, ++global) {
      std::fill(node.begin(), node.end(), 0);
      *reinterpret_cast<u32*>(node.data() + 8) = global;
      for (u32 dimension = 0; dimension < FixtureLayout::dim; ++dimension) {
        node[16 + dimension] = static_cast<byte_t>(global * 10 + dimension);
      }
      std::fill(node.begin() + FixtureLayout::old_code_offset, node.end(), 0xa5);
      output.seekp(static_cast<std::streamoff>(
        16 + slot * FixtureLayout::old_node_bytes));
      output.write(reinterpret_cast<const char*>(node.data()), node.size());
    }
    vamana::hot_graph::Header header;
    header.version = vamana::hot_graph::kVersion2;
    header.entry_bytes = FixtureLayout::graph_entry_bytes;
    header.max_degree = FixtureLayout::degree;
    header.compact_pointer_shard_bits = 1;
    header.entry_count = layout.counts[shard];
    header.reserved0 = layout.dynamics[shard];
    header.reserved1 = FixtureLayout::old_dynamic_record_bytes;
    header.reserved2 = FixtureLayout::old_node_bytes;
    output.seekp(static_cast<std::streamoff>(layout.graph_headers[shard]));
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    const u32 base = shard == 0 ? 0 : 2;
    std::vector<byte_t> entry(FixtureLayout::graph_entry_bytes);
    for (u64 slot = 0; slot < layout.counts[shard]; ++slot) {
      const auto& links = neighbors[base + slot];
      VamanaNode::encode_hot_graph_entry(
        entry.data(), static_cast<u8>(links.size()), links.data(),
        links.size(), 1, 0, false);
      output.seekp(static_cast<std::streamoff>(
        layout.graphs[shard] + slot * FixtureLayout::graph_entry_bytes));
      output.write(reinterpret_cast<const char*>(entry.data()), entry.size());
    }
    assert(output.good());
  }
}

void write_idmaps(const filepath_t& prefix) {
  const std::vector<std::vector<vamana::idmap::Entry>> entries{
    {{0, RemotePtr{0, 16}.raw_address, 0, 0},
     {2, RemotePtr{1, 16}.raw_address, 0, 0}},
    {{1, RemotePtr{0, 64}.raw_address, 0, 0}},
  };
  for (u32 owner = 0; owner < FixtureLayout::shards; ++owner) {
    const filepath_t path = index_path::owner_idmap_file(
      prefix, owner + 1, FixtureLayout::shards);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    vamana::idmap::Header header;
    header.owner_shard = owner;
    header.shard_count = FixtureLayout::shards;
    header.entry_count = entries[owner].size();
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(entries[owner].data()),
                 entries[owner].size() * sizeof(vamana::idmap::Entry));
    assert(output.good());
  }
}

void write_anchors(const filepath_t& prefix) {
  std::ofstream output(index_path::anchor_file(prefix),
                       std::ios::binary | std::ios::trunc);
  vamana::anchor::Header header;
  header.dim = FixtureLayout::dim;
  header.shard_count = FixtureLayout::shards;
  header.vector_dtype = static_cast<u32>(VectorDType::uint8);
  header.vector_bytes = FixtureLayout::dim;
  header.anchors_per_shard = 1;
  header.total_anchors = 2;
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  std::vector<f32> centroid(FixtureLayout::dim, 1.0f);
  std::vector<byte_t> vector(FixtureLayout::dim, 2);
  for (u32 shard = 0; shard < FixtureLayout::shards; ++shard) {
    vamana::anchor::ShardHeader shard_header{shard, 1};
    output.write(reinterpret_cast<const char*>(&shard_header), sizeof(shard_header));
    output.write(reinterpret_cast<const char*>(centroid.data()),
                 centroid.size() * sizeof(f32));
    vamana::anchor::EntryHeader entry;
    entry.rptr_raw = RemotePtr{shard, 16}.raw_address;
    entry.id = shard == 0 ? 0 : 2;
    output.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
    output.write(reinterpret_cast<const char*>(vector.data()), vector.size());
  }
  assert(output.good());
}

void write_metadata(const filepath_t& prefix, const FixtureLayout& layout) {
  nlohmann::json metadata{
    {"schema_version", 13},
    {"distance", "l2"},
    {"node_layout", "rabitq"},
    {"storage_format", "vamana_compact_v1"},
    {"num_vectors", 3},
    {"dim", FixtureLayout::dim},
    {"R", FixtureLayout::degree},
    {"num_memory_nodes", FixtureLayout::shards},
    {"node_size", FixtureLayout::old_node_bytes},
    {"vector_offset", 16},
    {"vector_bytes", FixtureLayout::dim},
    {"vector_storage_bytes", FixtureLayout::dim},
    {"vector_data_type", "uint8"},
    {"vector_component_size", 1},
    {"neighbors_offset", 16},
    {"graph_hot_bytes", 16},
    {"rabitq_offset", FixtureLayout::old_code_offset},
    {"rabitq_code_bits", 8},
    {"hot_graph_entry_size", FixtureLayout::graph_entry_bytes},
    {"hot_graph_pointer_bytes", 5},
    {"hot_graph_shard_bits", 1},
    {"hot_graph_header_offsets", layout.graph_headers},
    {"hot_graph_offsets", layout.graphs},
    {"hot_graph_entry_counts", layout.counts},
    {"hot_graph_dynamic_base_offsets", layout.dynamics},
    {"hot_graph_dynamic_record_bytes", FixtureLayout::old_dynamic_record_bytes},
    {"hot_graph_dynamic_hot_offset", FixtureLayout::old_node_bytes},
    {"allocation_size", FixtureLayout::old_dynamic_record_bytes},
    {"medoid", {{"memory_node", 1}, {"offset", 16}}},
    {"idmap_format", "owner_sharded_v1"},
    {"anchor_format", "owner_anchor_v1"},
    {"anchor_count_per_shard", 1},
  };
  std::ofstream output(filepath_t(prefix.string() + ".meta.json"));
  output << metadata;
  assert(output.good());
}

}  // namespace

int main() {
  const filepath_t root = std::filesystem::temp_directory_path() /
    "dvstor-legacy-migrator-test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const filepath_t source = root / "legacy";
  const filepath_t output = root / "gpu";
  const FixtureLayout layout;
  VamanaNode::disable_hot_graph();
  VamanaNode::init_static_storage(FixtureLayout::dim, FixtureLayout::degree,
                                  VectorDType::uint8);
  write_shards(source, layout);
  write_idmaps(source);
  write_anchors(source);
  write_metadata(source, layout);

  const auto result = tools::legacy_index::migrate_schema13_index({
    .source_prefix = source,
    .output_prefix = output,
    .io_threads = 2,
    .chunk_nodes = 1,
  });
  assert(result.node_count == 3);

  nlohmann::json metadata;
  std::ifstream metadata_input(filepath_t(output.string() + ".meta.json"));
  metadata_input >> metadata;
  assert(metadata.at("schema_version") == 14);
  assert(metadata.at("node_layout") == "plain");
  assert(metadata.at("node_size") == 32);
  assert(!metadata.contains("neighbors_offset"));
  for (auto iterator = metadata.begin(); iterator != metadata.end(); ++iterator) {
    assert(iterator.key().find("rabitq") == std::string::npos);
  }
  const auto graph_offsets = metadata.at("hot_graph_offsets").get<std::vector<u64>>();
  std::ifstream shard0(index_path::shard_file(output, 1, 2), std::ios::binary);
  std::vector<byte_t> node(32);
  shard0.seekg(16);
  shard0.read(reinterpret_cast<char*>(node.data()), node.size());
  assert(node[16] == 0 && node[23] == 7);
  std::vector<byte_t> graph(FixtureLayout::graph_entry_bytes);
  shard0.seekg(static_cast<std::streamoff>(graph_offsets[0]));
  shard0.read(reinterpret_cast<char*>(graph.data()), graph.size());
  const RemotePtr translated = vamana::hot_graph::decode_remote_ptr(
    graph.data() + vamana::hot_graph::neighbor_offset(0), 1);
  assert(translated == RemotePtr(1, 16));

  std::ifstream idmap(index_path::owner_idmap_file(output, 2, 2), std::ios::binary);
  vamana::idmap::Header idmap_header;
  vamana::idmap::Entry idmap_entry;
  idmap.read(reinterpret_cast<char*>(&idmap_header), sizeof(idmap_header));
  idmap.read(reinterpret_cast<char*>(&idmap_entry), sizeof(idmap_entry));
  assert(RemotePtr{idmap_entry.rptr_raw} == RemotePtr(0, 48));

  std::ifstream anchors(index_path::anchor_file(output), std::ios::binary);
  vamana::anchor::Header anchor_header;
  anchors.read(reinterpret_cast<char*>(&anchor_header), sizeof(anchor_header));
  std::vector<f32> centroid(FixtureLayout::dim);
  std::vector<byte_t> vector(FixtureLayout::dim);
  for (u32 shard = 0; shard < 2; ++shard) {
    vamana::anchor::ShardHeader shard_header;
    vamana::anchor::EntryHeader entry;
    anchors.read(reinterpret_cast<char*>(&shard_header), sizeof(shard_header));
    anchors.read(reinterpret_cast<char*>(centroid.data()), centroid.size() * sizeof(f32));
    anchors.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    anchors.read(reinterpret_cast<char*>(vector.data()), vector.size());
    assert(RemotePtr{entry.rptr_raw} == RemotePtr(shard, 16));
  }
  std::filesystem::remove_all(root);
  return 0;
}
