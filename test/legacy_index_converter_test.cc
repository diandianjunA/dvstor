#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "gpu_search/pq_index.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "tools/vamana_offline/legacy_index_converter.hh"
#include "vamana/centroid_state.hh"
#include "vamana/hot_graph.hh"
#include "vamana/idmap.hh"
#include "vamana/vamana_node.hh"

namespace {

namespace fs = std::filesystem;
namespace legacy_converter = tools::vamana_offline;

constexpr u32 kShards = 2;
constexpr u32 kDim = 4;
constexpr u32 kDegree = 2;
constexpr u64 kNodeBase = 16;
constexpr u32 kLegacyNodeBytes = 32;
constexpr u32 kLegacyVectorOffset = 16;
constexpr u32 kLegacyGraphEntryBytes = 24;
constexpr u64 kLegacyGraphHeaderOffset = 128;
constexpr u64 kLegacyGraphOffset = 192;
constexpr u64 kLegacyDynamicBase = 256;
constexpr u32 kLegacyShardBits = 1;
constexpr u32 kLegacyIdmapMagic = 0x504d4444;  // DDMP

#pragma pack(push, 1)
struct LegacyIdmapHeader {
  u32 magic{kLegacyIdmapMagic};
  u32 version{1};
  u32 owner_shard{};
  u32 shard_count{kShards};
  u64 entry_count{};
};

struct LegacyIdmapEntry {
  node_t id{};
  u64 remote_node{};
  u32 generation{};
  u32 flags{};
};
#pragma pack(pop)

static_assert(sizeof(LegacyIdmapHeader) == 24);
static_assert(sizeof(LegacyIdmapEntry) == 20);

struct FixtureNode {
  node_t id{};
  u32 shard{};
  u32 slot{};
  std::array<u8, kDim> vector{};
  std::array<node_t, kDegree> neighbors{};
};

const std::array<FixtureNode, 4> kNodes{{
  {.id = 0, .shard = 0, .slot = 0,
   .vector = {0, 2, 4, 6}, .neighbors = {1, 2}},
  {.id = 2, .shard = 0, .slot = 1,
   .vector = {4, 6, 8, 10}, .neighbors = {0, 3}},
  {.id = 1, .shard = 1, .slot = 0,
   .vector = {10, 12, 14, 16}, .neighbors = {0, 3}},
  {.id = 3, .shard = 1, .slot = 1,
   .vector = {14, 16, 18, 20}, .neighbors = {2, 1}},
}};

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    const auto stamp = std::chrono::steady_clock::now()
                         .time_since_epoch().count();
    path_ = fs::temp_directory_path() /
      ("dvstor_legacy_converter_test_" + std::to_string(stamp));
    if (!fs::create_directories(path_)) {
      throw std::runtime_error("failed to create converter test directory");
    }
  }

  ~TemporaryDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  const fs::path& path() const { return path_; }

private:
  fs::path path_;
};

template <typename T>
void store(byte_t* destination, T value) {
  std::memcpy(destination, &value, sizeof(value));
}

template <typename T>
T load(const byte_t* source) {
  T value{};
  std::memcpy(&value, source, sizeof(value));
  return value;
}

u64 legacy_raw_pointer(u32 shard, u64 byte_offset) {
  return (static_cast<u64>(shard) << 48) | byte_offset;
}

u64 legacy_node_offset(const FixtureNode& node) {
  return kNodeBase + static_cast<u64>(node.slot) * kLegacyNodeBytes;
}

const FixtureNode& fixture_node(node_t id) {
  const auto iterator = std::find_if(
    kNodes.begin(), kNodes.end(),
    [id](const FixtureNode& node) { return node.id == id; });
  assert(iterator != kNodes.end());
  return *iterator;
}

void encode_legacy_compact_pointer(u64 raw_pointer, byte_t* output) {
  const u32 shard = static_cast<u32>(raw_pointer >> 48);
  const u64 offset = (raw_pointer << 16) >> 16;
  assert(offset % 8 == 0);
  constexpr u32 offset_bits = 40 - kLegacyShardBits;
  const u64 packed =
    (static_cast<u64>(shard) << offset_bits) | (offset / 8);
  for (u32 index = 0; index < 5; ++index) {
    output[index] = static_cast<byte_t>(packed >> (8 * index));
  }
}

void write_legacy_idmaps(const fs::path& prefix) {
  for (u32 owner = 0; owner < kShards; ++owner) {
    std::vector<LegacyIdmapEntry> entries;
    for (const FixtureNode& node : kNodes) {
      if (node.id % kShards != owner) continue;
      entries.push_back({
        .id = node.id,
        .remote_node = legacy_raw_pointer(
          node.shard, legacy_node_offset(node)),
      });
    }
    const LegacyIdmapHeader header{
      .owner_shard = owner,
      .entry_count = entries.size(),
    };
    const fs::path path = index_path::owner_idmap_file(
      prefix, owner + 1, kShards);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(entries.data()),
                 static_cast<std::streamsize>(
                   entries.size() * sizeof(entries.front())));
    assert(output.good());
  }
}

void write_legacy_shard(const fs::path& prefix, u32 shard) {
  std::vector<byte_t> bytes(kLegacyDynamicBase, byte_t{});
  store<u64>(bytes.data(), kLegacyDynamicBase);
  if (shard == 0) {
    store<u64>(bytes.data() + sizeof(u64),
               legacy_raw_pointer(0, kNodeBase));
  }

  vamana::hot_graph::Header graph_header;
  graph_header.version = vamana::hot_graph::kVersion2;
  graph_header.entry_bytes = kLegacyGraphEntryBytes;
  graph_header.max_degree = kDegree;
  graph_header.compact_pointer_bytes = 5;
  graph_header.compact_pointer_shard_bits = kLegacyShardBits;
  graph_header.entry_count = 2;
  graph_header.node_base_offset = kNodeBase;
  graph_header.reserved0 = kLegacyDynamicBase;
  graph_header.reserved1 = 64;
  graph_header.reserved2 = kLegacyNodeBytes;
  std::memcpy(bytes.data() + kLegacyGraphHeaderOffset,
              &graph_header, sizeof(graph_header));

  for (const FixtureNode& node : kNodes) {
    if (node.shard != shard) continue;
    byte_t* fixed = bytes.data() + legacy_node_offset(node);
    store<u64>(fixed, 0);
    store<node_t>(fixed + 8, node.id);
    store<u32>(fixed + 12, 0);
    std::memcpy(fixed + kLegacyVectorOffset,
                node.vector.data(), node.vector.size());

    byte_t* graph = bytes.data() + kLegacyGraphOffset +
      static_cast<u64>(node.slot) * kLegacyGraphEntryBytes;
    graph[0] = kDegree;
    graph[1] = 0;
    store<u32>(graph + 4, 0);
    for (u32 edge = 0; edge < kDegree; ++edge) {
      const FixtureNode& neighbor = fixture_node(node.neighbors[edge]);
      encode_legacy_compact_pointer(
        legacy_raw_pointer(neighbor.shard, legacy_node_offset(neighbor)),
        graph + 8 + edge * 5);
    }
    const u16 checksum = vamana::hot_graph::checksum16(
      graph, kLegacyGraphEntryBytes);
    store<u16>(graph + 2, checksum);
  }

  const fs::path path = index_path::shard_file(
    prefix, shard + 1, kShards);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  assert(output.good());
}

void write_legacy_metadata(const fs::path& prefix) {
  const nlohmann::json metadata{
    {"data_file", "legacy-fixture.u8bin"},
    {"output_prefix", prefix.string()},
    {"distance", "l2"},
    {"num_vectors", kNodes.size()},
    {"dim", kDim},
    {"R", kDegree},
    {"beam_width", 8},
    {"beam_width_construction", 8},
    {"alpha", 1.2},
    {"num_memory_nodes", kShards},
    {"medoid", {{"memory_node", 0}, {"offset", kNodeBase}}},
    {"node_size", kLegacyNodeBytes},
    {"node_layout", "plain"},
    {"storage_format", "vamana_compact_v1"},
    {"schema_version", 15},
    {"graph_hot_bytes", 16},
    {"vector_offset", kLegacyVectorOffset},
    {"vector_storage_bytes", 8},
    {"vector_data_type", "uint8"},
    {"vector_component_size", 1},
    {"vector_bytes", kDim},
    {"partition_strategy", "metis"},
    {"partition_max_degree", kDegree},
    {"partition_imbalance", 1.05},
    {"partition_edge_cut", 4},
    {"partition_cross_shard_ratio", 0.5},
    {"hot_graph_neighbor_read_bytes", kLegacyGraphEntryBytes},
    {"hot_graph_neighbor_update_bytes", kLegacyGraphEntryBytes},
    {"hot_graph_entry_size", kLegacyGraphEntryBytes},
    {"hot_graph_pointer_bytes", 5},
    {"hot_graph_shard_bits", kLegacyShardBits},
    {"hot_graph_offsets", {kLegacyGraphOffset, kLegacyGraphOffset}},
    {"hot_graph_header_offsets",
     {kLegacyGraphHeaderOffset, kLegacyGraphHeaderOffset}},
    {"hot_graph_entry_counts", {2, 2}},
    {"hot_graph_dynamic_base_offsets",
     {kLegacyDynamicBase, kLegacyDynamicBase}},
    {"hot_graph_dynamic_record_bytes", 64},
    {"hot_graph_dynamic_hot_offset", kLegacyNodeBytes},
    {"allocation_size", 64},
    {"idmap_format", "owner_sharded_v1"},
    {"navigation_quantizer", ""},
    {"navigation_code_bytes", 0},
    {"pq_subquantizers", 0},
    {"pq_bits", 0},
    {"navigation_model_checksum", 0},
    {"navigation_format", ""},
    {"navigation_entry_points", 0},
    {"navigation_code_remote_offsets", nlohmann::json::array()},
    {"navigation_code_region_bytes", nlohmann::json::array()},
    {"navigation_graph_source", "storage_compact_graph"},
  };
  std::ofstream output(prefix.string() + ".meta.json");
  output << metadata.dump(2) << '\n';
  assert(output.good());
}

void write_legacy_fixture(const fs::path& prefix) {
  write_legacy_metadata(prefix);
  for (u32 shard = 0; shard < kShards; ++shard) {
    write_legacy_shard(prefix, shard);
  }
  write_legacy_idmaps(prefix);
}

void corrupt_first_legacy_graph_checksum(const fs::path& prefix) {
  const fs::path path = index_path::shard_file(prefix, 1, kShards);
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  assert(file.good());
  file.seekg(static_cast<std::streamoff>(kLegacyGraphOffset + 2));
  u16 checksum = 0;
  file.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
  assert(file.gcount() == static_cast<std::streamsize>(sizeof(checksum)));
  checksum ^= 1;
  file.seekp(static_cast<std::streamoff>(kLegacyGraphOffset + 2));
  file.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
  file.flush();
  assert(file.good());
}

template <typename Operation>
bool throws_runtime_error(Operation&& operation) {
  try {
    operation();
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

nlohmann::json read_json(const fs::path& path) {
  std::ifstream input(path);
  assert(input.good());
  nlohmann::json value;
  input >> value;
  return value;
}

fs::path add_legacy_pq_model(const fs::path& prefix) {
  gpu_search::pq::Model model;
  model.dim = kDim;
  model.subquantizers = 2;
  model.centroids.resize(
    static_cast<size_t>(model.subquantizers) *
    gpu_search::pq::kCentroidsPerSubquantizer * model.subvector_dim());
  for (u32 subquantizer = 0;
       subquantizer < model.subquantizers; ++subquantizer) {
    for (u32 centroid = 0;
         centroid < gpu_search::pq::kCentroidsPerSubquantizer; ++centroid) {
      const size_t base =
        (static_cast<size_t>(subquantizer) *
           gpu_search::pq::kCentroidsPerSubquantizer + centroid) *
        model.subvector_dim();
      model.centroids[base] = static_cast<f32>(centroid);
      model.centroids[base + 1] = static_cast<f32>(centroid * 2);
    }
  }
  std::string error;
  assert(gpu_search::pq::validate(model, &error));
  const fs::path model_path =
    index_path::navigation_model_file(prefix, model.subquantizers);
  assert(gpu_search::pq::write_model(model_path, model, &error));

  const fs::path metadata_path{prefix.string() + ".meta.json"};
  nlohmann::json metadata = read_json(metadata_path);
  metadata["navigation_quantizer"] = "opq_pq";
  metadata["navigation_code_bytes"] = model.code_bytes();
  metadata["pq_subquantizers"] = model.subquantizers;
  metadata["pq_bits"] = model.bits_per_code;
  metadata["navigation_model_checksum"] = model.checksum();
  metadata["navigation_model_file"] = model_path.string();
  metadata["navigation_format"] = "opq_pq_graph_v1";
  std::ofstream output(metadata_path, std::ios::trunc);
  output << metadata.dump(2) << '\n';
  assert(output.good());
  return model_path;
}

std::vector<byte_t> read_exact(const fs::path& path,
                               u64 offset, size_t bytes) {
  std::ifstream input(path, std::ios::binary);
  assert(input.good());
  input.seekg(static_cast<std::streamoff>(offset));
  std::vector<byte_t> result(bytes);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(result.size()));
  assert(input.gcount() == static_cast<std::streamsize>(result.size()));
  return result;
}

std::map<node_t, RemotePtr> verify_nodes(
    const fs::path& output_prefix, const nlohmann::json& metadata) {
  const u32 node_bytes = metadata.at("node_size").get<u32>();
  const u32 vector_offset = metadata.at("vector_offset").get<u32>();
  const u32 slot_incarnation_offset =
    metadata.at("slot_incarnation_offset").get<u32>();
  std::map<node_t, RemotePtr> pointers;
  for (const FixtureNode& expected : kNodes) {
    const u64 offset = kNodeBase +
      static_cast<u64>(expected.slot) * node_bytes;
    const fs::path shard_path = index_path::shard_file(
      output_prefix, expected.shard + 1, kShards);
    const std::vector<byte_t> node = read_exact(
      shard_path, offset, node_bytes);
    const u64 header = load<u64>(node.data());
    assert((header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0);
    assert(VamanaNode::header_incarnation(header) == 0);
    assert(load<node_t>(node.data() + VamanaNode::offset_id()) ==
           expected.id);
    assert(load<u32>(node.data() + VamanaNode::offset_generation()) == 0);
    assert(load<u32>(node.data() + slot_incarnation_offset) == 0);
    assert(std::equal(expected.vector.begin(), expected.vector.end(),
                      reinterpret_cast<const u8*>(
                        node.data() + vector_offset)));
    pointers.emplace(expected.id, RemotePtr{expected.shard, offset});
  }
  return pointers;
}

void verify_graph(const fs::path& output_prefix,
                  const nlohmann::json& metadata,
                  const std::map<node_t, RemotePtr>& pointers) {
  const std::vector<u64> graph_offsets =
    metadata.at("hot_graph_offsets").get<std::vector<u64>>();
  const u32 graph_bytes = metadata.at("hot_graph_entry_size").get<u32>();
  std::unordered_map<u64, node_t> ids;
  for (const auto& [id, pointer] : pointers) {
    ids.emplace(pointer.raw_address, id);
  }
  for (const FixtureNode& expected : kNodes) {
    const fs::path shard_path = index_path::shard_file(
      output_prefix, expected.shard + 1, kShards);
    const std::vector<byte_t> compact = read_exact(
      shard_path,
      graph_offsets.at(expected.shard) +
        static_cast<u64>(expected.slot) * graph_bytes,
      graph_bytes);
    std::vector<byte_t> decoded(VamanaNode::neighbor_read_size());
    assert(VamanaNode::decode_hot_graph_entry(
      compact.data(), decoded.data(), 0));
    assert(decoded[VamanaNode::stable_neighbor_count_offset_in_read()] ==
           kDegree);
    assert(decoded[
      VamanaNode::provisional_neighbor_count_offset_in_read()] == 0);
    const auto* neighbors = reinterpret_cast<const RemotePtr*>(
      decoded.data() + VamanaNode::neighbor_payload_offset_in_read());
    std::set<node_t> actual;
    for (u32 edge = 0; edge < kDegree; ++edge) {
      assert(neighbors[edge].incarnation() == 0);
      const auto iterator = ids.find(neighbors[edge].raw_address);
      assert(iterator != ids.end());
      actual.insert(iterator->second);
    }
    const std::set<node_t> wanted{
      expected.neighbors.begin(), expected.neighbors.end()};
    assert(actual == wanted);
  }
}

void verify_idmaps(const fs::path& output_prefix,
                   const nlohmann::json& metadata,
                   const std::map<node_t, RemotePtr>& pointers) {
  const u64 build_fingerprint =
    metadata.at("index_build_fingerprint").get<u64>();
  const std::vector<u64> shard_fingerprints =
    metadata.at("shard_build_fingerprints").get<std::vector<u64>>();
  const std::vector<u64> counts =
    metadata.at("hot_graph_entry_counts").get<std::vector<u64>>();
  std::set<node_t> all_ids;
  for (u32 owner = 0; owner < kShards; ++owner) {
    const fs::path path = index_path::owner_idmap_file(
      output_prefix, owner + 1, kShards);
    std::ifstream input(path, std::ios::binary);
    assert(input.good());
    vamana::idmap::Header header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    assert(input.gcount() == static_cast<std::streamsize>(sizeof(header)));
    const vamana::idmap::ValidationContext context{
      .build_fingerprint = build_fingerprint,
      .owner_shard_fingerprint = shard_fingerprints.at(owner),
      .node_base_offset = kNodeBase,
      .owner_shard = owner,
      .shard_count = kShards,
      .node_size = metadata.at("node_size").get<u32>(),
      .id_offset = static_cast<u32>(VamanaNode::offset_id()),
      .generation_offset = static_cast<u32>(
        VamanaNode::offset_generation()),
      .slot_incarnation_offset =
        metadata.at("slot_incarnation_offset").get<u32>(),
      .static_entry_counts = counts,
    };
    assert(vamana::idmap::valid_header(
      header, fs::file_size(path), context));
    assert(vamana::idmap::read_validated_payload(
      input, header, context,
      [&](const vamana::idmap::Entry& entry) {
        const auto expected = pointers.find(entry.id);
        return expected != pointers.end() &&
          expected->second.raw_address == entry.rptr_raw &&
          all_ids.insert(entry.id).second;
      }));
  }
  assert(all_ids == std::set<node_t>({0, 1, 2, 3}));
}

void verify_centroids(const fs::path& output_prefix,
                      const nlohmann::json& metadata,
                      const std::map<node_t, RemotePtr>& pointers) {
  const std::array<std::array<f64, kDim>, kShards> expected_sums{{
    {4, 8, 12, 16},
    {24, 28, 32, 36},
  }};
  const u64 build_fingerprint =
    metadata.at("index_build_fingerprint").get<u64>();
  const std::vector<u64> shard_fingerprints =
    metadata.at("shard_build_fingerprints").get<std::vector<u64>>();
  std::unordered_map<u64, node_t> ids;
  for (const auto& [id, pointer] : pointers) {
    ids.emplace(pointer.raw_address, id);
  }

  for (u32 shard = 0; shard < kShards; ++shard) {
    const fs::path path = index_path::centroid_state_file(
      output_prefix, shard + 1, kShards);
    std::ifstream input(path, std::ios::binary);
    assert(input.good());
    vamana::centroid_state::Header header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    assert(input.gcount() == static_cast<std::streamsize>(sizeof(header)));
    assert(header.magic == vamana::centroid_state::kMagic);
    assert(header.version == vamana::centroid_state::kVersion);
    assert(header.build_fingerprint == build_fingerprint);
    assert(header.shard_fingerprint == shard_fingerprints.at(shard));
    assert(header.shard == shard && header.shard_count == kShards);
    assert(header.dim == kDim && header.max_degree == kDegree);
    assert(header.vector_dtype == static_cast<u32>(VectorDType::uint8));
    assert(header.vector_count == 2 && header.entry_count == 2);
    assert(header.payload_bytes ==
           vamana::centroid_state::payload_bytes(kDim, 2));
    assert(vamana::centroid_state::valid_header_checksum(header));
    assert(fs::file_size(path) == sizeof(header) + header.payload_bytes);

    std::vector<byte_t> payload(header.payload_bytes);
    input.read(reinterpret_cast<char*>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
    assert(input.gcount() == static_cast<std::streamsize>(payload.size()));
    assert(vamana::centroid_state::checksum(payload) ==
           header.payload_checksum);
    const auto* sums = reinterpret_cast<const f64*>(payload.data());
    for (u32 dimension = 0; dimension < kDim; ++dimension) {
      assert(sums[dimension] == expected_sums[shard][dimension]);
    }
    const auto* entries =
      reinterpret_cast<const vamana::centroid_state::Entry*>(
        payload.data() + kDim * sizeof(f64));
    std::set<node_t> route_ids;
    for (u32 index = 0; index < header.entry_count; ++index) {
      assert(entries[index].generation == 0);
      assert(entries[index].reserved == 0);
      const auto iterator = ids.find(entries[index].remote_node);
      assert(iterator != ids.end());
      assert(fixture_node(iterator->second).shard == shard);
      route_ids.insert(iterator->second);
    }
    const std::set<node_t> expected_ids = shard == 0
      ? std::set<node_t>{0, 2}
      : std::set<node_t>{1, 3};
    assert(route_ids == expected_ids);
  }
}

void verify_metadata(const nlohmann::json& metadata,
                     const fs::path& output_prefix) {
  assert(metadata.at("schema_version") == 15);
  assert(metadata.at("storage_format") ==
         VamanaNode::storage_format_name());
  assert(metadata.at("output_prefix") == output_prefix.string());
  assert(metadata.at("num_vectors") == kNodes.size());
  assert(metadata.at("num_memory_nodes") == kShards);
  assert(metadata.at("vector_data_type") == "uint8");
  assert(metadata.at("node_size") == VamanaNode::total_size());
  assert(metadata.at("vector_offset") == VamanaNode::offset_vector());
  assert(metadata.at("slot_incarnation_offset") ==
         VamanaNode::offset_slot_incarnation());
  assert(metadata.at("hot_graph_entry_size") ==
         VamanaNode::hot_graph_entry_size());
  assert(metadata.at("hot_graph_pointer_bytes") == sizeof(RemotePtr));
  assert(metadata.at("idmap_format") == "owner_sharded_v2_bound");
  assert(metadata.at("centroid_state_format") ==
         "physical_shard_centroid_v2_bound");
  assert(metadata.at("index_build_fingerprint").get<u64>() != 0);
  const std::vector<u64> fingerprints =
    metadata.at("shard_build_fingerprints").get<std::vector<u64>>();
  assert(fingerprints.size() == kShards);
  assert(fingerprints[0] != 0 && fingerprints[1] != 0 &&
         fingerprints[0] != fingerprints[1]);
}

void verify_final_schema16(const fs::path& output_prefix,
                           const legacy_converter::LegacyIndexConversionResult&
                             result) {
  assert(result.wrote_graph);
  assert(result.built_pq);
  assert(result.subquantizers == 2);
  const nlohmann::json metadata = read_json(result.metadata_file);
  assert(metadata.at("schema_version") ==
         gpu_search::format::kMetadataSchemaVersion);
  assert(metadata.at("pq_subquantizers") == 2);
  assert(metadata.at("navigation_code_bytes") == 2);
  assert(metadata.at("navigation_model_checksum").get<u64>() != 0);

  gpu_search::format::View view;
  std::string error;
  assert(gpu_search::format::synthesize_distributed_view(
    output_prefix, view, &error));
  assert(view.layout.num_nodes == kNodes.size());
  assert(view.layout.num_shards == kShards);
  assert(view.layout.pq_subquantizers == 2);
  assert(view.layout.code_bytes == 2);
  assert(view.shards.size() == kShards);

  const std::vector<u64> shard_fingerprints =
    metadata.at("shard_build_fingerprints").get<std::vector<u64>>();
  for (u32 shard = 0; shard < kShards; ++shard) {
    const fs::path code_path = index_path::navigation_code_file(
      output_prefix, shard + 1, kShards, 2);
    gpu_search::format::CodeHeader header;
    assert(gpu_search::format::read_code_header(
      code_path, header, &error));
    assert(gpu_search::format::validate_code_header(header, &error));
    assert(header.memory_node == shard);
    assert(header.code_bytes == 2);
    assert(header.node_size == VamanaNode::total_size());
    assert(header.vector_dtype == static_cast<u32>(VectorDType::uint8));
    assert(header.entry_count == 2);
    assert(header.payload_bytes == 4);
    assert(header.model_checksum == view.layout.model_checksum);
    assert(header.build_fingerprint ==
           metadata.at("index_build_fingerprint").get<u64>());
    assert(header.shard_fingerprint == shard_fingerprints.at(shard));
    assert(fs::file_size(code_path) ==
           sizeof(gpu_search::format::CodeHeader) + header.payload_bytes);
  }
}

}  // namespace

int main() {
  TemporaryDirectory temporary;
  const fs::path input_prefix = temporary.path() / "legacy";
  const fs::path output_prefix = temporary.path() / "converted";
  write_legacy_fixture(input_prefix);

  VamanaNode::init_static_storage(kDim, kDegree, VectorDType::uint8);
  const legacy_converter::LegacyIndexConversionResult result =
    legacy_converter::convert_legacy_schema15_index({
      .input_prefix = input_prefix,
      .output_prefix = output_prefix,
      .reuse_model = {},
      .subquantizers = 0,
      .chunk_vectors = 1,
      .threads = 2,
      .dry_run = false,
      .graph_only = true,
    });
  assert(result.node_count == kNodes.size());
  assert(result.edge_count == kNodes.size() * kDegree);
  assert(result.shards == kShards);
  assert(result.wrote_graph);
  assert(!result.built_pq);

  assert(result.metadata_file ==
         fs::path(output_prefix.string() + ".meta.json"));
  const nlohmann::json metadata = read_json(result.metadata_file);
  verify_metadata(metadata, output_prefix);
  const std::map<node_t, RemotePtr> pointers =
    verify_nodes(output_prefix, metadata);
  verify_graph(output_prefix, metadata, pointers);
  verify_idmaps(output_prefix, metadata, pointers);
  verify_centroids(output_prefix, metadata, pointers);

  const fs::path corrupt_input = temporary.path() / "corrupt_legacy";
  const fs::path corrupt_output = temporary.path() / "corrupt_converted";
  write_legacy_fixture(corrupt_input);
  corrupt_first_legacy_graph_checksum(corrupt_input);
  assert(throws_runtime_error([&] {
    (void)legacy_converter::convert_legacy_schema15_index({
      .input_prefix = corrupt_input,
      .output_prefix = corrupt_output,
      .reuse_model = {},
      .subquantizers = 0,
      .chunk_vectors = 1,
      .threads = 1,
      .dry_run = false,
      .graph_only = true,
    });
  }));
  // Metadata is the commit marker. A rejected legacy graph must never leave
  // an output that a runtime could mistake for a complete conversion.
  assert(!fs::exists(fs::path(corrupt_output.string() + ".meta.json")));

  assert(throws_runtime_error([&] {
    (void)legacy_converter::convert_legacy_schema15_index({
      .input_prefix = input_prefix,
      .output_prefix = input_prefix,
      .reuse_model = {},
      .subquantizers = 0,
      .chunk_vectors = 1,
      .threads = 1,
      .dry_run = true,
      .graph_only = true,
    });
  }));

  const fs::path final_input = temporary.path() / "pq_legacy";
  const fs::path final_output = temporary.path() / "schema16";
  write_legacy_fixture(final_input);
  const fs::path legacy_model = add_legacy_pq_model(final_input);
  const legacy_converter::LegacyIndexConversionResult final_result =
    legacy_converter::convert_legacy_schema15_index({
      .input_prefix = final_input,
      .output_prefix = final_output,
      .reuse_model = legacy_model,
      .subquantizers = 2,
      .chunk_vectors = 1,
      .threads = 1,
      .dry_run = false,
      .graph_only = false,
    });
  assert(final_result.node_count == kNodes.size());
  assert(final_result.edge_count == kNodes.size() * kDegree);
  assert(final_result.shards == kShards);
  assert(final_result.legacy_model_file == legacy_model);
  verify_final_schema16(final_output, final_result);
  return 0;
}
