#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "gpu_search/pq_index.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "service/index_metadata.hh"
#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/graph_extent_indexer.hh"
#include "tools/vamana_offline/metis_repartitioner.hh"
#include "tools/vamana_offline/pq_indexer.hh"
#include "tools/vamana_offline/shard_writer.hh"
#include "vamana/centroid_state.hh"
#include "vamana/idmap.hh"
#include "vamana/vamana_node.hh"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace tools::vamana_offline;

namespace {

constexpr u32 kNodes = 512;
constexpr u32 kDim = 8;
constexpr u32 kDegree = 4;
constexpr u32 kShards = 4;
constexpr u32 kSubquantizers = 4;

struct TemporaryDirectory {
  fs::path path = fs::temp_directory_path() /
                  ("dvstor-metis-repartition-" + std::to_string(::getpid()));
  TemporaryDirectory() {
    std::error_code ignored;
    fs::remove_all(path, ignored);
    fs::create_directories(path);
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    fs::remove_all(path, ignored);
  }
};

template <typename T> T load(const byte_t *source) {
  T value{};
  std::memcpy(&value, source, sizeof(value));
  return value;
}

std::vector<char> read_all(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  assert(input.good());
  return {std::istreambuf_iterator<char>{input},
          std::istreambuf_iterator<char>{}};
}

json read_json(const fs::path &path) {
  std::ifstream input(path);
  json result;
  input >> result;
  assert(input.good() || input.eof());
  return result;
}

void create_dataset(const fs::path &path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(&kNodes), sizeof(kNodes));
  output.write(reinterpret_cast<const char *>(&kDim), sizeof(kDim));
  for (u32 node = 0; node < kNodes; ++node) {
    for (u32 dimension = 0; dimension < kDim; ++dimension) {
      const float value =
          static_cast<float>((node * 17 + dimension * 29) % 997) / 31.0F;
      output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }
  }
  output.close();
  assert(!output.fail());
}

fs::path create_reusable_pq_model(const fs::path &path) {
  gpu_search::pq::Model model;
  model.dim = kDim;
  model.subquantizers = kSubquantizers;
  model.centroids.resize(static_cast<size_t>(model.subquantizers) *
                         gpu_search::pq::kCentroidsPerSubquantizer *
                         model.subvector_dim());
  for (u32 subquantizer = 0; subquantizer < model.subquantizers;
       ++subquantizer) {
    for (u32 centroid = 0; centroid < gpu_search::pq::kCentroidsPerSubquantizer;
         ++centroid) {
      const size_t base = (static_cast<size_t>(subquantizer) *
                               gpu_search::pq::kCentroidsPerSubquantizer +
                           centroid) *
                          model.subvector_dim();
      model.centroids[base] = static_cast<float>(centroid % 16) * 2.0F;
      model.centroids[base + 1] = static_cast<float>(centroid / 16) * 2.0F;
    }
  }
  std::string error;
  assert(gpu_search::pq::validate(model, &error));
  assert(gpu_search::pq::write_model(path, model, &error));
  return path;
}

std::map<node_t, RemotePtr> read_idmaps(const fs::path &prefix,
                                        const json &metadata) {
  const u32 shards = metadata.at("num_memory_nodes").get<u32>();
  std::map<node_t, RemotePtr> result;
  for (u32 owner = 0; owner < shards; ++owner) {
    std::ifstream input(index_path::owner_idmap_file(prefix, owner + 1, shards),
                        std::ios::binary);
    vamana::idmap::Header header;
    input.read(reinterpret_cast<char *>(&header), sizeof(header));
    assert(input.gcount() == static_cast<std::streamsize>(sizeof(header)));
    for (u64 index = 0; index < header.entry_count; ++index) {
      vamana::idmap::Entry entry;
      input.read(reinterpret_cast<char *>(&entry), sizeof(entry));
      assert(input.gcount() == static_cast<std::streamsize>(sizeof(entry)));
      assert(result.emplace(entry.id, RemotePtr{entry.rptr_raw}).second);
    }
  }
  assert(result.size() == kNodes);
  return result;
}

std::map<node_t, std::set<node_t>>
read_logical_graph(const fs::path &prefix, const json &metadata,
                   const std::map<node_t, RemotePtr> &pointers) {
  const u32 shards = metadata.at("num_memory_nodes").get<u32>();
  const u32 node_bytes = metadata.at("node_size").get<u32>();
  const u32 graph_bytes = metadata.at("hot_graph_entry_size").get<u32>();
  const u32 shard_bits = metadata.at("hot_graph_shard_bits").get<u32>();
  const auto counts = metadata.at("hot_graph_entry_counts").get<vec<u64>>();
  const auto offsets = metadata.at("hot_graph_offsets").get<vec<u64>>();
  std::map<u64, node_t> ids_by_pointer;
  for (const auto &[id, pointer] : pointers) {
    ids_by_pointer.emplace(pointer.raw_address, id);
  }
  std::map<node_t, std::set<node_t>> result;
  vec<byte_t> compact(graph_bytes);
  vec<byte_t> decoded(VamanaNode::neighbor_read_size());
  VamanaNode::configure_hot_graph(
      offsets, counts, graph_bytes, shard_bits,
      metadata.at("hot_graph_dynamic_base_offsets").get<vec<u64>>(),
      metadata.at("hot_graph_dynamic_record_bytes").get<u32>(),
      metadata.at("hot_graph_dynamic_hot_offset").get<u32>(),
      metadata.at("dynamic_navigation_code_offset").get<u32>(),
      metadata.at("navigation_code_bytes").get<u32>());
  for (const auto &[id, pointer] : pointers) {
    const u64 slot =
        (pointer.byte_offset() - vamana::hot_graph::kNodeBaseOffset) /
        node_bytes;
    std::ifstream input(
        index_path::shard_file(prefix, pointer.memory_node() + 1, shards),
        std::ios::binary);
    input.seekg(static_cast<std::streamoff>(offsets.at(pointer.memory_node()) +
                                            slot * graph_bytes));
    input.read(reinterpret_cast<char *>(compact.data()),
               static_cast<std::streamsize>(compact.size()));
    assert(input.gcount() == static_cast<std::streamsize>(compact.size()));
    assert(
        VamanaNode::decode_hot_graph_entry(compact.data(), decoded.data(), 0));
    const u32 degree =
        decoded[VamanaNode::stable_neighbor_count_offset_in_read()];
    const byte_t *neighbors =
        decoded.data() + VamanaNode::neighbor_payload_offset_in_read();
    for (u32 edge = 0; edge < degree; ++edge) {
      const RemotePtr neighbor =
          load<RemotePtr>(neighbors + edge * sizeof(RemotePtr));
      const auto found = ids_by_pointer.find(neighbor.raw_address);
      assert(found != ids_by_pointer.end());
      result[id].insert(found->second);
    }
  }
  return result;
}

void verify_centroids(const fs::path &prefix, const json &metadata,
                      const Dataset &dataset,
                      const std::map<node_t, RemotePtr> &pointers,
                      u64 source_build_fingerprint) {
  const u64 build_fingerprint =
      metadata.at("index_build_fingerprint").get<u64>();
  assert(build_fingerprint != source_build_fingerprint);
  vec<vec<long double>> sums(kShards, vec<long double>(kDim, 0.0L));
  vec<u64> counts(kShards, 0);
  for (const auto &[id, pointer] : pointers) {
    ++counts[pointer.memory_node()];
    for (u32 dimension = 0; dimension < kDim; ++dimension) {
      sums[pointer.memory_node()][dimension] += static_cast<long double>(
          load<float>(dataset.raw_vector(id) + dimension * sizeof(float)));
    }
  }
  const auto shard_fingerprints =
      metadata.at("shard_build_fingerprints").get<vec<u64>>();
  for (u32 shard = 0; shard < kShards; ++shard) {
    std::ifstream input(
        index_path::centroid_state_file(prefix, shard + 1, kShards),
        std::ios::binary);
    vamana::centroid_state::Header header;
    input.read(reinterpret_cast<char *>(&header), sizeof(header));
    assert(input.gcount() == static_cast<std::streamsize>(sizeof(header)));
    assert(header.build_fingerprint == build_fingerprint);
    assert(header.shard_fingerprint == shard_fingerprints.at(shard));
    assert(header.vector_count == counts[shard]);
    vec<byte_t> payload(static_cast<size_t>(header.payload_bytes));
    input.read(reinterpret_cast<char *>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
    assert(input.gcount() == static_cast<std::streamsize>(payload.size()));
    assert(vamana::centroid_state::checksum(payload) ==
           header.payload_checksum);
    for (u32 dimension = 0; dimension < kDim; ++dimension) {
      const f64 actual = load<f64>(payload.data() + dimension * sizeof(f64));
      assert(std::abs(static_cast<long double>(actual) -
                      sums[shard][dimension]) < 1e-9L);
    }
  }
}

void expect_failure(const MetisRepartitionOptions &options) {
  bool rejected = false;
  try {
    (void)repartition_schema16_index(options);
  } catch (const std::exception &) {
    rejected = true;
  }
  assert(rejected);
}

} // namespace

int main() {
  TemporaryDirectory temporary;
  const fs::path data_path = temporary.path / "base.fbin";
  const fs::path source_prefix = temporary.path / "balanced";
  const fs::path output_prefix = temporary.path / "metis";
  create_dataset(data_path);

  VamanaBuildConfig build;
  build.data_path = data_path;
  build.vector_data_type = "float32";
  build.max_vectors = kNodes;
  build.R = kDegree;
  build.beam_width = 8;
  build.alpha = 1.2;
  build.num_memory_nodes = kShards;
  build.partition_strategy = "balanced";
  build.partition_max_degree = kDegree;
  build.partition_imbalance = 1.05;
  VamanaNode::init_static_storage(kDim, kDegree, VectorDType::float32);
  const Dataset dataset = read_dataset(build);
  VamanaGraph graph;
  graph.init(kNodes, kDim, kDegree, 64);
  graph.medoid = 0;
  for (u32 node = 0; node < kNodes; ++node) {
    graph.set_neighbors(node, {
                                  (node + 1) % kNodes,
                                  (node + kNodes - 1) % kNodes,
                                  (node + 2) % kNodes,
                                  (node + kNodes - 2) % kNodes,
                              });
  }
  write_vamana_shards(graph, dataset, build, source_prefix);
  const fs::path reusable_model =
      create_reusable_pq_model(temporary.path / "reusable.pq4");
  PqIndexOptions pq;
  pq.index_prefix = source_prefix;
  pq.reuse_model = reusable_model;
  pq.subquantizers = kSubquantizers;
  pq.chunk_vectors = 128;
  pq.threads = 1;
  pq.seed = 1234;
  pq.overwrite = true;
  (void)build_pq_index(pq);
  GraphExtentIndexOptions extent;
  extent.index_prefix = source_prefix;
  extent.overwrite = true;
  (void)build_graph_extent_index(extent);

  const auto source_metadata_bytes =
      read_all(fs::path{source_prefix.string() + ".meta.json"});
  const json source_metadata =
      read_json(fs::path{source_prefix.string() + ".meta.json"});
  const u64 source_fingerprint =
      source_metadata.at("index_build_fingerprint").get<u64>();
  const auto source_pointers = read_idmaps(source_prefix, source_metadata);
  const auto source_graph =
      read_logical_graph(source_prefix, source_metadata, source_pointers);

  MetisRepartitionOptions options;
  options.input_prefix = source_prefix;
  options.output_prefix = output_prefix;
  options.partition_max_degree = kDegree;
  options.partition_imbalance = 1.05;
  options.threads = 1;
  options.pq_chunk_vectors = 128;
  const auto converted = repartition_schema16_index(options);
  assert(converted.node_count == kNodes);
  assert(converted.edge_count == static_cast<u64>(kNodes) * kDegree);
  assert(converted.shards == kShards);
  assert(converted.graph_written && converted.pq_built &&
         converted.extent_built && !converted.resumed);

  const json output_metadata =
      read_json(fs::path{output_prefix.string() + ".meta.json"});
  assert(output_metadata.at("schema_version") == 16);
  assert(output_metadata.at("partition_strategy") == "metis");
  assert(output_metadata.at("repartition_source_build_fingerprint") ==
         source_fingerprint);
  const auto output_pointers = read_idmaps(output_prefix, output_metadata);
  size_t moved = 0;
  for (u32 id = 0; id < kNodes; ++id) {
    if (source_pointers.at(id).memory_node() !=
        output_pointers.at(id).memory_node()) {
      ++moved;
    }
  }
  assert(moved != 0);
  const auto output_graph =
      read_logical_graph(output_prefix, output_metadata, output_pointers);
  assert(output_graph == source_graph);
  verify_centroids(output_prefix, output_metadata, dataset, output_pointers,
                   source_fingerprint);
  assert(read_all(index_path::navigation_model_file(source_prefix,
                                                    kSubquantizers)) ==
         read_all(
             index_path::navigation_model_file(output_prefix, kSubquantizers)));
  assert(read_all(fs::path{source_prefix.string() + ".meta.json"}) ==
         source_metadata_bytes);

  service::index_metadata::Metadata runtime_metadata;
  std::string error;
  assert(service::index_metadata::load_metadata(output_prefix, runtime_metadata,
                                                &error));
  gpu_search::format::View view;
  assert(gpu_search::format::synthesize_distributed_view(output_prefix, view,
                                                         &error));

  const auto resumed = repartition_schema16_index(options);
  assert(resumed.resumed && !resumed.graph_written && !resumed.pq_built &&
         !resumed.extent_built);
  assert(resumed.output_build_fingerprint ==
         converted.output_build_fingerprint);

  MetisRepartitionOptions completed_graph_only = options;
  completed_graph_only.graph_only = true;
  const auto completed_graph_result =
      repartition_schema16_index(completed_graph_only);
  assert(completed_graph_result.resumed &&
         completed_graph_result.output_build_fingerprint ==
             converted.output_build_fingerprint);

  MetisRepartitionOptions in_place = options;
  in_place.output_prefix = source_prefix;
  expect_failure(in_place);

  const fs::path graph_only_prefix = temporary.path / "metis_graph_only";
  MetisRepartitionOptions graph_only = options;
  graph_only.output_prefix = graph_only_prefix;
  graph_only.graph_only = true;
  const auto graph_result = repartition_schema16_index(graph_only);
  assert(graph_result.graph_written && !graph_result.pq_built &&
         !graph_result.extent_built);
  assert(read_json(fs::path{graph_only_prefix.string() + ".meta.json"})
             .at("schema_version") == 15);
  assert(fs::is_regular_file(
      fs::path{graph_only_prefix.string() + ".graph.meta.json"}));
  const vec<fs::path> stale_temporaries{
      fs::path{graph_only_prefix.string() + ".pq4.pq-indexer.tmp.999999.0"},
      fs::path{graph_only_prefix.string() + ".gextent8.tmp.999999"},
      fs::path{graph_only_prefix.string() +
               ".graph.meta.json.repartition.tmp.999999"},
  };
  for (const fs::path &path : stale_temporaries) {
    std::ofstream stale(path);
    stale << "stale";
    assert(stale.good());
  }
  assert(fs::remove(fs::path{graph_only_prefix.string() + ".graph.meta.json"}));
  const auto cleaned_graph_result = repartition_schema16_index(graph_only);
  assert(cleaned_graph_result.resumed && !cleaned_graph_result.graph_written);
  for (const fs::path &path : stale_temporaries)
    assert(!fs::exists(path));
  assert(fs::is_regular_file(
      fs::path{graph_only_prefix.string() + ".graph.meta.json"}));

  const fs::path recovery_prefix = temporary.path / "metis_recovery";
  MetisRepartitionOptions recovery = graph_only;
  recovery.output_prefix = recovery_prefix;
  const auto first_recovery = repartition_schema16_index(recovery);
  assert(first_recovery.graph_written);
  assert(fs::remove(fs::path{recovery_prefix.string() + ".meta.json"}));
  assert(fs::remove(fs::path{recovery_prefix.string() + ".graph.meta.json"}));
  const auto recovered = repartition_schema16_index(recovery);
  assert(recovered.resumed && recovered.graph_written);
  assert(read_json(fs::path{recovery_prefix.string() + ".meta.json"})
             .at("schema_version") == 15);

  MetisRepartitionOptions validate = options;
  validate.output_prefix = temporary.path / "unused";
  validate.validate_only = true;
  const auto validated = repartition_schema16_index(validate);
  assert(validated.edge_count == static_cast<u64>(kNodes) * kDegree);
  assert(!fs::exists(
      fs::path{validate.output_prefix.string() + ".repartition.plan.json"}));
  return 0;
}
