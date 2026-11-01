#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include <unistd.h>

#include "tools/vamana_offline/checkpoint.hh"
#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/graph.hh"

using namespace tools::vamana_offline;

namespace {

struct TemporaryFile {
  std::filesystem::path path;
  ~TemporaryFile() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
};

void test_explicit_dtype_for_ambiguous_bin() {
  TemporaryFile input{
    std::filesystem::temp_directory_path() /
    ("dvstor_explicit_dtype_" + std::to_string(::getpid()) + ".bin")};
  {
    std::ofstream output(input.path, std::ios::binary | std::ios::trunc);
    const uint32_t rows = 2;
    const uint32_t dim = 3;
    const int8_t values[6] = {-3, -2, -1, 1, 2, 3};
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    output.write(reinterpret_cast<const char*>(values), sizeof(values));
  }
  VamanaBuildConfig config;
  config.data_path = input.path;
  config.vector_data_type = "int8";
  config.max_vectors = 2;
  const Dataset dataset = read_dataset(config);
  assert(dataset.dtype == VectorDType::int8);
  assert(dataset.size() == 2);
  assert(dataset.dim == 3);
  assert(dataset.raw_vector(1)[0] == static_cast<byte_t>(1));
}

void test_full_reverse_edge_prunes() {
  TemporaryFile input{
    std::filesystem::temp_directory_path() /
    ("dvstor_reverse_prune_" + std::to_string(::getpid()) + ".fbin")};
  {
    std::ofstream output(input.path, std::ios::binary | std::ios::trunc);
    const uint32_t rows = 4;
    const uint32_t dim = 1;
    const float values[4] = {0.0F, 10.0F, 11.0F, 1.0F};
    output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    output.write(reinterpret_cast<const char *>(values), sizeof(values));
  }
  VamanaBuildConfig config;
  config.data_path = input.path;
  config.vector_data_type = "float32";
  config.max_vectors = 4;
  const Dataset dataset = read_dataset(config);
  VamanaGraph graph;
  graph.init(4, 1, 2, 64);
  graph.set_neighbors(0, {1, 2});
  assert(append_reverse_edge_with_prune_unlocked(graph, dataset, 0, 3, 1.0F));
  assert(graph.contains_neighbor_unlocked(0, 3));
  assert(!append_reverse_edge_with_prune_unlocked(graph, dataset, 0, 3, 1.0F));

  VamanaGraph far_graph;
  far_graph.init(4, 1, 2, 64);
  far_graph.set_neighbors(0, {2, 3});
  assert(append_reverse_edge_with_prune_unlocked(far_graph, dataset, 0, 1, 1.0F));
  assert(far_graph.contains_neighbor_unlocked(0, 3));
  assert(!far_graph.contains_neighbor_unlocked(0, 1));

  VamanaGraph slack_graph;
  slack_graph.init(4, 1, 2, 64, 0, 3);
  slack_graph.set_neighbors(0, {1, 2});
  assert(!append_reverse_edge_with_prune_unlocked(slack_graph, dataset, 0, 3, 1.0F));
  assert(slack_graph.degree(0) == 3);
  assert(slack_graph.contains_neighbor_unlocked(0, 3));

  const std::filesystem::path checkpoint =
      std::filesystem::temp_directory_path() /
      ("dvstor_checkpoint_" + std::to_string(::getpid()));
  GraphCheckpointMeta meta;
  meta.num_nodes = 4;
  meta.dim = 1;
  meta.R = 2;
  meta.build_capacity = 3;
  meta.beam_width = 16;
  meta.alpha = 1.2;
  meta.build_slack_factor = 1.5;
  meta.vector_data_type = "float32";
  meta.medoid = 0;
  meta.seed = 42;
  meta.phase = "main_loop";
  meta.next_step = 2;
  meta.segment_size = 2;
  save_graph_checkpoint(checkpoint, slack_graph, meta, 2);
  const GraphCheckpointMeta loaded_meta = read_graph_checkpoint_meta(checkpoint);
  assert(loaded_meta.build_capacity == 3);
  assert(loaded_meta.beam_width == 16);
  assert(loaded_meta.alpha == 1.2);
  assert(loaded_meta.build_slack_factor == 1.5);
  assert(loaded_meta.vector_data_type == "float32");
  VamanaGraph restored;
  restored.init(4, 1, 2, 64, 0, 3);
  load_graph_checkpoint_arrays(checkpoint, restored, loaded_meta, 2);
  assert(restored.degrees.size() == slack_graph.degrees.size());
  assert(restored.neighbors.size() == slack_graph.neighbors.size());
  for (size_t i = 0; i < restored.degrees.size(); ++i) {
    assert(restored.degrees[i] == slack_graph.degrees[i]);
  }
  for (size_t i = 0; i < restored.neighbors.size(); ++i) {
    assert(restored.neighbors[i] == slack_graph.neighbors[i]);
  }
  remove_graph_checkpoint(checkpoint);
}

vec<u32> reference_occlusion_prune(
    const Dataset& dataset, u32 source,
    const vec<std::pair<float, u32>>& candidates, float alpha, u32 R) {
  vec<float> factors(candidates.size(), 0.0F);
  vec<u32> selected;
  for (float cur_alpha = 1.0F;
       cur_alpha <= alpha && selected.size() < R; cur_alpha *= 1.2F) {
    for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
      if (factors[i] > cur_alpha) continue;
      factors[i] = std::numeric_limits<float>::max();
      const u32 candidate = candidates[i].second;
      if (candidate == source ||
          std::find(selected.begin(), selected.end(), candidate) != selected.end()) {
        continue;
      }
      selected.push_back(candidate);
      for (size_t j = i + 1; j < candidates.size(); ++j) {
        if (factors[j] > alpha) continue;
        const float distance = dataset_distance(dataset, candidate, candidates[j].second);
        const float ratio = distance == 0.0F
            ? std::numeric_limits<float>::max()
            : candidates[j].first / distance;
        factors[j] = std::max(factors[j], ratio);
      }
    }
  }
  return selected;
}

void test_reference_prune_differential() {
  TemporaryFile input{
      std::filesystem::temp_directory_path() /
      ("dvstor_prune_differential_" + std::to_string(::getpid()) + ".fbin")};
  constexpr uint32_t rows = 32;
  constexpr uint32_t dim = 2;
  {
    std::ofstream output(input.path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    for (uint32_t i = 0; i < rows; ++i) {
      const float values[2] = {
          static_cast<float>(static_cast<int>(i * 7 % 11) - 5),
          static_cast<float>(static_cast<int>(i * 13 % 17) - 8)};
      output.write(reinterpret_cast<const char *>(values), sizeof(values));
    }
  }
  VamanaBuildConfig config;
  config.data_path = input.path;
  config.vector_data_type = "float32";
  config.max_vectors = rows;
  const Dataset dataset = read_dataset(config);
  for (u32 source : {0u, 5u}) {
    vec<std::pair<float, u32>> candidates;
    for (u32 id = 0; id < rows; ++id) {
      if (id == source) continue;
      candidates.push_back({dataset_distance(dataset, source, id), id});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first < b.first : a.second < b.second;
              });
    for (const float alpha : {1.0F, 1.2F}) {
      const auto expected = reference_occlusion_prune(
          dataset, source, candidates, alpha, 8);
      const auto actual = robust_prune(dataset, source, candidates, alpha, 8);
      assert(actual == expected);
    }
  }
}

void validate_chain_result(const vec<std::pair<float, u32>> &result,
                           uint32_t nodes) {
  assert(result.size() == nodes);
  assert(result.front().second == nodes - 1);
  assert(result.back().second == 0);
  std::vector<uint32_t> ids;
  ids.reserve(result.size());
  for (size_t index = 0; index < result.size(); ++index) {
    assert(result[index].second < nodes);
    if (index != 0)
      assert(result[index - 1].first <= result[index].first);
    ids.push_back(result[index].second);
  }
  std::sort(ids.begin(), ids.end());
  assert(std::adjacent_find(ids.begin(), ids.end()) == ids.end());
  for (uint32_t id = 0; id < nodes; ++id)
    assert(ids[id] == id);
}

} // namespace

int main() {
  test_explicit_dtype_for_ambiguous_bin();
  test_full_reverse_edge_prunes();
  test_reference_prune_differential();

  constexpr uint32_t kNodes = 4097;
  constexpr uint32_t kDim = 1;
  TemporaryFile input{
      std::filesystem::temp_directory_path() /
      ("dvstor_offline_chain_" + std::to_string(::getpid()) + ".fbin")};

  {
    std::ofstream output(input.path, std::ios::binary | std::ios::trunc);
    assert(output.good());
    output.write(reinterpret_cast<const char *>(&kNodes), sizeof(kNodes));
    output.write(reinterpret_cast<const char *>(&kDim), sizeof(kDim));
    for (uint32_t id = 0; id < kNodes; ++id) {
      const float value = static_cast<float>(kNodes - 1 - id);
      output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }
    output.close();
    assert(!output.fail());
  }

  VamanaBuildConfig config;
  config.data_path = input.path;
  config.vector_data_type = "float32";
  config.max_vectors = kNodes;
  const Dataset dataset = read_dataset(config);

  VamanaGraph graph;
  graph.init(kNodes, kDim, 1, 64);
  graph.medoid = 0;
  for (uint32_t id = 0; id + 1 < kNodes; ++id) {
    graph.set_neighbors(id, vec<u32>{id + 1});
  }
  graph.set_neighbors(kNodes - 1, {});

  // Both search entry points used to spin forever after filling the 2,048-slot
  // expanded/visited tables.
  validate_chain_result(beam_search(graph, dataset, kNodes - 1, 1), kNodes);
  const float query[1] = {0.0F};
  validate_chain_result(beam_search_float_query(graph, dataset, query, 1),
                        kNodes);

  // Reproduce the production tail shape where twelve workers were stuck.
  std::vector<std::thread> workers;
  workers.reserve(12);
  for (size_t worker = 0; worker < 12; ++worker) {
    workers.emplace_back([&]() {
      validate_chain_result(beam_search(graph, dataset, kNodes - 1, 1), kNodes);
    });
  }
  for (auto &worker : workers)
    worker.join();

  VamanaGraph star;
  star.init(kNodes, kDim, 96, 64);
  star.medoid = 0;
  vec<u32> star_neighbors;
  for (u32 id = 1; id <= 96; ++id) star_neighbors.push_back(id);
  star.set_neighbors(0, star_neighbors);
  const auto expanded = beam_search(star, dataset, kNodes - 1, 16);
  assert(expanded.size() == 17);
  const auto visited = beam_search_float_query(star, dataset, query, 16);
  assert(visited.size() == 97);

  config.R = 4;
  config.beam_width = 16;
  config.threads = 4;
  config.skip_sanity_check = true;
  config.checkpoint_segment_nodes = 1024;
  VamanaGraph built;
  build_vamana_graph(built, dataset, config);
  assert(built.R == 4);
  assert(built.capacity == 6);
  size_t total_edges = 0;
  vec<u32> neighbors;
  for (u32 id = 0; id < kNodes; ++id) {
    assert(built.degree(id) <= built.R);
    built.copy_neighbors(id, neighbors);
    total_edges += neighbors.size();
    for (const u32 neighbor : neighbors) {
      assert(neighbor < kNodes);
      assert(neighbor != id);
    }
  }
  assert(total_edges + 1 >= kNodes);

  std::cout << "offline beam-search growth regression passed\n";
  return 0;
}
