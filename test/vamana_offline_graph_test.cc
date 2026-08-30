#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include <unistd.h>

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

  std::cout << "offline beam-search growth regression passed\n";
  return 0;
}
