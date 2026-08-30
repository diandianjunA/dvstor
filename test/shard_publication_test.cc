#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "common/index_path.hh"
#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/shard_writer.hh"
#include "vamana/vamana_node.hh"

using namespace tools::vamana_offline;

namespace {

struct TemporaryDirectory {
  std::filesystem::path path;
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

std::vector<char> read_all(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  assert(input.good());
  return std::vector<char>{std::istreambuf_iterator<char>{input},
                           std::istreambuf_iterator<char>{}};
}

template <typename Function>
int run_child(Function&& function) {
  const pid_t child = ::fork();
  assert(child >= 0);
  if (child == 0) {
    function();
    ::_exit(0);
  }
  int status = 0;
  assert(::waitpid(child, &status, 0) == child);
  return status;
}

void assert_failed(int status) {
  assert(WIFEXITED(status));
  assert(WEXITSTATUS(status) != 0);
}

}  // namespace

int main() {
  const TemporaryDirectory temporary{
    std::filesystem::temp_directory_path() /
      ("dvstor_graph_publish_" + std::to_string(::getpid()))};
  std::filesystem::create_directories(temporary.path);
  const auto dataset_path = temporary.path / "base.fbin";
  {
    std::ofstream output(dataset_path,
                         std::ios::binary | std::ios::trunc);
    const uint32_t rows = 6;
    const uint32_t dim = 2;
    const float values[12] = {
      0.0F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F,
      3.0F, 0.0F, 4.0F, 0.0F, 5.0F, 0.0F};
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    output.write(reinterpret_cast<const char*>(values), sizeof(values));
    output.close();
    assert(!output.fail());
  }

  VamanaBuildConfig config;
  config.data_path = dataset_path;
  config.vector_data_type = "float32";
  config.max_vectors = 6;
  config.R = 2;
  config.beam_width = 4;
  config.num_memory_nodes = 2;
  config.partition_strategy = "balanced";
  VamanaNode::init_static_storage(2, config.R, VectorDType::float32);
  const Dataset dataset = read_dataset(config);
  VamanaGraph graph;
  graph.init(dataset.size(), dataset.dim, config.R, 8);
  graph.medoid = 0;
  for (uint32_t node = 0; node < dataset.size(); ++node) {
    graph.set_neighbors(
      node, vec<u32>{static_cast<u32>((node + 1) % dataset.size()),
                     static_cast<u32>((node + dataset.size() - 1) %
                                      dataset.size())});
  }

  const auto committed_prefix = temporary.path / "committed";
  write_vamana_shards(graph, dataset, config, committed_prefix);
  const auto metadata_path = std::filesystem::path{
    committed_prefix.string() + ".meta.json"};
  assert(std::filesystem::is_regular_file(metadata_path));
  for (uint32_t shard = 1; shard <= config.num_memory_nodes; ++shard) {
    assert(std::filesystem::is_regular_file(index_path::shard_file(
      committed_prefix, shard, config.num_memory_nodes)));
    assert(std::filesystem::is_regular_file(index_path::centroid_state_file(
      committed_prefix, shard, config.num_memory_nodes)));
    assert(std::filesystem::is_regular_file(index_path::owner_idmap_file(
      committed_prefix, shard, config.num_memory_nodes)));
  }
  for (const auto& entry : std::filesystem::directory_iterator(temporary.path)) {
    assert(entry.path().filename().string().find(".graph-build.tmp") ==
           std::string::npos);
  }

  const auto first_shard = index_path::shard_file(
    committed_prefix, 1, config.num_memory_nodes);
  const auto committed_bytes = read_all(first_shard);
  assert_failed(run_child([&] {
    write_vamana_shards(graph, dataset, config, committed_prefix);
  }));
  assert(read_all(first_shard) == committed_bytes);
  assert(std::filesystem::is_regular_file(metadata_path));

  const auto failed_prefix = temporary.path / "failed";
  const auto blocked_first_shard = index_path::shard_file(
    failed_prefix, 1, config.num_memory_nodes);
  std::filesystem::create_directory(blocked_first_shard);
  assert_failed(run_child([&] {
    write_vamana_shards(graph, dataset, config, failed_prefix);
  }));
  assert(!std::filesystem::exists(
    std::filesystem::path{failed_prefix.string() + ".meta.json"}));
  assert(std::filesystem::is_directory(blocked_first_shard));

  std::cout << "graph publication commit-marker regression passed\n";
  return 0;
}
