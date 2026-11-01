#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <unistd.h>

#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/partitioning.hh"

using namespace tools::vamana_offline;

namespace {

constexpr uint32_t kNumShards = 8;
constexpr uint32_t kPointsPerCluster = 1250;
constexpr uint32_t kNumVectors = kNumShards * kPointsPerCluster;
constexpr uint32_t kDim = 16;

struct TemporaryDirectory {
  std::filesystem::path path;
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

uint32_t true_cluster(uint32_t row) { return row / kPointsPerCluster; }

// kNumShards well-separated equal-size Gaussian blobs: blob c sits near
// 1000*c on every dimension, so a good 8-way kmeans recovers one shard per
// blob with near-exact natural balance.
void write_synthetic_dataset(const std::filesystem::path& path) {
  std::mt19937 rng(42);
  std::normal_distribution<float> noise(0.0F, 0.5F);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  assert(output.good());
  output.write(reinterpret_cast<const char*>(&kNumVectors), sizeof(kNumVectors));
  output.write(reinterpret_cast<const char*>(&kDim), sizeof(kDim));
  for (uint32_t row = 0; row < kNumVectors; ++row) {
    const float center = 1000.0F * static_cast<float>(true_cluster(row));
    for (uint32_t d = 0; d < kDim; ++d) {
      const float value = center + noise(rng);
      output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  output.close();
  assert(!output.fail());
}

Dataset load_dataset(const std::filesystem::path& path) {
  VamanaBuildConfig config;
  config.data_path = path;
  config.vector_data_type = "float32";
  config.max_vectors = kNumVectors;
  config.num_memory_nodes = kNumShards;
  return read_dataset(config);
}

void check_exact_balance(const vec<size_t>& counts) {
  assert(counts.size() == kNumShards);
  const size_t base = kNumVectors / kNumShards;
  const size_t rem = kNumVectors % kNumShards;
  size_t total = 0;
  size_t ceil_shards = 0;
  for (size_t c = 0; c < kNumShards; ++c) {
    assert(counts[c] == base || counts[c] == base + 1);
    if (counts[c] == base + 1) ++ceil_shards;
    total += counts[c];
  }
  assert(total == kNumVectors);
  assert(ceil_shards == rem);
}

void check_cluster_coherence(const vec<u32>& parts, double max_minority_ratio) {
  size_t minority = 0;
  for (uint32_t shard = 0; shard < kNumShards; ++shard) {
    size_t label_counts[kNumShards] = {0};
    for (uint32_t i = 0; i < kNumVectors; ++i) {
      if (parts[i] == shard) ++label_counts[true_cluster(i)];
    }
    size_t majority = 0;
    size_t total = 0;
    for (size_t count : label_counts) {
      majority = std::max(majority, count);
      total += count;
    }
    minority += total - majority;
  }
  const double ratio = static_cast<double>(minority) / kNumVectors;
  std::cout << "kmeans cluster minority ratio: " << ratio << "\n";
  assert(ratio <= max_minority_ratio);
}

// Symmetric 10-NN graph restricted to each true cluster, so a spatially
// coherent partition keeps most edges intra-shard while the round-robin
// balanced layout cuts ~7/8 of them.
vec<vec<u32>> build_cluster_knn_graph(const Dataset& dataset) {
  constexpr uint32_t kNeighbors = 10;
  vec<vec<u32>> neighbors(kNumVectors);
  vec<float> query(kDim);
  for (uint32_t i = 0; i < kNumVectors; ++i) {
    const uint32_t lo = true_cluster(i) * kPointsPerCluster;
    const uint32_t hi = lo + kPointsPerCluster;
    dataset_decode_vector(dataset, i, query.data());
    vec<std::pair<float, u32>> distances;
    distances.reserve(hi - lo - 1);
    for (uint32_t j = lo; j < hi; ++j) {
      if (j == i) continue;
      distances.emplace_back(dataset_distance_float_query(dataset, query.data(), j), j);
    }
    const size_t keep = std::min<size_t>(kNeighbors, distances.size());
    std::partial_sort(distances.begin(), distances.begin() + keep, distances.end());
    neighbors[i].reserve(keep);
    for (size_t e = 0; e < keep; ++e) neighbors[i].push_back(distances[e].second);
  }
  return neighbors;
}

}  // namespace

int main() {
  const TemporaryDirectory temporary{
    std::filesystem::temp_directory_path() /
      ("dvstor_kmeans_partition_" + std::to_string(::getpid()))};
  std::filesystem::create_directories(temporary.path);
  const auto dataset_path = temporary.path / "base.fbin";
  write_synthetic_dataset(dataset_path);
  const Dataset dataset = load_dataset(dataset_path);
  assert(dataset.size() == kNumVectors);

  PartitionStats stats;
  vec<u32> parts = compute_kmeans_partition(
    dataset, kNumVectors, kNumShards,
    /*sample_size=*/kNumVectors, /*iters=*/10, &stats);
  assert(parts.size() == kNumVectors);
  assert(stats.kmeans_sample_size == kNumVectors);
  assert(stats.kmeans_iterations >= 1);
  assert(stats.kmeans_inertia > 0.0);
  check_exact_balance(stats.part_node_counts);
  check_cluster_coherence(parts, 0.01);

  // Exercise the sampling path (sample_size < n); balance stays exact.
  PartitionStats sampled_stats;
  vec<u32> sampled_parts = compute_kmeans_partition(
    dataset, kNumVectors, kNumShards,
    /*sample_size=*/2000, /*iters=*/10, &sampled_stats);
  assert(sampled_stats.kmeans_sample_size == 2000);
  check_exact_balance(sampled_stats.part_node_counts);
  check_cluster_coherence(sampled_parts, 0.05);

  // Degenerate single-shard case.
  vec<u32> single = compute_kmeans_partition(dataset, kNumVectors, 1, 100, 2);
  assert(std::all_of(single.begin(), single.end(),
                     [](u32 part) { return part == 0; }));

  const vec<vec<u32>> neighbors = build_cluster_knn_graph(dataset);
  const auto kmeans_placements =
    assign_nodes_to_shards_from_partition(parts, kNumShards, 64);
  const auto balanced_placements =
    assign_nodes_to_shards_balanced(kNumVectors, kNumShards, 64);
  const double kmeans_ratio = compute_cross_shard_ratio(neighbors, kmeans_placements);
  const double balanced_ratio = compute_cross_shard_ratio(neighbors, balanced_placements);
  std::cout << "cross-shard ratio: kmeans=" << kmeans_ratio
            << " balanced=" << balanced_ratio << "\n";
  assert(balanced_ratio > 0.5);
  assert(kmeans_ratio < 0.5);
  assert(kmeans_ratio < balanced_ratio * 0.6);

  std::cout << "kmeans partition test passed\n";
  return 0;
}
