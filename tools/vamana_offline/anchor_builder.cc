#include "tools/vamana_offline/anchor_builder.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>

#include <library/utils.hh>

#include "common/index_path.hh"
#include "vamana/anchor_index.hh"

namespace tools::vamana_offline {

namespace {

struct Sample {
  u64 priority{};
  u32 node{};

  bool operator<(const Sample& other) const { return priority < other.priority; }
};

u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

}  // namespace

void write_anchor_sidecar(const VamanaGraph& graph,
                          const Dataset& dataset,
                          const VamanaBuildConfig& config,
                          const vec<NodePlacement>& placements,
                          const filepath_t& output_prefix) {
  const u32 target = config.anchor_count_per_shard;
  if (target == 0) {
    return;
  }

  vec<std::priority_queue<Sample>> samples(config.num_memory_nodes);
  for (u32 node = 0; node < graph.num_nodes; ++node) {
    const u32 shard = placements[node].memory_node;
    const u64 priority = mix64(static_cast<u64>(dataset.id(node)) ^
                               (static_cast<u64>(config.seed) << 32));
    auto& heap = samples[shard];
    if (heap.size() < target) {
      heap.push(Sample{priority, node});
    } else if (priority < heap.top().priority) {
      heap.pop();
      heap.push(Sample{priority, node});
    }
  }

  vec<vec<u32>> selected(config.num_memory_nodes);
  u64 total = 0;
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    auto& heap = samples[shard];
    auto& nodes = selected[shard];
    nodes.reserve(heap.size());
    while (!heap.empty()) {
      nodes.push_back(heap.top().node);
      heap.pop();
    }
    std::sort(nodes.begin(), nodes.end());
    total += nodes.size();
  }

  const filepath_t path = index_path::anchor_file(output_prefix);
  const filepath_t parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream output(path, std::ios::binary | std::ios::out | std::ios::trunc);
  lib_assert(output.good(), "failed to create anchor sidecar: " + path.string());

  vamana::anchor::Header header;
  header.dim = dataset.dim;
  header.shard_count = config.num_memory_nodes;
  header.vector_dtype = static_cast<u32>(dataset.dtype);
  header.vector_bytes = static_cast<u32>(dataset.vector_bytes);
  header.anchors_per_shard = target;
  header.total_anchors = total;
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));

  vec<float> decoded(dataset.dim);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    const auto& nodes = selected[shard];
    vamana::anchor::ShardHeader shard_header{shard, static_cast<u32>(nodes.size())};
    output.write(reinterpret_cast<const char*>(&shard_header), sizeof(shard_header));

    vec<float> centroid(dataset.dim, 0.0f);
    for (u32 node : nodes) {
      dataset_decode_vector(dataset, node, decoded.data());
      for (u32 d = 0; d < dataset.dim; ++d) {
        centroid[d] += decoded[d];
      }
    }
    if (!nodes.empty()) {
      const float scale = 1.0f / static_cast<float>(nodes.size());
      for (float& value : centroid) {
        value *= scale;
      }
    }
    output.write(reinterpret_cast<const char*>(centroid.data()),
                 static_cast<std::streamsize>(centroid.size() * sizeof(float)));

    for (u32 node : nodes) {
      vamana::anchor::EntryHeader entry;
      entry.rptr_raw = RemotePtr{placements[node].memory_node, placements[node].offset}.raw_address;
      entry.id = dataset.id(node);
      entry.degree = static_cast<u16>(std::min<size_t>(graph.degree(node),
                                                       std::numeric_limits<u16>::max()));
      output.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
      output.write(reinterpret_cast<const char*>(dataset.raw_vector(node)),
                   static_cast<std::streamsize>(dataset.vector_bytes));
    }
  }
  lib_assert(output.good(), "failed to write anchor sidecar: " + path.string());
  std::cerr << "anchor sidecar: " << path << " anchors=" << total
            << " per_shard_target=" << target << "\n";
}

}  // namespace tools::vamana_offline
