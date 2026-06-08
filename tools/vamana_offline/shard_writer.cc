#include "tools/vamana_offline/shard_writer.hh"

#include <algorithm>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include <library/utils.hh>

#include "common/index_path.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"
#include "tools/vamana_offline/partitioning.hh"
#include "tools/vamana_offline/progress.hh"

namespace tools::vamana_offline {

vec<NodePlacement> assign_nodes_to_shards(size_t num_vectors, u32 num_memory_nodes) {
  const size_t node_size = VamanaNode::total_size();
  const size_t aligned_size = (node_size + 7) & ~7ULL;
  return assign_nodes_to_shards_balanced(num_vectors, num_memory_nodes, aligned_size);
}

namespace {

struct PlacementResult {
  vec<NodePlacement> placements;
  PartitionStats stats;
  double cross_shard_ratio{0.0};
};

size_t graph_input_edges(const VamanaGraph& graph) {
  size_t total = 0;
  for (size_t i = 0; i < graph.num_nodes; ++i) total += graph.degree(i);
  return total;
}

vec<u32> compute_bfs_partition_graph(const VamanaGraph& graph,
                                     u32 num_parts,
                                     u32 start_node,
                                     PartitionStats* stats) {
  if (num_parts == 0) {
    throw std::runtime_error("BFS partition part count must be > 0");
  }
  const size_t num_nodes = graph.num_nodes;
  vec<u32> parts(num_nodes, 0);
  if (stats != nullptr) {
    stats->input_edges = graph_input_edges(graph);
    stats->unique_edges = stats->input_edges;
    stats->edge_cut = 0;
    stats->partition_cross_shard_ratio = 0.0;
    stats->part_node_counts.assign(num_parts, 0);
  }
  if (num_nodes == 0) return parts;

  vec<byte_t> visited(num_nodes, 0);
  vec<u32> bfs_order;
  bfs_order.reserve(num_nodes);
  std::deque<u32> queue;
  vec<u32> nbrs;

  const auto push_component = [&](u32 seed) {
    if (seed >= num_nodes || visited[seed]) return;
    visited[seed] = 1;
    queue.push_back(seed);
    while (!queue.empty()) {
      const u32 node = queue.front();
      queue.pop_front();
      bfs_order.push_back(node);
      graph.copy_neighbors(node, nbrs);
      for (u32 neighbor : nbrs) {
        if (neighbor < num_nodes && !visited[neighbor]) {
          visited[neighbor] = 1;
          queue.push_back(neighbor);
        }
      }
    }
  };

  push_component(start_node);
  for (u32 node = 0; node < num_nodes; ++node) push_component(node);

  for (size_t order_idx = 0; order_idx < bfs_order.size(); ++order_idx) {
    u32 part = static_cast<u32>((order_idx * static_cast<size_t>(num_parts)) / num_nodes);
    if (part >= num_parts) part = num_parts - 1;
    parts[bfs_order[order_idx]] = part;
    if (stats != nullptr) ++stats->part_node_counts[part];
  }

  if (stats != nullptr) {
    size_t total_edges = 0;
    size_t cut_edges = 0;
    for (size_t node = 0; node < num_nodes; ++node) {
      graph.copy_neighbors(node, nbrs);
      for (u32 neighbor : nbrs) {
        if (neighbor >= num_nodes) continue;
        ++total_edges;
        if (parts[node] != parts[neighbor]) ++cut_edges;
      }
    }
    stats->edge_cut = cut_edges;
    stats->partition_cross_shard_ratio =
      total_edges == 0 ? 0.0 : static_cast<double>(cut_edges) / static_cast<double>(total_edges);
  }
  return parts;
}

double compute_cross_shard_ratio_graph(const VamanaGraph& graph, const vec<NodePlacement>& placements) {
  size_t total_edges = 0;
  size_t cross_edges = 0;
  vec<u32> nbrs;
  for (size_t i = 0; i < graph.num_nodes; ++i) {
    const u32 source_shard = placements[i].memory_node;
    graph.copy_neighbors(i, nbrs);
    for (u32 neighbor : nbrs) {
      if (neighbor >= placements.size()) continue;
      ++total_edges;
      if (placements[neighbor].memory_node != source_shard) ++cross_edges;
    }
  }
  return total_edges == 0 ? 0.0 : static_cast<double>(cross_edges) / static_cast<double>(total_edges);
}

PlacementResult place_nodes(const VamanaGraph& graph,
                            const VamanaBuildConfig& config,
                            size_t aligned_size) {
  PlacementResult result;
  if (config.partition_strategy == "metis") {
    vec<u64> edges;
    const size_t reserve_edges =
      std::min<size_t>(graph.num_nodes * static_cast<size_t>(config.partition_max_degree),
                       static_cast<size_t>(std::numeric_limits<u32>::max()));
    edges.reserve(reserve_edges);
    vec<u32> nbrs;
    for (size_t i = 0; i < graph.num_nodes; ++i) {
      graph.copy_neighbors(i, nbrs);
      append_partition_edges(static_cast<u32>(i), nbrs, config.partition_max_degree, edges);
    }
    PartitionOptions options;
    options.num_parts = config.num_memory_nodes;
    options.max_degree = config.partition_max_degree;
    options.imbalance = config.partition_imbalance;
    vec<u32> parts = compute_metis_partition(graph.num_nodes, edges, options, &result.stats);
    result.placements = assign_nodes_to_shards_from_partition(parts, config.num_memory_nodes, aligned_size);
  } else if (config.partition_strategy == "bfs") {
    const u32 start_node = graph.medoid < graph.num_nodes ? static_cast<u32>(graph.medoid) : 0;
    vec<u32> parts = compute_bfs_partition_graph(graph, config.num_memory_nodes, start_node, &result.stats);
    result.placements = assign_nodes_to_shards_from_partition(parts, config.num_memory_nodes, aligned_size);
  } else {
    result.placements = assign_nodes_to_shards_balanced(graph.num_nodes, config.num_memory_nodes, aligned_size);
    result.stats.part_node_counts.assign(config.num_memory_nodes, 0);
    for (const auto& placement : result.placements) ++result.stats.part_node_counts[placement.memory_node];
  }
  result.cross_shard_ratio = compute_cross_shard_ratio_graph(graph, result.placements);
  return result;
}

void print_partition_stats(const VamanaBuildConfig& config, const PlacementResult& result) {
  std::cerr << "partition strategy: " << config.partition_strategy
            << " max_degree=" << config.partition_max_degree
            << " imbalance=" << config.partition_imbalance << "\n";
  if (config.partition_strategy == "metis") {
    std::cerr << "METIS partition edges: input=" << result.stats.input_edges
              << " unique=" << result.stats.unique_edges
              << " edge_cut=" << result.stats.edge_cut
              << " partition_cut_ratio=" << result.stats.partition_cross_shard_ratio << "\n";
  } else if (config.partition_strategy == "bfs") {
    std::cerr << "BFS partition edges: input=" << result.stats.input_edges
              << " edge_cut=" << result.stats.edge_cut
              << " partition_cut_ratio=" << result.stats.partition_cross_shard_ratio << "\n";
  }
  std::cerr << "partition node counts:";
  for (size_t count : result.stats.part_node_counts) std::cerr << " " << count;
  std::cerr << "\nactive neighbor cross-shard ratio: " << result.cross_shard_ratio << "\n";
}

void create_sized_file(const filepath_t& path, u64 size) {
  std::ofstream output(path, std::ios::binary | std::ios::out | std::ios::trunc);
  lib_assert(output.good(), "failed to create output shard file: " + path.string());
  if (size > 0) {
    output.seekp(static_cast<std::streamoff>(size - 1));
    output.put(0);
  }
  lib_assert(output.good(), "failed to size output shard file: " + path.string());
}

}  // namespace

void write_vamana_shards(const VamanaGraph& graph,
                         const Dataset& dataset,
                         const VamanaBuildConfig& config,
                         const filepath_t& output_prefix) {
  const size_t n = dataset.size();
  const u32 dim = dataset.dim;
  const size_t node_size = VamanaNode::total_size();
  const size_t aligned_size = (node_size + 7) & ~7ULL;

  ProgressReporter progress{"Exporting Vamana shards", n + config.num_memory_nodes};

  PlacementResult placement_result = place_nodes(graph, config, aligned_size);
  print_partition_stats(config, placement_result);
  const auto& placements = placement_result.placements;

  vec<u64> shard_sizes(config.num_memory_nodes, 16);
  for (const auto& p : placements) {
    shard_sizes[p.memory_node] = std::max<u64>(shard_sizes[p.memory_node], p.offset + aligned_size);
  }

  const filepath_t output_dir = output_prefix.parent_path();
  if (!output_dir.empty()) std::filesystem::create_directories(output_dir);

  vec<filepath_t> shard_paths(config.num_memory_nodes);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    shard_paths[shard] = index_path::shard_file(output_prefix, shard + 1, config.num_memory_nodes);
    create_sized_file(shard_paths[shard], shard_sizes[shard]);
  }

  vec<std::fstream> shard_files(config.num_memory_nodes);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    shard_files[shard].open(shard_paths[shard], std::ios::binary | std::ios::in | std::ios::out);
    lib_assert(shard_files[shard].good(), "failed to open output shard file: " + shard_paths[shard].string());
    shard_files[shard].seekp(0);
    shard_files[shard].write(reinterpret_cast<const char*>(&shard_sizes[shard]), sizeof(u64));
  }

  const RemotePtr medoid_ptr{placements[graph.medoid].memory_node, placements[graph.medoid].offset};
  shard_files[0].seekp(8);
  shard_files[0].write(reinterpret_cast<const char*>(&medoid_ptr.raw_address), sizeof(u64));

  vec<byte_t> node_buf(aligned_size, 0);
  vec<u32> nbrs;
  for (size_t i = 0; i < n; ++i) {
    std::fill(node_buf.begin(), node_buf.end(), 0);
    byte_t* buf = node_buf.data();

    u64 header = 0;
    if (i == graph.medoid) header |= VamanaNode::HEADER_IS_MEDOID;
    *reinterpret_cast<u64*>(buf) = header;
    *reinterpret_cast<u32*>(buf + VamanaNode::HEADER_SIZE) = dataset.id(i);

    graph.copy_neighbors(i, nbrs);
    const u8 edge_count = static_cast<u8>(std::min<size_t>(nbrs.size(), config.R));
    *reinterpret_cast<u8*>(buf + VamanaNode::offset_edge_count()) = edge_count;

    std::memcpy(buf + VamanaNode::offset_vector(), dataset.raw_vector(i), dataset.vector_bytes);

    auto* neighbor_buf = reinterpret_cast<u64*>(buf + VamanaNode::offset_neighbors());
    for (u8 j = 0; j < edge_count; ++j) {
      const u32 nbr = nbrs[j];
      RemotePtr nbr_ptr{placements[nbr].memory_node, placements[nbr].offset};
      neighbor_buf[j] = nbr_ptr.raw_address;
    }

    const auto& placement = placements[i];
    auto& file = shard_files[placement.memory_node];
    file.seekp(static_cast<std::streamoff>(placement.offset));
    file.write(reinterpret_cast<const char*>(node_buf.data()), static_cast<std::streamsize>(node_buf.size()));
    lib_assert(file.good(), "failed to write output shard node");
    progress.increment();
  }

  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    shard_files[shard].flush();
    lib_assert(shard_files[shard].good(), "failed to flush output shard file: " + shard_paths[shard].string());
    progress.increment();
  }

  nlohmann::json metadata{
    {"data_file", dataset.source_file.string()},
    {"output_prefix", output_prefix.string()},
    {"distance", config.ip_distance ? "ip" : "l2"},
    {"num_vectors", n},
    {"dim", dim},
    {"R", config.R},
    {"beam_width", config.beam_width},
    {"beam_width_construction", config.beam_width},
    {"alpha", config.alpha},
    {"num_memory_nodes", config.num_memory_nodes},
    {"medoid", {{"memory_node", medoid_ptr.memory_node()}, {"offset", medoid_ptr.byte_offset()}}},
    {"node_size", node_size},
    {"node_layout", VamanaNode::layout_name()},
    {"schema_version", 3},
    {"offline_builder_version", 2},
    {"random_graph_seed_scope", "per_node"},
    {"vector_data_type", vector_dtype_name(dataset.dtype)},
    {"vector_component_size", vector_dtype_component_size(dataset.dtype)},
    {"vector_bytes", dataset.vector_bytes},
    {"partition_strategy", config.partition_strategy},
    {"partition_max_degree", config.partition_max_degree},
    {"partition_imbalance", config.partition_imbalance},
    {"partition_edge_cut", placement_result.stats.edge_cut},
    {"partition_cross_shard_ratio", placement_result.cross_shard_ratio},
  };

  const filepath_t metadata_file = filepath_t(output_prefix.string() + ".meta.json");
  std::ofstream metadata_output(metadata_file);
  metadata_output << std::setw(2) << metadata << std::endl;
  progress.finish();
}


}  // namespace tools::vamana_offline
