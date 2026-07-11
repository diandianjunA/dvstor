#include "tools/vamana_offline/shard_writer.hh"

#include <algorithm>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

#include <library/utils.hh>

#include "common/index_path.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "vamana/idmap.hh"
#include "vamana/storage_layout_resolver.hh"
#include "vamana/vamana_node.hh"
#include "tools/vamana_offline/partitioning.hh"
#include "tools/vamana_offline/progress.hh"
#include "tools/vamana_offline/anchor_builder.hh"

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
  if (num_parts == 1) {
    if (stats != nullptr) stats->part_node_counts[0] = num_nodes;
    return parts;
  }

  vec<u32> nbrs;
  const u32 safe_start = (start_node < num_nodes) ? start_node : 0;

  // ── Step 1: Select seeds via farthest-point heuristic (k-center greedy) ──
  vec<u32> seeds;
  seeds.reserve(num_parts);
  seeds.push_back(safe_start);

  const u32 kUnreachable = std::numeric_limits<u32>::max();
  vec<u32> dist(num_nodes, kUnreachable);

  auto bfs_update_distances = [&](u32 source) {
    std::deque<u32> q;
    dist[source] = 0;
    q.push_back(source);
    while (!q.empty()) {
      const u32 node = q.front();
      q.pop_front();
      const u32 ndist = dist[node] + 1;
      graph.copy_neighbors(node, nbrs);
      for (u32 nbr : nbrs) {
        if (nbr < num_nodes && dist[nbr] > ndist) {
          dist[nbr] = ndist;
          q.push_back(nbr);
        }
      }
    }
  };

  // Initial BFS from medoid
  bfs_update_distances(safe_start);

  // Greedy farthest-point selection for remaining seeds
  while (seeds.size() < num_parts) {
    u32 farthest = 0;
    u32 max_dist = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
      if (dist[i] != kUnreachable && dist[i] > max_dist) {
        max_dist = dist[i];
        farthest = static_cast<u32>(i);
      }
    }
    seeds.push_back(farthest);
    // Incremental BFS from new seed — only updates nodes whose distance improves
    bfs_update_distances(farthest);
  }

  // ── Step 2: Multi-source BFS with load balancing ──
  const u32 kUnassigned = std::numeric_limits<u32>::max();
  std::fill(parts.begin(), parts.end(), kUnassigned);

  vec<byte_t> visited(num_nodes, 0);
  vec<u32> shard_size(num_parts, 0);
  const size_t target = num_nodes / num_parts;

  // Single FIFO queue: each entry is (node, preferred_shard)
  std::deque<std::pair<u32, u32>> queue;
  for (u32 s = 0; s < num_parts; ++s) {
    queue.emplace_back(seeds[s], s);
    visited[seeds[s]] = 1;
  }

  while (!queue.empty()) {
    const auto [node, pref_shard] = queue.front();
    queue.pop_front();

    if (parts[node] != kUnassigned) continue;

    // Load balance: if preferred shard is full, redirect to smallest shard
    u32 target_shard = pref_shard;
    if (shard_size[pref_shard] >= target) {
      target_shard = static_cast<u32>(
          std::min_element(shard_size.begin(), shard_size.end()) - shard_size.begin());
    }

    parts[node] = target_shard;
    ++shard_size[target_shard];

    // Tag neighbors with the assigned shard for territory continuity
    graph.copy_neighbors(node, nbrs);
    for (u32 nbr : nbrs) {
      if (nbr < num_nodes && !visited[nbr]) {
        visited[nbr] = 1;
        queue.emplace_back(nbr, target_shard);
      }
    }
  }

  // ── Step 3: Assign leftovers (isolated components never reached by BFS) ──
  for (size_t i = 0; i < num_nodes; ++i) {
    if (parts[i] == kUnassigned) {
      const u32 s = static_cast<u32>(
          std::min_element(shard_size.begin(), shard_size.end()) - shard_size.begin());
      parts[i] = s;
      ++shard_size[s];
    }
  }

  // ── Step 4: Compute stats ──
  if (stats != nullptr) {
    for (u32 s = 0; s < num_parts; ++s) {
      stats->part_node_counts[s] = shard_size[s];
    }
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
    std::cerr << "BFS partition (multi-source) edges: input=" << result.stats.input_edges
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
  VamanaNode::disable_hot_graph();
  const size_t node_size = VamanaNode::total_size();
  const size_t aligned_size = (node_size + 7) & ~7ULL;

  ProgressReporter progress{"Exporting Vamana shards", n + config.num_memory_nodes};

  PlacementResult placement_result = place_nodes(graph, config, aligned_size);
  print_partition_stats(config, placement_result);
  const auto& placements = placement_result.placements;
  write_anchor_sidecar(graph, dataset, config, placements, output_prefix);

  vec<u64> shard_sizes(config.num_memory_nodes, 16);
  vec<u64> shard_entry_counts(config.num_memory_nodes, 0);
  for (const auto& p : placements) {
    shard_sizes[p.memory_node] = std::max<u64>(shard_sizes[p.memory_node], p.offset + aligned_size);
    shard_entry_counts[p.memory_node] = std::max<u64>(
      shard_entry_counts[p.memory_node], (p.offset - 16) / aligned_size + 1);
  }

  const u32 hot_graph_shard_bits = vamana::hot_graph::shard_bits_for(config.num_memory_nodes);
  const u32 hot_graph_entry_size = static_cast<u32>(VamanaNode::hot_graph_entry_size());
  vec<u64> hot_graph_header_offsets(config.num_memory_nodes, 0);
  vec<u64> hot_graph_entry_offsets(config.num_memory_nodes, 0);
  vec<u64> hot_graph_dynamic_base_offsets(config.num_memory_nodes, 0);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    hot_graph_header_offsets[shard] = VamanaNode::align_storage(shard_sizes[shard]);
    hot_graph_entry_offsets[shard] =
      VamanaNode::align_storage(hot_graph_header_offsets[shard] + sizeof(vamana::hot_graph::Header));
    hot_graph_dynamic_base_offsets[shard] = VamanaNode::align_storage(
      hot_graph_entry_offsets[shard] + shard_entry_counts[shard] * hot_graph_entry_size);
    shard_sizes[shard] = hot_graph_dynamic_base_offsets[shard];
  }
  VamanaNode::configure_hot_graph(hot_graph_entry_offsets,
                                  shard_entry_counts,
                                  hot_graph_entry_size,
                                  hot_graph_shard_bits,
                                  hot_graph_dynamic_base_offsets,
                                  static_cast<u32>(VamanaNode::dynamic_record_size()),
                                  static_cast<u32>(VamanaNode::total_size()));

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
    vamana::hot_graph::Header header;
    header.version = vamana::hot_graph::kVersion2;
    header.entry_bytes = hot_graph_entry_size;
    header.max_degree = config.R;
    header.compact_pointer_shard_bits = hot_graph_shard_bits;
    header.entry_count = shard_entry_counts[shard];
    header.reserved0 = hot_graph_dynamic_base_offsets[shard];
    header.reserved1 = VamanaNode::allocation_size();
    header.reserved2 = static_cast<u32>(VamanaNode::total_size());
    shard_files[shard].seekp(static_cast<std::streamoff>(hot_graph_header_offsets[shard]));
    shard_files[shard].write(reinterpret_cast<const char*>(&header), sizeof(header));
  }

  const RemotePtr medoid_ptr{placements[graph.medoid].memory_node, placements[graph.medoid].offset};
  shard_files[0].seekp(8);
  shard_files[0].write(reinterpret_cast<const char*>(&medoid_ptr.raw_address), sizeof(u64));

  vec<byte_t> node_buf(aligned_size, 0);
  vec<byte_t> hot_graph_entry(hot_graph_entry_size, 0);
  vec<RemotePtr> hot_neighbors(config.R);
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
    *reinterpret_cast<u32*>(buf + VamanaNode::offset_generation()) = 0;

    std::memcpy(buf + VamanaNode::offset_vector(), dataset.raw_vector(i), dataset.vector_bytes);

    std::fill(hot_neighbors.begin(), hot_neighbors.end(), RemotePtr{});
    for (u8 j = 0; j < edge_count; ++j) {
      const u32 nbr = nbrs[j];
      RemotePtr nbr_ptr{placements[nbr].memory_node, placements[nbr].offset};
      hot_neighbors[j] = nbr_ptr;
    }

    const auto& placement = placements[i];
    auto& file = shard_files[placement.memory_node];
    file.seekp(static_cast<std::streamoff>(placement.offset));
    file.write(reinterpret_cast<const char*>(node_buf.data()), static_cast<std::streamsize>(node_buf.size()));
    lib_assert(file.good(), "failed to write output shard node");
    VamanaNode::encode_hot_graph_entry(hot_graph_entry.data(),
                                       edge_count,
                                       hot_neighbors.data(),
                                       edge_count,
                                       hot_graph_shard_bits,
                                       0);
    const u64 slot = (placement.offset - 16) / aligned_size;
    const u64 hot_offset = hot_graph_entry_offsets[placement.memory_node] +
      slot * hot_graph_entry_size;
    file.seekp(static_cast<std::streamoff>(hot_offset));
    file.write(reinterpret_cast<const char*>(hot_graph_entry.data()),
               static_cast<std::streamsize>(hot_graph_entry_size));
    lib_assert(file.good(), "failed to write compact hot graph entry");
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
    {"distance", "l2"},
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
    {"storage_format", VamanaNode::storage_format_name()},
    {"schema_version", 14},
    {"graph_hot_bytes", VamanaNode::graph_hot_bytes()},
    {"vector_offset", VamanaNode::offset_vector()},
    {"vector_storage_bytes", VamanaNode::vector_storage_bytes()},
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
    {"navigation_quantizer", ""},
    {"navigation_code_bytes", 0},
    {"pq_subquantizers", 0},
    {"pq_bits", 0},
    {"navigation_model_checksum", 0},
    {"navigation_format", ""},
    {"navigation_entry_points", 0},
    {"navigation_code_remote_offsets", vec<u64>{}},
    {"navigation_code_region_bytes", vec<u64>{}},
    {"navigation_code_materialization", ""},
    {"navigation_graph_source", "storage_compact_graph"},
    {"navigation_execution", ""},
  };
  metadata["hot_graph_neighbor_read_bytes"] = hot_graph_entry_size;
  metadata["hot_graph_neighbor_update_bytes"] = hot_graph_entry_size;
  metadata["hot_graph_entry_size"] = hot_graph_entry_size;
  metadata["hot_graph_pointer_bytes"] = vamana::hot_graph::kCompactPointerBytes;
  metadata["hot_graph_shard_bits"] = hot_graph_shard_bits;
  metadata["hot_graph_offsets"] = hot_graph_entry_offsets;
  metadata["hot_graph_header_offsets"] = hot_graph_header_offsets;
  metadata["hot_graph_entry_counts"] = shard_entry_counts;
  metadata["hot_graph_dynamic_base_offsets"] = hot_graph_dynamic_base_offsets;
  metadata["hot_graph_dynamic_record_bytes"] = VamanaNode::allocation_size();
  metadata["hot_graph_dynamic_hot_offset"] = VamanaNode::total_size();
  metadata["allocation_size"] = VamanaNode::allocation_size();
  metadata["idmap_format"] = "owner_sharded_v1";
  metadata["anchor_format"] = config.anchor_count_per_shard == 0 ? "" : "owner_anchor_v1";
  metadata["anchor_count_per_shard"] = config.anchor_count_per_shard;
  const filepath_t metadata_file = filepath_t(output_prefix.string() + ".meta.json");
  std::ofstream metadata_output(metadata_file);
  metadata_output << std::setw(2) << metadata << std::endl;

  {
    vec<vec<vamana::idmap::Entry>> owner_entries(config.num_memory_nodes);
    for (size_t i = 0; i < n; ++i) {
      const u32 owner = config.num_memory_nodes == 0
        ? 0
        : static_cast<u32>(dataset.id(i) % config.num_memory_nodes);
      owner_entries[owner].push_back(vamana::idmap::Entry{
        dataset.id(i),
        RemotePtr{placements[i].memory_node, placements[i].offset}.raw_address,
        0,
        0});
    }
    for (u32 owner = 0; owner < config.num_memory_nodes; ++owner) {
      const filepath_t idmap_path = index_path::owner_idmap_file(
        output_prefix, owner + 1, config.num_memory_nodes);
      std::ofstream idmap_output(idmap_path, std::ios::binary | std::ios::out | std::ios::trunc);
      lib_assert(idmap_output.good(), "failed to create idmap sidecar: " + idmap_path.string());
      vamana::idmap::Header header;
      header.owner_shard = owner;
      header.shard_count = config.num_memory_nodes;
      header.entry_count = owner_entries[owner].size();
      idmap_output.write(reinterpret_cast<const char*>(&header), sizeof(header));
      if (!owner_entries[owner].empty()) {
        idmap_output.write(reinterpret_cast<const char*>(owner_entries[owner].data()),
                           static_cast<std::streamsize>(
                             owner_entries[owner].size() * sizeof(vamana::idmap::Entry)));
      }
      lib_assert(idmap_output.good(), "failed to write idmap sidecar: " + idmap_path.string());
    }
  }
  progress.finish();
}


}  // namespace tools::vamana_offline
