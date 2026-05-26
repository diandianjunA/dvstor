#include "tools/vamana_offline/shard_writer.hh"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

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
    for (size_t i = 0; i < graph.neighbors.size(); ++i) {
      append_partition_edges(static_cast<u32>(i), graph.neighbors[i], config.partition_max_degree, edges);
    }
    PartitionOptions options;
    options.num_parts = config.num_memory_nodes;
    options.max_degree = config.partition_max_degree;
    options.imbalance = config.partition_imbalance;
    vec<u32> parts = compute_metis_partition(graph.num_nodes, edges, options, &result.stats);
    result.placements = assign_nodes_to_shards_from_partition(parts, config.num_memory_nodes, aligned_size);
  } else {
    result.placements = assign_nodes_to_shards_balanced(graph.num_nodes, config.num_memory_nodes, aligned_size);
    result.stats.part_node_counts.assign(config.num_memory_nodes, 0);
    for (const auto& placement : result.placements) {
      ++result.stats.part_node_counts[placement.memory_node];
    }
  }
  result.cross_shard_ratio = compute_cross_shard_ratio(graph.neighbors, result.placements);
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
  }
  std::cerr << "partition node counts:";
  for (size_t count : result.stats.part_node_counts) {
    std::cerr << " " << count;
  }
  std::cerr << "\nactive neighbor cross-shard ratio: " << result.cross_shard_ratio << "\n";
}

}  // namespace

void write_vamana_shards(const VamanaGraph& graph,
                         const Dataset& dataset,
                         const VamanaBuildConfig& config,
                         const RaBitQState& rabitq_state,
                         const vec<vec<byte_t>>& rabitq_data,
                         const filepath_t& output_prefix) {
  const size_t n = dataset.ids.size();
  const u32 dim = dataset.dim;
  const size_t node_size = VamanaNode::total_size();
  const size_t aligned_size = (node_size + 7) & ~7ULL;

  ProgressReporter progress{"Exporting Vamana shards", n + config.num_memory_nodes};

  PlacementResult placement_result = place_nodes(graph, config, aligned_size);
  print_partition_stats(config, placement_result);
  const auto& placements = placement_result.placements;

  // Compute shard sizes
  vec<u64> shard_sizes(config.num_memory_nodes, 16);
  for (const auto& p : placements) {
    shard_sizes[p.memory_node] = std::max<u64>(shard_sizes[p.memory_node], p.offset + aligned_size);
  }

  // Allocate shard buffers
  vec<vec<byte_t>> shard_buffers(config.num_memory_nodes);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    shard_buffers[shard].assign(shard_sizes[shard], 0);
    // Write free_ptr at offset 0 (points past last node)
    *reinterpret_cast<u64*>(shard_buffers[shard].data()) = shard_sizes[shard];
  }

  // Write medoid_ptr at offset 8 on shard 0
  const RemotePtr medoid_ptr{placements[graph.medoid].memory_node, placements[graph.medoid].offset};
  *reinterpret_cast<u64*>(shard_buffers[0].data() + 8) = medoid_ptr.raw_address;

  // Serialize each node
  for (size_t i = 0; i < n; ++i) {
    const auto& placement = placements[i];
    byte_t* buf = shard_buffers[placement.memory_node].data() + placement.offset;

    // Header (8B)
    u64 header = 0;
    if (i == graph.medoid) header |= VamanaNode::HEADER_IS_MEDOID;
    *reinterpret_cast<u64*>(buf) = header;

    // ID (4B)
    *reinterpret_cast<u32*>(buf + VamanaNode::HEADER_SIZE) = dataset.ids[i];

    // Edge count (1B)
    const u8 edge_count = static_cast<u8>(std::min<size_t>(graph.neighbors[i].size(), config.R));
    *reinterpret_cast<u8*>(buf + VamanaNode::offset_edge_count()) = edge_count;

    // Padding (3B) - already zeroed

    // Vector (dim * 4B)
    std::memcpy(buf + VamanaNode::offset_vector(),
                dataset.vector(i),
                dim * sizeof(float));

    // RaBitQ data
    std::memcpy(buf + VamanaNode::offset_rabitq(),
                rabitq_data[i].data(),
                rabitq_state.total_rabitq_bytes);

    // Neighbors (R * 8B) — write active + zero rest
    auto* neighbor_buf = reinterpret_cast<u64*>(buf + VamanaNode::offset_neighbors());
    for (u8 j = 0; j < edge_count; ++j) {
      const u32 nbr = graph.neighbors[i][j];
      RemotePtr nbr_ptr{placements[nbr].memory_node, placements[nbr].offset};
      neighbor_buf[j] = nbr_ptr.raw_address;
    }
    // Remaining slots already zeroed

    progress.increment();
  }

  // Write shard files
  const filepath_t output_dir = output_prefix.parent_path();
  if (!output_dir.empty()) {
    std::filesystem::create_directories(output_dir);
  }

  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    const filepath_t shard_file = index_path::shard_file(output_prefix, shard + 1, config.num_memory_nodes);
    std::ofstream output(shard_file, std::ios::binary | std::ios::out);
    lib_assert(output.good(), "failed to open output shard file: " + shard_file.string());
    output.write(reinterpret_cast<const char*>(shard_buffers[shard].data()),
                 static_cast<std::streamsize>(shard_buffers[shard].size()));
    lib_assert(output.good(), "failed to write output shard file: " + shard_file.string());
    progress.increment();
  }

  // Write rotation matrix to a separate file
  {
    const filepath_t rot_file = filepath_t(output_prefix.string() + ".rotation.bin");
    std::ofstream out(rot_file, std::ios::binary);
    lib_assert(out.good(), "failed to open rotation matrix file: " + rot_file.string());
    // Write dim, then column-major matrix data
    u32 d = dim;
    out.write(reinterpret_cast<const char*>(&d), sizeof(u32));
    out.write(reinterpret_cast<const char*>(rabitq_state.rotation_matrix.data()),
              static_cast<std::streamsize>(dim * dim * sizeof(float)));
    // Write rotated centroid
    out.write(reinterpret_cast<const char*>(rabitq_state.rotated_centroid.data()),
              static_cast<std::streamsize>(dim * sizeof(float)));
    // Write t_const
    double tc = rabitq_state.t_const;
    out.write(reinterpret_cast<const char*>(&tc), sizeof(double));
    lib_assert(out.good(), "failed to write rotation matrix file");
  }

  // Write metadata
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
    {"rabitq_bits", config.rabitq_bits},
    {"num_memory_nodes", config.num_memory_nodes},
    {"medoid", {{"memory_node", medoid_ptr.memory_node()}, {"offset", medoid_ptr.byte_offset()}}},
    {"node_size", node_size},
    {"node_layout", VamanaNode::layout_name()},
    {"rabitq_size", rabitq_state.total_rabitq_bytes},
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
