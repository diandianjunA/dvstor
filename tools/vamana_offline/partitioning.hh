#pragma once

#include <cstddef>
#include <string>

#include "common/types.hh"

namespace tools::vamana_offline {

struct NodePlacement {
  u32 memory_node{0};
  u64 offset{0};
};

struct PartitionOptions {
  u32 num_parts{1};
  u32 max_degree{16};
  double imbalance{1.03};
};

struct PartitionStats {
  size_t input_edges{0};
  size_t unique_edges{0};
  size_t edge_cut{0};
  double partition_cross_shard_ratio{0.0};
  vec<size_t> part_node_counts;
};

bool metis_partitioning_available();
u32 metis_index_bits();
str metis_unavailable_reason();
u64 pack_undirected_edge(u32 a, u32 b);
void append_partition_edges(u32 source, size_t num_nodes, const vec<u32>& neighbors,
                            u32 max_degree, vec<u64>& edges);
vec<u32> compute_metis_partition(size_t num_nodes,
                                 vec<u64>& edges,
                                 const PartitionOptions& options,
                                 PartitionStats* stats = nullptr);
vec<NodePlacement> assign_nodes_to_shards_balanced(size_t num_vectors,
                                                   u32 num_memory_nodes,
                                                   size_t aligned_node_size);
vec<u32> compute_bfs_partition(const vec<vec<u32>>& neighbors,
                               u32 num_parts,
                               u32 start_node,
                               PartitionStats* stats = nullptr);
vec<NodePlacement> assign_nodes_to_shards_from_partition(const vec<u32>& parts,
                                                         u32 num_memory_nodes,
                                                         size_t aligned_node_size);
double compute_cross_shard_ratio(const vec<vec<u32>>& neighbors,
                                 const vec<NodePlacement>& placements);

}  // namespace tools::vamana_offline
