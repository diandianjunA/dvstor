#include "tools/vamana_offline/partitioning.hh"

#include <algorithm>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>

#ifndef DVSTOR_HAVE_METIS
#define DVSTOR_HAVE_METIS 0
#endif

#if DVSTOR_HAVE_METIS
#define idx_t metis_idx_t
#define real_t metis_real_t
#include <metis.h>
#undef real_t
#undef idx_t
#endif

namespace tools::vamana_offline {
namespace {

size_t align_checked_idx(size_t value, const char* name) {
#if DVSTOR_HAVE_METIS
  if (value > static_cast<size_t>(std::numeric_limits<metis_idx_t>::max())) {
    throw std::runtime_error(str{name} + " exceeds METIS idx_t capacity; rebuild METIS with 64-bit idx_t");
  }
#else
  (void)name;
#endif
  return value;
}

#if DVSTOR_HAVE_METIS
u32 edge_u(u64 packed) {
  return static_cast<u32>(packed >> 32);
}

u32 edge_v(u64 packed) {
  return static_cast<u32>(packed & 0xffffffffu);
}

void fill_stats(size_t num_nodes, const vec<u32>& parts, const vec<u64>& edges, PartitionStats* stats) {
  if (stats == nullptr) {
    return;
  }
  stats->part_node_counts.assign(stats->part_node_counts.size(), 0);
  for (u32 part : parts) {
    if (part < stats->part_node_counts.size()) {
      ++stats->part_node_counts[part];
    }
  }
  size_t cut = 0;
  for (u64 edge : edges) {
    const u32 u = edge_u(edge);
    const u32 v = edge_v(edge);
    if (u < num_nodes && v < num_nodes && parts[u] != parts[v]) {
      ++cut;
    }
  }
  stats->edge_cut = cut;
  stats->partition_cross_shard_ratio =
    edges.empty() ? 0.0 : static_cast<double>(cut) / static_cast<double>(edges.size());
}
#endif

}  // namespace

bool metis_partitioning_available() {
  return DVSTOR_HAVE_METIS != 0;
}

u32 metis_index_bits() {
#if DVSTOR_HAVE_METIS
  return static_cast<u32>(sizeof(metis_idx_t) * 8);
#else
  return 0;
#endif
}

str metis_unavailable_reason() {
#if DVSTOR_HAVE_METIS
  return {};
#else
  return "METIS support is not built. Install libmetis-dev and configure with -DDVSTOR_METIS_PARTITION=ON.";
#endif
}

u64 pack_undirected_edge(u32 a, u32 b) {
  if (a == b) {
    return 0;
  }
  const u32 lo = std::min(a, b);
  const u32 hi = std::max(a, b);
  return (static_cast<u64>(lo) << 32) | static_cast<u64>(hi);
}

void append_partition_edges(u32 source, size_t num_nodes,
                            const vec<u32>& neighbors, u32 max_degree,
                            vec<u64>& edges) {
  if (source >= num_nodes) {
    throw std::runtime_error("partition edge source is outside graph");
  }
  const size_t limit = std::min<size_t>(neighbors.size(), max_degree);
  for (size_t i = 0; i < limit; ++i) {
    if (neighbors[i] >= num_nodes) {
      throw std::runtime_error("partition edge references node outside graph");
    }
    const u64 edge = pack_undirected_edge(source, neighbors[i]);
    if (edge != 0) {
      edges.push_back(edge);
    }
  }
}

vec<u32> compute_metis_partition(size_t num_nodes,
                                 vec<u64>& edges,
                                 const PartitionOptions& options,
                                 PartitionStats* stats) {
  if (options.num_parts == 0) {
    throw std::runtime_error("METIS partition part count must be > 0");
  }
  if (options.num_parts == 1 || num_nodes == 0) {
    vec<u32> parts(num_nodes, 0);
    if (stats != nullptr) {
      stats->input_edges = edges.size();
      stats->unique_edges = edges.size();
      stats->edge_cut = 0;
      stats->partition_cross_shard_ratio = 0.0;
      stats->part_node_counts.assign(options.num_parts, num_nodes);
    }
    return parts;
  }
  if (!metis_partitioning_available()) {
    throw std::runtime_error(metis_unavailable_reason());
  }

  if (stats != nullptr) {
    stats->input_edges = edges.size();
    stats->part_node_counts.assign(options.num_parts, 0);
  }

  edges.erase(std::remove(edges.begin(), edges.end(), 0), edges.end());
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  if (edges.size() > std::numeric_limits<size_t>::max() / 2) {
    throw std::runtime_error("partition adjacency entry count overflows");
  }

  if (stats != nullptr) {
    stats->unique_edges = edges.size();
  }

  align_checked_idx(num_nodes, "node count");
  align_checked_idx(edges.size() * 2, "adjacency entry count");

#if DVSTOR_HAVE_METIS
  vec<metis_idx_t> xadj(num_nodes + 1, 0);
  for (u64 edge : edges) {
    const u32 u = edge_u(edge);
    const u32 v = edge_v(edge);
    if (u >= num_nodes || v >= num_nodes) {
      throw std::runtime_error("partition edge references node outside graph");
    }
    ++xadj[static_cast<size_t>(u) + 1];
    ++xadj[static_cast<size_t>(v) + 1];
  }
  for (size_t i = 1; i < xadj.size(); ++i) {
    xadj[i] += xadj[i - 1];
  }

  vec<metis_idx_t> cursor = xadj;
  vec<metis_idx_t> adjncy(static_cast<size_t>(xadj.back()));
  for (u64 edge : edges) {
    const metis_idx_t u = static_cast<metis_idx_t>(edge_u(edge));
    const metis_idx_t v = static_cast<metis_idx_t>(edge_v(edge));
    adjncy[static_cast<size_t>(cursor[static_cast<size_t>(u)]++)] = v;
    adjncy[static_cast<size_t>(cursor[static_cast<size_t>(v)]++)] = u;
  }

  metis_idx_t nvtxs = static_cast<metis_idx_t>(num_nodes);
  metis_idx_t ncon = 1;
  metis_idx_t nparts = static_cast<metis_idx_t>(options.num_parts);
  metis_idx_t objval = 0;
  metis_real_t ubvec = static_cast<metis_real_t>(options.imbalance);
  vec<metis_idx_t> part(num_nodes, 0);
  metis_idx_t metis_options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(metis_options);
  metis_options[METIS_OPTION_NUMBERING] = 0;

  const int rc = METIS_PartGraphKway(&nvtxs,
                                     &ncon,
                                     xadj.data(),
                                     adjncy.data(),
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     &nparts,
                                     nullptr,
                                     &ubvec,
                                     metis_options,
                                     &objval,
                                     part.data());
  if (rc != METIS_OK) {
    throw std::runtime_error("METIS_PartGraphKway failed with code " + std::to_string(rc));
  }

  vec<u32> parts(num_nodes, 0);
  for (size_t i = 0; i < num_nodes; ++i) {
    parts[i] = static_cast<u32>(part[i]);
  }
  if (stats != nullptr) {
    stats->edge_cut = static_cast<size_t>(objval);
    fill_stats(num_nodes, parts, edges, stats);
  }
  return parts;
#else
  (void)num_nodes;
  (void)edges;
  (void)options;
  (void)stats;
  throw std::runtime_error(metis_unavailable_reason());
#endif
}

vec<NodePlacement> assign_nodes_to_shards_balanced(size_t num_vectors,
                                                   u32 num_memory_nodes,
                                                   size_t aligned_node_size) {
  if (num_memory_nodes == 0) {
    throw std::runtime_error("num_memory_nodes must be > 0");
  }
  vec<u64> shard_offsets(num_memory_nodes, 16);
  vec<NodePlacement> placements(num_vectors);

  for (size_t i = 0; i < num_vectors; ++i) {
    const auto min_it = std::min_element(shard_offsets.begin(), shard_offsets.end());
    const u32 shard = static_cast<u32>(std::distance(shard_offsets.begin(), min_it));
    placements[i] = {shard, *min_it};
    *min_it += aligned_node_size;
  }

  return placements;
}

vec<u32> compute_bfs_partition(const vec<vec<u32>>& neighbors,
                               u32 num_parts,
                               u32 start_node,
                               PartitionStats* stats) {
  if (num_parts == 0) {
    throw std::runtime_error("BFS partition part count must be > 0");
  }

  const size_t num_nodes = neighbors.size();
  vec<u32> parts(num_nodes, 0);
  if (stats != nullptr) {
    stats->input_edges = 0;
    stats->unique_edges = 0;
    stats->edge_cut = 0;
    stats->partition_cross_shard_ratio = 0.0;
    stats->part_node_counts.assign(num_parts, 0);
    for (const auto& nbrs : neighbors) {
      stats->input_edges += nbrs.size();
    }
    stats->unique_edges = stats->input_edges;
  }

  if (num_nodes == 0) return parts;
  if (num_parts == 1) {
    if (stats != nullptr) stats->part_node_counts[0] = num_nodes;
    return parts;
  }

  const u32 safe_start = (start_node < num_nodes) ? start_node : 0;

  // ── Step 1: Select seeds via farthest-point heuristic ──
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
      for (u32 nbr : neighbors[node]) {
        if (nbr < num_nodes && dist[nbr] > ndist) {
          dist[nbr] = ndist;
          q.push_back(nbr);
        }
      }
    }
  };

  bfs_update_distances(safe_start);

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
    bfs_update_distances(farthest);
  }

  // ── Step 2: Multi-source BFS with load balancing ──
  const u32 kUnassigned = std::numeric_limits<u32>::max();
  std::fill(parts.begin(), parts.end(), kUnassigned);

  vec<byte_t> visited(num_nodes, 0);
  vec<u32> shard_size(num_parts, 0);
  const size_t target = num_nodes / num_parts;

  std::deque<std::pair<u32, u32>> queue;
  for (u32 s = 0; s < num_parts; ++s) {
    queue.emplace_back(seeds[s], s);
    visited[seeds[s]] = 1;
  }

  while (!queue.empty()) {
    const auto [node, pref_shard] = queue.front();
    queue.pop_front();

    if (parts[node] != kUnassigned) continue;

    u32 target_shard = pref_shard;
    if (shard_size[pref_shard] >= target) {
      target_shard = static_cast<u32>(
          std::min_element(shard_size.begin(), shard_size.end()) - shard_size.begin());
    }

    parts[node] = target_shard;
    ++shard_size[target_shard];

    for (u32 nbr : neighbors[node]) {
      if (nbr < num_nodes && !visited[nbr]) {
        visited[nbr] = 1;
        queue.emplace_back(nbr, target_shard);
      }
    }
  }

  // ── Step 3: Assign leftovers (isolated components) ──
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
    for (size_t node = 0; node < neighbors.size(); ++node) {
      for (u32 neighbor : neighbors[node]) {
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

vec<NodePlacement> assign_nodes_to_shards_from_partition(const vec<u32>& parts,
                                                         u32 num_memory_nodes,
                                                         size_t aligned_node_size) {
  if (num_memory_nodes == 0) {
    throw std::runtime_error("num_memory_nodes must be > 0");
  }
  vec<u64> shard_offsets(num_memory_nodes, 16);
  vec<NodePlacement> placements(parts.size());
  for (size_t i = 0; i < parts.size(); ++i) {
    const u32 shard = parts[i];
    if (shard >= num_memory_nodes) {
      throw std::runtime_error("partition returned invalid shard id");
    }
    placements[i] = {shard, shard_offsets[shard]};
    shard_offsets[shard] += aligned_node_size;
  }
  return placements;
}

double compute_cross_shard_ratio(const vec<vec<u32>>& neighbors,
                                 const vec<NodePlacement>& placements) {
  size_t total_edges = 0;
  size_t cross_edges = 0;
  for (size_t i = 0; i < neighbors.size(); ++i) {
    const u32 source_shard = placements[i].memory_node;
    for (u32 neighbor : neighbors[i]) {
      if (neighbor >= placements.size()) {
        continue;
      }
      ++total_edges;
      if (placements[neighbor].memory_node != source_shard) {
        ++cross_edges;
      }
    }
  }
  return total_edges == 0 ? 0.0 : static_cast<double>(cross_edges) / static_cast<double>(total_edges);
}

}  // namespace tools::vamana_offline
