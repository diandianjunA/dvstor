#pragma once

#include <atomic>
#include <limits>
#include <memory>
#include <utility>

#include "tools/vamana_offline/dataset_io.hh"

namespace tools::vamana_offline {

struct VamanaGraph {
  static constexpr u32 kEmptyNeighbor = std::numeric_limits<u32>::max();

  size_t num_nodes{0};
  u32 dim{0};
  u32 R{0};
  size_t medoid{0};
  vec<u32> neighbors;
  vec<u8> degrees;
  std::unique_ptr<std::atomic_flag[]> lock_stripes;
  size_t lock_stripe_count{0};

  void init(size_t n, u32 d, u32 max_degree, size_t requested_lock_stripes = 1 << 20);
  size_t offset(size_t node) const { return node * static_cast<size_t>(R); }
  u8 degree(size_t node) const { return degrees[node]; }
  void copy_neighbors(size_t node, vec<u32>& out) const;
  bool contains_neighbor_unlocked(size_t node, u32 neighbor) const;
  bool try_append_neighbor_unlocked(size_t node, u32 neighbor);
  void set_neighbors(size_t node, const vec<u32>& new_neighbors);
  void lock_node(size_t node);
  void unlock_node(size_t node);
};

struct NodeLockGuard {
  VamanaGraph& graph;
  size_t node;
  NodeLockGuard(VamanaGraph& graph_, size_t node_) : graph(graph_), node(node_) { graph.lock_node(node); }
  ~NodeLockGuard() { graph.unlock_node(node); }
  NodeLockGuard(const NodeLockGuard&) = delete;
  NodeLockGuard& operator=(const NodeLockGuard&) = delete;
  NodeLockGuard(NodeLockGuard&&) = delete;
  NodeLockGuard& operator=(NodeLockGuard&&) = delete;
};

size_t compute_medoid(const Dataset& dataset);
vec<std::pair<float, u32>> beam_search(VamanaGraph& graph,
                                       const Dataset& dataset,
                                       u32 query_id,
                                       u32 beam_width);
vec<std::pair<float, u32>> beam_search_float_query(VamanaGraph& graph,
                                                   const Dataset& dataset,
                                                   const float* query,
                                                   u32 beam_width);
vec<u32> robust_prune(const Dataset& dataset,
                      u32 source,
                      const vec<std::pair<float, u32>>& sorted_candidates,
                      float alpha,
                      u32 R);
void build_vamana_graph(VamanaGraph& graph,
                        const Dataset& dataset,
                        const VamanaBuildConfig& config);

}  // namespace tools::vamana_offline
