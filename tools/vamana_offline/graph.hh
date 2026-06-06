#pragma once

#include <atomic>
#include <limits>
#include <memory>
#include <cuda_runtime.h>
#include <utility>

#include "gpu/gpu_kernel_launcher.hh"
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
  void set_neighbors(size_t node, const vec<u32>& new_neighbors);
  void lock_node(size_t node);
  void unlock_node(size_t node);
};

struct NodeLockGuard {
  VamanaGraph& graph;
  size_t node;
  NodeLockGuard(VamanaGraph& graph_, size_t node_) : graph(graph_), node(node_) { graph.lock_node(node); }
  ~NodeLockGuard() { graph.unlock_node(node); }
};

static constexpr u32 GPU_BATCH_THRESHOLD = 16;

struct BuilderGpuContext {
  cudaStream_t stream{nullptr};
  cudaEvent_t event{nullptr};
  u32* h_candidate_ids{nullptr};
  float* h_distances{nullptr};
  u32* d_candidate_ids{nullptr};
  float* d_distances{nullptr};
  const void* d_base_vectors{nullptr};
  u32 dim{0};
  u32 max_candidates{0};
  u32 dtype{0};

  void init(u32 dim_, u32 max_cand, VectorDType dtype_, const void* d_base_vectors_);
  void destroy();
  bool enabled() const { return stream != nullptr && d_base_vectors != nullptr; }
};

size_t compute_medoid(const Dataset& dataset, bool ip_distance);
vec<std::pair<float, u32>> beam_search(VamanaGraph& graph,
                                       const Dataset& dataset,
                                       u32 query_id,
                                       u32 beam_width,
                                       bool ip_distance,
                                       BuilderGpuContext* gpu_ctx = nullptr);
vec<std::pair<float, u32>> beam_search_float_query(VamanaGraph& graph,
                                                   const Dataset& dataset,
                                                   const float* query,
                                                   u32 beam_width,
                                                   bool ip_distance);
vec<u32> robust_prune(const Dataset& dataset,
                      u32 source,
                      const vec<std::pair<float, u32>>& sorted_candidates,
                      float alpha,
                      u32 R,
                      bool ip_distance);
void build_vamana_graph(VamanaGraph& graph,
                        const Dataset& dataset,
                        const VamanaBuildConfig& config,
                        BuilderGpuContext* gpu_contexts,
                        size_t num_gpu_contexts);

}  // namespace tools::vamana_offline
