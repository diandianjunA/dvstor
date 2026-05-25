#pragma once

#include <cuda_runtime.h>
#include <mutex>
#include <utility>

#include "gpu/gpu_kernel_launcher.hh"
#include "tools/vamana_offline/dataset_io.hh"

namespace tools::vamana_offline {

using DistFn = float (*)(const float*, const float*, u32);

struct VamanaGraph {
  size_t num_nodes{0};
  u32 dim{0};
  u32 R{0};
  size_t medoid{0};
  vec<vec<u32>> neighbors;
  vec<std::mutex> node_locks;

  void init(size_t n, u32 d, u32 max_degree) {
    num_nodes = n;
    dim = d;
    R = max_degree;
    neighbors.resize(n);
    node_locks = vec<std::mutex>(n);
  }
};

static constexpr u32 GPU_BATCH_THRESHOLD = 16;

struct BuilderGpuContext {
  cudaStream_t stream{nullptr};
  cudaEvent_t event{nullptr};
  float* h_query{nullptr};
  float* h_candidates{nullptr};
  float* h_distances{nullptr};
  float* h_candidate_dists{nullptr};
  u32* h_pruned_indices{nullptr};
  u32* h_pruned_count{nullptr};
  float* d_query{nullptr};
  float* d_candidates{nullptr};
  float* d_distances{nullptr};
  float* d_candidate_dists{nullptr};
  u32* d_pruned_indices{nullptr};
  u32* d_pruned_count{nullptr};
  u32 dim{0};
  u32 max_candidates{0};
  u32 max_R{0};

  void init(u32 dim_, u32 max_cand, u32 R) {
    dim = dim_;
    max_candidates = max_cand;
    max_R = R;

    stream = gpu::gpu_stream_create();
    event = gpu::gpu_event_create();

    h_query = static_cast<float*>(gpu::gpu_malloc_host(dim * sizeof(float)));
    h_candidates = static_cast<float*>(gpu::gpu_malloc_host(max_cand * dim * sizeof(float)));
    h_distances = static_cast<float*>(gpu::gpu_malloc_host(max_cand * sizeof(float)));
    h_candidate_dists = static_cast<float*>(gpu::gpu_malloc_host(max_cand * sizeof(float)));
    h_pruned_indices = static_cast<u32*>(gpu::gpu_malloc_host(R * sizeof(u32)));
    h_pruned_count = static_cast<u32*>(gpu::gpu_malloc_host(sizeof(u32)));

    d_query = static_cast<float*>(gpu::gpu_malloc(dim * sizeof(float)));
    d_candidates = static_cast<float*>(gpu::gpu_malloc(max_cand * dim * sizeof(float)));
    d_distances = static_cast<float*>(gpu::gpu_malloc(max_cand * sizeof(float)));
    d_candidate_dists = static_cast<float*>(gpu::gpu_malloc(max_cand * sizeof(float)));
    d_pruned_indices = static_cast<u32*>(gpu::gpu_malloc(R * sizeof(u32)));
    d_pruned_count = static_cast<u32*>(gpu::gpu_malloc(sizeof(u32)));
  }

  void destroy() {
    if (!stream) {
      return;
    }
    gpu::gpu_free_host(h_query);
    gpu::gpu_free_host(h_candidates);
    gpu::gpu_free_host(h_distances);
    gpu::gpu_free_host(h_candidate_dists);
    gpu::gpu_free_host(h_pruned_indices);
    gpu::gpu_free_host(h_pruned_count);
    gpu::gpu_free(d_query);
    gpu::gpu_free(d_candidates);
    gpu::gpu_free(d_distances);
    gpu::gpu_free(d_candidate_dists);
    gpu::gpu_free(d_pruned_indices);
    gpu::gpu_free(d_pruned_count);
    gpu::gpu_event_destroy(event);
    gpu::gpu_stream_destroy(stream);
    stream = nullptr;
  }
};

float l2_squared(const float* a, const float* b, u32 dim);
float ip_distance(const float* a, const float* b, u32 dim);
size_t compute_medoid(const Dataset& dataset, DistFn dist_fn);
vec<std::pair<float, u32>> beam_search(VamanaGraph& graph,
                                       const Dataset& dataset,
                                       const float* query,
                                       u32 beam_width,
                                       DistFn dist_fn,
                                       BuilderGpuContext* gpu_ctx = nullptr);
vec<u32> robust_prune(const Dataset& dataset,
                      u32 source,
                      const vec<std::pair<float, u32>>& sorted_candidates,
                      float alpha,
                      u32 R,
                      DistFn dist_fn);
void build_vamana_graph(VamanaGraph& graph,
                        const Dataset& dataset,
                        const VamanaBuildConfig& config,
                        DistFn dist_fn,
                        BuilderGpuContext* gpu_contexts,
                        size_t num_gpu_contexts);

}  // namespace tools::vamana_offline
