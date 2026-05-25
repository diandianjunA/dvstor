#pragma once

#include <cuda_runtime.h>
#include <mutex>

#include "gpu/gpu_kernel_launcher.hh"
#include "tools/vamana_offline/dataset_io.hh"

namespace tools::vamana_offline {

using DistFn = float (*)(const float*, const float*, u32);

struct VamanaGraph {
  size_t num_nodes{0};

static constexpr u32 GPU_BATCH_THRESHOLD = 16;

struct BuilderGpuContext {
  cudaStream_t stream{nullptr};

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
