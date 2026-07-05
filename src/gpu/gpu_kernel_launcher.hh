#pragma once

/**
 * GPU kernel launcher declarations.
 * These are host-callable functions that stage data and launch CUDA kernels.
 * Compiled by the host compiler (not nvcc), linked against dvstor_gpu_kernels.
 */

#include <cstdint>
#include <cstddef>

struct CUstream_st;
struct CUevent_st;
typedef CUstream_st* cudaStream_t;
typedef CUevent_st* cudaEvent_t;

namespace gpu {

void launch_batch_l2_distances(cudaStream_t stream, cudaEvent_t event,
                               const float* d_query, const float* d_candidates,
                               float* d_distances,
                               uint32_t n_candidates, uint32_t dim);

void launch_batch_typed_l2_distances(cudaStream_t stream, cudaEvent_t event,
                                     const float* d_query, const void* d_candidates,
                                     uint32_t candidate_dtype,
                                     float* d_distances,
                                     uint32_t n_candidates, uint32_t dim);

void launch_batch_typed_query_l2_distances(cudaStream_t stream, cudaEvent_t event,
                                           const void* d_query,
                                           uint32_t query_dtype,
                                           const void* d_candidates,
                                           uint32_t candidate_dtype,
                                           float* d_distances,
                                           uint32_t n_candidates, uint32_t dim);

void launch_batch_typed_multi_query_l2_distances(cudaStream_t stream, cudaEvent_t event,
                                                 const void* d_queries,
                                                 uint32_t query_dtype,
                                                 const uint32_t* d_candidate_query_ids,
                                                 const void* d_candidates,
                                                 uint32_t candidate_dtype,
                                                 float* d_distances,
                                                 uint32_t n_candidates,
                                                 uint32_t dim);

void launch_batch_id_l2_distances(cudaStream_t stream, cudaEvent_t event,
                                  const void* d_base_vectors,
                                  uint32_t query_id,
                                  const uint32_t* d_candidate_ids,
                                  float* d_distances,
                                  uint32_t n_candidates, uint32_t dim,
                                  uint32_t dtype);

void launch_batch_typed_query_l2_distances_indirect(cudaStream_t stream, cudaEvent_t event,
                                                    const void* d_query,
                                                    uint32_t query_dtype,
                                                    const void* const* d_candidate_ptrs,
                                                    uint32_t candidate_dtype,
                                                    float* d_distances,
                                                    uint32_t n_candidates, uint32_t dim);

void launch_robust_prune(cudaStream_t stream, cudaEvent_t event,
                         const float* d_source_vec,
                         const float* d_candidate_vecs,
                         const float* d_candidate_dists,
                         const uint32_t* d_candidate_order,
                         uint32_t n_candidates, uint32_t dim,
                         float alpha, uint32_t R,
                         uint32_t* d_pruned_indices, uint32_t* d_pruned_count);

void launch_robust_prune_typed(cudaStream_t stream, cudaEvent_t event,
                               const void* d_candidate_vecs,
                               uint32_t candidate_dtype,
                               const float* d_candidate_dists,
                               const uint32_t* d_candidate_order,
                               uint32_t n_candidates, uint32_t dim,
                               float alpha, uint32_t R,
                               uint32_t* d_pruned_indices, uint32_t* d_pruned_count);

void launch_batch_rabitq_asymmetric_distances(cudaStream_t stream, cudaEvent_t event,
                                    const float* d_rotated_query,
                                    const uint8_t* d_candidate_data,
                                    float* d_distances,
                                    float query_norm2, uint32_t n_candidates,
                                    uint32_t code_bits, uint32_t code_bytes,
                                    uint32_t entry_bytes);

void gpu_init(int device_id);
void gpu_shutdown();

void* gpu_malloc(size_t bytes);
void gpu_free(void* ptr);
void* gpu_malloc_host(size_t bytes);
void gpu_free_host(void* ptr);
cudaStream_t gpu_stream_create();
void gpu_stream_destroy(cudaStream_t stream);
cudaEvent_t gpu_event_create();
void gpu_event_destroy(cudaEvent_t event);
void gpu_memcpy_h2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream);
void gpu_memcpy_d2h_async(void* dst, const void* src, size_t bytes, cudaStream_t stream);
void gpu_stream_synchronize(cudaStream_t stream);

}  // namespace gpu
