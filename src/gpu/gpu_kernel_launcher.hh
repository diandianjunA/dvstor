#pragma once

/**
 * GPU kernel launcher declarations.
 * These are host-callable functions that stage data and launch CUDA kernels.
 * Compiled by the host compiler (not nvcc), linked against dvstor_gpu_kernels.
 */

#include <cstdint>
#include <cstddef>

// Forward-declare CUDA types to avoid including cuda_runtime.h in host code
struct CUstream_st;
struct CUevent_st;
typedef CUstream_st* cudaStream_t;
typedef CUevent_st* cudaEvent_t;
struct cublasContext;
typedef cublasContext* cublasHandle_t;

namespace gpu {

/**
 * Compute batch L2 squared distances between a query and N candidates.
 * Uses tile-based parallelism with uint4 coalesced loads (from Jasper).
 *
 * @param stream         CUDA stream for async execution
 * @param event          CUDA event to record after kernel completion
 * @param d_query        Device pointer to query vector (dim floats)
 * @param d_candidates   Device pointer to N candidate vectors (N * dim floats, contiguous)
 * @param d_distances    Device output pointer for N distances
 * @param n_candidates   Number of candidate vectors
 * @param dim            Vector dimension
 */
void launch_batch_l2_distances(cudaStream_t stream, cudaEvent_t event,
                               const float* d_query, const float* d_candidates,
                               float* d_distances,
                               uint32_t n_candidates, uint32_t dim);

/**
 * Compute batch RaBitQ approximate distances.
 * Uses precomputed query factors and quantized data vectors.
 *
 * @param stream         CUDA stream
 * @param event          CUDA event
 * @param d_rot_query    Device pointer to rotated query vector (dim floats)
 * @param d_query_factor Device pointer to RabitqQueryFactor struct
 * @param d_rabitq_vecs  Device pointer to N packed RaBitQ data vectors
 * @param d_distances    Device output for N distances
 * @param n_candidates   Number of candidates
 * @param dim            Vector dimension
 * @param bits_per_dim   RaBitQ bits per dimension (1, 2, 4, or 8)
 */
void launch_batch_rabitq_distances(cudaStream_t stream, cudaEvent_t event,
                                   const float* d_rot_query,
                                   const void* d_query_factor,
                                   const void* d_rabitq_vecs,
                                   float* d_distances,
                                   uint32_t n_candidates, uint32_t dim,
                                   uint32_t bits_per_dim);

void launch_batch_cached_rabitq_distances(cudaStream_t stream, cudaEvent_t event,
                                          const float* d_rot_query,
                                          const void* d_query_factor,
                                          const void* d_rabitq_base,
                                          const uint32_t* d_slot_ids,
                                          float* d_distances,
                                          uint32_t n_candidates, uint32_t dim,
                                          uint32_t bits_per_dim,
                                          uint32_t vec_stride);

void launch_gather_cached_rabitq(cudaStream_t stream,
                                 const void* d_rabitq_base,
                                 const uint32_t* d_slot_ids,
                                 void* d_out,
                                 uint32_t n_candidates,
                                 uint32_t vec_stride);

/**
 * GPU RobustPrune: select up to R neighbors from sorted candidates.
 * Alpha pruning: reject p* if alpha * dist(p_star, p_prime) <= dist(p, p_prime).
 *
 * @param stream           CUDA stream
 * @param event            CUDA event
 * @param d_source_vec     Device pointer to source vertex vector (dim floats)
 * @param d_candidate_vecs Device pointer to candidate vectors (n_candidates * dim floats)
 * @param d_candidate_dists Device pointer to distances from source to candidates (sorted ascending)
 * @param d_candidate_order Device pointer to candidate indices in sorted order, or nullptr for identity
 * @param n_candidates     Number of candidates
 * @param dim              Vector dimension
 * @param alpha            Diversity factor (typically 1.2)
 * @param R                Maximum out-degree
 * @param d_pruned_indices Device output: indices into candidate array of selected neighbors
 * @param d_pruned_count   Device output: number of selected neighbors (single uint32_t)
 */
void launch_robust_prune(cudaStream_t stream, cudaEvent_t event,
                         const float* d_source_vec,
                         const float* d_candidate_vecs,
                         const float* d_candidate_dists,
                         const uint32_t* d_candidate_order,
                         uint32_t n_candidates, uint32_t dim,
                         float alpha, uint32_t R,
                         uint32_t* d_pruned_indices, uint32_t* d_pruned_count);

/**
 * Quantize a single vector using RaBitQ.
 *
 * @param stream         CUDA stream
 * @param event          CUDA event
 * @param d_vector       Device pointer to input vector (dim floats)
 * @param d_rotation_mat Device pointer to rotation matrix P (dim*dim floats, column-major)
 * @param d_centroid     Device pointer to rotated centroid (dim floats)
 * @param d_output       Device output: packed RaBitQ data
 * @param dim            Vector dimension
 * @param bits_per_dim   Quantization bits per dimension
 * @param t_const        Pre-computed scaling constant
 */
void launch_rabitq_quantize_single(cudaStream_t stream, cudaEvent_t event,
                                   cublasHandle_t cublas_handle,
                                   const float* d_vector,
                                   const float* d_rotation_mat,
                                   const float* d_centroid,
                                   void* d_output,
                                   uint32_t dim, uint32_t bits_per_dim,
                                   double t_const);

/**
 * Prepare RaBitQ query factors for a single query vector.
 *
 * @param stream         CUDA stream
 * @param event          CUDA event
 * @param d_query        Device pointer to query vector (dim floats)
 * @param d_rotation_mat Device pointer to rotation matrix P
 * @param d_centroid     Device pointer to rotated centroid
 * @param d_rot_query    Device output: rotated query vector (dim floats)
 * @param d_query_factor Device output: RabitqQueryFactor struct
 * @param dim            Vector dimension
 * @param bits_per_dim   Quantization bits per dimension
 */
void launch_rabitq_query_prepare(cudaStream_t stream, cudaEvent_t event,
                                 cublasHandle_t cublas_handle,
                                 const float* d_query,
                                 const float* d_rotation_mat,
                                 const float* d_centroid,
                                 float* d_rot_query,
                                 void* d_query_factor,
                                 uint32_t dim, uint32_t bits_per_dim);

/**
 * Initialize GPU resources: set device, create cuBLAS handle.
 * Call once at startup.
 */
void gpu_init(int device_id);

/**
 * Cleanup GPU resources.
 */
void gpu_shutdown();

// =========================================================================
// GPU memory management utilities (usable from host code without cuda_runtime.h)
// =========================================================================

/** Allocate device memory (cudaMalloc). */
void* gpu_malloc(size_t bytes);

/** Free device memory (cudaFree). */
void gpu_free(void* ptr);

/** Allocate page-locked host memory (cudaMallocHost). */
void* gpu_malloc_host(size_t bytes);

/** Free page-locked host memory (cudaFreeHost). */
void gpu_free_host(void* ptr);

/** Create a CUDA stream. */
cudaStream_t gpu_stream_create();

/** Destroy a CUDA stream. */
void gpu_stream_destroy(cudaStream_t stream);

/** Create a CUDA event. */
cudaEvent_t gpu_event_create();

/** Destroy a CUDA event. */
void gpu_event_destroy(cudaEvent_t event);

/** Async host-to-device memcpy. */
void gpu_memcpy_h2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream);

/** Async device-to-host memcpy. */
void gpu_memcpy_d2h_async(void* dst, const void* src, size_t bytes, cudaStream_t stream);

/** Synchronize a CUDA stream. */
void gpu_stream_synchronize(cudaStream_t stream);

}  // namespace gpu
