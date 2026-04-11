#include "gpu_kernel_launcher.hh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "kernels/distance_kernels.cuh"

namespace gpu {

static constexpr uint32_t TILE_SIZE = 4;
static constexpr uint32_t BLOCK_SIZE = 512;

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,          \
                    __LINE__, static_cast<int>(status));                        \
            abort();                                                           \
        }                                                                      \
    } while (0)

// =============================================================================
// Initialization / Shutdown
// =============================================================================

void gpu_init(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

void gpu_shutdown() {
}

// =============================================================================
// Batch L2 Squared Distance
// =============================================================================

void launch_batch_l2_distances(cudaStream_t stream, cudaEvent_t event,
                               const float* d_query, const float* d_candidates,
                               float* d_distances,
                               uint32_t n_candidates, uint32_t dim) {
    if (n_candidates == 0) {
        CUDA_CHECK(cudaEventRecord(event, stream));
        return;
    }

    uint32_t total_threads = n_candidates * TILE_SIZE;
    uint32_t num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gpu_kernels::batch_l2_squared_distance_kernel<TILE_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_query, d_candidates, d_distances, n_candidates, dim);

    CUDA_CHECK(cudaEventRecord(event, stream));
}

// =============================================================================
// Batch RaBitQ Distance
// =============================================================================

void launch_batch_rabitq_distances(cudaStream_t stream, cudaEvent_t event,
                                   const float* d_rot_query,
                                   const void* d_query_factor,
                                   const void* d_rabitq_vecs,
                                   float* d_distances,
                                   uint32_t n_candidates, uint32_t dim,
                                   uint32_t bits_per_dim) {
    if (n_candidates == 0) {
        CUDA_CHECK(cudaEventRecord(event, stream));
        return;
    }

    uint32_t total_threads = n_candidates * TILE_SIZE;
    uint32_t num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto* qfactor = static_cast<const gpu_kernels::RabitqQueryFactor*>(d_query_factor);
    auto* vecs = static_cast<const uint8_t*>(d_rabitq_vecs);

    if (bits_per_dim == 8) {
        gpu_kernels::batch_rabitq_8bit_distance_kernel<TILE_SIZE>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_rot_query, qfactor, vecs, d_distances, n_candidates, dim);
    } else if (bits_per_dim == 1) {
        gpu_kernels::batch_rabitq_distance_kernel<TILE_SIZE, 1>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_rot_query, qfactor, vecs, nullptr, nullptr,
                d_distances, n_candidates, dim);
    } else if (bits_per_dim == 2) {
        gpu_kernels::batch_rabitq_distance_kernel<TILE_SIZE, 2>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_rot_query, qfactor, vecs, nullptr, nullptr,
                d_distances, n_candidates, dim);
    } else if (bits_per_dim == 4) {
        gpu_kernels::batch_rabitq_distance_kernel<TILE_SIZE, 4>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                d_rot_query, qfactor, vecs, nullptr, nullptr,
                d_distances, n_candidates, dim);
    } else {
        fprintf(stderr, "Unsupported bits_per_dim: %u\n", bits_per_dim);
        abort();
    }

    CUDA_CHECK(cudaEventRecord(event, stream));
}

// =============================================================================
// RobustPrune
// =============================================================================

void launch_robust_prune(cudaStream_t stream, cudaEvent_t event,
                         const float* d_source_vec,
                         const float* d_candidate_vecs,
                         const float* d_candidate_dists,
                         const uint32_t* d_candidate_order,
                         uint32_t n_candidates, uint32_t dim,
                         float alpha, uint32_t R,
                         uint32_t* d_pruned_indices, uint32_t* d_pruned_count) {
    if (n_candidates == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_pruned_count, 0, sizeof(uint32_t), stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        return;
    }

    // Single block kernel; shared memory for bool validity array
    size_t smem_size = n_candidates * sizeof(bool);
    uint32_t block_size = std::min(BLOCK_SIZE, n_candidates);

    gpu_kernels::robust_prune_kernel
        <<<1, block_size, smem_size, stream>>>(
            d_source_vec, d_candidate_vecs, d_candidate_dists, d_candidate_order,
            n_candidates, dim, alpha, R,
            d_pruned_indices, d_pruned_count);

    CUDA_CHECK(cudaEventRecord(event, stream));
}

// =============================================================================
// RaBitQ Quantize Single Vector
// =============================================================================

void launch_rabitq_quantize_single(cudaStream_t stream, cudaEvent_t event,
                                   cublasHandle_t cublas_handle,
                                   const float* d_vector,
                                   const float* d_rotation_mat,
                                   const float* d_centroid,
                                   void* d_output,
                                   uint32_t dim, uint32_t bits_per_dim,
                                   double t_const) {
    // Step 1: Rotate vector using cuBLAS (P^T * v)
    // d_rotation_mat is column-major P (dim x dim), so P^T * v = sgemv with transpose
    float* d_rot_vec = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_rot_vec, dim * sizeof(float), stream));

    cublasSetStream(cublas_handle, stream);
    float alpha_blas = 1.0f, beta_blas = 0.0f;
    CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_T,
                             dim, dim,
                             &alpha_blas,
                             d_rotation_mat, dim,
                             d_vector, 1,
                             &beta_blas,
                             d_rot_vec, 1));

    // Step 2: Quantize rotated vector
    uint32_t packed_bytes = (bits_per_dim * dim + 7) / 8;
    auto* output_bytes = static_cast<uint8_t*>(d_output);
    auto* output_add = reinterpret_cast<float*>(output_bytes + packed_bytes);
    auto* output_rescale = reinterpret_cast<float*>(output_bytes + packed_bytes + sizeof(float));

    // Shared memory for uncompressed codes
    size_t smem_size = dim * sizeof(uint8_t);

    if (bits_per_dim == 1) {
        gpu_kernels::rabitq_quantize_single_kernel<TILE_SIZE, 1>
            <<<1, TILE_SIZE, smem_size, stream>>>(
                d_rot_vec, d_centroid, output_bytes, output_add, output_rescale,
                dim, t_const);
    } else if (bits_per_dim == 2) {
        gpu_kernels::rabitq_quantize_single_kernel<TILE_SIZE, 2>
            <<<1, TILE_SIZE, smem_size, stream>>>(
                d_rot_vec, d_centroid, output_bytes, output_add, output_rescale,
                dim, t_const);
    } else if (bits_per_dim == 4) {
        gpu_kernels::rabitq_quantize_single_kernel<TILE_SIZE, 4>
            <<<1, TILE_SIZE, smem_size, stream>>>(
                d_rot_vec, d_centroid, output_bytes, output_add, output_rescale,
                dim, t_const);
    } else if (bits_per_dim == 8) {
        gpu_kernels::rabitq_quantize_single_kernel<TILE_SIZE, 8>
            <<<1, TILE_SIZE, smem_size, stream>>>(
                d_rot_vec, d_centroid, output_bytes, output_add, output_rescale,
                dim, t_const);
    } else {
        fprintf(stderr, "Unsupported bits_per_dim: %u\n", bits_per_dim);
        abort();
    }

    CUDA_CHECK(cudaFreeAsync(d_rot_vec, stream));
    CUDA_CHECK(cudaEventRecord(event, stream));
}

// =============================================================================
// RaBitQ Query Prepare
// =============================================================================

void launch_rabitq_query_prepare(cudaStream_t stream, cudaEvent_t event,
                                 cublasHandle_t cublas_handle,
                                 const float* d_query,
                                 const float* d_rotation_mat,
                                 const float* d_centroid,
                                 float* d_rot_query,
                                 void* d_query_factor,
                                 uint32_t dim, uint32_t bits_per_dim) {
    // Step 1: Rotate query using cuBLAS (P^T * q)
    cublasSetStream(cublas_handle, stream);
    float alpha_blas = 1.0f, beta_blas = 0.0f;
    CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_T,
                             dim, dim,
                             &alpha_blas,
                             d_rotation_mat, dim,
                             d_query, 1,
                             &beta_blas,
                             d_rot_query, 1));

    // Step 2: Compute query factors
    auto* factor = static_cast<gpu_kernels::RabitqQueryFactor*>(d_query_factor);

    if (bits_per_dim == 1) {
        gpu_kernels::rabitq_query_prepare_kernel<TILE_SIZE, 1>
            <<<1, TILE_SIZE, 0, stream>>>(d_rot_query, d_centroid, factor, dim);
    } else if (bits_per_dim == 2) {
        gpu_kernels::rabitq_query_prepare_kernel<TILE_SIZE, 2>
            <<<1, TILE_SIZE, 0, stream>>>(d_rot_query, d_centroid, factor, dim);
    } else if (bits_per_dim == 4) {
        gpu_kernels::rabitq_query_prepare_kernel<TILE_SIZE, 4>
            <<<1, TILE_SIZE, 0, stream>>>(d_rot_query, d_centroid, factor, dim);
    } else if (bits_per_dim == 8) {
        gpu_kernels::rabitq_query_prepare_kernel<TILE_SIZE, 8>
            <<<1, TILE_SIZE, 0, stream>>>(d_rot_query, d_centroid, factor, dim);
    } else {
        fprintf(stderr, "Unsupported bits_per_dim: %u\n", bits_per_dim);
        abort();
    }

    CUDA_CHECK(cudaEventRecord(event, stream));
}

// =============================================================================
// GPU Memory Management Utilities
// =============================================================================

void* gpu_malloc(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void gpu_free(void* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

void* gpu_malloc_host(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    return ptr;
}

void gpu_free_host(void* ptr) {
    if (ptr) CUDA_CHECK(cudaFreeHost(ptr));
}

cudaStream_t gpu_stream_create() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return stream;
}

void gpu_stream_destroy(cudaStream_t stream) {
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
}

cudaEvent_t gpu_event_create() {
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

void gpu_event_destroy(cudaEvent_t event) {
    if (event) CUDA_CHECK(cudaEventDestroy(event));
}

void gpu_memcpy_h2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
}

void gpu_memcpy_d2h_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
}

void gpu_stream_synchronize(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace gpu
