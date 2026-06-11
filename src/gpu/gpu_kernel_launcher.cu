#include "gpu_kernel_launcher.hh"

#include <cuda_runtime.h>
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

void gpu_init(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

void gpu_shutdown() {
}

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


namespace {

template <typename T>
void launch_id_distance_typed(cudaStream_t stream,
                              const void* d_base_vectors,
                              uint32_t query_id,
                              const uint32_t* d_candidate_ids,
                              float* d_distances,
                              uint32_t n_candidates,
                              uint32_t dim) {
    uint32_t total_threads = n_candidates * TILE_SIZE;
    uint32_t num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_kernels::batch_l2_id_distance_kernel<TILE_SIZE, T>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<const T*>(d_base_vectors),
            query_id,
            d_candidate_ids,
            d_distances,
            n_candidates,
            dim);
}

}  // namespace

void launch_batch_id_l2_distances(cudaStream_t stream, cudaEvent_t event,
                                  const void* d_base_vectors,
                                  uint32_t query_id,
                                  const uint32_t* d_candidate_ids,
                                  float* d_distances,
                                  uint32_t n_candidates, uint32_t dim,
                                  uint32_t dtype) {
    if (n_candidates == 0) {
        CUDA_CHECK(cudaEventRecord(event, stream));
        return;
    }
    if (dtype == 0) {
        launch_id_distance_typed<float>(stream, d_base_vectors, query_id, d_candidate_ids,
                                        d_distances, n_candidates, dim);
    } else if (dtype == 1) {
        launch_id_distance_typed<uint8_t>(stream, d_base_vectors, query_id, d_candidate_ids,
                                          d_distances, n_candidates, dim);
    } else if (dtype == 2) {
        launch_id_distance_typed<int8_t>(stream, d_base_vectors, query_id, d_candidate_ids,
                                         d_distances, n_candidates, dim);
    } else {
        fprintf(stderr, "Unsupported id-distance dtype: %u\n", dtype);
        abort();
    }
    CUDA_CHECK(cudaEventRecord(event, stream));
}

void launch_batch_typed_l2_distances(cudaStream_t stream, cudaEvent_t event,
                                     const float* d_query, const void* d_candidates,
                                     uint32_t candidate_dtype,
                                     float* d_distances,
                                     uint32_t n_candidates, uint32_t dim) {
    launch_batch_typed_query_l2_distances(stream, event, d_query, 0, d_candidates, candidate_dtype,
                                          d_distances, n_candidates, dim);
}

namespace {

template <typename QueryT, typename CandidateT>
void launch_typed_pair_distance(cudaStream_t stream,
                                const void* d_query,
                                const void* d_candidates,
                                float* d_distances,
                                uint32_t n_candidates,
                                uint32_t dim) {
    uint32_t total_threads = n_candidates * TILE_SIZE;
    uint32_t num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_kernels::batch_l2_typed_pair_distance_kernel<TILE_SIZE, QueryT, CandidateT>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<const QueryT*>(d_query),
            static_cast<const CandidateT*>(d_candidates),
            d_distances, n_candidates, dim);
}


template <typename QueryT, typename CandidateT>
void launch_typed_pair_distance_indirect(cudaStream_t stream,
                                         const void* d_query,
                                         const void* const* d_candidate_ptrs,
                                         float* d_distances,
                                         uint32_t n_candidates,
                                         uint32_t dim) {
    uint32_t total_threads = n_candidates * TILE_SIZE;
    uint32_t num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_kernels::batch_l2_typed_pair_distance_indirect_kernel<TILE_SIZE, QueryT, CandidateT>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<const QueryT*>(d_query),
            d_candidate_ptrs,
            d_distances, n_candidates, dim);
}

}  // namespace

void launch_batch_typed_query_l2_distances(cudaStream_t stream, cudaEvent_t event,
                                           const void* d_query,
                                           uint32_t query_dtype,
                                           const void* d_candidates,
                                           uint32_t candidate_dtype,
                                           float* d_distances,
                                           uint32_t n_candidates, uint32_t dim) {
    if (query_dtype == 0 && candidate_dtype == 0) {
        launch_batch_l2_distances(stream, event, static_cast<const float*>(d_query),
                                  static_cast<const float*>(d_candidates),
                                  d_distances, n_candidates, dim);
        return;
    }
    if (n_candidates == 0) {
        CUDA_CHECK(cudaEventRecord(event, stream));
        return;
    }

    if (query_dtype == 0 && candidate_dtype == 1) {
        launch_typed_pair_distance<float, uint8_t>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 0 && candidate_dtype == 2) {
        launch_typed_pair_distance<float, int8_t>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 1 && candidate_dtype == 0) {
        launch_typed_pair_distance<uint8_t, float>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 1 && candidate_dtype == 1) {
        launch_typed_pair_distance<uint8_t, uint8_t>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 1 && candidate_dtype == 2) {
        launch_typed_pair_distance<uint8_t, int8_t>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 0) {
        launch_typed_pair_distance<int8_t, float>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 1) {
        launch_typed_pair_distance<int8_t, uint8_t>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 2) {
        launch_typed_pair_distance<int8_t, int8_t>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else {
        fprintf(stderr, "Unsupported query/candidate dtype pair: %u/%u\n", query_dtype, candidate_dtype);
        abort();
    }

    CUDA_CHECK(cudaEventRecord(event, stream));
}


void launch_batch_typed_query_l2_distances_indirect(cudaStream_t stream, cudaEvent_t event,
                                                    const void* d_query,
                                                    uint32_t query_dtype,
                                                    const void* const* d_candidate_ptrs,
                                                    uint32_t candidate_dtype,
                                                    float* d_distances,
                                                    uint32_t n_candidates, uint32_t dim) {
    if (n_candidates == 0) {
        CUDA_CHECK(cudaEventRecord(event, stream));
        return;
    }

    if (query_dtype == 0 && candidate_dtype == 0) {
        launch_typed_pair_distance_indirect<float, float>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 0 && candidate_dtype == 1) {
        launch_typed_pair_distance_indirect<float, uint8_t>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 0 && candidate_dtype == 2) {
        launch_typed_pair_distance_indirect<float, int8_t>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 1 && candidate_dtype == 0) {
        launch_typed_pair_distance_indirect<uint8_t, float>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 1 && candidate_dtype == 1) {
        launch_typed_pair_distance_indirect<uint8_t, uint8_t>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 1 && candidate_dtype == 2) {
        launch_typed_pair_distance_indirect<uint8_t, int8_t>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 0) {
        launch_typed_pair_distance_indirect<int8_t, float>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 1) {
        launch_typed_pair_distance_indirect<int8_t, uint8_t>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 2) {
        launch_typed_pair_distance_indirect<int8_t, int8_t>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else {
        fprintf(stderr, "Unsupported query/candidate dtype pair: %u/%u\n", query_dtype, candidate_dtype);
        abort();
    }

    CUDA_CHECK(cudaEventRecord(event, stream));
}

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

    size_t smem_size = n_candidates * sizeof(bool);
    uint32_t block_size = std::min(BLOCK_SIZE, n_candidates);

    gpu_kernels::robust_prune_kernel
        <<<1, block_size, smem_size, stream>>>(
            d_source_vec, d_candidate_vecs, d_candidate_dists, d_candidate_order,
            n_candidates, dim, alpha, R,
            d_pruned_indices, d_pruned_count);

    CUDA_CHECK(cudaEventRecord(event, stream));
}

void launch_robust_prune_typed(cudaStream_t stream, cudaEvent_t event,
                               const void* d_candidate_vecs,
                               uint32_t candidate_dtype,
                               const float* d_candidate_dists,
                               const uint32_t* d_candidate_order,
                               uint32_t n_candidates, uint32_t dim,
                               float alpha, uint32_t R,
                               uint32_t* d_pruned_indices, uint32_t* d_pruned_count) {
    if (candidate_dtype == 0) {
        launch_robust_prune(stream, event, nullptr, static_cast<const float*>(d_candidate_vecs),
                            d_candidate_dists, d_candidate_order, n_candidates, dim, alpha, R,
                            d_pruned_indices, d_pruned_count);
        return;
    }
    if (n_candidates == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_pruned_count, 0, sizeof(uint32_t), stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        return;
    }

    size_t smem_size = n_candidates * sizeof(bool);
    uint32_t block_size = std::min(BLOCK_SIZE, n_candidates);

    if (candidate_dtype == 1) {
        gpu_kernels::robust_prune_typed_kernel<uint8_t>
            <<<1, block_size, smem_size, stream>>>(
                static_cast<const uint8_t*>(d_candidate_vecs), d_candidate_dists, d_candidate_order,
                n_candidates, dim, alpha, R, d_pruned_indices, d_pruned_count);
    } else if (candidate_dtype == 2) {
        gpu_kernels::robust_prune_typed_kernel<int8_t>
            <<<1, block_size, smem_size, stream>>>(
                static_cast<const int8_t*>(d_candidate_vecs), d_candidate_dists, d_candidate_order,
                n_candidates, dim, alpha, R, d_pruned_indices, d_pruned_count);
    } else {
        fprintf(stderr, "Unsupported robust-prune candidate dtype: %u\n", candidate_dtype);
        abort();
    }

    CUDA_CHECK(cudaEventRecord(event, stream));
}

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

void launch_batch_rabitq_distances(cudaStream_t stream, cudaEvent_t event,
                                    const uint64_t* d_query_code,
                                    const uint8_t* d_candidate_data,
                                    float* d_distances,
                                    float query_norm2, uint32_t n_candidates,
                                    uint32_t entry_bytes) {
    if (n_candidates == 0) { CUDA_CHECK(cudaEventRecord(event, stream)); return; }
    constexpr uint32_t BLOCK = 512;
    uint32_t blocks = (n_candidates + BLOCK - 1) / BLOCK;
    gpu_kernels::rabitq_popcount_kernel<<<blocks, BLOCK, 0, stream>>>(
        d_query_code, d_candidate_data, d_distances,
        query_norm2, n_candidates, entry_bytes);
    CUDA_CHECK(cudaEventRecord(event, stream));
}

}  // namespace gpu
