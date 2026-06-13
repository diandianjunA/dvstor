#include "gpu_kernel_launcher.hh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <type_traits>

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

template <typename LhsT, typename RhsT>
constexpr uint64_t max_integral_squared_difference() {
    static_assert(std::is_integral_v<LhsT> && std::is_integral_v<RhsT>);
    constexpr int64_t lhs_min = static_cast<int64_t>(std::numeric_limits<LhsT>::min());
    constexpr int64_t lhs_max = static_cast<int64_t>(std::numeric_limits<LhsT>::max());
    constexpr int64_t rhs_min = static_cast<int64_t>(std::numeric_limits<RhsT>::min());
    constexpr int64_t rhs_max = static_cast<int64_t>(std::numeric_limits<RhsT>::max());
    constexpr uint64_t diff1 = static_cast<uint64_t>(lhs_max - rhs_min);
    constexpr uint64_t diff2 = static_cast<uint64_t>(rhs_max - lhs_min);
    constexpr uint64_t max_diff = diff1 > diff2 ? diff1 : diff2;
    return max_diff * max_diff;
}

template <typename LhsT, typename RhsT>
bool int32_accumulator_is_safe(uint32_t dim) {
    constexpr uint64_t max_squared_difference =
        max_integral_squared_difference<LhsT, RhsT>();
    constexpr uint64_t max_safe_dim =
        static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) /
        max_squared_difference;
    return dim <= max_safe_dim;
}

static_assert(max_integral_squared_difference<uint8_t, uint8_t>() == 65025);
static_assert(max_integral_squared_difference<int8_t, int8_t>() == 65025);
static_assert(max_integral_squared_difference<uint8_t, int8_t>() == 146689);
static_assert(static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / 65025 >= 128);
static_assert(static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / 146689 >= 128);

template <typename T, typename AccumulatorT = int32_t>
void launch_id_distance_typed(cudaStream_t stream,
                              const void* d_base_vectors,
                              uint32_t query_id,
                              const uint32_t* d_candidate_ids,
                              float* d_distances,
                              uint32_t n_candidates,
                              uint32_t dim) {
    uint32_t total_threads = n_candidates * TILE_SIZE;
    uint32_t num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_kernels::batch_l2_id_distance_kernel<TILE_SIZE, T, AccumulatorT>
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
        if (int32_accumulator_is_safe<uint8_t, uint8_t>(dim)) {
            launch_id_distance_typed<uint8_t, int32_t>(stream, d_base_vectors, query_id,
                d_candidate_ids, d_distances, n_candidates, dim);
        } else {
            launch_id_distance_typed<uint8_t, int64_t>(stream, d_base_vectors, query_id,
                d_candidate_ids, d_distances, n_candidates, dim);
        }
    } else if (dtype == 2) {
        if (int32_accumulator_is_safe<int8_t, int8_t>(dim)) {
            launch_id_distance_typed<int8_t, int32_t>(stream, d_base_vectors, query_id,
                d_candidate_ids, d_distances, n_candidates, dim);
        } else {
            launch_id_distance_typed<int8_t, int64_t>(stream, d_base_vectors, query_id,
                d_candidate_ids, d_distances, n_candidates, dim);
        }
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

template <typename QueryT, typename CandidateT, typename IntegralAccumulator = int32_t>
void launch_typed_pair_distance(cudaStream_t stream,
                                const void* d_query,
                                const void* d_candidates,
                                float* d_distances,
                                uint32_t n_candidates,
                                uint32_t dim) {
    uint32_t total_threads = n_candidates * TILE_SIZE;
    uint32_t num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_kernels::batch_l2_typed_pair_distance_kernel<
        TILE_SIZE, QueryT, CandidateT, IntegralAccumulator>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<const QueryT*>(d_query),
            static_cast<const CandidateT*>(d_candidates),
            d_distances, n_candidates, dim);
}


template <typename QueryT, typename CandidateT, typename IntegralAccumulator = int32_t>
void launch_typed_pair_distance_indirect(cudaStream_t stream,
                                         const void* d_query,
                                         const void* const* d_candidate_ptrs,
                                         float* d_distances,
                                         uint32_t n_candidates,
                                         uint32_t dim) {
    uint32_t total_threads = n_candidates * TILE_SIZE;
    uint32_t num_blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_kernels::batch_l2_typed_pair_distance_indirect_kernel<
        TILE_SIZE, QueryT, CandidateT, IntegralAccumulator>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<const QueryT*>(d_query),
            d_candidate_ptrs,
            d_distances, n_candidates, dim);
}

template <typename QueryT, typename CandidateT>
void launch_integral_pair_distance(cudaStream_t stream,
                                   const void* d_query,
                                   const void* d_candidates,
                                   float* d_distances,
                                   uint32_t n_candidates,
                                   uint32_t dim) {
    if (int32_accumulator_is_safe<QueryT, CandidateT>(dim)) {
        launch_typed_pair_distance<QueryT, CandidateT, int32_t>(
            stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else {
        launch_typed_pair_distance<QueryT, CandidateT, int64_t>(
            stream, d_query, d_candidates, d_distances, n_candidates, dim);
    }
}

template <typename QueryT, typename CandidateT>
void launch_integral_pair_distance_indirect(cudaStream_t stream,
                                            const void* d_query,
                                            const void* const* d_candidate_ptrs,
                                            float* d_distances,
                                            uint32_t n_candidates,
                                            uint32_t dim) {
    if (int32_accumulator_is_safe<QueryT, CandidateT>(dim)) {
        launch_typed_pair_distance_indirect<QueryT, CandidateT, int32_t>(
            stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else {
        launch_typed_pair_distance_indirect<QueryT, CandidateT, int64_t>(
            stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    }
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
        launch_integral_pair_distance<uint8_t, uint8_t>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 1 && candidate_dtype == 2) {
        launch_integral_pair_distance<uint8_t, int8_t>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 0) {
        launch_typed_pair_distance<int8_t, float>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 1) {
        launch_integral_pair_distance<int8_t, uint8_t>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 2) {
        launch_integral_pair_distance<int8_t, int8_t>(stream, d_query, d_candidates, d_distances, n_candidates, dim);
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
        launch_integral_pair_distance_indirect<uint8_t, uint8_t>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 1 && candidate_dtype == 2) {
        launch_integral_pair_distance_indirect<uint8_t, int8_t>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 0) {
        launch_typed_pair_distance_indirect<int8_t, float>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 1) {
        launch_integral_pair_distance_indirect<int8_t, uint8_t>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
    } else if (query_dtype == 2 && candidate_dtype == 2) {
        launch_integral_pair_distance_indirect<int8_t, int8_t>(stream, d_query, d_candidate_ptrs, d_distances, n_candidates, dim);
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

void launch_batch_rabitq_asymmetric_distances(cudaStream_t stream, cudaEvent_t event,
                                    const float* d_rotated_query,
                                    const uint8_t* d_candidate_data,
                                    float* d_distances,
                                    float query_norm2, uint32_t n_candidates,
                                    uint32_t code_bits, uint32_t code_bytes,
                                    uint32_t entry_bytes) {
    if (n_candidates == 0) { CUDA_CHECK(cudaEventRecord(event, stream)); return; }
    constexpr uint32_t BLOCK = 256;
    if (code_bits <= 256) {
        const uint32_t blocks = (n_candidates + BLOCK / 8 - 1) / (BLOCK / 8);
        gpu_kernels::rabitq_asymmetric_kernel<8><<<blocks, BLOCK, 0, stream>>>(
            d_rotated_query, d_candidate_data, d_distances, query_norm2, n_candidates,
            code_bits, code_bytes, entry_bytes);
    } else if (code_bits <= 512) {
        const uint32_t blocks = (n_candidates + BLOCK / 16 - 1) / (BLOCK / 16);
        gpu_kernels::rabitq_asymmetric_kernel<16><<<blocks, BLOCK, 0, stream>>>(
            d_rotated_query, d_candidate_data, d_distances, query_norm2, n_candidates,
            code_bits, code_bytes, entry_bytes);
    } else {
        const uint32_t blocks = (n_candidates + BLOCK / 32 - 1) / (BLOCK / 32);
        gpu_kernels::rabitq_asymmetric_kernel<32><<<blocks, BLOCK, 0, stream>>>(
            d_rotated_query, d_candidate_data, d_distances, query_norm2, n_candidates,
            code_bits, code_bytes, entry_bytes);
    }
    CUDA_CHECK(cudaEventRecord(event, stream));
}

}  // namespace gpu
