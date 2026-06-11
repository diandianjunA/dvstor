#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <type_traits>

namespace cg = cooperative_groups;

namespace gpu_kernels {

template <typename T>
__device__ __forceinline__ float typed_component_to_float(T value) {
    return static_cast<float>(value);
}

template <uint32_t TILE_SIZE, typename QueryT, typename CandidateT>
__global__ void batch_l2_typed_pair_distance_kernel(
    const QueryT* __restrict__ query,
    const CandidateT* __restrict__ candidates,
    float* __restrict__ distances,
    uint32_t n_candidates,
    uint32_t dim)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_SIZE>(block);

    uint32_t tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    if (tile_id >= n_candidates) return;

    const CandidateT* cand_vec = candidates + static_cast<size_t>(tile_id) * dim;
    if constexpr (std::is_integral_v<QueryT> && std::is_integral_v<CandidateT>) {
        int local_sum = 0;
        for (uint32_t i = tile.thread_rank(); i < dim; i += TILE_SIZE) {
            const int diff = static_cast<int>(query[i]) - static_cast<int>(cand_vec[i]);
            local_sum += diff * diff;
        }
        int total = cg::reduce(tile, local_sum, cg::plus<int>());
        if (tile.thread_rank() == 0) {
            distances[tile_id] = static_cast<float>(total);
        }
    } else {
        float local_sum = 0.0f;
        for (uint32_t i = tile.thread_rank(); i < dim; i += TILE_SIZE) {
            const float diff = typed_component_to_float(query[i]) - typed_component_to_float(cand_vec[i]);
            local_sum += diff * diff;
        }
        float total = cg::reduce(tile, local_sum, cg::plus<float>());
        if (tile.thread_rank() == 0) {
            distances[tile_id] = total;
        }
    }
}


template <uint32_t TILE_SIZE, typename QueryT, typename CandidateT>
__global__ void batch_l2_typed_pair_distance_indirect_kernel(
    const QueryT* __restrict__ query,
    const void* const* __restrict__ candidate_ptrs,
    float* __restrict__ distances,
    uint32_t n_candidates,
    uint32_t dim)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_SIZE>(block);

    uint32_t tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    if (tile_id >= n_candidates) return;

    const CandidateT* cand_vec = static_cast<const CandidateT*>(candidate_ptrs[tile_id]);
    if constexpr (std::is_integral_v<QueryT> && std::is_integral_v<CandidateT>) {
        int local_sum = 0;
        for (uint32_t i = tile.thread_rank(); i < dim; i += TILE_SIZE) {
            const int diff = static_cast<int>(query[i]) - static_cast<int>(cand_vec[i]);
            local_sum += diff * diff;
        }
        int total = cg::reduce(tile, local_sum, cg::plus<int>());
        if (tile.thread_rank() == 0) {
            distances[tile_id] = static_cast<float>(total);
        }
    } else {
        float local_sum = 0.0f;
        for (uint32_t i = tile.thread_rank(); i < dim; i += TILE_SIZE) {
            const float diff = typed_component_to_float(query[i]) - typed_component_to_float(cand_vec[i]);
            local_sum += diff * diff;
        }
        float total = cg::reduce(tile, local_sum, cg::plus<float>());
        if (tile.thread_rank() == 0) {
            distances[tile_id] = total;
        }
    }
}


template <uint32_t TILE_SIZE, typename T>
__global__ void batch_l2_id_distance_kernel(
    const T* __restrict__ base_vectors,
    uint32_t query_id,
    const uint32_t* __restrict__ candidate_ids,
    float* __restrict__ distances,
    uint32_t n_candidates,
    uint32_t dim)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_SIZE>(block);

    uint32_t tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    if (tile_id >= n_candidates) return;

    const T* query = base_vectors + static_cast<size_t>(query_id) * dim;
    const uint32_t cand_id = candidate_ids[tile_id];
    const T* cand = base_vectors + static_cast<size_t>(cand_id) * dim;
    if constexpr (std::is_integral_v<T>) {
        int local_sum = 0;
        for (uint32_t i = tile.thread_rank(); i < dim; i += TILE_SIZE) {
            const int diff = static_cast<int>(query[i]) - static_cast<int>(cand[i]);
            local_sum += diff * diff;
        }
        int total = cg::reduce(tile, local_sum, cg::plus<int>());
        if (tile.thread_rank() == 0) distances[tile_id] = static_cast<float>(total);
    } else {
        float local_sum = 0.0f;
        for (uint32_t i = tile.thread_rank(); i < dim; i += TILE_SIZE) {
            const float diff = static_cast<float>(query[i]) - static_cast<float>(cand[i]);
            local_sum += diff * diff;
        }
        float total = cg::reduce(tile, local_sum, cg::plus<float>());
        if (tile.thread_rank() == 0) distances[tile_id] = total;
    }
}

template <uint32_t TILE_SIZE>
__global__ void batch_l2_squared_distance_kernel(
    const float* __restrict__ query,
    const float* __restrict__ candidates,
    float* __restrict__ distances,
    uint32_t n_candidates,
    uint32_t dim)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_SIZE>(block);

    uint32_t tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    if (tile_id >= n_candidates) return;

    const float* cand_vec = candidates + static_cast<size_t>(tile_id) * dim;
    const uint4* q_ptr = reinterpret_cast<const uint4*>(query);
    const uint4* c_ptr = reinterpret_cast<const uint4*>(cand_vec);
    constexpr uint32_t n_float_per_uint4 = 4;
    const uint32_t n_uint4 = dim / n_float_per_uint4;

    float local_sum = 0.0f;
    for (uint32_t i = tile.thread_rank(); i < n_uint4; i += TILE_SIZE) {
        uint4 q_data = q_ptr[i];
        uint4 c_data = c_ptr[i];

        float* q_f = reinterpret_cast<float*>(&q_data);
        float* c_f = reinterpret_cast<float*>(&c_data);
        for (int k = 0; k < 4; ++k) {
            const float diff = q_f[k] - c_f[k];
            local_sum += diff * diff;
        }
    }

    uint32_t base = n_uint4 * n_float_per_uint4;
    for (uint32_t i = base + tile.thread_rank(); i < dim; i += TILE_SIZE) {
        const float diff = query[i] - cand_vec[i];
        local_sum += diff * diff;
    }

    float total = cg::reduce(tile, local_sum, cg::plus<float>());
    if (tile.thread_rank() == 0) {
        distances[tile_id] = total;
    }
}

__global__ void robust_prune_kernel(
    const float* __restrict__ source_vec,
    const float* __restrict__ candidate_vecs,
    const float* __restrict__ candidate_dists,
    const uint32_t* __restrict__ candidate_order,
    uint32_t n_candidates,
    uint32_t dim,
    float alpha,
    uint32_t max_R,
    uint32_t* __restrict__ pruned_indices,
    uint32_t* __restrict__ pruned_count)
{
    extern __shared__ float smem[];
    bool* is_valid = reinterpret_cast<bool*>(smem);

    for (uint32_t i = threadIdx.x; i < n_candidates; i += blockDim.x) {
        is_valid[i] = true;
    }
    __syncthreads();

    __shared__ uint32_t write_idx;
    if (threadIdx.x == 0) {
        write_idx = 0;
    }
    __syncthreads();

    if (n_candidates <= max_R) {
        for (uint32_t i = threadIdx.x; i < n_candidates; i += blockDim.x) {
            pruned_indices[i] = candidate_order ? candidate_order[i] : i;
        }
        if (threadIdx.x == 0) {
            *pruned_count = n_candidates;
        }
        return;
    }

    for (uint32_t start = 0; start < n_candidates && write_idx < max_R; start++) {
        if (!is_valid[start]) continue;

        if (threadIdx.x == 0) {
            pruned_indices[write_idx] = candidate_order ? candidate_order[start] : start;
            write_idx++;
        }
        __syncthreads();

        const uint32_t pstar_idx = candidate_order ? candidate_order[start] : start;
        const float* pstar_vec = candidate_vecs + static_cast<size_t>(pstar_idx) * dim;

        for (uint32_t i = start + 1 + threadIdx.x; i < n_candidates; i += blockDim.x) {
            if (!is_valid[i]) continue;

            const uint32_t pprime_idx = candidate_order ? candidate_order[i] : i;
            const float* pprime_vec = candidate_vecs + static_cast<size_t>(pprime_idx) * dim;
            float dist_pstar_pprime = 0.0f;
            for (uint32_t d = 0; d < dim; d++) {
                float diff = pstar_vec[d] - pprime_vec[d];
                dist_pstar_pprime += diff * diff;
            }

            if (alpha * dist_pstar_pprime <= candidate_dists[i]) {
                is_valid[i] = false;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *pruned_count = write_idx;
    }
}

template <typename CandidateT>
__global__ void robust_prune_typed_kernel(
    const CandidateT* __restrict__ candidate_vecs,
    const float* __restrict__ candidate_dists,
    const uint32_t* __restrict__ candidate_order,
    uint32_t n_candidates,
    uint32_t dim,
    float alpha,
    uint32_t max_R,
    uint32_t* __restrict__ pruned_indices,
    uint32_t* __restrict__ pruned_count)
{
    extern __shared__ float smem[];
    bool* is_valid = reinterpret_cast<bool*>(smem);

    for (uint32_t i = threadIdx.x; i < n_candidates; i += blockDim.x) {
        is_valid[i] = true;
    }
    __syncthreads();

    __shared__ uint32_t write_idx;
    if (threadIdx.x == 0) {
        write_idx = 0;
    }
    __syncthreads();

    if (n_candidates <= max_R) {
        for (uint32_t i = threadIdx.x; i < n_candidates; i += blockDim.x) {
            pruned_indices[i] = candidate_order ? candidate_order[i] : i;
        }
        if (threadIdx.x == 0) {
            *pruned_count = n_candidates;
        }
        return;
    }

    for (uint32_t start = 0; start < n_candidates && write_idx < max_R; start++) {
        if (!is_valid[start]) continue;

        if (threadIdx.x == 0) {
            pruned_indices[write_idx] = candidate_order ? candidate_order[start] : start;
            write_idx++;
        }
        __syncthreads();

        const uint32_t pstar_idx = candidate_order ? candidate_order[start] : start;
        const CandidateT* pstar_vec = candidate_vecs + static_cast<size_t>(pstar_idx) * dim;

        for (uint32_t i = start + 1 + threadIdx.x; i < n_candidates; i += blockDim.x) {
            if (!is_valid[i]) continue;

            const uint32_t pprime_idx = candidate_order ? candidate_order[i] : i;
            const CandidateT* pprime_vec = candidate_vecs + static_cast<size_t>(pprime_idx) * dim;
            float dist_pstar_pprime = 0.0f;
            for (uint32_t d = 0; d < dim; d++) {
                float diff = typed_component_to_float(pstar_vec[d]) - typed_component_to_float(pprime_vec[d]);
                dist_pstar_pprime += diff * diff;
            }

            if (alpha * dist_pstar_pprime <= candidate_dists[i]) {
                is_valid[i] = false;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *pruned_count = write_idx;
    }
}

// RaBitQ: approximate L2 from 128-bit codes via arcsin formula.
// Codes and norms are interleaved: [code_lo(8B)][code_hi(8B)][norm(4B)] per candidate.
__global__ void rabitq_popcount_kernel(
    const uint64_t* __restrict__ query_code,
    const uint64_t* __restrict__ candidate_data,
    float* __restrict__ distances,
    float query_norm2, uint32_t n_candidates, uint32_t stride_qwords)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_candidates) return;
    uint64_t q_lo = query_code[0], q_hi = query_code[1];
    const uint64_t* entry = candidate_data + static_cast<size_t>(tid) * stride_qwords;
    uint64_t c_lo = entry[0], c_hi = entry[1];
    // Norm is 4 bytes starting at byte 16 of the 20-byte entry
    float vn2 = *reinterpret_cast<const float*>(reinterpret_cast<const char*>(entry) + 16);

    uint32_t pop = __popcll(q_lo ^ c_lo) + __popcll(q_hi ^ c_hi);
    float qn = sqrtf(fmaxf(query_norm2, 0.0f));
    float vn = sqrtf(fmaxf(vn2, 0.0f));
    float angle = 3.14159265f * float(pop) / 128.0f;
    float d2 = query_norm2 + vn2 - 2.0f * qn * vn * cosf(angle);
    distances[tid] = fmaxf(d2, 0.0f);
}

}  // namespace gpu_kernels
