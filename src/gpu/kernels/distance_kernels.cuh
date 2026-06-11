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

// Asymmetric RaBitQ with byte-level LUT optimization.
// Entry layout (32B): [code_lo(8B)][code_hi(8B)][x_norm(4B)][error_factor(4B)][reserved(8B)]
// Distance: ||q-x||² ≈ q_norm2 + x_norm² - 2*x_norm*e*(1/√128)*signed_dot
//
// LUT optimization: precompute lut[16][256] in shared memory where
//   lut[byte_pos][byte_val] = Σ_{bit j in byte} q_r[byte_pos*8+j] * sign(bit_j)
// Then signed_dot = Σ_{pos=0}^{15} lut[pos][code_byte[pos]]
// This reduces per-candidate work from 128 multiply-adds to 16 table lookups + adds.
__global__ void rabitq_asymmetric_kernel(
    const float* __restrict__ d_rotated_query,
    const uint8_t* __restrict__ candidate_data,
    float* __restrict__ distances,
    float query_norm2, uint32_t n_candidates, uint32_t entry_bytes)
{
    // LUT: 16 byte positions × 256 possible byte values = 4096 floats = 16KB shared mem
    __shared__ float s_lut[16][256];

    // Build LUT cooperatively across all threads in the block.
    // Total entries: 16 * 256 = 4096. Each thread handles multiple entries.
    const uint32_t lut_size = 16 * 256;
    for (uint32_t idx = threadIdx.x; idx < lut_size; idx += blockDim.x) {
        uint32_t pos = idx >> 8;        // byte position 0..15
        uint32_t byte_val = idx & 0xFF; // byte value 0..255
        // Bit layout: code bytes are stored big-endian within each u64 half.
        // code_lo stores bits 0..63 (byte 0 = bits 0..7 = MSB of code_lo)
        // code_hi stores bits 64..127 (byte 8 = bits 64..71 = MSB of code_hi)
        // Within each byte, bit 7 (MSB) corresponds to the lower query index.
        float sum = 0.0f;
        uint32_t base_bit = pos * 8;
        for (uint32_t b = 0; b < 8; ++b) {
            float q_val = d_rotated_query[base_bit + b];
            // Bit (7-b) in the byte corresponds to query index (base_bit + b)
            float sign = (byte_val & (1 << (7 - b))) ? 1.0f : -1.0f;
            sum += q_val * sign;
        }
        s_lut[pos][byte_val] = sum;
    }
    __syncthreads();

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_candidates) return;

    const uint8_t* entry = candidate_data + static_cast<size_t>(tid) * entry_bytes;
    float x_norm = *reinterpret_cast<const float*>(entry + 16);
    float e      = *reinterpret_cast<const float*>(entry + 20);

    // Read the 16-byte code and compute signed_dot via LUT lookups.
    // Code is stored as [lo(8B)][hi(8B)], each u64 in native (little-endian) order.
    // Byte 0 of code_lo (at entry[0]) on little-endian = bits 0-7 of the u64 value,
    // which is the LEAST significant byte. But our bit layout uses (63-j) encoding:
    //   bit j is at position (63-j) in the u64, so j=0 is at bit 63 (MSB).
    // On little-endian, MSB = byte[7], so j=0..7 → byte[7], j=8..15 → byte[6], etc.
    // To get byte_pos 0 (query indices 0..7), we need byte[7] of code_lo = entry[7].
    float signed_dot = 0.0f;
    // code_lo: query indices 0..63, stored at entry[0..7]
    // byte at entry[7] has the MSBs (bits 56-63 of u64 = query indices 0-7)
    // byte at entry[6] has bits 48-55 = query indices 8-15
    // ...
    // byte at entry[0] has bits 0-7 = query indices 56-63
    signed_dot += s_lut[0][entry[7]];
    signed_dot += s_lut[1][entry[6]];
    signed_dot += s_lut[2][entry[5]];
    signed_dot += s_lut[3][entry[4]];
    signed_dot += s_lut[4][entry[3]];
    signed_dot += s_lut[5][entry[2]];
    signed_dot += s_lut[6][entry[1]];
    signed_dot += s_lut[7][entry[0]];
    // code_hi: query indices 64..127, stored at entry[8..15]
    signed_dot += s_lut[8][entry[15]];
    signed_dot += s_lut[9][entry[14]];
    signed_dot += s_lut[10][entry[13]];
    signed_dot += s_lut[11][entry[12]];
    signed_dot += s_lut[12][entry[11]];
    signed_dot += s_lut[13][entry[10]];
    signed_dot += s_lut[14][entry[9]];
    signed_dot += s_lut[15][entry[8]];

    float x_norm2 = x_norm * x_norm;
    float ip_approx = x_norm * e * (1.0f / sqrtf(128.0f)) * signed_dot;
    float d2 = query_norm2 + x_norm2 - 2.0f * ip_approx;
    distances[tid] = fmaxf(d2, 0.0f);
}

}  // namespace gpu_kernels
