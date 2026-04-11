#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>

namespace cg = cooperative_groups;

namespace gpu_kernels {

// =============================================================================
// Batch L2 Squared Distance Kernel
// Adapted from Jasper's euclidean_distance_no_sqrt_chunked
// Uses tile-based parallelism with uint4 coalesced loads
// =============================================================================

template <uint32_t TILE_SIZE>
__global__ void batch_l2_squared_distance_kernel(
    const float* __restrict__ query,       // [dim]
    const float* __restrict__ candidates,  // [n_candidates * dim]
    float* __restrict__ distances,         // [n_candidates]
    uint32_t n_candidates,
    uint32_t dim)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_SIZE>(block);

    // Each tile computes one distance
    uint32_t tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    if (tile_id >= n_candidates) return;

    const float* cand_vec = candidates + tile_id * dim;

    // Chunked uint4 load for coalesced memory access
    const uint4* q_ptr = reinterpret_cast<const uint4*>(query);
    const uint4* c_ptr = reinterpret_cast<const uint4*>(cand_vec);
    const uint32_t n_float_per_uint4 = 4;  // uint4 = 4 floats = 16 bytes
    const uint32_t n_uint4 = dim / n_float_per_uint4;

    float local_sum = 0.0f;

    for (uint32_t i = tile.thread_rank(); i < n_uint4; i += TILE_SIZE) {
        uint4 q_data = q_ptr[i];
        uint4 c_data = c_ptr[i];

        float* q_f = reinterpret_cast<float*>(&q_data);
        float* c_f = reinterpret_cast<float*>(&c_data);

        for (int k = 0; k < 4; k++) {
            float diff = q_f[k] - c_f[k];
            local_sum += diff * diff;
        }
    }

    // Handle remaining dimensions (if dim not divisible by 4)
    uint32_t base = n_uint4 * n_float_per_uint4;
    for (uint32_t i = base + tile.thread_rank(); i < dim; i += TILE_SIZE) {
        float diff = query[i] - cand_vec[i];
        local_sum += diff * diff;
    }

    // Reduce across tile
    float total = cg::reduce(tile, local_sum, cg::plus<float>());

    if (tile.thread_rank() == 0) {
        distances[tile_id] = total;  // squared distance, no sqrt
    }
}

// =============================================================================
// RaBitQ Approximate Distance Kernel
// Adapted from Jasper's rabitq_l2_distance_device / one_distance_device
// =============================================================================

struct RabitqQueryFactor {
    float add;
    float k1xSumq;
    float kBxSumq;
};

// Generic multi-bit RaBitQ distance using bit extraction
template <uint32_t TILE_SIZE, uint32_t BITS_PER_DIM>
__global__ void batch_rabitq_distance_kernel(
    const float* __restrict__ rot_query,     // [dim] rotated query
    const RabitqQueryFactor* __restrict__ qfactor,  // query factor
    const uint8_t* __restrict__ rabitq_vecs, // [n * rabitq_vec_size] packed data
    const float* __restrict__ rabitq_add,    // [n] per-vector add factors
    const float* __restrict__ rabitq_rescale,// [n] per-vector rescale factors
    float* __restrict__ distances,           // [n] output
    uint32_t n_candidates,
    uint32_t dim)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_SIZE>(block);

    uint32_t tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    if (tile_id >= n_candidates) return;

    // Size of packed data per vector (excluding add/rescale floats)
    const uint32_t packed_bytes = (BITS_PER_DIM * dim + 7) / 8;
    const uint32_t vec_stride = packed_bytes + 2 * sizeof(float);  // data + add + rescale
    const uint8_t* vec_data = rabitq_vecs + tile_id * vec_stride;
    float data_add = *reinterpret_cast<const float*>(vec_data + packed_bytes);
    float data_rescale = *reinterpret_cast<const float*>(vec_data + packed_bytes + sizeof(float));

    // Compute dot product between rotated query and quantized data
    float dot_tmp = 0.0f;
    const float cb = -static_cast<float>((1 << BITS_PER_DIM) - 1) / 2.0f;

    for (uint32_t j = tile.thread_rank(); j < dim; j += TILE_SIZE) {
        // Extract quantized value for dimension j
        uint32_t bit_idx = j * BITS_PER_DIM;
        uint32_t byte_idx = bit_idx / 8;
        uint32_t bit_off = bit_idx % 8;
        uint16_t chunk = vec_data[byte_idx];
        if (bit_off + BITS_PER_DIM > 8 && byte_idx + 1 < packed_bytes) {
            chunk |= (uint16_t(vec_data[byte_idx + 1]) << 8);
        }
        uint8_t mask = (1u << BITS_PER_DIM) - 1u;
        float val = static_cast<float>((chunk >> bit_off) & mask);

        dot_tmp += val * rot_query[j];
    }

    float dot = cg::reduce(tile, dot_tmp, cg::plus<float>());

    if (tile.thread_rank() == 0) {
        float k1_factor = static_cast<float>((1 << BITS_PER_DIM) - 1);
        float dist = data_add + qfactor->add + data_rescale * (dot + qfactor->k1xSumq * k1_factor);
        distances[tile_id] = dist;
    }
}

// Specialized 8-bit RaBitQ distance using char4 loads (from Jasper)
template <uint32_t TILE_SIZE>
__global__ void batch_rabitq_8bit_distance_kernel(
    const float* __restrict__ rot_query,
    const RabitqQueryFactor* __restrict__ qfactor,
    const uint8_t* __restrict__ rabitq_vecs,  // packed 8-bit: dim bytes + 8 bytes (add+rescale)
    float* __restrict__ distances,
    uint32_t n_candidates,
    uint32_t dim)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_SIZE>(block);

    uint32_t tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    if (tile_id >= n_candidates) return;

    uint32_t vec_stride = dim + 2 * sizeof(float);
    const uint8_t* vec_data = rabitq_vecs + tile_id * vec_stride;
    float data_add = *reinterpret_cast<const float*>(vec_data + dim);
    float data_rescale = *reinterpret_cast<const float*>(vec_data + dim + sizeof(float));

    // Use char4/float4 for coalesced loads
    const char4* data_ptr = reinterpret_cast<const char4*>(vec_data);
    const float4* query_ptr = reinterpret_cast<const float4*>(rot_query);

    float dot_tmp = 0.0f;
    for (uint32_t i = tile.thread_rank(); i < dim / 4; i += TILE_SIZE) {
        float4 q = query_ptr[i];
        char4 d = data_ptr[i];
        dot_tmp += static_cast<float>(static_cast<uint8_t>(d.x)) * q.x +
                   static_cast<float>(static_cast<uint8_t>(d.y)) * q.y +
                   static_cast<float>(static_cast<uint8_t>(d.z)) * q.z +
                   static_cast<float>(static_cast<uint8_t>(d.w)) * q.w;
    }

    float dot = cg::reduce(tile, dot_tmp, cg::plus<float>());

    if (tile.thread_rank() == 0) {
        float dist = data_add + qfactor->add + data_rescale *
            (dot + qfactor->k1xSumq * static_cast<float>((1 << 8) - 1));
        distances[tile_id] = dist;
    }
}

// =============================================================================
// RobustPrune Kernel
// Adapted from Jasper's robust_prune_block
// =============================================================================

/**
 * RobustPrune for a single source vertex.
 * Input: n_candidates candidate vectors with distances sorted ascending.
 * Output: up to R selected neighbor indices.
 *
 * Algorithm:
 * 1. Always select the nearest candidate
 * 2. For each remaining candidate p*, check if any already-selected p' satisfies:
 *    alpha * dist(p', p*) <= dist(source, p*)
 *    If so, p* is redundant and skipped.
 */
__global__ void robust_prune_kernel(
    const float* __restrict__ source_vec,    // [dim]
    const float* __restrict__ candidate_vecs,// [n_candidates * dim]
    const float* __restrict__ candidate_dists,// [n_candidates] sorted ascending
    const uint32_t* __restrict__ candidate_order, // [n_candidates] or nullptr
    uint32_t n_candidates,
    uint32_t dim,
    float alpha,
    uint32_t max_R,
    uint32_t* __restrict__ pruned_indices,   // [max_R] output
    uint32_t* __restrict__ pruned_count)     // [1] output
{
    // Single-block kernel for one source vertex
    extern __shared__ float smem[];

    // We need shared memory for tracking validity
    bool* is_valid = reinterpret_cast<bool*>(smem);

    // Initialize validity
    for (uint32_t i = threadIdx.x; i < n_candidates; i += blockDim.x) {
        is_valid[i] = true;
    }
    __syncthreads();

    __shared__ uint32_t write_idx;
    if (threadIdx.x == 0) {
        write_idx = 0;
    }
    __syncthreads();

    // Fast path: fewer candidates than R, just take them all
    if (n_candidates <= max_R) {
        for (uint32_t i = threadIdx.x; i < n_candidates; i += blockDim.x) {
            pruned_indices[i] = candidate_order ? candidate_order[i] : i;
        }
        if (threadIdx.x == 0) {
            *pruned_count = n_candidates;
        }
        return;
    }

    // Main pruning loop
    for (uint32_t start = 0; start < n_candidates && write_idx < max_R; start++) {
        if (!is_valid[start]) continue;

        // Thread 0 selects this candidate
        uint32_t selected_idx;
        if (threadIdx.x == 0) {
            selected_idx = write_idx;
            pruned_indices[write_idx] = candidate_order ? candidate_order[start] : start;
            write_idx++;
        }
        __syncthreads();

        float dist_src_pstar = candidate_dists[start];

        // Check remaining candidates for redundancy (parallel)
        const uint32_t pstar_idx = candidate_order ? candidate_order[start] : start;
        const float* pstar_vec = candidate_vecs + pstar_idx * dim;

        for (uint32_t i = start + 1 + threadIdx.x; i < n_candidates; i += blockDim.x) {
            if (!is_valid[i]) continue;

            float dist_src_pprime = candidate_dists[i];

            // Compute dist(p*, p') -- distance between selected and candidate
            const uint32_t pprime_idx = candidate_order ? candidate_order[i] : i;
            const float* pprime_vec = candidate_vecs + pprime_idx * dim;
            float dist_pstar_pprime = 0.0f;
            for (uint32_t d = 0; d < dim; d++) {
                float diff = pstar_vec[d] - pprime_vec[d];
                dist_pstar_pprime += diff * diff;
            }

            // Redundancy check: if p* is closer to p' than source is, mark p' as invalid
            if (alpha * dist_pstar_pprime <= dist_src_pprime) {
                is_valid[i] = false;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *pruned_count = write_idx;
    }
}

// =============================================================================
// RaBitQ Quantize Single Vector Kernel
// =============================================================================

template <uint32_t TILE_SIZE, uint32_t BITS_PER_DIM>
__global__ void rabitq_quantize_single_kernel(
    const float* __restrict__ rot_vector,  // [dim] already rotated
    const float* __restrict__ centroid,     // [dim] rotated centroid
    uint8_t* __restrict__ output_data,     // packed output
    float* __restrict__ output_add,        // add factor
    float* __restrict__ output_rescale,    // rescale factor
    uint32_t dim,
    double t_const)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_SIZE>(block);

    // Only one vector to process, first tile does the work
    uint32_t my_tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    if (my_tile_id > 0) return;

    constexpr double kEps = 1e-5;

    // Compute residual vector and L2 norm
    extern __shared__ uint8_t uncompressed[];

    float l2_sqr_tmp = 0;
    for (uint32_t j = tile.thread_rank(); j < dim; j += TILE_SIZE) {
        float diff = rot_vector[j] - centroid[j];
        l2_sqr_tmp += diff * diff;
    }
    float l2_sqr = cg::reduce(tile, l2_sqr_tmp, cg::plus<float>());
    float l2_norm = sqrtf(l2_sqr);

    // Quantize: normalize and compute uncompressed code
    float ip_norm_tmp = 0;
    for (uint32_t j = tile.thread_rank(); j < dim; j += TILE_SIZE) {
        float abs_o = fabsf((rot_vector[j] - centroid[j]) / l2_norm);
        int val = static_cast<int>((t_const * abs_o) + kEps);
        if (val >= (1 << (BITS_PER_DIM - 1))) {
            val = (1 << (BITS_PER_DIM - 1)) - 1;
        }
        uncompressed[j] = static_cast<uint8_t>(val);
        ip_norm_tmp += (val + 0.5f) * abs_o;
    }
    float ip_norm = cg::reduce(tile, ip_norm_tmp, cg::plus<float>());
    float ip_norm_inv = (ip_norm == 0) ? 1.0f : (1.0f / ip_norm);

    // Add sign bits
    uint32_t const mask = (1 << (BITS_PER_DIM - 1)) - 1;
    for (uint32_t j = tile.thread_rank(); j < dim; j += TILE_SIZE) {
        float residual = rot_vector[j] - centroid[j];
        if (residual >= 0) {
            uncompressed[j] += 1 << (BITS_PER_DIM - 1);
        } else {
            uint8_t tmp = uncompressed[j];
            uncompressed[j] = (~tmp) & mask;
        }
    }
    tile.sync();

    // Compute factors (ip with centroid and reconstructed vector)
    float cb = -(static_cast<float>(1 << (BITS_PER_DIM - 1)) - 0.5f);
    float ip_resi_xucb_tmp = 0, ip_cent_xucb_tmp = 0;
    for (uint32_t j = tile.thread_rank(); j < dim; j += TILE_SIZE) {
        float residual = rot_vector[j] - centroid[j];
        float xu_cb = static_cast<float>(uncompressed[j]) + cb;
        ip_resi_xucb_tmp += residual * xu_cb;
        ip_cent_xucb_tmp += centroid[j] * xu_cb;
    }
    float ip_resi_xucb = cg::reduce(tile, ip_resi_xucb_tmp, cg::plus<float>());
    float ip_cent_xucb = cg::reduce(tile, ip_cent_xucb_tmp, cg::plus<float>());
    if (ip_resi_xucb == 0) ip_resi_xucb = FLT_MAX;

    if (tile.thread_rank() == 0) {
        // L2 metric factors
        *output_add = l2_sqr + 2 * l2_sqr * ip_cent_xucb / ip_resi_xucb;
        *output_rescale = ip_norm_inv * -2 * l2_norm;
    }

    // Pack uncompressed code into output
    uint32_t compressed_size = (BITS_PER_DIM * dim + 7) / 8;
    for (uint32_t j = tile.thread_rank(); j < compressed_size; j += TILE_SIZE) {
        uint8_t byte_val = 0;
        for (uint32_t k = 0; k < 8 / BITS_PER_DIM && (j * (8 / BITS_PER_DIM) + k) < dim; k++) {
            uint32_t dim_idx = j * (8 / BITS_PER_DIM) + k;
            if (dim_idx < dim) {
                byte_val |= (uncompressed[dim_idx] << (k * BITS_PER_DIM));
            }
        }
        output_data[j] = byte_val;
    }
}

// =============================================================================
// RaBitQ Query Prepare Kernel
// =============================================================================

template <uint32_t TILE_SIZE, uint32_t BITS_PER_DIM>
__global__ void rabitq_query_prepare_kernel(
    const float* __restrict__ rot_query,  // [dim] already rotated
    const float* __restrict__ centroid,    // [dim] rotated centroid
    RabitqQueryFactor* __restrict__ factor,
    uint32_t dim)
{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_SIZE>(block);

    uint32_t my_tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    if (my_tile_id > 0) return;

    float c_b = -static_cast<float>((1 << BITS_PER_DIM) - 1) / 2.0f;
    float c_1 = -static_cast<float>((1 << 1) - 1) / 2.0f;

    float sqr_norm_tmp = 0;
    float sumq_tmp = 0;
    for (uint32_t j = tile.thread_rank(); j < dim; j += TILE_SIZE) {
        float diff = rot_query[j] - centroid[j];
        sqr_norm_tmp += diff * diff;
        sumq_tmp += rot_query[j];
    }
    float sqr_norm = cg::reduce(tile, sqr_norm_tmp, cg::plus<float>());
    float sumq = cg::reduce(tile, sumq_tmp, cg::plus<float>());

    if (tile.thread_rank() == 0) {
        factor->add = sqr_norm;
        factor->k1xSumq = c_1 * sumq;
        factor->kBxSumq = c_b * sumq;
    }
}

}  // namespace gpu_kernels
