#pragma once

/**
 * GPU Buffer Manager: per compute-thread CUDA resource management.
 *
 * Each compute thread owns one GpuBufferManager that provides:
 *  - One CUDA stream + event per coroutine (for async GPU overlap)
 *  - Pinned host staging buffers for CPU->GPU data transfer
 *  - Device buffers mirroring the staging areas
 *  - Optional GPUDirect RDMA registration for candidate vector buffers
 */

#include <cstdint>

struct ibv_mr;
struct ibv_pd;

struct CUstream_st;
struct CUevent_st;
typedef CUstream_st* cudaStream_t;
typedef CUevent_st* cudaEvent_t;

namespace gpu {

struct CoroutineGpuState {
    cudaStream_t stream{nullptr};
    cudaEvent_t  event{nullptr};
    // Timing is kept separate from the completion event: the latter has timing
    // disabled so scheduler polling remains cheap.
    cudaEvent_t  kernel_start_event{nullptr};

    // Per-coroutine pinned staging buffers (host side)
    void*     h_query{nullptr};          // [dim * query component bytes]
    uint8_t*  h_candidate_vecs{nullptr}; // [max_batch * raw vector bytes]
    float*    h_candidate_dists{nullptr};// [max_batch]
    uint32_t* h_candidate_order{nullptr};// [max_batch]
    float*    h_distances{nullptr};      // [max_batch]
    const void** h_candidate_ptrs{nullptr}; // [max_batch]
    uint32_t* h_pruned_indices{nullptr}; // [R]
    uint32_t* h_pruned_count{nullptr};   // [1]

    // Per-coroutine device buffers
    void*     d_query{nullptr};
    uint8_t*  d_candidate_vecs{nullptr};
    uint8_t*  d_candidate_vecs_alt{nullptr};
    float*    d_candidate_dists{nullptr};
    uint32_t* d_candidate_order{nullptr};
    float*    d_distances{nullptr};
    const void** d_candidate_ptrs{nullptr};
    uint32_t* d_pruned_indices{nullptr};
    uint32_t* d_pruned_count{nullptr};

    ibv_mr*   d_candidate_vecs_mr{nullptr};
    ibv_mr*   d_candidate_vecs_alt_mr{nullptr};
    uint32_t  d_candidate_vecs_lkey{0};
    uint32_t  d_candidate_vecs_alt_lkey{0};
    bool      d_candidate_vecs_rdma_registered{false};
    bool      d_candidate_vecs_alt_rdma_registered{false};
    uint32_t  query_candidate_buffer_index{0};

    uint8_t* current_query_candidate_vecs() const {
        return query_candidate_buffer_index == 0 ? d_candidate_vecs : d_candidate_vecs_alt;
    }
    uint32_t current_query_candidate_vecs_lkey() const {
        return query_candidate_buffer_index == 0 ? d_candidate_vecs_lkey : d_candidate_vecs_alt_lkey;
    }
    bool current_query_candidate_vecs_registered() const {
        return query_candidate_buffer_index == 0 ? d_candidate_vecs_rdma_registered
                                                 : d_candidate_vecs_alt_rdma_registered;
    }
    void flip_query_candidate_buffer() { query_candidate_buffer_index ^= 1u; }
};

class GpuBufferManager {
public:
    GpuBufferManager() = default;
    ~GpuBufferManager();

    GpuBufferManager(const GpuBufferManager&) = delete;
    GpuBufferManager& operator=(const GpuBufferManager&) = delete;
    GpuBufferManager(GpuBufferManager&&) = delete;
    GpuBufferManager& operator=(GpuBufferManager&&) = delete;

    void init(uint32_t num_coroutines,
              uint32_t dim,
              uint32_t max_batch,
              uint32_t max_R,
              size_t query_vector_bytes = 0,
              size_t candidate_vector_bytes = 0,
              ibv_pd* rdma_pd = nullptr,
              bool enable_gpudirect_rdma = false);

    void destroy();

    CoroutineGpuState& state(uint32_t coroutine_id) { return states_[coroutine_id]; }
    const CoroutineGpuState& state(uint32_t coroutine_id) const { return states_[coroutine_id]; }

    cudaStream_t stream(uint32_t coroutine_id) const { return states_[coroutine_id].stream; }
    cudaEvent_t  event(uint32_t coroutine_id) const { return states_[coroutine_id].event; }

    uint32_t dim() const { return dim_; }
    uint32_t max_batch() const { return max_batch_; }
    uint32_t max_R() const { return max_R_; }
    size_t query_vector_bytes() const { return query_vector_bytes_; }
    size_t candidate_vector_bytes() const { return candidate_vector_bytes_; }
    bool initialized() const { return initialized_; }
    bool gpudirect_candidate_ready() const { return gpudirect_candidate_ready_; }

private:
    CoroutineGpuState* states_{nullptr};
    uint32_t num_coroutines_{0};
    uint32_t dim_{0};
    uint32_t max_batch_{0};
    uint32_t max_R_{0};
    size_t query_vector_bytes_{0};
    size_t candidate_vector_bytes_{0};

    bool gpudirect_rdma_enabled_{false};
    bool gpudirect_candidate_ready_{false};
    bool initialized_{false};
};

}  // namespace gpu
