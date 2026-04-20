#pragma once

/**
 * GPU Buffer Manager: per compute-thread CUDA resource management.
 *
 * Each compute thread owns one GpuBufferManager that provides:
 *  - One CUDA stream + event per coroutine (for async GPU overlap)
 *  - Pinned host staging buffers for CPU→GPU data transfer
 *  - Device buffers mirroring the staging areas
 *  - Shared device resources: RaBitQ rotation matrix, centroid
 *
 * The staging buffers are sized for the worst case:
 *  - query vector (dim floats)
 *  - batch candidate vectors or RaBitQ data (beam_width * per-vector size)
 *  - distance output (beam_width floats)
 *  - prune workspace (R+beam_width candidate vectors + distances + indices)
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpu/gpu_rabitq_cache.hh"

struct ibv_mr;
struct ibv_pd;

// Forward-declare CUDA types for host headers
struct CUstream_st;
struct CUevent_st;
typedef CUstream_st* cudaStream_t;
typedef CUevent_st* cudaEvent_t;
struct cublasContext;
typedef cublasContext* cublasHandle_t;

namespace gpu {

struct CoroutineGpuState {
    cudaStream_t stream{nullptr};
    cudaEvent_t  event{nullptr};

    // Per-coroutine pinned staging buffers (host side)
    float*    h_query{nullptr};          // [dim]
    float*    h_rot_query{nullptr};      // [dim]
    uint8_t*  h_rabitq_vecs{nullptr};    // [max_batch * rabitq_vec_size]
    float*    h_candidate_vecs{nullptr}; // [max_batch * dim] for prune
    float*    h_candidate_dists{nullptr};// [max_batch]
    uint32_t* h_candidate_order{nullptr};// [max_batch]
    uint32_t* h_cache_slot_ids{nullptr}; // [max_batch], mapped pinned host memory
    float*    h_distances{nullptr};      // [max_batch]
    uint32_t* h_pruned_indices{nullptr}; // [R]
    uint32_t* h_pruned_count{nullptr};   // [1]

    // Per-coroutine device buffers
    float*    d_query{nullptr};
    float*    d_rot_query{nullptr};
    uint8_t*  d_rabitq_vecs{nullptr};
    float*    d_candidate_vecs{nullptr};
    float*    d_candidate_dists{nullptr};
    uint32_t* d_candidate_order{nullptr};
    uint32_t* d_cache_slot_ids{nullptr}; // device alias of h_cache_slot_ids
    float*    d_distances{nullptr};
    uint32_t* d_pruned_indices{nullptr};
    uint32_t* d_pruned_count{nullptr};
    ibv_mr*   d_candidate_vecs_mr{nullptr};  // optional GPUDirect RDMA MR for d_candidate_vecs
    uint32_t  d_candidate_vecs_lkey{0};
    bool      d_candidate_vecs_rdma_registered{false};
    ibv_mr*   d_rabitq_vecs_mr{nullptr};
    uint32_t  d_rabitq_vecs_lkey{0};
    bool      d_rabitq_vecs_rdma_registered{false};

    // RaBitQ query factor (device, 3 floats)
    void*     d_query_factor{nullptr};
};

class GpuBufferManager {
public:
    GpuBufferManager() = default;
    ~GpuBufferManager();

    GpuBufferManager(const GpuBufferManager&) = delete;
    GpuBufferManager& operator=(const GpuBufferManager&) = delete;
    GpuBufferManager(GpuBufferManager&&) = delete;
    GpuBufferManager& operator=(GpuBufferManager&&) = delete;

    /**
     * Initialize all GPU resources for this thread.
     * Must be called after gpu_init() and before any kernel launch.
     *
     * @param num_coroutines  Number of coroutines per thread
     * @param dim             Vector dimensionality
     * @param max_batch       Maximum batch size (beam_width)
     * @param max_R           Maximum out-degree
     * @param rabitq_bits     Bits per dimension for RaBitQ
     */
    void init(uint32_t num_coroutines, uint32_t dim, uint32_t max_batch,
              uint32_t max_R, uint32_t rabitq_bits,
              ibv_pd* rdma_pd = nullptr,
              bool enable_gpudirect_rdma = false,
              size_t gpu_rabitq_cache_bytes = 0);

    /**
     * Release all GPU resources.
     */
    void destroy();

    // Accessors
    CoroutineGpuState& state(uint32_t coroutine_id) { return states_[coroutine_id]; }
    const CoroutineGpuState& state(uint32_t coroutine_id) const { return states_[coroutine_id]; }

    cudaStream_t stream(uint32_t coroutine_id) const { return states_[coroutine_id].stream; }
    cudaEvent_t  event(uint32_t coroutine_id) const { return states_[coroutine_id].event; }

    // Shared device resources (set once, read by all coroutines)
    float* d_rotation_matrix() const { return d_rotation_mat_; }
    float* d_centroid() const { return d_centroid_; }
    double t_const() const { return t_const_; }
    cublasHandle_t cublas_handle() const { return cublas_handle_; }

    /**
     * Upload rotation matrix to device. Called once during index load.
     * @param host_matrix  Column-major rotation matrix P (dim x dim floats)
     * @param dim          Dimension
     */
    void upload_rotation_matrix(const float* host_matrix, uint32_t dim);

    /**
     * Upload rotated centroid to device. Called once during index load.
     * @param host_centroid  Rotated centroid vector (dim floats)
     * @param dim            Dimension
     */
    void upload_centroid(const float* host_centroid, uint32_t dim);

    /**
     * Upload the complete RaBitQ parameter set and mark it ready for search.
     */
    void configure_rabitq(const float* host_matrix,
                          const float* host_centroid,
                          uint32_t dim,
                          double t_const);

    /**
     * Set t_const scaling factor (computed on host during index setup).
     */
    void set_t_const(double val) { t_const_ = val; }

    uint32_t dim() const { return dim_; }
    uint32_t max_batch() const { return max_batch_; }
    uint32_t max_R() const { return max_R_; }
    uint32_t rabitq_vec_size() const { return rabitq_vec_size_; }
    bool initialized() const { return initialized_; }
    bool rabitq_ready() const { return rabitq_ready_; }
    bool gpudirect_candidate_ready() const { return gpudirect_candidate_ready_; }
    bool gpudirect_rabitq_ready() const { return gpudirect_rabitq_ready_; }
    GpuRabitqCache& rabitq_cache() { return rabitq_cache_; }
    const GpuRabitqCache& rabitq_cache() const { return rabitq_cache_; }

private:
    CoroutineGpuState* states_{nullptr};
    uint32_t num_coroutines_{0};
    uint32_t dim_{0};
    uint32_t max_batch_{0};
    uint32_t max_R_{0};
    uint32_t rabitq_bits_{0};
    uint32_t rabitq_vec_size_{0};  // bytes per RaBitQ vector

    // Shared device resources
    float* d_rotation_mat_{nullptr};  // [dim * dim]
    float* d_centroid_{nullptr};      // [dim]
    double t_const_{0.0};
    cublasHandle_t cublas_handle_{nullptr};
    bool rabitq_ready_{false};
    bool gpudirect_rdma_enabled_{false};
    bool gpudirect_candidate_ready_{false};
    bool gpudirect_rabitq_ready_{false};
    GpuRabitqCache rabitq_cache_;

    bool initialized_{false};
};

}  // namespace gpu
