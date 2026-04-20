#include "gpu_buffer_manager.hh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

namespace gpu {

GpuBufferManager::~GpuBufferManager() {
    if (initialized_) {
        destroy();
    }
}

void GpuBufferManager::init(uint32_t num_coroutines, uint32_t dim,
                            uint32_t max_batch, uint32_t max_R,
                            uint32_t rabitq_bits,
                            ibv_pd* rdma_pd,
                            bool enable_gpudirect_rdma,
                            size_t gpu_rabitq_cache_bytes) {
    num_coroutines_ = num_coroutines;
    dim_ = dim;
    max_batch_ = max_batch;
    max_R_ = max_R;
    rabitq_bits_ = rabitq_bits;
    rabitq_vec_size_ = (rabitq_bits * dim + 7) / 8 + 2 * sizeof(float);
    rabitq_ready_ = false;
    gpudirect_rdma_enabled_ = false;
    gpudirect_candidate_ready_ = false;
    gpudirect_rabitq_ready_ = false;
    const bool try_gpudirect_rdma = enable_gpudirect_rdma && rdma_pd != nullptr;
    const size_t candidate_bytes = static_cast<size_t>(max_batch) * dim * sizeof(float);
    const size_t rabitq_bytes = static_cast<size_t>(max_batch) * rabitq_vec_size_;
    ibv_pd* pd = try_gpudirect_rdma ? rdma_pd : nullptr;

    states_ = new CoroutineGpuState[num_coroutines];
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));

    for (uint32_t i = 0; i < num_coroutines; ++i) {
        auto& s = states_[i];

        // Create stream (non-blocking to avoid synchronizing with default stream)
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming));

        // Pinned host staging buffers
        CUDA_CHECK(cudaMallocHost(&s.h_query, dim * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&s.h_rot_query, dim * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&s.h_rabitq_vecs, max_batch * rabitq_vec_size_));
        CUDA_CHECK(cudaMallocHost(&s.h_candidate_vecs, max_batch * dim * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&s.h_candidate_dists, max_batch * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&s.h_candidate_order, max_batch * sizeof(uint32_t)));
        CUDA_CHECK(cudaHostAlloc(&s.h_cache_slot_ids, max_batch * sizeof(uint32_t), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&s.d_cache_slot_ids, s.h_cache_slot_ids, 0));
        CUDA_CHECK(cudaMallocHost(&s.h_distances, max_batch * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&s.h_pruned_indices, max_R * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&s.h_pruned_count, sizeof(uint32_t)));

        // Device buffers
        CUDA_CHECK(cudaMalloc(&s.d_query, dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_rot_query, dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_rabitq_vecs, max_batch * rabitq_vec_size_));
        CUDA_CHECK(cudaMalloc(&s.d_candidate_vecs, max_batch * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_candidate_dists, max_batch * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_candidate_order, max_batch * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&s.d_distances, max_batch * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_pruned_indices, max_R * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&s.d_pruned_count, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&s.d_query_factor, 3 * sizeof(float)));  // RabitqQueryFactor

    }

    if (try_gpudirect_rdma) {
        bool candidate_success = true;
        for (uint32_t i = 0; i < num_coroutines; ++i) {
            auto& s = states_[i];
            s.d_candidate_vecs_mr = ibv_reg_mr(pd, s.d_candidate_vecs, candidate_bytes, IBV_ACCESS_LOCAL_WRITE);
            if (!s.d_candidate_vecs_mr) {
                candidate_success = false;
                continue;
            }
            s.d_candidate_vecs_lkey = s.d_candidate_vecs_mr->lkey;
            s.d_candidate_vecs_rdma_registered = true;
        }
        if (!candidate_success) {
            for (uint32_t i = 0; i < num_coroutines; ++i) {
                auto& s = states_[i];
                if (s.d_candidate_vecs_mr) {
                    ibv_dereg_mr(s.d_candidate_vecs_mr);
                    s.d_candidate_vecs_mr = nullptr;
                }
                s.d_candidate_vecs_lkey = 0;
                s.d_candidate_vecs_rdma_registered = false;
            }
        } else {
            gpudirect_candidate_ready_ = true;
        }

        bool rabitq_success = true;
        for (uint32_t i = 0; i < num_coroutines; ++i) {
            auto& s = states_[i];
            s.d_rabitq_vecs_mr = ibv_reg_mr(pd, s.d_rabitq_vecs, rabitq_bytes, IBV_ACCESS_LOCAL_WRITE);
            if (!s.d_rabitq_vecs_mr) {
                rabitq_success = false;
                continue;
            }
            s.d_rabitq_vecs_lkey = s.d_rabitq_vecs_mr->lkey;
            s.d_rabitq_vecs_rdma_registered = true;
        }
        if (!rabitq_success) {
            for (uint32_t i = 0; i < num_coroutines; ++i) {
                auto& s = states_[i];
                if (s.d_rabitq_vecs_mr) {
                    ibv_dereg_mr(s.d_rabitq_vecs_mr);
                    s.d_rabitq_vecs_mr = nullptr;
                }
                s.d_rabitq_vecs_lkey = 0;
                s.d_rabitq_vecs_rdma_registered = false;
            }
        } else {
            gpudirect_rabitq_ready_ = true;
        }

        gpudirect_rdma_enabled_ = gpudirect_candidate_ready_ || gpudirect_rabitq_ready_;
        if (gpudirect_rdma_enabled_) {
            std::fprintf(stderr, "[GPUDirect RDMA] enabled for %u coroutine GPU staging buffers\n", num_coroutines);
        }
    }

    if (gpudirect_rabitq_ready_ && gpu_rabitq_cache_bytes > 0) {
        if (!rabitq_cache_.init(gpu_rabitq_cache_bytes, rabitq_vec_size_, pd)) {
            std::fprintf(stderr, "[GPU RaBitQ cache] disabled; falling back to staging buffers\n");
        }
    }

    // Allocate and initialize default rotation matrix (identity) and zero centroid
    // These will be overwritten when an index is loaded.
    {
        size_t mat_bytes = dim * dim * sizeof(float);
        float* h_identity = new float[dim * dim]();
        for (uint32_t i = 0; i < dim; ++i) {
            h_identity[i * dim + i] = 1.0f;  // identity matrix (column-major)
        }
        CUDA_CHECK(cudaMalloc(&d_rotation_mat_, mat_bytes));
        CUDA_CHECK(cudaMemcpy(d_rotation_mat_, h_identity, mat_bytes, cudaMemcpyHostToDevice));
        delete[] h_identity;

        size_t vec_bytes = dim * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_centroid_, vec_bytes));
        CUDA_CHECK(cudaMemset(d_centroid_, 0, vec_bytes));
    }

    initialized_ = true;
}

void GpuBufferManager::destroy() {
    if (!initialized_) return;

    for (uint32_t i = 0; i < num_coroutines_; ++i) {
        auto& s = states_[i];

        // Device buffers
        if (s.d_query) cudaFree(s.d_query);
        if (s.d_rot_query) cudaFree(s.d_rot_query);
        if (s.d_rabitq_vecs) cudaFree(s.d_rabitq_vecs);
        if (s.d_candidate_vecs) cudaFree(s.d_candidate_vecs);
        if (s.d_candidate_dists) cudaFree(s.d_candidate_dists);
        if (s.d_candidate_order) cudaFree(s.d_candidate_order);
        if (s.d_distances) cudaFree(s.d_distances);
        if (s.d_pruned_indices) cudaFree(s.d_pruned_indices);
        if (s.d_pruned_count) cudaFree(s.d_pruned_count);
        if (s.d_query_factor) cudaFree(s.d_query_factor);
        if (s.d_candidate_vecs_mr) ibv_dereg_mr(s.d_candidate_vecs_mr);
        if (s.d_rabitq_vecs_mr) ibv_dereg_mr(s.d_rabitq_vecs_mr);

        // Pinned host buffers
        if (s.h_query) cudaFreeHost(s.h_query);
        if (s.h_rot_query) cudaFreeHost(s.h_rot_query);
        if (s.h_rabitq_vecs) cudaFreeHost(s.h_rabitq_vecs);
        if (s.h_candidate_vecs) cudaFreeHost(s.h_candidate_vecs);
        if (s.h_candidate_dists) cudaFreeHost(s.h_candidate_dists);
        if (s.h_candidate_order) cudaFreeHost(s.h_candidate_order);
        if (s.h_cache_slot_ids) cudaFreeHost(s.h_cache_slot_ids);
        if (s.h_distances) cudaFreeHost(s.h_distances);
        if (s.h_pruned_indices) cudaFreeHost(s.h_pruned_indices);
        if (s.h_pruned_count) cudaFreeHost(s.h_pruned_count);

        // Stream and event
        if (s.event) cudaEventDestroy(s.event);
        if (s.stream) cudaStreamDestroy(s.stream);
    }

    // Shared resources
    if (d_rotation_mat_) cudaFree(d_rotation_mat_);
    if (d_centroid_) cudaFree(d_centroid_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
    rabitq_cache_.destroy();

    delete[] states_;
    states_ = nullptr;
    cublas_handle_ = nullptr;
    rabitq_ready_ = false;
    gpudirect_rdma_enabled_ = false;
    gpudirect_candidate_ready_ = false;
    gpudirect_rabitq_ready_ = false;
    initialized_ = false;
}

void GpuBufferManager::upload_rotation_matrix(const float* host_matrix, uint32_t dim) {
    size_t bytes = dim * dim * sizeof(float);
    if (!d_rotation_mat_) {
        CUDA_CHECK(cudaMalloc(&d_rotation_mat_, bytes));
    }
    CUDA_CHECK(cudaMemcpy(d_rotation_mat_, host_matrix, bytes, cudaMemcpyHostToDevice));
}

void GpuBufferManager::upload_centroid(const float* host_centroid, uint32_t dim) {
    size_t bytes = dim * sizeof(float);
    if (!d_centroid_) {
        CUDA_CHECK(cudaMalloc(&d_centroid_, bytes));
    }
    CUDA_CHECK(cudaMemcpy(d_centroid_, host_centroid, bytes, cudaMemcpyHostToDevice));
}

void GpuBufferManager::configure_rabitq(const float* host_matrix,
                                        const float* host_centroid,
                                        uint32_t dim,
                                        double t_const) {
    upload_rotation_matrix(host_matrix, dim);
    upload_centroid(host_centroid, dim);
    set_t_const(t_const);
    rabitq_ready_ = true;
}

}  // namespace gpu
