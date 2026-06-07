#include "gpu_buffer_manager.hh"

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            abort();                                                           \
        }                                                                      \
    } while (0)

namespace gpu {

GpuBufferManager::~GpuBufferManager() {
    if (initialized_) {
        destroy();
    }
}

void GpuBufferManager::init(uint32_t num_coroutines,
                            uint32_t dim,
                            uint32_t max_batch,
                            uint32_t max_R,
                            size_t query_vector_bytes,
                            size_t candidate_vector_bytes,
                            ibv_pd* rdma_pd,
                            bool enable_gpudirect_rdma) {
    num_coroutines_ = num_coroutines;
    dim_ = dim;
    max_batch_ = max_batch;
    max_R_ = max_R;
    query_vector_bytes_ = query_vector_bytes == 0 ? static_cast<size_t>(dim) * sizeof(float)
                                                  : query_vector_bytes;
    candidate_vector_bytes_ = candidate_vector_bytes == 0 ? static_cast<size_t>(dim) * sizeof(float)
                                                          : candidate_vector_bytes;
    gpudirect_rdma_enabled_ = false;
    gpudirect_candidate_ready_ = false;

    const bool try_gpudirect_rdma = enable_gpudirect_rdma && rdma_pd != nullptr;
    const size_t candidate_bytes = static_cast<size_t>(max_batch) * candidate_vector_bytes_;
    ibv_pd* pd = try_gpudirect_rdma ? rdma_pd : nullptr;

    states_ = new CoroutineGpuState[num_coroutines];

    for (uint32_t i = 0; i < num_coroutines; ++i) {
        auto& s = states_[i];

        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming));

        CUDA_CHECK(cudaMallocHost(&s.h_query, query_vector_bytes_));
        CUDA_CHECK(cudaMallocHost(&s.h_candidate_vecs, max_batch * candidate_vector_bytes_));
        CUDA_CHECK(cudaMallocHost(&s.h_candidate_dists, max_batch * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&s.h_candidate_order, max_batch * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&s.h_distances, max_batch * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s.h_candidate_ptrs), max_batch * sizeof(void*)));
        CUDA_CHECK(cudaMallocHost(&s.h_pruned_indices, max_R * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&s.h_pruned_count, sizeof(uint32_t)));

        CUDA_CHECK(cudaMalloc(&s.d_query, query_vector_bytes_));
        CUDA_CHECK(cudaMalloc(&s.d_candidate_vecs, max_batch * candidate_vector_bytes_));
        CUDA_CHECK(cudaMalloc(&s.d_candidate_vecs_alt, max_batch * candidate_vector_bytes_));
        CUDA_CHECK(cudaMalloc(&s.d_candidate_dists, max_batch * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_candidate_order, max_batch * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&s.d_distances, max_batch * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.d_candidate_ptrs), max_batch * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&s.d_pruned_indices, max_R * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&s.d_pruned_count, sizeof(uint32_t)));
    }

    if (try_gpudirect_rdma) {
        bool candidate_success = true;
        for (uint32_t i = 0; i < num_coroutines; ++i) {
            auto& s = states_[i];
            s.d_candidate_vecs_mr = ibv_reg_mr(pd, s.d_candidate_vecs, candidate_bytes, IBV_ACCESS_LOCAL_WRITE);
            s.d_candidate_vecs_alt_mr = ibv_reg_mr(pd, s.d_candidate_vecs_alt, candidate_bytes, IBV_ACCESS_LOCAL_WRITE);
            if (!s.d_candidate_vecs_mr || !s.d_candidate_vecs_alt_mr) {
                candidate_success = false;
                continue;
            }
            s.d_candidate_vecs_lkey = s.d_candidate_vecs_mr->lkey;
            s.d_candidate_vecs_alt_lkey = s.d_candidate_vecs_alt_mr->lkey;
            s.d_candidate_vecs_rdma_registered = true;
            s.d_candidate_vecs_alt_rdma_registered = true;
        }
        if (!candidate_success) {
            for (uint32_t i = 0; i < num_coroutines; ++i) {
                auto& s = states_[i];
                if (s.d_candidate_vecs_mr) {
                    ibv_dereg_mr(s.d_candidate_vecs_mr);
                    s.d_candidate_vecs_mr = nullptr;
                }
                if (s.d_candidate_vecs_alt_mr) {
                    ibv_dereg_mr(s.d_candidate_vecs_alt_mr);
                    s.d_candidate_vecs_alt_mr = nullptr;
                }
                s.d_candidate_vecs_lkey = 0;
                s.d_candidate_vecs_alt_lkey = 0;
                s.d_candidate_vecs_rdma_registered = false;
                s.d_candidate_vecs_alt_rdma_registered = false;
            }
        } else {
            gpudirect_candidate_ready_ = true;
            gpudirect_rdma_enabled_ = true;
            std::fprintf(stderr, "[GPUDirect RDMA] enabled for %u coroutine candidate-vector buffers\n",
                         num_coroutines);
        }
    }

    initialized_ = true;
}

void GpuBufferManager::destroy() {
    if (!initialized_) return;

    for (uint32_t i = 0; i < num_coroutines_; ++i) {
        auto& s = states_[i];

        if (s.d_query) cudaFree(s.d_query);
        if (s.d_candidate_vecs) cudaFree(s.d_candidate_vecs);
        if (s.d_candidate_vecs_alt) cudaFree(s.d_candidate_vecs_alt);
        if (s.d_candidate_dists) cudaFree(s.d_candidate_dists);
        if (s.d_candidate_order) cudaFree(s.d_candidate_order);
        if (s.d_distances) cudaFree(s.d_distances);
        if (s.d_candidate_ptrs) cudaFree(s.d_candidate_ptrs);
        if (s.d_pruned_indices) cudaFree(s.d_pruned_indices);
        if (s.d_pruned_count) cudaFree(s.d_pruned_count);
        if (s.d_candidate_vecs_mr) ibv_dereg_mr(s.d_candidate_vecs_mr);
        if (s.d_candidate_vecs_alt_mr) ibv_dereg_mr(s.d_candidate_vecs_alt_mr);

        if (s.h_query) cudaFreeHost(s.h_query);
        if (s.h_candidate_vecs) cudaFreeHost(s.h_candidate_vecs);
        if (s.h_candidate_dists) cudaFreeHost(s.h_candidate_dists);
        if (s.h_candidate_order) cudaFreeHost(s.h_candidate_order);
        if (s.h_distances) cudaFreeHost(s.h_distances);
        if (s.h_candidate_ptrs) cudaFreeHost(s.h_candidate_ptrs);
        if (s.h_pruned_indices) cudaFreeHost(s.h_pruned_indices);
        if (s.h_pruned_count) cudaFreeHost(s.h_pruned_count);

        if (s.event) cudaEventDestroy(s.event);
        if (s.stream) cudaStreamDestroy(s.stream);
    }

    delete[] states_;
    states_ = nullptr;
    gpudirect_rdma_enabled_ = false;
    gpudirect_candidate_ready_ = false;
    initialized_ = false;
}

}  // namespace gpu
