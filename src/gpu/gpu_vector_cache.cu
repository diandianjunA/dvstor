#include "gpu_vector_cache.hh"

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

GpuVectorCache::~GpuVectorCache() {
    if (initialized_) {
        destroy();
    }
}

void GpuVectorCache::init(uint32_t vec_capacity, uint32_t rabitq_capacity,
                           uint32_t dim, uint32_t rabitq_vec_size,
                           ibv_pd* rdma_pd,
                           bool enable_gpudirect_rdma) {
    vec_capacity_ = vec_capacity;
    rabitq_capacity_ = rabitq_capacity;
    dim_ = dim;
    rabitq_vec_size_ = rabitq_vec_size;
    vec_pool_rdma_registered_ = false;
    rabitq_pool_rdma_registered_ = false;
    vec_pool_lkey_ = 0;
    rabitq_pool_lkey_ = 0;

    if (vec_capacity_ > 0) {
        const size_t vec_pool_bytes = static_cast<size_t>(vec_capacity_) * dim_ * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_vec_pool_, vec_pool_bytes));

        vec_keys_.resize(vec_capacity_);
        vec_ref_bits_.resize(vec_capacity_, false);
        vec_pin_counts_.resize(vec_capacity_, 0);
        vec_states_.resize(vec_capacity_, CacheSlotState::empty);
        vec_map_.reserve(vec_capacity_);

        if (enable_gpudirect_rdma && rdma_pd != nullptr) {
            vec_pool_mr_ = ibv_reg_mr(rdma_pd, d_vec_pool_, vec_pool_bytes, IBV_ACCESS_LOCAL_WRITE);
            if (vec_pool_mr_) {
                vec_pool_lkey_ = vec_pool_mr_->lkey;
                vec_pool_rdma_registered_ = true;
            }
        }
    }

    if (rabitq_capacity_ > 0) {
        const size_t rabitq_pool_bytes = static_cast<size_t>(rabitq_capacity_) * rabitq_vec_size_;
        CUDA_CHECK(cudaMalloc(&d_rabitq_pool_, rabitq_pool_bytes));

        rabitq_keys_.resize(rabitq_capacity_);
        rabitq_ref_bits_.resize(rabitq_capacity_, false);
        rabitq_pin_counts_.resize(rabitq_capacity_, 0);
        rabitq_states_.resize(rabitq_capacity_, CacheSlotState::empty);
        rabitq_map_.reserve(rabitq_capacity_);

        if (enable_gpudirect_rdma && rdma_pd != nullptr) {
            rabitq_pool_mr_ = ibv_reg_mr(rdma_pd, d_rabitq_pool_, rabitq_pool_bytes, IBV_ACCESS_LOCAL_WRITE);
            if (rabitq_pool_mr_) {
                rabitq_pool_lkey_ = rabitq_pool_mr_->lkey;
                rabitq_pool_rdma_registered_ = true;
            }
        }
    }

    stats_ = GpuVectorCacheStats{};
    initialized_ = true;
}

void GpuVectorCache::destroy() {
    if (vec_pool_mr_) {
        ibv_dereg_mr(vec_pool_mr_);
        vec_pool_mr_ = nullptr;
    }
    if (rabitq_pool_mr_) {
        ibv_dereg_mr(rabitq_pool_mr_);
        rabitq_pool_mr_ = nullptr;
    }

    if (d_vec_pool_) {
        CUDA_CHECK(cudaFree(d_vec_pool_));
        d_vec_pool_ = nullptr;
    }
    if (d_rabitq_pool_) {
        CUDA_CHECK(cudaFree(d_rabitq_pool_));
        d_rabitq_pool_ = nullptr;
    }

    vec_map_.clear();
    vec_keys_.clear();
    vec_ref_bits_.clear();
    vec_pin_counts_.clear();
    vec_states_.clear();
    vec_count_ = 0;
    vec_clock_hand_ = 0;

    rabitq_map_.clear();
    rabitq_keys_.clear();
    rabitq_ref_bits_.clear();
    rabitq_pin_counts_.clear();
    rabitq_states_.clear();
    rabitq_count_ = 0;
    rabitq_clock_hand_ = 0;

    vec_pool_lkey_ = 0;
    rabitq_pool_lkey_ = 0;
    vec_pool_rdma_registered_ = false;
    rabitq_pool_rdma_registered_ = false;
    initialized_ = false;
}

int32_t GpuVectorCache::probe_vec_slot(CacheKey key) {
    auto it = vec_map_.find(key);
    if (it == vec_map_.end()) {
        ++stats_.vec_misses;
        return -1;
    }

    const uint32_t slot = it->second;
    if (vec_states_[slot] != CacheSlotState::ready) {
        ++stats_.vec_misses;
        return -1;
    }

    vec_ref_bits_[slot] = true;
    ++vec_pin_counts_[slot];
    ++stats_.vec_hits;
    return static_cast<int32_t>(slot);
}

int32_t GpuVectorCache::probe_rabitq_slot(CacheKey key) {
    auto it = rabitq_map_.find(key);
    if (it == rabitq_map_.end()) {
        ++stats_.rabitq_misses;
        return -1;
    }

    const uint32_t slot = it->second;
    if (rabitq_states_[slot] != CacheSlotState::ready) {
        ++stats_.rabitq_misses;
        return -1;
    }

    rabitq_ref_bits_[slot] = true;
    ++rabitq_pin_counts_[slot];
    ++stats_.rabitq_hits;
    return static_cast<int32_t>(slot);
}

int32_t GpuVectorCache::lookup_vec(CacheKey key) {
    auto it = vec_map_.find(key);
    if (it == vec_map_.end()) {
        ++stats_.vec_misses;
        return -1;
    }

    const uint32_t slot = it->second;
    if (vec_states_[slot] != CacheSlotState::ready) {
        ++stats_.vec_misses;
        return -1;
    }

    vec_ref_bits_[slot] = true;
    ++stats_.vec_hits;
    return static_cast<int32_t>(slot);
}

void GpuVectorCache::gather_vec(int32_t slot, float* d_dst, uint32_t dst_offset,
                                 cudaStream_t stream) const {
    const size_t bytes = dim_ * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(
        d_dst + static_cast<size_t>(dst_offset) * dim_,
        d_vec_pool_ + static_cast<size_t>(slot) * dim_,
        bytes, cudaMemcpyDeviceToDevice, stream));
}

void GpuVectorCache::insert_vec(CacheKey key, const float* d_src,
                                 uint32_t src_offset, cudaStream_t stream) {
    int32_t slot = reserve_vec_slot_impl(key, false);
    if (slot < 0) {
        return;
    }

    const size_t bytes = dim_ * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(
        d_vec_pool_ + static_cast<size_t>(slot) * dim_,
        d_src + static_cast<size_t>(src_offset) * dim_,
        bytes, cudaMemcpyDeviceToDevice, stream));
    publish_vec_slot(key, static_cast<uint32_t>(slot));
}

int32_t GpuVectorCache::lookup_rabitq(CacheKey key) {
    auto it = rabitq_map_.find(key);
    if (it == rabitq_map_.end()) {
        ++stats_.rabitq_misses;
        return -1;
    }

    const uint32_t slot = it->second;
    if (rabitq_states_[slot] != CacheSlotState::ready) {
        ++stats_.rabitq_misses;
        return -1;
    }

    rabitq_ref_bits_[slot] = true;
    ++stats_.rabitq_hits;
    return static_cast<int32_t>(slot);
}

void GpuVectorCache::gather_rabitq(int32_t slot, uint8_t* d_dst,
                                    uint32_t dst_offset,
                                    cudaStream_t stream) const {
    const size_t bytes = rabitq_vec_size_;
    CUDA_CHECK(cudaMemcpyAsync(
        d_dst + static_cast<size_t>(dst_offset) * rabitq_vec_size_,
        d_rabitq_pool_ + static_cast<size_t>(slot) * rabitq_vec_size_,
        bytes, cudaMemcpyDeviceToDevice, stream));
}

void GpuVectorCache::insert_rabitq(CacheKey key, const uint8_t* d_src,
                                    uint32_t src_offset, cudaStream_t stream) {
    int32_t slot = reserve_rabitq_slot_impl(key, false);
    if (slot < 0) {
        return;
    }

    const size_t bytes = rabitq_vec_size_;
    CUDA_CHECK(cudaMemcpyAsync(
        d_rabitq_pool_ + static_cast<size_t>(slot) * rabitq_vec_size_,
        d_src + static_cast<size_t>(src_offset) * rabitq_vec_size_,
        bytes, cudaMemcpyDeviceToDevice, stream));
    publish_rabitq_slot(key, static_cast<uint32_t>(slot));
}

int32_t GpuVectorCache::reserve_vec_slot(CacheKey key) {
    return reserve_vec_slot_impl(key, true);
}

int32_t GpuVectorCache::reserve_rabitq_slot(CacheKey key) {
    return reserve_rabitq_slot_impl(key, true);
}

void GpuVectorCache::publish_vec_slot(CacheKey key, uint32_t slot) {
    vec_keys_[slot] = key;
    vec_map_[key] = slot;
    vec_ref_bits_[slot] = true;
    vec_states_[slot] = CacheSlotState::ready;
}

void GpuVectorCache::publish_rabitq_slot(CacheKey key, uint32_t slot) {
    rabitq_keys_[slot] = key;
    rabitq_map_[key] = slot;
    rabitq_ref_bits_[slot] = true;
    rabitq_states_[slot] = CacheSlotState::ready;
}

void GpuVectorCache::release_vec_slot(uint32_t slot) {
    if (slot < vec_pin_counts_.size() && vec_pin_counts_[slot] > 0) {
        --vec_pin_counts_[slot];
    }
}

void GpuVectorCache::release_rabitq_slot(uint32_t slot) {
    if (slot < rabitq_pin_counts_.size() && rabitq_pin_counts_[slot] > 0) {
        --rabitq_pin_counts_[slot];
    }
}

void GpuVectorCache::release_vec_slots(const std::vector<uint32_t>& slots) {
    for (uint32_t slot : slots) {
        release_vec_slot(slot);
    }
}

void GpuVectorCache::release_rabitq_slots(const std::vector<uint32_t>& slots) {
    for (uint32_t slot : slots) {
        release_rabitq_slot(slot);
    }
}

float* GpuVectorCache::vec_slot_ptr(uint32_t slot) const {
    return d_vec_pool_ + static_cast<size_t>(slot) * dim_;
}

uint8_t* GpuVectorCache::rabitq_slot_ptr(uint32_t slot) const {
    return d_rabitq_pool_ + static_cast<size_t>(slot) * rabitq_vec_size_;
}

int32_t GpuVectorCache::reserve_vec_slot_impl(CacheKey key, bool pin_slot) {
    auto it = vec_map_.find(key);
    if (it != vec_map_.end()) {
        const uint32_t slot = it->second;
        if (vec_states_[slot] == CacheSlotState::ready) {
            vec_ref_bits_[slot] = true;
            if (pin_slot) {
                ++vec_pin_counts_[slot];
            }
            return static_cast<int32_t>(slot);
        }
    }

    uint32_t slot;
    if (vec_count_ < vec_capacity_) {
        slot = vec_count_++;
    } else {
        const int32_t evicted = evict_vec_slot();
        if (evicted < 0) {
            return -1;
        }
        slot = static_cast<uint32_t>(evicted);
    }

    vec_keys_[slot] = key;
    vec_ref_bits_[slot] = true;
    vec_states_[slot] = CacheSlotState::filling;
    vec_pin_counts_[slot] = pin_slot ? 1u : 0u;
    return static_cast<int32_t>(slot);
}

int32_t GpuVectorCache::reserve_rabitq_slot_impl(CacheKey key, bool pin_slot) {
    auto it = rabitq_map_.find(key);
    if (it != rabitq_map_.end()) {
        const uint32_t slot = it->second;
        if (rabitq_states_[slot] == CacheSlotState::ready) {
            rabitq_ref_bits_[slot] = true;
            if (pin_slot) {
                ++rabitq_pin_counts_[slot];
            }
            return static_cast<int32_t>(slot);
        }
    }

    uint32_t slot;
    if (rabitq_count_ < rabitq_capacity_) {
        slot = rabitq_count_++;
    } else {
        const int32_t evicted = evict_rabitq_slot();
        if (evicted < 0) {
            return -1;
        }
        slot = static_cast<uint32_t>(evicted);
    }

    rabitq_keys_[slot] = key;
    rabitq_ref_bits_[slot] = true;
    rabitq_states_[slot] = CacheSlotState::filling;
    rabitq_pin_counts_[slot] = pin_slot ? 1u : 0u;
    return static_cast<int32_t>(slot);
}

int32_t GpuVectorCache::evict_vec_slot() {
    if (vec_capacity_ == 0) {
        return -1;
    }

    for (uint32_t i = 0; i < 2 * vec_capacity_; ++i) {
        const uint32_t hand = vec_clock_hand_;
        vec_clock_hand_ = (vec_clock_hand_ + 1) % vec_capacity_;

        if (vec_states_[hand] != CacheSlotState::ready || vec_pin_counts_[hand] > 0) {
            continue;
        }
        if (vec_ref_bits_[hand]) {
            vec_ref_bits_[hand] = false;
            continue;
        }

        vec_map_.erase(vec_keys_[hand]);
        vec_states_[hand] = CacheSlotState::empty;
        ++stats_.vec_evictions;
        return static_cast<int32_t>(hand);
    }
    return -1;
}

int32_t GpuVectorCache::evict_rabitq_slot() {
    if (rabitq_capacity_ == 0) {
        return -1;
    }

    for (uint32_t i = 0; i < 2 * rabitq_capacity_; ++i) {
        const uint32_t hand = rabitq_clock_hand_;
        rabitq_clock_hand_ = (rabitq_clock_hand_ + 1) % rabitq_capacity_;

        if (rabitq_states_[hand] != CacheSlotState::ready || rabitq_pin_counts_[hand] > 0) {
            continue;
        }
        if (rabitq_ref_bits_[hand]) {
            rabitq_ref_bits_[hand] = false;
            continue;
        }

        rabitq_map_.erase(rabitq_keys_[hand]);
        rabitq_states_[hand] = CacheSlotState::empty;
        ++stats_.rabitq_evictions;
        return static_cast<int32_t>(hand);
    }
    return -1;
}

}  // namespace gpu
