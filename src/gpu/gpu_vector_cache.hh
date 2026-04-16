#pragma once

/**
 * GPU Vector Cache: per compute-thread GPU-resident vector cache.
 *
 * Maintains two independent pools in GPU device memory:
 *  - Float vector pool (for exact L2 / rerank)
 *  - RaBitQ vector pool (for approximate distance)
 *
 * Host-side hash tables provide O(1) lookup by raw address key (uint64_t).
 * Clock (second-chance FIFO) eviction keeps hot vectors resident.
 *
 * Each ComputeThread owns one instance — cooperative coroutine
 * scheduling within a thread means no concurrent host access, no locks.
 */

#include <cstdint>
#include <unordered_map>
#include <vector>

struct ibv_mr;
struct ibv_pd;

// Forward-declare CUDA types
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace gpu {

using CacheKey = uint64_t;

enum class CacheSlotState : uint8_t {
    empty = 0,
    filling = 1,
    ready = 2,
};

struct GpuVectorCacheStats {
    size_t vec_hits{0};
    size_t vec_misses{0};
    size_t rabitq_hits{0};
    size_t rabitq_misses{0};
    size_t vec_evictions{0};
    size_t rabitq_evictions{0};
};

class GpuVectorCache {
public:
    GpuVectorCache() = default;
    ~GpuVectorCache();

    GpuVectorCache(const GpuVectorCache&) = delete;
    GpuVectorCache& operator=(const GpuVectorCache&) = delete;
    GpuVectorCache(GpuVectorCache&&) = delete;
    GpuVectorCache& operator=(GpuVectorCache&&) = delete;

    void init(uint32_t vec_capacity, uint32_t rabitq_capacity,
              uint32_t dim, uint32_t rabitq_vec_size,
              ibv_pd* rdma_pd = nullptr,
              bool enable_gpudirect_rdma = false);

    void destroy();

    // --- Legacy float vector cache API ---
    int32_t lookup_vec(CacheKey key);
    void gather_vec(int32_t slot, float* d_dst, uint32_t dst_offset,
                    cudaStream_t stream) const;
    void insert_vec(CacheKey key, const float* d_src, uint32_t src_offset,
                    cudaStream_t stream);

    // --- Legacy RaBitQ cache API ---
    int32_t lookup_rabitq(CacheKey key);
    void gather_rabitq(int32_t slot, uint8_t* d_dst, uint32_t dst_offset,
                       cudaStream_t stream) const;
    void insert_rabitq(CacheKey key, const uint8_t* d_src, uint32_t src_offset,
                       cudaStream_t stream);

    // --- Query-path direct slot management ---
    int32_t probe_vec_slot(CacheKey key);
    int32_t probe_rabitq_slot(CacheKey key);

    int32_t reserve_vec_slot(CacheKey key);
    int32_t reserve_rabitq_slot(CacheKey key);

    void publish_vec_slot(CacheKey key, uint32_t slot);
    void publish_rabitq_slot(CacheKey key, uint32_t slot);

    void release_vec_slot(uint32_t slot);
    void release_rabitq_slot(uint32_t slot);
    void release_vec_slots(const std::vector<uint32_t>& slots);
    void release_rabitq_slots(const std::vector<uint32_t>& slots);

    float* vec_pool_base() const { return d_vec_pool_; }
    uint8_t* rabitq_pool_base() const { return d_rabitq_pool_; }
    float* vec_slot_ptr(uint32_t slot) const;
    uint8_t* rabitq_slot_ptr(uint32_t slot) const;

    uint32_t vec_stride_bytes() const { return dim_ * sizeof(float); }
    uint32_t rabitq_stride_bytes() const { return rabitq_vec_size_; }

    uint32_t vec_pool_lkey() const { return vec_pool_lkey_; }
    uint32_t rabitq_pool_lkey() const { return rabitq_pool_lkey_; }
    bool vec_rdma_registered() const { return vec_pool_rdma_registered_; }
    bool rabitq_rdma_registered() const { return rabitq_pool_rdma_registered_; }

    bool vec_cache_enabled() const { return vec_capacity_ > 0; }
    bool rabitq_cache_enabled() const { return rabitq_capacity_ > 0; }
    bool initialized() const { return initialized_; }
    GpuVectorCacheStats& stats() { return stats_; }
    const GpuVectorCacheStats& stats() const { return stats_; }

private:
    std::unordered_map<CacheKey, uint32_t> vec_map_;
    std::vector<CacheKey> vec_keys_;
    std::vector<bool> vec_ref_bits_;
    std::vector<uint32_t> vec_pin_counts_;
    std::vector<CacheSlotState> vec_states_;
    uint32_t vec_clock_hand_{0};
    uint32_t vec_count_{0};

    std::unordered_map<CacheKey, uint32_t> rabitq_map_;
    std::vector<CacheKey> rabitq_keys_;
    std::vector<bool> rabitq_ref_bits_;
    std::vector<uint32_t> rabitq_pin_counts_;
    std::vector<CacheSlotState> rabitq_states_;
    uint32_t rabitq_clock_hand_{0};
    uint32_t rabitq_count_{0};

    float* d_vec_pool_{nullptr};
    uint8_t* d_rabitq_pool_{nullptr};
    ibv_mr* vec_pool_mr_{nullptr};
    ibv_mr* rabitq_pool_mr_{nullptr};
    uint32_t vec_pool_lkey_{0};
    uint32_t rabitq_pool_lkey_{0};
    bool vec_pool_rdma_registered_{false};
    bool rabitq_pool_rdma_registered_{false};

    uint32_t vec_capacity_{0};
    uint32_t rabitq_capacity_{0};
    uint32_t dim_{0};
    uint32_t rabitq_vec_size_{0};

    GpuVectorCacheStats stats_;
    bool initialized_{false};

    int32_t reserve_vec_slot_impl(CacheKey key, bool pin_slot);
    int32_t reserve_rabitq_slot_impl(CacheKey key, bool pin_slot);
    int32_t evict_vec_slot();
    int32_t evict_rabitq_slot();
};

}  // namespace gpu
