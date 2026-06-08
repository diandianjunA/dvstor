#pragma once

/**
 * GpuVectorCache: GPU-resident vector cache for Vamana graph traversal.
 *
 * Stores full-precision or quantized vectors in GPU memory to eliminate
 * redundant RDMA reads for hot nodes during beam search.
 *
 * Design (mirrors NeighborCache pattern):
 *  - CPU-side sharded metadata: open-addressing hash table with per-slot
 *    { atomic<u64> tag, atomic<u32> seq }. Vector data lives on GPU only.
 *  - Three-state tag: bits[63:3]=key, bit[2]=reserved, bit[1]=referenced,
 *    bit[0]=valid. RemotePtr byte_offsets are 8-byte aligned, so bit 2 is
 *    always 0 in a valid key — safe to use as a reservation flag.
 *  - Lock-free reads via seqlock (tag+seq validation).
 *  - Writes (allocate_slot, commit_slot) are serialized per shard
 *    via try_lock mutex.
 *  - Eviction: per-shard CLOCK (second-chance) bounded to 128 probes.
 *  - GPUDirect RDMA: entire GPU buffer is registered as an RDMA MR so
 *    batch_read_vectors can write directly into cache slots.
 *
 * Integration with search loop:
 *  1. find(key) → cache hit → use gpu_slot_ptr(slot_id) in h_candidate_ptrs
 *  2. allocate_slot(key) → reserve slot → RDMA directly to gpu_slot_addr()
 *  3. After RDMA completes → commit_slot(slot_id, key)
 *
 * The kTagReserved bit protects in-flight slots from CLOCK eviction during
 * the window between allocate_slot and commit_slot. This enables zero-copy
 * GPUDirect RDMA directly into cache slots.
 */

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/types.hh"
#include "remote_pointer.hh"

class GpuVectorCache {
public:
    static constexpr size_t kDefaultNumShards = 16;
    static constexpr size_t kMaxClockProbes = 128;
    // A Reserved slot whose alloc_epoch lags the shard epoch by more than this
    // is considered abandoned and eligible for CLOCK eviction.
    static constexpr u32 kStaleEpochDelta = 2;

    // Tag flag bits: RemotePtr byte_offsets are 8-byte aligned, so
    // bits [2:0] of raw_address are always 0 — safe to use as flags.
    // NeighborCache uses bits 0 (valid) and 1 (referenced).
    // We add bit 2 (reserved) for two-phase GPUDirect RDMA insertion.
    static constexpr u64 kTagValid      = 1ULL << 0;
    static constexpr u64 kTagReferenced = 1ULL << 1;
    static constexpr u64 kTagReserved   = 1ULL << 2;
    static constexpr u64 kTagFlagMask   = kTagValid | kTagReferenced | kTagReserved;
    static constexpr u64 kTagKeyMask    = ~kTagFlagMask;

    GpuVectorCache() = default;
    ~GpuVectorCache();

    GpuVectorCache(const GpuVectorCache&) = delete;
    GpuVectorCache& operator=(const GpuVectorCache&) = delete;
    GpuVectorCache(GpuVectorCache&&) = delete;
    GpuVectorCache& operator=(GpuVectorCache&&) = delete;

    // Initialize GPU buffer + CPU metadata.
    // pds: all unique RDMA protection domains across SharedContexts.
    // The GPU buffer is registered with each PD so every thread can
    // use GPUDirect RDMA into cache slots regardless of which
    // SharedContext it belongs to.
    void init(size_t total_slots, uint32_t vector_bytes,
              const std::vector<ibv_pd*>& pds = {});

    // Release GPU buffer, RDMA MR, and CPU metadata.
    void destroy();

    // ── Lock-free read (critical hot path) ──────────────────────────
    // Returns slot_id (>= 0) on hit, -1 on miss.
    int32_t find(RemotePtr key);

    // ── Two-phase insertion for GPUDirect RDMA ──────────────────────
    // Phase 1: reserve a slot (sets tag to key|kTagReserved, protected
    //          from eviction). Returns slot_id for use as RDMA dest.
    //          Returns -1 if try_lock fails (caller falls back to staging).
    // Phase 2: after RDMA completes, call commit_slot() to mark valid.
    int32_t allocate_slot(RemotePtr key);
    void commit_slot(int32_t slot_id, RemotePtr key);

    // ── GPU buffer accessors ────────────────────────────────────────
    uint8_t* gpu_buffer() const { return gpu_buffer_; }
    uint8_t* gpu_slot_ptr(int32_t slot_id) const;
    u64 gpu_slot_addr(int32_t slot_id) const;
    // Return the lkey registered for a specific PD (required when the
    // compute node has multiple SharedContexts, each with its own PD).
    uint32_t gpu_buffer_lkey(ibv_pd* pd) const;
    // Convenience: first registered lkey (valid when only one PD exists).
    uint32_t gpu_buffer_lkey() const { return gpu_buffer_lkey_0_; }
    bool gpu_buffer_registered() const { return !pd_lkeys_.empty(); }
    size_t total_slots() const { return total_slots_; }
    size_t slot_stride() const { return slot_stride_; }

    // ── Cache invalidation ──────────────────────────────────────────
    // Lock-free: atomically clears valid bit. Preserves probe chain.
    void invalidate(RemotePtr key);

    // Clear all entries (acquires all shard mutexes).
    void invalidate_all();

    // ── Statistics ──────────────────────────────────────────────────
    struct Stats {
        u64 hits{0};
        u64 misses{0};
        u64 evictions{0};
        u64 inserts{0};
        u64 alloc_failures{0};
    };
    Stats stats() const;

private:
    static u64 hash_key(RemotePtr key);
    size_t shard_index(RemotePtr key) const;

    static u64 make_reserved_tag(RemotePtr key) {
        return (key.raw_address & kTagKeyMask) | kTagReserved;
    }
    static u64 make_valid_tag(RemotePtr key) {
        return (key.raw_address & kTagKeyMask) | kTagValid;
    }
    static bool tag_key_matches(u64 tag, RemotePtr key) {
        return (tag & kTagKeyMask) == (key.raw_address & kTagKeyMask);
    }

    // GPU state
    u8* gpu_buffer_{nullptr};
    std::vector<ibv_mr*> gpu_buffer_mrs_;       // one MR per PD
    std::unordered_map<ibv_pd*, uint32_t> pd_lkeys_;  // PD → lkey
    uint32_t gpu_buffer_lkey_0_{0};             // first lkey (convenience)
    size_t total_slots_{0};
    size_t vector_bytes_{0};
    size_t slot_stride_{0};
    size_t num_shards_{0};

    // CPU metadata
    // Layout: tag(8B) | seq(4B) | alloc_epoch(4B) = 16B, one cache line friendly.
    // alloc_epoch records the CacheShard::alloc_epoch value at reservation time.
    // On commit it is set to 0.  A non-zero alloc_epoch on a Reserved slot means
    // the reservation was never completed (abandoned); CLOCK eviction may reclaim
    // such slots once the shard epoch has advanced by at least kStaleEpochDelta.
    struct SlotMeta {
        std::atomic<u64> tag;
        std::atomic<u32> seq;
        std::atomic<u32> alloc_epoch;  // 0 = not reserved, >0 = epoch at reserve time
        // Default-construct to zero
        SlotMeta() : tag(0), seq(0), alloc_epoch(0) {}
        // Non-copyable/non-movable due to atomic members
        SlotMeta(const SlotMeta&) = delete;
        SlotMeta& operator=(const SlotMeta&) = delete;
    };

    struct alignas(64) CacheShard {
        std::mutex write_mutex;
        SlotMeta* slots{nullptr};      // array allocated with new[]
        size_t num_slots{0};
        size_t clock_hand{0};
        u32 alloc_epoch{0};            // incremented on each allocate_slot call

        // Statistics counters — each on its own cache line to avoid false sharing
        // between threads updating different counters for the same shard.
        alignas(64) mutable std::atomic<u64> hits{0};
        alignas(64) mutable std::atomic<u64> misses{0};
        alignas(64) mutable std::atomic<u64> evictions{0};
        alignas(64) mutable std::atomic<u64> inserts{0};
        alignas(64) mutable std::atomic<u64> alloc_failures{0};
    };
    std::vector<std::unique_ptr<CacheShard>> shards_;
};

// =========================================================================
// Inline implementations
// =========================================================================

inline u64 GpuVectorCache::hash_key(RemotePtr key) {
    u64 h = key.raw_address;
    // murmur64 finalizer (same as NeighborCache)
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

inline size_t GpuVectorCache::shard_index(RemotePtr key) const {
    return hash_key(key) % num_shards_;
}

inline uint8_t* GpuVectorCache::gpu_slot_ptr(int32_t slot_id) const {
    return gpu_buffer_ + static_cast<size_t>(slot_id) * slot_stride_;
}

inline u64 GpuVectorCache::gpu_slot_addr(int32_t slot_id) const {
    return reinterpret_cast<u64>(gpu_buffer_ + static_cast<size_t>(slot_id) * slot_stride_);
}

inline void GpuVectorCache::init(size_t total_slots, uint32_t vector_bytes,
                                 const std::vector<ibv_pd*>& pds) {
    total_slots_ = total_slots;
    vector_bytes_ = vector_bytes;
    slot_stride_ = (static_cast<size_t>(vector_bytes) + 15) & ~static_cast<size_t>(15);
    num_shards_ = kDefaultNumShards;
    if (total_slots_ < num_shards_) {
        num_shards_ = std::max<size_t>(1, total_slots_);
    }

    const size_t gpu_bytes = total_slots_ * slot_stride_;
    cudaError_t err = cudaMalloc(&gpu_buffer_, gpu_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[GpuVectorCache] cudaMalloc(%zu) failed: %s\n",
                     gpu_bytes, cudaGetErrorString(err));
        std::abort();
    }
    cudaMemset(gpu_buffer_, 0, gpu_bytes);

    // Register the GPU buffer with each unique PD so every thread's QPs
    // can use GPUDirect RDMA into cache slots.
    for (ibv_pd* pd : pds) {
        if (pd == nullptr) continue;
        ibv_mr* mr = ibv_reg_mr(pd, gpu_buffer_, gpu_bytes, IBV_ACCESS_LOCAL_WRITE);
        if (mr != nullptr) {
            gpu_buffer_mrs_.push_back(mr);
            pd_lkeys_[pd] = mr->lkey;
            if (gpu_buffer_lkey_0_ == 0) {
                gpu_buffer_lkey_0_ = mr->lkey;
            }
        }
    }

    const size_t slots_per_shard = (total_slots_ + num_shards_ - 1) / num_shards_;
    shards_.reserve(num_shards_);
    for (size_t i = 0; i < num_shards_; ++i) {
        auto shard = std::make_unique<CacheShard>();
        shard->slots = new SlotMeta[slots_per_shard];  // zeroed via default ctor
        shard->num_slots = slots_per_shard;
        shards_.push_back(std::move(shard));
    }

    std::fprintf(stderr, "[GpuVectorCache] %zu slots, %zuB/slot (stride %zu), "
                 "%zu MB GPU, GPUDirect %s (%zu PDs)\n",
                 total_slots_, vector_bytes_, slot_stride_,
                 gpu_bytes / (1024 * 1024),
                 pd_lkeys_.empty() ? "disabled" : "enabled",
                 pd_lkeys_.size());
}

inline GpuVectorCache::~GpuVectorCache() {
    destroy();
}

inline void GpuVectorCache::destroy() {
    for (ibv_mr* mr : gpu_buffer_mrs_) {
        if (mr != nullptr) ibv_dereg_mr(mr);
    }
    gpu_buffer_mrs_.clear();
    pd_lkeys_.clear();
    gpu_buffer_lkey_0_ = 0;
    if (gpu_buffer_ != nullptr) {
        cudaFree(gpu_buffer_);
        gpu_buffer_ = nullptr;
    }
    for (auto& shard : shards_) {
        delete[] shard->slots;
        shard->slots = nullptr;
    }
    shards_.clear();
    total_slots_ = 0;
}

inline uint32_t GpuVectorCache::gpu_buffer_lkey(ibv_pd* pd) const {
    auto it = pd_lkeys_.find(pd);
    return it != pd_lkeys_.end() ? it->second : 0;
}

// Lock-free read — mirrors NeighborCache::find
inline int32_t GpuVectorCache::find(RemotePtr key) {
    const size_t shard_idx = shard_index(key);
    auto& shard = *shards_[shard_idx];
    const u64 search_key = key.raw_address & kTagKeyMask;
    const size_t num_slots = shard.num_slots;
    size_t start = hash_key(key) % num_slots;

    for (size_t probe = 0; probe < num_slots; ++probe) {
        size_t idx = (start + probe) % num_slots;
        auto& slot = shard.slots[idx];

        u64 tag1 = slot.tag.load(std::memory_order_acquire);
        if (tag1 == 0) {
            shard.misses.fetch_add(1, std::memory_order_relaxed);
            return -1;  // truly empty — end of probe chain
        }
        // Skip tombstones AND reserved (in-flight RDMA) slots
        if ((tag1 & kTagValid) == 0) {
            continue;
        }
        // Different key
        if ((tag1 & kTagKeyMask) != search_key) {
            continue;
        }

        // Key matches — seqlock validation.
        // seq1.load(acquire) already acts as a compiler+CPU barrier;
        // no non-atomic payload data is read here (data lives on GPU).
        u32 seq1 = slot.seq.load(std::memory_order_acquire);

        u32 seq2 = slot.seq.load(std::memory_order_acquire);
        u64 tag2 = slot.tag.load(std::memory_order_acquire);

        if (tag1 != tag2 || seq1 != seq2) {
            // Writer intervened — retry from current slot
            probe = static_cast<size_t>(-1);
            start = idx;
            continue;
        }

        // Hit — set reference bit for CLOCK eviction
        slot.tag.fetch_or(kTagReferenced, std::memory_order_release);

        shard.hits.fetch_add(1, std::memory_order_relaxed);
        return static_cast<int32_t>(shard_idx * num_slots + idx);
    }

    shard.misses.fetch_add(1, std::memory_order_relaxed);
    return -1;
}

// Reserve a slot for GPUDirect RDMA. Sets tag to (key|kTagReserved).
inline int32_t GpuVectorCache::allocate_slot(RemotePtr key) {
    const size_t shard_idx = shard_index(key);
    auto& shard = *shards_[shard_idx];
    std::unique_lock<std::mutex> lock(shard.write_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        shard.alloc_failures.fetch_add(1, std::memory_order_relaxed);
        return -1;  // contention — caller falls back to staging buffer
    }

    const u64 search_key = key.raw_address & kTagKeyMask;
    const size_t num_slots = shard.num_slots;
    size_t start = hash_key(key) % num_slots;

    // Bump epoch — used to detect abandoned reservations
    const u32 current_epoch = ++shard.alloc_epoch;

    // Find existing slot or first empty/tombstone
    size_t empty_slot = static_cast<size_t>(-1);
    for (size_t probe = 0; probe < num_slots; ++probe) {
        size_t idx = (start + probe) % num_slots;
        auto& slot = shard.slots[idx];
        u64 tag = slot.tag.load(std::memory_order_relaxed);

        if (tag == 0 || ((tag & (kTagValid | kTagReserved)) == 0)) {
            // Empty or tombstone (neither valid nor reserved)
            if (empty_slot == static_cast<size_t>(-1)) {
                empty_slot = idx;
            }
            if (tag == 0) break;  // prefer truly empty slot
            continue;
        }

        if ((tag & kTagKeyMask) == search_key) {
            // Already have a slot for this key — re-reserve it
            u32 new_seq = slot.seq.load(std::memory_order_relaxed) + 1;
            slot.seq.store(new_seq, std::memory_order_release);
            slot.alloc_epoch.store(current_epoch, std::memory_order_relaxed);
            slot.tag.store(make_reserved_tag(key), std::memory_order_release);
            shard.inserts.fetch_add(1, std::memory_order_relaxed);
            return static_cast<int32_t>(shard_idx * num_slots + idx);
        }
    }

    size_t target_idx = empty_slot;
    if (target_idx == static_cast<size_t>(-1)) {
        // Shard full — CLOCK eviction (bounded to kMaxClockProbes)
        const size_t clock_limit = std::min<size_t>(num_slots * 2, kMaxClockProbes);
        bool evicted = false;
        for (size_t attempt = 0; attempt < clock_limit; ++attempt) {
            size_t hand_idx = shard.clock_hand % num_slots;
            auto& slot = shard.slots[hand_idx];
            u64 tag = slot.tag.load(std::memory_order_relaxed);

            if (tag & kTagReserved) {
                // Check for abandoned reservation (epoch too old)
                u32 slot_epoch = slot.alloc_epoch.load(std::memory_order_relaxed);
                if (slot_epoch == 0 ||
                    (current_epoch - slot_epoch) <= kStaleEpochDelta) {
                    ++shard.clock_hand;  // still in-flight — never evict
                    continue;
                }
                // Fall through: stale reservation → eligible for eviction
            }
            if (tag & kTagReferenced) {
                slot.tag.fetch_and(~kTagReferenced, std::memory_order_relaxed);
                ++shard.clock_hand;
                continue;
            }
            target_idx = hand_idx;
            ++shard.clock_hand;
            shard.evictions.fetch_add(1, std::memory_order_relaxed);
            evicted = true;
            break;
        }
        if (!evicted) {
            shard.alloc_failures.fetch_add(1, std::memory_order_relaxed);
            return -1;  // no evictable slot
        }
    }

    // Reserve the target slot
    auto& target = shard.slots[target_idx];
    u32 new_seq = target.seq.load(std::memory_order_relaxed) + 1;
    target.seq.store(new_seq, std::memory_order_release);
    target.alloc_epoch.store(current_epoch, std::memory_order_relaxed);
    target.tag.store(make_reserved_tag(key), std::memory_order_release);
    shard.inserts.fetch_add(1, std::memory_order_relaxed);
    return static_cast<int32_t>(shard_idx * num_slots + target_idx);
}

// Commit after RDMA completes: atomically transitions tag from Reserved→Valid.
// Lock-free: allocate_slot skips kTagReserved slots during CLOCK eviction, so no
// writer can evict this slot while it's reserved.  We only need to atomically
// swap the flag bits — a CAS protects against the unlikely case where another
// thread re-allocated the same slot for the same key between our allocate and
// commit (in which case we simply skip the commit).
//
// Also sets kTagReferenced so newly cached data survives at least one CLOCK
// cycle before becoming eligible for eviction.
inline void GpuVectorCache::commit_slot(int32_t slot_id, RemotePtr key) {
    const size_t num_slots = shards_.empty() ? 0 : shards_[0]->num_slots;
    const size_t shard_idx = static_cast<size_t>(slot_id) / num_slots;
    const size_t slot_idx  = static_cast<size_t>(slot_id) % num_slots;
    auto& shard = *shards_[shard_idx];
    auto& slot = shard.slots[slot_idx];

    u64 expected = make_reserved_tag(key);
    u64 desired  = make_valid_tag(key) | kTagReferenced;
    // CAS: only commit if the slot is still reserved for this key.
    // If CAS fails the slot was evicted / re-allocated — data loss is
    // acceptable (vector will be re-fetched on next access).
    if (slot.tag.compare_exchange_strong(expected, desired,
                                         std::memory_order_release,
                                         std::memory_order_relaxed)) {
        // Only bump seq and clear epoch on success — otherwise we would
        // corrupt the metadata of whatever thread now owns this slot.
        u32 new_seq = slot.seq.load(std::memory_order_relaxed) + 1;
        slot.seq.store(new_seq, std::memory_order_release);
        slot.alloc_epoch.store(0, std::memory_order_relaxed);
    }
}

// Lock-free invalidation — mirrors NeighborCache::invalidate
inline void GpuVectorCache::invalidate(RemotePtr key) {
    const size_t shard_idx = shard_index(key);
    auto& shard = *shards_[shard_idx];
    const u64 search_key = key.raw_address & kTagKeyMask;
    const size_t num_slots = shard.num_slots;
    size_t start = hash_key(key) % num_slots;

    for (size_t probe = 0; probe < num_slots; ++probe) {
        size_t idx = (start + probe) % num_slots;
        auto& slot = shard.slots[idx];
        u64 tag = slot.tag.load(std::memory_order_relaxed);

        if (tag == 0) return;  // empty — end of chain, not found
        if ((tag & kTagValid) == 0) continue;  // tombstone or reserved
        if ((tag & kTagKeyMask) != search_key) continue;  // different key

        // Clear valid bit, preserve key for probe chain
        slot.seq.fetch_add(1, std::memory_order_relaxed);
        slot.tag.fetch_and(~kTagValid, std::memory_order_release);
        return;
    }
}

inline void GpuVectorCache::invalidate_all() {
    for (size_t i = 0; i < num_shards_; ++i) {
        auto& shard = *shards_[i];
        std::lock_guard<std::mutex> lock(shard.write_mutex);
        for (size_t j = 0; j < shard.num_slots; ++j) {
            shard.slots[j].tag.store(0, std::memory_order_relaxed);
            shard.slots[j].seq.store(0, std::memory_order_relaxed);
            shard.slots[j].alloc_epoch.store(0, std::memory_order_relaxed);
        }
        shard.clock_hand = 0;
        shard.alloc_epoch = 0;
    }
}

inline GpuVectorCache::Stats GpuVectorCache::stats() const {
    Stats s;
    for (const auto& shard : shards_) {
        s.hits      += shard->hits.load(std::memory_order_relaxed);
        s.misses    += shard->misses.load(std::memory_order_relaxed);
        s.evictions += shard->evictions.load(std::memory_order_relaxed);
        s.inserts   += shard->inserts.load(std::memory_order_relaxed);
        s.alloc_failures += shard->alloc_failures.load(std::memory_order_relaxed);
    }
    return s;
}
