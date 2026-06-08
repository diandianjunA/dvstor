#pragma once

/**
 * NeighborCache: CPU-side adjacency list cache for Vamana graph traversal.
 *
 * Stores edge_count + neighbor RemotePtrs for hot nodes to avoid the two-step
 * RDMA read (edge_count + neighbors) during beam search expansion.
 *
 * Concurrency:
 *  - Reads are lock-free (seqlock pattern on atomic tag)
 *  - Writes (insert) are serialized per shard via mutex
 *  - Invalidation is lock-free (atomic tag clear)
 *
 * Eviction: per-shard CLOCK (second-chance) using a reference bit.
 *
 * Only active when gpu_cache_optimization config flag is true.
 */

#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

#include "common/types.hh"
#include "remote_pointer.hh"

class NeighborCache {
public:
    static constexpr size_t kDefaultNumShards = 16;
    static constexpr size_t kDefaultCacheEntries = 100000;
    static constexpr size_t kEntryHeaderSize = 16;  // tag(8)+seq(4)+count(1)+pad(3)

    // Must be called after VamanaNode::init_static_storage() so R is known.
    NeighborCache(size_t total_entries, u32 R);
    ~NeighborCache();

    NeighborCache(const NeighborCache&) = delete;
    NeighborCache& operator=(const NeighborCache&) = delete;
    NeighborCache(NeighborCache&&) = delete;
    NeighborCache& operator=(NeighborCache&&) = delete;

    // Lock-free read. Returns true on hit; fills out_neighbors (wraps caller's
    // buffer) and out_count. The caller must provide a buffer of at least R
    // RemotePtrs via out_neighbors.
    bool find(RemotePtr key, RemotePtr* out_buffer, u8& out_count);

    // Insert or update. Acquires per-shard mutex.
    void insert(RemotePtr key, u8 count, const RemotePtr* neighbors);

    // Lock-free invalidation (atomically clears valid bit).
    void invalidate(RemotePtr key);

    // Clear all entries (not lock-free; acquires all shard mutexes).
    void invalidate_all();

    struct Stats {
        u64 hits{0};
        u64 misses{0};
        u64 evictions{0};
        u64 inserts{0};
    };
    Stats stats() const;

    size_t entry_size() const { return entry_size_; }
    size_t total_entries() const { return total_entries_; }

private:
    static constexpr u64 kTagValid = 1ULL << 0;
    static constexpr u64 kTagReferenced = 1ULL << 1;
    // Key occupies bits [63:2]; bits [1:0] are valid + referenced flags.
    // Safe because RemotePtr byte_offsets are 8-byte aligned (bit 0 always 0).
    static constexpr u64 kTagKeyMask = ~(kTagValid | kTagReferenced);

    // Entry layout: tag(8B) | seq(4B, 4B-aligned) | count(1B) | pad(3B) | neighbors[R*8B]
    // Total: kEntryHeaderSize + R * 8 bytes
    // seq is incremented on each write for seqlock consistency.

    struct alignas(64) CacheShard {
        std::mutex write_mutex;
        u8* entries;          // flat array: entry_size_ * num_entries bytes
        size_t num_entries;
        size_t clock_hand{0};
        mutable std::atomic<u64> hits{0};
        mutable std::atomic<u64> misses{0};
        mutable std::atomic<u64> evictions{0};
        mutable std::atomic<u64> inserts{0};
    };

    static u64 hash_key(RemotePtr key);
    size_t shard_index(RemotePtr key) const;

    // Per-entry field accessors (entry_ptr points to start of an entry)
    static std::atomic<u64>& entry_tag(u8* entry);
    static u8& entry_count(u8* entry);
    static std::atomic<u32>& entry_seq(u8* entry);
    static RemotePtr* entry_neighbors(u8* entry);

    // Tag helpers
    static u64 make_tag(RemotePtr key);
    static bool tag_matches(u64 tag, RemotePtr key);

    size_t entry_size_;       // = kEntryHeaderSize + R_ * sizeof(u64)
    size_t total_entries_;
    size_t num_shards_;
    u32 R_;
    std::vector<std::unique_ptr<CacheShard>> shards_;
    u8* raw_buffer_{nullptr}; // owns all shard memory
    size_t raw_buffer_size_{0};
};

// =========================================================================
// Implementation
// =========================================================================

inline u64 NeighborCache::hash_key(RemotePtr key) {
    u64 h = key.raw_address;
    // murmur64 finalizer
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

inline size_t NeighborCache::shard_index(RemotePtr key) const {
    return hash_key(key) % num_shards_;
}

inline std::atomic<u64>& NeighborCache::entry_tag(u8* entry) {
    return *reinterpret_cast<std::atomic<u64>*>(entry);
}

inline std::atomic<u32>& NeighborCache::entry_seq(u8* entry) {
    return *reinterpret_cast<std::atomic<u32>*>(entry + sizeof(u64));  // offset 8, 4B-aligned
}

inline u8& NeighborCache::entry_count(u8* entry) {
    return *reinterpret_cast<u8*>(entry + sizeof(u64) + sizeof(u32));  // offset 12
}

inline RemotePtr* NeighborCache::entry_neighbors(u8* entry) {
    return reinterpret_cast<RemotePtr*>(entry + kEntryHeaderSize);
}

inline u64 NeighborCache::make_tag(RemotePtr key) {
    // Key occupies bits [63:2]; bits [1:0] are valid + referenced flags.
    return (key.raw_address & kTagKeyMask) | kTagValid;
}

inline bool NeighborCache::tag_matches(u64 tag, RemotePtr key) {
    return (tag & kTagKeyMask) == (key.raw_address & kTagKeyMask);
}

inline NeighborCache::NeighborCache(size_t total_entries, u32 R)
    : total_entries_(total_entries), R_(R) {
    entry_size_ = kEntryHeaderSize + static_cast<size_t>(R) * sizeof(u64);
    num_shards_ = kDefaultNumShards;
    if (total_entries_ < num_shards_) {
        num_shards_ = std::max<size_t>(1, total_entries_);
    }

    const size_t entries_per_shard = (total_entries_ + num_shards_ - 1) / num_shards_;
    raw_buffer_size_ = num_shards_ * entries_per_shard * entry_size_;
    raw_buffer_ = new u8[raw_buffer_size_];
    std::memset(raw_buffer_, 0, raw_buffer_size_);

    shards_.reserve(num_shards_);
    for (size_t i = 0; i < num_shards_; ++i) {
        auto shard = std::make_unique<CacheShard>();
        shard->entries = raw_buffer_ + i * entries_per_shard * entry_size_;
        shard->num_entries = entries_per_shard;
        shards_.push_back(std::move(shard));
    }
}

inline NeighborCache::~NeighborCache() {
    delete[] raw_buffer_;
}

inline bool NeighborCache::find(RemotePtr key, RemotePtr* out_buffer, u8& out_count) {
    const size_t shard_idx = shard_index(key);
    auto& shard = *shards_[shard_idx];
    const u64 search_raw = key.raw_address;

    size_t start = hash_key(key) % shard.num_entries;

    for (size_t probe = 0; probe < shard.num_entries; ++probe) {
        size_t idx = (start + probe) % shard.num_entries;
        u8* entry = shard.entries + idx * entry_size_;
        auto& tag_atomic = entry_tag(entry);
        auto& seq_atomic = entry_seq(entry);

        u64 tag1 = tag_atomic.load(std::memory_order_acquire);
        if (tag1 == 0) {
            return false;  // truly empty — end of probe chain
        }
        if ((tag1 & kTagValid) == 0) {
            continue;  // tombstone — keep probing
        }

        // Check key match (ignoring valid + ref bits)
        if ((tag1 & kTagKeyMask) != (search_raw & kTagKeyMask)) {
            continue;  // different key
        }

        // Potential match — read data under seqlock
        u32 seq1 = seq_atomic.load(std::memory_order_acquire);
        // Acquire fence prevents data reads from being reordered before seq read,
        // and also prevents them from moving after the re-validation below.
        std::atomic_thread_fence(std::memory_order_acquire);

        u8 count = entry_count(entry);
        const RemotePtr* neighbors_ptr = entry_neighbors(entry);
        std::memcpy(out_buffer, neighbors_ptr, static_cast<size_t>(count) * sizeof(RemotePtr));
        out_count = count;

        // Acquire fence: prevent data reads from being reordered after seq/tag re-validation
        std::atomic_thread_fence(std::memory_order_acquire);
        u32 seq2 = seq_atomic.load(std::memory_order_acquire);
        u64 tag2 = tag_atomic.load(std::memory_order_acquire);

        if (tag1 != tag2 || seq1 != seq2) {
            // Writer intervened — retry from this slot
            probe = static_cast<size_t>(-1);  // will become 0 after ++
            start = idx;
            continue;
        }

        // Hit — set reference bit for CLOCK eviction
        tag_atomic.fetch_or(kTagReferenced, std::memory_order_release);

        shard.hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    shard.misses.fetch_add(1, std::memory_order_relaxed);
    return false;
}

inline void NeighborCache::insert(RemotePtr key, u8 count, const RemotePtr* neighbors) {
    const size_t shard_idx = shard_index(key);
    auto& shard = *shards_[shard_idx];
    std::unique_lock<std::mutex> lock(shard.write_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return;  // skip insertion under contention — avoid blocking coroutine thread
    }

    const u64 search_raw = key.raw_address;
    size_t start = hash_key(key) % shard.num_entries;

    // First, try to find existing entry for this key (update in place)
    size_t empty_slot = static_cast<size_t>(-1);
    for (size_t probe = 0; probe < shard.num_entries; ++probe) {
        size_t idx = (start + probe) % shard.num_entries;
        u8* entry = shard.entries + idx * entry_size_;
        auto& tag_atomic = entry_tag(entry);
        u64 tag = tag_atomic.load(std::memory_order_relaxed);

        if (tag == 0 || (tag & kTagValid) == 0) {
            if (empty_slot == static_cast<size_t>(-1)) {
                empty_slot = idx;
            }
            if (tag == 0) break;  // truly empty — prefer this slot
            continue;
        }

        if ((tag & kTagKeyMask) == (search_raw & kTagKeyMask)) {
            // Update existing entry in place — bump seq for seqlock
            auto& seq_atomic = entry_seq(entry);
            u32 new_seq = seq_atomic.load(std::memory_order_relaxed) + 1;
            tag_atomic.store(0, std::memory_order_release);  // invalidate readers
            entry_count(entry) = count;
            std::memcpy(entry_neighbors(entry), neighbors,
                        static_cast<size_t>(count) * sizeof(RemotePtr));
            seq_atomic.store(new_seq, std::memory_order_release);
            tag_atomic.store(make_tag(key), std::memory_order_release);
            shard.inserts.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }

    // Not found — need to insert into empty_slot or evict
    size_t target_idx = empty_slot;
    if (target_idx == static_cast<size_t>(-1)) {
        // Shard is full — bounded CLOCK eviction (max 128 probes to avoid long mutex hold)
        const size_t clock_limit = std::min<size_t>(shard.num_entries * 2, 128);
        for (size_t attempt = 0; attempt < clock_limit; ++attempt) {
            size_t hand_idx = shard.clock_hand % shard.num_entries;
            u8* entry = shard.entries + hand_idx * entry_size_;
            auto& tag_atomic = entry_tag(entry);

            u64 tag = tag_atomic.load(std::memory_order_relaxed);
            if (tag & kTagReferenced) {
                // Give a second chance — clear ref bit
                tag_atomic.fetch_and(~kTagReferenced, std::memory_order_relaxed);
                ++shard.clock_hand;
                continue;
            }
            // Evict this entry
            target_idx = hand_idx;
            ++shard.clock_hand;
            shard.evictions.fetch_add(1, std::memory_order_relaxed);
            break;
        }
        if (target_idx == static_cast<size_t>(-1)) {
            // No victim found within limit — skip insertion (don't block for too long)
            return;
        }
    }

    // Write new entry
    u8* target = shard.entries + target_idx * entry_size_;
    auto& tag_atomic = entry_tag(target);
    auto& seq_atomic = entry_seq(target);

    // Invalidate any existing entry, bump sequence
    u32 new_seq = seq_atomic.load(std::memory_order_relaxed) + 1;
    tag_atomic.store(0, std::memory_order_release);
    entry_count(target) = count;
    std::memcpy(entry_neighbors(target), neighbors,
                static_cast<size_t>(count) * sizeof(RemotePtr));
    seq_atomic.store(new_seq, std::memory_order_release);
    tag_atomic.store(make_tag(key), std::memory_order_release);
    shard.inserts.fetch_add(1, std::memory_order_relaxed);
}

inline void NeighborCache::invalidate(RemotePtr key) {
    const size_t shard_idx = shard_index(key);
    auto& shard = *shards_[shard_idx];
    const u64 search_raw = key.raw_address;

    size_t start = hash_key(key) % shard.num_entries;
    for (size_t probe = 0; probe < shard.num_entries; ++probe) {
        size_t idx = (start + probe) % shard.num_entries;
        u8* entry = shard.entries + idx * entry_size_;
        auto& tag_atomic = entry_tag(entry);

        u64 tag = tag_atomic.load(std::memory_order_relaxed);
        if (tag == 0) return;  // truly empty — end of chain, not found
        if ((tag & kTagValid) == 0) continue;  // tombstone, keep probing
        if ((tag & kTagKeyMask) == (search_raw & kTagKeyMask)) {
            // Clear valid bit only — preserves key for probe chain
            // Also bump seq so any in-flight readers detect the change
            auto& seq_atomic = entry_seq(entry);
            seq_atomic.fetch_add(1, std::memory_order_relaxed);
            tag_atomic.fetch_and(~kTagValid, std::memory_order_release);
            return;
        }
    }
}

inline void NeighborCache::invalidate_all() {
    for (size_t i = 0; i < num_shards_; ++i) {
        auto& shard = *shards_[i];
        std::lock_guard<std::mutex> lock(shard.write_mutex);
        std::memset(shard.entries, 0, shard.num_entries * entry_size_);
        shard.clock_hand = 0;
    }
}

inline NeighborCache::Stats NeighborCache::stats() const {
    Stats s;
    for (const auto& shard : shards_) {
        s.hits += shard->hits.load(std::memory_order_relaxed);
        s.misses += shard->misses.load(std::memory_order_relaxed);
        s.evictions += shard->evictions.load(std::memory_order_relaxed);
        s.inserts += shard->inserts.load(std::memory_order_relaxed);
    }
    return s;
}
