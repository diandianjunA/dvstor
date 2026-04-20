#include "gpu/gpu_rabitq_cache.hh"

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <cstdio>
#include <cstring>

namespace gpu {

#define CUDA_CHECK_CACHE(call)                                                \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err));                                  \
      return false;                                                           \
    }                                                                         \
  } while (0)

GpuRabitqCache::~GpuRabitqCache() { destroy(); }

bool GpuRabitqCache::init(size_t bytes, uint32_t stride, ibv_pd* pd) {
  destroy();
  if (bytes == 0 || stride == 0 || pd == nullptr) return false;

  stride_ = stride;
  slot_count_ = bytes / stride_;
  if (slot_count_ < 2) return false;

  const size_t pool_bytes = slot_count_ * static_cast<size_t>(stride_);
  CUDA_CHECK_CACHE(cudaMalloc(&pool_, pool_bytes));
  mr_ = ibv_reg_mr(pd, pool_, pool_bytes, IBV_ACCESS_LOCAL_WRITE);
  if (!mr_) {
    cudaFree(pool_);
    pool_ = nullptr;
    std::fprintf(stderr, "[GPU RaBitQ cache] failed to register GPU memory for RDMA\n");
    return false;
  }

  lkey_ = mr_->lkey;
  table_capacity_ = next_power_of_two(slot_count_ * 2);
  table_.assign(table_capacity_, Entry{});
  slot_keys_.assign(slot_count_, 0);
  slot_states_.assign(slot_count_, State::empty);
  slot_refs_.assign(slot_count_, 0);
  slot_use_counts_.assign(slot_count_, 0);
  next_slot_ = 0;
  clock_hand_ = 0;
  enabled_ = true;
  std::fprintf(stderr, "[GPU RaBitQ cache] enabled: slots=%zu stride=%u bytes=%zu\n",
               slot_count_, stride_, pool_bytes);
  return true;
}

void GpuRabitqCache::destroy() {
  if (mr_) {
    ibv_dereg_mr(mr_);
    mr_ = nullptr;
  }
  if (pool_) {
    cudaFree(pool_);
    pool_ = nullptr;
  }
  enabled_ = false;
  lkey_ = 0;
  stride_ = 0;
  slot_count_ = 0;
  table_capacity_ = 0;
  next_slot_ = 0;
  clock_hand_ = 0;
  table_.clear();
  slot_keys_.clear();
  slot_states_.clear();
  slot_refs_.clear();
  slot_use_counts_.clear();
}

GpuRabitqCacheResolve GpuRabitqCache::resolve_batch(const void* remote_ptrs,
                                                    uint32_t n,
                                                    uint32_t* out_slot_ids,
                                                    std::vector<uint32_t>& miss_indices,
                                                    std::vector<uint32_t>& miss_slots,
                                                    std::vector<uint64_t>& miss_addrs) {
  GpuRabitqCacheResolve result{};
  result.n = n;
  if (!enabled_) return result;

  miss_indices.clear();
  miss_slots.clear();
  miss_addrs.clear();
  miss_indices.reserve(n);
  miss_slots.reserve(n);
  miss_addrs.reserve(n);

  std::vector<uint32_t> newly_reserved;
  newly_reserved.reserve(n);

  for (uint32_t i = 0; i < n; ++i) {
    uint64_t key = 0;
    std::memcpy(&key, static_cast<const uint8_t*>(remote_ptrs) + static_cast<size_t>(i) * sizeof(uint64_t),
                sizeof(uint64_t));
    uint32_t slot = kInvalidSlot;
    if (lookup(key, slot)) {
      out_slot_ids[i] = slot;
      slot_refs_[slot] = 1;
      if (slot_states_[slot] == State::ready) {
        ++result.hit_count;
      } else {
        ++result.duplicate_loading_count;
        miss_indices.push_back(i);
        miss_slots.push_back(slot);
        miss_addrs.push_back(slot_addr(slot));
      }
      continue;
    }

    if (!allocate_slot(slot) || !insert_entry(key, slot)) {
      rollback_loading(newly_reserved);
      return result;
    }
    slot_keys_[slot] = key;
    slot_states_[slot] = State::loading;
    slot_refs_[slot] = 1;
    newly_reserved.push_back(slot);
    out_slot_ids[i] = slot;
    miss_indices.push_back(i);
    miss_slots.push_back(slot);
    miss_addrs.push_back(slot_addr(slot));
  }

  result.miss_count = static_cast<uint32_t>(miss_indices.size());
  result.ok = true;
  return result;
}

void GpuRabitqCache::publish_batch(const std::vector<uint32_t>& slots) {
  for (uint32_t slot : slots) {
    if (slot < slot_states_.size()) {
      slot_states_[slot] = State::ready;
      slot_refs_[slot] = 1;
    }
  }
}

void GpuRabitqCache::acquire_slots(const uint32_t* slots, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t slot = slots[i];
    if (slot < slot_use_counts_.size()) ++slot_use_counts_[slot];
  }
}

void GpuRabitqCache::release_slots(const uint32_t* slots, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t slot = slots[i];
    if (slot < slot_use_counts_.size() && slot_use_counts_[slot] > 0) {
      --slot_use_counts_[slot];
    }
  }
}

void GpuRabitqCache::rollback_loading(const std::vector<uint32_t>& slots) {
  for (uint32_t slot : slots) {
    if (slot >= slot_states_.size() || slot_states_[slot] != State::loading) continue;
    remove_entry(slot_keys_[slot]);
    free_slot(slot);
  }
}

size_t GpuRabitqCache::next_power_of_two(size_t value) {
  size_t out = 1;
  while (out < value) out <<= 1;
  return out;
}

uint64_t GpuRabitqCache::hash(uint64_t key) {
  key ^= key >> 33;
  key *= 0xff51afd7ed558ccdULL;
  key ^= key >> 33;
  key *= 0xc4ceb9fe1a85ec53ULL;
  key ^= key >> 33;
  return key;
}

bool GpuRabitqCache::lookup(uint64_t key, uint32_t& slot) const {
  size_t pos = hash(key) & (table_capacity_ - 1);
  for (size_t probe = 0; probe < table_capacity_; ++probe) {
    const auto& entry = table_[pos];
    if (entry.key == 0) return false;
    if (entry.key == key) {
      slot = entry.slot;
      return true;
    }
    pos = (pos + 1) & (table_capacity_ - 1);
  }
  return false;
}

bool GpuRabitqCache::insert_entry(uint64_t key, uint32_t slot) {
  size_t pos = hash(key) & (table_capacity_ - 1);
  for (size_t probe = 0; probe < table_capacity_; ++probe) {
    if (table_[pos].key == 0 || table_[pos].key == key) {
      table_[pos] = Entry{key, slot};
      return true;
    }
    pos = (pos + 1) & (table_capacity_ - 1);
  }
  return false;
}

void GpuRabitqCache::remove_entry(uint64_t key) {
  if (key == 0 || table_capacity_ == 0) return;
  size_t pos = hash(key) & (table_capacity_ - 1);
  for (size_t probe = 0; probe < table_capacity_; ++probe) {
    if (table_[pos].key == 0) return;
    if (table_[pos].key == key) {
      table_[pos] = Entry{};
      rehash_cluster((pos + 1) & (table_capacity_ - 1));
      return;
    }
    pos = (pos + 1) & (table_capacity_ - 1);
  }
}

void GpuRabitqCache::rehash_cluster(size_t pos) {
  while (table_[pos].key != 0) {
    const Entry entry = table_[pos];
    table_[pos] = Entry{};
    insert_entry(entry.key, entry.slot);
    pos = (pos + 1) & (table_capacity_ - 1);
  }
}

bool GpuRabitqCache::allocate_slot(uint32_t& slot) {
  if (next_slot_ < slot_count_) {
    slot = next_slot_++;
    return true;
  }
  for (size_t scanned = 0; scanned < slot_count_ * 2; ++scanned) {
    const uint32_t candidate = clock_hand_;
    clock_hand_ = (clock_hand_ + 1) % static_cast<uint32_t>(slot_count_);
    if (slot_states_[candidate] == State::loading || slot_use_counts_[candidate] > 0) continue;
    if (slot_refs_[candidate]) {
      slot_refs_[candidate] = 0;
      continue;
    }
    remove_entry(slot_keys_[candidate]);
    free_slot(candidate);
    slot = candidate;
    return true;
  }
  return false;
}

void GpuRabitqCache::free_slot(uint32_t slot) {
  slot_keys_[slot] = 0;
  slot_states_[slot] = State::empty;
  slot_refs_[slot] = 0;
  slot_use_counts_[slot] = 0;
}

uint64_t GpuRabitqCache::slot_addr(uint32_t slot) const {
  return reinterpret_cast<uint64_t>(base()) + static_cast<uint64_t>(slot) * stride_;
}

}  // namespace gpu
