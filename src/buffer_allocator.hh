#pragma once

#include <mutex>
#include <unordered_map>

#include "common/constants.hh"
#include "vamana/vamana_node.hh"

/**
 * Manages the memory-registered buffer globally per compute node.
 * Each thread has its own freelist which however may be accessed concurrently.
 */
class BufferAllocator {
public:
  explicit BufferAllocator(u32 num_threads, u64 buffer_size_bytes = COMPUTE_NODE_MAX_MEMORY) {
    // allocate a contiguous buffer for local memory
    local_buffer_.allocate(buffer_size_bytes);
    local_buffer_.touch_memory();

    buffer_ptr_ = local_buffer_.get_full_buffer();
    (void)num_threads;
  }

  HugePage<byte_t>& get_raw_buffer() { return local_buffer_; }

  // Allocate a buffer suitable for a full VamanaNode (including rabitq + neighbors)
  [[nodiscard]] byte_t* allocate_vamana_node(u32 thread_id) {
    (void)thread_id;
    return allocate_buffer(VamanaNode::total_size());
  }

  // General-purpose allocation for buffers of arbitrary size
  [[nodiscard]] byte_t* allocate_buffer(size_t size) {
    const size_t aligned_size = align(size);
    byte_t* ptr = nullptr;

    {
      std::lock_guard<std::mutex> lock(freelist_mutex_);
      auto it = freelists_by_size_.find(aligned_size);
      if (it != freelists_by_size_.end() && it->second.try_dequeue(ptr)) {
        return ptr;
      }
    }

    return allocate(size);
  }

  [[nodiscard]] u64* allocate_pointer() { return reinterpret_cast<u64*>(allocate(sizeof(u64))); }

  void free_buffer(byte_t* ptr, size_t size) {
    if (ptr == nullptr || size <= sizeof(u64)) {
      return;
    }

    const size_t aligned_size = align(size);
    std::lock_guard<std::mutex> lock(freelist_mutex_);
    freelists_by_size_[aligned_size].enqueue(ptr);
  }

  size_t allocated_memory() const { return bump_pointer_; }

private:
  static size_t align(size_t size) {
    while (size % CACHELINE_SIZE != 0) {
      ++size;
    }

    return size;
  }

  byte_t* allocate(size_t size) {
    lib_assert(size > 0, "unable to allocate 0 bytes");

    byte_t* ptr = buffer_ptr_ + bump_pointer_.fetch_add(align(size));
    lib_assert(bump_pointer_ <= local_buffer_.buffer_size, "out of local memory");

    // do not track 8B pointers
    if (size > sizeof(u64)) {
      allocated_buffers_.push_back(ptr);
    }

    return ptr;
  }

private:
  byte_t* buffer_ptr_;
  std::atomic<idx_t> bump_pointer_{0};  // points to free space
  HugePage<byte_t> local_buffer_;

  std::mutex freelist_mutex_;
  std::unordered_map<size_t, concurrent_queue<byte_t*>> freelists_by_size_;

  concurrent_vec<byte_t*> allocated_buffers_;  // track valid pointers (for cache eviction)
};
