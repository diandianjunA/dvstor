#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include <sys/mman.h>

#include <library/utils.hh>

#include "common/types.hh"

namespace tools::vamana_offline {

// mmap-backed buffer for the multi-GB arrays of the offline builder.
// Unlike vec<T>, allocation reserves zeroed pages without touching them, so
// the first parallel writer (pread workers / parallel fill) distributes the
// pages across NUMA nodes, and MADV_HUGEPAGE lets the kernel back the array
// with 2MB pages (THP is in madvise mode on the build hosts).
template <typename T>
class LargeBuffer {
  static_assert(std::is_trivially_destructible_v<T>,
                "LargeBuffer requires a trivially destructible element type");

public:
  LargeBuffer() = default;
  explicit LargeBuffer(size_t count) { reset(count); }
  ~LargeBuffer() { release(); }
  LargeBuffer(const LargeBuffer&) = delete;
  LargeBuffer& operator=(const LargeBuffer&) = delete;
  LargeBuffer(LargeBuffer&& other) noexcept { *this = std::move(other); }
  LargeBuffer& operator=(LargeBuffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void reset(size_t count) {
    release();
    if (count == 0) return;
    lib_assert(count <= std::numeric_limits<size_t>::max() / sizeof(T),
               "LargeBuffer allocation size overflows");
    const size_t bytes = count * sizeof(T);
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    lib_assert(p != MAP_FAILED,
               "LargeBuffer mmap failed for " + std::to_string(bytes) + " bytes");
    // Best effort: ignored when THP is disabled or set to "always".
    (void)::madvise(p, bytes, MADV_HUGEPAGE);
    ptr_ = static_cast<T*>(p);
    size_ = count;
  }

  void release() {
    if (ptr_ != nullptr) {
      (void)::munmap(ptr_, size_ * sizeof(T));
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t size() const { return size_; }
  T& operator[](size_t i) { return ptr_[i]; }
  const T& operator[](size_t i) const { return ptr_[i]; }
  T* begin() { return ptr_; }
  const T* begin() const { return ptr_; }
  T* end() { return ptr_ + size_; }
  const T* end() const { return ptr_ + size_; }

private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};

}  // namespace tools::vamana_offline
