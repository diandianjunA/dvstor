#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace gpu_search {

template <class T>
class BoundedMpmcRing {
  static_assert(std::is_trivially_copyable_v<T>);

  struct alignas(64) Cell {
    std::atomic<size_t> sequence{};
    T value{};
  };

public:
  explicit BoundedMpmcRing(size_t capacity)
      : capacity_(capacity), mask_(capacity - 1), cells_(std::make_unique<Cell[]>(capacity)) {
    if (capacity < 2 || !std::has_single_bit(capacity)) {
      throw std::invalid_argument("ring capacity must be a power of two and at least two");
    }
    for (size_t i = 0; i < capacity_; ++i) {
      cells_[i].sequence.store(i, std::memory_order_relaxed);
    }
  }

  BoundedMpmcRing(const BoundedMpmcRing&) = delete;
  BoundedMpmcRing& operator=(const BoundedMpmcRing&) = delete;

  bool try_push(const T& value) {
    size_t position = enqueue_position_.load(std::memory_order_relaxed);
    for (;;) {
      Cell& cell = cells_[position & mask_];
      const size_t sequence = cell.sequence.load(std::memory_order_acquire);
      const intptr_t difference = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(position);
      if (difference == 0) {
        if (enqueue_position_.compare_exchange_weak(position, position + 1,
                                                     std::memory_order_relaxed)) {
          cell.value = value;
          cell.sequence.store(position + 1, std::memory_order_release);
          return true;
        }
      } else if (difference < 0) {
        return false;
      } else {
        position = enqueue_position_.load(std::memory_order_relaxed);
      }
    }
  }

  bool try_pop(T& value) {
    size_t position = dequeue_position_.load(std::memory_order_relaxed);
    for (;;) {
      Cell& cell = cells_[position & mask_];
      const size_t sequence = cell.sequence.load(std::memory_order_acquire);
      const intptr_t difference = static_cast<intptr_t>(sequence) -
                                  static_cast<intptr_t>(position + 1);
      if (difference == 0) {
        if (dequeue_position_.compare_exchange_weak(position, position + 1,
                                                     std::memory_order_relaxed)) {
          value = cell.value;
          cell.sequence.store(position + capacity_, std::memory_order_release);
          return true;
        }
      } else if (difference < 0) {
        return false;
      } else {
        position = dequeue_position_.load(std::memory_order_relaxed);
      }
    }
  }

  size_t capacity() const { return capacity_; }

  size_t approximate_size() const {
    const size_t enqueued = enqueue_position_.load(std::memory_order_relaxed);
    const size_t dequeued = dequeue_position_.load(std::memory_order_relaxed);
    return enqueued >= dequeued ? enqueued - dequeued : 0;
  }

private:
  const size_t capacity_;
  const size_t mask_;
  std::unique_ptr<Cell[]> cells_;
  alignas(64) std::atomic<size_t> enqueue_position_{0};
  alignas(64) std::atomic<size_t> dequeue_position_{0};
};

}  // namespace gpu_search
