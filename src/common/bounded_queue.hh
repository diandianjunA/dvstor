#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "common/constants.hh"
#include "common/types.hh"

namespace bounded {

// A preallocated Vyukov-style bounded MPMC queue.  It is used as MPSC/SPSC in
// the update runtimes too, so queue ownership can be narrowed without changing
// the storage format or RPC protocol.  No operation allocates after
// construction.
template <class T>
class Queue {
public:
  explicit Queue(size_t requested_capacity)
      : capacity_(normalize_capacity(requested_capacity)),
        mask_(capacity_ - 1),
        cells_(std::make_unique<Cell[]>(capacity_)) {
    for (u64 index = 0; index < capacity_; ++index) {
      cells_[index].sequence.store(index, std::memory_order_relaxed);
    }
  }

  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;

  [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

  [[nodiscard]] size_t approximate_size() const noexcept {
    const u64 pushed = enqueue_position_.load(std::memory_order_acquire);
    const u64 popped = dequeue_position_.load(std::memory_order_acquire);
    return static_cast<size_t>(std::min<u64>(capacity_, pushed - popped));
  }

  [[nodiscard]] bool empty() const noexcept {
    return approximate_size() == 0;
  }

  bool try_push(const T& value) { return emplace(value); }
  bool try_push(T&& value) { return emplace(std::move(value)); }

  template <class U>
  void push_wait(U&& value) {
    for (;;) {
      if (emplace(std::forward<U>(value))) return;
      const u64 observed = pop_epoch_.load(std::memory_order_acquire);
      if (emplace(std::forward<U>(value))) return;
      pop_epoch_.wait(observed, std::memory_order_relaxed);
    }
  }

  bool try_pop(T& value) {
    u64 position = dequeue_position_.load(std::memory_order_relaxed);
    Cell* cell = nullptr;
    for (;;) {
      cell = &cells_[static_cast<size_t>(position) & mask_];
      const u64 sequence = cell->sequence.load(std::memory_order_acquire);
      const i64 difference = static_cast<i64>(sequence - (position + 1));
      if (difference == 0) {
        if (dequeue_position_.compare_exchange_weak(
              position, position + 1,
              std::memory_order_relaxed, std::memory_order_relaxed)) {
          break;
        }
      } else if (difference < 0) {
        return false;
      } else {
        position = dequeue_position_.load(std::memory_order_relaxed);
      }
    }

    value = std::move(cell->value);
    cell->sequence.store(position + capacity_, std::memory_order_release);
    pop_epoch_.fetch_add(1, std::memory_order_release);
    pop_epoch_.notify_one();
    return true;
  }

  void pop_wait(T& value) {
    for (;;) {
      if (try_pop(value)) return;
      const u64 observed = push_epoch_.load(std::memory_order_acquire);
      if (try_pop(value)) return;
      push_epoch_.wait(observed, std::memory_order_relaxed);
    }
  }

  bool pop_wait(T& value, const std::atomic<bool>& stop) {
    for (;;) {
      if (try_pop(value)) return true;
      if (stop.load(std::memory_order_acquire)) return false;
      const u64 observed = push_epoch_.load(std::memory_order_acquire);
      if (try_pop(value)) return true;
      if (stop.load(std::memory_order_acquire)) return false;
      push_epoch_.wait(observed, std::memory_order_relaxed);
    }
  }

  // Wake blocked producers/consumers after an external shutdown flag changes.
  void notify_all() noexcept {
    push_epoch_.fetch_add(1, std::memory_order_release);
    pop_epoch_.fetch_add(1, std::memory_order_release);
    push_epoch_.notify_all();
    pop_epoch_.notify_all();
  }

private:
  struct alignas(kCacheLineBytes) Cell {
    std::atomic<u64> sequence{};
    T value{};
  };

  template <class U>
  bool emplace(U&& value) {
    u64 position = enqueue_position_.load(std::memory_order_relaxed);
    Cell* cell = nullptr;
    for (;;) {
      cell = &cells_[static_cast<size_t>(position) & mask_];
      const u64 sequence = cell->sequence.load(std::memory_order_acquire);
      const i64 difference = static_cast<i64>(sequence - position);
      if (difference == 0) {
        if (enqueue_position_.compare_exchange_weak(
              position, position + 1,
              std::memory_order_relaxed, std::memory_order_relaxed)) {
          break;
        }
      } else if (difference < 0) {
        return false;
      } else {
        position = enqueue_position_.load(std::memory_order_relaxed);
      }
    }

    cell->value = std::forward<U>(value);
    cell->sequence.store(position + 1, std::memory_order_release);
    push_epoch_.fetch_add(1, std::memory_order_release);
    push_epoch_.notify_one();
    return true;
  }

  static size_t normalize_capacity(size_t requested) {
    requested = std::max<size_t>(2, requested);
    if (requested > (size_t{1} << 62)) return size_t{1} << 62;
    return std::bit_ceil(requested);
  }

  const size_t capacity_;
  const size_t mask_;
  std::unique_ptr<Cell[]> cells_;
  alignas(kCacheLineBytes) std::atomic<u64> enqueue_position_{0};
  alignas(kCacheLineBytes) std::atomic<u64> dequeue_position_{0};
  alignas(kCacheLineBytes) std::atomic<u64> push_epoch_{0};
  alignas(kCacheLineBytes) std::atomic<u64> pop_epoch_{0};
};

}  // namespace bounded
