#pragma once

#include <atomic>

namespace atomic_utils {

template <class T>
void update_max_relaxed(std::atomic<T>& target, T value) noexcept {
  T observed = target.load(std::memory_order_relaxed);
  while (observed < value &&
         !target.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {
  }
}

template <class T>
class CounterDecrementGuard {
public:
  explicit CounterDecrementGuard(std::atomic<T>& counter) : counter_(counter) {}
  ~CounterDecrementGuard() {
    counter_.fetch_sub(1, std::memory_order_acq_rel);
  }

  CounterDecrementGuard(const CounterDecrementGuard&) = delete;
  CounterDecrementGuard& operator=(const CounterDecrementGuard&) = delete;

private:
  std::atomic<T>& counter_;
};

}  // namespace atomic_utils
