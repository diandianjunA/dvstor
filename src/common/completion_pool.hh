#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "common/bounded_queue.hh"
#include "common/types.hh"

namespace bounded {

// Preallocated two-party completion cells.  A producer (the response executor)
// and a consumer (the synchronous public API) each own one reference.  A timed
// out/abandoned consumer therefore cannot create an ABA reuse while an RPC is
// still in flight.
class CompletionPool {
public:
  enum class Result : u32 {
    pending = 0,
    failure = 1,
    success = 2,
  };

  explicit CompletionPool(u32 capacity)
      : capacity_(capacity),
        cells_(std::make_unique<Cell[]>(capacity_)),
        free_(capacity_) {
    if (capacity_ == 0) {
      throw std::invalid_argument("completion pool capacity must be positive");
    }
    for (u32 id = 0; id < capacity_; ++id) {
      if (!free_.try_push(id)) {
        throw std::runtime_error("failed to initialize completion pool");
      }
    }
  }

  CompletionPool(const CompletionPool&) = delete;
  CompletionPool& operator=(const CompletionPool&) = delete;

  [[nodiscard]] u32 capacity() const noexcept { return capacity_; }

  u32 acquire() {
    u32 id = 0;
    free_.pop_wait(id);
    prepare(id);
    return id;
  }

  bool try_acquire(u32& id) {
    if (!free_.try_pop(id)) return false;
    prepare(id);
    return true;
  }

  Result wait(u32 id) const {
    validate(id);
    auto& state = cells_[id].state;
    u32 observed = state.load(std::memory_order_acquire);
    while (observed == static_cast<u32>(Result::pending)) {
      state.wait(observed, std::memory_order_relaxed);
      observed = state.load(std::memory_order_acquire);
    }
    return static_cast<Result>(observed);
  }

  template <typename Rep, typename Period>
  Result wait_for(
      u32 id, std::chrono::duration<Rep, Period> timeout) const {
    validate(id);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    auto& state = cells_[id].state;
    WaitShard& shard = wait_shards_[id & (kWaitShardCount - 1)];
    std::unique_lock<std::mutex> lock(shard.mutex);
    const bool completed = shard.changed.wait_until(lock, deadline, [&]() {
      return state.load(std::memory_order_acquire) !=
        static_cast<u32>(Result::pending);
    });
    if (!completed) return Result::pending;
    return static_cast<Result>(state.load(std::memory_order_acquire));
  }

  void complete(u32 id, bool success) {
    validate(id);
    Cell& cell = cells_[id];
    WaitShard& shard = wait_shards_[id & (kWaitShardCount - 1)];
    const u32 desired = static_cast<u32>(
      success ? Result::success : Result::failure);
    u32 expected = static_cast<u32>(Result::pending);
    {
      // Pair the predicate transition with the same striped mutex used by the
      // timed waiter. This closes the check-to-sleep lost-wakeup window without
      // allocating one condition variable (or one polling timer) per cell.
      std::lock_guard<std::mutex> lock(shard.mutex);
      if (!cell.state.compare_exchange_strong(
            expected, desired, std::memory_order_release,
            std::memory_order_acquire)) {
        throw std::logic_error("completion cell completed more than once");
      }
    }
    cell.state.notify_all();
    shard.changed.notify_all();
    release_reference(id);
  }

  // Called after wait(), or when a caller deliberately stops waiting.
  void release_consumer(u32 id) {
    validate(id);
    release_reference(id);
  }

  [[nodiscard]] Result result(u32 id) const {
    validate(id);
    return static_cast<Result>(
      cells_[id].state.load(std::memory_order_acquire));
  }

private:
  struct Cell {
    std::atomic<u32> state{static_cast<u32>(Result::pending)};
    std::atomic<u32> references{0};
  };

  struct WaitShard {
    mutable std::mutex mutex;
    mutable std::condition_variable changed;
  };

  // Keep unrelated synchronous writers out of one another's completion wake
  // domain. 1024 shards are still a fixed, small allocation compared with the
  // bounded mutation cells, while reducing notify_all fanout at 512--4096
  // concurrent callers.
  static constexpr size_t kWaitShardCount = 1024;
  static_assert((kWaitShardCount & (kWaitShardCount - 1)) == 0);

  void validate(u32 id) const {
    if (id >= capacity_) {
      throw std::out_of_range("invalid completion cell id");
    }
  }

  void prepare(u32 id) {
    Cell& cell = cells_[id];
    cell.state.store(static_cast<u32>(Result::pending),
                     std::memory_order_relaxed);
    cell.references.store(2, std::memory_order_release);
  }

  void release_reference(u32 id) {
    Cell& cell = cells_[id];
    const u32 previous = cell.references.fetch_sub(1, std::memory_order_acq_rel);
    if (previous == 0) {
      throw std::logic_error("completion cell reference underflow");
    }
    if (previous != 1) return;
    if (!free_.try_push(id)) {
      throw std::logic_error("completion pool free queue overflow");
    }
  }

  const u32 capacity_;
  std::unique_ptr<Cell[]> cells_;
  Queue<u32> free_;
  mutable std::array<WaitShard, kWaitShardCount> wait_shards_;
};

}  // namespace bounded
