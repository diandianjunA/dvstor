#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
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

  void complete(u32 id, bool success) {
    validate(id);
    Cell& cell = cells_[id];
    const u32 desired = static_cast<u32>(
      success ? Result::success : Result::failure);
    u32 expected = static_cast<u32>(Result::pending);
    if (!cell.state.compare_exchange_strong(
          expected, desired, std::memory_order_release,
          std::memory_order_acquire)) {
      throw std::logic_error("completion cell completed more than once");
    }
    cell.state.notify_all();
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
};

}  // namespace bounded
