#pragma once

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

#include "common/types.hh"

namespace bounded {

// Fixed-capacity sequence completion window.  Reservations are ordered, work
// may finish out of order, and finalized() only advances across a contiguous
// completed prefix.  Capacity is acquired before a mutation becomes visible.
class SlidingCompletionRing {
public:
  SlidingCompletionRing(size_t capacity,
                        u64 next_sequence = 1,
                        u64 finalized_sequence = 0)
      : capacity_(capacity),
        cells_(std::make_unique<Cell[]>(capacity_)),
        next_(next_sequence),
        finalized_(finalized_sequence) {
    if (capacity_ == 0 || next_sequence != finalized_sequence + 1) {
      throw std::invalid_argument(
        "completion ring requires positive capacity and an empty initial window");
    }
  }

  SlidingCompletionRing(const SlidingCompletionRing&) = delete;
  SlidingCompletionRing& operator=(const SlidingCompletionRing&) = delete;

  [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] u64 next_sequence() const noexcept {
    return next_.load(std::memory_order_acquire);
  }
  [[nodiscard]] u64 finalized() const noexcept {
    return finalized_.load(std::memory_order_acquire);
  }
  [[nodiscard]] size_t outstanding() const noexcept {
    const u64 done = finalized_.load(std::memory_order_acquire);
    const u64 next = next_.load(std::memory_order_acquire);
    // finalized_ can advance between independent atomic snapshots.  Loading
    // it first matches reserve_batch's ordering; retain the guard so a future
    // caller cannot observe unsigned underflow during wrap/shutdown telemetry.
    if (next <= done) return 0;
    return static_cast<size_t>(next - done - 1);
  }

  u64 reserve(u32 work_items) {
    return reserve_batch(span<const u32>{&work_items, 1});
  }

  // Atomically admits a whole foreground RPC batch.  Reserving each item
  // separately can deadlock when several workers each hold a partial window
  // while waiting for the rest of their batch: none of those sequences has
  // reached stage2 yet, so the finalized watermark cannot free the window.
  // This call waits without claiming anything until every item fits.
  u64 reserve_batch(span<const u32> work_items) {
    return reserve_batch(work_items, capacity_);
  }

  // admission_limit may be smaller than the physical ring. This keeps the
  // descriptor/intent allocation large while bounding the amount of visible,
  // not-yet-finalized stage2 work. Producers all recheck capacity when the
  // contiguous watermark advances; compare/exchange still admits only the
  // batches that fit, including mixed batch sizes without head-of-line idling.
  u64 reserve_batch(span<const u32> work_items, size_t admission_limit) {
    if (work_items.empty()) {
      throw std::invalid_argument("completion ring batch must not be empty");
    }
    if (admission_limit == 0 || admission_limit > capacity_ ||
        work_items.size() > admission_limit) {
      throw std::invalid_argument(
        "completion ring batch exceeds its admission window");
    }

    const u64 count = static_cast<u64>(work_items.size());
    u64 sequence = 0;
    for (;;) {
      const u64 done = finalized_.load(std::memory_order_acquire);
      sequence = next_.load(std::memory_order_acquire);
      // finalized_ advancing implies that next_ was advanced first. Loading
      // the watermark before next_ prevents a cross-atomic stale snapshot
      // from underflowing the unsigned outstanding calculation.
      if (sequence <= done) continue;
      if (sequence > std::numeric_limits<u64>::max() - count) {
        throw std::overflow_error("completion ring sequence overflow");
      }
      const u64 next_after_batch = sequence + count;
      if (next_after_batch - done - 1 > admission_limit) {
        finalized_.wait(done, std::memory_order_relaxed);
        continue;
      }
      if (next_.compare_exchange_weak(
            sequence, next_after_batch,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        break;
      }
    }

    for (size_t item = 0; item < work_items.size(); ++item) {
      const u64 item_sequence = sequence + static_cast<u64>(item);
      Cell& cell = cells_[index(item_sequence)];
      cell.remaining.store(work_items[item], std::memory_order_relaxed);
      cell.sequence.store(item_sequence, std::memory_order_release);
    }
    // This also advances across any zero-work prefix.  A later zero-work cell
    // will be consumed by the completion that makes its predecessor ready.
    advance();
    return sequence;
  }

  void complete(u64 sequence, u32 work_items = 1) {
    if (sequence == 0 || work_items == 0) return;
    Cell& cell = cells_[index(sequence)];
    if (cell.sequence.load(std::memory_order_acquire) != sequence) {
      throw std::logic_error("completion ring stale or unknown sequence");
    }
    const u32 previous = cell.remaining.fetch_sub(
      work_items, std::memory_order_acq_rel);
    if (previous < work_items) {
      cell.remaining.fetch_add(work_items, std::memory_order_relaxed);
      throw std::logic_error("completion ring work counter underflow");
    }
    if (previous == work_items) advance();
  }

  [[nodiscard]] u32 remaining(u64 sequence) const {
    if (sequence == 0) return 0;
    const Cell& cell = cells_[index(sequence)];
    if (cell.sequence.load(std::memory_order_acquire) != sequence) return 0;
    return cell.remaining.load(std::memory_order_acquire);
  }

private:
  struct Cell {
    std::atomic<u64> sequence{0};
    std::atomic<u32> remaining{0};
  };

  size_t index(u64 sequence) const noexcept {
    return static_cast<size_t>((sequence - 1) % capacity_);
  }

  void advance() {
    for (;;) {
      u64 watermark = finalized_.load(std::memory_order_acquire);
      const u64 candidate = watermark + 1;
      Cell& cell = cells_[index(candidate)];
      if (cell.sequence.load(std::memory_order_acquire) != candidate ||
          cell.remaining.load(std::memory_order_acquire) != 0) {
        return;
      }
      if (finalized_.compare_exchange_weak(
            watermark, candidate,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        finalized_.notify_all();
      }
    }
  }

  const size_t capacity_;
  std::unique_ptr<Cell[]> cells_;
  std::atomic<u64> next_;
  std::atomic<u64> finalized_;
};

}  // namespace bounded
