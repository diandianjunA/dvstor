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
    // This is the physical/durable sequence span, not an unfinished-work
    // count; use incomplete() for logical admission credit.
    if (next <= done) return 0;
    return static_cast<size_t>(next - done - 1);
  }
  // Exact number of reserved sequences that still have unfinished work.
  // Unlike outstanding(), this is not pinned by an earlier out-of-order
  // completion. Admission may use this service-credit count while the larger
  // physical ring remains protected by the contiguous sequence span.
  [[nodiscard]] size_t incomplete() const noexcept {
    return incomplete_.load(std::memory_order_acquire);
  }
  [[nodiscard]] u64 logical_full_failures() const noexcept {
    return logical_full_failures_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] u64 physical_full_failures() const noexcept {
    return physical_full_failures_.load(std::memory_order_relaxed);
  }

  u64 reserve(u32 work_items) {
    return reserve_batch(span<const u32>{&work_items, 1});
  }

  // Try-only counterpart used by foreground protocol workers.  A zero return
  // is an explicit transient-capacity result; sequence zero is never a valid
  // reservation.  Failure leaves next_, every cell, and finalized_ unchanged.
  u64 try_reserve(u32 work_items) {
    return try_reserve_batch(span<const u32>{&work_items, 1});
  }

  // Atomically admits a whole foreground RPC batch.  Reserving each item
  // separately can deadlock when several workers each hold a partial window
  // while waiting for the rest of their batch: none of those sequences has
  // reached stage2 yet, so the finalized watermark cannot free the window.
  // This call waits without claiming anything until every item fits.
  u64 reserve_batch(span<const u32> work_items) {
    return reserve_batch(work_items, capacity_);
  }

  // admission_limit may be smaller than the physical ring. It bounds the
  // exact number of unfinished sequences, while capacity_ independently
  // bounds next_sequence() - finalized() - 1 so modulo cells are never reused
  // before the contiguous durable watermark releases them.
  u64 reserve_batch(span<const u32> work_items, size_t admission_limit) {
    validate_batch(work_items, admission_limit);

    // Load both wait values before the try. Logical service credit can return
    // without moving the contiguous watermark, while physical cell capacity
    // can return only when finalized_ advances. Waiting on the specific failed
    // resource also prevents a physical-full producer from busy-retrying on
    // its own service-credit rollback notification.
    for (;;) {
      const u64 service_epoch = service_credit_epoch_.load(
        std::memory_order_acquire);
      const u64 done = finalized_.load(std::memory_order_acquire);
      ReservationFailure failure = ReservationFailure::none;
      const u64 sequence = try_reserve_batch_validated(
        work_items, admission_limit, &failure);
      if (sequence != 0) return sequence;
      if (failure == ReservationFailure::logical_full) {
        service_credit_epoch_.wait(
          service_epoch, std::memory_order_relaxed);
      } else {
        finalized_.wait(done, std::memory_order_relaxed);
      }
    }
  }

  u64 try_reserve_batch(span<const u32> work_items) {
    return try_reserve_batch(work_items, capacity_);
  }

  // Atomically claims the complete batch or claims nothing.  Capacity
  // pressure never waits on finalized_: callers retain ownership of their
  // artifacts and may return a protocol-level transient retry.
  u64 try_reserve_batch(
      span<const u32> work_items, size_t admission_limit) {
    validate_batch(work_items, admission_limit);
    return try_reserve_batch_validated(work_items, admission_limit);
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
    if (previous == work_items) {
      const size_t previous_incomplete = incomplete_.fetch_sub(
        1, std::memory_order_acq_rel);
      if (previous_incomplete == 0) {
        incomplete_.fetch_add(1, std::memory_order_relaxed);
        throw std::logic_error("completion ring incomplete counter underflow");
      }
      advance();
      signal_service_credit_change();
    }
  }

  [[nodiscard]] u32 remaining(u64 sequence) const {
    if (sequence == 0) return 0;
    const Cell& cell = cells_[index(sequence)];
    if (cell.sequence.load(std::memory_order_acquire) != sequence) return 0;
    return cell.remaining.load(std::memory_order_acquire);
  }

private:
  enum class ReservationFailure {
    none,
    logical_full,
    physical_full,
  };

  struct Cell {
    std::atomic<u64> sequence{0};
    std::atomic<u32> remaining{0};
  };

  void validate_batch(
      span<const u32> work_items, size_t admission_limit) const {
    if (work_items.empty()) {
      throw std::invalid_argument("completion ring batch must not be empty");
    }
    if (admission_limit == 0 || admission_limit > capacity_ ||
        work_items.size() > admission_limit) {
      throw std::invalid_argument(
        "completion ring batch exceeds its admission window");
    }
  }

  u64 try_reserve_batch_validated(
      span<const u32> work_items, size_t admission_limit,
      ReservationFailure* failure = nullptr) {
    if (failure != nullptr) *failure = ReservationFailure::none;
    size_t service_credits = 0;
    for (const u32 work : work_items) {
      service_credits += work != 0 ? 1 : 0;
    }
    if (!try_acquire_service_credits(service_credits, admission_limit)) {
      logical_full_failures_.fetch_add(1, std::memory_order_relaxed);
      if (failure != nullptr) {
        *failure = ReservationFailure::logical_full;
      }
      return 0;
    }
    const auto release_service_credits = [&]() {
      if (service_credits == 0) return;
      const size_t previous = incomplete_.fetch_sub(
        service_credits, std::memory_order_acq_rel);
      if (previous < service_credits) {
        incomplete_.fetch_add(service_credits, std::memory_order_relaxed);
        throw std::logic_error("completion ring credit rollback underflow");
      }
      signal_service_credit_change();
    };

    const u64 count = static_cast<u64>(work_items.size());
    for (;;) {
      const u64 done = finalized_.load(std::memory_order_acquire);
      u64 sequence = next_.load(std::memory_order_acquire);
      // finalized_ advancing implies that next_ was advanced first. Loading
      // the watermark before next_ prevents a cross-atomic stale snapshot
      // from underflowing the unsigned outstanding calculation.
      if (sequence <= done) continue;
      if (sequence > std::numeric_limits<u64>::max() - count) {
        release_service_credits();
        throw std::overflow_error("completion ring sequence overflow");
      }
      const u64 next_after_batch = sequence + count;
      if (next_after_batch - done - 1 > capacity_) {
        // Avoid a false transient result when the contiguous watermark moved
        // during this cross-atomic snapshot.  Otherwise report full without
        // touching next_ or any cell.
        if (finalized_.load(std::memory_order_acquire) != done) continue;
        release_service_credits();
        physical_full_failures_.fetch_add(1, std::memory_order_relaxed);
        if (failure != nullptr) {
          *failure = ReservationFailure::physical_full;
        }
        return 0;
      }
      if (next_.compare_exchange_weak(
            sequence, next_after_batch,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        for (size_t item = 0; item < work_items.size(); ++item) {
          const u64 item_sequence = sequence + static_cast<u64>(item);
          Cell& cell = cells_[index(item_sequence)];
          cell.remaining.store(work_items[item], std::memory_order_relaxed);
          cell.sequence.store(item_sequence, std::memory_order_release);
        }
        // This also advances across any zero-work prefix.  A later zero-work
        // cell will be consumed by the completion that makes its predecessor
        // ready.
        advance();
        return sequence;
      }
      // CAS contention (including a spurious failure) is not a capacity
      // result. Re-read both atomics and either commit the complete batch or
      // return zero only after observing a genuinely full window.
    }
  }

  size_t index(u64 sequence) const noexcept {
    return static_cast<size_t>((sequence - 1) % capacity_);
  }

  bool try_acquire_service_credits(
      size_t count, size_t admission_limit) {
    if (count == 0) return true;
    size_t current = incomplete_.load(std::memory_order_acquire);
    for (;;) {
      if (current > admission_limit || count > admission_limit - current) {
        return false;
      }
      if (incomplete_.compare_exchange_weak(
            current, current + count,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return true;
      }
    }
  }

  void signal_service_credit_change() noexcept {
    service_credit_epoch_.fetch_add(1, std::memory_order_release);
    service_credit_epoch_.notify_all();
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
  std::atomic<size_t> incomplete_{0};
  std::atomic<u64> service_credit_epoch_{0};
  std::atomic<u64> logical_full_failures_{0};
  std::atomic<u64> physical_full_failures_{0};
};

}  // namespace bounded
