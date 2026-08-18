#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "common/bounded_queue.hh"

namespace memory_node_storage_owner_maintenance_detail {

// Context identity is process-runtime scoped.  Stage2ContextHandle alone is
// worker-local, so a completion routed through a node-wide CQ must also carry
// the runtime and worker that own that slot.
struct Stage2ContextOwnerKey {
  std::uint64_t runtime_epoch{};
  std::uint32_t worker_id{};
  std::uint32_t slot{};
  std::uint64_t token{};

  bool operator==(const Stage2ContextOwnerKey&) const = default;
};

enum class Stage2ContextReadyReason : std::uint32_t {
  rpc_response = 1u << 0,
  rdma_completion = 1u << 1,
  transport_credit = 1u << 2,
  reverse_completion = 1u << 3,
  timer = 1u << 4,
};

constexpr std::uint32_t stage2_ready_reason_bits(
    Stage2ContextReadyReason reason) {
  return static_cast<std::uint32_t>(reason);
}

struct Stage2ReadyContextEvent {
  Stage2ContextOwnerKey owner;
  std::uint8_t reasons{};
};

// Per-worker bounded MPSC ready queue.  Producers update one generation-tagged
// slot word before publishing a ticket. Multiple CQEs for the same context
// coalesce into one ticket; the consumer atomically takes all reason bits.
//
// Queue overflow cannot lose readiness. The slot retains its QUEUED bit and an
// overflow flag asks the owner for one bounded slot scan. With capacity >= 2x
// the context count this is a recovery path only; normal completion dispatch
// is O(1) and never scans unrelated contexts.
class Stage2ReadyContextQueue {
 public:
  Stage2ReadyContextQueue(std::uint64_t runtime_epoch,
                          std::uint32_t worker_id,
                          std::size_t context_capacity)
      : runtime_epoch_(runtime_epoch),
        worker_id_(worker_id),
        context_capacity_(checked_context_capacity(
          runtime_epoch, context_capacity)),
        slots_(std::make_unique<Slot[]>(context_capacity_)),
        queue_(checked_queue_capacity(context_capacity_)) {}

  Stage2ReadyContextQueue(const Stage2ReadyContextQueue&) = delete;
  Stage2ReadyContextQueue& operator=(const Stage2ReadyContextQueue&) = delete;

  [[nodiscard]] Stage2ContextOwnerKey activate(std::uint32_t slot) {
    if (slot >= context_capacity_) {
      throw std::invalid_argument(
        "Stage2 ready-context activation is out of range");
    }
    const std::uint64_t token = allocate_token();
    std::uint64_t inactive = 0;
    if (!slots_[slot].state.compare_exchange_strong(
          inactive, pack(token, 0), std::memory_order_acq_rel,
          std::memory_order_acquire)) {
      throw std::logic_error(
        "Stage2 ready-context slot activated while still live");
    }
    return Stage2ContextOwnerKey{
      runtime_epoch_, worker_id_, slot, token};
  }

  [[nodiscard]] bool deactivate(const Stage2ContextOwnerKey& owner) {
    if (!belongs(owner)) return false;
    std::uint64_t observed = slots_[owner.slot].state.load(
      std::memory_order_acquire);
    for (;;) {
      if (unpack_token(observed) != owner.token) return false;
      if (slots_[owner.slot].state.compare_exchange_weak(
            observed, 0, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return true;
      }
    }
  }

  // Returns false only for a stale/foreign owner. A full ticket ring returns
  // true after retaining the event in the slot and setting overflow recovery.
  [[nodiscard]] bool notify(const Stage2ContextOwnerKey& owner,
                            Stage2ContextReadyReason reason) {
    if (!belongs(owner)) return false;
    const std::uint8_t reason_bits = static_cast<std::uint8_t>(
      stage2_ready_reason_bits(reason));
    if (reason_bits == 0 || (reason_bits & ~kReasonMask) != 0) return false;
    Slot& slot = slots_[owner.slot];
    std::uint64_t observed = slot.state.load(std::memory_order_acquire);
    for (;;) {
      if (unpack_token(observed) != owner.token) return false;
      const std::uint8_t flags = unpack_flags(observed);
      const bool already_queued = (flags & kQueued) != 0;
      const std::uint8_t desired_flags = flags | reason_bits | kQueued;
      if (!slot.state.compare_exchange_weak(
            observed, pack(owner.token, desired_flags),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }
      if (already_queued) return true;
      const Ticket ticket{owner.slot, owner.token};
      if (queue_.try_push(ticket)) return true;
      overflow_.store(true, std::memory_order_release);
      return true;
    }
  }

  [[nodiscard]] bool try_pop(Stage2ReadyContextEvent& event) {
    Ticket ticket;
    while (queue_.try_pop(ticket)) {
      if (try_claim(ticket.slot, ticket.token, event)) return true;
    }
    return false;
  }

  // Called only by the owning worker after the ordinary ring is drained.
  // Stale tickets left in the ring become harmless generation mismatches.
  [[nodiscard]] std::size_t recover_overflow(
      std::vector<Stage2ReadyContextEvent>& events) {
    if (!overflow_.exchange(false, std::memory_order_acq_rel)) return 0;
    const std::size_t before = events.size();
    for (std::uint32_t slot = 0; slot < context_capacity_; ++slot) {
      const std::uint64_t snapshot = slots_[slot].state.load(
        std::memory_order_acquire);
      const std::uint64_t token = unpack_token(snapshot);
      if (token == 0 || (unpack_flags(snapshot) & kQueued) == 0) {
        continue;
      }
      Stage2ReadyContextEvent event;
      if (try_claim(slot, token, event)) {
        events.push_back(event);
      }
    }
    return events.size() - before;
  }

  [[nodiscard]] bool overflowed() const {
    return overflow_.load(std::memory_order_acquire);
  }

 private:
  static constexpr std::uint8_t kQueued = 1u << 7;
  static constexpr std::uint8_t kReasonMask = kQueued - 1;
  static constexpr std::uint64_t kMaxToken = (std::uint64_t{1} << 56) - 1;

  struct Slot {
    std::atomic<std::uint64_t> state{0};
  };

  struct Ticket {
    std::uint32_t slot{};
    std::uint64_t token{};
  };

  static std::size_t checked_context_capacity(
      std::uint64_t runtime_epoch, std::size_t contexts) {
    if (runtime_epoch == 0 || contexts == 0 ||
        contexts > std::numeric_limits<std::uint32_t>::max() ||
        contexts > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::invalid_argument(
        "Stage2 ready-context queue geometry is invalid");
    }
    return contexts;
  }

  static std::size_t checked_queue_capacity(std::size_t contexts) {
    if (contexts == 0 ||
        contexts > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::invalid_argument(
        "Stage2 ready-context queue capacity overflow");
    }
    return contexts * 2;
  }

  static constexpr std::uint64_t pack(std::uint64_t token,
                                      std::uint8_t flags) {
    return (token << 8) | flags;
  }

  static constexpr std::uint64_t unpack_token(std::uint64_t state) {
    return state >> 8;
  }

  static constexpr std::uint8_t unpack_flags(std::uint64_t state) {
    return static_cast<std::uint8_t>(state);
  }

  [[nodiscard]] bool belongs(const Stage2ContextOwnerKey& owner) const {
    return owner.runtime_epoch == runtime_epoch_ &&
      owner.worker_id == worker_id_ && owner.slot < context_capacity_ &&
      owner.token != 0 && owner.token <= kMaxToken;
  }

  [[nodiscard]] bool try_claim(std::uint32_t slot_index,
                               std::uint64_t token,
                               Stage2ReadyContextEvent& event) {
    if (slot_index >= context_capacity_ || token == 0) return false;
    Slot& slot = slots_[slot_index];
    std::uint64_t observed = slot.state.load(std::memory_order_acquire);
    for (;;) {
      if (unpack_token(observed) != token) return false;
      const std::uint8_t flags = unpack_flags(observed);
      if ((flags & kQueued) == 0) return false;
      if (!slot.state.compare_exchange_weak(
            observed, pack(token, 0), std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        continue;
      }
      event = Stage2ReadyContextEvent{
        .owner = Stage2ContextOwnerKey{
          runtime_epoch_, worker_id_, slot_index, token},
        .reasons = static_cast<std::uint8_t>(flags & kReasonMask),
      };
      return event.reasons != 0;
    }
  }

  const std::uint64_t runtime_epoch_;
  const std::uint32_t worker_id_;
  const std::size_t context_capacity_;
  std::unique_ptr<Slot[]> slots_;
  bounded::Queue<Ticket> queue_;
  std::atomic<bool> overflow_{false};
  std::atomic<std::uint64_t> next_token_{1};

  [[nodiscard]] std::uint64_t allocate_token() {
    const std::uint64_t token = next_token_.fetch_add(
      1, std::memory_order_relaxed);
    if (token == 0 || token > kMaxToken) {
      throw std::overflow_error("Stage2 ready-context token exhausted");
    }
    return token;
  }
};

}  // namespace memory_node_storage_owner_maintenance_detail
