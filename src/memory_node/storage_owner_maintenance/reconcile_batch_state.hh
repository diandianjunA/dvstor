#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "memory_node/storage_owner_maintenance/stage2_tracker.hh"
#include "service/storage_owner_protocol.hh"

namespace memory_node_storage_owner_maintenance_detail {

enum class Stage2ReconcileBarrier : std::uint8_t {
  none,
  install,
  removal,
};

// Ordinary backlinks may RobustPrune the same hot parent selected as another
// insertion's mandatory reachability holder. Install ordinary mutations and
// mandatory promotions in one per-target transaction, with every ordinary op
// ordered before every promotion op for that target. The receiver audits all
// mandatory certificates against the transaction's final adjacency before it
// ACKs. A separate removal barrier can therefore retire provisional bridges
// without either the old cross-barrier eviction window or a third RTT.
[[nodiscard]] constexpr Stage2ReconcileBarrier
stage2_reconcile_first_barrier() noexcept {
  return Stage2ReconcileBarrier::install;
}

[[nodiscard]] constexpr Stage2ReconcileBarrier
stage2_reconcile_next_barrier(Stage2ReconcileBarrier barrier) noexcept {
  switch (barrier) {
    case Stage2ReconcileBarrier::install:
      return Stage2ReconcileBarrier::removal;
    case Stage2ReconcileBarrier::removal:
    case Stage2ReconcileBarrier::none:
      return Stage2ReconcileBarrier::none;
  }
  return Stage2ReconcileBarrier::none;
}

// The outer Stage2 tracker deliberately remains in prune_ready while these
// context-local transport barriers are outstanding.  Keeping this subphase in
// the context lets the scheduler run other searches instead of turning a
// network ACK into a worker-wide blocking wait.
enum class Stage2FinalizeSubphase : std::uint8_t {
  prepare,
  install_wait,
  removal_wait,
  placement_ready,
};

[[nodiscard]] constexpr Stage2FinalizeSubphase
stage2_reconcile_wait_subphase(Stage2ReconcileBarrier barrier) noexcept {
  switch (barrier) {
    case Stage2ReconcileBarrier::install:
      return Stage2FinalizeSubphase::install_wait;
    case Stage2ReconcileBarrier::removal:
      return Stage2FinalizeSubphase::removal_wait;
    case Stage2ReconcileBarrier::none:
      break;
  }
  return Stage2FinalizeSubphase::prepare;
}

[[nodiscard]] constexpr bool stage2_finalize_subphase_needs_lane(
    Stage2FinalizeSubphase subphase) noexcept {
  return subphase == Stage2FinalizeSubphase::prepare ||
    subphase == Stage2FinalizeSubphase::placement_ready;
}

// RPCs belonging to the same peer may complete out of order, so a target's
// operation run must never straddle two messages. Grouping retains the
// original per-target order within each install/removal barrier while
// allowing independent targets to be packed.
using Stage2ReconcileOp = service::storage_owner::ReconcileReverseOp;
inline std::optional<std::vector<std::vector<Stage2ReconcileOp>>>
pack_stage2_reconcile_target_runs(
    std::span<const Stage2ReconcileOp> ops, std::size_t wire_capacity) {
  if (wire_capacity == 0) return std::nullopt;

  std::vector<std::vector<Stage2ReconcileOp>> target_runs;
  target_runs.reserve(ops.size());
  std::unordered_map<std::uint64_t, std::size_t> target_to_run;
  target_to_run.reserve(ops.size());
  for (const Stage2ReconcileOp& op : ops) {
    auto [position, inserted] =
      target_to_run.emplace(op.target_raw, target_runs.size());
    if (inserted) target_runs.emplace_back();
    target_runs[position->second].push_back(op);
  }

  std::vector<std::vector<Stage2ReconcileOp>> chunks;
  for (auto& run : target_runs) {
    if (run.size() > wire_capacity) return std::nullopt;
    if (chunks.empty() ||
        chunks.back().size() + run.size() > wire_capacity) {
      chunks.emplace_back();
      chunks.back().reserve(wire_capacity);
    }
    chunks.back().insert(
      chunks.back().end(),
      std::make_move_iterator(run.begin()),
      std::make_move_iterator(run.end()));
  }
  return chunks;
}

struct Stage2ReconcileChunk {
  std::uint64_t request_id{};
  Stage2ContextHandle context{};
  std::uint32_t barrier_epoch{};
  std::uint32_t target_shard{};
  std::uint32_t begin{};
  std::uint32_t item_count{};
  std::uint32_t attempts_started{};
  std::uint64_t deadline_ns{};
  bool attempt_active{};
  bool posted{};
  bool complete{};

  [[nodiscard]] bool correlates(Stage2ContextHandle expected_context,
                                std::uint32_t expected_epoch) const noexcept {
    return context == expected_context && barrier_epoch == expected_epoch;
  }
};

// One context-local, bounded reconciliation barrier. The exact operation
// payload remains immutable from the first post through every transport retry.
// Offsets, rather than pointers, keep chunks valid if the payload vector grows
// while the barrier is initially assembled. A fresh epoch fences a late ACK
// from an earlier install/removal barrier in the same context slot.
class Stage2ReconcileBatchState {
 public:
  using Op = service::storage_owner::ReconcileReverseOp;

  void reserve(std::size_t op_capacity, std::size_t chunk_capacity) {
    ops_.reserve(op_capacity);
    chunks_.reserve(chunk_capacity);
  }

  void begin(Stage2ContextHandle context, Stage2ReconcileBarrier barrier) {
    advance_epoch();
    context_ = context;
    barrier_ = barrier;
    active_ = true;
    remaining_ = 0;
    ops_.clear();
    chunks_.clear();
  }

  [[nodiscard]] bool append_chunk(
      std::uint64_t request_id, std::uint32_t target_shard,
      std::span<const Op> ops) {
    if (!active_ || request_id == 0 || context_.generation == 0 ||
        barrier_ == Stage2ReconcileBarrier::none || ops.empty() ||
        ops.size() > static_cast<std::size_t>(
                       std::numeric_limits<std::uint32_t>::max()) ||
        ops_.size() > static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max()) -
                      ops.size()) {
      return false;
    }
    const std::uint32_t begin = static_cast<std::uint32_t>(ops_.size());
    ops_.insert(ops_.end(), ops.begin(), ops.end());
    chunks_.push_back(Stage2ReconcileChunk{
      .request_id = request_id,
      .context = context_,
      .barrier_epoch = epoch_,
      .target_shard = target_shard,
      .begin = begin,
      .item_count = static_cast<std::uint32_t>(ops.size()),
    });
    ++remaining_;
    return true;
  }

  [[nodiscard]] std::span<const Op> payload(
      const Stage2ReconcileChunk& chunk) const {
    if (!chunk.correlates(context_, epoch_) ||
        chunk.begin > ops_.size() ||
        chunk.item_count > ops_.size() - chunk.begin) {
      return {};
    }
    return std::span<const Op>{ops_.data() + chunk.begin, chunk.item_count};
  }

  [[nodiscard]] bool mark_complete(std::size_t index,
                                   Stage2ContextHandle context,
                                   std::uint32_t barrier_epoch) {
    if (!active_ || index >= chunks_.size()) return false;
    Stage2ReconcileChunk& chunk = chunks_[index];
    if (!chunk.correlates(context, barrier_epoch) ||
        !chunk.correlates(context_, epoch_)) {
      return false;
    }
    if (chunk.complete) return true;
    chunk.complete = true;
    chunk.posted = false;
    chunk.attempt_active = false;
    if (remaining_ == 0) return false;
    --remaining_;
    return true;
  }

  void clear() {
    advance_epoch();
    context_ = {};
    barrier_ = Stage2ReconcileBarrier::none;
    active_ = false;
    remaining_ = 0;
    ops_.clear();
    chunks_.clear();
  }

  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] bool complete() const noexcept {
    return active_ && remaining_ == 0;
  }
  [[nodiscard]] std::size_t remaining() const noexcept { return remaining_; }
  [[nodiscard]] std::uint32_t epoch() const noexcept { return epoch_; }
  [[nodiscard]] Stage2ContextHandle context() const noexcept {
    return context_;
  }
  [[nodiscard]] Stage2ReconcileBarrier barrier() const noexcept {
    return barrier_;
  }
  [[nodiscard]] std::vector<Stage2ReconcileChunk>& chunks() noexcept {
    return chunks_;
  }
  [[nodiscard]] const std::vector<Stage2ReconcileChunk>& chunks() const
      noexcept {
    return chunks_;
  }

 private:
  void advance_epoch() noexcept {
    ++epoch_;
    if (epoch_ == 0) ++epoch_;
  }

  Stage2ContextHandle context_{};
  Stage2ReconcileBarrier barrier_{Stage2ReconcileBarrier::none};
  std::uint32_t epoch_{};
  bool active_{};
  std::size_t remaining_{};
  std::vector<Op> ops_;
  std::vector<Stage2ReconcileChunk> chunks_;
};

}  // namespace memory_node_storage_owner_maintenance_detail
