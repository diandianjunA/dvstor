#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace memory_node_storage_owner_maintenance_detail {

enum class Stage2AdmissionDecision : std::uint8_t {
  admit,
  unavailable,
  foreground_pressure,
};

inline std::size_t saturating_admission_multiply(
    std::size_t lhs, std::size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return std::numeric_limits<std::size_t>::max();
  }
  return lhs * rhs;
}

// Stage2 execution is intentionally much smaller than the accepted
// descriptor window.  Four B8 contexts per executor is the safe production
// floor established by the bounded-pipeline fix; six is the largest adaptive
// trial.  In particular, this controller can never recreate the historical
// rpc-depth (128-context) expansion or admit a semantic B16/B32 context.
inline constexpr std::size_t kStage2BaselineContextsPerWorker = 4;
inline constexpr std::size_t kStage2MaximumContextsPerWorker = 6;
inline constexpr std::uint32_t kStage2BudgetPromotionObservations = 2;
inline constexpr std::uint32_t kStage2BudgetLaneRollbackObservations = 2;
inline constexpr std::uint32_t kStage2BudgetLowBacklogObservations = 4;
inline constexpr std::uint32_t kStage2BudgetCooldownObservations = 2;
inline constexpr std::uint32_t kStage2BudgetRateTrialObservations = 2;
inline constexpr std::uint32_t kStage2BudgetRateRollbackCooldown = 8;

enum class Stage2ExecutionBudgetAction : std::uint8_t {
  hold,
  promote,
  rollback_lane_pressure,
  rollback_low_backlog,
  rollback_rate_regression,
};

struct Stage2ExecutionBudgetPolicyState {
  std::size_t contexts_per_worker{kStage2BaselineContextsPerWorker};
  std::size_t promotion_ceiling_contexts_per_worker{
    kStage2MaximumContextsPerWorker};
  std::uint32_t promotion_streak{};
  std::uint32_t lane_pressure_streak{};
  std::uint32_t low_backlog_streak{};
  std::uint32_t cooldown{};
  std::uint64_t stable_finalized_rate_milli_per_sec{};
  std::uint64_t trial_baseline_rate_milli_per_sec{};
  std::uint32_t trial_regression_streak{};
  std::uint32_t trial_success_streak{};
  bool rate_trial_pending{};
};

struct Stage2ExecutionBudgetSample {
  std::size_t visible_backlog{};
  std::size_t accepted_window{};
  std::size_t active_search_lanes{};
  std::size_t search_lane_limit{};
  std::uint64_t search_lane_blocks_since_last{};
  // maintenance_processed delta normalized by the actual elapsed sample
  // duration. Milli-ops/s preserves useful precision without floating state.
  std::uint64_t finalized_rate_milli_per_sec{};
  bool finalized_rate_available{};
};

struct Stage2ExecutionBudgetDecision {
  Stage2ExecutionBudgetPolicyState state;
  Stage2ExecutionBudgetAction action{Stage2ExecutionBudgetAction::hold};
  bool high_backlog{};
  bool lane_headroom{};
  bool rate_trial_accepted{};
};

// Serializes the complete sampling transaction, not only its cadence
// timestamp. Publishing last_sample_ns at the beginning of a slow sample is
// insufficient: after one second another worker could otherwise overlap the
// lane-block exchange, streak updates, and limit publication. The owning
// guard releases on every return path.
class Stage2ExecutionBudgetSampleGuard {
 public:
  explicit Stage2ExecutionBudgetSampleGuard(std::atomic<bool>& busy)
      : busy_(&busy) {
    bool expected = false;
    owns_ = busy_->compare_exchange_strong(
      expected, true, std::memory_order_acquire,
      std::memory_order_relaxed);
  }

  Stage2ExecutionBudgetSampleGuard(
      const Stage2ExecutionBudgetSampleGuard&) = delete;
  Stage2ExecutionBudgetSampleGuard& operator=(
      const Stage2ExecutionBudgetSampleGuard&) = delete;

  ~Stage2ExecutionBudgetSampleGuard() {
    if (owns_) busy_->store(false, std::memory_order_release);
  }

  bool owns_sample() const { return owns_; }

 private:
  std::atomic<bool>* busy_{};
  bool owns_{};
};

inline constexpr std::size_t stage2_ceil_fraction(
    std::size_t value, std::size_t denominator) {
  return denominator == 0 ? value :
    value / denominator + (value % denominator != 0 ? 1 : 0);
}

inline constexpr std::uint32_t stage2_saturating_streak_increment(
    std::uint32_t value) {
  return value == std::numeric_limits<std::uint32_t>::max()
    ? value : value + 1;
}

inline constexpr std::uint64_t stage2_rate_ewma(
    std::uint64_t previous, std::uint64_t sample) {
  if (previous == 0) return sample;
  // previous + (sample - previous) / 4 without overflowing either direction.
  return sample >= previous
    ? previous + (sample - previous) / 4
    : previous - (previous - sample) / 4;
}

inline constexpr std::uint64_t stage2_normalized_rate_milli_per_sec(
    std::uint64_t completed_delta, std::uint64_t elapsed_ns) {
  constexpr std::uint64_t kMilliPerNanosecondSecond =
    1'000'000'000'000ULL;
  if (elapsed_ns == 0) return 0;
  if (completed_delta > std::numeric_limits<std::uint64_t>::max() /
                          kMilliPerNanosecondSecond) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return completed_delta * kMilliPerNanosecondSecond / elapsed_ns;
}

// Advance a one-second execution-budget sample. Promotions need sustained
// visible debt and at least 1/8 of the search-lane lease still unused. Lane
// blocking and an almost-drained queue use different rollback streaks, while
// a cooldown keeps the adjacent C32/C40/C48 levels from oscillating. No input
// describes foreground activity: the exact same policy and bound apply while
// clients are running and during an explicit maintenance drain.
inline Stage2ExecutionBudgetDecision decide_stage2_execution_budget(
    Stage2ExecutionBudgetPolicyState state,
    const Stage2ExecutionBudgetSample& sample,
    std::size_t baseline_contexts_per_worker,
    std::size_t maximum_contexts_per_worker) {
  const std::size_t baseline = std::max<std::size_t>(
    1, baseline_contexts_per_worker);
  const std::size_t maximum = std::max(
    baseline, maximum_contexts_per_worker);
  state.contexts_per_worker = std::clamp(
    state.contexts_per_worker, baseline, maximum);
  state.promotion_ceiling_contexts_per_worker = std::clamp(
    state.promotion_ceiling_contexts_per_worker, baseline, maximum);
  const std::size_t promotion_ceiling =
    state.promotion_ceiling_contexts_per_worker;

  const std::size_t high_backlog_threshold = std::max<std::size_t>(
    8, stage2_ceil_fraction(sample.accepted_window, 8));
  // Keep the hysteresis bands disjoint even on a tiny accepted window.  A
  // max(8, ...) floor on both sides makes queue==8 simultaneously high and
  // low when accepted_window<=64, which can alternate promotion and
  // low-backlog rollback forever on small test/deployment geometries.
  const std::size_t low_backlog_threshold = std::min(
    high_backlog_threshold - 1,
    std::max<std::size_t>(1,
      stage2_ceil_fraction(sample.accepted_window, 64)));
  const bool high_backlog =
    sample.visible_backlog >= high_backlog_threshold;
  const bool low_backlog =
    sample.visible_backlog <= low_backlog_threshold;
  const bool lane_headroom = sample.search_lane_limit != 0 &&
    sample.active_search_lanes < sample.search_lane_limit &&
    sample.active_search_lanes <=
      (sample.search_lane_limit / 8) * 7 +
        ((sample.search_lane_limit % 8) * 7) / 8;
  // A fully utilized lane pool is healthy service, not by itself a rollback
  // signal. It prevents another promotion through lane_headroom; contraction
  // requires at least one semantic B8 cohort of blocked lease attempts in
  // consecutive sample windows, so a rare handoff race cannot cause churn.
  const bool lane_pressure =
    sample.search_lane_blocks_since_last >= 8;

  Stage2ExecutionBudgetDecision decision{
    .state = state,
    .action = Stage2ExecutionBudgetAction::hold,
    .high_backlog = high_backlog,
    .lane_headroom = lane_headroom,
    .rate_trial_accepted = false,
  };
  auto& next = decision.state;
  if (next.cooldown != 0) {
    --next.cooldown;
    next.promotion_streak = 0;
    next.lane_pressure_streak = 0;
    next.low_backlog_streak = 0;
    return decision;
  }

  // Only a saturated/high-debt interval represents executor service
  // capacity. Low-backlog samples are arrival-limited and must not depress the
  // stable rate used by the next promotion trial.
  if (!next.rate_trial_pending && high_backlog &&
      sample.finalized_rate_available) {
    next.stable_finalized_rate_milli_per_sec = stage2_rate_ewma(
      next.stable_finalized_rate_milli_per_sec,
      sample.finalized_rate_milli_per_sec);
  }

  if (next.contexts_per_worker > baseline && lane_pressure) {
    next.lane_pressure_streak = stage2_saturating_streak_increment(
      next.lane_pressure_streak);
  } else {
    next.lane_pressure_streak = 0;
  }
  if (next.contexts_per_worker > baseline && low_backlog) {
    next.low_backlog_streak = stage2_saturating_streak_increment(
      next.low_backlog_streak);
  } else {
    next.low_backlog_streak = 0;
  }
  if (!next.rate_trial_pending &&
      next.contexts_per_worker < promotion_ceiling && high_backlog &&
      lane_headroom && !lane_pressure && sample.finalized_rate_available) {
    next.promotion_streak = stage2_saturating_streak_increment(
      next.promotion_streak);
  } else {
    next.promotion_streak = 0;
  }

  if (next.lane_pressure_streak >=
      kStage2BudgetLaneRollbackObservations) {
    --next.contexts_per_worker;
    decision.action = Stage2ExecutionBudgetAction::rollback_lane_pressure;
  } else if (next.low_backlog_streak >=
             kStage2BudgetLowBacklogObservations) {
    --next.contexts_per_worker;
    decision.action = Stage2ExecutionBudgetAction::rollback_low_backlog;
  } else if (next.rate_trial_pending && high_backlog &&
             sample.finalized_rate_available) {
    const std::uint64_t regression_threshold =
      next.trial_baseline_rate_milli_per_sec -
        next.trial_baseline_rate_milli_per_sec / 20;
    const bool regression =
      next.trial_baseline_rate_milli_per_sec != 0 &&
      sample.finalized_rate_milli_per_sec < regression_threshold;
    if (regression) {
      next.trial_regression_streak = stage2_saturating_streak_increment(
        next.trial_regression_streak);
      next.trial_success_streak = 0;
    } else {
      next.trial_success_streak = stage2_saturating_streak_increment(
        next.trial_success_streak);
      next.trial_regression_streak = 0;
    }
    if (next.trial_regression_streak >=
        kStage2BudgetRateTrialObservations) {
      --next.contexts_per_worker;
      // A repeatable >5% throughput regression is stronger evidence than a
      // transient resource signal. Fuse this process at the last good tier so
      // it cannot spend four seconds in the same known-bad tier every retry
      // cycle. Restarting the process deliberately resets the experiment.
      next.promotion_ceiling_contexts_per_worker =
        next.contexts_per_worker;
      next.stable_finalized_rate_milli_per_sec =
        next.trial_baseline_rate_milli_per_sec;
      next.rate_trial_pending = false;
      decision.action =
        Stage2ExecutionBudgetAction::rollback_rate_regression;
    } else if (next.trial_success_streak >=
               kStage2BudgetRateTrialObservations) {
      next.rate_trial_pending = false;
      next.stable_finalized_rate_milli_per_sec =
        sample.finalized_rate_milli_per_sec;
      next.trial_regression_streak = 0;
      next.trial_success_streak = 0;
      decision.rate_trial_accepted = true;
    }
  } else if (next.promotion_streak >=
             kStage2BudgetPromotionObservations) {
    next.trial_baseline_rate_milli_per_sec =
      next.stable_finalized_rate_milli_per_sec;
    next.rate_trial_pending = true;
    next.trial_regression_streak = 0;
    next.trial_success_streak = 0;
    ++next.contexts_per_worker;
    decision.action = Stage2ExecutionBudgetAction::promote;
  }

  if (decision.action != Stage2ExecutionBudgetAction::hold) {
    next.promotion_streak = 0;
    next.lane_pressure_streak = 0;
    next.low_backlog_streak = 0;
    if (decision.action != Stage2ExecutionBudgetAction::promote) {
      if (decision.action ==
          Stage2ExecutionBudgetAction::rollback_rate_regression) {
        // The explicit regression path already restored the prior tier's
        // baseline above.
      } else if (next.rate_trial_pending) {
        next.stable_finalized_rate_milli_per_sec =
          next.trial_baseline_rate_milli_per_sec;
      } else {
        // A non-trial lane/low-debt contraction changed tiers without a saved
        // lower-tier baseline. Relearn it before another promotion.
        next.stable_finalized_rate_milli_per_sec = 0;
      }
      next.rate_trial_pending = false;
      next.trial_regression_streak = 0;
      next.trial_success_streak = 0;
    }
    next.cooldown = decision.action ==
        Stage2ExecutionBudgetAction::rollback_rate_regression
      ? kStage2BudgetRateRollbackCooldown
      : kStage2BudgetCooldownObservations;
  }
  return decision;
}

// Bound accepted Stage2 descriptors independently of active execution
// resources. Stage1 may acknowledge every descriptor in this window after it
// is assigned a completion sequence and published to the maintenance queue;
// workers claim their separately bounded context/lane resources only later.
//
// The queue bound protects descriptor memory, while the completion-ring bound
// prevents a modulo cell from being reused before the contiguous durable
// watermark has crossed it. The accepted window must respect both. Neither
// maintenance worker count nor RPC depth belongs here: tying acceptance to
// active contexts recreates a completion-clocked foreground pipeline and
// prevents the queue from accumulating a useful Stage2 batch.
inline std::size_t stage2_accepted_sequence_limit(
    std::size_t maintenance_queue_depth,
    std::size_t completion_capacity) {
  return std::min(maintenance_queue_depth, completion_capacity);
}

// Stage2 contexts are execution resources, not foreground acceptance credit.
// Apply the same requested bound across foreground and drain: the adaptive
// controller may move it only from four through six, while expanding to the
// full RPC depth when the foreground becomes idle is forbidden. That old
// transition moved thousands of tasks out of the visible queue into
// whole-context barrier chains precisely when drain needed bounded independent
// work. The existing global/per-peer RDMA credits still bound posted work,
// while a depth-one configuration naturally retains the single-context floor.
inline std::size_t stage2_context_admission_limit(
    std::size_t maintenance_workers,
    std::size_t rpc_depth,
    bool foreground_pressure,
    std::size_t requested_contexts_per_worker =
      kStage2BaselineContextsPerWorker) {
  (void)foreground_pressure;
  const std::size_t workers = std::max<std::size_t>(1, maintenance_workers);
  const std::size_t depth = std::max<std::size_t>(1, rpc_depth);
  const std::size_t contexts_per_worker = std::min(
    depth, std::max<std::size_t>(1, requested_contexts_per_worker));
  return saturating_admission_multiply(workers, contexts_per_worker);
}

// The global context counter above is a debt/scratch bound, not a fair-share
// scheduler.  Without this local bound, the first OS worker to run can claim
// the entire global allowance even though it owns only its own small search
// lane pool; the other workers and their RDMA lanes then remain idle.  Apply
// the same per-worker share before touching the global counter.  This changes
// only which executor owns an admitted context, never the search, completion
// window, or amount of acknowledged work.
inline std::size_t stage2_worker_context_admission_limit(
    std::size_t rpc_depth,
    bool foreground_pressure,
    std::size_t requested_contexts_per_worker =
      kStage2BaselineContextsPerWorker) {
  (void)foreground_pressure;
  const std::size_t depth = std::max<std::size_t>(1, rpc_depth);
  return std::min(
    depth, std::max<std::size_t>(1, requested_contexts_per_worker));
}

// Active Stage2 tasks are a second, node-wide execution bound. Context count
// alone was insufficient while one context could consume a 16/32-item cohort:
// even 32 contexts could hide 1024 coupled search/prune chains from the queue.
// Budget four eight-task semantic slices per executor (256 tasks for the
// tested 8-worker production geometry) and leave later accepted descriptors
// visible. This is deliberately unrelated to Stage1 ACK admission; the exact
// task account also remains correct if later scheduling changes make context
// sizes non-uniform.
inline std::size_t stage2_active_task_limit(
    std::size_t maintenance_workers,
    std::size_t semantic_execution_batch,
    std::size_t accepted_descriptor_limit,
    std::size_t contexts_per_worker =
      kStage2BaselineContextsPerWorker) {
  const std::size_t workers = std::max<std::size_t>(1, maintenance_workers);
  const std::size_t batch = std::max<std::size_t>(1,
                                                   semantic_execution_batch);
  const std::size_t contexts = saturating_admission_multiply(
    workers, std::max<std::size_t>(1, contexts_per_worker));
  return std::min(
    accepted_descriptor_limit,
    saturating_admission_multiply(contexts, batch));
}

inline bool stage2_active_task_reservation_available(
    std::uint32_t active_tasks,
    std::uint32_t requested_tasks,
    std::uint32_t limit) {
  return requested_tasks != 0 && active_tasks <= limit &&
    requested_tasks <= limit - active_tasks;
}

inline bool try_reserve_stage2_active_tasks(
    std::atomic<std::uint32_t>& active_tasks,
    std::uint32_t requested_tasks,
    std::uint32_t limit) {
  std::uint32_t active = active_tasks.load(std::memory_order_acquire);
  while (stage2_active_task_reservation_available(
      active, requested_tasks, limit)) {
    if (active_tasks.compare_exchange_weak(
          active, active + requested_tasks,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

inline bool try_release_stage2_active_tasks(
    std::atomic<std::uint32_t>& active_tasks,
    std::uint32_t released_tasks) {
  if (released_tasks == 0) return false;
  std::uint32_t active = active_tasks.load(std::memory_order_acquire);
  while (released_tasks <= active) {
    if (active_tasks.compare_exchange_weak(
          active, active - released_tasks,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

// Credit-return callbacks can race and observe the same completion-ring
// availability. Runnable waiter coverage is a scheduler-only claim that makes
// those snapshots idempotent without reserving a maintenance sequence.
inline std::size_t stage1_waiter_uncovered_wake_capacity(
    std::size_t resource_available,
    std::size_t runnable_coverage) {
  return resource_available > runnable_coverage
    ? resource_available - runnable_coverage : 0;
}

// Wake an oversized FIFO head even when only one semantic credit is visible;
// its existing per-token fallback makes partial progress. Cover its *whole*
// demand as a scheduling baton: if another 31 credits arrive before this
// 32-item request runs, a second whole request must not be woken for the same
// eventual capacity. This is not durable debt or a sequence reservation.
inline std::size_t stage1_waiter_head_wake_coverage(
    std::size_t item_count,
    std::size_t uncovered_capacity) {
  if (item_count == 0 || uncovered_capacity == 0) return 0;
  return item_count;
}

// A Stage1 arm permit bridges the queue-capacity check and the try-only
// completion-ring transaction. Other producers must include those permits in
// their queue-capacity test or they could steal a runnable slot after arm has
// reserved a sequence.
inline bool maintenance_queue_permit_available(
    std::size_t runnable_tasks,
    std::size_t reserved_slots,
    std::size_t capacity) {
  return runnable_tasks < capacity &&
    reserved_slots < capacity - runnable_tasks;
}

// A control RPC is one admission transaction.  Claiming its queue permits
// item-by-item can leave a partial batch runnable while the caller is still
// waiting for the remaining items.  Those runnable tasks may in turn wait for
// the caller's authority commit, creating a closed wait cycle.  Admit all
// items together or leave the queue unchanged.
inline bool maintenance_queue_batch_permit_available(
    std::size_t runnable_tasks,
    std::size_t reserved_slots,
    std::size_t requested_slots,
    std::size_t capacity) {
  if (requested_slots == 0 || runnable_tasks > capacity ||
      reserved_slots > capacity - runnable_tasks) {
    return false;
  }
  return requested_slots <=
    capacity - runnable_tasks - reserved_slots;
}

// Update the shared permit account only when the entire batch fits. Keeping
// this arithmetic in one helper makes both the no-partial-admission property
// and the transient-failure no-op directly testable.
inline bool try_acquire_maintenance_queue_batch_permit(
    std::size_t runnable_tasks,
    std::size_t requested_slots,
    std::size_t capacity,
    std::size_t& reserved_slots) {
  if (!maintenance_queue_batch_permit_available(
        runnable_tasks, reserved_slots, requested_slots, capacity)) {
    return false;
  }
  reserved_slots += requested_slots;
  return true;
}

// Release is similarly checked so a failed completion-ring try cannot wrap
// the reservation account and silently expose more queue capacity than exists.
inline bool release_maintenance_queue_batch_permit(
    std::size_t released_slots,
    std::size_t& reserved_slots) {
  if (released_slots == 0 || released_slots > reserved_slots) return false;
  reserved_slots -= released_slots;
  return true;
}

// Avoid calling the pressure probe when this executor cannot admit work at
// all. The production probe may poll a shared peer CQ, so shutdown/full paths
// must remain side-effect free.
template <class ForegroundPressureProbe>
Stage2AdmissionDecision decide_stage2_admission(
    bool local_contexts_full,
    bool shutting_down,
    ForegroundPressureProbe&& foreground_pressure) {
  if (local_contexts_full || shutting_down) {
    return Stage2AdmissionDecision::unavailable;
  }
  return std::forward<ForegroundPressureProbe>(foreground_pressure)()
           ? Stage2AdmissionDecision::foreground_pressure
           : Stage2AdmissionDecision::admit;
}

}  // namespace memory_node_storage_owner_maintenance_detail
