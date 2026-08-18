#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>

namespace memory_node_storage_owner_maintenance_detail {

// 5 ms is long enough to collect roughly two extra arrivals at the measured
// per-shard update rates, but is still two orders of magnitude below the
// observed Stage2 tail. It is the target-four safety ceiling, not a configured
// target: the controller starts at the legacy delay and uses it only during a
// bounded trial under real completion-window pressure.
inline constexpr std::uint32_t kStage2AdaptivePackingMaxWaitUs = 5'000;
// A 512-task window is several times larger than the complete admission
// cohort in the production profile. This avoids promoting on only the first,
// unusually fast contexts to finish while still adapting within seconds.
inline constexpr std::uint64_t kStage2PackingEvaluationTasks = 512;
inline constexpr std::uint64_t kStage2PackingRollbackCooldownTasks = 2'048;
inline constexpr std::uint64_t kStage2PackingMaxRollbackCooldownTasks =
  32'768;
// Probe often at first, then exponentially reduce the tax after repeated
// confirmations. Absolute target-four regressions are still checked every
// 512 tasks; the periodic legacy probe detects only relative drift where both
// policies improved but target two improved more. At the steady 64-window
// interval the slow-path sample is 1/65 of work and still refreshes within
// roughly eleven seconds at 3K updates/s.
inline constexpr std::uint64_t kStage2PackingInitialProbeIntervalWindows = 8;
inline constexpr std::uint64_t kStage2PackingMaxProbeIntervalWindows = 64;

// Stage2 is a background graph-quality pipeline.  Once Stage1 acceptance is
// decoupled from active execution, the runnable queue is the authoritative
// batching signal: do not recreate the old completion-clocked two-item stream
// by treating every newly visible descriptor as a latency-sensitive tail.
// Eight is the smallest context that materially amortizes the per-context
// wave/reconcile machinery; 16 and 32 are backlog tiers, capped by the
// configured wire/scratch bound.
inline constexpr std::size_t kStage2BulkMinimumBatch = 8;
inline constexpr std::size_t kStage2BulkMiddleBatch = 16;
inline constexpr std::size_t kStage2BulkMaximumBatch = 32;
// A visible backlog may be collected and scheduled in 16/32-item cohorts,
// but the current Stage2 state machine is not a per-item pipeline yet.  One
// Stage2Context crosses authority, freeze/prune, reconcile, placement and
// membership barriers as a unit; a retryable item therefore stalls every
// sibling in that context.  Keep the semantic execution slice at eight until
// those barriers can detach completed/blocked items independently.  This
// still exposes a 4x larger batch than the old two-item completion clock while
// allowing a deep accepted queue to become several independently progressing
// contexts instead of one convoy.
inline constexpr std::size_t kStage2SemanticExecutionBatch = 8;
// A background tail must remain in the accepted queue long enough for the
// next arrivals to join it.  The old 50 us execution-credit deadline is far
// shorter than the measured 1--3 ms per-node arrival interval and therefore
// lets idle context slots consume descriptors one by one.  Predict the time
// needed to reach eight from the enqueue EWMA, bounded so sparse workloads
// still advance their durable watermark.
inline constexpr std::uint32_t kStage2BulkTailMinWaitUs = 1'000;
inline constexpr std::uint32_t kStage2BulkTailMaxWaitUs = 25'000;

inline constexpr bool stage2_bulk_packing_enabled(
    std::size_t configured_batch_limit) {
  return configured_batch_limit >= kStage2BulkMinimumBatch;
}

inline constexpr std::size_t stage2_execution_slice_limit(
    std::size_t selected_pop_limit,
    std::size_t configured_batch_limit) {
  return std::max<std::size_t>(1, std::min({
    std::max<std::size_t>(1, selected_pop_limit),
    std::max<std::size_t>(1, configured_batch_limit),
    kStage2SemanticExecutionBatch}));
}

// Select from already-visible work only.  In particular, this helper never
// predicts or waits for the next arrival.  A queue between tiers is split into
// independently progressing contexts (for example 24 -> 16 + 8) instead of
// being collapsed into one oversized context that would reduce executor
// parallelism.
inline constexpr std::size_t stage2_visible_backlog_target(
    std::size_t queued_tasks, std::size_t configured_batch_limit) {
  const std::size_t batch_limit =
    std::max<std::size_t>(1, configured_batch_limit);
  if (!stage2_bulk_packing_enabled(batch_limit)) {
    return std::min<std::size_t>(2, batch_limit);
  }
  if (batch_limit >= kStage2BulkMaximumBatch &&
      queued_tasks >= kStage2BulkMaximumBatch) {
    return kStage2BulkMaximumBatch;
  }
  if (batch_limit >= kStage2BulkMiddleBatch &&
      queued_tasks >= kStage2BulkMiddleBatch) {
    return kStage2BulkMiddleBatch;
  }
  return std::min(kStage2BulkMinimumBatch, batch_limit);
}

// Failed trials are evidence that the current workload does not amortize a
// larger context. Retrying at a fixed cadence repeatedly pays the same 512-task
// loss, so consecutive failures back off geometrically. Saturation preserves
// eventual re-probing after workload recovery without allowing unbounded state
// or a dataset-specific time constant.
inline constexpr std::uint64_t stage2_packing_rollback_cooldown_tasks(
    std::uint32_t consecutive_failures) {
  if (consecutive_failures == 0) return 0;
  constexpr std::uint32_t max_shift = 4;
  const std::uint32_t shift = std::min<std::uint32_t>(
    consecutive_failures - 1, max_shift);
  return std::min<std::uint64_t>(
    kStage2PackingRollbackCooldownTasks << shift,
    kStage2PackingMaxRollbackCooldownTasks);
}

enum class Stage2PackingFlushReason : std::uint8_t {
  none,
  full,
  target,
  deadline,
  low_pressure,
  cleanup,
};

struct Stage2PackingParameters {
  std::size_t target_batch{1};
  std::uint32_t estimated_arrival_interval_us{};
  std::uint64_t rollback_cooldown_tasks{};
  std::uint32_t consecutive_trial_failures{};
  bool larger_batch_trials_disabled{};
  bool target2_probe_active{};
  std::uint64_t probe_interval_windows{
    kStage2PackingInitialProbeIntervalWindows};
};

struct Stage2PackingDecision {
  bool ready{};
  std::size_t pop_limit{1};
  std::uint32_t target_batch{1};
  std::uint32_t wait_budget_us{};
  Stage2PackingFlushReason reason{Stage2PackingFlushReason::none};
  std::optional<std::chrono::steady_clock::time_point> deadline;
};

struct Stage2PackingTelemetry {
  std::uint64_t target_batch{1};
  std::uint64_t estimated_arrival_interval_us{};
  std::uint64_t waited_batches{};
  std::uint64_t wait_ns{};
  std::uint64_t target_flushes{};
  std::uint64_t deadline_flushes{};
  std::uint64_t full_flushes{};
  std::uint64_t low_pressure_flushes{};
  std::uint64_t cleanup_flushes{};
  std::uint64_t promotions{};
  std::uint64_t rollbacks{};
  std::uint64_t accepted_trial_windows{};
  std::uint64_t admitted_queue_depth_sum{};
  std::uint64_t admitted_queue_depth_max{};
  std::uint64_t batch_1_to_7{};
  std::uint64_t batch_8_to_15{};
  std::uint64_t batch_16_to_31{};
  std::uint64_t batch_32_plus{};
  std::uint64_t bulk_assembly_batches{};
  std::uint64_t bulk_assembly_wait_ns{};
};

// Select an immediate batch from visible backlog.  Only a sub-eight tail may
// retain the configured legacy deadline; small configured batch limits keep
// the old target-four feedback path for compatibility.
inline Stage2PackingDecision decide_stage2_packing(
    std::size_t queued_tasks,
    std::size_t configured_batch_limit,
    std::size_t adaptive_target,
    std::chrono::steady_clock::time_point oldest_queued_at,
    std::chrono::steady_clock::time_point now,
    std::uint32_t legacy_wait_us,
    std::uint32_t estimated_arrival_interval_us,
    bool high_pressure) {
  Stage2PackingDecision decision;
  const std::size_t batch_limit =
    std::max<std::size_t>(1, configured_batch_limit);
  const bool bulk_packing = stage2_bulk_packing_enabled(batch_limit);
  decision.target_batch = bulk_packing
    ? stage2_visible_backlog_target(queued_tasks, batch_limit)
    : std::clamp<std::size_t>(adaptive_target, 1, batch_limit);
  // A target-four policy is meaningful only while completion/queue pressure
  // supplies both enough arrivals and a measurable capacity signal. Outside
  // that trial domain select the legacy target and flush immediately: 185634
  // sent 99,009 of 99,028 contexts through the deadline path while both batch
  // size and end-to-end throughput regressed. There is no batching benefit to
  // justify putting isolated arrivals on a timer.
  if (!bulk_packing && !high_pressure) {
    decision.target_batch = std::min<std::size_t>(2, batch_limit);
  }
  // Under real pressure, target <=2 is the measured legacy/baseline path:
  // collect until its bounded deadline, then drain up to the wire limit. The
  // low-pressure branch below bypasses that timer, and the rollback path still
  // must not cap an already-available batch at two.
  decision.pop_limit = bulk_packing
    ? decision.target_batch
    : (decision.target_batch <= 2 ? batch_limit : decision.target_batch);
  if (queued_tasks == 0) return decision;

  if (queued_tasks >= batch_limit) {
    decision.ready = true;
    decision.pop_limit = batch_limit;
    decision.target_batch = batch_limit;
    decision.reason = Stage2PackingFlushReason::full;
    return decision;
  }
  // A visible bulk tier is ready now.  Do not add an arrival-rate wait: the
  // accepted backlog already contains everything needed to form this context.
  if (bulk_packing && queued_tasks >= decision.target_batch) {
    decision.ready = true;
    decision.reason = Stage2PackingFlushReason::target;
    return decision;
  }
  if (!bulk_packing && !high_pressure) {
    decision.ready = true;
    decision.reason = Stage2PackingFlushReason::low_pressure;
    return decision;
  }
  if (!bulk_packing && legacy_wait_us == 0) {
    decision.ready = true;
    decision.reason = Stage2PackingFlushReason::low_pressure;
    return decision;
  }
  if (!bulk_packing && decision.target_batch > 2 &&
      queued_tasks >= decision.target_batch) {
    decision.ready = true;
    decision.reason = Stage2PackingFlushReason::target;
    return decision;
  }

  // Target two is the rollback/baseline mode and retains the old bound.  A
  // target-four trial predicts only the time needed for its missing arrivals.
  // With no estimate yet, use the safety ceiling once; feedback will either
  // validate target four or restore the legacy path within one bounded
  // evaluation window.
  if (bulk_packing) {
    std::uint64_t predicted_us = kStage2BulkTailMaxWaitUs;
    if (estimated_arrival_interval_us != 0) {
      // The deadline is anchored at the oldest descriptor. Predict the time
      // from that first descriptor to a complete target, not merely the time
      // still missing at this observation; otherwise the already elapsed
      // interval is subtracted twice and a steady stream flushes at five or
      // six items before it can ever reach eight.
      predicted_us = static_cast<std::uint64_t>(
        estimated_arrival_interval_us) * (decision.target_batch - 1);
    }
    decision.wait_budget_us = static_cast<std::uint32_t>(std::clamp<
      std::uint64_t>(predicted_us,
                    kStage2BulkTailMinWaitUs,
                    kStage2BulkTailMaxWaitUs));
  } else if (decision.target_batch <= 2) {
    decision.wait_budget_us = std::min(
      legacy_wait_us, kStage2AdaptivePackingMaxWaitUs);
  } else {
    constexpr std::uint32_t adaptive_max_wait_us =
      kStage2AdaptivePackingMaxWaitUs;
    const std::size_t missing = decision.target_batch - queued_tasks;
    std::uint64_t predicted_us = adaptive_max_wait_us;
    if (estimated_arrival_interval_us != 0) {
      predicted_us = static_cast<std::uint64_t>(
        estimated_arrival_interval_us) * missing;
    }
    decision.wait_budget_us = static_cast<std::uint32_t>(std::clamp<
      std::uint64_t>(predicted_us,
                    std::min<std::uint32_t>(
                      legacy_wait_us, adaptive_max_wait_us),
                    adaptive_max_wait_us));
  }
  if (decision.wait_budget_us == 0) {
    decision.ready = true;
    decision.reason = Stage2PackingFlushReason::low_pressure;
    return decision;
  }
  decision.deadline = oldest_queued_at +
    std::chrono::microseconds(decision.wait_budget_us);
  if (now >= *decision.deadline) {
    decision.ready = true;
    decision.reason = Stage2PackingFlushReason::deadline;
  }
  return decision;
}

// Shared controller state is tiny and is touched only once per Stage1 enqueue,
// admitted context, and completed context.  No candidate/edge/RDMA hot loop
// takes this lock.  The feedback cost includes the bounded packing wait plus
// context wall time, divided by actual tasks; a larger batch is retained only
// when it reduces per-task cost and does not worsen completion debt.
class Stage2AdaptivePackingController {
public:
  void reset(std::size_t configured_batch_limit,
             std::uint32_t legacy_wait_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    configured_batch_limit_ =
      std::max<std::size_t>(1, configured_batch_limit);
    legacy_wait_us_ = legacy_wait_us;
    target_batch_ = stage2_bulk_packing_enabled(configured_batch_limit_)
      ? std::min(kStage2BulkMinimumBatch, configured_batch_limit_)
      : std::min<std::size_t>(2, configured_batch_limit_);
    last_arrival_ = {};
    arrival_interval_us_ = 0;
    baseline_cost_ns_per_task_ = 0.0;
    baseline_debt_delta_per_task_ = 0.0;
    cooldown_tasks_ = 0;
    consecutive_trial_failures_ = 0;
    accepted_windows_since_probe_ = 0;
    target2_probe_active_ = false;
    target4_revalidation_active_ = false;
    probe_interval_windows_ = kStage2PackingInitialProbeIntervalWindows;
    window_ = {};
    telemetry_ = {};
    telemetry_.target_batch = target_batch_;
  }

  void observe_enqueue(std::chrono::steady_clock::time_point now,
                       std::size_t task_count) {
    if (task_count == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (last_arrival_ != std::chrono::steady_clock::time_point{} &&
        now > last_arrival_) {
      const auto elapsed_us = std::chrono::duration_cast<
        std::chrono::microseconds>(now - last_arrival_).count();
      const std::uint64_t sample_us = std::max<std::uint64_t>(
        1, static_cast<std::uint64_t>(elapsed_us) / task_count);
      // A 1/8 EWMA follows workload drift within a few Stage1 RPCs while
      // filtering scheduler jitter. Clamp only the exported/predicted value;
      // the 5 ms hard deadline remains authoritative.
      arrival_interval_us_ = arrival_interval_us_ == 0
        ? sample_us
        : (arrival_interval_us_ * 7 + sample_us) / 8;
      arrival_interval_us_ = std::min<std::uint64_t>(
        arrival_interval_us_, kStage2AdaptivePackingMaxWaitUs);
    }
    last_arrival_ = now;
  }

  [[nodiscard]] Stage2PackingParameters parameters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return Stage2PackingParameters{
      .target_batch = target_batch_,
      .estimated_arrival_interval_us = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(arrival_interval_us_,
                                std::numeric_limits<std::uint32_t>::max())),
      .rollback_cooldown_tasks = cooldown_tasks_,
      .consecutive_trial_failures = consecutive_trial_failures_,
      .larger_batch_trials_disabled = larger_batch_trials_disabled_,
      .target2_probe_active = target2_probe_active_,
      .probe_interval_windows = probe_interval_windows_,
    };
  }

  void observe_admission(Stage2PackingFlushReason reason,
                         std::size_t actual_batch,
                         std::uint64_t oldest_wait_ns,
                         std::uint32_t wait_budget_us,
                         std::size_t selected_target = 0,
                         std::size_t queued_at_admission = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    switch (reason) {
      case Stage2PackingFlushReason::full:
        ++telemetry_.full_flushes;
        break;
      case Stage2PackingFlushReason::target:
        ++telemetry_.target_flushes;
        break;
      case Stage2PackingFlushReason::deadline:
        ++telemetry_.deadline_flushes;
        break;
      case Stage2PackingFlushReason::low_pressure:
        ++telemetry_.low_pressure_flushes;
        break;
      case Stage2PackingFlushReason::cleanup:
        ++telemetry_.cleanup_flushes;
        break;
      case Stage2PackingFlushReason::none:
        break;
    }
    if (actual_batch != 0) {
      telemetry_.target_batch = std::max<std::uint64_t>(
        telemetry_.target_batch, selected_target);
      telemetry_.admitted_queue_depth_sum += queued_at_admission;
      telemetry_.admitted_queue_depth_max = std::max<std::uint64_t>(
        telemetry_.admitted_queue_depth_max, queued_at_admission);
      if (actual_batch < kStage2BulkMinimumBatch) {
        ++telemetry_.batch_1_to_7;
      } else if (actual_batch < kStage2BulkMiddleBatch) {
        ++telemetry_.batch_8_to_15;
      } else if (actual_batch < kStage2BulkMaximumBatch) {
        ++telemetry_.batch_16_to_31;
      } else {
        ++telemetry_.batch_32_plus;
      }
      if (stage2_bulk_packing_enabled(configured_batch_limit_) &&
          oldest_wait_ns != 0) {
        ++telemetry_.bulk_assembly_batches;
        telemetry_.bulk_assembly_wait_ns += oldest_wait_ns;
      }
    }
    if (wait_budget_us != 0 && actual_batch != 0) {
      const std::uint64_t bounded_wait_ns = std::min<std::uint64_t>(
        oldest_wait_ns,
        static_cast<std::uint64_t>(wait_budget_us) * 1'000);
      if (bounded_wait_ns != 0) {
        ++telemetry_.waited_batches;
        telemetry_.wait_ns += bounded_wait_ns;
      }
    }
  }

  void observe_completion(std::size_t sampled_target,
                          bool high_pressure,
                          std::size_t actual_batch,
                          std::uint64_t effective_context_cost_ns,
                          std::size_t debt_at_admission,
                          std::size_t debt_at_completion) {
    if (!high_pressure || actual_batch == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    // The production >=8 path is deterministic and driven by visible queue
    // depth.  Feeding its 8/16/32 contexts into the legacy target2/target4
    // A/B controller would either ignore every sample or incorrectly trip the
    // old process-lifetime fuse.  Resource debt is still bounded elsewhere;
    // this controller has no execution-credit role.
    if (stage2_bulk_packing_enabled(configured_batch_limit_)) return;
    // A measured negative larger-batch cohort is a process-lifetime fuse.
    // Continuing to collect feedback cannot change the selected target and
    // would merely recreate the repeated trial tax this fuse removes.
    if (larger_batch_trials_disabled_) return;
    if (sampled_target != target_batch_) {
      // This context crossed an adaptation boundary. Its work is correct but
      // mixing it into either policy window would bias the rollback decision.
      return;
    }
    window_.tasks += actual_batch;
    ++window_.contexts;
    window_.cost_ns += effective_context_cost_ns;
    const std::int64_t debt_delta = debt_at_completion >= debt_at_admission
      ? static_cast<std::int64_t>(std::min<std::size_t>(
          debt_at_completion - debt_at_admission,
          static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())))
      : -static_cast<std::int64_t>(std::min<std::size_t>(
          debt_at_admission - debt_at_completion,
          static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())));
    window_.debt_delta += debt_delta;
    if (window_.tasks < kStage2PackingEvaluationTasks) return;

    const double cost_per_task =
      static_cast<double>(window_.cost_ns) /
      static_cast<double>(window_.tasks);
    const double debt_per_task =
      static_cast<double>(window_.debt_delta) /
      static_cast<double>(window_.tasks);
    const double actual_per_context =
      static_cast<double>(window_.tasks) /
      static_cast<double>(window_.contexts);

    if (cooldown_tasks_ != 0) {
      cooldown_tasks_ = cooldown_tasks_ > window_.tasks
        ? cooldown_tasks_ - window_.tasks : 0;
      window_ = {};
      return;
    }

    if (target_batch_ <= 2) {
      baseline_cost_ns_per_task_ = cost_per_task;
      baseline_debt_delta_per_task_ = debt_per_task;
      // One complete high-pressure cohort is the entire scheduled probe.
      // Sparse/zero-wait deployments may remain at legacy, but must not keep
      // reporting an active probe indefinitely.
      const bool completed_periodic_probe = target2_probe_active_;
      target2_probe_active_ = false;
      // Do not manufacture a target-four trial when arrivals cannot even
      // populate target two, or when the operator explicitly requested zero
      // batching delay.
      if (!larger_batch_trials_disabled_ && legacy_wait_us_ != 0 &&
          configured_batch_limit_ >= 4 &&
          actual_per_context >= 1.5) {
        target_batch_ = 4;
        target4_revalidation_active_ = completed_periodic_probe;
        telemetry_.target_batch = target_batch_;
        ++telemetry_.promotions;
      } else {
        target4_revalidation_active_ = false;
      }
      window_ = {};
      return;
    }

    const bool cost_improved = baseline_cost_ns_per_task_ != 0.0 &&
      cost_per_task <= baseline_cost_ns_per_task_ * 0.97;
    const bool cost_strongly_improved = baseline_cost_ns_per_task_ != 0.0 &&
      cost_per_task <= baseline_cost_ns_per_task_ * 0.90;
    const bool debt_worsened =
      debt_per_task > baseline_debt_delta_per_task_ + 0.25;
    const bool fill_sufficient = actual_per_context >= 3.5;
    if (!fill_sufficient || !cost_improved ||
        (debt_worsened && !cost_strongly_improved)) {
      target_batch_ = std::min<std::size_t>(2, configured_batch_limit_);
      target2_probe_active_ = false;
      target4_revalidation_active_ = false;
      accepted_windows_since_probe_ = 0;
      probe_interval_windows_ = kStage2PackingInitialProbeIntervalWindows;
      // The production trace showed every target-four promotion rolling back
      // without one accepted cohort. A negative measured window is therefore
      // authoritative for this process: do not periodically retry target four
      // (or any future target above two) and repeatedly pay the same tax.
      larger_batch_trials_disabled_ = true;
      telemetry_.target_batch = target_batch_;
      ++telemetry_.rollbacks;
      if (consecutive_trial_failures_ !=
          std::numeric_limits<std::uint32_t>::max()) {
        ++consecutive_trial_failures_;
      }
      cooldown_tasks_ = 0;
      baseline_cost_ns_per_task_ = 0.0;
      baseline_debt_delta_per_task_ = 0.0;
    } else {
      // Revalidate target four on every window and periodically refresh the
      // legacy cohort below. This catches both an absolute target-four
      // slowdown and an environment change that benefits legacy even more.
      ++telemetry_.accepted_trial_windows;
      consecutive_trial_failures_ = 0;
      cooldown_tasks_ = 0;
      if (target4_revalidation_active_) {
        probe_interval_windows_ = std::min<std::uint64_t>(
          kStage2PackingMaxProbeIntervalWindows,
          probe_interval_windows_ * 2);
        target4_revalidation_active_ = false;
      }
      ++accepted_windows_since_probe_;
      if (accepted_windows_since_probe_ >=
          probe_interval_windows_) {
        // Re-measure legacy instead of comparing target four forever with a
        // stale baseline. This scheduled probe is neither a rollback nor a
        // failed trial and therefore does not enter exponential cooldown.
        target_batch_ = std::min<std::size_t>(
          2, configured_batch_limit_);
        target2_probe_active_ = true;
        accepted_windows_since_probe_ = 0;
        telemetry_.target_batch = target_batch_;
      }
    }
    window_ = {};
  }

  [[nodiscard]] Stage2PackingTelemetry telemetry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stage2PackingTelemetry result = telemetry_;
    result.estimated_arrival_interval_us = arrival_interval_us_;
    return result;
  }

private:
  struct Window {
    std::uint64_t tasks{};
    std::uint64_t contexts{};
    std::uint64_t cost_ns{};
    std::int64_t debt_delta{};
  };

  mutable std::mutex mutex_;
  std::size_t configured_batch_limit_{1};
  std::uint32_t legacy_wait_us_{};
  std::size_t target_batch_{1};
  std::chrono::steady_clock::time_point last_arrival_{};
  std::uint64_t arrival_interval_us_{};
  double baseline_cost_ns_per_task_{};
  double baseline_debt_delta_per_task_{};
  std::uint64_t cooldown_tasks_{};
  std::uint32_t consecutive_trial_failures_{};
  std::uint64_t accepted_windows_since_probe_{};
  // Intentionally not cleared by reset(): reset may be used to reconfigure a
  // live controller, while a process-lifetime fuse must survive reconfigure.
  // A fresh process constructs a fresh controller with this value false.
  bool larger_batch_trials_disabled_{};
  bool target2_probe_active_{};
  bool target4_revalidation_active_{};
  std::uint64_t probe_interval_windows_{
    kStage2PackingInitialProbeIntervalWindows};
  Window window_{};
  Stage2PackingTelemetry telemetry_{};
};

// Compatibility helpers retained for focused tests and latency-sensitive
// deployments. Production admission uses decide_stage2_packing above.
inline constexpr bool stage2_maintenance_event_pending(
    std::uint64_t observed_epoch, std::uint64_t current_epoch) {
  return current_epoch != observed_epoch;
}

inline std::optional<std::chrono::steady_clock::time_point>
stage2_partial_batch_deadline(
    std::size_t queued_tasks,
    std::size_t batch_limit,
    std::chrono::steady_clock::time_point oldest_queued_at,
    std::uint32_t max_wait_us) {
  const std::size_t effective_batch_limit =
    std::max<std::size_t>(1, batch_limit);
  if (queued_tasks == 0 || queued_tasks >= effective_batch_limit ||
      max_wait_us == 0) {
    return std::nullopt;
  }
  return oldest_queued_at + std::chrono::microseconds(max_wait_us);
}

inline bool stage2_batch_ready(
    std::size_t queued_tasks,
    std::size_t batch_limit,
    std::chrono::steady_clock::time_point oldest_queued_at,
    std::chrono::steady_clock::time_point now,
    std::uint32_t max_wait_us) {
  if (queued_tasks == 0) return false;
  const std::size_t effective_batch_limit =
    std::max<std::size_t>(1, batch_limit);
  if (queued_tasks >= effective_batch_limit || max_wait_us == 0) return true;
  return now >= oldest_queued_at + std::chrono::microseconds(max_wait_us);
}

}  // namespace memory_node_storage_owner_maintenance_detail
