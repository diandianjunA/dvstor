#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>

namespace memory_node_storage_owner_maintenance_detail {

// Every admitted Stage2 context is one independently progressing B8 cohort.
// Larger configured wire batches are split into multiple contexts instead of
// creating 16/32-item barrier convoys.
inline constexpr std::size_t kStage2SemanticExecutionBatch = 8;
inline constexpr std::uint32_t kStage2TailMinimumPredictionUs = 1'000;
inline constexpr std::uint32_t kStage2TailMaximumPredictionUs = 25'000;

inline constexpr std::size_t stage2_execution_slice_limit(
    std::size_t selected_pop_limit,
    std::size_t configured_batch_limit) {
  return std::max<std::size_t>(1, std::min({
    std::max<std::size_t>(1, selected_pop_limit),
    std::max<std::size_t>(1, configured_batch_limit),
    kStage2SemanticExecutionBatch}));
}

inline constexpr std::size_t stage2_visible_backlog_target(
    std::size_t configured_batch_limit) {
  return std::min(
    kStage2SemanticExecutionBatch,
    std::max<std::size_t>(1, configured_batch_limit));
}

enum class Stage2PackingFlushReason : std::uint8_t {
  none,
  target,
  deadline,
  cleanup,
};

struct Stage2PackingParameters {
  std::uint32_t estimated_arrival_interval_us{};
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
  std::uint64_t cleanup_flushes{};
  std::uint64_t admitted_queue_depth_sum{};
  std::uint64_t admitted_queue_depth_max{};
  std::uint64_t batch_1_to_7{};
  std::uint64_t batch_8{};
  std::uint64_t assembly_batches{};
  std::uint64_t assembly_wait_ns{};
};

// Deterministic B8 packing. A complete cohort is runnable immediately. A
// sub-B8 tail predicts the time from its oldest descriptor to eight arrivals
// using the enqueue EWMA. The configured max wait is authoritative: zero
// flushes immediately, otherwise it caps the 1--25 ms arrival-aware budget.
inline Stage2PackingDecision decide_stage2_packing(
    std::size_t queued_tasks,
    std::size_t configured_batch_limit,
    std::chrono::steady_clock::time_point oldest_queued_at,
    std::chrono::steady_clock::time_point now,
    std::uint32_t configured_max_wait_us,
    std::uint32_t estimated_arrival_interval_us) {
  Stage2PackingDecision decision;
  const std::size_t target =
    stage2_visible_backlog_target(configured_batch_limit);
  decision.target_batch = static_cast<std::uint32_t>(target);
  decision.pop_limit = target;
  if (queued_tasks == 0) return decision;

  if (queued_tasks >= target) {
    decision.ready = true;
    decision.reason = Stage2PackingFlushReason::target;
    return decision;
  }

  if (configured_max_wait_us == 0) {
    decision.ready = true;
    decision.reason = Stage2PackingFlushReason::deadline;
    return decision;
  }

  std::uint64_t predicted_us = kStage2TailMaximumPredictionUs;
  if (estimated_arrival_interval_us != 0) {
    // The deadline is anchored at the oldest item, so predict the complete
    // cohort from its first arrival rather than subtracting elapsed time twice.
    predicted_us = static_cast<std::uint64_t>(
      estimated_arrival_interval_us) * (target - 1);
  }
  predicted_us = std::clamp<std::uint64_t>(
    predicted_us,
    kStage2TailMinimumPredictionUs,
    kStage2TailMaximumPredictionUs);
  decision.wait_budget_us = static_cast<std::uint32_t>(
    std::min<std::uint64_t>(predicted_us, configured_max_wait_us));
  decision.deadline = oldest_queued_at +
    std::chrono::microseconds(decision.wait_budget_us);
  if (now >= *decision.deadline) {
    decision.ready = true;
    decision.reason = Stage2PackingFlushReason::deadline;
  }
  return decision;
}

class Stage2PackingController {
 public:
  void reset(std::size_t configured_batch_limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    configured_batch_limit_ =
      std::max<std::size_t>(1, configured_batch_limit);
    last_arrival_ = {};
    arrival_interval_us_ = 0;
    telemetry_ = {};
    telemetry_.target_batch =
      stage2_visible_backlog_target(configured_batch_limit_);
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
      arrival_interval_us_ = arrival_interval_us_ == 0
        ? sample_us
        : (arrival_interval_us_ * 7 + sample_us) / 8;
      arrival_interval_us_ = std::min<std::uint64_t>(
        arrival_interval_us_, kStage2TailMaximumPredictionUs);
    }
    last_arrival_ = now;
  }

  [[nodiscard]] Stage2PackingParameters parameters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return Stage2PackingParameters{
      .estimated_arrival_interval_us = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(arrival_interval_us_,
                                std::numeric_limits<std::uint32_t>::max())),
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
      case Stage2PackingFlushReason::target:
        ++telemetry_.target_flushes;
        break;
      case Stage2PackingFlushReason::deadline:
        ++telemetry_.deadline_flushes;
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
      if (actual_batch < kStage2SemanticExecutionBatch) {
        ++telemetry_.batch_1_to_7;
      } else {
        ++telemetry_.batch_8;
      }
      if (oldest_wait_ns != 0) {
        ++telemetry_.assembly_batches;
        telemetry_.assembly_wait_ns += oldest_wait_ns;
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

  [[nodiscard]] Stage2PackingTelemetry telemetry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stage2PackingTelemetry result = telemetry_;
    result.estimated_arrival_interval_us = arrival_interval_us_;
    return result;
  }

 private:
  mutable std::mutex mutex_;
  std::size_t configured_batch_limit_{1};
  std::chrono::steady_clock::time_point last_arrival_{};
  std::uint64_t arrival_interval_us_{};
  Stage2PackingTelemetry telemetry_{};
};

}  // namespace memory_node_storage_owner_maintenance_detail
