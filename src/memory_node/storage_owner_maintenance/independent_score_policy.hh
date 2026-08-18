#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace memory_node_storage_owner_maintenance_detail {

// Each arm is large enough to average over context scheduling jitter while
// still completing during the normal update warm-up.
inline constexpr std::uint64_t kIndependentScoreEvaluationTasks = 512;
// Once accepted, one fresh no-spec control is inserted per 32 ordinary spec
// windows. A steady cycle is 32 ordinary spec windows, one no-spec control,
// and one spec revalidation, so the no-spec control tax is 1/34 (2.94%) and
// the comparison never relies only on a cold process-start baseline.
inline constexpr std::uint64_t kIndependentScoreRevalidationIntervalWindows =
  32;

enum class IndependentScoreMode : std::uint8_t {
  baseline,
  trial,
  confirmation_drain,
  confirmation,
  enabled,
  revalidation_control_drain,
  revalidation_control,
  revalidation_trial,
  disabled,
};

struct IndependentScoreSample {
  IndependentScoreMode mode{IndependentScoreMode::disabled};
  std::uint64_t generation{};
  std::uint64_t registration_id{};
  bool eligible{};

  [[nodiscard]] constexpr bool allows_speculation() const {
    return eligible &&
      (mode == IndependentScoreMode::trial ||
       mode == IndependentScoreMode::enabled ||
       mode == IndependentScoreMode::revalidation_trial);
  }
};

struct IndependentScoreTelemetry {
  IndependentScoreMode mode{IndependentScoreMode::baseline};
  std::uint64_t generation{1};
  std::uint64_t window_tasks{};
  std::uint64_t window_posted_rpcs{};
  std::uint64_t window_useful{};
  std::uint64_t drain_outstanding{};
  std::uint64_t trials_started{};
  std::uint64_t trials_accepted{};
  std::uint64_t confirmations{};
  std::uint64_t enabled_windows{};
  std::uint64_t revalidation_controls{};
  std::uint64_t revalidations_accepted{};
  std::uint64_t rollbacks{};
  double baseline_cost_ns_per_task{};
  double baseline_debt_delta_per_task{};
};

// Process-wide contemporaneous control:
//
//   no-spec baseline -> spec trial -> no-spec confirmation
//
// The trial must beat both surrounding controls by at least 5%, must not
// increase completion debt relative to either, and must contain both posted
// and context-locally useful work. Mode+generation are sampled at admission,
// so late contexts crossing any boundary are deliberately excluded. Accepted
// operation periodically runs a fresh no-spec -> spec pair; a single failed
// initial or revalidation cohort permanently opens the process-lifetime fuse.
class IndependentScoreController {
public:
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    window_ = {};
    leading_control_ = {};
    candidate_trial_ = {};
    latest_control_ = {};
    enabled_windows_since_control_ = 0;
    draining_generation_ = 0;
    active_registrations_.clear();
    active_registrations_.reserve(256);
    ++generation_;
    if (generation_ == 0) generation_ = 1;
    mode_ = permanently_disabled_
      ? IndependentScoreMode::disabled
      : IndependentScoreMode::baseline;
    telemetry_ = {};
    telemetry_.mode = mode_;
    telemetry_.generation = generation_;
    telemetry_.rollbacks = permanently_disabled_ ? 1 : 0;
  }

  [[nodiscard]] IndependentScoreSample sample(bool eligible) {
    std::lock_guard<std::mutex> lock(mutex_);
    finish_drain_if_ready();
    IndependentScoreSample result{
      .mode = mode_,
      .generation = generation_,
      .eligible = eligible && mode_ != IndependentScoreMode::disabled,
    };
    // Drain/washout contexts execute the no-spec correctness path but are not
    // part of either control window. They therefore own no feedback token.
    if (mode_ == IndependentScoreMode::confirmation_drain ||
        mode_ == IndependentScoreMode::revalidation_control_drain) {
      result.eligible = false;
      return result;
    }
    if (!result.eligible) return result;
    result.registration_id = allocate_registration_id();
    active_registrations_.emplace(
      result.registration_id, result.generation);
    return result;
  }

  void observe_completion(IndependentScoreSample sample,
                          std::size_t task_count,
                          std::uint64_t effective_context_cost_ns,
                          std::size_t debt_at_admission,
                          std::size_t debt_at_completion,
                          std::size_t posted_rpcs,
                          std::size_t useful) {
    if (sample.registration_id == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto registration = active_registrations_.find(
      sample.registration_id);
    if (registration == active_registrations_.end()) {
      // Exactly-once feedback: a duplicate completion cannot accidentally
      // release another old context and terminate washout early.
      return;
    }
    const std::uint64_t registered_generation = registration->second;
    active_registrations_.erase(registration);
    if (task_count == 0 || sample.generation != registered_generation ||
        sample.generation != generation_ || sample.mode != mode_ ||
        mode_ == IndependentScoreMode::disabled) {
      finish_drain_if_ready();
      return;
    }

    window_.tasks = saturating_add(window_.tasks, task_count);
    window_.cost_ns = saturating_add(
      window_.cost_ns, effective_context_cost_ns);
    window_.posted_rpcs = saturating_add(
      window_.posted_rpcs, posted_rpcs);
    window_.useful = saturating_add(window_.useful, useful);
    window_.debt_delta = saturating_add_signed(
      window_.debt_delta,
      completion_debt_delta(debt_at_admission, debt_at_completion));
    if (window_.tasks < kIndependentScoreEvaluationTasks) {
      publish_window();
      return;
    }

    const Metrics metrics = summarize(window_);
    switch (mode_) {
      case IndependentScoreMode::baseline:
        leading_control_ = metrics;
        latest_control_ = metrics;
        ++telemetry_.trials_started;
        transition_to(IndependentScoreMode::trial);
        break;

      case IndependentScoreMode::trial:
        candidate_trial_ = metrics;
        // An arm that does not even beat its leading control, or that posted
        // only wasted work, cannot be rescued by a later confirmation.
        if (!valid_spec_against(candidate_trial_, leading_control_)) {
          disable_permanently();
        } else {
          enter_drain(
            IndependentScoreMode::confirmation_drain,
            sample.generation);
        }
        break;

      case IndependentScoreMode::confirmation_drain:
        break;

      case IndependentScoreMode::confirmation:
        ++telemetry_.confirmations;
        latest_control_ = metrics;
        if (!valid_spec_against(candidate_trial_, leading_control_) ||
            !valid_spec_against(candidate_trial_, latest_control_)) {
          disable_permanently();
        } else {
          ++telemetry_.trials_accepted;
          enabled_windows_since_control_ = 0;
          transition_to(IndependentScoreMode::enabled);
        }
        break;

      case IndependentScoreMode::enabled:
        // A complete 512-task spec cohort with no real post or no locally
        // consumed exact score has no mechanism for saving a dependency wave.
        // Disable instead of paying cache scans until the next scheduled arm.
        // Between scheduled probes, retain an absolute fuse against the most
        // recent no-spec control. Uniform environment slowdown may close this
        // conservatively; allowing a known-above-control speculative path to
        // run for another 32 windows would be a much larger negative tax.
        if (!valid_absolute_against(metrics, latest_control_)) {
          disable_permanently();
          break;
        }
        ++telemetry_.enabled_windows;
        ++enabled_windows_since_control_;
        if (enabled_windows_since_control_ >=
            kIndependentScoreRevalidationIntervalWindows) {
          enter_drain(
            IndependentScoreMode::revalidation_control_drain,
            sample.generation);
        } else {
          window_ = {};
          publish_window();
        }
        break;

      case IndependentScoreMode::revalidation_control_drain:
        break;

      case IndependentScoreMode::revalidation_control:
        latest_control_ = metrics;
        ++telemetry_.revalidation_controls;
        transition_to(IndependentScoreMode::revalidation_trial);
        break;

      case IndependentScoreMode::revalidation_trial:
        if (!valid_spec_against(metrics, latest_control_)) {
          disable_permanently();
        } else {
          ++telemetry_.revalidations_accepted;
          enabled_windows_since_control_ = 0;
          transition_to(IndependentScoreMode::enabled);
        }
        break;

      case IndependentScoreMode::disabled:
        break;
    }
    finish_drain_if_ready();
  }

  [[nodiscard]] IndependentScoreTelemetry telemetry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    IndependentScoreTelemetry result = telemetry_;
    result.mode = mode_;
    result.generation = generation_;
    result.window_tasks = window_.tasks;
    result.window_posted_rpcs = window_.posted_rpcs;
    result.window_useful = window_.useful;
    result.drain_outstanding = draining_generation_ == 0 ? 0 :
      outstanding_for_generation(draining_generation_);
    result.baseline_cost_ns_per_task = latest_control_.cost_ns_per_task;
    result.baseline_debt_delta_per_task =
      latest_control_.debt_delta_per_task;
    return result;
  }

private:
  struct Window {
    std::uint64_t tasks{};
    std::uint64_t cost_ns{};
    std::uint64_t posted_rpcs{};
    std::uint64_t useful{};
    std::int64_t debt_delta{};
  };

  struct Metrics {
    double cost_ns_per_task{};
    double debt_delta_per_task{};
    std::uint64_t posted_rpcs{};
    std::uint64_t useful{};

    [[nodiscard]] bool has_effective_speculation() const {
      return posted_rpcs != 0 && useful != 0;
    }
  };

  static Metrics summarize(const Window& window) {
    return Metrics{
      .cost_ns_per_task = static_cast<double>(window.cost_ns) /
        static_cast<double>(window.tasks),
      .debt_delta_per_task = static_cast<double>(window.debt_delta) /
        static_cast<double>(window.tasks),
      .posted_rpcs = window.posted_rpcs,
      .useful = window.useful,
    };
  }

  static bool valid_spec_against(const Metrics& spec,
                                 const Metrics& control) {
    return spec.has_effective_speculation() &&
      control.cost_ns_per_task > 0.0 &&
      spec.cost_ns_per_task <= control.cost_ns_per_task * 0.95 &&
      spec.debt_delta_per_task <= control.debt_delta_per_task;
  }

  static bool valid_absolute_against(const Metrics& spec,
                                     const Metrics& control) {
    return spec.has_effective_speculation() &&
      control.cost_ns_per_task > 0.0 &&
      spec.cost_ns_per_task <= control.cost_ns_per_task &&
      spec.debt_delta_per_task <= control.debt_delta_per_task;
  }

  template <typename T, typename U>
  static constexpr T saturating_add(T value, U increment) {
    const T bounded_increment = increment >
        static_cast<U>(std::numeric_limits<T>::max())
      ? std::numeric_limits<T>::max()
      : static_cast<T>(increment);
    return value > std::numeric_limits<T>::max() - bounded_increment
      ? std::numeric_limits<T>::max()
      : value + bounded_increment;
  }

  static constexpr std::int64_t completion_debt_delta(
      std::size_t before, std::size_t after) {
    constexpr std::size_t max_delta = static_cast<std::size_t>(
      std::numeric_limits<std::int64_t>::max());
    if (after >= before) {
      return static_cast<std::int64_t>(std::min(after - before, max_delta));
    }
    return -static_cast<std::int64_t>(std::min(before - after, max_delta));
  }

  static constexpr std::int64_t saturating_add_signed(
      std::int64_t value, std::int64_t increment) {
    if (increment > 0 &&
        value > std::numeric_limits<std::int64_t>::max() - increment) {
      return std::numeric_limits<std::int64_t>::max();
    }
    if (increment < 0 &&
        value < std::numeric_limits<std::int64_t>::min() - increment) {
      return std::numeric_limits<std::int64_t>::min();
    }
    return value + increment;
  }

  void transition_to(IndependentScoreMode next) {
    mode_ = next;
    ++generation_;
    if (generation_ == 0) generation_ = 1;
    window_ = {};
    publish_window();
  }

  void enter_drain(IndependentScoreMode drain_mode,
                   std::uint64_t completed_spec_generation) {
    draining_generation_ = completed_spec_generation;
    transition_to(drain_mode);
  }

  void finish_drain_if_ready() {
    if (draining_generation_ == 0 ||
        outstanding_for_generation(draining_generation_) != 0) {
      return;
    }
    const IndependentScoreMode completed_drain = mode_;
    draining_generation_ = 0;
    if (completed_drain == IndependentScoreMode::confirmation_drain) {
      transition_to(IndependentScoreMode::confirmation);
    } else if (completed_drain ==
               IndependentScoreMode::revalidation_control_drain) {
      transition_to(IndependentScoreMode::revalidation_control);
    }
  }

  [[nodiscard]] std::uint64_t outstanding_for_generation(
      std::uint64_t generation) const {
    return static_cast<std::uint64_t>(std::count_if(
      active_registrations_.begin(), active_registrations_.end(),
      [generation](const auto& registration) {
        return registration.second == generation;
      }));
  }

  std::uint64_t allocate_registration_id() {
    do {
      ++next_registration_id_;
      if (next_registration_id_ == 0) ++next_registration_id_;
    } while (active_registrations_.contains(next_registration_id_));
    return next_registration_id_;
  }

  void disable_permanently() {
    permanently_disabled_ = true;
    ++telemetry_.rollbacks;
    transition_to(IndependentScoreMode::disabled);
  }

  void publish_window() {
    telemetry_.mode = mode_;
    telemetry_.generation = generation_;
    telemetry_.window_tasks = window_.tasks;
    telemetry_.window_posted_rpcs = window_.posted_rpcs;
    telemetry_.window_useful = window_.useful;
    telemetry_.baseline_cost_ns_per_task =
      latest_control_.cost_ns_per_task;
    telemetry_.baseline_debt_delta_per_task =
      latest_control_.debt_delta_per_task;
  }

  mutable std::mutex mutex_;
  IndependentScoreMode mode_{IndependentScoreMode::baseline};
  std::uint64_t generation_{1};
  bool permanently_disabled_{};
  std::uint64_t next_registration_id_{};
  std::uint64_t draining_generation_{};
  std::uint64_t enabled_windows_since_control_{};
  std::unordered_map<std::uint64_t, std::uint64_t>
    active_registrations_;
  Metrics leading_control_{};
  Metrics candidate_trial_{};
  Metrics latest_control_{};
  Window window_{};
  IndependentScoreTelemetry telemetry_{};
};

inline constexpr const char* independent_score_mode_name(
    IndependentScoreMode mode) {
  switch (mode) {
    case IndependentScoreMode::baseline: return "baseline";
    case IndependentScoreMode::trial: return "trial";
    case IndependentScoreMode::confirmation_drain:
      return "confirmation_drain";
    case IndependentScoreMode::confirmation: return "confirmation";
    case IndependentScoreMode::enabled: return "enabled";
    case IndependentScoreMode::revalidation_control_drain:
      return "revalidation_control_drain";
    case IndependentScoreMode::revalidation_control:
      return "revalidation_control";
    case IndependentScoreMode::revalidation_trial:
      return "revalidation_trial";
    case IndependentScoreMode::disabled: return "disabled";
  }
  return "unknown";
}

}  // namespace memory_node_storage_owner_maintenance_detail
