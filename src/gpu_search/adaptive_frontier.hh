#pragma once

#include <cstdint>

#include "gpu_search/types.hh"

namespace gpu_search::adaptive_frontier {

// The persistent query workspace has one fixed, warp-addressable frontier
// capacity.  Keeping the limit here lets configuration, host construction,
// and device control share one contract without duplicating a tuning knob.
inline constexpr u32 kFrontierCapacity = 32;

struct Feedback {
  u32 promoted{};
  u32 retained{};
  u32 stale{};
  u32 queue_rejects{};
  u32 critical_misses{};
  // Queue admission and exact certificate coverage are the latency/pressure
  // signals. A tail that is still in flight cannot be refilled and cannot
  // cover the next Commit wave. Do not compare a nonblocking poll's execution
  // time with merge duration: that is not issue-to-CQ latency and previously
  // caused false contractions.
  u32 tail_admitted{};
  // Width growth is granted only when the exact post-merge Stable-Run
  // certificate finds every handle in the next Commit prefix already backed
  // by a validated speculative ROB entry. Promotion/retention counts remain
  // secondary evidence for how much to grow; they cannot turn a partially
  // covered or still-inflight wave into a positive sample.
  u32 commit_waves_observed{};
  u32 commit_waves_covered{};
};

struct ControllerState {
  u32 commit_width{};
  u32 max_issue_width{};
  u32 current_issue_width{};
  // Number of complete queries observed since a controller collapsed to the
  // mandatory Commit width.  This state lives in one persistent query CTA:
  // it is neither exported to global memory nor shared between CTAs.
  u32 collapsed_queries{};
};

#if defined(__CUDACC__)
#define DVSTOR_ADAPTIVE_FRONTIER_HD __host__ __device__ __forceinline__
#else
#define DVSTOR_ADAPTIVE_FRONTIER_HD inline
#endif

DVSTOR_ADAPTIVE_FRONTIER_HD constexpr u32 clamp_width(
    u32 value, u32 lower, u32 upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

// Configuration value used when --gpu-graph-issue-width is left at zero.
// This is a hardware/workspace bound, not a dataset-specific tuning value.
DVSTOR_ADAPTIVE_FRONTIER_HD constexpr u32 automatic_max_issue_width(
    u32 traversal_beam_width) {
  return traversal_beam_width < kFrontierCapacity
    ? traversal_beam_width : kFrontierCapacity;
}

DVSTOR_ADAPTIVE_FRONTIER_HD constexpr ControllerState normalize(
    ControllerState state) {
  state.commit_width =
    clamp_width(state.commit_width, 1u, kFrontierCapacity);
  state.max_issue_width =
    clamp_width(state.max_issue_width, state.commit_width,
                kFrontierCapacity);
  state.current_issue_width =
    clamp_width(state.current_issue_width, state.commit_width,
                state.max_issue_width);
  return state;
}

// A collapsed controller periodically takes one single-slot online sample so
// that a workload phase change remains observable.  The interval is derived
// from the runtime Commit width: a wider mandatory frontier already supplies
// more RDMA demand and therefore needs proportionally fewer shadow probes per
// query.  No dataset or machine-specific interval is introduced.
DVSTOR_ADAPTIVE_FRONTIER_HD constexpr u32 collapsed_reprobe_period(
    u32 commit_width) {
  return clamp_width(commit_width, 1u, kFrontierCapacity);
}

// Start with one bounded online probe when a shadow frontier is available.
// This supplies query-local evidence without a dataset-specific width. After
// that first slot, growth/shrink decisions use only observed tail utility;
// an unprofitable frontier therefore falls back to commit width and is
// reopened only by the low-frequency query-boundary probe below.
DVSTOR_ADAPTIVE_FRONTIER_HD constexpr u32 initial_issue_width(
    u32 commit_width, u32 max_issue_width) {
  ControllerState bounds = normalize(
    ControllerState{commit_width, max_issue_width, commit_width});
  return bounds.commit_width < bounds.max_issue_width
    ? bounds.commit_width + 1u : bounds.commit_width;
}

DVSTOR_ADAPTIVE_FRONTIER_HD constexpr ControllerState make_controller_state(
    u32 commit_width, u32 max_issue_width) {
  ControllerState state = normalize(
    ControllerState{commit_width, max_issue_width, commit_width});
  state.current_issue_width =
    initial_issue_width(state.commit_width, state.max_issue_width);
  return state;
}

// Advance the CTA-local controller exactly once at the boundary of a valid
// query.  A profitable/non-collapsed controller carries its learned width
// directly into the next query.  An unprofitable controller stays at Commit
// width and reopens only one shadow slot every Commit-width queries.  This
// avoids making every short query pay the same known-useless bootstrap read,
// while retaining online recovery when frontier stability changes.
DVSTOR_ADAPTIVE_FRONTIER_HD constexpr ControllerState query_begin_state(
    ControllerState state) {
  state = normalize(state);
  if (state.max_issue_width == state.commit_width) {
    state.collapsed_queries = 0;
    return state;
  }
  if (state.current_issue_width != state.commit_width) {
    state.collapsed_queries = 0;
    return state;
  }

  const u32 period = collapsed_reprobe_period(state.commit_width);
  if (state.collapsed_queries >= period - 1u) {
    state.current_issue_width = state.commit_width + 1u;
    state.collapsed_queries = 0;
  } else {
    ++state.collapsed_queries;
  }
  return normalize(state);
}

DVSTOR_ADAPTIVE_FRONTIER_HD constexpr void begin_query(
    ControllerState& state) {
  state = query_begin_state(state);
}

DVSTOR_ADAPTIVE_FRONTIER_HD constexpr u64 ceil_ratio(
    u64 numerator, u64 denominator) {
  if (numerator == 0 || denominator == 0) return 0;
  return 1u + (numerator - 1u) / denominator;
}

// Marginal-utility feedback for the shadow frontier. Before the shadow is
// large enough to represent one complete Commit wave, positive node utility
// performs a bounded multiplicative bootstrap probe. Once that semantic
// capacity is reached, only an exact whole-prefix certificate can authorize
// further growth. This avoids the circular requirement that a one-slot probe
// must already cover a multi-slot Commit wave, while keeping full-wave
// readiness as the primary steady-state benefit signal. Shrink is
// proportional to measured waste or communication pressure. The law is
// dimensionless and derives every decision from query-local observations.
//
// A critical miss is demand evidence only when the same wave produced useful
// evidence; demand alone is ambiguous under a rapidly turning frontier.
DVSTOR_ADAPTIVE_FRONTIER_HD constexpr ControllerState adapt_issue_width(
    ControllerState state, const Feedback& feedback) {
  state = normalize(state);

  const u64 useful =
    static_cast<u64>(feedback.promoted) + feedback.retained;
  const u64 promotion_utility = feedback.promoted;
  const u64 waste = feedback.stale;
  // A critical miss is demand for the mandatory core, not evidence that a
  // shadow request was useful.  Adding it to `positive` double-counts the
  // same frontier transition and can make a wave with >50% stale records
  // appear profitable.  Promotion/retention versus stale/rejected is the
  // direct query-local utility signal; RDMA pressure is already represented
  // by queue rejects and the overlap gate below.
  const u64 evidence = useful + waste;
  const u32 shadow = state.current_issue_width - state.commit_width;
  const u64 observed_waves = feedback.commit_waves_observed;
  const u64 covered_waves =
    feedback.commit_waves_covered < observed_waves
      ? feedback.commit_waves_covered : observed_waves;
  const u64 uncovered_waves = observed_waves - covered_waves;
  const bool whole_wave_ready =
    observed_waves != 0 && uncovered_waves == 0;
  const u32 max_shadow = state.max_issue_width - state.commit_width;
  const u32 bootstrap_shadow =
    state.commit_width < max_shadow ? state.commit_width : max_shadow;
  const bool bootstrap_probe = shadow < bootstrap_shadow;

  // Pressure signals may coincide. Compute every proportional contraction and
  // apply the strongest once: one rejected request must not hide a much
  // larger stale frontier observed in the same certificate epoch.
  u32 contraction = 0;

  // Queue rejection is explicit QP/WQE pressure. It is kept separate from
  // stale payload evidence because a rejected suffix consumed no network
  // bytes and is corrected out of physical graph-read telemetry.
  if (shadow != 0 && feedback.queue_rejects != 0) {
    const u64 attempted =
      static_cast<u64>(feedback.tail_admitted) + feedback.queue_rejects;
    const u32 decrease = static_cast<u32>(ceil_ratio(
      static_cast<u64>(shadow) * feedback.queue_rejects,
      attempted == 0 ? feedback.queue_rejects : attempted));
    const u32 bounded_decrease =
      decrease == 0 ? 1u : (decrease < shadow ? decrease : shadow);
    contraction =
      bounded_decrease > contraction ? bounded_decrease : contraction;
  }

  if (shadow != 0 && waste > useful) {
    const u32 decrease = static_cast<u32>(ceil_ratio(
      static_cast<u64>(shadow) * waste, evidence));
    const u32 bounded_decrease =
      decrease == 0 ? 1u : (decrease < shadow ? decrease : shadow);
    contraction =
      bounded_decrease > contraction ? bounded_decrease : contraction;
  }
  if (contraction != 0) {
    state.current_issue_width -= contraction;
    return normalize(state);
  }

  if (evidence == 0) {
    // Whole-wave readiness is the primary benefit signal. In the absence of
    // node-level evidence, advance only by the minimum representable probe;
    // promotion/retention may accelerate later growth but is not allowed to
    // manufacture coverage.
    if (whole_wave_ready) ++state.current_issue_width;
    return normalize(state);
  }
  if (useful == waste) return state;

  if (useful > waste) {
    // Node-level utility can bootstrap enough slots to make a whole-wave
    // observation physically possible. Beyond that point it is secondary:
    // the exact Commit prefix must have been ready at certification.
    if (!bootstrap_probe && !whole_wave_ready) return state;
    // Retention says that a prefetched record remained in the predicted
    // frontier, but it did not eliminate a critical read.  It can preserve a
    // width and strengthen an exact whole-wave sample; by itself it must not
    // bootstrap a collapsed/partial shadow.  Otherwise a long-lived but
    // consistently late tail grows even when its promotion ratio is zero.
    if (bootstrap_probe && promotion_utility <= waste &&
        !whole_wave_ready) {
      return state;
    }
    // Scale the next probe by the already validated shadow, not by the whole
    // remaining capacity. This is a bounded multiplicative increase: a stable
    // frontier reaches useful RDMA depth within the finite expansion budget,
    // while a single one-slot sample cannot consume all QP/WQE capacity.
    const u64 growth_useful =
      whole_wave_ready ? useful : promotion_utility;
    const u64 net_useful = growth_useful > waste
      ? growth_useful - waste : 0;
    if (net_useful == 0) return state;
    const u32 growth = static_cast<u32>(ceil_ratio(
      static_cast<u64>(shadow == 0 ? 1u : shadow) * net_useful,
      evidence));
    state.current_issue_width += growth == 0 ? 1u : growth;
  }
  return normalize(state);
}

DVSTOR_ADAPTIVE_FRONTIER_HD constexpr void update_issue_width(
    ControllerState& state, const Feedback& feedback) {
  state = adapt_issue_width(state, feedback);
}

#undef DVSTOR_ADAPTIVE_FRONTIER_HD

}  // namespace gpu_search::adaptive_frontier
