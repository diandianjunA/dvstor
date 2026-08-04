#include <cassert>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "common/configuration.hh"
#include "gpu_search/adaptive_frontier.hh"

namespace {

namespace policy = gpu_search::adaptive_frontier;

configuration::IndexConfiguration make_config(
    u32 legacy_prefetch, u32 commit_width, u32 issue_width,
    u32 traversal_beam_width) {
  std::vector<std::string> arguments{
    "adaptive_frontier_policy_test",
    "--servers", "127.0.0.1:1234",
    "--storage-peers", "127.0.0.1:1234",
    "--index-prefix", "/tmp/index",
    "--threads", "1",
    "--dim", "128",
    "--k", "10",
    "--gpu-graph-prefetch-depth", std::to_string(legacy_prefetch),
    "--gpu-graph-commit-width", std::to_string(commit_width),
    "--gpu-graph-issue-width", std::to_string(issue_width),
    "--gpu-traversal-beam-width", std::to_string(traversal_beam_width),
  };
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (auto& argument : arguments) argv.push_back(argument.data());
  return configuration::IndexConfiguration(
    static_cast<int>(argv.size()), argv.data());
}

void test_configuration_resolution() {
  const auto legacy = make_config(16, 0, 0, 128);
  assert(legacy.gpu_graph_prefetch_depth == 16);
  assert(legacy.gpu_graph_commit_width == 16);
  assert(legacy.gpu_graph_issue_width == 32);

  const auto explicit_widths = make_config(16, 8, 24, 128);
  assert(explicit_widths.gpu_graph_commit_width == 8);
  assert(explicit_widths.gpu_graph_issue_width == 24);

  const auto beam_limited = make_config(16, 8, 0, 12);
  assert(beam_limited.gpu_graph_commit_width == 8);
  assert(beam_limited.gpu_graph_issue_width == 12);

  // Old valid configurations remain valid even when their legacy prefetch
  // depth was wider than the traversal beam.
  const auto wide_legacy = make_config(24, 0, 0, 16);
  assert(wide_legacy.gpu_graph_commit_width == 24);
  assert(wide_legacy.gpu_graph_issue_width == 24);

  const auto defaults = make_config(32, 0, 0, 128);
  assert(defaults.gpu_graph_commit_width > 0);
  assert(defaults.gpu_graph_commit_width <= defaults.gpu_graph_issue_width);
  assert(defaults.gpu_graph_issue_width <= policy::kFrontierCapacity);
}

void test_capacity_and_initial_state() {
  static_assert(policy::automatic_max_issue_width(1) == 1);
  static_assert(policy::automatic_max_issue_width(31) == 31);
  static_assert(policy::automatic_max_issue_width(32) == 32);
  static_assert(policy::automatic_max_issue_width(128) == 32);

  static_assert(policy::initial_issue_width(8, 8) == 8);
  static_assert(policy::initial_issue_width(8, 32) == 9);
  static_assert(policy::initial_issue_width(16, 32) == 17);
  static_assert(policy::initial_issue_width(31, 32) == 32);

  constexpr auto ordinary = policy::make_controller_state(16, 32);
  static_assert(ordinary.commit_width == 16);
  static_assert(ordinary.max_issue_width == 32);
  static_assert(ordinary.current_issue_width == 17);
  static_assert(ordinary.collapsed_queries == 0);

  constexpr auto hostile = policy::make_controller_state(0, 100);
  static_assert(hostile.commit_width == 1);
  static_assert(hostile.max_issue_width == 32);
  static_assert(hostile.current_issue_width == 2);
  static_assert(hostile.collapsed_queries == 0);
}

void test_feedback_direction_and_proportionality() {
  const policy::ControllerState base{8, 32, 20};

  const auto no_evidence =
    policy::adapt_issue_width(base, policy::Feedback{});
  assert(no_evidence.current_issue_width == 20);

  const auto balanced = policy::adapt_issue_width(
    base, policy::Feedback{3, 0, 2, 1, 0});
  // A queue reject is pressure, even when node-level useful/waste counts
  // happen to balance.
  assert(balanced.current_issue_width == 8);

  const auto promoted = policy::adapt_issue_width(
    base, policy::Feedback{4, 0, 0, 0, 0});
  // This state already has at least one complete Commit wave of shadow
  // capacity. Per-node promotion is now secondary evidence and cannot grow
  // without the exact whole-prefix certificate.
  assert(promoted.current_issue_width == 20);

  const auto retained = policy::adapt_issue_width(
    base, policy::Feedback{0, 4, 0, 0, 0});
  assert(retained.current_issue_width == promoted.current_issue_width);

  // A record that merely survived the next certificate did not hide a
  // critical read. Retention alone therefore keeps a bootstrap probe alive
  // but cannot grow it.
  const policy::ControllerState bootstrap{16, 32, 17};
  const auto retained_bootstrap = policy::adapt_issue_width(
    bootstrap, policy::Feedback{0, 1, 0, 0, 0});
  assert(retained_bootstrap.current_issue_width == 17);

  // Actual promotion remains sufficient online evidence during bootstrap.
  const auto promoted_bootstrap = policy::adapt_issue_width(
    bootstrap, policy::Feedback{1, 0, 0, 0, 0});
  assert(promoted_bootstrap.current_issue_width == 18);

  const auto stale = policy::adapt_issue_width(
    base, policy::Feedback{0, 0, 4, 0, 0});
  assert(stale.current_issue_width == 8);

  const auto rejected = policy::adapt_issue_width(
    base, policy::Feedback{0, 0, 0, 4, 0});
  assert(rejected.current_issue_width == stale.current_issue_width);

  const auto critical_miss = policy::adapt_issue_width(
    base, policy::Feedback{0, 0, 0, 0, 1});
  // Demand alone is ambiguous: on a high-turnover frontier every commit is
  // also a miss. Without useful prediction evidence it must not force growth.
  assert(critical_miss.current_issue_width == 20);

  const auto qualified_critical_miss = policy::adapt_issue_width(
    base, policy::Feedback{1, 0, 0, 0, 1});
  assert(qualified_critical_miss.current_issue_width == 20);

  const auto high_turnover = policy::adapt_issue_width(
    base, policy::Feedback{0, 0, 4, 0, 4});
  assert(high_turnover.current_issue_width == 8);

  // Net utility scales the already validated shadow.  With 3 useful and
  // 1 wasted slots, a 12-slot shadow grows by 12*(3-1)/4 = 6.
  policy::Feedback covered_partial_growth{};
  covered_partial_growth.promoted = 3;
  covered_partial_growth.stale = 1;
  covered_partial_growth.commit_waves_observed = 1;
  covered_partial_growth.commit_waves_covered = 1;
  const auto partial_growth = policy::adapt_issue_width(
    base, covered_partial_growth);
  assert(partial_growth.current_issue_width == 26);

  // A negative observation removes the observed wasted fraction immediately.
  const auto partial_shrink = policy::adapt_issue_width(
    base, policy::Feedback{1, 0, 3, 0, 0});
  assert(partial_shrink.current_issue_width == 11);

  policy::ControllerState updated = base;
  policy::update_issue_width(
    updated, policy::Feedback{1, 0, 3, 0, 0});
  assert(updated.current_issue_width == partial_shrink.current_issue_width);
}

void test_commit_wave_coverage_gate() {
  const policy::ControllerState base{8, 32, 20};

  // This is emitted only when the exact post-merge certificate finds the
  // complete Commit prefix in validated speculative ROB slots.
  policy::Feedback covered{};
  covered.promoted = 4;
  covered.tail_admitted = 4;
  covered.commit_waves_observed = 1;
  covered.commit_waves_covered = 1;
  const auto covered_growth =
    policy::adapt_issue_width(base, covered);
  assert(covered_growth.current_issue_width == 32);

  policy::Feedback covered_without_node_evidence = covered;
  covered_without_node_evidence.promoted = 0;
  const auto no_node_growth = policy::adapt_issue_width(
    base, covered_without_node_evidence);
  assert(no_node_growth.current_issue_width ==
         base.current_issue_width + 1);

  policy::Feedback uncovered = covered;
  uncovered.commit_waves_covered = 0;
  const auto uncovered_hold =
    policy::adapt_issue_width(base, uncovered);
  // Missing exact coverage does not manufacture a benefit, but it is not
  // itself queue pressure. The same reconciliation charges unmatched
  // validated records as stale, while late/inflight records feed the
  // exact coverage gate.
  assert(uncovered_hold.current_issue_width == base.current_issue_width);

  // Coverage is an all-prefix objective, but an accumulated feedback horizon
  // remains proportional when it contains more than one certificate.
  policy::Feedback partially_covered = covered;
  partially_covered.commit_waves_observed = 4;
  partially_covered.commit_waves_covered = 3;
  const auto partial_coverage_hold =
    policy::adapt_issue_width(base, partially_covered);
  assert(partial_coverage_hold.current_issue_width ==
         base.current_issue_width);

  // Bootstrap cannot require a full 16-entry certificate while only one
  // speculative slot exists. Positive exact-node utility probes
  // multiplicatively until the shadow can represent one complete Commit
  // wave; this bound is semantic, not dataset-specific.
  const policy::ControllerState bootstrap{16, 32, 17};
  policy::Feedback useful_but_uncovered{};
  useful_but_uncovered.promoted = 1;
  useful_but_uncovered.tail_admitted = 1;
  useful_but_uncovered.commit_waves_observed = 1;
  const auto bootstrap_growth =
    policy::adapt_issue_width(bootstrap, useful_but_uncovered);
  assert(bootstrap_growth.current_issue_width == 18);

  // Explicit QP/WQE rejection overrides a positive coverage sample.
  policy::Feedback rejected_covered = covered;
  rejected_covered.queue_rejects = 1;
  const auto pressure_shrink =
    policy::adapt_issue_width(base, rejected_covered);
  assert(pressure_shrink.current_issue_width == 17);
}

void test_combined_pressure_uses_strongest_contraction_once() {
  const policy::ControllerState base{8, 32, 20};
  policy::Feedback combined{};
  combined.promoted = 1;
  combined.stale = 10;
  combined.queue_rejects = 1;
  combined.tail_admitted = 10;

  // Rejection requests a two-slot contraction, while the observed stale
  // fraction requests ceil(12 * 10 / 11) = 11. The
  // controller applies the strongest query-local pressure exactly once,
  // instead of summing correlated signals and collapsing below the mandatory
  // Commit width.
  const auto next = policy::adapt_issue_width(base, combined);
  assert(next.current_issue_width == 9);
}

void test_finite_query_ramp_and_recovery() {
  auto state = policy::make_controller_state(16, 32);
  assert(state.current_issue_width == 17);

  // One exact retained/promoted sample per epoch doubles only the already
  // validated shadow: 1, 2, 4, 8, 16.  It reaches the hardware/workspace cap
  // within a finite ANN query without a hard-coded target width.
  const u32 expected[]{18, 20, 24, 32};
  for (u32 width : expected) {
    policy::Feedback covered{};
    covered.promoted = 1;
    covered.tail_admitted = 1;
    covered.commit_waves_observed = 1;
    covered.commit_waves_covered = 1;
    state = policy::adapt_issue_width(
      state, covered);
    assert(state.current_issue_width == width);
  }

  // A fully wasted observation removes the complete shadow immediately; the
  // mandatory commit width remains unchanged.
  state = policy::adapt_issue_width(
    state, policy::Feedback{0, 0, 16, 0, 0});
  assert(state.current_issue_width == state.commit_width);
}

void test_cta_local_collapsed_reprobe_cadence() {
  static_assert(policy::collapsed_reprobe_period(0) == 1);
  static_assert(policy::collapsed_reprobe_period(1) == 1);
  static_assert(policy::collapsed_reprobe_period(8) == 8);
  static_assert(policy::collapsed_reprobe_period(16) == 16);
  static_assert(policy::collapsed_reprobe_period(100) ==
                policy::kFrontierCapacity);

  policy::ControllerState state{8, 32, 8};
  // Seven complete queries retain the collapsed width. The eighth boundary
  // reopens exactly one tail slot, so a persistently unprofitable workload
  // samples once per runtime Commit-width queries rather than once per query.
  for (u32 query = 1; query < 8; ++query) {
    state = policy::query_begin_state(state);
    assert(state.current_issue_width == 8);
    assert(state.collapsed_queries == query);
  }
  state = policy::query_begin_state(state);
  assert(state.current_issue_width == 9);
  assert(state.collapsed_queries == 0);

  // A wasted re-probe collapses again; its next cadence starts from zero.
  state = policy::adapt_issue_width(
    state, policy::Feedback{0, 0, 1, 0, 0});
  assert(state.current_issue_width == 8);
  assert(state.collapsed_queries == 0);
  policy::begin_query(state);
  assert(state.current_issue_width == 8);
  assert(state.collapsed_queries == 1);

  // A fixed-width controller has no speculative state to probe.
  policy::ControllerState fixed{16, 16, 16, 123};
  fixed = policy::query_begin_state(fixed);
  assert(fixed.current_issue_width == 16);
  assert(fixed.collapsed_queries == 0);

  // Learned widths persist across query boundaries; begin_query only owns
  // collapsed recovery and never resets a profitable controller to commit+1.
  policy::ControllerState learned{8, 32, 24, 7};
  learned = policy::query_begin_state(learned);
  assert(learned.current_issue_width == 24);
  assert(learned.collapsed_queries == 0);
}

void test_boundaries_and_saturation() {
  const policy::Feedback patterns[]{
    {},
    {1, 0, 0, 0, 0},
    {0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0},
    {0, 0, 0, 1, 0},
    {0, 0, 0, 0, 1},
    {3, 2, 1, 1, 0},
    {1, 1, 3, 2, 1},
    {std::numeric_limits<u32>::max(),
     std::numeric_limits<u32>::max(),
     std::numeric_limits<u32>::max(),
     std::numeric_limits<u32>::max(),
     std::numeric_limits<u32>::max()},
  };

  for (u32 commit = 1; commit <= policy::kFrontierCapacity; ++commit) {
    for (u32 maximum = commit;
         maximum <= policy::kFrontierCapacity; ++maximum) {
      for (u32 current = 0;
           current <= policy::kFrontierCapacity + 1; ++current) {
        for (const auto& feedback : patterns) {
          const auto next = policy::adapt_issue_width(
            policy::ControllerState{commit, maximum, current}, feedback);
          assert(next.commit_width == commit);
          assert(next.max_issue_width == maximum);
          assert(next.current_issue_width >= commit);
          assert(next.current_issue_width <= maximum);
        }
      }
    }
  }

  const auto fixed = policy::adapt_issue_width(
    policy::ControllerState{32, 32, 32},
    policy::Feedback{0, 0, 100, 100, 0});
  assert(fixed.current_issue_width == 32);

  const u32 maximum = std::numeric_limits<u32>::max();
  const auto wide_counts = policy::adapt_issue_width(
    policy::ControllerState{8, 32, 20},
    policy::Feedback{maximum, maximum, maximum, maximum, maximum});
  assert(wide_counts.current_issue_width >= 8);
  assert(wide_counts.current_issue_width <= 32);
}

}  // namespace

int main() {
  test_configuration_resolution();
  test_capacity_and_initial_state();
  test_feedback_direction_and_proportionality();
  test_commit_wave_coverage_gate();
  test_combined_pressure_uses_strongest_contraction_once();
  test_finite_query_ramp_and_recovery();
  test_cta_local_collapsed_reprobe_cadence();
  test_boundaries_and_saturation();
  return 0;
}
