#include <cassert>
#include <limits>
#include <stdexcept>

#include "gpu_search/persistent_grid_plan.hh"

int main() {
  using gpu_search::evaluate_persistent_grid_candidate;
  using gpu_search::plan_persistent_grid;
  using gpu_search::u32;

  // A100/A800-shaped resource result.  The CPU service thread count is
  // deliberately absent from the planner; all 256 bounded query slots may be
  // represented by resident CTAs.
  const auto high_concurrency = plan_persistent_grid(
    {3, 1}, 4, 108, 256, 160);
  assert(high_concurrency.candidates[0].owner_blocks == 40);
  assert(high_concurrency.candidates[0].grid_capacity == 324);
  assert(high_concurrency.candidates[0].query_blocks == 256);
  assert(high_concurrency.candidates[0].total_blocks == 298);
  assert(high_concurrency.candidates[0].resident_query_warps == 1024);
  assert(high_concurrency.candidates[1].owner_blocks == 20);
  assert(high_concurrency.candidates[1].query_blocks == 86);
  assert(high_concurrency.candidates[1].resident_query_warps == 688);
  assert(high_concurrency.selected.threads == 128);

  // When both variants can cover every slot, prefer wider CTAs for the
  // per-query parallel work instead of treating 128 as a universal tuning.
  const auto latency_oriented = plan_persistent_grid(
    {3, 1}, 4, 108, 64, 160);
  assert(latency_oriented.selected.threads == 256);
  assert(latency_oriented.selected.query_blocks == 64);

  // The configured limit is a ceiling on the complete unified grid.  Owner
  // and two control CTAs consume that capacity before query CTAs.
  const auto capped = plan_persistent_grid({3, 1}, 1, 108, 256, 160);
  assert(capped.candidates[0].effective_blocks_per_sm == 1);
  assert(capped.candidates[0].query_blocks == 66);
  assert(capped.selected.threads == 256);

  // Equal resident query warp counts use the deterministic 256-thread
  // tie-breaker (16*4 == 8*8).
  const auto tied = plan_persistent_grid({2, 1}, 2, 11, 16, 8);
  assert(tied.candidates[0].query_blocks == 16);
  assert(tied.candidates[1].query_blocks == 8);
  assert(tied.selected.threads == 256);

  const auto no_query_capacity = evaluate_persistent_grid_candidate(
    256, 1, 1, 22, 32, 160);
  assert(no_query_capacity.owner_blocks == 20);
  assert(no_query_capacity.grid_capacity == 22);
  assert(!no_query_capacity.viable());

  bool rejected = false;
  try {
    (void)plan_persistent_grid({1, 1}, 1, 22, 32, 160);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);

  bool overflow_rejected = false;
  try {
    (void)evaluate_persistent_grid_candidate(
      128, 2, 2, std::numeric_limits<u32>::max(), 1, 1);
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  assert(overflow_rejected);
}
