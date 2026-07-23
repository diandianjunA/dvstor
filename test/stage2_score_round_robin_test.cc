#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "memory_node/peer_rdma_credit_policy.hh"
#include "memory_node/storage_owner_maintenance/search_io_state.hh"

using memory_node_storage_owner_maintenance_detail::
  Stage2ScoreRoundRobinCursor;
using memory_node_storage_owner_maintenance_detail::Stage2SearchIoPhase;
using memory_node_storage_owner_maintenance_detail::Stage2SearchIoState;
using memory_node_storage_owner_maintenance_detail::
  stage2_consumer_fits_physical_scratch;

namespace {

void test_one_dispatch_never_wraps_onto_the_same_request() {
  Stage2ScoreRoundRobinCursor cursor;
  cursor.begin_dispatch();
  assert(cursor.take(3) == std::optional<std::size_t>{0});
  assert(cursor.take(3) == std::optional<std::size_t>{1});
  assert(cursor.take(3) == std::optional<std::size_t>{2});
  assert(!cursor.take(3).has_value());

  cursor.begin_dispatch();
  assert(cursor.take(3) == std::optional<std::size_t>{0});
}

void test_retryable_front_does_not_starve_tail_after_swap_erase() {
  // Candidate zero models a snapshot that remains retryable. Every other
  // selected candidate resolves and is swap-erased exactly like
  // PartitionContinuationBatch::resolve_score_request(). With a reset-to-zero
  // collector the observed sequence would be 0 forever. The persistent cursor
  // reaches every tail candidate despite the changing vector layout.
  std::vector<int> pending{0, 1, 2, 3};
  std::vector<int> observed;
  Stage2ScoreRoundRobinCursor cursor;
  for (int dispatch = 0; dispatch < 5; ++dispatch) {
    cursor.begin_dispatch();
    const auto position = cursor.take(pending.size());
    assert(position.has_value());
    const int candidate = pending[*position];
    observed.push_back(candidate);
    if (candidate != 0) {
      pending[*position] = pending.back();
      pending.pop_back();
    }
  }
  assert((observed == std::vector<int>{0, 1, 2, 0, 3}));
  assert((pending == std::vector<int>{0}));
}

void test_cursor_normalizes_after_generation_size_change() {
  Stage2ScoreRoundRobinCursor cursor;
  cursor.next_position = 7;
  cursor.begin_dispatch();
  assert(cursor.take(2) == std::optional<std::size_t>{1});
  assert(cursor.take(2) == std::optional<std::size_t>{0});
  assert(!cursor.take(2).has_value());

  // Per-search cursors are independent; a retry in one logical search cannot
  // alter where another search resumes.
  Stage2ScoreRoundRobinCursor other;
  other.begin_dispatch();
  assert(other.take(4) == std::optional<std::size_t>{0});
  cursor.begin_dispatch();
  assert(cursor.take(4) == std::optional<std::size_t>{1});
  other.begin_dispatch();
  assert(other.take(4) == std::optional<std::size_t>{1});
}

void test_full_peer_is_skipped_without_hiding_other_peers_or_local_work() {
  // One logical search exposes a hot-peer request first, followed by another
  // hot-peer request, a local item, and a cold-peer request.  The collector
  // must keep examining after the second item is rejected; otherwise the hot
  // peer serializes every other destination behind its per-peer quota.
  constexpr std::uint32_t kLocal = 99;
  const std::array<std::uint32_t, 4> peers{0, 0, kLocal, 1};
  std::array<std::uint32_t, 2> used{};
  memory_node_detail::PeerRdmaReadDispatchQuota quota;
  quota.reset({.global_items = 2, .per_peer_items = 1}, used);

  Stage2ScoreRoundRobinCursor cursor;
  cursor.begin_dispatch();
  std::vector<std::size_t> selected;
  for (;;) {
    const auto position = cursor.take(peers.size());
    if (!position.has_value()) break;
    const bool remote = peers[*position] != kLocal;
    if (!quota.try_accept(peers[*position], remote)) continue;
    selected.push_back(*position);
  }
  assert((selected == std::vector<std::size_t>{0, 2, 3}));
  assert(quota.global_used == 2);
  assert(used[0] == 1 && used[1] == 1);

  // Rejection advances only this finite collector cursor.  It neither erases
  // the logical request nor permits the pass to wrap and select one request
  // twice; the next dispatch can revisit every still-unresolved request.
  cursor.begin_dispatch();
  std::array<bool, 4> observed{};
  for (;;) {
    const auto position = cursor.take(peers.size());
    if (!position.has_value()) break;
    assert(!observed[*position]);
    observed[*position] = true;
  }
  assert(std::all_of(observed.begin(), observed.end(), [](bool value) {
    return value;
  }));
}

void test_scratch_capacity_counts_physical_reads_not_consumers() {
  constexpr std::size_t kCapacity = 2;
  assert(stage2_consumer_fits_physical_scratch(true, 0, kCapacity));
  assert(stage2_consumer_fits_physical_scratch(true, 1, kCapacity));
  assert(!stage2_consumer_fits_physical_scratch(true, 2, kCapacity));

  // Once the physical wave is full, a local/terminal consumer or another
  // consumer of an already-selected remote pointer still needs no scratch.
  assert(stage2_consumer_fits_physical_scratch(false, 2, kCapacity));

  // A zero-capacity lane cannot admit a physical READ, but the same no-scratch
  // logical work remains legal. Production rejects zero-capacity lanes before
  // entering the dispatcher; keeping the predicate total makes its contract
  // explicit and independently testable.
  assert(!stage2_consumer_fits_physical_scratch(true, 0, 0));
  assert(stage2_consumer_fits_physical_scratch(false, 0, 0));
}

void test_home_rpc_wait_does_not_pin_registered_rdma_scratch() {
  Stage2SearchIoState state;
  assert(state.scratch_rebindable());

  state.initialized = true;
  state.phase = Stage2SearchIoPhase::graph_home_pending;
  assert(state.scratch_rebindable());

  // A live one-sided record always pins the lane, even if phase metadata were
  // corrupted to say that the context is waiting on a home RPC.
  state.pending_graph.push_back({});
  assert(!state.scratch_rebindable());
  state.pending_graph.clear();
  state.pending_vectors.push_back({});
  assert(!state.scratch_rebindable());

  state.pending_vectors.clear();
  state.phase = Stage2SearchIoPhase::score_body_pending;
  assert(!state.scratch_rebindable());
}

}  // namespace

int main() {
  test_one_dispatch_never_wraps_onto_the_same_request();
  test_retryable_front_does_not_starve_tail_after_swap_erase();
  test_cursor_normalizes_after_generation_size_change();
  test_full_peer_is_skipped_without_hiding_other_peers_or_local_work();
  test_scratch_capacity_counts_physical_reads_not_consumers();
  test_home_rpc_wait_does_not_pin_registered_rdma_scratch();
  return 0;
}
