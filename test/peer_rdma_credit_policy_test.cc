#include <algorithm>
#include <array>
#include <cassert>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <thread>

#include "memory_node/peer_rdma_credit_policy.hh"

using memory_node_detail::derive_peer_rdma_read_credit_plan;
using memory_node_detail::make_peer_wr_id;
using memory_node_detail::next_collision_free_peer_wr_id;
using memory_node_detail::peer_rdma_read_batch_completion_count;
using memory_node_detail::peer_rdma_read_batch_group_limit;
using memory_node_detail::peer_rdma_read_pair_chain_item;
using memory_node_detail::peer_rdma_read_pair_group_limit;
using memory_node_detail::peer_rdma_read_pair_wave_limit;
using memory_node_detail::peer_rdma_read_pair_work_request_count;
using memory_node_detail::PeerRdmaReadCreditRequest;
using memory_node_detail::release_peer_rdma_read_group;
using memory_node_detail::select_peer_data_qp;
using memory_node_detail::select_peer_data_qp_for_wave_chain;
using memory_node_detail::try_reserve_peer_rdma_read_group;
using memory_node_detail::try_reserve_peer_rdma_read_wave;

namespace {

void test_wr_id_wrap_skips_live_pending_and_completed_ids() {
  constexpr std::uint32_t owner = 0xfffffff0u;
  std::uint32_t next = std::numeric_limits<std::uint32_t>::max();
  std::unordered_set<std::uint64_t> unavailable{
    make_peer_wr_id(owner, std::numeric_limits<std::uint32_t>::max()),
    make_peer_wr_id(owner, 0),
    // A fixed peer-RPC ID in a different owner namespace must not interfere
    // with the generated transport namespace.
    make_peer_wr_id(3, std::numeric_limits<std::uint32_t>::max()),
  };
  const auto id = next_collision_free_peer_wr_id(
    owner, next, [&](const std::uint64_t candidate) {
      return unavailable.contains(candidate);
    });
  assert(id.has_value());
  assert(*id == make_peer_wr_id(owner, 1));
  assert(next == 2);

  // Reserving the chosen value closes the gap before it is inserted into the
  // pending-send map; a concurrent allocation probes to the next sequence.
  unavailable.insert(*id);
  const auto concurrent = next_collision_free_peer_wr_id(
    owner, next, [&](const std::uint64_t candidate) {
      return unavailable.contains(candidate);
    });
  assert(concurrent.has_value());
  assert(*concurrent == make_peer_wr_id(owner, 2));
}

void test_per_qp_configuration_aggregates_over_data_qps() {
  const auto plan = derive_peer_rdma_read_credit_plan(
    8, 4, 4, 16, 4096, 4096, 80);
  assert(plan.data_qps_per_peer == 3);
  assert(plan.per_qp == 8);
  assert(plan.per_peer == 24);
  assert(plan.global == 96);
}

void test_control_qp_is_not_counted_as_data_when_split() {
  const auto split = derive_peer_rdma_read_credit_plan(
    8, 2, 4, 16, 4096, 4096, 32);
  assert(split.data_qps_per_peer == 1);
  assert(split.per_peer == 8);

  const auto shared = derive_peer_rdma_read_credit_plan(
    8, 1, 4, 16, 9, 4096, 32);
  assert(shared.data_qps_per_peer == 1);
  assert(shared.per_qp == 8);
  assert(shared.per_peer == 8);
}

void test_async_read_tickets_stripe_every_data_qp() {
  assert(select_peer_data_qp(1, 0) == 0);
  assert(select_peer_data_qp(1, 17) == 0);
  assert(select_peer_data_qp(4, 0) == 1);
  assert(select_peer_data_qp(4, 1) == 2);
  assert(select_peer_data_qp(4, 2) == 3);
  assert(select_peer_data_qp(4, 3) == 1);
  assert(select_peer_data_qp(4, std::numeric_limits<std::uint32_t>::max())
         >= 1);
}

void test_each_shard_uses_an_independent_wave_qp_sequence() {
  constexpr std::uint32_t qps_per_peer = 4;
  constexpr std::size_t shard_count = 3;
  const std::array<std::uint32_t, shard_count> starts{7, 8, 9};
  std::array<std::uint32_t, shard_count> ordinals{};
  std::array<std::array<bool, qps_per_peer>, shard_count> seen{};

  // A global chain ticket would let the other shards perturb shard 0's
  // sequence in this schedule.  Independent per-shard ordinals must visit all
  // three data QPs exactly once for every shard before any QP is reused.
  constexpr std::array<std::size_t, 9> schedule{
    0, 1, 2, 1, 0, 2, 2, 0, 1};
  for (const std::size_t shard : schedule) {
    const std::uint32_t qp = select_peer_data_qp_for_wave_chain(
      qps_per_peer, starts[shard], ordinals[shard]++);
    assert(qp > 0 && qp < qps_per_peer);
    assert(!seen[shard][qp]);
    seen[shard][qp] = true;
  }
  for (std::size_t shard = 0; shard < shard_count; ++shard) {
    assert(ordinals[shard] == qps_per_peer - 1);
    for (std::uint32_t qp = 1; qp < qps_per_peer; ++qp) {
      assert(seen[shard][qp]);
    }
  }
}

void test_hardware_and_wqe_limits_cap_each_qp() {
  const auto rd_atomic_limited = derive_peer_rdma_read_credit_plan(
    32, 4, 2, 4, 4096, 4096, 16);
  assert(rd_atomic_limited.per_qp == 4);
  assert(rd_atomic_limited.per_peer == 12);

  const auto wqe_limited = derive_peer_rdma_read_credit_plan(
    32, 4, 2, 16, 3, 4096, 16);
  assert(wqe_limited.per_qp == 3);
  assert(wqe_limited.per_peer == 9);

  const auto shared_qp_wqe_limited = derive_peer_rdma_read_credit_plan(
    32, 1, 1, 16, 8, 4096, 4);
  assert(shared_qp_wqe_limited.per_qp == 7);
}

void test_shared_cq_caps_all_peer_read_credits() {
  const auto plan = derive_peer_rdma_read_credit_plan(
    8, 4, 4, 16, 4096, 64, 16);
  assert(plan.shared_cq_read_budget == 48);
  assert(plan.per_qp == 8);
  assert(plan.per_peer == 12);
  assert(plan.global == 48);
  assert(static_cast<std::uint64_t>(plan.per_peer) * 4 <=
         plan.shared_cq_read_budget);
}

void test_large_inputs_do_not_overflow_aggregate_limits() {
  const auto plan = derive_peer_rdma_read_credit_plan(
    std::numeric_limits<std::uint32_t>::max(),
    std::numeric_limits<std::uint32_t>::max(),
    std::numeric_limits<std::uint32_t>::max(),
    std::numeric_limits<std::uint32_t>::max(),
    std::numeric_limits<std::uint32_t>::max(),
    4096, 0);
  assert(plan.per_qp == std::numeric_limits<std::uint32_t>::max());
  assert(plan.per_peer == 0);
  assert(plan.global == 0);
}

void test_linked_read_batches_stay_inside_every_credit_domain() {
  const auto qp_limited = derive_peer_rdma_read_credit_plan(
    8, 4, 4, 16, 4096, 4096, 80);
  assert(peer_rdma_read_batch_group_limit(qp_limited) == 8);
  assert(peer_rdma_read_batch_completion_count(0, qp_limited) == 0);
  assert(peer_rdma_read_batch_completion_count(8, qp_limited) == 1);
  assert(peer_rdma_read_batch_completion_count(17, qp_limited) == 3);

  const auto cq_limited = derive_peer_rdma_read_credit_plan(
    32, 4, 4, 32, 4096, 20, 16);
  assert(cq_limited.per_qp == 32);
  assert(cq_limited.per_peer == 1);
  assert(cq_limited.global == 4);
  assert(peer_rdma_read_batch_group_limit(cq_limited) == 1);
  assert(peer_rdma_read_batch_completion_count(17, cq_limited) == 17);

  const memory_node_detail::PeerRdmaReadCreditPlan invalid{};
  assert(peer_rdma_read_batch_group_limit(invalid) == 0);
  assert(peer_rdma_read_batch_completion_count(17, invalid) == 0);

  // 64 balanced reads over four shards become two <=8-WR chains per shard,
  // hence eight successful CQEs instead of 64.
  assert(4 * peer_rdma_read_batch_completion_count(16, qp_limited) == 8);
}

void test_ordered_snapshot_pairs_never_split_a_credit_chain() {
  const memory_node_detail::PeerRdmaReadCreditPlan plan{
    .data_qps_per_peer = 3,
    .per_qp = 8,
    .per_peer = 16,
    .global = 32,
    .shared_cq_read_budget = 32,
  };
  assert(peer_rdma_read_pair_group_limit(plan) == 4);
  assert(peer_rdma_read_pair_work_request_count(4) == 8);
  for (std::uint32_t pair = 0; pair < 4; ++pair) {
    const auto full = peer_rdma_read_pair_chain_item(pair, 4);
    const auto after = peer_rdma_read_pair_chain_item(4 + pair, 4);
    assert(full.pair_index == pair);
    assert(!full.after_header);
    assert(after.pair_index == pair);
    assert(after.after_header);
  }
  std::atomic<std::uint32_t> peer{0};
  std::atomic<std::uint32_t> qp{0};
  std::atomic<std::uint32_t> global{0};
  const std::uint32_t pair_wr_count =
    peer_rdma_read_pair_work_request_count(4);
  assert(try_reserve_peer_rdma_read_group(
    peer, qp, global, plan, pair_wr_count));
  assert(peer.load() == 8);
  assert(qp.load() == 8);
  assert(global.load() == 8);

  // A one-credit transport cannot atomically preserve the pair. Production
  // must use its two-wave fallback instead of weakening validation.
  const memory_node_detail::PeerRdmaReadCreditPlan single{
    .data_qps_per_peer = 1,
    .per_qp = 1,
    .per_peer = 1,
    .global = 1,
    .shared_cq_read_budget = 1,
  };
  assert(peer_rdma_read_pair_group_limit(single) == 0);
}

void test_ordered_pair_wave_rounds_down_each_qp_before_aggregation() {
  const memory_node_detail::PeerRdmaReadCreditPlan odd_qp_limit{
    .data_qps_per_peer = 3,
    .per_qp = 7,
    .per_peer = 21,
    .global = 42,
    .shared_cq_read_budget = 42,
  };
  assert(peer_rdma_read_pair_group_limit(odd_qp_limit) == 3);
  // Every QP can carry only three complete pairs (six READs).  Ten pairs
  // would need a fourth chain, reuse one of the three QPs, and exceed 7 READs
  // on that QP even though floor(per_peer/2) is 10.
  assert(peer_rdma_read_pair_wave_limit(odd_qp_limit) == 9);

  const memory_node_detail::PeerRdmaReadCreditPlan global_limited{
    .data_qps_per_peer = 8,
    .per_qp = 8,
    .per_peer = 64,
    .global = 10,
    .shared_cq_read_budget = 10,
  };
  assert(peer_rdma_read_pair_wave_limit(global_limited) == 5);
}

void test_group_reservation_rolls_back_every_partial_domain() {
  const memory_node_detail::PeerRdmaReadCreditPlan plan{
    .data_qps_per_peer = 1,
    .per_qp = 8,
    .per_peer = 8,
    .global = 8,
    .shared_cq_read_budget = 8,
  };
  std::atomic<std::uint32_t> peer{0};
  std::atomic<std::uint32_t> qp{0};
  std::atomic<std::uint32_t> global{4};

  // The global domain cannot fit this chain. Earlier peer/QP reservations
  // must be gone when the call returns, so another producer can progress.
  assert(!try_reserve_peer_rdma_read_group(
    peer, qp, global, plan, 8));
  assert(peer.load() == 0);
  assert(qp.load() == 0);
  assert(global.load() == 4);

  global.store(0);
  assert(try_reserve_peer_rdma_read_group(
    peer, qp, global, plan, 8));
  assert(peer.load() == 8);
  assert(qp.load() == 8);
  assert(global.load() == 8);
}

void test_sync_and_async_reads_share_the_global_window() {
  const memory_node_detail::PeerRdmaReadCreditPlan plan{
    .data_qps_per_peer = 2,
    .per_qp = 8,
    .per_peer = 8,
    .global = 8,
    .shared_cq_read_budget = 8,
  };
  std::array<std::atomic<std::uint32_t>, 2> peers{};
  std::array<std::atomic<std::uint32_t>, 2> qps{};
  std::atomic<std::uint32_t> global{0};

  // Model a seven-WR resumable wave followed by one synchronous read. Both
  // use the same reservation primitive, so a second synchronous operation
  // must wait instead of exceeding the shared CQ/requester plan.
  assert(try_reserve_peer_rdma_read_group(
    peers[0], qps[0], global, plan, 7));
  assert(try_reserve_peer_rdma_read_group(
    peers[1], qps[1], global, plan, 1));
  assert(!try_reserve_peer_rdma_read_group(
    peers[1], qps[1], global, plan, 1));
  assert(global.load() == plan.global);
}

void test_concurrent_group_reservation_never_overcommits() {
  const memory_node_detail::PeerRdmaReadCreditPlan plan{
    .data_qps_per_peer = 2,
    .per_qp = 8,
    .per_peer = 8,
    .global = 8,
    .shared_cq_read_budget = 8,
  };
  std::atomic<std::uint32_t> peer{0};
  std::array<std::atomic<std::uint32_t>, 2> qps{};
  std::atomic<std::uint32_t> global{0};
  constexpr std::size_t kThreads = 16;
  std::barrier start(static_cast<std::ptrdiff_t>(kThreads + 1));
  std::array<bool, kThreads> reserved{};
  std::array<std::thread, kThreads> threads;
  for (std::size_t index = 0; index < kThreads; ++index) {
    threads[index] = std::thread([&, index]() {
      start.arrive_and_wait();
      reserved[index] = try_reserve_peer_rdma_read_group(
        peer, qps[index & 1], global, plan, 8);
    });
  }
  start.arrive_and_wait();
  for (std::thread& thread : threads) thread.join();

  const std::size_t winners = static_cast<std::size_t>(
    std::count(reserved.begin(), reserved.end(), true));
  assert(winners == 1);
  assert(peer.load() == 8);
  assert(qps[0].load() + qps[1].load() == 8);
  assert(global.load() == 8);
}

void test_wave_reservation_is_all_or_nothing() {
  const memory_node_detail::PeerRdmaReadCreditPlan plan{
    .data_qps_per_peer = 1,
    .per_qp = 8,
    .per_peer = 8,
    .global = 16,
    .shared_cq_read_budget = 16,
  };
  std::array<std::atomic<std::uint32_t>, 2> peers{};
  std::array<std::atomic<std::uint32_t>, 2> qps{};
  std::atomic<std::uint32_t> global{4};
  const std::array requests{
    PeerRdmaReadCreditRequest{&peers[0], &qps[0], 8},
    PeerRdmaReadCreditRequest{&peers[1], &qps[1], 8},
  };

  // The first chain fits, but the second does not fit the remaining global
  // window. The failed wave must return every credit acquired by its prefix.
  assert(!try_reserve_peer_rdma_read_wave(requests, global, plan));
  assert(peers[0].load() == 0 && peers[1].load() == 0);
  assert(qps[0].load() == 0 && qps[1].load() == 0);
  assert(global.load() == 4);

  global.store(0);
  assert(try_reserve_peer_rdma_read_wave(requests, global, plan));
  assert(peers[0].load() == 8 && peers[1].load() == 8);
  assert(qps[0].load() == 8 && qps[1].load() == 8);
  assert(global.load() == 16);
  release_peer_rdma_read_group(requests[0], global);
  release_peer_rdma_read_group(requests[1], global);
  assert(peers[0].load() == 0 && peers[1].load() == 0);
  assert(qps[0].load() == 0 && qps[1].load() == 0);
  assert(global.load() == 0);
}

}  // namespace

int main() {
  test_wr_id_wrap_skips_live_pending_and_completed_ids();
  test_per_qp_configuration_aggregates_over_data_qps();
  test_control_qp_is_not_counted_as_data_when_split();
  test_async_read_tickets_stripe_every_data_qp();
  test_each_shard_uses_an_independent_wave_qp_sequence();
  test_hardware_and_wqe_limits_cap_each_qp();
  test_shared_cq_caps_all_peer_read_credits();
  test_large_inputs_do_not_overflow_aggregate_limits();
  test_linked_read_batches_stay_inside_every_credit_domain();
  test_ordered_snapshot_pairs_never_split_a_credit_chain();
  test_ordered_pair_wave_rounds_down_each_qp_before_aggregation();
  test_group_reservation_rolls_back_every_partial_domain();
  test_sync_and_async_reads_share_the_global_window();
  test_concurrent_group_reservation_never_overcommits();
  test_wave_reservation_is_all_or_nothing();
  return 0;
}
