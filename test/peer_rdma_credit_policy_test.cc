#include <algorithm>
#include <array>
#include <cassert>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <limits>
#include <thread>

#include "memory_node/peer_rdma_credit_policy.hh"

using memory_node_detail::derive_peer_rdma_read_credit_plan;
using memory_node_detail::peer_rdma_read_batch_completion_count;
using memory_node_detail::peer_rdma_read_batch_group_limit;
using memory_node_detail::select_peer_data_qp;
using memory_node_detail::try_reserve_peer_rdma_read_group;

namespace {

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

}  // namespace

int main() {
  test_per_qp_configuration_aggregates_over_data_qps();
  test_control_qp_is_not_counted_as_data_when_split();
  test_async_read_tickets_stripe_every_data_qp();
  test_hardware_and_wqe_limits_cap_each_qp();
  test_shared_cq_caps_all_peer_read_credits();
  test_large_inputs_do_not_overflow_aggregate_limits();
  test_linked_read_batches_stay_inside_every_credit_domain();
  test_group_reservation_rolls_back_every_partial_domain();
  test_concurrent_group_reservation_never_overcommits();
  return 0;
}
