#include <cassert>
#include <cstdint>
#include <limits>

#include "memory_node/peer_rdma_credit_policy.hh"

using memory_node_detail::derive_peer_rdma_read_credit_plan;
using memory_node_detail::select_peer_data_qp;

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

}  // namespace

int main() {
  test_per_qp_configuration_aggregates_over_data_qps();
  test_control_qp_is_not_counted_as_data_when_split();
  test_async_read_tickets_stripe_every_data_qp();
  test_hardware_and_wqe_limits_cap_each_qp();
  test_shared_cq_caps_all_peer_read_credits();
  test_large_inputs_do_not_overflow_aggregate_limits();
  return 0;
}
