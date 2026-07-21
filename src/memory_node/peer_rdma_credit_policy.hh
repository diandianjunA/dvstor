#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace memory_node_detail {

struct PeerRdmaReadCreditPlan {
  std::uint32_t data_qps_per_peer{};
  std::uint32_t per_qp{};
  std::uint32_t per_peer{};
  std::uint32_t global{};
  std::uint32_t shared_cq_read_budget{};
};

// QP0 carries peer RPC traffic whenever there is more than one QP.  A
// single-QP deployment necessarily shares QP0 between control and data.
constexpr std::uint32_t peer_data_qps_per_peer(
    const std::uint32_t qps_per_peer) {
  return qps_per_peer <= 1 ? 1 : qps_per_peer - 1;
}

// QP0 is the control lane when dedicated data QPs exist.  A monotonically
// increasing ticket stripes one producer across every data QP, so a process
// with fewer maintenance workers than data QPs can still consume the credit
// plan derived below.  Unsigned wraparound preserves the bounded mapping.
constexpr std::uint32_t select_peer_data_qp(
    const std::uint32_t qps_per_peer,
    const std::uint32_t ticket) {
  return qps_per_peer <= 1
    ? 0
    : 1 + ticket % (qps_per_peer - 1);
}

// storage_owner_peer_rdma_tokens is a per-data-QP knob.  Derive all transport
// limits in one place so a multi-QP peer gets the intended aggregate credit,
// while no individual QP exceeds its requester-read-atomic or WQE capacity
// and all peers together fit in the shared send CQ budget.
constexpr PeerRdmaReadCreditPlan derive_peer_rdma_read_credit_plan(
    const std::uint32_t requested_per_data_qp,
    const std::uint32_t qps_per_peer,
    const std::uint32_t remote_peer_count,
    const std::uint32_t max_qp_rd_atomic,
    const std::uint32_t max_qp_send_wr,
    const std::uint32_t shared_send_cq_entries,
    const std::uint32_t reserved_non_read_cq_entries) {
  const std::uint32_t data_qps = peer_data_qps_per_peer(qps_per_peer);
  const std::uint32_t peers = std::max<std::uint32_t>(1, remote_peer_count);
  const std::uint32_t requested =
    std::max<std::uint32_t>(1, requested_per_data_qp);
  const std::uint32_t rd_atomic =
    std::max<std::uint32_t>(1, max_qp_rd_atomic);

  // With only QP0, retain one send WQE for forward progress of peer control
  // traffic.  Dedicated data QPs do not need that per-QP reservation.
  const std::uint32_t send_wr = std::max<std::uint32_t>(1, max_qp_send_wr);
  const std::uint32_t qp_control_reserve =
    qps_per_peer <= 1 && send_wr > 1 ? 1 : 0;
  const std::uint32_t qp_read_wqe_budget = send_wr - qp_control_reserve;
  const std::uint32_t per_qp = std::max<std::uint32_t>(
    1, std::min({requested, rd_atomic, qp_read_wqe_budget}));

  const std::uint32_t cq_entries =
    std::max<std::uint32_t>(1, shared_send_cq_entries);
  const std::uint32_t cq_reserve = std::min<std::uint32_t>(
    reserved_non_read_cq_entries, cq_entries - 1);
  const std::uint32_t cq_read_budget = cq_entries - cq_reserve;
  const std::uint32_t fair_peer_cq_budget = cq_read_budget / peers;

  const std::uint64_t aggregate_per_peer =
    static_cast<std::uint64_t>(per_qp) * data_qps;
  const std::uint32_t per_peer = static_cast<std::uint32_t>(
    std::min<std::uint64_t>(aggregate_per_peer, fair_peer_cq_budget));
  const std::uint64_t aggregate_global =
    static_cast<std::uint64_t>(per_peer) * peers;
  const std::uint32_t global = static_cast<std::uint32_t>(
    std::min<std::uint64_t>(aggregate_global, cq_read_budget));

  return PeerRdmaReadCreditPlan{
    .data_qps_per_peer = data_qps,
    .per_qp = per_qp,
    .per_peer = per_peer,
    .global = global,
    .shared_cq_read_budget = cq_read_budget,
  };
}

static_assert(peer_data_qps_per_peer(1) == 1);
static_assert(peer_data_qps_per_peer(4) == 3);
static_assert(select_peer_data_qp(1, 99) == 0);
static_assert(select_peer_data_qp(4, 0) == 1);
static_assert(select_peer_data_qp(4, 1) == 2);
static_assert(select_peer_data_qp(4, 2) == 3);
static_assert(select_peer_data_qp(4, 3) == 1);

}  // namespace memory_node_detail
