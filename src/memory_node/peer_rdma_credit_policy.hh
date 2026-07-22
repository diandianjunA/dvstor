#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>

namespace memory_node_detail {

constexpr std::uint64_t make_peer_wr_id(
    const std::uint32_t owner,
    const std::uint32_t sequence) {
  return (static_cast<std::uint64_t>(owner) << 32) | sequence;
}

// Work-request sequence numbers intentionally remain 32 bit because the
// upper half of wr_id is a transport-owner namespace.  A long-running node
// can wrap that sequence, however, so allocation must probe past every ID
// that is still pending, reserved by another producer, or waiting to be
// consumed by a synchronous caller.  `next_sequence` is protected by the
// caller's completion mutex; the predicate observes the same protected
// containers.  Returning nullopt means the complete 32-bit namespace is
// occupied, which is impossible for a correctly bounded verbs transport but
// is still made explicit instead of silently overwriting state.
template <typename IsUnavailable>
std::optional<std::uint64_t> next_collision_free_peer_wr_id(
    const std::uint32_t owner,
    std::uint32_t& next_sequence,
    IsUnavailable&& unavailable) {
  const std::uint32_t first = next_sequence;
  do {
    const std::uint32_t sequence = next_sequence++;
    const std::uint64_t candidate = make_peer_wr_id(owner, sequence);
    if (!unavailable(candidate)) return candidate;
  } while (next_sequence != first);
  return std::nullopt;
}

struct PeerRdmaReadCreditPlan {
  std::uint32_t data_qps_per_peer{};
  std::uint32_t per_qp{};
  std::uint32_t per_peer{};
  std::uint32_t global{};
  std::uint32_t shared_cq_read_budget{};
};

// A linked RC READ chain consumes one requester-read-atomic/WQE credit for
// every WR even though only its tail produces a successful CQE.  Keep one
// chain inside every independently enforced credit domain.  Larger waves are
// split into several chains; posting a completed chain before reserving the
// next one guarantees progress even for a single-QP deployment.
constexpr std::uint32_t peer_rdma_read_batch_group_limit(
    const PeerRdmaReadCreditPlan& plan) {
  return std::min({plan.per_qp, plan.per_peer, plan.global});
}

// A stable vector snapshot is one logical operation but two ordered RDMA
// READs: the full record prefix followed by an independent after-header.  A
// pair may never straddle QPs or linked chains, otherwise the after-header can
// overtake the body read and cease to be a seqlock validation.  Return the
// number of complete pairs that fit in every credit domain; zero tells the
// caller to retain the existing two-wave fallback on extremely constrained
// transports that cannot reserve two READ credits atomically.
constexpr std::uint32_t peer_rdma_read_pair_group_limit(
    const PeerRdmaReadCreditPlan& plan) {
  return peer_rdma_read_batch_group_limit(plan) / 2;
}

// Maximum number of ordered snapshot pairs that one all-or-nothing wave can
// distribute over every data QP of one peer.  Per-QP capacity must be rounded
// down before aggregation: with an odd per_qp limit, aggregating first and
// dividing later would invent one unusable READ credit per pair of QPs and can
// force the round-robin to reuse a QP beyond its actual limit.
constexpr std::uint32_t peer_rdma_read_pair_wave_limit(
    const PeerRdmaReadCreditPlan& plan) {
  const std::uint64_t aggregate_qp_pairs =
    static_cast<std::uint64_t>(plan.data_qps_per_peer) *
    (plan.per_qp / 2);
  return static_cast<std::uint32_t>(std::min<std::uint64_t>({
    plan.global / 2,
    plan.per_peer / 2,
    aggregate_qp_pairs,
  }));
}

struct PeerRdmaReadPairChainItem {
  std::uint32_t pair_index{};
  bool after_header{};
};

// A Stage2 transport dispatch may contain requests for several remote peers.
// `per_peer` is therefore a quota for each destination, not a cap on the
// complete dispatch.  Keep the aggregate and destination limits separate so
// a balanced wave can use the full shared-CQ window without allowing one hot
// peer (or one of its data QPs) to be overcommitted.
struct PeerRdmaReadDispatchLimits {
  std::uint32_t global_items{};
  std::uint32_t per_peer_items{};
};

constexpr PeerRdmaReadDispatchLimits peer_rdma_read_dispatch_limits(
    const PeerRdmaReadCreditPlan& plan) {
  const std::uint64_t aggregate_qp_items =
    static_cast<std::uint64_t>(plan.data_qps_per_peer) * plan.per_qp;
  return PeerRdmaReadDispatchLimits{
    .global_items = plan.global,
    .per_peer_items = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(plan.per_peer, aggregate_qp_items)),
  };
}

constexpr PeerRdmaReadDispatchLimits
peer_rdma_read_pair_dispatch_limits(
    const PeerRdmaReadCreditPlan& plan) {
  // Round down each QP before aggregating.  An odd final READ credit on one
  // QP cannot be combined with another QP's odd credit to form an ordered
  // body/after-header pair.
  const std::uint64_t aggregate_qp_pairs =
    static_cast<std::uint64_t>(plan.data_qps_per_peer) *
    (plan.per_qp / 2);
  return PeerRdmaReadDispatchLimits{
    .global_items = plan.global / 2,
    .per_peer_items = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(plan.per_peer / 2, aggregate_qp_pairs)),
  };
}

// Mutable accounting for one finite collector pass.  Local/terminal work and
// another consumer of an already selected physical pointer set
// `consumes_remote_item=false`; they remain admissible without stealing RDMA
// credit from a distinct remote read.  A rejected remote item owns no quota
// and remains pending in its logical search for a later dispatch.
struct PeerRdmaReadDispatchQuota {
  PeerRdmaReadDispatchLimits limits{};
  std::span<std::uint32_t> used_by_peer{};
  std::uint32_t global_used{};

  void reset(
      const PeerRdmaReadDispatchLimits new_limits,
      const std::span<std::uint32_t> new_used_by_peer) {
    limits = new_limits;
    used_by_peer = new_used_by_peer;
    std::fill(used_by_peer.begin(), used_by_peer.end(), 0);
    global_used = 0;
  }

  [[nodiscard]] bool try_accept(
      const std::uint32_t peer,
      const bool consumes_remote_item) {
    if (!consumes_remote_item) return true;
    if (peer >= used_by_peer.size() ||
        global_used >= limits.global_items ||
        used_by_peer[peer] >= limits.per_peer_items) {
      return false;
    }
    ++global_used;
    ++used_by_peer[peer];
    return true;
  }
};

// One mixed immutable-base/dynamic vector wave charges one READ for every
// body and one additional ordered after-header READ only for a recyclable
// dynamic record.  Total WR credit and pair packing are different limits:
// when per_qp is odd, the last credit of two QPs cannot be combined into one
// ordered pair. Keep both dimensions explicit so a caller never prepares an
// all-or-nothing wave that the transport can only partially place.
struct PeerRdmaSnapshotDispatchLimits {
  std::uint32_t global_wrs{};
  std::uint32_t per_peer_wrs{};
  std::uint32_t per_peer_pairs{};
};

constexpr PeerRdmaSnapshotDispatchLimits
peer_rdma_snapshot_dispatch_limits(const PeerRdmaReadCreditPlan& plan) {
  const std::uint64_t aggregate_qp_wrs =
    static_cast<std::uint64_t>(plan.data_qps_per_peer) * plan.per_qp;
  const std::uint64_t aggregate_qp_pairs =
    static_cast<std::uint64_t>(plan.data_qps_per_peer) *
    (plan.per_qp / 2);
  return PeerRdmaSnapshotDispatchLimits{
    .global_wrs = plan.global,
    .per_peer_wrs = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(plan.per_peer, aggregate_qp_wrs)),
    .per_peer_pairs = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(plan.per_peer / 2, aggregate_qp_pairs)),
  };
}

struct PeerRdmaSnapshotDispatchQuota {
  PeerRdmaSnapshotDispatchLimits limits{};
  std::span<std::uint32_t> wrs_by_peer{};
  std::span<std::uint32_t> pairs_by_peer{};
  std::uint32_t global_wrs{};

  void reset(
      const PeerRdmaSnapshotDispatchLimits new_limits,
      const std::span<std::uint32_t> new_wrs_by_peer,
      const std::span<std::uint32_t> new_pairs_by_peer) {
    assert(new_wrs_by_peer.size() == new_pairs_by_peer.size());
    limits = new_limits;
    wrs_by_peer = new_wrs_by_peer;
    pairs_by_peer = new_pairs_by_peer;
    std::fill(wrs_by_peer.begin(), wrs_by_peer.end(), 0);
    std::fill(pairs_by_peer.begin(), pairs_by_peer.end(), 0);
    global_wrs = 0;
  }

  // `wr_cost==0` represents local/terminal work or another consumer of an
  // already selected physical pointer. A distinct immutable base record costs
  // one body READ; a distinct dynamic record costs an indivisible two-READ
  // ordered pair.
  [[nodiscard]] bool try_accept(
      const std::uint32_t peer,
      const std::uint32_t wr_cost,
      const bool ordered_pair) {
    if (wr_cost == 0) return true;
    assert(wr_cost == (ordered_pair ? 2u : 1u));
    if (peer >= wrs_by_peer.size() ||
        wr_cost > limits.global_wrs -
          std::min(global_wrs, limits.global_wrs) ||
        wr_cost > limits.per_peer_wrs -
          std::min(wrs_by_peer[peer], limits.per_peer_wrs) ||
        (ordered_pair &&
         pairs_by_peer[peer] >= limits.per_peer_pairs)) {
      return false;
    }
    global_wrs += wr_cost;
    wrs_by_peer[peer] += wr_cost;
    pairs_by_peer[peer] += ordered_pair;
    return true;
  }
};

// Keep the collector's transport-mode choice and its admission decision in
// one object.  This prevents a new wave from initializing one quota using
// stale in-flight state and then admitting through the other quota.  In mixed
// mode an immutable base record costs one READ while a recyclable dynamic
// record reserves its body/after-header pair atomically.
struct PeerRdmaVectorSnapshotDispatchQuota {
  bool mixed_snapshots{};
  PeerRdmaSnapshotDispatchQuota mixed{};
  PeerRdmaReadDispatchQuota fallback{};

  void reset(
      const bool new_mixed_snapshots,
      const PeerRdmaSnapshotDispatchLimits mixed_limits,
      const PeerRdmaReadDispatchLimits fallback_limits,
      const std::span<std::uint32_t> wrs_by_peer,
      const std::span<std::uint32_t> pairs_by_peer) {
    mixed_snapshots = new_mixed_snapshots;
    if (mixed_snapshots) {
      mixed.reset(mixed_limits, wrs_by_peer, pairs_by_peer);
    } else {
      fallback.reset(fallback_limits, wrs_by_peer);
      std::fill(pairs_by_peer.begin(), pairs_by_peer.end(), 0);
    }
  }

  [[nodiscard]] bool try_accept(
      const std::uint32_t peer,
      const bool distinct_remote,
      const bool requires_after_header) {
    if (mixed_snapshots) {
      return mixed.try_accept(
        peer,
        distinct_remote ? (requires_after_header ? 2u : 1u) : 0u,
        distinct_remote && requires_after_header);
    }
    return fallback.try_accept(peer, distinct_remote);
  }
};

struct PeerRdmaSnapshotChainItem {
  std::uint32_t snapshot_index{};
  bool after_header{};
};

constexpr std::uint32_t peer_rdma_snapshot_work_request_count(
    const std::uint32_t snapshot_count,
    const std::uint32_t ordered_pair_count) {
  return ordered_pair_count <=
      std::numeric_limits<std::uint32_t>::max() - snapshot_count
    ? snapshot_count + ordered_pair_count
    : 0;
}

// A mixed chain is [all bodies, dynamic after-headers]. The first item in the
// second half carries IBV_SEND_FENCE in the transport implementation. The
// caller supplies the logical indices of pair-requiring records in the exact
// order used to build that second half.
constexpr PeerRdmaSnapshotChainItem peer_rdma_snapshot_chain_item(
    const std::uint32_t work_request_index,
    const std::uint32_t snapshot_count,
    const std::span<const std::uint32_t> ordered_pair_indices) {
  return work_request_index < snapshot_count
    ? PeerRdmaSnapshotChainItem{
        .snapshot_index = work_request_index,
        .after_header = false,
      }
    : PeerRdmaSnapshotChainItem{
        .snapshot_index = ordered_pair_indices[
          work_request_index - snapshot_count],
        .after_header = true,
      };
}

// Production and tests share this mapping.  One chain is laid out as
// [full_0 .. full_N-1, after_0(FENCE) .. after_N-1], not as interleaved pairs:
// the single fence delays the complete validation half until every body read
// has finished, while both halves remain internally batchable.  The tail CQE
// is consumed only after all 2*N WRs.
constexpr PeerRdmaReadPairChainItem peer_rdma_read_pair_chain_item(
    const std::uint32_t work_request_index,
    const std::uint32_t pair_count) {
  return PeerRdmaReadPairChainItem{
    .pair_index = work_request_index < pair_count
      ? work_request_index : work_request_index - pair_count,
    .after_header = work_request_index >= pair_count,
  };
}

constexpr std::uint32_t peer_rdma_read_pair_work_request_count(
    const std::uint32_t pair_count) {
  return pair_count <= std::numeric_limits<std::uint32_t>::max() / 2
    ? pair_count * 2
    : 0;
}

constexpr std::uint32_t peer_rdma_read_batch_completion_count(
    const std::uint32_t read_count,
    const PeerRdmaReadCreditPlan& plan) {
  const std::uint32_t group_limit =
    peer_rdma_read_batch_group_limit(plan);
  return read_count == 0 || group_limit == 0
    ? 0
    : 1 + (read_count - 1) / group_limit;
}

inline bool try_reserve_bounded_counter(
    std::atomic<std::uint32_t>& counter,
    const std::uint32_t limit,
    const std::uint32_t count) {
  if (count == 0 || count > limit) return false;
  std::uint32_t current = counter.load(std::memory_order_acquire);
  while (current <= limit - count) {
    if (counter.compare_exchange_weak(
          current, current + count,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

inline void release_reserved_counter(
    std::atomic<std::uint32_t>& counter,
    const std::uint32_t count) {
  const std::uint32_t previous = counter.fetch_sub(
    count, std::memory_order_acq_rel);
  if (previous < count) {
    assert(false && "RDMA credit counter underflow");
    std::abort();
  }
}

// Reserve the three credit domains as one logical operation. This is not a
// multi-word CAS, so a failed later domain rolls back every earlier one before
// returning. No caller may wait while retaining credits for an unposted WR;
// consequently concurrent producers cannot each hold a partial chain and
// deadlock on the remaining global credits.
inline bool try_reserve_peer_rdma_read_group(
    std::atomic<std::uint32_t>& peer_outstanding,
    std::atomic<std::uint32_t>& qp_outstanding,
    std::atomic<std::uint32_t>& global_outstanding,
    const PeerRdmaReadCreditPlan& plan,
    const std::uint32_t count) {
  if (count == 0 ||
      count > peer_rdma_read_batch_group_limit(plan)) {
    return false;
  }
  if (!try_reserve_bounded_counter(
        peer_outstanding, plan.per_peer, count)) {
    return false;
  }
  if (!try_reserve_bounded_counter(qp_outstanding, plan.per_qp, count)) {
    release_reserved_counter(peer_outstanding, count);
    return false;
  }
  if (!try_reserve_bounded_counter(
        global_outstanding, plan.global, count)) {
    release_reserved_counter(qp_outstanding, count);
    release_reserved_counter(peer_outstanding, count);
    return false;
  }
  return true;
}

struct PeerRdmaReadCreditRequest {
  std::atomic<std::uint32_t>* peer_outstanding{};
  std::atomic<std::uint32_t>* qp_outstanding{};
  std::uint32_t count{};
};

inline void release_peer_rdma_read_group(
    const PeerRdmaReadCreditRequest& request,
    std::atomic<std::uint32_t>& global_outstanding) {
  release_reserved_counter(*request.qp_outstanding, request.count);
  release_reserved_counter(*request.peer_outstanding, request.count);
  release_reserved_counter(global_outstanding, request.count);
}

// Reserve every chain needed by one scheduler wave before posting its first
// WR. If any domain is temporarily full, unwind the complete prefix. This is
// the transaction boundary required by a resumable caller: false means zero
// posted WRs and zero credit retained by this attempt, so retrying the same
// immutable request wave cannot duplicate an RDMA operation.
inline bool try_reserve_peer_rdma_read_wave(
    const std::span<const PeerRdmaReadCreditRequest> requests,
    std::atomic<std::uint32_t>& global_outstanding,
    const PeerRdmaReadCreditPlan& plan) {
  std::size_t reserved = 0;
  for (; reserved < requests.size(); ++reserved) {
    const PeerRdmaReadCreditRequest& request = requests[reserved];
    if (request.peer_outstanding != nullptr &&
        request.qp_outstanding != nullptr &&
        try_reserve_peer_rdma_read_group(
          *request.peer_outstanding, *request.qp_outstanding,
          global_outstanding, plan, request.count)) {
      continue;
    }
    while (reserved != 0) {
      release_peer_rdma_read_group(requests[--reserved],
                                   global_outstanding);
    }
    return false;
  }
  return true;
}

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

// One request wave takes a single rotating start ticket for each destination
// shard. Chain ordinals are then local to that shard: requests for a second
// shard cannot perturb the first shard's QP sequence or cause early reuse.
constexpr std::uint32_t select_peer_data_qp_for_wave_chain(
    const std::uint32_t qps_per_peer,
    const std::uint32_t wave_start_ticket,
    const std::uint32_t chain_ordinal) {
  return select_peer_data_qp(
    qps_per_peer, wave_start_ticket + chain_ordinal);
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
static_assert(select_peer_data_qp_for_wave_chain(4, 1, 0) == 2);
static_assert(select_peer_data_qp_for_wave_chain(4, 1, 1) == 3);
static_assert(select_peer_data_qp_for_wave_chain(4, 1, 2) == 1);
static_assert(peer_rdma_read_batch_group_limit(
                PeerRdmaReadCreditPlan{3, 8, 16, 32, 32}) == 8);
static_assert(peer_rdma_read_pair_wave_limit(
                PeerRdmaReadCreditPlan{3, 7, 21, 42, 42}) == 9);
static_assert(peer_rdma_read_batch_completion_count(
                17, PeerRdmaReadCreditPlan{1, 8, 8, 8, 8}) == 3);

}  // namespace memory_node_detail
