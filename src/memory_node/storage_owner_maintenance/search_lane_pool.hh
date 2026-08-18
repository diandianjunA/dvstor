#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "memory_node/storage_owner_maintenance/stage2_tracker.hh"

namespace memory_node_storage_owner_maintenance_detail {

// Legacy single-worker sizing helper retained for callers/tests that size one
// executor in isolation.  Return size_t so a host whose credit/context bound
// crosses UINT32_MAX cannot silently wrap before the runtime validates the
// StorageOwnerThread wire index type.
constexpr std::size_t stage2_search_lane_count(
    const std::size_t context_capacity,
    const std::size_t per_peer_read_window,
    const std::size_t global_read_window) {
  const std::size_t peer_window = std::max<std::size_t>(
    1, per_peer_read_window);
  // One suspended lane can occupy at most one complete peer window. Derive the
  // useful number of independent dependency chains from the real global/peer
  // credit ratio instead of a machine-specific constant. Context capacity is
  // the hard memory bound: registered scratch cannot grow with backlog.
  const std::size_t independent_windows = std::max<std::size_t>(
    1, global_read_window == 0
      ? 1
      : 1 + (global_read_window - 1) / peer_window);
  return std::min<std::size_t>(
    std::max<std::size_t>(1, context_capacity),
    independent_windows);
}

// One continuation lane can have either one ordinary graph/snapshot wave or
// one ordered stable-vector wave in flight.  The latter consumes two RDMA READ
// WRs per logical pair.  This is the actual peak transport footprint used to
// turn the process-wide READ window into a process-wide lane budget.
constexpr std::size_t stage2_search_lane_peak_rdma_wrs(
    const std::size_t configured_batch,
    const std::size_t ordinary_wave_limit,
    const std::size_t ordered_pair_wave_limit) {
  const std::size_t batch = std::max<std::size_t>(1, configured_batch);
  const std::size_t ordinary_peak = std::min(batch, ordinary_wave_limit);
  const std::size_t pair_count = std::min(batch, ordered_pair_wave_limit);
  const std::size_t ordered_pair_peak =
    pair_count > std::numeric_limits<std::size_t>::max() / 2
      ? std::numeric_limits<std::size_t>::max()
      : pair_count * 2;
  return std::max<std::size_t>(
    1, std::max(ordinary_peak, ordered_pair_peak));
}

constexpr std::size_t saturating_stage2_lane_product(
    const std::size_t lhs,
    const std::size_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs
    ? std::numeric_limits<std::size_t>::max()
    : lhs * rhs;
}

// Search continuations retain no RDMA credit merely by holding a lane: every
// posted wave is still admitted transactionally by the per-QP, per-peer, and
// process-wide credit counters.  The lease therefore exists to bound
// registered scratch ownership and runnable continuation state, not to model
// a reservation of each lane's worst-case wave.  Give the lease calculation a
// floor of four active continuations per executor when that scratch has
// already been allocated, and cap the active lease at 32 per node.  This
// preserves enough latency-hiding headroom without allowing synchronous
// Stage2/reverse waits to occupy every maintenance executor at once.
constexpr std::size_t stage2_global_search_lane_lease_limit(
    const std::size_t worker_count,
    const std::size_t physical_lanes_per_worker,
    const std::size_t credit_derived_lanes) {
  if (worker_count == 0 || physical_lanes_per_worker == 0) return 0;
  constexpr std::size_t kPipelineLanesPerWorker = 4;
  constexpr std::size_t kPipelineLeaseCap = 32;
  const std::size_t total_physical_lanes =
    saturating_stage2_lane_product(
      worker_count, physical_lanes_per_worker);
  const std::size_t pipeline_floor = std::min(
    total_physical_lanes,
    std::min(
      kPipelineLeaseCap,
      saturating_stage2_lane_product(
        worker_count, kPipelineLanesPerWorker)));
  return std::min(
    total_physical_lanes,
    std::min(
      kPipelineLeaseCap,
      std::max(credit_derived_lanes, pipeline_floor)));
}

// Stage2 READ credits are process-wide, whereas maintenance workers are OS
// threads.  Compute one global number of useful suspended dependency chains;
// the old per-worker computation multiplied the same global credit window by
// every worker.
//
// A lane is continuation/scratch ownership, not a reservation of its largest
// possible RDMA wave. try_post_peer_* reserves the actual wave atomically and
// rolls the whole reservation back on pressure. One lane per peak-sized credit
// window is nevertheless insufficient: when every such lane is waiting on the
// HCA, an OS worker has no ready continuation with which to hide that latency.
// Keep two bounded lanes per useful credit window (one in flight and one ready)
// and, when context capacity permits, at least two lanes per worker. This is a
// fixed double buffer, not backlog-dependent growth; the context pool remains
// the hard memory bound and dynamic credit admission remains authoritative.
constexpr std::size_t stage2_global_search_lane_count(
    const std::size_t worker_count,
    const std::size_t contexts_per_worker,
    const std::size_t useful_wave_rdma_wrs,
    const std::size_t global_read_window) {
  if (worker_count == 0) return 0;
  const std::size_t contexts = std::max<std::size_t>(1, contexts_per_worker);
  const std::size_t maximum_lanes =
    saturating_stage2_lane_product(worker_count, contexts);
  const std::size_t useful_wave =
    std::max<std::size_t>(1, useful_wave_rdma_wrs);
  const std::size_t credit_windows = global_read_window == 0
    ? 1
    : 1 + (global_read_window - 1) / useful_wave;
  const std::size_t buffered_credit_windows =
    saturating_stage2_lane_product(credit_windows, 2);
  const std::size_t worker_double_buffer =
    saturating_stage2_lane_product(
      worker_count, std::min<std::size_t>(contexts, 2));
  return std::min(
    maximum_lanes,
    std::max(worker_double_buffer, buffered_credit_windows));
}

// Deterministically spread the global lane budget.  The first remainder
// workers receive one extra lane; no worker exceeds its bounded context count.
// size_t is intentional: runtime performs an explicit u32 range check before
// constructing transport state instead of accepting a truncating cast.
constexpr std::size_t stage2_search_lanes_for_worker(
    const std::size_t worker_index,
    const std::size_t worker_count,
    const std::size_t contexts_per_worker,
    const std::size_t global_lane_count) {
  if (worker_count == 0 || worker_index >= worker_count) return 0;
  const std::size_t contexts = std::max<std::size_t>(1, contexts_per_worker);
  const std::size_t maximum_lanes =
    saturating_stage2_lane_product(worker_count, contexts);
  const std::size_t distributed = std::min(
    maximum_lanes, std::max(worker_count, global_lane_count));
  const std::size_t base = distributed / worker_count;
  const std::size_t remainder = distributed % worker_count;
  return std::min(contexts, base + (worker_index < remainder));
}

constexpr std::size_t stage2_round_robin_context_index(
    const std::size_t scan_begin,
    const std::size_t offset,
    const std::size_t context_count) {
  if (context_count == 0) return 0;
  const std::size_t begin = scan_begin % context_count;
  const std::size_t normalized_offset = offset % context_count;
  const std::size_t tail = context_count - begin;
  return normalized_offset < tail
    ? begin + normalized_offset : normalized_offset - tail;
}

// A lane can be rebound only after both independent owners have released it:
// the HCA must have retired every WR that can still write registered scratch,
// and the search dispatcher must have copied out/reset its continuation.  In
// particular, CQ readiness alone is insufficient because a retryable pointer
// can leave a live beam in an idle transport phase.
constexpr bool stage2_search_lane_rebindable(
    const bool rdma_ready,
    const bool search_state_idle) {
  return rdma_ready && search_state_idle;
}

// Fixed-capacity lane ownership for one maintenance OS worker.  A generation-
// tagged context handle, rather than only its reusable slot, fences late CQEs
// from a previous Stage2 context.  The caller must pass rdma_ready=true only
// after that lane's post balance reached zero; a lane can therefore never be
// recycled while an HCA still owns its registered scratch.
class Stage2SearchLanePool {
 public:
  explicit Stage2SearchLanePool(const std::size_t capacity) {
    if (capacity == 0 ||
        capacity > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("stage2 search lane capacity is out of range");
    }
    // Validate before allocation.  Constructing owners_(capacity) in the
    // initializer list would attempt a multi-billion-element allocation before
    // rejecting a value that cannot be represented by the public lane type.
    owners_.resize(capacity);
  }

  [[nodiscard]] std::optional<std::uint32_t> try_acquire(
      const Stage2ContextHandle handle) {
    for (std::size_t lane = 0; lane < owners_.size(); ++lane) {
      if (owners_[lane].has_value() && *owners_[lane] == handle) {
        return static_cast<std::uint32_t>(lane);
      }
    }
    for (std::size_t lane = 0; lane < owners_.size(); ++lane) {
      if (!owners_[lane].has_value()) {
        owners_[lane] = handle;
        ++size_;
        return static_cast<std::uint32_t>(lane);
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] bool owns(const std::uint32_t lane,
                          const Stage2ContextHandle handle) const {
    return lane < owners_.size() && owners_[lane].has_value() &&
      *owners_[lane] == handle;
  }

  [[nodiscard]] bool release(const std::uint32_t lane,
                             const Stage2ContextHandle handle,
                             const bool rdma_ready) {
    if (!rdma_ready || !owns(lane, handle)) return false;
    owners_[lane].reset();
    --size_;
    return true;
  }

  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] std::size_t capacity() const { return owners_.size(); }
  [[nodiscard]] bool full() const { return size_ == owners_.size(); }

 private:
  std::vector<std::optional<Stage2ContextHandle>> owners_;
  std::size_t size_{};
};

static_assert(stage2_search_lane_count(0, 24, 96) == 1);
static_assert(stage2_search_lane_count(1, 24, 96) == 1);
static_assert(stage2_search_lane_count(3, 24, 96) == 3);
static_assert(stage2_search_lane_count(16, 24, 96) == 4);
static_assert(stage2_search_lane_count(16, 8, 96) == 12);
static_assert(stage2_search_lane_count(16, 0, 0) == 1);
static_assert(stage2_search_lane_peak_rdma_wrs(32, 96, 48) == 64);
static_assert(stage2_search_lane_peak_rdma_wrs(32, 24, 0) == 24);
static_assert(stage2_global_search_lane_lease_limit(8, 16, 16) == 32);
static_assert(stage2_global_search_lane_lease_limit(8, 2, 16) == 16);
static_assert(stage2_global_search_lane_lease_limit(4, 16, 8) == 16);
static_assert(stage2_global_search_lane_lease_limit(1, 2, 1) == 2);
static_assert(stage2_global_search_lane_lease_limit(64, 1, 1) == 32);
static_assert(stage2_global_search_lane_lease_limit(8, 16, 128) == 32);
static_assert(stage2_global_search_lane_count(4, 16, 1, 96) == 64);
static_assert(stage2_global_search_lane_count(4, 16, 2, 96) == 64);
static_assert(stage2_global_search_lane_count(4, 16, 56, 224) == 8);
static_assert(stage2_global_search_lane_count(4, 16, 64, 96) == 8);
static_assert(stage2_global_search_lane_count(5, 16, 1, 4) == 10);
static_assert(stage2_global_search_lane_count(5, 1, 1, 4096) == 5);
static_assert(stage2_search_lanes_for_worker(0, 5, 16, 7) == 2);
static_assert(stage2_search_lanes_for_worker(1, 5, 16, 7) == 2);
static_assert(stage2_search_lanes_for_worker(2, 5, 16, 7) == 1);
static_assert(stage2_round_robin_context_index(3, 0, 5) == 3);
static_assert(stage2_round_robin_context_index(3, 1, 5) == 4);
static_assert(stage2_round_robin_context_index(3, 2, 5) == 0);
static_assert(stage2_search_lane_rebindable(true, true));
static_assert(!stage2_search_lane_rebindable(false, true));
static_assert(!stage2_search_lane_rebindable(true, false));

}  // namespace memory_node_storage_owner_maintenance_detail
