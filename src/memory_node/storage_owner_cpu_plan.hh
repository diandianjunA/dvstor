#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace memory_node_detail {

struct StorageOwnerCpuPlan {
  // CPU lanes reserved for foreground graph work.  This value participates
  // in the strict process-wide CPU partition below.
  std::uint32_t foreground_workers{};
  // Synchronous authority requests spend most of their lifetime waiting for
  // a physical-home response.  Coordinators are therefore allowed to
  // multiplex over the foreground CPU lanes, but remain bounded by both the
  // registered RPC window and a small per-lane factor.
  std::uint32_t foreground_coordinators{};
  std::uint32_t maintenance_workers{};
  // Admission debt is sized from the pre-rebalance Stage2 lane count.  This
  // keeps a CPU transfer from reverse processing to Stage2 from silently
  // increasing acknowledged-but-unfinished work; only the executor count
  // above changes.
  std::uint32_t maintenance_admission_workers{};
  std::uint32_t peer_stage1_workers{};
  std::uint32_t peer_reverse_workers{};
  std::uint32_t peer_cleanup_workers{};
  std::uint32_t peer_placement_workers{};
  std::uint32_t peer_progress_threads{};
  std::uint32_t foreground_progress_threads{};
};

// Derive one stable division of the CPUs assigned to this storage process.
// Stage1 has one coordinator and exactly one physical-home executor per
// insert.  Therefore its CPU split is independent of the number of remote
// shards: foreground and peer Stage1 workers share the budget left after the
// fixed progress, maintenance, and reverse-update roles.  Configured values
// remain upper bounds; no benchmark-specific scheduling knob is introduced.
inline StorageOwnerCpuPlan derive_storage_owner_cpu_plan(
    std::uint32_t available_cpus,
    std::uint32_t configured_threads,
    std::uint32_t rpc_parallelism,
    std::uint32_t configured_maintenance_workers,
    std::uint32_t remote_peer_count) {
  const std::uint32_t budget = std::max<std::uint32_t>(1, available_cpus);
  const std::uint32_t cpu_parallelism = std::max<std::uint32_t>(
    1, configured_threads / 2);

  StorageOwnerCpuPlan plan;
  // One CQ progress thread and one response sender are the only fixed peer
  // roles. Graph requests use the bounded async send lanes directly; the
  // legacy outgoing relay no longer consumes a thread or a core.
  plan.peer_progress_threads = remote_peer_count == 0 ? 0 : 2;
  // service_storage_runtime runs forever on the lifecycle thread and polls
  // the foreground send/receive CQs, so it needs its own CPU just like the
  // explicit worker threads below.
  plan.foreground_progress_threads = 1;
  // Stage2 executes several synchronous one-sided-RDMA waves for every
  // inserted vector, while reverse updates are coalesced before they reach a
  // peer worker.  Giving the latter twice as many CPUs as Stage2 therefore
  // leaves the latency-bearing side unable to keep enough reads in flight on
  // a colocated shard.  Rebalance the same fixed CPU pool; do not enlarge it:
  // transfer at most two reverse lanes above a two-worker service floor to
  // Stage2, up to a conservative one-fifth-of-the-process Stage2 target.  The
  // transfer cap keeps reverse-heavy upsert/delete workloads provisioned on
  // larger hosts where the insert-only headroom observation does not justify
  // draining the full reverse pool.  The configured maintenance value remains
  // a hard upper bound, and the Stage1 budget below is unchanged.
  const std::uint32_t baseline_maintenance_workers = std::min(
    std::max<std::uint32_t>(1, configured_maintenance_workers),
    std::max<std::uint32_t>(1, budget >= 8 ? budget / 10 : 1));
  const std::uint32_t desired_maintenance_workers = std::min(
    std::max<std::uint32_t>(1, configured_maintenance_workers),
    std::max<std::uint32_t>(1, budget >= 8 ? budget / 5 : 1));
  plan.maintenance_admission_workers = baseline_maintenance_workers;
  if (remote_peer_count == 0) {
    // There is no reverse pool from which to transfer a lane.  Retain the
    // baseline split: raising Stage2 here would consume foreground CPUs on
    // mid-sized local-only deployments and would no longer be a fixed-pool
    // rebalance.
    plan.maintenance_workers = baseline_maintenance_workers;
    plan.peer_reverse_workers = 0;
  } else {
    const std::uint32_t baseline_reverse_workers = std::min(
      std::uint32_t{8},
      std::max<std::uint32_t>(1, budget >= 8 ? budget / 5 : 1));
    const std::uint32_t reverse_service_floor = std::min(
      std::uint32_t{2}, baseline_reverse_workers);
    const std::uint32_t transferable_reverse_workers =
      baseline_reverse_workers - reverse_service_floor;
    const std::uint32_t requested_transfer =
      desired_maintenance_workers - baseline_maintenance_workers;
    constexpr std::uint32_t kMaxReverseToMaintenanceTransfer = 2;
    const std::uint32_t transferred_workers = std::min({
      transferable_reverse_workers,
      requested_transfer,
      kMaxReverseToMaintenanceTransfer});
    plan.maintenance_workers =
      baseline_maintenance_workers + transferred_workers;
    plan.peer_reverse_workers =
      baseline_reverse_workers - transferred_workers;
  }
  // Cleanup activation is on every replacement/deletion path and may wait
  // behind an in-progress same-token retry. Give it a bounded CPU-scaled
  // pool so an unrelated token is never forced through one process-wide
  // worker. The fixed 8-CPU quantum and cap keep cheap control work from
  // stealing the Stage1/foreground data path on either small or large hosts.
  plan.peer_cleanup_workers = remote_peer_count == 0 ? 0 : std::min({
    std::uint32_t{8},
    std::max<std::uint32_t>(1, rpc_parallelism),
    std::max<std::uint32_t>(1, budget / 8)});
  // Placement and node-control mutate global physical state and deliberately
  // remain in their own serial ordering domain.
  plan.peer_placement_workers = remote_peer_count == 0 ? 0 : 1;

  const std::uint64_t reserved =
    static_cast<std::uint64_t>(plan.peer_progress_threads) +
    plan.foreground_progress_threads +
    plan.maintenance_workers + plan.peer_reverse_workers +
    plan.peer_cleanup_workers + plan.peer_placement_workers;
  const std::uint32_t stage1_budget = reserved < budget
    ? static_cast<std::uint32_t>(budget - reserved) : 0;
  // A foreground coordinator performs authority bookkeeping and local work,
  // while the physical-home executor performs the full graph construction.
  // Do not let adding compute clients continuously transfer CPU lanes from
  // the latter to the former.  Once the coordinator side reaches half of the
  // configured CPU parallelism, extra RPC slots are multiplexed as waiting
  // coordinators below instead of changing the CPU partition.
  const std::uint32_t foreground_cpu_limit =
    std::max<std::uint32_t>(1, cpu_parallelism / 2);
  const std::uint32_t foreground_limit = std::min(
    std::max<std::uint32_t>(1, rpc_parallelism), foreground_cpu_limit);

  const auto finish_foreground_coordinators = [&]() {
    if (remote_peer_count == 0) {
      // A local-only coordinator executes graph work continuously, so
      // oversubscribing it cannot hide any transport wait.
      plan.foreground_coordinators = plan.foreground_workers;
      return;
    }
    constexpr std::uint32_t kCoordinatorMultiplexPerCpu = 4;
    const std::uint64_t cpu_bounded =
      static_cast<std::uint64_t>(plan.foreground_workers) *
      kCoordinatorMultiplexPerCpu;
    plan.foreground_coordinators = std::max<std::uint32_t>(
      plan.foreground_workers,
      std::min<std::uint32_t>(
        foreground_limit,
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
          cpu_bounded, std::numeric_limits<std::uint32_t>::max()))));
  };

  if (remote_peer_count == 0) {
    plan.foreground_workers = std::min(
      foreground_limit, std::max<std::uint32_t>(1, stage1_budget));
    finish_foreground_coordinators();
    return plan;
  }

  // One worker on each side is a functional minimum.  With a usable CPU
  // budget, divide the remainder in proportion to each side's real
  // concurrency ceiling; critically, remote_peer_count is not a divisor.
  const std::uint32_t usable_stage1_budget =
    std::max<std::uint32_t>(2, stage1_budget);
  const std::uint64_t combined_limit =
    static_cast<std::uint64_t>(foreground_limit) + cpu_parallelism;
  if (usable_stage1_budget >= combined_limit) {
    plan.foreground_workers = foreground_limit;
    plan.peer_stage1_workers = cpu_parallelism;
    finish_foreground_coordinators();
    return plan;
  }

  const std::uint32_t foreground_share = static_cast<std::uint32_t>(
    static_cast<std::uint64_t>(usable_stage1_budget) * foreground_limit /
    combined_limit);
  plan.foreground_workers = std::min(
    foreground_limit, std::max<std::uint32_t>(1, foreground_share));
  const std::uint32_t peer_stage1_budget =
    usable_stage1_budget - plan.foreground_workers;
  plan.peer_stage1_workers = std::min(
    cpu_parallelism, std::max<std::uint32_t>(1, peer_stage1_budget));
  finish_foreground_coordinators();
  return plan;
}

}  // namespace memory_node_detail
