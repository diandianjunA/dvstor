#pragma once

#include <algorithm>
#include <cstdint>

namespace memory_node_detail {

struct StorageOwnerCpuPlan {
  std::uint32_t foreground_workers{};
  std::uint32_t maintenance_workers{};
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
  plan.maintenance_workers = std::min(
    std::max<std::uint32_t>(1, configured_maintenance_workers),
    std::max<std::uint32_t>(1, budget >= 8 ? budget / 10 : 1));
  plan.peer_reverse_workers = remote_peer_count == 0 ? 0 : std::min(
    std::uint32_t{8},
    std::max<std::uint32_t>(1, budget >= 8 ? budget / 5 : 1));
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
  const std::uint32_t foreground_limit = std::min(
    std::max<std::uint32_t>(1, rpc_parallelism), cpu_parallelism);

  if (remote_peer_count == 0) {
    plan.foreground_workers = std::min(
      foreground_limit, std::max<std::uint32_t>(1, stage1_budget));
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
  return plan;
}

}  // namespace memory_node_detail
