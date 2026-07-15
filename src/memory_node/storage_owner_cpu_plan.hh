#pragma once

#include <algorithm>
#include <cstdint>

namespace memory_node_detail {

struct StorageOwnerCpuPlan {
  std::uint32_t foreground_workers{};
  std::uint32_t maintenance_workers{};
  std::uint32_t peer_search_workers{};
  std::uint32_t peer_reverse_workers{};
  std::uint32_t peer_progress_threads{};
  std::uint32_t foreground_progress_threads{};
};

// Derive one stable division of the CPUs assigned to this storage process.
// Peer search receives the remainder because every finalized insert performs
// one complete search on every remote shard.  The configured values remain
// upper bounds on foreground/maintenance parallelism; no benchmark knob is
// introduced for colocated-process scheduling.
inline StorageOwnerCpuPlan derive_storage_owner_cpu_plan(
    std::uint32_t available_cpus,
    std::uint32_t configured_threads,
    std::uint32_t rpc_parallelism,
    std::uint32_t configured_maintenance_workers,
    std::uint32_t remote_peer_count) {
  const std::uint32_t budget = std::max<std::uint32_t>(1, available_cpus);
  const std::uint32_t cpu_parallelism = std::max<std::uint32_t>(
    1, configured_threads / 2);
  const std::uint32_t fanout = std::max<std::uint32_t>(1, remote_peer_count + 1);

  StorageOwnerCpuPlan plan;
  plan.peer_progress_threads = remote_peer_count == 0 ? 0 : 3;
  // service_storage_runtime runs forever on the lifecycle thread and polls
  // the foreground send/receive CQs, so it needs its own CPU just like the
  // explicit worker threads below.
  plan.foreground_progress_threads = 1;
  plan.foreground_workers = std::min(
    std::max<std::uint32_t>(1, rpc_parallelism),
    std::max<std::uint32_t>(1,
      std::min(cpu_parallelism, budget / fanout)));
  plan.maintenance_workers = std::min(
    std::max<std::uint32_t>(1, configured_maintenance_workers),
    std::max<std::uint32_t>(1, budget >= 8 ? budget / 10 : 1));
  plan.peer_reverse_workers = remote_peer_count == 0 ? 0 : std::min(
    std::uint32_t{8},
    std::max<std::uint32_t>(1, budget >= 8 ? budget / 5 : 1));

  const std::uint64_t reserved =
    static_cast<std::uint64_t>(plan.peer_progress_threads) +
    plan.foreground_progress_threads +
    plan.foreground_workers + plan.maintenance_workers +
    plan.peer_reverse_workers;
  const std::uint32_t search_budget = reserved < budget
    ? static_cast<std::uint32_t>(budget - reserved) : 1;
  plan.peer_search_workers = remote_peer_count == 0 ? 0 :
    std::min(cpu_parallelism, std::max<std::uint32_t>(1, search_budget));
  return plan;
}

}  // namespace memory_node_detail
