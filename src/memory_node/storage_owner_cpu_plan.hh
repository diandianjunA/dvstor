#pragma once

#include <algorithm>
#include <array>
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

struct PhysicalHomeWorkerSplit {
  std::uint32_t stage1{};
  std::uint32_t stage2_home{};
};

// Isolate latency-bearing Stage1 publication from the much more numerous
// read-only Stage2 home operations without changing the CPU plan's total.
// Very small deployments keep the legacy shared executor because taking its
// only Stage1 lane would make foreground progress impossible.
inline PhysicalHomeWorkerSplit split_physical_home_workers(
    std::uint32_t total) {
  // One dedicated Stage2 lane is the liveness floor. Remaining lanes stay
  // shared and are borrowed by Stage2 only when queue age proves pressure;
  // a static one-third reservation needlessly starves latency-bearing Stage1.
  const std::uint32_t stage2_home = total >= 2 ? 1 : 0;
  return PhysicalHomeWorkerSplit{
    .stage1 = total - stage2_home,
    .stage2_home = stage2_home,
  };
}

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

  const std::uint32_t baseline_maintenance_workers = std::min(
    std::max<std::uint32_t>(1, configured_maintenance_workers),
    std::max<std::uint32_t>(1, budget >= 8 ? budget / 10 : 1));
  plan.maintenance_admission_workers = baseline_maintenance_workers;
  // Placement and node-control mutate global physical state and deliberately
  // remain in their own serial ordering domain.
  plan.peer_placement_workers = remote_peer_count == 0 ? 0 : 1;

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
    // There is no peer service pool from which to borrow CPUs.  Retain the
    // conservative admission-sized maintenance pool and spend the remainder
    // on the continuously-running local foreground executor.
    plan.maintenance_workers = baseline_maintenance_workers;
    const std::uint64_t reserved =
      static_cast<std::uint64_t>(plan.foreground_progress_threads) +
      plan.maintenance_workers;
    const std::uint32_t stage1_budget = reserved < budget
      ? static_cast<std::uint32_t>(budget - reserved) : 0;
    plan.foreground_workers = std::min(
      foreground_limit, std::max<std::uint32_t>(1, stage1_budget));
    finish_foreground_coordinators();
    return plan;
  }

  // Nine CPUs are the functional remote-service profile: two progress
  // threads, one lifecycle poller, one placement authority, and one lane for
  // each of Stage2, reverse, cleanup, foreground Stage1, and physical-home
  // Stage1.  Start there and assign every additional CPU exactly once.  This
  // construction is intentionally incremental: increasing the process CPU
  // budget can grow or saturate a worker domain, but can never shrink it.
  plan.maintenance_workers = 1;
  plan.peer_reverse_workers = 1;
  plan.peer_cleanup_workers = 1;
  plan.foreground_workers = 1;
  plan.peer_stage1_workers = 1;
  constexpr std::uint32_t kRemoteFunctionalCpuFloor = 9;

  const std::uint32_t maintenance_cap =
    std::max<std::uint32_t>(1, configured_maintenance_workers);
  const std::uint32_t cleanup_cap = std::min(
    std::uint32_t{8}, std::max<std::uint32_t>(1, rpc_parallelism));
  constexpr std::uint32_t kMaxReverseToMaintenanceTransfer = 2;

  const auto maintenance_admission_target_for =
      [&](std::uint32_t cpu_budget) {
    return std::min(
      maintenance_cap,
      std::max<std::uint32_t>(1, cpu_budget >= 8 ? cpu_budget / 10 : 1));
  };
  const auto maintenance_target_for = [&](std::uint32_t cpu_budget) {
    const std::uint32_t admission =
      maintenance_admission_target_for(cpu_budget);
    return std::min(
      maintenance_cap,
      std::max<std::uint32_t>(
        std::min<std::uint32_t>(maintenance_cap, 8),
        admission > std::numeric_limits<std::uint32_t>::max() - 2
          ? std::numeric_limits<std::uint32_t>::max()
          : admission + 2));
  };
  const auto reverse_target_for = [&](std::uint32_t cpu_budget) {
    const std::uint32_t baseline_reverse = std::min(
      std::uint32_t{8},
      std::max<std::uint32_t>(1,
        cpu_budget >= 8 ? cpu_budget / 5 : 1));
    const std::uint32_t reverse_floor = std::min(
      std::uint32_t{2}, baseline_reverse);
    const std::uint32_t transferable = baseline_reverse - reverse_floor;
    const std::uint32_t admission =
      maintenance_admission_target_for(cpu_budget);
    const std::uint32_t desired_maintenance = std::min(
      maintenance_cap,
      std::max<std::uint32_t>(1,
        cpu_budget >= 8 ? cpu_budget / 5 : 1));
    const std::uint32_t requested_transfer =
      desired_maintenance - admission;
    return baseline_reverse - std::min({
      transferable,
      requested_transfer,
      kMaxReverseToMaintenanceTransfer});
  };

  enum class WorkerRole : std::uint8_t {
    kMaintenance,
    kPeerStage1,
    kForeground,
    kCleanup,
    kReverse,
  };
  const auto try_grow = [&](WorkerRole role,
                            std::uint32_t cpu_budget) {
    std::uint32_t* workers = nullptr;
    std::uint32_t cap = 0;
    switch (role) {
      case WorkerRole::kMaintenance:
        workers = &plan.maintenance_workers;
        cap = maintenance_cap;
        break;
      case WorkerRole::kPeerStage1:
        workers = &plan.peer_stage1_workers;
        cap = cpu_parallelism;
        break;
      case WorkerRole::kForeground:
        workers = &plan.foreground_workers;
        cap = foreground_limit;
        break;
      case WorkerRole::kCleanup:
        workers = &plan.peer_cleanup_workers;
        cap = cleanup_cap;
        break;
      case WorkerRole::kReverse:
        workers = &plan.peer_reverse_workers;
        cap = reverse_target_for(cpu_budget);
        break;
    }
    if (*workers >= cap) return false;
    ++*workers;
    return true;
  };

  // Establish the full durable-pipeline service profile one CPU at a time.
  // Interleave Stage1, Stage2, and reverse growth so a small host never spends
  // all of its first CPUs on one dependency while the other stages remain at
  // their one-lane liveness minima.  The sequence is a prefix allocation: a
  // larger CPU budget contains every assignment made by a smaller one.  Once
  // complete it provides 8 Stage2, 2 reverse, 3 foreground, 5 physical-home,
  // and 2 cleanup lanes (subject to the operator's concurrency caps).
  constexpr std::array<WorkerRole, 15> kServiceProfileGrowth{
    WorkerRole::kForeground,
    WorkerRole::kPeerStage1,
    WorkerRole::kReverse,
    WorkerRole::kMaintenance,
    WorkerRole::kMaintenance,
    WorkerRole::kPeerStage1,
    WorkerRole::kForeground,
    WorkerRole::kMaintenance,
    WorkerRole::kPeerStage1,
    WorkerRole::kMaintenance,
    WorkerRole::kMaintenance,
    WorkerRole::kMaintenance,
    WorkerRole::kMaintenance,
    WorkerRole::kCleanup,
    WorkerRole::kPeerStage1,
  };
  // Spend the remaining elastic budget on the actual service mix.  Three
  // quarters goes to the two Stage1 sides in their 1:2 concurrency ratio;
  // cleanup and reverse receive bounded 15%/10% shares.  Once a bounded role
  // saturates, skip its slots and continue the same cycle, preserving prefix
  // monotonicity for every CPU count and retaining reverse capacity on large
  // hosts.  The cycle is independent of peer count because the queues are
  // shared, work-conserving domains.
  constexpr std::array<WorkerRole, 20> kElasticCycle{
    WorkerRole::kPeerStage1, WorkerRole::kForeground,
    WorkerRole::kPeerStage1, WorkerRole::kCleanup,
    WorkerRole::kPeerStage1, WorkerRole::kReverse,
    WorkerRole::kPeerStage1, WorkerRole::kForeground,
    WorkerRole::kPeerStage1, WorkerRole::kCleanup,
    WorkerRole::kPeerStage1, WorkerRole::kForeground,
    WorkerRole::kPeerStage1, WorkerRole::kReverse,
    WorkerRole::kPeerStage1, WorkerRole::kCleanup,
    WorkerRole::kPeerStage1, WorkerRole::kForeground,
    WorkerRole::kPeerStage1, WorkerRole::kForeground,
  };
  // Replay one deterministic assignment for each CPU above the functional
  // floor.  Replaying the same prefix is what makes the monotonicity guarantee
  // structural even when an operator cap previously left CPUs unassigned:
  // newly-raised maintenance and reverse targets can consume at most the one
  // newly-added CPU, never latent slack from earlier budgets.
  std::size_t service_index = 0;
  std::size_t cycle_index = 0;
  for (std::uint64_t virtual_cpu = kRemoteFunctionalCpuFloor + 1;
       virtual_cpu <= budget; ++virtual_cpu) {
    const std::uint32_t cpu_budget =
      static_cast<std::uint32_t>(virtual_cpu);
    bool assigned = false;

    while (service_index < kServiceProfileGrowth.size()) {
      const WorkerRole role = kServiceProfileGrowth[service_index++];
      if (try_grow(role, cpu_budget)) {
        assigned = true;
        break;
      }
    }
    if (assigned) continue;

    // Above the full service profile, admission debt grows at one lane per ten
    // CPUs. Keep two executor lanes beyond that window when configuration
    // permits, then preserve the generic reverse curve. Only the old
    // at-most-two requested maintenance transfers may reduce reverse service.
    if (plan.maintenance_workers < maintenance_target_for(cpu_budget)) {
      ++plan.maintenance_workers;
      continue;
    }
    if (plan.peer_reverse_workers < reverse_target_for(cpu_budget)) {
      ++plan.peer_reverse_workers;
      continue;
    }

    for (std::size_t attempt = 0; attempt < kElasticCycle.size();
         ++attempt) {
      const WorkerRole role =
        kElasticCycle[cycle_index++ % kElasticCycle.size()];
      if (try_grow(role, cpu_budget)) {
        assigned = true;
        break;
      }
    }
  }

  finish_foreground_coordinators();
  return plan;
}

}  // namespace memory_node_detail
