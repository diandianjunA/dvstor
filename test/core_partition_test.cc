#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <vector>

#include "common/core_partition.hh"
#include "memory_node/storage_owner_cpu_plan.hh"

int main() {
  const auto pinned_cpu_lanes = [](const auto& plan) {
    return static_cast<std::uint64_t>(plan.foreground_workers) +
      plan.maintenance_workers + plan.peer_stage1_workers +
      plan.peer_reverse_workers + plan.peer_cleanup_workers +
      plan.peer_placement_workers + plan.peer_progress_threads +
      plan.foreground_progress_threads;
  };

  for (std::uint32_t total = 0; total <= 128; ++total) {
    const auto split =
      memory_node_detail::split_physical_home_workers(total);
    assert(split.stage1 + split.stage2_home == total);
    if (total == 0) assert(split.stage1 == 0 && split.stage2_home == 0);
    if (total == 1) assert(split.stage1 == 1 && split.stage2_home == 0);
    if (total >= 2) {
      assert(split.stage1 >= 1);
      assert(split.stage2_home >= 1);
    }
    if (total >= 2) {
      assert(split.stage2_home == 1);
    }
  }

  // Strict order for four physical cores followed by their four SMT siblings.
  const std::vector<std::uint32_t> ordered{4, 5, 0, 1, 12, 13, 8, 9};
  std::set<std::uint32_t> union_cpus;
  for (std::uint32_t rank = 0; rank < 3; ++rank) {
    const auto part = core_assignment_detail::partition_ordered_cores(
      ordered, true, rank, 3);
    assert(!part.empty());
    assert(part.size() % 2 == 0);
    const std::size_t physical = part.size() / 2;
    for (std::size_t i = 0; i < physical; ++i) {
      // The test topology's sibling IDs differ by eight.
      assert(part[physical + i] == part[i] + 8);
    }
    for (const auto cpu : part) {
      assert(union_cpus.insert(cpu).second);
    }
  }
  assert(union_cpus == std::set<std::uint32_t>(ordered.begin(), ordered.end()));

  const std::vector<std::uint32_t> no_smt{7, 3, 9, 1, 5};
  const auto first = core_assignment_detail::partition_ordered_cores(
    no_smt, false, 0, 2);
  const auto second = core_assignment_detail::partition_ordered_cores(
    no_smt, false, 1, 2);
  assert(first == std::vector<std::uint32_t>({7, 3, 9}));
  assert(second == std::vector<std::uint32_t>({1, 5}));

  const auto colocated = memory_node_detail::derive_storage_owner_cpu_plan(
    22, 64, 16, 8, 4);
  assert(colocated.foreground_workers == 3);
  assert(colocated.foreground_coordinators == 12);
  assert(colocated.maintenance_workers == 8);
  assert(colocated.maintenance_admission_workers == 2);
  assert(colocated.peer_reverse_workers == 2);
  assert(colocated.peer_stage1_workers == 4);
  assert(colocated.peer_cleanup_workers == 1);
  assert(colocated.peer_placement_workers == 1);
  assert(colocated.peer_progress_threads == 2);
  assert(colocated.foreground_progress_threads == 1);
  assert(colocated.foreground_workers + colocated.maintenance_workers +
           colocated.peer_reverse_workers + colocated.peer_stage1_workers +
           colocated.peer_cleanup_workers +
           colocated.peer_placement_workers +
           colocated.peer_progress_threads +
           colocated.foreground_progress_threads == 22);

  // The five-way colocated deployment gives some shard ranks 24 logical
  // CPUs. The latency-bearing Stage2 pool stays at eight while the two extra
  // CPUs preserve one additional physical-home and cleanup lane.
  const auto colocated_24 = memory_node_detail::derive_storage_owner_cpu_plan(
    24, 64, 16, 8, 4);
  assert(colocated_24.foreground_workers == 3);
  assert(colocated_24.foreground_coordinators == 12);
  assert(colocated_24.maintenance_workers == 8);
  assert(colocated_24.maintenance_admission_workers == 2);
  assert(colocated_24.peer_reverse_workers == 2);
  assert(colocated_24.peer_stage1_workers == 5);
  assert(colocated_24.peer_cleanup_workers == 2);
  assert(colocated_24.peer_placement_workers == 1);
  assert(colocated_24.peer_progress_threads == 2);
  assert(colocated_24.foreground_progress_threads == 1);
  assert(colocated_24.foreground_workers +
           colocated_24.maintenance_workers +
           colocated_24.peer_reverse_workers +
           colocated_24.peer_stage1_workers +
           colocated_24.peer_cleanup_workers +
           colocated_24.peer_placement_workers +
           colocated_24.peer_progress_threads +
           colocated_24.foreground_progress_threads == 24);

  // Configuration remains authoritative below the colocated eight-worker
  // profile, and preserves the combined Stage2/reverse CPU pool.
  const auto maintenance_two =
    memory_node_detail::derive_storage_owner_cpu_plan(22, 64, 16, 2, 4);
  const auto maintenance_three =
    memory_node_detail::derive_storage_owner_cpu_plan(22, 64, 16, 3, 4);
  assert(maintenance_two.maintenance_workers == 2);
  assert(maintenance_two.maintenance_admission_workers == 2);
  assert(maintenance_two.peer_reverse_workers == 4);
  assert(maintenance_three.maintenance_workers == 3);
  assert(maintenance_three.maintenance_admission_workers == 2);
  assert(maintenance_three.peer_reverse_workers == 3);
  assert(maintenance_two.maintenance_workers +
           maintenance_two.peer_reverse_workers == 6);
  assert(maintenance_three.maintenance_workers +
           maintenance_three.peer_reverse_workers == 6);
  assert(colocated.maintenance_workers + colocated.peer_reverse_workers == 10);

  const auto dedicated = memory_node_detail::derive_storage_owner_cpu_plan(
    112, 64, 16, 8, 4);
  assert(dedicated.foreground_workers == 16);
  assert(dedicated.foreground_coordinators == 16);
  assert(dedicated.maintenance_workers == 8);
  assert(dedicated.peer_reverse_workers == 8);
  assert(dedicated.peer_stage1_workers == 32);
  assert(dedicated.peer_cleanup_workers == 8);
  assert(dedicated.peer_placement_workers == 1);
  assert(dedicated.foreground_progress_threads == 1);

  // A high operator ceiling on a larger host must not be interpreted as
  // permission to collapse the incoming reverse-update pool.  Only the same
  // two-lane conservative transfer is allowed; admission remains tied to the
  // pre-transfer count.  The common 112-CPU/configured-8 plan above remains
  // unchanged at 8/8 because it already reaches its configured Stage2 bound.
  const auto large_high_maintenance =
    memory_node_detail::derive_storage_owner_cpu_plan(
      64, 128, 64, 64, 4);
  assert(large_high_maintenance.maintenance_admission_workers == 6);
  assert(large_high_maintenance.maintenance_workers == 8);
  assert(large_high_maintenance.peer_reverse_workers == 6);
  const auto dedicated_high_maintenance =
    memory_node_detail::derive_storage_owner_cpu_plan(
      112, 128, 64, 64, 4);
  assert(dedicated_high_maintenance.maintenance_admission_workers == 11);
  assert(dedicated_high_maintenance.maintenance_workers == 13);
  assert(dedicated_high_maintenance.peer_reverse_workers == 6);

  // Stage1 chooses one physical home. Adding shards does not fan out its
  // coordinator work and must not reduce foreground concurrency.
  const auto one_remote = memory_node_detail::derive_storage_owner_cpu_plan(
    22, 64, 16, 8, 1);
  assert(one_remote.foreground_workers == colocated.foreground_workers);
  assert(one_remote.foreground_coordinators ==
         colocated.foreground_coordinators);
  assert(one_remote.peer_stage1_workers == colocated.peer_stage1_workers);

  const auto local_only = memory_node_detail::derive_storage_owner_cpu_plan(
    22, 64, 16, 8, 0);
  assert(local_only.foreground_workers == 16);
  assert(local_only.foreground_coordinators ==
         local_only.foreground_workers);
  assert(local_only.maintenance_workers == 2);
  assert(local_only.maintenance_admission_workers == 2);
  assert(local_only.peer_stage1_workers == 0);
  assert(local_only.peer_cleanup_workers == 0);
  assert(local_only.peer_placement_workers == 0);

  // Compute-node scale changes the registered RPC window, not the number of
  // CPU lanes or an unbounded thread pool. Waiting coordinators are capped at
  // four per foreground CPU and never exceed the configured thread ceiling.
  const auto many_clients = memory_node_detail::derive_storage_owner_cpu_plan(
    22, 64, 4096, 8, 4);
  assert(many_clients.foreground_workers == 3);
  assert(many_clients.foreground_coordinators == 12);
  assert(many_clients.foreground_coordinators <=
         many_clients.foreground_workers * 4);

  const auto shallow_rpc = memory_node_detail::derive_storage_owner_cpu_plan(
    22, 64, 2, 8, 4);
  assert(shallow_rpc.foreground_coordinators <= 2);
  assert(shallow_rpc.foreground_coordinators >=
         shallow_rpc.foreground_workers);

  // Exercise the policy, not only the production 22/24-CPU points.  Nine
  // CPUs are the functional minimum for a remote plan's independently pinned
  // progress, maintenance, reverse, control, and two Stage1 lanes.  At and
  // above that minimum the plan must never invent CPUs, regardless of shard
  // count or operator concurrency bounds.
  for (std::uint32_t cpus = 9; cpus <= 160; ++cpus) {
    for (const std::uint32_t configured_threads : {1u, 8u, 64u, 256u}) {
      for (const std::uint32_t rpc_parallelism : {1u, 2u, 16u, 4096u}) {
        for (const std::uint32_t configured_maintenance :
             {1u, 2u, 3u, 8u, 64u}) {
          const auto one_peer =
            memory_node_detail::derive_storage_owner_cpu_plan(
              cpus, configured_threads, rpc_parallelism,
              configured_maintenance, 1);
          const auto many_peers =
            memory_node_detail::derive_storage_owner_cpu_plan(
              cpus, configured_threads, rpc_parallelism,
              configured_maintenance, 64);
          assert(pinned_cpu_lanes(one_peer) <= cpus);
          assert(one_peer.maintenance_admission_workers >= 1);
          assert(one_peer.maintenance_admission_workers <=
                 one_peer.maintenance_workers);
          assert(one_peer.maintenance_workers <= configured_maintenance);
          assert(one_peer.peer_reverse_workers >= 1);
          assert(one_peer.peer_reverse_workers <= 8);
          assert(one_peer.foreground_workers >= 1);
          assert(one_peer.peer_stage1_workers >= 1);
          assert(one_peer.foreground_coordinators >=
                 one_peer.foreground_workers);
          assert(one_peer.foreground_coordinators <=
                 one_peer.foreground_workers * 4);

          // Worker pools service all peer queues; adding shards must not
          // divide the fixed CPU partition or grow it per peer.
          assert(one_peer.foreground_workers ==
                 many_peers.foreground_workers);
          assert(one_peer.foreground_coordinators ==
                 many_peers.foreground_coordinators);
          assert(one_peer.maintenance_workers ==
                 many_peers.maintenance_workers);
          assert(one_peer.maintenance_admission_workers ==
                 many_peers.maintenance_admission_workers);
          assert(one_peer.peer_stage1_workers ==
                 many_peers.peer_stage1_workers);
          assert(one_peer.peer_reverse_workers ==
                 many_peers.peer_reverse_workers);
          assert(pinned_cpu_lanes(one_peer) ==
                 pinned_cpu_lanes(many_peers));
        }
      }
    }
  }

  // Local-only plans have no reverse/control pool to donate.  Their actual
  // and admission Stage2 counts therefore remain identical, and a viable
  // three-CPU plan also stays inside its physical budget.
  for (std::uint32_t cpus = 3; cpus <= 160; ++cpus) {
    for (const std::uint32_t configured_maintenance : {1u, 2u, 8u, 64u}) {
      const auto plan = memory_node_detail::derive_storage_owner_cpu_plan(
        cpus, 64, 16, configured_maintenance, 0);
      assert(pinned_cpu_lanes(plan) <= cpus);
      assert(plan.maintenance_workers ==
             plan.maintenance_admission_workers);
      assert(plan.peer_stage1_workers == 0);
      assert(plan.peer_reverse_workers == 0);
      assert(plan.peer_cleanup_workers == 0);
      assert(plan.peer_placement_workers == 0);
      assert(plan.peer_progress_threads == 0);
    }
  }
}
