#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <vector>

#include "common/core_partition.hh"
#include "memory_node/storage_owner_cpu_plan.hh"
#include "memory_node/storage_owner_runtime/exact_update_contract.hh"

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

  // The generic reverse curve continues smoothly after the production point;
  // the 24-CPU profile remains unchanged while 25/32/64 CPUs retain the
  // reverse capacity expected by reverse-heavy update/delete workloads.
  const auto portable_25 = memory_node_detail::derive_storage_owner_cpu_plan(
    25, 64, 16, 8, 4);
  const auto portable_32 = memory_node_detail::derive_storage_owner_cpu_plan(
    32, 64, 16, 8, 4);
  const auto portable_64 = memory_node_detail::derive_storage_owner_cpu_plan(
    64, 64, 16, 8, 4);
  assert(portable_25.maintenance_workers == 8);
  assert(portable_25.peer_reverse_workers >= 3);
  assert(portable_32.maintenance_workers == 8);
  assert(portable_32.peer_reverse_workers >= 4);
  assert(portable_64.peer_reverse_workers == 6);

  // Configuration remains authoritative below the full eight-worker Stage2
  // service profile. CPUs not requested by Stage2 stay with the generic
  // reverse-service curve, preserving reverse-heavy update/delete capacity.
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

  // CPU portability is a policy invariant, not a collection of host-size
  // exceptions. For a fixed operator configuration, adding one CPU may grow
  // or saturate each independently pinned service domain, but must never take
  // a worker away from one. Exercise the full range across shallow/deep RPC
  // windows and small/large configured thread ceilings.
  for (const std::uint32_t configured_threads : {1u, 8u, 64u, 256u}) {
    for (const std::uint32_t rpc_parallelism : {1u, 2u, 16u, 4096u}) {
      for (const std::uint32_t configured_maintenance :
           {1u, 2u, 3u, 8u, 64u}) {
        auto previous = memory_node_detail::derive_storage_owner_cpu_plan(
          9, configured_threads, rpc_parallelism,
          configured_maintenance, 4);
        for (std::uint32_t cpus = 10; cpus <= 160; ++cpus) {
          const auto current =
            memory_node_detail::derive_storage_owner_cpu_plan(
              cpus, configured_threads, rpc_parallelism,
              configured_maintenance, 4);
          assert(current.maintenance_workers >=
                 previous.maintenance_workers);
          assert(current.maintenance_admission_workers >=
                 previous.maintenance_admission_workers);
          assert(current.peer_reverse_workers >=
                 previous.peer_reverse_workers);
          assert(current.peer_cleanup_workers >=
                 previous.peer_cleanup_workers);
          assert(current.foreground_workers >=
                 previous.foreground_workers);
          assert(current.foreground_coordinators >=
                 previous.foreground_coordinators);
          assert(current.peer_stage1_workers >=
                 previous.peer_stage1_workers);
          assert(pinned_cpu_lanes(current) >= pinned_cpu_lanes(previous));
          assert(pinned_cpu_lanes(current) - pinned_cpu_lanes(previous) <= 1);
          assert(pinned_cpu_lanes(current) <= cpus);
          previous = current;
        }
      }
    }
  }

  // The production concurrency ceilings can consume every CPU through the
  // colocated and medium-host range; no CPU is hidden by the monotonic
  // allocator before a configured role actually saturates.
  for (std::uint32_t cpus = 9; cpus <= 64; ++cpus) {
    const auto plan = memory_node_detail::derive_storage_owner_cpu_plan(
      cpus, 64, 16, 8, 4);
    assert(pinned_cpu_lanes(plan) == cpus);
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

  // Coupled append-only updates have no target-side RPC, Stage1, Stage2, or
  // cleanup CPU domain. Every diagnostic plan remains in budget; a remote
  // deployment needs only the authority and lifecycle/completion lanes.
  for (const std::uint32_t peers : {0u, 4u}) {
    for (std::uint32_t cpus = 1; cpus <= 64; ++cpus) {
      const auto exact =
        memory_node_detail::derive_storage_owner_exact_cpu_plan(
          cpus, 256, peers);
      assert(exact.foreground_workers >= 1);
      assert(exact.maintenance_workers == 0);
      assert(exact.maintenance_admission_workers == 0);
      assert(exact.peer_stage1_workers == 0);
      assert(exact.peer_reverse_workers == 0);
      assert(exact.peer_cleanup_workers == 0);
      assert(exact.peer_placement_workers == 0);
      assert(exact.peer_progress_threads == 0);
      assert(exact.foreground_workers == exact.foreground_coordinators);
      assert(pinned_cpu_lanes(exact) <= cpus);
      if (peers != 0 && cpus >=
          memory_node_detail::exact_update_remote_cpu_floor()) {
        assert(memory_node_detail::
          exact_update_plan_has_remote_correctness_floor(exact));
      }
      assert(memory_node_detail::peer_runtime_thread_counts_supported(exact));
      const auto peer_threads =
        memory_node_detail::derive_peer_runtime_thread_plan(exact);
      assert(peer_threads.cq_progress_threads +
               peer_threads.response_dispatch_threads ==
             exact.peer_progress_threads);
      assert(peer_threads.placement_control_threads ==
             exact.peer_placement_workers);
    }
  }
  const auto local_exact =
    memory_node_detail::derive_storage_owner_exact_cpu_plan(16, 256, 0);
  assert(local_exact.foreground_workers == 15);
  assert(local_exact.peer_reverse_workers == 0);
  assert(local_exact.peer_placement_workers == 0);
  assert(local_exact.peer_progress_threads == 0);
  assert(pinned_cpu_lanes(local_exact) == 16);

  using namespace memory_node_storage_owner_runtime_detail;
  static_assert(kExactUpdateContract.append_only);
  static_assert(!kExactUpdateContract.supports_upsert);
  static_assert(!kExactUpdateContract.supports_erase);
  static_assert(!kExactUpdateContract.stage2_enabled);
  static_assert(!kExactUpdateContract.migration_enabled);
  static_assert(!kExactUpdateContract.publishes_maintenance_debt);
  static_assert(!kExactUpdateContract.stage1_peer_artifacts_enabled);
  static_assert(!kExactUpdateContract.cleanup_peer_artifacts_enabled);
  static_assert(
    !kExactUpdateContract.migration_allocation_receipts_enabled);
  static_assert(!kExactUpdateContract.stage2_home_outbox_enabled);
  static_assert(kExactUpdateContract.public_maintenance_sequence == 0);
  static_assert(exact_update_mutation_cookie(0, 1, 0, 1) != 0);
  assert(exact_update_mutation_cookie(7, 99, 123, 2) ==
         exact_update_mutation_cookie(7, 99, 123, 2));
}
