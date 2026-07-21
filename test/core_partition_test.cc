#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <vector>

#include "common/core_partition.hh"
#include "memory_node/storage_owner_cpu_plan.hh"

int main() {
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
  assert(colocated.maintenance_workers == 2);
  assert(colocated.peer_reverse_workers == 4);
  assert(colocated.peer_stage1_workers == 7);
  assert(colocated.peer_cleanup_workers == 2);
  assert(colocated.peer_placement_workers == 1);
  assert(colocated.peer_progress_threads == 2);
  assert(colocated.foreground_progress_threads == 1);
  assert(colocated.foreground_workers + colocated.maintenance_workers +
           colocated.peer_reverse_workers + colocated.peer_stage1_workers +
           colocated.peer_cleanup_workers +
           colocated.peer_placement_workers +
           colocated.peer_progress_threads +
           colocated.foreground_progress_threads == 22);

  const auto dedicated = memory_node_detail::derive_storage_owner_cpu_plan(
    112, 64, 16, 8, 4);
  assert(dedicated.foreground_workers == 16);
  assert(dedicated.maintenance_workers == 8);
  assert(dedicated.peer_reverse_workers == 8);
  assert(dedicated.peer_stage1_workers == 32);
  assert(dedicated.peer_cleanup_workers == 8);
  assert(dedicated.peer_placement_workers == 1);
  assert(dedicated.foreground_progress_threads == 1);

  // Stage1 chooses one physical home. Adding shards does not fan out its
  // coordinator work and must not reduce foreground concurrency.
  const auto one_remote = memory_node_detail::derive_storage_owner_cpu_plan(
    22, 64, 16, 8, 1);
  assert(one_remote.foreground_workers == colocated.foreground_workers);
  assert(one_remote.peer_stage1_workers == colocated.peer_stage1_workers);

  const auto local_only = memory_node_detail::derive_storage_owner_cpu_plan(
    22, 64, 16, 8, 0);
  assert(local_only.foreground_workers == 16);
  assert(local_only.peer_stage1_workers == 0);
  assert(local_only.peer_cleanup_workers == 0);
  assert(local_only.peer_placement_workers == 0);
}
