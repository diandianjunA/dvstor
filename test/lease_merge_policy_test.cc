#include <array>
#include <cassert>
#include <cstdint>

#include "memory_node/storage_owner_index/atomic_cut_policy.hh"
#include "memory_node/storage_owner_maintenance/certificate_lease.hh"
#include "memory_node/storage_owner_maintenance/dependency_scheduler.hh"

namespace {

using memory_node_storage_owner_maintenance_detail::CertificateAcquireResult;
using memory_node_storage_owner_maintenance_detail::CertificateLeaseTable;
using memory_node_storage_owner_maintenance_detail::MaintenanceDependencyTracker;

void test_lease_lifecycle_capacity_and_replay() {
  CertificateLeaseTable leases(2);
  assert(leases.acquire(10, 100, 1) == CertificateAcquireResult::acquired);
  assert(leases.acquire(10, 100, 1) == CertificateAcquireResult::replay);
  assert(leases.acquire(10, 100, 2) == CertificateAcquireResult::conflict);
  assert(leases.contains(10, 100, 1));
  assert(leases.active_at(10) == 1);

  assert(leases.acquire(10, 101, 2) == CertificateAcquireResult::acquired);
  assert(leases.acquire(10, 102, 3) ==
         CertificateAcquireResult::capacity_exhausted);
  assert(leases.acquire(11, 102, 3) == CertificateAcquireResult::acquired);
  assert(leases.size() == 3);
  assert(leases.leases_at(10).size() == 2);
  assert(leases.commit(20, {{200, 4}}, {}));
  assert(leases.size() == 4);
  assert(leases.commit(20, {{200, 4}}, {}));
  assert(leases.size() == 4);
  assert(leases.commit(20, {}, {{200, 4}}));
  assert(leases.size() == 3);
  assert(leases.commit(20, {{201, 5}}, {{200, 9}}));
  assert(leases.contains(20, 201, 5));
  assert(leases.size() == 4);

  assert(!leases.release(10, 100, 9));
  assert(leases.contains(10, 100, 1));
  assert(leases.release(10, 100, 1));
  assert(leases.release(10, 100, 1));
  assert(!leases.contains(10, 100));
  assert(leases.acquire(10, 102, 3) == CertificateAcquireResult::acquired);
}

void test_cleanup_depends_only_on_matching_incarnation() {
  MaintenanceDependencyTracker dependencies;
  assert(dependencies.register_finalize(7, 42, 3, 1000));
  assert(!dependencies.register_finalize(8, 42, 3, 1000));
  assert(!dependencies.cleanup_ready(42, 3, 1000));
  assert(dependencies.cleanup_ready(42, 4, 1000));
  assert(dependencies.cleanup_ready(42, 3, 1001));
  assert(dependencies.cleanup_ready(43, 3, 1000));
  assert(dependencies.size() == 1);

  assert(dependencies.complete_finalize(7));
  assert(!dependencies.complete_finalize(7));
  assert(dependencies.cleanup_ready(42, 3, 1000));
  assert(dependencies.size() == 0);
}

void test_atomic_cut_pins_leases_and_commits_certificate_sets() {
  using service::storage_owner::ReconcileReverseOp;
  using service::storage_owner::ReconcileReverseOpKind;
  const auto op = [](ReconcileReverseOpKind kind, std::uint64_t old_raw,
                     std::uint64_t new_raw, std::uint64_t sequence) {
    return ReconcileReverseOp{
      .target_raw = 10,
      .old_candidate_raw = old_raw,
      .new_candidate_raw = new_raw,
      .placement_sequence = sequence,
      .id = 42,
      .generation = 7,
      .kind = static_cast<std::uint32_t>(kind),
    };
  };
  const RemotePtr existing{1, 0x1000};
  const RemotePtr ordinary{1, 0x2000};
  const RemotePtr leased{1, 0x3000};
  const RemotePtr acquiring{1, 0x4000};
  const vec<ReconcileReverseOp> ops{
    op(ReconcileReverseOpKind::add, 0, ordinary.raw_address, 9),
    op(ReconcileReverseOpKind::acquire_certificate_lease, 0,
       acquiring.raw_address, 2),
    op(ReconcileReverseOpKind::remove_provisional_if_present,
       leased.raw_address, 0, 1),
  };
  const vec<RemotePtr> stable{existing};
  const vec<RemotePtr> provisional{leased, acquiring};
  const vec<std::pair<RemotePtr, std::uint64_t>> active{
    std::pair{leased, std::uint64_t{1}}};
  std::uint32_t prune_calls = 0;
  auto prune = [&](const vec<RemotePtr>&) {
    ++prune_calls;
    return vec<RemotePtr>{existing, ordinary};
  };
  const auto plan = memory_node_storage_owner_index_detail::plan_atomic_cut(
    span<const ReconcileReverseOp>{ops}, span<const RemotePtr>{stable},
    span<const RemotePtr>{provisional},
    span<const std::pair<RemotePtr, std::uint64_t>>{active}, u32{2}, prune);
  assert(plan.valid);
  assert(prune_calls == 1);
  assert(plan.stable.size() == 2);
  assert(std::find(plan.stable.begin(), plan.stable.end(), leased) !=
         plan.stable.end());
  assert(std::find(plan.stable.begin(), plan.stable.end(), acquiring) !=
         plan.stable.end());
  assert(plan.provisional.empty());
  assert(plan.acquired.size() == 1);
  assert(plan.released.empty());
  assert(plan.results[1].accepted == 1);
  assert(plan.results[2].removed == 1);

  const vec<ReconcileReverseOp> release{
    op(ReconcileReverseOpKind::release_certificate_lease, 0,
       leased.raw_address, 1)};
  const auto released = memory_node_storage_owner_index_detail::plan_atomic_cut(
    span<const ReconcileReverseOp>{release},
    span<const RemotePtr>{plan.stable},
    span<const RemotePtr>{plan.provisional},
    span<const std::pair<RemotePtr, std::uint64_t>>{active}, 2,
    [&](const vec<RemotePtr>& candidates) { return candidates; });
  assert(released.valid);
  assert(released.released.size() == 1);
  assert(released.results[0].accepted == 1);
}

}  // namespace

int main() {
  test_lease_lifecycle_capacity_and_replay();
  test_cleanup_depends_only_on_matching_incarnation();
  test_atomic_cut_pins_leases_and_commits_certificate_sets();
}
