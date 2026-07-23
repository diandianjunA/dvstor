#include <cassert>

#include "memory_node/storage_owner_maintenance/centroid_lifecycle_policy.hh"

namespace detail = memory_node_storage_owner_maintenance_detail;

int main() {
  using Decision = detail::CentroidRemoveIdentityDecision;
  assert(detail::classify_centroid_remove_identity(
           7, 7, 7, true) == Decision::apply_exact);
  // A delayed remove for incarnation 7 observes a fully published occupant 8:
  // the old membership postcondition holds, and occupant 8 must not be debited.
  assert(detail::classify_centroid_remove_identity(
           7, 8, 8, false) == Decision::already_absent);
  // A torn header/slot pair is not enough evidence to declare ABA completion.
  assert(detail::classify_centroid_remove_identity(
           7, 7, 8, true) == Decision::retry_inconsistent);
  assert(detail::classify_centroid_remove_identity(
           7, 8, 7, true) == Decision::retry_inconsistent);
  // The requested incarnation still exists, but its logical identity differs:
  // this is malformed/current-identity failure, not a successful stale remove.
  assert(detail::classify_centroid_remove_identity(
           7, 7, 7, false) == Decision::reject_current_identity);

  assert(!detail::cleanup_tombstone_allowed(false, true, true));
  assert(!detail::cleanup_tombstone_allowed(true, false, true));
  assert(!detail::cleanup_tombstone_allowed(true, true, false));
  assert(detail::cleanup_tombstone_allowed(true, true, true));

  assert(!detail::migrated_source_tombstone_allowed(false, false));
  assert(!detail::migrated_source_tombstone_allowed(true, false));
  assert(!detail::migrated_source_tombstone_allowed(false, true));
  assert(detail::migrated_source_tombstone_allowed(true, true));
  return 0;
}
