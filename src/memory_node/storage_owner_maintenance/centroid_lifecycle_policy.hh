#pragma once

namespace memory_node_storage_owner_maintenance_detail {

enum class CentroidRemoveIdentityDecision {
  apply_exact,
  already_absent,
  retry_inconsistent,
  reject_current_identity,
};

// A delayed remove names one exact physical incarnation.  Once both stored
// incarnation tags agree on a different occupant, its postcondition already
// holds and the new vector/count must not be touched.  A torn pair remains
// retryable, while an ID/generation mismatch within the requested incarnation
// is a genuine identity error rather than an ABA success.
inline CentroidRemoveIdentityDecision classify_centroid_remove_identity(
    unsigned requested_incarnation,
    unsigned header_incarnation,
    unsigned slot_incarnation,
    bool id_and_generation_match) {
  const bool header_exact = header_incarnation == requested_incarnation;
  const bool slot_exact = slot_incarnation == requested_incarnation;
  if (header_exact != slot_exact) {
    return CentroidRemoveIdentityDecision::retry_inconsistent;
  }
  if (!header_exact) {
    return header_incarnation == slot_incarnation
      ? CentroidRemoveIdentityDecision::already_absent
      : CentroidRemoveIdentityDecision::retry_inconsistent;
  }
  return id_and_generation_match
    ? CentroidRemoveIdentityDecision::apply_exact
    : CentroidRemoveIdentityDecision::reject_current_identity;
}

// These are deliberately small safety predicates rather than timing policy.
// Callers may retry or batch each preceding operation arbitrarily, but a
// tombstone is never legal while the route still accounts the same physical
// identity.
inline bool cleanup_tombstone_allowed(bool authority_retired,
                                      bool retiring,
                                      bool centroid_withdrawn) {
  return authority_retired && retiring && centroid_withdrawn;
}

// Stage1 records are intentionally absent from the exact physical centroid
// until their final placement is authority-visible. A migrated source may be
// tombstoned only after the final identity has been counted and published.
inline bool migrated_source_tombstone_allowed(bool placement_committed,
                                               bool final_accounted) {
  return placement_committed && final_accounted;
}

}  // namespace memory_node_storage_owner_maintenance_detail
