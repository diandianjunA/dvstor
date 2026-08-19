#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "service/storage_owner_protocol.hh"

namespace memory_node_storage_owner_runtime_detail {

// Observable and lifecycle contract for the unoptimized coupled baseline.
// Keep this pure so configuration/unit tests can prove that selecting the
// baseline cannot accidentally turn into a Stage2-drain variant.
struct ExactUpdateContract {
  bool append_only{};
  bool supports_upsert{};
  bool supports_erase{};
  bool stage2_enabled{};
  bool migration_enabled{};
  bool publishes_maintenance_debt{};
  bool stage1_peer_artifacts_enabled{};
  bool cleanup_peer_artifacts_enabled{};
  bool migration_allocation_receipts_enabled{};
  bool stage2_home_outbox_enabled{};
  std::uint64_t public_maintenance_sequence{};
};

inline constexpr ExactUpdateContract kExactUpdateContract{
  .append_only = true,
  .supports_upsert = false,
  .supports_erase = false,
  .stage2_enabled = false,
  .migration_enabled = false,
  .publishes_maintenance_debt = false,
  .stage1_peer_artifacts_enabled = false,
  .cleanup_peer_artifacts_enabled = false,
  .migration_allocation_receipts_enabled = false,
  .stage2_home_outbox_enabled = false,
  .public_maintenance_sequence = 0,
};

// The strict coupled baseline is deliberately append-only. A fresh insert is
// fully coordinated by its logical authority: allocation and centroid
// membership are local, while cross-shard search and backlinks use one-sided
// RDMA. Generic upsert/erase can name an old record on another physical shard;
// its authoritative centroid accumulator and reclaim queue are host-only
// objects on that shard, so accepting either operation would silently re-add
// a target-side CPU execution domain. Reject them before taking an authority
// lease instead of weakening the advertised single-owner contract.
inline constexpr bool exact_mutation_kind_allowed(
    service::storage_owner::MutationKind kind) {
  return service::storage_owner::mutation_supported_by_completion_mode(
    true, kind);
}

inline constexpr bool exact_peer_request_allowed(
    service::storage_owner::PeerRpcType) {
  return false;
}

inline constexpr bool exact_peer_response_allowed(
    service::storage_owner::PeerRpcType) {
  return false;
}

inline constexpr bool exact_dynamic_control_action_allowed(
    service::storage_owner::DynamicNodeControlAction) {
  return false;
}

// Retry loops must not expose every attempted parent as a graph invalidation.
// Record only a completed reconciliation target, and keep duplicate successful
// retries bounded to one cache invalidation.
inline bool record_exact_completed_invalidation(
    std::vector<std::uint64_t>* invalidations,
    std::uint64_t target_raw) {
  if (invalidations == nullptr || target_raw == 0) return false;
  if (std::find(invalidations->begin(), invalidations->end(), target_raw) !=
      invalidations->end()) {
    return false;
  }
  invalidations->push_back(target_raw);
  return true;
}

// Graph-reconcile and local-centroid records historically call this value a
// maintenance/placement sequence, but the synchronous path needs only a
// non-zero, deterministic retry correlation value. It is not a reclaim epoch
// or public watermark.
inline constexpr std::uint64_t exact_update_mutation_cookie(
    std::uint32_t source_client,
    std::uint64_t operation_id,
    std::uint32_t vector_id,
    std::uint32_t generation) {
  std::uint64_t value = operation_id ^
    (static_cast<std::uint64_t>(source_client) << 32) ^
    static_cast<std::uint64_t>(vector_id) ^
    (static_cast<std::uint64_t>(generation) << 17);
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  value ^= value >> 31;
  return value == 0 ? 1 : value;
}

}  // namespace memory_node_storage_owner_runtime_detail
