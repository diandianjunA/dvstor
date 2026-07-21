#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"

#include <algorithm>
#include <cstring>

using namespace memory_node_storage_owner_index_detail;

namespace {

bool same_reconcile_neighbors(const vec<RemotePtr>& lhs,
                              const vec<RemotePtr>& rhs) {
  return lhs.size() == rhs.size() &&
    std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

}  // namespace

bool MemoryNode::reconcile_local_reverse_ops(
    const span<const service::storage_owner::ReconcileReverseOp> ops,
    const Configuration& config,
    vec<service::storage_owner::ReconcileReverseResult>& results) {
  using service::storage_owner::ReconcileReverseOpKind;
  using service::storage_owner::ReconcileReverseResult;

  results.assign(ops.size(), {});
  bool structurally_valid = true;
  for (size_t op_index = 0; op_index < ops.size(); ++op_index) {
    const auto& op = ops[op_index];
    ReconcileReverseResult& result = results[op_index];
    result.placement_sequence = op.placement_sequence;

    const RemotePtr target{op.target_raw};
    if (!valid_local_storage_node_pointer(target)) {
      result.stale = 1;
      structurally_valid = false;
      continue;
    }

    const auto pointer_sane = [&](const RemotePtr candidate) {
      if (candidate.is_null() ||
          candidate.memory_node() >= num_storage_nodes_ ||
          !VamanaNode::hot_graph_entry_available(candidate)) {
        return false;
      }
      if (local_shard(candidate.memory_node()) &&
          !valid_local_storage_node_pointer(candidate)) {
        return false;
      }
      const auto vector = vamana::StorageLayoutResolver::vector(candidate);
      return vector.offset <= mn_memory_bytes_ &&
        vector.size <= mn_memory_bytes_ - vector.offset;
    };

    const IncarnationLockResult target_lock = try_lock_node(target);
    if (target_lock == IncarnationLockResult::stale) {
      // The operation names a retired physical identity. Report the existing
      // per-op stale outcome; never acquire or mutate its replacement.
      result.stale = 1;
      continue;
    }
    if (target_lock == IncarnationLockResult::busy) {
      // Preserve transient contention as an RPC retry instead of pretending
      // that the reconciliation postcondition was reached.
      structurally_valid = false;
      continue;
    }
    const u64 target_header = load_local_node_header_acquire(target);
    const bool target_stable =
      VamanaNode::stable_graph_mutation_allowed(target_header);
    GraphAdjacency adjacency;
    if (!read_graph_adjacency(target, adjacency)) {
      result.stale = 1;
      unlock_node(target);
      continue;
    }
    const vec<RemotePtr> before_stable = adjacency.stable;
    const vec<RemotePtr> before_provisional = adjacency.provisional;

    const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
    if (kind == ReconcileReverseOpKind::ensure_reachable) {
      // Protected slots are bounded long-lived structural state. Reclaim
      // deleted/reused tagged children lazily at the next reservation so a
      // stale protected pointer cannot permanently consume capacity under a
      // high-frequency update workload. Live provisional Stage1 children are
      // intentionally retained.
      adjacency.provisional.erase(
        std::remove_if(
          adjacency.provisional.begin(), adjacency.provisional.end(),
          [this](RemotePtr child) {
            NodeSnapshot child_snapshot;
            if (read_node_snapshot(child, child_snapshot)) {
              return child_snapshot.deleted;
            }
            // A locked same-incarnation child is uncertain and must be
            // retained. A tag mismatch, in contrast, proves that this edge
            // names a reclaimed slot and is safe to drop.
            u64 header = 0;
            if (local_shard(child.memory_node())) {
              if (!valid_local_storage_node_pointer(child)) return true;
              header = load_local_node_header_acquire(child);
            } else if (child.memory_node() < num_storage_nodes_) {
              const auto address =
                vamana::StorageLayoutResolver::header(child);
              if (address.offset > mn_memory_bytes_ ||
                  sizeof(header) > mn_memory_bytes_ - address.offset) {
                return true;
              }
              remote_read_bytes(child.memory_node(), address.offset,
                                &header, sizeof(header), 0);
            } else {
              return true;
            }
            return VamanaNode::header_incarnation(header) !=
              child.incarnation();
          }),
        adjacency.provisional.end());
    }
    const RemotePtr old_candidate{op.old_candidate_raw};
    const RemotePtr new_candidate{op.new_candidate_raw};
    const bool old_present = !old_candidate.is_null() &&
      (reconcile_contains(adjacency.stable, old_candidate) ||
       reconcile_contains(adjacency.provisional, old_candidate));
    const bool needs_new = reconcile_kind_needs_new_identity(kind);

    NodeSnapshot old_snapshot;
    NodeSnapshot new_snapshot;
    bool old_identity_matches = !old_present;
    if (old_present && pointer_sane(old_candidate) &&
        read_node_snapshot(old_candidate, old_snapshot)) {
      old_identity_matches =
        old_snapshot.id == op.id &&
        old_snapshot.generation == op.generation;
    }

    bool new_identity_live = !needs_new;
    if (needs_new && pointer_sane(new_candidate) &&
        read_node_snapshot(new_candidate, new_snapshot)) {
      new_identity_live =
        !new_snapshot.deleted &&
        (kind == ReconcileReverseOpKind::ensure_reachable ||
         (new_snapshot.header & VamanaNode::HEADER_PROVISIONAL) == 0) &&
        new_snapshot.id == op.id &&
        new_snapshot.generation == op.generation;
    }

    const bool replacement_equivalent =
      old_present && old_identity_matches && new_identity_live &&
      old_snapshot.vector_data.size() == new_snapshot.vector_data.size() &&
      old_snapshot.vector_data.size() >= VamanaNode::vector_bytes() &&
      std::memcmp(old_snapshot.vector_data.data(),
                  new_snapshot.vector_data.data(),
                  VamanaNode::vector_bytes()) == 0;

    const auto robust_prune = [&](const vec<RemotePtr>& candidates) {
      vec<NodeSnapshot> snapshots =
        read_node_snapshots_batched(candidates, config);
      hashset_t<RemotePtr> skip;
      skip.insert(target);
      const auto target_vector =
        vamana::StorageLayoutResolver::vector(target);
      lib_assert(target_vector.offset <= mn_memory_bytes_ &&
                   target_vector.size <= mn_memory_bytes_ -
                     target_vector.offset,
                 "reconcile reverse target vector exceeds shard bounds");
      return robust_prune_snapshots_cpu(
        index_buffer_.get_full_buffer() + target_vector.offset,
        VamanaNode::vector_dtype(),
        span<const NodeSnapshot>{snapshots.data(), snapshots.size()},
        skip, config, config.R);
    };

    result = reconcile_reverse_adjacency(
      op, target_stable, old_identity_matches, new_identity_live,
      replacement_equivalent, config.R, VamanaNode::provisional_slots(),
      adjacency.stable,
      adjacency.provisional, robust_prune);
    const bool changed =
      !same_reconcile_neighbors(before_stable, adjacency.stable) ||
      !same_reconcile_neighbors(before_provisional,
                                adjacency.provisional);
    if (!result.stale && changed) {
      write_graph_adjacency(target, adjacency.stable,
                            adjacency.provisional,
                            adjacency.generation, false);
    }
    unlock_node(target);
  }
  return structurally_valid;
}
