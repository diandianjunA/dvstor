#include "memory_node/storage_owner_index/detail.hh"

#include "memory_node/storage_owner_index/atomic_cut_policy.hh"
#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

using namespace memory_node_storage_owner_index_detail;

bool MemoryNode::reconcile_lease_merge_ops(
    span<const service::storage_owner::ReconcileReverseOp> ops,
    const Configuration& config,
    vec<service::storage_owner::ReconcileReverseResult>& results) {
  using service::storage_owner::ReconcileReverseOpKind;
  using service::storage_owner::ReconcileReverseResult;

  results.assign(ops.size(), {});
  for (size_t index = 0; index < ops.size(); ++index) {
    results[index].placement_sequence = ops[index].placement_sequence;
    results[index].stale = 1;
  }
  if (ops.empty()) return true;
  if (storage_owner_certificate_leases_ == nullptr) return false;

  dense_hashmap_t<u64, vec<size_t>> grouped;
  grouped.reserve(ops.size());
  vec<u64> target_order;
  target_order.reserve(ops.size());
  for (size_t index = 0; index < ops.size(); ++index) {
    const auto& op = ops[index];
    const RemotePtr target{op.target_raw};
    const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
    const bool supported =
      kind == ReconcileReverseOpKind::add ||
      kind == ReconcileReverseOpKind::remove_if_present ||
      kind == ReconcileReverseOpKind::acquire_certificate_lease ||
      kind == ReconcileReverseOpKind::release_certificate_lease ||
      kind == ReconcileReverseOpKind::remove_provisional_if_present;
    if (!supported || op.reserved != 0 || op.placement_sequence == 0 ||
        !valid_local_storage_node_pointer(target)) {
      return false;
    }
    if (kind == ReconcileReverseOpKind::remove_if_present ||
        kind == ReconcileReverseOpKind::remove_provisional_if_present) {
      if (op.old_candidate_raw == 0) return false;
    } else if (op.new_candidate_raw == 0) {
      return false;
    }
    auto position = grouped.find(target.raw_address);
    if (position == grouped.end()) {
      target_order.push_back(target.raw_address);
      position = grouped.emplace(target.raw_address, vec<size_t>{}).first;
    }
    position->second.push_back(index);
  }

  bool valid = true;
  for (const u64 target_raw : target_order) {
    const RemotePtr target{target_raw};
    const vec<size_t>& indices = grouped.find(target_raw)->second;
    const IncarnationLockResult lock = try_lock_node(target);
    if (lock == IncarnationLockResult::stale) {
      vec<std::pair<u64, u64>> retired_leases;
      for (const auto& [child, sequence] :
           storage_owner_certificate_leases_->leases_at(target.raw_address)) {
        retired_leases.emplace_back(child, sequence);
      }
      (void)storage_owner_certificate_leases_->commit(
        target.raw_address, {}, retired_leases);
      for (const size_t index : indices) {
        results[index] = reconcile_retired_target_result(ops[index]);
      }
      continue;
    }
    if (lock == IncarnationLockResult::busy) {
      valid = false;
      continue;
    }

    u64 target_header = 0;
    node_t target_id = 0;
    u32 target_generation = 0;
    GraphAdjacency adjacency;
    if (!read_locked_node_identity(
          target, target_header, target_id, target_generation) ||
        !read_graph_adjacency(target, adjacency)) {
      unlock_node(target);
      valid = false;
      continue;
    }
    const bool target_stable =
      VamanaNode::stable_graph_mutation_allowed(target_header);
    const bool target_accounted =
      (target_header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0;
    const vec<RemotePtr> before_stable = adjacency.stable;
    const vec<RemotePtr> before_provisional = adjacency.provisional;

    const auto lease_rows =
      storage_owner_certificate_leases_->leases_at(target.raw_address);
    vec<std::pair<RemotePtr, u64>> active;
    active.reserve(lease_rows.size());
    for (const auto& [child_raw, sequence] : lease_rows) {
      active.emplace_back(RemotePtr{child_raw}, sequence);
    }

    bool identities_valid = true;
    const auto pointer_sane = [&](RemotePtr candidate) {
      if (candidate.is_null() || candidate == target ||
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
    for (const auto& [child, sequence] : active) {
      const bool released = std::any_of(
        indices.begin(), indices.end(), [&](size_t index) {
          const auto& op = ops[index];
          return op.kind == static_cast<u32>(
                   ReconcileReverseOpKind::release_certificate_lease) &&
            RemotePtr{op.new_candidate_raw} == child &&
            op.placement_sequence == sequence;
        });
      if (released) continue;
      NodeSnapshot snapshot;
      if (!pointer_sane(child) || !read_node_snapshot(child, snapshot) ||
          snapshot.deleted) {
        identities_valid = false;
        break;
      }
    }
    for (const size_t index : indices) {
      if (!identities_valid) break;
      const auto& op = ops[index];
      const auto kind = static_cast<ReconcileReverseOpKind>(op.kind);
      if (kind != ReconcileReverseOpKind::add &&
          kind != ReconcileReverseOpKind::acquire_certificate_lease) {
        continue;
      }
      const RemotePtr candidate{op.new_candidate_raw};
      NodeSnapshot snapshot;
      if (!pointer_sane(candidate) ||
          !read_node_snapshot(candidate, snapshot) || snapshot.deleted ||
          snapshot.id != op.id || snapshot.generation != op.generation ||
          (kind == ReconcileReverseOpKind::add &&
           (snapshot.header & VamanaNode::HEADER_PROVISIONAL) != 0) ||
          !target_stable ||
          (kind == ReconcileReverseOpKind::acquire_certificate_lease &&
           !target_accounted)) {
        identities_valid = false;
      }
    }
    if (!identities_valid) {
      unlock_node(target);
      valid = false;
      continue;
    }

    const auto target_vector = vamana::StorageLayoutResolver::vector(target);
    const byte_t* target_vector_data =
      index_buffer_.get_full_buffer() + target_vector.offset;
    bool prune_retryable = false;
    auto robust_prune = [&](const vec<RemotePtr>& candidates) {
      vec<StableNodeSnapshotState> states;
      const vec<NodeSnapshot> snapshots = read_node_snapshots_batched(
        candidates, config, "lease_merge_atomic_cut", &states);
      if (std::find(states.begin(), states.end(),
                    StableNodeSnapshotState::retryable) != states.end()) {
        prune_retryable = true;
        return candidates;
      }
      hashset_t<RemotePtr> skip;
      skip.insert(target);
      return robust_prune_snapshots_cpu(
        target_vector_data, VamanaNode::vector_dtype(),
        span<const NodeSnapshot>{snapshots}, skip, config, config.R);
    };
    vec<service::storage_owner::ReconcileReverseOp> target_ops;
    target_ops.reserve(indices.size());
    for (const size_t index : indices) target_ops.push_back(ops[index]);
    auto plan = plan_atomic_cut(
      span<const service::storage_owner::ReconcileReverseOp>{target_ops},
      span<const RemotePtr>{before_stable},
      span<const RemotePtr>{before_provisional},
      span<const std::pair<RemotePtr, u64>>{active}, config.R, robust_prune);
    if (prune_retryable || !plan.valid) {
      unlock_node(target);
      valid = false;
      continue;
    }

    const bool changed = before_stable != plan.stable ||
      before_provisional != plan.provisional;
    if (changed) {
      write_graph_adjacency(target, plan.stable, plan.provisional,
                            adjacency.generation, false);
    }
    vec<std::pair<u64, u64>> acquired;
    vec<std::pair<u64, u64>> released;
    acquired.reserve(plan.acquired.size());
    released.reserve(plan.released.size());
    for (const auto& [child, sequence] : plan.acquired) {
      acquired.emplace_back(child.raw_address, sequence);
    }
    for (const auto& [child, sequence] : plan.released) {
      released.emplace_back(child.raw_address, sequence);
    }
    if (!storage_owner_certificate_leases_->commit(
          target.raw_address, acquired, released)) {
      if (changed) {
        write_graph_adjacency(target, before_stable, before_provisional,
                              adjacency.generation, false);
      }
      unlock_node(target);
      valid = false;
      continue;
    }
    storage_owner_lease_merge_atomic_cuts_.fetch_add(
      1, std::memory_order_relaxed);
    storage_owner_lease_merge_target_ops_.fetch_add(
      indices.size(), std::memory_order_relaxed);
    for (size_t slot = 0; slot < indices.size(); ++slot) {
      results[indices[slot]] = plan.results[slot];
    }
    unlock_node(target);
  }
  return valid;
}
