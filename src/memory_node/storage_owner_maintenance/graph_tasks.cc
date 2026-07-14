#include "memory_node/storage_owner_maintenance/detail.hh"

using namespace memory_node_storage_owner_maintenance_detail;

bool MemoryNode::try_lock_node(RemotePtr rptr) {
  if (rptr.is_null() || !local_shard(rptr.memory_node())) {
    return false;
  }

  auto* header_ptr = reinterpret_cast<u64*>(
    index_buffer_.get_full_buffer() + vamana::StorageLayoutResolver::header(rptr).offset);
  std::atomic_ref<u64> ref(*header_ptr);
  for (u32 attempt = 0; attempt < 8; ++attempt) {
    u64 header = ref.load(std::memory_order_acquire);
    if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
      return false;
    }
    const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
    if (ref.compare_exchange_weak(header, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

bool MemoryNode::storage_owner_task_current(node_t id, u32 generation, RemotePtr target) {
  std::lock_guard<std::mutex> lock(idmap_mutex_);
  const auto it = idmap_.find(id);
  return it != idmap_.end() &&
         !it->second.deleted &&
         it->second.generation == generation &&
         it->second.current == target;
}

vec<RemotePtr> MemoryNode::read_preserved_neighbor_list(RemotePtr rptr) {
  if (rptr.is_null()) {
    return {};
  }
  vec<byte_t> entry(VamanaNode::hot_graph_entry_size());
  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
  if (local_shard(rptr.memory_node())) {
    std::memcpy(entry.data(),
                index_buffer_.get_full_buffer() + hot_offset,
                entry.size());
  } else {
    remote_read_bytes(rptr.memory_node(), hot_offset, entry.data(), entry.size(), 0);
  }

  const u8 edge_count = entry[0];
  if (edge_count > VamanaNode::R) {
    return {};
  }
  const u16 expected = vamana::hot_graph::load_u16_le(entry.data() + 2);
  const u16 actual = vamana::hot_graph::checksum16(entry.data(), entry.size());
  if (expected != actual) {
    return {};
  }
  vec<RemotePtr> neighbors;
  neighbors.reserve(edge_count);
  for (u32 i = 0; i < edge_count; ++i) {
    RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
      entry.data() + vamana::hot_graph::neighbor_offset(i),
      VamanaNode::HOT_GRAPH_SHARD_BITS);
    if (!neighbor.is_null()) {
      neighbors.push_back(neighbor);
    }
  }
  return neighbors;
}

bool MemoryNode::remove_local_neighbor(RemotePtr target_ptr,
                                       RemotePtr deleted_ptr,
                                       const Configuration&) {
  if (target_ptr.is_null() || deleted_ptr.is_null() || !local_shard(target_ptr.memory_node())) {
    return false;
  }

  lock_node(target_ptr);
  const bool target_deleted =
    (*reinterpret_cast<const u64*>(local_node_ptr(target_ptr)) &
     VamanaNode::HEADER_DELETED) != 0;
  if (target_deleted) {
    unlock_node(target_ptr);
    return true;
  }

  vec<RemotePtr> neighbors = read_neighbor_list(target_ptr);
  const auto old_size = neighbors.size();
  neighbors.erase(
    std::remove(neighbors.begin(), neighbors.end(), deleted_ptr),
    neighbors.end());
  if (neighbors.size() != old_size) {
    write_neighbor_list(target_ptr, neighbors);
  }
  unlock_node(target_ptr);
  return true;
}

bool MemoryNode::remove_local_neighbors_batched(
    const dense_hashmap_t<u64, vec<RemotePtr>>& removals,
    const Configuration&) {
  bool success = true;
  for (const auto& [target_raw, deleted_ptrs] : removals) {
    const RemotePtr target_ptr{target_raw};
    if (target_ptr.is_null() || !local_shard(target_ptr.memory_node())) {
      success = false;
      continue;
    }

    lock_node(target_ptr);
    const bool target_deleted =
      (*reinterpret_cast<const u64*>(local_node_ptr(target_ptr)) &
       VamanaNode::HEADER_DELETED) != 0;
    if (target_deleted) {
      unlock_node(target_ptr);
      continue;
    }

    vec<RemotePtr> neighbors = read_neighbor_list(target_ptr);
    const auto old_size = neighbors.size();
    neighbors.erase(
      std::remove_if(neighbors.begin(), neighbors.end(), [&](const RemotePtr& neighbor) {
        return std::find(deleted_ptrs.begin(), deleted_ptrs.end(), neighbor) !=
               deleted_ptrs.end();
      }),
      neighbors.end());
    if (neighbors.size() != old_size) {
      write_neighbor_list(target_ptr, neighbors);
    }
    unlock_node(target_ptr);
  }
  return success;
}

bool MemoryNode::stitch_inserted_storage_owner_nodes(
    const vec<StorageOwnerMaintenanceTask>& tasks,
    const Configuration& config,
    vec<StorageOwnerMaintenanceTask>& retry_tasks,
    u64& processed_count) {
  retry_tasks.clear();
  processed_count = 0;
  if (tasks.empty()) {
    return true;
  }

  storage_owner_stitch_batches_.fetch_add(1, std::memory_order_relaxed);
  storage_owner_stitch_batched_items_.fetch_add(tasks.size(), std::memory_order_relaxed);

  vec<StorageOwnerMaintenanceTask> valid_tasks;
  vec<NodeSnapshot> targets;
  vec<vec<NodeSnapshot>> candidate_snapshots;
  valid_tasks.reserve(tasks.size());
  targets.reserve(tasks.size());
  candidate_snapshots.reserve(tasks.size());

  for (const StorageOwnerMaintenanceTask& task : tasks) {
    if (!local_shard(task.target.memory_node())) {
      complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
      ++processed_count;
      continue;
    }
    if (!storage_owner_task_current(task.id, task.generation, task.target)) {
      storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
      complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
      ++processed_count;
      continue;
    }

    NodeSnapshot target_snapshot;
    if (!read_node_snapshot(task.target, target_snapshot)) {
      retry_tasks.push_back(task);
      continue;
    }
    if (target_snapshot.deleted) {
      storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
      complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
      ++processed_count;
      continue;
    }

    valid_tasks.push_back(task);
    targets.push_back(std::move(target_snapshot));
    candidate_snapshots.push_back(
      read_node_snapshots_batched(read_neighbor_list(task.target), config));
  }

  if (valid_tasks.empty()) {
    return retry_tasks.empty();
  }

  if (peer_context_ != nullptr && num_storage_nodes_ > 1) {
    struct PendingStitchRequest {
      u32 shard{};
      u64 request_id{};
      u32 item_count{};
    };
    vec<PendingStitchRequest> pending;
    pending.reserve(num_storage_nodes_ - 1);
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      if (shard == storage_id_) {
        continue;
      }
      PendingStitchRequest request;
      request.shard = shard;
      if (!post_stitch_search_request(shard, targets, request.request_id,
                                      request.item_count, config)) {
        retry_tasks.insert(retry_tasks.end(), valid_tasks.begin(), valid_tasks.end());
        return false;
      }
      if (request.item_count != 0) {
        pending.push_back(request);
      }
    }
    for (const PendingStitchRequest& request : pending) {
      vec<vec<NodeSnapshot>> shard_candidates;
      if (!wait_for_peer_stitch_search_response(request.request_id, request.shard,
                                                request.item_count, shard_candidates, config)) {
        retry_tasks.insert(retry_tasks.end(), valid_tasks.begin(), valid_tasks.end());
        return false;
      }
      u64 candidate_count = 0;
      for (size_t i = 0; i < shard_candidates.size() && i < candidate_snapshots.size(); ++i) {
        candidate_count += shard_candidates[i].size();
        candidate_snapshots[i].insert(
          candidate_snapshots[i].end(),
          std::make_move_iterator(shard_candidates[i].begin()),
          std::make_move_iterator(shard_candidates[i].end()));
      }
      storage_owner_stitch_external_requests_.fetch_add(1, std::memory_order_relaxed);
      storage_owner_stitch_external_candidates_.fetch_add(candidate_count,
                                                          std::memory_order_relaxed);
    }
  }

  dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_updates;
  vec<StorageOwnerMaintenanceTask> finalized_tasks;
  finalized_tasks.reserve(valid_tasks.size());
  for (size_t item = 0; item < valid_tasks.size(); ++item) {
    const StorageOwnerMaintenanceTask& task = valid_tasks[item];
    const NodeSnapshot& target_snapshot = targets[item];
    const vec<NodeSnapshot>& candidates = candidate_snapshots[item];

    hashset_t<RemotePtr> skip;
    skip.insert(task.target);
    vec<NodeSnapshot> local_candidates;
    vec<NodeSnapshot> external_candidates;
    local_candidates.reserve(candidates.size());
    external_candidates.reserve(candidates.size());
    for (const NodeSnapshot& candidate : candidates) {
      if (candidate.rptr.is_null()) {
        continue;
      }
      if (local_shard(candidate.rptr.memory_node())) {
        local_candidates.push_back(candidate);
      } else {
        external_candidates.push_back(candidate);
      }
    }

    vec<RemotePtr> external_neighbors = robust_prune_snapshots_cpu(
      target_snapshot.vector_data.data(),
      VamanaNode::vector_dtype(),
      external_candidates,
      skip,
      config,
      storage_owner_cross_shard_degree_);
    const u32 local_degree = external_neighbors.size() >= config.R
                               ? 0
                               : config.R - static_cast<u32>(external_neighbors.size());
    vec<RemotePtr> final_neighbors;
    if (local_degree != 0) {
      final_neighbors = robust_prune_snapshots_cpu(
        target_snapshot.vector_data.data(),
        VamanaNode::vector_dtype(),
        local_candidates,
        skip,
        config,
        local_degree);
    }
    final_neighbors.insert(final_neighbors.end(),
                           external_neighbors.begin(),
                           external_neighbors.end());
    lib_assert(final_neighbors.size() <= config.R,
               "online stitch exceeded graph degree");

    lock_node(task.target);
    if (!storage_owner_task_current(task.id, task.generation, task.target)) {
      unlock_node(task.target);
      storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
      complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
      ++processed_count;
      continue;
    }
    NodeSnapshot latest_snapshot;
    if (!read_node_snapshot(task.target, latest_snapshot)) {
      unlock_node(task.target);
      retry_tasks.push_back(task);
      continue;
    }
    if (latest_snapshot.deleted) {
      unlock_node(task.target);
      storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
      complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
      ++processed_count;
      continue;
    }
    const vec<RemotePtr> current_neighbors = read_neighbor_list(task.target);
    if (!same_neighbors(current_neighbors, final_neighbors)) {
      write_neighbor_list(task.target, final_neighbors);
    }
    unlock_node(task.target);

    dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>> task_remote_updates;
    for (const RemotePtr& neighbor : final_neighbors) {
      if (!local_shard(neighbor.memory_node())) {
        task_remote_updates[neighbor.memory_node()].push_back(
          service::storage_owner::ReverseUpdateOp{neighbor.raw_address, task.target.raw_address});
      }
    }
    for (auto& [target_shard, ops] : task_remote_updates) {
      auto& merged = remote_updates[target_shard];
      merged.insert(merged.end(), ops.begin(), ops.end());
    }
    finalized_tasks.push_back(task);
  }

  if (!send_reverse_update_fanout_and_wait(remote_updates, config)) {
    retry_tasks.insert(retry_tasks.end(), finalized_tasks.begin(), finalized_tasks.end());
    return false;
  }

  for (const StorageOwnerMaintenanceTask& task : finalized_tasks) {
    complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
    const u64 finalize_latency_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - task.queued_at).count());
    storage_owner_maintenance_finalize_latency_ns_.fetch_add(finalize_latency_ns,
                                                             std::memory_order_relaxed);
    storage_owner_maintenance_finalize_latency_buckets_[
      finalize_latency_bucket(finalize_latency_ns)].fetch_add(1, std::memory_order_relaxed);
    atomic_utils::update_max_relaxed(
      storage_owner_maintenance_finalize_max_latency_ns_, finalize_latency_ns);
  }
  if (!finalized_tasks.empty()) {
    storage_owner_maintenance_finalized_live_.fetch_add(finalized_tasks.size(),
                                                        std::memory_order_relaxed);
    processed_count += finalized_tasks.size();
  }
  return retry_tasks.empty();
}

bool MemoryNode::stitch_inserted_storage_owner_node(const StorageOwnerMaintenanceTask& task,
                                                    const Configuration& config) {
  vec<StorageOwnerMaintenanceTask> tasks;
  tasks.push_back(task);
  vec<StorageOwnerMaintenanceTask> retry_tasks;
  u64 processed_count = 0;
  return stitch_inserted_storage_owner_nodes(tasks, config, retry_tasks, processed_count) &&
         retry_tasks.empty();
}

bool MemoryNode::cleanup_deleted_storage_owner_nodes(
    const vec<StorageOwnerMaintenanceTask>& tasks,
    const Configuration& config,
    vec<StorageOwnerMaintenanceTask>& retry_tasks,
    u64& processed_count) {
  retry_tasks.clear();
  processed_count = 0;
  dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_cleanup;
  vec<StorageOwnerMaintenanceTask> ready_tasks;
  ready_tasks.reserve(tasks.size());

  for (const StorageOwnerMaintenanceTask& task : tasks) {
    if (task.target.is_null()) {
      complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
      ++processed_count;
      continue;
    }

    NodeSnapshot deleted_snapshot;
    if (!read_node_snapshot(task.target, deleted_snapshot)) {
      retry_tasks.push_back(task);
      continue;
    }
    if (!deleted_snapshot.deleted) {
      storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
      complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
      ++processed_count;
      continue;
    }

    bool local_ok = true;
    const vec<RemotePtr> old_neighbors = read_preserved_neighbor_list(task.target);
    dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>> task_remote_cleanup;
    for (const RemotePtr& neighbor : old_neighbors) {
      if (neighbor.is_null()) {
        continue;
      }
      if (local_shard(neighbor.memory_node())) {
        local_ok &= remove_local_neighbor(neighbor, task.target, config);
      } else {
        task_remote_cleanup[neighbor.memory_node()].push_back(
          service::storage_owner::ReverseUpdateOp{
            neighbor.raw_address, task.target.raw_address});
      }
    }
    if (!local_ok) {
      retry_tasks.push_back(task);
      continue;
    }
    for (auto& [target_shard, ops] : task_remote_cleanup) {
      auto& merged = remote_cleanup[target_shard];
      merged.insert(merged.end(), ops.begin(), ops.end());
    }
    ready_tasks.push_back(task);
  }

  const bool remote_ok = send_cleanup_deleted_fanout_and_wait(remote_cleanup, config);
  if (!remote_ok) {
    retry_tasks.insert(retry_tasks.end(), ready_tasks.begin(), ready_tasks.end());
    return false;
  }

  for (const StorageOwnerMaintenanceTask& task : ready_tasks) {
    retire_local_dynamic_node(task.target, task.maintenance_sequence);
    complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
    ++processed_count;
  }
  return retry_tasks.empty();
}
