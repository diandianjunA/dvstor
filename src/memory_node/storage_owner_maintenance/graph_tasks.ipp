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
  NodeSnapshot snapshot;
  if (!read_node_snapshot(target_ptr, snapshot)) {
    unlock_node(target_ptr);
    return false;
  }
  if (snapshot.deleted) {
    unlock_node(target_ptr);
    return true;
  }

  vec<RemotePtr> neighbors = read_neighbor_list(target_ptr);
  const auto old_size = neighbors.size();
  neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), deleted_ptr), neighbors.end());
  if (neighbors.size() != old_size) {
    write_neighbor_list(target_ptr, neighbors);
  }
  unlock_node(target_ptr);
  return true;
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
  vec<vec<RemotePtr>> candidate_pools;
  valid_tasks.reserve(tasks.size());
  targets.reserve(tasks.size());
  candidate_pools.reserve(tasks.size());

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
    candidate_pools.push_back(read_neighbor_list(task.target));
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
      vec<vec<RemotePtr>> shard_candidates;
      if (!wait_for_peer_stitch_search_response(request.request_id, request.shard,
                                                request.item_count, shard_candidates, config)) {
        retry_tasks.insert(retry_tasks.end(), valid_tasks.begin(), valid_tasks.end());
        return false;
      }
      u64 candidate_count = 0;
      for (size_t i = 0; i < shard_candidates.size() && i < candidate_pools.size(); ++i) {
        candidate_count += shard_candidates[i].size();
        candidate_pools[i].insert(candidate_pools[i].end(),
                                  shard_candidates[i].begin(),
                                  shard_candidates[i].end());
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
    vec<RemotePtr>& candidates = candidate_pools[item];

    hashset_t<RemotePtr> skip;
    skip.insert(task.target);
    const u32 candidate_limit = static_cast<u32>(
      std::max<size_t>(config.R, candidates.size()));
    vec<RemotePtr> final_neighbors = robust_prune_cpu(target_snapshot.vector_data.data(),
                                                      VamanaNode::vector_dtype(),
                                                      candidates,
                                                      skip,
                                                      config,
                                                      nullptr,
                                                      candidate_limit);

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
    bool local_reverse_ok = true;
    for (const RemotePtr& neighbor : final_neighbors) {
      if (local_shard(neighbor.memory_node())) {
        vec<RemotePtr> reverse_candidate{task.target};
        if (!apply_local_reverse_update(neighbor, reverse_candidate, config, false)) {
          local_reverse_ok = false;
          break;
        }
      } else {
        task_remote_updates[neighbor.memory_node()].push_back(
          service::storage_owner::ReverseUpdateOp{neighbor.raw_address, task.target.raw_address});
      }
    }
    if (!local_reverse_ok) {
      retry_tasks.push_back(task);
      continue;
    }
    for (auto& [target_shard, ops] : task_remote_updates) {
      auto& merged = remote_updates[target_shard];
      merged.insert(merged.end(), ops.begin(), ops.end());
    }
    finalized_tasks.push_back(task);
  }

  for (auto& [target_shard, ops] : remote_updates) {
    if (!send_reverse_update_batch(target_shard, ops, config)) {
      retry_tasks.insert(retry_tasks.end(), finalized_tasks.begin(), finalized_tasks.end());
      return false;
    }
  }

  for (const StorageOwnerMaintenanceTask& task : finalized_tasks) {
    complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
    const u64 finalize_latency_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - task.queued_at).count());
    storage_owner_maintenance_finalize_latency_ns_.fetch_add(finalize_latency_ns,
                                                             std::memory_order_relaxed);
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

bool MemoryNode::cleanup_deleted_storage_owner_node(const StorageOwnerMaintenanceTask& task,
                                                    const Configuration& config) {
  if (task.target.is_null()) {
    return true;
  }

  NodeSnapshot deleted_snapshot;
  if (!read_node_snapshot(task.target, deleted_snapshot)) {
    return false;
  }
  if (!deleted_snapshot.deleted) {
    storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  const vec<RemotePtr> old_neighbors = read_preserved_neighbor_list(task.target);
  dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_cleanup;
  bool ok = true;
  for (const RemotePtr& neighbor : old_neighbors) {
    if (neighbor.is_null()) {
      continue;
    }
    if (local_shard(neighbor.memory_node())) {
      ok &= remove_local_neighbor(neighbor, task.target, config);
    } else {
      remote_cleanup[neighbor.memory_node()].push_back(
        service::storage_owner::ReverseUpdateOp{neighbor.raw_address, task.target.raw_address});
    }
  }
  for (auto& [target_shard, ops] : remote_cleanup) {
    ok &= send_cleanup_deleted_batch(target_shard, ops, config);
  }
  if (ok) {
    retire_local_dynamic_node(task.target, task.maintenance_sequence);
  }
  return ok;
}
