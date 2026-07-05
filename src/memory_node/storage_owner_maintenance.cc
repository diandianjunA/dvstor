#include "memory_node/memory_node.hh"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

namespace {

size_t maintenance_candidate_limit(const configuration::IndexConfiguration& config) {
  if (config.storage_owner_prune_max_candidates == 0) {
    return std::max<size_t>(256, static_cast<size_t>(config.R) * 4);
  }
  return std::max<size_t>(config.R, config.storage_owner_prune_max_candidates);
}

bool append_unique(vec<RemotePtr>& out, RemotePtr ptr, size_t limit) {
  if (ptr.is_null() || out.size() >= limit) {
    return false;
  }
  if (std::find(out.begin(), out.end(), ptr) != out.end()) {
    return false;
  }
  out.push_back(ptr);
  return true;
}

void append_unique_all(vec<RemotePtr>& out,
                       const vec<RemotePtr>& input,
                       size_t limit,
                       RemotePtr skip = RemotePtr{}) {
  for (const RemotePtr& ptr : input) {
    if (ptr == skip) {
      continue;
    }
    append_unique(out, ptr, limit);
  }
}

bool same_neighbors(const vec<RemotePtr>& lhs, const vec<RemotePtr>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

bool contains_ptr(const vec<RemotePtr>& values, RemotePtr ptr) {
  return std::find(values.begin(), values.end(), ptr) != values.end();
}

}  // namespace

bool MemoryNode::storage_owner_maintenance_enabled(const Configuration& config) {
  return config.storage_owner_maintenance_mode == "budgeted" &&
         config.storage_owner_maintenance_workers > 0 &&
         config.storage_owner_maintenance_budget_us > 0 &&
         config.storage_owner_maintenance_period_us > 0 &&
         config.storage_owner_maintenance_queue_depth > 0 &&
         config.storage_owner_maintenance_budget_us <= config.storage_owner_maintenance_period_us;
}

void MemoryNode::start_storage_owner_maintenance_runtime(const Configuration& config) {
  if (!use_storage_owner_insert_ || !storage_owner_maintenance_enabled(config)) {
    return;
  }

  storage_owner_maintenance_shutdown_.store(false, std::memory_order_release);
  storage_owner_maintenance_queue_limit_ =
    std::max<size_t>(1, config.storage_owner_maintenance_queue_depth);

  const u32 worker_count = std::max<u32>(1, config.storage_owner_maintenance_workers);
  const size_t snapshot_stride = memory_node_detail::storage_owner_snapshot_stride();
  const size_t neighbor_stride = align_up(VamanaNode::neighbor_read_size());
  const size_t coroutine_scratch_stride =
    align_up(std::max<size_t>(VamanaNode::total_size(),
                              std::max(neighbor_stride,
                                       snapshot_stride *
                                         std::max<u32>(1, config.storage_owner_search_snapshot_batch))));
  const size_t scratch_bytes = std::max<size_t>(64ull * 1024ull * 1024ull, coroutine_scratch_stride);

  storage_owner_maintenance_worker_states_.reserve(worker_count);
  for (u32 worker_id = 0; worker_id < worker_count; ++worker_id) {
    auto worker = std::make_unique<StorageOwnerThread>(worker_id, 1, config.max_send_queue_wr);
    if (peer_context_) {
      worker->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
    }
    storage_owner_maintenance_worker_states_.push_back(std::move(worker));
  }

  for (u32 worker_id = 0; worker_id < worker_count; ++worker_id) {
    storage_owner_maintenance_workers_.emplace_back(
      [this, worker_id]() { storage_owner_maintenance_worker_loop(worker_id); });
  }

  print_status("storage-owner maintenance workers: " + std::to_string(worker_count));
  print_status("storage-owner maintenance tuning: mode=" + config.storage_owner_maintenance_mode +
               " budget_us=" + std::to_string(config.storage_owner_maintenance_budget_us) +
               " period_us=" + std::to_string(config.storage_owner_maintenance_period_us) +
               " queue_depth=" + std::to_string(storage_owner_maintenance_queue_limit_));
}

void MemoryNode::stop_storage_owner_maintenance_runtime() {
  storage_owner_maintenance_shutdown_.store(true, std::memory_order_release);
  storage_owner_maintenance_cv_.notify_all();

  for (auto& worker : storage_owner_maintenance_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }

  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_tasks_.clear();
    storage_owner_maintenance_reverse_candidates_.clear();
  }
  storage_owner_maintenance_workers_.clear();
  storage_owner_maintenance_worker_states_.clear();

  if (storage_owner_maintenance_enqueued_.load(std::memory_order_relaxed) != 0 ||
      storage_owner_maintenance_dropped_.load(std::memory_order_relaxed) != 0) {
    print_status("storage-owner maintenance summary: enqueued=" +
                 std::to_string(storage_owner_maintenance_enqueued_.load(std::memory_order_relaxed)) +
                 " processed=" +
                 std::to_string(storage_owner_maintenance_processed_.load(std::memory_order_relaxed)) +
                 " failed=" +
                 std::to_string(storage_owner_maintenance_failed_.load(std::memory_order_relaxed)) +
                 " dropped=" +
                 std::to_string(storage_owner_maintenance_dropped_.load(std::memory_order_relaxed)));
  }
}

bool MemoryNode::enqueue_storage_owner_maintenance(StorageOwnerMaintenanceTask&& task,
                                                   const Configuration& config) {
  if (!storage_owner_maintenance_enabled(config) ||
      task.target.is_null() ||
      !local_shard(task.target.memory_node())) {
    return false;
  }

  task.queued_at = std::chrono::steady_clock::now();
  const size_t candidate_limit = maintenance_candidate_limit(config);
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
      return false;
    }

    if (task.kind == StorageOwnerMaintenanceKind::reverse_edge_repair) {
      auto existing = storage_owner_maintenance_reverse_candidates_.find(task.target.raw_address);
      if (existing != storage_owner_maintenance_reverse_candidates_.end()) {
        append_unique_all(existing->second, task.candidates, candidate_limit);
        return true;
      }
      if (storage_owner_maintenance_tasks_.size() >= storage_owner_maintenance_queue_limit_) {
        storage_owner_maintenance_dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      vec<RemotePtr> candidates;
      candidates.reserve(std::min(candidate_limit, task.candidates.size()));
      append_unique_all(candidates, task.candidates, candidate_limit);
      if (candidates.empty()) {
        return false;
      }
      storage_owner_maintenance_reverse_candidates_.emplace(task.target.raw_address, std::move(candidates));
      task.candidates.clear();
    } else if (storage_owner_maintenance_tasks_.size() >= storage_owner_maintenance_queue_limit_) {
      storage_owner_maintenance_dropped_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    storage_owner_maintenance_tasks_.push_back(std::move(task));
    storage_owner_maintenance_enqueued_.fetch_add(1, std::memory_order_relaxed);
  }
  storage_owner_maintenance_cv_.notify_one();
  return true;
}

bool MemoryNode::enqueue_inserted_node_repair(RemotePtr target, const Configuration& config) {
  StorageOwnerMaintenanceTask task;
  task.kind = StorageOwnerMaintenanceKind::inserted_node_repair;
  task.target = target;
  return enqueue_storage_owner_maintenance(std::move(task), config);
}

bool MemoryNode::enqueue_reverse_edge_repair(RemotePtr target,
                                             const vec<RemotePtr>& candidates,
                                             const Configuration& config) {
  if (candidates.empty()) {
    return false;
  }
  StorageOwnerMaintenanceTask task;
  task.kind = StorageOwnerMaintenanceKind::reverse_edge_repair;
  task.target = target;
  task.candidates = candidates;
  return enqueue_storage_owner_maintenance(std::move(task), config);
}

bool MemoryNode::enqueue_tombstone_cleanup(RemotePtr deleted_ptr,
                                           const vec<RemotePtr>& candidate_neighbors,
                                           const Configuration& config) {
  if (candidate_neighbors.empty()) {
    return false;
  }
  StorageOwnerMaintenanceTask task;
  task.kind = StorageOwnerMaintenanceKind::tombstone_cleanup;
  task.target = deleted_ptr;
  task.candidates = candidate_neighbors;
  return enqueue_storage_owner_maintenance(std::move(task), config);
}

void MemoryNode::storage_owner_maintenance_worker_loop(u32 worker_id) {
  lib_assert(worker_id < storage_owner_maintenance_worker_states_.size(),
             "storage-owner maintenance worker state missing");
  StorageOwnerThread& thread = *storage_owner_maintenance_worker_states_[worker_id];
  current_storage_owner_thread_ = &thread;

  const Configuration& config = *storage_worker_config_;
  const auto budget = std::chrono::microseconds(config.storage_owner_maintenance_budget_us);
  const auto period = std::chrono::microseconds(config.storage_owner_maintenance_period_us);
  auto window_started = std::chrono::steady_clock::now();

  for (;;) {
    StorageOwnerMaintenanceTask task;
    {
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait(lock, [&]() {
        return storage_owner_maintenance_shutdown_.load(std::memory_order_acquire) ||
               !storage_owner_maintenance_tasks_.empty();
      });
      if (storage_owner_maintenance_tasks_.empty()) {
        if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
          break;
        }
        continue;
      }
      task = std::move(storage_owner_maintenance_tasks_.front());
      storage_owner_maintenance_tasks_.pop_front();
      if (task.kind == StorageOwnerMaintenanceKind::reverse_edge_repair) {
        auto candidates = storage_owner_maintenance_reverse_candidates_.find(task.target.raw_address);
        if (candidates != storage_owner_maintenance_reverse_candidates_.end()) {
          task.candidates = std::move(candidates->second);
          storage_owner_maintenance_reverse_candidates_.erase(candidates);
        }
      }
    }

    bool ok = true;
    switch (task.kind) {
      case StorageOwnerMaintenanceKind::inserted_node_repair:
        ok = repair_inserted_storage_owner_node(task.target, config);
        break;
      case StorageOwnerMaintenanceKind::reverse_edge_repair:
        ok = repair_storage_owner_neighbors(task.target, task.candidates, config);
        break;
      case StorageOwnerMaintenanceKind::tombstone_cleanup:
        for (const RemotePtr& neighbor : task.candidates) {
          if (!local_shard(neighbor.memory_node())) {
            continue;
          }
          vec<RemotePtr> deleted_candidate{task.target};
          ok &= repair_storage_owner_neighbors(neighbor, deleted_candidate, config);
        }
        break;
    }

    if (ok) {
      storage_owner_maintenance_processed_.fetch_add(1, std::memory_order_relaxed);
    } else {
      storage_owner_maintenance_failed_.fetch_add(1, std::memory_order_relaxed);
    }

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - window_started);
    if (elapsed >= budget) {
      if (elapsed < period) {
        std::this_thread::sleep_for(period - elapsed);
      } else {
        std::this_thread::yield();
      }
      window_started = std::chrono::steady_clock::now();
    }
  }

  current_storage_owner_thread_ = nullptr;
}

bool MemoryNode::repair_storage_owner_neighbors(RemotePtr target_ptr,
                                                const vec<RemotePtr>& candidate_ptrs,
                                                const Configuration& config) {
  if (target_ptr.is_null() || !local_shard(target_ptr.memory_node())) {
    return false;
  }

  for (u32 attempt = 0; attempt < 2; ++attempt) {
    lock_node(target_ptr);
    NodeSnapshot target_snapshot;
    if (!read_node_snapshot(target_ptr, target_snapshot)) {
      unlock_node(target_ptr);
      return false;
    }
    if (target_snapshot.deleted) {
      unlock_node(target_ptr);
      return true;
    }
    const vec<RemotePtr> current_neighbors = read_neighbor_list(target_ptr);
    unlock_node(target_ptr);

    vec<RemotePtr> repair_candidates;
    const size_t candidate_limit =
      maintenance_candidate_limit(config) + std::min<size_t>(config.R, current_neighbors.size());
    repair_candidates.reserve(std::min(candidate_limit, current_neighbors.size() + candidate_ptrs.size()));
    append_unique_all(repair_candidates, current_neighbors, candidate_limit, target_ptr);
    append_unique_all(repair_candidates, candidate_ptrs, candidate_limit, target_ptr);

    hashset_t<RemotePtr> skip;
    skip.insert(target_ptr);
    vec<RemotePtr> repaired = robust_prune_cpu(target_snapshot.vector_data.data(),
                                               VamanaNode::vector_dtype(),
                                               repair_candidates,
                                               skip,
                                               config,
                                               nullptr);

    lock_node(target_ptr);
    NodeSnapshot latest_snapshot;
    if (!read_node_snapshot(target_ptr, latest_snapshot)) {
      unlock_node(target_ptr);
      return false;
    }
    if (latest_snapshot.deleted) {
      unlock_node(target_ptr);
      return true;
    }
    const vec<RemotePtr> latest_neighbors = read_neighbor_list(target_ptr);
    if (!same_neighbors(current_neighbors, latest_neighbors) && attempt == 0) {
      unlock_node(target_ptr);
      continue;
    }
    if (!same_neighbors(latest_neighbors, repaired)) {
      write_neighbor_list(target_ptr, repaired);
    }
    unlock_node(target_ptr);
    return true;
  }
  return true;
}

bool MemoryNode::repair_inserted_storage_owner_node(RemotePtr target_ptr, const Configuration& config) {
  if (target_ptr.is_null() || !local_shard(target_ptr.memory_node())) {
    return false;
  }

  NodeSnapshot target_snapshot;
  if (!read_node_snapshot(target_ptr, target_snapshot) || target_snapshot.deleted) {
    return true;
  }

  RemotePtr medoid = read_global_medoid();
  if (medoid.is_null()) {
    return true;
  }

  vec<element_t> query = decode_storage_vector_to_float(
    target_snapshot.vector_data.data(), VamanaNode::vector_dtype(), VamanaNode::DIM);
  vec<RemotePtr> search_candidates =
    beam_search_candidates(span<const element_t>{query.data(), query.size()}, medoid, config, nullptr);

  vec<RemotePtr> previous_neighbors;
  vec<RemotePtr> repaired;
  bool changed = false;
  for (u32 attempt = 0; attempt < 2; ++attempt) {
    lock_node(target_ptr);
    if (!read_node_snapshot(target_ptr, target_snapshot) || target_snapshot.deleted) {
      unlock_node(target_ptr);
      return true;
    }
    const vec<RemotePtr> current_neighbors = read_neighbor_list(target_ptr);
    unlock_node(target_ptr);

    vec<RemotePtr> repair_candidates = search_candidates;
    const size_t candidate_limit =
      maintenance_candidate_limit(config) + std::min<size_t>(config.R, current_neighbors.size());
    append_unique_all(repair_candidates, current_neighbors, candidate_limit, target_ptr);

    hashset_t<RemotePtr> skip;
    skip.insert(target_ptr);
    repaired = robust_prune_cpu(reinterpret_cast<const byte_t*>(query.data()),
                                VectorDType::float32,
                                repair_candidates,
                                skip,
                                config,
                                nullptr);
    if (repaired.empty() && !current_neighbors.empty()) {
      return true;
    }

    lock_node(target_ptr);
    NodeSnapshot latest_snapshot;
    if (!read_node_snapshot(target_ptr, latest_snapshot) || latest_snapshot.deleted) {
      unlock_node(target_ptr);
      return true;
    }
    const vec<RemotePtr> latest_neighbors = read_neighbor_list(target_ptr);
    if (!same_neighbors(current_neighbors, latest_neighbors) && attempt == 0) {
      unlock_node(target_ptr);
      continue;
    }
    changed = !same_neighbors(latest_neighbors, repaired);
    if (changed) {
      write_neighbor_list(target_ptr, repaired);
    }
    previous_neighbors = latest_neighbors;
    unlock_node(target_ptr);
    break;
  }

  if (!changed) {
    return true;
  }

  std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_updates;
  for (const RemotePtr& neighbor : repaired) {
    if (contains_ptr(previous_neighbors, neighbor)) {
      continue;
    }
    if (local_shard(neighbor.memory_node())) {
      vec<RemotePtr> reverse_candidate{target_ptr};
      if (!apply_local_reverse_update(neighbor, reverse_candidate, config, false)) {
        return false;
      }
    } else {
      remote_updates[neighbor.memory_node()].push_back(
        service::storage_owner::ReverseUpdateOp{neighbor.raw_address, target_ptr.raw_address});
    }
  }
  for (auto& [target_shard, ops] : remote_updates) {
    if (!send_reverse_update_batch(target_shard, ops, config)) {
      return false;
    }
  }
  return true;
}
