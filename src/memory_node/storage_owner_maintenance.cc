#include "memory_node/memory_node.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include "vamana/storage_layout_resolver.hh"

namespace {

size_t maintenance_candidate_limit(const configuration::IndexConfiguration& config) {
  return std::max<size_t>(config.R, config.storage_owner_maintenance_max_candidates);
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
         config.storage_owner_maintenance_policy == "delta_log" &&
         config.storage_owner_maintenance_workers > 0 &&
         config.storage_owner_maintenance_budget_us > 0 &&
         config.storage_owner_maintenance_period_us > 0 &&
         config.storage_owner_maintenance_queue_depth > 0 &&
         config.storage_owner_maintenance_max_dirty_per_period > 0 &&
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
               " policy=" + config.storage_owner_maintenance_policy +
               " budget_us=" + std::to_string(config.storage_owner_maintenance_budget_us) +
               " period_us=" + std::to_string(config.storage_owner_maintenance_period_us) +
               " dirty_per_period=" +
               std::to_string(config.storage_owner_maintenance_max_dirty_per_period) +
               " max_candidates=" +
               std::to_string(config.storage_owner_maintenance_max_candidates) +
               " max_expansions=" +
               std::to_string(config.storage_owner_maintenance_max_expansions) +
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
    storage_owner_maintenance_dirty_queue_.clear();
    storage_owner_maintenance_dirty_.clear();
  }
  storage_owner_maintenance_workers_.clear();
  storage_owner_maintenance_worker_states_.clear();

  if (storage_owner_maintenance_enqueued_.load(std::memory_order_relaxed) != 0 ||
      storage_owner_maintenance_coalesced_.load(std::memory_order_relaxed) != 0 ||
      storage_owner_maintenance_dropped_.load(std::memory_order_relaxed) != 0) {
    print_status("storage-owner maintenance summary: enqueued=" +
                 std::to_string(storage_owner_maintenance_enqueued_.load(std::memory_order_relaxed)) +
                 " coalesced=" +
                 std::to_string(storage_owner_maintenance_coalesced_.load(std::memory_order_relaxed)) +
                 " processed=" +
                 std::to_string(storage_owner_maintenance_processed_.load(std::memory_order_relaxed)) +
                 " failed=" +
                 std::to_string(storage_owner_maintenance_failed_.load(std::memory_order_relaxed)) +
                 " dropped=" +
                 std::to_string(storage_owner_maintenance_dropped_.load(std::memory_order_relaxed)) +
                 " skipped_busy=" +
                 std::to_string(storage_owner_maintenance_skipped_busy_.load(std::memory_order_relaxed)) +
                 " local_repairs=" +
                 std::to_string(storage_owner_maintenance_local_repairs_.load(std::memory_order_relaxed)) +
                 " expand_repairs=" +
                 std::to_string(storage_owner_maintenance_expand_repairs_.load(std::memory_order_relaxed)));
  }
}

bool MemoryNode::enqueue_storage_owner_maintenance(StorageOwnerMaintenanceTask&& task,
                                                   const Configuration& config) {
  if (!storage_owner_maintenance_enabled(config) ||
      task.target.is_null() ||
      !local_shard(task.target.memory_node())) {
    return false;
  }

  task.kind = StorageOwnerMaintenanceKind::dirty_node;
  task.queued_at = std::chrono::steady_clock::now();
  task.dirty_count = std::max<u32>(1, task.dirty_count);
  const size_t candidate_limit = maintenance_candidate_limit(config);

  bool should_notify = false;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
      return false;
    }

    const u64 key = task.target.raw_address;
    auto existing = storage_owner_maintenance_dirty_.find(key);
    if (existing != storage_owner_maintenance_dirty_.end()) {
      append_unique_all(existing->second.candidates, task.candidates, candidate_limit, task.target);
      const u32 max_dirty = std::numeric_limits<u32>::max() - 1;
      existing->second.dirty_count =
        existing->second.dirty_count > max_dirty - task.dirty_count
          ? max_dirty
          : existing->second.dirty_count + task.dirty_count;
      existing->second.expand_hint = existing->second.expand_hint || task.expand_hint;
      existing->second.tombstone_hint = existing->second.tombstone_hint || task.tombstone_hint;
      storage_owner_maintenance_coalesced_.fetch_add(1, std::memory_order_relaxed);
      should_notify = !existing->second.queued;
    } else {
      vec<RemotePtr> candidates;
      candidates.reserve(std::min(candidate_limit, task.candidates.size()));
      append_unique_all(candidates, task.candidates, candidate_limit, task.target);
      task.candidates = std::move(candidates);
      task.queued = storage_owner_maintenance_dirty_queue_.size() < storage_owner_maintenance_queue_limit_;
      if (task.queued) {
        storage_owner_maintenance_dirty_queue_.push_back(key);
        storage_owner_maintenance_enqueued_.fetch_add(1, std::memory_order_relaxed);
      }
      should_notify = true;
      storage_owner_maintenance_dirty_.emplace(key, std::move(task));
    }
  }

  if (should_notify) {
    storage_owner_maintenance_cv_.notify_one();
  }
  return true;
}

bool MemoryNode::enqueue_inserted_node_repair(RemotePtr target,
                                              const vec<RemotePtr>& candidates,
                                              const Configuration& config) {
  StorageOwnerMaintenanceTask task;
  task.target = target;
  task.candidates = candidates;
  task.dirty_count = 1;
  return enqueue_storage_owner_maintenance(std::move(task), config);
}

bool MemoryNode::enqueue_reverse_edge_repair(RemotePtr target,
                                             const vec<RemotePtr>& candidates,
                                             const Configuration& config) {
  if (candidates.empty()) {
    return false;
  }
  StorageOwnerMaintenanceTask task;
  task.target = target;
  task.candidates = candidates;
  task.dirty_count = 1;
  task.expand_hint = candidates.size() >= config.storage_owner_maintenance_max_expansions;
  return enqueue_storage_owner_maintenance(std::move(task), config);
}

bool MemoryNode::enqueue_tombstone_cleanup(RemotePtr deleted_ptr,
                                           const vec<RemotePtr>& candidate_neighbors,
                                           const Configuration& config) {
  if (deleted_ptr.is_null() || candidate_neighbors.empty()) {
    return false;
  }

  bool enqueued = false;
  std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_dirty;
  for (const RemotePtr& neighbor : candidate_neighbors) {
    if (neighbor.is_null()) {
      continue;
    }
    if (local_shard(neighbor.memory_node())) {
      StorageOwnerMaintenanceTask task;
      task.target = neighbor;
      task.candidates.push_back(deleted_ptr);
      task.dirty_count = 1;
      task.tombstone_hint = true;
      task.expand_hint = true;
      enqueued = enqueue_storage_owner_maintenance(std::move(task), config) || enqueued;
    } else {
      remote_dirty[neighbor.memory_node()].push_back(
        service::storage_owner::ReverseUpdateOp{neighbor.raw_address, deleted_ptr.raw_address});
      enqueued = true;
    }
  }

  for (auto& [target_shard, ops] : remote_dirty) {
    (void)send_maintenance_dirty_batch(target_shard, ops, config);
  }
  return enqueued;
}

bool MemoryNode::storage_owner_maintenance_foreground_busy(const Configuration& config) {
  const double high_watermark = std::clamp(config.storage_owner_maintenance_high_watermark, 0.01, 1.0);
  const size_t insert_high_watermark =
    std::max<size_t>(1, static_cast<size_t>(std::ceil(config.storage_owner_rpc_depth * high_watermark)));
  const size_t reverse_high_watermark =
    std::max<size_t>(1, static_cast<size_t>(std::ceil(peer_reverse_outgoing_queue_limit_ * high_watermark)));

  {
    std::unique_lock<std::mutex> lock(storage_insert_tasks_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return true;
    }
    if (storage_insert_tasks_.size() >= insert_high_watermark) {
      return true;
    }
  }
  {
    std::unique_lock<std::mutex> lock(peer_reverse_outgoing_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return true;
    }
    if (peer_reverse_outgoing_.size() >= reverse_high_watermark) {
      return true;
    }
  }
  {
    std::unique_lock<std::mutex> lock(peer_reverse_tasks_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return true;
    }
    if (peer_reverse_tasks_.size() >= reverse_high_watermark) {
      return true;
    }
  }
  return false;
}

void MemoryNode::storage_owner_maintenance_worker_loop(u32 worker_id) {
  lib_assert(worker_id < storage_owner_maintenance_worker_states_.size(),
             "storage-owner maintenance worker state missing");
  StorageOwnerThread& thread = *storage_owner_maintenance_worker_states_[worker_id];
  current_storage_owner_thread_ = &thread;

  const Configuration& config = *storage_worker_config_;
  const auto budget = std::chrono::microseconds(config.storage_owner_maintenance_budget_us);
  const auto period = std::chrono::microseconds(config.storage_owner_maintenance_period_us);
  const u32 max_dirty_per_period =
    std::max<u32>(1, config.storage_owner_maintenance_max_dirty_per_period);

  auto queue_unqueued_locked = [&]() {
    if (!storage_owner_maintenance_dirty_queue_.empty()) {
      return;
    }
    for (auto& [key, dirty] : storage_owner_maintenance_dirty_) {
      if (!dirty.queued) {
        dirty.queued = true;
        storage_owner_maintenance_dirty_queue_.push_back(key);
        storage_owner_maintenance_enqueued_.fetch_add(1, std::memory_order_relaxed);
        break;
      }
    }
  };

  for (;;) {
    const auto window_started = std::chrono::steady_clock::now();
    u32 processed_this_period = 0;

    while (processed_this_period < max_dirty_per_period) {
      if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
        current_storage_owner_thread_ = nullptr;
        return;
      }

      if (storage_owner_maintenance_foreground_busy(config)) {
        storage_owner_maintenance_skipped_busy_.fetch_add(1, std::memory_order_relaxed);
        break;
      }

      StorageOwnerMaintenanceTask task;
      {
        std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
        queue_unqueued_locked();
        if (storage_owner_maintenance_dirty_queue_.empty()) {
          storage_owner_maintenance_cv_.wait_until(lock, window_started + period, [&]() {
            if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
              return true;
            }
            queue_unqueued_locked();
            return !storage_owner_maintenance_dirty_queue_.empty();
          });
          if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
            current_storage_owner_thread_ = nullptr;
            return;
          }
          queue_unqueued_locked();
          if (storage_owner_maintenance_dirty_queue_.empty()) {
            break;
          }
        }

        const u64 key = storage_owner_maintenance_dirty_queue_.front();
        storage_owner_maintenance_dirty_queue_.pop_front();
        auto dirty = storage_owner_maintenance_dirty_.find(key);
        if (dirty == storage_owner_maintenance_dirty_.end()) {
          continue;
        }
        task = std::move(dirty->second);
        task.queued = false;
        storage_owner_maintenance_dirty_.erase(dirty);
      }

      const bool ok = repair_dirty_storage_owner_node(task, config);
      if (ok) {
        storage_owner_maintenance_processed_.fetch_add(1, std::memory_order_relaxed);
      } else {
        storage_owner_maintenance_failed_.fetch_add(1, std::memory_order_relaxed);
      }
      ++processed_this_period;

      const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - window_started);
      if (elapsed >= budget) {
        break;
      }
    }

    const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - window_started);
    if (elapsed < period) {
      std::this_thread::sleep_for(period - elapsed);
    } else {
      std::this_thread::yield();
    }
  }
}

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

bool MemoryNode::repair_dirty_storage_owner_node(const StorageOwnerMaintenanceTask& task,
                                                 const Configuration& config) {
  const bool allow_expand = task.expand_hint || task.tombstone_hint || task.dirty_count >= 4;
  return repair_storage_owner_neighbors(task.target, task.candidates, allow_expand, config);
}

bool MemoryNode::repair_storage_owner_neighbors(RemotePtr target_ptr,
                                                const vec<RemotePtr>& candidate_ptrs,
                                                bool allow_expand,
                                                const Configuration& config) {
  if (target_ptr.is_null() || !local_shard(target_ptr.memory_node())) {
    return false;
  }

  const size_t candidate_limit = maintenance_candidate_limit(config);
  for (u32 attempt = 0; attempt < 2; ++attempt) {
    if (!try_lock_node(target_ptr)) {
      StorageOwnerMaintenanceTask deferred;
      deferred.target = target_ptr;
      deferred.candidates = candidate_ptrs;
      deferred.dirty_count = 1;
      deferred.expand_hint = allow_expand;
      storage_owner_maintenance_skipped_busy_.fetch_add(1, std::memory_order_relaxed);
      (void)enqueue_storage_owner_maintenance(std::move(deferred), config);
      return true;
    }

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
    repair_candidates.reserve(
      std::min(candidate_limit, current_neighbors.size() + candidate_ptrs.size()));
    append_unique_all(repair_candidates, current_neighbors, candidate_limit, target_ptr);
    append_unique_all(repair_candidates, candidate_ptrs, candidate_limit, target_ptr);

    if (allow_expand && config.storage_owner_maintenance_max_expansions > 0) {
      vec<RemotePtr> expansion_roots;
      expansion_roots.reserve(config.storage_owner_maintenance_max_expansions);
      append_unique_all(expansion_roots,
                        current_neighbors,
                        config.storage_owner_maintenance_max_expansions,
                        target_ptr);
      append_unique_all(expansion_roots,
                        candidate_ptrs,
                        config.storage_owner_maintenance_max_expansions,
                        target_ptr);
      for (const RemotePtr& root : expansion_roots) {
        if (repair_candidates.size() >= candidate_limit) {
          break;
        }
        append_unique_all(repair_candidates, read_neighbor_list(root), candidate_limit, target_ptr);
      }
      storage_owner_maintenance_expand_repairs_.fetch_add(1, std::memory_order_relaxed);
    }

    hashset_t<RemotePtr> skip;
    skip.insert(target_ptr);
    vec<RemotePtr> repaired = robust_prune_cpu(target_snapshot.vector_data.data(),
                                               VamanaNode::vector_dtype(),
                                               repair_candidates,
                                               skip,
                                               config,
                                               nullptr,
                                               static_cast<u32>(candidate_limit));

    static std::atomic<u64> audit_sequence{0};
    const u64 sequence = audit_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    if (config.storage_owner_maintenance_exact_audit_rate != 0 &&
        sequence % config.storage_owner_maintenance_exact_audit_rate == 0) {
      RemotePtr medoid = read_global_medoid();
      if (!medoid.is_null()) {
        vec<element_t> query = decode_storage_vector_to_float(
          target_snapshot.vector_data.data(), VamanaNode::vector_dtype(), VamanaNode::DIM);
        vec<RemotePtr> exact_candidates =
          beam_search_candidates(span<const element_t>{query.data(), query.size()}, medoid, config, nullptr);
        vec<RemotePtr> exact_repaired = robust_prune_cpu(target_snapshot.vector_data.data(),
                                                         VamanaNode::vector_dtype(),
                                                         exact_candidates,
                                                         skip,
                                                         config,
                                                         nullptr,
                                                         static_cast<u32>(candidate_limit));
        if (!exact_repaired.empty() &&
            storage_owner_candidate_overlap(repaired, exact_repaired, config.R) <
              config.storage_owner_maintenance_audit_min_overlap) {
          repaired = std::move(exact_repaired);
        }
      }
    }

    if (!try_lock_node(target_ptr)) {
      StorageOwnerMaintenanceTask deferred;
      deferred.target = target_ptr;
      deferred.candidates = candidate_ptrs;
      deferred.dirty_count = 1;
      deferred.expand_hint = allow_expand;
      storage_owner_maintenance_skipped_busy_.fetch_add(1, std::memory_order_relaxed);
      (void)enqueue_storage_owner_maintenance(std::move(deferred), config);
      return true;
    }

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
    if (!same_neighbors(current_neighbors, latest_neighbors)) {
      if (attempt == 0) {
        unlock_node(target_ptr);
        continue;
      }
      vec<RemotePtr> merged_candidates;
      merged_candidates.reserve(candidate_limit);
      append_unique_all(merged_candidates, latest_neighbors, candidate_limit, target_ptr);
      append_unique_all(merged_candidates, candidate_ptrs, candidate_limit, target_ptr);
      append_unique_all(merged_candidates, repaired, candidate_limit, target_ptr);
      repaired = robust_prune_cpu(latest_snapshot.vector_data.data(),
                                  VamanaNode::vector_dtype(),
                                  merged_candidates,
                                  skip,
                                  config,
                                  nullptr,
                                  static_cast<u32>(candidate_limit));
    }

    vec<RemotePtr> added_neighbors;
    if (!same_neighbors(latest_neighbors, repaired)) {
      write_neighbor_list(target_ptr, repaired);
      storage_owner_maintenance_local_repairs_.fetch_add(1, std::memory_order_relaxed);
      for (const RemotePtr& neighbor : repaired) {
        if (!contains_ptr(latest_neighbors, neighbor)) {
          added_neighbors.push_back(neighbor);
        }
      }
    }
    unlock_node(target_ptr);

    if (!added_neighbors.empty()) {
      std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_dirty;
      for (const RemotePtr& neighbor : added_neighbors) {
        if (local_shard(neighbor.memory_node())) {
          vec<RemotePtr> reverse_candidate{target_ptr};
          (void)enqueue_reverse_edge_repair(neighbor, reverse_candidate, config);
        } else {
          remote_dirty[neighbor.memory_node()].push_back(
            service::storage_owner::ReverseUpdateOp{neighbor.raw_address, target_ptr.raw_address});
        }
      }
      for (auto& [target_shard, ops] : remote_dirty) {
        (void)send_maintenance_dirty_batch(target_shard, ops, config);
      }
    }
    return true;
  }
  return true;
}

bool MemoryNode::repair_inserted_storage_owner_node(RemotePtr target_ptr,
                                                    const vec<RemotePtr>& candidates,
                                                    const Configuration& config) {
  return repair_storage_owner_neighbors(target_ptr, candidates, false, config);
}
