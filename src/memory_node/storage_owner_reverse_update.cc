#include "memory_node/memory_node.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

namespace {

using Configuration = configuration::IndexConfiguration;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;

}  // namespace

void MemoryNode::start_storage_owner_repair_runtime(const Configuration& config) {
  if (!use_storage_owner_insert_ ||
      config.storage_owner_repair_workers == 0 ||
      config.storage_owner_repair_budget_us == 0) {
    return;
  }

  storage_owner_repair_shutdown_.store(false, std::memory_order_release);
  storage_owner_repair_queue_limit_ =
    std::max<size_t>(1, static_cast<size_t>(config.storage_owner_repair_queue_depth));

  const u32 worker_count = std::max<u32>(1, config.storage_owner_repair_workers);
  const size_t snapshot_stride = align_up(VamanaNode::vector_bytes());
  const size_t neighbor_stride = align_up(VamanaNode::neighbor_read_size());
  const size_t coroutine_scratch_stride =
    align_up(std::max<size_t>(VamanaNode::total_size(),
                              std::max(neighbor_stride,
                                       snapshot_stride *
                                         std::max<u32>(1, config.storage_owner_search_snapshot_batch))));
  const size_t scratch_bytes = std::max<size_t>(64ull * 1024ull * 1024ull, coroutine_scratch_stride);
  storage_owner_repair_worker_states_.reserve(worker_count);
  for (u32 i = 0; i < worker_count; ++i) {
    auto worker = std::make_unique<StorageOwnerThread>(i, 1, config.max_send_queue_wr);
    if (peer_context_) {
      worker->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
    }
    storage_owner_repair_worker_states_.push_back(std::move(worker));
  }
  for (u32 i = 0; i < worker_count; ++i) {
    storage_owner_repair_workers_.emplace_back([this, i]() { storage_owner_repair_worker_loop(i); });
  }
  print_status("storage-owner ALDI repair workers: " + std::to_string(worker_count) +
               " queue_depth=" + std::to_string(storage_owner_repair_queue_limit_));
}

void MemoryNode::stop_storage_owner_repair_runtime() {
  storage_owner_repair_shutdown_.store(true, std::memory_order_release);
  storage_owner_repair_cv_.notify_all();
  for (auto& worker : storage_owner_repair_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  storage_owner_repair_workers_.clear();
  storage_owner_repair_worker_states_.clear();
  {
    std::lock_guard<std::mutex> lock(storage_owner_repair_mutex_);
    storage_owner_repair_order_.clear();
    storage_owner_repair_candidates_.clear();
  }
}

bool MemoryNode::enqueue_storage_owner_repair(RemotePtr target_ptr,
                                              const vec<RemotePtr>& candidate_ptrs,
                                              const Configuration& config) {
  if (target_ptr.is_null() || !local_shard(target_ptr.memory_node()) ||
      candidate_ptrs.empty() ||
      config.storage_owner_repair_workers == 0 ||
      config.storage_owner_repair_budget_us == 0) {
    return false;
  }

  std::lock_guard<std::mutex> lock(storage_owner_repair_mutex_);
  const u64 target_raw = target_ptr.raw_address;
  auto it = storage_owner_repair_candidates_.find(target_raw);
  if (it == storage_owner_repair_candidates_.end()) {
    if (storage_owner_repair_order_.size() >= storage_owner_repair_queue_limit_) {
      return false;
    }
    storage_owner_repair_order_.push_back(target_raw);
    it = storage_owner_repair_candidates_.emplace(target_raw, vec<RemotePtr>{}).first;
  }

  vec<RemotePtr>& queued = it->second;
  const size_t cap = std::max<size_t>(config.R, config.storage_owner_pending_edge_cap);
  for (const RemotePtr candidate : candidate_ptrs) {
    if (candidate.is_null() || candidate == target_ptr) {
      continue;
    }
    if (std::find(queued.begin(), queued.end(), candidate) == queued.end()) {
      queued.push_back(candidate);
      if (queued.size() >= cap) {
        break;
      }
    }
  }
  storage_owner_repair_cv_.notify_one();
  return true;
}

void MemoryNode::storage_owner_repair_worker_loop(u32 worker_id) {
  current_storage_owner_thread_ = worker_id < storage_owner_repair_worker_states_.size()
                                    ? storage_owner_repair_worker_states_[worker_id].get()
                                    : nullptr;
  const Configuration& config = *storage_worker_config_;
  const auto budget = std::chrono::microseconds(config.storage_owner_repair_budget_us);
  auto window_started = std::chrono::steady_clock::now();
  for (;;) {
    StorageOwnerRepairTask task;
    {
      std::unique_lock<std::mutex> lock(storage_owner_repair_mutex_);
      storage_owner_repair_cv_.wait(lock, [&]() {
        return storage_owner_repair_shutdown_.load(std::memory_order_acquire) ||
               !storage_owner_repair_order_.empty();
      });
      if (storage_owner_repair_shutdown_.load(std::memory_order_acquire) &&
          storage_owner_repair_order_.empty()) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
      const u64 target_raw = storage_owner_repair_order_.front();
      storage_owner_repair_order_.pop_front();
      auto it = storage_owner_repair_candidates_.find(target_raw);
      if (it == storage_owner_repair_candidates_.end()) {
        continue;
      }
      task.target = RemotePtr{target_raw};
      task.candidates = std::move(it->second);
      storage_owner_repair_candidates_.erase(it);
    }

    (void)repair_storage_owner_neighbors(task.target, task.candidates, config);
    if (budget.count() > 0 &&
        std::chrono::steady_clock::now() - window_started >= budget) {
      std::this_thread::yield();
      window_started = std::chrono::steady_clock::now();
    }
  }
}

bool MemoryNode::repair_storage_owner_neighbors(RemotePtr target_ptr,
                                                const vec<RemotePtr>& candidate_ptrs,
                                                const Configuration& config) {
  if (target_ptr.is_null() || !local_shard(target_ptr.memory_node()) || candidate_ptrs.empty()) {
    return true;
  }

  lock_node(target_ptr);
  NodeSnapshot target_snapshot;
  read_node_snapshot(target_ptr, target_snapshot);
  if (target_snapshot.deleted) {
    unlock_node(target_ptr);
    return true;
  }

  vec<RemotePtr> repair_candidates = read_neighbor_list(target_ptr);
  repair_candidates.reserve(repair_candidates.size() + candidate_ptrs.size());
  for (const RemotePtr candidate : candidate_ptrs) {
    if (candidate.is_null() || candidate == target_ptr) {
      continue;
    }
    if (std::find(repair_candidates.begin(), repair_candidates.end(), candidate) ==
        repair_candidates.end()) {
      repair_candidates.push_back(candidate);
    }
  }

  hashset_t<RemotePtr> skip;
  skip.insert(target_ptr);
  const vec<RemotePtr> repaired = repair_candidates.size() <= config.R
                                    ? repair_candidates
                                    : robust_prune_cpu(target_snapshot.vector_data.data(),
                                                       VamanaNode::vector_dtype(),
                                                       repair_candidates,
                                                       skip,
                                                       config,
                                                       nullptr);
  write_neighbor_list(target_ptr, repaired);
  unlock_node(target_ptr);
  return true;
}

bool MemoryNode::apply_local_reverse_update(RemotePtr target_ptr,
                                const vec<RemotePtr>& candidate_ptrs,
                                const Configuration& config) {
  lib_assert(local_shard(target_ptr.memory_node()), "target reverse update must be local");
  if (candidate_ptrs.empty()) {
    return true;
  }

  const auto update_started = std::chrono::steady_clock::now();
  const auto lock_started = std::chrono::steady_clock::now();
  lock_node(target_ptr);
  const u64 lock_wait_ns = elapsed_ns_since(lock_started);
  vec<RemotePtr> updated_neighbors;
  bool changed = false;
  bool pruned = false;
  size_t current_count = 0;
  size_t filtered_count = 0;
  u64 snapshot_ns = 0;
  u64 neighbor_read_ns = 0;
  u64 filter_ns = 0;
  u64 prune_ns = 0;
  u64 write_ns = 0;

  NodeSnapshot target_snapshot;
  auto step_started = std::chrono::steady_clock::now();
  read_node_snapshot(target_ptr, target_snapshot);
  snapshot_ns = elapsed_ns_since(step_started);
  if (target_snapshot.deleted) {
    unlock_node(target_ptr);
    return true;
  }

  step_started = std::chrono::steady_clock::now();
  vec<RemotePtr> current_neighbors = read_neighbor_list(target_ptr);
  neighbor_read_ns = elapsed_ns_since(step_started);
  current_count = current_neighbors.size();

  step_started = std::chrono::steady_clock::now();
  vec<RemotePtr> filtered_candidates;
  filtered_candidates.reserve(candidate_ptrs.size());
  for (const RemotePtr& candidate_ptr : candidate_ptrs) {
    if (candidate_ptr.is_null()) {
      continue;
    }
    bool already_present = false;
    for (const RemotePtr& existing : current_neighbors) {
      if (existing == candidate_ptr) {
        already_present = true;
        break;
      }
    }
    if (!already_present &&
        std::find(filtered_candidates.begin(), filtered_candidates.end(), candidate_ptr) == filtered_candidates.end()) {
      filtered_candidates.push_back(candidate_ptr);
    }
  }
  filter_ns = elapsed_ns_since(step_started);
  filtered_count = filtered_candidates.size();

  if (!filtered_candidates.empty()) {
    changed = true;

    if (current_neighbors.size() + filtered_candidates.size() <= config.R) {
      current_neighbors.insert(current_neighbors.end(), filtered_candidates.begin(), filtered_candidates.end());
      updated_neighbors = std::move(current_neighbors);
    } else {
      vec<RemotePtr> repair_candidates;
      repair_candidates.reserve(current_neighbors.size() + filtered_candidates.size());
      for (const RemotePtr& ptr : current_neighbors) {
        if (!ptr.is_null() &&
            std::find(repair_candidates.begin(), repair_candidates.end(), ptr) == repair_candidates.end()) {
          repair_candidates.push_back(ptr);
        }
      }
      for (const RemotePtr& ptr : filtered_candidates) {
        if (!ptr.is_null() &&
            std::find(repair_candidates.begin(), repair_candidates.end(), ptr) == repair_candidates.end()) {
          repair_candidates.push_back(ptr);
        }
      }
      (void)enqueue_storage_owner_repair(target_ptr, repair_candidates, config);

      // Evict-farthest: for each new candidate, compute distance from target
      // and replace the farthest existing neighbor if the candidate is closer.
      // This is O(R) distance calls per candidate instead of O(R²) pair distances
      // from full RobustPrune, trading a small diversity loss for large speedup.
      pruned = true;
      step_started = std::chrono::steady_clock::now();

      // 1. Collect non-null current neighbors (do this once, reuse below)
      vec<RemotePtr> non_null_neighbors;
      non_null_neighbors.reserve(current_neighbors.size());
      for (const auto& n : current_neighbors) {
        if (!n.is_null()) non_null_neighbors.push_back(n);
      }

      // 2. Batch-read all current neighbor snapshots + compute distances (O(R), SIMD)
      vec<distance_t> neighbor_dists;
      neighbor_dists.reserve(non_null_neighbors.size());
      if (!non_null_neighbors.empty()) {
        vec<NodeSnapshot> neighbor_snapshots =
            read_node_snapshots_batched(non_null_neighbors, config);
        for (const auto& snap : neighbor_snapshots) {
          neighbor_dists.push_back(distance_between_vectors(
              target_snapshot.vector_data.data(), VamanaNode::vector_dtype(),
              snap.vector_data.data(), VamanaNode::vector_dtype(), config));
        }
      }

      // 3. Initialise updated_neighbors from filtered list (no extra allocation)
      updated_neighbors = std::move(non_null_neighbors);

      // 4. For each candidate, evict farthest if candidate is closer
      {
        vec<RemotePtr> non_null_candidates;
        non_null_candidates.reserve(filtered_candidates.size());
        for (const auto& c : filtered_candidates) {
          if (!c.is_null()) non_null_candidates.push_back(c);
        }

        vec<NodeSnapshot> candidate_snapshots;
        if (!non_null_candidates.empty()) {
          candidate_snapshots = read_node_snapshots_batched(non_null_candidates, config);
        }

        for (size_t ci = 0; ci < candidate_snapshots.size(); ++ci) {
          const auto& cand_snap = candidate_snapshots[ci];
          const distance_t cand_dist = distance_between_vectors(
              target_snapshot.vector_data.data(), VamanaNode::vector_dtype(),
              cand_snap.vector_data.data(), VamanaNode::vector_dtype(), config);

          if (updated_neighbors.size() < config.R) {
            updated_neighbors.push_back(cand_snap.rptr);
            neighbor_dists.push_back(cand_dist);
          } else {
            // updated_neighbors.size() >= R, and neighbor_dists tracks the same
            // set, so at least one element exists.
            lib_assert(!neighbor_dists.empty(),
                       "neighbor_dists non-empty when updated_neighbors >= R");
            size_t farthest_idx = 0;
            distance_t farthest_dist = neighbor_dists[0];
            for (size_t j = 1; j < neighbor_dists.size(); ++j) {
              if (neighbor_dists[j] > farthest_dist) {
                farthest_dist = neighbor_dists[j];
                farthest_idx = j;
              }
            }
            if (cand_dist < farthest_dist) {
              updated_neighbors[farthest_idx] = cand_snap.rptr;
              neighbor_dists[farthest_idx] = cand_dist;
            }
          }
        }
      }

      prune_ns = elapsed_ns_since(step_started);
    }
  }

  if (changed) {
    step_started = std::chrono::steady_clock::now();
    write_neighbor_list(target_ptr, updated_neighbors);
    write_ns = elapsed_ns_since(step_started);
  }
  unlock_node(target_ptr);

  const u64 update_ns = elapsed_ns_since(update_started);
  if (update_ns > 1000ull * 1000ull * 1000ull) {
    static std::atomic<u32> slow_update_logs{0};
    const u32 log_index = slow_update_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] slow reverse-update target"
                << " self_shard=" << storage_id_
                << " target_raw=" << target_ptr.raw_address
                << " candidates=" << candidate_ptrs.size()
                << " current_neighbors=" << current_count
                << " filtered_candidates=" << filtered_count
                << " changed=" << (changed ? 1 : 0)
                << " pruned=" << (pruned ? 1 : 0)
                << " elapsed_ms=" << (update_ns / 1000000.0)
                << " lock_wait_ms=" << (lock_wait_ns / 1000000.0)
                << " snapshot_ms=" << (snapshot_ns / 1000000.0)
                << " neighbor_read_ms=" << (neighbor_read_ns / 1000000.0)
                << " filter_ms=" << (filter_ns / 1000000.0)
                << " prune_ms=" << (prune_ns / 1000000.0)
                << " write_ms=" << (write_ns / 1000000.0)
                << std::endl;
    }
  }
  return true;
}
