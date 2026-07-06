#include "memory_node/memory_node.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <unordered_map>

#include "vamana/hot_graph.hh"
#include "vamana/storage_layout_resolver.hh"

namespace {

constexpr u64 kMaintenanceObservationPeriodNs = 5ull * 1000ull * 1000ull * 1000ull;
constexpr size_t kForegroundQueueYieldMultiplier = 2;

u64 steady_now_ns() {
  return static_cast<u64>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
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

void update_max_relaxed(std::atomic<u64>& target, u64 value) {
  u64 observed = target.load(std::memory_order_relaxed);
  while (observed < value &&
         !target.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {
  }
}

double ratio_or_zero(u64 numerator, u64 denominator) {
  if (denominator == 0) {
    return 0.0;
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

bool queue_near_limit(size_t size, size_t limit) {
  if (limit == 0) {
    return size != 0;
  }
  const size_t threshold = std::max<size_t>(1, (limit * 3) / 4);
  return size >= threshold;
}

bool counter_above_fraction(u32 value, u32 limit, u32 numerator, u32 denominator) {
  const u32 threshold = std::max<u32>(1, (limit * numerator) / denominator);
  return value >= threshold;
}

struct CounterReleaseGuard {
  explicit CounterReleaseGuard(std::atomic<u32>& counter) : counter(counter) {}
  ~CounterReleaseGuard() {
    counter.fetch_sub(1, std::memory_order_acq_rel);
  }

  std::atomic<u32>& counter;
};

}  // namespace

bool MemoryNode::storage_owner_maintenance_enabled(const Configuration& config) {
  return config.storage_owner_maintenance_mode == "finalize" &&
         config.storage_owner_maintenance_workers > 0;
}

void MemoryNode::start_storage_owner_maintenance_runtime(const Configuration& config) {
  if (!use_storage_owner_insert_ || !storage_owner_maintenance_enabled(config)) {
    return;
  }

  storage_owner_maintenance_shutdown_.store(false, std::memory_order_release);
  storage_owner_maintenance_enqueued_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_enqueued_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_cleanup_enqueued_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_processed_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalized_live_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_failed_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_stale_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_cleanup_processed_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_max_backlog_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_pressure_yields_.store(0, std::memory_order_relaxed);
  storage_owner_stitch_external_requests_.store(0, std::memory_order_relaxed);
  storage_owner_stitch_external_candidates_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_active_workers_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_latency_ns_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_max_latency_ns_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_started_ns_.store(steady_now_ns(), std::memory_order_release);
  storage_owner_maintenance_last_observation_ns_.store(0, std::memory_order_relaxed);
  const u32 worker_count = std::max<u32>(1, config.storage_owner_maintenance_workers);
  const size_t snapshot_stride = memory_node_detail::storage_owner_snapshot_stride();
  const size_t neighbor_stride = align_up(VamanaNode::neighbor_read_size());
  const size_t snapshot_batch = std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  const size_t coroutine_scratch_stride =
    align_up(snapshot_stride * snapshot_batch +
             std::max(VamanaNode::total_size(), neighbor_stride));
  const size_t scratch_bytes =
    std::max<size_t>(64ull * 1024ull * 1024ull, coroutine_scratch_stride);

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

  print_status("storage-owner maintenance workers: " + std::to_string(worker_count) +
               " (configured=" + std::to_string(config.storage_owner_maintenance_workers) +
               ", resource_gated_parallel_stitching=true)");
  print_status("storage-owner maintenance tuning: mode=" + config.storage_owner_maintenance_mode +
               " local_stitch=" + (config.storage_owner_update_mode == "local_stitch" ? "true" : "false"));
}

void MemoryNode::stop_storage_owner_maintenance_runtime() {
  storage_owner_maintenance_shutdown_.store(true, std::memory_order_release);
  storage_owner_maintenance_cv_.notify_all();

  for (auto& worker : storage_owner_maintenance_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }

  size_t remaining = 0;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    remaining = storage_owner_maintenance_tasks_.size();
    storage_owner_maintenance_tasks_.clear();
  }
  storage_owner_maintenance_workers_.clear();
  storage_owner_maintenance_worker_states_.clear();

  log_storage_owner_maintenance_observation(remaining, true);
}

void MemoryNode::log_storage_owner_maintenance_observation(size_t remaining, bool final) {
  if (storage_owner_maintenance_enqueued_.load(std::memory_order_relaxed) == 0) {
    return;
  }

  const u64 finalize_enqueued =
    storage_owner_maintenance_finalize_enqueued_.load(std::memory_order_relaxed);
  const u64 cleanup_enqueued =
    storage_owner_maintenance_cleanup_enqueued_.load(std::memory_order_relaxed);
  const u64 finalized_live =
    storage_owner_maintenance_finalized_live_.load(std::memory_order_relaxed);
  const u64 stale =
    storage_owner_maintenance_stale_.load(std::memory_order_relaxed);
  const u64 live_required = finalize_enqueued > stale ? finalize_enqueued - stale : 0;
  const u64 total_finalize_latency_ns =
    storage_owner_maintenance_finalize_latency_ns_.load(std::memory_order_relaxed);
  const u64 max_finalize_latency_ns =
    storage_owner_maintenance_finalize_max_latency_ns_.load(std::memory_order_relaxed);
  const u64 started_ns =
    storage_owner_maintenance_started_ns_.load(std::memory_order_acquire);
  const u64 elapsed_ns = started_ns == 0 ? 0 : steady_now_ns() - started_ns;
  const double elapsed_s = static_cast<double>(elapsed_ns) / 1e9;
  const double repair_rate = elapsed_s > 0.0
                               ? static_cast<double>(finalized_live) / elapsed_s
                               : 0.0;
  const double avg_finalize_ms = finalized_live == 0
                                   ? 0.0
                                   : static_cast<double>(total_finalize_latency_ns) /
                                       static_cast<double>(finalized_live) / 1e6;

  print_status(str("storage-owner maintenance ") + (final ? "summary" : "observation") +
               ": enqueued=" +
               std::to_string(storage_owner_maintenance_enqueued_.load(std::memory_order_relaxed)) +
               " stitch_enqueued=" +
               std::to_string(finalize_enqueued) +
               " cleanup_enqueued=" +
               std::to_string(cleanup_enqueued) +
               " stitch_tasks_done=" +
               std::to_string(storage_owner_maintenance_processed_.load(std::memory_order_relaxed)) +
               " stitched_live=" +
               std::to_string(finalized_live) +
               " cleanup_processed=" +
               std::to_string(storage_owner_maintenance_cleanup_processed_.load(std::memory_order_relaxed)) +
               " stale=" +
               std::to_string(stale) +
               " stitch_completion_ratio=" +
               std::to_string(ratio_or_zero(finalized_live, finalize_enqueued)) +
               " live_stitch_completion_ratio=" +
               std::to_string(ratio_or_zero(finalized_live, live_required)) +
               " avg_stitch_delay_ms=" +
               std::to_string(avg_finalize_ms) +
               " max_stitch_delay_ms=" +
               std::to_string(static_cast<double>(max_finalize_latency_ns) / 1e6) +
               " stitch_rate_per_sec=" +
               std::to_string(repair_rate) +
               " failed=" +
               std::to_string(storage_owner_maintenance_failed_.load(std::memory_order_relaxed)) +
               " max_backlog=" +
               std::to_string(storage_owner_maintenance_max_backlog_.load(std::memory_order_relaxed)) +
               " pressure_yields=" +
               std::to_string(storage_owner_maintenance_pressure_yields_.load(std::memory_order_relaxed)) +
               " external_search_requests=" +
               std::to_string(storage_owner_stitch_external_requests_.load(std::memory_order_relaxed)) +
               " external_candidates=" +
               std::to_string(storage_owner_stitch_external_candidates_.load(std::memory_order_relaxed)) +
               " remaining=" + std::to_string(remaining));
}

void MemoryNode::maybe_log_storage_owner_maintenance_observation() {
  const u64 now = steady_now_ns();
  u64 last = storage_owner_maintenance_last_observation_ns_.load(std::memory_order_acquire);
  while (now - last >= kMaintenanceObservationPeriodNs) {
    if (storage_owner_maintenance_last_observation_ns_.compare_exchange_weak(
          last, now, std::memory_order_acq_rel, std::memory_order_acquire)) {
      size_t remaining = 0;
      {
        std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
        remaining = storage_owner_maintenance_tasks_.size();
      }
      log_storage_owner_maintenance_observation(remaining, false);
      return;
    }
  }
}

bool MemoryNode::enqueue_storage_owner_maintenance(StorageOwnerMaintenanceTask&& task,
                                                   const Configuration& config) {
  if (!storage_owner_maintenance_enabled(config) || task.target.is_null()) {
    return false;
  }

  task.queued_at = std::chrono::steady_clock::now();
  size_t backlog = 0;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
      return false;
    }
    storage_owner_maintenance_tasks_.push_back(std::move(task));
    backlog = storage_owner_maintenance_tasks_.size();
  }
  storage_owner_maintenance_enqueued_.fetch_add(1, std::memory_order_relaxed);
  update_max_relaxed(storage_owner_maintenance_max_backlog_, static_cast<u64>(backlog));
  storage_owner_maintenance_cv_.notify_one();
  return true;
}

bool MemoryNode::enqueue_insert_stitch(node_t id,
                                       u32 generation,
                                       RemotePtr target,
                                       const Configuration& config) {
  StorageOwnerMaintenanceTask task;
  task.kind = StorageOwnerMaintenanceKind::stitch_insert;
  task.id = id;
  task.generation = generation;
  task.target = target;
  const bool queued = enqueue_storage_owner_maintenance(std::move(task), config);
  if (queued) {
    storage_owner_maintenance_finalize_enqueued_.fetch_add(1, std::memory_order_relaxed);
  }
  return queued;
}

bool MemoryNode::enqueue_deleted_node_cleanup(RemotePtr deleted_ptr, const Configuration& config) {
  StorageOwnerMaintenanceTask task;
  task.kind = StorageOwnerMaintenanceKind::cleanup_deleted_node;
  task.target = deleted_ptr;
  const bool queued = enqueue_storage_owner_maintenance(std::move(task), config);
  if (queued) {
    storage_owner_maintenance_cleanup_enqueued_.fetch_add(1, std::memory_order_relaxed);
  }
  return queued;
}

void MemoryNode::mark_storage_owner_foreground_activity() {
  storage_owner_foreground_last_active_ns_.store(steady_now_ns(), std::memory_order_release);
}

bool MemoryNode::storage_owner_maintenance_foreground_busy(const Configuration&) {
  const bool foreground_active =
    storage_owner_insert_active_workers_.load(std::memory_order_acquire) != 0;

  {
    std::unique_lock<std::mutex> lock(storage_insert_tasks_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return true;
    }
    const size_t foreground_queue_yield_threshold =
      std::max<size_t>(4, storage_owner_threads_.size() * kForegroundQueueYieldMultiplier);
    if (storage_insert_tasks_.size() >= foreground_queue_yield_threshold) {
      return true;
    }
  }

  {
    std::unique_lock<std::mutex> lock(peer_reverse_tasks_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return true;
    }
    if (queue_near_limit(peer_reverse_tasks_.size(), peer_reverse_task_queue_limit_)) {
      return true;
    }
  }

  {
    std::unique_lock<std::mutex> lock(peer_reverse_outgoing_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return true;
    }
    if (queue_near_limit(peer_reverse_outgoing_.size(), peer_reverse_outgoing_queue_limit_)) {
      return true;
    }
  }

  if (peer_context_) {
    poll_peer_send_cq();
    const u32 pressure_num = foreground_active ? 1 : 3;
    const u32 pressure_den = foreground_active ? 2 : 4;
    if (counter_above_fraction(peer_async_rdma_outstanding_.load(std::memory_order_acquire),
                               peer_rdma_read_global_credit_limit(),
                               pressure_num,
                               pressure_den)) {
      return true;
    }
    const u32 per_peer_limit = peer_rdma_read_credit_limit();
    for (const auto& counter : peer_rdma_read_outstanding_) {
      if (counter_above_fraction(counter.load(std::memory_order_acquire),
                                 per_peer_limit,
                                 pressure_num,
                                 pressure_den)) {
        return true;
      }
    }
  }

  return false;
}

bool MemoryNode::try_acquire_storage_owner_maintenance_slot(const Configuration& config) {
  if (storage_owner_maintenance_foreground_busy(config)) {
    return false;
  }

  const u32 max_workers = std::max<u32>(1, config.storage_owner_maintenance_workers);
  u32 active = storage_owner_maintenance_active_workers_.load(std::memory_order_acquire);
  while (active < max_workers) {
    if (storage_owner_maintenance_active_workers_.compare_exchange_weak(
          active, active + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
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

  for (;;) {
    StorageOwnerMaintenanceTask task;
    {
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait(lock, [&]() {
        return storage_owner_maintenance_shutdown_.load(std::memory_order_acquire) ||
               !storage_owner_maintenance_tasks_.empty();
      });
      if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
    }

    maybe_log_storage_owner_maintenance_observation();
    if (!try_acquire_storage_owner_maintenance_slot(config)) {
      storage_owner_maintenance_pressure_yields_.fetch_add(1, std::memory_order_relaxed);
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(lock, std::chrono::milliseconds(1), [&]() {
        return storage_owner_maintenance_shutdown_.load(std::memory_order_acquire);
      });
      continue;
    }
    CounterReleaseGuard active_slot(storage_owner_maintenance_active_workers_);

    {
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
      if (storage_owner_maintenance_tasks_.empty()) {
        continue;
      }
      if (storage_owner_maintenance_foreground_busy(config)) {
        storage_owner_maintenance_pressure_yields_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      task = std::move(storage_owner_maintenance_tasks_.front());
      storage_owner_maintenance_tasks_.pop_front();
    }

    bool ok = true;
    switch (task.kind) {
      case StorageOwnerMaintenanceKind::stitch_insert:
        ok = stitch_inserted_storage_owner_node(task, config);
        if (ok) {
          storage_owner_maintenance_processed_.fetch_add(1, std::memory_order_relaxed);
        }
        break;
      case StorageOwnerMaintenanceKind::cleanup_deleted_node:
        ok = cleanup_deleted_storage_owner_node(task, config);
        if (ok) {
          storage_owner_maintenance_cleanup_processed_.fetch_add(1, std::memory_order_relaxed);
        }
        break;
    }
    if (!ok) {
      storage_owner_maintenance_failed_.fetch_add(1, std::memory_order_relaxed);
      if (!storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
        {
          std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
          if (!storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
            storage_owner_maintenance_tasks_.push_back(std::move(task));
            update_max_relaxed(storage_owner_maintenance_max_backlog_,
                               static_cast<u64>(storage_owner_maintenance_tasks_.size()));
          }
        }
        storage_owner_maintenance_cv_.notify_one();
        std::this_thread::yield();
      }
    }
    maybe_log_storage_owner_maintenance_observation();
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
  if (!VamanaNode::compact_storage()) {
    return read_neighbor_list(rptr);
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

  if (VamanaNode::HOT_GRAPH_FORMAT_VERSION >= 2) {
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

  vec<byte_t> decoded(VamanaNode::neighbor_read_size());
  if (!VamanaNode::decode_hot_graph_entry(entry.data(), decoded.data())) {
    return {};
  }
  const u8 edge_count =
    *reinterpret_cast<const u8*>(decoded.data() + VamanaNode::neighbor_count_offset_in_read());
  const auto* slots = reinterpret_cast<const RemotePtr*>(
    decoded.data() + VamanaNode::neighbor_payload_offset_in_read());
  vec<RemotePtr> neighbors;
  neighbors.reserve(edge_count);
  for (u32 i = 0; i < edge_count && i < VamanaNode::R; ++i) {
    if (!slots[i].is_null()) {
      neighbors.push_back(slots[i]);
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

bool MemoryNode::stitch_inserted_storage_owner_node(const StorageOwnerMaintenanceTask& task,
                                                    const Configuration& config) {
  if (!local_shard(task.target.memory_node())) {
    return true;
  }
  if (!storage_owner_task_current(task.id, task.generation, task.target)) {
    storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  NodeSnapshot target_snapshot;
  if (!read_node_snapshot(task.target, target_snapshot)) {
    return false;
  }
  if (target_snapshot.deleted) {
    storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  vec<RemotePtr> candidates = read_neighbor_list(task.target);
  vec<NodeSnapshot> targets;
  targets.push_back(target_snapshot);
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
        return false;
      }
      if (request.item_count != 0) {
        pending.push_back(request);
      }
    }
    for (const PendingStitchRequest& request : pending) {
      vec<RemotePtr> shard_candidates;
      if (!wait_for_peer_stitch_search_response(request.request_id, request.shard,
                                                request.item_count, shard_candidates, config)) {
        return false;
      }
      storage_owner_stitch_external_requests_.fetch_add(1, std::memory_order_relaxed);
      storage_owner_stitch_external_candidates_.fetch_add(shard_candidates.size(),
                                                          std::memory_order_relaxed);
      candidates.insert(candidates.end(), shard_candidates.begin(), shard_candidates.end());
    }
  }

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
    return true;
  }
  NodeSnapshot latest_snapshot;
  if (!read_node_snapshot(task.target, latest_snapshot)) {
    unlock_node(task.target);
    return false;
  }
  if (latest_snapshot.deleted) {
    unlock_node(task.target);
    storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  const vec<RemotePtr> current_neighbors = read_neighbor_list(task.target);
  if (!same_neighbors(current_neighbors, final_neighbors)) {
    write_neighbor_list(task.target, final_neighbors);
  }
  unlock_node(task.target);

  std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_updates;
  for (const RemotePtr& neighbor : final_neighbors) {
    if (local_shard(neighbor.memory_node())) {
      vec<RemotePtr> reverse_candidate{task.target};
      if (!apply_local_reverse_update(neighbor, reverse_candidate, config, false)) {
        return false;
      }
    } else {
      remote_updates[neighbor.memory_node()].push_back(
        service::storage_owner::ReverseUpdateOp{neighbor.raw_address, task.target.raw_address});
    }
  }
  for (auto& [target_shard, ops] : remote_updates) {
    if (!send_reverse_update_batch(target_shard, ops, config)) {
      return false;
    }
  }

  const u64 finalize_latency_ns = static_cast<u64>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - task.queued_at).count());
  storage_owner_maintenance_finalized_live_.fetch_add(1, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_latency_ns_.fetch_add(finalize_latency_ns,
                                                           std::memory_order_relaxed);
  update_max_relaxed(storage_owner_maintenance_finalize_max_latency_ns_, finalize_latency_ns);
  return true;
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
  std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_cleanup;
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
  return ok;
}
