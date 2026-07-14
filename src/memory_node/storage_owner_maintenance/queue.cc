#include "memory_node/storage_owner_maintenance/detail.hh"

using namespace memory_node_storage_owner_maintenance_detail;

bool MemoryNode::enqueue_storage_owner_maintenance(StorageOwnerMaintenanceTask&& task,
                                                   const Configuration& config) {
  if (!storage_owner_maintenance_enabled(config) || task.target.is_null()) {
    return false;
  }

  task.queued_at = std::chrono::steady_clock::now();
  size_t backlog = 0;
  {
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_cv_.wait(lock, [&]() {
      return storage_owner_maintenance_shutdown_.load(std::memory_order_acquire) ||
             storage_owner_stitch_tasks_.size() + storage_owner_cleanup_tasks_.size() <
               config.storage_owner_maintenance_queue_depth;
    });
    if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
      return false;
    }
    if (task.kind == StorageOwnerMaintenanceKind::stitch_insert) {
      storage_owner_stitch_tasks_.push_back(std::move(task));
    } else {
      storage_owner_cleanup_tasks_.push_back(std::move(task));
    }
    backlog = storage_owner_stitch_tasks_.size() + storage_owner_cleanup_tasks_.size();
  }
  storage_owner_maintenance_enqueued_.fetch_add(1, std::memory_order_relaxed);
  atomic_utils::update_max_relaxed(
    storage_owner_maintenance_max_backlog_, static_cast<u64>(backlog));
  storage_owner_maintenance_cv_.notify_one();
  return true;
}

bool MemoryNode::enqueue_insert_stitch(node_t id,
                                       u32 generation,
                                       RemotePtr target,
                                       u64 maintenance_sequence,
                                       const Configuration& config) {
  StorageOwnerMaintenanceTask task;
  task.kind = StorageOwnerMaintenanceKind::stitch_insert;
  task.id = id;
  task.generation = generation;
  task.maintenance_sequence = maintenance_sequence;
  task.target = target;
  const bool queued = enqueue_storage_owner_maintenance(std::move(task), config);
  if (queued) {
    storage_owner_maintenance_finalize_enqueued_.fetch_add(1, std::memory_order_relaxed);
  }
  return queued;
}

bool MemoryNode::enqueue_deleted_node_cleanup(RemotePtr deleted_ptr,
                                              u64 maintenance_sequence,
                                              const Configuration& config) {
  StorageOwnerMaintenanceTask task;
  task.kind = StorageOwnerMaintenanceKind::cleanup_deleted_node;
  task.maintenance_sequence = maintenance_sequence;
  task.target = deleted_ptr;
  const bool queued = enqueue_storage_owner_maintenance(std::move(task), config);
  if (queued) {
    storage_owner_maintenance_cleanup_enqueued_.fetch_add(1, std::memory_order_relaxed);
  }
  return queued;
}

u64 MemoryNode::begin_storage_owner_maintenance_sequence(u32 work_items) {
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  lib_assert(control->magic == gpu_search::format::kStorageControlMagic &&
               control->version == gpu_search::format::kStorageControlVersion,
             "storage-owner maintenance control block is not initialized");
  std::atomic_ref<u64> next_sequence(control->next_maintenance_sequence);
  const u64 sequence = next_sequence.fetch_add(1, std::memory_order_acq_rel);
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_sequence_mutex_);
    const auto [iterator, inserted] =
      storage_owner_maintenance_sequence_remaining_.emplace(sequence, work_items);
    lib_assert(inserted && iterator->second == work_items,
               "duplicate storage-owner maintenance sequence");
    advance_storage_owner_durable_sequence_locked();
  }
  return sequence;
}

void MemoryNode::advance_storage_owner_durable_sequence_locked() {
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  std::atomic_ref<u64> durable(control->durable_maintenance_sequence);
  u64 watermark = durable.load(std::memory_order_acquire);
  for (;;) {
    const auto iterator = storage_owner_maintenance_sequence_remaining_.find(watermark + 1);
    if (iterator == storage_owner_maintenance_sequence_remaining_.end() ||
        iterator->second != 0) {
      break;
    }
    storage_owner_maintenance_sequence_remaining_.erase(iterator);
    ++watermark;
  }
  durable.store(watermark, std::memory_order_release);
}

void MemoryNode::complete_storage_owner_maintenance_sequence(u64 sequence) {
  if (sequence == 0) return;
  std::lock_guard<std::mutex> lock(storage_owner_maintenance_sequence_mutex_);
  const auto iterator = storage_owner_maintenance_sequence_remaining_.find(sequence);
  lib_assert(iterator != storage_owner_maintenance_sequence_remaining_.end() &&
               iterator->second != 0,
             "invalid storage-owner maintenance completion sequence");
  --iterator->second;
  advance_storage_owner_durable_sequence_locked();
}

u64 MemoryNode::schedule_storage_owner_maintenance(
    node_t id,
    u32 generation,
    service::storage_owner::MutationKind kind,
    RemotePtr new_ptr,
    RemotePtr old_ptr,
    const Configuration& config) {
  const bool maintenance_enabled = storage_owner_maintenance_enabled(config);
  const bool needs_stitch = maintenance_enabled &&
    kind != service::storage_owner::MutationKind::erase && !new_ptr.is_null();
  const bool needs_cleanup = maintenance_enabled &&
    kind != service::storage_owner::MutationKind::insert && !old_ptr.is_null();
  const u64 sequence = begin_storage_owner_maintenance_sequence(
    static_cast<u32>(needs_stitch) + static_cast<u32>(needs_cleanup));
  if (needs_stitch &&
      !enqueue_insert_stitch(id, generation, new_ptr, sequence, config)) {
    lib_failure("failed to enqueue storage-owner stitch maintenance");
  }
  if (needs_cleanup &&
      !enqueue_deleted_node_cleanup(old_ptr, sequence, config)) {
    lib_failure("failed to enqueue storage-owner cleanup maintenance");
  }
  return sequence;
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
