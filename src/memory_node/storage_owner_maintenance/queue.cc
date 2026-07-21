#include "memory_node/storage_owner_maintenance/detail.hh"
#include "memory_node/storage_owner_maintenance/admission_policy.hh"
#include "memory_node/storage_owner_maintenance/cleanup_scheduler.hh"

using namespace memory_node_storage_owner_maintenance_detail;

u64 MemoryNode::arm_storage_owner_maintenance(
    StorageOwnerMaintenanceTask&& task, const Configuration& config) {
  vec<StorageOwnerMaintenanceTask> tasks;
  tasks.push_back(std::move(task));
  const u64 sequence = arm_storage_owner_maintenance_batch(tasks, config);
  if (sequence == 0 && !tasks.empty()) {
    task = std::move(tasks.front());
  }
  return sequence;
}

u64 MemoryNode::arm_storage_owner_maintenance_batch(
    vec<StorageOwnerMaintenanceTask>& tasks,
    const Configuration& config) {
  if (!storage_owner_maintenance_enabled(config) || tasks.empty() ||
      tasks.size() > config.storage_owner_maintenance_queue_depth ||
      tasks.size() > storage_owner_maintenance_admission_limit_) {
    return 0;
  }
  for (const StorageOwnerMaintenanceTask& task : tasks) {
    if (task.target.is_null() ||
        task.kind != StorageOwnerMaintenanceKind::finalize_insert) {
      return 0;
    }
  }

  // The physical-home control RPC is one admission transaction.  Reserve all
  // queue permits before touching the completion ring; otherwise concurrent
  // RPCs can each own a partial batch and wait forever for the credit that
  // those same pre-commit tasks are unable to release.
  const size_t task_count = tasks.size();
  {
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_cv_.wait(lock, [&]() {
      return storage_owner_maintenance_shutdown_.load(
               std::memory_order_acquire) ||
        maintenance_queue_batch_permit_available(
          storage_owner_stage2_tasks_.size() +
            storage_owner_cleanup_tasks_.size(),
          storage_owner_maintenance_reserved_slots_,
          task_count,
          config.storage_owner_maintenance_queue_depth);
    });
    if (storage_owner_maintenance_shutdown_.load(
          std::memory_order_acquire)) {
      return 0;
    }
    storage_owner_maintenance_reserved_slots_ += task_count;
  }

  vec<u32> work_items(task_count, 1);
  const u64 first_sequence = begin_storage_owner_maintenance_batch(
    span<const u32>{work_items});
  const auto queued_at = std::chrono::steady_clock::now();
  size_t backlog = 0;
  bool enqueued = false;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    lib_assert(storage_owner_maintenance_reserved_slots_ >= task_count,
               "Stage1 arm lost its reserved maintenance queue batch");
    storage_owner_maintenance_reserved_slots_ -= task_count;
    if (!storage_owner_maintenance_shutdown_.load(
          std::memory_order_acquire)) {
      for (size_t item = 0; item < task_count; ++item) {
        tasks[item].maintenance_sequence =
          first_sequence + static_cast<u64>(item);
        tasks[item].queued_at = queued_at;
        storage_owner_stage2_tasks_.push_back(std::move(tasks[item]));
      }
      backlog = storage_owner_stage2_tasks_.size() +
        storage_owner_cleanup_tasks_.size();
      enqueued = true;
    }
  }
  if (!enqueued) {
    // Shutdown is the only failure after sequence allocation. Finalize every
    // unused ticket synchronously so none can pin the durable watermark.
    for (size_t item = 0; item < task_count; ++item) {
      complete_storage_owner_maintenance_sequence(
        first_sequence + static_cast<u64>(item));
    }
    storage_owner_maintenance_cv_.notify_all();
    return 0;
  }

  storage_owner_maintenance_enqueued_.fetch_add(
    task_count, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_enqueued_.fetch_add(
    task_count, std::memory_order_relaxed);
  atomic_utils::update_max_relaxed(
    storage_owner_maintenance_max_backlog_, static_cast<u64>(backlog));
  storage_owner_maintenance_cv_.notify_all();
  return first_sequence;
}

u64 MemoryNode::activate_storage_owner_cleanup(
    StorageOwnerMaintenanceTask&& task, const Configuration& config) {
  vec<StorageOwnerMaintenanceTask> tasks;
  tasks.push_back(std::move(task));
  const u64 sequence = activate_storage_owner_cleanup_batch(tasks, config);
  if (sequence == 0 && !tasks.empty()) {
    task = std::move(tasks.front());
  }
  return sequence;
}

u64 MemoryNode::activate_storage_owner_cleanup_batch(
    vec<StorageOwnerMaintenanceTask>& tasks,
    const Configuration& config) {
  if (!storage_owner_maintenance_enabled(config) || tasks.empty() ||
      tasks.size() > config.storage_owner_maintenance_queue_depth ||
      tasks.size() > storage_owner_maintenance_admission_limit_) {
    return 0;
  }
  for (const StorageOwnerMaintenanceTask& task : tasks) {
    if (task.target.is_null() ||
        task.kind != StorageOwnerMaintenanceKind::cleanup_deleted_node) {
      return 0;
    }
  }

  const size_t task_count = tasks.size();
  {
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_cv_.wait(lock, [&]() {
      return storage_owner_maintenance_shutdown_.load(
               std::memory_order_acquire) ||
        maintenance_queue_batch_permit_available(
          storage_owner_stage2_tasks_.size() +
            storage_owner_cleanup_tasks_.size(),
          storage_owner_maintenance_reserved_slots_,
          task_count,
          config.storage_owner_maintenance_queue_depth);
    });
    if (storage_owner_maintenance_shutdown_.load(
          std::memory_order_acquire)) {
      return 0;
    }
    storage_owner_maintenance_reserved_slots_ += task_count;
  }

  vec<u32> work_items(task_count, 1);
  const u64 first_sequence = begin_storage_owner_maintenance_batch(
    span<const u32>{work_items});
  const auto queued_at = std::chrono::steady_clock::now();
  size_t backlog = 0;
  bool enqueued = false;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    lib_assert(storage_owner_maintenance_reserved_slots_ >= task_count,
               "cleanup activation lost its reserved maintenance queue batch");
    storage_owner_maintenance_reserved_slots_ -= task_count;
    if (!storage_owner_maintenance_shutdown_.load(
          std::memory_order_acquire)) {
      // The task becomes runnable before the physical node is quiesced or
      // tombstoned. Its worker must first reparent every protected child and
      // may publish DELETED only after those reservations are ACKed.
      for (size_t item = 0; item < task_count; ++item) {
        tasks[item].maintenance_sequence =
          first_sequence + static_cast<u64>(item);
        tasks[item].queued_at = queued_at;
        cleanup_schedule_push(
          storage_owner_cleanup_tasks_, std::move(tasks[item]));
      }
      backlog = storage_owner_stage2_tasks_.size() +
        storage_owner_cleanup_tasks_.size();
      enqueued = true;
    }
  }
  if (!enqueued) {
    for (size_t item = 0; item < task_count; ++item) {
      complete_storage_owner_maintenance_sequence(
        first_sequence + static_cast<u64>(item));
    }
    storage_owner_maintenance_cv_.notify_all();
    return 0;
  }

  storage_owner_maintenance_enqueued_.fetch_add(
    task_count, std::memory_order_relaxed);
  storage_owner_maintenance_cleanup_enqueued_.fetch_add(
    task_count, std::memory_order_relaxed);
  atomic_utils::update_max_relaxed(
    storage_owner_maintenance_max_backlog_, static_cast<u64>(backlog));
  storage_owner_maintenance_cv_.notify_all();
  return first_sequence;
}

u64 MemoryNode::begin_storage_owner_maintenance_sequence(u32 work_items) {
  return begin_storage_owner_maintenance_batch(
    span<const u32>{&work_items, 1});
}

u64 MemoryNode::begin_storage_owner_maintenance_batch(
    span<const u32> work_items) {
  lib_assert(storage_owner_maintenance_completion_ring_ != nullptr,
             "storage-owner completion ring is not initialized");
  lib_assert(storage_owner_maintenance_admission_limit_ != 0,
             "storage-owner completion admission window is not initialized");
  const u64 sequence =
    storage_owner_maintenance_completion_ring_->reserve_batch(
      work_items, storage_owner_maintenance_admission_limit_);
  publish_storage_owner_maintenance_watermarks();
  return sequence;
}

void MemoryNode::publish_storage_owner_maintenance_watermarks() {
  lib_assert(storage_owner_maintenance_completion_ring_ != nullptr,
             "storage-owner completion ring is not initialized");
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  std::atomic_ref<u64> next(control->next_maintenance_sequence);
  std::atomic_ref<u64> durable(control->durable_maintenance_sequence);
  u64 observed_next = next.load(std::memory_order_acquire);
  const u64 desired_next =
    storage_owner_maintenance_completion_ring_->next_sequence();
  while (observed_next < desired_next &&
         !next.compare_exchange_weak(
           observed_next, desired_next,
           std::memory_order_release, std::memory_order_acquire)) {
  }
  u64 observed_durable = durable.load(std::memory_order_acquire);
  const u64 desired_durable =
    storage_owner_maintenance_completion_ring_->finalized();
  while (observed_durable < desired_durable &&
         !durable.compare_exchange_weak(
           observed_durable, desired_durable,
           std::memory_order_release, std::memory_order_acquire)) {
  }
}

void MemoryNode::complete_storage_owner_maintenance_sequence(u64 sequence) {
  complete_storage_owner_maintenance_sequence(sequence, 1);
}

void MemoryNode::complete_storage_owner_maintenance_sequence(
    u64 sequence, u32 work_items) {
  if (sequence == 0 || work_items == 0) return;
  lib_assert(storage_owner_maintenance_completion_ring_ != nullptr,
             "storage-owner completion ring is not initialized");
  storage_owner_maintenance_completion_ring_->complete(
    sequence, work_items);
  publish_storage_owner_maintenance_watermarks();
  storage_owner_maintenance_cv_.notify_all();
}

bool MemoryNode::storage_owner_cleanup_ready(u64 sequence) const {
  if (sequence <= 1) {
    return cleanup_predecessors_durable(sequence, 0);
  }
  const auto* control = reinterpret_cast<const gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  auto& durable_storage = const_cast<u64&>(control->durable_maintenance_sequence);
  const u64 durable = std::atomic_ref<u64>(durable_storage).load(
    std::memory_order_acquire);
  return cleanup_predecessors_durable(sequence, durable);
}

void MemoryNode::mark_storage_owner_foreground_activity() {
  storage_owner_foreground_last_active_ns_.store(steady_now_ns(), std::memory_order_release);
}

bool MemoryNode::storage_owner_maintenance_foreground_busy(const Configuration&) {
  const bool foreground_active =
    storage_owner_insert_active_workers_.load(std::memory_order_acquire) != 0;

  if (storage_insert_tasks_ != nullptr) {
    const size_t foreground_queue_yield_threshold =
      std::max<size_t>(4, storage_owner_threads_.size() * kForegroundQueueYieldMultiplier);
    if (storage_insert_tasks_->approximate_size() >=
        foreground_queue_yield_threshold) {
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

bool MemoryNode::try_acquire_storage_owner_maintenance_slot(
    const Configuration& config, bool foreground_pressure) {
  // This counter is the global number of live stage2 contexts, not the number
  // of physical workers. Each maintenance executor owns a fixed rpc-depth
  // context pool; peer send credits are independently bounded and try-only.
  // Under foreground pressure, retain one escape context per dedicated
  // maintenance worker. A zero-context policy deadlocks when Stage1 producers
  // themselves are waiting for the ordered maintenance window to advance.
  const size_t admitted_contexts = stage2_context_admission_limit(
    storage_owner_maintenance_worker_states_.size(),
    config.storage_owner_rpc_depth, foreground_pressure);
  const u32 max_contexts = static_cast<u32>(std::min<u64>(
    admitted_contexts, std::numeric_limits<u32>::max()));
  u32 active = storage_owner_maintenance_active_workers_.load(std::memory_order_acquire);
  while (active < max_contexts) {
    if (storage_owner_maintenance_active_workers_.compare_exchange_weak(
          active, active + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}
