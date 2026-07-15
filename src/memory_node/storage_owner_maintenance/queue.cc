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
                                       const vec<RemotePtr>* stage1_candidates,
                                       const vec<RemotePtr>* stage1_neighbors,
                                       const Configuration& config) {
  StorageOwnerMaintenanceTask task;
  task.kind = StorageOwnerMaintenanceKind::stitch_insert;
  task.id = id;
  task.generation = generation;
  task.maintenance_sequence = maintenance_sequence;
  task.target = target;
  if (stage1_candidates != nullptr) {
    task.stage1_candidates = *stage1_candidates;
  }
  if (stage1_neighbors != nullptr) {
    task.stitch_base_neighbors = *stage1_neighbors;
  }
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
  return begin_storage_owner_maintenance_batch(
    span<const u32>{&work_items, 1});
}

u64 MemoryNode::begin_storage_owner_maintenance_batch(
    span<const u32> work_items) {
  lib_assert(storage_owner_maintenance_completion_ring_ != nullptr,
             "storage-owner completion ring is not initialized");
  const u64 sequence =
    storage_owner_maintenance_completion_ring_->reserve_batch(work_items);
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
    return true;
  }
  const auto* control = reinterpret_cast<const gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  auto& durable_storage = const_cast<u64&>(control->durable_maintenance_sequence);
  const u64 durable = std::atomic_ref<u64>(durable_storage).load(
    std::memory_order_acquire);
  return durable >= sequence - 1;
}

u32 MemoryNode::storage_owner_maintenance_work_items(
    service::storage_owner::MutationKind kind,
    const Configuration& config) const {
  // Even when stage2 is disabled, keep one completion unit until the
  // maintenance intent has been published.  Returning zero lets reserve_batch
  // finalize and recycle the modulo slot before schedule_storage_owner_maintenance
  // writes the intent, so concurrent foreground workers can race while writing
  // the same non-atomic intent fields.
  if (!storage_owner_maintenance_enabled(config)) return 1;
  switch (kind) {
    case service::storage_owner::MutationKind::insert:
    case service::storage_owner::MutationKind::erase:
      return 1;
    case service::storage_owner::MutationKind::upsert:
      return 2;
  }
  return 0;
}

u64 MemoryNode::schedule_storage_owner_maintenance(
    node_t id,
    u32 generation,
    service::storage_owner::MutationKind kind,
    RemotePtr new_ptr,
    RemotePtr old_ptr,
    u64 reserved_sequence,
    u32 reserved_work_items,
    const Configuration& config,
    const vec<RemotePtr>* stage1_candidates,
    const vec<RemotePtr>* stage1_neighbors) {
  const bool maintenance_enabled = storage_owner_maintenance_enabled(config);
  const bool needs_stitch = maintenance_enabled &&
    kind != service::storage_owner::MutationKind::erase && !new_ptr.is_null();
  const bool needs_cleanup = maintenance_enabled &&
    kind != service::storage_owner::MutationKind::insert && !old_ptr.is_null();
  const u32 actual_work_items =
    static_cast<u32>(needs_stitch) + static_cast<u32>(needs_cleanup);
  lib_assert(reserved_sequence != 0 && actual_work_items <= reserved_work_items,
             "storage-owner maintenance exceeded its pre-stage1 reservation");
  lib_assert(storage_owner_maintenance_intents_ != nullptr &&
               storage_owner_maintenance_intent_capacity_ != 0,
             "storage-owner maintenance intent ring is not initialized");
  auto& intent = storage_owner_maintenance_intents_[
    static_cast<size_t>((reserved_sequence - 1) %
                        storage_owner_maintenance_intent_capacity_)];
  intent.id = id;
  intent.generation = generation;
  intent.kind = kind;
  intent.new_ptr = new_ptr;
  intent.old_ptr = old_ptr;
  intent.published_at = std::chrono::steady_clock::now();
  intent.sequence.store(reserved_sequence, std::memory_order_release);
  if (needs_stitch &&
      !enqueue_insert_stitch(
        id, generation, new_ptr, reserved_sequence,
        stage1_candidates, stage1_neighbors, config)) {
    lib_failure("failed to enqueue storage-owner stitch maintenance");
  }
  if (needs_cleanup &&
      !enqueue_deleted_node_cleanup(
        old_ptr, reserved_sequence, config)) {
    lib_failure("failed to enqueue storage-owner cleanup maintenance");
  }
  if (reserved_work_items > actual_work_items) {
    complete_storage_owner_maintenance_sequence(
      reserved_sequence, reserved_work_items - actual_work_items);
  }
  return reserved_sequence;
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
  // This counter is the global number of live stage2 contexts, not the number
  // of physical workers. Each maintenance executor owns a fixed rpc-depth
  // context pool; peer send credits are independently bounded and try-only.
  const u64 configured_contexts =
    static_cast<u64>(std::max<u32>(1, config.storage_owner_maintenance_workers)) *
    std::max<u32>(1, config.storage_owner_rpc_depth);
  const u32 max_contexts = static_cast<u32>(std::min<u64>(
    configured_contexts, std::numeric_limits<u32>::max()));
  u32 active = storage_owner_maintenance_active_workers_.load(std::memory_order_acquire);
  while (active < max_contexts) {
    if (storage_owner_maintenance_active_workers_.compare_exchange_weak(
          active, active + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}
