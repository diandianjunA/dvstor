#include "memory_node/storage_owner_maintenance/detail.hh"

using namespace memory_node_storage_owner_maintenance_detail;

void MemoryNode::storage_owner_maintenance_worker_loop(u32 worker_id) {
  lib_assert(worker_id < storage_owner_maintenance_worker_states_.size(),
             "storage-owner maintenance worker state missing");
  StorageOwnerThread& thread = *storage_owner_maintenance_worker_states_[worker_id];
  current_storage_owner_thread_ = &thread;
  const Configuration& config = *storage_worker_config_;

  for (;;) {
    {
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait(lock, [&]() {
        return storage_owner_maintenance_shutdown_.load(std::memory_order_acquire) ||
               !storage_owner_stitch_tasks_.empty() ||
               !storage_owner_cleanup_tasks_.empty();
      });
      if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
    }

    maybe_log_storage_owner_maintenance_observation();
    {
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
      if (storage_owner_cleanup_tasks_.empty() && !storage_owner_stitch_tasks_.empty()) {
        const size_t batch_limit = std::max<u32>(1, config.storage_owner_batch_max);
        const auto ready_at = storage_owner_stitch_tasks_.front().queued_at +
                              std::chrono::nanoseconds(kStitchCompactionMaxDelayNs);
        const auto now = std::chrono::steady_clock::now();
        const bool candidate_ready =
          storage_owner_stitch_tasks_.size() >= batch_limit || now >= ready_at;
        if (!candidate_ready) {
          storage_owner_maintenance_cv_.wait_until(lock, ready_at, [&]() {
            return storage_owner_maintenance_shutdown_.load(std::memory_order_acquire) ||
                   !storage_owner_cleanup_tasks_.empty() ||
                   storage_owner_stitch_tasks_.size() >= batch_limit;
          });
          continue;
        }
      }
    }
    if (!try_acquire_storage_owner_maintenance_slot(config)) {
      storage_owner_maintenance_pressure_yields_.fetch_add(1, std::memory_order_relaxed);
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(lock, std::chrono::milliseconds(1), [&]() {
        return storage_owner_maintenance_shutdown_.load(std::memory_order_acquire);
      });
      continue;
    }
    atomic_utils::CounterDecrementGuard active_slot(
      storage_owner_maintenance_active_workers_);

    vec<StorageOwnerMaintenanceTask> stitch_batch;
    vec<StorageOwnerMaintenanceTask> cleanup_batch;
    {
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
      if (storage_owner_stitch_tasks_.empty() && storage_owner_cleanup_tasks_.empty()) {
        continue;
      }
      const auto ready_cleanup = std::find_if(
        storage_owner_cleanup_tasks_.begin(),
        storage_owner_cleanup_tasks_.end(),
        [&](const StorageOwnerMaintenanceTask& task) {
          return storage_owner_cleanup_ready(task.maintenance_sequence);
        });
      const bool cleanup_ready = ready_cleanup != storage_owner_cleanup_tasks_.end();
      const size_t batch_limit = std::max<u32>(1, config.storage_owner_batch_max);
      const bool choose_stitch =
        !storage_owner_stitch_tasks_.empty() &&
        (!cleanup_ready ||
         storage_owner_stitch_tasks_.front().queued_at <=
           ready_cleanup->queued_at);
      if (choose_stitch) {
        stitch_batch.reserve(batch_limit);
        while (!storage_owner_stitch_tasks_.empty() && stitch_batch.size() < batch_limit) {
          stitch_batch.push_back(std::move(storage_owner_stitch_tasks_.front()));
          storage_owner_stitch_tasks_.pop_front();
        }
      } else if (cleanup_ready) {
        cleanup_batch.reserve(batch_limit);
        for (auto iterator = storage_owner_cleanup_tasks_.begin();
             iterator != storage_owner_cleanup_tasks_.end() &&
               cleanup_batch.size() < batch_limit;) {
          if (!storage_owner_cleanup_ready(iterator->maintenance_sequence)) {
            ++iterator;
            continue;
          }
          cleanup_batch.push_back(std::move(*iterator));
          iterator = storage_owner_cleanup_tasks_.erase(iterator);
        }
      } else {
        storage_owner_maintenance_cv_.wait_for(
          lock, std::chrono::milliseconds(1));
        continue;
      }
    }
    storage_owner_maintenance_cv_.notify_all();

    if (!stitch_batch.empty()) {
      vec<StorageOwnerMaintenanceTask> retry_tasks;
      u64 processed_count = 0;
      const bool ok =
        stitch_inserted_storage_owner_nodes(stitch_batch, config, retry_tasks, processed_count);
      if (processed_count != 0) {
        storage_owner_maintenance_processed_.fetch_add(processed_count, std::memory_order_relaxed);
      }
      if (!ok || !retry_tasks.empty()) {
        storage_owner_maintenance_failed_.fetch_add(1, std::memory_order_relaxed);
        if (!storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
          size_t backlog = 0;
          {
            std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
            if (!storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
              for (auto& task : retry_tasks) {
                storage_owner_stitch_tasks_.push_back(std::move(task));
              }
              backlog = storage_owner_stitch_tasks_.size() + storage_owner_cleanup_tasks_.size();
            }
          }
          atomic_utils::update_max_relaxed(
            storage_owner_maintenance_max_backlog_, static_cast<u64>(backlog));
          storage_owner_maintenance_cv_.notify_one();
          std::this_thread::yield();
        }
      }
    }
    if (!cleanup_batch.empty()) {
      vec<StorageOwnerMaintenanceTask> retry_tasks;
      u64 processed_count = 0;
      const bool ok = cleanup_deleted_storage_owner_nodes(
        cleanup_batch, config, retry_tasks, processed_count);
      if (processed_count != 0) {
        storage_owner_maintenance_cleanup_processed_.fetch_add(
          processed_count, std::memory_order_relaxed);
      }
      if (!ok || !retry_tasks.empty()) {
        storage_owner_maintenance_failed_.fetch_add(1, std::memory_order_relaxed);
        if (!storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
          size_t backlog = 0;
          {
            std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
            if (!storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
              for (auto& task : retry_tasks) {
                storage_owner_cleanup_tasks_.push_back(std::move(task));
              }
              backlog = storage_owner_stitch_tasks_.size() + storage_owner_cleanup_tasks_.size();
            }
          }
          atomic_utils::update_max_relaxed(
            storage_owner_maintenance_max_backlog_, static_cast<u64>(backlog));
          storage_owner_maintenance_cv_.notify_one();
          std::this_thread::yield();
        }
      }
    }
    maybe_log_storage_owner_maintenance_observation();
  }
}
