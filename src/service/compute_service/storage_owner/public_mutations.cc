#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

size_t ComputeService::insert(const vec<InsertItem>& batch) {
  if (storage_insert_owners_.empty()) {
    throw std::runtime_error("storage_owner insert runtime is not initialized");
  }
    vec<std::future<bool>> futures;
    vec<std::shared_ptr<service::breakdown::Sample>> samples;
    futures.reserve(batch.size());
    samples.reserve(batch.size());
    for (const auto& item : batch) {
      if (item.values.size() != config_.dim) {
        throw std::invalid_argument("insert dimension mismatch");
      }

      auto sample = std::make_shared<service::breakdown::Sample>(
        service::breakdown::Operation::insert, breakdown_enabled_);
      const auto now = std::chrono::steady_clock::now();
      sample->enqueued_at = now;
      sample->mark_started(now, now);

      auto task = std::make_unique<StorageInsertTask>();
      task->item = item;
      task->kind = service::storage_owner::MutationKind::insert;
      task->sample = sample;
      task->enqueued_at = sample->enqueued_at;
      futures.push_back(task->result.get_future());
      samples.push_back(sample);
      const auto route = route_storage_owner_update(item);
      task->anchor_hints = route.hints;
      task->anchor_bucket_hint = route.bucket_hint;
      const u32 owner_storage = route.owner;
      auto& state = *storage_insert_owners_[owner_storage];
      {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.queue.push_back(std::move(task));
      }
      state.cv.notify_one();
    }

    size_t inserted = 0;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.storage_owner_rpc_timeout_ms);
    for (size_t i = 0; i < futures.size(); ++i) {
      auto& future = futures[i];
      if (future.wait_until(deadline) != std::future_status::ready) {
        const u32 log_index = storage_insert_timeout_logs_.fetch_add(1, std::memory_order_relaxed);
        if (log_index < 8) {
          std::cerr << "[storage-owner] insert RPC timed out after "
                    << config_.storage_owner_rpc_timeout_ms << " ms" << std::endl;
        }
        if (samples[i] && !samples[i]->finished_flag) {
          samples[i]->mark_finished(std::chrono::steady_clock::now());
        }
        continue;
      }
      inserted += future.get() ? 1u : 0u;
    }
    for (const auto& sample : samples) {
      if (sample && sample->finished_flag) {
        std::lock_guard<std::mutex> lock(breakdown_mutex_);
        service::breakdown::add_sample(
          completed_breakdown_report_.insert, *sample);
      }
    }
    vectors_inserted_.fetch_add(inserted, std::memory_order_relaxed);
    return inserted;
}

size_t ComputeService::upsert(const vec<InsertItem>& batch) {
  if (storage_insert_owners_.empty()) {
    throw std::runtime_error("storage_owner mutation runtime is not initialized");
  }
  vec<std::future<bool>> futures;
  futures.reserve(batch.size());
  for (const auto& item : batch) {
    if (item.values.size() != config_.dim) {
      throw std::invalid_argument("upsert dimension mismatch");
    }
    auto task = std::make_unique<StorageInsertTask>();
    task->item = item;
    task->kind = service::storage_owner::MutationKind::upsert;
    futures.push_back(task->result.get_future());
    const u32 owner_storage = storage_owner_for_id(item.id);
    const auto route = route_storage_owner_update(item, owner_storage);
    task->anchor_hints = route.hints;
    task->anchor_bucket_hint = route.bucket_hint;
    auto& state = *storage_insert_owners_[owner_storage];
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.queue.push_back(std::move(task));
    }
    state.cv.notify_one();
  }
  size_t updated = 0;
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(config_.storage_owner_rpc_timeout_ms);
  for (auto& future : futures) {
    if (future.wait_until(deadline) != std::future_status::ready) {
      const u32 log_index = storage_insert_timeout_logs_.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 8) {
        std::cerr << "[storage-owner] upsert RPC timed out after "
                  << config_.storage_owner_rpc_timeout_ms << " ms" << std::endl;
      }
      continue;
    }
    updated += future.get() ? 1u : 0u;
  }
  vectors_inserted_.fetch_add(updated, std::memory_order_relaxed);
  return updated;
}

size_t ComputeService::erase(const vec<node_t>& ids) {
  if (storage_insert_owners_.empty()) {
    throw std::runtime_error("storage_owner mutation runtime is not initialized");
  }
  vec<std::future<bool>> futures;
  futures.reserve(ids.size());
  for (const node_t id : ids) {
    auto task = std::make_unique<StorageInsertTask>();
    task->item.id = id;
    task->item.values.assign(config_.dim, 0.0f);
    task->kind = service::storage_owner::MutationKind::erase;
    futures.push_back(task->result.get_future());
    const u32 owner_storage = storage_owner_for_id(id);
    auto& state = *storage_insert_owners_[owner_storage];
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.queue.push_back(std::move(task));
    }
    state.cv.notify_one();
  }
  size_t erased = 0;
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(config_.storage_owner_rpc_timeout_ms);
  for (auto& future : futures) {
    if (future.wait_until(deadline) != std::future_status::ready) {
      const u32 log_index = storage_insert_timeout_logs_.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 8) {
        std::cerr << "[storage-owner] delete RPC timed out after "
                  << config_.storage_owner_rpc_timeout_ms << " ms" << std::endl;
      }
      continue;
    }
    erased += future.get() ? 1u : 0u;
  }
  return erased;
}
