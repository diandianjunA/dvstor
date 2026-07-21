#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

size_t ComputeService::submit_storage_owner_mutations(
    const vec<InsertItem>& items,
    service::storage_owner::MutationKind kind) {
  if (!config_.enable_updates) {
    throw std::runtime_error(
      "compute updates are disabled by enable-updates=false");
  }
  if (storage_insert_owners_.empty()) {
    throw std::runtime_error("storage_owner mutation runtime is not initialized");
  }
  for (const auto& item : items) {
    if (item.id >= config_.vector_id_namespace_size) {
      throw std::out_of_range(
        "mutation id " + std::to_string(item.id) +
        " exceeds the configured vector namespace [0," +
        std::to_string(config_.vector_id_namespace_size) + ")");
    }
  }
  if (kind != service::storage_owner::MutationKind::erase) {
    for (const auto& item : items) {
      if (item.values.size() != config_.dim) {
        throw std::invalid_argument("mutation dimension mismatch");
      }
    }
  }

  // A writer normally submits one item at a time. Reuse this buffer across
  // calls so the synchronous API does not allocate on every operation while
  // still supporting arbitrary public batch sizes.
  thread_local vec<u32> pending;
  thread_local vec<byte_t> canonical_bytes;
  thread_local vec<element_t> canonical_vector;
  pending.clear();
  pending.reserve(std::min<size_t>(
    items.size(), storage_completion_pool_->capacity()));
  size_t committed = 0;

  const auto consume_one = [&]() {
    lib_assert(!pending.empty(), "missing storage-owner completion");
    const u32 completion_id = pending.back();
    const auto result = storage_completion_pool_->wait_for(
      completion_id,
      std::chrono::milliseconds(config_.storage_owner_rpc_timeout_ms));
    if (result == bounded::CompletionPool::Result::pending) {
      const u64 log_index = storage_insert_late_rpc_completions_.fetch_add(
        1, std::memory_order_relaxed);
      if (log_index < 8) {
        std::cerr << "[storage-owner] mutation RPC timed out after "
                  << config_.storage_owner_rpc_timeout_ms
                  << " ms; caller stopped waiting but the bounded producer "
                     "cell remains valid until its late completion"
                  << std::endl;
      }
      storage_completion_pool_->release_consumer(completion_id);
      pending.pop_back();
      return;
    }
    auto& sample = storage_completion_samples_[completion_id];
    if (sample.collects_breakdown()) {
      sample.add_subcategory(
        service::breakdown::Subcategory::cpu_storage_owner_caller_wake,
        duration_ns_clamped(
          sample.finished_at, std::chrono::steady_clock::now()));
    }
    sample.mark_finished(std::chrono::steady_clock::now());
    if (sample.finished_flag) {
      service::breakdown::add_sample(
        completed_breakdown_report_.insert, sample);
    }
    if (sample.end_to_end_ns >
        static_cast<u64>(config_.storage_owner_rpc_timeout_ms) * 1'000'000ull) {
      const u64 log_index = storage_insert_late_rpc_completions_.fetch_add(
        1, std::memory_order_relaxed);
      if (log_index < 8) {
        std::cerr << "[storage-owner] mutation completed after configured "
                     "RPC timeout: elapsed_ms="
                  << (sample.end_to_end_ns / 1'000'000.0) << std::endl;
      }
    }
    committed += result == bounded::CompletionPool::Result::success ? 1 : 0;
    storage_completion_pool_->release_consumer(completion_id);
    pending.pop_back();
  };

  for (const auto& item : items) {
    const auto operation_started = std::chrono::steady_clock::now();
    // Logical authority never follows physical placement. Every compute node
    // derives the same shard for base and dynamic IDs, while that authority's
    // storage-side directory resolves the current centroid-selected record.
    const u32 owner_storage = static_cast<u32>(item.id % num_servers_);
    lib_assert(owner_storage < storage_insert_owners_.size(),
               "storage-owner route selected an invalid owner");
    auto& state = *storage_insert_owners_[owner_storage];
    struct ProducerAnnouncement {
      std::atomic<u32>& pending;
      bool active{true};

      explicit ProducerAnnouncement(std::atomic<u32>& value)
          : pending(value) {
        pending.fetch_add(1, std::memory_order_acq_rel);
      }
      ~ProducerAnnouncement() {
        if (active) retire();
      }
      void retire() {
        const u32 previous = pending.fetch_sub(1, std::memory_order_release);
        lib_assert(previous != 0,
                   "storage-owner producer announcement underflow");
        active = false;
      }
    } producer_announcement{state.pending_producers};
    u32 stage1_home = owner_storage;
    if (kind != service::storage_owner::MutationKind::erase) {
      if (persistent_search_ == nullptr) {
        throw std::runtime_error(
          "centroid home selection requires the persistent query engine");
      }
      canonical_bytes.resize(VamanaNode::vector_bytes());
      encode_float_vector_to_storage(
        item.values.data(), config_.dim, VamanaNode::vector_dtype(),
        canonical_bytes.data());
      canonical_vector.resize(config_.dim);
      decode_storage_vector_to_float(
        canonical_bytes.data(), VamanaNode::vector_dtype(), config_.dim,
        canonical_vector.data());
      const auto selected = persistent_search_->select_centroid_home(
        span<const element_t>{canonical_vector});
      if (!selected.has_value() || *selected >= num_servers_) {
        throw std::runtime_error(
          "no live physical-shard centroid is available for mutation");
      }
      stage1_home = *selected;
    }

    u32 completion_id = 0;
    while (!storage_completion_pool_->try_acquire(completion_id)) {
      if (pending.empty()) {
        completion_id = storage_completion_pool_->acquire();
        break;
      }
      consume_one();
    }
    auto& sample = storage_completion_samples_[completion_id];
    sample = service::breakdown::Sample(
      service::breakdown::Operation::insert,
      breakdown_enabled_.load(std::memory_order_acquire));
    sample.enqueued_at = operation_started;
    sample.mark_started(operation_started, operation_started);
    const auto route_finished = std::chrono::steady_clock::now();
    sample.add_subcategory(
      service::breakdown::Subcategory::cpu_storage_owner_route,
      duration_ns(operation_started, route_finished));

    u32 task_id = 0;
    state.free_tasks->pop_wait(task_id);
    auto& task = state.tasks[task_id];
    task.id = item.id;
    if (kind == service::storage_owner::MutationKind::erase) {
      task.encoded_vector.clear();
    } else {
      task.encoded_vector.assign(
        canonical_bytes.begin(), canonical_bytes.end());
    }
    task.kind = kind;
    task.completion_id = completion_id;
    task.stage1_home = stage1_home;
    task.enqueued_at = std::chrono::steady_clock::now();
    task.sender_dequeued_at = {};
    state.queue->push_wait(task_id);
    state.published_tasks.fetch_add(1, std::memory_order_release);
    // Queue publication precedes the release-store. A sender that observes no
    // remaining producer therefore either already drained this task or can
    // acquire it on its final queue probe.
    producer_announcement.retire();
    pending.push_back(completion_id);
  }

  while (!pending.empty()) consume_one();
  return committed;
}

size_t ComputeService::insert(const vec<InsertItem>& batch) {
  const size_t inserted = submit_storage_owner_mutations(
    batch, service::storage_owner::MutationKind::insert);
  vectors_inserted_.fetch_add(inserted, std::memory_order_relaxed);
  return inserted;
}

size_t ComputeService::upsert(const vec<InsertItem>& batch) {
  const size_t updated = submit_storage_owner_mutations(
    batch, service::storage_owner::MutationKind::upsert);
  vectors_inserted_.fetch_add(updated, std::memory_order_relaxed);
  return updated;
}

size_t ComputeService::erase(const vec<node_t>& ids) {
  vec<InsertItem> items;
  items.reserve(ids.size());
  for (const node_t id : ids) {
    InsertItem item;
    item.id = id;
    items.push_back(std::move(item));
  }
  return submit_storage_owner_mutations(
    items, service::storage_owner::MutationKind::erase);
}

bool ComputeService::wait_for_storage_maintenance(
    std::chrono::milliseconds timeout,
    vec<u64>* target_sequences,
    vec<u64>* durable_sequences) {
  if (!config_.enable_updates || storage_maintenance_targets_ == nullptr) {
    if (target_sequences != nullptr) target_sequences->clear();
    if (durable_sequences != nullptr) durable_sequences->clear();
    return true;
  }
  vec<u64> targets(num_servers_, 0);
  for (u32 shard = 0; shard < num_servers_; ++shard) {
    targets[shard] = storage_maintenance_targets_[shard].load(
      std::memory_order_acquire);
  }
  vec<u64> effective_targets;
  const bool complete = persistent_search_ != nullptr &&
    persistent_search_->wait_for_maintenance(
      span<const u64>{targets}, timeout, durable_sequences,
      &effective_targets);
  if (target_sequences != nullptr) {
    *target_sequences = std::move(effective_targets);
  }
  return complete;
}
