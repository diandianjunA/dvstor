#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

size_t ComputeService::submit_storage_owner_mutations(
    const vec<InsertItem>& items,
    service::storage_owner::MutationKind kind) {
  if (storage_insert_owners_.empty()) {
    throw std::runtime_error("storage_owner mutation runtime is not initialized");
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
  pending.clear();
  pending.reserve(std::min<size_t>(
    items.size(), storage_completion_pool_->capacity()));
  size_t committed = 0;

  const auto consume_one = [&]() {
    lib_assert(!pending.empty(), "missing storage-owner completion");
    const u32 completion_id = pending.back();
    const auto result = storage_completion_pool_->wait(completion_id);
    auto& sample = storage_completion_samples_[completion_id];
    if (sample.collects_breakdown()) {
      sample.add_subcategory(
        service::breakdown::Subcategory::cpu_storage_owner_caller_wake,
        duration_ns_clamped(
          sample.finished_at, std::chrono::steady_clock::now()));
    }
    sample.mark_finished(std::chrono::steady_clock::now());
    if (sample.finished_flag) {
      std::lock_guard<std::mutex> lock(breakdown_mutex_);
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
    vamana::anchor::Route route;
    u32 owner_storage = 0;
    const std::optional<u32> known_owner =
      known_storage_owner_for_id(item.id);
    if (known_owner.has_value()) {
      owner_storage = *known_owner;
      if (kind == service::storage_owner::MutationKind::erase) {
        route.owner = owner_storage;
      } else {
        route = route_storage_owner_update(item, owner_storage);
      }
    } else {
      // Every first mutation claims an owner while holding the ID-map shard
      // lock. Inserts may propose the anchor-selected owner; upserts and
      // erases use the deterministic ID fallback. A racing mutation observes
      // the existing claim and routes to that actual owner, keeping generation
      // a single monotonic stream even across tombstones.
      if (kind == service::storage_owner::MutationKind::insert) {
        route = route_storage_owner_update(item, std::nullopt);
        owner_storage = claim_storage_owner_for_mutation(
          item.id, route.owner);
        if (owner_storage != route.owner) {
          route = route_storage_owner_update(item, owner_storage);
        }
      } else {
        const u32 proposed_owner = num_servers_ == 0
          ? 0
          : static_cast<u32>(item.id % num_servers_);
        owner_storage = claim_storage_owner_for_mutation(
          item.id, proposed_owner);
        if (kind == service::storage_owner::MutationKind::erase) {
          route.owner = owner_storage;
        } else {
          route = route_storage_owner_update(item, owner_storage);
        }
      }
    }
    lib_assert(owner_storage < storage_insert_owners_.size(),
               "storage-owner route selected an invalid owner");

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

    // Capacity is acquired before the descriptor can reach storage stage1.
    // Backpressure happens here, never after owner memory has been mutated.
    if (persistent_search_ != nullptr) {
      persistent_search_->reserve_mutation_capacity(1);
    }

    auto& state = *storage_insert_owners_[owner_storage];
    u32 task_id = 0;
    state.free_tasks->pop_wait(task_id);
    auto& task = state.tasks[task_id];
    task.item.id = item.id;
    if (kind == service::storage_owner::MutationKind::erase) {
      task.item.values.clear();
    } else {
      task.item.values.assign(item.values.begin(), item.values.end());
    }
    task.kind = kind;
    task.completion_id = completion_id;
    task.anchor_hints.assign(route.hints.begin(), route.hints.end());
    task.anchor_bucket_hint = route.bucket_hint;
    task.enqueued_at = std::chrono::steady_clock::now();
    task.sender_dequeued_at = {};
    state.queue->push_wait(task_id);
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
