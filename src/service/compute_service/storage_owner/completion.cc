#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

void ComputeService::handle_storage_owner_send_completion(
    u32 owner_storage, u32 slot_id) {
  if (owner_storage >= storage_insert_owners_.size()) return;
  auto& state = *storage_insert_owners_[owner_storage];
  if (slot_id >= state.slots.size()) return;
  auto& slot = state.slots[slot_id];
  if (!slot.in_use) return;
  slot.send_done = true;
  slot.send_completed_at = std::chrono::steady_clock::now();
  queue_storage_owner_completion(slot);
}

void ComputeService::handle_storage_owner_response(
    u32 owner_storage, u32 response_slot_id) {
  if (owner_storage >= storage_insert_owners_.size()) return;
  auto& state = *storage_insert_owners_[owner_storage];
  if (response_slot_id >= state.response_slots.size()) return;

  auto& response_slot = state.response_slots[response_slot_id];
  const auto* response = reinterpret_cast<const
    service::storage_owner::InsertBatchResponseHeader*>(
      response_slot.buffer.data());
  const bool header_ok =
    (response->magic == service::storage_owner::kInsertMagic ||
     response->magic == service::storage_owner::kMutationMagic) &&
    response->owner_storage == owner_storage;

  StorageOwnerRpcSlot* matched = nullptr;
  if (header_ok) {
    for (auto& slot : state.slots) {
      if (slot.in_use && !slot.response_done &&
          slot.batch_id == response->batch_id) {
        matched = &slot;
        break;
      }
    }
  }
  if (matched == nullptr) {
    static std::atomic<u32> unknown_response_logs{0};
    const u32 log_index = unknown_response_logs.fetch_add(
      1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] unmatched insert response"
                << " owner=" << owner_storage
                << " response_slot=" << response_slot_id
                << " magic=0x" << std::hex << response->magic << std::dec
                << " response_owner=" << response->owner_storage
                << " batch_id=" << response->batch_id
                << " item_count=" << response->item_count << std::endl;
    }
    post_storage_owner_response_receive(owner_storage, response_slot_id);
    return;
  }

  const size_t response_size =
    service::storage_owner::insert_batch_response_bytes(response->item_count);
  if (response_size > response_slot.buffer.size()) {
    post_storage_owner_response_receive(owner_storage, response_slot_id);
    return;
  }
  matched->response_done = true;
  matched->response_slot_id = response_slot_id;
  matched->response_completed_at = std::chrono::steady_clock::now();
  matched->cq_progress_gap_ns = storage_insert_current_cq_gap_ns_;
  queue_storage_owner_completion(*matched);
  // The receive is reposted only after the response executor has finished
  // parsing this buffer. This removes the large CQ-thread memcpy.
}

void ComputeService::post_storage_owner_response_receive(
    u32 owner_storage, u32 response_slot_id) {
  if (owner_storage >= storage_insert_owners_.size()) return;
  auto& state = *storage_insert_owners_[owner_storage];
  if (response_slot_id >= state.response_slots.size()) return;
  auto& response_slot = state.response_slots[response_slot_id];
  cm_.server_qps[owner_storage]->post_receive(
    *response_slot.region,
    static_cast<u32>(response_slot.buffer.size()),
    storage_owner_wr_id(owner_storage, response_slot_id));
}

bool ComputeService::queue_storage_owner_completion(
    StorageOwnerRpcSlot& slot) {
  if (!slot.in_use || !slot.send_done || !slot.response_done ||
      slot.results_completed || slot.completion_claimed) {
    return false;
  }
  slot.completion_claimed = true;
  const bool queued = storage_ready_slots_->try_push(
    StorageOwnerReadySlot{slot.owner_storage, slot.slot_id});
  lib_assert(queued,
             "storage-owner ready queue exhausted despite RPC-slot bound");
  return true;
}

void ComputeService::run_storage_insert_completion_loop() {
  for (;;) {
    StorageOwnerReadySlot ready;
    if (!storage_ready_slots_->pop_wait(
          ready, storage_insert_progress_done_)) {
      return;
    }

    StorageOwnerPublicationBatch publication;
    commit_storage_owner_slot(
      ready.owner_storage, ready.slot_id, publication);
    release_storage_owner_slot(ready.owner_storage, ready.slot_id);

    if (publication.mutations.empty()) {
      if (persistent_search_ != nullptr && publication.reserved_items != 0) {
        persistent_search_->release_mutation_capacity(
          publication.reserved_items);
      }
    } else {
      publish_storage_owner_mutations(std::move(publication));
    }
  }
}

void ComputeService::commit_storage_owner_slot(
    u32 owner_storage,
    u32 slot_id,
    StorageOwnerPublicationBatch& publication) {
  auto& state = *storage_insert_owners_[owner_storage];
  lib_assert(slot_id < state.slots.size(),
             "storage-owner completion references an invalid slot");
  auto& slot = state.slots[slot_id];
  lib_assert(slot.in_use && slot.send_done && slot.response_done &&
               slot.completion_claimed && !slot.results_completed &&
               slot.response_slot_id < state.response_slots.size(),
             "storage-owner completion claimed a slot in an invalid state");
  slot.results_completed = true;
  const auto response_executor_started = std::chrono::steady_clock::now();

  const byte_t* response_buffer =
    state.response_slots[slot.response_slot_id].buffer.data();
  const auto* response = reinterpret_cast<const
    service::storage_owner::InsertBatchResponseHeader*>(response_buffer);
  const auto* request = reinterpret_cast<const
    service::storage_owner::InsertBatchRequestHeader*>(
      slot.request_buffer.data());
  bool response_ok =
    (response->magic == service::storage_owner::kInsertMagic ||
     response->magic == service::storage_owner::kMutationMagic) &&
    response->magic == request->magic &&
    response->owner_storage == slot.owner_storage &&
    response->batch_id == slot.batch_id &&
    response->item_count == slot.item_count;
  u32 invalidation_count = 0;
  if (response_ok) {
    invalidation_count =
      *service::storage_owner::response_invalidation_count(
        response_buffer, slot.item_count);
    response_ok = invalidation_count <=
      service::storage_owner::response_invalidation_capacity(
        slot.item_count);
    if (!response_ok) invalidation_count = 0;
  }

  const u32* statuses =
    service::storage_owner::response_statuses(response_buffer);
  const auto* results =
    service::storage_owner::response_mutation_results(
      response_buffer, slot.item_count);
  const bool mutation_request =
    request->magic == service::storage_owner::kMutationMagic;
  const byte_t* request_vectors = mutation_request
    ? service::storage_owner::mutation_request_vectors(
        slot.request_buffer.data(), slot.item_count)
    : service::storage_owner::request_vectors(
        slot.request_buffer.data(), slot.item_count);

  bool collect_breakdown = false;
  for (const u32 task_id : slot.tasks) {
    collect_breakdown = collect_breakdown ||
      storage_completion_samples_[state.tasks[task_id].completion_id]
        .collects_breakdown();
  }
  const auto* breakdown = collect_breakdown && response_ok
    ? service::storage_owner::response_breakdown(
        response_buffer, slot.item_count)
    : nullptr;
  if (!response_ok) {
    static std::atomic<u32> bad_response_logs{0};
    const u32 log_index = bad_response_logs.fetch_add(
      1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] invalid insert response"
                << " owner=" << slot.owner_storage
                << " slot=" << slot.slot_id
                << " magic=0x" << std::hex << response->magic << std::dec
                << " response_owner=" << response->owner_storage
                << " expected_owner=" << slot.owner_storage
                << " batch_id=" << response->batch_id
                << " expected_batch_id=" << slot.batch_id
                << " item_count=" << response->item_count
                << " expected_item_count=" << slot.item_count << std::endl;
    }
  }

  const u64 memory_breakdown_ns = breakdown == nullptr
    ? 0 : breakdown->total();
  const u64 send_ns = collect_breakdown
    ? duration_ns_clamped(slot.send_posted_at, slot.send_completed_at) : 0;
  const u64 response_wait_ns = collect_breakdown
    ? duration_ns_clamped(
        slot.send_completed_at, slot.response_completed_at) : 0;
  const u64 response_wait_unaccounted_ns =
    collect_breakdown && response_wait_ns > memory_breakdown_ns
      ? response_wait_ns - memory_breakdown_ns : 0;

  publication.mutations.reserve(slot.item_count);
  publication.invalidated_graph_nodes.reserve(
    static_cast<size_t>(slot.item_count) * config_.R);
  if (persistent_search_ != nullptr && response_ok) {
    const u64* invalidated_raws =
      service::storage_owner::response_invalidated_raws(
        response_buffer, slot.item_count);
    for (u32 index = 0; index < invalidation_count; ++index) {
      if (invalidated_raws[index] != 0) {
        publication.invalidated_graph_nodes.push_back(
          invalidated_raws[index]);
      }
    }
  }

  u32 committed_items = 0;
  for (u32 i = 0; i < slot.item_count; ++i) {
    const u32 task_id = slot.tasks[i];
    auto& task = state.tasks[task_id];
    auto& sample = storage_completion_samples_[task.completion_id];
    const bool committed = response_ok && statuses[i] == 0;
    if (response_ok && !committed) {
      static std::atomic<u32> failed_status_logs{0};
      const u32 log_index = failed_status_logs.fetch_add(
        1, std::memory_order_relaxed);
      if (log_index < 16) {
        std::cerr << "[storage-owner] mutation failed"
                  << " owner=" << slot.owner_storage
                  << " slot=" << slot.slot_id
                  << " batch_id=" << slot.batch_id
                  << " item=" << i
                  << " status=" << statuses[i] << std::endl;
      }
    }
    if (sample.collects_breakdown()) {
      add_storage_owner_sender_breakdown(
        &sample,
        duration_ns_clamped(task.enqueued_at, task.sender_dequeued_at),
        0,
        slot.request_prepare_ns,
        send_ns,
        response_wait_unaccounted_ns,
        slot.item_count);
      sample.add_subcategory(
        service::breakdown::Subcategory::cpu_storage_owner_dequeue_to_post,
        duration_ns_clamped(task.sender_dequeued_at, slot.send_posted_at));
      sample.add_subcategory(
        service::breakdown::Subcategory::cpu_storage_owner_cq_progress_gap,
        per_item_ns(slot.cq_progress_gap_ns, slot.item_count));
      sample.add_subcategory(
        service::breakdown::Subcategory::cpu_storage_owner_response_executor_queue,
        duration_ns_clamped(
          slot.response_completed_at, response_executor_started));
      if (breakdown != nullptr) {
        add_storage_owner_breakdown(&sample, *breakdown, slot.item_count);
        if (i == 0) add_storage_owner_counters(&sample, *breakdown);
      }
    }

    if (committed) {
      const auto& result = results[i];
      publish_compute_side_id(
        task.item.id,
        RemotePtr{task.kind == service::storage_owner::MutationKind::erase
                    ? result.old_rptr_raw : result.new_rptr_raw},
        task.kind == service::storage_owner::MutationKind::erase,
        slot.owner_storage,
        result.generation);
      if (persistent_search_ != nullptr) {
        gpu_search::DeltaMutation mutation;
        mutation.id = task.item.id;
        mutation.kind = task.kind;
        mutation.generation = result.generation;
        mutation.remote_node = result.new_rptr_raw;
        mutation.old_remote_node = result.old_rptr_raw;
        mutation.anchor_hint = task.anchor_bucket_hint.raw_address;
        mutation.maintenance_sequence = result.maintenance_sequence;
        mutation.owner_storage = owner_storage;
        mutation.enqueued_at = slot.response_completed_at;
        if (mutation.kind != service::storage_owner::MutationKind::erase) {
          const byte_t* vector = request_vectors +
            static_cast<size_t>(i) * VamanaNode::vector_bytes();
          mutation.vector.assign(
            vector, vector + VamanaNode::vector_bytes());
        }
        publication.mutations.push_back(std::move(mutation));
        ++committed_items;
      }
    }

  }

  const auto response_processed_at = std::chrono::steady_clock::now();
  const u64 response_process_ns = duration_ns(
    response_executor_started, response_processed_at);
  for (u32 i = 0; i < slot.item_count; ++i) {
    const u32 task_id = slot.tasks[i];
    auto& task = state.tasks[task_id];
    auto& sample = storage_completion_samples_[task.completion_id];
    if (sample.collects_breakdown()) {
      sample.add_subcategory(
        service::breakdown::Subcategory::cpu_storage_owner_response_process,
        per_item_ns(response_process_ns, slot.item_count));
    }
    sample.mark_finished(response_processed_at);
    complete_storage_owner_task(
      owner_storage, task_id, response_ok && statuses[i] == 0);
  }

  lib_assert(committed_items <= slot.gpu_reserved_items,
             "committed storage mutations exceeded reserved GPU capacity");
  publication.reserved_items = committed_items;
  const u32 release_reserved_items =
    slot.gpu_reserved_items - committed_items;
  if (persistent_search_ != nullptr && release_reserved_items != 0) {
    persistent_search_->release_mutation_capacity(release_reserved_items);
  }
}

void ComputeService::release_storage_owner_slot(
    u32 owner_storage, u32 slot_id) {
  auto& state = *storage_insert_owners_[owner_storage];
  lib_assert(slot_id < state.slots.size(),
             "storage-owner release references an invalid slot");
  auto& slot = state.slots[slot_id];
  lib_assert(slot.in_use && slot.results_completed &&
               slot.response_slot_id < state.response_slots.size(),
             "storage-owner released a slot before completion");
  const bool queued = storage_released_slots_->try_push(
    StorageOwnerReleasedSlot{
      owner_storage, slot_id, slot.response_slot_id});
  lib_assert(queued,
             "storage-owner release queue exhausted despite RPC-slot bound");
}

void ComputeService::publish_storage_owner_mutations(
    StorageOwnerPublicationBatch&& publication) {
  if (persistent_search_ == nullptr || publication.mutations.empty()) {
    if (persistent_search_ != nullptr && publication.reserved_items != 0) {
      persistent_search_->release_mutation_capacity(
        publication.reserved_items);
    }
    return;
  }

  std::sort(publication.invalidated_graph_nodes.begin(),
            publication.invalidated_graph_nodes.end());
  publication.invalidated_graph_nodes.erase(
    std::unique(publication.invalidated_graph_nodes.begin(),
                publication.invalidated_graph_nodes.end()),
    publication.invalidated_graph_nodes.end());
  try {
    const u64 epoch = persistent_search_->delta().reserve_epoch();
    if (!persistent_search_->publish_mutations(
          std::move(publication.mutations), epoch,
          publication.invalidated_graph_nodes)) {
      persistent_search_->mark_committed_mutation_gap(
        "persistent GPU mutation publication returned false");
    }
  } catch (const std::exception& error) {
    persistent_search_->mark_committed_mutation_gap(error.what());
    static std::atomic<u32> gpu_delta_failure_logs{0};
    const u32 log_index = gpu_delta_failure_logs.fetch_add(
      1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] committed mutation batch was not "
                   "published to GPU delta: "
                << error.what() << std::endl;
    }
  }
  persistent_search_->release_mutation_capacity(publication.reserved_items);
}

void ComputeService::complete_storage_owner_task(
    u32 owner_storage, u32 task_id, bool success) {
  auto& state = *storage_insert_owners_[owner_storage];
  lib_assert(task_id < state.task_capacity,
             "storage-owner completion references an invalid task");
  auto& task = state.tasks[task_id];
  const u32 completion_id = task.completion_id;
  task.item.values.clear();
  task.anchor_hints.clear();
  task.anchor_bucket_hint = {};
  task.enqueued_at = {};
  task.sender_dequeued_at = {};
  task.completion_id = std::numeric_limits<u32>::max();
  const bool freed = state.free_tasks->try_push(task_id);
  lib_assert(freed, "storage-owner task pool overflow");
  storage_completion_pool_->complete(completion_id, success);
}

void ComputeService::fail_storage_owner_tasks(
    u32 owner_storage, vec<u32>& tasks) {
  if (tasks.empty()) return;
  if (persistent_search_ != nullptr) {
    persistent_search_->release_mutation_capacity(tasks.size());
  }
  const auto finished_at = std::chrono::steady_clock::now();
  for (const u32 task_id : tasks) {
    auto& task = storage_insert_owners_[owner_storage]->tasks[task_id];
    auto& sample = storage_completion_samples_[task.completion_id];
    if (!sample.finished_flag) sample.mark_finished(finished_at);
    complete_storage_owner_task(owner_storage, task_id, false);
  }
  tasks.clear();
}
