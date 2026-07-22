#include "service/compute_service/detail.hh"
#include "service/compute_service/storage_owner/response_validation.hh"

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
    u32 owner_storage, u32 response_slot_id, u32 received_bytes) {
  if (owner_storage >= storage_insert_owners_.size()) return;
  auto& state = *storage_insert_owners_[owner_storage];
  if (response_slot_id >= state.response_slots.size()) return;

  auto& response_slot = state.response_slots[response_slot_id];
  const auto* response = reinterpret_cast<const
    service::storage_owner::InsertBatchResponseHeader*>(
      response_slot.buffer.data());
  StorageOwnerRpcSlot* matched = nullptr;
  if (received_bytes >=
      sizeof(service::storage_owner::InsertBatchResponseHeader)) {
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
                << " item_count=" << response->item_count
                << " received_bytes=" << received_bytes << std::endl;
    }
    post_storage_owner_response_receive(owner_storage, response_slot_id);
    return;
  }

  const auto* request = reinterpret_cast<const
    service::storage_owner::InsertBatchRequestHeader*>(
      matched->request_buffer.data());
  const auto validation = validate_storage_owner_response(
    *response,
    received_bytes,
    response_slot.buffer.size(),
    request->magic,
    owner_storage,
    matched->item_count,
    matched->batch_id,
    matched->response_size);
  lib_assert(validation != StorageOwnerResponseValidation::unmatched,
             "batch-id matched response was classified as unmatched");
  matched->response_done = true;
  matched->response_valid =
    validation == StorageOwnerResponseValidation::matched_valid;
  matched->response_slot_id = response_slot_id;
  matched->response_completed_at = std::chrono::steady_clock::now();
  matched->cq_progress_gap_ns = storage_insert_current_cq_gap_ns_;
  const u64 rpc_wall_ns = duration_ns_clamped(
    matched->send_posted_at, matched->response_completed_at);
  ++state.completed_rpc_batches;
  state.completed_rpc_items += matched->item_count;
  state.completed_rpc_wall_ns += rpc_wall_ns;
  state.max_rpc_wall_ns = std::max(state.max_rpc_wall_ns, rpc_wall_ns);
  storage_owner_completed_batches_.fetch_add(1, std::memory_order_relaxed);
  storage_owner_completed_items_.fetch_add(
    matched->item_count, std::memory_order_relaxed);
  storage_owner_completed_rpc_wall_ns_.fetch_add(
    rpc_wall_ns, std::memory_order_relaxed);
  u64 observed_max = storage_owner_max_rpc_wall_ns_.load(
    std::memory_order_relaxed);
  while (observed_max < rpc_wall_ns &&
         !storage_owner_max_rpc_wall_ns_.compare_exchange_weak(
           observed_max, rpc_wall_ns, std::memory_order_relaxed,
           std::memory_order_relaxed)) {
  }
  if (state.completed_rpc_batches >= 32 &&
      (state.completed_rpc_batches & (state.completed_rpc_batches - 1)) == 0) {
    const double average_items = static_cast<double>(
      state.completed_rpc_items) / state.completed_rpc_batches;
    const double average_wall_us = static_cast<double>(
      state.completed_rpc_wall_ns) /
      (1000.0 * static_cast<double>(state.completed_rpc_batches));
    std::cerr << "[storage-owner] sender completion telemetry owner="
              << owner_storage
              << " batches=" << state.completed_rpc_batches
              << " items=" << state.completed_rpc_items
              << " avg_batch=" << average_items
              << " avg_rpc_wall_us=" << average_wall_us
              << " max_rpc_wall_us=" << (state.max_rpc_wall_ns / 1000.0)
              << " max_active_rpcs=" << state.max_active_rpcs
              << " max_published=" << state.max_published_tasks
              << std::endl;
  }
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

    commit_storage_owner_slot(ready.owner_storage, ready.slot_id);

    release_storage_owner_slot(ready.owner_storage, ready.slot_id);
  }
}

void ComputeService::commit_storage_owner_slot(
    u32 owner_storage, u32 slot_id) {
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
  bool response_ok = slot.response_valid &&
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
  const auto* mutation_results =
    service::storage_owner::response_mutation_results(
      response_buffer, slot.item_count);
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

  for (u32 i = 0; i < slot.item_count; ++i) {
    const u32 task_id = slot.tasks[i];
    auto& task = state.tasks[task_id];
    auto& sample = storage_completion_samples_[task.completion_id];
    const bool committed = response_ok && statuses[i] == 0;
    if (committed && mutation_results[i].maintenance_sequence != 0 &&
        storage_maintenance_targets_ != nullptr) {
      const RemotePtr new_pointer{mutation_results[i].new_rptr_raw};
      const RemotePtr old_pointer{mutation_results[i].old_rptr_raw};
      const RemotePtr maintenance_home =
        !new_pointer.is_null() ? new_pointer : old_pointer;
      if (!maintenance_home.is_null() &&
          maintenance_home.memory_node() < num_servers_) {
        auto& target =
          storage_maintenance_targets_[maintenance_home.memory_node()];
        u64 observed = target.load(std::memory_order_relaxed);
        while (observed < mutation_results[i].maintenance_sequence &&
               !target.compare_exchange_weak(
                 observed, mutation_results[i].maintenance_sequence,
                 std::memory_order_release, std::memory_order_relaxed)) {
        }
      }
    }
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

void ComputeService::complete_storage_owner_task(
    u32 owner_storage, u32 task_id, bool success) {
  auto& state = *storage_insert_owners_[owner_storage];
  lib_assert(task_id < state.task_capacity,
             "storage-owner completion references an invalid task");
  auto& task = state.tasks[task_id];
  const u32 completion_id = task.completion_id;
  task.encoded_vector.clear();
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
  const auto finished_at = std::chrono::steady_clock::now();
  for (const u32 task_id : tasks) {
    auto& task = storage_insert_owners_[owner_storage]->tasks[task_id];
    auto& sample = storage_completion_samples_[task.completion_id];
    if (!sample.finished_flag) sample.mark_finished(finished_at);
    complete_storage_owner_task(owner_storage, task_id, false);
  }
  tasks.clear();
}
