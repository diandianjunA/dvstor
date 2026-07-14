#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

void ComputeService::run_storage_insert_sender(u32 owner_storage) {
  auto& state = *storage_insert_owners_[owner_storage];
  for (;;) {
    vec<std::unique_ptr<StorageInsertTask>> owned_tasks;
    u64 batch_wait_ns = 0;
    std::chrono::steady_clock::time_point owner_selected_at{};
    u32 slot_id = 0;
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      state.cv.wait(lock, [&]() {
        return (storage_insert_shutdown_.load(std::memory_order_acquire) && state.queue.empty()) ||
               (!state.queue.empty() && !state.free_slots.empty());
      });

      if (storage_insert_shutdown_.load(std::memory_order_acquire) && state.queue.empty()) {
        return;
      }
      if (state.queue.empty() || state.free_slots.empty()) {
        continue;
      }
      if (breakdown_enabled_) {
        owner_selected_at = std::chrono::steady_clock::now();
      }

      if (config_.storage_owner_batch_wait_us > 0 &&
          state.queue.size() < config_.storage_owner_batch_max &&
          !storage_insert_shutdown_.load(std::memory_order_acquire)) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::microseconds(config_.storage_owner_batch_wait_us);
        std::chrono::steady_clock::time_point batch_wait_start{};
        if (breakdown_enabled_) {
          batch_wait_start = std::chrono::steady_clock::now();
        }
        while (state.queue.size() < config_.storage_owner_batch_max &&
               !storage_insert_shutdown_.load(std::memory_order_acquire)) {
          if (state.cv.wait_until(lock, deadline) == std::cv_status::timeout) {
            break;
          }
        }
        if (breakdown_enabled_) {
          batch_wait_ns = duration_ns(batch_wait_start, std::chrono::steady_clock::now());
        }
      }

      slot_id = state.free_slots.front();
      state.free_slots.pop_front();
      const size_t batch_size = std::min<size_t>(state.queue.size(), config_.storage_owner_batch_max);
      owned_tasks.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        if (state.queue.front()->sample &&
            state.queue.front()->sample->collects_breakdown()) {
          state.queue.front()->sender_dequeued_at = owner_selected_at;
        }
        owned_tasks.push_back(std::move(state.queue.front()));
        state.queue.pop_front();
      }
    }

    if (owned_tasks.empty()) {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.free_slots.push_back(slot_id);
      state.cv.notify_one();
      continue;
    }
    post_storage_owner_batch(owner_storage, slot_id, std::move(owned_tasks), batch_wait_ns);
  }
}

void ComputeService::post_storage_owner_batch(
    u32 owner_storage,
    u32 slot_id,
    vec<std::unique_ptr<StorageInsertTask>>&& tasks,
    u64 batch_wait_ns) {
  if (tasks.empty()) {
    return;
  }

  const u32 item_count = static_cast<u32>(tasks.size());
  if (persistent_search_ != nullptr &&
      !persistent_search_->try_reserve_mutation_capacity(item_count)) {
    static std::atomic<u32> capacity_rejection_logs{0};
    const u32 log_index = capacity_rejection_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] write rejected before remote commit: "
                << "bounded GPU mutation tier is full" << std::endl;
    }
    fail_storage_owner_tasks(tasks);
    auto& state = *storage_insert_owners_[owner_storage];
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.free_slots.push_back(slot_id);
    }
    state.cv.notify_one();
    return;
  }
  const bool anchor_mode = config_.storage_owner_update_mode == "local_stitch";
  const u32 anchor_hint_count = anchor_mode
                                  ? config_.storage_owner_anchor_hints : 0;
  const u64 batch_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  bool collect_breakdown = false;
  for (const auto& task : tasks) {
    if (task->sample && task->sample->collects_breakdown()) {
      collect_breakdown = true;
      break;
    }
  }
  const auto prepare_start = collect_breakdown
    ? std::chrono::steady_clock::now()
    : std::chrono::steady_clock::time_point{};
  bool mutation_request = false;
  for (const auto& task : tasks) {
    mutation_request = mutation_request || task->kind != service::storage_owner::MutationKind::insert;
  }
  const size_t request_size = mutation_request
    ? service::storage_owner::mutation_batch_request_bytes(item_count, config_.dim, anchor_hint_count)
    : service::storage_owner::insert_batch_request_bytes(item_count, config_.dim, anchor_hint_count);
  const size_t response_size = service::storage_owner::insert_batch_response_bytes(item_count);
  auto& state = *storage_insert_owners_[owner_storage];
  auto& slot = state.slots[slot_id];
  lib_assert(request_size <= slot.request_buffer.size() &&
             response_size <= slot.response_buffer.size(),
             "storage_owner RPC slot buffer is too small for this batch");
  lib_assert(request_size <= std::numeric_limits<u32>::max() &&
             response_size <= std::numeric_limits<u32>::max(),
             "storage_owner RPC message is too large for verbs SGEs");
  slot.tasks = std::move(tasks);
  slot.samples.clear();
  slot.samples.reserve(slot.tasks.size());
  for (const auto& task : slot.tasks) {
    slot.samples.push_back(task->sample);
  }

  auto* request = reinterpret_cast<service::storage_owner::InsertBatchRequestHeader*>(slot.request_buffer.data());
  request->magic = mutation_request ? service::storage_owner::kMutationMagic
                                    : service::storage_owner::kInsertMagic;
  request->dim = config_.dim;
  request->owner_storage = owner_storage;
  request->source_client = cm_.client_id;
  request->item_count = item_count;
  request->vector_dtype = static_cast<u32>(VamanaNode::vector_dtype());
  request->vector_bytes = static_cast<u32>(VamanaNode::vector_bytes());
  request->anchor_hint_count = anchor_hint_count;
  request->batch_id = batch_id;

  node_t* ids = mutation_request
    ? service::storage_owner::mutation_request_ids(slot.request_buffer.data())
    : service::storage_owner::request_ids(slot.request_buffer.data());
  byte_t* vectors = mutation_request
    ? service::storage_owner::mutation_request_vectors(slot.request_buffer.data(), item_count)
    : service::storage_owner::request_vectors(slot.request_buffer.data(), item_count);
  u32* kinds = mutation_request ? service::storage_owner::mutation_request_kinds(slot.request_buffer.data())
                                : nullptr;
  u64* anchor_hints = mutation_request
    ? service::storage_owner::mutation_request_anchor_hints(slot.request_buffer.data(), item_count)
    : service::storage_owner::request_anchor_hints(slot.request_buffer.data(), item_count);
  for (u32 i = 0; i < item_count; ++i) {
    ids[i] = slot.tasks[i]->item.id;
    if (kinds != nullptr) {
      kinds[i] = static_cast<u32>(slot.tasks[i]->kind);
    }
    byte_t* vector_output = vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes();
    if (slot.tasks[i]->kind == service::storage_owner::MutationKind::erase) {
      std::memset(vector_output, 0, VamanaNode::vector_bytes());
    } else {
      encode_float_vector_to_storage(slot.tasks[i]->item.values.data(), config_.dim,
                                     VamanaNode::vector_dtype(), vector_output);
    }
    for (u32 hint = 0; hint < anchor_hint_count; ++hint) {
      anchor_hints[static_cast<size_t>(i) * anchor_hint_count + hint] =
        hint < slot.tasks[i]->anchor_hints.size()
          ? slot.tasks[i]->anchor_hints[hint].raw_address : 0;
    }
  }

  const u64 request_prepare_ns = collect_breakdown
    ? duration_ns(prepare_start, std::chrono::steady_clock::now())
    : 0;
  std::memset(slot.response_buffer.data(), 0, response_size);

  const u64 wr_id = storage_owner_wr_id(owner_storage, slot_id);
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    slot.in_use = true;
    slot.send_done = false;
    slot.response_done = false;
    slot.results_completed = false;
    slot.completion_claimed = false;
    slot.gpu_reserved_items = persistent_search_ == nullptr ? 0 : item_count;
    slot.item_count = item_count;
    slot.batch_id = batch_id;
    slot.batch_wait_ns = batch_wait_ns;
    slot.request_prepare_ns = request_prepare_ns;
    slot.request_size = request_size;
    slot.response_size = response_size;
    slot.send_posted_at = std::chrono::steady_clock::now();
    state.batch_to_slot[batch_id] = slot_id;
  }
  storage_insert_inflight_.fetch_add(1, std::memory_order_acq_rel);
  auto& qp = *cm_.server_qps[owner_storage];
  qp.post_send_with_id(*slot.request_region,
                       static_cast<u32>(request_size),
                       IBV_WR_SEND,
                       wr_id,
                       true,
                       nullptr,
                       0,
                       0);
}
