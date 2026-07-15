#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

void ComputeService::run_storage_insert_progress_loop() {
  vec<ibv_wc> send_wcs(std::max<i32>(1, config_.max_send_queue_wr));
  vec<ibv_wc> recv_wcs(std::max<i32>(1, config_.max_recv_queue_wr));
  u32 first_owner = 0;
  auto previous_poll = std::chrono::steady_clock::now();

  for (;;) {
    bool progressed = false;
    reclaim_storage_owner_slots();

    const auto poll_started = std::chrono::steady_clock::now();
    storage_insert_current_cq_gap_ns_ =
      duration_ns(previous_poll, poll_started);
    previous_poll = poll_started;

    const i32 send_count = Context::poll_send_cq(
      send_wcs.data(), static_cast<i32>(send_wcs.size()),
      context_.get_send_cq(), [&](u64 wr_id) {
        const auto [owner_storage, slot_id] = decode_64bit(wr_id);
        handle_storage_owner_send_completion(owner_storage, slot_id);
      });
    progressed = send_count > 0;

    const i32 recv_count = context_.poll_recv_cq(
      recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
    progressed = progressed || recv_count > 0;
    for (i32 i = 0; i < recv_count; ++i) {
      const auto [owner_storage, slot_id] = decode_64bit(recv_wcs[i].wr_id);
      handle_storage_owner_response(owner_storage, slot_id);
    }

    progressed = drain_storage_owner_submissions(first_owner) || progressed;
    reclaim_storage_owner_slots();

    bool submissions_empty = true;
    for (const auto& state : storage_insert_owners_) {
      submissions_empty = submissions_empty && state->queue->empty();
    }
    if (storage_insert_shutdown_.load(std::memory_order_acquire) &&
        submissions_empty &&
        storage_insert_inflight_.load(std::memory_order_acquire) == 0) {
      storage_insert_progress_done_.store(true, std::memory_order_release);
      storage_insert_progress_done_.notify_all();
      storage_ready_slots_->notify_all();
      return;
    }
    if (!progressed) std::this_thread::yield();
  }
}

bool ComputeService::drain_storage_owner_submissions(u32& first_owner) {
  const u32 owner_count = static_cast<u32>(storage_insert_owners_.size());
  if (owner_count == 0) return false;
  bool progressed = false;
  for (u32 offset = 0; offset < owner_count; ++offset) {
    const u32 owner = (first_owner + offset) % owner_count;
    auto& state = *storage_insert_owners_[owner];
    while (!state.free_slots.empty()) {
      u32 first_task = 0;
      if (!state.queue->try_pop(first_task)) break;

      const u32 slot_id = state.free_slots.back();
      state.free_slots.pop_back();
      auto& slot = state.slots[slot_id];
      slot.tasks.clear();
      slot.tasks.push_back(first_task);
      u32 task_id = 0;
      while (slot.tasks.size() < config_.storage_owner_batch_max &&
             state.queue->try_pop(task_id)) {
        slot.tasks.push_back(task_id);
      }
      const auto dequeued_at = std::chrono::steady_clock::now();
      for (const u32 id : slot.tasks) {
        state.tasks[id].sender_dequeued_at = dequeued_at;
      }
      post_storage_owner_batch(owner, slot_id);
      progressed = true;
    }
  }
  first_owner = (first_owner + 1) % owner_count;
  return progressed;
}

void ComputeService::reclaim_storage_owner_slots() {
  StorageOwnerReleasedSlot released;
  while (storage_released_slots_->try_pop(released)) {
    lib_assert(released.owner_storage < storage_insert_owners_.size(),
               "storage-owner release references an invalid owner");
    auto& state = *storage_insert_owners_[released.owner_storage];
    lib_assert(released.slot_id < state.slots.size() &&
                 released.response_slot_id < state.response_slots.size(),
               "storage-owner release references an invalid slot");
    auto& slot = state.slots[released.slot_id];
    lib_assert(slot.in_use && slot.results_completed,
               "storage-owner released a slot before completion");
    slot.in_use = false;
    slot.send_done = false;
    slot.response_done = false;
    slot.results_completed = false;
    slot.completion_claimed = false;
    slot.response_slot_id = std::numeric_limits<u32>::max();
    slot.gpu_reserved_items = 0;
    slot.item_count = 0;
    slot.batch_id = 0;
    slot.request_prepare_ns = 0;
    slot.cq_progress_gap_ns = 0;
    slot.request_size = 0;
    slot.response_size = 0;
    slot.send_posted_at = {};
    slot.send_completed_at = {};
    slot.response_completed_at = {};
    slot.tasks.clear();
    state.free_slots.push_back(released.slot_id);
    post_storage_owner_response_receive(
      released.owner_storage, released.response_slot_id);
    storage_insert_inflight_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

void ComputeService::post_storage_owner_batch(
    u32 owner_storage,
    u32 slot_id) {
  auto& state = *storage_insert_owners_[owner_storage];
  auto& slot = state.slots[slot_id];
  if (slot.tasks.empty()) return;

  const u32 item_count = static_cast<u32>(slot.tasks.size());
  const bool anchor_mode = config_.storage_owner_update_mode == "local_stitch";
  const u32 anchor_hint_count = anchor_mode
    ? config_.storage_owner_anchor_hints : 0;
  const u64 batch_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  bool collect_breakdown = false;
  bool mutation_request = false;
  for (const u32 task_id : slot.tasks) {
    const auto& task = state.tasks[task_id];
    const auto& sample = storage_completion_samples_[task.completion_id];
    collect_breakdown = collect_breakdown || sample.collects_breakdown();
    mutation_request = mutation_request ||
      task.kind != service::storage_owner::MutationKind::insert;
  }
  const auto prepare_start = collect_breakdown
    ? std::chrono::steady_clock::now()
    : std::chrono::steady_clock::time_point{};
  const size_t request_size = mutation_request
    ? service::storage_owner::mutation_batch_request_bytes(
        item_count, config_.dim, anchor_hint_count)
    : service::storage_owner::insert_batch_request_bytes(
        item_count, config_.dim, anchor_hint_count);
  const size_t response_size =
    service::storage_owner::insert_batch_response_bytes(item_count);
  lib_assert(request_size <= slot.request_buffer.size(),
             "storage_owner RPC request slot is too small for this batch");
  lib_assert(request_size <= std::numeric_limits<u32>::max() &&
               response_size <= std::numeric_limits<u32>::max(),
             "storage_owner RPC message is too large for verbs SGEs");
  auto* request = reinterpret_cast<
    service::storage_owner::InsertBatchRequestHeader*>(
      slot.request_buffer.data());
  request->magic = mutation_request
    ? service::storage_owner::kMutationMagic
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
    ? service::storage_owner::mutation_request_vectors(
        slot.request_buffer.data(), item_count)
    : service::storage_owner::request_vectors(
        slot.request_buffer.data(), item_count);
  u32* kinds = mutation_request
    ? service::storage_owner::mutation_request_kinds(slot.request_buffer.data())
    : nullptr;
  u64* anchor_hints = mutation_request
    ? service::storage_owner::mutation_request_anchor_hints(
        slot.request_buffer.data(), item_count)
    : service::storage_owner::request_anchor_hints(
        slot.request_buffer.data(), item_count);
  for (u32 i = 0; i < item_count; ++i) {
    const auto& task = state.tasks[slot.tasks[i]];
    ids[i] = task.item.id;
    if (kinds != nullptr) kinds[i] = static_cast<u32>(task.kind);
    byte_t* vector_output =
      vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes();
    if (task.kind == service::storage_owner::MutationKind::erase) {
      std::memset(vector_output, 0, VamanaNode::vector_bytes());
    } else {
      encode_float_vector_to_storage(
        task.item.values.data(), config_.dim,
        VamanaNode::vector_dtype(), vector_output);
    }
    for (u32 hint = 0; hint < anchor_hint_count; ++hint) {
      anchor_hints[static_cast<size_t>(i) * anchor_hint_count + hint] =
        hint < task.anchor_hints.size()
          ? task.anchor_hints[hint].raw_address : 0;
    }
  }

  slot.in_use = true;
  slot.send_done = false;
  slot.response_done = false;
  slot.results_completed = false;
  slot.completion_claimed = false;
  slot.response_slot_id = std::numeric_limits<u32>::max();
  slot.gpu_reserved_items = item_count;
  slot.item_count = item_count;
  slot.batch_id = batch_id;
  slot.request_prepare_ns = collect_breakdown
    ? duration_ns(prepare_start, std::chrono::steady_clock::now()) : 0;
  slot.request_size = request_size;
  slot.response_size = response_size;
  slot.cq_progress_gap_ns = 0;
  slot.send_posted_at = std::chrono::steady_clock::now();
  storage_insert_inflight_.fetch_add(1, std::memory_order_acq_rel);

  cm_.server_qps[owner_storage]->post_send_with_id(
    *slot.request_region,
    static_cast<u32>(request_size),
    IBV_WR_SEND,
    storage_owner_wr_id(owner_storage, slot_id),
    true,
    nullptr,
    0,
    0);
}
