#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

void ComputeService::start_storage_insert_runtime() {
  if (!storage_insert_owners_.empty()) {
    return;
  }

  const u32 owner_count = std::max<u32>(1, num_servers_);
  const u32 rpc_depth = std::max<u32>(1, config_.storage_owner_rpc_depth);
  const bool anchor_mode = config_.storage_owner_update_mode == "local_stitch";
  const size_t request_bytes = std::max(
    service::storage_owner::insert_batch_request_bytes(
      config_.storage_owner_batch_max, config_.dim,
      anchor_mode ? config_.storage_owner_anchor_hints : 0),
    service::storage_owner::mutation_batch_request_bytes(
      config_.storage_owner_batch_max, config_.dim,
      anchor_mode ? config_.storage_owner_anchor_hints : 0));
  const size_t response_bytes =
    service::storage_owner::insert_batch_response_bytes(config_.storage_owner_batch_max);
  lib_assert(request_bytes <= std::numeric_limits<u32>::max() &&
             response_bytes <= std::numeric_limits<u32>::max(),
             "storage_owner RPC message is too large for verbs SGEs; reduce batch size or vector dimension");
  const size_t max_inflight = static_cast<size_t>(owner_count) * rpc_depth;
  lib_assert(max_inflight <= static_cast<size_t>(config_.max_send_queue_wr),
             "storage_owner RPC depth exceeds compute send CQ capacity");
  lib_assert(max_inflight <= static_cast<size_t>(config_.max_recv_queue_wr),
             "storage_owner RPC depth exceeds compute receive CQ capacity");

  storage_insert_shutdown_.store(false, std::memory_order_release);
  storage_insert_senders_done_.store(false, std::memory_order_release);
  storage_insert_completion_done_.store(false, std::memory_order_release);
  storage_insert_inflight_.store(0, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(storage_insert_publication_mutex_);
    storage_insert_publication_queue_.clear();
  }
  storage_insert_owners_.reserve(owner_count);
  for (u32 owner = 0; owner < owner_count; ++owner) {
    auto state = std::make_unique<StorageOwnerSenderState>();
    state->slots.resize(rpc_depth);
    for (u32 slot_id = 0; slot_id < rpc_depth; ++slot_id) {
      auto& slot = state->slots[slot_id];
      slot.owner_storage = owner;
      slot.slot_id = slot_id;
      slot.request_buffer.assign(request_bytes, 0);
      slot.response_buffer.assign(response_bytes, 0);
      slot.request_region =
        std::make_unique<LocalMemoryRegion>(context_, slot.request_buffer.data(), slot.request_buffer.size());
      slot.tasks.reserve(config_.storage_owner_batch_max);
      slot.samples.reserve(config_.storage_owner_batch_max);
      state->free_slots.push_back(slot_id);
    }
    state->response_slots.resize(rpc_depth);
    for (u32 response_slot_id = 0; response_slot_id < rpc_depth; ++response_slot_id) {
      auto& response_slot = state->response_slots[response_slot_id];
      response_slot.owner_storage = owner;
      response_slot.slot_id = response_slot_id;
      response_slot.buffer.assign(response_bytes, 0);
      response_slot.region =
        std::make_unique<LocalMemoryRegion>(context_, response_slot.buffer.data(), response_slot.buffer.size());
    }
    storage_insert_owners_.push_back(std::move(state));
  }
  for (u32 owner = 0; owner < owner_count; ++owner) {
    for (u32 response_slot_id = 0; response_slot_id < rpc_depth; ++response_slot_id) {
      post_storage_owner_response_receive(owner, response_slot_id);
    }
  }

  storage_insert_publication_thread_ =
    std::thread([this]() { run_storage_insert_publication_loop(); });
  storage_insert_completion_thread_ =
    std::thread([this]() { run_storage_insert_completion_loop(); });
  print_status(
    "storage-owner acknowledgement=durable stage1 commit; "
    "GPU visibility=ordered asynchronous publication");
  for (u32 owner = 0; owner < owner_count; ++owner) {
    storage_insert_owners_[owner]->thread =
      std::thread([this, owner]() { run_storage_insert_sender(owner); });
  }
}

void ComputeService::stop_storage_insert_runtime() {
  storage_insert_shutdown_.store(true, std::memory_order_release);
  for (auto& state : storage_insert_owners_) {
    if (state) {
      state->cv.notify_all();
    }
  }

  for (auto& state : storage_insert_owners_) {
    if (state && state->thread.joinable()) {
      state->thread.join();
    }
  }

  storage_insert_senders_done_.store(true, std::memory_order_release);
  storage_insert_publication_cv_.notify_all();
  if (storage_insert_completion_thread_.joinable()) {
    storage_insert_completion_thread_.join();
  }
  storage_insert_publication_cv_.notify_all();
  if (storage_insert_publication_thread_.joinable()) {
    storage_insert_publication_thread_.join();
  }

  for (auto& state : storage_insert_owners_) {
    if (!state) {
      continue;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    while (!state->queue.empty()) {
      vec<std::unique_ptr<StorageInsertTask>> task;
      task.push_back(std::move(state->queue.front()));
      state->queue.pop_front();
      fail_storage_owner_tasks(task);
    }
    for (auto& slot : state->slots) {
      if (slot.in_use && !slot.results_completed) {
        fail_storage_owner_tasks(slot.tasks);
      }
      if (persistent_search_ != nullptr && slot.gpu_reserved_items != 0) {
        persistent_search_->release_mutation_capacity(slot.gpu_reserved_items);
        slot.gpu_reserved_items = 0;
      }
      slot.in_use = false;
      slot.send_done = true;
      slot.response_done = true;
      slot.results_completed = true;
      slot.completion_claimed = false;
    }
  }
  storage_insert_inflight_.store(0, std::memory_order_release);
  storage_insert_completion_done_.store(true, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(storage_insert_publication_mutex_);
    storage_insert_publication_queue_.clear();
  }
}

void ComputeService::release_storage_insert_runtime() {
  for (auto& state : storage_insert_owners_) {
    if (!state) {
      continue;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    for (auto& slot : state->slots) {
      slot.request_region.reset();
      slot.response_region.reset();
    }
    for (auto& response_slot : state->response_slots) {
      response_slot.region.reset();
    }
    state->slots.clear();
    state->response_slots.clear();
    state->batch_to_slot.clear();
    state->free_slots.clear();
  }
  storage_insert_owners_.clear();
}
