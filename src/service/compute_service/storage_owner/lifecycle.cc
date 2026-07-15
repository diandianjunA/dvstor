#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

void ComputeService::start_storage_insert_runtime() {
  if (!storage_insert_owners_.empty()) return;

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
    service::storage_owner::insert_batch_response_bytes(
      config_.storage_owner_batch_max);
  lib_assert(request_bytes <= std::numeric_limits<u32>::max() &&
               response_bytes <= std::numeric_limits<u32>::max(),
             "storage_owner RPC message is too large for verbs SGEs; "
             "reduce batch size or vector dimension");
  const size_t max_inflight = static_cast<size_t>(owner_count) * rpc_depth;
  lib_assert(max_inflight <= static_cast<size_t>(config_.max_send_queue_wr),
             "storage_owner RPC depth exceeds compute send CQ capacity");
  lib_assert(max_inflight <= static_cast<size_t>(config_.max_recv_queue_wr),
             "storage_owner RPC depth exceeds compute receive CQ capacity");

  // The bound is derived from protocol concurrency rather than exposed as a
  // benchmark tuning knob. It comfortably covers closed-loop writer bursts
  // while propagating sustained pressure back to callers.
  const u64 requested_task_capacity = std::max<u64>(
    256,
    static_cast<u64>(rpc_depth) * config_.storage_owner_batch_max * 8);
  lib_assert(requested_task_capacity <= std::numeric_limits<u32>::max(),
             "storage-owner submission capacity exceeds u32");
  const u32 task_capacity = static_cast<u32>(requested_task_capacity);
  const u64 total_task_capacity_u64 =
    static_cast<u64>(owner_count) * task_capacity;
  lib_assert(total_task_capacity_u64 <= std::numeric_limits<u32>::max(),
             "storage-owner completion capacity exceeds u32");

  storage_insert_shutdown_.store(false, std::memory_order_release);
  storage_insert_progress_done_.store(false, std::memory_order_release);
  storage_insert_inflight_.store(0, std::memory_order_release);
  storage_ready_slots_ =
    std::make_unique<bounded::Queue<StorageOwnerReadySlot>>(max_inflight);
  storage_released_slots_ =
    std::make_unique<bounded::Queue<StorageOwnerReleasedSlot>>(max_inflight);
  storage_completion_pool_ = std::make_unique<bounded::CompletionPool>(
    static_cast<u32>(total_task_capacity_u64));
  storage_completion_samples_ =
    std::make_unique<service::breakdown::Sample[]>(total_task_capacity_u64);

  storage_insert_owners_.reserve(owner_count);
  for (u32 owner = 0; owner < owner_count; ++owner) {
    auto state = std::make_unique<StorageOwnerSenderState>();
    state->task_capacity = task_capacity;
    state->queue = std::make_unique<bounded::Queue<u32>>(task_capacity);
    state->free_tasks = std::make_unique<bounded::Queue<u32>>(task_capacity);
    state->tasks = std::make_unique<StorageInsertTask[]>(task_capacity);
    for (u32 task_id = 0; task_id < task_capacity; ++task_id) {
      auto& task = state->tasks[task_id];
      task.item.values.reserve(config_.dim);
      task.anchor_hints.reserve(config_.storage_owner_anchor_hints);
      lib_assert(state->free_tasks->try_push(task_id),
                 "failed to initialize storage-owner task pool");
    }

    state->slots.resize(rpc_depth);
    state->free_slots.reserve(rpc_depth);
    for (u32 slot_id = 0; slot_id < rpc_depth; ++slot_id) {
      auto& slot = state->slots[slot_id];
      slot.owner_storage = owner;
      slot.slot_id = slot_id;
      slot.request_buffer.assign(request_bytes, 0);
      slot.request_region = std::make_unique<LocalMemoryRegion>(
        context_, slot.request_buffer.data(), slot.request_buffer.size());
      slot.tasks.reserve(config_.storage_owner_batch_max);
      state->free_slots.push_back(slot_id);
    }
    state->response_slots.resize(rpc_depth);
    for (u32 response_slot_id = 0;
         response_slot_id < rpc_depth; ++response_slot_id) {
      auto& response_slot = state->response_slots[response_slot_id];
      response_slot.owner_storage = owner;
      response_slot.slot_id = response_slot_id;
      response_slot.buffer.assign(response_bytes, 0);
      response_slot.region = std::make_unique<LocalMemoryRegion>(
        context_, response_slot.buffer.data(), response_slot.buffer.size());
    }
    storage_insert_owners_.push_back(std::move(state));
  }

  for (u32 owner = 0; owner < owner_count; ++owner) {
    for (u32 response_slot_id = 0;
         response_slot_id < rpc_depth; ++response_slot_id) {
      post_storage_owner_response_receive(owner, response_slot_id);
    }
  }

  storage_insert_completion_thread_ =
    std::thread([this]() { run_storage_insert_completion_loop(); });
  storage_insert_progress_thread_ =
    std::thread([this]() { run_storage_insert_progress_loop(); });
  if (!config_.disable_thread_pinning) {
    const u32 progress_core = core_assignment_.get_available_core();
    const u32 completion_core = core_assignment_.get_available_core();
    pin_thread(storage_insert_progress_thread_, progress_core);
    pin_thread(storage_insert_completion_thread_, completion_core);
    print_status("storage-owner update cores: progress=" +
                 std::to_string(progress_core) + " response/gpu=" +
                 std::to_string(completion_core));
  }
  print_status(
    "storage-owner acknowledgement=owner-memory stage1 publication; "
    "GPU visibility=ordered asynchronous publication; "
    "submission=bounded owner rings; progress=single work-conserving executor");
}

void ComputeService::stop_storage_insert_runtime() {
  storage_insert_shutdown_.store(true, std::memory_order_release);
  for (auto& state : storage_insert_owners_) {
    if (state && state->queue) state->queue->notify_all();
    if (state && state->free_tasks) state->free_tasks->notify_all();
  }
  if (storage_ready_slots_) storage_ready_slots_->notify_all();
  if (storage_released_slots_) storage_released_slots_->notify_all();

  if (storage_insert_progress_thread_.joinable()) {
    storage_insert_progress_thread_.join();
  }
  if (storage_ready_slots_) storage_ready_slots_->notify_all();
  if (storage_insert_completion_thread_.joinable()) {
    storage_insert_completion_thread_.join();
  }

  for (u32 owner = 0; owner < storage_insert_owners_.size(); ++owner) {
    auto& state = *storage_insert_owners_[owner];
    u32 task_id = 0;
    while (state.queue && state.queue->try_pop(task_id)) {
      vec<u32> failed{task_id};
      fail_storage_owner_tasks(owner, failed);
    }
    for (auto& slot : state.slots) {
      if (slot.in_use && !slot.results_completed) {
        fail_storage_owner_tasks(owner, slot.tasks);
      }
      slot.in_use = false;
      slot.send_done = false;
      slot.response_done = false;
      slot.results_completed = true;
      slot.completion_claimed = false;
      slot.gpu_reserved_items = 0;
    }
  }
  storage_insert_inflight_.store(0, std::memory_order_release);
  storage_insert_progress_done_.store(true, std::memory_order_release);
}

void ComputeService::release_storage_insert_runtime() {
  for (auto& state : storage_insert_owners_) {
    if (!state) continue;
    for (auto& slot : state->slots) {
      slot.request_region.reset();
      slot.tasks.clear();
    }
    for (auto& response_slot : state->response_slots) {
      response_slot.region.reset();
    }
    state->slots.clear();
    state->response_slots.clear();
    state->free_slots.clear();
    state->queue.reset();
    state->free_tasks.reset();
    state->tasks.reset();
  }
  storage_insert_owners_.clear();
  storage_ready_slots_.reset();
  storage_released_slots_.reset();
  storage_completion_pool_.reset();
  storage_completion_samples_.reset();
}
