#include "service/compute_service/detail.hh"
#include "service/compute_service/storage_owner/batch_policy.hh"

using namespace compute_service_detail;

void ComputeService::run_storage_insert_progress_loop() {
  vec<ibv_wc> send_wcs(std::max<i32>(1, config_.max_send_queue_wr));
  vec<ibv_wc> recv_wcs(std::max<i32>(1, config_.max_recv_queue_wr));
  u32 first_owner = 0;

  for (;;) {
    // Shutdown must remain bounded even when a remote process has disappeared.
    // Outstanding request/response buffers stay registered until the QPs are
    // destroyed by ComputeService::~ComputeService(), so it is safe to stop CQ
    // progress here and fail the still-owned tasks in stop_storage_insert_runtime().
    if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
      storage_insert_progress_done_.store(true, std::memory_order_release);
      storage_insert_progress_done_.notify_all();
      storage_ready_slots_->notify_all();
      return;
    }
    bool progressed = false;
    reclaim_storage_owner_slots();

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
      const auto [encoded_owner, slot_id] = decode_64bit(recv_wcs[i].wr_id);
      const u32 owner_storage = storage_owner_wr_owner(encoded_owner);
      if (owner_storage >= storage_insert_owners_.size()) continue;
      auto& state = *storage_insert_owners_[owner_storage];
      const bool completion_pool =
        storage_owner_is_completion_wr(encoded_owner);
      const byte_t* receive_buffer = nullptr;
      if (completion_pool) {
        if (slot_id >= state.completion_slot_count) continue;
        receive_buffer = state.completion_buffer.data() +
          static_cast<size_t>(slot_id) *
            sizeof(service::storage_owner::MutationCompletionV2);
      } else {
        if (slot_id >= state.response_slots.size()) continue;
        receive_buffer = state.response_slots[slot_id].buffer.data();
      }

      // Every receive WQE on an RC QP is interchangeable. Copy and classify
      // the envelope before reposting the same physical slot; WR-ID denotes
      // buffer ownership only, never the incoming message type.
      u32 magic = 0;
      if (recv_wcs[i].byte_len >= sizeof(magic)) {
        std::memcpy(&magic, receive_buffer, sizeof(magic));
      }
      service::storage_owner::MutationBatchAckV2 ack{};
      service::storage_owner::MutationCompletionV2 completion{};
      if (magic == service::storage_owner::kMutationCompletionMagic &&
          recv_wcs[i].byte_len == sizeof(completion)) {
        std::memcpy(&completion, receive_buffer, sizeof(completion));
      } else if (recv_wcs[i].byte_len == sizeof(ack)) {
        std::memcpy(&ack, receive_buffer, sizeof(ack));
      }
      if (completion_pool) {
        post_storage_owner_completion_receive(owner_storage, slot_id);
      } else {
        post_storage_owner_response_receive(owner_storage, slot_id);
      }
      if (magic == service::storage_owner::kMutationCompletionMagic) {
        handle_storage_owner_token_completion(
          owner_storage, completion, recv_wcs[i].byte_len);
      } else {
        handle_storage_owner_response(
          owner_storage, ack, recv_wcs[i].byte_len);
      }
    }

    progressed = drain_storage_owner_submissions(first_owner) || progressed;
    reclaim_storage_owner_slots();

    if (!progressed) std::this_thread::yield();
  }
}

bool ComputeService::drain_storage_owner_submissions(u32& first_owner) {
  const u32 owner_count = static_cast<u32>(storage_insert_owners_.size());
  if (owner_count == 0) return false;
  const u64 max_wait_ns =
    static_cast<u64>(config_.storage_owner_batch_max_wait_us) * 1000ull;
  const bool flow_control =
    config_.storage_owner_flow_control_backlog_low != 0 &&
    storage_flow_control_next_sequences_ != nullptr;
  const u64 flow_control_max_delay_ns =
    static_cast<u64>(config_.storage_owner_flow_control_max_delay_us) *
      1000ull;
  const auto now_ns = []() {
    return static_cast<u64>(std::chrono::duration_cast<
      std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
  };
  bool progressed = false;
  for (u32 offset = 0; offset < owner_count; ++offset) {
    const u32 owner = (first_owner + offset) % owner_count;
    auto& state = *storage_insert_owners_[owner];
    const u64 owner_now_ns = now_ns();
    if (!storage_owner_retry_deadline_elapsed(
          state.retry_not_before_ns.load(std::memory_order_acquire),
          owner_now_ns)) {
      continue;
    }
    const u32 batch_max = std::max<u32>(
      1, config_.storage_owner_batch_max);
    const u32 initially_ready = state.published_tasks.load(
      std::memory_order_acquire);
    state.max_published_tasks = std::max(
      state.max_published_tasks, initially_ready);
    while (!state.free_slots.empty()) {
      const u64 observed_now_ns = now_ns();
      const u32 free = static_cast<u32>(state.free_slots.size());
      const u32 active = static_cast<u32>(state.slots.size()) - free;
      const u32 ready = state.published_tasks.load(
        std::memory_order_acquire);
      if (ready == 0) {
        state.oldest_published_observed_ns = 0;
        break;
      }
      if (state.oldest_published_observed_ns == 0) {
        state.oldest_published_observed_ns = observed_now_ns;
      }
      const StorageOwnerBatchDecision decision = decide_storage_owner_batch(
        ready, active, free,
        state.pending_producers.load(std::memory_order_acquire), batch_max,
        state.oldest_published_observed_ns, observed_now_ns, max_wait_ns);
      if (decision.take == 0) break;
      state.max_active_rpcs = std::max(state.max_active_rpcs, active);

      const u32 slot_id = state.free_slots.back();
      state.free_slots.pop_back();
      auto& slot = state.slots[slot_id];
      slot.tasks.clear();
      const u32 dequeued = dequeue_storage_owner_visible_prefix(
        *state.queue, decision.take, slot.tasks);
      if (dequeued == 0) {
        // A producer can be preempted after reserving the FIFO head but before
        // publishing it. Later cells may already contribute to
        // published_tasks, yet they cannot be popped past that transient
        // hole. Preserve both slot and counter and retry on a later progress
        // pass; blocking here would also stop CQ progress.
        ++state.queue_visibility_stalls;
        state.free_slots.push_back(slot_id);
        break;
      }
      state.partial_visible_batches += dequeued < decision.take;
      const u32 previous = state.published_tasks.fetch_sub(
        dequeued, std::memory_order_acq_rel);
      lib_assert(previous >= dequeued,
                 "storage-owner published task counter underflow");
      const auto dequeued_at = std::chrono::steady_clock::now();
      const u64 dequeued_at_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          dequeued_at.time_since_epoch()).count());
      ++state.rpc_batches;
      state.rpc_items += dequeued;
      state.full_batches += dequeued == batch_max;
      state.tail_escape_batches += decision.tail_escape;
      state.max_wait_flush_batches += decision.max_wait_flush;
      const u32 remaining_published = state.published_tasks.load(
        std::memory_order_acquire);
      state.oldest_published_observed_ns =
        next_storage_owner_batch_observed_ns(
          state.oldest_published_observed_ns, remaining_published,
          dequeued_at_ns);
      if (state.rpc_batches >= 32 &&
          (state.rpc_batches & (state.rpc_batches - 1)) == 0) {
        const double average_batch = static_cast<double>(state.rpc_items) /
          static_cast<double>(state.rpc_batches);
        std::cerr << "[storage-owner] sender batch telemetry owner="
                  << owner
                  << " batches=" << state.rpc_batches
                  << " items=" << state.rpc_items
                  << " avg_batch=" << average_batch
                  << " full_batches=" << state.full_batches
                  << " tail_escape_batches=" << state.tail_escape_batches
                  << " max_wait_flush_batches="
                  << state.max_wait_flush_batches
                  << " queue_visibility_stalls="
                  << state.queue_visibility_stalls
                  << " partial_visible_batches="
                  << state.partial_visible_batches
                  << " published="
                  << state.published_tasks.load(std::memory_order_relaxed);
        if (state.flow_control_throttled_batches != 0) {
          std::cerr << " flow_control_throttled_batches="
                    << state.flow_control_throttled_batches
                    << " flow_control_avg_delay_us="
                    << (static_cast<double>(state.flow_control_delay_ns_total) /
                        static_cast<double>(
                          state.flow_control_throttled_batches) / 1000.0)
                    << " flow_control_max_backlog="
                    << state.flow_control_max_backlog;
        }
        std::cerr << std::endl;
      }
      post_storage_owner_batch(owner, slot_id);
      state.max_active_rpcs = std::max(
        state.max_active_rpcs,
        static_cast<u32>(state.slots.size() - state.free_slots.size()));
      progressed = true;
      if (flow_control) {
        // Read durable before next so a concurrent refresh can only
        // overestimate the backlog (see the poller's store order).
        const u64 durable =
          storage_flow_control_durable_sequences_[owner].load(
            std::memory_order_acquire);
        const u64 next =
          storage_flow_control_next_sequences_[owner].load(
            std::memory_order_acquire);
        const u64 target =
          storage_maintenance_targets_[owner].load(
            std::memory_order_acquire);
        const u64 backlog =
          storage_owner_flow_control_backlog(next, target, durable);
        const u64 delay_ns = storage_owner_flow_control_delay_ns(
          backlog,
          config_.storage_owner_flow_control_backlog_low,
          config_.storage_owner_flow_control_backlog_high,
          flow_control_max_delay_ns);
        if (delay_ns != 0) {
          // Pace instead of blocking: one batch per delay interval per owner.
          // Raise-only so a concurrent busy-ACK backoff with a later deadline
          // is never weakened. A token completion may still clear the gate
          // early; the next posted batch simply re-arms it.
          const u64 not_before = now_ns() + delay_ns;
          u64 observed_deadline =
            state.retry_not_before_ns.load(std::memory_order_acquire);
          while (observed_deadline < not_before &&
                 !state.retry_not_before_ns.compare_exchange_weak(
                   observed_deadline, not_before,
                   std::memory_order_release, std::memory_order_acquire)) {
          }
          ++state.flow_control_throttled_batches;
          state.flow_control_delay_ns_total += delay_ns;
          state.flow_control_max_backlog =
            std::max(state.flow_control_max_backlog, backlog);
          break;
        }
      }
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
    lib_assert(released.slot_id < state.slots.size(),
               "storage-owner release references an invalid slot");
    auto& slot = state.slots[released.slot_id];
    lib_assert(slot.in_use && slot.results_completed,
               "storage-owner released a slot before completion");
    slot.in_use = false;
    slot.send_done = false;
    slot.response_done = false;
    slot.results_completed = false;
    slot.completion_claimed = false;
    slot.response_valid = false;
    slot.response_ack = {};
    slot.item_count = 0;
    slot.batch_id = 0;
    slot.send_posted_at = {};
    slot.tasks.clear();
    state.free_slots.push_back(released.slot_id);
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
  const u64 batch_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  bool mutation_request = false;
  for (const u32 task_id : slot.tasks) {
    const auto& task = state.tasks[task_id];
    mutation_request = mutation_request ||
      task.kind != service::storage_owner::MutationKind::insert;
  }
  const size_t request_size = mutation_request
    ? service::storage_owner::mutation_batch_request_bytes(item_count)
    : service::storage_owner::insert_batch_request_bytes(item_count);
  const size_t response_size =
    sizeof(service::storage_owner::MutationBatchAckV2);
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
  request->source_client = storage_operation_source_;
  request->item_count = item_count;
  request->vector_dtype = static_cast<u32>(VamanaNode::vector_dtype());
  request->vector_bytes = static_cast<u32>(VamanaNode::vector_bytes());
  request->protocol_version = service::storage_owner::kMutationProtocolVersion;
  request->batch_id = batch_id;

  node_t* ids = mutation_request
    ? service::storage_owner::mutation_request_ids(slot.request_buffer.data())
    : service::storage_owner::request_ids(slot.request_buffer.data());
  u64* operation_ids = mutation_request
    ? service::storage_owner::mutation_request_operation_ids(
        slot.request_buffer.data(), item_count)
    : service::storage_owner::request_operation_ids(
        slot.request_buffer.data(), item_count);
  byte_t* vectors = mutation_request
    ? service::storage_owner::mutation_request_vectors(
        slot.request_buffer.data(), item_count)
    : service::storage_owner::request_vectors(
        slot.request_buffer.data(), item_count);
  u32* kinds = mutation_request
    ? service::storage_owner::mutation_request_kinds(slot.request_buffer.data())
    : nullptr;
  u32* stage1_homes = mutation_request
    ? service::storage_owner::mutation_request_stage1_homes(
        slot.request_buffer.data(), item_count)
    : service::storage_owner::request_stage1_homes(
        slot.request_buffer.data(), item_count);
  for (u32 i = 0; i < item_count; ++i) {
    const auto& task = state.tasks[slot.tasks[i]];
    ids[i] = task.id;
    operation_ids[i] = task.operation_id;
    stage1_homes[i] = task.stage1_home;
    if (kinds != nullptr) kinds[i] = static_cast<u32>(task.kind);
    byte_t* vector_output =
      vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes();
    if (task.kind == service::storage_owner::MutationKind::erase) {
      std::memset(vector_output, 0, VamanaNode::vector_bytes());
    } else {
      lib_assert(task.encoded_vector.size() == VamanaNode::vector_bytes(),
                 "storage-owner task lost its canonical encoded vector");
      std::memcpy(vector_output, task.encoded_vector.data(),
                  VamanaNode::vector_bytes());
    }
  }

  slot.in_use = true;
  slot.send_done = false;
  slot.response_done = false;
  slot.results_completed = false;
  slot.completion_claimed = false;
  slot.response_valid = false;
  slot.response_ack = {};
  slot.item_count = item_count;
  slot.batch_id = batch_id;
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
  storage_owner_submitted_batches_.fetch_add(1, std::memory_order_relaxed);
  storage_owner_submitted_items_.fetch_add(
    item_count, std::memory_order_relaxed);
}
