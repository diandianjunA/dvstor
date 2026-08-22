#include "memory_node/storage_owner_runtime/detail.hh"
#include "gpu_search/maintenance_telemetry.hh"
#include "service/storage_owner_client_helpers.hh"

using namespace memory_node_storage_owner_runtime_detail;

void MemoryNode::process_storage_owner_insert_task(const StorageOwnerInsertTask& task) {
  const Configuration& config = *storage_worker_config_;
  lib_assert(current_storage_owner_thread_ != nullptr,
             "storage-owner request executed without a foreground worker");
  auto& scratch = current_storage_owner_thread_->request_scratch;
  scratch.clear();

  lib_assert(task.client_id < num_clients_ &&
               task.slot_id < insert_runtime_.request_slot_count,
             "storage-owner task references an invalid request slot");
  struct BatchContextCreditGuard {
    std::atomic<u32>* credits{};
    ~BatchContextCreditGuard() {
      if (credits != nullptr) {
        credits->fetch_add(1, std::memory_order_release);
      }
    }
  } context_credit_guard{
    &storage_client_batch_context_credits_[task.client_id]};
  const byte_t* payload = task.payload.data();
  const auto* request =
    reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(payload);
  const bool mutation = request->magic == service::storage_owner::kMutationMagic;
  const size_t expected_bytes = mutation
    ? service::storage_owner::mutation_batch_request_bytes(request->item_count)
    : service::storage_owner::insert_batch_request_bytes(request->item_count);
  lib_assert(request->item_count == task.item_count &&
               request->batch_id == task.batch_id &&
               task.completion_slots.size() == request->item_count &&
               request->protocol_version ==
                 service::storage_owner::kMutationProtocolVersion &&
               task.byte_len >= expected_bytes,
             "storage-owner request slot changed before task execution");

  const node_t* ids = mutation
    ? service::storage_owner::mutation_request_ids(payload)
    : service::storage_owner::request_ids(payload);
  const u32* stage1_homes = mutation
    ? service::storage_owner::mutation_request_stage1_homes(
        payload, request->item_count)
    : service::storage_owner::request_stage1_homes(
        payload, request->item_count);
  const u64* operation_ids = mutation
    ? service::storage_owner::mutation_request_operation_ids(
        payload, request->item_count)
    : service::storage_owner::request_operation_ids(
        payload, request->item_count);
  const byte_t* vectors = mutation
    ? service::storage_owner::mutation_request_vectors(payload, request->item_count)
    : service::storage_owner::request_vectors(payload, request->item_count);
  const u32* wire_kinds = mutation
    ? service::storage_owner::mutation_request_kinds(payload) : nullptr;
  scratch.kinds.resize(request->item_count, service::storage_owner::MutationKind::insert);
  for (u32 item = 0; item < request->item_count; ++item) {
    if (wire_kinds != nullptr) {
      scratch.kinds[item] =
        static_cast<service::storage_owner::MutationKind>(wire_kinds[item]);
    }
  }

  InsertBreakdownCounters breakdown{};
  const auto process_started = std::chrono::steady_clock::now();
  breakdown.storage_owner_queue_wait_ns = static_cast<u64>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      process_started - task.received_at).count());

  vec<u8> completion_emitted(request->item_count, 0);
  const auto emit_completion = [&](size_t item) {
    lib_assert(item < request->item_count,
               "storage-owner completion callback crossed its request");
    if (completion_emitted[item] != 0) return;
    completion_emitted[item] = 1;
    const auto& result = scratch.results[item];
    post_storage_owner_token_completion(
      task.client_id,
      task.completion_slots[item],
      service::storage_owner::MutationCompletionV2{
        .owner_storage = storage_id_,
        .source_client = request->source_client,
        .operation_id = operation_ids[item],
        .new_rptr_raw = result.new_rptr_raw,
        .old_rptr_raw = result.old_rptr_raw,
        .maintenance_sequence = result.maintenance_sequence,
        .generation = result.generation,
        .status = scratch.statuses[item],
      });
  };

  const bool synchronous_exact =
    config.synchronous_exact_updates_enabled();
  if (synchronous_exact) {
    for (u32 item = 0; item < request->item_count; ++item) {
      lib_assert(stage1_homes[item] == storage_id_,
                 "coupled update was routed to a non-authority home; "
                 "compute/memory-node completion modes do not match");
    }
  }
  const bool ok = synchronous_exact
    ? execute_storage_owner_batch_items_exact(
        ids,
        scratch.kinds.data(),
        vectors,
        operation_ids,
        request->source_client,
        request->item_count,
        breakdown,
        config,
        &scratch.invalidated_neighbors,
        &scratch.statuses,
        &scratch.results,
        emit_completion)
    : execute_storage_owner_batch_items(
        ids,
        scratch.kinds.data(),
        vectors,
        stage1_homes,
        operation_ids,
        request->source_client,
        request->item_count,
        breakdown,
        config,
        &scratch.invalidated_neighbors,
        &scratch.statuses,
        &scratch.results,
        emit_completion);

  if (synchronous_exact) {
    const u64 remote_read_ns =
      breakdown.storage_owner_search_neighbor_read_ns +
      breakdown.storage_owner_search_snapshot_read_ns +
      breakdown.storage_owner_prune_snapshot_read_ns;
    exact_insert_items_.fetch_add(request->item_count,
                                  std::memory_order_relaxed);
    exact_insert_total_ns_.fetch_add(breakdown.total(),
                                     std::memory_order_relaxed);
    exact_insert_remote_read_ns_.fetch_add(remote_read_ns,
                                           std::memory_order_relaxed);
    exact_insert_remote_reverse_ns_.fetch_add(
      breakdown.storage_owner_remote_reverse_ns,
      std::memory_order_relaxed);
    exact_insert_search_ns_.fetch_add(breakdown.storage_owner_search_ns,
                                      std::memory_order_relaxed);
    exact_insert_prune_ns_.fetch_add(breakdown.storage_owner_prune_ns,
                                     std::memory_order_relaxed);
    exact_insert_allocate_write_ns_.fetch_add(
      breakdown.storage_owner_allocate_node_ns +
        breakdown.storage_owner_write_node_ns,
      std::memory_order_relaxed);
    exact_insert_local_reverse_ns_.fetch_add(
      breakdown.storage_owner_local_reverse_ns,
      std::memory_order_relaxed);

    // The completion callback above has already emitted client-visible ACKs.
    // Serializing this cold publication cannot inflate the measured path.
    std::lock_guard<std::mutex> telemetry_lock(exact_insert_telemetry_mutex_);
    auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
      index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
    gpu_search::maintenance_telemetry::publish(
      reinterpret_cast<byte_t*>(control),
      gpu_search::maintenance_telemetry::Snapshot{
        .shard_id = storage_id_,
        .published_steady_ns = static_cast<u64>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()),
        .exact_insert_items = exact_insert_items_.load(
          std::memory_order_relaxed),
        .exact_insert_total_ns = exact_insert_total_ns_.load(
          std::memory_order_relaxed),
        .exact_insert_remote_read_ns = exact_insert_remote_read_ns_.load(
          std::memory_order_relaxed),
        .exact_insert_remote_reverse_ns =
          exact_insert_remote_reverse_ns_.load(std::memory_order_relaxed),
        .exact_insert_search_ns = exact_insert_search_ns_.load(
          std::memory_order_relaxed),
        .exact_insert_prune_ns = exact_insert_prune_ns_.load(
          std::memory_order_relaxed),
        .exact_insert_allocate_write_ns =
          exact_insert_allocate_write_ns_.load(std::memory_order_relaxed),
        .exact_insert_local_reverse_ns =
          exact_insert_local_reverse_ns_.load(std::memory_order_relaxed),
      });
  }

  // Non-committing terminal paths are emitted here. Successful fresh inserts
  // normally complete earlier, directly from authority commit_plan().
  for (size_t item = 0; item < request->item_count; ++item) {
    if (!ok && scratch.statuses[item] ==
          static_cast<u32>(service::storage_owner::MutationStatus::ok)) {
      scratch.statuses[item] = static_cast<u32>(
        service::storage_owner::MutationStatus::failed);
    }
    emit_completion(item);
  }
}

void MemoryNode::post_storage_owner_response(StorageOwnerResponseReady response) {
  if (response.queued_at == std::chrono::steady_clock::time_point{}) {
    response.queued_at = std::chrono::steady_clock::now();
  }
  lib_assert(response.client_id < num_clients_ &&
               response.slot_id < insert_runtime_.request_slot_count &&
               response.client_id < storage_client_send_mutexes_.size(),
             "storage-owner response references an invalid RPC slot");

  const Configuration& config = *storage_worker_config_;
  byte_t* response_buffer = insert_runtime_.buffer.get_full_buffer() +
    insert_response_slot_offset(config, response.client_id, response.slot_id);
  const auto* response_header =
    reinterpret_cast<const service::storage_owner::InsertBatchResponseHeader*>(
      response_buffer);

  std::lock_guard<std::mutex> send_lock(
    *storage_client_send_mutexes_[response.client_id]);
  if (response_header->item_count > 0 &&
      response_header->item_count <= config.storage_owner_batch_max) {
    auto* breakdown = service::storage_owner::response_breakdown(
      response_buffer, response_header->item_count);
    breakdown->storage_owner_response_send_ns += static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - response.queued_at).count());
  }
  cm_.client_qps[response.client_id]->post_send_with_id(
    *insert_runtime_.region,
    response.byte_len,
    IBV_WR_SEND,
    encode_64bit(response.client_id, response.slot_id),
    true,
    nullptr,
    0,
    insert_response_slot_offset(config, response.client_id, response.slot_id));
}

void MemoryNode::post_storage_owner_token_completion(
    u32 client_id,
    u32 slot_id,
    const service::storage_owner::MutationCompletionV2& completion) {
  lib_assert(client_id < num_clients_ &&
               client_id < storage_client_completion_free_slots_.size(),
             "storage-owner token completion references an invalid client");
  lib_assert(slot_id < insert_runtime_.completion_slot_count,
             "storage-owner token completion references an invalid credit");
  const size_t offset = insert_completion_slot_offset(client_id, slot_id);
  auto* output = reinterpret_cast<
    service::storage_owner::MutationCompletionV2*>(
      insert_runtime_.buffer.get_full_buffer() + offset);
  *output = completion;

  std::lock_guard<std::mutex> send_lock(
    *storage_client_send_mutexes_[client_id]);
  cm_.client_qps[client_id]->post_send_with_id(
    *insert_runtime_.region,
    sizeof(service::storage_owner::MutationCompletionV2),
    IBV_WR_SEND,
    service::storage_owner_client::storage_owner_completion_wr_id(
      client_id, slot_id),
    true,
    nullptr,
    0,
    offset);
}
