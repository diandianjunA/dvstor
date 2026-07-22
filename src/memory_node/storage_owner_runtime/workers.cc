#include "memory_node/storage_owner_runtime/detail.hh"

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
  const byte_t* payload = insert_runtime_.buffer.get_full_buffer() +
    insert_request_slot_offset(task.client_id, task.slot_id);
  const auto* request =
    reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(payload);
  const bool mutation = request->magic == service::storage_owner::kMutationMagic;
  const size_t expected_bytes = mutation
    ? service::storage_owner::mutation_batch_request_bytes(request->item_count)
    : service::storage_owner::insert_batch_request_bytes(request->item_count);
  lib_assert(request->item_count == task.item_count &&
               request->batch_id == task.batch_id &&
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

  const bool ok = execute_storage_owner_batch_items(
        ids,
        scratch.kinds.data(),
        vectors,
        stage1_homes,
        request->source_client,
        request->batch_id,
        request->item_count,
        breakdown,
        config,
        &scratch.invalidated_neighbors,
        &scratch.statuses,
        &scratch.results);

  const auto response_build_started = std::chrono::steady_clock::now();
  const u32 item_count = request->item_count;
  const size_t response_size = service::storage_owner::insert_batch_response_bytes(item_count);
  lib_assert(response_size <= std::numeric_limits<u32>::max(),
             "storage_owner response is too large for verbs SGEs");
  byte_t* response_buffer = insert_runtime_.buffer.get_full_buffer() +
    insert_response_slot_offset(config, task.client_id, task.slot_id);
  auto* response =
    reinterpret_cast<service::storage_owner::InsertBatchResponseHeader*>(response_buffer);
  response->magic = request->magic;
  response->owner_storage = storage_id_;
  response->item_count = item_count;
  response->batch_id = request->batch_id;
  u32* response_statuses = service::storage_owner::response_statuses(response_buffer);
  auto* response_results =
    service::storage_owner::response_mutation_results(response_buffer, item_count);
  for (u32 item = 0; item < item_count; ++item) {
    response_statuses[item] = ok
      ? scratch.statuses[item]
      : static_cast<u32>(service::storage_owner::MutationStatus::failed);
    response_results[item] = ok
      ? scratch.results[item]
      : service::storage_owner::MutationResult{};
  }
  const u32 invalidation_capacity =
    service::storage_owner::response_invalidation_capacity(item_count);
  scratch.response_invalidations.reserve(invalidation_capacity);
  for (const auto& item_invalidations : scratch.invalidated_neighbors) {
    scratch.response_invalidations.insert(
      scratch.response_invalidations.end(),
      item_invalidations.begin(),
      item_invalidations.end());
  }
  std::sort(scratch.response_invalidations.begin(),
            scratch.response_invalidations.end());
  scratch.response_invalidations.erase(
    std::unique(scratch.response_invalidations.begin(),
                scratch.response_invalidations.end()),
    scratch.response_invalidations.end());
  lib_assert(scratch.response_invalidations.size() <= invalidation_capacity,
             "storage_owner invalidation response exceeds its capacity");
  const u32 invalidation_count =
    static_cast<u32>(scratch.response_invalidations.size());
  *service::storage_owner::response_invalidation_count(response_buffer, item_count) =
    invalidation_count;
  u64* invalidated =
    service::storage_owner::response_invalidated_raws(response_buffer, item_count);
  std::copy(scratch.response_invalidations.begin(),
            scratch.response_invalidations.end(),
            invalidated);
  breakdown.storage_owner_response_build_ns += elapsed_ns_since(response_build_started);
  *service::storage_owner::response_breakdown(response_buffer, item_count) = breakdown;
  post_storage_owner_response(
    {task.client_id,
     task.slot_id,
     static_cast<u32>(response_size),
     std::chrono::steady_clock::now()});
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
