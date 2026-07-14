#include "memory_node/storage_owner_runtime/detail.hh"

using namespace memory_node_storage_owner_runtime_detail;

void MemoryNode::process_storage_owner_insert_tasks(const vec<StorageOwnerInsertTask>& tasks) {
  if (tasks.empty()) {
    return;
  }

  const Configuration& config = *storage_worker_config_;
  vec<node_t> batch_ids;
  vec<service::storage_owner::MutationKind> batch_kinds;
  vec<element_t> batch_vectors;
  vec<u64> batch_anchor_hints;
  const u32 expected_anchor_hint_count = storage_owner_local_stitch_mode(config)
                                           ? config.storage_owner_anchor_hints : 0;
  vec<u32> item_counts;
  vec<u32> response_magics;
  batch_ids.reserve(config.storage_owner_batch_max);
  batch_kinds.reserve(config.storage_owner_batch_max);
  batch_vectors.reserve(static_cast<size_t>(config.storage_owner_batch_max) * config.dim);
  item_counts.reserve(tasks.size());
  response_magics.reserve(tasks.size());

  for (const auto& task : tasks) {
    lib_assert(task.client_id < num_clients_ &&
                 task.slot_id < insert_runtime_.request_slot_count,
               "storage-owner task references an invalid request slot");
    const byte_t* payload = insert_runtime_.buffer.get_full_buffer() +
      insert_request_slot_offset(task.client_id, task.slot_id);
    const auto* request = reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(payload);
    const bool mutation = request->magic == service::storage_owner::kMutationMagic;
    const size_t expected_bytes = mutation
      ? service::storage_owner::mutation_batch_request_bytes(
          request->item_count, config.dim, request->anchor_hint_count)
      : service::storage_owner::insert_batch_request_bytes(
          request->item_count, config.dim, request->anchor_hint_count);
    lib_assert(request->item_count == task.item_count &&
                 request->batch_id == task.batch_id &&
                 task.byte_len >= expected_bytes,
               "storage-owner request slot changed before task execution");
    const node_t* ids = mutation
      ? service::storage_owner::mutation_request_ids(payload)
      : service::storage_owner::request_ids(payload);
    const byte_t* vectors = mutation
      ? service::storage_owner::mutation_request_vectors(payload, request->item_count)
      : service::storage_owner::request_vectors(payload, request->item_count);
    const u32* kinds = mutation ? service::storage_owner::mutation_request_kinds(payload)
                                : nullptr;
    const u64* hints = mutation
      ? service::storage_owner::mutation_request_anchor_hints(payload, request->item_count)
      : service::storage_owner::request_anchor_hints(payload, request->item_count);
    item_counts.push_back(request->item_count);
    response_magics.push_back(request->magic);
    batch_ids.insert(batch_ids.end(), ids, ids + request->item_count);
    for (u32 i = 0; i < request->item_count; ++i) {
      batch_kinds.push_back(kinds == nullptr
        ? service::storage_owner::MutationKind::insert
        : static_cast<service::storage_owner::MutationKind>(kinds[i]));
    }
    const size_t kind_base = batch_kinds.size() - request->item_count;
    for (u32 i = 0; i < request->item_count; ++i) {
      for (u32 hint = 0; hint < request->anchor_hint_count; ++hint) {
        batch_anchor_hints.push_back(hints[static_cast<size_t>(i) * request->anchor_hint_count + hint]);
      }
    }
    const size_t old_size = batch_vectors.size();
    batch_vectors.resize(old_size + static_cast<size_t>(request->item_count) * config.dim);
    for (u32 i = 0; i < request->item_count; ++i) {
      if (batch_kinds[kind_base + i] == service::storage_owner::MutationKind::erase) continue;
      decode_storage_vector_to_float(vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes(),
                                     VamanaNode::vector_dtype(),
                                     config.dim,
                                     batch_vectors.data() + old_size + static_cast<size_t>(i) * config.dim);
    }
  }

  InsertBreakdownCounters breakdown{};
  const auto process_started = std::chrono::steady_clock::now();
  for (const auto& task : tasks) {
    breakdown.storage_owner_queue_wait_ns += static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(process_started - task.received_at).count());
  }

  vec<vec<u64>> invalidated_neighbors;
  vec<u32> statuses(batch_ids.size(), static_cast<u32>(service::storage_owner::MutationStatus::failed));
  vec<service::storage_owner::MutationResult> mutation_results(batch_ids.size());
  const bool use_sync_local_stitch =
    storage_owner_batch_prefers_sync_local_stitch(config,
                                                  batch_kinds,
                                                  batch_anchor_hints,
                                                  expected_anchor_hint_count,
                                                  batch_ids.size());
  const bool ok = current_storage_owner_thread_ != nullptr && !use_sync_local_stitch
                    ? execute_storage_owner_batch_items_async(batch_ids.data(),
                                                               batch_kinds.data(),
                                                               batch_vectors.data(),
                                                               batch_anchor_hints.empty() ? nullptr : batch_anchor_hints.data(),
                                                               expected_anchor_hint_count,
                                                               batch_ids.size(),
                                                               *current_storage_owner_thread_,
                                                               breakdown,
                                                               config,
                                                               &invalidated_neighbors,
                                                               &statuses,
                                                               &mutation_results)
                    : execute_storage_owner_batch_items(batch_ids.data(),
                                                        batch_kinds.data(),
                                                        batch_vectors.data(),
                                                        batch_anchor_hints.empty() ? nullptr : batch_anchor_hints.data(),
                                                        expected_anchor_hint_count,
                                                        batch_ids.size(),
                                                        breakdown,
                                                        config,
                                                        &invalidated_neighbors,
                                                        &statuses,
                                                        &mutation_results);
  size_t status_base = 0;
  for (size_t task_idx = 0; task_idx < tasks.size(); ++task_idx) {
    const auto& task = tasks[task_idx];
    const byte_t* payload = insert_runtime_.buffer.get_full_buffer() +
      insert_request_slot_offset(task.client_id, task.slot_id);
    const auto* request = reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(payload);
    const u32 item_count = item_counts[task_idx];
    const size_t response_size = service::storage_owner::insert_batch_response_bytes(item_count);
    lib_assert(response_size <= std::numeric_limits<u32>::max(),
               "storage_owner async response is too large for verbs SGEs");
    byte_t* response_buffer = insert_runtime_.buffer.get_full_buffer() +
      insert_response_slot_offset(config, task.client_id, task.slot_id);
    auto* response = reinterpret_cast<service::storage_owner::InsertBatchResponseHeader*>(response_buffer);
    response->magic = response_magics[task_idx];
    response->owner_storage = storage_id_;
    response->item_count = item_count;
    response->batch_id = request->batch_id;
    u32* response_statuses = service::storage_owner::response_statuses(response_buffer);
    for (u32 i = 0; i < item_count; ++i) {
      response_statuses[i] = ok ? statuses[status_base + i]
                                : static_cast<u32>(service::storage_owner::MutationStatus::failed);
    }
    auto* response_results = service::storage_owner::response_mutation_results(
      response_buffer, item_count);
    for (u32 i = 0; i < item_count; ++i) {
      response_results[i] = ok ? mutation_results[status_base + i]
                               : service::storage_owner::MutationResult{};
    }
    *service::storage_owner::response_breakdown(response_buffer, item_count) =
      scale_breakdown(breakdown, item_count, static_cast<u32>(std::max<size_t>(1, batch_ids.size())));
    const u32 invalidation_capacity = service::storage_owner::response_invalidation_capacity(item_count);
    vec<u64> task_invalidations;
    task_invalidations.reserve(invalidation_capacity);
    for (u32 item = 0; item < item_count; ++item) {
      const auto& item_invalidations = invalidated_neighbors[status_base + item];
      task_invalidations.insert(task_invalidations.end(),
                                item_invalidations.begin(), item_invalidations.end());
    }
    std::sort(task_invalidations.begin(), task_invalidations.end());
    task_invalidations.erase(
      std::unique(task_invalidations.begin(), task_invalidations.end()),
      task_invalidations.end());
    lib_assert(task_invalidations.size() <= invalidation_capacity,
               "storage_owner async invalidation response exceeds its capacity");
    const u32 invalidation_count = static_cast<u32>(task_invalidations.size());
    *service::storage_owner::response_invalidation_count(response_buffer, item_count) = invalidation_count;
    u64* invalidated = service::storage_owner::response_invalidated_raws(response_buffer, item_count);
    for (u32 i = 0; i < invalidation_count; ++i) {
      invalidated[i] = task_invalidations[i];
    }
    status_base += item_count;
    enqueue_storage_owner_response(
      {task.client_id, task.slot_id, static_cast<u32>(response_size)});
  }
}

void MemoryNode::enqueue_storage_owner_response(StorageOwnerResponseReady response) {
  std::lock_guard<std::mutex> lock(storage_responses_mutex_);
  storage_responses_ready_.push_back(response);
}
