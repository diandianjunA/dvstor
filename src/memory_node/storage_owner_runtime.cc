#include "memory_node/memory_node.hh"

#include <algorithm>
#include <iostream>
#include <limits>

namespace {

bool storage_owner_anchor_mode(const configuration::IndexConfiguration& config) {
  return config.storage_owner_update_mode == "local_stitch";
}

bool storage_owner_local_stitch_mode(const configuration::IndexConfiguration& config) {
  return config.storage_owner_update_mode == "local_stitch";
}

}  // namespace

void MemoryNode::setup_insert_runtime(const Configuration& config) {
  lib_assert(static_cast<u64>(config.storage_owner_batch_max) * VamanaNode::R <=
               std::numeric_limits<u32>::max(),
             "storage_owner invalidation capacity is too large for the wire format");
  const size_t insert_request_bytes = align_up(std::max(
    service::storage_owner::insert_batch_request_bytes(
      config.storage_owner_batch_max, VamanaNode::DIM,
      storage_owner_anchor_mode(config) ? config.storage_owner_anchor_hints : 0),
    service::storage_owner::mutation_batch_request_bytes(
      config.storage_owner_batch_max, VamanaNode::DIM,
      storage_owner_anchor_mode(config) ? config.storage_owner_anchor_hints : 0)));
  insert_runtime_.request_bytes = insert_request_bytes;
  insert_runtime_.request_slot_count = std::max<u32>(1, config.storage_owner_rpc_depth);
  const size_t insert_response_bytes =
    align_up(service::storage_owner::insert_batch_response_bytes(config.storage_owner_batch_max));
  lib_assert(insert_request_bytes <= std::numeric_limits<u32>::max() &&
             insert_response_bytes <= std::numeric_limits<u32>::max(),
             "storage_owner RPC message is too large for verbs SGEs; reduce batch size or vector dimension");
  const size_t slot_count =
    static_cast<size_t>(num_clients_) * insert_runtime_.request_slot_count;
  lib_assert(slot_count <= static_cast<size_t>(config.max_recv_queue_wr),
             "storage_owner RPC receive slots exceed memory-node receive CQ capacity");
  insert_runtime_.response_offset = insert_runtime_.request_bytes * slot_count;
  insert_runtime_.buffer.allocate(insert_runtime_.response_offset + insert_response_bytes * slot_count);
  insert_runtime_.buffer.touch_memory();
  insert_runtime_.region = std::make_unique<LocalMemoryRegion>(
    context_, insert_runtime_.buffer.get_full_buffer(), insert_runtime_.buffer.buffer_size);
}

void MemoryNode::start_storage_owner_insert_workers(const Configuration& config) {
  if (!use_storage_owner_insert_) {
    return;
  }
  print_status("storage-owner peer RDMA read credits per peer: " +
               std::to_string(peer_rdma_read_credit_limit()) +
               " per QP: " + std::to_string(peer_rdma_read_credit_limit_per_qp()) +
               " (requested=" + std::to_string(storage_owner_peer_rdma_tokens_) + ")");
  print_status("storage-owner online insert tuning: construction_beam=" +
               std::to_string(config.storage_owner_construction_beam_width == 0
                                ? config.beam_width_construction
                                : std::min(config.beam_width_construction,
                                           config.storage_owner_construction_beam_width)) +
               " snapshot_batch=" + std::to_string(config.storage_owner_search_snapshot_batch) +
               " prune_max_candidates=" + std::to_string(config.storage_owner_prune_max_candidates) +
               " update_mode=" + config.storage_owner_update_mode);
  const u32 worker_count = std::max<u32>(1, std::min<u32>(8, std::max<u32>(1, num_compute_threads_ / 2)));
  const u32 coroutines_per_worker = std::max<u32>(1, config.insert_coroutines == 0 ? config.num_coroutines
                                                                                    : config.insert_coroutines);
  const size_t snapshot_bytes = memory_node_detail::storage_owner_snapshot_bytes();
  const size_t snapshot_stride = memory_node_detail::storage_owner_snapshot_stride();
  const size_t neighbor_stride = align_up(VamanaNode::neighbor_read_size());
  const size_t snapshot_batch =
    std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  // Keep one general-purpose slot beyond the batch area. This prevents a
  // neighbor/node fallback in the same coroutine from aliasing batched reads.
  const size_t coroutine_scratch_stride =
    align_up(snapshot_stride * snapshot_batch +
             std::max(VamanaNode::total_size(), neighbor_stride));
  const size_t scratch_bytes =
    std::max<size_t>(64ull * 1024ull * 1024ull,
                     coroutine_scratch_stride * std::max<u32>(1, coroutines_per_worker));
  print_status("storage-owner coroutine scratch: snapshot_bytes=" +
               std::to_string(snapshot_bytes) +
               " snapshot_stride=" + std::to_string(snapshot_stride) +
               " batch=" + std::to_string(snapshot_batch) +
               " per_coroutine=" + std::to_string(coroutine_scratch_stride));
  storage_owner_threads_.reserve(worker_count);
  for (u32 i = 0; i < worker_count; ++i) {
    auto thread = std::make_unique<StorageOwnerThread>(i, coroutines_per_worker, config.max_send_queue_wr);
    if (peer_context_) {
      thread->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
    }
    storage_owner_threads_.push_back(std::move(thread));
  }
  storage_owner_async_candidates_.clear();
  storage_owner_async_candidates_.resize(worker_count);
  for (auto& worker_candidates : storage_owner_async_candidates_) {
    worker_candidates.resize(coroutines_per_worker);
  }
  for (u32 i = 0; i < worker_count; ++i) {
    storage_insert_workers_.emplace_back([this, i]() { storage_owner_insert_worker_loop(i); });
  }
}

void MemoryNode::storage_owner_insert_worker_loop(u32 worker_id) {
  current_storage_owner_thread_ = storage_owner_threads_[worker_id].get();
  const Configuration& config = *storage_worker_config_;
  for (;;) {
    vec<StorageOwnerInsertTask> tasks;
    u32 total_items = 0;
    {
      std::unique_lock<std::mutex> lock(storage_insert_tasks_mutex_);
      storage_insert_tasks_cv_.wait(lock, [&]() {
        return storage_insert_shutdown_.load(std::memory_order_acquire) || !storage_insert_tasks_.empty();
      });
      if (storage_insert_shutdown_.load(std::memory_order_acquire) && storage_insert_tasks_.empty()) {
        current_storage_owner_thread_ = nullptr;
        return;
      }

      while (!storage_insert_tasks_.empty()) {
        const u32 next_items = storage_insert_tasks_.front().item_count;
        if (!tasks.empty() && total_items + next_items > std::max<u32>(config.storage_owner_batch_max, 64)) {
          break;
        }
        total_items += next_items;
        tasks.push_back(std::move(storage_insert_tasks_.front()));
        storage_insert_tasks_.pop_front();
      }
    }

    mark_storage_owner_foreground_activity();
    storage_owner_insert_active_workers_.fetch_add(1, std::memory_order_acq_rel);
    process_storage_owner_insert_tasks(tasks);
    storage_owner_insert_active_workers_.fetch_sub(1, std::memory_order_acq_rel);
    mark_storage_owner_foreground_activity();
  }
}

void MemoryNode::process_storage_owner_insert_tasks(const vec<StorageOwnerInsertTask>& tasks) {
  if (tasks.empty()) {
    return;
  }

  const Configuration& config = *storage_worker_config_;
  vec<node_t> batch_ids;
  vec<service::storage_owner::MutationKind> batch_kinds;
  vec<element_t> batch_vectors;
  vec<u64> batch_anchor_hints;
  const u32 expected_anchor_hint_count = storage_owner_anchor_mode(config)
                                           ? config.storage_owner_anchor_hints : 0;
  vec<u32> item_counts;
  vec<u32> response_magics;
  batch_ids.reserve(std::max<u32>(config.storage_owner_batch_max, 64));
  batch_kinds.reserve(std::max<u32>(config.storage_owner_batch_max, 64));
  batch_vectors.reserve(static_cast<size_t>(std::max<u32>(config.storage_owner_batch_max, 64)) * config.dim);
  item_counts.reserve(tasks.size());
  response_magics.reserve(tasks.size());

  for (const auto& task : tasks) {
    const auto* request = reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(task.payload.data());
    const bool mutation = request->magic == service::storage_owner::kMutationMagic;
    const node_t* ids = mutation
      ? service::storage_owner::mutation_request_ids(task.payload.data())
      : service::storage_owner::request_ids(task.payload.data());
    const byte_t* vectors = mutation
      ? service::storage_owner::mutation_request_vectors(task.payload.data(), request->item_count)
      : service::storage_owner::request_vectors(task.payload.data(), request->item_count);
    const u32* kinds = mutation ? service::storage_owner::mutation_request_kinds(task.payload.data())
                                : nullptr;
    const u64* hints = mutation
      ? service::storage_owner::mutation_request_anchor_hints(task.payload.data(), request->item_count)
      : service::storage_owner::request_anchor_hints(task.payload.data(), request->item_count);
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

  vec<u64> invalidated_neighbors;
  vec<u32> statuses(batch_ids.size(), static_cast<u32>(service::storage_owner::MutationStatus::failed));
  vec<service::storage_owner::MutationResult> mutation_results(batch_ids.size());
  const bool ok = current_storage_owner_thread_ != nullptr
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
    const auto* request = reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(task.payload.data());
    const u32 item_count = item_counts[task_idx];
    const size_t response_size = service::storage_owner::insert_batch_response_bytes(item_count);
    lib_assert(response_size <= std::numeric_limits<u32>::max(),
               "storage_owner async response is too large for verbs SGEs");
    vec<byte_t> response_buffer(response_size);
    auto* response = reinterpret_cast<service::storage_owner::InsertBatchResponseHeader*>(response_buffer.data());
    response->magic = response_magics[task_idx];
    response->owner_storage = storage_id_;
    response->item_count = item_count;
    response->batch_id = request->batch_id;
    u32* response_statuses = service::storage_owner::response_statuses(response_buffer.data());
    for (u32 i = 0; i < item_count; ++i) {
      response_statuses[i] = ok ? statuses[status_base + i]
                                : static_cast<u32>(service::storage_owner::MutationStatus::failed);
    }
    auto* response_results = service::storage_owner::response_mutation_results(
      response_buffer.data(), item_count);
    for (u32 i = 0; i < item_count; ++i) {
      response_results[i] = ok ? mutation_results[status_base + i]
                               : service::storage_owner::MutationResult{};
    }
    status_base += item_count;
    *service::storage_owner::response_breakdown(response_buffer.data(), item_count) =
      scale_breakdown(breakdown, item_count, static_cast<u32>(std::max<size_t>(1, batch_ids.size())));
    const u32 invalidation_capacity = service::storage_owner::response_invalidation_capacity(item_count);
    const u32 invalidation_count = static_cast<u32>(std::min<size_t>(invalidated_neighbors.size(), invalidation_capacity));
    *service::storage_owner::response_invalidation_count(response_buffer.data(), item_count) = invalidation_count;
    u64* invalidated = service::storage_owner::response_invalidated_raws(response_buffer.data(), item_count);
    for (u32 i = 0; i < invalidation_count; ++i) {
      invalidated[i] = invalidated_neighbors[i];
    }

    LocalMemoryRegion response_region{context_, response_buffer.data(), response_buffer.size()};
    {
      std::lock_guard<std::mutex> lock(storage_send_mutex_);
      cm_.client_qps[task.client_id]->post_send(
        response_region, static_cast<u32>(response_size), IBV_WR_SEND, true, nullptr, 0, 0);
      context_.poll_send_cq_until_completion();
    }
  }
}

bool MemoryNode::execute_storage_owner_batch_items_async(const node_t* ids,
                                             const service::storage_owner::MutationKind* kinds,
                                             const element_t* vectors,
                                             const u64* anchor_hints,
                                             u32 anchor_hint_count,
                                             size_t item_count,
                                             StorageOwnerThread& thread,
                                             InsertBreakdownCounters& breakdown,
                                             const Configuration& config,
                                             vec<u64>* invalidated_neighbors,
                                             vec<u32>* statuses,
                                             vec<service::storage_owner::MutationResult>* results) {
  if (item_count == 0) {
    return true;
  }

  vec<StorageOwnerInsertJob> jobs;
  jobs.reserve(item_count);
  for (size_t idx = 0; idx < item_count; ++idx) {
    StorageOwnerInsertJob job;
    job.id = ids[idx];
    job.kind = kinds == nullptr ? service::storage_owner::MutationKind::insert : kinds[idx];
    job.vector_data.resize(static_cast<size_t>(VamanaNode::DIM) * sizeof(element_t));
    std::memcpy(job.vector_data.data(),
                vectors + idx * VamanaNode::DIM,
                static_cast<size_t>(VamanaNode::DIM) * sizeof(element_t));
    if (anchor_hints != nullptr) {
      job.anchor_hints.reserve(anchor_hint_count);
      for (u32 hint = 0; hint < anchor_hint_count; ++hint) {
        const RemotePtr ptr{anchor_hints[idx * anchor_hint_count + hint]};
        if (!ptr.is_null()) {
          job.anchor_hints.push_back(ptr);
        }
      }
    }
    jobs.push_back(std::move(job));
  }

  dense_hashmap_t<u64, vec<RemotePtr>> local_updates;
  dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_updates;

  const u32 coroutine_count = static_cast<u32>(std::max<size_t>(1, thread.post_balances.size()));
  lib_assert(thread.id < storage_owner_async_candidates_.size(),
             "storage_owner async candidate slots not initialized for worker");
  lib_assert(storage_owner_async_candidates_[thread.id].size() >= coroutine_count,
             "storage_owner async candidate slots not initialized for coroutines");

  thread.coroutines.clear();
  thread.coroutines.reserve(coroutine_count);
  for (u32 i = 0; i < coroutine_count; ++i) {
    thread.coroutines.emplace_back(std::make_unique<StorageOwnerInsertCoroutine>(dummy_storage_owner_insert_coroutine()));
  }

  size_t next_job = 0;
  for (;;) {
    bool all_done = true;
    poll_peer_send_cq();

    for (u32 coroutine_id = 0; coroutine_id < coroutine_count; ++coroutine_id) {
      auto& coroutine = *thread.coroutines[coroutine_id];
      if (coroutine.handle.done()) {
        if (next_job < jobs.size()) {
          coroutine.handle.destroy();
          thread.set_current_coroutine(coroutine_id);
          coroutine.handle = execute_storage_owner_insert_job_async(
            thread, jobs[next_job++], local_updates, remote_updates, breakdown, config).handle;
          all_done = false;
        }
      } else if (thread.is_ready(coroutine_id)) {
        thread.set_current_coroutine(coroutine_id);
        coroutine.handle.resume();
        all_done = false;
      } else {
        all_done = false;
      }
    }

    if (all_done) {
      break;
    }
  }

  for (const auto& coroutine : thread.coroutines) {
    lib_assert(coroutine->handle.done(), "storage-owner insert coroutine not done yet");
    coroutine->handle.destroy();
  }
  thread.coroutines.clear();

  bool ok = true;
  if (statuses != nullptr) {
    statuses->assign(item_count, static_cast<u32>(service::storage_owner::MutationStatus::failed));
  }
  if (results != nullptr) {
    results->assign(item_count, {});
  }
  for (size_t i = 0; i < jobs.size(); ++i) {
    const auto& job = jobs[i];
    if (statuses != nullptr) {
      (*statuses)[i] = static_cast<u32>(job.status);
    }
    if (results != nullptr) {
      (*results)[i] = service::storage_owner::MutationResult{
        job.new_ptr.raw_address,
        job.old_ptr.raw_address,
        job.generation,
        0};
    }
  }
  auto t_local_reverse = std::chrono::steady_clock::now();
  for (auto& [target_raw, candidates] : local_updates) {
    ok &= apply_local_reverse_update(RemotePtr{target_raw}, candidates, config);
  }
  breakdown.storage_owner_local_reverse_ns += elapsed_ns_since(t_local_reverse);
  auto t_remote_reverse = std::chrono::steady_clock::now();
  for (auto& [target_shard, ops] : remote_updates) {
    ok &= send_reverse_update_batch(target_shard, ops, config);
  }
  breakdown.storage_owner_remote_reverse_ns += elapsed_ns_since(t_remote_reverse);
  if (invalidated_neighbors != nullptr) {
    invalidated_neighbors->reserve(invalidated_neighbors->size() + local_updates.size());
    for (const auto& [target_raw, _] : local_updates) {
      invalidated_neighbors->push_back(target_raw);
    }
    for (const auto& [_, ops] : remote_updates) {
      for (const auto& op : ops) {
        invalidated_neighbors->push_back(op.target_raw);
      }
    }
  }
  return ok;
}

StorageOwnerInsertCoroutine MemoryNode::dummy_storage_owner_insert_coroutine() {
  co_return;
}

size_t MemoryNode::insert_request_slot_offset(u32 client_id, u32 slot_id) const {
  const size_t slot_index =
    static_cast<size_t>(client_id) * insert_runtime_.request_slot_count + slot_id;
  return slot_index * insert_runtime_.request_bytes;
}

size_t MemoryNode::insert_response_slot_offset(const Configuration& config, u32 client_id, u32 slot_id) const {
  const size_t slot_index =
    static_cast<size_t>(client_id) * insert_runtime_.request_slot_count + slot_id;
  return insert_runtime_.response_offset + slot_index * response_slot_bytes(config);
}

void MemoryNode::service_storage_runtime(const Configuration& config) {
  print_status("storage-owner insert runtime enabled on shard " + std::to_string(storage_id_));
  vec<ibv_wc> recv_wcs(std::max<i32>(1, config.max_recv_queue_wr));

  for (u32 client_id = 0; client_id < num_clients_; ++client_id) {
    for (u32 slot_id = 0; slot_id < insert_runtime_.request_slot_count; ++slot_id) {
      cm_.client_qps[client_id]->post_receive(
        *insert_runtime_.region,
        static_cast<u32>(insert_runtime_.request_bytes),
        encode_64bit(client_id, slot_id),
        insert_request_slot_offset(client_id, slot_id));
    }
  }

  for (;;) {
    const i32 num_received = context_.poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
    if (num_received == 0) {
      std::this_thread::yield();
      continue;
    }

    for (i32 i = 0; i < num_received; ++i) {
      const auto [client_id, slot_id] = decode_64bit(recv_wcs[i].wr_id);
      if (client_id >= num_clients_ || slot_id >= insert_runtime_.request_slot_count) {
        continue;
      }
      const size_t offset = insert_request_slot_offset(client_id, slot_id);
      const byte_t* payload = insert_runtime_.buffer.get_full_buffer() + offset;
      const size_t bytes = recv_wcs[i].byte_len;

      bool handled_async = false;
      if (bytes >= sizeof(service::storage_owner::InsertBatchRequestHeader)) {
        const auto* request = reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(payload);
        const bool magic_ok = request->magic == service::storage_owner::kInsertMagic ||
                              request->magic == service::storage_owner::kMutationMagic;
        const u32 expected_anchor_hint_count = storage_owner_anchor_mode(config)
                                                 ? config.storage_owner_anchor_hints : 0;
        const size_t expected_bytes = request->magic == service::storage_owner::kMutationMagic
          ? service::storage_owner::mutation_batch_request_bytes(
              request->item_count, config.dim, request->anchor_hint_count)
          : service::storage_owner::insert_batch_request_bytes(
              request->item_count, config.dim, request->anchor_hint_count);
        if (magic_ok &&
            request->dim == config.dim &&
            request->owner_storage == storage_id_ &&
            request->item_count > 0 &&
            request->item_count <= config.storage_owner_batch_max &&
            request->vector_dtype == static_cast<u32>(VamanaNode::vector_dtype()) &&
            request->vector_bytes == VamanaNode::vector_bytes() &&
            request->anchor_hint_count == expected_anchor_hint_count &&
            bytes >= expected_bytes) {
          StorageOwnerInsertTask task;
          task.client_id = client_id;
          task.item_count = request->item_count;
          task.batch_id = request->batch_id;
          task.received_at = std::chrono::steady_clock::now();
          task.payload.assign(payload, payload + bytes);
          mark_storage_owner_foreground_activity();
          {
            std::lock_guard<std::mutex> lock(storage_insert_tasks_mutex_);
            storage_insert_tasks_.push_back(std::move(task));
          }
          storage_insert_tasks_cv_.notify_one();
          handled_async = true;
        }
      }

      cm_.client_qps[client_id]->post_receive(
        *insert_runtime_.region,
        static_cast<u32>(insert_runtime_.request_bytes),
        encode_64bit(client_id, slot_id),
        insert_request_slot_offset(client_id, slot_id));

      if (handled_async) {
        continue;
      }

      const size_t response_bytes = handle_storage_insert_request(client_id, payload, bytes, config);
      lib_assert(response_bytes > 0, "invalid storage-owner insert request");
      lib_assert(response_bytes <= response_slot_bytes(config) &&
                 response_bytes <= std::numeric_limits<u32>::max(),
                 "storage_owner response exceeds the registered response slot");

      cm_.client_qps[client_id]->post_send(
        *insert_runtime_.region,
        static_cast<u32>(response_bytes),
        IBV_WR_SEND,
        true,
        nullptr,
        0,
        insert_response_slot_offset(config, client_id, slot_id));
      context_.poll_send_cq_until_completion();
    }
  }
}

size_t MemoryNode::response_slot_bytes(const Configuration& config) const {
  return align_up(service::storage_owner::insert_batch_response_bytes(config.storage_owner_batch_max));
}

size_t MemoryNode::handle_storage_insert_request(u32 client_id, const byte_t* payload, size_t bytes, const Configuration& config) {
  if (bytes < sizeof(service::storage_owner::InsertBatchRequestHeader)) {
    return 0;
  }

  const auto* request = reinterpret_cast<const service::storage_owner::InsertBatchRequestHeader*>(payload);
  const bool mutation = request->magic == service::storage_owner::kMutationMagic;
  const bool magic_ok = request->magic == service::storage_owner::kInsertMagic || mutation;
  const size_t expected_bytes = mutation
    ? service::storage_owner::mutation_batch_request_bytes(
        request->item_count, config.dim, request->anchor_hint_count)
    : service::storage_owner::insert_batch_request_bytes(
        request->item_count, config.dim, request->anchor_hint_count);
  if (!magic_ok ||
      request->dim != config.dim ||
      request->owner_storage != storage_id_ ||
      request->item_count == 0 ||
      request->item_count > config.storage_owner_batch_max ||
      request->vector_dtype != static_cast<u32>(VamanaNode::vector_dtype()) ||
      request->vector_bytes != VamanaNode::vector_bytes() ||
      request->anchor_hint_count != (storage_owner_anchor_mode(config)
                                      ? config.storage_owner_anchor_hints : 0) ||
      bytes < expected_bytes) {
    return 0;
  }

  auto* response_ptr = reinterpret_cast<service::storage_owner::InsertBatchResponseHeader*>(
    insert_runtime_.buffer.get_full_buffer() + insert_runtime_.response_offset +
    static_cast<size_t>(client_id) *
      response_slot_bytes(config));
  response_ptr->magic = request->magic;
  response_ptr->owner_storage = storage_id_;
  response_ptr->item_count = request->item_count;
  response_ptr->batch_id = request->batch_id;
  u32* statuses = service::storage_owner::response_statuses(response_ptr);

  const node_t* ids = mutation ? service::storage_owner::mutation_request_ids(payload)
                               : service::storage_owner::request_ids(payload);
  const u32* kinds_raw = mutation ? service::storage_owner::mutation_request_kinds(payload)
                                  : nullptr;
  const byte_t* raw_vectors = mutation
    ? service::storage_owner::mutation_request_vectors(payload, request->item_count)
    : service::storage_owner::request_vectors(payload, request->item_count);
  const u64* anchor_hints = mutation
    ? service::storage_owner::mutation_request_anchor_hints(payload, request->item_count)
    : service::storage_owner::request_anchor_hints(payload, request->item_count);
  vec<service::storage_owner::MutationKind> kinds(request->item_count, service::storage_owner::MutationKind::insert);
  for (u32 i = 0; i < request->item_count && kinds_raw != nullptr; ++i) {
    kinds[i] = static_cast<service::storage_owner::MutationKind>(kinds_raw[i]);
  }
  vec<element_t> decoded_vectors(static_cast<size_t>(request->item_count) * config.dim);
  for (u32 i = 0; i < request->item_count; ++i) {
    if (kinds[i] == service::storage_owner::MutationKind::erase) continue;
    decode_storage_vector_to_float(raw_vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes(),
                                   VamanaNode::vector_dtype(),
                                   config.dim,
                                   decoded_vectors.data() + static_cast<size_t>(i) * config.dim);
  }
  InsertBreakdownCounters breakdown{};
  vec<u64> invalidated_neighbors;
  vec<u32> item_statuses(request->item_count, static_cast<u32>(service::storage_owner::MutationStatus::failed));
  vec<service::storage_owner::MutationResult> mutation_results(request->item_count);
  mark_storage_owner_foreground_activity();
  storage_owner_insert_active_workers_.fetch_add(1, std::memory_order_acq_rel);
  const bool ok = execute_storage_owner_batch_items(ids, kinds.data(), decoded_vectors.data(),
                                                    anchor_hints, request->anchor_hint_count,
                                                    request->item_count,
                                                    breakdown, config, &invalidated_neighbors,
                                                    &item_statuses, &mutation_results);
  storage_owner_insert_active_workers_.fetch_sub(1, std::memory_order_acq_rel);
  mark_storage_owner_foreground_activity();
  for (u32 i = 0; i < request->item_count; ++i) {
    statuses[i] = ok ? item_statuses[i]
                     : static_cast<u32>(service::storage_owner::MutationStatus::failed);
  }
  auto* response_results = service::storage_owner::response_mutation_results(response_ptr, request->item_count);
  for (u32 i = 0; i < request->item_count; ++i) {
    response_results[i] = ok ? mutation_results[i] : service::storage_owner::MutationResult{};
  }
  *service::storage_owner::response_breakdown(response_ptr, request->item_count) = breakdown;
  const u32 invalidation_capacity = service::storage_owner::response_invalidation_capacity(request->item_count);
  const u32 invalidation_count = static_cast<u32>(std::min<size_t>(invalidated_neighbors.size(), invalidation_capacity));
  *service::storage_owner::response_invalidation_count(response_ptr, request->item_count) = invalidation_count;
  u64* invalidated = service::storage_owner::response_invalidated_raws(response_ptr, request->item_count);
  for (u32 i = 0; i < invalidation_count; ++i) {
    invalidated[i] = invalidated_neighbors[i];
  }
  return service::storage_owner::insert_batch_response_bytes(request->item_count);
}

bool MemoryNode::execute_storage_owner_batch_items(const node_t* ids,
                                       const service::storage_owner::MutationKind* kinds,
                                       const element_t* vectors,
                                       const u64* anchor_hints,
                                       u32 anchor_hint_count,
                                       size_t item_count,
                                       InsertBreakdownCounters& breakdown,
                                       const Configuration& config,
                                       vec<u64>* invalidated_neighbors,
                                       vec<u32>* statuses,
                                       vec<service::storage_owner::MutationResult>* results) {
  if (item_count == 0) {
    return true;
  }

  RemotePtr medoid_ptr{};
  bool medoid_loaded = false;
  const bool maintenance_enabled = storage_owner_maintenance_enabled(config);
  dense_hashmap_t<u64, vec<RemotePtr>> local_updates;
  dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>> remote_updates;
  if (statuses != nullptr) {
    statuses->assign(item_count, static_cast<u32>(service::storage_owner::MutationStatus::failed));
  }
  if (results != nullptr) {
    results->assign(item_count, {});
  }

  for (size_t idx = 0; idx < item_count; ++idx) {
    const auto kind = kinds == nullptr ? service::storage_owner::MutationKind::insert : kinds[idx];
    FreshnessEntry old_entry{};
    u32 generation = 0;
    const auto status = prepare_mutation(ids[idx], kind, &old_entry, &generation);
    if (results != nullptr) {
      (*results)[idx].old_rptr_raw = old_entry.current.raw_address;
      (*results)[idx].generation = generation;
    }
    if (status != service::storage_owner::MutationStatus::ok) {
      if (statuses != nullptr) {
        (*statuses)[idx] = static_cast<u32>(status);
      }
      continue;
    }
    if (kind == service::storage_owner::MutationKind::erase) {
      const bool deleted = mark_node_deleted(old_entry.current, old_entry.generation);
      if (deleted) {
        publish_mutation(ids[idx], old_entry.current, old_entry.generation, true);
        if (maintenance_enabled) {
          (void)enqueue_deleted_node_cleanup(old_entry.current, config);
        }
      }
      if (statuses != nullptr) {
        (*statuses)[idx] = static_cast<u32>(deleted ? service::storage_owner::MutationStatus::ok
                                                    : service::storage_owner::MutationStatus::failed);
      }
      continue;
    }
    const element_t* vec_ptr = vectors + idx * VamanaNode::DIM;
    const auto components = span<const element_t>{vec_ptr, VamanaNode::DIM};
    vec<RemotePtr> item_anchor_hints;
    if (anchor_hints != nullptr) {
      item_anchor_hints.reserve(anchor_hint_count);
      for (u32 hint = 0; hint < anchor_hint_count; ++hint) {
        const RemotePtr ptr{anchor_hints[idx * anchor_hint_count + hint]};
        if (!ptr.is_null()) item_anchor_hints.push_back(ptr);
      }
    }
    const bool local_stitch = storage_owner_local_stitch_mode(config);
    const bool use_anchors = storage_owner_anchor_mode(config) &&
                             !item_anchor_hints.empty();
    vec<RemotePtr> candidates;
    if (use_anchors) {
      auto t_search = std::chrono::steady_clock::now();
      candidates = anchor_search_candidates(components, item_anchor_hints, config,
                                            &breakdown, local_stitch);
      breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);
    } else if (!medoid_loaded) {
      auto t_medoid = std::chrono::steady_clock::now();
      medoid_ptr = read_global_medoid();
      medoid_loaded = true;
      breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
    }

    if (medoid_loaded && medoid_ptr.is_null()) {
      const RemotePtr new_ptr = allocate_local_node();
      if (results != nullptr) {
        (*results)[idx].new_rptr_raw = new_ptr.raw_address;
      }
      auto t_write = std::chrono::steady_clock::now();
      write_new_node(new_ptr, ids[idx], components, {}, generation);
      breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
      if (kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
        mark_node_deleted(old_entry.current, old_entry.generation);
      }
      publish_mutation(ids[idx], new_ptr, generation, false);
      if (maintenance_enabled) {
        (void)enqueue_insert_stitch(ids[idx], generation, new_ptr, config);
        (void)enqueue_deleted_node_cleanup(old_entry.current, config);
      }
      RemotePtr observed;
      if (try_set_global_medoid(RemotePtr{}, new_ptr, observed) || observed.is_null()) {
        medoid_ptr = new_ptr;
        if (statuses != nullptr) {
          (*statuses)[idx] = static_cast<u32>(service::storage_owner::MutationStatus::ok);
        }
        continue;
      }
      medoid_ptr = observed;
    }

    if (!use_anchors) {
      auto t_search = std::chrono::steady_clock::now();
      candidates = beam_search_candidates(components, medoid_ptr, config, &breakdown);
      breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);
    }
    hashset_t<RemotePtr> empty_skip;
    auto t_prune = std::chrono::steady_clock::now();
    vec<RemotePtr> selected_neighbors = robust_prune_cpu(reinterpret_cast<const byte_t*>(components.data()),
                                                         VectorDType::float32, candidates, empty_skip, config, &breakdown);
    breakdown.storage_owner_prune_ns += elapsed_ns_since(t_prune);
    const RemotePtr new_ptr = allocate_local_node();
    if (results != nullptr) {
      (*results)[idx].new_rptr_raw = new_ptr.raw_address;
    }
    auto t_write = std::chrono::steady_clock::now();
    write_new_node(new_ptr, ids[idx], components, selected_neighbors, generation);
    breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
    if (kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
      mark_node_deleted(old_entry.current, old_entry.generation);
    }
    publish_mutation(ids[idx], new_ptr, generation, false);
    if (maintenance_enabled) {
      (void)enqueue_insert_stitch(ids[idx], generation, new_ptr, config);
      (void)enqueue_deleted_node_cleanup(old_entry.current, config);
    }
    if (statuses != nullptr) {
      (*statuses)[idx] = static_cast<u32>(service::storage_owner::MutationStatus::ok);
    }

    if (!maintenance_enabled || local_stitch) {
      for (const RemotePtr& neighbor_ptr : selected_neighbors) {
        if (local_shard(neighbor_ptr.memory_node())) {
          local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
        } else if (!local_stitch) {
          remote_updates[neighbor_ptr.memory_node()].push_back(
            service::storage_owner::ReverseUpdateOp{neighbor_ptr.raw_address, new_ptr.raw_address});
        }
      }
    }
  }

  auto t_local_reverse = std::chrono::steady_clock::now();
  for (auto& [target_raw, candidates] : local_updates) {
    if (!apply_local_reverse_update(RemotePtr{target_raw}, candidates, config)) {
      return false;
    }
  }
  breakdown.storage_owner_local_reverse_ns += elapsed_ns_since(t_local_reverse);
  auto t_remote_reverse = std::chrono::steady_clock::now();
  for (auto& [target_shard, ops] : remote_updates) {
    if (!send_reverse_update_batch(target_shard, ops, config)) {
      return false;
    }
  }
  breakdown.storage_owner_remote_reverse_ns += elapsed_ns_since(t_remote_reverse);
  if (invalidated_neighbors != nullptr) {
    invalidated_neighbors->reserve(invalidated_neighbors->size() + local_updates.size());
    for (const auto& [target_raw, _] : local_updates) {
      invalidated_neighbors->push_back(target_raw);
    }
    for (const auto& [_, ops] : remote_updates) {
      for (const auto& op : ops) {
        invalidated_neighbors->push_back(op.target_raw);
      }
    }
  }
  return true;
}
