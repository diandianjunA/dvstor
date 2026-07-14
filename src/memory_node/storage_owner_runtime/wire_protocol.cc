#include "memory_node/storage_owner_runtime/detail.hh"

using namespace memory_node_storage_owner_runtime_detail;

void MemoryNode::service_storage_runtime(const Configuration& config) {
  print_status("storage-owner insert runtime enabled on shard " + std::to_string(storage_id_));
  vec<ibv_wc> recv_wcs(std::max<i32>(1, config.max_recv_queue_wr));
  vec<ibv_wc> send_wcs(std::max<i32>(1, config.max_send_queue_wr));

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
    bool progressed = false;
    for (;;) {
      StorageOwnerResponseReady response;
      {
        std::lock_guard<std::mutex> lock(storage_responses_mutex_);
        if (storage_responses_ready_.empty()) {
          break;
        }
        response = storage_responses_ready_.front();
        storage_responses_ready_.pop_front();
      }
      byte_t* response_buffer = insert_runtime_.buffer.get_full_buffer() +
        insert_response_slot_offset(config, response.client_id, response.slot_id);
      const auto* response_header =
        reinterpret_cast<const service::storage_owner::InsertBatchResponseHeader*>(
          response_buffer);
      if (response.queued_at != std::chrono::steady_clock::time_point{} &&
          response_header->item_count > 0 &&
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
      progressed = true;
    }

    const i32 num_sent = context_.poll_send_cq(
      send_wcs.data(), static_cast<i32>(send_wcs.size()));
    progressed = progressed || num_sent > 0;
    for (i32 i = 0; i < num_sent; ++i) {
      const auto [client_id, slot_id] = decode_64bit(send_wcs[i].wr_id);
      if (client_id >= num_clients_ || slot_id >= insert_runtime_.request_slot_count) {
        continue;
      }
      cm_.client_qps[client_id]->post_receive(
        *insert_runtime_.region,
        static_cast<u32>(insert_runtime_.request_bytes),
        encode_64bit(client_id, slot_id),
        insert_request_slot_offset(client_id, slot_id));
    }

    const i32 num_received = context_.poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
    progressed = progressed || num_received > 0;

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
        const u32 expected_anchor_hint_count = storage_owner_local_stitch_mode(config)
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
          task.slot_id = slot_id;
          task.item_count = request->item_count;
          task.batch_id = request->batch_id;
          task.byte_len = bytes;
          task.received_at = std::chrono::steady_clock::now();
          mark_storage_owner_foreground_activity();
          {
            std::lock_guard<std::mutex> lock(storage_insert_tasks_mutex_);
            storage_insert_tasks_.push_back(std::move(task));
          }
          storage_insert_tasks_cv_.notify_one();
          handled_async = true;
        }
      }

      if (handled_async) {
        continue;
      }

      const size_t response_bytes = handle_storage_insert_request(
        client_id, slot_id, payload, bytes, config);
      lib_assert(response_bytes > 0, "invalid storage-owner insert request");
      lib_assert(response_bytes <= response_slot_bytes(config) &&
                 response_bytes <= std::numeric_limits<u32>::max(),
                 "storage_owner response exceeds the registered response slot");
      enqueue_storage_owner_response(
        {client_id,
         slot_id,
         static_cast<u32>(response_bytes),
         std::chrono::steady_clock::now()});
    }

    if (!progressed) {
      std::this_thread::yield();
    }
  }
}

size_t MemoryNode::response_slot_bytes(const Configuration& config) const {
  return align_up(service::storage_owner::insert_batch_response_bytes(config.storage_owner_batch_max));
}

size_t MemoryNode::handle_storage_insert_request(u32 client_id,
                                                 u32 slot_id,
                                                 const byte_t* payload,
                                                 size_t bytes,
                                                 const Configuration& config) {
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
      request->anchor_hint_count != (storage_owner_local_stitch_mode(config)
                                      ? config.storage_owner_anchor_hints : 0) ||
      bytes < expected_bytes) {
    return 0;
  }

  auto* response_ptr = reinterpret_cast<service::storage_owner::InsertBatchResponseHeader*>(
    insert_runtime_.buffer.get_full_buffer() +
    insert_response_slot_offset(config, client_id, slot_id));
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
  vec<vec<u64>> invalidated_neighbors;
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
  vec<u64> response_invalidations;
  response_invalidations.reserve(invalidation_capacity);
  for (const auto& item_invalidations : invalidated_neighbors) {
    response_invalidations.insert(response_invalidations.end(),
                                  item_invalidations.begin(), item_invalidations.end());
  }
  std::sort(response_invalidations.begin(), response_invalidations.end());
  response_invalidations.erase(
    std::unique(response_invalidations.begin(), response_invalidations.end()),
    response_invalidations.end());
  lib_assert(response_invalidations.size() <= invalidation_capacity,
             "storage_owner invalidation response exceeds its capacity");
  const u32 invalidation_count = static_cast<u32>(response_invalidations.size());
  *service::storage_owner::response_invalidation_count(response_ptr, request->item_count) = invalidation_count;
  u64* invalidated = service::storage_owner::response_invalidated_raws(response_ptr, request->item_count);
  for (u32 i = 0; i < invalidation_count; ++i) {
    invalidated[i] = response_invalidations[i];
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
                                       vec<vec<u64>>* invalidated_neighbors,
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
  if (invalidated_neighbors != nullptr) {
    invalidated_neighbors->assign(item_count, {});
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
        const u64 maintenance_sequence = schedule_storage_owner_maintenance(
          ids[idx], old_entry.generation, kind, RemotePtr{}, old_entry.current, config);
        if (results != nullptr) {
          (*results)[idx].maintenance_sequence = maintenance_sequence;
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
    const bool local_stitch = storage_owner_local_stitch_mode(config);
    if (anchor_hints != nullptr) {
      item_anchor_hints.reserve(anchor_hint_count);
      for (u32 hint = 0; hint < anchor_hint_count; ++hint) {
        const RemotePtr ptr{anchor_hints[idx * anchor_hint_count + hint]};
        if (!ptr.is_null() &&
            (!local_stitch || local_shard(ptr.memory_node()))) {
          item_anchor_hints.push_back(ptr);
        }
      }
    }
    if (local_stitch && item_anchor_hints.empty() &&
        storage_owner_anchor_index_ != nullptr &&
        !storage_owner_anchor_index_->empty()) {
      item_anchor_hints = storage_owner_anchor_index_->nearest_anchors(
        components,
        storage_id_,
        std::max<u32>(1, config.storage_owner_anchor_hints));
    }
    const bool use_anchors = local_stitch && !item_anchor_hints.empty();
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
      const u64 maintenance_sequence = schedule_storage_owner_maintenance(
        ids[idx], generation, kind, new_ptr, old_entry.current, config);
      if (results != nullptr) {
        (*results)[idx].maintenance_sequence = maintenance_sequence;
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
    hashset_t<RemotePtr> local_empty_skip;
    hashset_t<RemotePtr>& empty_skip =
      current_storage_owner_thread_ != nullptr
        ? current_storage_owner_thread_->coroutine_scratch_state().empty_skip
        : local_empty_skip;
    empty_skip.clear();
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
    const u64 maintenance_sequence = schedule_storage_owner_maintenance(
      ids[idx], generation, kind, new_ptr, old_entry.current, config);
    if (results != nullptr) {
      (*results)[idx].maintenance_sequence = maintenance_sequence;
    }
    if (statuses != nullptr) {
      (*statuses)[idx] = static_cast<u32>(service::storage_owner::MutationStatus::ok);
    }

    if (!maintenance_enabled) {
      for (const RemotePtr& neighbor_ptr : selected_neighbors) {
        if (local_shard(neighbor_ptr.memory_node())) {
          local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
          if (invalidated_neighbors != nullptr) {
            (*invalidated_neighbors)[idx].push_back(neighbor_ptr.raw_address);
          }
        } else if (!local_stitch) {
          remote_updates[neighbor_ptr.memory_node()].push_back(
            service::storage_owner::ReverseUpdateOp{neighbor_ptr.raw_address, new_ptr.raw_address});
          if (invalidated_neighbors != nullptr) {
            (*invalidated_neighbors)[idx].push_back(neighbor_ptr.raw_address);
          }
        }
      }
    }
  }

  hashset_t<u64> changed_local_nodes;
  auto t_local_reverse = std::chrono::steady_clock::now();
  for (auto& [target_raw, candidates] : local_updates) {
    bool graph_changed = false;
    if (!apply_partition_local_reverse_update(
          RemotePtr{target_raw}, candidates, config, &graph_changed)) {
      return false;
    }
    if (graph_changed) {
      changed_local_nodes.insert(target_raw);
    }
  }
  breakdown.storage_owner_local_reverse_ns += elapsed_ns_since(t_local_reverse);
  if (invalidated_neighbors != nullptr) {
    for (auto& item_invalidations : *invalidated_neighbors) {
      item_invalidations.erase(
        std::remove_if(item_invalidations.begin(), item_invalidations.end(),
                       [&](u64 raw) {
                         const RemotePtr pointer{raw};
                         return local_shard(pointer.memory_node()) &&
                                !changed_local_nodes.contains(raw);
                       }),
        item_invalidations.end());
    }
  }
  auto t_remote_reverse = std::chrono::steady_clock::now();
  for (auto& [target_shard, ops] : remote_updates) {
    if (!send_reverse_update_batch(target_shard, ops, config)) {
      return false;
    }
  }
  breakdown.storage_owner_remote_reverse_ns += elapsed_ns_since(t_remote_reverse);
  return true;
}
