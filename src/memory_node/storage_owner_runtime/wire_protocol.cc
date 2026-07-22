#include <stdexcept>

#include "memory_node/peer_rpc/stage1_control_fanout_policy.hh"
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
        const size_t expected_bytes = request->magic == service::storage_owner::kMutationMagic
          ? service::storage_owner::mutation_batch_request_bytes(request->item_count)
          : service::storage_owner::insert_batch_request_bytes(request->item_count);
        if (magic_ok &&
            request->dim == config.dim &&
            request->owner_storage == storage_id_ &&
            request->item_count > 0 &&
            request->item_count <= config.storage_owner_batch_max &&
            request->vector_dtype == static_cast<u32>(VamanaNode::vector_dtype()) &&
            request->vector_bytes == VamanaNode::vector_bytes() &&
            request->protocol_version ==
              service::storage_owner::kMutationProtocolVersion &&
            bytes >= expected_bytes) {
          StorageOwnerInsertTask task;
          task.client_id = client_id;
          task.slot_id = slot_id;
          task.item_count = request->item_count;
          task.batch_id = request->batch_id;
          task.byte_len = bytes;
          task.received_at = std::chrono::steady_clock::now();
          mark_storage_owner_foreground_activity();
          // At most one descriptor exists per occupied receive slot. The
          // queue is preallocated to that exact protocol bound, so ingress
          // never blocks the CQ progress loop.
          lib_assert(storage_insert_tasks_->try_push(std::move(task)),
                     "storage-owner ingress queue exhausted despite RPC-slot bound");
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
      post_storage_owner_response(
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
    ? service::storage_owner::mutation_batch_request_bytes(request->item_count)
    : service::storage_owner::insert_batch_request_bytes(request->item_count);
  if (!magic_ok ||
      request->dim != config.dim ||
      request->owner_storage != storage_id_ ||
      request->item_count == 0 ||
      request->item_count > config.storage_owner_batch_max ||
      request->vector_dtype != static_cast<u32>(VamanaNode::vector_dtype()) ||
      request->vector_bytes != VamanaNode::vector_bytes() ||
      request->protocol_version !=
        service::storage_owner::kMutationProtocolVersion ||
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
  const u32* stage1_homes = mutation
    ? service::storage_owner::mutation_request_stage1_homes(
        payload, request->item_count)
    : service::storage_owner::request_stage1_homes(
        payload, request->item_count);
  const byte_t* raw_vectors = mutation
    ? service::storage_owner::mutation_request_vectors(payload, request->item_count)
    : service::storage_owner::request_vectors(payload, request->item_count);
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
                                                    raw_vectors, stage1_homes,
                                                    request->source_client,
                                                    request->batch_id,
                                                    request->item_count,
                                                    breakdown, config, &invalidated_neighbors,
                                                    &item_statuses, &mutation_results);
  storage_owner_insert_active_workers_.fetch_sub(1, std::memory_order_acq_rel);
  mark_storage_owner_foreground_activity();
  const auto response_build_started = std::chrono::steady_clock::now();
  for (u32 i = 0; i < request->item_count; ++i) {
    statuses[i] = ok ? item_statuses[i]
                     : static_cast<u32>(service::storage_owner::MutationStatus::failed);
  }
  auto* response_results = service::storage_owner::response_mutation_results(response_ptr, request->item_count);
  for (u32 i = 0; i < request->item_count; ++i) {
    response_results[i] = ok ? mutation_results[i] : service::storage_owner::MutationResult{};
  }
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
  breakdown.storage_owner_response_build_ns += elapsed_ns_since(response_build_started);
  *service::storage_owner::response_breakdown(response_ptr, request->item_count) = breakdown;
  return service::storage_owner::insert_batch_response_bytes(request->item_count);
}

bool MemoryNode::execute_storage_owner_batch_items(const node_t* ids,
                                       const service::storage_owner::MutationKind* kinds,
                                       const element_t* vectors,
                                       const byte_t* raw_vectors,
                                       const u32* stage1_homes,
                                       u32 source_client,
                                       u64 client_batch_id,
                                       size_t item_count,
                                       InsertBreakdownCounters& breakdown,
                                       const Configuration& config,
                                       vec<vec<u64>>* invalidated_neighbors,
                                       vec<u32>* statuses,
                                       vec<service::storage_owner::MutationResult>* results) {
  if (item_count == 0) {
    return true;
  }
  lib_assert(stage1_homes != nullptr,
             "two-stage mutation request omitted physical Stage1 homes");

  lib_assert(ids != nullptr && kinds != nullptr && vectors != nullptr &&
               raw_vectors != nullptr && client_batch_id != 0,
             "two-stage authority request omitted mutation identity or vectors");
  lib_assert(storage_owner_maintenance_enabled(config),
             "two-stage updates require the Stage2 maintenance runtime");
  if (statuses != nullptr) {
    statuses->assign(item_count, static_cast<u32>(service::storage_owner::MutationStatus::failed));
  }
  if (results != nullptr) {
    results->assign(item_count, {});
  }
  if (invalidated_neighbors != nullptr) {
    invalidated_neighbors->assign(item_count, {});
  }

  using namespace service::storage_owner;
  using BeginState =
    memory_node_storage_owner_index_detail::AuthorityBeginState;
  struct MutationPlan {
    MutationKind kind{MutationKind::insert};
    u32 stage1_home{};
    AuthorityOperationToken operation;
    AuthorityBeginResult begin;
    Stage1ExecuteItem stage1_item;
    Stage1ExecuteResult stage1_result;
    Stage1ArmResult arm_result;
    CleanupActivateItem cleanup_item;
    CleanupActivateResult cleanup_result;
    bool active{};
    bool committed_replay{};
    bool authority_committed{};
    bool fused_stage1{};
  };
  vec<MutationPlan> plans(item_count);
  dense_hashmap_t<u32, vec<size_t>> stage1_groups;

  const auto map_begin_failure = [](BeginState state) {
    switch (state) {
      case BeginState::already_exists:
        return MutationStatus::already_exists;
      case BeginState::not_found:
        return MutationStatus::not_found;
      case BeginState::already_deleted:
        return MutationStatus::already_deleted;
      case BeginState::busy:
      case BeginState::conflict:
      case BeginState::prepared:
      case BeginState::replay:
      case BeginState::committed_replay:
        return MutationStatus::failed;
    }
    return MutationStatus::failed;
  };

  // Logical authority acquisition is cheap and sharded by ID. It never
  // allocates graph memory and therefore remains independent of METIS skew.
  for (size_t index = 0; index < item_count; ++index) {
    // IDs index a configured, capacity-bounded logical namespace. Reject a
    // malformed remote request before it can create an immortal authority
    // tombstone; similarly, never trust a wire-provided physical home enough
    // to index the peer arrays. The per-item failed status is already set.
    if (ids[index] >= config.vector_id_namespace_size ||
        stage1_homes[index] >= num_storage_nodes_) {
      continue;
    }
    MutationPlan& plan = plans[index];
    plan.kind = kinds[index];
    plan.stage1_home = stage1_homes[index];
    plan.operation = AuthorityOperationToken{
      source_client, static_cast<u32>(index), client_batch_id};
    const auto prepare_started = std::chrono::steady_clock::now();
    plan.begin = begin_authority_mutation(
      ids[index], plan.kind, plan.operation, plan.stage1_home);
    breakdown.storage_owner_prepare_mutation_ns +=
      elapsed_ns_since(prepare_started);

    if (plan.begin.state == BeginState::committed_replay) {
      plan.committed_replay = true;
      if (statuses != nullptr) {
        (*statuses)[index] = static_cast<u32>(MutationStatus::ok);
      }
      if (results != nullptr) {
        const auto& replay = plan.begin.replay_result;
        (*results)[index] = MutationResult{
          .new_rptr_raw = replay.new_pointer.raw_address,
          .old_rptr_raw = replay.old_pointer.raw_address,
          .generation = replay.generation,
          .maintenance_sequence = replay.maintenance_sequence,
        };
      }
      continue;
    }
    if (!plan.begin.acquired()) {
      if (statuses != nullptr) {
        (*statuses)[index] = static_cast<u32>(
          map_begin_failure(plan.begin.state));
      }
      continue;
    }
    plan.active = true;
    if (results != nullptr) {
      (*results)[index].old_rptr_raw =
        plan.begin.previous.current.raw_address;
      (*results)[index].generation = plan.begin.generation;
    }
    if (plan.kind == MutationKind::erase) continue;

    plan.stage1_item = Stage1ExecuteItem{
      .client_batch_id = client_batch_id,
      .old_raw = plan.begin.previous.current.raw_address,
      // The complete physical-home subgroup below decides whether this stays
      // a legacy prepare or carries a fresh-insert fused-arm request. Starting
      // at zero prevents an individual item from opting into fusion before
      // the subgroup's cleanup dependency has been checked.
      .initial_placement_version = 0,
      .source_client = source_client,
      .item_index = static_cast<u32>(index),
      .id = ids[index],
      .generation = plan.begin.generation,
      .kind = static_cast<u32>(plan.kind),
      .authority_shard = storage_id_,
    };
    stage1_groups[plan.stage1_home].push_back(index);
  }

  // Fuse prepare and bounded Stage2 admission only when the complete
  // physical-home subgroup consists of fresh inserts. A mixed insert/upsert
  // subgroup stays on the legacy prepare -> cleanup -> arm path so the old
  // generation is never retired after its successor becomes runnable.
  for (const auto& [home, indices] : stage1_groups) {
    (void)home;
    const bool fused = std::all_of(
      indices.begin(), indices.end(), [&](const size_t index) {
        const MutationPlan& plan = plans[index];
        return plan.kind == MutationKind::insert &&
          plan.begin.previous.current.is_null();
      });
    if (!fused) continue;
    for (const size_t index : indices) {
      MutationPlan& plan = plans[index];
      lib_assert(plan.begin.previous.placement_version !=
                   std::numeric_limits<u64>::max(),
                 "authority placement version overflow");
      plan.fused_stage1 = true;
      plan.stage1_item.initial_placement_version =
        plan.begin.previous.placement_version + 1;
    }
  }

  // Group each item by its one centroid-selected home. Distinct remote-home
  // groups in the batch are posted as one fanout before waiting for any
  // response, so centroid skew does not turn the batch into a serial RPC
  // chain. Local work uses the identical semantic operation table and runs
  // only after those remote messages have been primed, overlapping both
  // physical-home execution paths.
  dense_hashmap_t<u32, vec<Stage1ExecuteItem>> remote_stage1_items;
  dense_hashmap_t<u32, vec<byte_t>> remote_stage1_vectors;
  vec<size_t> local_stage1_indices;
  for (const auto& [home, indices] : stage1_groups) {
    if (home == storage_id_) {
      local_stage1_indices.insert(
        local_stage1_indices.end(), indices.begin(), indices.end());
      continue;
    }
    vec<Stage1ExecuteItem>& wire_items = remote_stage1_items[home];
    vec<byte_t>& wire_vectors = remote_stage1_vectors[home];
    wire_items.reserve(indices.size());
    wire_vectors.resize(indices.size() * VamanaNode::vector_bytes());
    for (size_t slot = 0; slot < indices.size(); ++slot) {
      const size_t index = indices[slot];
      wire_items.push_back(plans[index].stage1_item);
      std::memcpy(
        wire_vectors.data() + slot * VamanaNode::vector_bytes(),
        raw_vectors + index * VamanaNode::vector_bytes(),
        VamanaNode::vector_bytes());
    }
  }

  u64 local_stage1_elapsed_ns = 0;
  const auto execute_local_stage1 = [&]() {
    const auto local_started = std::chrono::steady_clock::now();
    const auto finish_local_timing = [&]() {
      local_stage1_elapsed_ns += elapsed_ns_since(local_started);
    };
    for (const size_t index : local_stage1_indices) {
      const Stage1OperationKey key{
        .authority_shard = storage_id_,
        .source_client = plans[index].stage1_item.source_client,
        .item_index = plans[index].stage1_item.item_index,
        .client_batch_id = plans[index].stage1_item.client_batch_id,
      };
      do {
        while (!try_track_stage1_inflight_request(key)) {
          if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
            finish_local_timing();
            return false;
          }
          std::unique_lock<std::mutex> lock(
            storage_owner_maintenance_mutex_);
          storage_owner_maintenance_cv_.wait_for(
            lock, std::chrono::microseconds(100));
        }
        try {
          plans[index].stage1_result =
            prepare_and_maybe_arm_local_stage1_item(
              storage_id_, plans[index].stage1_item,
              raw_vectors + index * VamanaNode::vector_bytes(), config,
              &breakdown);
        } catch (...) {
          finish_stage1_inflight_request(key);
          throw;
        }
        finish_stage1_inflight_request(key);
        if (plans[index].stage1_result.status ==
            static_cast<u32>(MutationStatus::retry)) {
          if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
            finish_local_timing();
            return false;
          }
          std::unique_lock<std::mutex> lock(
            storage_owner_maintenance_mutex_);
          storage_owner_maintenance_cv_.wait_for(
            lock, std::chrono::microseconds(100));
        }
      } while (plans[index].stage1_result.status ==
               static_cast<u32>(MutationStatus::retry));
    }

    vec<size_t> fused_indices;
    vec<Stage1ArmItem> fused_items;
    fused_indices.reserve(local_stage1_indices.size());
    fused_items.reserve(local_stage1_indices.size());
    for (const size_t index : local_stage1_indices) {
      MutationPlan& plan = plans[index];
      if (!plan.fused_stage1 ||
          plan.stage1_result.status !=
            static_cast<u32>(MutationStatus::ok)) {
        continue;
      }
      fused_indices.push_back(index);
      fused_items.push_back(Stage1ArmItem{
        .token = plan.operation,
        .target_raw = plan.stage1_result.target_raw,
        .initial_placement_version =
          plan.stage1_item.initial_placement_version,
        .id = ids[index],
        .generation = plan.begin.generation,
        .action = static_cast<u32>(Stage1ArmAction::arm),
      });
    }
    if (!fused_items.empty()) {
      const auto arm_started = std::chrono::steady_clock::now();
      for (;;) {
        vec<Stage1ArmResult> arm_results;
        (void)arm_local_stage1_items(
          storage_id_, span<const Stage1ArmItem>{fused_items},
          arm_results, config);
        const auto disposition =
          memory_node_peer_rpc_detail::classify_stage1_control_response(
            span<const Stage1ArmItem>{fused_items},
            span<const Stage1ArmResult>{arm_results});
        if (disposition == memory_node_peer_rpc_detail::
              Stage1ControlResponseDisposition::resolved) {
          for (size_t slot = 0; slot < fused_indices.size(); ++slot) {
            MutationPlan& plan = plans[fused_indices[slot]];
            plan.arm_result = arm_results[slot];
            plan.stage1_result.maintenance_sequence =
              arm_results[slot].maintenance_sequence;
          }
          break;
        }
        if (disposition == memory_node_peer_rpc_detail::
              Stage1ControlResponseDisposition::malformed) {
          throw std::runtime_error(
            "local fused Stage1 arm returned a malformed result");
        }
        if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
          finish_local_timing();
          return false;
        }
        std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
        storage_owner_maintenance_cv_.wait_for(
          lock, std::chrono::microseconds(100));
      }
      breakdown.storage_owner_stage1_arm_wait_ns +=
        elapsed_ns_since(arm_started);
    }
    finish_local_timing();
    return true;
  };
  dense_hashmap_t<u32, vec<Stage1ExecuteResult>> remote_stage1_results;
  const auto stage1_execute_started = std::chrono::steady_clock::now();
  const bool stage1_transport_ok = execute_remote_stage1_fanout_and_wait(
    remote_stage1_items, remote_stage1_vectors,
    remote_stage1_results, execute_local_stage1, config);
  // Local search/prune time is already represented by its detailed counters.
  // Subtract the overlapped interval so the aggregate breakdown remains a
  // partition rather than double-counting concurrent local and remote work.
  const u64 stage1_critical_path_ns = elapsed_ns_since(stage1_execute_started);
  breakdown.storage_owner_stage1_execute_wait_ns +=
    stage1_critical_path_ns > local_stage1_elapsed_ns
      ? stage1_critical_path_ns - local_stage1_elapsed_ns : 0;
  if (!stage1_transport_ok) {
    // A fused request may already own a runnable Stage2 descriptor at its
    // physical home. Transport uncertainty is not a semantic prepare failure.
    if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
      return false;
    }
    throw std::runtime_error(
      "Stage1 transport stopped before every semantic token resolved");
  }
  for (const auto& [home, indices] : stage1_groups) {
    if (home == storage_id_) continue;
    const auto found = remote_stage1_results.find(home);
    if (found == remote_stage1_results.end() ||
        found->second.size() != indices.size()) {
      throw std::runtime_error(
        "Stage1 fanout resolved without a complete home result set");
    }
    for (size_t slot = 0; slot < indices.size(); ++slot) {
      plans[indices[slot]].stage1_result = found->second[slot];
    }
  }
  for (size_t index = 0; index < item_count; ++index) {
    MutationPlan& plan = plans[index];
    if (!plan.fused_stage1 ||
        plan.stage1_result.status !=
          static_cast<u32>(MutationStatus::ok)) {
      continue;
    }
    lib_assert(plan.stage1_result.maintenance_sequence != 0,
               "fused Stage1 success omitted its runnable maintenance fence");
    plan.arm_result = Stage1ArmResult{
      .token = plan.operation,
      .target_raw = plan.stage1_result.target_raw,
      .maintenance_sequence = plan.stage1_result.maintenance_sequence,
      .status = static_cast<u32>(MutationStatus::ok),
    };
  }

  const auto control_stage1 = [&](size_t index, Stage1ArmAction action,
                                  u64 placement_version,
                                  Stage1ArmResult* arm_result) {
    MutationPlan& plan = plans[index];
    if (plan.kind == MutationKind::erase) return true;
    const Stage1ArmItem item{
      .token = {
        .source_client = source_client,
        .item_index = static_cast<u32>(index),
        .client_batch_id = client_batch_id,
      },
      .target_raw = plan.stage1_result.target_raw,
      .initial_placement_version = placement_version,
      .id = ids[index],
      .generation = plan.begin.generation,
      .action = static_cast<u32>(action),
    };
    vec<Stage1ArmResult> control_results;
    bool transported = false;
    const auto control_started = std::chrono::steady_clock::now();
    if (plan.stage1_home == storage_id_) {
      transported = arm_local_stage1_items(
        storage_id_, span<const Stage1ArmItem>{&item, 1},
        control_results, config);
    } else {
      transported = arm_remote_stage1_batch(
        plan.stage1_home, source_client,
        span<const Stage1ArmItem>{&item, 1}, control_results, config);
    }
    const u64 control_ns = elapsed_ns_since(control_started);
    if (action == Stage1ArmAction::release) {
      breakdown.storage_owner_stage1_release_wait_ns += control_ns;
    } else {
      breakdown.storage_owner_stage1_arm_wait_ns += control_ns;
    }
    if (!transported || control_results.size() != 1) return false;
    const Stage1ArmResult& output = control_results.front();
    const bool same_token =
      output.token.source_client == item.token.source_client &&
      output.token.item_index == item.token.item_index &&
      output.token.client_batch_id == item.token.client_batch_id;
    if (!same_token || output.reserved != 0 ||
        output.status != static_cast<u32>(MutationStatus::ok)) {
      return false;
    }
    if (action == Stage1ArmAction::arm &&
        (output.target_raw != item.target_raw ||
         output.maintenance_sequence == 0)) {
      return false;
    }
    if (arm_result != nullptr) *arm_result = output;
    return true;
  };

  const auto release_stage1 = [&](size_t index, u64 placement_version) {
    while (!control_stage1(
      index, Stage1ArmAction::release, placement_version, nullptr)) {
      if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
        return false;
      }
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::microseconds(100));
    }
    return true;
  };

  const auto release_cleanup = [&](span<const CleanupActivateItem> activated) {
    if (activated.empty()) return true;
    const auto release_started = std::chrono::steady_clock::now();
    vec<CleanupActivateItem> releases(activated.begin(), activated.end());
    for (CleanupActivateItem& item : releases) {
      item.action = static_cast<u32>(CleanupActivateAction::release);
    }
    vec<CleanupActivateResult> release_results;
    bool released = false;
    while (!released &&
           !storage_insert_shutdown_.load(std::memory_order_acquire)) {
      released = activate_cleanup_fanout_and_wait(
        span<const CleanupActivateItem>{releases}, release_results, config);
      if (!released) {
        std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
        storage_owner_maintenance_cv_.wait_for(
          lock, std::chrono::microseconds(100));
      }
    }
    breakdown.storage_owner_cleanup_control_wait_ns +=
      elapsed_ns_since(release_started);
    return released;
  };

  // A failed physical prepare has no authority-visible side effect. Abort by
  // semantic token first (which is idempotent even when no artifact exists),
  // then release the per-ID authority lease.
  for (size_t index = 0; index < item_count; ++index) {
    MutationPlan& plan = plans[index];
    if (!plan.active || plan.kind == MutationKind::erase ||
        plan.stage1_result.status == static_cast<u32>(MutationStatus::ok)) {
      continue;
    }
    while (!control_stage1(
      index, Stage1ArmAction::abort, 0, nullptr)) {
      if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
        return false;
      }
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::microseconds(100));
    }
    // Keep the authority lease closed until the abort fence itself has been
    // released. A fresh retry of the same public token can then acquire the
    // lease without racing a stale physical receipt.
    if (!release_stage1(index, 0)) return false;
    (void)abort_authority_mutation(ids[index], plan.operation);
    plan.active = false;
  }

  vec<CleanupActivateItem> cleanup_items;
  vec<size_t> cleanup_indices;
  for (size_t index = 0; index < item_count; ++index) {
    MutationPlan& plan = plans[index];
    if (!plan.active || plan.begin.previous.current.is_null()) continue;
    cleanup_indices.push_back(index);
    plan.cleanup_item = CleanupActivateItem{
      .token = {
        .source_client = source_client,
        .item_index = static_cast<u32>(index),
        .client_batch_id = client_batch_id,
      },
      .old_raw = plan.begin.previous.current.raw_address,
      .id = ids[index],
      .old_generation = plan.begin.previous.generation,
      .authority_shard = storage_id_,
      .action = static_cast<u32>(CleanupActivateAction::activate),
    };
    cleanup_items.push_back(plan.cleanup_item);
  }
  vec<CleanupActivateResult> cleanup_results;
  bool cleanup_ok = cleanup_items.empty();
  const auto cleanup_started = std::chrono::steady_clock::now();
  while (!cleanup_ok &&
         !storage_insert_shutdown_.load(std::memory_order_acquire)) {
    cleanup_ok = activate_cleanup_fanout_and_wait(
      span<const CleanupActivateItem>{cleanup_items}, cleanup_results,
      config);
    if (!cleanup_ok) {
      for (const CleanupActivateResult& result : cleanup_results) {
        const auto status = static_cast<MutationStatus>(result.status);
        lib_assert(status == MutationStatus::failed ||
                     status == MutationStatus::ok,
                   "old physical generation no longer matches its authority directory");
      }
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::microseconds(100));
    }
  }
  if (!cleanup_ok) return false;
  if (cleanup_results.size() == cleanup_indices.size()) {
    for (size_t slot = 0; slot < cleanup_indices.size(); ++slot) {
      plans[cleanup_indices[slot]].cleanup_result = cleanup_results[slot];
    }
  }
  if (!cleanup_items.empty()) {
    breakdown.storage_owner_cleanup_control_wait_ns +=
      elapsed_ns_since(cleanup_started);
  }

  // Arm only after old-generation cleanup is runnable. Arm itself reserves a
  // completion sequence and immediately enqueues Stage2 before returning the
  // sequence; therefore authority commit never depends on a later activation
  // RPC and the completion ring contains no dormant Stage1 ticket.
  for (size_t index = 0; index < item_count; ++index) {
    MutationPlan& plan = plans[index];
    if (!plan.active) continue;
    const bool needs_cleanup = !plan.begin.previous.current.is_null();
    if (needs_cleanup &&
        (plan.cleanup_result.status != static_cast<u32>(MutationStatus::ok) ||
         plan.cleanup_result.maintenance_sequence == 0)) {
      while (!control_stage1(
        index, Stage1ArmAction::abort, 0, nullptr)) {
        if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
          return false;
        }
        std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
        storage_owner_maintenance_cv_.wait_for(
          lock, std::chrono::microseconds(100));
      }
      if (!release_stage1(index, 0)) return false;
      if (!release_cleanup(span<const CleanupActivateItem>{
            &plan.cleanup_item, 1})) {
        return false;
      }
      (void)abort_authority_mutation(ids[index], plan.operation);
      plan.active = false;
      continue;
    }

  }

  const auto commit_plan = [&](size_t index) {
    MutationPlan& plan = plans[index];
    if (!plan.active || plan.authority_committed) return;
    const bool deleted = plan.kind == MutationKind::erase;
    const RemotePtr desired = deleted
      ? RemotePtr{} : RemotePtr{plan.stage1_result.target_raw};
    const u64 maintenance_sequence = deleted
      ? plan.cleanup_result.maintenance_sequence
      : plan.arm_result.maintenance_sequence;
    lib_assert(maintenance_sequence != 0,
               "authority commit omitted its runnable maintenance fence");
    const auto commit_started = std::chrono::steady_clock::now();
    const AuthorityCommitState commit = commit_authority_mutation(
      ids[index], plan.operation, desired, plan.begin.generation, deleted,
      maintenance_sequence);
    breakdown.storage_owner_publish_mutation_ns +=
      elapsed_ns_since(commit_started);
    lib_assert(commit == AuthorityCommitState::committed ||
                 commit == AuthorityCommitState::replay,
               "token-fenced authority commit lost its active lease");
    plan.authority_committed = true;

    if (statuses != nullptr) {
      (*statuses)[index] = static_cast<u32>(MutationStatus::ok);
    }
    if (results != nullptr) {
      (*results)[index] = MutationResult{
        .new_rptr_raw = desired.raw_address,
        .old_rptr_raw = plan.begin.previous.current.raw_address,
        .generation = plan.begin.generation,
        .maintenance_sequence = maintenance_sequence,
      };
    }
  };

  // Cleanup-only mutations do not participate in a Stage1 arm group. Fresh
  // fused inserts already crossed their physical home's bounded admission
  // transaction in the Execute response. Commit both now so neither retains
  // an authority lease while an unrelated legacy home waits for credit.
  for (size_t index = 0; index < item_count; ++index) {
    if (plans[index].active &&
        (plans[index].kind == MutationKind::erase ||
         plans[index].fused_stage1)) {
      commit_plan(index);
    }
  }

  // Batch arm by physical home and post every remote batch together. Each
  // physical home remains an independent atomic admission transaction: when
  // its ACK arrives, commit exactly that authority subset immediately. No
  // home waits while the coordinator accumulates resources from every other
  // home, and only a still-unresolved home is retried with the same tokens.
  dense_hashmap_t<u32, vec<size_t>> arm_groups;
  dense_hashmap_t<u32, vec<Stage1ArmItem>> arm_items_by_home;
  for (size_t index = 0; index < item_count; ++index) {
    MutationPlan& plan = plans[index];
    if (!plan.active || plan.kind == MutationKind::erase ||
        plan.fused_stage1) {
      continue;
    }
    arm_groups[plan.stage1_home].push_back(index);
    lib_assert(plan.begin.previous.placement_version !=
                 std::numeric_limits<u64>::max(),
               "authority placement version overflow");
    arm_items_by_home[plan.stage1_home].push_back(Stage1ArmItem{
      .token = plan.operation,
      .target_raw = plan.stage1_result.target_raw,
      .initial_placement_version =
        plan.begin.previous.placement_version + 1,
      .id = ids[index],
      .generation = plan.begin.generation,
      .action = static_cast<u32>(Stage1ArmAction::arm),
    });
  }
  const auto arm_started = std::chrono::steady_clock::now();
  const bool armed = control_stage1_fanout_and_wait(
    arm_items_by_home, source_client,
    [&](u32 home, span<const Stage1ArmItem> arm_items,
        span<const Stage1ArmResult> arm_results) {
      const auto found = arm_groups.find(home);
      lib_assert(found != arm_groups.end() &&
                   found->second.size() == arm_items.size() &&
                   arm_results.size() == arm_items.size(),
                 "Stage1 arm fanout lost its home-to-authority mapping");
      for (size_t slot = 0; slot < found->second.size(); ++slot) {
        plans[found->second[slot]].arm_result = arm_results[slot];
      }
      // This callback runs as soon as this home's validated ACK is consumed;
      // it is deliberately not deferred until all other homes have armed.
      for (const size_t index : found->second) commit_plan(index);
    },
    config);
  breakdown.storage_owner_stage1_arm_wait_ns +=
    elapsed_ns_since(arm_started);
  if (!armed) {
    return false;
  }

  // Every active mutation must have crossed its own physical admission
  // transaction and authority linearization point. Keeping this audit here
  // makes it impossible to reintroduce a global commit barrier accidentally.
  for (size_t index = 0; index < item_count; ++index) {
    lib_assert(!plans[index].active || plans[index].authority_committed,
               "active mutation escaped per-home authority commit");
  }

  // Release compact Stage1 receipts only after the authority commit.  For a
  // remote home the release RPC uses the same RC control QP as every execute
  // retry.  Its ACK is therefore an ordered watermark: all older retries were
  // received, registered as in-flight, and quiesced before the receipt was
  // erased.  A Stage2-local "count is currently zero" observation cannot
  // establish that transport fact.  Batch by home so this costs at most one
  // control RTT per participating shard rather than one RTT per item.
  dense_hashmap_t<u32, vec<Stage1ArmItem>> committed_release_groups;
  for (size_t index = 0; index < item_count; ++index) {
    const MutationPlan& plan = plans[index];
    if (!plan.active || plan.kind == MutationKind::erase) continue;
    committed_release_groups[plan.stage1_home].push_back(Stage1ArmItem{
      .token = plan.operation,
      .target_raw = plan.stage1_result.target_raw,
      .initial_placement_version =
        plan.begin.previous.placement_version + 1,
      .id = ids[index],
      .generation = plan.begin.generation,
      .action = static_cast<u32>(Stage1ArmAction::release),
    });
  }
  const auto release_started = std::chrono::steady_clock::now();
  const bool released = control_stage1_fanout_and_wait(
    committed_release_groups, source_client, {}, config);
  breakdown.storage_owner_stage1_release_wait_ns +=
    elapsed_ns_since(release_started);
  if (!released) {
    // The directory commit is already durable. Returning a transport failure
    // is safe: replay of this public token observes committed_replay and never
    // performs physical work twice.
    return false;
  }

  vec<CleanupActivateItem> committed_cleanup_items;
  committed_cleanup_items.reserve(cleanup_items.size());
  for (const MutationPlan& plan : plans) {
    if (plan.active && !plan.begin.previous.current.is_null()) {
      committed_cleanup_items.push_back(plan.cleanup_item);
    }
  }
  return release_cleanup(
    span<const CleanupActivateItem>{committed_cleanup_items});
}
