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
      .source_client = source_client,
      .item_index = static_cast<u32>(index),
      .id = ids[index],
      .generation = plan.begin.generation,
      .kind = static_cast<u32>(plan.kind),
      .authority_shard = storage_id_,
    };
    stage1_groups[plan.stage1_home].push_back(index);
  }

  // Group each item by its one centroid-selected home. Distinct remote-home
  // groups in the batch are posted as one fanout before waiting for any
  // response, so centroid skew does not turn the batch into a serial RPC
  // chain. Local work uses the identical semantic operation table.
  dense_hashmap_t<u32, vec<Stage1ExecuteItem>> remote_stage1_items;
  dense_hashmap_t<u32, vec<byte_t>> remote_stage1_vectors;
  for (const auto& [home, indices] : stage1_groups) {
    if (home == storage_id_) {
      for (const size_t index : indices) {
        plans[index].stage1_result = prepare_local_stage1_item(
          storage_id_, plans[index].stage1_item,
          raw_vectors + index * VamanaNode::vector_bytes(), config,
          &breakdown);
      }
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
  dense_hashmap_t<u32, vec<Stage1ExecuteResult>> remote_stage1_results;
  (void)execute_remote_stage1_fanout_and_wait(
    remote_stage1_items, remote_stage1_vectors,
    remote_stage1_results, config);
  for (const auto& [home, indices] : stage1_groups) {
    if (home == storage_id_) continue;
    const auto found = remote_stage1_results.find(home);
    if (found == remote_stage1_results.end() ||
        found->second.size() != indices.size()) {
      continue;
    }
    for (size_t slot = 0; slot < indices.size(); ++slot) {
      plans[indices[slot]].stage1_result = found->second[slot];
    }
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
    if (plan.stage1_home == storage_id_) {
      transported = arm_local_stage1_items(
        storage_id_, span<const Stage1ArmItem>{&item, 1},
        control_results, config);
    } else {
      transported = arm_remote_stage1_batch(
        plan.stage1_home, source_client,
        span<const Stage1ArmItem>{&item, 1}, control_results, config);
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
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::microseconds(100));
    }
  };

  const auto release_cleanup = [&](span<const CleanupActivateItem> activated) {
    if (activated.empty()) return;
    vec<CleanupActivateItem> releases(activated.begin(), activated.end());
    for (CleanupActivateItem& item : releases) {
      item.action = static_cast<u32>(CleanupActivateAction::release);
    }
    vec<CleanupActivateResult> release_results;
    bool released = false;
    while (!released) {
      released = activate_cleanup_fanout_and_wait(
        span<const CleanupActivateItem>{releases}, release_results, config);
      if (!released) {
        std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
        storage_owner_maintenance_cv_.wait_for(
          lock, std::chrono::microseconds(100));
      }
    }
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
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::microseconds(100));
    }
    // Keep the authority lease closed until the abort fence itself has been
    // released. A fresh retry of the same public token can then acquire the
    // lease without racing a stale physical receipt.
    release_stage1(index, 0);
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
  while (!cleanup_ok) {
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
  if (cleanup_results.size() == cleanup_indices.size()) {
    for (size_t slot = 0; slot < cleanup_indices.size(); ++slot) {
      plans[cleanup_indices[slot]].cleanup_result = cleanup_results[slot];
    }
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
        std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
        storage_owner_maintenance_cv_.wait_for(
          lock, std::chrono::microseconds(100));
      }
      release_stage1(index, 0);
      release_cleanup(span<const CleanupActivateItem>{
        &plan.cleanup_item, 1});
      (void)abort_authority_mutation(ids[index], plan.operation);
      plan.active = false;
      continue;
    }

  }

  // Batch arm by physical home. A skewed centroid assignment therefore costs
  // one bounded control message rather than one RTT per item. Partial arm is
  // safe: the whole semantic batch is replayed and already-armed items return
  // their cached sequences without allocating another descriptor.
  dense_hashmap_t<u32, vec<size_t>> arm_groups;
  for (size_t index = 0; index < item_count; ++index) {
    const MutationPlan& plan = plans[index];
    if (plan.active && plan.kind != MutationKind::erase) {
      arm_groups[plan.stage1_home].push_back(index);
    }
  }
  for (const auto& [home, indices] : arm_groups) {
    vec<Stage1ArmItem> arm_items;
    arm_items.reserve(indices.size());
    for (const size_t index : indices) {
      const MutationPlan& plan = plans[index];
      lib_assert(plan.begin.previous.placement_version !=
                   std::numeric_limits<u64>::max(),
                 "authority placement version overflow");
      arm_items.push_back(Stage1ArmItem{
        .token = plan.operation,
        .target_raw = plan.stage1_result.target_raw,
        .initial_placement_version =
          plan.begin.previous.placement_version + 1,
        .id = ids[index],
        .generation = plan.begin.generation,
        .action = static_cast<u32>(Stage1ArmAction::arm),
      });
    }

    bool armed = false;
    while (!armed) {
      vec<Stage1ArmResult> arm_results;
      const bool transported = home == storage_id_
        ? arm_local_stage1_items(
            storage_id_, span<const Stage1ArmItem>{arm_items},
            arm_results, config)
        : arm_remote_stage1_batch(
            home, source_client, span<const Stage1ArmItem>{arm_items},
            arm_results, config);
      armed = transported && arm_results.size() == arm_items.size();
      for (size_t slot = 0; armed && slot < arm_items.size(); ++slot) {
        const Stage1ArmItem& input = arm_items[slot];
        const Stage1ArmResult& output = arm_results[slot];
        armed = output.token.source_client == input.token.source_client &&
          output.token.item_index == input.token.item_index &&
          output.token.client_batch_id == input.token.client_batch_id &&
          output.target_raw == input.target_raw && output.reserved == 0 &&
          output.status == static_cast<u32>(MutationStatus::ok) &&
          output.maintenance_sequence != 0;
      }
      if (armed) {
        for (size_t slot = 0; slot < indices.size(); ++slot) {
          plans[indices[slot]].arm_result = arm_results[slot];
        }
        break;
      }
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::microseconds(100));
    }
  }

  // Directory commit is the sole logical linearization point. Old cleanup is
  // runnable and the new Stage1 record is both query-visible and armed before
  // this point. A prematurely scheduled Stage2 observes the authority lease
  // as busy at its placement CAS and retries until this commit releases it.
  for (size_t index = 0; index < item_count; ++index) {
    MutationPlan& plan = plans[index];
    if (!plan.active) continue;
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
  }

  // A committed public replay terminates at the authority directory and never
  // needs the physical Stage1 receipt again. Release in the same per-home
  // batches used by arm so high update rates pay one bounded control RTT per
  // centroid home, not one RTT per vector. A lost response is harmless: the
  // release postcondition (missing receipt) is itself ACKed on retry.
  for (const auto& [home, indices] : arm_groups) {
    vec<Stage1ArmItem> release_items;
    release_items.reserve(indices.size());
    for (const size_t index : indices) {
      const MutationPlan& plan = plans[index];
      release_items.push_back(Stage1ArmItem{
        .token = plan.operation,
        .target_raw = plan.stage1_result.target_raw,
        .initial_placement_version =
          plan.begin.previous.placement_version + 1,
        .id = ids[index],
        .generation = plan.begin.generation,
        .action = static_cast<u32>(Stage1ArmAction::release),
      });
    }

    bool released = false;
    while (!released) {
      vec<Stage1ArmResult> release_results;
      const bool transported = home == storage_id_
        ? arm_local_stage1_items(
            storage_id_, span<const Stage1ArmItem>{release_items},
            release_results, config)
        : arm_remote_stage1_batch(
            home, source_client, span<const Stage1ArmItem>{release_items},
            release_results, config);
      released = transported &&
        release_results.size() == release_items.size();
      for (size_t slot = 0; released && slot < release_items.size(); ++slot) {
        const Stage1ArmItem& input = release_items[slot];
        const Stage1ArmResult& output = release_results[slot];
        released = output.token.source_client == input.token.source_client &&
          output.token.item_index == input.token.item_index &&
          output.token.client_batch_id == input.token.client_batch_id &&
          output.target_raw == input.target_raw && output.reserved == 0 &&
          output.status == static_cast<u32>(MutationStatus::ok);
      }
      if (!released) {
        std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
        storage_owner_maintenance_cv_.wait_for(
          lock, std::chrono::microseconds(100));
      }
    }
  }

  vec<CleanupActivateItem> committed_cleanup_items;
  committed_cleanup_items.reserve(cleanup_items.size());
  for (const MutationPlan& plan : plans) {
    if (plan.active && !plan.begin.previous.current.is_null()) {
      committed_cleanup_items.push_back(plan.cleanup_item);
    }
  }
  release_cleanup(
    span<const CleanupActivateItem>{committed_cleanup_items});
  return true;
}
