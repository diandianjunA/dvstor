#include <stdexcept>

#include "memory_node/peer_rpc/stage1_control_fanout_policy.hh"
#include "memory_node/storage_owner_runtime/detail.hh"
#include "service/storage_owner_client_helpers.hh"

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
      const auto [encoded_client, slot_id] = decode_64bit(send_wcs[i].wr_id);
      const bool token_completion =
        service::storage_owner_client::storage_owner_is_completion_wr(
          encoded_client);
      const u32 client_id =
        service::storage_owner_client::storage_owner_wr_owner(
          encoded_client);
      if (token_completion) {
        if (client_id < storage_client_completion_free_slots_.size() &&
            slot_id < insert_runtime_.completion_slot_count) {
          lib_assert(storage_client_completion_free_slots_[client_id]->try_push(
                       slot_id),
                     "storage-owner completion credit returned twice");
        }
        continue;
      }
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
          task.byte_len = expected_bytes;
          task.payload.assign(payload, payload + expected_bytes);
          task.completion_slots.reserve(request->item_count);
          task.received_at = std::chrono::steady_clock::now();

          byte_t* ack_buffer = insert_runtime_.buffer.get_full_buffer() +
            insert_response_slot_offset(config, client_id, slot_id);
          auto* ack = reinterpret_cast<
            service::storage_owner::MutationBatchAckV2*>(ack_buffer);
          *ack = service::storage_owner::MutationBatchAckV2{
            .magic = request->magic,
            .owner_storage = storage_id_,
            .item_count = request->item_count,
            .status = static_cast<u32>(
              service::storage_owner::MutationBatchAckStatus::busy),
            .protocol_version =
              service::storage_owner::kMutationProtocolVersion,
            .batch_id = request->batch_id,
          };

          // Serialize acceptance before every completion SEND on this RC QP.
          // The worker may dequeue immediately, but its completion post takes
          // the same mutex and therefore cannot overtake this ACK.
          bool context_reserved = false;
          auto& context_credits =
            storage_client_batch_context_credits_[client_id];
          u32 available = context_credits.load(std::memory_order_acquire);
          while (available != 0 &&
                 !context_credits.compare_exchange_weak(
                   available, available - 1,
                   std::memory_order_acq_rel,
                   std::memory_order_acquire)) {
          }
          context_reserved = available != 0;
          if (context_reserved) {
            u32 completion_slot = 0;
            while (task.completion_slots.size() < request->item_count &&
                   storage_client_completion_free_slots_[client_id]->try_pop(
                     completion_slot)) {
              task.completion_slots.push_back(completion_slot);
            }
            if (task.completion_slots.size() != request->item_count) {
              for (const u32 reserved_slot : task.completion_slots) {
                lib_assert(
                  storage_client_completion_free_slots_[client_id]->try_push(
                    reserved_slot),
                  "failed to roll back storage-owner completion reservation");
              }
              task.completion_slots.clear();
              context_credits.fetch_add(1, std::memory_order_release);
              context_reserved = false;
            }
          }
          std::lock_guard<std::mutex> send_lock(
            *storage_client_send_mutexes_[client_id]);
          if (context_reserved &&
              storage_insert_tasks_->try_push(std::move(task))) {
            ack->status = static_cast<u32>(
              service::storage_owner::MutationBatchAckStatus::accepted);
            mark_storage_owner_foreground_activity();
          } else if (context_reserved) {
            // bounded::Queue does not move its input unless it successfully
            // claims a cell, so the reservation remains available here.
            for (const u32 reserved_slot : task.completion_slots) {
              lib_assert(
                storage_client_completion_free_slots_[client_id]->try_push(
                  reserved_slot),
                "failed to roll back rejected completion reservation");
            }
            context_credits.fetch_add(1, std::memory_order_release);
          }
          cm_.client_qps[client_id]->post_send_with_id(
            *insert_runtime_.region,
            sizeof(service::storage_owner::MutationBatchAckV2),
            IBV_WR_SEND,
            encode_64bit(client_id, slot_id),
            true,
            nullptr,
            0,
            insert_response_slot_offset(config, client_id, slot_id));
          handled_async = true;
        }
      }

      if (handled_async) {
        continue;
      }

      // Reject a parseable malformed request without entering the authority
      // pipeline. The ACK carries its batch id so the compute side can fail
      // the exact logical tasks instead of waiting for an RPC timeout.
      if (bytes >= sizeof(service::storage_owner::InsertBatchRequestHeader)) {
        const auto* request = reinterpret_cast<const
          service::storage_owner::InsertBatchRequestHeader*>(payload);
        const size_t response_offset =
          insert_response_slot_offset(config, client_id, slot_id);
        auto* ack = reinterpret_cast<
          service::storage_owner::MutationBatchAckV2*>(
            insert_runtime_.buffer.get_full_buffer() + response_offset);
        *ack = service::storage_owner::MutationBatchAckV2{
          .magic = request->magic,
          .owner_storage = storage_id_,
          .item_count = request->item_count,
          .status = static_cast<u32>(
            service::storage_owner::MutationBatchAckStatus::malformed),
          .protocol_version = service::storage_owner::kMutationProtocolVersion,
          .batch_id = request->batch_id,
        };
        std::lock_guard<std::mutex> send_lock(
          *storage_client_send_mutexes_[client_id]);
        cm_.client_qps[client_id]->post_send_with_id(
          *insert_runtime_.region,
          sizeof(*ack), IBV_WR_SEND,
          encode_64bit(client_id, slot_id), true, nullptr, 0,
          response_offset);
      } else {
        // There is no safe batch identity to echo. Replenish the transport
        // receive and let the sender's existing fail-stop/timeout policy
        // diagnose the truncated connection payload.
        cm_.client_qps[client_id]->post_receive(
          *insert_runtime_.region,
          static_cast<u32>(insert_runtime_.request_bytes),
          encode_64bit(client_id, slot_id),
          insert_request_slot_offset(client_id, slot_id));
      }
    }

    if (!progressed) {
      std::this_thread::yield();
    }
  }
}

size_t MemoryNode::response_slot_bytes(const Configuration& config) const {
  (void)config;
  return align_up(sizeof(service::storage_owner::MutationBatchAckV2));
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
  const u64* operation_ids = mutation
    ? service::storage_owner::mutation_request_operation_ids(
        payload, request->item_count)
    : service::storage_owner::request_operation_ids(
        payload, request->item_count);
  const byte_t* raw_vectors = mutation
    ? service::storage_owner::mutation_request_vectors(payload, request->item_count)
    : service::storage_owner::request_vectors(payload, request->item_count);
  vec<service::storage_owner::MutationKind> kinds(request->item_count, service::storage_owner::MutationKind::insert);
  for (u32 i = 0; i < request->item_count && kinds_raw != nullptr; ++i) {
    kinds[i] = static_cast<service::storage_owner::MutationKind>(kinds_raw[i]);
  }
  InsertBreakdownCounters breakdown{};
  vec<vec<u64>> invalidated_neighbors;
  vec<u32> item_statuses(request->item_count, static_cast<u32>(service::storage_owner::MutationStatus::failed));
  vec<service::storage_owner::MutationResult> mutation_results(request->item_count);
  mark_storage_owner_foreground_activity();
  storage_owner_insert_active_workers_.fetch_add(1, std::memory_order_acq_rel);
  const bool ok = execute_storage_owner_batch_items(ids, kinds.data(), raw_vectors,
                                                    stage1_homes,
                                                    operation_ids,
                                                    request->source_client,
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
                                       const byte_t* raw_vectors,
                                       const u32* stage1_homes,
                                       const u64* operation_ids,
                                       u32 source_client,
                                       size_t item_count,
                                       InsertBreakdownCounters& breakdown,
                                       const Configuration& config,
                                       vec<vec<u64>>* invalidated_neighbors,
                                       vec<u32>* statuses,
                                       vec<service::storage_owner::MutationResult>* results,
                                       const std::function<void(size_t)>&
                                         on_terminal) {
  if (item_count == 0) {
    return true;
  }
  lib_assert(stage1_homes != nullptr,
             "two-stage mutation request omitted physical Stage1 homes");

  lib_assert(ids != nullptr && kinds != nullptr && raw_vectors != nullptr &&
               operation_ids != nullptr,
             "two-stage authority request omitted mutation identity or vectors");
  for (size_t index = 0; index < item_count; ++index) {
    lib_assert(operation_ids[index] != 0,
               "two-stage authority request contains a zero operation id");
  }
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
    bool stage1_receipt_released{};
  };
  vec<MutationPlan> plans(item_count);
  dense_hashmap_t<u32, vec<size_t>> stage1_groups;
  const auto plan_index_for_token = [&](u32 token_source,
                                        u64 token_operation) -> size_t {
    for (size_t candidate = 0; candidate < plans.size(); ++candidate) {
      if (plans[candidate].operation.source_client == token_source &&
          plans[candidate].operation.client_batch_id == token_operation) {
        return candidate;
      }
    }
    return plans.size();
  };

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
      if (on_terminal) on_terminal(index);
      continue;
    }
    MutationPlan& plan = plans[index];
    plan.kind = kinds[index];
    plan.stage1_home = stage1_homes[index];
    plan.operation = AuthorityOperationToken{
      source_client, 0, operation_ids[index]};
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
      if (on_terminal) on_terminal(index);
      continue;
    }
    if (!plan.begin.acquired()) {
      if (statuses != nullptr) {
        (*statuses)[index] = static_cast<u32>(
          map_begin_failure(plan.begin.state));
      }
      if (on_terminal) on_terminal(index);
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
      .client_batch_id = operation_ids[index],
      .old_raw = plan.begin.previous.current.raw_address,
      // The complete physical-home subgroup below decides whether this stays
      // a legacy prepare or carries a fresh-insert fused-arm request. Starting
      // at zero prevents an individual item from opting into fusion before
      // the subgroup's cleanup dependency has been checked.
      .initial_placement_version = 0,
      .source_client = source_client,
      .item_index = 0,
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

  // Authority publication is intentionally per item, not batch-atomic.  In
  // particular, a fresh fused insert must be committed as soon as its own
  // physical home proves publication into the bounded accepted Stage2
  // backlog; retaining that lease while another home waits for accepted
  // capacity creates a cross-home cycle.
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
    if (on_terminal) on_terminal(index);
  };

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
  const auto execute_local_stage1_attempt = [&](size_t index) {
    MutationPlan& plan = plans[index];
    const Stage1OperationKey key{
      .authority_shard = storage_id_,
      .source_client = plan.stage1_item.source_client,
      .item_index = plan.stage1_item.item_index,
      .client_batch_id = plan.stage1_item.client_batch_id,
    };
    if (!try_track_stage1_inflight_request(key)) {
      // Do not wait here: remote fused Execute requests were already posted.
      // Returning an explicit local retry lets the remote ACK loop commit and
      // release those homes before this coordinator waits for local capacity.
      plan.stage1_result = Stage1ExecuteResult{
        .client_batch_id = plan.stage1_item.client_batch_id,
        .source_client = plan.stage1_item.source_client,
        .item_index = plan.stage1_item.item_index,
        .status = static_cast<u32>(MutationStatus::retry),
      };
      return;
    }
    try {
      plan.stage1_result = prepare_and_maybe_arm_local_stage1_item(
        storage_id_, plan.stage1_item,
        raw_vectors + index * VamanaNode::vector_bytes(), config,
        &breakdown);
    } catch (...) {
      finish_stage1_inflight_request(key);
      throw;
    }
    finish_stage1_inflight_request(key);
  };
  const auto execute_local_stage1 = [&]() {
    const auto local_started = std::chrono::steady_clock::now();
    const auto finish_local_timing = [&]() {
      local_stage1_elapsed_ns += elapsed_ns_since(local_started);
    };
    for (const size_t index : local_stage1_indices) {
      execute_local_stage1_attempt(index);
    }

    // Do not reserve local completion credit in this overlap callback. Remote
    // homes have already been posted, but their fused ACKs are consumed only
    // after this function returns. A blocking local ARM here could therefore
    // prevent the remote per-home authority commits that release that credit.
    finish_local_timing();
    return true;
  };
  dense_hashmap_t<u32, vec<Stage1ExecuteResult>> remote_stage1_results;
  const auto resolve_remote_stage1_home = [&] (
      u32 home, span<const Stage1ExecuteItem> home_items,
      span<const Stage1ExecuteResult> home_results) {
    const auto found = stage1_groups.find(home);
    lib_assert(found != stage1_groups.end() && !home_items.empty() &&
                 home_results.size() == home_items.size(),
               "Stage1 execute fanout lost its home-to-authority mapping");
    for (size_t slot = 0; slot < home_items.size(); ++slot) {
      // A physical-home response may contain only the tokens that resolved in
      // this wave. Map by the stable public operation id; transport ordinals
      // are intentionally excluded from authority identity.
      const size_t index = plan_index_for_token(
        home_items[slot].source_client, home_items[slot].client_batch_id);
      lib_assert(index < plans.size(),
                 "Stage1 partial callback referenced an invalid mutation");
      MutationPlan& plan = plans[index];
      lib_assert(plan.stage1_home == home &&
                   plan.stage1_item.client_batch_id ==
                     home_items[slot].client_batch_id &&
                   plan.stage1_item.source_client ==
                     home_items[slot].source_client &&
                   plan.stage1_item.item_index == home_items[slot].item_index,
                 "Stage1 home callback crossed a mutation token");
      plan.stage1_result = home_results[slot];
      if (!plan.fused_stage1 ||
          plan.stage1_result.status !=
            static_cast<u32>(MutationStatus::ok)) {
        continue;
      }
      lib_assert(plan.stage1_result.maintenance_sequence != 0,
                 "fused Stage1 home ACK omitted bounded admission");
      plan.arm_result = Stage1ArmResult{
        .token = plan.operation,
        .target_raw = plan.stage1_result.target_raw,
        .maintenance_sequence = plan.stage1_result.maintenance_sequence,
        .status = static_cast<u32>(MutationStatus::ok),
      };
      // The response lease has already been acknowledged. Commit this home
      // now, while unrelated homes may still be waiting or retrying.
      commit_plan(index);
    }
  };

  // The asynchronous per-home Execute state machine owns the only normal-path
  // remote release. It invokes this notification only after validating the
  // release envelope and every token/status, then acknowledging the response
  // lease. Map by the release input token: an idempotent missing-receipt ACK is
  // allowed to return target_raw == 0.
  const auto resolve_remote_stage1_release = [&] (
      u32 home, span<const Stage1ArmItem> release_items) {
    for (const Stage1ArmItem& item : release_items) {
      const size_t index = plan_index_for_token(
        item.token.source_client, item.token.client_batch_id);
      lib_assert(index < plans.size(),
                 "Stage1 release ACK referenced an invalid mutation index");
      const MutationPlan& plan = plans[index];
      lib_assert(home != storage_id_ && plan.stage1_home == home &&
                   plan.fused_stage1 && plan.authority_committed &&
                   plan.operation.source_client == item.token.source_client &&
                   plan.operation.item_index == item.token.item_index &&
                   plan.operation.client_batch_id ==
                     item.token.client_batch_id &&
                   plan.stage1_result.target_raw == item.target_raw &&
                   plan.stage1_item.initial_placement_version ==
                     item.initial_placement_version &&
                   ids[index] == item.id &&
                   plan.begin.generation == item.generation &&
                   static_cast<Stage1ArmAction>(item.action) ==
                     Stage1ArmAction::release,
                 "Stage1 release ACK crossed a committed mutation token");
    }
    for (const Stage1ArmItem& item : release_items) {
      const size_t index = plan_index_for_token(
        item.token.source_client, item.token.client_batch_id);
      lib_assert(index < plans.size(),
                 "Stage1 release completion lost its stable operation id");
      plans[index].stage1_receipt_released = true;
    }
  };

  const auto collect_remote_stage1_release_debt = [&]() {
    dense_hashmap_t<u32, vec<Stage1ArmItem>> groups;
    for (size_t index = 0; index < item_count; ++index) {
      const MutationPlan& plan = plans[index];
      if (!plan.active || !plan.fused_stage1 ||
          !plan.authority_committed || plan.stage1_receipt_released ||
          plan.stage1_home == storage_id_) {
        continue;
      }
      groups[plan.stage1_home].push_back(Stage1ArmItem{
        .token = plan.operation,
        .target_raw = plan.stage1_result.target_raw,
        .initial_placement_version =
          plan.stage1_item.initial_placement_version,
        .id = ids[index],
        .generation = plan.begin.generation,
        .action = static_cast<u32>(Stage1ArmAction::release),
      });
    }
    return groups;
  };

  const auto stage1_execute_started = std::chrono::steady_clock::now();
  bool stage1_transport_ok = false;
  try {
    stage1_transport_ok = execute_remote_stage1_fanout_and_wait(
      remote_stage1_items, remote_stage1_vectors,
      remote_stage1_results, resolve_remote_stage1_home,
      resolve_remote_stage1_release, execute_local_stage1, config);
  } catch (...) {
    // A callback can fail after committing only part of a remote home. Drain
    // exactly those committed receipt debts before propagating the exception;
    // uncommitted prepares remain replayable under their authority leases.
    auto debt = collect_remote_stage1_release_debt();
    if (!debt.empty()) {
      const auto release_started = std::chrono::steady_clock::now();
      const bool debt_released = control_stage1_fanout_and_wait(
        debt, source_client,
        [&](u32 home, span<const Stage1ArmItem> items,
            span<const Stage1ArmResult>) {
          resolve_remote_stage1_release(home, items);
        },
        config);
      breakdown.storage_owner_stage1_release_wait_ns +=
        elapsed_ns_since(release_started);
      if (!debt_released &&
          storage_insert_shutdown_.load(std::memory_order_acquire)) {
        return false;
      }
    }
    throw;
  }
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

  // A local capacity retry must not hold already-posted remote homes behind
  // the overlap callback. Those homes are now committed and their receipts
  // released, so retrying local prepare cannot participate in a cross-home
  // hold-and-wait cycle. The uncontended path still performed its full local
  // search concurrently with the remote Execute requests above.
  for (const size_t index : local_stage1_indices) {
    while (plans[index].stage1_result.status ==
           static_cast<u32>(MutationStatus::retry)) {
      if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
        return false;
      }
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::microseconds(100));
      lock.unlock();
      execute_local_stage1_attempt(index);
    }
  }

  // Remote homes have now either committed their fused authority subset or
  // returned a terminal prepare result. Only now may a local fused home wait
  // for admission: no remote completion credit remains pinned behind an
  // unconsumed ACK. Search/prune was still overlapped above; only this bounded
  // local admission step is ordered after the remote event loop.
  vec<size_t> local_fused_indices;
  vec<Stage1ArmItem> local_fused_items;
  local_fused_indices.reserve(local_stage1_indices.size());
  local_fused_items.reserve(local_stage1_indices.size());
  for (const size_t index : local_stage1_indices) {
    MutationPlan& plan = plans[index];
    if (!plan.fused_stage1 ||
        plan.stage1_result.status !=
          static_cast<u32>(MutationStatus::ok)) {
      continue;
    }
    local_fused_indices.push_back(index);
    local_fused_items.push_back(Stage1ArmItem{
      .token = plan.operation,
      .target_raw = plan.stage1_result.target_raw,
      .initial_placement_version =
        plan.stage1_item.initial_placement_version,
      .id = ids[index],
      .generation = plan.begin.generation,
      .action = static_cast<u32>(Stage1ArmAction::arm),
    });
  }
  if (!local_fused_items.empty()) {
    const auto arm_started = std::chrono::steady_clock::now();
    vec<Stage1ArmResult> arm_results;
    (void)arm_local_stage1_items(
      storage_id_, span<const Stage1ArmItem>{local_fused_items},
      arm_results, config);
    const auto batch_disposition =
      memory_node_peer_rpc_detail::classify_stage1_control_response(
        span<const Stage1ArmItem>{local_fused_items},
        span<const Stage1ArmResult>{arm_results});
    if (batch_disposition == memory_node_peer_rpc_detail::
          Stage1ControlResponseDisposition::malformed) {
      throw std::runtime_error(
        "local fused Stage1 arm returned a malformed result");
    }
    if (batch_disposition == memory_node_peer_rpc_detail::
          Stage1ControlResponseDisposition::resolved) {
      // Uncontended fast path: retain one completion-ring reservation and one
      // maintenance-queue transaction for the complete local subgroup.
      for (size_t slot = 0; slot < local_fused_indices.size(); ++slot) {
        MutationPlan& plan = plans[local_fused_indices[slot]];
        plan.arm_result = arm_results[slot];
        plan.stage1_result.maintenance_sequence =
          arm_results[slot].maintenance_sequence;
        commit_plan(local_fused_indices[slot]);
      }
    } else {
      // The authority is also this physical home for roughly one shard's
      // worth of updates. Do not leave that subset behind the old atomic ARM
      // retry loop: sweep size-one try-admissions, commit every success
      // immediately so Stage2 can return its credit, and retain only retrying
      // slots for the next sweep.
      vec<size_t> pending_slots(local_fused_items.size());
      for (size_t slot = 0; slot < pending_slots.size(); ++slot) {
        pending_slots[slot] = slot;
      }
      vec<Stage1ArmResult> one_result;
      one_result.reserve(1);
      while (!pending_slots.empty()) {
        bool made_progress = false;
        vec<size_t> retry_slots;
        retry_slots.reserve(pending_slots.size());
        for (const size_t slot : pending_slots) {
          const Stage1ArmItem& arm_item = local_fused_items[slot];
          (void)arm_local_stage1_items(
            storage_id_, span<const Stage1ArmItem>{&arm_item, 1},
            one_result, config);
          const auto disposition =
            memory_node_peer_rpc_detail::classify_stage1_control_response(
              span<const Stage1ArmItem>{&arm_item, 1},
              span<const Stage1ArmResult>{one_result});
          if (disposition == memory_node_peer_rpc_detail::
                Stage1ControlResponseDisposition::malformed) {
            throw std::runtime_error(
              "local partial fused Stage1 arm returned a malformed result");
          }
          if (disposition == memory_node_peer_rpc_detail::
                Stage1ControlResponseDisposition::retry) {
            retry_slots.push_back(slot);
            continue;
          }
          MutationPlan& plan = plans[local_fused_indices[slot]];
          plan.arm_result = one_result.front();
          plan.stage1_result.maintenance_sequence =
            one_result.front().maintenance_sequence;
          commit_plan(local_fused_indices[slot]);
          made_progress = true;
        }
        pending_slots = std::move(retry_slots);
        if (pending_slots.empty()) break;
        if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
          return false;
        }
        if (!made_progress) {
          std::unique_lock<std::mutex> lock(
            storage_owner_maintenance_mutex_);
          storage_owner_maintenance_cv_.wait_for(
            lock, std::chrono::microseconds(100));
        }
      }
    }
    breakdown.storage_owner_stage1_arm_wait_ns +=
      elapsed_ns_since(arm_started);

    vec<Stage1ArmItem> local_release_items = local_fused_items;
    for (Stage1ArmItem& item : local_release_items) {
      item.action = static_cast<u32>(Stage1ArmAction::release);
    }
    const auto release_started = std::chrono::steady_clock::now();
    for (;;) {
      vec<Stage1ArmResult> release_results;
      const bool transported = arm_local_stage1_items(
        storage_id_, span<const Stage1ArmItem>{local_release_items},
        release_results, config);
      if (transported) {
        const auto disposition =
          memory_node_peer_rpc_detail::classify_stage1_control_response(
            span<const Stage1ArmItem>{local_release_items},
            span<const Stage1ArmResult>{release_results});
        if (disposition == memory_node_peer_rpc_detail::
              Stage1ControlResponseDisposition::resolved) {
          for (const size_t index : local_fused_indices) {
            plans[index].stage1_receipt_released = true;
          }
          break;
        }
        if (disposition == memory_node_peer_rpc_detail::
              Stage1ControlResponseDisposition::malformed) {
          throw std::runtime_error(
            "local fused Stage1 release returned a malformed result");
        }
      }
      if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
        return false;
      }
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::microseconds(100));
    }
    breakdown.storage_owner_stage1_release_wait_ns +=
      elapsed_ns_since(release_started);
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
        .item_index = 0,
        .client_batch_id = operation_ids[index],
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
        .item_index = 0,
        .client_batch_id = operation_ids[index],
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

  // Cleanup-only mutations do not participate in a Stage1 arm group. Fresh
  // fused inserts were already committed by their per-home Execute callback
  // (or by the local ARM completion); the idempotent call here is an audit that
  // no successful fused item escaped its immediate linearization point.
  for (size_t index = 0; index < item_count; ++index) {
    if (plans[index].active &&
        (plans[index].kind == MutationKind::erase ||
         plans[index].fused_stage1)) {
      commit_plan(index);
    }
  }

  // Batch arm by physical home and post every remote batch together. Each
  // physical home remains an independent atomic accepted-backlog transaction:
  // when its ACK arrives, commit exactly that authority subset immediately. No
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
    if (!plan.active || plan.kind == MutationKind::erase ||
        plan.stage1_receipt_released) {
      continue;
    }
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
