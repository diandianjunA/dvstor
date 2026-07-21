#include "memory_node/peer_rpc/detail.hh"

namespace authority = memory_node_storage_owner_index_detail;

service::storage_owner::Stage1ExecuteResult
MemoryNode::prepare_local_stage1_item(
    u32 authority_shard,
    const service::storage_owner::Stage1ExecuteItem& item,
    const byte_t* raw_vector,
    const Configuration& config,
    InsertBreakdownCounters* breakdown) {
  using namespace service::storage_owner;
  Stage1ExecuteResult result{
    .client_batch_id = item.client_batch_id,
    .source_client = item.source_client,
    .item_index = item.item_index,
  };
  const auto kind = static_cast<MutationKind>(item.kind);
  if (raw_vector == nullptr || authority_shard >= num_storage_nodes_ ||
      item.authority_shard != authority_shard || item.generation == 0 ||
      (kind != MutationKind::insert && kind != MutationKind::upsert)) {
    result.status = static_cast<u32>(MutationStatus::failed);
    return result;
  }
  const Stage1OperationKey key{
    .authority_shard = authority_shard,
    .source_client = item.source_client,
    .item_index = item.item_index,
    .client_batch_id = item.client_batch_id,
  };
  Stage1PreparedResultShard& prepared_shard = stage1_prepared_results_[
    Stage1OperationKeyHash{}(key) & (kStage1PreparedShardCount - 1)];
  auto& prepared_records = prepared_shard.records;
  // Build the only O(DIM) receipt payload before taking the key shard. A
  // successful new claim moves it into the map in O(1); duplicate validation
  // still compares against the caller-owned wire bytes without allocating
  // while the shard is locked.
  vec<byte_t> receipt_vector(
    raw_vector, raw_vector + VamanaNode::vector_bytes());

  {
    std::lock_guard<std::mutex> lock(prepared_shard.mutex);
    const auto existing = prepared_records.find(key);
    if (existing != prepared_records.end()) {
      const Stage1PreparedResult& prepared = existing->second;
      if (prepared.aborted) {
        // Abort remains terminal until the authority explicitly releases this
        // receipt. A delayed execute that was dequeued before abort therefore
        // cannot rebuild an orphaned provisional record.
        result.status = static_cast<u32>(MutationStatus::failed);
        return result;
      }
      const bool same_operation = prepared.id == item.id &&
        prepared.generation == item.generation && prepared.kind == kind &&
        prepared.old_ptr.raw_address == item.old_raw;
      if (!same_operation) {
        result.status = static_cast<u32>(MutationStatus::failed);
        return result;
      }
      if (prepared.armed) {
        // The semantic token and fixed identity bind every retry to the input
        // already handed to Stage2. The exact vector bytes are intentionally
        // released at arm ACK; committed public replays terminate at the
        // authority and never execute Stage1 again.
        return prepared.result;
      }
      if (prepared.vector_data.size() != VamanaNode::vector_bytes() ||
          std::memcmp(prepared.vector_data.data(), raw_vector,
                      VamanaNode::vector_bytes()) != 0) {
        result.status = static_cast<u32>(MutationStatus::failed);
        return result;
      }
      if (!prepared.prepared) {
        // Another transport request for this semantic operation is already
        // computing. Return bounded backpressure; the authority retries the
        // same token and receives the cached result once it is ready.
        result.status = static_cast<u32>(MutationStatus::failed);
        return result;
      }
      return prepared.result;
    }
    // Admission is O(1). Terminal receipts are never aged or scanned: they are
    // true in-flight protocol state and leave only through an explicit release
    // ACK from their authority.
    if (prepared_records.size() >=
        stage1_prepared_results_limit_per_shard_) {
      result.status = static_cast<u32>(MutationStatus::failed);
      return result;
    }
    Stage1PreparedResult reservation;
    reservation.result = result;
    reservation.id = item.id;
    reservation.generation = item.generation;
    reservation.kind = kind;
    reservation.old_ptr = RemotePtr{item.old_raw};
    reservation.vector_data = std::move(receipt_vector);
    prepared_records.emplace(key, std::move(reservation));
  }

  const auto erase_reservation = [&]() {
    std::lock_guard<std::mutex> lock(prepared_shard.mutex);
    const auto position = prepared_records.find(key);
    if (position != prepared_records.end() &&
        !position->second.prepared) {
      prepared_records.erase(position);
    }
  };

  vec<element_t> components(VamanaNode::DIM);
  decode_storage_vector_to_float(
    raw_vector, VamanaNode::vector_dtype(), VamanaNode::DIM,
    components.data());
  const vec<RemotePtr> entries = local_centroid_route_entries();
  if (entries.empty()) {
    erase_reservation();
    result.status = static_cast<u32>(MutationStatus::failed);
    return result;
  }

  vec<BeamEntry> stage1_beam;
  vec<RemotePtr> remote_frontier;
  const auto search_started = std::chrono::steady_clock::now();
  vec<RemotePtr> candidates = partition_local_search_candidates(
    span<const element_t>{components}, entries, config, breakdown,
    raw_vector, &stage1_beam, &remote_frontier);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_ns += elapsed_ns_since(search_started);
  }
  hashset_t<RemotePtr> skip;
  if (kind == MutationKind::upsert && item.old_raw != 0) {
    // The previous generation is tombstoned before authority commit. It must
    // not become a provisional backlink target for its replacement.
    skip.insert(RemotePtr{item.old_raw});
  }
  const auto prune_started = std::chrono::steady_clock::now();
  vec<RemotePtr> neighbors = robust_prune_cpu(
    raw_vector, VamanaNode::vector_dtype(), candidates, skip, config,
    breakdown, config.R);
  if (breakdown != nullptr) {
    breakdown->storage_owner_prune_ns += elapsed_ns_since(prune_started);
  }
  const auto allocate_started = std::chrono::steady_clock::now();
  const RemotePtr target = allocate_local_node();
  if (breakdown != nullptr) {
    breakdown->storage_owner_allocate_node_ns +=
      elapsed_ns_since(allocate_started);
  }
  const auto write_started = std::chrono::steady_clock::now();
  write_new_node_on_shard(
    target, item.id, span<const element_t>{components}, neighbors,
    item.generation, true);
  if (breakdown != nullptr) {
    breakdown->storage_owner_write_node_ns += elapsed_ns_since(write_started);
  }
  vec<RemotePtr> backlink_targets = install_local_provisional_backlinks(
    target, span<const RemotePtr>{neighbors});
  if (backlink_targets.empty()) {
    const u64 retirement_sequence =
      begin_storage_owner_maintenance_sequence(1);
    (void)mark_node_deleted(target, item.generation);
    retire_local_dynamic_node(target, retirement_sequence);
    complete_storage_owner_maintenance_sequence(retirement_sequence);
    erase_reservation();
    result.status = static_cast<u32>(MutationStatus::failed);
    return result;
  }

  result.target_raw = target.raw_address;
  result.status = static_cast<u32>(MutationStatus::ok);
  {
    std::lock_guard<std::mutex> lock(prepared_shard.mutex);
    const auto position = prepared_records.find(key);
    lib_assert(position != prepared_records.end() &&
                 !position->second.prepared,
               "Stage1 reservation disappeared while preparing a node");
    Stage1PreparedResult& prepared = position->second;
    prepared.result = result;
    prepared.neighbors = std::move(neighbors);
    prepared.beam = std::move(stage1_beam);
    prepared.remote_frontier = std::move(remote_frontier);
    prepared.backlink_targets = std::move(backlink_targets);
    prepared.prepared = true;
  }
  return result;
}

bool MemoryNode::handle_peer_stage1_execute_request(
    u32 source_shard,
    const service::storage_owner::PeerRpcHeader& header,
    const byte_t* payload,
    const Configuration& config) {
  using namespace service::storage_owner;
  if (payload == nullptr || header.item_count == 0 ||
      header.item_count > config.storage_owner_batch_max ||
      source_shard >= num_storage_nodes_) {
    return false;
  }

  const size_t response_bytes =
    stage1_execute_response_bytes(header.item_count);
  vec<byte_t> response(response_bytes, 0);
  auto* response_header =
    reinterpret_cast<PeerRpcHeader*>(response.data());
  response_header->magic = kPeerRpcMagic;
  response_header->version = kPeerRpcVersion;
  response_header->type = static_cast<u32>(
    PeerRpcType::stage1_execute_response);
  response_header->source_shard = storage_id_;
  response_header->item_count = header.item_count;
  response_header->request_id = header.request_id;
  response_header->status = static_cast<u32>(InsertStatus::ok);
  auto* output = stage1_execute_results(response.data());
  const auto* items = stage1_execute_items(payload);
  const byte_t* vectors = stage1_execute_vectors(
    payload, header.item_count);

  for (u32 index = 0; index < header.item_count; ++index) {
    const Stage1ExecuteItem& item = items[index];
    const byte_t* raw_vector = vectors +
      static_cast<size_t>(index) * VamanaNode::vector_bytes();
    output[index] = prepare_local_stage1_item(
      source_shard, item, raw_vector, config);
    if (output[index].status != static_cast<u32>(MutationStatus::ok)) {
      response_header->status = static_cast<u32>(InsertStatus::overloaded);
    }
  }

  send_peer_rpc_message(source_shard, response.data(), response.size());
  return response_header->status == static_cast<u32>(InsertStatus::ok);
}

bool MemoryNode::handle_peer_stage1_arm_request(
    u32 source_shard,
    const service::storage_owner::PeerRpcHeader& header,
    const service::storage_owner::Stage1ArmItem* items,
    const Configuration& config) {
  using namespace service::storage_owner;
  if (items == nullptr || header.item_count == 0 ||
      source_shard >= num_storage_nodes_) {
    return false;
  }

  vec<Stage1ArmResult> results;
  const bool processed = arm_local_stage1_items(
    source_shard, span<const Stage1ArmItem>{items, header.item_count},
    results, config);
  const size_t bytes = stage1_arm_response_bytes(header.item_count);
  vec<byte_t> response(bytes, 0);
  auto* response_header = reinterpret_cast<PeerRpcHeader*>(
    response.data());
  response_header->magic = kPeerRpcMagic;
  response_header->version = kPeerRpcVersion;
  response_header->type = static_cast<u32>(
    PeerRpcType::stage1_arm_response);
  response_header->source_shard = storage_id_;
  response_header->item_count = header.item_count;
  response_header->request_id = header.request_id;
  // Retryable item failures are carried per item. Keeping the envelope
  // consumable lets the authority retry only the same semantic arm token.
  response_header->status = static_cast<u32>(InsertStatus::ok);
  if (results.size() == header.item_count) {
    std::memcpy(stage1_arm_results(response.data()), results.data(),
                results.size() * sizeof(results[0]));
  }
  send_peer_rpc_message(source_shard, response.data(), response.size());
  return processed;
}

bool MemoryNode::arm_local_stage1_items(
    u32 authority_shard,
    span<const service::storage_owner::Stage1ArmItem> items,
    vec<service::storage_owner::Stage1ArmResult>& results,
    const Configuration& config) {
  using namespace service::storage_owner;
  results.assign(items.size(), {});
  if (authority_shard >= num_storage_nodes_ || items.empty()) return false;

  const auto clear_heavy_artifact = [](Stage1PreparedResult& prepared) {
    vec<byte_t>{}.swap(prepared.vector_data);
    vec<RemotePtr>{}.swap(prepared.neighbors);
    vec<BeamEntry>{}.swap(prepared.beam);
    vec<RemotePtr>{}.swap(prepared.remote_frontier);
    vec<RemotePtr>{}.swap(prepared.backlink_targets);
  };

  bool structurally_valid = true;
  for (size_t result_index = 0; result_index < items.size();
       ++result_index) {
    const Stage1ArmItem& item = items[result_index];
    Stage1ArmResult& output = results[result_index];
    output.token = item.token;
    output.target_raw = item.target_raw;
    const auto action = static_cast<Stage1ArmAction>(item.action);
    const RemotePtr declared_target{item.target_raw};
    if (!authority::valid_authority_operation(item.token) ||
        item.generation == 0 || item.reserved != 0 ||
        (action != Stage1ArmAction::arm &&
         action != Stage1ArmAction::abort &&
         action != Stage1ArmAction::release) ||
        (action == Stage1ArmAction::arm &&
         (item.initial_placement_version == 0 ||
          !valid_local_storage_node_pointer(declared_target))) ||
        (action == Stage1ArmAction::abort &&
         (item.initial_placement_version != 0 ||
          (!declared_target.is_null() &&
           !valid_local_storage_node_pointer(declared_target)))) ||
        (action == Stage1ArmAction::release &&
         !declared_target.is_null() &&
         !valid_local_storage_node_pointer(declared_target))) {
      structurally_valid = false;
      continue;
    }
    const Stage1OperationKey key{
      .authority_shard = authority_shard,
      .source_client = item.token.source_client,
      .item_index = item.token.item_index,
      .client_batch_id = item.token.client_batch_id,
    };
    Stage1PreparedResultShard& prepared_shard = stage1_prepared_results_[
      Stage1OperationKeyHash{}(key) & (kStage1PreparedShardCount - 1)];
    auto& prepared_records = prepared_shard.records;
    if (action == Stage1ArmAction::release) {
      std::lock_guard<std::mutex> lock(prepared_shard.mutex);
      const auto position = prepared_records.find(key);
      if (position == prepared_records.end()) {
        // A lost release response is retried with the same token. Missing is
        // therefore the successful idempotent postcondition. Remote release
        // tasks first cross the per-authority worker-completion barrier, so an
        // earlier queued execute cannot appear after this ACK.
        output.status = static_cast<u32>(MutationStatus::ok);
        continue;
      }
      const Stage1PreparedResult& prepared = position->second;
      const bool target_matches = declared_target.is_null() ||
        prepared.result.target_raw == item.target_raw;
      const bool terminal_matches =
        (prepared.armed && item.initial_placement_version != 0 &&
         prepared.initial_placement_version ==
           item.initial_placement_version) ||
        (prepared.aborted && item.initial_placement_version == 0);
      if (prepared.id != item.id || prepared.generation != item.generation ||
          prepared.arming || !target_matches || !terminal_matches) {
        continue;
      }
      output.target_raw = prepared.result.target_raw;
      output.maintenance_sequence = prepared.maintenance_sequence;
      prepared_records.erase(position);
      output.status = static_cast<u32>(MutationStatus::ok);
      continue;
    }
    if (action == Stage1ArmAction::abort) {
      RemotePtr target;
      u32 generation = 0;
      vec<RemotePtr> backlink_targets;
      {
        std::lock_guard<std::mutex> lock(prepared_shard.mutex);
        const auto position = prepared_records.find(key);
        if (position == prepared_records.end()) {
          // Persist a compact abort fence even when execute failed before
          // materializing an artifact. Parallel workers may finish messages in
          // a different order than their RC delivery; the fence rejects every
          // earlier execute until a release crosses the completion barrier.
          if (prepared_records.size() >=
              stage1_prepared_results_limit_per_shard_) {
            continue;
          }
          Stage1PreparedResult receipt;
          receipt.result.client_batch_id = item.token.client_batch_id;
          receipt.result.source_client = item.token.source_client;
          receipt.result.item_index = item.token.item_index;
          receipt.result.target_raw = item.target_raw;
          receipt.result.status = static_cast<u32>(MutationStatus::failed);
          receipt.id = item.id;
          receipt.generation = item.generation;
          receipt.prepared = true;
          receipt.aborted = true;
          prepared_records.emplace(key, std::move(receipt));
          output.status = static_cast<u32>(MutationStatus::ok);
          continue;
        }
        Stage1PreparedResult& prepared = position->second;
        const bool same_artifact = prepared.id == item.id &&
          prepared.generation == item.generation &&
          (declared_target.is_null() ||
           prepared.result.target_raw == item.target_raw);
        if (!same_artifact) {
          continue;
        }
        output.target_raw = prepared.result.target_raw;
        if (prepared.aborted) {
          output.maintenance_sequence = prepared.maintenance_sequence;
          output.status = static_cast<u32>(MutationStatus::ok);
          continue;
        }
        if (!prepared.prepared || prepared.arming || prepared.armed) {
          continue;
        }
        prepared.arming = true;
        target = RemotePtr{prepared.result.target_raw};
        generation = prepared.generation;
        backlink_targets = prepared.backlink_targets;
      }
      const bool removed = remove_local_provisional_backlinks(
        target, span<const RemotePtr>{backlink_targets});
      u64 retirement_sequence = 0;
      if (removed) {
        // This synchronous path never hands the ticket to a future task. It
        // publishes DELETED, registers the RCU retirement, and completes the
        // fence before acknowledging abort.
        retirement_sequence = begin_storage_owner_maintenance_sequence(1);
        bool deleted = mark_node_deleted(target, generation);
        if (!deleted) {
          NodeSnapshot snapshot;
          deleted = read_node_snapshot(target, snapshot) &&
            snapshot.id == item.id && snapshot.generation == generation &&
            snapshot.deleted;
        }
        if (deleted) {
          retire_local_dynamic_node(target, retirement_sequence);
        }
        complete_storage_owner_maintenance_sequence(retirement_sequence);
        if (!deleted) retirement_sequence = 0;
      }
      {
        std::lock_guard<std::mutex> lock(prepared_shard.mutex);
        const auto position = prepared_records.find(key);
        if (position != prepared_records.end()) {
          if (removed) {
            Stage1PreparedResult& prepared = position->second;
            prepared.arming = false;
            if (retirement_sequence != 0) {
              prepared.result.status = static_cast<u32>(
                MutationStatus::failed);
              prepared.maintenance_sequence = retirement_sequence;
              prepared.aborted = true;
              clear_heavy_artifact(prepared);
            }
          } else {
            position->second.arming = false;
          }
        }
      }
      if (retirement_sequence != 0) {
        output.target_raw = target.raw_address;
        output.maintenance_sequence = retirement_sequence;
        output.status = static_cast<u32>(MutationStatus::ok);
      }
      continue;
    }

    StorageOwnerMaintenanceTask task;
    {
      std::lock_guard<std::mutex> lock(prepared_shard.mutex);
      const auto position = prepared_records.find(key);
      if (position == prepared_records.end()) {
        // A physical snapshot cannot prove that a runnable task owns a
        // completion sequence. Only the bounded arm receipt may replay ACK.
        continue;
      }
      Stage1PreparedResult& prepared = position->second;
      if (prepared.aborted) {
        continue;
      }
      if (prepared.result.status != static_cast<u32>(MutationStatus::ok) ||
          prepared.result.target_raw != item.target_raw ||
          prepared.id != item.id || prepared.generation != item.generation) {
        continue;
      }
      if (prepared.armed) {
        if (prepared.initial_placement_version ==
              item.initial_placement_version &&
            prepared.maintenance_sequence != 0) {
          output.maintenance_sequence = prepared.maintenance_sequence;
          output.status = static_cast<u32>(MutationStatus::ok);
        }
        continue;
      }
      // Do not hold the operation-table lock while the bounded maintenance
      // queue applies backpressure. This keeps unrelated Stage1 executes and
      // retries moving under sustained update load.
      if (prepared.arming) continue;
      prepared.arming = true;
      prepared.initial_placement_version =
        item.initial_placement_version;
      task.kind = StorageOwnerMaintenanceKind::finalize_insert;
      task.id = prepared.id;
      task.generation = prepared.generation;
      task.target = RemotePtr{prepared.result.target_raw};
      task.authority_shard = authority_shard;
      task.source_client = item.token.source_client;
      task.operation_item_index = item.token.item_index;
      task.operation_batch_id = item.token.client_batch_id;
      task.initial_placement_version = item.initial_placement_version;
      // arming is the exclusive ownership bit. Move the O(L + R) continuation
      // artifact into the task so the key shard is held only for hash/field
      // updates, never for a deep vector copy.
      task.stage1_base_neighbors = std::move(prepared.neighbors);
      task.stage1_beam = std::move(prepared.beam);
      task.stage1_remote_frontier = std::move(prepared.remote_frontier);
      task.stage1_backlink_targets = std::move(
        prepared.backlink_targets);
    }
    const u64 maintenance_sequence =
      arm_storage_owner_maintenance(std::move(task), config);
    if (maintenance_sequence == 0) {
      std::lock_guard<std::mutex> lock(prepared_shard.mutex);
      const auto position = prepared_records.find(key);
      if (position != prepared_records.end()) {
        Stage1PreparedResult& prepared = position->second;
        lib_assert(prepared.arming,
                   "failed Stage1 arm lost exclusive artifact ownership");
        prepared.neighbors = std::move(task.stage1_base_neighbors);
        prepared.beam = std::move(task.stage1_beam);
        prepared.remote_frontier = std::move(task.stage1_remote_frontier);
        prepared.backlink_targets = std::move(
          task.stage1_backlink_targets);
        prepared.arming = false;
        prepared.initial_placement_version = 0;
      }
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(prepared_shard.mutex);
      const auto position = prepared_records.find(key);
      lib_assert(position != prepared_records.end(),
                 "Stage1 artifact disappeared during arm");
      position->second.arming = false;
      position->second.armed = true;
      position->second.maintenance_sequence = maintenance_sequence;
      // The maintenance task now owns the continuation artifact. Retain only
      // the compact semantic replay receipt until the authority explicitly
      // releases it; no vector or O(L + R) search state is retained.
      clear_heavy_artifact(position->second);
    }
    output.maintenance_sequence = maintenance_sequence;
    output.status = static_cast<u32>(MutationStatus::ok);
  }
  return structurally_valid;
}
