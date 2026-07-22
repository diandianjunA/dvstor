#include "memory_node/peer_rpc/detail.hh"
#include "memory_node/peer_rpc/stage1_control_fanout_policy.hh"

namespace authority = memory_node_storage_owner_index_detail;

bool MemoryNode::try_send_peer_stage1_retry_response(
    u32 destination_shard,
    const service::storage_owner::PeerRpcHeader& header,
    span<const byte_t> request) {
  using namespace service::storage_owner;
  const auto request_type = static_cast<PeerRpcType>(header.type);
  const size_t response_bytes = request_type ==
      PeerRpcType::stage1_execute_request
    ? stage1_execute_response_bytes(header.item_count)
    : request_type == PeerRpcType::stage1_arm_request
      ? stage1_arm_response_bytes(header.item_count) : 0;
  if (destination_shard >= num_storage_nodes_ ||
      destination_shard == storage_id_ || response_bytes == 0 ||
      response_bytes > peer_rpc_runtime_.message_bytes) {
    return false;
  }

  u32 slot_id = 0;
  if (!try_acquire_peer_rpc_send_slot(
        destination_shard, PeerRpcSendClass::control, slot_id)) {
    return false;
  }
  byte_t* destination = peer_rpc_runtime_.buffer.get_full_buffer() +
    peer_rpc_async_send_offset(destination_shard, slot_id);
  if (!memory_node_peer_rpc_detail::write_stage1_retry_response(
        storage_id_, header, request,
        span<byte_t>{destination, response_bytes})) {
    release_peer_rpc_send_slot(destination_shard, slot_id);
    return false;
  }
  // This path runs on the CQ progress thread. Use an asynchronous registered
  // send slot; send_peer_rpc_message() intentionally forbids its blocking
  // completion wait here.
  post_peer_rpc_send_slot(destination_shard, slot_id, response_bytes);
  return true;
}

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
      (kind != MutationKind::insert && kind != MutationKind::upsert) ||
      (memory_node_peer_rpc_detail::stage1_execute_uses_fused_arm(item) &&
       !memory_node_peer_rpc_detail::valid_fused_stage1_execute_item(item))) {
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
        prepared.old_ptr.raw_address == item.old_raw &&
        prepared.execute_initial_placement_version ==
          item.initial_placement_version;
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
        result.status = static_cast<u32>(MutationStatus::retry);
        return result;
      }
      return prepared.result;
    }
    // Admission is O(1). Terminal receipts are never aged or scanned: they are
    // true in-flight protocol state and leave only through an explicit release
    // ACK from their authority.
    if (prepared_records.size() >=
        stage1_prepared_results_limit_per_shard_) {
      result.status = static_cast<u32>(MutationStatus::retry);
      return result;
    }
    Stage1PreparedResult reservation;
    reservation.result = result;
    reservation.id = item.id;
    reservation.generation = item.generation;
    reservation.kind = kind;
    reservation.old_ptr = RemotePtr{item.old_raw};
    reservation.execute_initial_placement_version =
      item.initial_placement_version;
    reservation.vector_data = std::move(receipt_vector);
    prepared_records.emplace(key, std::move(reservation));
  }

  const auto seal_failed_reservation = [&]() {
    std::lock_guard<std::mutex> lock(prepared_shard.mutex);
    const auto position = prepared_records.find(key);
    if (position != prepared_records.end() &&
        !position->second.prepared) {
      Stage1PreparedResult& prepared = position->second;
      prepared.result = result;
      prepared.result.status = static_cast<u32>(MutationStatus::failed);
      prepared.prepared = true;
      prepared.aborted = true;
      vec<byte_t>{}.swap(prepared.vector_data);
    }
  };

  vec<element_t> components(VamanaNode::DIM);
  decode_storage_vector_to_float(
    raw_vector, VamanaNode::vector_dtype(), VamanaNode::DIM,
    components.data());
  const vec<RemotePtr> entries = local_centroid_route_entries();
  if (entries.empty()) {
    result.status = static_cast<u32>(MutationStatus::failed);
    seal_failed_reservation();
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
    result.status = static_cast<u32>(MutationStatus::failed);
    seal_failed_reservation();
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

service::storage_owner::Stage1ExecuteResult
MemoryNode::prepare_and_maybe_arm_local_stage1_item(
    u32 authority_shard,
    const service::storage_owner::Stage1ExecuteItem& item,
    const byte_t* raw_vector,
    const Configuration& config,
    InsertBreakdownCounters* breakdown) {
  using namespace service::storage_owner;
  // Keep the per-item helper prepare-only. The Execute request handler may
  // subsequently arm a homogeneous fresh-insert batch in one atomic bounded
  // admission operation; legacy upsert batches retain their standalone
  // cleanup -> arm ordering.
  return prepare_local_stage1_item(
    authority_shard, item, raw_vector, config, breakdown);
}

bool MemoryNode::try_track_stage1_inflight_request(
    const Stage1OperationKey& key) {
  Stage1InflightRequestShard& inflight = stage1_inflight_requests_[
    Stage1OperationKeyHash{}(key) & (kStage1PreparedShardCount - 1)];
  std::lock_guard<std::mutex> lock(inflight.mutex);
  auto position = inflight.counts.find(key);
  if (position == inflight.counts.end()) {
    // Use the same per-shard bound as the semantic receipt table. Queue
    // admission is therefore item-bounded, not merely RPC-bounded.
    if (inflight.counts.size() >=
        stage1_prepared_results_limit_per_shard_) {
      return false;
    }
    position = inflight.counts.emplace(key, 0).first;
  }
  lib_assert(position->second != std::numeric_limits<u32>::max(),
             "Stage1 per-operation in-flight count overflow");
  ++position->second;
  return true;
}

void MemoryNode::finish_stage1_inflight_request(
    const Stage1OperationKey& key) {
  Stage1InflightRequestShard& inflight = stage1_inflight_requests_[
    Stage1OperationKeyHash{}(key) & (kStage1PreparedShardCount - 1)];
  {
    std::lock_guard<std::mutex> lock(inflight.mutex);
    const auto position = inflight.counts.find(key);
    lib_assert(position != inflight.counts.end() && position->second != 0,
               "Stage1 completion lost its per-operation request count");
    if (--position->second == 0) inflight.counts.erase(position);
  }
  inflight.changed.notify_all();
}

bool MemoryNode::stage1_inflight_quiescent(
    const Stage1OperationKey& key) {
  Stage1InflightRequestShard& inflight = stage1_inflight_requests_[
    Stage1OperationKeyHash{}(key) & (kStage1PreparedShardCount - 1)];
  std::lock_guard<std::mutex> lock(inflight.mutex);
  return inflight.counts.find(key) == inflight.counts.end();
}

bool MemoryNode::wait_for_stage1_inflight_quiescence(
    const Stage1OperationKey& key) {
  Stage1InflightRequestShard& inflight = stage1_inflight_requests_[
    Stage1OperationKeyHash{}(key) & (kStage1PreparedShardCount - 1)];
  std::unique_lock<std::mutex> lock(inflight.mutex);
  inflight.changed.wait(lock, [&]() {
    return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
      storage_owner_maintenance_shutdown_.load(std::memory_order_acquire) ||
      inflight.counts.find(key) == inflight.counts.end();
  });
  return inflight.counts.find(key) == inflight.counts.end();
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

  vec<size_t> fused_result_indices;
  vec<Stage1ArmItem> fused_arm_items;
  fused_result_indices.reserve(header.item_count);
  fused_arm_items.reserve(header.item_count);

  for (u32 index = 0; index < header.item_count; ++index) {
    const Stage1ExecuteItem& item = items[index];
    const byte_t* raw_vector = vectors +
      static_cast<size_t>(index) * VamanaNode::vector_bytes();
    output[index] = prepare_and_maybe_arm_local_stage1_item(
      source_shard, item, raw_vector, config);
    if (!memory_node_peer_rpc_detail::stage1_execute_uses_fused_arm(item)) {
      continue;
    }
    if (output[index].status == static_cast<u32>(MutationStatus::retry)) {
      continue;
    }
    if (output[index].status == static_cast<u32>(MutationStatus::ok)) {
      fused_result_indices.push_back(index);
      fused_arm_items.push_back(Stage1ArmItem{
        .token = {
          .source_client = item.source_client,
          .item_index = item.item_index,
          .client_batch_id = item.client_batch_id,
        },
        .target_raw = output[index].target_raw,
        .initial_placement_version = item.initial_placement_version,
        .id = item.id,
        .generation = item.generation,
        .action = static_cast<u32>(Stage1ArmAction::arm),
      });
    }
  }

  if (!fused_arm_items.empty()) {
    // Transport batching does not make sibling mutation tokens atomic. Arm
    // every prepared token now even when another prepare returned transient
    // retry. The authority commits/releases this successful subset and sends a
    // compact retry containing only the unfinished semantic tokens. This
    // prevents one hot token from repeatedly consuming CPU for every ready
    // sibling while preserving the same bounded batch admission for the ready
    // subset itself.
    // Keep the one-lock/one-sequence-range fast path when the complete ready
    // subset fits. Under completion-window pressure the atomic batch returns
    // retry without consuming any task, so fall back to size-one admission.
    // That fallback admits every token for which credit exists and produces a
    // truthful mixed ok/retry response; receipt replay consumes no new credit.
    vec<Stage1ArmResult> arm_results;
    (void)arm_local_stage1_items(
      source_shard, span<const Stage1ArmItem>{fused_arm_items},
      arm_results, config);
    const bool batch_fast_path =
      arm_results.size() == fused_arm_items.size() &&
      std::all_of(
        arm_results.begin(), arm_results.end(), [](const Stage1ArmResult& result) {
          return result.status == static_cast<u32>(MutationStatus::ok);
        });
    if (!batch_fast_path) arm_results.clear();
    arm_results.reserve(fused_arm_items.size());
    vec<Stage1ArmResult> one_result;
    one_result.reserve(1);
    for (size_t slot = 0; slot < fused_result_indices.size(); ++slot) {
      Stage1ExecuteResult& execute = output[fused_result_indices[slot]];
      const Stage1ArmItem& arm = fused_arm_items[slot];
      const Stage1ArmResult* result = nullptr;
      if (batch_fast_path) {
        result = &arm_results[slot];
      } else {
        (void)arm_local_stage1_items(
          source_shard, span<const Stage1ArmItem>{&arm, 1},
          one_result, config);
        if (one_result.size() == 1) result = &one_result.front();
      }
      if (result == nullptr) {
        execute.maintenance_sequence = 0;
        execute.status = static_cast<u32>(MutationStatus::failed);
        continue;
      }
      const bool same_token =
        result->token.source_client == arm.token.source_client &&
        result->token.item_index == arm.token.item_index &&
        result->token.client_batch_id == arm.token.client_batch_id;
      if (!same_token || result->target_raw != arm.target_raw ||
          result->reserved != 0 ||
          result->status > static_cast<u32>(MutationStatus::retry) ||
          (result->status == static_cast<u32>(MutationStatus::ok) &&
           result->maintenance_sequence == 0) ||
          (result->status != static_cast<u32>(MutationStatus::ok) &&
           result->maintenance_sequence != 0)) {
        execute.maintenance_sequence = 0;
        execute.status = static_cast<u32>(MutationStatus::failed);
        continue;
      }
      execute.maintenance_sequence = result->maintenance_sequence;
      execute.status = result->status;
    }
  }

  response_header->status = static_cast<u32>(InsertStatus::ok);
  for (u32 index = 0; index < header.item_count; ++index) {
    if (output[index].status != static_cast<u32>(MutationStatus::ok)) {
      response_header->status = static_cast<u32>(InsertStatus::overloaded);
    } else {
      peer_stage1_items_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  send_peer_rpc_message(source_shard, response.data(), response.size());
  return response_header->status == static_cast<u32>(InsertStatus::ok);
}

bool MemoryNode::handle_peer_stage1_arm_request(
    u32 source_shard,
    const service::storage_owner::PeerRpcHeader& header,
    const service::storage_owner::Stage1ArmItem* items,
    bool release_quiesced,
    const Configuration& config) {
  using namespace service::storage_owner;
  if (items == nullptr || header.item_count == 0 ||
      source_shard >= num_storage_nodes_) {
    return false;
  }

  vec<Stage1ArmResult> results;
  bool processed = true;
  if (release_quiesced) {
    processed = arm_local_stage1_items(
      source_shard, span<const Stage1ArmItem>{items, header.item_count},
      results, config);
  } else {
    // A release is an ordered observation, not work that should occupy a
    // Stage1 executor while an older same-token Execute finishes. Probe each
    // token independently: quiescent receipts can be erased now, while only
    // the still-live tokens return retry. This preserves every per-token QP
    // watermark and prevents one slow Execute from replaying an entire release
    // group or delaying the next compact Execute wave.
    processed = true;
    results.assign(header.item_count, {});
    vec<Stage1ArmResult> one_result;
    one_result.reserve(1);
    for (u32 index = 0; index < header.item_count; ++index) {
      const Stage1OperationKey key{
        .authority_shard = source_shard,
        .source_client = items[index].token.source_client,
        .item_index = items[index].token.item_index,
        .client_batch_id = items[index].token.client_batch_id,
      };
      if (!stage1_inflight_quiescent(key)) {
        results[index].token = items[index].token;
        results[index].target_raw = items[index].target_raw;
        results[index].status = static_cast<u32>(MutationStatus::retry);
        processed = false;
        continue;
      }
      const bool one_processed = arm_local_stage1_items(
        source_shard, span<const Stage1ArmItem>{items + index, 1},
        one_result, config);
      if (one_result.size() != 1) {
        results[index].token = items[index].token;
        results[index].target_raw = items[index].target_raw;
        results[index].status = static_cast<u32>(MutationStatus::failed);
        processed = false;
        continue;
      }
      results[index] = one_result.front();
      processed &= one_processed;
    }
  }
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
    for (const Stage1ArmResult& result : results) {
      if (result.status == static_cast<u32>(MutationStatus::ok)) {
        peer_stage1_items_.fetch_add(1, std::memory_order_relaxed);
      }
    }
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

  const bool arm_batch = std::all_of(
    items.begin(), items.end(), [](const Stage1ArmItem& item) {
      return static_cast<Stage1ArmAction>(item.action) ==
        Stage1ArmAction::arm;
    });
  if (arm_batch) {
    struct ClaimedArm {
      size_t result_index{};
      Stage1OperationKey key;
      StorageOwnerMaintenanceTask task;
    };
    vec<ClaimedArm> claimed;
    claimed.reserve(items.size());
    bool structurally_valid = true;
    bool structural_conflict = false;
    bool transient_unready = false;
    bool whole_batch_claimed = true;

    const auto restore_claims = [&]() {
      for (ClaimedArm& claim : claimed) {
        Stage1PreparedResultShard& prepared_shard =
          stage1_prepared_results_[
            Stage1OperationKeyHash{}(claim.key) &
              (kStage1PreparedShardCount - 1)];
        std::lock_guard<std::mutex> lock(prepared_shard.mutex);
        const auto position = prepared_shard.records.find(claim.key);
        lib_assert(position != prepared_shard.records.end() &&
                     position->second.arming &&
                     !position->second.armed,
                   "atomic Stage1 arm lost its claimed receipt");
        Stage1PreparedResult& prepared = position->second;
        prepared.neighbors = std::move(claim.task.stage1_base_neighbors);
        prepared.beam = std::move(claim.task.stage1_beam);
        prepared.remote_frontier = std::move(
          claim.task.stage1_remote_frontier);
        prepared.backlink_targets = std::move(
          claim.task.stage1_backlink_targets);
        prepared.arming = false;
        prepared.initial_placement_version = 0;
      }
    };

    for (size_t result_index = 0; result_index < items.size();
         ++result_index) {
      const Stage1ArmItem& item = items[result_index];
      Stage1ArmResult& output = results[result_index];
      output.token = item.token;
      output.target_raw = item.target_raw;
      const RemotePtr declared_target{item.target_raw};
      if (!authority::valid_authority_operation(item.token) ||
          item.generation == 0 || item.reserved != 0 ||
          item.initial_placement_version == 0 ||
          !valid_local_storage_node_pointer(declared_target)) {
        structurally_valid = false;
        whole_batch_claimed = false;
        continue;
      }

      const Stage1OperationKey key{
        .authority_shard = authority_shard,
        .source_client = item.token.source_client,
        .item_index = item.token.item_index,
        .client_batch_id = item.token.client_batch_id,
      };
      Stage1PreparedResultShard& prepared_shard =
        stage1_prepared_results_[
          Stage1OperationKeyHash{}(key) &
            (kStage1PreparedShardCount - 1)];
      std::lock_guard<std::mutex> lock(prepared_shard.mutex);
      const auto position = prepared_shard.records.find(key);
      if (position == prepared_shard.records.end()) {
        transient_unready = true;
        whole_batch_claimed = false;
        continue;
      }
      Stage1PreparedResult& prepared = position->second;
      if (prepared.aborted ||
          prepared.result.status != static_cast<u32>(MutationStatus::ok) ||
          prepared.result.target_raw != item.target_raw ||
          prepared.id != item.id ||
          prepared.generation != item.generation ||
          (prepared.execute_initial_placement_version != 0 &&
           prepared.execute_initial_placement_version !=
             item.initial_placement_version)) {
        structural_conflict = true;
        whole_batch_claimed = false;
        continue;
      }
      if (prepared.armed) {
        if (prepared.initial_placement_version ==
              item.initial_placement_version &&
            prepared.maintenance_sequence != 0) {
          output.maintenance_sequence = prepared.maintenance_sequence;
          output.status = static_cast<u32>(MutationStatus::ok);
        } else {
          structural_conflict = true;
          whole_batch_claimed = false;
        }
        continue;
      }
      if (!prepared.prepared || prepared.arming) {
        transient_unready = true;
        whole_batch_claimed = false;
        continue;
      }

      prepared.arming = true;
      prepared.initial_placement_version = item.initial_placement_version;
      StorageOwnerMaintenanceTask task;
      task.kind = StorageOwnerMaintenanceKind::finalize_insert;
      task.id = prepared.id;
      task.generation = prepared.generation;
      task.target = RemotePtr{prepared.result.target_raw};
      task.authority_shard = authority_shard;
      task.source_client = item.token.source_client;
      task.operation_item_index = item.token.item_index;
      task.operation_batch_id = item.token.client_batch_id;
      task.initial_placement_version = item.initial_placement_version;
      task.stage1_base_neighbors = std::move(prepared.neighbors);
      task.stage1_beam = std::move(prepared.beam);
      task.stage1_remote_frontier = std::move(prepared.remote_frontier);
      task.stage1_backlink_targets = std::move(
        prepared.backlink_targets);
      claimed.push_back(ClaimedArm{
        .result_index = result_index,
        .key = key,
        .task = std::move(task),
      });
    }

    // A semantic control RPC is all-or-nothing with respect to new
    // completion credits. Replays consume no credit. If an item is merely
    // unpublished or concurrently arming, restore every claim and return the
    // explicit retry status for the whole atomic home batch. Identity,
    // generation, target, and placement-version conflicts remain failed.
    if (!whole_batch_claimed) {
      restore_claims();
      if (structurally_valid && !structural_conflict && transient_unready) {
        for (Stage1ArmResult& output : results) {
          output.maintenance_sequence = 0;
          output.status = static_cast<u32>(MutationStatus::retry);
        }
        return true;
      }
      return false;
    }

    vec<StorageOwnerMaintenanceTask> tasks;
    tasks.reserve(claimed.size());
    for (ClaimedArm& claim : claimed) {
      tasks.push_back(std::move(claim.task));
    }
    u64 first_sequence = 0;
    if (!tasks.empty()) {
      first_sequence = arm_storage_owner_maintenance_batch(tasks, config);
      if (first_sequence == 0) {
        // Queue/completion admission is try-only. No task was consumed and no
        // sequence was published, so restore every claimed heavy artifact and
        // let the bounded Stage1 control retry path back off before rearming.
        lib_assert(tasks.size() == claimed.size(),
                   "failed atomic Stage1 admission changed batch ownership");
        for (size_t item = 0; item < claimed.size(); ++item) {
          claimed[item].task = std::move(tasks[item]);
        }
        restore_claims();
        for (Stage1ArmResult& output : results) {
          output.maintenance_sequence = 0;
          output.status = static_cast<u32>(MutationStatus::retry);
        }
        return true;
      }
    }

    for (size_t item = 0; item < claimed.size(); ++item) {
      ClaimedArm& claim = claimed[item];
      const u64 sequence = first_sequence + static_cast<u64>(item);
      Stage1PreparedResultShard& prepared_shard =
        stage1_prepared_results_[
          Stage1OperationKeyHash{}(claim.key) &
            (kStage1PreparedShardCount - 1)];
      {
        std::lock_guard<std::mutex> lock(prepared_shard.mutex);
        const auto position = prepared_shard.records.find(claim.key);
        lib_assert(position != prepared_shard.records.end() &&
                     position->second.arming &&
                     !position->second.armed,
                   "atomic Stage1 arm lost its admitted receipt");
        Stage1PreparedResult& prepared = position->second;
        prepared.arming = false;
        prepared.armed = true;
        prepared.maintenance_sequence = sequence;
        prepared.result.maintenance_sequence = sequence;
        clear_heavy_artifact(prepared);
      }
      Stage1ArmResult& output = results[claim.result_index];
      output.maintenance_sequence = sequence;
      output.status = static_cast<u32>(MutationStatus::ok);
    }
    return structurally_valid;
  }

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
         !authority::receipt_release_pointer_addressable(
           declared_target, storage_id_, num_storage_nodes_,
           mn_memory_bytes_, true))) {
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
        // arrives after every same-token Execute retry on the authority's RC
        // QP. The worker only enters this erase path after its per-token
        // quiescence probe succeeds; otherwise it returns an explicit retry.
        // An older Execute therefore cannot recreate the receipt after ACK.
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
          !target_matches) {
        continue;
      }
      if (prepared.arming) {
        output.status = static_cast<u32>(MutationStatus::retry);
        continue;
      }
      if (!terminal_matches) {
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
            output.status = static_cast<u32>(MutationStatus::retry);
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
        if (!prepared.prepared || prepared.arming) {
          output.status = static_cast<u32>(MutationStatus::retry);
          continue;
        }
        if (prepared.armed) {
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
      } else {
        output.status = static_cast<u32>(MutationStatus::retry);
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
        output.status = static_cast<u32>(MutationStatus::retry);
        continue;
      }
      Stage1PreparedResult& prepared = position->second;
      if (prepared.aborted) {
        continue;
      }
      if (prepared.result.status != static_cast<u32>(MutationStatus::ok) ||
          prepared.result.target_raw != item.target_raw ||
          prepared.id != item.id || prepared.generation != item.generation ||
          (prepared.execute_initial_placement_version != 0 &&
           prepared.execute_initial_placement_version !=
             item.initial_placement_version)) {
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
      if (!prepared.prepared || prepared.arming) {
        output.status = static_cast<u32>(MutationStatus::retry);
        continue;
      }
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
      // A full queue/window is transient and arm_storage_owner_maintenance
      // returns ownership instead of waiting on a Stage2 watermark.
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
      output.status = static_cast<u32>(MutationStatus::retry);
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
      position->second.result.maintenance_sequence = maintenance_sequence;
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

bool MemoryNode::release_resolved_local_stage1_receipt(
    const StorageOwnerMaintenanceTask& task,
    const Configuration& config) {
  using namespace service::storage_owner;
  if (task.authority_shard >= num_storage_nodes_ ||
      task.operation_batch_id == 0 || task.generation == 0 ||
      task.target.is_null() || !local_shard(task.target.memory_node())) {
    return false;
  }

  // A remote authority may have an older same-token execute retry that has
  // not reached this process's receive CQ yet.  A local in-flight count of
  // zero cannot prove that transport watermark.  The authority therefore
  // sends a release marker after commit on the same RC QP as every Execute;
  // the peer Stage1 worker probes per-token quiescence and either erases or
  // returns retry without blocking its executor.
  // Stage2 must not race that ordered marker with a locally inferred erase.
  if (task.authority_shard != storage_id_) return true;

  const Stage1OperationKey key{
    .authority_shard = task.authority_shard,
    .source_client = task.source_client,
    .item_index = task.operation_item_index,
    .client_batch_id = task.operation_batch_id,
  };
  if (!wait_for_stage1_inflight_quiescence(key)) return false;

  const Stage1ArmItem release{
    .token = {
      .source_client = task.source_client,
      .item_index = task.operation_item_index,
      .client_batch_id = task.operation_batch_id,
    },
    .target_raw = task.target.raw_address,
    .initial_placement_version = task.initial_placement_version,
    .id = task.id,
    .generation = task.generation,
    .action = static_cast<u32>(Stage1ArmAction::release),
  };
  vec<Stage1ArmResult> results;
  if (!arm_local_stage1_items(
        task.authority_shard, span<const Stage1ArmItem>{&release, 1},
        results, config) ||
      results.size() != 1) {
    return false;
  }
  const Stage1ArmResult& output = results.front();
  return output.token.source_client == release.token.source_client &&
    output.token.item_index == release.token.item_index &&
    output.token.client_batch_id == release.token.client_batch_id &&
    output.target_raw == release.target_raw && output.reserved == 0 &&
    output.status == static_cast<u32>(MutationStatus::ok);
}
