#include "memory_node/peer_rpc/detail.hh"
#include "memory_node/peer_rpc/stage1_control_fanout_policy.hh"
#include "memory_node/storage_owner_index/stage1_reachability_policy.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

namespace authority = memory_node_storage_owner_index_detail;

bool MemoryNode::handle_peer_stage2_expand_score_request(
    u32 source_shard,
    const service::storage_owner::PeerRpcHeader& header,
    const byte_t* payload,
    const Configuration& config) {
  using namespace service::storage_owner;
  using memory_node_storage_owner_index_detail::StableNodeSnapshotState;
  using memory_node_storage_owner_index_detail::classify_stable_node_snapshot;
  if (payload == nullptr || source_shard >= num_storage_nodes_ ||
      source_shard == storage_id_ || header.item_count == 0 ||
      header.item_count > config.storage_owner_batch_max) {
    return false;
  }

  const size_t response_capacity =
    stage2_expand_score_response_bytes(header.item_count);
  if (response_capacity > peer_rpc_runtime_.message_bytes) return false;
  // The asynchronous sender recycles bounded high-water buffers after copying
  // them into registered send slots. Stage2 emits many medium-sized responses;
  // avoiding a malloc/free pair for each graph wave is important at high QPS.
  vec<byte_t> response = acquire_peer_graph_response_buffer(response_capacity);
  auto* response_header = reinterpret_cast<PeerRpcHeader*>(response.data());
  response_header->magic = kPeerRpcMagic;
  response_header->version = kPeerRpcVersion;
  response_header->type = static_cast<u32>(
    PeerRpcType::stage2_expand_score_response);
  response_header->source_shard = storage_id_;
  response_header->item_count = header.item_count;
  response_header->request_id = header.request_id;
  response_header->status = static_cast<u32>(InsertStatus::ok);
  response_header->reserved = 0;

  const auto* items = stage2_expand_score_items(payload);
  const byte_t* queries = stage2_expand_score_queries(
    payload, header.item_count);
  auto* results = stage2_expand_score_results(response.data());
  auto* neighbors = stage2_expand_score_neighbors(
    response.data(), header.item_count);
  const size_t neighbor_stride = VamanaNode::graph_entry_capacity();
  const VectorDType dtype = VamanaNode::vector_dtype();
  GraphAdjacency adjacency;
  u32 compact_neighbor_count = 0;

  // Home expansion needs only a stable vector, not an owning NodeSnapshot.
  // Score directly from the local registered node under the same header /
  // incarnation seqlock used by read_node_snapshot(). This removes one vector
  // allocation and 128-byte copy per neighbor while preserving the exact
  // stable/terminal/retryable classification.
  const auto score_local_neighbor = [&] (
      RemotePtr neighbor, const byte_t* query,
      Stage2ExpandScoreNeighbor& output) {
    constexpr u32 kMaxReadAttempts = 3;
    const byte_t* node = local_node_ptr(neighbor);
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(neighbor);
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0 ||
          VamanaNode::header_incarnation(before) !=
            neighbor.incarnation()) {
        std::this_thread::yield();
        continue;
      }
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        node + VamanaNode::offset_slot_incarnation());
      const distance_t distance = distance_between_vectors(
        query, dtype, node + VamanaNode::offset_vector(), dtype, config);
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(neighbor);
      const StableNodeSnapshotState state = classify_stable_node_snapshot(
        neighbor, before, after, slot_incarnation);
      if (state == StableNodeSnapshotState::stable) {
        output.distance = distance;
        output.disposition = static_cast<u32>(
          Stage2HomeDisposition::stable);
        return;
      }
      if (state == StableNodeSnapshotState::terminal) {
        output.disposition = static_cast<u32>(
          Stage2HomeDisposition::terminal);
        return;
      }
      std::this_thread::yield();
    }
    output.disposition = static_cast<u32>(
      Stage2HomeDisposition::retryable);
  };

  for (u32 item_index = 0; item_index < header.item_count; ++item_index) {
    const Stage2ExpandScoreItem& item = items[item_index];
    Stage2ExpandScoreResult& result = results[item_index];
    result = {};
    result.pointer_raw = item.pointer_raw;
    result.generation = item.generation;
    result.search_index = item.search_index;
    result.neighbor_offset = compact_neighbor_count;
    result.operation = item.operation;
    const RemotePtr pointer{item.pointer_raw};
    if (item.operation > static_cast<u32>(Stage2HomeOperation::score_only)) {
      result.disposition = static_cast<u32>(
        Stage2HomeDisposition::terminal);
      continue;
    }
    if (!valid_local_storage_node_pointer(pointer)) {
      result.disposition = static_cast<u32>(
        Stage2HomeDisposition::terminal);
      continue;
    }

    const byte_t* query = queries +
      static_cast<size_t>(item_index) * VamanaNode::vector_bytes();
    if (item.operation == static_cast<u32>(Stage2HomeOperation::score_only)) {
      Stage2ExpandScoreNeighbor score{};
      score_local_neighbor(pointer, query, score);
      result.distance = score.distance;
      result.disposition = score.disposition;
      continue;
    }

    adjacency.stable.clear();
    adjacency.provisional.clear();
    if (!read_graph_adjacency(pointer, adjacency)) {
      const u64 before = load_local_node_header_acquire(pointer);
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        local_node_ptr(pointer) + VamanaNode::offset_slot_incarnation());
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(pointer);
      const StableNodeSnapshotState state = classify_stable_node_snapshot(
        pointer, before, after, slot_incarnation);
      result.disposition = static_cast<u32>(
        state == StableNodeSnapshotState::terminal
          ? Stage2HomeDisposition::terminal
          : Stage2HomeDisposition::retryable);
      continue;
    }

    result.disposition = static_cast<u32>(Stage2HomeDisposition::stable);
    if (adjacency.deleted) continue;
    const size_t total_neighbors = std::min(
      neighbor_stride, adjacency.stable.size() +
        adjacency.provisional.size());
    result.neighbor_count = static_cast<u32>(total_neighbors);
    size_t output_index = 0;
    const auto emit_neighbor = [&](RemotePtr neighbor) {
      if (output_index >= total_neighbors) return;
      Stage2ExpandScoreNeighbor& output =
        neighbors[static_cast<size_t>(compact_neighbor_count) +
                  output_index++];
      output = {};
      output.pointer_raw = neighbor.raw_address;
      if (!storage_node_pointer_addressable(neighbor)) {
        output.disposition = static_cast<u32>(
          Stage2HomeDisposition::terminal);
        return;
      }
      if (!local_shard(neighbor.memory_node())) {
        output.disposition = static_cast<u32>(
          Stage2HomeDisposition::unscored);
        return;
      }
      score_local_neighbor(neighbor, query, output);
    };
    for (RemotePtr neighbor : adjacency.stable) emit_neighbor(neighbor);
    for (RemotePtr neighbor : adjacency.provisional) emit_neighbor(neighbor);
    compact_neighbor_count += result.neighbor_count;
  }

  response.resize(stage2_expand_score_response_bytes(
    header.item_count, compact_neighbor_count));

  PeerReverseUpdateResponse outbound;
  outbound.destination_shard = source_shard;
  outbound.header = *response_header;
  outbound.payload = std::move(response);
  outbound.graph_response = true;
  outbound.queued_at = std::chrono::steady_clock::now();
  const bool queued = try_enqueue_peer_reverse_update_response(
    std::move(outbound));
  if (!queued) {
    peer_stage2_home_response_queue_drops_.fetch_add(
      1, std::memory_order_relaxed);
  }
  return queued;
}

bool MemoryNode::handle_peer_stage2_score_many_request(
    u32 source_shard,
    const service::storage_owner::PeerRpcHeader& header,
    const byte_t* payload,
    const Configuration& config) {
  using namespace service::storage_owner;
  using memory_node_storage_owner_index_detail::StableNodeSnapshotState;
  using memory_node_storage_owner_index_detail::classify_stable_node_snapshot;
  if (payload == nullptr || source_shard >= num_storage_nodes_ ||
      source_shard == storage_id_ || header.item_count == 0 ||
      header.item_count > std::max<u32>(
        1, config.storage_owner_search_snapshot_batch)) {
    return false;
  }
  const auto* own_header = stage2_score_many_header(payload);
  if (own_header->query_count == 0 ||
      own_header->query_count > header.item_count ||
      own_header->reserved != 0) {
    return false;
  }

  const size_t response_capacity =
    stage2_score_many_response_bytes(header.item_count);
  if (response_capacity > peer_rpc_runtime_.message_bytes) return false;
  vec<byte_t> response = acquire_peer_graph_response_buffer(response_capacity);
  auto* response_header = reinterpret_cast<PeerRpcHeader*>(response.data());
  response_header->magic = kPeerRpcMagic;
  response_header->version = kPeerRpcVersion;
  response_header->type = static_cast<u32>(
    PeerRpcType::stage2_score_many_response);
  response_header->source_shard = storage_id_;
  response_header->item_count = header.item_count;
  response_header->request_id = header.request_id;
  response_header->status = static_cast<u32>(InsertStatus::ok);
  response_header->reserved = 0;

  const auto* items = stage2_score_many_items(payload);
  const byte_t* queries = stage2_score_many_queries(
    payload, header.item_count);
  auto* results = stage2_score_many_results(response.data());
  const VectorDType dtype = VamanaNode::vector_dtype();
  constexpr u32 kMaxReadAttempts = 3;
  for (u32 item_index = 0; item_index < header.item_count; ++item_index) {
    const Stage2ScoreManyItem& item = items[item_index];
    Stage2ScoreManyResult& result = results[item_index];
    result = {};
    result.pointer_raw = item.pointer_raw;
    result.generation = item.generation;
    result.search_index = item.search_index;
    result.disposition = static_cast<u32>(
      Stage2HomeDisposition::terminal);
    const RemotePtr pointer{item.pointer_raw};
    if (item.query_index >= own_header->query_count ||
        !valid_local_storage_node_pointer(pointer)) {
      continue;
    }

    const byte_t* query = queries +
      static_cast<size_t>(item.query_index) * VamanaNode::vector_bytes();
    const byte_t* node = local_node_ptr(pointer);
    bool classified = false;
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(pointer);
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0 ||
          VamanaNode::header_incarnation(before) != pointer.incarnation()) {
        std::this_thread::yield();
        continue;
      }
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        node + VamanaNode::offset_slot_incarnation());
      const distance_t distance = distance_between_vectors(
        query, dtype, node + VamanaNode::offset_vector(), dtype, config);
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(pointer);
      const StableNodeSnapshotState state = classify_stable_node_snapshot(
        pointer, before, after, slot_incarnation);
      if (state == StableNodeSnapshotState::stable) {
        result.distance = distance;
        result.disposition = static_cast<u32>(
          Stage2HomeDisposition::stable);
        classified = true;
        break;
      }
      if (state == StableNodeSnapshotState::terminal) {
        classified = true;
        break;
      }
      std::this_thread::yield();
    }
    if (!classified) {
      result.disposition = static_cast<u32>(
        Stage2HomeDisposition::retryable);
    }
  }

  PeerReverseUpdateResponse outbound;
  outbound.destination_shard = source_shard;
  outbound.header = *response_header;
  outbound.payload = std::move(response);
  outbound.graph_response = true;
  outbound.queued_at = std::chrono::steady_clock::now();
  const bool queued = try_enqueue_peer_reverse_update_response(
    std::move(outbound));
  if (!queued) {
    peer_stage2_home_response_queue_drops_.fetch_add(
      1, std::memory_order_relaxed);
  }
  return queued;
}

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
  const auto release_retryable_reservation = [&]() {
    std::lock_guard<std::mutex> lock(prepared_shard.mutex);
    const auto position = prepared_records.find(key);
    lib_assert(position != prepared_records.end() &&
                 !position->second.prepared &&
                 position->second.id == item.id &&
                 position->second.generation == item.generation,
               "retryable Stage1 attempt lost its private reservation");
    prepared_records.erase(position);
    result.status = static_cast<u32>(MutationStatus::retry);
  };

  // From this point onward this semantic token owns a new physical-home
  // execution. Duplicate/retry receipts returned above must not be charged a
  // second time. Keep aggregate timing out of the completion packet so remote
  // and locally coordinated tokens are measured uniformly.
  const auto physical_stage1_started = std::chrono::steady_clock::now();
  u64 physical_search_ns = 0;
  u64 physical_prune_ns = 0;
  u64 physical_allocate_write_ns = 0;
  u64 physical_backlink_ns = 0;
  const auto record_physical_stage1 = [&](size_t candidate_count,
                                          size_t frontier_count,
                                          size_t neighbor_count) {
    physical_stage1_items_.fetch_add(1, std::memory_order_relaxed);
    physical_stage1_total_ns_.fetch_add(
      elapsed_ns_since(physical_stage1_started), std::memory_order_relaxed);
    physical_stage1_search_ns_.fetch_add(
      physical_search_ns, std::memory_order_relaxed);
    physical_stage1_prune_ns_.fetch_add(
      physical_prune_ns, std::memory_order_relaxed);
    physical_stage1_allocate_write_ns_.fetch_add(
      physical_allocate_write_ns, std::memory_order_relaxed);
    physical_stage1_backlink_ns_.fetch_add(
      physical_backlink_ns, std::memory_order_relaxed);
    physical_stage1_candidates_.fetch_add(
      candidate_count, std::memory_order_relaxed);
    physical_stage1_remote_frontier_items_.fetch_add(
      frontier_count, std::memory_order_relaxed);
    physical_stage1_neighbors_.fetch_add(
      neighbor_count, std::memory_order_relaxed);
  };

  vec<element_t> components(VamanaNode::DIM);
  decode_storage_vector_to_float(
    raw_vector, VamanaNode::vector_dtype(), VamanaNode::DIM,
    components.data());
  vec<BeamEntry> stage1_beam;
  vec<RemotePtr> remote_frontier;
  hashset_t<RemotePtr> skip;
  vec<RemotePtr> bridge_exclusions;
  if (kind == MutationKind::upsert && item.old_raw != 0) {
    // The previous generation is tombstoned before authority commit. It must
    // not become a provisional backlink target for its replacement.
    const RemotePtr old_pointer{item.old_raw};
    skip.insert(old_pointer);
    bridge_exclusions.push_back(old_pointer);
  }
  vec<RemotePtr> candidates;
  vec<RemotePtr> neighbors;
  vec<RemotePtr> bridge_targets;
  const auto refresh_stage1_candidates = [&]() {
    const vec<RemotePtr> entries = local_centroid_route_entries();
    if (entries.empty()) {
      candidates.clear();
      neighbors.clear();
      bridge_targets.clear();
      stage1_beam.clear();
      remote_frontier.clear();
      return false;
    }
    const auto search_started = std::chrono::steady_clock::now();
    candidates = partition_local_search_candidates(
      span<const element_t>{components}, entries, config, breakdown,
      raw_vector, &stage1_beam, &remote_frontier);
    const u64 search_ns = elapsed_ns_since(search_started);
    physical_search_ns += search_ns;
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_ns += search_ns;
    }
    const auto prune_started = std::chrono::steady_clock::now();
    neighbors = robust_prune_cpu(
      raw_vector, VamanaNode::vector_dtype(), candidates, skip, config,
      breakdown, config.R);
    const u64 prune_ns = elapsed_ns_since(prune_started);
    physical_prune_ns += prune_ns;
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_ns += prune_ns;
    }
    // RobustPrune determines the new node's outgoing graph, but an incoming
    // provisional certificate may use any stable, reachable member of the
    // construction beam.  SpaceV often prunes a wide beam down to 1--4
    // neighbors; using only that set concentrates every pending insertion in
    // a handful of protected planes and can never make progress once their
    // six slots fill.  Keep the two planes independent and distribute bridges
    // over the complete point-in-time reachability witness.
    bridge_targets =
      memory_node_storage_owner_index_detail::
        make_stage1_reachability_bridge_targets(
          span<const RemotePtr>{neighbors},
          span<const RemotePtr>{candidates},
          span<const RemotePtr>{bridge_exclusions});
    return !neighbors.empty() && !bridge_targets.empty();
  };

  // A route/prune snapshot can transiently contain no usable stable parent.
  // Do not hold the physical Stage1 RPC thread (and an authority fanout token)
  // forever waiting for that snapshot to change. The authority retries the
  // same semantic operation after the bounded control backoff.
  constexpr u32 kCandidateRefreshLimit = 2;
  u32 candidate_refreshes = 0;
  while (!refresh_stage1_candidates()) {
    if (storage_insert_shutdown_.load(std::memory_order_acquire) ||
        storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
      result.status = static_cast<u32>(MutationStatus::failed);
      seal_failed_reservation();
      record_physical_stage1(0, 0, 0);
      return result;
    }
    if (candidate_refreshes == kCandidateRefreshLimit) {
      release_retryable_reservation();
      record_physical_stage1(
        candidates.size(), remote_frontier.size(), neighbors.size());
      return result;
    }
    ++candidate_refreshes;
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_cv_.wait_for(
      lock, std::chrono::microseconds(100));
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
  physical_allocate_write_ns = elapsed_ns_since(allocate_started);
  if (breakdown != nullptr) {
    breakdown->storage_owner_write_node_ns += elapsed_ns_since(write_started);
  }
  vec<RemotePtr> backlink_targets;
  // A physical-home attempt must remain bounded. In particular, a local
  // Stage1 attempt runs as overlap work while remote-home Execute responses
  // are already waiting to be consumed by the authority coordinator. An
  // unbounded refresh loop here prevents those already-armed remote tokens
  // from being committed and makes their Stage2 authority gates wait forever.
  constexpr u64 kBridgeRefreshLimit = 2;
  u64 bridge_refreshes = 0;
  for (;;) {
    const auto backlink_started = std::chrono::steady_clock::now();
    backlink_targets = install_local_provisional_backlinks(
      target, span<const RemotePtr>{bridge_targets});
    physical_backlink_ns += elapsed_ns_since(backlink_started);
    if (!backlink_targets.empty()) break;

    // Search/prune observes an unlocked graph snapshot. Under a high-rate GPU
    // query/update workload, every selected parent can retire or change
    // incarnation before the provisional reachability certificate is
    // installed. That is snapshot invalidation, not a permanent mutation
    // failure. Refresh the candidates and rewrite the still-private
    // provisional node's outgoing graph before trying a new parent set.
    ++bridge_refreshes;
    if (bridge_refreshes > kBridgeRefreshLimit) {
      const u64 retirement_sequence =
        begin_storage_owner_maintenance_sequence(1);
      (void)mark_node_deleted(target, item.generation);
      retire_local_dynamic_node(target, retirement_sequence);
      complete_storage_owner_maintenance_sequence(retirement_sequence);
      release_retryable_reservation();
      record_physical_stage1(
        candidates.size(), remote_frontier.size(), neighbors.size());
      return result;
    }
    if (bridge_refreshes <= 8 ||
        (bridge_refreshes & (bridge_refreshes - 1)) == 0) {
      std::cerr << "[storage-owner] Stage1 reachability snapshot invalidated; "
                   "refreshing candidates"
                << " id=" << item.id
                << " generation=" << item.generation
                << " candidate_count=" << candidates.size()
                << " neighbor_count=" << neighbors.size()
                << " bridge_target_count=" << bridge_targets.size()
                << " refresh=" << bridge_refreshes << '\n';
    }
    if (storage_insert_shutdown_.load(std::memory_order_acquire) ||
        storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
      const u64 retirement_sequence =
        begin_storage_owner_maintenance_sequence(1);
      (void)mark_node_deleted(target, item.generation);
      retire_local_dynamic_node(target, retirement_sequence);
      complete_storage_owner_maintenance_sequence(retirement_sequence);
      result.status = static_cast<u32>(MutationStatus::failed);
      seal_failed_reservation();
      record_physical_stage1(
        candidates.size(), remote_frontier.size(), neighbors.size());
      return result;
    }

    {
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::microseconds(100));
    }
    if (!refresh_stage1_candidates()) continue;

    const authority::IncarnationLockResult target_lock =
      try_lock_node(target);
    if (target_lock == authority::IncarnationLockResult::busy) continue;
    lib_assert(target_lock == authority::IncarnationLockResult::locked,
               "private Stage1 node became stale before authority commit");
    const u64 target_header = load_local_node_header_acquire(target);
    lib_assert((target_header & VamanaNode::HEADER_PROVISIONAL) != 0,
               "private Stage1 retry lost its provisional state");
    write_graph_adjacency(
      target, neighbors, {}, item.generation, false);
    unlock_node(target);
  }
  result.target_raw = target.raw_address;
  result.status = static_cast<u32>(MutationStatus::ok);
  const size_t physical_candidate_count = candidates.size();
  const size_t physical_frontier_count = remote_frontier.size();
  const size_t physical_neighbor_count = neighbors.size();
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
  record_physical_stage1(
    physical_candidate_count, physical_frontier_count,
    physical_neighbor_count);
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
    const Configuration& config,
    bool* admission_deferred) {
  using namespace service::storage_owner;
  if (admission_deferred != nullptr) *admission_deferred = false;
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

  bool saw_admission_block = false;
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
    bool batch_admission_blocked = false;
    const bool batch_structurally_valid = arm_local_stage1_items(
      source_shard, span<const Stage1ArmItem>{fused_arm_items},
      arm_results, config, &batch_admission_blocked);
    saw_admission_block |= batch_admission_blocked;
    const bool batch_fast_path =
      batch_structurally_valid &&
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
      bool arm_structurally_valid = batch_structurally_valid;
      if (batch_fast_path) {
        result = &arm_results[slot];
      } else {
        bool one_admission_blocked = false;
        arm_structurally_valid = arm_local_stage1_items(
          source_shard, span<const Stage1ArmItem>{&arm, 1},
          one_result, config, &one_admission_blocked);
        saw_admission_block |= one_admission_blocked;
        if (one_result.size() == 1) result = &one_result.front();
      }
      (void)memory_node_peer_rpc_detail::propagate_fused_stage1_arm_result(
        arm_structurally_valid, arm, result, execute);
    }
  }

  // A prepared fresh-insert request that made no semantic progress solely
  // because the ordered Stage2 window is full is not a transport failure.
  // Keep its original dedup lease/request ID parked on the physical home. A
  // durable completion will wake it and the cached ANN artifact will be armed
  // exactly once; the authority's existing late-response registry consumes
  // that success. This removes completion-credit polling without moving the
  // public ACK boundary or reserving additional Stage2 debt.
  const bool complete_fused_prepare =
    fused_result_indices.size() == header.item_count;
  const bool all_admission_retry = complete_fused_prepare &&
    std::all_of(output, output + header.item_count,
                [](const Stage1ExecuteResult& result) {
                  return result.status ==
                    static_cast<u32>(MutationStatus::retry) &&
                    result.maintenance_sequence == 0;
                });
  if (saw_admission_block && all_admission_retry &&
      admission_deferred != nullptr &&
      !peer_reverse_shutdown_.load(std::memory_order_acquire) &&
      !storage_owner_maintenance_shutdown_.load(
        std::memory_order_acquire)) {
    *admission_deferred = true;
    return false;
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
    const Configuration& config,
    bool* admission_blocked) {
  using namespace service::storage_owner;
  if (admission_blocked != nullptr) *admission_blocked = false;
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
      bool capacity_blocked = false;
      first_sequence = arm_storage_owner_maintenance_batch(
        tasks, config, &capacity_blocked);
      if (first_sequence == 0) {
        // The cancellable accepted-backlog reservation consumed neither a
        // task nor a sequence. Restore every claimed heavy artifact and let
        // the bounded Stage1 control retry path back off before rearming.
        lib_assert(tasks.size() == claimed.size(),
                   "failed atomic Stage1 admission changed batch ownership");
        for (size_t item = 0; item < claimed.size(); ++item) {
          claimed[item].task = std::move(tasks[item]);
        }
        restore_claims();
        if (admission_blocked != nullptr) {
          *admission_blocked = capacity_blocked;
        }
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
