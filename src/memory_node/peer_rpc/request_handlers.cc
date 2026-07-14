#include "memory_node/peer_rpc/detail.hh"

bool MemoryNode::apply_peer_reverse_update_task(const PeerReverseUpdateTask& task, const Configuration& config) {
  vec<PeerReverseUpdateTask> tasks;
  tasks.push_back(task);
  return apply_peer_reverse_update_tasks(tasks, config);
}

bool MemoryNode::apply_peer_reverse_update_tasks(const vec<PeerReverseUpdateTask>& tasks, const Configuration& config) {
  if (tasks.empty()) {
    return true;
  }

  const auto apply_started = std::chrono::steady_clock::now();
  dense_hashmap_t<u64, vec<RemotePtr>> grouped;
  size_t item_count = 0;
  for (const PeerReverseUpdateTask& task : tasks) {
    item_count += task.ops.size();
  }
  grouped.reserve(item_count);
  for (const PeerReverseUpdateTask& task : tasks) {
    for (const auto& op : task.ops) {
      const RemotePtr target{op.target_raw};
      const RemotePtr candidate{op.candidate_raw};
      lib_assert(local_shard(target.memory_node()), "reverse-update target routed to wrong shard");
      grouped[target.raw_address].push_back(candidate);
    }
  }

  const bool success = apply_local_reverse_updates_batched(grouped, config);
  const u64 apply_ns = elapsed_ns_since(apply_started);
  if (apply_ns > 1000ull * 1000ull * 1000ull) {
    static std::atomic<u32> slow_apply_logs{0};
    const u32 log_index = slow_apply_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-peer] slow reverse-update apply"
                << " self_shard=" << storage_id_
                << " task_count=" << tasks.size()
                << " item_count=" << item_count
                << " grouped_targets=" << grouped.size()
                << " elapsed_ms=" << (apply_ns / 1000000.0)
                << std::endl;
    }
  }
  return success;
}

void MemoryNode::send_peer_reverse_update_response(const PeerReverseUpdateResponse& response) {
  const auto response_send_started = std::chrono::steady_clock::now();
  send_peer_rpc_message(response.destination_shard, &response.header, sizeof(response.header));
  const u64 response_send_ns = elapsed_ns_since(response_send_started);
  if (response_send_ns > 1000ull * 1000ull * 1000ull) {
    static std::atomic<u32> slow_response_send_logs{0};
    const u32 log_index = slow_response_send_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-peer] slow reverse-update response-send"
                << " self_shard=" << storage_id_
                << " destination_shard=" << response.destination_shard
                << " request_id=" << response.header.request_id
                << " item_count=" << response.header.item_count
                << " queued_ms="
                << (std::chrono::duration_cast<std::chrono::nanoseconds>(
                      response_send_started - response.queued_at).count() / 1000000.0)
                << " elapsed_ms=" << (response_send_ns / 1000000.0)
                << std::endl;
    }
  }
}

bool MemoryNode::handle_peer_reverse_update_request(u32 source_shard,
                                        const service::storage_owner::PeerRpcHeader& header,
                                        const service::storage_owner::ReverseUpdateOp* ops,
                                        const Configuration& config) {
  PeerReverseUpdateTask task;
  task.source_shard = source_shard;
  task.header = header;
  task.received_at = std::chrono::steady_clock::now();
  task.ops.assign(ops, ops + header.item_count);
  const bool success = apply_peer_reverse_update_task(task, config);
  if ((header.reserved & kPeerRpcFlagNoResponse) == 0) {
    PeerReverseUpdateResponse response;
    response.destination_shard = source_shard;
    response.header = make_peer_reverse_update_response(header, success);
    response.queued_at = std::chrono::steady_clock::now();
    send_peer_reverse_update_response(response);
  }
  return success;
}

bool MemoryNode::handle_peer_cleanup_deleted_request(
    u32 source_shard,
    const service::storage_owner::PeerRpcHeader& header,
    const service::storage_owner::ReverseUpdateOp* ops,
    const Configuration& config) {
  (void)source_shard;
  if (ops == nullptr || header.item_count == 0) {
    return true;
  }

  dense_hashmap_t<u64, vec<RemotePtr>> grouped;
  grouped.reserve(header.item_count);
  for (u32 i = 0; i < header.item_count; ++i) {
    const RemotePtr target{ops[i].target_raw};
    const RemotePtr deleted{ops[i].candidate_raw};
    lib_assert(local_shard(target.memory_node()), "cleanup-deleted target routed to wrong shard");
    grouped[target.raw_address].push_back(deleted);
  }

  bool success = true;
  for (auto& [target_raw, deleted_ptrs] : grouped) {
    for (const RemotePtr& deleted : deleted_ptrs) {
      success &= remove_local_neighbor(RemotePtr{target_raw}, deleted, config);
    }
  }
  return success;
}

bool MemoryNode::handle_peer_stitch_search_request(
    u32 source_shard,
    const service::storage_owner::PeerRpcHeader& header,
    const byte_t* payload,
    const Configuration& config) {
  const size_t response_bytes = service::storage_owner::stitch_search_response_bytes(header.item_count);
  vec<byte_t> response(response_bytes, 0);
  auto* response_header =
    reinterpret_cast<service::storage_owner::PeerRpcHeader*>(response.data());
  response_header->magic = service::storage_owner::kPeerRpcMagic;
  response_header->version = service::storage_owner::kPeerRpcVersion;
  response_header->type = static_cast<u32>(service::storage_owner::PeerRpcType::stitch_search_response);
  response_header->source_shard = storage_id_;
  response_header->item_count = header.item_count;
  response_header->request_id = header.request_id;
  response_header->status = static_cast<u32>(service::storage_owner::InsertStatus::ok);

  if (storage_owner_anchor_index_ == nullptr || storage_owner_anchor_index_->empty()) {
    response_header->status = static_cast<u32>(service::storage_owner::InsertStatus::failed);
    send_peer_rpc_message(source_shard, response.data(), response.size());
    return false;
  }

  const auto* items = service::storage_owner::stitch_search_items(payload);
  const byte_t* vectors = service::storage_owner::stitch_search_vectors(payload, header.item_count);
  u32* counts = service::storage_owner::stitch_search_response_counts(response.data());
  auto* candidate_slots =
    service::storage_owner::stitch_search_response_candidates(response.data(), header.item_count);
  byte_t* candidate_vectors =
    service::storage_owner::stitch_search_response_candidate_vectors(response.data(), header.item_count);

  for (u32 i = 0; i < header.item_count; ++i) {
    const byte_t* raw_vector = vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes();
    vec<element_t> query = decode_storage_vector_to_float(
      raw_vector, VamanaNode::vector_dtype(), VamanaNode::DIM);
    vec<RemotePtr> anchors =
      storage_owner_anchor_index_->nearest_anchors(
        span<const element_t>{query.data(), query.size()},
        storage_id_,
        config.storage_owner_anchor_hints);
    vec<RemotePtr> candidates =
      anchor_search_candidates(span<const element_t>{query.data(), query.size()},
                               anchors,
                               config,
                               nullptr,
                               true);
    u32 written = 0;
    hashset_t<RemotePtr> seen;
    for (const RemotePtr& candidate : candidates) {
      if (written >= config.R) {
        break;
      }
      if (candidate.is_null() || !local_shard(candidate.memory_node()) ||
          candidate.raw_address == items[i].target_raw ||
          !seen.insert(candidate).second) {
        continue;
      }
      NodeSnapshot snapshot;
      if (!read_node_snapshot(candidate, snapshot) || snapshot.deleted) {
        continue;
      }
      const size_t slot = static_cast<size_t>(i) * VamanaNode::R + written;
      candidate_slots[slot].raw = candidate.raw_address;
      candidate_slots[slot].generation = snapshot.generation;
      std::memcpy(candidate_vectors + slot * VamanaNode::vector_bytes(),
                  snapshot.vector_data.data(),
                  VamanaNode::vector_bytes());
      ++written;
    }
    counts[i] = written;
  }

  send_peer_rpc_message(source_shard, response.data(), response.size());
  return true;
}

void MemoryNode::send_peer_stitch_search_failed_response(
    u32 destination_shard,
    const service::storage_owner::PeerRpcHeader& request) {
  const size_t response_bytes =
    service::storage_owner::stitch_search_response_bytes(request.item_count);
  vec<byte_t> response(response_bytes, 0);
  auto* response_header =
    reinterpret_cast<service::storage_owner::PeerRpcHeader*>(response.data());
  response_header->magic = service::storage_owner::kPeerRpcMagic;
  response_header->version = service::storage_owner::kPeerRpcVersion;
  response_header->type =
    static_cast<u32>(service::storage_owner::PeerRpcType::stitch_search_response);
  response_header->source_shard = storage_id_;
  response_header->item_count = request.item_count;
  response_header->request_id = request.request_id;
  response_header->status = static_cast<u32>(service::storage_owner::InsertStatus::failed);
  send_peer_rpc_message(destination_shard, response.data(), response.size());
}

bool MemoryNode::handle_peer_rpc_request(const PeerRpcMessage& message, const Configuration& config) {
  if (message.payload.size() < sizeof(service::storage_owner::PeerRpcHeader)) {
    return false;
  }
  const auto* header =
    reinterpret_cast<const service::storage_owner::PeerRpcHeader*>(message.payload.data());
  if (header->magic != service::storage_owner::kPeerRpcMagic ||
      header->version != service::storage_owner::kPeerRpcVersion) {
    return false;
  }

  if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_request)) {
    const size_t expected_bytes = service::storage_owner::reverse_update_request_bytes(header->item_count);
    if (message.payload.size() < expected_bytes) {
      return false;
    }
    const auto* ops = service::storage_owner::reverse_update_ops(message.payload.data());
    return handle_peer_reverse_update_request(message.source_shard, *header, ops, config);
  }
  if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::cleanup_deleted_request)) {
    const size_t expected_bytes = service::storage_owner::reverse_update_request_bytes(header->item_count);
    if (message.payload.size() < expected_bytes) {
      return false;
    }
    const auto* ops = service::storage_owner::reverse_update_ops(message.payload.data());
    return handle_peer_cleanup_deleted_request(message.source_shard, *header, ops, config);
  }
  if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::stitch_search_request)) {
    const size_t expected_bytes = service::storage_owner::stitch_search_request_bytes(header->item_count);
    if (message.payload.size() < expected_bytes) {
      return false;
    }
    PeerStitchSearchTask task;
    task.source_shard = message.source_shard;
    task.header = *header;
    task.received_at = std::chrono::steady_clock::now();
    task.payload.assign(message.payload.data(), message.payload.data() + expected_bytes);
    if (!enqueue_peer_stitch_search_task(std::move(task))) {
      send_peer_stitch_search_failed_response(message.source_shard, *header);
      return false;
    }
    return true;
  }

  return false;
}

bool MemoryNode::enqueue_peer_reverse_update_task(PeerReverseUpdateTask&& task) {
  const u64 item_count = task.ops.size();
  size_t queue_size = 0;
  std::unique_lock<std::mutex> lock(peer_reverse_tasks_mutex_);
  peer_reverse_tasks_cv_.wait(lock, [&]() {
    return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
           peer_reverse_tasks_.size() < peer_reverse_task_queue_limit_;
  });
  if (peer_reverse_shutdown_.load(std::memory_order_acquire)) {
    return false;
  }
  peer_reverse_tasks_.push_back(std::move(task));
  queue_size = peer_reverse_tasks_.size();
  lock.unlock();
  peer_reverse_update_enqueued_.fetch_add(1, std::memory_order_relaxed);
  peer_reverse_update_items_enqueued_.fetch_add(item_count, std::memory_order_relaxed);
  atomic_utils::update_max_relaxed(
    peer_reverse_update_max_queue_, static_cast<u64>(queue_size));
  peer_reverse_tasks_cv_.notify_one();
  return true;
}

bool MemoryNode::enqueue_peer_stitch_search_task(PeerStitchSearchTask&& task) {
  std::lock_guard<std::mutex> lock(peer_stitch_search_tasks_mutex_);
  if (peer_reverse_shutdown_.load(std::memory_order_acquire) ||
      peer_stitch_search_tasks_.size() >= peer_stitch_search_task_queue_limit_) {
    return false;
  }
  peer_stitch_search_tasks_.push_back(std::move(task));
  peer_stitch_search_enqueued_.fetch_add(1, std::memory_order_relaxed);
  atomic_utils::update_max_relaxed(
    peer_stitch_search_max_queue_,
    static_cast<u64>(peer_stitch_search_tasks_.size()));
  peer_stitch_search_tasks_cv_.notify_one();
  return true;
}

void MemoryNode::enqueue_peer_reverse_update_response(u32 destination_shard,
                                          const service::storage_owner::PeerRpcHeader& request,
                                          bool success) {
  PeerReverseUpdateResponse response;
  response.destination_shard = destination_shard;
  response.header = make_peer_reverse_update_response(request, success);
  response.queued_at = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(peer_reverse_responses_mutex_);
    peer_reverse_responses_.push_back(std::move(response));
  }
  peer_reverse_responses_cv_.notify_one();
}
