#include "memory_node/peer_rpc/detail.hh"

bool MemoryNode::handle_peer_rpc_requests(vec<PeerRpcMessage>& requests, const Configuration& config) {
  bool progressed = false;
  for (const auto& request : requests) {
    progressed = handle_peer_rpc_request(request, config) || progressed;
  }
  requests.clear();
  return progressed;
}

bool MemoryNode::pump_peer_rpcs_locked(const Configuration&,
                           vec<PeerRpcMessage>& requests,
                           bool wait_for_event) {
  if (!peer_context_) {
    return false;
  }

  bool progressed = false;
  vec<ibv_wc> recv_wcs(std::max<i32>(1, peer_context_->get_config().max_recv_queue_wr));
  do {
    const i32 num_received =
      peer_context_->poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
    if (num_received <= 0) {
      break;
    }
    progressed = true;
    for (i32 i = 0; i < num_received; ++i) {
      const auto [peer_id, slot_id] = decode_64bit(recv_wcs[i].wr_id);
      if (peer_id >= num_storage_nodes_ || slot_id >= peer_rpc_runtime_.recv_slots_per_peer) {
        continue;
      }
      const size_t offset = peer_rpc_receive_offset(peer_id, slot_id);
      const byte_t* payload = peer_rpc_runtime_.buffer.get_full_buffer() + offset;
      const size_t bytes = recv_wcs[i].byte_len;
      if (bytes < sizeof(service::storage_owner::PeerRpcHeader)) {
        repost_peer_rpc_receive(peer_id, slot_id);
        continue;
      }
      const auto* header = reinterpret_cast<const service::storage_owner::PeerRpcHeader*>(payload);
      if (header->magic != service::storage_owner::kPeerRpcMagic ||
          header->version != service::storage_owner::kPeerRpcVersion) {
        repost_peer_rpc_receive(peer_id, slot_id);
        continue;
      }

      if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_request) ||
          header->type == static_cast<u32>(service::storage_owner::PeerRpcType::cleanup_deleted_request) ||
          header->type == static_cast<u32>(service::storage_owner::PeerRpcType::stitch_search_request)) {
        PeerRpcMessage request;
        request.source_shard = peer_id;
        request.payload.assign(payload, payload + bytes);
        requests.push_back(std::move(request));
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_response) ||
                 header->type == static_cast<u32>(service::storage_owner::PeerRpcType::cleanup_deleted_response)) {
        if (peer_rpc_pending_responses_.contains(header->request_id)) {
          peer_rpc_responses_[header->request_id] = *header;
        }
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::stitch_search_response)) {
        if (peer_rpc_pending_responses_.contains(header->request_id)) {
          peer_rpc_responses_[header->request_id] = *header;
          peer_rpc_response_payloads_[header->request_id].assign(
            payload, payload + bytes);
        }
      }

      repost_peer_rpc_receive(peer_id, slot_id);
    }
  } while (wait_for_event);

  return progressed;
}

bool MemoryNode::pump_peer_rpcs(const Configuration& config, bool wait_for_event) {
  std::unique_lock<std::mutex> rpc_lock(peer_rpc_mutex_, std::defer_lock);
  if (wait_for_event) {
    rpc_lock.lock();
  } else if (!rpc_lock.try_lock()) {
    return false;
  }
  vec<PeerRpcMessage> requests;
  const bool progressed = pump_peer_rpcs_locked(config, requests, wait_for_event);
  rpc_lock.unlock();
  return handle_peer_rpc_requests(requests, config) || progressed;
}

bool MemoryNode::wait_for_peer_reverse_update_response(u64 request_id,
                                           u32 target_shard,
                                           u32 item_count,
                                           service::storage_owner::PeerRpcType response_type,
                                           const Configuration& config) {
  const auto wait_started = std::chrono::steady_clock::now();
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  std::unique_lock<std::mutex> lock(peer_rpc_mutex_);
  for (;;) {
    const auto it = peer_rpc_responses_.find(request_id);
    if (it != peer_rpc_responses_.end()) {
      const auto& header = it->second;
      const bool success =
        header.magic == service::storage_owner::kPeerRpcMagic &&
        header.version == service::storage_owner::kPeerRpcVersion &&
        header.type == static_cast<u32>(response_type) &&
        header.source_shard == target_shard &&
        header.item_count == item_count &&
        header.status == static_cast<u32>(service::storage_owner::InsertStatus::ok);
      peer_rpc_responses_.erase(it);
      peer_rpc_pending_responses_.erase(request_id);
      lock.unlock();
      log_slow_peer_reverse_update_response(wait_started, request_id, target_shard, item_count, success);
      return success;
    }

    if (peer_rpc_responses_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
      peer_rpc_pending_responses_.erase(request_id);
      peer_rpc_responses_.erase(request_id);
      peer_rpc_response_payloads_.erase(request_id);
      static std::atomic<u32> timeout_logs{0};
      const u32 log_index = timeout_logs.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 8) {
        std::cerr << "[storage-peer] graph-update RPC timed out after "
                  << config.storage_owner_rpc_timeout_ms << " ms"
                  << " self_shard=" << storage_id_
                  << " target_shard=" << target_shard
                  << " response_type=" << static_cast<u32>(response_type)
                  << " request_id=" << request_id
                  << " item_count=" << item_count << std::endl;
      }
      return false;
    }
  }
}

bool MemoryNode::wait_for_peer_stitch_search_response(u64 request_id,
                                                      u32 target_shard,
                                                      u32 item_count,
                                                      vec<vec<NodeSnapshot>>& candidates_by_item,
                                                      const Configuration& config) {
  const u32 candidate_capacity = storage_owner_cross_shard_degree_;
  lib_assert(candidate_capacity > 0 && candidate_capacity <= VamanaNode::R,
             "invalid online cross-shard stitch degree");
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  std::unique_lock<std::mutex> lock(peer_rpc_mutex_);
  for (;;) {
    const auto it = peer_rpc_responses_.find(request_id);
    if (it != peer_rpc_responses_.end()) {
      const auto header = it->second;
      auto payload_it = peer_rpc_response_payloads_.find(request_id);
      const bool success =
        header.magic == service::storage_owner::kPeerRpcMagic &&
        header.version == service::storage_owner::kPeerRpcVersion &&
        header.type == static_cast<u32>(
          service::storage_owner::PeerRpcType::stitch_search_response) &&
        header.source_shard == target_shard &&
        header.item_count == item_count &&
        header.reserved == candidate_capacity &&
        header.status == static_cast<u32>(service::storage_owner::InsertStatus::ok) &&
        payload_it != peer_rpc_response_payloads_.end() &&
        payload_it->second.size() >= service::storage_owner::stitch_search_response_bytes(
          item_count, candidate_capacity);
      if (success) {
        candidates_by_item.assign(item_count, {});
        const byte_t* payload = payload_it->second.data();
        const u32* counts = service::storage_owner::stitch_search_response_counts(payload);
        const auto* candidates =
          service::storage_owner::stitch_search_response_candidates(payload, item_count);
        const byte_t* vectors =
          service::storage_owner::stitch_search_response_candidate_vectors(
            payload, item_count, candidate_capacity);
        for (u32 item = 0; item < item_count; ++item) {
          const u32 count = std::min<u32>(counts[item], candidate_capacity);
          candidates_by_item[item].reserve(count);
          for (u32 i = 0; i < count; ++i) {
            const size_t slot = static_cast<size_t>(item) * candidate_capacity + i;
            const auto& candidate = candidates[slot];
            if (candidate.raw != 0) {
              NodeSnapshot snapshot;
              snapshot.rptr = RemotePtr{candidate.raw};
              snapshot.generation = candidate.generation;
              snapshot.vector_data.resize(VamanaNode::vector_bytes());
              std::memcpy(snapshot.vector_data.data(),
                          vectors + slot * VamanaNode::vector_bytes(),
                          VamanaNode::vector_bytes());
              candidates_by_item[item].push_back(std::move(snapshot));
            }
          }
        }
      }
      peer_rpc_responses_.erase(it);
      if (payload_it != peer_rpc_response_payloads_.end()) {
        peer_rpc_response_payloads_.erase(payload_it);
      }
      peer_rpc_pending_responses_.erase(request_id);
      return success;
    }

    if (peer_rpc_responses_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
      peer_rpc_pending_responses_.erase(request_id);
      peer_rpc_responses_.erase(request_id);
      peer_rpc_response_payloads_.erase(request_id);
      static std::atomic<u32> timeout_logs{0};
      const u32 log_index = timeout_logs.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 8) {
        std::cerr << "[storage-peer] stitch-search RPC timed out after "
                  << config.storage_owner_rpc_timeout_ms << " ms"
                  << " self_shard=" << storage_id_
                  << " target_shard=" << target_shard
                  << " request_id=" << request_id
                  << " item_count=" << item_count << std::endl;
      }
      return false;
    }
  }
}

bool MemoryNode::post_stitch_search_request(u32 target_shard,
                                            const vec<NodeSnapshot>& targets,
                                            u64& request_id,
                                            u32& item_count,
                                            const Configuration& config) {
  if (targets.empty() || target_shard == storage_id_) {
    return true;
  }

  const u32 max_items = std::max<u32>(1, config.storage_owner_batch_max);
  lib_assert(targets.size() <= max_items,
             "stitch-search request batch exceeds storage_owner_batch_max");
  item_count = static_cast<u32>(std::min<size_t>(targets.size(), max_items));
  const size_t bytes = service::storage_owner::stitch_search_request_bytes(item_count);
  vec<byte_t> message(bytes, 0);
  auto* header = reinterpret_cast<service::storage_owner::PeerRpcHeader*>(message.data());
  header->magic = service::storage_owner::kPeerRpcMagic;
  header->version = service::storage_owner::kPeerRpcVersion;
  header->type = static_cast<u32>(service::storage_owner::PeerRpcType::stitch_search_request);
  header->source_shard = storage_id_;
  header->item_count = item_count;
  header->request_id = next_peer_request_id_.fetch_add(1, std::memory_order_relaxed);
  header->reserved = storage_owner_cross_shard_degree_;
  request_id = header->request_id;
  {
    std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
    peer_rpc_pending_responses_.insert(request_id);
  }

  auto* items = service::storage_owner::stitch_search_items(message.data());
  byte_t* vectors = service::storage_owner::stitch_search_vectors(message.data(), item_count);
  for (u32 i = 0; i < item_count; ++i) {
    const NodeSnapshot& snapshot = targets[i];
    items[i].target_raw = snapshot.rptr.raw_address;
    items[i].id = snapshot.id;
    items[i].generation = snapshot.generation;
    std::memcpy(vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes(),
                snapshot.vector_data.data(),
                VamanaNode::vector_bytes());
  }

  send_peer_rpc_message(target_shard, message.data(), bytes);
  return true;
}

u64 MemoryNode::allocate_peer_request_id() {
  for (;;) {
    const u64 request_id = next_peer_request_id_.fetch_add(
      1, std::memory_order_relaxed);
    if (request_id != 0) return request_id;
  }
}

bool MemoryNode::post_stitch_search_request_async(
    u32 target_shard,
    const vec<NodeSnapshot>& targets,
    u64 request_id,
    u32& item_count,
    const Configuration& config) {
  item_count = 0;
  if (targets.empty() || target_shard == storage_id_) return true;
  if (peer_async_responses_ == nullptr || target_shard >= num_storage_nodes_ ||
      request_id == 0 || targets.size() > config.storage_owner_batch_max) {
    return false;
  }

  item_count = static_cast<u32>(targets.size());
  const size_t bytes = service::storage_owner::stitch_search_request_bytes(
    item_count);
  if (bytes > peer_rpc_runtime_.message_bytes) return false;
  for (const NodeSnapshot& target : targets) {
    if (target.vector_data.size() < VamanaNode::vector_bytes()) return false;
  }

  const auto registration = peer_async_responses_->register_send_attempt(
    request_id,
    target_shard,
    service::storage_owner::PeerRpcType::stitch_search_response,
    item_count);
  if (registration == memory_node_detail::PeerResponseRegistration::already_complete) {
    return true;
  }
  if (registration != memory_node_detail::PeerResponseRegistration::registered &&
      registration != memory_node_detail::PeerResponseRegistration::retry) {
    return false;
  }

  u32 slot_id = 0;
  if (!try_acquire_peer_rpc_send_slot(
        target_shard, PeerRpcSendClass::stitch_search, slot_id)) {
    return false;
  }
  const size_t offset = peer_rpc_async_send_offset(target_shard, slot_id);
  byte_t* message = peer_rpc_runtime_.buffer.get_full_buffer() + offset;
  std::memset(message, 0, bytes);
  auto* header = reinterpret_cast<service::storage_owner::PeerRpcHeader*>(message);
  *header = {};
  header->magic = service::storage_owner::kPeerRpcMagic;
  header->version = service::storage_owner::kPeerRpcVersion;
  header->type = static_cast<u32>(
    service::storage_owner::PeerRpcType::stitch_search_request);
  header->source_shard = storage_id_;
  header->item_count = item_count;
  header->request_id = request_id;
  header->reserved = storage_owner_cross_shard_degree_;

  auto* items = service::storage_owner::stitch_search_items(message);
  byte_t* vectors = service::storage_owner::stitch_search_vectors(
    message, item_count);
  for (u32 i = 0; i < item_count; ++i) {
    const NodeSnapshot& snapshot = targets[i];
    items[i].target_raw = snapshot.rptr.raw_address;
    items[i].id = snapshot.id;
    items[i].generation = snapshot.generation;
    std::memcpy(vectors + static_cast<size_t>(i) * VamanaNode::vector_bytes(),
                snapshot.vector_data.data(),
                VamanaNode::vector_bytes());
  }
  post_peer_rpc_send_slot(target_shard, slot_id, bytes);
  return true;
}

bool MemoryNode::post_peer_op_batch_async(
    u32 target_shard,
    const vec<service::storage_owner::ReverseUpdateOp>& ops,
    service::storage_owner::PeerRpcType request_type,
    u64 request_id,
    u32& item_count,
    const Configuration& config) {
  item_count = 0;
  if (ops.empty() || target_shard == storage_id_) return true;
  const bool reverse = request_type ==
    service::storage_owner::PeerRpcType::reverse_update_request;
  const bool cleanup = request_type ==
    service::storage_owner::PeerRpcType::cleanup_deleted_request;
  if ((!reverse && !cleanup) || peer_async_responses_ == nullptr ||
      target_shard >= num_storage_nodes_ || request_id == 0) {
    return false;
  }

  const u64 max_items = std::max<u64>(
    1, static_cast<u64>(config.R) * config.storage_owner_batch_max);
  if (ops.size() > max_items ||
      ops.size() > std::numeric_limits<u32>::max()) {
    return false;
  }
  item_count = static_cast<u32>(ops.size());
  const size_t bytes = service::storage_owner::reverse_update_request_bytes(
    item_count);
  if (bytes > peer_rpc_runtime_.message_bytes) return false;

  const auto response_type = cleanup
    ? service::storage_owner::PeerRpcType::cleanup_deleted_response
    : service::storage_owner::PeerRpcType::reverse_update_response;
  const auto registration = peer_async_responses_->register_send_attempt(
    request_id, target_shard, response_type, item_count);
  if (registration == memory_node_detail::PeerResponseRegistration::already_complete) {
    return true;
  }
  if (registration != memory_node_detail::PeerResponseRegistration::registered &&
      registration != memory_node_detail::PeerResponseRegistration::retry) {
    return false;
  }

  u32 slot_id = 0;
  if (!try_acquire_peer_rpc_send_slot(
        target_shard, PeerRpcSendClass::graph_update, slot_id)) {
    return false;
  }
  const size_t offset = peer_rpc_async_send_offset(target_shard, slot_id);
  byte_t* message = peer_rpc_runtime_.buffer.get_full_buffer() + offset;
  std::memset(message, 0, bytes);
  auto* header = reinterpret_cast<service::storage_owner::PeerRpcHeader*>(message);
  *header = {};
  header->magic = service::storage_owner::kPeerRpcMagic;
  header->version = service::storage_owner::kPeerRpcVersion;
  header->type = static_cast<u32>(request_type);
  header->source_shard = storage_id_;
  header->item_count = item_count;
  header->request_id = request_id;
  auto* payload_ops = service::storage_owner::reverse_update_ops(message);
  std::memcpy(payload_ops, ops.data(),
              static_cast<size_t>(item_count) * sizeof(*payload_ops));
  post_peer_rpc_send_slot(target_shard, slot_id, bytes);
  return true;
}

MemoryNode::TryPeerResponse MemoryNode::try_consume_peer_rpc_response(
    u64 request_id,
    u32 expected_shard,
    service::storage_owner::PeerRpcType expected_type,
    u32 expected_item_count,
    service::storage_owner::PeerRpcHeader& header,
    vec<byte_t>& payload) {
  if (peer_async_responses_ == nullptr) return TryPeerResponse::stale;

  memory_node_detail::PeerResponseDescriptor response;
  const TryPeerResponse result = peer_async_responses_->try_take(
    request_id, expected_shard, expected_type, expected_item_count, response);
  if (result == TryPeerResponse::pending || result == TryPeerResponse::stale) {
    return result;
  }

  header = response.header;
  const bool valid_descriptor =
    response.peer_id < num_storage_nodes_ &&
    response.receive_slot < peer_rpc_runtime_.recv_slots_per_peer &&
    response.bytes >= sizeof(service::storage_owner::PeerRpcHeader) &&
    response.bytes <= peer_rpc_runtime_.message_bytes;
  if (valid_descriptor) {
    const size_t offset = peer_rpc_receive_offset(
      response.peer_id, response.receive_slot);
    const byte_t* source = peer_rpc_runtime_.buffer.get_full_buffer() + offset;
    payload.assign(source, source + response.bytes);
  } else {
    payload.clear();
    (void)peer_async_responses_->mark_retryable(
      request_id, expected_shard, expected_type, expected_item_count);
  }
  repost_peer_rpc_receive(response.peer_id, response.receive_slot);
  return valid_descriptor ? result : TryPeerResponse::failure;
}

bool MemoryNode::rearm_peer_rpc_response(
    u64 request_id,
    u32 expected_shard,
    service::storage_owner::PeerRpcType expected_type,
    u32 expected_item_count) {
  return peer_async_responses_ != nullptr &&
    peer_async_responses_->mark_retryable(
      request_id, expected_shard, expected_type, expected_item_count);
}

void MemoryNode::cancel_peer_rpc_response(u64 request_id) {
  if (peer_async_responses_ == nullptr) return;
  const auto response = peer_async_responses_->cancel(request_id);
  if (response.has_value()) {
    repost_peer_rpc_receive(response->peer_id, response->receive_slot);
  }
}

bool MemoryNode::enqueue_reverse_update_batch(u32 target_shard,
                                  const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                  const Configuration& config) {
  if (ops.empty()) {
    return true;
  }

  const u64 max_items_u64 = std::max<u64>(
    1, static_cast<u64>(config.R) * config.storage_owner_batch_max);
  lib_assert(max_items_u64 <= std::numeric_limits<u32>::max(),
             "storage-owner reverse outbox batch exceeds wire capacity");
  const size_t max_items = static_cast<size_t>(max_items_u64);
  for (size_t begin = 0; begin < ops.size(); begin += max_items) {
    const size_t count = std::min(max_items, ops.size() - begin);
    std::unique_lock<std::mutex> lock(peer_reverse_outgoing_mutex_);
    peer_reverse_outgoing_cv_.wait(lock, [&]() {
      return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
             peer_reverse_outgoing_.size() < peer_reverse_outgoing_queue_limit_;
    });
    if (peer_reverse_shutdown_.load(std::memory_order_acquire)) {
      return false;
    }
    // Allocate/copy only after a bounded queue slot is owned. Every queued
    // vector is also capped by the fixed wire maximum, so both cardinality and
    // retained bytes have an explicit upper bound.
    PeerReverseOutgoingTask task;
    task.target_shard = target_shard;
    task.ops.assign(ops.begin() + static_cast<std::ptrdiff_t>(begin),
                    ops.begin() + static_cast<std::ptrdiff_t>(begin + count));
    task.queued_at = std::chrono::steady_clock::now();
    peer_reverse_outgoing_.push_back(std::move(task));
    lock.unlock();
    peer_reverse_outgoing_cv_.notify_one();
  }
  return true;
}

bool MemoryNode::send_peer_op_batch_direct(u32 target_shard,
                                      const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                      service::storage_owner::PeerRpcType rpc_type,
                                      bool wait_for_response,
                                      const Configuration& config) {
  if (ops.empty()) {
    return true;
  }

  const u64 max_items_u64 =
    std::max<u64>(1, static_cast<u64>(config.R) * config.storage_owner_batch_max);
  lib_assert(max_items_u64 <= std::numeric_limits<u32>::max(),
             "storage-owner reverse-update RPC batch is too large for the wire format");
  const u32 max_items = static_cast<u32>(max_items_u64);
  for (size_t begin = 0; begin < ops.size(); begin += max_items) {
    const u32 item_count = static_cast<u32>(std::min<size_t>(ops.size() - begin, max_items));
    const size_t bytes = service::storage_owner::reverse_update_request_bytes(item_count);
    lib_assert(bytes <= peer_rpc_runtime_.message_bytes,
               "storage-owner peer RPC message exceeds the registered slot size");
    vec<byte_t> message(bytes);
    auto* header = reinterpret_cast<service::storage_owner::PeerRpcHeader*>(message.data());
    header->magic = service::storage_owner::kPeerRpcMagic;
    header->version = service::storage_owner::kPeerRpcVersion;
    header->type = static_cast<u32>(rpc_type);
    header->source_shard = storage_id_;
    header->item_count = item_count;
    header->request_id = next_peer_request_id_.fetch_add(1, std::memory_order_relaxed);
    if (!wait_for_response) {
      header->reserved |= kPeerRpcFlagNoResponse;
    } else {
      lib_assert(rpc_type == service::storage_owner::PeerRpcType::reverse_update_request ||
                   rpc_type == service::storage_owner::PeerRpcType::cleanup_deleted_request,
                 "peer graph-update response requested for unsupported RPC type");
      std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
      peer_rpc_pending_responses_.insert(header->request_id);
    }
    auto* payload_ops = service::storage_owner::reverse_update_ops(message.data());
    std::memcpy(payload_ops,
                ops.data() + begin,
                static_cast<size_t>(item_count) * sizeof(service::storage_owner::ReverseUpdateOp));
    const auto send_started = std::chrono::steady_clock::now();
    send_peer_rpc_message(target_shard, message.data(), bytes);
    const u64 send_ns = elapsed_ns_since(send_started);
    if (send_ns > 1000ull * 1000ull * 1000ull) {
      static std::atomic<u32> slow_send_logs{0};
      const u32 log_index = slow_send_logs.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 16) {
        std::cerr << "[storage-peer] slow peer op send"
                  << " self_shard=" << storage_id_
                  << " target_shard=" << target_shard
                  << " rpc_type=" << static_cast<u32>(rpc_type)
                  << " request_id=" << header->request_id
                  << " item_count=" << item_count
                  << " elapsed_ms=" << (send_ns / 1000000.0)
                  << std::endl;
      }
    }
    if (wait_for_response) {
      const auto response_type =
        rpc_type == service::storage_owner::PeerRpcType::cleanup_deleted_request
          ? service::storage_owner::PeerRpcType::cleanup_deleted_response
          : service::storage_owner::PeerRpcType::reverse_update_response;
      if (!wait_for_peer_reverse_update_response(
            header->request_id, target_shard, item_count, response_type, config)) {
        return false;
      }
    }
  }
  return true;
}

bool MemoryNode::send_reverse_update_batch_direct(u32 target_shard,
                                      const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                      bool wait_for_response,
                                      const Configuration& config) {
  return send_peer_op_batch_direct(target_shard,
                                   ops,
                                   service::storage_owner::PeerRpcType::reverse_update_request,
                                   wait_for_response,
                                   config);
}

bool MemoryNode::send_reverse_update_batch(u32 target_shard,
                               const vec<service::storage_owner::ReverseUpdateOp>& ops,
                               const Configuration& config) {
  if (config.storage_owner_reverse_mode == "async") {
    return enqueue_reverse_update_batch(target_shard, ops, config);
  }
  return send_reverse_update_batch_direct(target_shard, ops, true, config);
}

bool MemoryNode::send_reverse_update_fanout_and_wait(
    const dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>>& updates,
    const Configuration& config) {
  return send_peer_op_fanout_and_wait(
    updates,
    service::storage_owner::PeerRpcType::reverse_update_request,
    service::storage_owner::PeerRpcType::reverse_update_response,
    config);
}

bool MemoryNode::send_peer_op_fanout_and_wait(
    const dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>>& updates,
    service::storage_owner::PeerRpcType request_type,
    service::storage_owner::PeerRpcType response_type,
    const Configuration& config) {
  lib_assert(
    (request_type == service::storage_owner::PeerRpcType::reverse_update_request &&
     response_type == service::storage_owner::PeerRpcType::reverse_update_response) ||
      (request_type == service::storage_owner::PeerRpcType::cleanup_deleted_request &&
       response_type == service::storage_owner::PeerRpcType::cleanup_deleted_response),
    "invalid peer graph-update request/response pair");
  struct PendingResponse {
    u64 request_id{};
    u32 target_shard{};
    u32 item_count{};
  };

  const u64 max_items_u64 =
    std::max<u64>(1, static_cast<u64>(config.R) * config.storage_owner_batch_max);
  lib_assert(max_items_u64 <= std::numeric_limits<u32>::max(),
             "storage-owner reverse-update RPC batch is too large for the wire format");
  const u32 max_items = static_cast<u32>(max_items_u64);
  vec<PendingResponse> pending;

  for (const auto& [target_shard, ops] : updates) {
    lib_assert(target_shard < num_storage_nodes_ && target_shard != storage_id_,
               "graph-update fanout target must be a remote storage shard");
    for (size_t begin = 0; begin < ops.size(); begin += max_items) {
      const u32 item_count = static_cast<u32>(
        std::min<size_t>(ops.size() - begin, max_items));
      const size_t bytes = service::storage_owner::reverse_update_request_bytes(item_count);
      lib_assert(bytes <= peer_rpc_runtime_.message_bytes,
                 "storage-owner graph-update fanout exceeds the registered slot size");
      vec<byte_t> message(bytes, 0);
      auto* header = reinterpret_cast<service::storage_owner::PeerRpcHeader*>(
        message.data());
      header->magic = service::storage_owner::kPeerRpcMagic;
      header->version = service::storage_owner::kPeerRpcVersion;
      header->type = static_cast<u32>(request_type);
      header->source_shard = storage_id_;
      header->item_count = item_count;
      header->request_id = next_peer_request_id_.fetch_add(
        1, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
        peer_rpc_pending_responses_.insert(header->request_id);
      }
      auto* payload_ops = service::storage_owner::reverse_update_ops(message.data());
      std::memcpy(payload_ops,
                  ops.data() + begin,
                  static_cast<size_t>(item_count) *
                    sizeof(service::storage_owner::ReverseUpdateOp));
      send_peer_rpc_message(target_shard, message.data(), bytes);
      pending.push_back({header->request_id, target_shard, item_count});
    }
  }

  bool success = true;
  for (const PendingResponse& response : pending) {
    success &= wait_for_peer_reverse_update_response(
      response.request_id,
      response.target_shard,
      response.item_count,
      response_type,
      config);
  }
  return success;
}

bool MemoryNode::send_cleanup_deleted_fanout_and_wait(
    const dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>>& updates,
    const Configuration& config) {
  return send_peer_op_fanout_and_wait(
    updates,
    service::storage_owner::PeerRpcType::cleanup_deleted_request,
    service::storage_owner::PeerRpcType::cleanup_deleted_response,
    config);
}

void MemoryNode::log_slow_peer_reverse_update_response(std::chrono::steady_clock::time_point wait_started,
                                           u64 request_id,
                                           u32 target_shard,
                                           u32 item_count,
                                           bool success) const {
  const u64 wait_ns = static_cast<u64>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - wait_started).count());
  if (wait_ns <= 1000ull * 1000ull * 1000ull) {
    return;
  }
  static std::atomic<u32> slow_response_logs{0};
  const u32 log_index = slow_response_logs.fetch_add(1, std::memory_order_relaxed);
  if (log_index >= 16) {
    return;
  }
  std::cerr << "[storage-peer] slow reverse-update response"
            << " self_shard=" << storage_id_
            << " target_shard=" << target_shard
            << " request_id=" << request_id
            << " item_count=" << item_count
            << " success=" << (success ? 1 : 0)
            << " elapsed_ms=" << (wait_ns / 1000000.0)
            << std::endl;
}
