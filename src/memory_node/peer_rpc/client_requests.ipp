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
      if (header->magic != service::storage_owner::kPeerRpcMagic) {
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
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_response)) {
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
                                           const Configuration& config) {
  const auto wait_started = std::chrono::steady_clock::now();
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  std::unique_lock<std::mutex> lock(peer_rpc_mutex_);
  for (;;) {
    const auto it = peer_rpc_responses_.find(request_id);
    if (it != peer_rpc_responses_.end()) {
      const bool success = it->second.status == static_cast<u32>(service::storage_owner::InsertStatus::ok);
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
        std::cerr << "[storage-peer] reverse-update RPC timed out after "
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

bool MemoryNode::wait_for_peer_stitch_search_response(u64 request_id,
                                                      u32 target_shard,
                                                      u32 item_count,
                                                      vec<vec<RemotePtr>>& candidates_by_item,
                                                      const Configuration& config) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  std::unique_lock<std::mutex> lock(peer_rpc_mutex_);
  for (;;) {
    const auto it = peer_rpc_responses_.find(request_id);
    if (it != peer_rpc_responses_.end()) {
      const auto header = it->second;
      auto payload_it = peer_rpc_response_payloads_.find(request_id);
      const bool success =
        header.status == static_cast<u32>(service::storage_owner::InsertStatus::ok) &&
        payload_it != peer_rpc_response_payloads_.end() &&
        payload_it->second.size() >= service::storage_owner::stitch_search_response_bytes(item_count);
      if (success) {
        candidates_by_item.assign(item_count, {});
        const byte_t* payload = payload_it->second.data();
        const u32* counts = service::storage_owner::stitch_search_response_counts(payload);
        const u64* raws = service::storage_owner::stitch_search_response_candidates(payload, item_count);
        for (u32 item = 0; item < item_count; ++item) {
          const u32 count = std::min<u32>(counts[item], VamanaNode::R);
          candidates_by_item[item].reserve(count);
          for (u32 i = 0; i < count; ++i) {
            const u64 raw = raws[static_cast<size_t>(item) * VamanaNode::R + i];
            if (raw != 0) {
              candidates_by_item[item].emplace_back(raw);
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
  header->type = static_cast<u32>(service::storage_owner::PeerRpcType::stitch_search_request);
  header->source_shard = storage_id_;
  header->item_count = item_count;
  header->request_id = next_peer_request_id_.fetch_add(1, std::memory_order_relaxed);
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

bool MemoryNode::enqueue_reverse_update_batch(u32 target_shard,
                                  const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                  const Configuration&) {
  if (ops.empty()) {
    return true;
  }

  PeerReverseOutgoingTask task;
  task.target_shard = target_shard;
  task.ops = ops;
  task.queued_at = std::chrono::steady_clock::now();
  {
    std::unique_lock<std::mutex> lock(peer_reverse_outgoing_mutex_);
    peer_reverse_outgoing_cv_.wait(lock, [&]() {
      return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
             peer_reverse_outgoing_.size() < peer_reverse_outgoing_queue_limit_;
    });
    if (peer_reverse_shutdown_.load(std::memory_order_acquire)) {
      return false;
    }
    peer_reverse_outgoing_.push_back(std::move(task));
  }
  peer_reverse_outgoing_cv_.notify_one();
  return true;
}

bool MemoryNode::enqueue_cleanup_deleted_batch(
    u32 target_shard,
    const vec<service::storage_owner::ReverseUpdateOp>& ops,
    const Configuration&) {
  if (ops.empty()) {
    return true;
  }

  PeerReverseOutgoingTask task;
  task.target_shard = target_shard;
  task.rpc_type = service::storage_owner::PeerRpcType::cleanup_deleted_request;
  task.ops = ops;
  task.queued_at = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(peer_reverse_outgoing_mutex_);
    if (peer_reverse_shutdown_.load(std::memory_order_acquire)) {
      return false;
    }
    peer_reverse_outgoing_.push_back(std::move(task));
  }
  peer_reverse_outgoing_cv_.notify_one();
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
    header->type = static_cast<u32>(rpc_type);
    header->source_shard = storage_id_;
    header->item_count = item_count;
    header->request_id = next_peer_request_id_.fetch_add(1, std::memory_order_relaxed);
    if (!wait_for_response) {
      header->reserved |= kPeerRpcFlagNoResponse;
    } else if (rpc_type == service::storage_owner::PeerRpcType::reverse_update_request) {
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
    if (wait_for_response &&
        rpc_type == service::storage_owner::PeerRpcType::reverse_update_request &&
        !wait_for_peer_reverse_update_response(header->request_id, target_shard, item_count, config)) {
      return false;
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

bool MemoryNode::send_cleanup_deleted_batch(
    u32 target_shard,
    const vec<service::storage_owner::ReverseUpdateOp>& ops,
    const Configuration& config) {
  if (config.storage_owner_reverse_mode == "async") {
    return enqueue_cleanup_deleted_batch(target_shard, ops, config);
  }
  return send_peer_op_batch_direct(target_shard,
                                   ops,
                                   service::storage_owner::PeerRpcType::cleanup_deleted_request,
                                   false,
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
