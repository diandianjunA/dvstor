#include "memory_node/memory_node.hh"

#include <algorithm>
#include <iostream>

void MemoryNode::setup_peer_rpc_runtime(const Configuration& config) {
  if (!peer_context_ || num_storage_nodes_ <= 1) {
    return;
  }

  const size_t reverse_update_bytes =
    service::storage_owner::reverse_update_request_bytes(config.R * config.storage_owner_batch_max);
  peer_rpc_runtime_.message_bytes = align_up(
    std::max({reverse_update_bytes,
              service::storage_owner::reverse_update_response_bytes()}));
  const u32 remote_peer_count = num_storage_nodes_ - 1;
  const u32 max_recv_wr = static_cast<u32>(std::max<i32>(1, config.max_recv_queue_wr));
  const u32 max_slots_per_peer = std::max<u32>(1, max_recv_wr / remote_peer_count);
  const u32 desired_slots_per_peer = std::max<u32>(16, config.storage_owner_rpc_depth * 4);
  peer_rpc_runtime_.recv_slots_per_peer = std::min(desired_slots_per_peer, max_slots_per_peer);
  peer_rpc_runtime_.send_slots_per_peer = std::min(
    std::max<u32>(1, config.storage_owner_rpc_depth),
    peer_rpc_runtime_.recv_slots_per_peer);
  peer_rpc_runtime_.recv_region_bytes =
    peer_rpc_runtime_.message_bytes * num_storage_nodes_ * peer_rpc_runtime_.recv_slots_per_peer;
  peer_rpc_runtime_.sync_send_offset = peer_rpc_runtime_.recv_region_bytes;
  peer_rpc_runtime_.async_send_offset =
    peer_rpc_runtime_.sync_send_offset + peer_rpc_runtime_.message_bytes * num_storage_nodes_;
  const size_t async_send_bytes = peer_rpc_runtime_.message_bytes * num_storage_nodes_ *
                                  peer_rpc_runtime_.send_slots_per_peer;
  peer_rpc_runtime_.buffer.allocate(peer_rpc_runtime_.async_send_offset + async_send_bytes);
  peer_rpc_runtime_.buffer.touch_memory();
  peer_rpc_runtime_.region = std::make_unique<LocalMemoryRegion>(
    *peer_context_, peer_rpc_runtime_.buffer.get_full_buffer(), peer_rpc_runtime_.buffer.buffer_size);
  print_status("storage-owner peer RPC receive slots per peer: " +
               std::to_string(peer_rpc_runtime_.recv_slots_per_peer) +
               " (requested=" + std::to_string(desired_slots_per_peer) + ")");
  for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
    if (peer_id == storage_id_) continue;
    for (u32 slot_id = 0; slot_id < peer_rpc_runtime_.recv_slots_per_peer; ++slot_id) {
      peer_control_qp(peer_id)->post_receive(
        *peer_rpc_runtime_.region,
        static_cast<u32>(peer_rpc_runtime_.message_bytes),
        encode_64bit(peer_id, slot_id),
        peer_rpc_receive_offset(peer_id, slot_id));
    }
  }
}

void MemoryNode::start_peer_reverse_update_runtime(const Configuration& config) {
  if (!peer_context_ || num_storage_nodes_ <= 1) {
    return;
  }

  peer_reverse_shutdown_.store(false, std::memory_order_release);
  peer_reverse_workers_done_.store(false, std::memory_order_release);
  peer_reverse_task_queue_limit_ =
    std::max<size_t>(1024, static_cast<size_t>(config.storage_owner_reverse_queue_depth));
  peer_reverse_outgoing_queue_limit_ = peer_reverse_task_queue_limit_;

  const u32 worker_count = std::max<u32>(1, std::min<u32>(8, std::max<u32>(1, num_compute_threads_ / 2)));
  const size_t snapshot_stride = align_up(VamanaNode::vector_bytes());
  const size_t neighbor_stride = align_up(VamanaNode::neighbor_read_size());
  const size_t coroutine_scratch_stride =
    align_up(std::max<size_t>(VamanaNode::total_size(),
                              std::max(neighbor_stride,
                                       snapshot_stride *
                                         std::max<u32>(1, config.storage_owner_search_snapshot_batch))));
  const size_t scratch_bytes = std::max<size_t>(64ull * 1024ull * 1024ull, coroutine_scratch_stride);
  peer_reverse_worker_states_.reserve(worker_count);
  for (u32 i = 0; i < worker_count; ++i) {
    auto worker = std::make_unique<StorageOwnerThread>(i, 1, config.max_send_queue_wr);
    worker->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
    peer_reverse_worker_states_.push_back(std::move(worker));
  }

  peer_rpc_progress_thread_ = std::thread([this]() { peer_rpc_progress_loop(); });
  peer_reverse_response_thread_ = std::thread([this]() { peer_reverse_response_loop(); });
  peer_reverse_outgoing_thread_ = std::thread([this]() { peer_reverse_outgoing_loop(); });
  for (u32 i = 0; i < worker_count; ++i) {
    peer_reverse_workers_.emplace_back([this, i]() { peer_reverse_update_worker_loop(i); });
  }
  print_status("storage-owner peer reverse-update workers: " + std::to_string(worker_count));
  print_status("storage-owner peer reverse-update tuning: mode=" + config.storage_owner_reverse_mode +
               " queue_depth=" + std::to_string(peer_reverse_task_queue_limit_) +
               " flush_us=" + std::to_string(config.storage_owner_reverse_flush_us) +
               " coalesce_max=" + std::to_string(config.storage_owner_reverse_coalesce_max));
}

void MemoryNode::stop_peer_reverse_update_runtime() {
  peer_reverse_shutdown_.store(true, std::memory_order_release);
  peer_reverse_tasks_cv_.notify_all();
  peer_reverse_responses_cv_.notify_all();
  peer_reverse_outgoing_cv_.notify_all();
  peer_rpc_responses_cv_.notify_all();

  if (peer_reverse_outgoing_thread_.joinable()) {
    peer_reverse_outgoing_thread_.join();
  }
  for (auto& worker : peer_reverse_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  peer_reverse_workers_done_.store(true, std::memory_order_release);
  peer_reverse_responses_cv_.notify_all();
  if (peer_reverse_response_thread_.joinable()) {
    peer_reverse_response_thread_.join();
  }
  if (peer_rpc_progress_thread_.joinable()) {
    peer_rpc_progress_thread_.join();
  }
}

size_t MemoryNode::peer_rpc_receive_offset(u32 peer_id, u32 slot_id) const {
  const size_t slot_index =
    static_cast<size_t>(peer_id) * peer_rpc_runtime_.recv_slots_per_peer + slot_id;
  return slot_index * peer_rpc_runtime_.message_bytes;
}

size_t MemoryNode::peer_rpc_sync_send_offset(u32 peer_id) const {
  return peer_rpc_runtime_.sync_send_offset +
         static_cast<size_t>(peer_id) * peer_rpc_runtime_.message_bytes;
}

size_t MemoryNode::peer_rpc_async_send_offset(u32 peer_id, u32 slot_id) const {
  const size_t slot_index =
    static_cast<size_t>(peer_id) * peer_rpc_runtime_.send_slots_per_peer + slot_id;
  return peer_rpc_runtime_.async_send_offset + slot_index * peer_rpc_runtime_.message_bytes;
}

void MemoryNode::repost_peer_rpc_receive(u32 peer_id, u32 slot_id) {
  if (!peer_context_ || peer_id == storage_id_ || slot_id >= peer_rpc_runtime_.recv_slots_per_peer) {
    return;
  }
  peer_control_qp(peer_id)->post_receive(
    *peer_rpc_runtime_.region,
    static_cast<u32>(peer_rpc_runtime_.message_bytes),
    encode_64bit(peer_id, slot_id),
    peer_rpc_receive_offset(peer_id, slot_id));
}

void MemoryNode::send_peer_rpc_message(u32 peer_id, const void* payload, size_t bytes) {
  lib_assert(peer_context_ != nullptr, "peer context not initialized");
  lib_assert(bytes <= peer_rpc_runtime_.message_bytes, "peer rpc message too large");
  const u64 wr_id = next_peer_sync_wr_id();
  const size_t offset = peer_rpc_sync_send_offset(peer_id);
  std::lock_guard<std::mutex> rpc_send_lock(peer_rpc_send_mutex_);
  std::memcpy(peer_rpc_runtime_.buffer.get_full_buffer() + offset, payload, bytes);
  {
    std::lock_guard<std::mutex> send_lock(*peer_qp_send_mutexes_[peer_id][0]);
    peer_control_qp(peer_id)->post_send_with_id(
      *peer_rpc_runtime_.region,
      static_cast<u32>(bytes),
      IBV_WR_SEND,
      wr_id,
      true,
      nullptr,
      0,
      offset);
  }
  wait_peer_sync_completion(wr_id);
}

service::storage_owner::PeerRpcHeader MemoryNode::make_peer_reverse_update_response(
    const service::storage_owner::PeerRpcHeader& request,
    bool success) const {
  service::storage_owner::PeerRpcHeader response{};
  response.magic = service::storage_owner::kPeerRpcMagic;
  response.type = static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_response);
  response.source_shard = storage_id_;
  response.item_count = request.item_count;
  response.request_id = request.request_id;
  response.status = static_cast<u32>(success ? service::storage_owner::InsertStatus::ok
                                             : service::storage_owner::InsertStatus::failed);
  return response;
}

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
  std::unordered_map<u64, vec<RemotePtr>> grouped;
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

  bool success = true;
  for (const auto& [target_raw, candidates] : grouped) {
    success &= apply_local_reverse_update(RemotePtr{target_raw}, candidates, config);
  }
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

bool MemoryNode::handle_peer_rpc_request(const PeerRpcMessage& message, const Configuration& config) {
  if (message.payload.size() < sizeof(service::storage_owner::PeerRpcHeader)) {
    return false;
  }
  const auto* header =
    reinterpret_cast<const service::storage_owner::PeerRpcHeader*>(message.payload.data());
  if (header->magic != service::storage_owner::kPeerRpcMagic) {
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

  return false;
}

bool MemoryNode::enqueue_peer_reverse_update_task(PeerReverseUpdateTask&& task) {
  std::unique_lock<std::mutex> lock(peer_reverse_tasks_mutex_);
  peer_reverse_tasks_cv_.wait(lock, [&]() {
    return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
           peer_reverse_tasks_.size() < peer_reverse_task_queue_limit_;
  });
  if (peer_reverse_shutdown_.load(std::memory_order_acquire)) {
    return false;
  }
  peer_reverse_tasks_.push_back(std::move(task));
  lock.unlock();
  peer_reverse_tasks_cv_.notify_one();
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

void MemoryNode::peer_rpc_progress_loop() {
  vec<ibv_wc> recv_wcs(std::max<i32>(1, peer_context_->get_config().max_recv_queue_wr));
  for (;;) {
    poll_peer_send_cq();
    const i32 num_received =
      peer_context_->poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
    if (num_received <= 0) {
      if (peer_reverse_workers_done_.load(std::memory_order_acquire) &&
          peer_reverse_responses_.empty() &&
          peer_reverse_outgoing_.empty()) {
        return;
      }
      std::this_thread::yield();
      continue;
    }

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

      if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_request)) {
        const size_t expected_bytes = service::storage_owner::reverse_update_request_bytes(header->item_count);
        if (bytes >= expected_bytes) {
          const auto* ops = service::storage_owner::reverse_update_ops(payload);
          PeerReverseUpdateTask task;
          task.source_shard = peer_id;
          task.header = *header;
          task.received_at = std::chrono::steady_clock::now();
          task.ops.assign(ops, ops + header->item_count);
          enqueue_peer_reverse_update_task(std::move(task));
        }
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_response)) {
        {
          std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
          peer_rpc_responses_[header->request_id] = *header;
        }
        peer_rpc_responses_cv_.notify_all();
      }

      repost_peer_rpc_receive(peer_id, slot_id);
    }
  }
}

void MemoryNode::peer_reverse_update_worker_loop(u32 worker_id) {
  current_storage_owner_thread_ = peer_reverse_worker_states_[worker_id].get();
  const Configuration& config = *storage_worker_config_;
  for (;;) {
    vec<PeerReverseUpdateTask> tasks;
    tasks.reserve(8);
    {
      std::unique_lock<std::mutex> lock(peer_reverse_tasks_mutex_);
      peer_reverse_tasks_cv_.wait(lock, [&]() {
        return peer_reverse_shutdown_.load(std::memory_order_acquire) || !peer_reverse_tasks_.empty();
      });
      if (peer_reverse_shutdown_.load(std::memory_order_acquire) && peer_reverse_tasks_.empty()) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
      tasks.push_back(std::move(peer_reverse_tasks_.front()));
      peer_reverse_tasks_.pop_front();
      size_t coalesced_ops = tasks.back().ops.size();
      if (config.storage_owner_reverse_flush_us > 0 && peer_reverse_tasks_.empty() &&
          !peer_reverse_shutdown_.load(std::memory_order_acquire)) {
        peer_reverse_tasks_cv_.wait_for(lock,
                                        std::chrono::microseconds(config.storage_owner_reverse_flush_us),
                                        [&]() {
                                          return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
                                                 !peer_reverse_tasks_.empty();
                                        });
      }
      while (!peer_reverse_tasks_.empty() &&
             coalesced_ops < config.storage_owner_reverse_coalesce_max) {
        const size_t next_ops = peer_reverse_tasks_.front().ops.size();
        if (!tasks.empty() && coalesced_ops + next_ops > config.storage_owner_reverse_coalesce_max) {
          break;
        }
        tasks.push_back(std::move(peer_reverse_tasks_.front()));
        peer_reverse_tasks_.pop_front();
        coalesced_ops += next_ops;
      }
    }
    peer_reverse_tasks_cv_.notify_one();

    const bool success = apply_peer_reverse_update_tasks(tasks, config);
    for (const PeerReverseUpdateTask& task : tasks) {
      if ((task.header.reserved & kPeerRpcFlagNoResponse) == 0) {
        enqueue_peer_reverse_update_response(task.source_shard, task.header, success);
      }
    }
  }
}

void MemoryNode::peer_reverse_response_loop() {
  for (;;) {
    PeerReverseUpdateResponse response;
    {
      std::unique_lock<std::mutex> lock(peer_reverse_responses_mutex_);
      peer_reverse_responses_cv_.wait(lock, [&]() {
        return peer_reverse_workers_done_.load(std::memory_order_acquire) || !peer_reverse_responses_.empty();
      });
      if (peer_reverse_workers_done_.load(std::memory_order_acquire) && peer_reverse_responses_.empty()) {
        return;
      }
      response = std::move(peer_reverse_responses_.front());
      peer_reverse_responses_.pop_front();
    }
    send_peer_reverse_update_response(response);
  }
}

void MemoryNode::peer_reverse_outgoing_loop() {
  const Configuration& config = *storage_worker_config_;
  const u32 coalesce_max = std::max<u32>(1, config.storage_owner_reverse_coalesce_max);
  for (;;) {
    PeerReverseOutgoingTask task;
    {
      std::unique_lock<std::mutex> lock(peer_reverse_outgoing_mutex_);
      peer_reverse_outgoing_cv_.wait(lock, [&]() {
        return peer_reverse_shutdown_.load(std::memory_order_acquire) || !peer_reverse_outgoing_.empty();
      });
      if (peer_reverse_shutdown_.load(std::memory_order_acquire) && peer_reverse_outgoing_.empty()) {
        return;
      }

      task = std::move(peer_reverse_outgoing_.front());
      peer_reverse_outgoing_.pop_front();
      size_t coalesced_ops = task.ops.size();
      if (config.storage_owner_reverse_flush_us > 0 && peer_reverse_outgoing_.empty() &&
          !peer_reverse_shutdown_.load(std::memory_order_acquire)) {
        peer_reverse_outgoing_cv_.wait_for(lock,
                                           std::chrono::microseconds(config.storage_owner_reverse_flush_us),
                                           [&]() {
                                             return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
                                                    !peer_reverse_outgoing_.empty();
                                           });
      }

      size_t scanned = 0;
      constexpr size_t kOutboxCoalesceScanLimit = 64;
      for (auto it = peer_reverse_outgoing_.begin();
           it != peer_reverse_outgoing_.end() && coalesced_ops < coalesce_max &&
           scanned < kOutboxCoalesceScanLimit;) {
        ++scanned;
        if (it->target_shard != task.target_shard) {
          ++it;
          continue;
        }
        const size_t next_ops = it->ops.size();
        if (coalesced_ops + next_ops > coalesce_max) {
          break;
        }
        task.ops.insert(task.ops.end(), it->ops.begin(), it->ops.end());
        coalesced_ops += next_ops;
        it = peer_reverse_outgoing_.erase(it);
      }
    }
    peer_reverse_outgoing_cv_.notify_one();

    const auto send_started = std::chrono::steady_clock::now();
    const bool success = send_reverse_update_batch_direct(task.target_shard, task.ops, false, config);
    const u64 send_ns = elapsed_ns_since(send_started);
    if (!success || send_ns > 1000ull * 1000ull * 1000ull) {
      static std::atomic<u32> slow_outbox_logs{0};
      const u32 log_index = slow_outbox_logs.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 16) {
        const u64 queued_ns = static_cast<u64>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            send_started - task.queued_at).count());
        std::cerr << "[storage-peer] slow reverse-update outbox send"
                  << " self_shard=" << storage_id_
                  << " target_shard=" << task.target_shard
                  << " item_count=" << task.ops.size()
                  << " success=" << (success ? 1 : 0)
                  << " queued_ms=" << (queued_ns / 1000000.0)
                  << " elapsed_ms=" << (send_ns / 1000000.0)
                  << std::endl;
      }
    }
  }
}

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

      if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_request)) {
        PeerRpcMessage request;
        request.source_shard = peer_id;
        request.payload.assign(payload, payload + bytes);
        requests.push_back(std::move(request));
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_response)) {
        peer_rpc_responses_[header->request_id] = *header;
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
      lock.unlock();
      log_slow_peer_reverse_update_response(wait_started, request_id, target_shard, item_count, success);
      return success;
    }

    if (peer_rpc_responses_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
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


bool MemoryNode::send_reverse_update_batch_direct(u32 target_shard,
                                      const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                      bool wait_for_response,
                                      const Configuration& config) {
  if (ops.empty()) {
    return true;
  }

  const u32 max_items = std::max<u32>(1, config.R * config.storage_owner_batch_max);
  for (size_t begin = 0; begin < ops.size(); begin += max_items) {
    const u32 item_count = static_cast<u32>(std::min<size_t>(ops.size() - begin, max_items));
    const size_t bytes = service::storage_owner::reverse_update_request_bytes(item_count);
    vec<byte_t> message(bytes);
    auto* header = reinterpret_cast<service::storage_owner::PeerRpcHeader*>(message.data());
    header->magic = service::storage_owner::kPeerRpcMagic;
    header->type = static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_request);
    header->source_shard = storage_id_;
    header->item_count = item_count;
    header->request_id = next_peer_request_id_.fetch_add(1, std::memory_order_relaxed);
    if (!wait_for_response) {
      header->reserved |= kPeerRpcFlagNoResponse;
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
        std::cerr << "[storage-peer] slow reverse-update send"
                  << " self_shard=" << storage_id_
                  << " target_shard=" << target_shard
                  << " request_id=" << header->request_id
                  << " item_count=" << item_count
                  << " elapsed_ms=" << (send_ns / 1000000.0)
                  << std::endl;
      }
    }
    if (wait_for_response &&
        !wait_for_peer_reverse_update_response(header->request_id, target_shard, item_count, config)) {
      return false;
    }
  }
  return true;
}

bool MemoryNode::send_reverse_update_batch(u32 target_shard,
                               const vec<service::storage_owner::ReverseUpdateOp>& ops,
                               const Configuration& config) {
  if (config.storage_owner_reverse_mode == "async") {
    return enqueue_reverse_update_batch(target_shard, ops, config);
  }
  return send_reverse_update_batch_direct(target_shard, ops, true, config);
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
