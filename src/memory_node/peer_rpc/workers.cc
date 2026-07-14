#include "memory_node/peer_rpc/detail.hh"

void MemoryNode::peer_rpc_progress_loop() {
  const Configuration& config = *storage_worker_config_;
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
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::cleanup_deleted_request)) {
        const size_t expected_bytes = service::storage_owner::reverse_update_request_bytes(header->item_count);
        if (bytes >= expected_bytes) {
          const auto* ops = service::storage_owner::reverse_update_ops(payload);
          (void)handle_peer_cleanup_deleted_request(peer_id, *header, ops, config);
        }
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::stitch_search_request)) {
        const size_t expected_bytes = service::storage_owner::stitch_search_request_bytes(header->item_count);
        if (bytes >= expected_bytes) {
          PeerStitchSearchTask task;
          task.source_shard = peer_id;
          task.header = *header;
          task.received_at = std::chrono::steady_clock::now();
          task.payload.assign(payload, payload + expected_bytes);
          if (!enqueue_peer_stitch_search_task(std::move(task))) {
            send_peer_stitch_search_failed_response(peer_id, *header);
          }
        }
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_response)) {
        bool accepted = false;
        {
          std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
          if (peer_rpc_pending_responses_.contains(header->request_id)) {
            peer_rpc_responses_[header->request_id] = *header;
            accepted = true;
          }
        }
        if (accepted) peer_rpc_responses_cv_.notify_all();
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::stitch_search_response)) {
        bool accepted = false;
        {
          std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
          if (peer_rpc_pending_responses_.contains(header->request_id)) {
            peer_rpc_responses_[header->request_id] = *header;
            peer_rpc_response_payloads_[header->request_id].assign(
              payload, payload + bytes);
            accepted = true;
          }
        }
        if (accepted) peer_rpc_responses_cv_.notify_all();
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

void MemoryNode::peer_stitch_search_worker_loop(u32 worker_id) {
  current_storage_owner_thread_ = peer_stitch_search_worker_states_[worker_id].get();
  const Configuration& config = *storage_worker_config_;
  const u32 worker_count =
    static_cast<u32>(std::max<size_t>(1, peer_stitch_search_worker_states_.size()));
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(peer_stitch_search_tasks_mutex_);
      peer_stitch_search_tasks_cv_.wait(lock, [&]() {
        return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
               !peer_stitch_search_tasks_.empty();
      });
      if (peer_reverse_shutdown_.load(std::memory_order_acquire) &&
          peer_stitch_search_tasks_.empty()) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
    }

    bool foreground_active =
      storage_owner_insert_active_workers_.load(std::memory_order_acquire) != 0;
    {
      std::unique_lock<std::mutex> lock(storage_insert_tasks_mutex_, std::try_to_lock);
      foreground_active = foreground_active || !lock.owns_lock() || !storage_insert_tasks_.empty();
    }
    u32 worker_limit = foreground_active
                         ? std::max<u32>(1, worker_count / 4)
                         : worker_count;
    u32 active = peer_stitch_search_active_workers_.load(std::memory_order_acquire);
    while (active >= worker_limit) {
      std::unique_lock<std::mutex> lock(peer_stitch_search_tasks_mutex_);
      peer_stitch_search_tasks_cv_.wait_for(lock, std::chrono::milliseconds(1), [&]() {
        return peer_reverse_shutdown_.load(std::memory_order_acquire);
      });
      if (peer_reverse_shutdown_.load(std::memory_order_acquire) &&
          peer_stitch_search_tasks_.empty()) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
      foreground_active =
        storage_owner_insert_active_workers_.load(std::memory_order_acquire) != 0;
      {
        std::unique_lock<std::mutex> insert_lock(storage_insert_tasks_mutex_, std::try_to_lock);
        foreground_active = foreground_active || !insert_lock.owns_lock() ||
                            !storage_insert_tasks_.empty();
      }
      const u32 refreshed_limit = foreground_active
                                    ? std::max<u32>(1, worker_count / 4)
                                    : worker_count;
      worker_limit = refreshed_limit;
      active = peer_stitch_search_active_workers_.load(std::memory_order_acquire);
      if (active < refreshed_limit) {
        break;
      }
    }
    active = peer_stitch_search_active_workers_.load(std::memory_order_acquire);
    while (active < worker_limit) {
      if (peer_stitch_search_active_workers_.compare_exchange_weak(
            active, active + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        break;
      }
    }
    if (active >= worker_limit) {
      continue;
    }
    atomic_utils::CounterDecrementGuard active_slot(
      peer_stitch_search_active_workers_);

    PeerStitchSearchTask task;
    {
      std::lock_guard<std::mutex> lock(peer_stitch_search_tasks_mutex_);
      if (peer_stitch_search_tasks_.empty()) {
        continue;
      }
      task = std::move(peer_stitch_search_tasks_.front());
      peer_stitch_search_tasks_.pop_front();
    }
    peer_stitch_search_tasks_cv_.notify_one();

    const bool success = handle_peer_stitch_search_request(
      task.source_shard, task.header, task.payload.data(), config);
    peer_stitch_search_processed_.fetch_add(1, std::memory_order_relaxed);
    if (success) {
      peer_stitch_search_items_.fetch_add(task.header.item_count, std::memory_order_relaxed);
    }
    peer_stitch_search_tasks_cv_.notify_one();
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
        if (it->target_shard != task.target_shard || it->rpc_type != task.rpc_type) {
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
    const bool success =
      send_peer_op_batch_direct(task.target_shard, task.ops, task.rpc_type, false, config);
    const u64 send_ns = elapsed_ns_since(send_started);
    if (!success || send_ns > 1000ull * 1000ull * 1000ull) {
      static std::atomic<u32> slow_outbox_logs{0};
      const u32 log_index = slow_outbox_logs.fetch_add(1, std::memory_order_relaxed);
      if (log_index < 16) {
        const u64 queued_ns = static_cast<u64>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            send_started - task.queued_at).count());
        std::cerr << "[storage-peer] slow peer outbox send"
                  << " self_shard=" << storage_id_
                  << " target_shard=" << task.target_shard
                  << " rpc_type=" << static_cast<u32>(task.rpc_type)
                  << " item_count=" << task.ops.size()
                  << " success=" << (success ? 1 : 0)
                  << " queued_ms=" << (queued_ns / 1000000.0)
                  << " elapsed_ms=" << (send_ns / 1000000.0)
                  << std::endl;
      }
    }
  }
}
