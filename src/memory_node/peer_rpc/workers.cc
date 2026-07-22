#include "memory_node/peer_rpc/detail.hh"

void MemoryNode::peer_rpc_progress_loop() {
  lib_assert(storage_worker_config_ != nullptr,
             "peer RPC progress started without storage configuration");
  const Configuration& config = *storage_worker_config_;
  current_peer_rpc_progress_thread_ = true;
  peer_rpc_progress_running_.store(true, std::memory_order_release);
  vec<ibv_wc> recv_wcs(std::max<i32>(1, peer_context_->get_config().max_recv_queue_wr));
  for (;;) {
    poll_peer_send_cq();
    const i32 num_received =
      peer_context_->poll_recv_cq(recv_wcs.data(), static_cast<i32>(recv_wcs.size()));
    if (num_received <= 0) {
      bool responses_empty = false;
      bool sends_empty = false;
      if (peer_reverse_response_done_.load(std::memory_order_acquire)) {
        responses_empty = peer_reverse_responses_ == nullptr ||
                          peer_reverse_responses_->empty();
        {
          std::lock_guard<std::mutex> lock(peer_completion_mutex_);
          sends_empty = peer_pending_sends_.empty();
        }
      }
      if (peer_reverse_response_done_.load(std::memory_order_acquire) &&
          responses_empty && sends_empty) {
        peer_rpc_progress_running_.store(false, std::memory_order_release);
        peer_completion_cv_.notify_all();
        current_peer_rpc_progress_thread_ = false;
        return;
      }
      std::this_thread::yield();
      continue;
    }

    for (i32 i = 0; i < num_received; ++i) {
      bool hold_receive_slot = false;
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
          header->version != service::storage_owner::kPeerRpcVersion ||
          header->source_shard != peer_id) {
        repost_peer_rpc_receive(peer_id, slot_id);
        continue;
      }

      const bool is_request =
        header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::reverse_update_request) ||
        header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::cleanup_deleted_request) ||
        header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::reconcile_reverse_request) ||
        header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::centroid_membership_request) ||
        header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::stage1_execute_request) ||
        header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::stage1_arm_request) ||
        header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::cleanup_activate_request) ||
        header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::authority_placement_request) ||
        header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::dynamic_node_control_request);
      const bool is_stage2_home_request =
        header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::stage2_expand_score_request);
      if ((is_request || is_stage2_home_request) &&
          peer_reverse_shutdown_.load(std::memory_order_acquire)) {
        repost_peer_rpc_receive(peer_id, slot_id);
        continue;
      }

      if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_request)) {
        const size_t expected_bytes = service::storage_owner::reverse_update_request_bytes(header->item_count);
        if (header->item_count != 0 && header->reserved == 0 &&
            bytes == expected_bytes) {
          const auto decision = peer_request_deduplicator_->begin(
            peer_id, *header, true);
          if (decision.action == memory_node_detail::PeerRequestAction::execute) {
            const auto* ops = service::storage_owner::reverse_update_ops(payload);
            PeerReverseUpdateTask task;
            task.source_shard = peer_id;
            task.header = *header;
            task.dedup_lease = decision.lease;
            task.received_at = std::chrono::steady_clock::now();
            task.ops.assign(ops, ops + header->item_count);
            if (!enqueue_peer_reverse_update_task(std::move(task))) {
              lib_assert(peer_request_deduplicator_->abandon(
                           decision.lease, peer_id, *header),
                         "reverse request lost its dedup lease");
              enqueue_peer_reverse_update_response(peer_id, *header, false);
            }
          } else if (decision.action ==
                       memory_node_detail::PeerRequestAction::replay) {
            PeerReverseUpdateResponse response;
            response.destination_shard = peer_id;
            response.header = decision.response;
            response.queued_at = std::chrono::steady_clock::now();
            (void)try_enqueue_peer_reverse_update_response(
              std::move(response));
          }
        }
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::cleanup_deleted_request)) {
        const size_t expected_bytes = service::storage_owner::reverse_update_request_bytes(header->item_count);
        if (header->item_count != 0 && header->reserved == 0 &&
            bytes == expected_bytes) {
          const auto decision = peer_request_deduplicator_->begin(
            peer_id, *header, true);
          if (decision.action == memory_node_detail::PeerRequestAction::execute) {
            const auto* ops = service::storage_owner::reverse_update_ops(payload);
            PeerReverseUpdateTask task;
            task.source_shard = peer_id;
            task.header = *header;
            task.dedup_lease = decision.lease;
            task.received_at = std::chrono::steady_clock::now();
            task.ops.assign(ops, ops + header->item_count);
            if (!enqueue_peer_reverse_update_task(std::move(task))) {
              lib_assert(peer_request_deduplicator_->abandon(
                           decision.lease, peer_id, *header),
                         "cleanup request lost its dedup lease");
              enqueue_peer_reverse_update_response(peer_id, *header, false);
            }
          } else if (decision.action ==
                       memory_node_detail::PeerRequestAction::replay) {
            PeerReverseUpdateResponse response;
            response.destination_shard = peer_id;
            response.header = decision.response;
            response.queued_at = std::chrono::steady_clock::now();
            (void)try_enqueue_peer_reverse_update_response(
              std::move(response));
          }
        }
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reconcile_reverse_request)) {
        const size_t expected_bytes =
          service::storage_owner::reconcile_reverse_request_bytes(
            header->item_count);
        const u64 max_items = std::max<u64>(
          1, static_cast<u64>(config.R) *
               config.storage_owner_batch_max);
        if (header->item_count != 0 &&
            header->item_count <= max_items &&
            header->reserved == 0 && bytes == expected_bytes) {
          // Reconciliation replies contain one postcondition per operation.
          // Re-execute an identical request ID instead of replaying the
          // generic header-only cache; every operation is idempotent.
          const auto decision = peer_request_deduplicator_->begin(
            peer_id, *header, false);
          if (decision.action ==
              memory_node_detail::PeerRequestAction::execute) {
            const auto* ops =
              service::storage_owner::reconcile_reverse_ops(payload);
            PeerReverseUpdateTask task;
            task.source_shard = peer_id;
            task.header = *header;
            task.dedup_lease = decision.lease;
            task.received_at = std::chrono::steady_clock::now();
            task.reconcile_ops.assign(ops, ops + header->item_count);
            if (!enqueue_peer_reverse_update_task(std::move(task))) {
              lib_assert(peer_request_deduplicator_->abandon(
                           decision.lease, peer_id, *header),
                         "reconcile request lost its dedup lease");
              enqueue_peer_reverse_update_response(
                peer_id, *header, false);
            }
          }
        }
      } else if (header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::centroid_membership_request)) {
        const size_t expected_bytes =
          service::storage_owner::centroid_membership_request_bytes(
            header->item_count);
        if (header->item_count != 0 &&
            header->item_count <= config.storage_owner_batch_max &&
            header->reserved == 0 && bytes == expected_bytes) {
          const auto decision = peer_request_deduplicator_->begin(
            peer_id, *header, true);
          if (decision.action ==
              memory_node_detail::PeerRequestAction::execute) {
            const auto* ops =
              service::storage_owner::centroid_membership_ops(payload);
            PeerReverseUpdateTask task;
            task.source_shard = peer_id;
            task.header = *header;
            task.dedup_lease = decision.lease;
            task.received_at = std::chrono::steady_clock::now();
            task.centroid_ops.assign(ops, ops + header->item_count);
            if (!enqueue_peer_reverse_update_task(std::move(task))) {
              lib_assert(peer_request_deduplicator_->abandon(
                           decision.lease, peer_id, *header),
                         "centroid request lost its dedup lease");
              enqueue_peer_reverse_update_response(
                peer_id, *header, false);
            }
          } else if (decision.action ==
                       memory_node_detail::PeerRequestAction::replay) {
            PeerReverseUpdateResponse response;
            response.destination_shard = peer_id;
            response.header = decision.response;
            response.queued_at = std::chrono::steady_clock::now();
            (void)try_enqueue_peer_reverse_update_response(
              std::move(response));
          }
        }
      } else if (header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::cleanup_activate_request) ||
                 header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::authority_placement_request) ||
                 header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::dynamic_node_control_request)) {
        const auto request_type = static_cast<
          service::storage_owner::PeerRpcType>(header->type);
        size_t expected_bytes = 0;
        if (request_type ==
            service::storage_owner::PeerRpcType::cleanup_activate_request) {
          expected_bytes =
            service::storage_owner::cleanup_activate_request_bytes(
              header->item_count);
        } else if (request_type ==
                   service::storage_owner::PeerRpcType::authority_placement_request) {
          expected_bytes =
            service::storage_owner::authority_placement_request_bytes(
              header->item_count);
        } else {
          expected_bytes =
            service::storage_owner::dynamic_node_control_request_bytes(
              header->item_count);
        }
        if (header->item_count != 0 &&
            header->item_count <= config.storage_owner_batch_max &&
            header->reserved == 0 && bytes == expected_bytes) {
          // These replies carry per-item payloads, so the generic header-only
          // replay cache cannot serve them.  The physical cleanup token table
          // and authority CAS make a same-ID re-execution side-effect safe.
          const auto decision = peer_request_deduplicator_->begin(
            peer_id, *header, false);
          if (decision.action ==
              memory_node_detail::PeerRequestAction::execute) {
            PeerPhysicalControlTask task;
            task.source_shard = peer_id;
            task.header = *header;
            task.dedup_lease = decision.lease;
            task.received_at = std::chrono::steady_clock::now();
            task.payload.assign(payload, payload + expected_bytes);
            if (!enqueue_peer_physical_control_task(std::move(task))) {
              lib_assert(peer_request_deduplicator_->abandon(
                           decision.lease, peer_id, *header),
                         "physical-control request lost its dedup lease");
            }
          }
        }
      } else if (header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::stage2_expand_score_request)) {
        const size_t expected_bytes =
          service::storage_owner::stage2_expand_score_request_bytes(
            header->item_count);
        if (header->item_count != 0 &&
            header->item_count <= config.storage_owner_batch_max &&
            header->reserved == 0 && bytes == expected_bytes) {
          const auto decision = peer_request_deduplicator_->begin(
            peer_id, *header, false);
          if (decision.action ==
              memory_node_detail::PeerRequestAction::execute) {
            PeerStage1Task task;
            task.source_shard = peer_id;
            task.header = *header;
            task.dedup_lease = decision.lease;
            task.received_at = std::chrono::steady_clock::now();
            task.payload.assign(payload, payload + expected_bytes);
            if (!enqueue_peer_stage1_task(std::move(task))) {
              lib_assert(peer_request_deduplicator_->abandon(
                           decision.lease, peer_id, *header),
                         "Stage2 home request lost its dedup lease");
            }
          }
        }
      } else if (header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::stage1_execute_request) ||
                 header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::stage1_arm_request)) {
        const auto request_type = static_cast<
          service::storage_owner::PeerRpcType>(header->type);
        const size_t expected_bytes = request_type ==
            service::storage_owner::PeerRpcType::stage1_execute_request
          ? service::storage_owner::stage1_execute_request_bytes(
              header->item_count)
          : service::storage_owner::stage1_arm_request_bytes(
              header->item_count);
        if (header->item_count != 0 &&
            header->item_count <= config.storage_owner_batch_max &&
            header->reserved == 0 && bytes == expected_bytes) {
          const auto decision = peer_request_deduplicator_->begin(
            peer_id, *header, false);
          if (decision.action == memory_node_detail::PeerRequestAction::execute) {
            PeerStage1Task task;
            task.source_shard = peer_id;
            task.header = *header;
            task.dedup_lease = decision.lease;
            task.received_at = std::chrono::steady_clock::now();
            task.payload.assign(payload, payload + expected_bytes);
            if (!enqueue_peer_stage1_task(std::move(task))) {
              // Keep CQ progress nonblocking. The source retries the same ID;
              // abandoning here lets that retry execute once capacity returns.
              lib_assert(peer_request_deduplicator_->abandon(
                           decision.lease, peer_id, *header),
                         "Stage1 request lost its dedup lease");
              const bool sent = try_send_peer_stage1_retry_response(
                peer_id, *header,
                span<const byte_t>{payload, expected_bytes});
              (sent ? peer_stage1_admission_retry_responses_
                    : peer_stage1_retry_response_drops_)
                .fetch_add(1, std::memory_order_relaxed);
            }
          } else if (decision.action ==
                       memory_node_detail::PeerRequestAction::duplicate_inflight) {
            // The original request may be parked behind the bounded Stage2
            // credit window and owns the only semantic execution/dedup lease.
            // Coalesce a same-ID retransmission into its eventual late
            // response. Sending an immediate retry here changes a normal
            // at-most-once duplicate into a 10ms polling loop that steals the
            // CPU/CQ capacity needed to release that very window.
            peer_stage1_duplicate_coalesced_.fetch_add(
              1, std::memory_order_relaxed);
          } else if (decision.action ==
                       memory_node_detail::PeerRequestAction::full) {
            // No original execution owns a full-table rejection, so return an
            // explicit retry rather than making the caller wait for a response
            // that cannot arrive.
            const bool sent = try_send_peer_stage1_retry_response(
              peer_id, *header,
              span<const byte_t>{payload, expected_bytes});
            if (sent) {
              peer_stage1_admission_retry_responses_.fetch_add(
                1, std::memory_order_relaxed);
            } else {
              peer_stage1_retry_response_drops_.fetch_add(
                1, std::memory_order_relaxed);
            }
          }
        }
      } else if (header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::stage2_expand_score_response)) {
        const bool valid_response = header->item_count != 0 &&
          header->item_count <= config.storage_owner_batch_max &&
          header->reserved == 0 &&
          bytes == service::storage_owner::stage2_expand_score_response_bytes(
            header->item_count);
        if (valid_response && peer_async_responses_ != nullptr &&
            peer_async_responses_->try_deliver(
              peer_id, slot_id, bytes, *header)) {
          hold_receive_slot = true;
          peer_completion_cv_.notify_all();
          storage_owner_maintenance_cv_.notify_all();
        }
      } else if (header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::reconcile_reverse_response)) {
        const bool valid_response = header->item_count != 0 &&
          header->reserved == 0 &&
          bytes == service::storage_owner::reconcile_reverse_response_bytes(
            header->item_count);
        if (valid_response && peer_async_responses_ != nullptr &&
            peer_async_responses_->try_deliver(
              peer_id, slot_id, bytes, *header)) {
          hold_receive_slot = true;
          peer_completion_cv_.notify_all();
          storage_owner_maintenance_cv_.notify_all();
        }
      } else if (header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::stage1_arm_response)) {
        const bool valid_response = header->item_count != 0 &&
          header->reserved == 0 &&
          bytes == service::storage_owner::stage1_arm_response_bytes(
            header->item_count);
        if (valid_response && peer_async_responses_ != nullptr &&
            peer_async_responses_->try_deliver(
              peer_id, slot_id, bytes, *header)) {
          hold_receive_slot = true;
          peer_completion_cv_.notify_all();
          storage_owner_maintenance_cv_.notify_all();
        }
      } else if (header->type == static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_response) ||
                 header->type == static_cast<u32>(service::storage_owner::PeerRpcType::cleanup_deleted_response) ||
                 header->type == static_cast<u32>(service::storage_owner::PeerRpcType::centroid_membership_response)) {
        const bool valid_response = header->item_count != 0 &&
          header->reserved == 0 &&
          bytes == service::storage_owner::reverse_update_response_bytes();
        if (valid_response && peer_async_responses_ != nullptr &&
            peer_async_responses_->try_deliver(
              peer_id, slot_id, bytes, *header)) {
          hold_receive_slot = true;
          peer_completion_cv_.notify_all();
          storage_owner_maintenance_cv_.notify_all();
        }
      } else if (header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::cleanup_activate_response) ||
                 header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::authority_placement_response) ||
                 header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::dynamic_node_control_response)) {
        const bool cleanup_response = header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::cleanup_activate_response);
        const bool allocation_response = header->type == static_cast<u32>(
          service::storage_owner::PeerRpcType::dynamic_node_control_response);
        const size_t expected_bytes = cleanup_response
          ? service::storage_owner::cleanup_activate_response_bytes(
              header->item_count)
          : allocation_response
            ? service::storage_owner::dynamic_node_control_response_bytes(
                header->item_count)
            : service::storage_owner::authority_placement_response_bytes(
                header->item_count);
        const bool valid_response = header->item_count != 0 &&
          header->item_count <= config.storage_owner_batch_max &&
          header->reserved == 0 && bytes == expected_bytes;
        if (valid_response && peer_async_responses_ != nullptr &&
            peer_async_responses_->try_deliver(
              peer_id, slot_id, bytes, *header)) {
          hold_receive_slot = true;
          peer_completion_cv_.notify_all();
          storage_owner_maintenance_cv_.notify_all();
        }
      } else if (header->type == static_cast<u32>(
                   service::storage_owner::PeerRpcType::stage1_execute_response)) {
        const bool valid_response = header->item_count != 0 &&
          header->item_count <= config.storage_owner_batch_max &&
          header->reserved == 0 &&
          bytes == service::storage_owner::stage1_execute_response_bytes(
            header->item_count);
        if (valid_response && peer_async_responses_ != nullptr &&
            peer_async_responses_->try_deliver(
              peer_id, slot_id, bytes, *header)) {
          hold_receive_slot = true;
          peer_completion_cv_.notify_all();
          storage_owner_maintenance_cv_.notify_all();
        }
      }

      if (!hold_receive_slot) repost_peer_rpc_receive(peer_id, slot_id);
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
      const u32 request_type = tasks.back().header.type;
      size_t coalesced_ops = tasks.back().item_count();
      const bool carries_per_op_results = request_type == static_cast<u32>(
        service::storage_owner::PeerRpcType::reconcile_reverse_request);
      while (!carries_per_op_results && !peer_reverse_tasks_.empty() &&
             coalesced_ops < config.storage_owner_reverse_coalesce_max) {
        if (peer_reverse_tasks_.front().header.type != request_type) {
          break;
        }
        const size_t next_ops =
          peer_reverse_tasks_.front().item_count();
        if (!tasks.empty() && coalesced_ops + next_ops > config.storage_owner_reverse_coalesce_max) {
          break;
        }
        tasks.push_back(std::move(peer_reverse_tasks_.front()));
        peer_reverse_tasks_.pop_front();
        coalesced_ops += next_ops;
      }
    }
    peer_reverse_tasks_cv_.notify_one();

    u64 processed_items = 0;
    for (const PeerReverseUpdateTask& task : tasks) {
      processed_items += task.item_count();
    }
    const bool reconcile_request = tasks.size() == 1 &&
      tasks.front().header.type == static_cast<u32>(
        service::storage_owner::PeerRpcType::reconcile_reverse_request);
    bool success = false;
    vec<service::storage_owner::ReconcileReverseResult> reconcile_results;
    if (reconcile_request) {
      success = reconcile_local_reverse_ops(
        span<const service::storage_owner::ReconcileReverseOp>{
          tasks.front().reconcile_ops}, config, reconcile_results);
    } else {
      success = apply_peer_reverse_update_tasks(tasks, config);
    }
    peer_reverse_update_processed_.fetch_add(tasks.size(), std::memory_order_relaxed);
    peer_reverse_update_items_processed_.fetch_add(
      processed_items, std::memory_order_relaxed);
    if (!success) {
      peer_reverse_update_failed_.fetch_add(1, std::memory_order_relaxed);
    }
    for (const PeerReverseUpdateTask& task : tasks) {
      if (reconcile_request) {
        lib_assert(reconcile_results.size() == task.reconcile_ops.size(),
                   "reverse reconciliation lost per-operation results");
        const size_t bytes =
          service::storage_owner::reconcile_reverse_response_bytes(
            task.header.item_count);
        vec<byte_t> response(bytes, 0);
        auto* response_header = reinterpret_cast<
          service::storage_owner::PeerRpcHeader*>(response.data());
        *response_header = make_peer_reverse_update_response(
          task.header, success);
        std::memcpy(
          service::storage_owner::reconcile_reverse_results(response.data()),
          reconcile_results.data(),
          reconcile_results.size() * sizeof(reconcile_results.front()));
        lib_assert(peer_request_deduplicator_->abandon(
                     task.dedup_lease, task.source_shard, task.header),
                   "reconcile completion lost its dedup lease");
        send_peer_rpc_message(
          task.source_shard, response.data(), response.size());
        continue;
      }
      const auto response_header = make_peer_reverse_update_response(
        task.header, success);
      lib_assert(peer_request_deduplicator_->complete(
                   task.dedup_lease, task.source_shard, task.header,
                   response_header),
                 "graph-update completion lost its dedup lease");
      enqueue_peer_reverse_update_response(task.source_shard, task.header, success);
    }
  }
}

void MemoryNode::peer_stage1_worker_loop(u32 worker_id) {
  current_storage_owner_thread_ = peer_stage1_worker_states_[worker_id].get();
  const Configuration& config = *storage_worker_config_;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(peer_stage1_tasks_mutex_);
      peer_stage1_tasks_cv_.wait(lock, [&]() {
        return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
               !peer_stage1_tasks_.empty() ||
               !peer_stage2_home_tasks_.empty();
      });
      if (peer_reverse_shutdown_.load(std::memory_order_acquire) &&
          peer_stage1_tasks_.empty() && peer_stage2_home_tasks_.empty()) {
        current_storage_owner_thread_ = nullptr;
        return;
      }
    }

    peer_stage1_active_workers_.fetch_add(1, std::memory_order_acq_rel);
    atomic_utils::CounterDecrementGuard active_slot(
      peer_stage1_active_workers_);

    PeerStage1Task task;
    {
      std::lock_guard<std::mutex> lock(peer_stage1_tasks_mutex_);
      if (peer_stage1_tasks_.empty() && peer_stage2_home_tasks_.empty()) {
        continue;
      }
      if (!peer_stage1_tasks_.empty()) {
        task = std::move(peer_stage1_tasks_.front());
        peer_stage1_tasks_.pop_front();
      } else {
        task = std::move(peer_stage2_home_tasks_.front());
        peer_stage2_home_tasks_.pop_front();
      }
    }
    peer_stage1_tasks_cv_.notify_one();

    const auto request_type = static_cast<
      service::storage_owner::PeerRpcType>(task.header.type);
    if (request_type ==
        service::storage_owner::PeerRpcType::stage2_expand_score_request) {
      (void)handle_peer_stage2_expand_score_request(
        task.source_shard, task.header, task.payload.data(), config);
      // The operation is read-only and generation fenced at the caller.  Do
      // not retain a large payload response in the generic dedup table;
      // retrying the identical request is semantically and physically safe.
      lib_assert(peer_request_deduplicator_->abandon(
                   task.dedup_lease, task.source_shard, task.header),
                 "Stage2 home completion lost its dedup lease");
      peer_stage1_processed_.fetch_add(1, std::memory_order_relaxed);
      peer_stage1_tasks_cv_.notify_one();
      continue;
    }
    bool release_barrier = false;
    if (request_type ==
        service::storage_owner::PeerRpcType::stage1_arm_request) {
      const auto* items = service::storage_owner::stage1_arm_items(
        task.payload.data());
      for (u32 item = 0; item < task.header.item_count; ++item) {
        release_barrier |= items[item].action == static_cast<u32>(
          service::storage_owner::Stage1ArmAction::release);
      }
    }
    lib_assert(task.source_shard < peer_stage1_completion_states_.size() &&
                 peer_stage1_completion_states_[task.source_shard] != nullptr &&
                 task.source_sequence != 0,
               "peer Stage1 task omitted its RC receive-order sequence");
    PeerOrderedCompletionState& completion =
      *peer_stage1_completion_states_[task.source_shard];
    const auto record_source_completion = [&]() {
      if (task.source_sequence_completed) return;
      {
        std::lock_guard<std::mutex> lock(completion.mutex);
        if (task.source_sequence == completion.completed_prefix + 1) {
          ++completion.completed_prefix;
          while (completion.completed_out_of_order.erase(
                   completion.completed_prefix + 1) != 0) {
            ++completion.completed_prefix;
          }
        } else {
          lib_assert(task.source_sequence > completion.completed_prefix + 1,
                     "peer Stage1 task completed its sequence twice");
          const bool inserted = completion.completed_out_of_order.insert(
            task.source_sequence).second;
          lib_assert(inserted,
                     "peer Stage1 out-of-order sequence completed twice");
        }
      }
      task.source_sequence_completed = true;
      completion.changed.notify_all();
    };
    bool release_quiesced = true;
    if (release_barrier) {
      const auto* items = service::storage_owner::stage1_arm_items(
        task.payload.data());
      for (u32 item = 0; item < task.header.item_count; ++item) {
        const Stage1OperationKey key{
          .authority_shard = task.source_shard,
          .source_client = items[item].token.source_client,
          .item_index = items[item].token.item_index,
          .client_batch_id = items[item].token.client_batch_id,
        };
        if (!stage1_inflight_quiescent(key)) {
          release_quiesced = false;
          break;
        }
      }
    }
    bool success = false;
    bool admission_deferred = false;
    if (release_barrier && !release_quiesced) {
      peer_stage1_release_deferred_batches_.fetch_add(
        1, std::memory_order_relaxed);
      peer_stage1_release_deferred_items_.fetch_add(
        task.header.item_count, std::memory_order_relaxed);
    }
    if (release_quiesced && request_type ==
        service::storage_owner::PeerRpcType::stage1_execute_request) {
      success = handle_peer_stage1_execute_request(
        task.source_shard, task.header, task.payload.data(), config,
        &admission_deferred);
    } else if (request_type ==
               service::storage_owner::PeerRpcType::stage1_arm_request) {
      success = handle_peer_stage1_arm_request(
        task.source_shard, task.header,
        service::storage_owner::stage1_arm_items(task.payload.data()),
        release_quiesced, config);
    }
    const bool had_admission_wake_coverage =
      task.admission_wake_coverage != 0;
    const auto release_admission_wake_coverage_locked = [&]() {
      if (task.admission_wake_coverage == 0) return;
      lib_assert(peer_stage1_admission_wake_coverage_ >=
                   task.admission_wake_coverage,
                 "Stage1 runnable waiter coverage underflow");
      peer_stage1_admission_wake_coverage_ -=
        task.admission_wake_coverage;
      task.admission_wake_coverage = 0;
    };
    if (admission_deferred) {
      // Retain the exact request/dedup lease and semantic in-flight tokens.
      // The completion sequence is diagnostic only, and must be retired now
      // so one saturated physical home cannot grow its out-of-order set behind
      // a long-lived parked request.
      const bool reparked = task.admission_waiter_owned;
      record_source_completion();
      bool parked = false;
      size_t waiter_count = 0;
      {
        std::lock_guard<std::mutex> lock(peer_stage1_tasks_mutex_);
        // Drop this request's scheduler baton in the same critical section
        // that republishes it as a waiter. No younger waiter can observe the
        // transient uncovered credit and overtake it in between.
        release_admission_wake_coverage_locked();
        const size_t waiter_item_limit = std::max<size_t>(
          1, storage_owner_maintenance_admission_limit_);
        const size_t task_items = task.header.item_count;
        const bool owns_waiter_items = task.admission_waiter_owned;
        if (!peer_reverse_shutdown_.load(std::memory_order_acquire) &&
            !storage_owner_maintenance_shutdown_.load(
              std::memory_order_acquire) &&
            task_items <= waiter_item_limit &&
            (owns_waiter_items ||
             peer_stage1_admission_owned_items_ <=
               waiter_item_limit - task_items)) {
          if (!owns_waiter_items) {
            task.admission_waiter_owned = true;
            peer_stage1_admission_owned_items_ += task_items;
          }
          peer_stage1_admission_waiters_.push_back(std::move(task));
          peer_stage1_admission_waiter_items_ += task_items;
          peer_stage1_admission_waiter_items_hint_.store(
            peer_stage1_admission_waiter_items_,
            std::memory_order_release);
          waiter_count = peer_stage1_admission_waiters_.size();
          parked = true;
        }
      }
      if (parked) {
        peer_stage1_admission_parked_.fetch_add(
          1, std::memory_order_relaxed);
        if (reparked) {
          peer_stage1_admission_reparked_.fetch_add(
            1, std::memory_order_relaxed);
        }
        atomic_utils::update_max_relaxed(
          peer_stage1_max_admission_waiters_,
          static_cast<u64>(waiter_count));
        // Do not abandon the dedup lease or finish operation_tokens. A same-ID
        // retry is coalesced, and the eventual late success remains fenced by
        // the original in-flight ownership.
        // Recheck after parking to close the lost-wakeup race in which the
        // final completion/queue-pop edge happened between arm failure and
        // insertion into the waiter deque. The wake routine accounts runnable
        // soft coverage, so this cannot create a completion-burst stampede.
        wake_peer_stage1_admission_waiters();
        continue;
      }

      // The waiter pool is intentionally bounded independently of the large
      // peer dedup table so Stage2 reverse/placement traffic always retains
      // headroom. Fall back to an explicit retry without losing the original
      // request's cleanup path.
      const bool sent = try_send_peer_stage1_retry_response(
        task.source_shard, task.header,
        span<const byte_t>{task.payload.data(), task.payload.size()});
      (sent ? peer_stage1_admission_retry_responses_
            : peer_stage1_retry_response_drops_)
        .fetch_add(1, std::memory_order_relaxed);
    }
    if (task.admission_wake_coverage != 0) {
      std::lock_guard<std::mutex> lock(peer_stage1_tasks_mutex_);
      release_admission_wake_coverage_locked();
    }
    // Stage1 payloads live in the bounded operation table and arm is
    // idempotent there, so a same-ID retry can be executed safely. Large
    // payload responses are deliberately not copied into the generic dedup
    // cache.
    lib_assert(peer_request_deduplicator_->abandon(
                 task.dedup_lease, task.source_shard, task.header),
               "Stage1 completion lost its dedup lease");
    for (const auto& token : task.operation_tokens) {
      const Stage1OperationKey key{
        .authority_shard = task.source_shard,
        .source_client = token.source_client,
        .item_index = token.item_index,
        .client_batch_id = token.client_batch_id,
      };
      finish_stage1_inflight_request(key);
    }
    if (task.admission_waiter_owned) {
      std::lock_guard<std::mutex> lock(peer_stage1_tasks_mutex_);
      const size_t task_items = task.header.item_count;
      lib_assert(peer_stage1_admission_owned_items_ >= task_items,
                 "Stage1 semantic waiter item account underflow");
      peer_stage1_admission_owned_items_ -= task_items;
      task.admission_waiter_owned = false;
    }
    record_source_completion();
    // `processed` counts consumed RPCs, including an explicit retry response.
    // The handlers count each per-token ok result directly, including mixed
    // responses; charging item_count only when the aggregate bool was true
    // made partial progress invisible and distorted retry amplification.
    peer_stage1_processed_.fetch_add(1, std::memory_order_relaxed);
    (void)success;
    if (had_admission_wake_coverage) {
      // A woken request can resolve without consuming its advertised soft
      // coverage (for example a structural retry during shutdown). Pass any
      // still-visible credit to the next FIFO waiter.
      wake_peer_stage1_admission_waiters();
    }
    peer_stage1_tasks_cv_.notify_one();
  }
}

void MemoryNode::peer_reverse_response_loop() {
  for (;;) {
    PeerReverseUpdateResponse response;
    lib_assert(peer_reverse_responses_ != nullptr,
               "peer reverse response queue is not initialized");
    if (!peer_reverse_responses_->pop_wait(
          response, peer_reverse_workers_done_)) {
      peer_reverse_response_done_.store(true, std::memory_order_release);
      return;
    }
    send_peer_reverse_update_response(response);
  }
}
