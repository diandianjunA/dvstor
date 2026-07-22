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
      if (is_request &&
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
                       memory_node_detail::PeerRequestAction::duplicate_inflight ||
                     decision.action ==
                       memory_node_detail::PeerRequestAction::full) {
            // Never make an idempotent caller discover transient receiver
            // pressure only through a 500 ms attempt timeout. A token-complete
            // retry response is nonblocking and races safely with the original
            // success response in the same response registry.
            const bool sent = try_send_peer_stage1_retry_response(
              peer_id, *header,
              span<const byte_t>{payload, expected_bytes});
            if (sent) {
              (decision.action ==
                   memory_node_detail::PeerRequestAction::duplicate_inflight
                 ? peer_stage1_duplicate_retry_responses_
                 : peer_stage1_admission_retry_responses_)
                .fetch_add(1, std::memory_order_relaxed);
            } else {
              peer_stage1_retry_response_drops_.fetch_add(
                1, std::memory_order_relaxed);
            }
          }
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
               !peer_stage1_tasks_.empty();
      });
      if (peer_reverse_shutdown_.load(std::memory_order_acquire) &&
          peer_stage1_tasks_.empty()) {
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
      if (peer_stage1_tasks_.empty()) {
        continue;
      }
      task = std::move(peer_stage1_tasks_.front());
      peer_stage1_tasks_.pop_front();
    }
    peer_stage1_tasks_cv_.notify_one();

    const auto request_type = static_cast<
      service::storage_owner::PeerRpcType>(task.header.type);
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
    if (release_barrier && !release_quiesced) {
      peer_stage1_release_deferred_batches_.fetch_add(
        1, std::memory_order_relaxed);
      peer_stage1_release_deferred_items_.fetch_add(
        task.header.item_count, std::memory_order_relaxed);
    }
    if (release_quiesced && request_type ==
        service::storage_owner::PeerRpcType::stage1_execute_request) {
      success = handle_peer_stage1_execute_request(
        task.source_shard, task.header, task.payload.data(), config);
    } else if (request_type ==
               service::storage_owner::PeerRpcType::stage1_arm_request) {
      success = handle_peer_stage1_arm_request(
        task.source_shard, task.header,
        service::storage_owner::stage1_arm_items(task.payload.data()),
        release_quiesced, config);
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
    completion.changed.notify_all();
    // `processed` counts consumed RPCs, including an explicit retry response.
    // The handlers count each per-token ok result directly, including mixed
    // responses; charging item_count only when the aggregate bool was true
    // made partial progress invisible and distorted retry amplification.
    peer_stage1_processed_.fetch_add(1, std::memory_order_relaxed);
    (void)success;
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
