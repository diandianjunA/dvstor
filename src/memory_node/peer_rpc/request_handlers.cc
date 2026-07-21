#include "memory_node/peer_rpc/detail.hh"

bool MemoryNode::apply_peer_reverse_update_tasks(const vec<PeerReverseUpdateTask>& tasks, const Configuration& config) {
  if (tasks.empty()) {
    return true;
  }

  const auto apply_started = std::chrono::steady_clock::now();
  const auto request_type = static_cast<service::storage_owner::PeerRpcType>(
    tasks.front().header.type);
  lib_assert(request_type == service::storage_owner::PeerRpcType::reverse_update_request ||
               request_type == service::storage_owner::PeerRpcType::cleanup_deleted_request ||
               request_type == service::storage_owner::PeerRpcType::reconcile_reverse_request ||
               request_type == service::storage_owner::PeerRpcType::centroid_membership_request,
             "invalid peer graph-update request type");
  size_t item_count = 0;
  for (const PeerReverseUpdateTask& task : tasks) {
    lib_assert(task.header.type == tasks.front().header.type,
               "mixed peer graph-update request types in one apply batch");
    item_count += task.item_count();
  }
  if (request_type ==
      service::storage_owner::PeerRpcType::reconcile_reverse_request) {
    vec<service::storage_owner::ReconcileReverseOp> ops;
    ops.reserve(item_count);
    for (const PeerReverseUpdateTask& task : tasks) {
      ops.insert(ops.end(), task.reconcile_ops.begin(),
                 task.reconcile_ops.end());
    }
    vec<service::storage_owner::ReconcileReverseResult> results;
    return reconcile_local_reverse_ops(
      span<const service::storage_owner::ReconcileReverseOp>{ops},
      config, results);
  }
  if (request_type ==
      service::storage_owner::PeerRpcType::centroid_membership_request) {
    vec<service::storage_owner::CentroidMembershipOp> ops;
    ops.reserve(item_count);
    for (const PeerReverseUpdateTask& task : tasks) {
      ops.insert(ops.end(), task.centroid_ops.begin(),
                 task.centroid_ops.end());
    }
    return apply_local_centroid_membership_ops(
      span<const service::storage_owner::CentroidMembershipOp>{ops});
  }

  if (request_type ==
      service::storage_owner::PeerRpcType::cleanup_deleted_request) {
    vec<service::storage_owner::ReverseUpdateOp> ops;
    ops.reserve(item_count);
    for (const PeerReverseUpdateTask& task : tasks) {
      ops.insert(ops.end(), task.ops.begin(), task.ops.end());
    }
    return remove_local_neighbors_identity_fenced(
      span<const service::storage_owner::ReverseUpdateOp>{ops}, config);
  }

  dense_hashmap_t<u64, vec<RemotePtr>> grouped;
  grouped.reserve(item_count);
  for (const PeerReverseUpdateTask& task : tasks) {
    for (const auto& op : task.ops) {
      const RemotePtr target{op.target_raw};
      const RemotePtr candidate{op.candidate_raw};
      // Peer payloads are a trust boundary even when all healthy stage2
      // senders generate aligned pointers. Reject a wrong-shard, unaligned,
      // out-of-range, or not-yet-allocated target before acquiring a node
      // lock; malformed/mixed-version traffic must not become a local OOB.
      if (!valid_local_storage_node_pointer(target)) {
        return false;
      }
      grouped[target.raw_address].push_back(candidate);
    }
  }

  const bool success = apply_local_reverse_updates_batched(grouped, config);
  const u64 apply_ns = elapsed_ns_since(apply_started);
  if (apply_ns > 1000ull * 1000ull * 1000ull) {
    static std::atomic<u32> slow_apply_logs{0};
    const u32 log_index = slow_apply_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-peer] slow graph-update apply"
                << " self_shard=" << storage_id_
                << " rpc_type=" << static_cast<u32>(request_type)
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

bool MemoryNode::enqueue_peer_reverse_update_task(PeerReverseUpdateTask&& task) {
  const u64 item_count = task.item_count();
  size_t queue_size = 0;
  std::unique_lock<std::mutex> lock(peer_reverse_tasks_mutex_);
  if (peer_reverse_shutdown_.load(std::memory_order_acquire) ||
      peer_reverse_tasks_.size() >= peer_reverse_task_queue_limit_) {
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

bool MemoryNode::enqueue_peer_stage1_task(PeerStage1Task&& task) {
  using namespace service::storage_owner;
  const auto request_type = static_cast<PeerRpcType>(task.header.type);
  task.operation_tokens.clear();
  task.operation_tokens.reserve(task.header.item_count);
  if (request_type == PeerRpcType::stage1_execute_request) {
    const Stage1ExecuteItem* items = stage1_execute_items(
      task.payload.data());
    for (u32 item = 0; item < task.header.item_count; ++item) {
      task.operation_tokens.push_back(AuthorityOperationToken{
        .source_client = items[item].source_client,
        .item_index = items[item].item_index,
        .client_batch_id = items[item].client_batch_id,
      });
    }
  } else if (request_type == PeerRpcType::stage1_arm_request) {
    const Stage1ArmItem* items = stage1_arm_items(task.payload.data());
    bool saw_release = false;
    bool saw_non_release = false;
    for (u32 item = 0; item < task.header.item_count; ++item) {
      const auto action = static_cast<Stage1ArmAction>(items[item].action);
      if (action != Stage1ArmAction::arm &&
          action != Stage1ArmAction::abort &&
          action != Stage1ArmAction::release) {
        return false;
      }
      saw_release |= action == Stage1ArmAction::release;
      saw_non_release |= action != Stage1ArmAction::release;
      // A release is a quiescence observer, not work that can recreate or
      // mutate the receipt. Excluding it prevents duplicate releases from
      // waiting on one another forever.
      if (action != Stage1ArmAction::release) {
        task.operation_tokens.push_back(items[item].token);
      }
    }
    // Production callers send homogeneous control batches. Reject a mixed
    // release/mutation message at the trust boundary so its wait semantics
    // cannot be ambiguous.
    if (saw_release && saw_non_release) return false;
  } else {
    return false;
  }

  std::lock_guard<std::mutex> lock(peer_stage1_tasks_mutex_);
  if (peer_reverse_shutdown_.load(std::memory_order_acquire) ||
      peer_stage1_tasks_.size() >= peer_stage1_task_queue_limit_ ||
      task.source_shard >= peer_stage1_next_source_sequences_.size() ||
      peer_stage1_next_source_sequences_[task.source_shard] ==
        std::numeric_limits<u64>::max()) {
    return false;
  }
  vec<Stage1OperationKey> tracked_keys;
  tracked_keys.reserve(task.operation_tokens.size());
  for (const AuthorityOperationToken& token : task.operation_tokens) {
    const Stage1OperationKey key{
      .authority_shard = task.source_shard,
      .source_client = token.source_client,
      .item_index = token.item_index,
      .client_batch_id = token.client_batch_id,
    };
    if (!try_track_stage1_inflight_request(key)) {
      for (auto position = tracked_keys.rbegin();
           position != tracked_keys.rend(); ++position) {
        finish_stage1_inflight_request(*position);
      }
      // Reject the entire RPC before assigning a receive-order sequence.
      // The authority retains the same semantic token and retries after
      // bounded Stage1 state drains.
      return false;
    }
    tracked_keys.push_back(key);
  }
  task.source_sequence =
    ++peer_stage1_next_source_sequences_[task.source_shard];
  peer_stage1_tasks_.push_back(std::move(task));
  peer_stage1_enqueued_.fetch_add(1, std::memory_order_relaxed);
  atomic_utils::update_max_relaxed(
    peer_stage1_max_queue_,
    static_cast<u64>(peer_stage1_tasks_.size()));
  peer_stage1_tasks_cv_.notify_one();
  return true;
}

void MemoryNode::enqueue_peer_reverse_update_response(u32 destination_shard,
                                                      const service::storage_owner::PeerRpcHeader& request,
                                                      bool success) {
  PeerReverseUpdateResponse response;
  response.destination_shard = destination_shard;
  response.header = make_peer_reverse_update_response(request, success);
  response.queued_at = std::chrono::steady_clock::now();
  (void)try_enqueue_peer_reverse_update_response(std::move(response));
}

bool MemoryNode::try_enqueue_peer_reverse_update_response(
    PeerReverseUpdateResponse&& response) {
  if (peer_reverse_responses_ != nullptr &&
      peer_reverse_responses_->try_push(std::move(response))) {
    return true;
  }
  // A successful reverse operation remains in the bounded receiver dedup
  // cache. Dropping only this ACK is safe: the source retries the identical
  // request ID and receives a replay without applying the graph operation
  // twice. Failed operations are retryable by definition as well.
  static std::atomic<u32> dropped_response_logs{0};
  const u32 log_index = dropped_response_logs.fetch_add(
    1, std::memory_order_relaxed);
  if (log_index < 16) {
    std::cerr << "[storage-peer] bounded response queue full; "
                 "dropping ACK for same-ID replay" << std::endl;
  }
  return false;
}
