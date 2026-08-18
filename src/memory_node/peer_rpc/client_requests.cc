#include <stdexcept>

#include "memory_node/peer_rpc/detail.hh"
#include "memory_node/peer_rpc/stage1_control_fanout_policy.hh"
#include "memory_node/storage_owner_index/reconcile_reverse_policy.hh"

u64 MemoryNode::allocate_peer_request_id() {
  for (;;) {
    const u64 request_id = next_peer_request_id_.fetch_add(
      1, std::memory_order_relaxed);
    if (request_id != 0) return request_id;
  }
}

bool MemoryNode::try_post_peer_rpc_request_attempt(
    u32 target_shard,
    service::storage_owner::PeerRpcType request_type,
    service::storage_owner::PeerRpcType response_type,
    u64 request_id,
    u32 item_count,
    const void* items,
    size_t item_bytes,
    size_t request_bytes,
    PeerRpcSendClass send_class) {
  using service::storage_owner::PeerRpcHeader;
  if (target_shard >= num_storage_nodes_ || target_shard == storage_id_ ||
      request_id == 0 || item_count == 0 || items == nullptr ||
      request_bytes < sizeof(PeerRpcHeader) ||
      request_bytes > peer_rpc_runtime_.message_bytes ||
      item_bytes != request_bytes - sizeof(PeerRpcHeader) ||
      peer_async_responses_ == nullptr) {
    return false;
  }

  u32 maintenance_wake_owner =
    memory_node_detail::kNoMaintenanceWakeOwner;
  if (current_storage_owner_maintenance_worker_) {
    lib_assert(current_storage_owner_thread_ != nullptr,
               "maintenance RPC registration has no executor state");
    maintenance_wake_owner = current_storage_owner_thread_->id;
  }
  const auto registration = peer_async_responses_->register_send_attempt(
    request_id, target_shard, response_type, item_count,
    maintenance_wake_owner);
  if (registration ==
      memory_node_detail::PeerResponseRegistration::already_complete) {
    return true;
  }
  if (registration !=
        memory_node_detail::PeerResponseRegistration::registered &&
      registration != memory_node_detail::PeerResponseRegistration::retry) {
    return false;
  }

  u32 slot_id = 0;
  if (!try_acquire_peer_rpc_send_slot(target_shard, send_class, slot_id)) {
    return false;
  }
  const size_t offset = peer_rpc_async_send_offset(target_shard, slot_id);
  byte_t* message = peer_rpc_runtime_.buffer.get_full_buffer() + offset;
  std::memset(message, 0, request_bytes);
  auto* header = reinterpret_cast<PeerRpcHeader*>(message);
  header->magic = service::storage_owner::kPeerRpcMagic;
  header->version = service::storage_owner::kPeerRpcVersion;
  header->type = static_cast<u32>(request_type);
  header->source_shard = storage_id_;
  header->item_count = item_count;
  header->request_id = request_id;
  std::memcpy(message + sizeof(*header), items, item_bytes);
  post_peer_rpc_send_slot(target_shard, slot_id, request_bytes);
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
  return try_post_peer_rpc_request_attempt(
    target_shard, request_type, response_type, request_id, item_count,
    ops.data(), static_cast<size_t>(item_count) * sizeof(ops[0]), bytes,
    PeerRpcSendClass::graph_update);
}

MemoryNode::TryPeerResponse MemoryNode::try_consume_peer_rpc_response(
    u64 request_id,
    u32 expected_shard,
    service::storage_owner::PeerRpcType expected_type,
    u32 expected_item_count,
    service::storage_owner::PeerRpcHeader& header,
    vec<byte_t>& payload,
    PeerResponseLease& lease) {
  lease = {};
  if (peer_async_responses_ == nullptr) return TryPeerResponse::stale;

  memory_node_detail::PeerResponseDescriptor response;
  const TryPeerResponse result = peer_async_responses_->try_take(
    request_id, expected_shard, expected_type, expected_item_count,
    response, lease);
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
    try {
      payload.assign(source, source + response.bytes);
    } catch (...) {
      // try_take() moved the registry entry into its consuming state.  Restore
      // both receive-WQ ownership and retryability before propagating an
      // allocation/copy exception; cancel() cannot reclaim a consuming lease.
      repost_peer_rpc_receive(response.peer_id, response.receive_slot);
      lib_assert(peer_async_responses_->mark_receive_reposted(lease),
                 "exceptional peer response copy lost its receive lease");
      lib_assert(peer_async_responses_->retry(lease),
                 "exceptional peer response copy could not be rearmed");
      lease = {};
      throw;
    }
  } else {
    payload.clear();
  }
  repost_peer_rpc_receive(response.peer_id, response.receive_slot);
  lib_assert(peer_async_responses_->mark_receive_reposted(lease),
             "peer response receive descriptor lost its lease");
  if (!valid_descriptor) {
    lib_assert(peer_async_responses_->retry(lease),
               "malformed peer response could not be rearmed");
    lease = {};
  }
  return valid_descriptor ? result : TryPeerResponse::failure;
}

bool MemoryNode::acknowledge_peer_rpc_response(PeerResponseLease lease) {
  return peer_async_responses_ != nullptr &&
    peer_async_responses_->ack_consumed(lease);
}

bool MemoryNode::rearm_peer_rpc_response(PeerResponseLease lease) {
  return peer_async_responses_ != nullptr &&
    peer_async_responses_->retry(lease);
}

bool MemoryNode::await_late_peer_rpc_response(PeerResponseLease lease) {
  return peer_async_responses_ != nullptr &&
    peer_async_responses_->await_late_delivery(lease);
}

void MemoryNode::cancel_peer_rpc_response(u64 request_id) {
  if (peer_async_responses_ == nullptr) return;
  const auto response = peer_async_responses_->cancel(request_id);
  if (response.has_value()) {
    repost_peer_rpc_receive(response->peer_id, response->receive_slot);
  }
}

bool MemoryNode::send_reconcile_reverse_fanout_and_wait(
    const dense_hashmap_t<
      u32, vec<service::storage_owner::ReconcileReverseOp>>& updates,
    vec<service::storage_owner::ReconcileReverseResult>& results,
    const Configuration& config) {
  using service::storage_owner::InsertStatus;
  using service::storage_owner::PeerRpcHeader;
  using service::storage_owner::PeerRpcType;
  using service::storage_owner::ReconcileReverseOp;
  using service::storage_owner::ReconcileReverseResult;

  results.clear();
  struct PendingResponse {
    u64 request_id{};
    u32 target_shard{};
    u32 item_count{};
    vec<ReconcileReverseOp> expected_ops;
    vec<ReconcileReverseResult> accepted_results;
    std::chrono::steady_clock::time_point deadline{};
    u32 attempts_started{};
    bool attempt_active{};
    bool posted{};
    bool complete{};
  };
  const size_t payload_bytes =
    peer_rpc_runtime_.message_bytes -
    sizeof(service::storage_owner::PeerRpcHeader);
  const u32 wire_capacity = static_cast<u32>(
    payload_bytes / sizeof(service::storage_owner::ReconcileReverseOp));
  lib_assert(wire_capacity > 0,
             "peer RPC slot cannot hold one reverse reconciliation op");
  vec<PendingResponse> pending;

  for (const auto& [target_shard, ops] : updates) {
    lib_assert(target_shard < num_storage_nodes_ &&
                 target_shard != storage_id_,
               "reverse reconciliation target must be remote");
    for (size_t begin = 0; begin < ops.size(); begin += wire_capacity) {
      const u32 item_count = static_cast<u32>(std::min<size_t>(
        wire_capacity, ops.size() - begin));
      PendingResponse response;
      response.request_id = allocate_peer_request_id();
      response.target_shard = target_shard;
      response.item_count = item_count;
      response.expected_ops.assign(
        ops.begin() + static_cast<std::ptrdiff_t>(begin),
        ops.begin() + static_cast<std::ptrdiff_t>(begin + item_count));
      response.accepted_results.reserve(item_count);
      pending.push_back(std::move(response));
    }
  }

  constexpr u32 kTransportAttempts = 3;
  const auto timeout =
    std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  bool success = true;
  size_t remaining = pending.size();
  vec<byte_t> payload;
  payload.reserve(peer_rpc_runtime_.message_bytes);
  // Drive every chunk in round-robin order. Completed responses retain an
  // RDMA receive slot until copied, so waiting on one request in list order
  // could otherwise let out-of-order responses exhaust a peer's receive WQ.
  while (remaining != 0 &&
         !peer_reverse_shutdown_.load(std::memory_order_acquire)) {
    bool made_progress = false;
    for (PendingResponse& response : pending) {
      if (response.complete) continue;
      auto now = std::chrono::steady_clock::now();
      if (!response.attempt_active) {
        if (response.attempts_started == kTransportAttempts) {
          cancel_peer_rpc_response(response.request_id);
          response.complete = true;
          --remaining;
          success = false;
          made_progress = true;
          continue;
        }
        ++response.attempts_started;
        response.attempt_active = true;
        response.posted = false;
        response.deadline = now + timeout;
      }

      if (!response.posted) {
        const size_t request_bytes =
          service::storage_owner::reconcile_reverse_request_bytes(
            response.item_count);
        response.posted = try_post_peer_rpc_request_attempt(
          response.target_shard, PeerRpcType::reconcile_reverse_request,
          PeerRpcType::reconcile_reverse_response, response.request_id,
          response.item_count, response.expected_ops.data(),
          response.expected_ops.size() * sizeof(response.expected_ops[0]),
          request_bytes, PeerRpcSendClass::graph_update);
        made_progress = response.posted || made_progress;
        now = std::chrono::steady_clock::now();
        if (!response.posted && now >= response.deadline) {
          cancel_peer_rpc_response(response.request_id);
          response.attempt_active = false;
          made_progress = true;
        }
        if (!response.posted) continue;
      }

      PeerRpcHeader response_header{};
      PeerResponseLease response_lease{};
      const TryPeerResponse state = try_consume_peer_rpc_response(
        response.request_id, response.target_shard,
        PeerRpcType::reconcile_reverse_response, response.item_count,
        response_header, payload, response_lease);
      now = std::chrono::steady_clock::now();
      if (state == TryPeerResponse::pending) {
        if (now >= response.deadline) {
          cancel_peer_rpc_response(response.request_id);
          response.attempt_active = false;
          response.posted = false;
          made_progress = true;
        }
        continue;
      }
      if (state == TryPeerResponse::stale) {
        response.attempt_active = false;
        response.posted = false;
        made_progress = true;
        continue;
      }

      const size_t expected_bytes =
        service::storage_owner::reconcile_reverse_response_bytes(
          response.item_count);
      bool valid = state == TryPeerResponse::success &&
        payload.size() == expected_bytes &&
        response_header.magic == service::storage_owner::kPeerRpcMagic &&
        response_header.version == service::storage_owner::kPeerRpcVersion &&
        response_header.type == static_cast<u32>(
          PeerRpcType::reconcile_reverse_response) &&
        response_header.source_shard == response.target_shard &&
        response_header.item_count == response.item_count &&
        response_header.request_id == response.request_id &&
        response_header.status == static_cast<u32>(InsertStatus::ok) &&
        response_header.reserved == 0;
      if (valid) {
        const auto* wire_results =
          service::storage_owner::reconcile_reverse_results(payload.data());
        for (u32 index = 0; index < response.item_count; ++index) {
          const ReconcileReverseResult& observed = wire_results[index];
          valid = observed.accepted <= 1 && observed.replaced <= 1 &&
            observed.removed <= 1 && observed.stale <= 1 &&
            observed.reserved == 0 &&
            memory_node_storage_owner_index_detail::
              reconcile_reverse_postcondition_holds(
                response.expected_ops[index], observed);
          if (!valid) break;
        }
        if (valid) {
          response.accepted_results.assign(
            wire_results, wire_results + response.item_count);
        }
      }

      if (valid) {
        lib_assert(acknowledge_peer_rpc_response(response_lease),
                   "validated reconcile response lost its registry lease");
        response.complete = true;
        --remaining;
      } else {
        if (response_lease.valid()) {
          lib_assert(rearm_peer_rpc_response(response_lease),
                     "invalid reconcile response lost its registry lease");
        }
        response.attempt_active = false;
        response.posted = false;
      }
      made_progress = true;
    }

    if (!made_progress && remaining != 0) {
      std::unique_lock<std::mutex> lock(peer_completion_mutex_);
      peer_completion_cv_.wait_for(lock, std::chrono::microseconds(100));
    }
  }

  for (PendingResponse& response : pending) {
    if (!response.complete) {
      cancel_peer_rpc_response(response.request_id);
      success = false;
    }
  }
  if (success) {
    for (const PendingResponse& response : pending) {
      results.insert(results.end(), response.accepted_results.begin(),
                     response.accepted_results.end());
    }
  } else {
    results.clear();
  }
  return success;
}

bool MemoryNode::apply_centroid_membership_fanout_and_wait(
    span<const service::storage_owner::CentroidMembershipOp> ops,
    const Configuration& config) {
  if (ops.empty()) return true;

  dense_hashmap_t<u32, vec<service::storage_owner::CentroidMembershipOp>>
    remote_ops;
  vec<service::storage_owner::CentroidMembershipOp> local_ops;
  for (const auto& op : ops) {
    const RemotePtr pointer{op.node_raw};
    if (pointer.is_null() || pointer.memory_node() >= num_storage_nodes_) {
      return false;
    }
    if (pointer.memory_node() == storage_id_) {
      local_ops.push_back(op);
    } else {
      remote_ops[pointer.memory_node()].push_back(op);
    }
  }

  // Apply local work before sending. The operation is idempotent through the
  // node's CENTROID_ACCOUNTED bit, so retrying after any remote failure is
  // safe and does not double-count the vector.
  if (!local_ops.empty() && !apply_local_centroid_membership_ops(local_ops)) {
    return false;
  }
  if (remote_ops.empty()) return true;

  struct PendingResponse {
    u64 request_id{};
    u32 target_shard{};
    u32 item_count{};
    vec<service::storage_owner::CentroidMembershipOp> items;
    std::chrono::steady_clock::time_point deadline{};
    u32 attempts_started{};
    bool attempt_active{};
    bool posted{};
    bool complete{};
  };
  const size_t payload_bytes = peer_rpc_runtime_.message_bytes -
    sizeof(service::storage_owner::PeerRpcHeader);
  const u32 wire_capacity = static_cast<u32>(
    payload_bytes / sizeof(service::storage_owner::CentroidMembershipOp));
  lib_assert(wire_capacity > 0,
             "peer RPC slot cannot hold one centroid membership op");
  vec<PendingResponse> pending;

  // Materialize every shard/chunk before driving the bounded async lanes so a
  // retry always reuses the exact same semantic payload.
  for (const auto& [target_shard, shard_ops] : remote_ops) {
    for (size_t begin = 0; begin < shard_ops.size();
         begin += wire_capacity) {
      const u32 item_count = static_cast<u32>(std::min<size_t>(
        wire_capacity, shard_ops.size() - begin));
      PendingResponse response;
      response.request_id = allocate_peer_request_id();
      response.target_shard = target_shard;
      response.item_count = item_count;
      response.items.assign(
        shard_ops.begin() + static_cast<std::ptrdiff_t>(begin),
        shard_ops.begin() + static_cast<std::ptrdiff_t>(begin + item_count));
      pending.push_back(std::move(response));
    }
  }

  using service::storage_owner::InsertStatus;
  using service::storage_owner::PeerRpcHeader;
  using service::storage_owner::PeerRpcType;
  constexpr u32 kTransportAttempts = 3;
  const auto timeout =
    std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  bool success = true;
  size_t remaining = pending.size();
  vec<byte_t> payload;
  payload.reserve(peer_rpc_runtime_.message_bytes);
  // Consume whichever response arrives first instead of holding later receive
  // slots while waiting for an earlier list element.
  while (remaining != 0 &&
         !peer_reverse_shutdown_.load(std::memory_order_acquire)) {
    bool made_progress = false;
    for (PendingResponse& response : pending) {
      if (response.complete) continue;
      auto now = std::chrono::steady_clock::now();
      if (!response.attempt_active) {
        if (response.attempts_started == kTransportAttempts) {
          cancel_peer_rpc_response(response.request_id);
          response.complete = true;
          --remaining;
          success = false;
          made_progress = true;
          continue;
        }
        ++response.attempts_started;
        response.attempt_active = true;
        response.posted = false;
        response.deadline = now + timeout;
      }

      if (!response.posted) {
        const size_t request_bytes =
          service::storage_owner::centroid_membership_request_bytes(
            response.item_count);
        response.posted = try_post_peer_rpc_request_attempt(
          response.target_shard, PeerRpcType::centroid_membership_request,
          PeerRpcType::centroid_membership_response, response.request_id,
          response.item_count, response.items.data(),
          response.items.size() * sizeof(response.items[0]), request_bytes,
          PeerRpcSendClass::graph_update);
        made_progress = response.posted || made_progress;
        now = std::chrono::steady_clock::now();
        if (!response.posted && now >= response.deadline) {
          cancel_peer_rpc_response(response.request_id);
          response.attempt_active = false;
          made_progress = true;
        }
        if (!response.posted) continue;
      }

      PeerRpcHeader response_header{};
      PeerResponseLease response_lease{};
      const TryPeerResponse state = try_consume_peer_rpc_response(
        response.request_id, response.target_shard,
        PeerRpcType::centroid_membership_response, response.item_count,
        response_header, payload, response_lease);
      now = std::chrono::steady_clock::now();
      if (state == TryPeerResponse::pending) {
        if (now >= response.deadline) {
          cancel_peer_rpc_response(response.request_id);
          response.attempt_active = false;
          response.posted = false;
          made_progress = true;
        }
        continue;
      }
      if (state == TryPeerResponse::stale) {
        response.attempt_active = false;
        response.posted = false;
        made_progress = true;
        continue;
      }

      const bool valid = state == TryPeerResponse::success &&
        payload.size() ==
          service::storage_owner::centroid_membership_response_bytes() &&
        response_header.magic == service::storage_owner::kPeerRpcMagic &&
        response_header.version == service::storage_owner::kPeerRpcVersion &&
        response_header.type == static_cast<u32>(
          PeerRpcType::centroid_membership_response) &&
        response_header.source_shard == response.target_shard &&
        response_header.item_count == response.item_count &&
        response_header.request_id == response.request_id &&
        response_header.status == static_cast<u32>(InsertStatus::ok) &&
        response_header.reserved == 0;
      if (valid) {
        lib_assert(acknowledge_peer_rpc_response(response_lease),
                   "validated centroid response lost its registry lease");
        response.complete = true;
        --remaining;
      } else {
        if (response_lease.valid()) {
          lib_assert(rearm_peer_rpc_response(response_lease),
                     "invalid centroid response lost its registry lease");
        }
        response.attempt_active = false;
        response.posted = false;
      }
      made_progress = true;
    }

    if (!made_progress && remaining != 0) {
      std::unique_lock<std::mutex> lock(peer_completion_mutex_);
      peer_completion_cv_.wait_for(lock, std::chrono::microseconds(100));
    }
  }

  for (PendingResponse& response : pending) {
    if (!response.complete) {
      cancel_peer_rpc_response(response.request_id);
      success = false;
    }
  }
  return success;
}

bool MemoryNode::execute_remote_stage1_fanout_and_wait(
    const dense_hashmap_t<
      u32, vec<service::storage_owner::Stage1ExecuteItem>>& items_by_home,
    const dense_hashmap_t<u32, vec<byte_t>>& vectors_by_home,
    dense_hashmap_t<
      u32, vec<service::storage_owner::Stage1ExecuteResult>>& results_by_home,
    const std::function<void(
      u32, span<const service::storage_owner::Stage1ExecuteItem>,
      span<const service::storage_owner::Stage1ExecuteResult>)>&
      on_home_resolved,
    const std::function<void(
      u32,
      span<const service::storage_owner::Stage1ArmItem>)>&
      on_home_release_resolved,
    const std::function<bool()>& overlap_work,
    const Configuration& config) {
  using namespace service::storage_owner;
  using memory_node_peer_rpc_detail::Stage1HomeRetryBackoff;
  const auto attempt_timeout =
    memory_node_peer_rpc_detail::stage1_peer_attempt_timeout(
      config.storage_owner_rpc_timeout_ms);
  results_by_home.clear();
  if (items_by_home.empty()) return !overlap_work || overlap_work();

  struct PendingStage1 {
    u32 home{};
    u64 request_id{};
    u32 item_count{};
    vec<byte_t> message;
    vec<byte_t> response_payload;
    // Maps each slot in the current compact wire wave back to the immutable
    // physical-home input order. retry_indices is populated only while an
    // ordered release for the just-committed subset is in flight.
    vec<u32> active_indices;
    vec<u32> retry_indices;
    // Reserved to the immutable home cardinality before any response lease is
    // registered. Partitioning a consumed response must be allocation-free:
    // cancel cannot reclaim a registry entry in the consuming state.
    vec<u32> resolved_slots_scratch;
    vec<u32> retry_slots_scratch;
    vec<Stage1ExecuteItem> resolved_items_scratch;
    vec<Stage1ExecuteResult> resolved_results_scratch;
    vec<Stage1ArmItem> release_items;
    vec<Stage1ArmItem> resolved_release_items_scratch;
    vec<Stage1ArmItem> compact_release_items_scratch;
    std::chrono::steady_clock::time_point deadline{};
    Stage1HomeRetryBackoff retry_backoff;
    bool posted{};
    bool execute_resolved{};
    bool resolved{};
  };
  vec<PendingStage1> pending;
  pending.reserve(items_by_home.size());
  vec<AuthorityOperationToken> seen_tokens;
  size_t semantic_token_count = 0;
  for (const auto& [home, items] : items_by_home) {
    (void)home;
    semantic_token_count += items.size();
  }
  seen_tokens.reserve(semantic_token_count);

  for (const auto& [home, items] : items_by_home) {
    const auto vectors_position = vectors_by_home.find(home);
    if (home >= num_storage_nodes_ || home == storage_id_ || items.empty() ||
        items.size() > config.storage_owner_batch_max ||
        vectors_position == vectors_by_home.end() ||
        vectors_position->second.size() !=
          items.size() * VamanaNode::vector_bytes()) {
      return false;
    }
    const bool fused_batch =
      memory_node_peer_rpc_detail::stage1_execute_uses_fused_arm(
        items.front());
    if (!memory_node_peer_rpc_detail::stage1_execute_tokens_unique(
          span<const Stage1ExecuteItem>{items})) {
      // A semantic token may appear in exactly one current subset. If it
      // appeared twice, committing one slot and retrying/releasing the other
      // would violate the Execute-before-release QP proof.
      return false;
    }
    for (size_t item_slot = 0; item_slot < items.size(); ++item_slot) {
      const Stage1ExecuteItem& item = items[item_slot];
      if (item.authority_shard != storage_id_ ||
          item.client_batch_id == 0 ||
          memory_node_peer_rpc_detail::stage1_execute_uses_fused_arm(item) !=
            fused_batch ||
          (fused_batch &&
           !memory_node_peer_rpc_detail::valid_fused_stage1_execute_item(
             item))) {
        return false;
      }
      const AuthorityOperationToken token{
        .source_client = item.source_client,
        .item_index = item.item_index,
        .client_batch_id = item.client_batch_id,
      };
      if (std::find_if(
            seen_tokens.begin(), seen_tokens.end(), [&](const auto& seen) {
              return seen.source_client == token.source_client &&
                seen.item_index == token.item_index &&
                seen.client_batch_id == token.client_batch_id;
            }) != seen_tokens.end()) {
        return false;
      }
      seen_tokens.push_back(token);
    }

    PendingStage1 request;
    request.home = home;
    request.request_id = allocate_peer_request_id();
    request.item_count = static_cast<u32>(items.size());
    request.active_indices.reserve(items.size());
    request.retry_indices.reserve(items.size());
    request.resolved_slots_scratch.reserve(items.size());
    request.retry_slots_scratch.reserve(items.size());
    request.resolved_items_scratch.reserve(items.size());
    request.resolved_results_scratch.reserve(items.size());
    request.release_items.reserve(items.size());
    request.resolved_release_items_scratch.reserve(items.size());
    request.compact_release_items_scratch.reserve(items.size());
    for (u32 index = 0; index < request.item_count; ++index) {
      request.active_indices.push_back(index);
    }
    request.message.resize(stage1_execute_request_bytes(request.item_count));
    if (request.message.size() > peer_rpc_runtime_.message_bytes) return false;
    // Reserve before the request ID becomes visible to the response registry.
    // Once try_take() leases a receive descriptor, copying its payload must be
    // allocation-free: an allocation failure in the consuming state cannot be
    // cancelled and would otherwise pin both the descriptor and registry slot.
    request.response_payload.reserve(peer_rpc_runtime_.message_bytes);
    auto* header = reinterpret_cast<PeerRpcHeader*>(request.message.data());
    header->magic = kPeerRpcMagic;
    header->version = kPeerRpcVersion;
    header->type = static_cast<u32>(PeerRpcType::stage1_execute_request);
    header->source_shard = storage_id_;
    header->item_count = request.item_count;
    header->request_id = request.request_id;
    std::memcpy(stage1_execute_items(request.message.data()), items.data(),
                items.size() * sizeof(Stage1ExecuteItem));
    std::memcpy(stage1_execute_vectors(
                  request.message.data(), request.item_count),
                vectors_position->second.data(),
                vectors_position->second.size());
    const auto [result_position, inserted] = results_by_home.emplace(
      home, vec<Stage1ExecuteResult>(items.size()));
    (void)result_position;
    lib_assert(inserted, "duplicate physical home in Stage1 fanout");
    pending.push_back(std::move(request));
  }

  const auto begin_compact_execute_wave = [&](PendingStage1& request) {
    lib_assert(!request.retry_indices.empty() && !request.posted,
               "Stage1 compact retry omitted its unresolved tokens");
    const auto& original_items = items_by_home.at(request.home);
    const auto& original_vectors = vectors_by_home.at(request.home);
    // Both vectors retain the immutable home cardinality reserved before the
    // first request is registered. Copy the compact mapping in-place instead
    // of moving its allocation away: response processing after a later ACK
    // must remain allocation-free as the subset shrinks repeatedly.
    request.active_indices.assign(
      request.retry_indices.begin(), request.retry_indices.end());
    request.retry_indices.clear();
    request.release_items.clear();
    request.execute_resolved = false;
    request.item_count = static_cast<u32>(request.active_indices.size());
    request.request_id = allocate_peer_request_id();
    request.message.assign(
      stage1_execute_request_bytes(request.item_count), byte_t{});
    auto* header = reinterpret_cast<PeerRpcHeader*>(request.message.data());
    header->magic = kPeerRpcMagic;
    header->version = kPeerRpcVersion;
    header->type = static_cast<u32>(PeerRpcType::stage1_execute_request);
    header->source_shard = storage_id_;
    header->item_count = request.item_count;
    header->request_id = request.request_id;
    Stage1ExecuteItem* compact_items = stage1_execute_items(
      request.message.data());
    byte_t* compact_vectors = stage1_execute_vectors(
      request.message.data(), request.item_count);
    for (u32 slot = 0; slot < request.item_count; ++slot) {
      const u32 original_index = request.active_indices[slot];
      lib_assert(original_index < original_items.size(),
                 "Stage1 compact retry lost its original item mapping");
      compact_items[slot] = original_items[original_index];
      std::memcpy(
        compact_vectors + static_cast<size_t>(slot) *
          VamanaNode::vector_bytes(),
        original_vectors.data() + static_cast<size_t>(original_index) *
          VamanaNode::vector_bytes(),
        VamanaNode::vector_bytes());
    }
    request.retry_backoff.reset();
  };

  // Stage1 owns a dedicated async send-credit lane. This keeps the distinct
  // single-home groups of one mutation batch in flight together and prevents
  // a foreground coordinator from serializing on the shared synchronous
  // response buffer. Each item still executes on exactly one centroid home.
  const auto try_post_execute = [&](const PendingStage1& request) {
    if (peer_async_responses_ == nullptr) return false;
    const auto registration = peer_async_responses_->register_send_attempt(
      request.request_id, request.home,
      PeerRpcType::stage1_execute_response, request.item_count);
    if (registration ==
        memory_node_detail::PeerResponseRegistration::already_complete) {
      return true;
    }
    if (registration !=
          memory_node_detail::PeerResponseRegistration::registered &&
        registration !=
          memory_node_detail::PeerResponseRegistration::retry) {
      return false;
    }

    u32 slot_id = 0;
    if (!try_acquire_peer_rpc_send_slot(
          request.home, PeerRpcSendClass::stage1, slot_id)) {
      return false;
    }
    const size_t offset = peer_rpc_async_send_offset(
      request.home, slot_id);
    std::memcpy(peer_rpc_runtime_.buffer.get_full_buffer() + offset,
                request.message.data(), request.message.size());
    post_peer_rpc_send_slot(
      request.home, slot_id, request.message.size());
    return true;
  };
  const auto try_post_release = [&](const PendingStage1& request) {
    lib_assert(request.execute_resolved && !request.release_items.empty(),
               "Stage1 release post omitted its resolved Execute home");
    const u32 release_count = static_cast<u32>(
      request.release_items.size());
    return try_post_peer_rpc_request_attempt(
      request.home, PeerRpcType::stage1_arm_request,
      PeerRpcType::stage1_arm_response, request.request_id, release_count,
      request.release_items.data(),
      request.release_items.size() * sizeof(request.release_items[0]),
      stage1_arm_request_bytes(release_count), PeerRpcSendClass::control);
  };

  // Prime every remote home before running authority-local Stage1 work.  The
  // physical-home workers can then search and publish in parallel with the
  // local subset of this batch.  A temporarily exhausted send lane is not an
  // error: the event loop below posts that home as soon as a slot completes.
  const auto initial_post_time = std::chrono::steady_clock::now();
  for (PendingStage1& request : pending) {
    if (!try_post_execute(request)) continue;
    request.posted = true;
    request.deadline = initial_post_time + attempt_timeout;
  }

  try {
    if (overlap_work && !overlap_work()) {
      for (const PendingStage1& request : pending) {
        cancel_peer_rpc_response(request.request_id);
      }
      return false;
    }
  } catch (...) {
    // Local failure must not leave response descriptors or retry identity
    // registered forever.  The semantic Stage1 receipts themselves remain
    // replayable and are handled by the public mutation retry.
    for (const PendingStage1& request : pending) {
      cancel_peer_rpc_response(request.request_id);
    }
    throw;
  }

  const auto cancel_unresolved = [&]() {
    for (const PendingStage1& request : pending) {
      if (!request.resolved) cancel_peer_rpc_response(request.request_id);
    }
  };

  // Progress every home in one event loop.  Consuming completed descriptors
  // promptly is part of flow control: a response holds its registered receive
  // WR until acknowledge_peer_rpc_response(), so waiting indefinitely on one
  // retrying home before inspecting the others can exhaust receive credits.
  // A fused home advances through Execute ACK -> authority callback -> ordered
  // release ACK without blocking this scan. Thus one slow receipt release
  // cannot delay another home's Execute ACK or authority commit. Per-phase
  // retries keep the same request ID and semantic item tokens.
  size_t remaining = pending.size();
  try {
    while (remaining != 0 &&
           !peer_reverse_shutdown_.load(std::memory_order_acquire) &&
           !storage_insert_shutdown_.load(std::memory_order_acquire)) {
      bool made_progress = false;
      const auto now = std::chrono::steady_clock::now();
      for (PendingStage1& request : pending) {
        if (request.resolved) continue;

        if (!request.posted) {
          if (!request.retry_backoff.ready(now)) continue;
          const bool posted = request.execute_resolved
            ? try_post_release(request) : try_post_execute(request);
          if (!posted) continue;
          request.posted = true;
          request.deadline = now + attempt_timeout;
          made_progress = true;
        }

        if (request.execute_resolved) {
          request.response_payload.clear();
          PeerResponseLease response_lease{};
          PeerRpcHeader response_header{};
          const u32 release_count = static_cast<u32>(
            request.release_items.size());
          const TryPeerResponse state = try_consume_peer_rpc_response(
            request.request_id, request.home,
            PeerRpcType::stage1_arm_response, release_count,
            response_header, request.response_payload, response_lease);
          if (state == TryPeerResponse::pending) {
            if (std::chrono::steady_clock::now() >= request.deadline) {
              request.posted = false;
              request.retry_backoff.schedule(
                std::chrono::steady_clock::now());
              made_progress = true;
            }
            continue;
          }
          if (state == TryPeerResponse::stale) {
            request.posted = false;
            request.retry_backoff.schedule(
              std::chrono::steady_clock::now());
            made_progress = true;
            continue;
          }

          const size_t expected_bytes =
            stage1_arm_response_bytes(release_count);
          const bool valid_envelope = state == TryPeerResponse::success &&
            request.response_payload.size() == expected_bytes &&
            response_header.magic == kPeerRpcMagic &&
            response_header.version == kPeerRpcVersion &&
            response_header.type == static_cast<u32>(
              PeerRpcType::stage1_arm_response) &&
            response_header.source_shard == request.home &&
            response_header.item_count == release_count &&
            response_header.request_id == request.request_id &&
            response_header.status == static_cast<u32>(InsertStatus::ok) &&
            response_header.reserved == 0;
          if (!valid_envelope) {
            if (response_lease.valid()) {
              (void)rearm_peer_rpc_response(response_lease);
            }
            throw std::runtime_error(
              "malformed ordered Stage1 release response envelope");
          }
          const Stage1ArmResult* wire = stage1_arm_results(
            request.response_payload.data());
          vec<u32>& resolved_release_slots =
            request.resolved_slots_scratch;
          vec<u32>& retry_release_slots = request.retry_slots_scratch;
          if (!memory_node_peer_rpc_detail::
                partition_stage1_control_response(
                  span<const Stage1ArmItem>{request.release_items},
                  span<const Stage1ArmResult>{wire, release_count},
                  resolved_release_slots, retry_release_slots)) {
            (void)rearm_peer_rpc_response(response_lease);
            throw std::runtime_error(
              "malformed token-fenced ordered Stage1 release response");
          }
          if (resolved_release_slots.empty()) {
            // An all-retry response may race an older same-ID operation. Keep
            // the request identity pending so its late successful response can
            // still satisfy this exact release item_count during backoff.
            lib_assert(await_late_peer_rpc_response(response_lease),
                       "retryable Stage1 release lost its response lease");
            request.posted = false;
            request.retry_backoff.schedule(
              std::chrono::steady_clock::now());
            made_progress = true;
            continue;
          }
          lib_assert(acknowledge_peer_rpc_response(response_lease),
                     "validated Stage1 release lost its response lease");
          vec<Stage1ArmItem>& resolved_release_items =
            request.resolved_release_items_scratch;
          resolved_release_items.clear();
          for (const u32 slot : resolved_release_slots) {
            resolved_release_items.push_back(request.release_items[slot]);
          }
          if (on_home_release_resolved) {
            on_home_release_resolved(
              request.home,
              span<const Stage1ArmItem>{resolved_release_items});
          }
          request.posted = false;
          if (!retry_release_slots.empty()) {
            vec<Stage1ArmItem>& compact_release_items =
              request.compact_release_items_scratch;
            compact_release_items.clear();
            for (const u32 slot : retry_release_slots) {
              compact_release_items.push_back(request.release_items[slot]);
            }
            request.release_items.assign(
              compact_release_items.begin(), compact_release_items.end());
            request.request_id = allocate_peer_request_id();
            request.retry_backoff.reset();
            made_progress = true;
            continue;
          }
          if (!request.retry_indices.empty()) {
            // The release ACK is the RC-QP watermark for the successful
            // subset. Start a new transport generation only now; semantic
            // tokens remain unchanged, while the fresh request ID prevents a
            // late full-wave response from aliasing this compact item_count.
            begin_compact_execute_wave(request);
          } else {
            request.resolved = true;
            --remaining;
          }
          made_progress = true;
          continue;
        }

        request.response_payload.clear();
        PeerResponseLease response_lease{};
        PeerRpcHeader response_header{};
        const TryPeerResponse state = try_consume_peer_rpc_response(
          request.request_id, request.home,
          PeerRpcType::stage1_execute_response, request.item_count,
          response_header, request.response_payload, response_lease);
        if (state == TryPeerResponse::pending) {
          if (std::chrono::steady_clock::now() >= request.deadline) {
            // Prepare is token-idempotent. Do not cancel or semantically abort
            // an uncertain request; repost the identical request ID so either
            // response resolves the same registry entry.
            request.posted = false;
            request.retry_backoff.schedule(
              std::chrono::steady_clock::now());
            made_progress = true;
          }
          continue;
        }
        if (state == TryPeerResponse::stale) {
          request.posted = false;
          request.retry_backoff.schedule(
            std::chrono::steady_clock::now());
          made_progress = true;
          continue;
        }

        // A failed aggregate header can still carry authoritative per-item
        // Stage1 statuses. Keep the descriptor leased until all payload
        // records have been validated.
        const size_t expected_bytes =
          stage1_execute_response_bytes(request.item_count);
        const auto* header = request.response_payload.size() == expected_bytes
          ? reinterpret_cast<const PeerRpcHeader*>(
              request.response_payload.data()) : nullptr;
        if (header == nullptr || header->magic != kPeerRpcMagic ||
            header->version != kPeerRpcVersion ||
            header->type != static_cast<u32>(
              PeerRpcType::stage1_execute_response) ||
            header->source_shard != request.home ||
            header->item_count != request.item_count ||
            header->request_id != request.request_id ||
            (header->status != static_cast<u32>(InsertStatus::ok) &&
             header->status != static_cast<u32>(InsertStatus::overloaded)) ||
            header->reserved != 0) {
          if (response_lease.valid()) {
            (void)rearm_peer_rpc_response(response_lease);
          }
          throw std::runtime_error(
            "invalid Stage1 response under an uncertain prepare");
        }

        const auto* output = stage1_execute_results(
          request.response_payload.data());
        const Stage1ExecuteItem* active_input = stage1_execute_items(
          request.message.data());
        vec<u32>& resolved_slots = request.resolved_slots_scratch;
        vec<u32>& retry_slots = request.retry_slots_scratch;
        if (!memory_node_peer_rpc_detail::partition_stage1_execute_response(
              span<const Stage1ExecuteItem>{
                active_input, request.item_count},
              span<const Stage1ExecuteResult>{output, request.item_count},
              request.home,
              resolved_slots, retry_slots)) {
          if (response_lease.valid()) {
            (void)rearm_peer_rpc_response(response_lease);
          }
          throw std::runtime_error(
            "malformed Stage1 result under an uncertain prepare");
        }
        request.posted = false;
        made_progress = true;

        // A CQ-level duplicate/admission retry is intentionally all-retry.
        // Rearm, rather than ACK + re-register, so the original handler can
        // deliver a late successful response without falling into a registry
        // gap. Keep the same transport generation and exact item_count.
        if (resolved_slots.empty()) {
          lib_assert(await_late_peer_rpc_response(response_lease),
                     "all-retry Stage1 Execute lost its response lease");
          request.retry_backoff.schedule(
            std::chrono::steady_clock::now());
          continue;
        }

        // A validated mixed/resolved descriptor is in the registry's consuming
        // state. It cannot be made retryable by cancel/re-register, so ACK loss
        // is an invariant violation rather than a transport transition.
        lib_assert(acknowledge_peer_rpc_response(response_lease),
                   "validated partial Stage1 Execute lost its response lease");

        const auto& original_input = items_by_home.at(request.home);
        vec<Stage1ExecuteResult>& full_results =
          results_by_home.at(request.home);
        vec<Stage1ExecuteItem>& resolved_items =
          request.resolved_items_scratch;
        vec<Stage1ExecuteResult>& resolved_results =
          request.resolved_results_scratch;
        resolved_items.clear();
        resolved_results.clear();
        request.retry_indices.clear();
        request.retry_indices.reserve(retry_slots.size());
        for (const u32 slot : retry_slots) {
          lib_assert(slot < request.active_indices.size(),
                     "Stage1 retry slot escaped its compact wave");
          request.retry_indices.push_back(request.active_indices[slot]);
        }
        for (const u32 slot : resolved_slots) {
          lib_assert(slot < request.active_indices.size(),
                     "Stage1 resolved slot escaped its compact wave");
          const u32 original_index = request.active_indices[slot];
          lib_assert(original_index < original_input.size() &&
                       original_index < full_results.size(),
                     "Stage1 result lost its original home mapping");
          full_results[original_index] = output[slot];
          resolved_items.push_back(original_input[original_index]);
          resolved_results.push_back(output[slot]);
          Stage1ArmItem release;
          if (memory_node_peer_rpc_detail::make_fused_stage1_release_item(
                original_input[original_index], output[slot],
                release)) {
            request.release_items.push_back(release);
          }
        }

        // Linearize every resolved token immediately. A retrying sibling is
        // only a member of the next transport wave and cannot reopen these
        // per-item authority commits. The callback is synchronous on this
        // foreground coordinator, never on a CQ thread.
        if (on_home_resolved) {
          on_home_resolved(
            request.home, span<const Stage1ExecuteItem>{resolved_items},
            span<const Stage1ExecuteResult>{resolved_results});
        }

        // The authority callback has committed every successful fused token.
        // Its release is posted on the same RC QP before a compact Execute
        // generation can reuse the home, preserving the ordered receipt
        // watermark without delaying unrelated homes in this event loop.
        if (request.release_items.empty()) {
          if (!request.retry_indices.empty()) {
            begin_compact_execute_wave(request);
          } else {
            request.resolved = true;
            --remaining;
          }
          continue;
        }
        request.execute_resolved = true;
        request.request_id = allocate_peer_request_id();
        request.retry_backoff.reset();
      }

      if (!made_progress && remaining != 0) {
        std::unique_lock<std::mutex> lock(peer_completion_mutex_);
        peer_completion_cv_.wait_for(lock, std::chrono::microseconds(100));
      }
    }
  } catch (...) {
    cancel_unresolved();
    throw;
  }

  cancel_unresolved();
  return remaining == 0;
}

bool MemoryNode::arm_remote_stage1_batch(
    u32 stage1_home,
    u32 source_client,
    span<const service::storage_owner::Stage1ArmItem> items,
    vec<service::storage_owner::Stage1ArmResult>& results,
    const Configuration& config) {
  using namespace service::storage_owner;
  results.clear();
  if (stage1_home >= num_storage_nodes_ || stage1_home == storage_id_ ||
      items.empty() || items.size() > config.storage_owner_batch_max) {
    return false;
  }
  for (const Stage1ArmItem& item : items) {
    if (item.token.source_client != source_client) return false;
  }

  // Abort/release recovery used to run a separate three-attempt synchronous
  // path whose *individual* waits each consumed the full public 30 s timeout.
  // Reuse the normal per-home state machine so recovery has the same bounded
  // attempt deadline, same-ID replay, reserved control credit, and backoff as
  // the hot path. This also keeps one transport correctness contract for all
  // Stage1 control actions.
  dense_hashmap_t<u32, vec<Stage1ArmItem>> items_by_home;
  items_by_home[stage1_home].assign(items.begin(), items.end());
  bool captured = false;
  const bool resolved = control_stage1_fanout_and_wait(
    items_by_home, source_client,
    [&](u32 home, span<const Stage1ArmItem> resolved_items,
        span<const Stage1ArmResult> resolved_results) {
      lib_assert(home == stage1_home && resolved_items.size() == items.size() &&
                   resolved_results.size() == items.size(),
                 "single-home Stage1 control lost its result mapping");
      results.assign(resolved_results.begin(), resolved_results.end());
      captured = true;
    },
    config);
  return resolved && captured;
}

bool MemoryNode::control_stage1_fanout_and_wait(
    const dense_hashmap_t<
      u32, vec<service::storage_owner::Stage1ArmItem>>& items_by_home,
    u32 source_client,
    const std::function<void(
      u32,
      span<const service::storage_owner::Stage1ArmItem>,
      span<const service::storage_owner::Stage1ArmResult>)>&
      on_home_resolved,
    const Configuration& config) {
  using namespace service::storage_owner;
  using memory_node_peer_rpc_detail::Stage1ControlHomeProgress;
  using memory_node_peer_rpc_detail::Stage1ControlResponseDisposition;
  using memory_node_peer_rpc_detail::Stage1HomeRetryBackoff;
  using memory_node_peer_rpc_detail::classify_stage1_control_response;

  if (items_by_home.empty()) return true;
  struct PendingControl {
    u32 home{};
    u64 request_id{};
    vec<Stage1ArmItem> items;
    vec<byte_t> response_payload;
    Stage1ControlHomeProgress progress;
    Stage1HomeRetryBackoff retry_backoff;
    std::chrono::steady_clock::time_point deadline{};
    bool local{};
    bool release{};
  };
  vec<PendingControl> pending;
  pending.reserve(items_by_home.size());

  for (const auto& [home, items] : items_by_home) {
    if (home >= num_storage_nodes_ || items.empty() ||
        items.size() > config.storage_owner_batch_max ||
        items.size() > std::numeric_limits<u32>::max() ||
        stage1_arm_request_bytes(static_cast<u32>(items.size())) >
          peer_rpc_runtime_.message_bytes) {
      return false;
    }
    for (const Stage1ArmItem& item : items) {
      const auto action = static_cast<Stage1ArmAction>(item.action);
      if (item.token.source_client != source_client ||
          item.token.client_batch_id == 0 || item.generation == 0 ||
          item.reserved != 0 ||
          (action != Stage1ArmAction::arm &&
           action != Stage1ArmAction::abort &&
           action != Stage1ArmAction::release)) {
        return false;
      }
    }
    const bool saw_release = std::any_of(
      items.begin(), items.end(), [](const Stage1ArmItem& item) {
        return static_cast<Stage1ArmAction>(item.action) ==
          Stage1ArmAction::release;
      });
    const bool saw_non_release = std::any_of(
      items.begin(), items.end(), [](const Stage1ArmItem& item) {
        return static_cast<Stage1ArmAction>(item.action) !=
          Stage1ArmAction::release;
      });
    // The receiver deliberately rejects mixed release/mutation batches because
    // only a homogeneous release is an ordered quiescence watermark.
    if (saw_release && saw_non_release) return false;
    PendingControl request;
    request.home = home;
    request.local = home == storage_id_;
    request.release = saw_release;
    request.request_id = request.local ? 0 : allocate_peer_request_id();
    request.items = items;
    if (!request.local) {
      // Reserve before registering a request ID. Payload copying after a
      // response lease is acquired is then allocation-free and exception-safe.
      request.response_payload.reserve(peer_rpc_runtime_.message_bytes);
    }
    pending.push_back(std::move(request));
  }

  const auto timeout =
    memory_node_peer_rpc_detail::stage1_peer_attempt_timeout(
      config.storage_owner_rpc_timeout_ms);
  const auto try_post = [&](PendingControl& request) {
    lib_assert(!request.local && !request.progress.resolved(),
               "invalid remote Stage1 control post state");
    const auto now = std::chrono::steady_clock::now();
    if (!request.retry_backoff.ready(now)) return false;
    const u32 item_count = static_cast<u32>(request.items.size());
    const bool posted = try_post_peer_rpc_request_attempt(
      request.home, PeerRpcType::stage1_arm_request,
      PeerRpcType::stage1_arm_response, request.request_id, item_count,
      request.items.data(), request.items.size() * sizeof(request.items[0]),
      stage1_arm_request_bytes(item_count), request.release
        ? PeerRpcSendClass::control : PeerRpcSendClass::stage1);
    if (posted) {
      request.progress.mark_posted();
      request.deadline = now + timeout;
    }
    return posted;
  };

  // Prime every remote physical home before touching the local group. The
  // local admission path may wait for bounded Stage2 capacity, but that wait
  // now overlaps all remote arm/release work instead of preceding its sends.
  for (PendingControl& request : pending) {
    if (!request.local) (void)try_post(request);
  }

  const auto cancel_unresolved = [&]() {
    for (const PendingControl& request : pending) {
      if (!request.local && !request.progress.resolved()) {
        cancel_peer_rpc_response(request.request_id);
      }
    }
  };
  const auto resolve = [&](PendingControl& request,
                           span<const Stage1ArmResult> outputs) {
    const auto disposition = classify_stage1_control_response(
      span<const Stage1ArmItem>{request.items}, outputs);
    if (disposition == Stage1ControlResponseDisposition::malformed) {
      throw std::runtime_error(
        "malformed token-fenced Stage1 control response");
    }
    if (disposition == Stage1ControlResponseDisposition::retry) {
      request.progress.mark_retry();
      request.retry_backoff.schedule(std::chrono::steady_clock::now());
      return false;
    }
    if (on_home_resolved) {
      on_home_resolved(
        request.home, span<const Stage1ArmItem>{request.items}, outputs);
    }
    // "Resolved" includes the caller's per-home authority action, not just
    // receipt parsing. If that callback throws, the public token remains the
    // recovery mechanism and no later home is incorrectly reported complete.
    lib_assert(request.progress.mark_resolved(),
               "Stage1 control home resolved twice");
    return true;
  };

  size_t remaining = pending.size();
  try {
    while (remaining != 0 &&
           !peer_reverse_shutdown_.load(std::memory_order_acquire) &&
           !storage_insert_shutdown_.load(std::memory_order_acquire)) {
      bool made_progress = false;
      bool local_wait_needed = false;

      // Consume remote homes first. Any response already durable before local
      // admission is processed and committed immediately; no callback waits
      // for a second physical home to resolve.
      for (PendingControl& request : pending) {
        if (request.local || request.progress.resolved()) continue;
        if (request.progress.needs_post()) {
          made_progress = try_post(request) || made_progress;
          if (request.progress.needs_post()) continue;
        }

        PeerRpcHeader response_header{};
        request.response_payload.clear();
        PeerResponseLease response_lease{};
        const u32 item_count = static_cast<u32>(request.items.size());
        const TryPeerResponse state = try_consume_peer_rpc_response(
          request.request_id, request.home,
          PeerRpcType::stage1_arm_response, item_count,
          response_header, request.response_payload, response_lease);
        if (state == TryPeerResponse::pending) {
          if (std::chrono::steady_clock::now() >= request.deadline) {
            // The physical transaction is token-idempotent. Keep both the
            // request ID and semantic tokens and repost only this unresolved
            // home; a late response still resolves the same registry entry.
            request.progress.mark_retry();
            request.retry_backoff.schedule(
              std::chrono::steady_clock::now());
            made_progress = true;
          }
          continue;
        }
        if (state == TryPeerResponse::stale) {
          request.progress.mark_retry();
          request.retry_backoff.schedule(
            std::chrono::steady_clock::now());
          made_progress = true;
          continue;
        }

        const size_t expected_bytes = stage1_arm_response_bytes(item_count);
        const bool valid_envelope = state == TryPeerResponse::success &&
          request.response_payload.size() == expected_bytes &&
          response_header.magic == kPeerRpcMagic &&
          response_header.version == kPeerRpcVersion &&
          response_header.type == static_cast<u32>(
            PeerRpcType::stage1_arm_response) &&
          response_header.source_shard == request.home &&
          response_header.item_count == item_count &&
          response_header.request_id == request.request_id &&
          response_header.status == static_cast<u32>(InsertStatus::ok) &&
          response_header.reserved == 0;
        if (!valid_envelope) {
          if (response_lease.valid()) {
            (void)rearm_peer_rpc_response(response_lease);
          }
          throw std::runtime_error(
            "malformed Stage1 control response envelope");
        }

        const Stage1ArmResult* wire = stage1_arm_results(
          request.response_payload.data());
        const span<const Stage1ArmResult> outputs{wire, item_count};
        const auto disposition = classify_stage1_control_response(
          span<const Stage1ArmItem>{request.items}, outputs);
        if (disposition == Stage1ControlResponseDisposition::malformed) {
          (void)rearm_peer_rpc_response(response_lease);
          throw std::runtime_error(
            "malformed token-fenced Stage1 control response");
        }
        if (disposition == Stage1ControlResponseDisposition::retry) {
          lib_assert(await_late_peer_rpc_response(response_lease),
                     "retryable Stage1 control response lost its lease");
          request.progress.mark_retry();
          request.retry_backoff.schedule(
            std::chrono::steady_clock::now());
          made_progress = true;
          continue;
        }

        lib_assert(acknowledge_peer_rpc_response(response_lease),
                   "validated Stage1 control response lost its lease");
        if (resolve(request, outputs)) --remaining;
        made_progress = true;
      }

      // There is at most one local physical home. It is deliberately driven
      // after polling remote completions so a ready remote subset can commit
      // before a locally saturated maintenance queue wakes up.
      for (PendingControl& request : pending) {
        if (!request.local || request.progress.resolved()) continue;
        if (!request.retry_backoff.ready(std::chrono::steady_clock::now())) {
          local_wait_needed = true;
          continue;
        }
        vec<Stage1ArmResult> local_results;
        const bool processed = arm_local_stage1_items(
          storage_id_, span<const Stage1ArmItem>{request.items},
          local_results, config);
        made_progress = true;
        if (local_results.size() != request.items.size()) {
          throw std::runtime_error(
            "local Stage1 control returned a malformed result count");
        }
        // Per-item status is the semantic contract. `processed == false`
        // accompanies structural invalidity, whose failed result must pass
        // through the classifier and fail fast rather than becoming an
        // implicit infinite retry.
        (void)processed;
        if (resolve(request, span<const Stage1ArmResult>{local_results})) {
          --remaining;
        } else {
          local_wait_needed = true;
        }
      }

      if (remaining == 0) break;
      if (local_wait_needed) {
        // Local semantic backpressure has no CQ edge. Yield on the shared
        // maintenance condition rather than spinning between identical arm
        // attempts, while remote sends remain in flight.
        std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
        storage_owner_maintenance_cv_.wait_for(
          lock, std::chrono::microseconds(100));
      } else if (!made_progress) {
        std::unique_lock<std::mutex> lock(peer_completion_mutex_);
        peer_completion_cv_.wait_for(lock, std::chrono::microseconds(100));
      }
    }
  } catch (...) {
    cancel_unresolved();
    throw;
  }

  cancel_unresolved();
  return remaining == 0;
}
