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

  const auto registration = peer_async_responses_->register_send_attempt(
    request_id, target_shard, response_type, item_count);
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
    payload.assign(source, source + response.bytes);
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
    const std::function<bool()>& overlap_work,
    const Configuration& config) {
  using namespace service::storage_owner;
  results_by_home.clear();
  if (items_by_home.empty()) return !overlap_work || overlap_work();

  struct PendingStage1 {
    u32 home{};
    u64 request_id{};
    u32 item_count{};
    vec<byte_t> message;
    std::chrono::steady_clock::time_point deadline{};
    bool posted{};
    bool resolved{};
  };
  vec<PendingStage1> pending;
  pending.reserve(items_by_home.size());

  for (const auto& [home, items] : items_by_home) {
    const auto vectors_position = vectors_by_home.find(home);
    if (home >= num_storage_nodes_ || home == storage_id_ || items.empty() ||
        items.size() > config.storage_owner_batch_max ||
        vectors_position == vectors_by_home.end() ||
        vectors_position->second.size() !=
          items.size() * VamanaNode::vector_bytes()) {
      return false;
    }
    for (const Stage1ExecuteItem& item : items) {
      if (item.authority_shard != storage_id_ ||
          item.client_batch_id == 0 ||
          item.initial_placement_version != 0) {
        return false;
      }
    }

    PendingStage1 request;
    request.home = home;
    request.request_id = allocate_peer_request_id();
    request.item_count = static_cast<u32>(items.size());
    request.message.resize(stage1_execute_request_bytes(request.item_count));
    if (request.message.size() > peer_rpc_runtime_.message_bytes) return false;
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
    pending.push_back(std::move(request));
  }

  // Stage1 owns a dedicated async send-credit lane. This keeps the distinct
  // single-home groups of one mutation batch in flight together and prevents
  // a foreground coordinator from serializing on the shared synchronous
  // response buffer. Each item still executes on exactly one centroid home.
  const auto try_post = [&](const PendingStage1& request) {
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

  // Prime every remote home before running authority-local Stage1 work.  The
  // physical-home workers can then search and publish in parallel with the
  // local subset of this batch.  A temporarily exhausted send lane is not an
  // error: the event loop below posts that home as soon as a slot completes.
  const auto initial_post_time = std::chrono::steady_clock::now();
  for (PendingStage1& request : pending) {
    if (!try_post(request)) continue;
    request.posted = true;
    request.deadline = initial_post_time +
      std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
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

  // Progress every home in one event loop.  Consuming completed descriptors
  // promptly is part of flow control: a response holds its registered receive
  // WR until acknowledge_peer_rpc_response(), so waiting indefinitely on one
  // retrying home before inspecting the others can exhaust receive credits.
  // Per-home retries keep the same request ID and semantic item tokens.
  size_t remaining = pending.size();
  while (remaining != 0 &&
         !peer_reverse_shutdown_.load(std::memory_order_acquire) &&
         !storage_insert_shutdown_.load(std::memory_order_acquire)) {
    bool made_progress = false;
    const auto now = std::chrono::steady_clock::now();
    for (PendingStage1& request : pending) {
      if (request.resolved) continue;

      if (!request.posted) {
        if (!try_post(request)) continue;
        request.posted = true;
        request.deadline = now +
          std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
        made_progress = true;
      }

      vec<byte_t> response;
      PeerResponseLease response_lease{};
      PeerRpcHeader response_header{};
      const TryPeerResponse state = try_consume_peer_rpc_response(
        request.request_id, request.home,
        PeerRpcType::stage1_execute_response, request.item_count,
        response_header, response, response_lease);
      if (state == TryPeerResponse::pending) {
        if (std::chrono::steady_clock::now() >= request.deadline) {
          // Prepare is token-idempotent. Do not cancel or semantically abort
          // an uncertain request; repost the identical request ID so either
          // response resolves the same registry entry.
          request.posted = false;
          made_progress = true;
        }
        continue;
      }
      if (state == TryPeerResponse::stale) {
        request.posted = false;
        made_progress = true;
        continue;
      }

      // A failed aggregate header can still carry authoritative per-item
      // Stage1 statuses. Keep the descriptor leased until all payload records
      // have been validated.
      const size_t expected_bytes =
        stage1_execute_response_bytes(request.item_count);
      const auto* header = response.size() == expected_bytes
        ? reinterpret_cast<const PeerRpcHeader*>(response.data()) : nullptr;
      if (header == nullptr || header->magic != kPeerRpcMagic ||
          header->version != kPeerRpcVersion ||
          header->type != static_cast<u32>(
            PeerRpcType::stage1_execute_response) ||
          header->source_shard != request.home ||
          header->item_count != request.item_count ||
          header->request_id != request.request_id) {
        if (response_lease.valid()) {
          (void)rearm_peer_rpc_response(response_lease);
        }
        cancel_peer_rpc_response(request.request_id);
        throw std::runtime_error(
          "invalid Stage1 response under an uncertain prepare");
      }

      const auto* output = stage1_execute_results(response.data());
      vec<Stage1ExecuteResult> shard_results(
        output, output + request.item_count);
      const auto& input = items_by_home.at(request.home);
      bool valid = true;
      bool retryable = false;
      for (u32 index = 0; index < request.item_count; ++index) {
        valid &= shard_results[index].client_batch_id ==
                   input[index].client_batch_id &&
          shard_results[index].source_client == input[index].source_client &&
          shard_results[index].item_index == input[index].item_index &&
          shard_results[index].reserved == 0 &&
          shard_results[index].status <=
            static_cast<u32>(MutationStatus::retry);
        retryable |= shard_results[index].status ==
          static_cast<u32>(MutationStatus::retry);
        if (shard_results[index].status ==
            static_cast<u32>(MutationStatus::ok)) {
          const RemotePtr target{shard_results[index].target_raw};
          valid &= !target.is_null() &&
            target.memory_node() == request.home &&
            shard_results[index].maintenance_sequence == 0;
        }
      }
      if (!valid) {
        if (response_lease.valid()) {
          (void)rearm_peer_rpc_response(response_lease);
        }
        cancel_peer_rpc_response(request.request_id);
        throw std::runtime_error(
          "malformed Stage1 result under an uncertain prepare");
      }
      if (!acknowledge_peer_rpc_response(response_lease)) {
        request.posted = false;
        made_progress = true;
        continue;
      }
      request.posted = false;
      made_progress = true;
      if (retryable) continue;

      const bool inserted = results_by_home.emplace(
        request.home, std::move(shard_results)).second;
      lib_assert(inserted, "Stage1 home resolved twice");
      request.resolved = true;
      --remaining;
    }

    if (!made_progress && remaining != 0) {
      std::unique_lock<std::mutex> lock(peer_completion_mutex_);
      peer_completion_cv_.wait_for(lock, std::chrono::microseconds(100));
    }
  }

  for (const PendingStage1& request : pending) {
    if (!request.resolved) cancel_peer_rpc_response(request.request_id);
  }
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
  const u32 item_count = static_cast<u32>(items.size());
  const u64 request_id = allocate_peer_request_id();
  constexpr u32 kTransportAttempts = 3;
  bool posted = false;
  for (u32 attempt = 0; attempt < kTransportAttempts; ++attempt) {
    if (!posted) {
      posted = post_peer_control_request_attempt(
        stage1_home, PeerRpcType::stage1_arm_request,
        PeerRpcType::stage1_arm_response, request_id, item_count,
        items.data(), items.size() * sizeof(items[0]),
        stage1_arm_request_bytes(item_count), config);
      if (!posted) continue;
    }
    PeerRpcHeader response_header;
    vec<byte_t> payload;
    PeerResponseLease response_lease{};
    const TryPeerResponse state = wait_peer_control_response(
      request_id, stage1_home, PeerRpcType::stage1_arm_response,
      item_count, response_header, payload, response_lease, config);
    if (state == TryPeerResponse::success &&
        payload.size() == stage1_arm_response_bytes(item_count)) {
      const Stage1ArmResult* wire = stage1_arm_results(payload.data());
      results.assign(wire, wire + item_count);
      if (acknowledge_peer_rpc_response(response_lease)) return true;
    }
    posted = false;
    if (response_lease.valid()) {
      (void)rearm_peer_rpc_response(response_lease);
    }
  }
  cancel_peer_rpc_response(request_id);
  return false;
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
  using memory_node_peer_rpc_detail::classify_stage1_control_response;

  if (items_by_home.empty()) return true;
  struct PendingControl {
    u32 home{};
    u64 request_id{};
    vec<Stage1ArmItem> items;
    vec<byte_t> response_payload;
    Stage1ControlHomeProgress progress;
    std::chrono::steady_clock::time_point deadline{};
    bool local{};
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
    PendingControl request;
    request.home = home;
    request.local = home == storage_id_;
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
    std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  const auto try_post = [&](PendingControl& request) {
    lib_assert(!request.local && !request.progress.resolved(),
               "invalid remote Stage1 control post state");
    const u32 item_count = static_cast<u32>(request.items.size());
    const bool posted = try_post_peer_rpc_request_attempt(
      request.home, PeerRpcType::stage1_arm_request,
      PeerRpcType::stage1_arm_response, request.request_id, item_count,
      request.items.data(), request.items.size() * sizeof(request.items[0]),
      stage1_arm_request_bytes(item_count), PeerRpcSendClass::stage1);
    if (posted) {
      request.progress.mark_posted();
      request.deadline = std::chrono::steady_clock::now() + timeout;
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
            made_progress = true;
          }
          continue;
        }
        if (state == TryPeerResponse::stale) {
          request.progress.mark_retry();
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
          lib_assert(rearm_peer_rpc_response(response_lease),
                     "retryable Stage1 control response lost its lease");
          request.progress.mark_retry();
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
