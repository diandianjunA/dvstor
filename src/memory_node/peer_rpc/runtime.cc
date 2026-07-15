#include "memory_node/peer_rpc/detail.hh"

void MemoryNode::setup_peer_rpc_runtime(const Configuration& config) {
  if (!peer_context_ || num_storage_nodes_ <= 1) {
    return;
  }

  const u64 max_reverse_update_ops =
    static_cast<u64>(config.R) * static_cast<u64>(config.storage_owner_batch_max);
  lib_assert(max_reverse_update_ops <= std::numeric_limits<u32>::max(),
             "storage-owner reverse-update RPC batch is too large for the wire format");
  const size_t reverse_update_bytes =
    service::storage_owner::reverse_update_request_bytes(static_cast<u32>(max_reverse_update_ops));
  const size_t stitch_request_bytes =
    service::storage_owner::stitch_search_request_bytes(config.storage_owner_batch_max);
  const size_t stitch_response_bytes =
    service::storage_owner::stitch_search_response_bytes(
      config.storage_owner_batch_max,
      config.resolved_storage_owner_construction_width());
  peer_rpc_runtime_.message_bytes = align_up(
    std::max({reverse_update_bytes,
              service::storage_owner::reverse_update_response_bytes(),
              stitch_request_bytes,
              stitch_response_bytes}));
  lib_assert(peer_rpc_runtime_.message_bytes <= std::numeric_limits<u32>::max(),
             "storage-owner peer RPC message is too large for verbs SGEs");
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
  {
    std::lock_guard<std::mutex> lock(peer_rpc_send_slots_mutex_);
    peer_rpc_free_send_slots_.clear();
    peer_rpc_free_send_slots_.resize(num_storage_nodes_);
    peer_rpc_sync_send_mutexes_.clear();
    peer_rpc_sync_send_mutexes_.resize(num_storage_nodes_);
    for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
      if (peer_id == storage_id_) {
        continue;
      }
      peer_rpc_sync_send_mutexes_[peer_id] = std::make_unique<std::mutex>();
      for (u32 slot_id = 0;
           slot_id < peer_rpc_runtime_.send_slots_per_peer;
           ++slot_id) {
        const size_t send_class = static_cast<size_t>(
          peer_rpc_send_slot_class(slot_id));
        peer_rpc_free_send_slots_[peer_id][send_class].push_back(slot_id);
      }
    }
  }
  const size_t registry_peer_count = std::max<u32>(1, num_storage_nodes_ - 1);
  const size_t response_capacity = std::max<size_t>(
    1024,
    static_cast<size_t>(config.storage_owner_rpc_depth) *
      std::max<u32>(1, config.storage_owner_maintenance_workers) *
      registry_peer_count * 4);
  peer_async_responses_ =
    std::make_unique<PeerAsyncResponseRegistry>(response_capacity);
  const size_t dedup_capacity = std::max<size_t>(
    1024,
    static_cast<size_t>(config.storage_owner_reverse_queue_depth) *
      registry_peer_count * 2);
  peer_request_deduplicator_ =
    std::make_unique<PeerRequestDeduplicator>(dedup_capacity);
  print_status("storage-owner peer RPC receive slots per peer: " +
               std::to_string(peer_rpc_runtime_.recv_slots_per_peer) +
               " (requested=" + std::to_string(desired_slots_per_peer) + ")");
  print_status("storage-owner peer RPC concurrent sends per peer: " +
               std::to_string(peer_rpc_runtime_.send_slots_per_peer));
  print_status("storage-owner peer RPC send credits: search/graph are split "
               "at depth >= 2; responses use a dedicated sync buffer");
  print_status("storage-owner peer async response capacity: " +
               std::to_string(peer_async_responses_->capacity()));
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

  {
    std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
    peer_rpc_pending_responses_.clear();
    peer_rpc_responses_.clear();
    peer_rpc_response_payloads_.clear();
  }

  peer_reverse_shutdown_.store(false, std::memory_order_release);
  peer_reverse_workers_done_.store(false, std::memory_order_release);
  peer_reverse_response_done_.store(false, std::memory_order_release);
  peer_reverse_task_queue_limit_ =
    std::max<size_t>(1024, static_cast<size_t>(config.storage_owner_reverse_queue_depth));
  peer_stitch_search_task_queue_limit_ = peer_reverse_task_queue_limit_;
  peer_reverse_outgoing_queue_limit_ = peer_reverse_task_queue_limit_;
  peer_reverse_responses_ =
    std::make_unique<bounded::Queue<PeerReverseUpdateResponse>>(
      peer_reverse_task_queue_limit_);
  peer_reverse_update_enqueued_.store(0, std::memory_order_relaxed);
  peer_reverse_update_processed_.store(0, std::memory_order_relaxed);
  peer_reverse_update_items_enqueued_.store(0, std::memory_order_relaxed);
  peer_reverse_update_items_processed_.store(0, std::memory_order_relaxed);
  peer_reverse_update_failed_.store(0, std::memory_order_relaxed);
  peer_reverse_update_max_queue_.store(0, std::memory_order_relaxed);
  peer_stitch_search_enqueued_.store(0, std::memory_order_relaxed);
  peer_stitch_search_processed_.store(0, std::memory_order_relaxed);
  peer_stitch_search_items_.store(0, std::memory_order_relaxed);
  peer_stitch_search_max_queue_.store(0, std::memory_order_relaxed);
  peer_stitch_search_active_workers_.store(0, std::memory_order_relaxed);

  const u32 cpu_parallelism = std::max<u32>(1, num_compute_threads_ / 2);
  const u32 reverse_worker_count = std::min<u32>(8, cpu_parallelism);
  const u32 stitch_worker_count = std::min(
    cpu_parallelism,
    std::max<u32>(1, config.storage_owner_maintenance_workers));
  const size_t snapshot_stride = align_up(VamanaNode::vector_bytes());
  const size_t neighbor_stride = align_up(VamanaNode::neighbor_read_size());
  const size_t coroutine_scratch_stride =
    align_up(std::max<size_t>(VamanaNode::total_size(),
                              std::max(neighbor_stride,
                                       snapshot_stride *
                                         std::max<u32>(1, config.storage_owner_search_snapshot_batch))));
  const size_t scratch_bytes = coroutine_scratch_stride;
  peer_reverse_worker_states_.reserve(reverse_worker_count);
  for (u32 i = 0; i < reverse_worker_count; ++i) {
    auto worker = std::make_unique<StorageOwnerThread>(i, 1, config.max_send_queue_wr);
    worker->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
    peer_reverse_worker_states_.push_back(std::move(worker));
  }
  peer_stitch_search_worker_states_.reserve(stitch_worker_count);
  for (u32 i = 0; i < stitch_worker_count; ++i) {
    auto stitch_worker = std::make_unique<StorageOwnerThread>(
      reverse_worker_count + i, 1, config.max_send_queue_wr);
    peer_stitch_search_worker_states_.push_back(std::move(stitch_worker));
  }

  peer_rpc_progress_thread_ = std::thread([this]() { peer_rpc_progress_loop(); });
  peer_reverse_response_thread_ = std::thread([this]() { peer_reverse_response_loop(); });
  peer_reverse_outgoing_thread_ = std::thread([this]() { peer_reverse_outgoing_loop(); });
  if (!config.disable_thread_pinning) {
    pin_thread(peer_rpc_progress_thread_, core_assignment_.get_available_core());
    pin_thread(peer_reverse_response_thread_, core_assignment_.get_available_core());
    pin_thread(peer_reverse_outgoing_thread_, core_assignment_.get_available_core());
  }
  for (u32 i = 0; i < reverse_worker_count; ++i) {
    peer_reverse_workers_.emplace_back([this, i]() { peer_reverse_update_worker_loop(i); });
    if (!config.disable_thread_pinning) {
      pin_thread(peer_reverse_workers_.back(),
                 core_assignment_.get_available_core());
    }
  }
  for (u32 i = 0; i < stitch_worker_count; ++i) {
    peer_stitch_search_workers_.emplace_back([this, i]() { peer_stitch_search_worker_loop(i); });
    if (!config.disable_thread_pinning) {
      pin_thread(peer_stitch_search_workers_.back(),
                 core_assignment_.get_available_core());
    }
  }
  print_status("storage-owner peer reverse-update workers: " +
               std::to_string(reverse_worker_count));
  print_status("storage-owner peer stitch-search workers: " +
               std::to_string(stitch_worker_count) +
               " (dedicated background CPU partition)");
  print_status("storage-owner peer reverse-update tuning: mode=" + config.storage_owner_reverse_mode +
               " queue_depth=" + std::to_string(peer_reverse_task_queue_limit_) +
               " coalesce_max=" + std::to_string(config.storage_owner_reverse_coalesce_max));
}

void MemoryNode::stop_peer_reverse_update_runtime() {
  peer_reverse_shutdown_.store(true, std::memory_order_release);
  peer_reverse_tasks_cv_.notify_all();
  peer_stitch_search_tasks_cv_.notify_all();
  if (peer_reverse_responses_) peer_reverse_responses_->notify_all();
  peer_reverse_outgoing_cv_.notify_all();
  peer_rpc_responses_cv_.notify_all();
  peer_completion_cv_.notify_all();

  if (peer_reverse_outgoing_thread_.joinable()) {
    peer_reverse_outgoing_thread_.join();
  }
  for (auto& worker : peer_reverse_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  for (auto& worker : peer_stitch_search_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  peer_reverse_workers_done_.store(true, std::memory_order_release);
  if (peer_reverse_responses_) peer_reverse_responses_->notify_all();
  if (peer_reverse_response_thread_.joinable()) {
    peer_reverse_response_thread_.join();
  }
  if (peer_rpc_progress_thread_.joinable()) {
    peer_rpc_progress_thread_.join();
  }
  if (peer_async_responses_ != nullptr) {
    for (const auto& response : peer_async_responses_->drain_completed()) {
      repost_peer_rpc_receive(response.peer_id, response.receive_slot);
    }
  }
  peer_reverse_workers_.clear();
  peer_stitch_search_workers_.clear();
  peer_reverse_worker_states_.clear();
  peer_stitch_search_worker_states_.clear();
  peer_reverse_responses_.reset();
  {
    std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
    peer_rpc_pending_responses_.clear();
    peer_rpc_responses_.clear();
    peer_rpc_response_payloads_.clear();
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

MemoryNode::PeerRpcSendClass MemoryNode::peer_rpc_send_slot_class(
    u32 slot_id) const {
  const u32 slot_count = peer_rpc_runtime_.send_slots_per_peer;
  if (slot_count <= 1) return PeerRpcSendClass::control;
  return slot_id % 2 == 0 ? PeerRpcSendClass::stitch_search
                          : PeerRpcSendClass::graph_update;
}

bool MemoryNode::try_acquire_peer_rpc_send_slot(
    u32 peer_id,
    PeerRpcSendClass send_class,
    u32& slot_id) {
  lib_assert(peer_id < peer_rpc_free_send_slots_.size() && peer_id != storage_id_,
             "invalid peer RPC send-slot owner");
  std::lock_guard<std::mutex> lock(peer_rpc_send_slots_mutex_);
  auto& lanes = peer_rpc_free_send_slots_[peer_id];
  auto try_lane = [&](PeerRpcSendClass lane) {
    auto& free_slots = lanes[static_cast<size_t>(lane)];
    if (free_slots.empty()) return false;
    slot_id = free_slots.front();
    free_slots.pop_front();
    return true;
  };

  if (peer_rpc_runtime_.send_slots_per_peer == 1) {
    return try_lane(PeerRpcSendClass::control);
  }
  if (send_class == PeerRpcSendClass::control) {
    return try_lane(PeerRpcSendClass::stitch_search) ||
           try_lane(PeerRpcSendClass::graph_update);
  }
  return try_lane(send_class);
}

void MemoryNode::release_peer_rpc_send_slot(u32 peer_id, u32 slot_id) {
  {
    std::lock_guard<std::mutex> lock(peer_rpc_send_slots_mutex_);
    lib_assert(peer_id < peer_rpc_free_send_slots_.size() &&
                 slot_id < peer_rpc_runtime_.send_slots_per_peer,
               "invalid peer RPC send-slot release");
    const size_t send_class = static_cast<size_t>(
      peer_rpc_send_slot_class(slot_id));
    peer_rpc_free_send_slots_[peer_id][send_class].push_back(slot_id);
  }
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

void MemoryNode::post_peer_rpc_send_slot(u32 peer_id,
                                         u32 slot_id,
                                         size_t bytes) {
  lib_assert(peer_context_ != nullptr, "peer context not initialized");
  lib_assert(bytes <= peer_rpc_runtime_.message_bytes, "peer rpc message too large");
  const u64 wr_id = next_peer_async_wr_id();
  const size_t offset = peer_rpc_async_send_offset(peer_id, slot_id);
  register_peer_pending_send_locked(
    wr_id,
    PeerPendingSend{
      .target_shard = peer_id,
      .target_qp_idx = 0,
      .release_rpc_slot = true,
      .rpc_slot_id = slot_id,
    });
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
}

void MemoryNode::send_peer_rpc_message(u32 peer_id, const void* payload, size_t bytes) {
  lib_assert(peer_context_ != nullptr, "peer context not initialized");
  lib_assert(peer_id < peer_rpc_sync_send_mutexes_.size() &&
               peer_rpc_sync_send_mutexes_[peer_id] != nullptr,
             "peer RPC sync send buffer is not initialized");
  lib_assert(bytes <= peer_rpc_runtime_.message_bytes,
             "peer rpc message too large");
  lib_assert(!current_peer_rpc_progress_thread_,
             "peer CQ progress thread must not execute a blocking response send");
  std::lock_guard<std::mutex> sync_lock(
    *peer_rpc_sync_send_mutexes_[peer_id]);
  const size_t offset = peer_rpc_sync_send_offset(peer_id);
  std::memcpy(peer_rpc_runtime_.buffer.get_full_buffer() + offset, payload, bytes);
  const u64 wr_id = next_peer_sync_wr_id();
  register_peer_pending_send_locked(
    wr_id,
    PeerPendingSend{
      .target_shard = peer_id,
      .target_qp_idx = 0,
    });
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
  response.version = service::storage_owner::kPeerRpcVersion;
  const auto request_type = static_cast<service::storage_owner::PeerRpcType>(request.type);
  const auto response_type =
    request_type == service::storage_owner::PeerRpcType::cleanup_deleted_request
      ? service::storage_owner::PeerRpcType::cleanup_deleted_response
      : service::storage_owner::PeerRpcType::reverse_update_response;
  response.type = static_cast<u32>(response_type);
  response.source_shard = storage_id_;
  response.item_count = request.item_count;
  response.request_id = request.request_id;
  response.status = static_cast<u32>(success ? service::storage_owner::InsertStatus::ok
                                             : service::storage_owner::InsertStatus::failed);
  return response;
}
