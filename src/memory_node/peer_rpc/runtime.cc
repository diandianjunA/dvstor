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
    service::storage_owner::stitch_search_response_bytes(config.storage_owner_batch_max);
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

  {
    std::lock_guard<std::mutex> lock(peer_rpc_mutex_);
    peer_rpc_pending_responses_.clear();
    peer_rpc_responses_.clear();
    peer_rpc_response_payloads_.clear();
  }

  peer_reverse_shutdown_.store(false, std::memory_order_release);
  peer_reverse_workers_done_.store(false, std::memory_order_release);
  peer_reverse_task_queue_limit_ =
    std::max<size_t>(1024, static_cast<size_t>(config.storage_owner_reverse_queue_depth));
  peer_stitch_search_task_queue_limit_ = peer_reverse_task_queue_limit_;
  peer_reverse_outgoing_queue_limit_ = peer_reverse_task_queue_limit_;
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
  peer_stitch_search_worker_states_.reserve(worker_count);
  for (u32 i = 0; i < worker_count; ++i) {
    auto worker = std::make_unique<StorageOwnerThread>(i, 1, config.max_send_queue_wr);
    worker->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
    peer_reverse_worker_states_.push_back(std::move(worker));

    auto stitch_worker = std::make_unique<StorageOwnerThread>(worker_count + i, 1, config.max_send_queue_wr);
    stitch_worker->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
    peer_stitch_search_worker_states_.push_back(std::move(stitch_worker));
  }

  peer_rpc_progress_thread_ = std::thread([this]() { peer_rpc_progress_loop(); });
  peer_reverse_response_thread_ = std::thread([this]() { peer_reverse_response_loop(); });
  peer_reverse_outgoing_thread_ = std::thread([this]() { peer_reverse_outgoing_loop(); });
  for (u32 i = 0; i < worker_count; ++i) {
    peer_reverse_workers_.emplace_back([this, i]() { peer_reverse_update_worker_loop(i); });
    peer_stitch_search_workers_.emplace_back([this, i]() { peer_stitch_search_worker_loop(i); });
  }
  print_status("storage-owner peer reverse-update workers: " + std::to_string(worker_count));
  print_status("storage-owner peer stitch-search workers: " + std::to_string(worker_count));
  print_status("storage-owner peer reverse-update tuning: mode=" + config.storage_owner_reverse_mode +
               " queue_depth=" + std::to_string(peer_reverse_task_queue_limit_) +
               " flush_us=" + std::to_string(config.storage_owner_reverse_flush_us) +
               " coalesce_max=" + std::to_string(config.storage_owner_reverse_coalesce_max));
}

void MemoryNode::stop_peer_reverse_update_runtime() {
  peer_reverse_shutdown_.store(true, std::memory_order_release);
  peer_reverse_tasks_cv_.notify_all();
  peer_stitch_search_tasks_cv_.notify_all();
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
  for (auto& worker : peer_stitch_search_workers_) {
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
  peer_reverse_workers_.clear();
  peer_stitch_search_workers_.clear();
  peer_reverse_worker_states_.clear();
  peer_stitch_search_worker_states_.clear();
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
  response.version = service::storage_owner::kPeerRpcVersion;
  response.type = static_cast<u32>(service::storage_owner::PeerRpcType::reverse_update_response);
  response.source_shard = storage_id_;
  response.item_count = request.item_count;
  response.request_id = request.request_id;
  response.status = static_cast<u32>(success ? service::storage_owner::InsertStatus::ok
                                             : service::storage_owner::InsertStatus::failed);
  return response;
}
