#include "memory_node/memory_node.hh"

#include <algorithm>
#include <iostream>

void MemoryNode::setup_storage_peers(Configuration& config) {
  if (!use_storage_owner_insert_ || num_storage_nodes_ <= 1) {
    return;
  }

  lib_assert(config.storage_peers.size() == num_storage_nodes_,
             "storage_owner mode requires one storage peer endpoint per storage node");
  const auto self_endpoint = parse_endpoint(config.storage_peers[storage_id_], config.port);

  peer_config_ = std::make_unique<configuration::Configuration>(config);
  peer_config_->port = self_endpoint.port;
  peer_config_->is_server = true;
  peer_context_ = std::make_unique<Context>(*peer_config_);
  peer_context_->bind_to_port(self_endpoint.port);

  peer_qps_per_peer_ = std::max<u32>(1, std::min<u32>(MAX_QPS, std::max<u32>(1, num_compute_threads_)));
  peer_qps_.resize(num_storage_nodes_);
  peer_remote_tokens_.resize(num_storage_nodes_);
  peer_rdma_read_qp_outstanding_.clear();
  peer_rdma_read_qp_outstanding_.reserve(num_storage_nodes_);
  for (u32 i = 0; i < num_storage_nodes_; ++i) {
    auto& qp_credits = peer_rdma_read_qp_outstanding_.emplace_back(peer_qps_per_peer_);
    for (auto& credit : qp_credits) {
      credit.store(0, std::memory_order_relaxed);
    }
    if (i != storage_id_) {
      peer_qps_[i].resize(peer_qps_per_peer_);
      peer_remote_tokens_[i] = std::make_unique<MemoryRegionToken>();
    }
  }

  for (u32 peer_id = 0; peer_id < storage_id_; ++peer_id) {
    for (u32 qp_idx = 0; qp_idx < peer_qps_per_peer_; ++qp_idx) {
      const auto endpoint = parse_endpoint(config.storage_peers[peer_id], config.port);
      const u32 encoded_id = storage_id_ * peer_qps_per_peer_ + qp_idx;
      peer_qps_[peer_id][qp_idx] =
        peer_context_->connect_to_server(endpoint.address, endpoint.port, encoded_id);
    }
  }
  const u32 incoming_peer_count = num_storage_nodes_ - storage_id_ - 1;
  for (u32 i = 0; i < incoming_peer_count * peer_qps_per_peer_; ++i) {
    auto [qp, encoded_id] = peer_context_->wait_for_connection();
    const u32 peer_id = encoded_id / peer_qps_per_peer_;
    const u32 remote_qp_idx = encoded_id % peer_qps_per_peer_;
    lib_assert(peer_id < num_storage_nodes_, "invalid peer storage id");
    lib_assert(peer_id > storage_id_, "unexpected lower peer connection");
    lib_assert(remote_qp_idx < peer_qps_per_peer_, "invalid peer QP index");
    lib_assert(peer_qps_[peer_id][remote_qp_idx] == nullptr, "duplicate peer QP connection");
    peer_qps_[peer_id][remote_qp_idx] = std::move(qp);
  }
  peer_context_->close_server_socket();
  print_status("storage-owner peer RDMA QPs per peer: " + std::to_string(peer_qps_per_peer_));

  peer_index_region_ = std::make_unique<MemoryRegion>(*peer_context_);
  peer_index_region_->register_memory(index_buffer_.get_full_buffer(), index_buffer_.buffer_size, true);

  const MemoryRegionToken local_token = peer_index_region_->createToken();
  std::cerr << "[storage-peer][token] self_shard=" << storage_id_
            << " local_base=" << local_token.address
            << " local_rkey=" << local_token.rkey
            << " local_bytes=" << index_buffer_.buffer_size << std::endl;
  for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
    if (peer_id == storage_id_) continue;
    LocalMemoryRegion peer_token_region{*peer_context_, peer_remote_tokens_[peer_id].get(), sizeof(MemoryRegionToken)};
    peer_control_qp(peer_id)->post_receive(peer_token_region);
    peer_control_qp(peer_id)->post_send_inlined(&local_token, sizeof(local_token), IBV_WR_SEND);
    peer_context_->poll_send_cq_until_completion();
    peer_context_->receive();
    std::cerr << "[storage-peer][token] self_shard=" << storage_id_
              << " peer_shard=" << peer_id
              << " remote_base=" << peer_remote_tokens_[peer_id]->address
              << " remote_rkey=" << peer_remote_tokens_[peer_id]->rkey << std::endl;
  }

  const size_t scratch_bytes = std::max<size_t>(64ull * 1024ull * 1024ull, align_up(VamanaNode::total_size() * 4));
  peer_scratch_buffer_.allocate(scratch_bytes);
  peer_scratch_buffer_.touch_memory();
  peer_scratch_region_ =
    std::make_unique<LocalMemoryRegion>(*peer_context_, peer_scratch_buffer_.get_full_buffer(), scratch_bytes);
  peer_send_wcs_.resize(std::max<i32>(1, peer_context_->get_config().max_send_queue_wr));

  setup_peer_rpc_runtime(config);

  for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
    if (peer_id == storage_id_) continue;
    u64 header_words[2]{};
    remote_read_bytes(peer_id, 0, header_words, sizeof(header_words), 0);
    std::cerr << "[storage-peer][probe] self_shard=" << storage_id_
              << " peer_shard=" << peer_id
              << " free_ptr=" << header_words[0]
              << " medoid_raw=" << header_words[1] << std::endl;
  }
}

QP& MemoryNode::peer_control_qp(u32 shard_id) {
  lib_assert(shard_id < peer_qps_.size(), "invalid peer shard id: " + std::to_string(shard_id));
  lib_assert(!peer_qps_[shard_id].empty() && peer_qps_[shard_id][0] != nullptr,
             "peer control QP is not initialized for shard " + std::to_string(shard_id));
  return peer_qps_[shard_id][0];
}

u32 MemoryNode::peer_data_qp_index(u32 worker_id) const {
  lib_assert(peer_qps_per_peer_ > 0, "peer QP count is not initialized");
  return worker_id % peer_qps_per_peer_;
}

QP& MemoryNode::peer_data_qp(u32 shard_id, u32 qp_idx) {
  lib_assert(shard_id < peer_qps_.size(), "invalid peer shard id: " + std::to_string(shard_id));
  lib_assert(qp_idx < peer_qps_[shard_id].size() && peer_qps_[shard_id][qp_idx] != nullptr,
             "peer data QP is not initialized for shard " + std::to_string(shard_id) +
               " qp_idx=" + std::to_string(qp_idx));
  return peer_qps_[shard_id][qp_idx];
}

u64 MemoryNode::peer_coroutine_wr_id(u32 thread_id, u32 coroutine_id) {
  return encode_64bit(thread_id, coroutine_id);
}

u32 MemoryNode::peer_rdma_read_credit_limit_per_qp() const {
  return std::max<u32>(1, std::min<u32>(storage_owner_peer_rdma_tokens_, kPeerSafeRdAtomic));
}

u32 MemoryNode::peer_rdma_read_credit_limit() const {
  const u32 per_peer_safe = std::max<u32>(1, peer_qps_per_peer_) * kPeerSafeRdAtomic;
  return std::max<u32>(1, std::min<u32>(storage_owner_peer_rdma_tokens_, per_peer_safe));
}

u32 MemoryNode::peer_rdma_read_global_credit_limit() const {
  const u32 remote_peer_count = num_storage_nodes_ > 1 ? num_storage_nodes_ - 1 : 1;
  return std::max<u32>(1, peer_rdma_read_credit_limit() * remote_peer_count);
}

bool MemoryNode::try_acquire_counter(std::atomic<u32>& counter, u32 limit) {
  u32 current = counter.load(std::memory_order_acquire);
  while (current < limit) {
    if (counter.compare_exchange_weak(current,
                                      current + 1,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

bool MemoryNode::try_acquire_peer_rdma_read_credit(u32 shard_id, u32 qp_idx) {
  lib_assert(shard_id < peer_rdma_read_qp_outstanding_.size(), "invalid peer shard id");
  lib_assert(qp_idx < peer_rdma_read_qp_outstanding_[shard_id].size(), "invalid peer QP index");
  if (!try_acquire_counter(peer_rdma_read_outstanding_[shard_id], peer_rdma_read_credit_limit())) {
    return false;
  }
  if (try_acquire_counter(peer_rdma_read_qp_outstanding_[shard_id][qp_idx],
                          peer_rdma_read_credit_limit_per_qp())) {
    return true;
  }
  peer_rdma_read_outstanding_[shard_id].fetch_sub(1, std::memory_order_acq_rel);
  return false;
}

void MemoryNode::acquire_peer_rdma_read_credit(u32 shard_id, u32 qp_idx) {
  while (!try_acquire_peer_rdma_read_credit(shard_id, qp_idx)) {
    poll_peer_send_cq();
    std::this_thread::yield();
  }
}

u64 MemoryNode::next_peer_sync_wr_id() {
  const u32 id = peer_sync_wr_id_counter_.fetch_add(1, std::memory_order_relaxed);
  return encode_64bit(kPeerSyncWrOwner, id);
}

u64 MemoryNode::next_peer_async_wr_id() {
  const u32 id = peer_async_wr_id_counter_.fetch_add(1, std::memory_order_relaxed);
  return encode_64bit(kPeerAsyncWrOwner, id);
}

void MemoryNode::register_peer_pending_send_locked(u64 wr_id, PeerPendingSend pending) {
  peer_pending_sends_[wr_id] = pending;
}

void MemoryNode::handle_peer_send_completion(u64 wr_id) {
  const auto pending_it = peer_pending_sends_.find(wr_id);
  if (pending_it != peer_pending_sends_.end()) {
    const PeerPendingSend pending = pending_it->second;
    peer_pending_sends_.erase(pending_it);
    if (pending.rdma_read_credit) {
      peer_rdma_read_outstanding_[pending.target_shard].fetch_sub(1, std::memory_order_acq_rel);
      if (pending.target_shard < peer_rdma_read_qp_outstanding_.size() &&
          pending.target_qp_idx < peer_rdma_read_qp_outstanding_[pending.target_shard].size()) {
        peer_rdma_read_qp_outstanding_[pending.target_shard][pending.target_qp_idx].fetch_sub(
          1, std::memory_order_acq_rel);
      }
    }
    if (pending.async) {
      if (pending.thread_id < storage_owner_threads_.size() && storage_owner_threads_[pending.thread_id]) {
        auto& balance = storage_owner_threads_[pending.thread_id]->post_balances[pending.coroutine_id];
        --balance;
        peer_async_rdma_outstanding_.fetch_sub(1, std::memory_order_acq_rel);
      }
      return;
    }
  }

  const auto [owner, id] = decode_64bit(wr_id);
  if (owner == kPeerSyncWrOwner) {
    peer_sync_completions_.insert(wr_id);
    return;
  }
  if (owner < storage_owner_threads_.size() && storage_owner_threads_[owner]) {
    auto& balance = storage_owner_threads_[owner]->post_balances[id];
    --balance;
    peer_async_rdma_outstanding_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

void MemoryNode::poll_peer_send_cq() {
  if (!peer_context_) {
    return;
  }
  std::lock_guard<std::mutex> lock(peer_send_mutex_);
  Context::poll_send_cq(peer_send_wcs_.data(),
                        static_cast<i32>(peer_send_wcs_.size()),
                        peer_context_->get_send_cq(),
                        [&](u64 wr_id) { handle_peer_send_completion(wr_id); });
}

bool MemoryNode::consume_peer_sync_completion(u64 wr_id) {
  std::lock_guard<std::mutex> lock(peer_send_mutex_);
  const auto it = peer_sync_completions_.find(wr_id);
  if (it == peer_sync_completions_.end()) {
    return false;
  }
  peer_sync_completions_.erase(it);
  return true;
}

void MemoryNode::wait_peer_sync_completion(u64 wr_id) {
  while (!consume_peer_sync_completion(wr_id)) {
    poll_peer_send_cq();
    std::this_thread::yield();
  }
}

void MemoryNode::post_peer_read_async(StorageOwnerThread& thread,
                                      u32 shard_id,
                                      u64 remote_offset,
                                      byte_t* dst,
                                      size_t bytes,
                                      size_t local_offset) {
  if (bytes == 0) {
    return;
  }
  lib_assert(peer_context_ != nullptr, "storage peer context is not initialized");
  lib_assert(thread.has_peer_scratch(), "storage-owner thread scratch is not initialized");
  lib_assert(shard_id < num_storage_nodes_, "invalid peer shard id: " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id] != nullptr,
             "peer token is not initialized for shard " + std::to_string(shard_id));
  lib_assert(remote_offset + bytes <= mn_memory_bytes_, "peer RDMA read exceeds shard bounds");
  const u32 qp_idx = peer_data_qp_index(thread.id);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  acquire_peer_rdma_read_credit(shard_id, qp_idx);
  while (peer_async_rdma_outstanding_.load(std::memory_order_acquire) >= peer_rdma_read_global_credit_limit()) {
    poll_peer_send_cq();
    std::this_thread::yield();
  }
  peer_async_rdma_outstanding_.fetch_add(1, std::memory_order_acq_rel);
  thread.track_post();
  const u64 wr_id = next_peer_async_wr_id();
  std::lock_guard<std::mutex> send_lock(peer_send_mutex_);
  register_peer_pending_send_locked(
    wr_id,
    PeerPendingSend{shard_id, qp_idx, thread.id, thread.running_coroutine, true, true});
  qp->post_send(reinterpret_cast<u64>(dst),
                static_cast<u32>(bytes),
                thread.scratch_region->get_lkey(),
                IBV_WR_RDMA_READ,
                true,
                false,
                peer_remote_tokens_[shard_id].get(),
                remote_offset,
                local_offset,
                wr_id);
}

void MemoryNode::remote_read_bytes(u32 shard_id, u64 remote_offset, void* dst, size_t bytes, size_t scratch_offset) {
  if (bytes == 0) return;
  lib_assert(peer_context_ != nullptr, "storage peer context is not initialized");
  lib_assert(shard_id < num_storage_nodes_, "invalid peer shard id: " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id] != nullptr,
             "peer token is not initialized for shard " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id]->address != 0 && peer_remote_tokens_[shard_id]->rkey != 0,
             "peer token is invalid for shard " + std::to_string(shard_id));
  lib_assert(remote_offset + bytes <= mn_memory_bytes_,
             "peer RDMA read exceeds shard bounds: shard=" + std::to_string(shard_id) +
               " offset=" + std::to_string(remote_offset) +
               " bytes=" + std::to_string(bytes) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  static std::atomic<u32> debug_reads{0};
  const u32 debug_idx = debug_reads.fetch_add(1, std::memory_order_relaxed);
  if (debug_idx < 16) {
    std::cerr << "[storage-peer][read] self_shard=" << storage_id_
              << " target_shard=" << shard_id
              << " remote_base=" << peer_remote_tokens_[shard_id]->address
              << " rkey=" << peer_remote_tokens_[shard_id]->rkey
              << " offset=" << remote_offset
              << " bytes=" << bytes << std::endl;
  }
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  const u32 qp_idx = peer_data_qp_index(owner_thread != nullptr ? owner_thread->id : 0);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  HugePage<byte_t>& scratch_buffer =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? owner_thread->scratch_buffer : peer_scratch_buffer_;
  LocalMemoryRegion& scratch_region =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? *owner_thread->scratch_region : *peer_scratch_region_;
  lib_assert(scratch_offset + bytes <= scratch_buffer.buffer_size, "peer scratch buffer exhausted");
  byte_t* scratch = scratch_buffer.get_full_buffer() + scratch_offset;
  acquire_peer_rdma_read_credit(shard_id, qp_idx);
  const u64 wr_id = next_peer_sync_wr_id();
  {
    std::lock_guard<std::mutex> send_lock(peer_send_mutex_);
    register_peer_pending_send_locked(
      wr_id,
      PeerPendingSend{shard_id, qp_idx, 0, 0, false, true});
    qp->post_send(reinterpret_cast<u64>(scratch),
                  static_cast<u32>(bytes),
                  scratch_region.get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  peer_remote_tokens_[shard_id].get(),
                  remote_offset,
                  0,
                  wr_id);
  }
  wait_peer_sync_completion(wr_id);
  std::memcpy(dst, scratch, bytes);
}

void MemoryNode::remote_write_bytes(u32 shard_id, u64 remote_offset, const void* src, size_t bytes, size_t scratch_offset) {
  if (bytes == 0) return;
  lib_assert(peer_context_ != nullptr, "storage peer context is not initialized");
  lib_assert(shard_id < num_storage_nodes_, "invalid peer shard id: " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id] != nullptr,
             "peer token is not initialized for shard " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id]->address != 0 && peer_remote_tokens_[shard_id]->rkey != 0,
             "peer token is invalid for shard " + std::to_string(shard_id));
  lib_assert(remote_offset + bytes <= mn_memory_bytes_,
             "peer RDMA write exceeds shard bounds: shard=" + std::to_string(shard_id) +
               " offset=" + std::to_string(remote_offset) +
               " bytes=" + std::to_string(bytes) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  static std::atomic<u32> debug_writes{0};
  const u32 debug_idx = debug_writes.fetch_add(1, std::memory_order_relaxed);
  if (debug_idx < 16) {
    std::cerr << "[storage-peer][write] self_shard=" << storage_id_
              << " target_shard=" << shard_id
              << " remote_base=" << peer_remote_tokens_[shard_id]->address
              << " rkey=" << peer_remote_tokens_[shard_id]->rkey
              << " offset=" << remote_offset
              << " bytes=" << bytes << std::endl;
  }
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  const u32 qp_idx = peer_data_qp_index(owner_thread != nullptr ? owner_thread->id : 0);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  HugePage<byte_t>& scratch_buffer =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? owner_thread->scratch_buffer : peer_scratch_buffer_;
  LocalMemoryRegion& scratch_region =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? *owner_thread->scratch_region : *peer_scratch_region_;
  lib_assert(scratch_offset + bytes <= scratch_buffer.buffer_size, "peer scratch buffer exhausted");
  byte_t* scratch = scratch_buffer.get_full_buffer() + scratch_offset;
  std::memcpy(scratch, src, bytes);
  const u64 wr_id = next_peer_sync_wr_id();
  {
    std::lock_guard<std::mutex> send_lock(peer_send_mutex_);
    qp->post_send(reinterpret_cast<u64>(scratch),
                  static_cast<u32>(bytes),
                  scratch_region.get_lkey(),
                  IBV_WR_RDMA_WRITE,
                  true,
                  false,
                  peer_remote_tokens_[shard_id].get(),
                  remote_offset,
                  0,
                  wr_id);
  }
  wait_peer_sync_completion(wr_id);
}

u64 MemoryNode::remote_compare_and_swap(u32 shard_id, u64 remote_offset, u64 expected, u64 desired, size_t scratch_offset) {
  lib_assert(peer_context_ != nullptr, "storage peer context is not initialized");
  lib_assert(shard_id < num_storage_nodes_, "invalid peer shard id: " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id] != nullptr,
             "peer token is not initialized for shard " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id]->address != 0 && peer_remote_tokens_[shard_id]->rkey != 0,
             "peer token is invalid for shard " + std::to_string(shard_id));
  lib_assert(remote_offset + sizeof(u64) <= mn_memory_bytes_,
             "peer CAS exceeds shard bounds: shard=" + std::to_string(shard_id) +
               " offset=" + std::to_string(remote_offset) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  static std::atomic<u32> debug_cas{0};
  const u32 debug_idx = debug_cas.fetch_add(1, std::memory_order_relaxed);
  if (debug_idx < 16) {
    std::cerr << "[storage-peer][cas] self_shard=" << storage_id_
              << " target_shard=" << shard_id
              << " remote_base=" << peer_remote_tokens_[shard_id]->address
              << " rkey=" << peer_remote_tokens_[shard_id]->rkey
              << " offset=" << remote_offset
              << " expected=" << expected
              << " desired=" << desired << std::endl;
  }
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  const u32 qp_idx = peer_data_qp_index(owner_thread != nullptr ? owner_thread->id : 0);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  HugePage<byte_t>& scratch_buffer =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? owner_thread->scratch_buffer : peer_scratch_buffer_;
  LocalMemoryRegion& scratch_region =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? *owner_thread->scratch_region : *peer_scratch_region_;
  lib_assert(scratch_offset + sizeof(u64) <= scratch_buffer.buffer_size, "peer scratch buffer exhausted");
  auto* scratch = reinterpret_cast<u64*>(scratch_buffer.get_full_buffer() + scratch_offset);
  *scratch = 0;
  acquire_peer_rdma_read_credit(shard_id, qp_idx);
  const u64 wr_id = next_peer_sync_wr_id();
  {
    std::lock_guard<std::mutex> send_lock(peer_send_mutex_);
    register_peer_pending_send_locked(
      wr_id,
      PeerPendingSend{shard_id, qp_idx, 0, 0, false, true});
    qp->post_CAS(reinterpret_cast<u64>(scratch),
                 scratch_region.get_lkey(),
                 peer_remote_tokens_[shard_id].get(),
                 remote_offset,
                 expected,
                 desired,
                 true,
                 wr_id);
  }
  wait_peer_sync_completion(wr_id);
  return *scratch;
}

std::pair<bool, u64> MemoryNode::try_lock_remote_header(RemotePtr rptr) {
  u64 header = 0;
  remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), &header, sizeof(header), 0);
  if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
    return {false, header};
  }
  const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
  const u64 original = remote_compare_and_swap(rptr.memory_node(), rptr.byte_offset(), header, desired, align_up(sizeof(header)));
  return {original == header, original};
}
