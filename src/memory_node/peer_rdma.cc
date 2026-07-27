#include "memory_node/memory_node.hh"

#include <algorithm>

namespace {

struct PeerScratchLaneRange {
  uintptr_t begin{};
  uintptr_t end{};
};

PeerScratchLaneRange peer_scratch_lane_range(
    const memory_node_detail::StorageOwnerThread& thread) {
  lib_assert(thread.has_peer_scratch(),
             "storage-owner thread scratch is not initialized");
  lib_assert(thread.running_coroutine < thread.post_balances.size(),
             "storage-owner peer read has an invalid coroutine lane");
  lib_assert(thread.scratch_stride != 0,
             "storage-owner peer read has a zero scratch stride");
  lib_assert(
    static_cast<size_t>(thread.running_coroutine) <=
      std::numeric_limits<size_t>::max() / thread.scratch_stride,
    "storage-owner scratch lane offset overflow");
  const size_t lane_offset =
    static_cast<size_t>(thread.running_coroutine) * thread.scratch_stride;
  lib_assert(lane_offset <= thread.scratch_buffer.buffer_size &&
               thread.scratch_stride <=
                 thread.scratch_buffer.buffer_size - lane_offset,
             "storage-owner scratch lane exceeds registered memory");
  const uintptr_t registered_begin = reinterpret_cast<uintptr_t>(
    thread.scratch_buffer.get_full_buffer());
  lib_assert(lane_offset <=
               std::numeric_limits<uintptr_t>::max() - registered_begin,
             "storage-owner scratch lane address overflow");
  const uintptr_t lane_begin = registered_begin + lane_offset;
  lib_assert(thread.scratch_stride <=
               std::numeric_limits<uintptr_t>::max() - lane_begin,
             "storage-owner scratch lane end overflow");
  return PeerScratchLaneRange{
    .begin = lane_begin,
    .end = lane_begin + thread.scratch_stride,
  };
}

std::pair<uintptr_t, uintptr_t> checked_local_read_range(
    byte_t* destination,
    const size_t local_offset,
    const size_t bytes,
    const char* boundary) {
  lib_assert(destination != nullptr,
             str(boundary) + " has a null destination");
  const uintptr_t destination_address =
    reinterpret_cast<uintptr_t>(destination);
  lib_assert(local_offset <=
               std::numeric_limits<uintptr_t>::max() - destination_address,
             str(boundary) + " local offset overflow");
  const uintptr_t begin = destination_address + local_offset;
  lib_assert(bytes <= std::numeric_limits<uintptr_t>::max() - begin,
             str(boundary) + " local range overflow");
  return {begin, begin + bytes};
}

}  // namespace

void MemoryNode::setup_storage_peers(Configuration& config) {
  if (num_storage_nodes_ <= 1) {
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

  peer_qps_per_peer_ = std::max<u32>(
    1, std::min<u32>(kMaxPeerQps,
                     config.storage_owner_peer_qps_per_peer));
  peer_qps_.resize(num_storage_nodes_);
  peer_qp_send_mutexes_.resize(num_storage_nodes_);
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
      peer_qp_send_mutexes_[i].reserve(peer_qps_per_peer_);
      for (u32 qp_idx = 0; qp_idx < peer_qps_per_peer_; ++qp_idx) {
        peer_qp_send_mutexes_[i].push_back(std::make_unique<std::mutex>());
      }
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
  if (peer_qps_per_peer_ > 1) {
    print_status("storage-owner peer QP0 reserved for RPC; data RDMA uses QP1.." +
                 std::to_string(peer_qps_per_peer_ - 1));
  }

  peer_index_region_ = std::make_unique<MemoryRegion>(*peer_context_);
  peer_index_region_->register_memory(index_buffer_.get_full_buffer(), index_buffer_.buffer_size, true);

  const MemoryRegionToken local_token = peer_index_region_->createToken();
  for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
    if (peer_id == storage_id_) continue;
    LocalMemoryRegion peer_token_region{*peer_context_, peer_remote_tokens_[peer_id].get(), sizeof(MemoryRegionToken)};
    peer_control_qp(peer_id)->post_receive(peer_token_region);
    peer_control_qp(peer_id)->post_send_inlined(&local_token, sizeof(local_token), IBV_WR_SEND);
    peer_context_->poll_send_cq_until_completion();
    peer_context_->receive();
  }

  const size_t scratch_bytes = std::max<size_t>(64ull * 1024ull * 1024ull, align_up(VamanaNode::total_size() * 4));
  peer_scratch_buffer_.allocate(scratch_bytes);
  peer_scratch_buffer_.touch_memory();
  peer_scratch_region_ =
    std::make_unique<LocalMemoryRegion>(*peer_context_, peer_scratch_buffer_.get_full_buffer(), scratch_bytes);
  peer_send_wcs_.resize(std::max<i32>(1, peer_context_->get_config().max_send_queue_wr));

  setup_peer_rpc_runtime(config);
  peer_rdma_read_credits_ = derive_peer_rdma_read_credit_plan();
  lib_assert(peer_rdma_read_credits_.per_qp != 0 &&
               peer_rdma_read_credits_.per_peer != 0 &&
               peer_rdma_read_credits_.global != 0,
             "peer send CQ is too small for one RDMA read credit per peer "
             "after reserving peer control traffic");

  for (u32 peer_id = 0; peer_id < num_storage_nodes_; ++peer_id) {
    if (peer_id == storage_id_) continue;
    u64 header_words[2]{};
    remote_read_bytes(peer_id, 0, header_words, sizeof(header_words), 0);
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
  if (peer_qps_per_peer_ == 1) {
    return 0;
  }
  return 1 + worker_id % (peer_qps_per_peer_ - 1);
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

memory_node_detail::PeerRdmaReadCreditPlan
MemoryNode::derive_peer_rdma_read_credit_plan() const {
  const u32 remote_peer_count =
    num_storage_nodes_ > 1 ? num_storage_nodes_ - 1 : 1;
  const u32 data_qp_count =
    memory_node_detail::peer_data_qps_per_peer(peer_qps_per_peer_);

  u32 max_qp_send_wr = std::numeric_limits<u32>::max();
  bool observed_data_qp = false;
  const u32 first_data_qp = peer_qps_per_peer_ <= 1 ? 0 : 1;
  for (u32 peer_id = 0; peer_id < peer_qps_.size(); ++peer_id) {
    if (peer_id == storage_id_) continue;
    for (u32 qp_idx = first_data_qp; qp_idx < peer_qps_per_peer_; ++qp_idx) {
      if (qp_idx >= peer_qps_[peer_id].size() ||
          peer_qps_[peer_id][qp_idx] == nullptr) {
        continue;
      }
      observed_data_qp = true;
      max_qp_send_wr = std::min(
        max_qp_send_wr, peer_qps_[peer_id][qp_idx]->max_send_wr());
    }
  }
  if (!observed_data_qp) {
    max_qp_send_wr = static_cast<u32>(std::max<i32>(
      1, peer_context_ != nullptr
           ? peer_context_->get_config().max_send_queue_wr : 1));
  }

  const u32 device_rd_atomic = peer_context_ != nullptr
    ? std::min<u32>(kPeerSafeRdAtomic,
                    peer_context_->max_qp_read_atomic())
    : kPeerSafeRdAtomic;
  const u32 send_cq_entries = peer_context_ != nullptr &&
      peer_context_->get_send_cq() != nullptr
    ? static_cast<u32>(std::max<i32>(
        1, peer_context_->get_send_cq()->cqe))
    : max_qp_send_wr;

  // Async peer RPC slots can all own a signaled SEND concurrently.  Keep one
  // additional synchronous control completion per peer and one non-read data
  // completion per data QP out of the read-CQE budget.
  const u64 non_read_reserve = static_cast<u64>(remote_peer_count) *
    (static_cast<u64>(peer_rpc_runtime_.send_slots_per_peer) + 1 +
     data_qp_count);
  const u32 reserved_non_read_cq_entries = static_cast<u32>(
    std::min<u64>(non_read_reserve, std::numeric_limits<u32>::max()));

  return memory_node_detail::derive_peer_rdma_read_credit_plan(
    storage_owner_peer_rdma_tokens_, peer_qps_per_peer_, remote_peer_count,
    device_rd_atomic, max_qp_send_wr, send_cq_entries,
    reserved_non_read_cq_entries);
}

const memory_node_detail::PeerRdmaReadCreditPlan&
MemoryNode::peer_rdma_read_credit_plan() const {
  return peer_rdma_read_credits_;
}

u32 MemoryNode::peer_rdma_read_credit_limit_per_qp() const {
  return peer_rdma_read_credits_.per_qp;
}

u32 MemoryNode::peer_rdma_read_credit_limit() const {
  return peer_rdma_read_credits_.per_peer;
}

u32 MemoryNode::peer_rdma_read_global_credit_limit() const {
  return peer_rdma_read_credits_.global;
}

bool MemoryNode::try_acquire_peer_rdma_read_credit(u32 shard_id, u32 qp_idx) {
  lib_assert(shard_id < peer_rdma_read_qp_outstanding_.size(), "invalid peer shard id");
  lib_assert(qp_idx < peer_rdma_read_qp_outstanding_[shard_id].size(), "invalid peer QP index");
  // Synchronous reads and atomics share the same NIC requester/CQ resources
  // as resumable Stage2 waves.  Reserve all three domains transactionally;
  // otherwise a synchronous caller can overrun the global plan while async
  // searches already occupy it.
  return memory_node_detail::try_reserve_peer_rdma_read_group(
    peer_rdma_read_outstanding_[shard_id],
    peer_rdma_read_qp_outstanding_[shard_id][qp_idx],
    peer_async_rdma_outstanding_, peer_rdma_read_credit_plan(), 1);
}

void MemoryNode::acquire_peer_rdma_read_credit(u32 shard_id, u32 qp_idx) {
  while (!try_acquire_peer_rdma_read_credit(shard_id, qp_idx)) {
    poll_peer_send_cq();
    std::this_thread::yield();
  }
}

bool MemoryNode::try_acquire_peer_rdma_read_group(
    u32 shard_id, u32 qp_idx, u32 read_count) {
  lib_assert(shard_id < peer_rdma_read_qp_outstanding_.size(),
             "invalid peer shard id");
  lib_assert(qp_idx < peer_rdma_read_qp_outstanding_[shard_id].size(),
             "invalid peer QP index");
  return memory_node_detail::try_reserve_peer_rdma_read_group(
    peer_rdma_read_outstanding_[shard_id],
    peer_rdma_read_qp_outstanding_[shard_id][qp_idx],
    peer_async_rdma_outstanding_, peer_rdma_read_credit_plan(),
    read_count);
}

void MemoryNode::acquire_peer_rdma_read_group(
    u32 shard_id, u32 qp_idx, u32 read_count) {
  while (!try_acquire_peer_rdma_read_group(
      shard_id, qp_idx, read_count)) {
    // A failed reservation retained no partial credit in any domain. CQ
    // progress can therefore release another producer's posted chain.
    poll_peer_send_cq();
    std::this_thread::yield();
  }
}

u64 MemoryNode::next_peer_sync_wr_id() {
  std::lock_guard<std::mutex> lock(peer_completion_mutex_);
  const auto wr_id = memory_node_detail::next_collision_free_peer_wr_id(
    kPeerSyncWrOwner, peer_sync_wr_id_counter_, [&](const u64 candidate) {
      return peer_reserved_wr_ids_.contains(candidate) ||
        peer_pending_sends_.contains(candidate) ||
        peer_sync_completions_.contains(candidate);
    });
  lib_assert(wr_id.has_value(),
             "peer sync WR-ID namespace is completely occupied");
  const auto [_, inserted] = peer_reserved_wr_ids_.insert(*wr_id);
  lib_assert(inserted, "peer sync WR-ID reservation collided");
  return *wr_id;
}

u64 MemoryNode::next_peer_async_wr_id() {
  std::lock_guard<std::mutex> lock(peer_completion_mutex_);
  const auto wr_id = memory_node_detail::next_collision_free_peer_wr_id(
    kPeerAsyncWrOwner, peer_async_wr_id_counter_, [&](const u64 candidate) {
      return peer_reserved_wr_ids_.contains(candidate) ||
        peer_pending_sends_.contains(candidate) ||
        peer_sync_completions_.contains(candidate);
    });
  lib_assert(wr_id.has_value(),
             "peer async WR-ID namespace is completely occupied");
  const auto [_, inserted] = peer_reserved_wr_ids_.insert(*wr_id);
  lib_assert(inserted, "peer async WR-ID reservation collided");
  return *wr_id;
}

void MemoryNode::register_peer_pending_send_locked(u64 wr_id, PeerPendingSend pending) {
  std::lock_guard<std::mutex> lock(peer_completion_mutex_);
  const auto [owner, ignored_sequence] = decode_64bit(wr_id);
  (void)ignored_sequence;
  if (owner == kPeerSyncWrOwner || owner == kPeerAsyncWrOwner) {
    lib_assert(peer_reserved_wr_ids_.erase(wr_id) == 1,
               "generated peer WR-ID was not reserved before registration");
  }
  lib_assert(!peer_sync_completions_.contains(wr_id),
             "peer WR-ID collides with an unconsumed completion");
  const auto [_, inserted] = peer_pending_sends_.emplace(wr_id, pending);
  lib_assert(inserted,
             "peer WR-ID collides with an active pending send");
}

void MemoryNode::handle_peer_send_completion(u64 wr_id) {
  const auto [completion_owner, completion_sequence] = decode_64bit(wr_id);
  PeerPendingSend pending;
  bool has_pending = false;
  {
    std::lock_guard<std::mutex> lock(peer_completion_mutex_);
    const auto pending_it = peer_pending_sends_.find(wr_id);
    if (pending_it != peer_pending_sends_.end()) {
      pending = pending_it->second;
      peer_pending_sends_.erase(pending_it);
      if (completion_owner == kPeerSyncWrOwner) {
        // Keep the ID live across the pending -> completed transition.  Credit
        // release happens outside this mutex, so without this reservation a
        // sequence wrap could reallocate the ID in that short window.
        const auto [_, inserted] = peer_reserved_wr_ids_.insert(wr_id);
        lib_assert(inserted,
                   "peer sync completion transition reservation collided");
      }
      has_pending = true;
    }
  }
  if (has_pending) {
    if (pending.release_rpc_slot) {
      release_peer_rpc_send_slot(pending.target_shard, pending.rpc_slot_id);
      return;
    }
    const u32 read_count = std::max<u32>(1, pending.rdma_read_count);
    bool released_read_credit = false;
    if (pending.rdma_read_credit) {
      lib_assert(pending.target_shard <
                   peer_rdma_read_outstanding_.size() &&
                   pending.target_shard <
                     peer_rdma_read_qp_outstanding_.size(),
                 "peer RDMA completion has an invalid credit shard");
      lib_assert(pending.target_qp_idx <
                   peer_rdma_read_qp_outstanding_[pending.target_shard].size(),
                 "peer RDMA completion has an invalid credit QP");
      const u32 previous_peer =
        peer_rdma_read_outstanding_[pending.target_shard].fetch_sub(
          read_count, std::memory_order_acq_rel);
      lib_assert(previous_peer >= read_count,
                 "peer RDMA completion underflowed peer read credits");
      const u32 previous_qp =
        peer_rdma_read_qp_outstanding_
          [pending.target_shard][pending.target_qp_idx].fetch_sub(
            read_count, std::memory_order_acq_rel);
      lib_assert(previous_qp >= read_count,
                 "peer RDMA completion underflowed QP read credits");
      const u32 previous_global = peer_async_rdma_outstanding_.fetch_sub(
        read_count, std::memory_order_acq_rel);
      lib_assert(previous_global >= read_count,
                 "peer RDMA completion underflowed global read credits");
      released_read_credit = true;
    }
    if (pending.async) {
      lib_assert(pending.thread != nullptr, "async peer RDMA completion has no owner thread");
      lib_assert(pending.coroutine_id < pending.thread->post_balances.size(),
                 "async peer RDMA completion has invalid coroutine id");
      auto& balance = pending.thread->post_balances[pending.coroutine_id];
      const i32 previous = balance.fetch_sub(1, std::memory_order_acq_rel);
      lib_assert(previous > 0,
                 "async peer RDMA completion underflowed its search lane");
      if (released_read_credit || previous == 1) {
        // A completed chain can make another context credit-runnable before
        // this lane reaches zero.  Notify on every credit release; the same
        // notification also publishes this lane's updated post balance.
        storage_owner_maintenance_cv_.notify_all();
      }
      return;
    }
    if (released_read_credit) {
      // Synchronous reads/atomics share the same credit domains as Stage2.
      // Their caller waits on a different condition variable, so explicitly
      // wake maintenance contexts whose previously rejected wave now fits.
      storage_owner_maintenance_cv_.notify_all();
    }
  }

  if (completion_owner == kPeerSyncWrOwner) {
    {
      std::lock_guard<std::mutex> lock(peer_completion_mutex_);
      // A non-pending sync WR (currently none in normal operation) may still
      // carry an allocation reservation.  Consume it before publishing the
      // completion so wraparound cannot confuse the two states.
      lib_assert(peer_reserved_wr_ids_.erase(wr_id) == 1,
                 "peer sync completion lost its transition reservation");
      const auto [_, inserted] = peer_sync_completions_.insert(wr_id);
      lib_assert(inserted, "duplicate peer sync completion WR-ID");
    }
    peer_completion_cv_.notify_all();
    return;
  }
  if (completion_owner < storage_owner_threads_.size() &&
      storage_owner_threads_[completion_owner]) {
    auto& balance = storage_owner_threads_[completion_owner]
      ->post_balances[completion_sequence];
    const i32 previous = balance.fetch_sub(1, std::memory_order_acq_rel);
    lib_assert(previous > 0,
               "peer RDMA completion underflowed its coroutine lane");
    const u32 previous_global = peer_async_rdma_outstanding_.fetch_sub(
      1, std::memory_order_acq_rel);
    lib_assert(previous_global > 0,
               "peer RDMA completion underflowed legacy global credits");
    // The credit release can unblock a different context even when this
    // particular coroutine still owns more completions.
    storage_owner_maintenance_cv_.notify_all();
  }
}

void MemoryNode::poll_peer_send_cq() {
  if (!peer_context_) {
    return;
  }
  std::lock_guard<std::mutex> lock(peer_send_cq_mutex_);
  Context::poll_send_cq(peer_send_wcs_.data(),
                        static_cast<i32>(peer_send_wcs_.size()),
                        peer_context_->get_send_cq(),
                        [&](u64 wr_id) { handle_peer_send_completion(wr_id); });
}

bool MemoryNode::consume_peer_sync_completion(u64 wr_id) {
  std::lock_guard<std::mutex> lock(peer_completion_mutex_);
  const auto it = peer_sync_completions_.find(wr_id);
  if (it == peer_sync_completions_.end()) {
    return false;
  }
  peer_sync_completions_.erase(it);
  return true;
}

void MemoryNode::wait_peer_sync_completion(u64 wr_id) {
  if (peer_rpc_progress_running_.load(std::memory_order_acquire) &&
      !current_peer_rpc_progress_thread_) {
    std::unique_lock<std::mutex> lock(peer_completion_mutex_);
    peer_completion_cv_.wait(lock, [&]() {
      return peer_sync_completions_.contains(wr_id) ||
             !peer_rpc_progress_running_.load(std::memory_order_acquire);
    });
    const auto completion = peer_sync_completions_.find(wr_id);
    if (completion != peer_sync_completions_.end()) {
      peer_sync_completions_.erase(completion);
      return;
    }
  }
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
  lib_assert(shard_id < num_storage_nodes_ && shard_id != storage_id_,
             "invalid remote peer shard id: " + std::to_string(shard_id));
  lib_assert(peer_remote_tokens_[shard_id] != nullptr,
             "peer token is not initialized for shard " + std::to_string(shard_id));
  lib_assert(remote_offset <= mn_memory_bytes_ &&
               bytes <= mn_memory_bytes_ - remote_offset,
             "peer RDMA read exceeds shard bounds");
  lib_assert(bytes <= std::numeric_limits<u32>::max(),
             "peer RDMA read exceeds verbs SGE length");
  lib_assert(remote_offset <= std::numeric_limits<u64>::max() -
                 peer_remote_tokens_[shard_id]->address,
             "peer RDMA remote address overflow");
  const u64 remote_address =
    peer_remote_tokens_[shard_id]->address + remote_offset;
  lib_assert(bytes <= std::numeric_limits<u64>::max() - remote_address,
             "peer RDMA remote range overflow");
  const PeerScratchLaneRange lane = peer_scratch_lane_range(thread);
  const auto [local_begin, local_end] = checked_local_read_range(
    dst, local_offset, bytes, "peer RDMA read");
  lib_assert(local_begin >= lane.begin && local_end <= lane.end,
             "peer RDMA read exceeds its registered scratch lane");
  const u32 qp_idx = memory_node_detail::select_peer_data_qp(
    peer_qps_per_peer_, thread.id + thread.next_peer_data_qp_ticket++);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  acquire_peer_rdma_read_group(shard_id, qp_idx, 1);
  thread.track_post();
  const u64 wr_id = next_peer_async_wr_id();
  register_peer_pending_send_locked(
    wr_id,
    PeerPendingSend{shard_id, qp_idx, thread.id, thread.running_coroutine, &thread, true, true});
  std::lock_guard<std::mutex> send_lock(*peer_qp_send_mutexes_[shard_id][qp_idx]);
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

void MemoryNode::post_peer_reads_async(
    StorageOwnerThread& thread,
    span<const PeerReadRequest> requests) {
  const bool posted = post_peer_reads_async_impl(thread, requests, false);
  lib_assert(posted, "blocking peer RDMA read post unexpectedly deferred");
}

bool MemoryNode::try_post_peer_reads_async(
    StorageOwnerThread& thread,
    span<const PeerReadRequest> requests) {
  return post_peer_reads_async_impl(thread, requests, true);
}

bool MemoryNode::post_peer_reads_async_impl(
    StorageOwnerThread& thread,
    span<const PeerReadRequest> requests,
    const bool try_only) {
  if (requests.empty()) return true;
  lib_assert(peer_context_ != nullptr,
             "storage peer context is not initialized");
  lib_assert(thread.has_peer_scratch(),
             "storage-owner thread scratch is not initialized");
  const PeerScratchLaneRange lane = peer_scratch_lane_range(thread);

  struct AssignedRead {
    PeerReadRequest request;
    u32 qp_idx{};
  };
  // A caller wave is bounded by storage_owner_search_snapshot_batch. Retain
  // only that bounded high-water capacity on its owning maintenance thread.
  thread_local vec<vec<AssignedRead>> groups;
  const size_t group_count =
    static_cast<size_t>(num_storage_nodes_) * peer_qps_per_peer_;
  groups.resize(group_count);
  for (vec<AssignedRead>& group : groups) group.clear();

  const u32 chain_limit =
    memory_node_detail::peer_rdma_read_batch_group_limit(
      peer_rdma_read_credit_plan());
  lib_assert(chain_limit != 0,
             "peer RDMA read batch has no transport credit");
  thread_local vec<u32> active_qp_by_shard;
  thread_local vec<u32> active_fill_by_shard;
  thread_local vec<u32> wave_qp_start_by_shard;
  thread_local vec<u32> wave_qp_chain_by_shard;
  thread_local vec<u8> wave_qp_initialized_by_shard;
  active_qp_by_shard.resize(num_storage_nodes_);
  active_fill_by_shard.assign(num_storage_nodes_, 0);
  wave_qp_start_by_shard.resize(num_storage_nodes_);
  wave_qp_chain_by_shard.assign(num_storage_nodes_, 0);
  wave_qp_initialized_by_shard.assign(num_storage_nodes_, 0);

  for (const PeerReadRequest& request : requests) {
    if (request.bytes == 0) continue;
    lib_assert(request.destination != nullptr,
               "peer RDMA read batch has a null destination");
    lib_assert(request.shard_id < num_storage_nodes_ &&
                 request.shard_id != storage_id_,
               "peer RDMA read batch has an invalid remote shard");
    lib_assert(peer_remote_tokens_[request.shard_id] != nullptr,
               "peer RDMA read batch has no remote token");
    lib_assert(request.remote_offset <= mn_memory_bytes_ &&
                 request.bytes <= mn_memory_bytes_ - request.remote_offset,
               "peer RDMA read batch exceeds shard bounds");
    lib_assert(request.bytes <= std::numeric_limits<u32>::max(),
               "peer RDMA read batch item exceeds verbs SGE length");
    lib_assert(request.remote_offset <= std::numeric_limits<u64>::max() -
                   peer_remote_tokens_[request.shard_id]->address,
               "peer RDMA read batch remote address overflow");
    const u64 remote_address =
      peer_remote_tokens_[request.shard_id]->address +
      request.remote_offset;
    lib_assert(request.bytes <=
                 std::numeric_limits<u64>::max() - remote_address,
               "peer RDMA read batch remote range overflow");
    const auto [local_begin, local_end] = checked_local_read_range(
      request.destination, request.local_offset, request.bytes,
      "peer RDMA read batch item");
    lib_assert(local_begin >= lane.begin && local_end <= lane.end,
               "peer RDMA read batch exceeds its registered scratch lane");
    // Rotate QPs once per bounded linked chain, not once per WR. Striping
    // every individual item makes every chain a singleton when a node owns
    // many QPs and defeats CQ/WQE batching. Independent waves and workers
    // still rotate across every data QP, while one chain retains exact
    // per-(remote shard, QP) ownership.
    if (active_fill_by_shard[request.shard_id] == 0 ||
        active_fill_by_shard[request.shard_id] == chain_limit) {
      if (wave_qp_initialized_by_shard[request.shard_id] == 0) {
        // Take one rotating start ticket per shard and wave. Chains of this
        // shard then advance by their own ordinal, so interleaved requests
        // for other shards cannot consume its QP sequence and make it reuse
        // one QP before visiting every data lane.
        wave_qp_start_by_shard[request.shard_id] =
          thread.id + thread.next_peer_data_qp_ticket++;
        wave_qp_initialized_by_shard[request.shard_id] = 1;
      }
      active_qp_by_shard[request.shard_id] =
        memory_node_detail::select_peer_data_qp_for_wave_chain(
          peer_qps_per_peer_,
          wave_qp_start_by_shard[request.shard_id],
          wave_qp_chain_by_shard[request.shard_id]++);
      active_fill_by_shard[request.shard_id] = 0;
    }
    const u32 qp_idx = active_qp_by_shard[request.shard_id];
    ++active_fill_by_shard[request.shard_id];
    const size_t group_index =
      static_cast<size_t>(request.shard_id) * peer_qps_per_peer_ + qp_idx;
    groups[group_index].push_back({request, qp_idx});
  }

  thread_local vec<ibv_send_wr> work_requests;
  thread_local vec<ibv_sge> scatter_gather_entries;

  thread_local vec<memory_node_detail::PeerRdmaReadCreditRequest>
    credit_requests;
  credit_requests.clear();
  u64 wave_read_count = 0;
  thread_local vec<u64> wave_reads_by_shard;
  thread_local vec<u64> wave_reads_by_qp;
  wave_reads_by_shard.assign(num_storage_nodes_, 0);
  wave_reads_by_qp.assign(group_count, 0);
  for (const vec<AssignedRead>& group : groups) {
    if (group.empty()) continue;
    const u32 shard_id = group.front().request.shard_id;
    const u32 qp_idx = group.front().qp_idx;
    const size_t group_index =
      static_cast<size_t>(shard_id) * peer_qps_per_peer_ + qp_idx;
    for (size_t begin = 0; begin < group.size(); begin += chain_limit) {
      const size_t end = std::min(
        group.size(), begin + static_cast<size_t>(chain_limit));
      const u32 read_count = static_cast<u32>(end - begin);
      credit_requests.push_back({
        .peer_outstanding = &peer_rdma_read_outstanding_[shard_id],
        .qp_outstanding =
          &peer_rdma_read_qp_outstanding_[shard_id][qp_idx],
        .count = read_count,
      });
      wave_read_count += read_count;
      wave_reads_by_shard[shard_id] += read_count;
      wave_reads_by_qp[group_index] += read_count;
    }
  }

  if (try_only) {
    // A try-post is all-or-nothing.  Reject a programming error explicitly
    // instead of returning false forever for a wave that can never fit the
    // static transport window.  The resumable Stage2 caller chunks waves to
    // these same per-QP, per-peer and global limits.
    lib_assert(wave_read_count <= peer_rdma_read_global_credit_limit(),
               "nonblocking peer RDMA read wave exceeds global credit");
    for (u32 shard_id = 0; shard_id < num_storage_nodes_; ++shard_id) {
      lib_assert(wave_reads_by_shard[shard_id] <=
                   peer_rdma_read_credit_limit(),
                 "nonblocking peer RDMA read wave exceeds peer credit");
    }
    for (u64 count : wave_reads_by_qp) {
      lib_assert(count <= peer_rdma_read_credit_limit_per_qp(),
                 "nonblocking peer RDMA read wave exceeds QP credit");
    }

    if (!memory_node_detail::try_reserve_peer_rdma_read_wave(
          span<const memory_node_detail::PeerRdmaReadCreditRequest>{
            credit_requests},
          peer_async_rdma_outstanding_, peer_rdma_read_credit_plan())) {
      // The helper rolls back the complete prefix before returning to the
      // scheduler. A false result therefore owns no partial credit and has
      // posted no WR.
      return false;
    }
  }

  for (vec<AssignedRead>& group : groups) {
    if (group.empty()) continue;
    const u32 shard_id = group.front().request.shard_id;
    const u32 qp_idx = group.front().qp_idx;
    QP& qp = peer_data_qp(shard_id, qp_idx);
    for (size_t begin = 0; begin < group.size(); begin += chain_limit) {
      const size_t end = std::min(
        group.size(), begin + static_cast<size_t>(chain_limit));
      const u32 read_count = static_cast<u32>(end - begin);

      // Reserve the complete chain all-or-nothing. A producer never waits
      // while holding credits for an unposted prefix, so two maintenance
      // workers cannot split the global window and deadlock each other.
      if (!try_only) {
        acquire_peer_rdma_read_group(shard_id, qp_idx, read_count);
      }

      work_requests.resize(read_count);
      scatter_gather_entries.resize(read_count);
      const MemoryRegionToken& token = *peer_remote_tokens_[shard_id];
      for (u32 item = 0; item < read_count; ++item) {
        const PeerReadRequest& request = group[begin + item].request;
        ibv_sge& sge = scatter_gather_entries[item];
        ibv_send_wr& wr = work_requests[item];
        sge = {};
        wr = {};
        sge.addr = reinterpret_cast<u64>(request.destination) +
          request.local_offset;
        sge.length = static_cast<u32>(request.bytes);
        sge.lkey = thread.scratch_region->get_lkey();
        wr.opcode = IBV_WR_RDMA_READ;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.wr.rdma.remote_addr = token.address + request.remote_offset;
        wr.wr.rdma.rkey = token.rkey;
        wr.next = item + 1 < read_count
          ? &work_requests[item + 1] : nullptr;
      }

      // RC preserves the order of this per-(remote shard, QP) chain. A
      // successful tail CQE means every preceding unsignaled READ completed;
      // verbs still reports an error CQE for a failed unsignaled WR, and the
      // shared CQ poller fail-stops on every non-success WC.
      const u64 wr_id = next_peer_async_wr_id();
      work_requests.back().send_flags = IBV_SEND_SIGNALED;
      work_requests.back().wr_id = wr_id;
      thread.track_post();
      register_peer_pending_send_locked(
        wr_id,
        PeerPendingSend{
          .target_shard = shard_id,
          .target_qp_idx = qp_idx,
          .thread_id = thread.id,
          .coroutine_id = thread.running_coroutine,
          .thread = &thread,
          .async = true,
          .rdma_read_credit = true,
          .rdma_read_count = read_count,
        });
      ibv_send_wr* bad_work_request = nullptr;
      {
        std::lock_guard<std::mutex> send_lock(
          *peer_qp_send_mutexes_[shard_id][qp_idx]);
        lib_assert(
          ibv_post_send(qp->get_ibv_qp(), work_requests.data(),
                        &bad_work_request) == 0,
          "cannot post linked peer RDMA read batch");
      }
    }
  }
  return true;
}

void MemoryNode::post_peer_read_pairs_async(
    StorageOwnerThread& thread,
    span<const PeerReadPairRequest> requests) {
  const bool posted = post_peer_read_pairs_async_impl(
    thread, requests, false);
  lib_assert(posted,
             "blocking ordered peer snapshot post unexpectedly deferred");
}

bool MemoryNode::try_post_peer_read_pairs_async(
    StorageOwnerThread& thread,
    span<const PeerReadPairRequest> requests) {
  return post_peer_read_pairs_async_impl(thread, requests, true);
}

bool MemoryNode::post_peer_read_pairs_async_impl(
    StorageOwnerThread& thread,
    span<const PeerReadPairRequest> requests,
    const bool try_only) {
  if (requests.empty()) return true;
  lib_assert(peer_context_ != nullptr,
             "storage peer context is not initialized");
  lib_assert(thread.has_peer_scratch(),
             "storage-owner thread scratch is not initialized");

  const u32 pair_chain_limit =
    memory_node_detail::peer_rdma_read_pair_group_limit(
      peer_rdma_read_credit_plan());
  lib_assert(pair_chain_limit != 0,
             "ordered peer snapshot pair has fewer than two transport "
             "credits; caller must use the two-wave fallback");

  struct AssignedPair {
    PeerReadPairRequest request;
    u32 qp_idx{};
  };
  thread_local vec<vec<AssignedPair>> groups;
  const size_t group_count =
    static_cast<size_t>(num_storage_nodes_) * peer_qps_per_peer_;
  groups.resize(group_count);
  for (vec<AssignedPair>& group : groups) group.clear();

  thread_local vec<u32> active_qp_by_shard;
  thread_local vec<u32> active_fill_by_shard;
  thread_local vec<u32> wave_qp_start_by_shard;
  thread_local vec<u32> wave_qp_chain_by_shard;
  thread_local vec<u8> wave_qp_initialized_by_shard;
  active_qp_by_shard.resize(num_storage_nodes_);
  active_fill_by_shard.assign(num_storage_nodes_, 0);
  wave_qp_start_by_shard.resize(num_storage_nodes_);
  wave_qp_chain_by_shard.assign(num_storage_nodes_, 0);
  wave_qp_initialized_by_shard.assign(num_storage_nodes_, 0);

  const PeerScratchLaneRange lane = peer_scratch_lane_range(thread);
  const auto local_range = [&](const PeerReadRequest& request) {
    return checked_local_read_range(
      request.destination, request.local_offset, request.bytes,
      "ordered peer snapshot");
  };

  for (const PeerReadPairRequest& pair : requests) {
    const PeerReadRequest& full = pair.full_snapshot;
    const PeerReadRequest& after = pair.after_header;
    lib_assert(full.destination != nullptr && after.destination != nullptr,
               "ordered peer snapshot pair has a null destination");
    lib_assert(full.bytes >= VamanaNode::HEADER_SIZE &&
                 after.bytes == VamanaNode::HEADER_SIZE,
               "ordered peer snapshot pair has invalid read sizes");
    lib_assert(full.shard_id == after.shard_id &&
                 full.remote_offset == after.remote_offset,
               "ordered peer snapshot pair crossed a remote record");
    lib_assert(full.shard_id < num_storage_nodes_ &&
                 full.shard_id != storage_id_,
               "ordered peer snapshot pair has an invalid remote shard");
    lib_assert(peer_remote_tokens_[full.shard_id] != nullptr,
               "ordered peer snapshot pair has no remote token");
    lib_assert(full.remote_offset <= mn_memory_bytes_ &&
                 full.bytes <= mn_memory_bytes_ - full.remote_offset &&
                 after.bytes <= mn_memory_bytes_ - after.remote_offset,
               "ordered peer snapshot pair exceeds shard bounds");
    lib_assert(full.bytes <= std::numeric_limits<u32>::max(),
               "ordered peer full snapshot exceeds verbs SGE length");
    lib_assert(full.remote_offset <= std::numeric_limits<u64>::max() -
                   peer_remote_tokens_[full.shard_id]->address,
               "ordered peer snapshot remote address overflow");
    const u64 remote_address =
      peer_remote_tokens_[full.shard_id]->address + full.remote_offset;
    lib_assert(full.bytes <=
                 std::numeric_limits<u64>::max() - remote_address,
               "ordered peer snapshot remote range overflow");

    const auto [full_begin, full_end] = local_range(full);
    const auto [after_begin, after_end] = local_range(after);
    lib_assert(full_begin >= lane.begin && full_end <= lane.end &&
                 after_begin >= lane.begin && after_end <= lane.end,
               "ordered peer snapshot pair exceeds registered scratch lane");
    lib_assert(full_end <= after_begin || after_end <= full_begin,
               "ordered peer after-header overlaps the before snapshot");

    if (active_fill_by_shard[full.shard_id] == 0 ||
        active_fill_by_shard[full.shard_id] == pair_chain_limit) {
      if (wave_qp_initialized_by_shard[full.shard_id] == 0) {
        wave_qp_start_by_shard[full.shard_id] =
          thread.id + thread.next_peer_data_qp_ticket++;
        wave_qp_initialized_by_shard[full.shard_id] = 1;
      }
      active_qp_by_shard[full.shard_id] =
        memory_node_detail::select_peer_data_qp_for_wave_chain(
          peer_qps_per_peer_,
          wave_qp_start_by_shard[full.shard_id],
          wave_qp_chain_by_shard[full.shard_id]++);
      active_fill_by_shard[full.shard_id] = 0;
    }
    const u32 qp_idx = active_qp_by_shard[full.shard_id];
    ++active_fill_by_shard[full.shard_id];
    const size_t group_index =
      static_cast<size_t>(full.shard_id) * peer_qps_per_peer_ + qp_idx;
    groups[group_index].push_back({pair, qp_idx});
  }

  thread_local vec<ibv_send_wr> work_requests;
  thread_local vec<ibv_sge> scatter_gather_entries;

  thread_local vec<memory_node_detail::PeerRdmaReadCreditRequest>
    credit_requests;
  credit_requests.clear();
  u64 wave_read_count = 0;
  thread_local vec<u64> wave_reads_by_shard;
  thread_local vec<u64> wave_reads_by_qp;
  wave_reads_by_shard.assign(num_storage_nodes_, 0);
  wave_reads_by_qp.assign(group_count, 0);
  for (const vec<AssignedPair>& group : groups) {
    if (group.empty()) continue;
    const u32 shard_id = group.front().request.full_snapshot.shard_id;
    const u32 qp_idx = group.front().qp_idx;
    const size_t group_index =
      static_cast<size_t>(shard_id) * peer_qps_per_peer_ + qp_idx;
    for (size_t begin = 0; begin < group.size();
         begin += pair_chain_limit) {
      const size_t end = std::min(
        group.size(), begin + static_cast<size_t>(pair_chain_limit));
      const u32 pair_count = static_cast<u32>(end - begin);
      const u32 read_count =
        memory_node_detail::peer_rdma_read_pair_work_request_count(
          pair_count);
      lib_assert(read_count != 0,
                 "ordered peer snapshot pair WR count overflow");
      credit_requests.push_back({
        .peer_outstanding = &peer_rdma_read_outstanding_[shard_id],
        .qp_outstanding =
          &peer_rdma_read_qp_outstanding_[shard_id][qp_idx],
        .count = read_count,
      });
      wave_read_count += read_count;
      wave_reads_by_shard[shard_id] += read_count;
      wave_reads_by_qp[group_index] += read_count;
    }
  }

  if (try_only) {
    lib_assert(wave_read_count <= peer_rdma_read_global_credit_limit(),
               "nonblocking ordered snapshot wave exceeds global credit");
    for (u32 shard_id = 0; shard_id < num_storage_nodes_; ++shard_id) {
      lib_assert(wave_reads_by_shard[shard_id] <=
                   peer_rdma_read_credit_limit(),
                 "nonblocking ordered snapshot wave exceeds peer credit");
    }
    for (u64 count : wave_reads_by_qp) {
      lib_assert(count <= peer_rdma_read_credit_limit_per_qp(),
                 "nonblocking ordered snapshot wave exceeds QP credit");
    }

    if (!memory_node_detail::try_reserve_peer_rdma_read_wave(
          span<const memory_node_detail::PeerRdmaReadCreditRequest>{
            credit_requests},
          peer_async_rdma_outstanding_, peer_rdma_read_credit_plan())) {
      return false;
    }
  }

  for (vec<AssignedPair>& group : groups) {
    if (group.empty()) continue;
    const u32 shard_id = group.front().request.full_snapshot.shard_id;
    const u32 qp_idx = group.front().qp_idx;
    QP& qp = peer_data_qp(shard_id, qp_idx);
    for (size_t begin = 0; begin < group.size();
         begin += pair_chain_limit) {
      const size_t end = std::min(
        group.size(), begin + static_cast<size_t>(pair_chain_limit));
      const u32 pair_count = static_cast<u32>(end - begin);
      const u32 read_count =
        memory_node_detail::peer_rdma_read_pair_work_request_count(
          pair_count);
      lib_assert(read_count != 0,
                 "ordered peer snapshot pair WR count overflow");

      // Reserve both reads of every pair in one all-or-nothing credit group.
      // A pair is never split across QPs/chains, and the only signaled WR is
      // the chain tail.  RC ordering therefore guarantees full snapshot ->
      // after-header before the caller observes completion.
      if (!try_only) {
        acquire_peer_rdma_read_group(shard_id, qp_idx, read_count);
      }
      work_requests.resize(read_count);
      scatter_gather_entries.resize(read_count);
      const MemoryRegionToken& token = *peer_remote_tokens_[shard_id];
      for (u32 wr_index = 0; wr_index < read_count; ++wr_index) {
        const auto chain_item =
          memory_node_detail::peer_rdma_read_pair_chain_item(
            wr_index, pair_count);
        const PeerReadPairRequest& pair =
          group[begin + chain_item.pair_index].request;
        const PeerReadRequest& request = chain_item.after_header
          ? pair.after_header : pair.full_snapshot;
        ibv_sge& sge = scatter_gather_entries[wr_index];
        ibv_send_wr& wr = work_requests[wr_index];
        sge = {};
        wr = {};
        sge.addr = reinterpret_cast<u64>(request.destination) +
          request.local_offset;
        sge.length = static_cast<u32>(request.bytes);
        sge.lkey = thread.scratch_region->get_lkey();
        wr.opcode = IBV_WR_RDMA_READ;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.wr.rdma.remote_addr = token.address + request.remote_offset;
        wr.wr.rdma.rkey = token.rkey;
        // The first validation READ fences the complete preceding full-body
        // half of this chain.  Later headers remain after that boundary while
        // the HCA can batch them together; fencing every pair would serialize
        // the chain into N dependent READ round trips.
        if (wr_index == pair_count) {
          wr.send_flags = IBV_SEND_FENCE;
        }
        wr.next = wr_index + 1 < read_count
          ? &work_requests[wr_index + 1] : nullptr;
      }

      const u64 wr_id = next_peer_async_wr_id();
      work_requests.back().send_flags |= IBV_SEND_SIGNALED;
      work_requests.back().wr_id = wr_id;
      thread.track_post();
      register_peer_pending_send_locked(
        wr_id,
        PeerPendingSend{
          .target_shard = shard_id,
          .target_qp_idx = qp_idx,
          .thread_id = thread.id,
          .coroutine_id = thread.running_coroutine,
          .thread = &thread,
          .async = true,
          .rdma_read_credit = true,
          .rdma_read_count = read_count,
        });
      ibv_send_wr* bad_work_request = nullptr;
      {
        std::lock_guard<std::mutex> send_lock(
          *peer_qp_send_mutexes_[shard_id][qp_idx]);
        lib_assert(
          ibv_post_send(qp->get_ibv_qp(), work_requests.data(),
                        &bad_work_request) == 0,
          "cannot post ordered peer snapshot pair batch");
      }
    }
  }
  return true;
}

bool MemoryNode::try_post_peer_snapshot_reads_async(
    StorageOwnerThread& thread,
    span<const PeerReadSnapshotRequest> requests) {
  if (requests.empty()) return true;
  lib_assert(peer_context_ != nullptr,
             "storage peer context is not initialized");
  lib_assert(thread.has_peer_scratch(),
             "storage-owner thread scratch is not initialized");

  const auto& plan = peer_rdma_read_credit_plan();
  const u32 chain_limit =
    memory_node_detail::peer_rdma_read_batch_group_limit(plan);
  lib_assert(chain_limit != 0,
             "mixed peer snapshot has no transport credit");
  const PeerScratchLaneRange lane = peer_scratch_lane_range(thread);

  struct AssignedSnapshot {
    PeerReadSnapshotRequest request;
    u32 qp_idx{};
  };
  thread_local vec<vec<PeerReadSnapshotRequest>> by_shard;
  thread_local vec<vec<AssignedSnapshot>> groups;
  const size_t group_count =
    static_cast<size_t>(num_storage_nodes_) * peer_qps_per_peer_;
  by_shard.resize(num_storage_nodes_);
  groups.resize(group_count);
  for (vec<PeerReadSnapshotRequest>& shard : by_shard) shard.clear();
  for (vec<AssignedSnapshot>& group : groups) group.clear();

  const auto validate_read = [&](const PeerReadRequest& request,
                                 const char* boundary) {
    lib_assert(request.destination != nullptr,
               str(boundary) + " has a null destination");
    lib_assert(request.shard_id < num_storage_nodes_ &&
                 request.shard_id != storage_id_,
               str(boundary) + " has an invalid remote shard");
    lib_assert(peer_remote_tokens_[request.shard_id] != nullptr,
               str(boundary) + " has no remote token");
    lib_assert(request.remote_offset <= mn_memory_bytes_ &&
                 request.bytes <= mn_memory_bytes_ - request.remote_offset,
               str(boundary) + " exceeds shard bounds");
    lib_assert(request.bytes <= std::numeric_limits<u32>::max(),
               str(boundary) + " exceeds verbs SGE length");
    lib_assert(request.remote_offset <= std::numeric_limits<u64>::max() -
                   peer_remote_tokens_[request.shard_id]->address,
               str(boundary) + " remote address overflow");
    const u64 remote_address =
      peer_remote_tokens_[request.shard_id]->address +
      request.remote_offset;
    lib_assert(request.bytes <=
                 std::numeric_limits<u64>::max() - remote_address,
               str(boundary) + " remote range overflow");
    const auto [local_begin, local_end] = checked_local_read_range(
      request.destination, request.local_offset, request.bytes, boundary);
    lib_assert(local_begin >= lane.begin && local_end <= lane.end,
               str(boundary) + " exceeds registered scratch lane");
    return std::pair{local_begin, local_end};
  };

  thread_local vec<u32> wrs_by_shard;
  thread_local vec<u32> pairs_by_shard;
  wrs_by_shard.assign(num_storage_nodes_, 0);
  pairs_by_shard.assign(num_storage_nodes_, 0);
  u64 wave_wr_count = 0;
  for (const PeerReadSnapshotRequest& snapshot : requests) {
    const PeerReadRequest& full = snapshot.full_snapshot;
    lib_assert(full.bytes >= VamanaNode::HEADER_SIZE,
               "mixed peer snapshot body is shorter than its header");
    const auto [full_begin, full_end] =
      validate_read(full, "mixed peer snapshot body");
    if (snapshot.after_header.has_value()) {
      const PeerReadRequest& after = *snapshot.after_header;
      lib_assert(after.bytes == VamanaNode::HEADER_SIZE &&
                   after.shard_id == full.shard_id &&
                   after.remote_offset == full.remote_offset,
                 "mixed peer snapshot validation crossed a remote record");
      const auto [after_begin, after_end] =
        validate_read(after, "mixed peer snapshot after-header");
      lib_assert(full_end <= after_begin || after_end <= full_begin,
                 "mixed peer snapshot after-header overlaps its body");
      ++pairs_by_shard[full.shard_id];
    }
    const u32 cost = snapshot.after_header.has_value() ? 2u : 1u;
    lib_assert(wrs_by_shard[full.shard_id] <=
                 std::numeric_limits<u32>::max() - cost,
               "mixed peer snapshot peer WR count overflow");
    wrs_by_shard[full.shard_id] += cost;
    wave_wr_count += cost;
    by_shard[full.shard_id].push_back(snapshot);
  }

  const auto limits =
    memory_node_detail::peer_rdma_snapshot_dispatch_limits(plan);
  lib_assert(wave_wr_count <= limits.global_wrs,
             "mixed peer snapshot wave exceeds global credit");
  for (u32 shard_id = 0; shard_id < num_storage_nodes_; ++shard_id) {
    lib_assert(wrs_by_shard[shard_id] <= limits.per_peer_wrs,
               "mixed peer snapshot wave exceeds peer credit");
    lib_assert(pairs_by_shard[shard_id] <= limits.per_peer_pairs,
               "mixed peer snapshot wave cannot keep every pair on one QP");
  }

  // Pairs cost two indivisible WR slots. Place them before singles so an odd
  // per-QP remainder can always be filled by one immutable-base body instead
  // of fragmenting capacity that a later pair would require.
  const u32 data_qps = plan.data_qps_per_peer;
  lib_assert(data_qps != 0,
             "mixed peer snapshot has no data QP");
  thread_local vec<u32> qp_loads;
  qp_loads.resize(data_qps);
  for (u32 shard_id = 0; shard_id < num_storage_nodes_; ++shard_id) {
    vec<PeerReadSnapshotRequest>& shard = by_shard[shard_id];
    if (shard.empty()) continue;
    std::stable_sort(
      shard.begin(), shard.end(),
      [](const PeerReadSnapshotRequest& left,
         const PeerReadSnapshotRequest& right) {
        return left.after_header.has_value() >
          right.after_header.has_value();
      });
    std::fill(qp_loads.begin(), qp_loads.end(), 0);
    const u32 wave_start =
      thread.id + thread.next_peer_data_qp_ticket++;
    for (const PeerReadSnapshotRequest& snapshot : shard) {
      const u32 cost = snapshot.after_header.has_value() ? 2u : 1u;
      u32 bin = data_qps;
      for (u32 candidate = 0; candidate < data_qps; ++candidate) {
        if (cost <= chain_limit &&
            qp_loads[candidate] <= chain_limit - cost) {
          bin = candidate;
          break;
        }
      }
      lib_assert(bin != data_qps,
                 "mixed peer snapshot QP packing exceeded static credit");
      qp_loads[bin] += cost;
      const u32 qp_idx =
        memory_node_detail::select_peer_data_qp_for_wave_chain(
          peer_qps_per_peer_, wave_start, bin);
      const size_t group_index =
        static_cast<size_t>(shard_id) * peer_qps_per_peer_ + qp_idx;
      groups[group_index].push_back({snapshot, qp_idx});
    }
  }

  thread_local vec<memory_node_detail::PeerRdmaReadCreditRequest>
    credit_requests;
  credit_requests.clear();
  for (const vec<AssignedSnapshot>& group : groups) {
    if (group.empty()) continue;
    u32 pair_count = 0;
    for (const AssignedSnapshot& assigned : group) {
      pair_count += assigned.request.after_header.has_value();
    }
    const u32 read_count =
      memory_node_detail::peer_rdma_snapshot_work_request_count(
        static_cast<u32>(group.size()), pair_count);
    lib_assert(read_count != 0 && read_count <= chain_limit,
               "mixed peer snapshot chain exceeds QP credit");
    const u32 shard_id = group.front().request.full_snapshot.shard_id;
    const u32 qp_idx = group.front().qp_idx;
    credit_requests.push_back({
      .peer_outstanding = &peer_rdma_read_outstanding_[shard_id],
      .qp_outstanding =
        &peer_rdma_read_qp_outstanding_[shard_id][qp_idx],
      .count = read_count,
    });
  }
  if (!memory_node_detail::try_reserve_peer_rdma_read_wave(
        span<const memory_node_detail::PeerRdmaReadCreditRequest>{
          credit_requests},
        peer_async_rdma_outstanding_, plan)) {
    return false;
  }

  thread_local vec<ibv_send_wr> work_requests;
  thread_local vec<ibv_sge> scatter_gather_entries;
  thread_local vec<u32> pair_indices;
  for (vec<AssignedSnapshot>& group : groups) {
    if (group.empty()) continue;
    const u32 shard_id = group.front().request.full_snapshot.shard_id;
    const u32 qp_idx = group.front().qp_idx;
    QP& qp = peer_data_qp(shard_id, qp_idx);
    pair_indices.clear();
    for (u32 snapshot_index = 0;
         snapshot_index < group.size(); ++snapshot_index) {
      if (group[snapshot_index].request.after_header.has_value()) {
        pair_indices.push_back(snapshot_index);
      }
    }
    const u32 snapshot_count = static_cast<u32>(group.size());
    const u32 read_count =
      memory_node_detail::peer_rdma_snapshot_work_request_count(
        snapshot_count, static_cast<u32>(pair_indices.size()));
    work_requests.resize(read_count);
    scatter_gather_entries.resize(read_count);
    const MemoryRegionToken& token = *peer_remote_tokens_[shard_id];
    for (u32 wr_index = 0; wr_index < read_count; ++wr_index) {
      const auto chain_item =
        memory_node_detail::peer_rdma_snapshot_chain_item(
          wr_index, snapshot_count,
          span<const u32>{pair_indices});
      const PeerReadSnapshotRequest& snapshot =
        group[chain_item.snapshot_index].request;
      const PeerReadRequest& request = chain_item.after_header
        ? *snapshot.after_header : snapshot.full_snapshot;
      ibv_sge& sge = scatter_gather_entries[wr_index];
      ibv_send_wr& wr = work_requests[wr_index];
      sge = {};
      wr = {};
      sge.addr = reinterpret_cast<u64>(request.destination) +
        request.local_offset;
      sge.length = static_cast<u32>(request.bytes);
      sge.lkey = thread.scratch_region->get_lkey();
      wr.opcode = IBV_WR_RDMA_READ;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.wr.rdma.remote_addr = token.address + request.remote_offset;
      wr.wr.rdma.rkey = token.rkey;
      if (!pair_indices.empty() && wr_index == snapshot_count) {
        // One fence orders every dynamic validation header after every body in
        // this QP chain without serializing the headers against each other.
        wr.send_flags = IBV_SEND_FENCE;
      }
      wr.next = wr_index + 1 < read_count
        ? &work_requests[wr_index + 1] : nullptr;
    }
    const u64 wr_id = next_peer_async_wr_id();
    work_requests.back().send_flags |= IBV_SEND_SIGNALED;
    work_requests.back().wr_id = wr_id;
    thread.track_post();
    register_peer_pending_send_locked(
      wr_id,
      PeerPendingSend{
        .target_shard = shard_id,
        .target_qp_idx = qp_idx,
        .thread_id = thread.id,
        .coroutine_id = thread.running_coroutine,
        .thread = &thread,
        .async = true,
        .rdma_read_credit = true,
        .rdma_read_count = read_count,
      });
    ibv_send_wr* bad_work_request = nullptr;
    {
      std::lock_guard<std::mutex> send_lock(
        *peer_qp_send_mutexes_[shard_id][qp_idx]);
      lib_assert(
        ibv_post_send(qp->get_ibv_qp(), work_requests.data(),
                      &bad_work_request) == 0,
        "cannot post mixed peer snapshot batch");
    }
  }
  return true;
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
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  const u32 qp_idx = peer_data_qp_index(owner_thread != nullptr ? owner_thread->id : 0);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  HugePage<byte_t>& scratch_buffer =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? owner_thread->scratch_buffer : peer_scratch_buffer_;
  LocalMemoryRegion& scratch_region =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? *owner_thread->scratch_region : *peer_scratch_region_;
  const size_t scratch_capacity =
    owner_thread != nullptr && owner_thread->has_peer_scratch()
      ? owner_thread->scratch_stride : scratch_buffer.buffer_size;
  lib_assert(scratch_offset <= scratch_capacity &&
               bytes <= scratch_capacity - scratch_offset,
             "peer scratch buffer exhausted");
  byte_t* scratch = owner_thread != nullptr && owner_thread->has_peer_scratch()
    ? owner_thread->coroutine_scratch(scratch_offset)
    : scratch_buffer.get_full_buffer() + scratch_offset;
  acquire_peer_rdma_read_credit(shard_id, qp_idx);
  const u64 wr_id = next_peer_sync_wr_id();
  {
    register_peer_pending_send_locked(
      wr_id,
      PeerPendingSend{shard_id, qp_idx, 0, 0, nullptr, false, true});
    std::lock_guard<std::mutex> send_lock(*peer_qp_send_mutexes_[shard_id][qp_idx]);
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
  if (dst != scratch) std::memcpy(dst, scratch, bytes);
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
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  const u32 qp_idx = peer_data_qp_index(owner_thread != nullptr ? owner_thread->id : 0);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  HugePage<byte_t>& scratch_buffer =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? owner_thread->scratch_buffer : peer_scratch_buffer_;
  LocalMemoryRegion& scratch_region =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? *owner_thread->scratch_region : *peer_scratch_region_;
  const size_t scratch_capacity =
    owner_thread != nullptr && owner_thread->has_peer_scratch()
      ? owner_thread->scratch_stride : scratch_buffer.buffer_size;
  lib_assert(scratch_offset <= scratch_capacity &&
               bytes <= scratch_capacity - scratch_offset,
             "peer scratch buffer exhausted");
  byte_t* scratch = owner_thread != nullptr && owner_thread->has_peer_scratch()
    ? owner_thread->coroutine_scratch(scratch_offset)
    : scratch_buffer.get_full_buffer() + scratch_offset;
  if (src != scratch) std::memcpy(scratch, src, bytes);
  const u64 wr_id = next_peer_sync_wr_id();
  {
    register_peer_pending_send_locked(
      wr_id,
      PeerPendingSend{shard_id, qp_idx, 0, 0, nullptr, false, false});
    std::lock_guard<std::mutex> send_lock(*peer_qp_send_mutexes_[shard_id][qp_idx]);
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
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  const u32 qp_idx = peer_data_qp_index(owner_thread != nullptr ? owner_thread->id : 0);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  HugePage<byte_t>& scratch_buffer =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? owner_thread->scratch_buffer : peer_scratch_buffer_;
  LocalMemoryRegion& scratch_region =
    owner_thread != nullptr && owner_thread->has_peer_scratch() ? *owner_thread->scratch_region : *peer_scratch_region_;
  const size_t scratch_capacity =
    owner_thread != nullptr && owner_thread->has_peer_scratch()
      ? owner_thread->scratch_stride : scratch_buffer.buffer_size;
  lib_assert(scratch_offset <= scratch_capacity &&
               sizeof(u64) <= scratch_capacity - scratch_offset,
             "peer scratch buffer exhausted");
  auto* scratch = reinterpret_cast<u64*>(
    owner_thread != nullptr && owner_thread->has_peer_scratch()
      ? owner_thread->coroutine_scratch(scratch_offset)
      : scratch_buffer.get_full_buffer() + scratch_offset);
  *scratch = 0;
  acquire_peer_rdma_read_credit(shard_id, qp_idx);
  const u64 wr_id = next_peer_sync_wr_id();
  {
    register_peer_pending_send_locked(
      wr_id,
      PeerPendingSend{shard_id, qp_idx, 0, 0, nullptr, false, true});
    std::lock_guard<std::mutex> send_lock(*peer_qp_send_mutexes_[shard_id][qp_idx]);
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

u64 MemoryNode::remote_fetch_add(u32 shard_id,
                                 u64 remote_offset,
                                 u64 increment,
                                 size_t scratch_offset) {
  lib_assert(peer_context_ != nullptr,
             "storage peer context is not initialized");
  lib_assert(shard_id < num_storage_nodes_ &&
               shard_id != storage_id_,
             "remote FAA requires a remote storage shard");
  lib_assert(peer_remote_tokens_[shard_id] != nullptr &&
               peer_remote_tokens_[shard_id]->address != 0 &&
               peer_remote_tokens_[shard_id]->rkey != 0,
             "peer token is invalid for remote FAA");
  lib_assert(remote_offset + sizeof(u64) <= mn_memory_bytes_,
             "peer FAA exceeds shard bounds");
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  const u32 qp_idx = peer_data_qp_index(
    owner_thread != nullptr ? owner_thread->id : 0);
  QP& qp = peer_data_qp(shard_id, qp_idx);
  HugePage<byte_t>& scratch_buffer =
    owner_thread != nullptr && owner_thread->has_peer_scratch()
      ? owner_thread->scratch_buffer : peer_scratch_buffer_;
  LocalMemoryRegion& scratch_region =
    owner_thread != nullptr && owner_thread->has_peer_scratch()
      ? *owner_thread->scratch_region : *peer_scratch_region_;
  const size_t scratch_capacity =
    owner_thread != nullptr && owner_thread->has_peer_scratch()
      ? owner_thread->scratch_stride : scratch_buffer.buffer_size;
  lib_assert(scratch_offset <= scratch_capacity &&
               sizeof(u64) <= scratch_capacity - scratch_offset,
             "peer scratch buffer exhausted");
  auto* scratch = reinterpret_cast<u64*>(
    owner_thread != nullptr && owner_thread->has_peer_scratch()
      ? owner_thread->coroutine_scratch(scratch_offset)
      : scratch_buffer.get_full_buffer() + scratch_offset);
  *scratch = 0;
  acquire_peer_rdma_read_credit(shard_id, qp_idx);
  const u64 wr_id = next_peer_sync_wr_id();
  {
    register_peer_pending_send_locked(
      wr_id, PeerPendingSend{
        shard_id, qp_idx, 0, 0, nullptr, false, true});
    std::lock_guard<std::mutex> send_lock(
      *peer_qp_send_mutexes_[shard_id][qp_idx]);
    qp->post_FAA(reinterpret_cast<u64>(scratch),
                 scratch_region.get_lkey(),
                 peer_remote_tokens_[shard_id].get(),
                 remote_offset, increment, true, wr_id);
  }
  wait_peer_sync_completion(wr_id);
  return *scratch;
}

std::pair<bool, u64> MemoryNode::try_lock_remote_header(RemotePtr rptr) {
  u64 header = 0;
  remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), &header, sizeof(header), 0);
  if (VamanaNode::header_incarnation(header) != rptr.incarnation() ||
      (header & VamanaNode::HEADER_NODE_LOCK) != 0) {
    return {false, header};
  }
  const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
  const u64 original = remote_compare_and_swap(rptr.memory_node(), rptr.byte_offset(), header, desired, align_up(sizeof(header)));
  return {original == header, original};
}
