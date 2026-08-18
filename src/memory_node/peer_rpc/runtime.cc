#include "memory_node/peer_rpc/detail.hh"
#include "memory_node/peer_rpc/stage1_control_fanout_policy.hh"
#include "memory_node/storage_owner_cpu_plan.hh"
#include "memory_node/storage_owner_index/graph_pointer_validation.hh"

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
  const size_t cleanup_activate_request_bytes =
    service::storage_owner::cleanup_activate_request_bytes(
      config.storage_owner_batch_max);
  const size_t cleanup_activate_response_bytes =
    service::storage_owner::cleanup_activate_response_bytes(
      config.storage_owner_batch_max);
  const size_t authority_placement_request_bytes =
    service::storage_owner::authority_placement_request_bytes(
      config.storage_owner_batch_max);
  const size_t authority_placement_response_bytes =
    service::storage_owner::authority_placement_response_bytes(
      config.storage_owner_batch_max);
  const size_t stage1_request_bytes =
    service::storage_owner::stage1_execute_request_bytes(
      config.storage_owner_batch_max);
  const size_t stage1_response_bytes =
    service::storage_owner::stage1_execute_response_bytes(
      config.storage_owner_batch_max);
  const size_t stage1_arm_request_bytes =
    service::storage_owner::stage1_arm_request_bytes(
      config.storage_owner_batch_max);
  const size_t stage1_arm_response_bytes =
    service::storage_owner::stage1_arm_response_bytes(
      config.storage_owner_batch_max);
  const size_t stage2_expand_score_request_bytes =
    service::storage_owner::stage2_expand_score_request_bytes(
      config.storage_owner_batch_max);
  const size_t stage2_expand_score_response_bytes =
    service::storage_owner::stage2_expand_score_response_bytes(
      config.storage_owner_batch_max);
  const u32 stage2_score_many_items = std::max<u32>(
    1, config.storage_owner_search_snapshot_batch);
  const size_t stage2_score_many_request_bytes =
    service::storage_owner::stage2_score_many_request_bytes(
      stage2_score_many_items, stage2_score_many_items);
  const size_t stage2_score_many_response_bytes =
    service::storage_owner::stage2_score_many_response_bytes(
      stage2_score_many_items);
  peer_rpc_runtime_.message_bytes = align_up(
    std::max({reverse_update_bytes,
              service::storage_owner::reconcile_reverse_request_bytes(1),
              service::storage_owner::centroid_membership_request_bytes(1),
              cleanup_activate_request_bytes,
              cleanup_activate_response_bytes,
              authority_placement_request_bytes,
              authority_placement_response_bytes,
              service::storage_owner::reverse_update_response_bytes(),
              stage1_request_bytes,
              stage1_response_bytes,
              stage1_arm_request_bytes,
              stage1_arm_response_bytes,
              stage2_expand_score_request_bytes,
              stage2_expand_score_response_bytes,
              stage2_score_many_request_bytes,
              stage2_score_many_response_bytes}));
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
    peer_rpc_speculative_credit_owner_.assign(num_storage_nodes_, 0);
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
  const size_t response_producers =
    static_cast<size_t>(std::max<u32>(1, config.storage_owner_maintenance_workers)) +
    static_cast<size_t>(std::max<u32>(1, num_compute_threads_));
  lib_assert(response_producers <=
               std::numeric_limits<size_t>::max() /
                 std::max<size_t>(1, config.storage_owner_rpc_depth),
             "peer control response registry producer capacity overflow");
  const size_t producer_depth = response_producers *
    std::max<size_t>(1, config.storage_owner_rpc_depth);
  lib_assert(producer_depth <=
               std::numeric_limits<size_t>::max() / registry_peer_count / 4,
             "peer control response registry peer capacity overflow");
  const size_t response_capacity = std::max<size_t>(
    1024, producer_depth * registry_peer_count * 4);
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
  print_status("storage-owner peer RPC send credits: Stage1/graph are split "
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

  peer_reverse_shutdown_.store(false, std::memory_order_release);
  peer_reverse_workers_done_.store(false, std::memory_order_release);
  peer_reverse_response_done_.store(false, std::memory_order_release);
  peer_reverse_task_queue_limit_ =
    std::max<size_t>(1024, static_cast<size_t>(config.storage_owner_reverse_queue_depth));
  peer_stage1_task_queue_limit_ = peer_reverse_task_queue_limit_;
  peer_physical_control_task_queue_limit_ =
    peer_reverse_task_queue_limit_;
  {
    std::lock_guard<std::mutex> lock(peer_cleanup_control_tasks_mutex_);
    peer_cleanup_control_tasks_.clear();
    peer_cleanup_next_source_sequences_.assign(num_storage_nodes_, 0);
  }
  {
    std::lock_guard<std::mutex> lock(peer_placement_control_tasks_mutex_);
    peer_placement_control_tasks_.clear();
  }
  peer_reverse_responses_ =
    std::make_unique<bounded::Queue<PeerReverseUpdateResponse>>(
      peer_reverse_task_queue_limit_);
  // Healthy senders hold at most one independent request-lifetime credit per
  // peer. Size the isolated low-priority response queue to one slot per remote
  // sender (with the queue's two-cell implementation floor), rather than
  // duplicating the potentially large authoritative response capacity.
  peer_speculative_responses_ =
    std::make_unique<bounded::Queue<PeerReverseUpdateResponse>>(
      std::max<size_t>(2, num_storage_nodes_ > 0
        ? num_storage_nodes_ - 1 : 1));
  peer_reverse_update_enqueued_.store(0, std::memory_order_relaxed);
  peer_reverse_update_processed_.store(0, std::memory_order_relaxed);
  peer_reverse_update_items_enqueued_.store(0, std::memory_order_relaxed);
  peer_reverse_update_items_processed_.store(0, std::memory_order_relaxed);
  peer_reverse_update_failed_.store(0, std::memory_order_relaxed);
  peer_reverse_update_max_queue_.store(0, std::memory_order_relaxed);
  peer_stage1_enqueued_.store(0, std::memory_order_relaxed);
  peer_stage1_processed_.store(0, std::memory_order_relaxed);
  peer_stage1_items_.store(0, std::memory_order_relaxed);
  peer_stage1_max_queue_.store(0, std::memory_order_relaxed);
  peer_stage2_home_enqueued_.store(0, std::memory_order_relaxed);
  peer_stage2_home_processed_.store(0, std::memory_order_relaxed);
  peer_stage2_home_items_.store(0, std::memory_order_relaxed);
  peer_stage2_home_max_queue_.store(0, std::memory_order_relaxed);
  peer_stage2_home_response_queue_drops_.store(0,
                                               std::memory_order_relaxed);
  peer_stage2_home_response_send_wait_ns_.store(0,
                                                std::memory_order_relaxed);
  peer_stage2_home_queue_wait_ns_.store(0, std::memory_order_relaxed);
  peer_stage2_home_execution_ns_.store(0, std::memory_order_relaxed);
  peer_stage1_release_deferred_batches_.store(0, std::memory_order_relaxed);
  peer_stage1_release_deferred_items_.store(0, std::memory_order_relaxed);
  peer_stage1_duplicate_retry_responses_.store(0, std::memory_order_relaxed);
  peer_stage1_admission_retry_responses_.store(0, std::memory_order_relaxed);
  peer_stage1_retry_response_drops_.store(0, std::memory_order_relaxed);
  peer_stage1_admission_parked_.store(0, std::memory_order_relaxed);
  peer_stage1_admission_woken_.store(0, std::memory_order_relaxed);
  peer_stage1_admission_reparked_.store(0, std::memory_order_relaxed);
  peer_stage1_duplicate_coalesced_.store(0, std::memory_order_relaxed);
  peer_stage1_max_admission_waiters_.store(0, std::memory_order_relaxed);
  peer_stage1_admission_waiter_items_hint_.store(
    0, std::memory_order_relaxed);
  peer_stage1_active_workers_.store(0, std::memory_order_relaxed);
  peer_stage2_home_active_workers_.store(0, std::memory_order_relaxed);
  peer_stage2_home_speculative_active_workers_.store(
    0, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(peer_stage1_tasks_mutex_);
    peer_stage1_tasks_.clear();
    peer_stage2_home_tasks_.clear();
    peer_stage2_home_speculative_tasks_.clear();
    peer_stage2_home_speculative_source_active_.assign(
      num_storage_nodes_, false);
    peer_stage1_admission_waiters_.clear();
    peer_stage1_admission_waiter_items_ = 0;
    peer_stage1_admission_owned_items_ = 0;
    peer_stage1_admission_wake_coverage_ = 0;
    peer_stage1_next_source_sequences_.assign(num_storage_nodes_, 0);
  }
  peer_stage1_completion_states_.clear();
  peer_stage1_completion_states_.reserve(num_storage_nodes_);
  for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
    peer_stage1_completion_states_.push_back(
      std::make_unique<PeerOrderedCompletionState>());
  }
  peer_cleanup_completion_states_.clear();
  peer_cleanup_completion_states_.reserve(num_storage_nodes_);
  for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
    peer_cleanup_completion_states_.push_back(
      std::make_unique<PeerOrderedCompletionState>());
  }

  const u32 rpc_parallelism = std::max<u32>(
    1, static_cast<u32>(num_clients_) *
       std::max<u32>(1, config.storage_owner_rpc_depth));
  const auto cpu_plan = memory_node_detail::derive_storage_owner_cpu_plan(
    core_assignment_.available_core_count(), num_compute_threads_,
    rpc_parallelism, config.storage_owner_maintenance_workers,
    num_storage_nodes_ > 0 ? num_storage_nodes_ - 1 : 0);
  const u32 reverse_worker_count = cpu_plan.peer_reverse_workers;
  // Reserve one Stage2-home lane to break dependency cycles. The remaining
  // lanes keep Stage1 priority and may steal only authoritative Stage2-home
  // work after a bounded delay. Speculation stays on the reserved lane, so a
  // later Stage1 publication always has at least one unpolluted shared lane.
  const u32 physical_home_worker_count = cpu_plan.peer_stage1_workers;
  const auto physical_home_split =
    memory_node_detail::split_physical_home_workers(
      physical_home_worker_count);
  const u32 stage2_home_worker_count = physical_home_split.stage2_home;
  const u32 stage1_rpc_worker_count = physical_home_split.stage1;
  // A single physical-home service lane cannot isolate an in-flight
  // speculative score from an authoritative dependency. Disable receiver and
  // local sender speculation in that environment; two or more lanes retain
  // one-per-source bounded lookahead with strict queue priority.
  peer_stage2_home_speculation_enabled_ =
    memory_node_peer_rpc_detail::independent_score_receiver_enabled(
      physical_home_worker_count,
      peer_rpc_runtime_.send_slots_per_peer);
  peer_stage2_home_speculative_task_limit_ =
    peer_stage2_home_speculation_enabled_
      ? std::max<size_t>(1, num_storage_nodes_ > 0
          ? num_storage_nodes_ - 1 : 1)
      : 0;
  peer_stage2_home_speculative_execution_limit_ =
    peer_stage2_home_speculation_enabled_ ? 1 : 0;
  peer_stage2_home_dedicated_ = false;
  peer_graph_response_buffer_limit_ = std::max<size_t>(
    1, static_cast<size_t>(physical_home_worker_count) * 2);
  {
    std::lock_guard<std::mutex> lock(peer_graph_response_buffers_mutex_);
    peer_graph_response_buffers_.clear();
  }
  const u32 cleanup_worker_count = cpu_plan.peer_cleanup_workers;
  const size_t stage1_total_worker_count =
    static_cast<size_t>(cpu_plan.foreground_coordinators) +
    static_cast<size_t>(physical_home_worker_count);
  lib_assert(stage1_total_worker_count <=
               std::numeric_limits<size_t>::max() /
                 std::max<size_t>(1, config.storage_owner_batch_max) / 4,
             "Stage1 artifact table worker capacity overflow");
  const size_t stage1_active_capacity = stage1_total_worker_count *
    std::max<size_t>(1, config.storage_owner_batch_max) * 4;
  lib_assert(static_cast<size_t>(config.storage_owner_maintenance_queue_depth) <=
               std::numeric_limits<size_t>::max() - stage1_active_capacity,
             "Stage1 artifact table queue capacity overflow");
  stage1_prepared_results_limit_ = std::max<size_t>(
    1024,
    static_cast<size_t>(config.storage_owner_maintenance_queue_depth) +
      stage1_active_capacity);
  stage1_prepared_results_limit_per_shard_ = std::max<size_t>(
    16, (stage1_prepared_results_limit_ + kStage1PreparedShardCount - 1) /
      kStage1PreparedShardCount);
  for (Stage1PreparedResultShard& shard : stage1_prepared_results_) {
    std::lock_guard<std::mutex> lock(shard.mutex);
    shard.records.clear();
    shard.records.reserve(stage1_prepared_results_limit_per_shard_);
  }
  for (Stage1InflightRequestShard& shard : stage1_inflight_requests_) {
    std::lock_guard<std::mutex> lock(shard.mutex);
    shard.counts.clear();
    shard.counts.reserve(stage1_prepared_results_limit_per_shard_);
  }
  const size_t cleanup_dedupe_total = std::max<size_t>(
    1024,
    static_cast<size_t>(config.storage_owner_maintenance_queue_depth) * 2);
  cleanup_activation_dedupe_limit_per_shard_ = std::max<size_t>(
    16, (cleanup_dedupe_total + kCleanupActivationShardCount - 1) /
      kCleanupActivationShardCount);
  for (CleanupActivationDedupeShard& shard :
       cleanup_activation_dedupe_) {
    std::lock_guard<std::mutex> lock(shard.mutex);
    shard.records.clear();
    shard.records.reserve(cleanup_activation_dedupe_limit_per_shard_);
  }
  dynamic_allocation_dedupe_limit_ = cleanup_dedupe_total;
  dynamic_allocation_receipts_.reset(dynamic_allocation_dedupe_limit_);
  const size_t snapshot_stride = align_up(VamanaNode::vector_bytes());
  const size_t neighbor_stride = align_up(VamanaNode::neighbor_read_size());
  const size_t batched_read_stride =
    memory_node_storage_owner_index_detail::batched_read_slot_stride(
      snapshot_stride);
  const size_t coroutine_scratch_stride =
    align_up(std::max<size_t>(VamanaNode::total_size(),
                              std::max(neighbor_stride,
                                       batched_read_stride *
                                         std::max<u32>(1, config.storage_owner_search_snapshot_batch))));
  const size_t scratch_bytes = coroutine_scratch_stride;
  peer_reverse_worker_states_.reserve(reverse_worker_count);
  for (u32 i = 0; i < reverse_worker_count; ++i) {
    auto worker = std::make_unique<StorageOwnerThread>(i, 1, config.max_send_queue_wr);
    worker->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
    peer_reverse_worker_states_.push_back(std::move(worker));
  }
  peer_stage1_worker_states_.reserve(stage1_rpc_worker_count);
  for (u32 i = 0; i < stage1_rpc_worker_count; ++i) {
    auto stage1_worker = std::make_unique<StorageOwnerThread>(
      reverse_worker_count + i, 1, config.max_send_queue_wr);
    peer_stage1_worker_states_.push_back(std::move(stage1_worker));
  }
  peer_stage2_home_worker_states_.reserve(stage2_home_worker_count);
  for (u32 i = 0; i < stage2_home_worker_count; ++i) {
    auto home_worker = std::make_unique<StorageOwnerThread>(
      reverse_worker_count + stage1_rpc_worker_count + i, 1,
      config.max_send_queue_wr);
    peer_stage2_home_worker_states_.push_back(std::move(home_worker));
  }

  peer_rpc_progress_thread_ = std::thread([this]() { peer_rpc_progress_loop(); });
  peer_reverse_response_thread_ = std::thread([this]() { peer_reverse_response_loop(); });
  peer_cleanup_control_workers_.reserve(cleanup_worker_count);
  for (u32 i = 0; i < cleanup_worker_count; ++i) {
    peer_cleanup_control_workers_.emplace_back(
      [this]() { peer_cleanup_control_worker_loop(); });
  }
  peer_placement_control_thread_ =
    std::thread([this]() { peer_placement_control_worker_loop(); });
  if (!config.disable_thread_pinning) {
    pin_thread(peer_rpc_progress_thread_, core_assignment_.get_available_core());
    pin_thread(peer_reverse_response_thread_, core_assignment_.get_available_core());
    pin_thread(peer_placement_control_thread_,
               core_assignment_.get_available_core());
  }
  for (auto& worker : peer_cleanup_control_workers_) {
    if (!config.disable_thread_pinning) {
      pin_thread(worker, core_assignment_.get_available_core());
    }
  }
  for (u32 i = 0; i < reverse_worker_count; ++i) {
    peer_reverse_workers_.emplace_back([this, i]() { peer_reverse_update_worker_loop(i); });
    if (!config.disable_thread_pinning) {
      pin_thread(peer_reverse_workers_.back(),
                 core_assignment_.get_available_core());
    }
  }
  for (u32 i = 0; i < stage1_rpc_worker_count; ++i) {
    peer_stage1_workers_.emplace_back([this, i]() {
      peer_stage1_worker_loop(i);
    });
    if (!config.disable_thread_pinning) {
      pin_thread(peer_stage1_workers_.back(),
                 core_assignment_.get_available_core());
    }
  }
  for (u32 i = 0; i < stage2_home_worker_count; ++i) {
    peer_stage2_home_workers_.emplace_back([this, i]() {
      peer_stage2_home_worker_loop(i);
    });
    if (!config.disable_thread_pinning) {
      pin_thread(peer_stage2_home_workers_.back(),
                 core_assignment_.get_available_core());
    }
  }
  print_status("storage-owner peer reverse-update workers: " +
               std::to_string(reverse_worker_count));
  print_status("storage-owner peer Stage1 workers: " +
               std::to_string(stage1_rpc_worker_count) +
               "; Stage2-home workers: " +
               std::to_string(stage2_home_worker_count) +
               " (one reserved; shared lanes borrow authoritative work only)");
  print_status("storage-owner independent exact-score receiver: " +
               std::string(peer_stage2_home_speculation_enabled_
                 ? "enabled" : "disabled") +
               " low_priority_limit=" +
               std::to_string(peer_stage2_home_speculative_task_limit_) +
               " (authoritative-first, one outstanding per sender, "
               "one executing globally)");
  print_status("storage-owner physical control workers: cleanup=" +
               std::to_string(cleanup_worker_count) + " placement=" +
               std::to_string(cpu_plan.peer_placement_workers) +
               " (separate blocking domains)");
  print_status("storage-owner Stage1 artifact capacity: " +
               std::to_string(stage1_prepared_results_limit_) +
               " (64-way, per shard=" +
               std::to_string(stage1_prepared_results_limit_per_shard_) + ")" +
               " cleanup replay per shard=" +
               std::to_string(cleanup_activation_dedupe_limit_per_shard_) +
               " active migration receipts=" +
               std::to_string(dynamic_allocation_dedupe_limit_));
  print_status("storage-owner peer reverse-update tuning: queue_depth=" +
               std::to_string(peer_reverse_task_queue_limit_) +
               " coalesce_max=" + std::to_string(config.storage_owner_reverse_coalesce_max));
}

void MemoryNode::stop_peer_reverse_update_runtime() {
  {
    std::lock_guard<std::mutex> lock(peer_stage1_tasks_mutex_);
    // Publish shutdown while holding the same mutex used by the worker exit
    // predicate and waiter parking. Otherwise every Stage1 worker can observe
    // shutdown+an empty runnable queue and exit in the gap before parked
    // waiters are moved, stranding their dedup leases and semantic tokens.
    peer_reverse_shutdown_.store(true, std::memory_order_release);
    while (!peer_stage1_admission_waiters_.empty()) {
      const size_t item_count = std::max<size_t>(
        1, peer_stage1_admission_waiters_.front().header.item_count);
      lib_assert(peer_stage1_admission_waiter_items_ >= item_count,
                 "Stage1 shutdown waiter item account underflow");
      peer_stage1_admission_waiter_items_ -= item_count;
      peer_stage1_tasks_.push_back(
        std::move(peer_stage1_admission_waiters_.front()));
      peer_stage1_admission_waiters_.pop_front();
    }
    lib_assert(peer_stage1_admission_waiter_items_ == 0,
               "Stage1 shutdown retained waiter item credit");
    peer_stage1_admission_waiter_items_hint_.store(
      0, std::memory_order_release);
  }
  peer_reverse_tasks_cv_.notify_all();
  peer_stage1_tasks_cv_.notify_all();
  for (const auto& completion : peer_stage1_completion_states_) {
    if (completion != nullptr) completion->changed.notify_all();
  }
  for (const auto& completion : peer_cleanup_completion_states_) {
    if (completion != nullptr) completion->changed.notify_all();
  }
  peer_cleanup_control_tasks_cv_.notify_all();
  peer_placement_control_tasks_cv_.notify_all();
  if (peer_reverse_responses_) peer_reverse_responses_->notify_all();
  if (peer_speculative_responses_) peer_speculative_responses_->notify_all();
  peer_response_wait_cv_.notify_all();
  peer_completion_cv_.notify_all();
  for (auto& worker : peer_reverse_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  for (auto& worker : peer_stage1_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  for (auto& worker : peer_stage2_home_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  {
    std::lock_guard<std::mutex> lock(peer_stage1_tasks_mutex_);
    lib_assert(peer_stage1_admission_wake_coverage_ == 0,
               "Stage1 shutdown retained runnable waiter coverage");
    lib_assert(peer_stage1_admission_owned_items_ == 0,
               "Stage1 shutdown retained semantic waiter ownership");
    lib_assert(peer_stage2_home_tasks_.empty() &&
                 peer_stage2_home_speculative_tasks_.empty(),
               "Stage2-home shutdown did not drain both priority queues");
    lib_assert(std::none_of(
                 peer_stage2_home_speculative_source_active_.begin(),
                 peer_stage2_home_speculative_source_active_.end(),
                 [](bool active) { return active; }),
               "Stage2-home shutdown retained per-source speculative debt");
    lib_assert(peer_stage2_home_speculative_active_workers_.load(
                 std::memory_order_acquire) == 0,
               "Stage2-home shutdown retained speculative execution credit");
  }
  for (auto& worker : peer_cleanup_control_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  if (peer_placement_control_thread_.joinable()) {
    peer_placement_control_thread_.join();
  }
  // Publish the terminal predicate under the same mutex used by the
  // authoritative/speculative response wait.  Without this handshake the
  // notify can land after the waiter tests an empty predicate but before it
  // actually sleeps, leaving shutdown blocked in join().
  {
    std::lock_guard<std::mutex> lock(peer_response_wait_mutex_);
    peer_reverse_workers_done_.store(true, std::memory_order_release);
  }
  if (peer_reverse_responses_) peer_reverse_responses_->notify_all();
  if (peer_speculative_responses_) peer_speculative_responses_->notify_all();
  peer_response_wait_cv_.notify_all();
  if (peer_reverse_response_thread_.joinable()) {
    peer_reverse_response_thread_.join();
  }
  lib_assert((peer_reverse_responses_ == nullptr ||
                peer_reverse_responses_->empty()) &&
               (peer_speculative_responses_ == nullptr ||
                peer_speculative_responses_->empty()),
             "peer response shutdown did not drain both priority queues");
  if (peer_rpc_progress_thread_.joinable()) {
    peer_rpc_progress_thread_.join();
  }
  if (peer_async_responses_ != nullptr) {
    const auto probes = peer_async_responses_->probe_telemetry();
    const double average = probes.lookups == 0
      ? 0.0
      : static_cast<double>(probes.probes) /
          static_cast<double>(probes.lookups);
    print_status("storage-owner peer response hash probes: average=" +
                 std::to_string(average) +
                 " max=" + std::to_string(probes.max_probe) +
                 " buckets=" +
                 std::to_string(peer_async_responses_->bucket_capacity()));
  }
  if (peer_request_deduplicator_ != nullptr) {
    const auto probes = peer_request_deduplicator_->probe_telemetry();
    const double average = probes.lookups == 0
      ? 0.0
      : static_cast<double>(probes.probes) /
          static_cast<double>(probes.lookups);
    print_status("storage-owner peer dedup hash probes: average=" +
                 std::to_string(average) +
                 " max=" + std::to_string(probes.max_probe) +
                 " buckets=" +
                 std::to_string(peer_request_deduplicator_->bucket_capacity()));
  }
  if (peer_async_responses_ != nullptr) {
    for (const auto& response : peer_async_responses_->drain_completed()) {
      repost_peer_rpc_receive(response.peer_id, response.receive_slot);
    }
  }
  peer_reverse_workers_.clear();
  peer_stage1_workers_.clear();
  peer_stage2_home_workers_.clear();
  peer_cleanup_control_workers_.clear();
  peer_reverse_worker_states_.clear();
  peer_stage1_worker_states_.clear();
  peer_stage2_home_worker_states_.clear();
  peer_stage2_home_dedicated_ = false;
  peer_stage2_home_speculation_enabled_ = false;
  {
    std::lock_guard<std::mutex> lock(peer_graph_response_buffers_mutex_);
    peer_graph_response_buffers_.clear();
  }
  peer_reverse_responses_.reset();
  peer_speculative_responses_.reset();
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
  if (slot_count >= 3 && slot_id == slot_count - 1) {
    return PeerRpcSendClass::control;
  }
  return slot_id % 2 == 0 ? PeerRpcSendClass::stage1
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
    // No surplus lane exists at depth one. Low-priority work must not borrow
    // the sole correctness/control SEND through the generic fallback.
    if (send_class == PeerRpcSendClass::speculative) return false;
    return try_lane(PeerRpcSendClass::control);
  }
  if (send_class == PeerRpcSendClass::speculative) {
    auto& free_slots = lanes[static_cast<size_t>(
      PeerRpcSendClass::graph_update)];
    // A speculative RPC may use otherwise-idle transport capacity, but it
    // must never consume the final graph/reverse slot needed by exact Stage2
    // progress. This also bounds process-wide speculative debt per peer by
    // the statically provisioned surplus instead of by cache size.
    if (free_slots.size() <= 1) return false;
    slot_id = free_slots.front();
    free_slots.pop_front();
    return true;
  }
  if (send_class == PeerRpcSendClass::control) {
    return try_lane(PeerRpcSendClass::control) ||
           try_lane(PeerRpcSendClass::stage1) ||
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
  // A SEND CQE returns process-wide transport capacity. Maintenance retries
  // sleep on their owner channels, while synchronous/control callers retain
  // peer_completion_cv_; publish the same capacity edge to both domains.
  notify_one_storage_owner_maintenance_executor();
  peer_completion_cv_.notify_all();
}

bool MemoryNode::try_reserve_peer_rpc_speculative_credit(
    u32 peer_id, u64 request_id) {
  if (request_id == 0 || peer_id >= num_storage_nodes_ ||
      peer_id == storage_id_) {
    return false;
  }
  std::lock_guard<std::mutex> lock(peer_rpc_send_slots_mutex_);
  if (peer_id >= peer_rpc_speculative_credit_owner_.size() ||
      peer_rpc_speculative_credit_owner_[peer_id] != 0) {
    return false;
  }
  peer_rpc_speculative_credit_owner_[peer_id] = request_id;
  return true;
}

void MemoryNode::release_peer_rpc_speculative_credit(
    u32 peer_id, u64 request_id) {
  std::lock_guard<std::mutex> lock(peer_rpc_send_slots_mutex_);
  lib_assert(request_id != 0 &&
               peer_id < peer_rpc_speculative_credit_owner_.size() &&
               peer_rpc_speculative_credit_owner_[peer_id] == request_id,
             "invalid peer speculative RPC credit release");
  peer_rpc_speculative_credit_owner_[peer_id] = 0;
}

void MemoryNode::fail_closed_peer_rpc_speculative_credit(
    u32 peer_id, u64 request_id) {
  std::lock_guard<std::mutex> lock(peer_rpc_send_slots_mutex_);
  lib_assert(request_id != 0 &&
               peer_id < peer_rpc_speculative_credit_owner_.size() &&
               peer_rpc_speculative_credit_owner_[peer_id] == request_id,
             "invalid peer speculative RPC fail-closed transition");
  // Canceling a local response-registry cell cannot remove an already posted
  // request from the remote Stage2-home queue. Keep this peer disabled for
  // the rest of the process unless a real response was observed; otherwise a
  // slow peer could accumulate an unbounded sequence of orphan lookahead
  // requests as each local deadline reused the same credit.
  peer_rpc_speculative_credit_owner_[peer_id] =
    std::numeric_limits<u64>::max();
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
  service::storage_owner::PeerRpcType response_type =
    service::storage_owner::PeerRpcType::reverse_update_response;
  if (request_type ==
      service::storage_owner::PeerRpcType::cleanup_deleted_request) {
    response_type =
      service::storage_owner::PeerRpcType::cleanup_deleted_response;
  } else if (request_type ==
             service::storage_owner::PeerRpcType::reconcile_reverse_request) {
    response_type =
      service::storage_owner::PeerRpcType::reconcile_reverse_response;
  } else if (request_type ==
             service::storage_owner::PeerRpcType::centroid_membership_request) {
    response_type =
      service::storage_owner::PeerRpcType::centroid_membership_response;
  }
  response.type = static_cast<u32>(response_type);
  response.source_shard = storage_id_;
  response.item_count = request.item_count;
  response.request_id = request.request_id;
  response.status = static_cast<u32>(success ? service::storage_owner::InsertStatus::ok
                                             : service::storage_owner::InsertStatus::failed);
  return response;
}
