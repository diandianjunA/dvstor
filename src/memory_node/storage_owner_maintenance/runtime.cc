#include "memory_node/storage_owner_maintenance/detail.hh"

using namespace memory_node_storage_owner_maintenance_detail;

bool MemoryNode::storage_owner_maintenance_enabled(const Configuration& config) {
  return config.storage_owner_maintenance_mode == "finalize" &&
         config.storage_owner_maintenance_workers > 0;
}

void MemoryNode::start_storage_owner_maintenance_runtime(const Configuration& config) {
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  lib_assert(control->magic == gpu_search::format::kStorageControlMagic &&
               control->version == gpu_search::format::kStorageControlVersion,
             "storage-owner maintenance control block is not initialized");
  std::atomic_ref<u64> next(control->next_maintenance_sequence);
  std::atomic_ref<u64> durable(control->durable_maintenance_sequence);
  const u64 initial_next = next.load(std::memory_order_acquire);
  const u64 initial_durable = durable.load(std::memory_order_acquire);
  lib_assert(initial_next == initial_durable + 1,
             "storage-owner cannot restart with unfinished in-memory maintenance");
  // Upserts can reserve stitch plus cleanup. Halving the descriptor bound
  // guarantees that every admitted sequence can publish all of its intents.
  const size_t completion_capacity = std::max<size_t>(
    std::max<size_t>(1, config.storage_owner_batch_max),
    config.storage_owner_maintenance_queue_depth / 2);
  storage_owner_maintenance_completion_ring_ =
    std::make_unique<bounded::SlidingCompletionRing>(
      completion_capacity, initial_next, initial_durable);
  storage_owner_maintenance_intent_capacity_ = completion_capacity;
  storage_owner_maintenance_intents_ =
    std::make_unique<StorageOwnerMaintenanceIntent[]>(completion_capacity);

  if (!storage_owner_maintenance_enabled(config)) {
    return;
  }

  storage_owner_maintenance_shutdown_.store(false, std::memory_order_release);
  storage_owner_maintenance_enqueued_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_enqueued_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_cleanup_enqueued_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_processed_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalized_live_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_failed_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_rpc_timeouts_.store(0, std::memory_order_relaxed);
  storage_owner_reverse_aggregate_batches_.store(0, std::memory_order_relaxed);
  storage_owner_reverse_aggregate_logical_requests_.store(
    0, std::memory_order_relaxed);
  storage_owner_reverse_aggregate_ops_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_stale_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_cleanup_processed_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_max_backlog_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_pressure_yields_.store(0, std::memory_order_relaxed);
  storage_owner_stitch_external_requests_.store(0, std::memory_order_relaxed);
  storage_owner_stitch_external_candidates_.store(0, std::memory_order_relaxed);
  storage_owner_stitch_batches_.store(0, std::memory_order_relaxed);
  storage_owner_stitch_batched_items_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_active_workers_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_latency_ns_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_max_latency_ns_.store(0, std::memory_order_relaxed);
  for (auto& bucket : storage_owner_maintenance_finalize_latency_buckets_) {
    bucket.store(0, std::memory_order_relaxed);
  }
  storage_owner_maintenance_started_ns_.store(steady_now_ns(), std::memory_order_release);
  storage_owner_maintenance_last_observation_ns_.store(0, std::memory_order_relaxed);
  const u32 worker_count = std::max<u32>(1, config.storage_owner_maintenance_workers);
  const size_t contexts_per_worker =
    std::max<size_t>(1, config.storage_owner_rpc_depth);
  const size_t remote_peer_count =
    num_storage_nodes_ > 1 ? static_cast<size_t>(num_storage_nodes_ - 1) : 1;
  lib_assert(static_cast<size_t>(worker_count) <=
               std::numeric_limits<size_t>::max() / contexts_per_worker,
             "stage2 reverse outbox worker/context capacity overflow");
  const size_t worker_context_capacity =
    static_cast<size_t>(worker_count) * contexts_per_worker;
  lib_assert(worker_context_capacity <=
               std::numeric_limits<size_t>::max() / remote_peer_count,
             "stage2 reverse outbox peer capacity overflow");
  const size_t reverse_outbox_capacity =
    worker_context_capacity * remote_peer_count;
  const u64 reverse_wire_max_u64 = std::max<u64>(
    1, static_cast<u64>(config.R) * config.storage_owner_batch_max);
  lib_assert(reverse_wire_max_u64 <= std::numeric_limits<u32>::max(),
             "stage2 reverse aggregate exceeds schema-15 item_count");
  const u32 reverse_wire_max = static_cast<u32>(reverse_wire_max_u64);
  storage_owner_reverse_outbox_ = std::make_unique<Stage2ReverseOutbox>(
    reverse_outbox_capacity, reverse_outbox_capacity, num_storage_nodes_,
    reverse_wire_max);
  storage_owner_reverse_completions_.clear();
  storage_owner_reverse_completions_.reserve(worker_count);
  const size_t completions_per_worker =
    contexts_per_worker * remote_peer_count;
  for (u32 worker_id = 0; worker_id < worker_count; ++worker_id) {
    storage_owner_reverse_completions_.push_back(
      std::make_unique<bounded::Queue<Stage2ReverseCompletion>>(
        completions_per_worker));
  }
  const size_t repair_capacity = std::max<size_t>(
    completion_capacity,
    2 * static_cast<size_t>(worker_count) *
      std::max<size_t>(1, config.storage_owner_rpc_depth) *
      std::max<size_t>(1, config.storage_owner_batch_max));
  storage_owner_repair_tasks_ =
    std::make_unique<bounded::Queue<StorageOwnerMaintenanceTask>>(
      repair_capacity);
  const size_t snapshot_stride = memory_node_detail::storage_owner_snapshot_stride();
  const size_t neighbor_stride = align_up(VamanaNode::neighbor_read_size());
  const size_t snapshot_batch = std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  const size_t coroutine_scratch_stride =
    align_up(snapshot_stride * snapshot_batch +
             std::max(VamanaNode::total_size(), neighbor_stride));
  const size_t scratch_bytes = coroutine_scratch_stride;

  storage_owner_maintenance_worker_states_.reserve(worker_count);
  for (u32 worker_id = 0; worker_id < worker_count; ++worker_id) {
    auto worker = std::make_unique<StorageOwnerThread>(worker_id, 1, config.max_send_queue_wr);
    if (peer_context_) {
      worker->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
    }
    storage_owner_maintenance_worker_states_.push_back(std::move(worker));
  }

  for (u32 worker_id = 0; worker_id < worker_count; ++worker_id) {
    storage_owner_maintenance_workers_.emplace_back(
      [this, worker_id]() { storage_owner_maintenance_worker_loop(worker_id); });
    if (!config.disable_thread_pinning) {
      pin_thread(storage_owner_maintenance_workers_.back(),
                 core_assignment_.get_available_core());
    }
  }

  print_status("storage-owner maintenance workers: " + std::to_string(worker_count) +
               " (configured=" + std::to_string(config.storage_owner_maintenance_workers) +
               ", work_conserving=true)");
  print_status("storage-owner stage2 reverse outbox descriptors: " +
               std::to_string(storage_owner_reverse_outbox_->capacity()) +
               " aggregates=" +
               std::to_string(storage_owner_reverse_outbox_->aggregate_capacity()) +
               " wire_max_ops=" + std::to_string(reverse_wire_max) +
               " (shared per-peer work-conserving aggregation)");
  print_status("storage-owner maintenance tuning: mode=" + config.storage_owner_maintenance_mode +
               " local_stitch=" + (config.storage_owner_update_mode == "local_stitch" ? "true" : "false") +
               " compaction_batch_target=" + std::to_string(config.storage_owner_batch_max) +
               " backlog_limit=" +
               std::to_string(config.storage_owner_maintenance_queue_depth));
}

void MemoryNode::stop_storage_owner_maintenance_runtime() {
  // Foreground insert workers are joined before this call, so no new intent
  // can appear.  Keep peer progress alive and drain every queued/active stage2
  // context before asking executors to exit. The wait is bounded because peer
  // runtimes have no cross-shard shutdown barrier in schema-15: one shard may
  // already be offline, in which case infinite same-ID retry must not deadlock
  // process shutdown. A non-drained summary remains an acceptance failure.
  if (!storage_owner_maintenance_workers_.empty()) {
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    const u64 rpc_timeout_ms = storage_worker_config_ == nullptr
      ? 1000 : storage_worker_config_->storage_owner_rpc_timeout_ms;
    const auto drain_timeout = std::chrono::milliseconds(
      std::max<u64>(5000, std::min<u64>(60'000, rpc_timeout_ms * 3)));
    const bool drained = storage_owner_maintenance_cv_.wait_for(
      lock, drain_timeout, [&]() {
      return storage_owner_stitch_tasks_.empty() &&
             storage_owner_cleanup_tasks_.empty() &&
             (storage_owner_repair_tasks_ == nullptr ||
              storage_owner_repair_tasks_->empty()) &&
             storage_owner_maintenance_active_workers_.load(
               std::memory_order_acquire) == 0;
    });
    if (!drained) {
      std::cerr << "[storage-owner] maintenance shutdown drain timed out; "
                   "final summary will report unfinished stage2 work"
                << std::endl;
    }
  }
  storage_owner_maintenance_shutdown_.store(true, std::memory_order_release);
  storage_owner_maintenance_cv_.notify_all();

  for (auto& worker : storage_owner_maintenance_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }

  size_t stitch_remaining = 0;
  size_t cleanup_remaining = 0;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    stitch_remaining = storage_owner_stitch_tasks_.size();
    cleanup_remaining = storage_owner_cleanup_tasks_.size();
    storage_owner_stitch_tasks_.clear();
    storage_owner_cleanup_tasks_.clear();
  }
  if (storage_owner_repair_tasks_ != nullptr) {
    StorageOwnerMaintenanceTask abandoned;
    while (storage_owner_repair_tasks_->try_pop(abandoned)) {
      ++cleanup_remaining;
      abandoned = StorageOwnerMaintenanceTask{};
    }
  }
  storage_owner_maintenance_workers_.clear();
  storage_owner_maintenance_worker_states_.clear();
  storage_owner_repair_tasks_.reset();
  storage_owner_reverse_outbox_.reset();
  storage_owner_reverse_completions_.clear();

  log_storage_owner_maintenance_observation(stitch_remaining, cleanup_remaining, true);
}

void MemoryNode::log_storage_owner_maintenance_observation(size_t stitch_remaining,
                                                           size_t cleanup_remaining,
                                                           bool final) {
  if (storage_owner_maintenance_enqueued_.load(std::memory_order_relaxed) == 0) {
    return;
  }

  const u64 finalize_enqueued =
    storage_owner_maintenance_finalize_enqueued_.load(std::memory_order_relaxed);
  const u64 cleanup_enqueued =
    storage_owner_maintenance_cleanup_enqueued_.load(std::memory_order_relaxed);
  const u64 finalized_live =
    storage_owner_maintenance_finalized_live_.load(std::memory_order_relaxed);
  const u64 stale =
    storage_owner_maintenance_stale_.load(std::memory_order_relaxed);
  const u64 live_required = finalize_enqueued > stale ? finalize_enqueued - stale : 0;
  const u64 total_finalize_latency_ns =
    storage_owner_maintenance_finalize_latency_ns_.load(std::memory_order_relaxed);
  const u64 max_finalize_latency_ns =
    storage_owner_maintenance_finalize_max_latency_ns_.load(std::memory_order_relaxed);
  const u64 started_ns =
    storage_owner_maintenance_started_ns_.load(std::memory_order_acquire);
  const u64 elapsed_ns = started_ns == 0 ? 0 : steady_now_ns() - started_ns;
  const double elapsed_s = static_cast<double>(elapsed_ns) / 1e9;
  const double repair_rate = elapsed_s > 0.0
                               ? static_cast<double>(finalized_live) / elapsed_s
                               : 0.0;
  const double avg_finalize_ms = finalized_live == 0
                                   ? 0.0
                                   : static_cast<double>(total_finalize_latency_ns) /
                                       static_cast<double>(finalized_live) / 1e6;
  u64 p99_target = finalized_live == 0 ? 0 : (finalized_live * 99 + 99) / 100;
  u64 p99_accumulated = 0;
  size_t p99_bucket = 0;
  for (; p99_bucket < storage_owner_maintenance_finalize_latency_buckets_.size();
       ++p99_bucket) {
    p99_accumulated += storage_owner_maintenance_finalize_latency_buckets_[p99_bucket].load(
      std::memory_order_relaxed);
    if (p99_accumulated >= p99_target) {
      break;
    }
  }
  const bool p99_finalize_over_30s = finalized_live != 0 &&
    p99_bucket >= kFinalizeLatencyBucketUpperNs.size() - 1;
  const size_t p99_finite_bucket = std::min(
    p99_bucket, kFinalizeLatencyBucketUpperNs.size() - 2);
  const double p99_finalize_ms = finalized_live == 0
    ? 0.0
    : static_cast<double>(
        kFinalizeLatencyBucketUpperNs[p99_finite_bucket]) / 1e6;
  std::string stitch_delay_histogram;
  for (size_t bucket = 0;
       bucket < storage_owner_maintenance_finalize_latency_buckets_.size();
       ++bucket) {
    if (bucket != 0) stitch_delay_histogram.push_back(',');
    stitch_delay_histogram += std::to_string(
      storage_owner_maintenance_finalize_latency_buckets_[bucket].load(
        std::memory_order_relaxed));
  }
  const u64 stitch_batches = storage_owner_stitch_batches_.load(std::memory_order_relaxed);
  const u64 stitch_batched_items =
    storage_owner_stitch_batched_items_.load(std::memory_order_relaxed);
  const u64 reverse_aggregate_batches =
    storage_owner_reverse_aggregate_batches_.load(std::memory_order_relaxed);
  const u64 reverse_aggregate_logical_requests =
    storage_owner_reverse_aggregate_logical_requests_.load(
      std::memory_order_relaxed);
  const u64 reverse_aggregate_ops =
    storage_owner_reverse_aggregate_ops_.load(std::memory_order_relaxed);
  const u64 peer_stitch_enqueued = peer_stitch_search_enqueued_.load(std::memory_order_relaxed);
  const u64 peer_stitch_processed = peer_stitch_search_processed_.load(std::memory_order_relaxed);
  const u64 peer_stitch_items = peer_stitch_search_items_.load(std::memory_order_relaxed);
  const u64 peer_reverse_enqueued =
    peer_reverse_update_enqueued_.load(std::memory_order_relaxed);
  const u64 peer_reverse_processed =
    peer_reverse_update_processed_.load(std::memory_order_relaxed);
  const u64 peer_reverse_items_enqueued =
    peer_reverse_update_items_enqueued_.load(std::memory_order_relaxed);
  const u64 peer_reverse_items_processed =
    peer_reverse_update_items_processed_.load(std::memory_order_relaxed);
  const u64 peer_reverse_remaining = peer_reverse_enqueued > peer_reverse_processed
    ? peer_reverse_enqueued - peer_reverse_processed
    : 0;
  const double peer_stitch_rate = elapsed_s > 0.0
                                    ? static_cast<double>(peer_stitch_items) / elapsed_s
                                    : 0.0;
  const double peer_reverse_rate = elapsed_s > 0.0
    ? static_cast<double>(peer_reverse_items_processed) / elapsed_s
    : 0.0;
  const u64 active_contexts =
    storage_owner_maintenance_active_workers_.load(std::memory_order_acquire);
  const u64 repair_remaining = storage_owner_repair_tasks_ == nullptr
    ? 0
    : static_cast<u64>(storage_owner_repair_tasks_->approximate_size());
  const u64 maintenance_done =
    storage_owner_maintenance_processed_.load(std::memory_order_relaxed) +
    storage_owner_maintenance_cleanup_processed_.load(
      std::memory_order_relaxed);
  const u64 maintenance_enqueued =
    storage_owner_maintenance_enqueued_.load(std::memory_order_relaxed);
  const u64 counter_remaining = maintenance_enqueued > maintenance_done
    ? maintenance_enqueued - maintenance_done : 0;
  // Queue cardinality alone becomes zero as soon as work enters a context,
  // even if that context is still waiting on remote shards. Keep the log SLO
  // conservative so MAX_STAGE2_REMAINING=0 cannot pass on pending RPCs.
  const u64 remaining = std::max<u64>(
    counter_remaining,
    static_cast<u64>(stitch_remaining + cleanup_remaining) +
      active_contexts + repair_remaining);
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const u64 reclaim_pending =
    std::atomic_ref<u64>(control->reclaim_pending_nodes).load(std::memory_order_acquire);
  const u64 reclaim_reused =
    std::atomic_ref<u64>(control->reclaim_reused_nodes).load(std::memory_order_acquire);
  const u64 dynamic_high_watermark =
    std::atomic_ref<u64>(control->dynamic_high_watermark).load(std::memory_order_acquire);
  const u64 reclaim_ack = minimum_compute_reclaim_ack();

  print_status(str("storage-owner maintenance ") + (final ? "summary" : "observation") +
               ": enqueued=" +
               std::to_string(storage_owner_maintenance_enqueued_.load(std::memory_order_relaxed)) +
               " stitch_enqueued=" +
               std::to_string(finalize_enqueued) +
               " cleanup_enqueued=" +
               std::to_string(cleanup_enqueued) +
               " stitch_tasks_done=" +
               std::to_string(storage_owner_maintenance_processed_.load(std::memory_order_relaxed)) +
               " stitched_live=" +
               std::to_string(finalized_live) +
               " cleanup_processed=" +
               std::to_string(storage_owner_maintenance_cleanup_processed_.load(std::memory_order_relaxed)) +
               " stale=" +
               std::to_string(stale) +
               " stitch_completion_ratio=" +
               std::to_string(ratio_or_zero(finalized_live, finalize_enqueued)) +
               " live_stitch_completion_ratio=" +
               std::to_string(ratio_or_zero(finalized_live, live_required)) +
               " avg_stitch_delay_ms=" +
               std::to_string(avg_finalize_ms) +
               " p99_stitch_delay_upper_ms=" +
               std::to_string(p99_finalize_ms) +
               " p99_stitch_delay_over_30s=" +
               (p99_finalize_over_30s ? "true" : "false") +
               " stitch_delay_histogram=" + stitch_delay_histogram +
               " max_stitch_delay_ms=" +
               std::to_string(static_cast<double>(max_finalize_latency_ns) / 1e6) +
               " compaction_batch_target=" +
               std::to_string(storage_worker_config_ != nullptr
                                ? storage_worker_config_->storage_owner_batch_max
                                : 0) +
               " compaction_max_delay_ms=" +
               std::to_string(kStitchCompactionMaxDelayNs / 1000000ull) +
               " stitch_rate_per_sec=" +
               std::to_string(repair_rate) +
               " failed=" +
               std::to_string(storage_owner_maintenance_failed_.load(std::memory_order_relaxed)) +
               " rpc_timeouts=" +
               std::to_string(storage_owner_maintenance_rpc_timeouts_.load(
                 std::memory_order_relaxed)) +
               " reverse_aggregate_batches=" +
               std::to_string(reverse_aggregate_batches) +
               " reverse_aggregate_logical_requests=" +
               std::to_string(reverse_aggregate_logical_requests) +
               " reverse_aggregate_ops=" +
               std::to_string(reverse_aggregate_ops) +
               " avg_reverse_aggregate_logicals=" +
               std::to_string(ratio_or_zero(
                 reverse_aggregate_logical_requests,
                 reverse_aggregate_batches)) +
               " avg_reverse_aggregate_ops=" +
               std::to_string(ratio_or_zero(
                 reverse_aggregate_ops, reverse_aggregate_batches)) +
               " max_backlog=" +
               std::to_string(storage_owner_maintenance_max_backlog_.load(std::memory_order_relaxed)) +
               " pressure_yields=" +
               std::to_string(storage_owner_maintenance_pressure_yields_.load(std::memory_order_relaxed)) +
               " external_search_requests=" +
               std::to_string(storage_owner_stitch_external_requests_.load(std::memory_order_relaxed)) +
               " external_candidates=" +
               std::to_string(storage_owner_stitch_external_candidates_.load(std::memory_order_relaxed)) +
               " stitch_batches=" +
               std::to_string(stitch_batches) +
               " avg_stitch_batch_size=" +
               std::to_string(ratio_or_zero(stitch_batched_items, stitch_batches)) +
               " peer_stitch_enqueued=" +
               std::to_string(peer_stitch_enqueued) +
               " peer_stitch_processed=" +
               std::to_string(peer_stitch_processed) +
               " peer_stitch_items=" +
               std::to_string(peer_stitch_items) +
               " peer_stitch_rate_per_sec=" +
               std::to_string(peer_stitch_rate) +
               " peer_stitch_max_queue=" +
               std::to_string(peer_stitch_search_max_queue_.load(std::memory_order_relaxed)) +
               " peer_reverse_enqueued=" +
               std::to_string(peer_reverse_enqueued) +
               " peer_reverse_processed=" +
               std::to_string(peer_reverse_processed) +
               " peer_reverse_remaining=" +
               std::to_string(peer_reverse_remaining) +
               " peer_reverse_items_enqueued=" +
               std::to_string(peer_reverse_items_enqueued) +
               " peer_reverse_items_processed=" +
               std::to_string(peer_reverse_items_processed) +
               " peer_reverse_rate_per_sec=" +
               std::to_string(peer_reverse_rate) +
               " peer_reverse_failed=" +
               std::to_string(peer_reverse_update_failed_.load(std::memory_order_relaxed)) +
               " peer_reverse_max_queue=" +
               std::to_string(peer_reverse_update_max_queue_.load(std::memory_order_relaxed)) +
               " reclaim_ack=" +
               std::to_string(reclaim_ack) +
               " reclaim_pending=" +
               std::to_string(reclaim_pending) +
               " reclaim_reused=" +
               std::to_string(reclaim_reused) +
               " dynamic_high_watermark=" +
               std::to_string(dynamic_high_watermark) +
               " stitch_remaining=" +
               std::to_string(stitch_remaining) +
               " cleanup_remaining=" +
               std::to_string(cleanup_remaining) +
               " active_contexts=" +
               std::to_string(active_contexts) +
               " repair_remaining=" +
               std::to_string(repair_remaining) +
               " remaining=" + std::to_string(remaining));
}

void MemoryNode::maybe_log_storage_owner_maintenance_observation() {
  const u64 now = steady_now_ns();
  u64 last = storage_owner_maintenance_last_observation_ns_.load(std::memory_order_acquire);
  while (now - last >= kMaintenanceObservationPeriodNs) {
    if (storage_owner_maintenance_last_observation_ns_.compare_exchange_weak(
          last, now, std::memory_order_acq_rel, std::memory_order_acquire)) {
      size_t stitch_remaining = 0;
      size_t cleanup_remaining = 0;
      {
        std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
        stitch_remaining = storage_owner_stitch_tasks_.size();
        cleanup_remaining = storage_owner_cleanup_tasks_.size();
      }
      log_storage_owner_maintenance_observation(stitch_remaining, cleanup_remaining, false);
      return;
    }
  }
}
