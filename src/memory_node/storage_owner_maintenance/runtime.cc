#include "memory_node/storage_owner_maintenance/detail.hh"

using namespace memory_node_storage_owner_maintenance_detail;

bool MemoryNode::storage_owner_maintenance_enabled(const Configuration& config) {
  return config.storage_owner_maintenance_mode == "finalize" &&
         config.storage_owner_maintenance_workers > 0;
}

void MemoryNode::start_storage_owner_maintenance_runtime(const Configuration& config) {
  if (!storage_owner_maintenance_enabled(config)) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_sequence_mutex_);
    storage_owner_maintenance_sequence_remaining_.clear();
  }

  storage_owner_maintenance_shutdown_.store(false, std::memory_order_release);
  storage_owner_maintenance_enqueued_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_enqueued_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_cleanup_enqueued_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_processed_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalized_live_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_failed_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_stale_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_cleanup_processed_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_max_backlog_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_pressure_yields_.store(0, std::memory_order_relaxed);
  storage_owner_stitch_external_requests_.store(0, std::memory_order_relaxed);
  storage_owner_stitch_external_candidates_.store(0, std::memory_order_relaxed);
  storage_owner_stitch_batches_.store(0, std::memory_order_relaxed);
  storage_owner_stitch_batched_items_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_active_workers_.store(0, std::memory_order_relaxed);
  storage_owner_next_stitch_release_ns_.store(
    steady_now_ns() +
      (static_cast<u64>(storage_id_ % std::max<u32>(1, num_storage_nodes_)) *
       kStitchCompactionPaceSlotNs),
    std::memory_order_relaxed);
  storage_owner_maintenance_finalize_latency_ns_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_max_latency_ns_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_started_ns_.store(steady_now_ns(), std::memory_order_release);
  storage_owner_maintenance_last_observation_ns_.store(0, std::memory_order_relaxed);
  const u32 worker_count = std::max<u32>(1, config.storage_owner_maintenance_workers);
  const size_t snapshot_stride = memory_node_detail::storage_owner_snapshot_stride();
  const size_t neighbor_stride = align_up(VamanaNode::neighbor_read_size());
  const size_t snapshot_batch = std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  const size_t coroutine_scratch_stride =
    align_up(snapshot_stride * snapshot_batch +
             std::max(VamanaNode::total_size(), neighbor_stride));
  const size_t scratch_bytes =
    std::max<size_t>(64ull * 1024ull * 1024ull, coroutine_scratch_stride);

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
  }

  print_status("storage-owner maintenance workers: " + std::to_string(worker_count) +
               " (configured=" + std::to_string(config.storage_owner_maintenance_workers) +
               ", resource_gated_parallel_stitching=true)");
  print_status("storage-owner maintenance tuning: mode=" + config.storage_owner_maintenance_mode +
               " local_stitch=" + (config.storage_owner_update_mode == "local_stitch" ? "true" : "false") +
               " compaction_batch_target=" + std::to_string(config.storage_owner_batch_max) +
               " compaction_max_delay_ms=" +
               std::to_string(kStitchCompactionMaxDelayNs / 1000000ull) +
               " compaction_pace=adaptive backlog_limit=" +
               std::to_string(config.storage_owner_maintenance_queue_depth));
}

void MemoryNode::stop_storage_owner_maintenance_runtime() {
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
  storage_owner_maintenance_workers_.clear();
  storage_owner_maintenance_worker_states_.clear();

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
  const u64 stitch_batches = storage_owner_stitch_batches_.load(std::memory_order_relaxed);
  const u64 stitch_batched_items =
    storage_owner_stitch_batched_items_.load(std::memory_order_relaxed);
  const u64 peer_stitch_enqueued = peer_stitch_search_enqueued_.load(std::memory_order_relaxed);
  const u64 peer_stitch_processed = peer_stitch_search_processed_.load(std::memory_order_relaxed);
  const u64 peer_stitch_items = peer_stitch_search_items_.load(std::memory_order_relaxed);
  const double peer_stitch_rate = elapsed_s > 0.0
                                    ? static_cast<double>(peer_stitch_items) / elapsed_s
                                    : 0.0;
  const size_t remaining = stitch_remaining + cleanup_remaining;
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
               " max_stitch_delay_ms=" +
               std::to_string(static_cast<double>(max_finalize_latency_ns) / 1e6) +
               " compaction_batch_target=" +
               std::to_string(storage_worker_config_ != nullptr
                                ? storage_worker_config_->storage_owner_batch_max
                                : 0) +
               " compaction_max_delay_ms=" +
               std::to_string(kStitchCompactionMaxDelayNs / 1000000ull) +
               " compaction_pace_slot_ms=" +
               std::to_string(kStitchCompactionPaceSlotNs / 1000000ull) +
               " stitch_rate_per_sec=" +
               std::to_string(repair_rate) +
               " failed=" +
               std::to_string(storage_owner_maintenance_failed_.load(std::memory_order_relaxed)) +
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
