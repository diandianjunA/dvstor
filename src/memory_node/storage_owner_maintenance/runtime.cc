#include "memory_node/storage_owner_maintenance/detail.hh"
#include "memory_node/storage_owner_maintenance/admission_policy.hh"
#include "memory_node/storage_owner_cpu_plan.hh"
#include "memory_node/storage_owner_index/graph_pointer_validation.hh"
#include "gpu_search/maintenance_telemetry.hh"

using namespace memory_node_storage_owner_maintenance_detail;

bool MemoryNode::storage_owner_maintenance_enabled(const Configuration& config) {
  return config.storage_owner_maintenance_workers > 0;
}

void MemoryNode::start_storage_owner_maintenance_runtime(const Configuration& config) {
  lib_assert(storage_owner_maintenance_enabled(config),
             "the two-stage update protocol requires a Stage2 executor");
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
  const u32 rpc_parallelism = std::max<u32>(
    1, static_cast<u32>(num_clients_) *
       std::max<u32>(1, config.storage_owner_rpc_depth));
  const auto cpu_plan = memory_node_detail::derive_storage_owner_cpu_plan(
    core_assignment_.available_core_count(), num_compute_threads_,
    rpc_parallelism, config.storage_owner_maintenance_workers,
    num_storage_nodes_ > 0 ? num_storage_nodes_ - 1 : 0);
  const u32 worker_count = cpu_plan.maintenance_workers;
  // Every reserved sequence is either already queued/runnable or completed by
  // its synchronous retirement path. Stage1 preparation owns no sequence, so
  // the full descriptor bound is safe. Keep the smaller admission window tied
  // to the workers that the CPU plan actually supplied, not merely the
  // requested worker count: otherwise a constrained node can acknowledge a
  // burst sized for executors that do not exist and create seconds of avoidable
  // Stage2 debt before bounded backpressure begins.
  const size_t completion_capacity = std::max<size_t>(
    std::max<size_t>(1, config.storage_owner_batch_max),
    config.storage_owner_maintenance_queue_depth);
  storage_owner_maintenance_completion_ring_ =
    std::make_unique<bounded::SlidingCompletionRing>(
      completion_capacity, initial_next, initial_durable);
  const size_t requested_admission_limit =
    stage2_sequence_admission_limit(
      cpu_plan.maintenance_admission_workers,
      config.storage_owner_rpc_depth,
      config.storage_owner_batch_max);
  storage_owner_maintenance_admission_limit_ = static_cast<size_t>(
    std::min(completion_capacity, requested_admission_limit));
  storage_owner_maintenance_intent_capacity_ = completion_capacity;
  storage_owner_maintenance_intents_ =
    std::make_unique<StorageOwnerMaintenanceIntent[]>(completion_capacity);

  storage_owner_maintenance_shutdown_.store(false, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_reserved_slots_ = 0;
  }
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
  storage_owner_stage2_batches_.store(0, std::memory_order_relaxed);
  storage_owner_stage2_batched_items_.store(0, std::memory_order_relaxed);
  for (auto& timing : storage_owner_stage2_phase_timing_) {
    timing.attempts.store(0, std::memory_order_relaxed);
    timing.task_attempts.store(0, std::memory_order_relaxed);
    timing.elapsed_ns.store(0, std::memory_order_relaxed);
  }
  storage_owner_maintenance_worker_idle_waits_.store(
    0, std::memory_order_relaxed);
  storage_owner_maintenance_worker_idle_ns_.store(
    0, std::memory_order_relaxed);
  storage_owner_maintenance_active_workers_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_latency_ns_.store(0, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_max_latency_ns_.store(0, std::memory_order_relaxed);
  for (auto& bucket : storage_owner_maintenance_finalize_latency_buckets_) {
    bucket.store(0, std::memory_order_relaxed);
  }
  storage_owner_maintenance_started_ns_.store(steady_now_ns(), std::memory_order_release);
  storage_owner_maintenance_last_observation_ns_.store(0, std::memory_order_relaxed);
  // Replace any telemetry tail left by an earlier process before accepting
  // the first mutation.  A benchmark with a zero-length write warmup still
  // needs a valid all-zero baseline; waiting for the first five-second
  // observation would otherwise force it back to host-local log files.
  gpu_search::maintenance_telemetry::publish(
    reinterpret_cast<byte_t*>(control),
    gpu_search::maintenance_telemetry::Snapshot{
      .shard_id = storage_id_,
      .published_steady_ns = steady_now_ns(),
      .admission_window = storage_owner_maintenance_admission_limit_,
    });
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
             "stage2 reverse aggregate exceeds bounded wire item_count");
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
  const size_t graph_stride = align_up(VamanaNode::hot_graph_entry_size());
  const size_t snapshot_batch = std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  const size_t batch_slot_stride =
    memory_node_storage_owner_index_detail::batched_read_slot_stride(
      snapshot_stride);
  const size_t coroutine_scratch_stride =
    align_up(batch_slot_stride * snapshot_batch +
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

  print_status("storage-owner peer scratch: snapshot_slot=" +
               std::to_string(snapshot_stride) + " graph_slot=" +
               std::to_string(graph_stride) + " batch=" +
               std::to_string(snapshot_batch) + " bytes_per_worker=" +
               std::to_string(coroutine_scratch_stride));

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
               ", admission_baseline_workers=" +
               std::to_string(cpu_plan.maintenance_admission_workers) +
               ", work_conserving=true)");
  print_status("storage-owner stage2 reverse outbox descriptors: " +
               std::to_string(storage_owner_reverse_outbox_->capacity()) +
               " aggregates=" +
               std::to_string(storage_owner_reverse_outbox_->aggregate_capacity()) +
               " wire_max_ops=" + std::to_string(reverse_wire_max) +
               " (shared per-peer work-conserving aggregation)");
  print_status("storage-owner maintenance tuning: protocol=centroid-home-two-stage"
               " compaction_batch_target=" + std::to_string(config.storage_owner_batch_max) +
               " backlog_limit=" +
               std::to_string(config.storage_owner_maintenance_queue_depth) +
               " admission_window=" +
               std::to_string(storage_owner_maintenance_admission_limit_) +
               " physical_completion_capacity=" +
               std::to_string(completion_capacity));
}

void MemoryNode::stop_storage_owner_maintenance_runtime() {
  // Foreground insert workers are joined before this call, so no new intent
  // can appear.  Keep peer progress alive and drain every queued/active stage2
  // context before asking executors to exit. The wait is bounded because peer
  // Peer runtimes have no cross-shard shutdown barrier: one shard may
  // already be offline, in which case infinite same-ID retry must not deadlock
  // process shutdown. A non-drained summary is reported as an incomplete drain.
  if (!storage_owner_maintenance_workers_.empty()) {
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    const u64 rpc_timeout_ms = storage_worker_config_ == nullptr
      ? 1000 : storage_worker_config_->storage_owner_rpc_timeout_ms;
    const auto drain_timeout = std::chrono::milliseconds(
      std::max<u64>(5000, std::min<u64>(60'000, rpc_timeout_ms * 3)));
    const bool drained = storage_owner_maintenance_cv_.wait_for(
      lock, drain_timeout, [&]() {
      return storage_owner_stage2_tasks_.empty() &&
             storage_owner_cleanup_tasks_.empty() &&
             storage_owner_maintenance_reserved_slots_ == 0 &&
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
  for (Stage1InflightRequestShard& inflight : stage1_inflight_requests_) {
    inflight.changed.notify_all();
  }

  for (auto& worker : storage_owner_maintenance_workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }

  size_t stage2_remaining = 0;
  size_t cleanup_remaining = 0;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    stage2_remaining = storage_owner_stage2_tasks_.size();
    cleanup_remaining = storage_owner_cleanup_tasks_.size();
    storage_owner_stage2_tasks_.clear();
    storage_owner_cleanup_tasks_.clear();
  }
  if (storage_owner_repair_tasks_ != nullptr) {
    StorageOwnerMaintenanceTask abandoned;
    while (storage_owner_repair_tasks_->try_pop(abandoned)) {
      ++cleanup_remaining;
      abandoned = StorageOwnerMaintenanceTask{};
    }
  }

  // Keep the joined worker vector intact until after the final observation:
  // its size is the denominator for cumulative worker-idle time.  Clearing it
  // first would report idle_ratio=0 and busy_ratio=1 for every shutdown
  // summary, regardless of the observed waits.
  log_storage_owner_maintenance_observation(
    stage2_remaining, cleanup_remaining, true);
  storage_owner_maintenance_workers_.clear();
  storage_owner_maintenance_worker_states_.clear();
  storage_owner_repair_tasks_.reset();
  storage_owner_reverse_outbox_.reset();
  storage_owner_reverse_completions_.clear();
}

void MemoryNode::log_storage_owner_maintenance_observation(size_t stage2_remaining,
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
  std::string stage2_delay_histogram;
  for (size_t bucket = 0;
       bucket < storage_owner_maintenance_finalize_latency_buckets_.size();
       ++bucket) {
    if (bucket != 0) stage2_delay_histogram.push_back(',');
    stage2_delay_histogram += std::to_string(
      storage_owner_maintenance_finalize_latency_buckets_[bucket].load(
        std::memory_order_relaxed));
  }
  const u64 stage2_batches = storage_owner_stage2_batches_.load(std::memory_order_relaxed);
  const u64 stage2_batched_items =
    storage_owner_stage2_batched_items_.load(std::memory_order_relaxed);
  std::array<u64, kStorageOwnerStage2TimingPhaseCount>
    stage2_phase_attempts{};
  std::array<u64, kStorageOwnerStage2TimingPhaseCount>
    stage2_phase_task_attempts{};
  std::array<u64, kStorageOwnerStage2TimingPhaseCount>
    stage2_phase_elapsed_ns{};
  for (size_t phase = 0;
       phase < kStorageOwnerStage2TimingPhaseCount; ++phase) {
    stage2_phase_attempts[phase] =
      storage_owner_stage2_phase_timing_[phase].attempts.load(
        std::memory_order_relaxed);
    stage2_phase_task_attempts[phase] =
      storage_owner_stage2_phase_timing_[phase].task_attempts.load(
        std::memory_order_relaxed);
    stage2_phase_elapsed_ns[phase] =
      storage_owner_stage2_phase_timing_[phase].elapsed_ns.load(
        std::memory_order_relaxed);
  }
  const auto phase_index = [](StorageOwnerStage2TimingPhase phase) {
    return static_cast<size_t>(phase);
  };
  const auto avg_phase_us_per_task = [&](
      StorageOwnerStage2TimingPhase phase) {
    const size_t index = phase_index(phase);
    return stage2_phase_task_attempts[index] == 0
      ? 0.0
      : static_cast<double>(stage2_phase_elapsed_ns[index]) /
          static_cast<double>(stage2_phase_task_attempts[index]) / 1e3;
  };
  const auto phase_elapsed_ms = [&](StorageOwnerStage2TimingPhase phase) {
    return static_cast<double>(stage2_phase_elapsed_ns[phase_index(phase)]) /
      1e6;
  };
  const u64 worker_idle_waits =
    storage_owner_maintenance_worker_idle_waits_.load(
      std::memory_order_relaxed);
  const u64 worker_idle_ns =
    storage_owner_maintenance_worker_idle_ns_.load(
      std::memory_order_relaxed);
  const double worker_observed_idle_ratio =
    elapsed_ns == 0 || storage_owner_maintenance_workers_.empty()
      ? 0.0
      : std::min(
          1.0,
          static_cast<double>(worker_idle_ns) /
            (static_cast<double>(elapsed_ns) *
             static_cast<double>(storage_owner_maintenance_workers_.size())));
  std::string stage2_phase_timing_log;
  const auto append_phase_timing = [&](
      const char* name, StorageOwnerStage2TimingPhase phase) {
    const size_t index = phase_index(phase);
    stage2_phase_timing_log += " stage2_phase_" + std::string(name) +
      "_attempts=" + std::to_string(stage2_phase_attempts[index]) +
      " stage2_phase_" + std::string(name) + "_task_attempts=" +
      std::to_string(stage2_phase_task_attempts[index]) +
      " stage2_phase_" + std::string(name) + "_elapsed_ms=" +
      std::to_string(phase_elapsed_ms(phase)) +
      " avg_stage2_phase_" + std::string(name) + "_us_per_task=" +
      std::to_string(avg_phase_us_per_task(phase));
  };
  append_phase_timing(
    "search", StorageOwnerStage2TimingPhase::continuation_search);
  append_phase_timing(
    "freeze_prune", StorageOwnerStage2TimingPhase::freeze_prune);
  append_phase_timing(
    "reverse_prepare", StorageOwnerStage2TimingPhase::reverse_prepare);
  append_phase_timing(
    "placement_authority",
    StorageOwnerStage2TimingPhase::placement_authority);
  append_phase_timing(
    "completion_handoff",
    StorageOwnerStage2TimingPhase::completion_handoff);
  append_phase_timing(
    "finalize", StorageOwnerStage2TimingPhase::finalize);
  stage2_phase_timing_log +=
    " maintenance_worker_idle_waits=" +
      std::to_string(worker_idle_waits) +
    " maintenance_worker_idle_ms=" +
      std::to_string(static_cast<double>(worker_idle_ns) / 1e6) +
    " maintenance_worker_observed_idle_ratio=" +
      std::to_string(worker_observed_idle_ratio) +
    " maintenance_worker_observed_busy_ratio=" +
      std::to_string(1.0 - worker_observed_idle_ratio);
  const u64 stage1_search_budget_exhausted =
    storage_owner_stage1_search_budget_exhausted_.load(
      std::memory_order_relaxed);
  const u64 stage2_search_budget_exhausted =
    storage_owner_stage2_search_budget_exhausted_.load(
      std::memory_order_relaxed);
  const u64 stage2_continuations =
    storage_owner_stage2_continuations_.load(std::memory_order_relaxed);
  const u64 stage2_remote_frontier_items =
    storage_owner_stage2_remote_frontier_items_.load(
      std::memory_order_relaxed);
  const u64 stage2_remote_expansions =
    storage_owner_stage2_remote_expansions_.load(std::memory_order_relaxed);
  const u64 stage2_scored_candidates =
    storage_owner_stage2_scored_candidates_.load(std::memory_order_relaxed);
  const u64 stage2_graph_read_waves =
    storage_owner_stage2_graph_read_waves_.load(std::memory_order_relaxed);
  const u64 stage2_graph_unique_reads =
    storage_owner_stage2_graph_unique_reads_.load(std::memory_order_relaxed);
  const u64 stage2_vector_read_waves =
    storage_owner_stage2_vector_read_waves_.load(std::memory_order_relaxed);
  const u64 stage2_vector_unique_reads =
    storage_owner_stage2_vector_unique_reads_.load(
      std::memory_order_relaxed);
  const u64 stage2_migrations =
    storage_owner_stage2_migrations_.load(std::memory_order_relaxed);
  const u64 stage2_final_edges =
    storage_owner_stage2_final_edges_.load(std::memory_order_relaxed);
  const u64 stage2_cross_edges_stage1_home =
    storage_owner_stage2_cross_edges_stage1_home_.load(
      std::memory_order_relaxed);
  const u64 stage2_cross_edges_final_home =
    storage_owner_stage2_cross_edges_final_home_.load(
      std::memory_order_relaxed);
  const u64 reverse_aggregate_batches =
    storage_owner_reverse_aggregate_batches_.load(std::memory_order_relaxed);
  const u64 reverse_aggregate_logical_requests =
    storage_owner_reverse_aggregate_logical_requests_.load(
      std::memory_order_relaxed);
  const u64 reverse_aggregate_ops =
    storage_owner_reverse_aggregate_ops_.load(std::memory_order_relaxed);
  const u64 peer_stage1_enqueued =
    peer_stage1_enqueued_.load(std::memory_order_relaxed);
  const u64 peer_stage1_processed =
    peer_stage1_processed_.load(std::memory_order_relaxed);
  const u64 peer_stage1_items =
    peer_stage1_items_.load(std::memory_order_relaxed);
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
  const double peer_stage1_rate = elapsed_s > 0.0
    ? static_cast<double>(peer_stage1_items) / elapsed_s
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
  const u64 completion_outstanding =
    storage_owner_maintenance_completion_ring_ == nullptr
      ? 0
      : static_cast<u64>(
          storage_owner_maintenance_completion_ring_->outstanding());
  // Queue cardinality alone becomes zero as soon as work enters a context,
  // even if that context is still waiting on remote shards. Report the larger
  // raw count so an observation does not hide pending RPC work.
  const u64 remaining = std::max<u64>(
    counter_remaining,
    static_cast<u64>(stage2_remaining + cleanup_remaining) +
      active_contexts + repair_remaining);
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const u64 reclaim_pending =
    std::atomic_ref<u64>(control->reclaim_pending_nodes).load(std::memory_order_acquire);
  const u64 reclaim_reused =
    std::atomic_ref<u64>(control->reclaim_reused_nodes).load(std::memory_order_acquire);
  const u64 dynamic_high_watermark =
    std::atomic_ref<u64>(control->dynamic_high_watermark).load(std::memory_order_acquire);

  gpu_search::maintenance_telemetry::Snapshot telemetry_snapshot{
    .shard_id = storage_id_,
    .published_steady_ns = steady_now_ns(),
    .stage2_enqueued = finalize_enqueued,
    .stage2_finalized_live = finalized_live,
    .stale = stale,
    .remaining = remaining,
    .peer_reverse_remaining = peer_reverse_remaining,
    .failed = storage_owner_maintenance_failed_.load(
      std::memory_order_relaxed),
    .peer_reverse_failed = peer_reverse_update_failed_.load(
      std::memory_order_relaxed),
    .admission_window = storage_owner_maintenance_admission_limit_,
    .completion_outstanding = completion_outstanding,
    .max_backlog = storage_owner_maintenance_max_backlog_.load(
      std::memory_order_relaxed),
    .stage1_search_budget_exhausted = stage1_search_budget_exhausted,
    .stage2_search_budget_exhausted = stage2_search_budget_exhausted,
    .stage2_continuations = stage2_continuations,
    .stage2_remote_frontier_items = stage2_remote_frontier_items,
    .stage2_remote_expansions = stage2_remote_expansions,
    .stage2_scored_candidates = stage2_scored_candidates,
    .stage2_migrations = stage2_migrations,
    .stage2_final_edges = stage2_final_edges,
    .stage2_cross_edges_stage1_home = stage2_cross_edges_stage1_home,
    .stage2_cross_edges_final_home = stage2_cross_edges_final_home,
    .pressure_yields = storage_owner_maintenance_pressure_yields_.load(
      std::memory_order_relaxed),
    .stage2_batches = stage2_batches,
    .stage2_batched_items = stage2_batched_items,
    .stage2_graph_read_waves = stage2_graph_read_waves,
    .stage2_graph_unique_reads = stage2_graph_unique_reads,
    .stage2_vector_read_waves = stage2_vector_read_waves,
    .stage2_vector_unique_reads = stage2_vector_unique_reads,
  };
  for (size_t bucket = 0;
       bucket < telemetry_snapshot.stage2_delay_histogram.size(); ++bucket) {
    telemetry_snapshot.stage2_delay_histogram[bucket] =
      storage_owner_maintenance_finalize_latency_buckets_[bucket].load(
        std::memory_order_relaxed);
  }
  gpu_search::maintenance_telemetry::publish(
    reinterpret_cast<byte_t*>(control), telemetry_snapshot);

  print_status(str("storage-owner maintenance ") + (final ? "summary" : "observation") +
               ": enqueued=" +
               std::to_string(storage_owner_maintenance_enqueued_.load(std::memory_order_relaxed)) +
               " stage2_enqueued=" +
               std::to_string(finalize_enqueued) +
               " cleanup_enqueued=" +
               std::to_string(cleanup_enqueued) +
               " stage2_tasks_done=" +
               std::to_string(storage_owner_maintenance_processed_.load(std::memory_order_relaxed)) +
               " stage2_finalized_live=" +
               std::to_string(finalized_live) +
               " cleanup_processed=" +
               std::to_string(storage_owner_maintenance_cleanup_processed_.load(std::memory_order_relaxed)) +
               " stale=" +
               std::to_string(stale) +
               " stage2_completion_ratio=" +
               std::to_string(ratio_or_zero(finalized_live, finalize_enqueued)) +
               " live_stage2_completion_ratio=" +
               std::to_string(ratio_or_zero(finalized_live, live_required)) +
               " avg_stage2_delay_ms=" +
               std::to_string(avg_finalize_ms) +
               " p99_stage2_delay_upper_ms=" +
               std::to_string(p99_finalize_ms) +
               " p99_stage2_delay_over_30s=" +
               (p99_finalize_over_30s ? "true" : "false") +
               " stage2_delay_histogram=" + stage2_delay_histogram +
               " max_stage2_delay_ms=" +
               std::to_string(static_cast<double>(max_finalize_latency_ns) / 1e6) +
               " compaction_batch_target=" +
               std::to_string(storage_worker_config_ != nullptr
                                ? storage_worker_config_->storage_owner_batch_max
                                : 0) +
               " compaction_max_delay_ms=" +
               std::to_string(kStage2CompactionMaxDelayNs / 1000000ull) +
               " stage2_rate_per_sec=" +
               std::to_string(repair_rate) +
               " stage1_search_budget_exhausted=" +
               std::to_string(stage1_search_budget_exhausted) +
               " stage2_search_budget_exhausted=" +
               std::to_string(stage2_search_budget_exhausted) +
               " stage2_continuations=" +
               std::to_string(stage2_continuations) +
               " stage2_remote_frontier_items=" +
               std::to_string(stage2_remote_frontier_items) +
               " avg_stage2_remote_frontier=" +
               std::to_string(ratio_or_zero(
                 stage2_remote_frontier_items, stage2_continuations)) +
               " stage2_remote_expansions=" +
               std::to_string(stage2_remote_expansions) +
               " avg_stage2_remote_expansions=" +
               std::to_string(ratio_or_zero(
                 stage2_remote_expansions, stage2_continuations)) +
               " stage2_scored_candidates=" +
               std::to_string(stage2_scored_candidates) +
               " avg_stage2_scored_candidates=" +
               std::to_string(ratio_or_zero(
                 stage2_scored_candidates, stage2_continuations)) +
               " stage2_graph_read_waves=" +
               std::to_string(stage2_graph_read_waves) +
               " avg_stage2_expansions_per_graph_wave=" +
               std::to_string(ratio_or_zero(
                 stage2_remote_expansions, stage2_graph_read_waves)) +
               " stage2_graph_unique_reads=" +
               std::to_string(stage2_graph_unique_reads) +
               " stage2_vector_read_waves=" +
               std::to_string(stage2_vector_read_waves) +
               " avg_stage2_scores_per_vector_wave=" +
               std::to_string(ratio_or_zero(
                 stage2_scored_candidates, stage2_vector_read_waves)) +
               " stage2_vector_unique_reads=" +
               std::to_string(stage2_vector_unique_reads) +
               " stage2_migrations=" +
               std::to_string(stage2_migrations) +
               " home_match_rate=" +
               std::to_string(finalized_live == 0 ? 0.0 :
                 1.0 - ratio_or_zero(stage2_migrations, finalized_live)) +
               " stage2_final_edges=" +
               std::to_string(stage2_final_edges) +
               " stage2_cross_edges_stage1_home=" +
               std::to_string(stage2_cross_edges_stage1_home) +
               " stage2_cross_edges_final_home=" +
               std::to_string(stage2_cross_edges_final_home) +
               " cross_edge_reduction_ratio=" +
               std::to_string(stage2_cross_edges_stage1_home == 0 ? 0.0 :
                 1.0 - ratio_or_zero(stage2_cross_edges_final_home,
                                     stage2_cross_edges_stage1_home)) +
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
               " admission_window=" +
               std::to_string(storage_owner_maintenance_admission_limit_) +
               " completion_outstanding=" +
               std::to_string(completion_outstanding) +
               " pressure_yields=" +
               std::to_string(storage_owner_maintenance_pressure_yields_.load(std::memory_order_relaxed)) +
               " stage2_batches=" +
               std::to_string(stage2_batches) +
               " avg_stage2_batch_size=" +
               std::to_string(ratio_or_zero(stage2_batched_items, stage2_batches)) +
               stage2_phase_timing_log +
               " peer_stage1_enqueued=" +
               std::to_string(peer_stage1_enqueued) +
               " peer_stage1_processed=" +
               std::to_string(peer_stage1_processed) +
               " peer_stage1_items=" +
               std::to_string(peer_stage1_items) +
               " avg_peer_stage1_items=" +
               std::to_string(ratio_or_zero(
                 peer_stage1_items, peer_stage1_processed)) +
               " peer_stage1_rate_per_sec=" +
               std::to_string(peer_stage1_rate) +
               " peer_stage1_max_queue=" +
               std::to_string(peer_stage1_max_queue_.load(
                 std::memory_order_relaxed)) +
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
               " reclaim_pending=" +
               std::to_string(reclaim_pending) +
               " reclaim_reused=" +
               std::to_string(reclaim_reused) +
               " dynamic_high_watermark=" +
               std::to_string(dynamic_high_watermark) +
               " stage2_remaining=" +
               std::to_string(stage2_remaining) +
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
      size_t stage2_remaining = 0;
      size_t cleanup_remaining = 0;
      {
        std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
        stage2_remaining = storage_owner_stage2_tasks_.size();
        cleanup_remaining = storage_owner_cleanup_tasks_.size();
      }
      log_storage_owner_maintenance_observation(stage2_remaining, cleanup_remaining, false);
      return;
    }
  }
}
