#include "memory_node/storage_owner_maintenance/detail.hh"
#include "memory_node/storage_owner_maintenance/admission_policy.hh"
#include "memory_node/storage_owner_maintenance/cleanup_scheduler.hh"

using namespace memory_node_storage_owner_maintenance_detail;

u64 MemoryNode::arm_storage_owner_maintenance(
    StorageOwnerMaintenanceTask&& task, const Configuration& config) {
  vec<StorageOwnerMaintenanceTask> tasks;
  tasks.push_back(std::move(task));
  const u64 sequence = arm_storage_owner_maintenance_batch(tasks, config);
  if (sequence == 0 && !tasks.empty()) {
    task = std::move(tasks.front());
  }
  return sequence;
}

u64 MemoryNode::arm_storage_owner_maintenance_batch(
    vec<StorageOwnerMaintenanceTask>& tasks,
    const Configuration& config,
    bool* capacity_blocked) {
  if (capacity_blocked != nullptr) *capacity_blocked = false;
  // Peer receive/progress threads are intentionally started before the Stage2
  // executor so RC traffic can queue during node startup. A faster peer may
  // therefore deliver Stage1 in that narrow window; return an explicit retry
  // rather than entering try_begin() before its completion ring exists.
  if (!storage_owner_maintenance_enabled(config) ||
      storage_owner_maintenance_completion_ring_ == nullptr ||
      storage_owner_maintenance_admission_limit_ == 0 || tasks.empty() ||
      storage_insert_shutdown_.load(std::memory_order_acquire) ||
      tasks.size() > config.storage_owner_maintenance_queue_depth ||
      tasks.size() > storage_owner_maintenance_admission_limit_) {
    return 0;
  }
  for (const StorageOwnerMaintenanceTask& task : tasks) {
    if (task.target.is_null() ||
        task.kind != StorageOwnerMaintenanceKind::finalize_insert) {
      return 0;
    }
  }

  // Publish the already validated physical-home request into the bounded
  // accepted backlog before the authority can ACK it. This permit is wholly
  // independent of active Stage2 contexts/search lanes: a maintenance worker
  // claims those resources only after it pops a visible descriptor. Parking
  // here therefore means the complete configured backlog is full, not merely
  // that the execution window is busy. Retrying the whole RPC would repeat
  // deduplication, graph publication bookkeeping, and response construction
  // while making no durable progress.
  const size_t task_count = tasks.size();
  {
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_cv_.wait(lock, [&]() {
      return storage_owner_maintenance_shutdown_.load(
               std::memory_order_acquire) ||
        storage_insert_shutdown_.load(std::memory_order_acquire) ||
        maintenance_queue_batch_permit_available(
          storage_owner_stage2_tasks_.size() +
            storage_owner_cleanup_tasks_.size(),
          storage_owner_maintenance_reserved_slots_, task_count,
          config.storage_owner_maintenance_queue_depth);
    });
    if (storage_owner_maintenance_shutdown_.load(
          std::memory_order_acquire) ||
        storage_insert_shutdown_.load(std::memory_order_acquire)) {
      return 0;
    }
    storage_owner_maintenance_reserved_slots_ += task_count;
  }

  vec<u32> work_items(task_count, 1);
  // Do not enter SlidingCompletionRing::reserve_batch() here: its atomic wait
  // has no cancellation channel, while shutdown joins foreground workers
  // before stopping Stage2.  A full accepted window would otherwise make
  // graceful shutdown wait forever.  Keep the queue permit while retrying the
  // all-or-nothing reservation, but periodically recheck both shutdown flags.
  u64 first_sequence = 0;
  while (first_sequence == 0 &&
         !storage_insert_shutdown_.load(std::memory_order_acquire) &&
         !storage_owner_maintenance_shutdown_.load(
           std::memory_order_acquire)) {
    first_sequence = try_begin_storage_owner_maintenance_batch(
      span<const u32>{work_items});
    if (first_sequence != 0) break;
    if (capacity_blocked != nullptr) *capacity_blocked = true;
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_cv_.wait_for(
      lock, std::chrono::milliseconds(1));
  }
  if (first_sequence == 0) {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    lib_assert(storage_owner_maintenance_reserved_slots_ >= task_count,
               "cancelled Stage1 arm lost its maintenance queue permit");
    storage_owner_maintenance_reserved_slots_ -= task_count;
    storage_owner_maintenance_cv_.notify_all();
    return 0;
  }
  const auto queued_at = std::chrono::steady_clock::now();
  size_t backlog = 0;
  bool enqueued = false;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    lib_assert(storage_owner_maintenance_reserved_slots_ >= task_count,
               "Stage1 arm lost its reserved maintenance queue batch");
    storage_owner_maintenance_reserved_slots_ -= task_count;
    if (!storage_owner_maintenance_shutdown_.load(
          std::memory_order_acquire)) {
      for (size_t item = 0; item < task_count; ++item) {
        tasks[item].maintenance_sequence =
          first_sequence + static_cast<u64>(item);
        tasks[item].queued_at = queued_at;
        storage_owner_stage2_tasks_.push_back(std::move(tasks[item]));
      }
      storage_owner_stage2_packing_.observe_enqueue(queued_at, task_count);
      backlog = storage_owner_stage2_tasks_.size() +
        storage_owner_cleanup_tasks_.size();
      enqueued = true;
    }
  }
  if (!enqueued) {
    // Shutdown is the only failure after sequence allocation. Finalize every
    // unused ticket synchronously so none can pin the durable watermark.
    for (size_t item = 0; item < task_count; ++item) {
      complete_storage_owner_maintenance_sequence(
        first_sequence + static_cast<u64>(item));
    }
    notify_storage_owner_maintenance();
    return 0;
  }

  storage_owner_maintenance_enqueued_.fetch_add(
    task_count, std::memory_order_relaxed);
  storage_owner_maintenance_finalize_enqueued_.fetch_add(
    task_count, std::memory_order_relaxed);
  atomic_utils::update_max_relaxed(
    storage_owner_maintenance_max_backlog_, static_cast<u64>(backlog));
  notify_one_storage_owner_maintenance_executor();
  return first_sequence;
}

u64 MemoryNode::activate_storage_owner_cleanup(
    StorageOwnerMaintenanceTask&& task, const Configuration& config) {
  vec<StorageOwnerMaintenanceTask> tasks;
  tasks.push_back(std::move(task));
  const u64 sequence = activate_storage_owner_cleanup_batch(tasks, config);
  if (sequence == 0 && !tasks.empty()) {
    task = std::move(tasks.front());
  }
  return sequence;
}

u64 MemoryNode::activate_storage_owner_cleanup_batch(
    vec<StorageOwnerMaintenanceTask>& tasks,
    const Configuration& config) {
  if (!storage_owner_maintenance_enabled(config) || tasks.empty() ||
      storage_insert_shutdown_.load(std::memory_order_acquire) ||
      tasks.size() > config.storage_owner_maintenance_queue_depth ||
      tasks.size() > storage_owner_maintenance_admission_limit_) {
    return 0;
  }
  for (const StorageOwnerMaintenanceTask& task : tasks) {
    if (task.target.is_null() ||
        task.kind != StorageOwnerMaintenanceKind::cleanup_deleted_node) {
      return 0;
    }
  }

  const size_t task_count = tasks.size();
  {
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_cv_.wait(lock, [&]() {
      return storage_owner_maintenance_shutdown_.load(
               std::memory_order_acquire) ||
        storage_insert_shutdown_.load(std::memory_order_acquire) ||
        maintenance_queue_batch_permit_available(
          storage_owner_stage2_tasks_.size() +
            storage_owner_cleanup_tasks_.size(),
          storage_owner_maintenance_reserved_slots_,
          task_count,
          config.storage_owner_maintenance_queue_depth);
    });
    if (storage_owner_maintenance_shutdown_.load(
          std::memory_order_acquire) ||
        storage_insert_shutdown_.load(std::memory_order_acquire)) {
      return 0;
    }
    storage_owner_maintenance_reserved_slots_ += task_count;
  }

  vec<u32> work_items(task_count, 1);
  u64 first_sequence = 0;
  while (first_sequence == 0 &&
         !storage_insert_shutdown_.load(std::memory_order_acquire) &&
         !storage_owner_maintenance_shutdown_.load(
           std::memory_order_acquire)) {
    first_sequence = try_begin_storage_owner_maintenance_batch(
      span<const u32>{work_items});
    if (first_sequence != 0) break;
    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    storage_owner_maintenance_cv_.wait_for(
      lock, std::chrono::milliseconds(1));
  }
  if (first_sequence == 0) {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    lib_assert(storage_owner_maintenance_reserved_slots_ >= task_count,
               "cancelled cleanup activation lost its queue permit");
    storage_owner_maintenance_reserved_slots_ -= task_count;
    storage_owner_maintenance_cv_.notify_all();
    return 0;
  }
  const auto queued_at = std::chrono::steady_clock::now();
  size_t backlog = 0;
  bool enqueued = false;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    lib_assert(storage_owner_maintenance_reserved_slots_ >= task_count,
               "cleanup activation lost its reserved maintenance queue batch");
    storage_owner_maintenance_reserved_slots_ -= task_count;
    if (!storage_owner_maintenance_shutdown_.load(
          std::memory_order_acquire)) {
      // The task becomes runnable before the physical node is quiesced or
      // tombstoned. Its worker must first reparent every protected child and
      // may publish DELETED only after those reservations are ACKed.
      for (size_t item = 0; item < task_count; ++item) {
        tasks[item].maintenance_sequence =
          first_sequence + static_cast<u64>(item);
        tasks[item].queued_at = queued_at;
        cleanup_schedule_push(
          storage_owner_cleanup_tasks_, std::move(tasks[item]));
      }
      backlog = storage_owner_stage2_tasks_.size() +
        storage_owner_cleanup_tasks_.size();
      enqueued = true;
    }
  }
  if (!enqueued) {
    for (size_t item = 0; item < task_count; ++item) {
      complete_storage_owner_maintenance_sequence(
        first_sequence + static_cast<u64>(item));
    }
    notify_storage_owner_maintenance();
    return 0;
  }

  storage_owner_maintenance_enqueued_.fetch_add(
    task_count, std::memory_order_relaxed);
  storage_owner_maintenance_cleanup_enqueued_.fetch_add(
    task_count, std::memory_order_relaxed);
  atomic_utils::update_max_relaxed(
    storage_owner_maintenance_max_backlog_, static_cast<u64>(backlog));
  notify_one_storage_owner_maintenance_executor();
  return first_sequence;
}

u64 MemoryNode::begin_storage_owner_maintenance_sequence(u32 work_items) {
  return begin_storage_owner_maintenance_batch(
    span<const u32>{&work_items, 1});
}

u64 MemoryNode::begin_storage_owner_maintenance_batch(
    span<const u32> work_items) {
  lib_assert(storage_owner_maintenance_completion_ring_ != nullptr,
             "storage-owner completion ring is not initialized");
  lib_assert(storage_owner_maintenance_admission_limit_ != 0,
             "storage-owner accepted descriptor window is not initialized");
  // This claims bounded accepted-backlog capacity and a durable-watermark
  // sequence. It deliberately does not claim a context, search lane, RPC slot,
  // or other active Stage2 execution resource.
  const u64 sequence =
    storage_owner_maintenance_completion_ring_->reserve_batch(
      work_items, storage_owner_maintenance_admission_limit_);
  publish_storage_owner_maintenance_watermarks();
  return sequence;
}

u64 MemoryNode::try_begin_storage_owner_maintenance_batch(
    span<const u32> work_items) {
  lib_assert(storage_owner_maintenance_completion_ring_ != nullptr,
             "storage-owner completion ring is not initialized");
  lib_assert(storage_owner_maintenance_admission_limit_ != 0,
             "storage-owner accepted descriptor window is not initialized");
  const u64 sequence =
    storage_owner_maintenance_completion_ring_->try_reserve_batch(
      work_items, storage_owner_maintenance_admission_limit_);
  if (sequence != 0) publish_storage_owner_maintenance_watermarks();
  return sequence;
}

void MemoryNode::publish_storage_owner_maintenance_watermarks() {
  lib_assert(storage_owner_maintenance_completion_ring_ != nullptr,
             "storage-owner completion ring is not initialized");
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  std::atomic_ref<u64> next(control->next_maintenance_sequence);
  std::atomic_ref<u64> durable(control->durable_maintenance_sequence);
  u64 observed_next = next.load(std::memory_order_acquire);
  const u64 desired_next =
    storage_owner_maintenance_completion_ring_->next_sequence();
  while (observed_next < desired_next &&
         !next.compare_exchange_weak(
           observed_next, desired_next,
           std::memory_order_release, std::memory_order_acquire)) {
  }
  u64 observed_durable = durable.load(std::memory_order_acquire);
  const u64 desired_durable =
    storage_owner_maintenance_completion_ring_->finalized();
  while (observed_durable < desired_durable &&
         !durable.compare_exchange_weak(
           observed_durable, desired_durable,
           std::memory_order_release, std::memory_order_acquire)) {
  }
}

void MemoryNode::complete_storage_owner_maintenance_sequence(u64 sequence) {
  complete_storage_owner_maintenance_sequence(sequence, 1);
}

void MemoryNode::complete_storage_owner_maintenance_sequence(
    u64 sequence, u32 work_items) {
  if (sequence == 0 || work_items == 0) return;
  lib_assert(storage_owner_maintenance_completion_ring_ != nullptr,
             "storage-owner completion ring is not initialized");
  storage_owner_maintenance_completion_ring_->complete(
    sequence, work_items);
  publish_storage_owner_maintenance_watermarks();
  if (current_storage_owner_maintenance_worker_) {
    notify_storage_owner_maintenance_capacity();
  } else {
    notify_storage_owner_maintenance();
  }
  wake_peer_stage1_admission_waiters();
}

void MemoryNode::wake_peer_stage1_admission_waiters() {
  if (peer_stage1_admission_waiter_items_hint_.load(
        std::memory_order_acquire) == 0) {
    return;
  }
  if (storage_owner_maintenance_completion_ring_ == nullptr ||
      storage_owner_maintenance_admission_limit_ == 0 ||
      storage_worker_config_ == nullptr) {
    return;
  }
  const size_t incomplete =
    storage_owner_maintenance_completion_ring_->incomplete();
  if (incomplete >= storage_owner_maintenance_admission_limit_) return;
  size_t queue_available = 0;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    const size_t occupied = storage_owner_stage2_tasks_.size() +
      storage_owner_cleanup_tasks_.size() +
      storage_owner_maintenance_reserved_slots_;
    const size_t queue_depth =
      storage_worker_config_->storage_owner_maintenance_queue_depth;
    if (occupied >= queue_depth) return;
    queue_available = queue_depth - occupied;
  }
  size_t available = std::min(
    storage_owner_maintenance_admission_limit_ - incomplete,
    queue_available);
  u64 woken = 0;
  {
    std::lock_guard<std::mutex> lock(peer_stage1_tasks_mutex_);
    available = stage1_waiter_uncovered_wake_capacity(
      available, peer_stage1_admission_wake_coverage_);
    if (available == 0) return;
    // Reserve no hidden debt here: this is only a runnable notification. The
    // normal try-only arm remains the sole allocator. One visible credit can
    // wake an oversized FIFO head, whose *whole* demand becomes soft coverage;
    // its existing per-token fallback then makes truthful partial progress.
    // Requiring item_count visible credits before waking can starve forever in
    // a closed loop where other producers consume each returned credit before
    // a full wire batch accumulates.
    std::deque<PeerStage1Task> ready_waiters;
    while (available != 0 &&
           !peer_stage1_admission_waiters_.empty()) {
      const size_t item_count = std::max<size_t>(
        1, peer_stage1_admission_waiters_.front().header.item_count);
      const size_t coverage = stage1_waiter_head_wake_coverage(
        item_count, available);
      available = coverage >= available ? 0 : available - coverage;
      lib_assert(peer_stage1_admission_waiter_items_ >= item_count,
                 "Stage1 admission waiter item account underflow");
      peer_stage1_admission_waiter_items_ -= item_count;
      peer_stage1_admission_waiter_items_hint_.store(
        peer_stage1_admission_waiter_items_, std::memory_order_release);
      PeerStage1Task ready =
        std::move(peer_stage1_admission_waiters_.front());
      ready.admission_wake_coverage = static_cast<u32>(coverage);
      peer_stage1_admission_wake_coverage_ += coverage;
      ready_waiters.push_back(std::move(ready));
      peer_stage1_admission_waiters_.pop_front();
      ++woken;
    }
    // A returned completion credit is the event that made these old waiters
    // runnable. Put them ahead of fresh peer traffic so a hot receive stream
    // cannot repeatedly steal the credit and force wake -> repark CPU churn.
    // Moving the temporary queue from the back preserves waiter FIFO order.
    while (!ready_waiters.empty()) {
      peer_stage1_tasks_.push_front(std::move(ready_waiters.back()));
      ready_waiters.pop_back();
    }
  }
  if (woken != 0) {
    peer_stage1_admission_woken_.fetch_add(
      woken, std::memory_order_relaxed);
    peer_stage1_tasks_cv_.notify_all();
  }
}

bool MemoryNode::storage_owner_cleanup_ready(u64 sequence) const {
  if (sequence <= 1) {
    return cleanup_predecessors_durable(sequence, 0);
  }
  const auto* control = reinterpret_cast<const gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  auto& durable_storage = const_cast<u64&>(control->durable_maintenance_sequence);
  const u64 durable = std::atomic_ref<u64>(durable_storage).load(
    std::memory_order_acquire);
  return cleanup_predecessors_durable(sequence, durable);
}

void MemoryNode::mark_storage_owner_foreground_activity() {
  storage_owner_foreground_last_active_ns_.store(steady_now_ns(), std::memory_order_release);
}

bool MemoryNode::storage_owner_maintenance_foreground_busy(const Configuration&) {
  const bool foreground_active =
    storage_owner_insert_active_workers_.load(std::memory_order_acquire) != 0;

  if (storage_insert_tasks_ != nullptr) {
    const size_t foreground_queue_yield_threshold =
      std::max<size_t>(4, storage_owner_threads_.size() * kForegroundQueueYieldMultiplier);
    if (storage_insert_tasks_->approximate_size() >=
        foreground_queue_yield_threshold) {
      return true;
    }
  }

  {
    std::unique_lock<std::mutex> lock(peer_reverse_tasks_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return true;
    }
    if (queue_near_limit(peer_reverse_tasks_.size(), peer_reverse_task_queue_limit_)) {
      return true;
    }
  }

  if (peer_context_) {
    poll_peer_send_cq();
    const u32 pressure_num = foreground_active ? 1 : 3;
    const u32 pressure_den = foreground_active ? 2 : 4;
    if (counter_above_fraction(peer_async_rdma_outstanding_.load(std::memory_order_acquire),
                               peer_rdma_read_global_credit_limit(),
                               pressure_num,
                               pressure_den)) {
      return true;
    }
    const u32 per_peer_limit = peer_rdma_read_credit_limit();
    for (const auto& counter : peer_rdma_read_outstanding_) {
      if (counter_above_fraction(counter.load(std::memory_order_acquire),
                                 per_peer_limit,
                                 pressure_num,
                                 pressure_den)) {
        return true;
      }
    }
  }

  return false;
}

void MemoryNode::maybe_adjust_storage_owner_stage2_execution_budget() {
  constexpr u64 kBudgetSamplePeriodNs = 1'000'000'000ULL;
  // Advisory read-only fast path: almost every scheduler pass is inside the
  // current one-second window and must not contend on the transaction guard.
  // The owner rechecks the cadence under the guard below, so this cannot admit
  // an overlapping sample.
  const u64 fast_now = steady_now_ns();
  const u64 fast_previous =
    storage_owner_stage2_budget_last_sample_ns_.load(
      std::memory_order_acquire);
  if (fast_previous != 0 &&
      (fast_now <= fast_previous ||
       fast_now - fast_previous < kBudgetSamplePeriodNs)) {
    return;
  }

  Stage2ExecutionBudgetSampleGuard sample_guard(
    storage_owner_stage2_budget_sample_busy_);
  if (!sample_guard.owns_sample()) return;

  // Sampling is process-wide rather than per worker. A one-second cadence is
  // fast enough to reach C48 during a sustained buildup but slow enough that
  // the two-sample promotion and cooldown represent real hysteresis rather
  // than scheduler-loop noise.
  const u64 now = steady_now_ns();
  u64 previous_sample = storage_owner_stage2_budget_last_sample_ns_.load(
    std::memory_order_acquire);
  if (previous_sample != 0 &&
      (now <= previous_sample ||
       now - previous_sample < kBudgetSamplePeriodNs)) {
    return;
  }
  // A single strong CAS elects the sampler. If another worker published a
  // newer timestamp after our steady-clock read, return instead of subtracting
  // the newer value from the older one or moving the sample clock backwards.
  if (!storage_owner_stage2_budget_last_sample_ns_.compare_exchange_strong(
        previous_sample, now,
        std::memory_order_acq_rel, std::memory_order_acquire)) {
    return;
  }

  size_t visible_backlog = 0;
  {
    std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
    visible_backlog = storage_owner_stage2_tasks_.size();
  }
  const u64 lane_blocks = storage_owner_search_lane_lease_blocked_.load(
    std::memory_order_acquire);
  const u64 previous_lane_blocks =
    storage_owner_stage2_budget_last_lane_blocks_.exchange(
      lane_blocks, std::memory_order_acq_rel);
  const u64 lane_blocks_since_last = lane_blocks >= previous_lane_blocks
    ? lane_blocks - previous_lane_blocks : lane_blocks;
  const u64 processed = storage_owner_maintenance_processed_.load(
    std::memory_order_acquire);
  const u64 previous_processed =
    storage_owner_stage2_budget_last_processed_.exchange(
      processed, std::memory_order_acq_rel);
  const u64 processed_delta = processed >= previous_processed
    ? processed - previous_processed : 0;
  const u64 sample_elapsed_ns = previous_sample != 0 && now > previous_sample
    ? now - previous_sample : 0;
  const bool finalized_rate_available = sample_elapsed_ns != 0 &&
    processed >= previous_processed;
  const u64 finalized_rate_milli_per_sec =
    finalized_rate_available
      ? stage2_normalized_rate_milli_per_sec(
          processed_delta, sample_elapsed_ns)
      : 0;

  Stage2ExecutionBudgetPolicyState state{
    .contexts_per_worker =
      storage_owner_maintenance_contexts_per_worker_limit_.load(
        std::memory_order_acquire),
    .promotion_ceiling_contexts_per_worker =
      storage_owner_stage2_budget_promotion_ceiling_.load(
        std::memory_order_acquire),
    .promotion_streak =
      storage_owner_stage2_budget_promotion_streak_.load(
        std::memory_order_relaxed),
    .lane_pressure_streak =
      storage_owner_stage2_budget_lane_pressure_streak_.load(
        std::memory_order_relaxed),
    .low_backlog_streak =
      storage_owner_stage2_budget_low_backlog_streak_.load(
        std::memory_order_relaxed),
    .cooldown = storage_owner_stage2_budget_cooldown_.load(
      std::memory_order_relaxed),
    .stable_finalized_rate_milli_per_sec =
      storage_owner_stage2_budget_stable_rate_milli_.load(
        std::memory_order_relaxed),
    .trial_baseline_rate_milli_per_sec =
      storage_owner_stage2_budget_trial_baseline_rate_milli_.load(
        std::memory_order_relaxed),
    .trial_regression_streak =
      storage_owner_stage2_budget_trial_regression_streak_.load(
        std::memory_order_relaxed),
    .trial_success_streak =
      storage_owner_stage2_budget_trial_success_streak_.load(
        std::memory_order_relaxed),
    .rate_trial_pending =
      storage_owner_stage2_budget_rate_trial_pending_.load(
        std::memory_order_relaxed),
  };
  const Stage2ExecutionBudgetDecision decision =
    decide_stage2_execution_budget(
      state,
      Stage2ExecutionBudgetSample{
        .visible_backlog = visible_backlog,
        .accepted_window = storage_owner_maintenance_admission_limit_,
        .active_search_lanes =
          storage_owner_search_lane_leases_.load(std::memory_order_acquire),
        .search_lane_limit =
          storage_owner_search_lane_lease_limit_.load(
            std::memory_order_acquire),
        .search_lane_blocks_since_last = lane_blocks_since_last,
        .finalized_rate_milli_per_sec = finalized_rate_milli_per_sec,
        .finalized_rate_available = finalized_rate_available,
      },
      storage_owner_maintenance_contexts_per_worker_baseline_,
      storage_owner_maintenance_contexts_per_worker_max_);

  storage_owner_stage2_budget_promotion_streak_.store(
    decision.state.promotion_streak, std::memory_order_relaxed);
  storage_owner_stage2_budget_promotion_ceiling_.store(
    static_cast<u32>(
      decision.state.promotion_ceiling_contexts_per_worker),
    std::memory_order_release);
  storage_owner_stage2_budget_lane_pressure_streak_.store(
    decision.state.lane_pressure_streak, std::memory_order_relaxed);
  storage_owner_stage2_budget_low_backlog_streak_.store(
    decision.state.low_backlog_streak, std::memory_order_relaxed);
  storage_owner_stage2_budget_cooldown_.store(
    decision.state.cooldown, std::memory_order_relaxed);
  storage_owner_stage2_budget_stable_rate_milli_.store(
    decision.state.stable_finalized_rate_milli_per_sec,
    std::memory_order_relaxed);
  storage_owner_stage2_budget_trial_baseline_rate_milli_.store(
    decision.state.trial_baseline_rate_milli_per_sec,
    std::memory_order_relaxed);
  storage_owner_stage2_budget_trial_regression_streak_.store(
    decision.state.trial_regression_streak, std::memory_order_relaxed);
  storage_owner_stage2_budget_trial_success_streak_.store(
    decision.state.trial_success_streak, std::memory_order_relaxed);
  storage_owner_stage2_budget_rate_trial_pending_.store(
    decision.state.rate_trial_pending, std::memory_order_relaxed);
  if (decision.high_backlog) {
    storage_owner_stage2_budget_high_backlog_samples_.fetch_add(
      1, std::memory_order_relaxed);
  }
  if (decision.lane_headroom) {
    storage_owner_stage2_budget_lane_headroom_samples_.fetch_add(
      1, std::memory_order_relaxed);
  }
  if (decision.rate_trial_accepted) {
    storage_owner_stage2_budget_rate_trials_accepted_.fetch_add(
      1, std::memory_order_relaxed);
  }

  if (decision.action == Stage2ExecutionBudgetAction::hold) return;

  const size_t next_task_limit = stage2_active_task_limit(
    storage_owner_maintenance_worker_states_.size(),
    kStage2SemanticExecutionBatch,
    storage_owner_maintenance_admission_limit_,
    decision.state.contexts_per_worker);
  lib_assert(next_task_limit != 0 &&
               next_task_limit <=
                 storage_owner_maintenance_active_task_limit_max_,
             "adaptive Stage2 task target exceeded its hard maximum");
  const u32 next_context_limit = static_cast<u32>(
    decision.state.contexts_per_worker);
  const u32 next_tasks = static_cast<u32>(next_task_limit);
  if (decision.action == Stage2ExecutionBudgetAction::promote) {
    // Publish task capacity first: a worker seeing the larger context target
    // must also see enough exact B8 reservations for that target.
    storage_owner_maintenance_active_task_limit_.store(
      next_tasks, std::memory_order_release);
    storage_owner_maintenance_contexts_per_worker_limit_.store(
      next_context_limit, std::memory_order_release);
    storage_owner_stage2_budget_promotions_.fetch_add(
      1, std::memory_order_relaxed);
    notify_storage_owner_maintenance_capacity();
    return;
  }

  // Contract context admission first. Existing contexts are never canceled;
  // their reservations may temporarily exceed the new target and drain
  // naturally, while no replacement work is admitted above that target.
  storage_owner_maintenance_contexts_per_worker_limit_.store(
    next_context_limit, std::memory_order_release);
  storage_owner_maintenance_active_task_limit_.store(
    next_tasks, std::memory_order_release);
  storage_owner_stage2_budget_rollbacks_.fetch_add(
    1, std::memory_order_relaxed);
  if (decision.action ==
      Stage2ExecutionBudgetAction::rollback_lane_pressure) {
    storage_owner_stage2_budget_lane_rollbacks_.fetch_add(
      1, std::memory_order_relaxed);
  } else if (decision.action ==
             Stage2ExecutionBudgetAction::rollback_low_backlog) {
    storage_owner_stage2_budget_low_backlog_rollbacks_.fetch_add(
      1, std::memory_order_relaxed);
  } else {
    lib_assert(decision.action ==
                 Stage2ExecutionBudgetAction::rollback_rate_regression,
               "unknown Stage2 execution-budget rollback reason");
    storage_owner_stage2_budget_rate_rollbacks_.fetch_add(
      1, std::memory_order_relaxed);
  }
}

bool MemoryNode::try_acquire_storage_owner_maintenance_slot(
    const Configuration& config, bool foreground_pressure) {
  // This counter is the global number of live stage2 contexts, not the number
  // of physical workers. Each maintenance executor owns a fixed rpc-depth
  // context pool; peer send credits are independently bounded and try-only.
  // The adaptive target is identical during foreground work and drain. It
  // can move only one context per worker at a time between the verified C32
  // floor and C48 ceiling; becoming idle can never expose the configured RPC
  // depth as a context limit.
  const size_t contexts_per_worker_limit =
    storage_owner_maintenance_contexts_per_worker_limit_.load(
      std::memory_order_acquire);
  const size_t admitted_contexts = stage2_context_admission_limit(
    storage_owner_maintenance_worker_states_.size(),
    config.storage_owner_rpc_depth, foreground_pressure,
    contexts_per_worker_limit);
  const u32 max_contexts = static_cast<u32>(std::min<u64>(
    admitted_contexts, std::numeric_limits<u32>::max()));
  u32 active = storage_owner_maintenance_active_workers_.load(std::memory_order_acquire);
  while (active < max_contexts) {
    if (storage_owner_maintenance_active_workers_.compare_exchange_weak(
          active, active + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}
