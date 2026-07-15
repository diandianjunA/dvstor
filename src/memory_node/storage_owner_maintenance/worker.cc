#include "memory_node/storage_owner_maintenance/detail.hh"
#include "memory_node/storage_owner_maintenance/admission_policy.hh"
#include "memory_node/storage_owner_maintenance/cleanup_policy.hh"
#include "memory_node/storage_owner_maintenance/stage2_tracker.hh"

#include <algorithm>
#include <limits>

using namespace memory_node_storage_owner_maintenance_detail;

void MemoryNode::storage_owner_maintenance_worker_loop(u32 worker_id) {
  lib_assert(worker_id < storage_owner_maintenance_worker_states_.size(),
             "storage-owner maintenance worker state missing");
  StorageOwnerThread& thread = *storage_owner_maintenance_worker_states_[worker_id];
  current_storage_owner_thread_ = &thread;
  const Configuration& config = *storage_worker_config_;
  lib_assert(num_storage_nodes_ > 0 && num_storage_nodes_ <= 64,
             "asynchronous stage2 supports at most 64 storage shards");

  using ReverseUpdateOp = service::storage_owner::ReverseUpdateOp;
  using PeerRpcType = service::storage_owner::PeerRpcType;

  struct Stage2Context {
    bool active{};
    Stage2ContextHandle handle{};
    StorageOwnerMaintenanceKind kind{
      StorageOwnerMaintenanceKind::stitch_insert};
    vec<StorageOwnerMaintenanceTask> tasks;
    vec<NodeSnapshot> targets;
    vec<NodeSnapshot> candidate_storage;
    vec<u32> candidate_counts;
    vec<vec<ReverseUpdateOp>> remote_ops_by_peer;
    vec<u64> search_request_ids;
    vec<u64> reverse_request_ids;
    vec<u32> search_item_counts;
    vec<u8> search_waiting_response;
    vec<byte_t> response_payload;
  };

  // A worker owns its contexts and both trackers, so the response path needs
  // no lock beyond the bounded peer-response registry. Global admission below
  // limits all workers together to the dedicated peer RPC depth.
  const size_t context_capacity =
    std::max<size_t>(1, config.storage_owner_rpc_depth);
  const size_t remote_peer_count = num_storage_nodes_ - 1;
  const size_t request_capacity =
    context_capacity * std::max<size_t>(1, remote_peer_count);
  Stage2StateTracker states(context_capacity, num_storage_nodes_);
  Stage2RequestTracker requests(request_capacity);
  vec<Stage2Context> contexts(context_capacity);
  const size_t candidate_capacity_per_item =
    static_cast<size_t>(config.R) +
    remote_peer_count * storage_owner_cross_shard_degree_;
  for (Stage2Context& context : contexts) {
    context.tasks.reserve(config.storage_owner_batch_max);
    context.targets.reserve(config.storage_owner_batch_max);
    context.candidate_storage.resize(
      static_cast<size_t>(config.storage_owner_batch_max) *
      candidate_capacity_per_item);
    for (NodeSnapshot& candidate : context.candidate_storage) {
      candidate.vector_data.resize(VamanaNode::vector_bytes());
    }
    context.candidate_counts.resize(config.storage_owner_batch_max);
    context.remote_ops_by_peer.resize(num_storage_nodes_);
    for (auto& ops : context.remote_ops_by_peer) {
      ops.reserve(static_cast<size_t>(config.R) *
                  config.storage_owner_batch_max);
    }
    context.search_request_ids.resize(num_storage_nodes_);
    context.reverse_request_ids.resize(num_storage_nodes_);
    context.search_item_counts.resize(num_storage_nodes_);
    context.search_waiting_response.resize(num_storage_nodes_);
    context.response_payload.reserve(peer_rpc_runtime_.message_bytes);
  }

  const u64 rpc_timeout_ns =
    static_cast<u64>(config.storage_owner_rpc_timeout_ms) * 1000ull * 1000ull;
  const u64 retry_backoff_ns = std::min<u64>(
    rpc_timeout_ns, 1000ull * 1000ull);
  lib_assert(storage_owner_reverse_outbox_ != nullptr &&
               worker_id < storage_owner_reverse_completions_.size() &&
               storage_owner_reverse_completions_[worker_id] != nullptr,
             "stage2 reverse aggregation runtime is not initialized");
  const u32 reverse_wire_max_ops =
    storage_owner_reverse_outbox_->wire_max_ops();
  vec<ReverseUpdateOp> reverse_wire_ops(reverse_wire_max_ops);
  vec<Stage2ReverseCompletion> reverse_completion_scratch(
    reverse_wire_max_ops);
  vec<byte_t> reverse_response_payload;
  reverse_response_payload.reserve(peer_rpc_runtime_.message_bytes);

  const auto reset_context = [&](Stage2Context& context) {
    context.active = false;
    context.tasks.clear();
    context.targets.clear();
    std::fill(context.candidate_counts.begin(),
              context.candidate_counts.end(), 0);
    for (auto& ops : context.remote_ops_by_peer) {
      ops.clear();
    }
    std::fill(context.search_request_ids.begin(),
              context.search_request_ids.end(), 0);
    std::fill(context.reverse_request_ids.begin(),
              context.reverse_request_ids.end(), 0);
    std::fill(context.search_item_counts.begin(),
              context.search_item_counts.end(), 0);
    std::fill(context.search_waiting_response.begin(),
              context.search_waiting_response.end(), 0);
    context.response_payload.clear();
  };

  const auto record_finalized_live = [this](
      std::chrono::steady_clock::time_point queued_at) {
    const u64 latency_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - queued_at).count());
    storage_owner_maintenance_finalize_latency_ns_.fetch_add(
      latency_ns, std::memory_order_relaxed);
    storage_owner_maintenance_finalize_latency_buckets_[
      finalize_latency_bucket(latency_ns)].fetch_add(
        1, std::memory_order_relaxed);
    atomic_utils::update_max_relaxed(
      storage_owner_maintenance_finalize_max_latency_ns_, latency_ns);
    storage_owner_maintenance_finalized_live_.fetch_add(
      1, std::memory_order_relaxed);
  };

  const auto commit_rebased_stitch_neighbors = [&, this](
      StorageOwnerMaintenanceTask& task) {
    if (!task.stitch_prepared || task.target.is_null() ||
        !local_shard(task.target.memory_node())) {
      return false;
    }
    lock_node(task.target);
    const bool target_deleted =
      (load_local_node_header_acquire(task.target) &
       VamanaNode::HEADER_DELETED) != 0;
    const bool current = !target_deleted &&
      storage_owner_task_current(task.id, task.generation, task.target);

    // Never rewrite a stale/tombstoned node here. A prepared stitch may
    // already have installed reverse edges, and the union of those targets
    // with the adjacency preserved at deletion can exceed R. Sequence
    // ownership is transferred to explicit cleanup below instead.
    if (!current) {
      unlock_node(task.target);
      return false;
    }

    // Remote search and reverse ACKs run without holding the target lock. A
    // newer insert may therefore have already installed a reverse edge into
    // this node. Rebase the pruned result on exactly the neighbors added since
    // our stage1 snapshot, then run the same robust-prune rule while locked.
    vec<RemotePtr> rebased_candidates = task.stitch_neighbors;
    const vec<RemotePtr> observed_neighbors =
      read_neighbor_list(task.target);
    for (const RemotePtr& neighbor : observed_neighbors) {
      const bool existed_at_stage1 = std::find(
        task.stitch_base_neighbors.begin(),
        task.stitch_base_neighbors.end(), neighbor) !=
          task.stitch_base_neighbors.end();
      const bool already_selected = std::find(
        rebased_candidates.begin(), rebased_candidates.end(), neighbor) !=
          rebased_candidates.end();
      if (!existed_at_stage1 && !already_selected) {
        rebased_candidates.push_back(neighbor);
      }
    }
    vec<NodeSnapshot> rebased_snapshots =
      read_node_snapshots_batched(rebased_candidates, config);
    hashset_t<RemotePtr> skip;
    skip.insert(task.target);
    const auto target_vector_address =
      vamana::StorageLayoutResolver::vector(task.target);
    lib_assert(target_vector_address.offset + target_vector_address.size <=
                 mn_memory_bytes_,
               "stage2 final target vector exceeds shard bounds");
    task.stitch_neighbors = robust_prune_snapshots_cpu(
      index_buffer_.get_full_buffer() + target_vector_address.offset,
      VamanaNode::vector_dtype(), rebased_snapshots, skip, config, config.R);
    write_neighbor_list(task.target, task.stitch_neighbors);
    unlock_node(task.target);
    return true;
  };

  const auto handoff_stitch_cleanup = [&, this](
      StorageOwnerMaintenanceTask& task,
      vec<RemotePtr>&& cleanup_neighbors) {
    if (cleanup_neighbors.empty()) {
      return false;
    }

    StorageOwnerMaintenanceTask cleanup;
    cleanup.kind = StorageOwnerMaintenanceKind::cleanup_deleted_node;
    cleanup.maintenance_sequence = task.maintenance_sequence;
    cleanup.target = task.target;
    cleanup.cleanup_repair_only = true;
    cleanup.cleanup_neighbors = std::move(cleanup_neighbors);
    cleanup.queued_at = std::chrono::steady_clock::now();

    lib_assert(storage_owner_repair_tasks_ != nullptr,
               "stage2 repair queue is not initialized");
    lib_assert(storage_owner_repair_tasks_->try_push(std::move(cleanup)),
               "bounded stage2 repair queue capacity invariant failed");
    size_t external_backlog = 0;
    {
      std::lock_guard<std::mutex> lock(storage_owner_maintenance_mutex_);
      external_backlog = storage_owner_stitch_tasks_.size() +
                         storage_owner_cleanup_tasks_.size();
    }
    const size_t backlog = external_backlog +
      storage_owner_repair_tasks_->approximate_size();
    storage_owner_maintenance_enqueued_.fetch_add(
      1, std::memory_order_relaxed);
    storage_owner_maintenance_cleanup_enqueued_.fetch_add(
      1, std::memory_order_relaxed);
    atomic_utils::update_max_relaxed(
      storage_owner_maintenance_max_backlog_, static_cast<u64>(backlog));
    storage_owner_maintenance_cv_.notify_all();
    return true;
  };

  const auto complete_stale_stitch = [&](StorageOwnerMaintenanceTask& task) {
    // A partially prepared retry may already have installed reverse edges.
    // Transfer sequence completion to repair so every backlink is removed.
    vec<RemotePtr> applied_neighbors;
    if (task.stitch_prepared) {
      applied_neighbors = std::move(task.stitch_neighbors);
    }
    const bool handed_off = handoff_stitch_cleanup(
      task, std::move(applied_neighbors));
    storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
    storage_owner_maintenance_processed_.fetch_add(1, std::memory_order_relaxed);
    if (!handed_off) {
      complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
    }
  };

  const auto complete_stale_cleanup = [&](const StorageOwnerMaintenanceTask& task) {
    storage_owner_maintenance_stale_.fetch_add(1, std::memory_order_relaxed);
    storage_owner_maintenance_cleanup_processed_.fetch_add(
      1, std::memory_order_relaxed);
    complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
  };

  const auto note_send_result = [&](u64 request_id, bool sent) {
    if (sent) {
      return;
    }
    const u64 now = steady_now_ns();
    // The logical request remains registered even when no dedicated send
    // credit was available. Retry the identical ID after a short backoff.
    (void)requests.mark_retry(request_id, now, now + retry_backoff_ns);
    storage_owner_maintenance_pressure_yields_.fetch_add(
      1, std::memory_order_relaxed);
  };

  const auto post_search = [&](Stage2Context& context, u32 shard) {
    u32 item_count = 0;
    const bool sent = post_stitch_search_request_async(
      shard, context.targets, context.search_request_ids[shard], item_count,
      config);
    lib_assert(item_count == context.tasks.size(),
               "stage2 stitch-search post changed the batch item count");
    context.search_item_counts[shard] = item_count;
    note_send_result(context.search_request_ids[shard], sent);
    context.search_waiting_response[shard] = sent ? 1 : 0;
    return sent;
  };

  const auto reverse_request_type = [&](const Stage2Context& context) {
    return context.kind == StorageOwnerMaintenanceKind::cleanup_deleted_node
      ? PeerRpcType::cleanup_deleted_request
      : PeerRpcType::reverse_update_request;
  };

  const auto enqueue_reverse_dispatch = [&](Stage2Context& context,
                                             u32 shard,
                                             u64 ready_at_ns) {
    lib_assert(storage_owner_reverse_outbox_ != nullptr,
               "stage2 reverse outbox is not initialized");
    lib_assert(shard < num_storage_nodes_ && shard != storage_id_,
               "stage2 reverse dispatch targets an invalid peer");
    lib_assert(context.remote_ops_by_peer[shard].size() <=
                 std::numeric_limits<u32>::max(),
               "stage2 reverse dispatch item count exceeds schema-15");
    const u32 item_count = static_cast<u32>(
      context.remote_ops_by_peer[shard].size());
    const Stage2ReverseDispatch dispatch{
      .logical_request_id = context.reverse_request_ids[shard],
      .context = context.handle,
      .worker_id = worker_id,
      .peer_index = shard,
      .request_type = reverse_request_type(context),
      .item_count = item_count,
      .ops = context.remote_ops_by_peer[shard].data(),
      .ready_at_ns = ready_at_ns,
    };
    const Stage2ReverseEnqueueResult result =
      storage_owner_reverse_outbox_->try_enqueue(dispatch);
    lib_assert(result == Stage2ReverseEnqueueResult::enqueued ||
                 result == Stage2ReverseEnqueueResult::duplicate,
               "bounded stage2 reverse outbox capacity/correlation invariant failed");
    if (result == Stage2ReverseEnqueueResult::enqueued) {
      storage_owner_maintenance_cv_.notify_all();
    }
    return result;
  };

  const auto register_search_requests = [&](Stage2Context& context,
                                            u64 expected_mask) {
    const u64 now = steady_now_ns();
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      const u64 bit = u64{1} << shard;
      if ((expected_mask & bit) == 0) {
        continue;
      }
      const u64 request_id = allocate_peer_request_id();
      context.search_request_ids[shard] = request_id;
      const auto result = requests.try_register(
        request_id, context.handle, Stage2RequestKind::remote_search, shard,
        now, now + rpc_timeout_ns, states);
      lib_assert(result == Stage2RequestRegisterResult::registered,
                 "stage2 search request tracker capacity invariant failed");
    }
    // Attempt every shard before returning to the rest of the pipeline.
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      if (context.search_request_ids[shard] != 0) {
        (void)post_search(context, shard);
      }
    }
  };

  const auto register_reverse_requests = [&](Stage2Context& context,
                                             u64 expected_mask) {
    const u64 now = steady_now_ns();
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      const u64 bit = u64{1} << shard;
      if ((expected_mask & bit) == 0) {
        continue;
      }
      const u64 request_id = allocate_peer_request_id();
      context.reverse_request_ids[shard] = request_id;
      // Reverse retry/deadline state belongs to the aggregate wire request;
      // this logical record exists only for ACK fan-out into the context mask.
      const auto result = requests.try_register(
        request_id, context.handle, Stage2RequestKind::reverse_update, shard,
        now, std::numeric_limits<u64>::max(), states);
      lib_assert(result == Stage2RequestRegisterResult::registered,
                 "stage2 reverse request tracker capacity invariant failed");
      const auto queued = enqueue_reverse_dispatch(context, shard, now);
      lib_assert(queued == Stage2ReverseEnqueueResult::enqueued,
                 "new stage2 reverse request was already present in outbox");
    }
  };

  const auto aggregate_response_type = [](PeerRpcType request_type) {
    return request_type == PeerRpcType::cleanup_deleted_request
      ? PeerRpcType::cleanup_deleted_response
      : PeerRpcType::reverse_update_response;
  };

  const auto poll_owned_reverse_aggregates = [&]() {
    bool progressed = false;
    size_t cursor = 0;
    for (;;) {
      const auto aggregate =
        storage_owner_reverse_outbox_->claim_awaiting_response(
          worker_id, cursor);
      if (!aggregate.has_value()) break;

      service::storage_owner::PeerRpcHeader header{};
      reverse_response_payload.clear();
      const PeerRpcType response_type =
        aggregate_response_type(aggregate->request_type);
      const TryPeerResponse response = try_consume_peer_rpc_response(
        aggregate->wire_request_id, aggregate->peer_index, response_type,
        aggregate->item_count, header, reverse_response_payload);
      const u64 now = steady_now_ns();
      if (response == TryPeerResponse::success) {
        const auto completion_count =
          storage_owner_reverse_outbox_->copy_completions(
            worker_id, aggregate->wire_request_id,
            std::span<Stage2ReverseCompletion>{
              reverse_completion_scratch.data(),
              reverse_completion_scratch.size()});
        lib_assert(completion_count.has_value() &&
                     *completion_count == aggregate->logical_count,
                   "stage2 reverse aggregate lost ACK fan-out metadata");
        // The copied completions are value snapshots. Release every logical
        // outbox entry before making an ACK visible to a destination worker:
        // that worker may consume its final ACK, reuse the context slot, and
        // enqueue replacement work immediately. Keeping the old entries until
        // after fan-out would create a transient false-full at exact capacity.
        lib_assert(storage_owner_reverse_outbox_->finish_success(
                     worker_id, aggregate->wire_request_id),
                   "stage2 reverse aggregate ACK release failed");
        for (size_t index = 0; index < *completion_count; ++index) {
          const Stage2ReverseCompletion& completion =
            reverse_completion_scratch[index];
          lib_assert(completion.worker_id <
                       storage_owner_reverse_completions_.size() &&
                       storage_owner_reverse_completions_[
                         completion.worker_id] != nullptr,
                     "stage2 reverse completion targets an invalid worker");
          lib_assert(storage_owner_reverse_completions_[
                       completion.worker_id]->try_push(completion),
                     "bounded stage2 reverse completion capacity invariant failed");
        }
        storage_owner_maintenance_cv_.notify_all();
        progressed = true;
        continue;
      }

      if (response == TryPeerResponse::failure ||
          response == TryPeerResponse::stale) {
        if (response == TryPeerResponse::stale) {
          cancel_peer_rpc_response(aggregate->wire_request_id);
          // stale also covers a retired tombstone that another request has
          // already reused. Rearm is therefore best-effort; the next exact-ID
          // send attempt atomically reinstalls or revives the registry entry.
          (void)rearm_peer_rpc_response(
            aggregate->wire_request_id,
            aggregate->peer_index,
            response_type,
            aggregate->item_count);
        }
        lib_assert(storage_owner_reverse_outbox_->release_poll(
                     worker_id, aggregate->wire_request_id, true,
                     now + retry_backoff_ns),
                   "stage2 reverse aggregate failure retry release failed");
        storage_owner_maintenance_failed_.fetch_add(
          1, std::memory_order_relaxed);
        progressed = true;
        continue;
      }

      const bool timed_out = now >= aggregate->deadline_ns;
      if (timed_out) {
        storage_owner_maintenance_rpc_timeouts_.fetch_add(
          1, std::memory_order_relaxed);
        storage_owner_maintenance_failed_.fetch_add(
          1, std::memory_order_relaxed);
      }
      lib_assert(storage_owner_reverse_outbox_->release_poll(
                   worker_id, aggregate->wire_request_id, timed_out, now),
                 "stage2 reverse aggregate poll release failed");
      progressed = timed_out || progressed;
    }
    return progressed;
  };

  const auto form_reverse_aggregates = [&]() {
    bool progressed = false;
    const u64 now = steady_now_ns();
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      if (shard == storage_id_) continue;
      for (;;) {
        if (!storage_owner_reverse_outbox_->can_form_aggregate(shard, now)) {
          break;
        }
        const u64 wire_request_id = allocate_peer_request_id();
        const auto aggregate = storage_owner_reverse_outbox_->form_aggregate(
          shard, worker_id, wire_request_id, now);
        if (!aggregate.has_value()) break;
        storage_owner_reverse_aggregate_batches_.fetch_add(
          1, std::memory_order_relaxed);
        storage_owner_reverse_aggregate_logical_requests_.fetch_add(
          aggregate->logical_count, std::memory_order_relaxed);
        storage_owner_reverse_aggregate_ops_.fetch_add(
          aggregate->item_count, std::memory_order_relaxed);
        progressed = true;
      }
    }
    return progressed;
  };

  const auto post_owned_reverse_aggregates = [&]() {
    bool progressed = false;
    size_t cursor = 0;
    for (;;) {
      const u64 now = steady_now_ns();
      const auto aggregate =
        storage_owner_reverse_outbox_->claim_ready_to_post(
          worker_id, now, cursor);
      if (!aggregate.has_value()) break;

      reverse_wire_ops.resize(aggregate->item_count);
      lib_assert(storage_owner_reverse_outbox_->copy_ops(
                   worker_id, aggregate->wire_request_id,
                   std::span<ReverseUpdateOp>{reverse_wire_ops.data(),
                                              reverse_wire_ops.size()}),
                 "stage2 reverse aggregate payload copy failed");
      u32 item_count = 0;
      const bool sent = post_peer_op_batch_async(
        aggregate->peer_index, reverse_wire_ops, aggregate->request_type,
        aggregate->wire_request_id, item_count, config);
      lib_assert(item_count == aggregate->item_count,
                 "stage2 reverse aggregate post changed item_count");
      const u64 posted_at = steady_now_ns();
      lib_assert(storage_owner_reverse_outbox_->finish_post(
                   worker_id, aggregate->wire_request_id, sent,
                   posted_at + (sent ? rpc_timeout_ns : retry_backoff_ns)),
                 "stage2 reverse aggregate post release failed");
      if (sent) {
        progressed = true;
      } else {
        storage_owner_maintenance_pressure_yields_.fetch_add(
          1, std::memory_order_relaxed);
      }
    }
    reverse_wire_ops.resize(reverse_wire_max_ops);
    return progressed;
  };

  const auto drive_reverse_outbox = [&]() {
    bool progressed = poll_owned_reverse_aggregates();
    progressed = form_reverse_aggregates() || progressed;
    progressed = post_owned_reverse_aggregates() || progressed;
    return progressed;
  };

  const auto drain_reverse_completions = [&]() {
    bool progressed = false;
    Stage2ReverseCompletion completion;
    auto& completion_queue = *storage_owner_reverse_completions_[worker_id];
    while (completion_queue.try_pop(completion)) {
      lib_assert(completion.worker_id == worker_id,
                 "stage2 reverse completion reached the wrong worker");
      const auto metadata = requests.find(completion.logical_request_id);
      lib_assert(metadata.has_value() &&
                   metadata->context == completion.context &&
                   metadata->kind == Stage2RequestKind::reverse_update &&
                   metadata->peer_index == completion.peer_index,
                 "stage2 reverse completion lost logical correlation");
      const Stage2EventResult result = requests.record_response(
        completion.logical_request_id, states);
      lib_assert(result == Stage2EventResult::accepted ||
                   result == Stage2EventResult::ready_to_finalize ||
                   result == Stage2EventResult::duplicate,
                 "stage2 rejected an aggregate reverse-update ACK");
      lib_assert(requests.erase(completion.logical_request_id),
                 "stage2 reverse completion request release failed");
      completion = {};
      progressed = true;
    }
    return progressed;
  };

  const auto prepare_local = [&](Stage2Context& context) {
    if (context.kind == StorageOwnerMaintenanceKind::cleanup_deleted_node) {
      const auto transition = states.begin_remote_search(context.handle, 0);
      lib_assert(transition == Stage2EventResult::phase_advanced,
                 "cleanup stage2 failed to enter prune_ready");
      return true;
    }

    context.targets.clear();
    std::fill(context.candidate_counts.begin(),
              context.candidate_counts.end(), 0);
    size_t ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (!local_shard(task.target.memory_node())) {
        storage_owner_maintenance_processed_.fetch_add(
          1, std::memory_order_relaxed);
        complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
        continue;
      }
      if (!storage_owner_task_current(task.id, task.generation, task.target)) {
        complete_stale_stitch(task);
        continue;
      }

      NodeSnapshot target_snapshot;
      const bool readable = read_node_snapshot(task.target, target_snapshot);
      lib_assert(readable, "local stage2 target snapshot was unreadable");
      if (target_snapshot.deleted) {
        complete_stale_stitch(task);
        continue;
      }

      if (ready != item) {
        context.tasks[ready] = std::move(task);
      }
      context.targets.push_back(std::move(target_snapshot));
      context.tasks[ready].stitch_base_neighbors =
        read_neighbor_list(context.tasks[ready].target);
      vec<NodeSnapshot> local_candidates = read_node_snapshots_batched(
        context.tasks[ready].stitch_base_neighbors, config);
      lib_assert(local_candidates.size() <= candidate_capacity_per_item,
                 "stage2 local candidate capacity invariant failed");
      for (const NodeSnapshot& source : local_candidates) {
        const size_t slot = ready * candidate_capacity_per_item +
                            context.candidate_counts[ready]++;
        NodeSnapshot& destination = context.candidate_storage[slot];
        destination.rptr = source.rptr;
        destination.header = source.header;
        destination.id = source.id;
        destination.generation = source.generation;
        destination.deleted = source.deleted;
        lib_assert(source.vector_data.size() >= VamanaNode::vector_bytes(),
                   "stage2 local candidate vector is incomplete");
        std::memcpy(destination.vector_data.data(),
                    source.vector_data.data(), VamanaNode::vector_bytes());
      }
      ++ready;
    }
    context.tasks.resize(ready);

    u64 expected_mask = 0;
    if (!context.tasks.empty() && peer_context_ != nullptr &&
        num_storage_nodes_ > 1) {
      for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
        if (shard != storage_id_) {
          expected_mask |= u64{1} << shard;
        }
      }
    }
    const auto transition =
      states.begin_remote_search(context.handle, expected_mask);
    lib_assert(transition == Stage2EventResult::phase_advanced,
               "stage2 failed to enter remote_search_pending");
    register_search_requests(context, expected_mask);
    return true;
  };

  const auto parse_search_response = [&](Stage2Context& context,
                                         const service::storage_owner::PeerRpcHeader& header,
                                         const vec<byte_t>& payload) {
    const u32 candidate_capacity = storage_owner_cross_shard_degree_;
    if (candidate_capacity == 0 || candidate_capacity > VamanaNode::R ||
        header.reserved != candidate_capacity ||
        header.item_count != context.tasks.size()) {
      return false;
    }
    const size_t expected_bytes =
      service::storage_owner::stitch_search_response_bytes(
        header.item_count, candidate_capacity);
    if (payload.size() < expected_bytes) {
      return false;
    }

    const u32* counts =
      service::storage_owner::stitch_search_response_counts(payload.data());
    const auto* candidates =
      service::storage_owner::stitch_search_response_candidates(
        payload.data(), header.item_count);
    const byte_t* vectors =
      service::storage_owner::stitch_search_response_candidate_vectors(
        payload.data(), header.item_count, candidate_capacity);
    for (u32 item = 0; item < header.item_count; ++item) {
      if (counts[item] > candidate_capacity) {
        return false;
      }
      if (static_cast<size_t>(context.candidate_counts[item]) +
            counts[item] > candidate_capacity_per_item) {
        return false;
      }
      for (u32 candidate_index = 0;
           candidate_index < counts[item]; ++candidate_index) {
        const size_t slot = static_cast<size_t>(item) * candidate_capacity +
                            candidate_index;
        const auto& candidate = candidates[slot];
        if (candidate.raw == 0) {
          continue;
        }
        const RemotePtr pointer{candidate.raw};
        if (pointer.memory_node() >= num_storage_nodes_ ||
            pointer.memory_node() != header.source_shard ||
            !vamana::StorageLayoutResolver::ptr_in_bounds(
              pointer, mn_memory_bytes_) ||
            !VamanaNode::hot_graph_entry_available(pointer)) {
          return false;
        }
        const bool dynamic =
          pointer.memory_node() <
            VamanaNode::HOT_GRAPH_DYNAMIC_BASE_OFFSETS.size() &&
          pointer.byte_offset() >=
            VamanaNode::HOT_GRAPH_DYNAMIC_BASE_OFFSETS[
              pointer.memory_node()];
        if (dynamic && candidate.generation == 0) {
          return false;
        }
      }
    }
    u64 candidate_count = 0;
    for (u32 item = 0; item < header.item_count; ++item) {
      for (u32 candidate_index = 0;
           candidate_index < counts[item]; ++candidate_index) {
        const size_t slot = static_cast<size_t>(item) * candidate_capacity +
                            candidate_index;
        const auto& candidate = candidates[slot];
        if (candidate.raw == 0) {
          continue;
        }
        const size_t destination_slot =
          static_cast<size_t>(item) * candidate_capacity_per_item +
          context.candidate_counts[item]++;
        NodeSnapshot& snapshot =
          context.candidate_storage[destination_slot];
        snapshot.rptr = RemotePtr{candidate.raw};
        snapshot.header = 0;
        snapshot.id = 0;
        snapshot.generation = candidate.generation;
        snapshot.deleted = false;
        std::memcpy(snapshot.vector_data.data(),
                    vectors + slot * VamanaNode::vector_bytes(),
                    VamanaNode::vector_bytes());
        ++candidate_count;
      }
    }
    storage_owner_stitch_external_requests_.fetch_add(
      1, std::memory_order_relaxed);
    storage_owner_stitch_external_candidates_.fetch_add(
      candidate_count, std::memory_order_relaxed);
    return true;
  };

  const auto retry_search_if_due = [&](Stage2Context& context, u32 shard) {
    const u64 request_id = context.search_request_ids[shard];
    const u64 now = steady_now_ns();
    if (request_id == 0 || !requests.retry_due(request_id, now)) {
      return false;
    }
    if (context.search_waiting_response[shard] != 0) {
      storage_owner_maintenance_rpc_timeouts_.fetch_add(
        1, std::memory_order_relaxed);
      storage_owner_maintenance_failed_.fetch_add(
        1, std::memory_order_relaxed);
    }
    u32 item_count = 0;
    const bool sent = post_stitch_search_request_async(
      shard, context.targets, request_id, item_count, config);
    lib_assert(item_count == context.search_item_counts[shard],
               "stage2 search retry changed item count");
    (void)requests.mark_retry(
      request_id, now, now + (sent ? rpc_timeout_ns : retry_backoff_ns));
    context.search_waiting_response[shard] = sent ? 1 : 0;
    if (!sent) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
    }
    return true;
  };

  const auto poll_search_responses = [&](Stage2Context& context) {
    bool progressed = false;
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      const u64 request_id = context.search_request_ids[shard];
      if (request_id == 0 || !requests.find(request_id).has_value()) {
        continue;
      }
      service::storage_owner::PeerRpcHeader header{};
      context.response_payload.clear();
      const TryPeerResponse response = try_consume_peer_rpc_response(
        request_id, shard, PeerRpcType::stitch_search_response,
        context.search_item_counts[shard], header, context.response_payload);
      if (response == TryPeerResponse::success &&
          parse_search_response(context, header, context.response_payload)) {
        const Stage2EventResult result =
          requests.record_response(request_id, states);
        lib_assert(result == Stage2EventResult::accepted ||
                     result == Stage2EventResult::phase_advanced ||
                     result == Stage2EventResult::duplicate,
                   "stage2 rejected a valid search response");
        (void)requests.erase(request_id);
        progressed = true;
        continue;
      }
      if (response == TryPeerResponse::failure ||
          response == TryPeerResponse::stale ||
          response == TryPeerResponse::success) {
        context.search_waiting_response[shard] = 0;
        if (response == TryPeerResponse::stale) {
          cancel_peer_rpc_response(request_id);
          (void)rearm_peer_rpc_response(
            request_id, shard,
            PeerRpcType::stitch_search_response,
            context.search_item_counts[shard]);
        } else if (response == TryPeerResponse::success) {
          (void)rearm_peer_rpc_response(
            request_id, shard,
            PeerRpcType::stitch_search_response,
            context.search_item_counts[shard]);
        }
        storage_owner_maintenance_failed_.fetch_add(
          1, std::memory_order_relaxed);
        const u64 now = steady_now_ns();
        (void)requests.mark_retry(
          request_id, now, now + retry_backoff_ns);
        progressed = true;
        continue;
      }
      progressed = retry_search_if_due(context, shard) || progressed;
    }
    return progressed;
  };

  const auto prepare_stitch_reverse = [&](Stage2Context& context) {
    dense_hashmap_t<u64, vec<RemotePtr>> local_updates;
    for (auto& ops : context.remote_ops_by_peer) {
      ops.clear();
    }

    size_t ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      const bool target_deleted =
        (load_local_node_header_acquire(task.target) &
         VamanaNode::HEADER_DELETED) != 0;
      if (target_deleted ||
          !storage_owner_task_current(task.id, task.generation, task.target)) {
        complete_stale_stitch(task);
        continue;
      }

      vec<RemotePtr> final_neighbors;
      if (task.stitch_prepared) {
        final_neighbors = task.stitch_neighbors;
      } else {
        hashset_t<RemotePtr> skip;
        skip.insert(task.target);
        final_neighbors = robust_prune_snapshots_cpu(
          context.targets[item].vector_data.data(),
          VamanaNode::vector_dtype(),
          span<const NodeSnapshot>{
            context.candidate_storage.data() +
              item * candidate_capacity_per_item,
            context.candidate_counts[item]},
          skip,
          config, config.R);
        lib_assert(final_neighbors.size() <= config.R,
                   "online stitch exceeded graph degree");
      }
      task.stitch_prepared = true;
      task.stitch_neighbors = std::move(final_neighbors);

      for (const RemotePtr& neighbor : task.stitch_neighbors) {
        if (local_shard(neighbor.memory_node())) {
          local_updates[neighbor.raw_address].push_back(task.target);
        } else {
          context.remote_ops_by_peer[neighbor.memory_node()].push_back(
            ReverseUpdateOp{neighbor.raw_address, task.target.raw_address});
        }
      }
      if (ready != item) {
        context.tasks[ready] = std::move(task);
      }
      ++ready;
    }
    context.tasks.resize(ready);

    if (!apply_local_reverse_updates_batched(local_updates, config)) {
      storage_owner_maintenance_failed_.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }

    u64 expected_mask = 0;
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      if (shard != storage_id_ &&
          !context.remote_ops_by_peer[shard].empty()) {
        expected_mask |= u64{1} << shard;
      }
    }
    const Stage2EventResult transition =
      states.begin_reverse(context.handle, expected_mask);
    lib_assert(transition == Stage2EventResult::phase_advanced ||
                 transition == Stage2EventResult::ready_to_finalize,
               "stage2 stitch failed to enter reverse_pending");
    register_reverse_requests(context, expected_mask);
    return true;
  };

  const auto prepare_cleanup_reverse = [&](Stage2Context& context) {
    dense_hashmap_t<u64, vec<RemotePtr>> local_removals;
    for (auto& ops : context.remote_ops_by_peer) {
      ops.clear();
    }

    size_t ready = 0;
    for (size_t item = 0; item < context.tasks.size(); ++item) {
      StorageOwnerMaintenanceTask& task = context.tasks[item];
      if (task.target.is_null()) {
        storage_owner_maintenance_cleanup_processed_.fetch_add(
          1, std::memory_order_relaxed);
        complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
        continue;
      }
      NodeSnapshot deleted_snapshot;
      const bool readable = read_node_snapshot(task.target, deleted_snapshot);
      lib_assert(readable, "local cleanup snapshot was unreadable");
      if (!deleted_snapshot.deleted && !task.cleanup_repair_only) {
        complete_stale_cleanup(task);
        continue;
      }

      // A prepared stitch can become stale only after a later erase/upsert
      // tombstones the same physical node. That later mutation owns an
      // ordinary cleanup intent for the preserved adjacency. This repair must
      // therefore undo only the backlinks attempted by the stale stitch; a
      // preserved+supplemental union can contain 2R operations per item and
      // cannot fit the unchanged schema-15 R*batch peer message.
      lib_assert(!task.cleanup_repair_only || deleted_snapshot.deleted,
                 "stale stitch repair requires a successor tombstone cleanup");
      lib_assert(task.cleanup_repair_only || task.cleanup_neighbors.empty(),
                 "ordinary tombstone cleanup unexpectedly carried repair neighbors");
      vec<RemotePtr> preserved_neighbors;
      if (deleted_snapshot.deleted && !task.cleanup_repair_only) {
        preserved_neighbors = read_preserved_neighbor_list(task.target);
      }
      vec<RemotePtr> old_neighbors = select_cleanup_neighbors(
        task.cleanup_repair_only,
        span<const RemotePtr>{preserved_neighbors.data(),
                              preserved_neighbors.size()},
        span<const RemotePtr>{task.cleanup_neighbors.data(),
                              task.cleanup_neighbors.size()});
      lib_assert(old_neighbors.size() <= config.R,
                 "stage2 cleanup exceeded the schema-15 per-item wire bound");
      for (const RemotePtr& neighbor : old_neighbors) {
        if (neighbor.is_null() ||
            neighbor.memory_node() >= num_storage_nodes_) {
          continue;
        }
        if (local_shard(neighbor.memory_node())) {
          local_removals[neighbor.raw_address].push_back(task.target);
        } else {
          context.remote_ops_by_peer[neighbor.memory_node()].push_back(
            ReverseUpdateOp{neighbor.raw_address, task.target.raw_address});
        }
      }
      if (ready != item) {
        context.tasks[ready] = std::move(task);
      }
      ++ready;
    }
    context.tasks.resize(ready);

    if (!remove_local_neighbors_batched(local_removals, config)) {
      storage_owner_maintenance_failed_.fetch_add(
        1, std::memory_order_relaxed);
      return false;
    }

    u64 expected_mask = 0;
    for (u32 shard = 0; shard < num_storage_nodes_; ++shard) {
      if (shard != storage_id_ &&
          !context.remote_ops_by_peer[shard].empty()) {
        expected_mask |= u64{1} << shard;
      }
    }
    const Stage2EventResult transition =
      states.begin_reverse(context.handle, expected_mask);
    lib_assert(transition == Stage2EventResult::phase_advanced ||
                 transition == Stage2EventResult::ready_to_finalize,
               "cleanup stage2 failed to enter reverse_pending");
    register_reverse_requests(context, expected_mask);
    return true;
  };

  const auto finalize_context = [&](Stage2Context& context) {
    const Stage2EventResult transition = states.finalize(context.handle);
    lib_assert(transition == Stage2EventResult::phase_advanced,
               "stage2 finalized before all reverse ACKs");

    if (context.kind == StorageOwnerMaintenanceKind::stitch_insert) {
      for (StorageOwnerMaintenanceTask& task : context.tasks) {
        const bool current = commit_rebased_stitch_neighbors(task);

        if (!current) {
          storage_owner_maintenance_stale_.fetch_add(
            1, std::memory_order_relaxed);
        } else {
          record_finalized_live(task.queued_at);
        }
        storage_owner_maintenance_processed_.fetch_add(
          1, std::memory_order_relaxed);

        bool handed_off = false;
        if (!current) {
          vec<RemotePtr> stale_cleanup = std::move(task.stitch_neighbors);
          handed_off = handoff_stitch_cleanup(
            task, std::move(stale_cleanup));
        }
        if (!handed_off) {
          complete_storage_owner_maintenance_sequence(
            task.maintenance_sequence);
        }
      }
    } else {
      for (StorageOwnerMaintenanceTask& task : context.tasks) {
        if (!task.cleanup_repair_only) {
          retire_local_dynamic_node(task.target, task.maintenance_sequence);
        }
        storage_owner_maintenance_cleanup_processed_.fetch_add(
          1, std::memory_order_relaxed);
        complete_storage_owner_maintenance_sequence(task.maintenance_sequence);
      }
    }

    const Stage2ContextHandle handle = context.handle;
    reset_context(context);
    lib_assert(states.release(handle),
               "stage2 context release violated finalized generation");
    storage_owner_maintenance_active_workers_.fetch_sub(
      1, std::memory_order_acq_rel);
    storage_owner_maintenance_cv_.notify_all();
  };

  const auto drive_context = [&](Stage2Context& context) {
    bool progressed = false;
    for (;;) {
      const auto snapshot = states.snapshot(context.handle);
      lib_assert(snapshot.has_value(), "active stage2 context became stale");
      switch (snapshot->phase) {
        case Stage2Phase::local_ready:
          (void)prepare_local(context);
          progressed = true;
          continue;
        case Stage2Phase::remote_search_pending:
          progressed = poll_search_responses(context) || progressed;
          if (states.snapshot(context.handle)->phase ==
              Stage2Phase::remote_search_pending) {
            return progressed;
          }
          continue;
        case Stage2Phase::prune_ready: {
          const bool prepared =
            context.kind == StorageOwnerMaintenanceKind::stitch_insert
              ? prepare_stitch_reverse(context)
              : prepare_cleanup_reverse(context);
          if (!prepared) {
            return progressed;
          }
          progressed = true;
          continue;
        }
        case Stage2Phase::reverse_pending:
          if (states.snapshot(context.handle)->phase ==
              Stage2Phase::reverse_pending &&
              states.snapshot(context.handle)->completed_reverse_mask !=
                states.snapshot(context.handle)->expected_reverse_mask) {
            return progressed;
          }
          finalize_context(context);
          return true;
        case Stage2Phase::finalized:
          lib_failure("stage2 context remained active after finalization");
      }
    }
  };

  const auto try_admit_context = [&]() -> Stage2Context* {
    const Stage2AdmissionDecision admission = decide_stage2_admission(
      states.full(),
      storage_owner_maintenance_shutdown_.load(std::memory_order_acquire),
      [&]() { return storage_owner_maintenance_foreground_busy(config); });
    if (admission == Stage2AdmissionDecision::unavailable) {
      return nullptr;
    }
    if (admission == Stage2AdmissionDecision::foreground_pressure) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      return nullptr;
    }
    if (!try_acquire_storage_owner_maintenance_slot(config)) {
      storage_owner_maintenance_pressure_yields_.fetch_add(
        1, std::memory_order_relaxed);
      return nullptr;
    }

    const size_t batch_limit =
      std::max<size_t>(1, config.storage_owner_batch_max);
    const auto acquire_context = [&]() -> Stage2Context* {
      const auto handle = states.try_acquire();
      lib_assert(handle.has_value(),
                 "stage2 context tracker unexpectedly exhausted");
      Stage2Context& context = contexts[handle->slot];
      reset_context(context);
      context.active = true;
      context.handle = *handle;
      return &context;
    };

    // Repair continuations own an already-admitted maintenance sequence and
    // therefore take priority over new stitch work. This removes the stale
    // stitch's attempted backlinks before advancing the watermark and proves
    // the dedicated queue cannot grow across successive admission waves.
    if (storage_owner_repair_tasks_ != nullptr) {
      StorageOwnerMaintenanceTask repair;
      if (storage_owner_repair_tasks_->try_pop(repair)) {
        if (storage_owner_cleanup_ready(repair.maintenance_sequence)) {
          Stage2Context& context = *acquire_context();
          context.kind = StorageOwnerMaintenanceKind::cleanup_deleted_node;
          context.tasks.push_back(std::move(repair));
          repair = StorageOwnerMaintenanceTask{};
          while (context.tasks.size() < batch_limit &&
                 storage_owner_repair_tasks_->try_pop(repair)) {
            if (!storage_owner_cleanup_ready(repair.maintenance_sequence)) {
              lib_assert(storage_owner_repair_tasks_->try_push(
                           std::move(repair)),
                         "failed to return a not-yet-ready repair descriptor");
              break;
            }
            context.tasks.push_back(std::move(repair));
            repair = StorageOwnerMaintenanceTask{};
          }
          storage_owner_maintenance_cv_.notify_all();
          return &context;
        }
        lib_assert(storage_owner_repair_tasks_->try_push(std::move(repair)),
                   "failed to return a not-yet-ready repair descriptor");
      }
    }

    std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
    const auto ready_cleanup = std::find_if(
      storage_owner_cleanup_tasks_.begin(), storage_owner_cleanup_tasks_.end(),
      [&](const StorageOwnerMaintenanceTask& task) {
        return storage_owner_cleanup_ready(task.maintenance_sequence);
      });
    const bool cleanup_ready =
      ready_cleanup != storage_owner_cleanup_tasks_.end();
    const bool choose_stitch =
      !storage_owner_stitch_tasks_.empty() &&
      (!cleanup_ready || storage_owner_stitch_tasks_.front().queued_at <=
                            ready_cleanup->queued_at);
    if (!choose_stitch && !cleanup_ready) {
      storage_owner_maintenance_active_workers_.fetch_sub(
        1, std::memory_order_acq_rel);
      storage_owner_maintenance_cv_.notify_all();
      return nullptr;
    }

    Stage2Context& context = *acquire_context();
    context.kind = choose_stitch
      ? StorageOwnerMaintenanceKind::stitch_insert
      : StorageOwnerMaintenanceKind::cleanup_deleted_node;

    if (choose_stitch) {
      while (!storage_owner_stitch_tasks_.empty() &&
             context.tasks.size() < batch_limit) {
        context.tasks.push_back(
          std::move(storage_owner_stitch_tasks_.front()));
        storage_owner_stitch_tasks_.pop_front();
      }
      storage_owner_stitch_batches_.fetch_add(1, std::memory_order_relaxed);
      storage_owner_stitch_batched_items_.fetch_add(
        context.tasks.size(), std::memory_order_relaxed);
    } else {
      for (auto iterator = storage_owner_cleanup_tasks_.begin();
           iterator != storage_owner_cleanup_tasks_.end() &&
             context.tasks.size() < batch_limit;) {
        if (!storage_owner_cleanup_ready(iterator->maintenance_sequence)) {
          ++iterator;
          continue;
        }
        context.tasks.push_back(std::move(*iterator));
        iterator = storage_owner_cleanup_tasks_.erase(iterator);
      }
    }
    lock.unlock();
    storage_owner_maintenance_cv_.notify_all();
    lib_assert(!context.tasks.empty(),
               "stage2 admitted an empty maintenance context");
    return &context;
  };

  for (;;) {
    if (storage_owner_maintenance_shutdown_.load(std::memory_order_acquire)) {
      if (storage_owner_reverse_outbox_ != nullptr) {
        (void)storage_owner_reverse_outbox_->erase_queued_worker(worker_id);
        for (;;) {
          const auto wire_request_id =
            storage_owner_reverse_outbox_->discard_owned_aggregate(worker_id);
          if (!wire_request_id.has_value()) break;
          cancel_peer_rpc_response(*wire_request_id);
        }
      }
      for (Stage2Context& context : contexts) {
        if (!context.active) {
          continue;
        }
        for (u64 request_id : context.search_request_ids) {
          if (request_id != 0) cancel_peer_rpc_response(request_id);
        }
        storage_owner_maintenance_active_workers_.fetch_sub(
          1, std::memory_order_acq_rel);
      }
      current_storage_owner_thread_ = nullptr;
      return;
    }

    // Drain every currently sendable per-peer descriptor before polling
    // contexts. A second pass below catches work produced by pruning in this
    // iteration; neither pass waits for a timer to form a batch.
    bool progressed = drive_reverse_outbox();
    progressed = drain_reverse_completions() || progressed;
    for (Stage2Context& context : contexts) {
      if (context.active) {
        progressed = drive_context(context) || progressed;
      }
    }

    while (Stage2Context* context = try_admit_context()) {
      progressed = true;
      (void)drive_context(*context);
    }

    progressed = drive_reverse_outbox() || progressed;
    progressed = drain_reverse_completions() || progressed;

    maybe_log_storage_owner_maintenance_observation();
    if (!progressed) {
      std::unique_lock<std::mutex> lock(storage_owner_maintenance_mutex_);
      storage_owner_maintenance_cv_.wait_for(
        lock, std::chrono::milliseconds(1));
    }
  }
}
