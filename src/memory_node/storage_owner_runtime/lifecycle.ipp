void MemoryNode::setup_insert_runtime(const Configuration& config) {
  lib_assert(static_cast<u64>(config.storage_owner_batch_max) * VamanaNode::R <=
               std::numeric_limits<u32>::max(),
             "storage_owner invalidation capacity is too large for the wire format");
  const size_t insert_request_bytes = align_up(std::max(
    service::storage_owner::insert_batch_request_bytes(
      config.storage_owner_batch_max, VamanaNode::DIM,
      storage_owner_local_stitch_mode(config) ? config.storage_owner_anchor_hints : 0),
    service::storage_owner::mutation_batch_request_bytes(
      config.storage_owner_batch_max, VamanaNode::DIM,
      storage_owner_local_stitch_mode(config) ? config.storage_owner_anchor_hints : 0)));
  insert_runtime_.request_bytes = insert_request_bytes;
  insert_runtime_.request_slot_count = std::max<u32>(1, config.storage_owner_rpc_depth);
  const size_t insert_response_bytes =
    align_up(service::storage_owner::insert_batch_response_bytes(config.storage_owner_batch_max));
  lib_assert(insert_request_bytes <= std::numeric_limits<u32>::max() &&
             insert_response_bytes <= std::numeric_limits<u32>::max(),
             "storage_owner RPC message is too large for verbs SGEs; reduce batch size or vector dimension");
  const size_t slot_count =
    static_cast<size_t>(num_clients_) * insert_runtime_.request_slot_count;
  lib_assert(slot_count <= static_cast<size_t>(config.max_recv_queue_wr),
             "storage_owner RPC receive slots exceed memory-node receive CQ capacity");
  insert_runtime_.response_offset = insert_runtime_.request_bytes * slot_count;
  insert_runtime_.buffer.allocate(insert_runtime_.response_offset + insert_response_bytes * slot_count);
  insert_runtime_.buffer.touch_memory();
  insert_runtime_.region = std::make_unique<LocalMemoryRegion>(
    context_, insert_runtime_.buffer.get_full_buffer(), insert_runtime_.buffer.buffer_size);
}

void MemoryNode::start_storage_owner_insert_workers(const Configuration& config) {
  print_status("storage-owner peer RDMA read credits per peer: " +
               std::to_string(peer_rdma_read_credit_limit()) +
               " per QP: " + std::to_string(peer_rdma_read_credit_limit_per_qp()) +
               " (requested=" + std::to_string(storage_owner_peer_rdma_tokens_) + ")");
  print_status("storage-owner online insert tuning: construction_beam=" +
               std::to_string(config.storage_owner_construction_beam_width == 0
                                ? config.beam_width_construction
                                : std::min(config.beam_width_construction,
                                           config.storage_owner_construction_beam_width)) +
               " snapshot_batch=" + std::to_string(config.storage_owner_search_snapshot_batch) +
               " prune_max_candidates=" + std::to_string(config.storage_owner_prune_max_candidates) +
               " update_mode=" + config.storage_owner_update_mode);
  const u32 worker_count = std::max<u32>(1, std::min<u32>(8, std::max<u32>(1, num_compute_threads_ / 2)));
  const u32 coroutines_per_worker = std::max<u32>(1, config.storage_owner_coroutines);
  const size_t snapshot_bytes = memory_node_detail::storage_owner_snapshot_bytes();
  const size_t snapshot_stride = memory_node_detail::storage_owner_snapshot_stride();
  const size_t neighbor_stride = align_up(VamanaNode::neighbor_read_size());
  const size_t snapshot_batch =
    std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  // Keep one general-purpose slot beyond the batch area. This prevents a
  // neighbor/node fallback in the same coroutine from aliasing batched reads.
  const size_t coroutine_scratch_stride =
    align_up(snapshot_stride * snapshot_batch +
             std::max(VamanaNode::total_size(), neighbor_stride));
  const size_t scratch_bytes =
    std::max<size_t>(64ull * 1024ull * 1024ull,
                     coroutine_scratch_stride * std::max<u32>(1, coroutines_per_worker));
  print_status("storage-owner coroutine scratch: snapshot_bytes=" +
               std::to_string(snapshot_bytes) +
               " snapshot_stride=" + std::to_string(snapshot_stride) +
               " batch=" + std::to_string(snapshot_batch) +
               " per_coroutine=" + std::to_string(coroutine_scratch_stride));
  storage_owner_threads_.reserve(worker_count);
  for (u32 i = 0; i < worker_count; ++i) {
    auto thread = std::make_unique<StorageOwnerThread>(i, coroutines_per_worker, config.max_send_queue_wr);
    if (peer_context_) {
      thread->init_peer_scratch(*peer_context_, scratch_bytes, coroutine_scratch_stride);
    }
    storage_owner_threads_.push_back(std::move(thread));
  }
  storage_owner_async_candidates_.clear();
  storage_owner_async_candidates_.resize(worker_count);
  for (auto& worker_candidates : storage_owner_async_candidates_) {
    worker_candidates.resize(coroutines_per_worker);
  }
  for (u32 i = 0; i < worker_count; ++i) {
    storage_insert_workers_.emplace_back([this, i]() { storage_owner_insert_worker_loop(i); });
  }
}

void MemoryNode::storage_owner_insert_worker_loop(u32 worker_id) {
  current_storage_owner_thread_ = storage_owner_threads_[worker_id].get();
  const Configuration& config = *storage_worker_config_;
  for (;;) {
    vec<StorageOwnerInsertTask> tasks;
    u32 total_items = 0;
    {
      std::unique_lock<std::mutex> lock(storage_insert_tasks_mutex_);
      storage_insert_tasks_cv_.wait(lock, [&]() {
        return storage_insert_shutdown_.load(std::memory_order_acquire) || !storage_insert_tasks_.empty();
      });
      if (storage_insert_shutdown_.load(std::memory_order_acquire) && storage_insert_tasks_.empty()) {
        current_storage_owner_thread_ = nullptr;
        return;
      }

      while (!storage_insert_tasks_.empty()) {
        const u32 next_items = storage_insert_tasks_.front().item_count;
        if (!tasks.empty() && total_items + next_items > std::max<u32>(config.storage_owner_batch_max, 64)) {
          break;
        }
        total_items += next_items;
        tasks.push_back(std::move(storage_insert_tasks_.front()));
        storage_insert_tasks_.pop_front();
      }
    }

    mark_storage_owner_foreground_activity();
    storage_owner_insert_active_workers_.fetch_add(1, std::memory_order_acq_rel);
    process_storage_owner_insert_tasks(tasks);
    storage_owner_insert_active_workers_.fetch_sub(1, std::memory_order_acq_rel);
    mark_storage_owner_foreground_activity();
  }
}
