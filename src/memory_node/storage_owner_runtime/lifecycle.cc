#include "memory_node/storage_owner_runtime/detail.hh"
#include "memory_node/storage_owner_cpu_plan.hh"

using namespace memory_node_storage_owner_runtime_detail;

void MemoryNode::setup_insert_runtime(const Configuration& config) {
  lib_assert(static_cast<u64>(config.storage_owner_batch_max) * VamanaNode::R <=
               std::numeric_limits<u32>::max(),
             "storage_owner invalidation capacity is too large for the wire format");
  const size_t insert_request_bytes = align_up(std::max(
    service::storage_owner::insert_batch_request_bytes(
      config.storage_owner_batch_max),
    service::storage_owner::mutation_batch_request_bytes(
      config.storage_owner_batch_max)));
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
  storage_insert_tasks_ =
    std::make_unique<bounded::Queue<StorageOwnerInsertTask>>(slot_count);
  insert_runtime_.buffer.allocate(insert_runtime_.response_offset + insert_response_bytes * slot_count);
  insert_runtime_.buffer.touch_memory();
  insert_runtime_.region = std::make_unique<LocalMemoryRegion>(
    context_, insert_runtime_.buffer.get_full_buffer(), insert_runtime_.buffer.buffer_size);
  storage_client_send_mutexes_.clear();
  storage_client_send_mutexes_.reserve(num_clients_);
  for (u32 client_id = 0; client_id < num_clients_; ++client_id) {
    storage_client_send_mutexes_.push_back(std::make_unique<std::mutex>());
  }
}

void MemoryNode::start_storage_owner_insert_workers(const Configuration& config) {
  print_status("storage-owner peer RDMA read credits per peer: " +
               std::to_string(peer_rdma_read_credit_limit()) +
               " per QP: " + std::to_string(peer_rdma_read_credit_limit_per_qp()) +
               " (requested=" + std::to_string(storage_owner_peer_rdma_tokens_) + ")");
  print_status("storage-owner online insert tuning: construction_beam=" +
               std::to_string(config.resolved_storage_owner_construction_width()) +
               " snapshot_batch=" + std::to_string(config.storage_owner_search_snapshot_batch) +
               " update_mode=" + config.storage_owner_update_mode);
  if (storage_owner_local_stitch_mode(config)) {
    print_status("storage-owner stage1=direct local commit without peer RDMA; "
                 "reverse edges=batched stage2");
  }
  print_status("storage-owner responses=foreground direct post; "
               "completion=repost by service poller");
  const u32 rpc_parallelism = std::max<u32>(
    1, static_cast<u32>(num_clients_) * insert_runtime_.request_slot_count);
  const auto cpu_plan = memory_node_detail::derive_storage_owner_cpu_plan(
    core_assignment_.available_core_count(), num_compute_threads_,
    rpc_parallelism, config.storage_owner_maintenance_workers,
    num_storage_nodes_ > 0 ? num_storage_nodes_ - 1 : 0);
  const u32 worker_count = cpu_plan.foreground_workers;
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
    coroutine_scratch_stride * std::max<u32>(1, coroutines_per_worker);
  print_status("storage-owner coroutine scratch: snapshot_bytes=" +
               std::to_string(snapshot_bytes) +
               " snapshot_stride=" + std::to_string(snapshot_stride) +
               " batch=" + std::to_string(snapshot_batch) +
               " per_coroutine=" + std::to_string(coroutine_scratch_stride));
  print_status("storage-owner foreground workers: " +
               std::to_string(worker_count) +
               " (assigned_cpus=" +
               std::to_string(core_assignment_.available_core_count()) +
               ", rpc_parallelism=" + std::to_string(rpc_parallelism) + ")");
  print_status("storage-owner CPU plan: foreground=" +
               std::to_string(cpu_plan.foreground_workers) +
               " maintenance=" +
               std::to_string(cpu_plan.maintenance_workers) +
               " peer_search=" +
               std::to_string(cpu_plan.peer_search_workers) +
               " peer_reverse=" +
               std::to_string(cpu_plan.peer_reverse_workers) +
               " peer_progress=" +
               std::to_string(cpu_plan.peer_progress_threads) +
               " foreground_progress=" +
               std::to_string(cpu_plan.foreground_progress_threads));
  storage_owner_threads_.reserve(worker_count);
  for (u32 i = 0; i < worker_count; ++i) {
    auto thread = std::make_unique<StorageOwnerThread>(i, coroutines_per_worker, config.max_send_queue_wr);
    if (peer_context_ && !storage_owner_local_stitch_mode(config)) {
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
    if (!config.disable_thread_pinning) {
      pin_thread(storage_insert_workers_.back(),
                 core_assignment_.get_available_core());
    }
  }
}

void MemoryNode::storage_owner_insert_worker_loop(u32 worker_id) {
  current_storage_owner_thread_ = storage_owner_threads_[worker_id].get();
  for (;;) {
    StorageOwnerInsertTask task;
    if (!storage_insert_tasks_->pop_wait(task, storage_insert_shutdown_)) {
      current_storage_owner_thread_ = nullptr;
      return;
    }

    mark_storage_owner_foreground_activity();
    storage_owner_insert_active_workers_.fetch_add(1, std::memory_order_acq_rel);
    process_storage_owner_insert_task(task);
    storage_owner_insert_active_workers_.fetch_sub(1, std::memory_order_acq_rel);
    mark_storage_owner_foreground_activity();
  }
}
