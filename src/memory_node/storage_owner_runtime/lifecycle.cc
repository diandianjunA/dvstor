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
  const auto peer_read_credits = peer_rdma_read_credit_plan();
  print_status("storage-owner peer RDMA read credits: per_data_qp=" +
               std::to_string(peer_read_credits.per_qp) +
               " data_qps_per_peer=" +
               std::to_string(peer_read_credits.data_qps_per_peer) +
               " per_peer=" + std::to_string(peer_read_credits.per_peer) +
               " global=" + std::to_string(peer_read_credits.global) +
               " shared_cq_read_budget=" +
               std::to_string(peer_read_credits.shared_cq_read_budget) +
               " ordered_snapshot_pairs_per_chain=" +
               std::to_string(
                 memory_node_detail::peer_rdma_read_pair_group_limit(
                   peer_read_credits)) +
               " (requested_per_data_qp=" +
               std::to_string(storage_owner_peer_rdma_tokens_) + ")");
  print_status("storage-owner online insert tuning: construction_beam=" +
               std::to_string(config.resolved_storage_owner_construction_width()) +
               " snapshot_batch=" + std::to_string(config.storage_owner_search_snapshot_batch) +
               " protocol=centroid-home-two-stage");
  print_status("storage-owner stage1=physical-home local search and backlinks; "
               "stage2=generation-fenced home expand+score continuation");
  print_status("storage-owner responses=foreground direct post; "
               "completion=repost by service poller");
  const u32 rpc_parallelism = std::max<u32>(
    1, static_cast<u32>(num_clients_) * insert_runtime_.request_slot_count);
  const auto cpu_plan = memory_node_detail::derive_storage_owner_cpu_plan(
    core_assignment_.available_core_count(), num_compute_threads_,
    rpc_parallelism, config.storage_owner_maintenance_workers,
    num_storage_nodes_ > 0 ? num_storage_nodes_ - 1 : 0);
  const u32 worker_count = cpu_plan.foreground_coordinators;
  print_status("storage-owner foreground pipeline: coordinators=" +
               std::to_string(worker_count) +
               " cpu_lanes=" +
               std::to_string(cpu_plan.foreground_workers) +
               " (assigned_cpus=" +
               std::to_string(core_assignment_.available_core_count()) +
               ", rpc_parallelism=" + std::to_string(rpc_parallelism) + ")");
  print_status("storage-owner CPU plan: foreground_cpu=" +
               std::to_string(cpu_plan.foreground_workers) +
               " foreground_coordinators=" +
               std::to_string(cpu_plan.foreground_coordinators) +
               " maintenance=" +
               std::to_string(cpu_plan.maintenance_workers) +
               " peer_stage1=" +
               std::to_string(cpu_plan.peer_stage1_workers) +
               " (split into isolated Stage1/Stage2-home domains)" +
               " peer_reverse=" +
               std::to_string(cpu_plan.peer_reverse_workers) +
               " peer_cleanup=" +
               std::to_string(cpu_plan.peer_cleanup_workers) +
               " peer_placement=" +
               std::to_string(cpu_plan.peer_placement_workers) +
               " peer_progress=" +
               std::to_string(cpu_plan.peer_progress_threads) +
               " foreground_progress=" +
               std::to_string(cpu_plan.foreground_progress_threads));
  storage_owner_threads_.reserve(worker_count);
  for (u32 i = 0; i < worker_count; ++i) {
    // Foreground Stage1 is synchronous and partition-local. Cross-shard reads
    // belong exclusively to the Stage2 executor, so one scratch state per
    // worker is sufficient and no foreground peer-RDMA scratch is allocated.
    auto thread = std::make_unique<StorageOwnerThread>(
      i, 1, config.max_send_queue_wr);
    storage_owner_threads_.push_back(std::move(thread));
  }
  vec<u32> foreground_cpus;
  if (!config.disable_thread_pinning) {
    foreground_cpus.reserve(cpu_plan.foreground_workers);
    for (u32 lane = 0; lane < cpu_plan.foreground_workers; ++lane) {
      foreground_cpus.push_back(core_assignment_.get_available_core());
    }
  }
  for (u32 i = 0; i < worker_count; ++i) {
    storage_insert_workers_.emplace_back([this, i]() { storage_owner_insert_worker_loop(i); });
    if (!config.disable_thread_pinning) {
      lib_assert(!foreground_cpus.empty(),
                 "foreground coordinator has no assigned CPU lane");
      // Coordinators blocked on peer responses do not consume an additional
      // process CPU.  Reusing this fixed lane set preserves the CPU plan while
      // allowing the registered request window to remain in flight.
      pin_thread(storage_insert_workers_.back(),
                 foreground_cpus[i % foreground_cpus.size()]);
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
