struct MemoryNode::GlobalMedoidReadAwaitable {
  bool ready{};
  byte_t* buffer{};
  MemoryNode* node{};

  bool await_ready() const { return ready; }
  static void await_suspend(std::coroutine_handle<>) {}
  RemotePtr await_resume() const {
    if (node->storage_id_ == 0) {
      return RemotePtr{
        *reinterpret_cast<u64*>(node->index_buffer_.get_full_buffer() + 8)};
    }
    return RemotePtr{*reinterpret_cast<const u64*>(buffer)};
  }
};

RemotePtr MemoryNode::allocate_local_node() {
  size_t node_size = VamanaNode::allocation_size();
  while (node_size % 8 != 0) {
    node_size += 4;
  }

  if (storage_owner_reclaim_candidates_.load(std::memory_order_acquire) != 0) {
    std::lock_guard<std::mutex> lock(storage_owner_reclaim_mutex_);
    auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
      index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
    std::atomic_ref<u64> durable(control->durable_maintenance_sequence);
    const auto reclaimed = storage_owner_reclaim_queue_.acquire(
      durable.load(std::memory_order_acquire), minimum_compute_reclaim_ack());
    if (reclaimed.has_value()) {
      lib_assert(reclaimed->memory_node() == storage_id_ &&
                   reclaimed->byte_offset() >= gpu_dynamic_node_base_ &&
                   (reclaimed->byte_offset() - gpu_dynamic_node_base_) % node_size == 0,
                 "storage-owner reclaim queue returned an invalid dynamic node");
      storage_owner_reclaim_candidates_.store(
        storage_owner_reclaim_queue_.size(), std::memory_order_release);
      std::atomic_ref<u64>(control->reclaim_pending_nodes).store(
        storage_owner_reclaim_queue_.size(), std::memory_order_release);
      std::atomic_ref<u64>(control->reclaim_reused_nodes).store(
        storage_owner_reclaim_queue_.reused(), std::memory_order_release);
      return *reclaimed;
    }
  }

  auto* free_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer());
  std::atomic_ref<u64> alloc_ref(*free_ptr);
  const u64 offset = alloc_ref.fetch_add(node_size, std::memory_order_acq_rel);
  lib_assert(offset + node_size <= mn_memory_bytes_, "storage node out of memory");
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  std::atomic_ref<u64> high_watermark(control->dynamic_high_watermark);
  u64 observed = high_watermark.load(std::memory_order_relaxed);
  while (observed < offset + node_size &&
         !high_watermark.compare_exchange_weak(
           observed, offset + node_size,
           std::memory_order_release, std::memory_order_relaxed)) {}
  return RemotePtr{storage_id_, offset};
}

void MemoryNode::retire_local_dynamic_node(RemotePtr pointer,
                                           u64 maintenance_sequence) {
  if (pointer.is_null() || pointer.memory_node() != storage_id_ ||
      pointer.byte_offset() < gpu_dynamic_node_base_ || maintenance_sequence == 0) {
    return;
  }
  const u64 node_size = VamanaNode::allocation_size();
  lib_assert((pointer.byte_offset() - gpu_dynamic_node_base_) % node_size == 0,
             "cannot retire a misaligned storage-owner dynamic node");
  std::lock_guard<std::mutex> lock(storage_owner_reclaim_mutex_);
  storage_owner_reclaim_queue_.retire(pointer, maintenance_sequence);
  storage_owner_reclaim_candidates_.store(
    storage_owner_reclaim_queue_.size(), std::memory_order_release);
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  std::atomic_ref<u64>(control->reclaim_pending_nodes).store(
    storage_owner_reclaim_queue_.size(), std::memory_order_release);
}

u64 MemoryNode::minimum_compute_reclaim_ack() const {
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const u32 client_count = control->compute_client_count;
  if (client_count == 0 || client_count > gpu_search::format::kMaxComputeClients) {
    return 0;
  }
  u64 minimum = std::numeric_limits<u64>::max();
  for (u32 client = 0; client < client_count; ++client) {
    std::atomic_ref<u64> ack(control->reclaim_ack_sequences[client]);
    minimum = std::min(minimum, ack.load(std::memory_order_acquire));
  }
  return minimum;
}

bool MemoryNode::load_owner_idmap(const filepath_t& index_prefix) {
  idmap_.clear();
  mutations_inflight_.clear();
  if (index_prefix.empty()) {
    return true;
  }
  const filepath_t path = index_path::owner_idmap_file(index_prefix, storage_id_ + 1, num_storage_nodes_);
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    std::cerr << "[storage-owner] missing idmap sidecar: " << path << std::endl;
    return false;
  }
  vamana::idmap::Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!input.good() || header.magic != vamana::idmap::kMagic ||
      header.version != vamana::idmap::kVersion ||
      header.owner_shard != storage_id_ ||
      header.shard_count != num_storage_nodes_) {
    std::cerr << "[storage-owner] invalid idmap sidecar: " << path << std::endl;
    return false;
  }
  const u64 dynamic_headroom = std::max<u64>(1024ull * 1024ull, header.entry_count / 20);
  const u64 reserve_count = header.entry_count + dynamic_headroom;
  // Reserve dynamic-write headroom up front; storage-owner inserts keep adding
  // idmap entries and dense maps grow by reallocating large contiguous blocks.
  idmap_.reserve(static_cast<size_t>(reserve_count));
  for (u64 i = 0; i < header.entry_count; ++i) {
    vamana::idmap::Entry entry;
    input.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    if (!input.good()) return false;
    idmap_[entry.id] = FreshnessEntry{
      RemotePtr{entry.rptr_raw},
      entry.generation,
      (entry.flags & vamana::idmap::kDeleted) != 0};
  }
  print_status("storage-owner idmap loaded entries=" +
               std::to_string(idmap_.size()) +
               " reserved=" + std::to_string(reserve_count));
  return true;
}

void MemoryNode::publish_mutation(node_t id, RemotePtr ptr, u32 generation, bool deleted) {
  std::lock_guard<std::mutex> lock(idmap_mutex_);
  idmap_[id] = FreshnessEntry{ptr, generation, deleted};
  mutations_inflight_.erase(id);
}

service::storage_owner::MutationStatus MemoryNode::prepare_mutation(
    node_t id,
    service::storage_owner::MutationKind kind,
    FreshnessEntry* old_entry,
    u32* new_generation) {
  std::lock_guard<std::mutex> lock(idmap_mutex_);
  if (mutations_inflight_.contains(id)) {
    return service::storage_owner::MutationStatus::failed;
  }
  auto it = idmap_.find(id);
  const bool exists = it != idmap_.end();
  const bool live = exists && !it->second.deleted;
  if (old_entry != nullptr) {
    *old_entry = exists ? it->second : FreshnessEntry{};
    if (old_entry->deleted) {
      old_entry->current = RemotePtr{};
    }
  }
  const u32 previous_generation = exists ? it->second.generation : 0;
  if (new_generation != nullptr) {
    *new_generation = previous_generation + 1;
  }
  if (kind == service::storage_owner::MutationKind::insert && live) {
    return service::storage_owner::MutationStatus::already_exists;
  }
  if (kind == service::storage_owner::MutationKind::erase) {
    if (!exists) return service::storage_owner::MutationStatus::not_found;
    if (!live) return service::storage_owner::MutationStatus::already_deleted;
  }
  mutations_inflight_.insert(id);
  return service::storage_owner::MutationStatus::ok;
}

bool MemoryNode::mark_node_deleted(RemotePtr rptr, u32 generation) {
  if (rptr.is_null()) return true;
  const auto header_addr = vamana::StorageLayoutResolver::header(rptr);
  const bool local = local_shard(rptr.memory_node());
  bool locked = false;
  if (local) {
    auto* header_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + header_addr.offset);
    std::atomic_ref<u64> ref(*header_ptr);
    ref.fetch_or(static_cast<u64>(VamanaNode::HEADER_DELETED), std::memory_order_acq_rel);
    lock_node(rptr);
    locked = true;
  } else {
    lock_node(rptr);
    locked = true;
    u64 header = 0;
    remote_read_bytes(rptr.memory_node(), header_addr.offset, &header, sizeof(header), 0);
    header |= static_cast<u64>(VamanaNode::HEADER_DELETED);
    remote_write_bytes(rptr.memory_node(), header_addr.offset, &header, sizeof(header), 0);
  }
  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
  if (local_shard(rptr.memory_node())) {
    byte_t* entry = index_buffer_.get_full_buffer() + hot_offset;
    entry[1] |= VamanaNode::HOT_GRAPH_DELETED;
    vamana::hot_graph::store_u32_le(entry + 4, generation);
    const u16 checksum =
      vamana::hot_graph::checksum16(entry, VamanaNode::hot_graph_entry_size());
    vamana::hot_graph::store_u16_le(entry + 2, checksum);
  } else {
    vec<byte_t> entry(VamanaNode::hot_graph_entry_size(), 0);
    remote_read_bytes(rptr.memory_node(), hot_offset, entry.data(), entry.size(), 0);
    entry[1] |= VamanaNode::HOT_GRAPH_DELETED;
    vamana::hot_graph::store_u32_le(entry.data() + 4, generation);
    const u16 checksum = vamana::hot_graph::checksum16(entry.data(), entry.size());
    vamana::hot_graph::store_u16_le(entry.data() + 2, checksum);
    remote_write_bytes(rptr.memory_node(), hot_offset, entry.data(), entry.size(), 0);
  }
  if (locked) {
    unlock_node(rptr);
  }
  return true;
}

RemotePtr MemoryNode::read_global_medoid() {
  if (storage_id_ == 0) {
    return RemotePtr{*reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8)};
  }

  u64 raw = 0;
  remote_read_bytes(0, 8, &raw, sizeof(raw), 0);
  return RemotePtr{raw};
}

MemoryNode::GlobalMedoidReadAwaitable MemoryNode::async_read_global_medoid(
    StorageOwnerThread& thread) {
  if (storage_id_ == 0) {
    return GlobalMedoidReadAwaitable{true, nullptr, this};
  }
  byte_t* buffer = thread.coroutine_scratch();
  post_peer_read_async(thread, 0, 8, buffer, sizeof(u64));
  return GlobalMedoidReadAwaitable{false, buffer, this};
}

void MemoryNode::write_global_medoid(const RemotePtr& medoid) {
  if (storage_id_ == 0) {
    *reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8) = medoid.raw_address;
    return;
  }
  remote_write_bytes(0, 8, &medoid.raw_address, sizeof(medoid.raw_address), 0);
}

bool MemoryNode::try_set_global_medoid(const RemotePtr& expected, const RemotePtr& desired, RemotePtr& observed) {
  if (storage_id_ == 0) {
    auto* slot = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8);
    std::atomic_ref<u64> ref(*slot);
    u64 current = expected.raw_address;
    const bool ok =
      ref.compare_exchange_strong(current, desired.raw_address, std::memory_order_acq_rel, std::memory_order_acquire);
    observed = RemotePtr{current};
    return ok;
  }

  const u64 original = remote_compare_and_swap(0, 8, expected.raw_address, desired.raw_address, 0);
  observed = RemotePtr{original};
  return original == expected.raw_address;
}
