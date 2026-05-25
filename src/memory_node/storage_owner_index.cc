#include "memory_node/memory_node.hh"

#include <algorithm>
#include <iostream>

namespace {

using Configuration = configuration::IndexConfiguration;
using NodeSnapshot = memory_node_detail::NodeSnapshot;

size_t aligned_snapshot_bytes() {
  size_t value = VamanaNode::size_until_vector_end();
  while (value % CACHELINE_SIZE != 0) {
    ++value;
  }
  return value;
}

u32 storage_owner_construction_width(const Configuration& config) {
  const u32 configured = config.storage_owner_construction_beam_width == 0
                           ? config.beam_width_construction
                           : config.storage_owner_construction_beam_width;
  return std::max<u32>(1, std::min(config.beam_width_construction, configured));
}

u32 storage_owner_snapshot_batch_size(const Configuration& config) {
  return std::max<u32>(1, config.storage_owner_search_snapshot_batch);
}

u32 storage_owner_prune_candidate_limit(const Configuration& config) {
  if (config.storage_owner_prune_max_candidates == 0) {
    return std::numeric_limits<u32>::max();
  }
  return std::max(config.R, config.storage_owner_prune_max_candidates);
}

void parse_node_snapshot(RemotePtr rptr, const byte_t* ptr, NodeSnapshot& snapshot) {
  snapshot = NodeSnapshot{};
  snapshot.rptr = rptr;
  snapshot.components.resize(VamanaNode::DIM);
  snapshot.header = *reinterpret_cast<const u64*>(ptr);
  snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
  snapshot.edge_count = *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
  std::memcpy(snapshot.components.data(), ptr + VamanaNode::offset_vector(), VamanaNode::DIM * sizeof(element_t));
}

}  // namespace

RemotePtr MemoryNode::allocate_local_node() {
  size_t node_size = VamanaNode::total_size();
  while (node_size % 8 != 0) {
    node_size += 4;
  }

  auto* free_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer());
  std::atomic_ref<u64> alloc_ref(*free_ptr);
  const u64 offset = alloc_ref.fetch_add(node_size, std::memory_order_acq_rel);
  lib_assert(offset + node_size <= mn_memory_bytes_, "storage node out of memory");
  return RemotePtr{storage_id_, offset};
}

RemotePtr MemoryNode::read_global_medoid() {
  if (storage_id_ == 0) {
    return RemotePtr{*reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8)};
  }

  u64 raw = 0;
  remote_read_bytes(0, 8, &raw, sizeof(raw), 0);
  return RemotePtr{raw};
}

auto MemoryNode::async_read_global_medoid(StorageOwnerThread& thread) {
  struct Awaitable {
    bool ready{};
    byte_t* buffer{};
    MemoryNode* node{};

    bool await_ready() const { return ready; }
    static void await_suspend(std::coroutine_handle<>) {}
    RemotePtr await_resume() const {
      if (node->storage_id_ == 0) {
        return RemotePtr{*reinterpret_cast<u64*>(node->index_buffer_.get_full_buffer() + 8)};
      }
      return RemotePtr{*reinterpret_cast<const u64*>(buffer)};
    }
  };

  if (storage_id_ == 0) {
    return Awaitable{true, nullptr, this};
  }
  byte_t* buffer = thread.coroutine_scratch();
  post_peer_read_async(thread, 0, 8, buffer, sizeof(u64));
  return Awaitable{false, buffer, this};
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

bool MemoryNode::read_node_snapshot(RemotePtr rptr, NodeSnapshot& snapshot) {
  lib_assert(rptr.memory_node() < num_storage_nodes_,
             "invalid remote shard id in read_node_snapshot: " + std::to_string(rptr.memory_node()));
  lib_assert(rptr.byte_offset() + VamanaNode::size_until_vector_end() <= mn_memory_bytes_,
             "node snapshot read exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
               " offset=" + std::to_string(rptr.byte_offset()) +
               " size=" + std::to_string(VamanaNode::size_until_vector_end()) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  snapshot = NodeSnapshot{};
  snapshot.rptr = rptr;
  snapshot.components.resize(VamanaNode::DIM);

  if (current_storage_owner_thread_ != nullptr &&
      current_storage_owner_thread_->cache.lookup_snapshot(rptr, snapshot)) {
    return true;
  }

  const size_t read_size = VamanaNode::size_until_vector_end();
  if (local_shard(rptr.memory_node())) {
    const byte_t* ptr = local_node_ptr(rptr);
    snapshot.header = *reinterpret_cast<const u64*>(ptr);
    snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
    snapshot.edge_count = *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
    std::memcpy(snapshot.components.data(), ptr + VamanaNode::offset_vector(), VamanaNode::DIM * sizeof(element_t));
    if (current_storage_owner_thread_ != nullptr) {
      current_storage_owner_thread_->cache.insert_snapshot(snapshot);
    }
    return true;
  }

  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                          ? owner_thread->scratch_buffer.get_full_buffer()
                          : peer_scratch_buffer_.get_full_buffer();
  remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), read_buffer, read_size, 0);
  const byte_t* ptr = read_buffer;
  snapshot.header = *reinterpret_cast<const u64*>(ptr);
  snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
  snapshot.edge_count = *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
  std::memcpy(snapshot.components.data(), ptr + VamanaNode::offset_vector(), VamanaNode::DIM * sizeof(element_t));
  if (current_storage_owner_thread_ != nullptr) {
    current_storage_owner_thread_->cache.insert_snapshot(snapshot);
  }
  return true;
}

vec<RemotePtr> MemoryNode::read_neighbor_list(RemotePtr rptr) {
  lib_assert(rptr.memory_node() < num_storage_nodes_,
             "invalid remote shard id in read_neighbor_list: " + std::to_string(rptr.memory_node()));
  lib_assert(rptr.byte_offset() + VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE <= mn_memory_bytes_,
             "neighbor-list read exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
               " offset=" + std::to_string(rptr.byte_offset()) +
               " size=" + std::to_string(VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  vec<RemotePtr> neighbors;
  if (current_storage_owner_thread_ != nullptr &&
      current_storage_owner_thread_->cache.lookup_neighbors(rptr, neighbors)) {
    return neighbors;
  }

  if (local_shard(rptr.memory_node())) {
    const byte_t* ptr = local_node_ptr(rptr);
    const u8 edge_count = *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
    const auto* slots = reinterpret_cast<const RemotePtr*>(ptr + VamanaNode::offset_neighbors());
    neighbors.reserve(edge_count);
    for (u32 i = 0; i < edge_count; ++i) {
      if (!slots[i].is_null()) {
        neighbors.push_back(slots[i]);
      }
    }
    if (current_storage_owner_thread_ != nullptr) {
      current_storage_owner_thread_->cache.insert_neighbors(rptr, neighbors);
    }
    return neighbors;
  }

  u8 edge_count = 0;
  remote_read_bytes(rptr.memory_node(), rptr.byte_offset() + VamanaNode::offset_edge_count(), &edge_count, sizeof(edge_count), 0);
  vec<RemotePtr> slots(VamanaNode::R);
  remote_read_bytes(rptr.memory_node(),
                    rptr.byte_offset() + VamanaNode::offset_neighbors(),
                    slots.data(),
                    VamanaNode::NEIGHBORS_SIZE,
                    align_up(sizeof(u64)));
  neighbors.reserve(edge_count);
  for (u32 i = 0; i < edge_count && i < slots.size(); ++i) {
    if (!slots[i].is_null()) {
      neighbors.push_back(slots[i]);
    }
  }
  if (current_storage_owner_thread_ != nullptr) {
    current_storage_owner_thread_->cache.insert_neighbors(rptr, neighbors);
  }
  return neighbors;
}

auto MemoryNode::async_read_node_snapshot(RemotePtr rptr, StorageOwnerThread& thread) {
  struct Awaitable {
    bool ready{};
    RemotePtr rptr;
    byte_t* buffer{};
    NodeSnapshot snapshot;
    MemoryNode* node{};
    StorageOwnerThread* thread{};

    bool await_ready() const { return ready; }
    static void await_suspend(std::coroutine_handle<>) {}
    NodeSnapshot await_resume() {
      if (ready) {
        return std::move(snapshot);
      }
      snapshot = NodeSnapshot{};
      snapshot.rptr = rptr;
      snapshot.components.resize(VamanaNode::DIM);
      snapshot.header = *reinterpret_cast<const u64*>(buffer);
      snapshot.id = *reinterpret_cast<const u32*>(buffer + VamanaNode::offset_id());
      snapshot.edge_count = *reinterpret_cast<const u8*>(buffer + VamanaNode::offset_edge_count());
      std::memcpy(snapshot.components.data(), buffer + VamanaNode::offset_vector(), VamanaNode::DIM * sizeof(element_t));
      thread->cache.insert_snapshot(snapshot);
      return std::move(snapshot);
    }
  };

  NodeSnapshot cached;
  if (thread.cache.lookup_snapshot(rptr, cached)) {
    return Awaitable{true, rptr, nullptr, std::move(cached), this, &thread};
  }

  if (local_shard(rptr.memory_node())) {
    NodeSnapshot snapshot;
    read_node_snapshot(rptr, snapshot);
    return Awaitable{true, rptr, nullptr, std::move(snapshot), this, &thread};
  }

  byte_t* buffer = thread.coroutine_scratch();
  post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer, VamanaNode::size_until_vector_end());
  return Awaitable{false, rptr, buffer, {}, this, &thread};
}

auto MemoryNode::async_read_node_snapshots(const vec<RemotePtr>& rptrs,
                                           const Configuration& config,
                                           StorageOwnerThread& thread) {
  struct PendingRead {
    RemotePtr rptr;
    byte_t* buffer{};
  };

  struct Awaitable {
    bool ready{true};
    vec<NodeSnapshot> snapshots;
    vec<PendingRead> pending;
    StorageOwnerThread* thread{};

    bool await_ready() const { return ready; }
    static void await_suspend(std::coroutine_handle<>) {}
    vec<NodeSnapshot> await_resume() {
      for (const PendingRead& read : pending) {
        NodeSnapshot snapshot;
        parse_node_snapshot(read.rptr, read.buffer, snapshot);
        thread->cache.insert_snapshot(snapshot);
        snapshots.push_back(std::move(snapshot));
      }
      return std::move(snapshots);
    }
  };

  Awaitable awaitable;
  awaitable.snapshots.reserve(rptrs.size());
  awaitable.pending.reserve(rptrs.size());
  awaitable.thread = &thread;

  const size_t snapshot_size = VamanaNode::size_until_vector_end();
  const size_t snapshot_stride = aligned_snapshot_bytes();
  const u32 max_batch = storage_owner_snapshot_batch_size(config);
  lib_assert(rptrs.size() <= max_batch, "storage-owner snapshot batch exceeds configured limit");

  u32 remote_slot = 0;
  for (const RemotePtr& rptr : rptrs) {
    if (rptr.is_null()) {
      continue;
    }

    NodeSnapshot snapshot;
    if (thread.cache.lookup_snapshot(rptr, snapshot)) {
      awaitable.snapshots.push_back(std::move(snapshot));
      continue;
    }

    if (local_shard(rptr.memory_node())) {
      read_node_snapshot(rptr, snapshot);
      awaitable.snapshots.push_back(std::move(snapshot));
      continue;
    }

    const size_t scratch_offset = static_cast<size_t>(remote_slot) * snapshot_stride;
    lib_assert(scratch_offset + snapshot_size <= thread.scratch_stride,
               "storage-owner coroutine scratch stride is too small for snapshot batch");
    byte_t* buffer = thread.coroutine_scratch(scratch_offset);
    post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer, snapshot_size);
    awaitable.pending.push_back(PendingRead{rptr, buffer});
    awaitable.ready = false;
    ++remote_slot;
  }

  return awaitable;
}

vec<MemoryNode::NodeSnapshot> MemoryNode::read_node_snapshots_batched(const vec<RemotePtr>& rptrs,
                                                                      const Configuration& config) {
  vec<NodeSnapshot> snapshots;
  snapshots.reserve(rptrs.size());
  if (rptrs.empty()) {
    return snapshots;
  }

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    for (const RemotePtr& rptr : rptrs) {
      NodeSnapshot snapshot;
      if (!rptr.is_null() && read_node_snapshot(rptr, snapshot)) {
        snapshots.push_back(std::move(snapshot));
      }
    }
    return snapshots;
  }

  struct PendingRead {
    RemotePtr rptr;
    byte_t* buffer{};
  };

  const size_t snapshot_size = VamanaNode::size_until_vector_end();
  const size_t snapshot_stride = aligned_snapshot_bytes();
  const size_t max_batch = storage_owner_snapshot_batch_size(config);

  for (size_t begin = 0; begin < rptrs.size(); begin += max_batch) {
    const size_t end = std::min(rptrs.size(), begin + max_batch);
    vec<PendingRead> pending;
    pending.reserve(end - begin);
    u32 remote_slot = 0;

    for (size_t idx = begin; idx < end; ++idx) {
      const RemotePtr& rptr = rptrs[idx];
      if (rptr.is_null()) {
        continue;
      }

      NodeSnapshot snapshot;
      if (thread->cache.lookup_snapshot(rptr, snapshot)) {
        snapshots.push_back(std::move(snapshot));
        continue;
      }

      if (local_shard(rptr.memory_node())) {
        read_node_snapshot(rptr, snapshot);
        snapshots.push_back(std::move(snapshot));
        continue;
      }

      const size_t scratch_offset = static_cast<size_t>(remote_slot) * snapshot_stride;
      lib_assert(scratch_offset + snapshot_size <= thread->scratch_stride,
                 "storage-owner coroutine scratch stride is too small for snapshot batch");
      byte_t* buffer = thread->coroutine_scratch(scratch_offset);
      post_peer_read_async(*thread, rptr.memory_node(), rptr.byte_offset(), buffer, snapshot_size);
      pending.push_back(PendingRead{rptr, buffer});
      ++remote_slot;
    }

    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }

    for (const PendingRead& read : pending) {
      NodeSnapshot snapshot;
      parse_node_snapshot(read.rptr, read.buffer, snapshot);
      thread->cache.insert_snapshot(snapshot);
      snapshots.push_back(std::move(snapshot));
    }
  }

  return snapshots;
}

auto MemoryNode::async_read_neighbor_list(RemotePtr rptr, StorageOwnerThread& thread) {
  struct Awaitable {
    bool ready{};
    RemotePtr rptr;
    byte_t* buffer{};
    vec<RemotePtr> neighbors;
    MemoryNode* node{};
    StorageOwnerThread* thread{};

    bool await_ready() const { return ready; }
    static void await_suspend(std::coroutine_handle<>) {}
    vec<RemotePtr> await_resume() {
      if (ready) {
        return std::move(neighbors);
      }
      const u8 edge_count = *reinterpret_cast<const u8*>(buffer);
      const auto* slots = reinterpret_cast<const RemotePtr*>(buffer + align_up(sizeof(u8)));
      neighbors.reserve(edge_count);
      for (u32 i = 0; i < edge_count && i < VamanaNode::R; ++i) {
        if (!slots[i].is_null()) {
          neighbors.push_back(slots[i]);
        }
      }
      thread->cache.insert_neighbors(rptr, neighbors);
      return std::move(neighbors);
    }
  };

  vec<RemotePtr> cached;
  if (thread.cache.lookup_neighbors(rptr, cached)) {
    return Awaitable{true, rptr, nullptr, std::move(cached), this, &thread};
  }

  if (local_shard(rptr.memory_node())) {
    vec<RemotePtr> neighbors = read_neighbor_list(rptr);
    return Awaitable{true, rptr, nullptr, std::move(neighbors), this, &thread};
  }

  byte_t* buffer = thread.coroutine_scratch();
  post_peer_read_async(thread,
                       rptr.memory_node(),
                       rptr.byte_offset() + VamanaNode::offset_edge_count(),
                       buffer,
                       sizeof(u8));
  post_peer_read_async(thread,
                       rptr.memory_node(),
                       rptr.byte_offset() + VamanaNode::offset_neighbors(),
                       buffer,
                       VamanaNode::NEIGHBORS_SIZE,
                       align_up(sizeof(u8)));
  return Awaitable{false, rptr, buffer, {}, this, &thread};
}

void MemoryNode::write_neighbor_list(RemotePtr rptr, const vec<RemotePtr>& neighbors) {
  lib_assert(rptr.memory_node() < num_storage_nodes_,
             "invalid remote shard id in write_neighbor_list: " + std::to_string(rptr.memory_node()));
  lib_assert(rptr.byte_offset() + VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE <= mn_memory_bytes_,
             "neighbor-list write exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
               " offset=" + std::to_string(rptr.byte_offset()) +
               " size=" + std::to_string(VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  const u8 edge_count = static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
  if (local_shard(rptr.memory_node())) {
    byte_t* ptr = local_node_ptr(rptr);
    *reinterpret_cast<u8*>(ptr + VamanaNode::offset_edge_count()) = edge_count;
    std::memset(ptr + VamanaNode::offset_edge_count() + sizeof(u8), 0, VamanaNode::PADDING_SIZE);
    auto* slots = reinterpret_cast<RemotePtr*>(ptr + VamanaNode::offset_neighbors());
    for (u32 i = 0; i < edge_count; ++i) {
      slots[i] = neighbors[i];
    }
    for (u32 i = edge_count; i < VamanaNode::R; ++i) {
      slots[i].reset();
    }
    invalidate_storage_owner_cache(rptr);
    return;
  }

  byte_t meta[sizeof(u8) + VamanaNode::PADDING_SIZE]{};
  meta[0] = edge_count;
  remote_write_bytes(rptr.memory_node(), rptr.byte_offset() + VamanaNode::offset_edge_count(), meta, sizeof(meta), 0);

  vec<RemotePtr> slots(VamanaNode::R);
  for (u32 i = 0; i < edge_count; ++i) {
    slots[i] = neighbors[i];
  }
  remote_write_bytes(rptr.memory_node(),
                     rptr.byte_offset() + VamanaNode::offset_neighbors(),
                     slots.data(),
                     VamanaNode::NEIGHBORS_SIZE,
                     align_up(sizeof(meta)));
  invalidate_storage_owner_cache(rptr);
}

void MemoryNode::write_new_node(RemotePtr rptr,
                    node_t id,
                    const span<const element_t> components,
                    const vec<byte_t>& rabitq_data,
                    const vec<RemotePtr>& neighbors) {
  byte_t* ptr = local_node_ptr(rptr);
  std::memset(ptr, 0, VamanaNode::total_size());
  *reinterpret_cast<u64*>(ptr) = 0;
  *reinterpret_cast<u32*>(ptr + VamanaNode::offset_id()) = id;
  *reinterpret_cast<u8*>(ptr + VamanaNode::offset_edge_count()) = static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
  std::memcpy(ptr + VamanaNode::offset_vector(), components.data(), VamanaNode::DIM * sizeof(element_t));
  if (!rabitq_data.empty()) {
    std::memcpy(ptr + VamanaNode::offset_rabitq(), rabitq_data.data(), std::min<size_t>(rabitq_data.size(), VamanaNode::RABITQ_SIZE));
  }
  auto* slots = reinterpret_cast<RemotePtr*>(ptr + VamanaNode::offset_neighbors());
  for (u32 i = 0; i < neighbors.size() && i < VamanaNode::R; ++i) {
    slots[i] = neighbors[i];
  }
  invalidate_storage_owner_cache(rptr);
}

void MemoryNode::lock_node(RemotePtr rptr) {
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(local_node_ptr(rptr));
    std::atomic_ref<u64> ref(*header_ptr);
    for (;;) {
      u64 header = ref.load(std::memory_order_acquire);
      if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
        std::this_thread::yield();
        continue;
      }
      const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
      if (ref.compare_exchange_weak(header, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
      }
    }
  }

  for (;;) {
    auto [success, header] = try_lock_remote_header(rptr);
    if (success) {
      return;
    }
    if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
      std::this_thread::yield();
    }
  }
}

void MemoryNode::unlock_node(RemotePtr rptr) {
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(local_node_ptr(rptr));
    std::atomic_ref<u64> ref(*header_ptr);
    ref.fetch_and(~static_cast<u64>(VamanaNode::HEADER_NODE_LOCK), std::memory_order_acq_rel);
    return;
  }

  const byte_t unlock = 0;
  remote_write_bytes(rptr.memory_node(), rptr.byte_offset() + VamanaNode::HEADER_UNTIL_LOCK, &unlock, 1, 0);
}

vec<RemotePtr> MemoryNode::beam_search_candidates(const span<const element_t> query,
                                                  RemotePtr medoid,
                                                  const Configuration& config,
                                                  InsertBreakdownCounters* breakdown) {
  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;

  NodeSnapshot medoid_snapshot;
  auto t_snapshot = std::chrono::steady_clock::now();
  read_node_snapshot(medoid, medoid_snapshot);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
  }
  auto t_distance = std::chrono::steady_clock::now();
  const distance_t medoid_dist = distance_fn()(query, medoid_snapshot.components, config.dim);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
  }
  beam.push_back({medoid, medoid_dist, false});
  visited.insert(medoid);

  for (;;) {
    i32 best_idx = -1;
    distance_t best_dist = std::numeric_limits<distance_t>::max();
    auto t_select = std::chrono::steady_clock::now();
    for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
      if (!beam[i].expanded && beam[i].distance < best_dist) {
        best_dist = beam[i].distance;
        best_idx = i;
      }
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_select_ns += elapsed_ns_since(t_select);
    }
    if (best_idx < 0) {
      break;
    }

    beam[best_idx].expanded = true;
    auto t_neighbor_read = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = read_neighbor_list(beam[best_idx].rptr);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(t_neighbor_read);
    }
    vec<RemotePtr> unvisited_neighbors;
    unvisited_neighbors.reserve(neighbors.size());
    for (const RemotePtr& neighbor : neighbors) {
      if (neighbor.is_null() || visited.contains(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      unvisited_neighbors.push_back(neighbor);
    }

    const u32 snapshot_batch = storage_owner_snapshot_batch_size(config);
    const u32 construction_width = storage_owner_construction_width(config);
    for (size_t begin = 0; begin < unvisited_neighbors.size(); begin += snapshot_batch) {
      const size_t end = std::min(unvisited_neighbors.size(), begin + snapshot_batch);
      vec<RemotePtr> batch;
      batch.reserve(end - begin);
      batch.insert(batch.end(), unvisited_neighbors.begin() + begin, unvisited_neighbors.begin() + end);
      t_snapshot = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        t_distance = std::chrono::steady_clock::now();
        const distance_t dist = distance_fn()(query, snapshot.components, config.dim);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
        }
        auto t_beam_update = std::chrono::steady_clock::now();
        insert_into_beam(beam, snapshot.rptr, dist, construction_width);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
        }
      }
    }
  }

  vec<RemotePtr> candidates;
  candidates.reserve(beam.size());
  auto t_sort = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) { return lhs.distance < rhs.distance; });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
  }
  for (const auto& entry : beam) {
    candidates.push_back(entry.rptr);
  }
  return candidates;
}

auto MemoryNode::beam_search_candidates_async(const span<const element_t> query,
                                              RemotePtr medoid,
                                              const Configuration& config,
                                              StorageOwnerThread& thread,
                                              InsertBreakdownCounters* breakdown) -> StorageOwnerInsertCoroutine {
  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;

  auto t_snapshot = std::chrono::steady_clock::now();
  NodeSnapshot medoid_snapshot = co_await async_read_node_snapshot(medoid, thread);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
  }
  auto t_distance = std::chrono::steady_clock::now();
  const distance_t medoid_dist = distance_fn()(query, medoid_snapshot.components, config.dim);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
  }
  beam.push_back({medoid, medoid_dist, false});
  visited.insert(medoid);

  for (;;) {
    i32 best_idx = -1;
    distance_t best_dist = std::numeric_limits<distance_t>::max();
    auto t_select = std::chrono::steady_clock::now();
    for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
      if (!beam[i].expanded && beam[i].distance < best_dist) {
        best_dist = beam[i].distance;
        best_idx = i;
      }
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_select_ns += elapsed_ns_since(t_select);
    }
    if (best_idx < 0) {
      break;
    }

    beam[best_idx].expanded = true;
    auto t_neighbor_read = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = co_await async_read_neighbor_list(beam[best_idx].rptr, thread);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(t_neighbor_read);
    }
    vec<RemotePtr> unvisited_neighbors;
    unvisited_neighbors.reserve(neighbors.size());
    for (const RemotePtr& neighbor : neighbors) {
      if (neighbor.is_null() || visited.contains(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      unvisited_neighbors.push_back(neighbor);
    }

    const u32 snapshot_batch = storage_owner_snapshot_batch_size(config);
    const u32 construction_width = storage_owner_construction_width(config);
    for (size_t begin = 0; begin < unvisited_neighbors.size(); begin += snapshot_batch) {
      const size_t end = std::min(unvisited_neighbors.size(), begin + snapshot_batch);
      vec<RemotePtr> batch;
      batch.reserve(end - begin);
      batch.insert(batch.end(), unvisited_neighbors.begin() + begin, unvisited_neighbors.begin() + end);
      t_snapshot = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = co_await async_read_node_snapshots(batch, config, thread);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        t_distance = std::chrono::steady_clock::now();
        const distance_t dist = distance_fn()(query, snapshot.components, config.dim);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
        }
        auto t_beam_update = std::chrono::steady_clock::now();
        insert_into_beam(beam, snapshot.rptr, dist, construction_width);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
        }
      }
    }
  }

  auto& out = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  out.clear();
  out.reserve(beam.size());
  auto t_sort = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) { return lhs.distance < rhs.distance; });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
  }
  for (const auto& entry : beam) {
    out.push_back(entry.rptr);
  }
}

vec<RemotePtr> MemoryNode::robust_prune_cpu(const span<const element_t> source,
                                            const vec<RemotePtr>& candidates,
                                            const hashset_t<RemotePtr>& skip,
                                            const Configuration& config,
                                            InsertBreakdownCounters* breakdown) {
  struct CandidateInfo {
    RemotePtr rptr;
    distance_t dist{};
    vec<element_t> components;
  };

  vec<CandidateInfo> infos;
  infos.reserve(candidates.size());
  vec<RemotePtr> filtered;
  filtered.reserve(std::min<size_t>(candidates.size(), storage_owner_prune_candidate_limit(config)));
  const u32 prune_candidate_limit = storage_owner_prune_candidate_limit(config);
  for (const RemotePtr& candidate : candidates) {
    if (candidate.is_null() || skip.contains(candidate)) {
      continue;
    }
    filtered.push_back(candidate);
    if (filtered.size() >= prune_candidate_limit) {
      break;
    }
  }

  const u32 snapshot_batch = storage_owner_snapshot_batch_size(config);
  for (size_t begin = 0; begin < filtered.size(); begin += snapshot_batch) {
    const size_t end = std::min(filtered.size(), begin + snapshot_batch);
    vec<RemotePtr> batch;
    batch.reserve(end - begin);
    batch.insert(batch.end(), filtered.begin() + begin, filtered.begin() + end);
    auto t_snapshot = std::chrono::steady_clock::now();
    vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_snapshot);
    }
    for (NodeSnapshot& snapshot : snapshots) {
      auto t_distance = std::chrono::steady_clock::now();
      const distance_t dist = distance_fn()(source, snapshot.components, config.dim);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_distance_ns += elapsed_ns_since(t_distance);
      }
      infos.push_back({snapshot.rptr, dist, std::move(snapshot.components)});
    }
  }

  auto t_sort = std::chrono::steady_clock::now();
  std::sort(infos.begin(), infos.end(), [](const CandidateInfo& lhs, const CandidateInfo& rhs) {
    return lhs.dist < rhs.dist;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_prune_sort_ns += elapsed_ns_since(t_sort);
  }

  vec<RemotePtr> selected;
  selected.reserve(config.R);
  vec<span<const element_t>> selected_components;
  selected_components.reserve(config.R);

  for (const auto& candidate : infos) {
    if (selected.size() >= config.R) {
      break;
    }

    bool pruned = false;
    for (idx_t i = 0; i < selected_components.size(); ++i) {
      auto t_pair_distance = std::chrono::steady_clock::now();
      const distance_t pair_dist = distance_fn()(candidate.components, selected_components[i], config.dim);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_pair_distance_ns += elapsed_ns_since(t_pair_distance);
      }
      if (config.alpha * pair_dist <= candidate.dist) {
        pruned = true;
        break;
      }
    }

    if (!pruned) {
      selected.push_back(candidate.rptr);
      selected_components.push_back(candidate.components);
    }
  }

  return selected;
}

auto MemoryNode::execute_storage_owner_insert_job_async(StorageOwnerThread& thread,
                                            StorageOwnerInsertJob& job,
                                            std::unordered_map<u64, vec<RemotePtr>>& local_updates,
                                            std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>>& remote_updates,
                                            InsertBreakdownCounters& breakdown,
                                            const Configuration& config) -> StorageOwnerInsertCoroutine {
  const auto components = span<const element_t>{job.components.data(), VamanaNode::DIM};
  auto t_quantize = std::chrono::steady_clock::now();
  const vec<byte_t> rabitq_data = quantize_rabitq_cpu(components, config);
  breakdown.storage_owner_quantize_ns += elapsed_ns_since(t_quantize);

  auto t_medoid = std::chrono::steady_clock::now();
  RemotePtr medoid_ptr = co_await async_read_global_medoid(thread);
  breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
  if (medoid_ptr.is_null()) {
    const RemotePtr new_ptr = allocate_local_node();
    auto t_write = std::chrono::steady_clock::now();
    write_new_node(new_ptr, job.id, components, rabitq_data, {});
    breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
    RemotePtr observed;
    if (try_set_global_medoid(RemotePtr{}, new_ptr, observed) || observed.is_null()) {
      job.ok = true;
      co_return;
    }
    medoid_ptr = observed;
  }

  auto t_search = std::chrono::steady_clock::now();
  auto search = beam_search_candidates_async(components, medoid_ptr, config, thread, &breakdown);
  co_await std::suspend_always{};
  while (!search.handle.done()) {
    if (thread.is_ready(thread.running_coroutine)) {
      search.handle.resume();
    } else {
      co_await std::suspend_always{};
    }
  }
  search.handle.destroy();
  breakdown.storage_owner_search_ns += elapsed_ns_since(t_search);

  const vec<RemotePtr>& candidates = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  hashset_t<RemotePtr> empty_skip;
  auto t_prune = std::chrono::steady_clock::now();
  vec<RemotePtr> selected_neighbors = robust_prune_cpu(components, candidates, empty_skip, config, &breakdown);
  breakdown.storage_owner_prune_ns += elapsed_ns_since(t_prune);
  const RemotePtr new_ptr = allocate_local_node();
  auto t_write = std::chrono::steady_clock::now();
  write_new_node(new_ptr, job.id, components, rabitq_data, selected_neighbors);
  breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);

  for (const RemotePtr& neighbor_ptr : selected_neighbors) {
    if (local_shard(neighbor_ptr.memory_node())) {
      local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
    } else {
      remote_updates[neighbor_ptr.memory_node()].push_back(
        service::storage_owner::ReverseUpdateOp{neighbor_ptr.raw_address, new_ptr.raw_address});
    }
  }
  job.ok = true;
}

vec<byte_t> MemoryNode::quantize_rabitq_cpu(const span<const element_t> components, const Configuration& config) const {
  vec<byte_t> output(VamanaNode::RABITQ_SIZE, 0);
  if (!config.use_rabitq_search() || !rabitq_artifacts_ready_) {
    return output;
  }

  const u32 dim = config.dim;
  const u32 bits = config.rabitq_bits;
  const u32 packed_bytes = (bits * dim + 7) / 8;
  vec<float> rotated(dim, 0.0f);
  for (u32 col = 0; col < dim; ++col) {
    float sum = 0.0f;
    for (u32 row = 0; row < dim; ++row) {
      sum += rabitq_artifacts_.rotation_matrix[row + static_cast<size_t>(col) * dim] * components[row];
    }
    rotated[col] = sum;
  }

  constexpr double kEps = 1e-5;
  vec<uint8_t> uncompressed(dim, 0);
  float l2_sqr = 0.0f;
  for (u32 j = 0; j < dim; ++j) {
    const float diff = rotated[j] - rabitq_artifacts_.rotated_centroid[j];
    l2_sqr += diff * diff;
  }
  const float l2_norm = std::sqrt(std::max(l2_sqr, 1e-12f));

  float ip_norm = 0.0f;
  for (u32 j = 0; j < dim; ++j) {
    const float abs_o = std::fabs((rotated[j] - rabitq_artifacts_.rotated_centroid[j]) / l2_norm);
    int val = static_cast<int>((rabitq_artifacts_.t_const * abs_o) + kEps);
    if (val >= (1 << (bits - 1))) {
      val = (1 << (bits - 1)) - 1;
    }
    uncompressed[j] = static_cast<uint8_t>(val);
    ip_norm += (static_cast<float>(val) + 0.5f) * abs_o;
  }
  const float ip_norm_inv = ip_norm == 0.0f ? 1.0f : (1.0f / ip_norm);

  const uint32_t mask = (1u << (bits - 1)) - 1u;
  for (u32 j = 0; j < dim; ++j) {
    const float residual = rotated[j] - rabitq_artifacts_.rotated_centroid[j];
    if (residual >= 0.0f) {
      uncompressed[j] = static_cast<uint8_t>(uncompressed[j] + (1u << (bits - 1)));
    } else {
      uncompressed[j] = static_cast<uint8_t>((~uncompressed[j]) & mask);
    }
  }

  const float cb = -(static_cast<float>(1 << (bits - 1)) - 0.5f);
  float ip_resi_xucb = 0.0f;
  float ip_cent_xucb = 0.0f;
  for (u32 j = 0; j < dim; ++j) {
    const float residual = rotated[j] - rabitq_artifacts_.rotated_centroid[j];
    const float xu_cb = static_cast<float>(uncompressed[j]) + cb;
    ip_resi_xucb += residual * xu_cb;
    ip_cent_xucb += rabitq_artifacts_.rotated_centroid[j] * xu_cb;
  }
  if (ip_resi_xucb == 0.0f) {
    ip_resi_xucb = FLT_MAX;
  }

  const float add = l2_sqr + 2.0f * l2_sqr * ip_cent_xucb / ip_resi_xucb;
  const float rescale = ip_norm_inv * -2.0f * l2_norm;

  const u32 values_per_byte = 8 / bits;
  for (u32 byte = 0; byte < packed_bytes; ++byte) {
    uint8_t packed = 0;
    for (u32 k = 0; k < values_per_byte; ++k) {
      const u32 dim_idx = byte * values_per_byte + k;
      if (dim_idx < dim) {
        packed |= static_cast<uint8_t>(uncompressed[dim_idx] << (k * bits));
      }
    }
    output[byte] = packed;
  }
  std::memcpy(output.data() + packed_bytes, &add, sizeof(float));
  std::memcpy(output.data() + packed_bytes + sizeof(float), &rescale, sizeof(float));
  return output;
}

bool MemoryNode::apply_local_reverse_update(RemotePtr target_ptr,
                                const vec<RemotePtr>& candidate_ptrs,
                                const Configuration& config) {
  lib_assert(local_shard(target_ptr.memory_node()), "target reverse update must be local");
  if (candidate_ptrs.empty()) {
    return true;
  }

  const auto update_started = std::chrono::steady_clock::now();
  const auto lock_started = std::chrono::steady_clock::now();
  lock_node(target_ptr);
  const u64 lock_wait_ns = elapsed_ns_since(lock_started);
  vec<RemotePtr> updated_neighbors;
  bool changed = false;
  bool pruned = false;
  size_t current_count = 0;
  size_t filtered_count = 0;
  u64 snapshot_ns = 0;
  u64 neighbor_read_ns = 0;
  u64 filter_ns = 0;
  u64 prune_ns = 0;
  u64 write_ns = 0;

  NodeSnapshot target_snapshot;
  auto step_started = std::chrono::steady_clock::now();
  read_node_snapshot(target_ptr, target_snapshot);
  snapshot_ns = elapsed_ns_since(step_started);

  step_started = std::chrono::steady_clock::now();
  vec<RemotePtr> current_neighbors = read_neighbor_list(target_ptr);
  neighbor_read_ns = elapsed_ns_since(step_started);
  current_count = current_neighbors.size();

  step_started = std::chrono::steady_clock::now();
  vec<RemotePtr> filtered_candidates;
  filtered_candidates.reserve(candidate_ptrs.size());
  for (const RemotePtr& candidate_ptr : candidate_ptrs) {
    if (candidate_ptr.is_null()) {
      continue;
    }
    bool already_present = false;
    for (const RemotePtr& existing : current_neighbors) {
      if (existing == candidate_ptr) {
        already_present = true;
        break;
      }
    }
    if (!already_present &&
        std::find(filtered_candidates.begin(), filtered_candidates.end(), candidate_ptr) == filtered_candidates.end()) {
      filtered_candidates.push_back(candidate_ptr);
    }
  }
  filter_ns = elapsed_ns_since(step_started);
  filtered_count = filtered_candidates.size();

  if (!filtered_candidates.empty()) {
    changed = true;

    if (current_neighbors.size() + filtered_candidates.size() <= config.R) {
      current_neighbors.insert(current_neighbors.end(), filtered_candidates.begin(), filtered_candidates.end());
      updated_neighbors = std::move(current_neighbors);
    } else {
      pruned = true;
      vec<RemotePtr> prune_candidates = current_neighbors;
      prune_candidates.insert(prune_candidates.end(), filtered_candidates.begin(), filtered_candidates.end());
      hashset_t<RemotePtr> skip{target_ptr};
      step_started = std::chrono::steady_clock::now();
      updated_neighbors = robust_prune_cpu(target_snapshot.components, prune_candidates, skip, config);
      prune_ns = elapsed_ns_since(step_started);
    }
  }

  if (changed) {
    step_started = std::chrono::steady_clock::now();
    write_neighbor_list(target_ptr, updated_neighbors);
    write_ns = elapsed_ns_since(step_started);
  }
  unlock_node(target_ptr);

  const u64 update_ns = elapsed_ns_since(update_started);
  if (update_ns > 1000ull * 1000ull * 1000ull) {
    static std::atomic<u32> slow_update_logs{0};
    const u32 log_index = slow_update_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] slow reverse-update target"
                << " self_shard=" << storage_id_
                << " target_raw=" << target_ptr.raw_address
                << " candidates=" << candidate_ptrs.size()
                << " current_neighbors=" << current_count
                << " filtered_candidates=" << filtered_count
                << " changed=" << (changed ? 1 : 0)
                << " pruned=" << (pruned ? 1 : 0)
                << " elapsed_ms=" << (update_ns / 1000000.0)
                << " lock_wait_ms=" << (lock_wait_ns / 1000000.0)
                << " snapshot_ms=" << (snapshot_ns / 1000000.0)
                << " neighbor_read_ms=" << (neighbor_read_ns / 1000000.0)
                << " filter_ms=" << (filter_ns / 1000000.0)
                << " prune_ms=" << (prune_ns / 1000000.0)
                << " write_ms=" << (write_ns / 1000000.0)
                << std::endl;
    }
  }
  return true;
}
