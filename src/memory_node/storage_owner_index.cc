#include "memory_node/memory_node.hh"
#include "memory_node/storage_owner_helpers.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "common/index_path.hh"
#include "vamana/idmap.hh"
#include "vamana/storage_layout_resolver.hh"

namespace {

using Configuration = configuration::IndexConfiguration;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;

using memory_node_detail::aligned_snapshot_bytes;
using memory_node_detail::kSnapshotPrefixBytes;
using memory_node_detail::parse_remote_snapshot;
using memory_node_detail::snapshot_buffer_bytes;
using memory_node_detail::storage_owner_snapshot_batch_size;

}  // namespace

bool MemoryNode::GlobalMedoidReadAwaitable::await_ready() const {
  return ready;
}

void MemoryNode::GlobalMedoidReadAwaitable::await_suspend(std::coroutine_handle<>) {}

RemotePtr MemoryNode::GlobalMedoidReadAwaitable::await_resume() const {
  if (ready) {
    return value;
  }
  return RemotePtr{*reinterpret_cast<const u64*>(buffer)};
}

bool MemoryNode::NodeSnapshotReadAwaitable::await_ready() const {
  return ready;
}

void MemoryNode::NodeSnapshotReadAwaitable::await_suspend(std::coroutine_handle<>) {}

MemoryNode::NodeSnapshot MemoryNode::NodeSnapshotReadAwaitable::await_resume() {
  if (ready) {
    return std::move(snapshot);
  }
  parse_remote_snapshot(rptr, buffer, snapshot);
  return std::move(snapshot);
}

bool MemoryNode::NodeSnapshotsReadAwaitable::await_ready() const {
  return ready;
}

void MemoryNode::NodeSnapshotsReadAwaitable::await_suspend(std::coroutine_handle<>) {}

vec<MemoryNode::NodeSnapshot> MemoryNode::NodeSnapshotsReadAwaitable::await_resume() {
  for (const PendingRead& read : pending) {
    NodeSnapshot snapshot;
    parse_remote_snapshot(read.rptr, read.buffer, snapshot);
    snapshots.push_back(std::move(snapshot));
  }
  return std::move(snapshots);
}

bool MemoryNode::NeighborListReadAwaitable::await_ready() const {
  return ready;
}

void MemoryNode::NeighborListReadAwaitable::await_suspend(std::coroutine_handle<>) {}

vec<RemotePtr> MemoryNode::NeighborListReadAwaitable::await_resume() {
  if (ready) {
    return std::move(neighbors);
  }
  vec<byte_t> decoded;
  const byte_t* parse_buffer = buffer;
  if (hot_graph) {
    decoded.resize(VamanaNode::neighbor_read_size());
    const bool ok = VamanaNode::decode_hot_graph_entry(buffer, decoded.data());
    if (!ok) {
      return node->read_neighbor_list(rptr);
    }
    parse_buffer = decoded.data();
  }
  const u8 edge_count = *reinterpret_cast<const u8*>(
    parse_buffer + VamanaNode::neighbor_count_offset_in_read());
  const auto* slots = reinterpret_cast<const RemotePtr*>(
    parse_buffer + VamanaNode::neighbor_payload_offset_in_read());
  neighbors.reserve(edge_count);
  for (u32 i = 0; i < edge_count && i < VamanaNode::R; ++i) {
    if (!slots[i].is_null()) {
      neighbors.push_back(slots[i]);
    }
  }
  return std::move(neighbors);
}

RemotePtr MemoryNode::allocate_local_node() {
  size_t node_size = VamanaNode::allocation_size();
  while (node_size % 8 != 0) {
    node_size += 4;
  }

  auto* free_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer());
  std::atomic_ref<u64> alloc_ref(*free_ptr);
  const u64 offset = alloc_ref.fetch_add(node_size, std::memory_order_acq_rel);
  lib_assert(offset + node_size <= mn_memory_bytes_, "storage node out of memory");
  return RemotePtr{storage_id_, offset};
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
  idmap_.reserve(static_cast<size_t>(header.entry_count));
  for (u64 i = 0; i < header.entry_count; ++i) {
    vamana::idmap::Entry entry;
    input.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    if (!input.good()) return false;
    idmap_[entry.id] = FreshnessEntry{
      RemotePtr{entry.rptr_raw},
      entry.generation,
      (entry.flags & vamana::idmap::kDeleted) != 0};
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

MemoryNode::GlobalMedoidReadAwaitable MemoryNode::async_read_global_medoid(StorageOwnerThread& thread) {
  if (storage_id_ == 0) {
    return GlobalMedoidReadAwaitable{
      true,
      RemotePtr{*reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + 8)},
      nullptr};
  }
  byte_t* buffer = thread.coroutine_scratch();
  post_peer_read_async(thread, 0, 8, buffer, sizeof(u64));
  return GlobalMedoidReadAwaitable{false, {}, buffer};
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
  const auto vector_addr = vamana::StorageLayoutResolver::vector(rptr);
  lib_assert(vector_addr.offset + vector_addr.size <= mn_memory_bytes_,
             "node snapshot read exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
               " offset=" + std::to_string(rptr.byte_offset()) +
               " size=" + std::to_string(vector_addr.size) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  snapshot = NodeSnapshot{};
  snapshot.rptr = rptr;
  snapshot.vector_data.resize(VamanaNode::vector_bytes());

  const size_t read_size = VamanaNode::vector_bytes();
  if (local_shard(rptr.memory_node())) {
    const byte_t* base = index_buffer_.get_full_buffer();
    const byte_t* ptr = base + rptr.byte_offset();
    snapshot.header = *reinterpret_cast<const u64*>(ptr);
    snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
    snapshot.generation = VamanaNode::compact_storage()
      ? *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_generation()) : 0;
    snapshot.edge_count = VamanaNode::compact_storage()
      ? 0 : *reinterpret_cast<const u8*>(ptr + VamanaNode::offset_edge_count());
    snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
    std::memcpy(snapshot.vector_data.data(), base + vector_addr.offset, VamanaNode::vector_bytes());
    return true;
  }

  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                          ? owner_thread->scratch_buffer.get_full_buffer()
                          : peer_scratch_buffer_.get_full_buffer();
  byte_t prefix[VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE]{};
  remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), prefix, sizeof(prefix), 0);
  remote_read_bytes(rptr.memory_node(),
                    vector_addr.offset,
                    read_buffer,
                    read_size,
                    0);
  snapshot.vector_data.resize(VamanaNode::vector_bytes());
  std::memcpy(snapshot.vector_data.data(), read_buffer, VamanaNode::vector_bytes());
  snapshot.header = *reinterpret_cast<const u64*>(prefix);
  snapshot.id = *reinterpret_cast<const u32*>(prefix + VamanaNode::offset_id());
  snapshot.generation = VamanaNode::compact_storage()
    ? *reinterpret_cast<const u32*>(prefix + VamanaNode::offset_generation()) : 0;
  snapshot.edge_count = VamanaNode::compact_storage()
    ? 0 : *reinterpret_cast<const u8*>(prefix + VamanaNode::offset_edge_count());
  snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
  return true;
}

vec<RemotePtr> MemoryNode::read_neighbor_list_aos(RemotePtr rptr) {
  lib_assert(rptr.memory_node() < num_storage_nodes_,
             "invalid remote shard id in read_neighbor_list: " + std::to_string(rptr.memory_node()));
  const auto neighbor_addr = vamana::StorageLayoutResolver::neighbor_slots(rptr);
  lib_assert(neighbor_addr.offset + neighbor_addr.size <= mn_memory_bytes_,
             "neighbor-list read exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
               " offset=" + std::to_string(rptr.byte_offset()) +
               " size=" + std::to_string(neighbor_addr.size) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  vec<RemotePtr> neighbors;
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
    return neighbors;
  }

  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                          ? owner_thread->scratch_buffer.get_full_buffer()
                          : peer_scratch_buffer_.get_full_buffer();
  remote_read_bytes(rptr.memory_node(),
                    vamana::StorageLayoutResolver::neighbor_read(rptr).address.offset,
                    read_buffer,
                    VamanaNode::neighbor_read_size(),
                    0);
  const u8 edge_count = *reinterpret_cast<const u8*>(read_buffer + VamanaNode::neighbor_count_offset_in_read());
  const auto* slots = reinterpret_cast<const RemotePtr*>(read_buffer + VamanaNode::neighbor_payload_offset_in_read());
  neighbors.reserve(edge_count);
  for (u32 i = 0; i < edge_count && i < VamanaNode::R; ++i) {
    if (!slots[i].is_null()) {
      neighbors.push_back(slots[i]);
    }
  }
  return neighbors;
}

vec<RemotePtr> MemoryNode::read_neighbor_list(RemotePtr rptr) {
  if (!VamanaNode::compact_storage()) {
    return read_neighbor_list_aos(rptr);
  }

  vec<byte_t> local_entry;
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = nullptr;
  if (local_shard(rptr.memory_node())) {
    local_entry.resize(VamanaNode::hot_graph_entry_size());
    read_buffer = local_entry.data();
  } else {
    read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                    ? owner_thread->scratch_buffer.get_full_buffer()
                    : peer_scratch_buffer_.get_full_buffer();
  }
  vec<byte_t> decoded(VamanaNode::neighbor_read_size());
  bool decoded_ok = false;
  constexpr u32 kMaxReadAttempts = 3;
  for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
    if (local_shard(rptr.memory_node())) {
      std::memcpy(read_buffer,
                  index_buffer_.get_full_buffer() + VamanaNode::hot_graph_entry_offset(rptr),
                  VamanaNode::hot_graph_entry_size());
    } else {
      remote_read_bytes(rptr.memory_node(),
                        VamanaNode::hot_graph_entry_offset(rptr),
                        read_buffer,
                        VamanaNode::hot_graph_entry_size(),
                        0);
    }
    decoded_ok = VamanaNode::decode_hot_graph_entry(read_buffer, decoded.data());
    if (decoded_ok) {
      break;
    }
    std::this_thread::yield();
  }
  if (!decoded_ok) {
    return {};
  }
  const byte_t* parse_buffer = decoded.data();
  const u8 edge_count = *reinterpret_cast<const u8*>(parse_buffer + VamanaNode::neighbor_count_offset_in_read());
  const auto* slots = reinterpret_cast<const RemotePtr*>(parse_buffer + VamanaNode::neighbor_payload_offset_in_read());
  vec<RemotePtr> neighbors;
  neighbors.reserve(edge_count);
  for (u32 i = 0; i < edge_count && i < VamanaNode::R; ++i) {
    if (!slots[i].is_null()) {
      neighbors.push_back(slots[i]);
    }
  }
  return neighbors;
}

MemoryNode::NodeSnapshotReadAwaitable MemoryNode::async_read_node_snapshot(RemotePtr rptr,
                                                                           StorageOwnerThread& thread) {
  if (local_shard(rptr.memory_node())) {
    NodeSnapshot snapshot;
    read_node_snapshot(rptr, snapshot);
    return NodeSnapshotReadAwaitable{true, rptr, nullptr, std::move(snapshot)};
  }

  byte_t* buffer = thread.coroutine_scratch();
  if (VamanaNode::compact_storage()) {
    post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                         VamanaNode::size_until_vector_end());
  } else {
    post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                         kSnapshotPrefixBytes);
    post_peer_read_async(thread, rptr.memory_node(),
                         vamana::StorageLayoutResolver::vector(rptr).offset,
                         buffer + kSnapshotPrefixBytes, VamanaNode::vector_bytes());
  }
  return NodeSnapshotReadAwaitable{false, rptr, buffer, {}};
}

MemoryNode::NodeSnapshotsReadAwaitable MemoryNode::async_read_node_snapshots(
    const vec<RemotePtr>& rptrs,
    const Configuration& config,
    StorageOwnerThread& thread) {
  NodeSnapshotsReadAwaitable awaitable;
  awaitable.snapshots.reserve(rptrs.size());
  awaitable.pending.reserve(rptrs.size());

  const size_t snapshot_size = snapshot_buffer_bytes();
  const size_t snapshot_stride = aligned_snapshot_bytes();
  const u32 max_batch = storage_owner_snapshot_batch_size(config, &thread);
  lib_assert(rptrs.size() <= max_batch, "storage-owner snapshot batch exceeds configured limit");

  u32 remote_slot = 0;
  for (const RemotePtr& rptr : rptrs) {
    if (rptr.is_null()) {
      continue;
    }

    NodeSnapshot snapshot;
    if (local_shard(rptr.memory_node())) {
      read_node_snapshot(rptr, snapshot);
      awaitable.snapshots.push_back(std::move(snapshot));
      continue;
    }

    const size_t scratch_offset = static_cast<size_t>(remote_slot) * snapshot_stride;
    lib_assert(scratch_offset + snapshot_size <= thread.scratch_stride,
               "storage-owner coroutine scratch stride is too small for snapshot batch: "
               "offset=" + std::to_string(scratch_offset) +
               " snapshot=" + std::to_string(snapshot_size) +
               " stride=" + std::to_string(thread.scratch_stride) +
               " remote_slot=" + std::to_string(remote_slot) +
               " batch=" + std::to_string(rptrs.size()));
    byte_t* buffer = thread.coroutine_scratch(scratch_offset);
    if (VamanaNode::compact_storage()) {
      post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                           VamanaNode::size_until_vector_end());
    } else {
      post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                           kSnapshotPrefixBytes);
      post_peer_read_async(thread, rptr.memory_node(),
                             vamana::StorageLayoutResolver::vector(rptr).offset,
                             buffer + kSnapshotPrefixBytes, VamanaNode::vector_bytes());
    }
    awaitable.pending.push_back(NodeSnapshotsReadAwaitable::PendingRead{rptr, buffer});
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

  const size_t snapshot_size = snapshot_buffer_bytes();
  const size_t snapshot_stride = aligned_snapshot_bytes();
  const size_t max_batch = storage_owner_snapshot_batch_size(config, thread);

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
      if (local_shard(rptr.memory_node())) {
        read_node_snapshot(rptr, snapshot);
        snapshots.push_back(std::move(snapshot));
        continue;
      }

      const size_t scratch_offset = static_cast<size_t>(remote_slot) * snapshot_stride;
      lib_assert(scratch_offset + snapshot_size <= thread->scratch_stride,
                 "storage-owner coroutine scratch stride is too small for snapshot batch: "
                 "offset=" + std::to_string(scratch_offset) +
                 " snapshot=" + std::to_string(snapshot_size) +
                 " stride=" + std::to_string(thread->scratch_stride) +
                 " remote_slot=" + std::to_string(remote_slot) +
                 " chunk=" + std::to_string(end - begin));
      byte_t* buffer = thread->coroutine_scratch(scratch_offset);
      if (VamanaNode::compact_storage()) {
        post_peer_read_async(*thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                             VamanaNode::size_until_vector_end());
      } else {
        post_peer_read_async(*thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                             kSnapshotPrefixBytes);
        post_peer_read_async(*thread, rptr.memory_node(),
                             vamana::StorageLayoutResolver::vector(rptr).offset,
                             buffer + kSnapshotPrefixBytes, VamanaNode::vector_bytes());
      }
      pending.push_back(PendingRead{rptr, buffer});
      ++remote_slot;
    }

    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }

    for (const PendingRead& read : pending) {
      NodeSnapshot snapshot;
      parse_remote_snapshot(read.rptr, read.buffer, snapshot);
      snapshots.push_back(std::move(snapshot));
    }
  }

  return snapshots;
}

MemoryNode::NeighborListReadAwaitable MemoryNode::async_read_neighbor_list(RemotePtr rptr,
                                                                           StorageOwnerThread& thread) {
  if (local_shard(rptr.memory_node())) {
    vec<RemotePtr> neighbors = read_neighbor_list(rptr);
    return NeighborListReadAwaitable{true, rptr, nullptr, std::move(neighbors), this, false};
  }

  byte_t* buffer = thread.coroutine_scratch();
  const auto neighbor_read = vamana::StorageLayoutResolver::neighbor_read(rptr);
  const bool use_hot_graph = neighbor_read.compact;
  post_peer_read_async(thread,
                       rptr.memory_node(),
                       neighbor_read.address.offset,
                       buffer,
                       neighbor_read.address.size);
  return NeighborListReadAwaitable{false, rptr, buffer, {}, this, use_hot_graph};
}

void MemoryNode::write_hot_graph_entry(RemotePtr rptr, u32 id, const vec<RemotePtr>& neighbors) {
  if (!VamanaNode::hot_graph_entry_available(rptr)) {
    return;
  }
  const size_t entry_size = VamanaNode::hot_graph_entry_size();
  vec<byte_t> entry(entry_size, 0);
  const u8 edge_count = static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
  VamanaNode::encode_hot_graph_entry(entry.data(), id, edge_count,
                                     neighbors.data(), edge_count);
  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
  if (local_shard(rptr.memory_node())) {
    lib_assert(hot_offset + entry_size <= mn_memory_bytes_,
               "hot graph write exceeds shard bounds");
    std::memcpy(index_buffer_.get_full_buffer() + hot_offset, entry.data(), entry_size);
    return;
  }
  remote_write_bytes(rptr.memory_node(), hot_offset, entry.data(), entry_size, 0);
}

void MemoryNode::write_neighbor_list(RemotePtr rptr, const vec<RemotePtr>& neighbors) {
  lib_assert(rptr.memory_node() < num_storage_nodes_,
             "invalid remote shard id in write_neighbor_list: " + std::to_string(rptr.memory_node()));
  const auto neighbor_addr = vamana::StorageLayoutResolver::neighbor_slots(rptr);
  lib_assert(neighbor_addr.offset + neighbor_addr.size <= mn_memory_bytes_,
             "neighbor-list write exceeds shard bounds: shard=" + std::to_string(rptr.memory_node()) +
               " offset=" + std::to_string(rptr.byte_offset()) +
               " size=" + std::to_string(neighbor_addr.size) +
               " capacity=" + std::to_string(mn_memory_bytes_));
  const u8 edge_count = static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
  if (VamanaNode::compact_storage()) {
    write_hot_graph_entry(rptr, 0, neighbors);
    return;
  }
  if (local_shard(rptr.memory_node())) {
    byte_t* ptr = local_node_ptr(rptr);
    const u32 id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
    *reinterpret_cast<u8*>(ptr + VamanaNode::offset_edge_count()) = edge_count;
    std::memset(ptr + VamanaNode::offset_edge_count() + sizeof(u8), 0, VamanaNode::PADDING_SIZE);
    auto* slots = reinterpret_cast<RemotePtr*>(ptr + VamanaNode::offset_neighbors());
    for (u32 i = 0; i < edge_count; ++i) {
      slots[i] = neighbors[i];
    }
    for (u32 i = edge_count; i < VamanaNode::R; ++i) {
      slots[i].reset();
    }
    write_hot_graph_entry(rptr, id, neighbors);
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
  write_hot_graph_entry(rptr, 0, neighbors);
}

void MemoryNode::write_new_node(RemotePtr rptr,
                    node_t id,
                    const span<const element_t> components,
                    const vec<RemotePtr>& neighbors,
                    u32 generation) {
  byte_t* ptr = local_node_ptr(rptr);
  std::memset(ptr, 0, VamanaNode::total_size());
  *reinterpret_cast<u64*>(ptr) = 0;
  *reinterpret_cast<u32*>(ptr + VamanaNode::offset_id()) = id;
  if (VamanaNode::compact_storage()) {
    *reinterpret_cast<u32*>(ptr + VamanaNode::offset_generation()) = generation;
  } else {
    *reinterpret_cast<u8*>(ptr + VamanaNode::offset_edge_count()) =
      static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
  }
  encode_float_vector_to_storage(components.data(), VamanaNode::DIM, VamanaNode::vector_dtype(),
                                 ptr + VamanaNode::offset_vector());
  if (!VamanaNode::compact_storage()) {
    auto* slots = reinterpret_cast<RemotePtr*>(ptr + VamanaNode::offset_neighbors());
    for (u32 i = 0; i < neighbors.size() && i < VamanaNode::R; ++i) {
      slots[i] = neighbors[i];
    }
  }
  write_hot_graph_entry(rptr, id, neighbors);
  if (VamanaNode::HAS_RABITQ_CODE) {
      VamanaNode::RabitqCode code;
      float norm = 0.0f;
      float error = 0.0f;
      VamanaNode::compute_rabitq_entry(
          ptr + VamanaNode::offset_vector(), VamanaNode::vector_dtype(), code, norm, error);
      std::memcpy(ptr + VamanaNode::offset_rabitq_code(), code.data(), code.size());
      *reinterpret_cast<float*>(ptr + VamanaNode::offset_rabitq_norm()) = norm;
      *reinterpret_cast<float*>(ptr + VamanaNode::offset_rabitq_error()) = error;
  }
}

void MemoryNode::lock_node(RemotePtr rptr) {
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + vamana::StorageLayoutResolver::header(rptr).offset);
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
    auto* header_ptr = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + vamana::StorageLayoutResolver::header(rptr).offset);
    std::atomic_ref<u64> ref(*header_ptr);
    ref.fetch_and(~static_cast<u64>(VamanaNode::HEADER_NODE_LOCK), std::memory_order_acq_rel);
    return;
  }

  const byte_t unlock = 0;
  remote_write_bytes(rptr.memory_node(),
                     vamana::StorageLayoutResolver::header(rptr).offset +
                       VamanaNode::HEADER_UNTIL_LOCK,
                     &unlock, 1, 0);
}
