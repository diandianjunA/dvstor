#include "memory_node/memory_node.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "common/index_path.hh"
#include "vamana/idmap.hh"
#include "vamana/storage_layout_resolver.hh"

namespace {

using Configuration = configuration::IndexConfiguration;
using BeamEntry = memory_node_detail::BeamEntry;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using StorageOwnerCoroutineScratch = memory_node_detail::StorageOwnerCoroutineScratch;
using StorageOwnerPruneCandidateInfo = memory_node_detail::StorageOwnerPruneCandidateInfo;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;

size_t snapshot_buffer_bytes() {
  return memory_node_detail::storage_owner_snapshot_bytes();
}

size_t aligned_snapshot_bytes() {
  return memory_node_detail::storage_owner_snapshot_stride();
}

u32 storage_owner_construction_width(const Configuration& config) {
  const u32 configured = config.storage_owner_construction_beam_width == 0
                           ? config.beam_width_construction
                           : config.storage_owner_construction_beam_width;
  return std::max<u32>(1, std::min(config.beam_width_construction, configured));
}

u32 storage_owner_snapshot_batch_size(const Configuration& config,
                                      const StorageOwnerThread* thread = nullptr) {
  const u32 configured = std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  if (thread == nullptr || !thread->has_peer_scratch()) {
    return configured;
  }
  const size_t stride = aligned_snapshot_bytes();
  const size_t capacity = stride == 0 ? 0 : thread->scratch_stride / stride;
  lib_assert(capacity > 0,
             "storage-owner coroutine scratch cannot hold one snapshot: snapshot_stride=" +
             std::to_string(stride) + " scratch_stride=" +
             std::to_string(thread->scratch_stride));
  return static_cast<u32>(std::min<size_t>(configured, capacity));
}

u32 storage_owner_prune_candidate_limit(const Configuration& config) {
  if (config.storage_owner_prune_max_candidates == 0) {
    return std::numeric_limits<u32>::max();
  }
  return std::max(config.R, config.storage_owner_prune_max_candidates);
}

bool anchor_update_enabled(const Configuration& config, const vec<RemotePtr>& hints) {
  return config.storage_owner_update_mode == "local_stitch" && !hints.empty();
}

bool local_stitch_enabled(const Configuration& config) {
  return config.storage_owner_update_mode == "local_stitch";
}

void parse_remote_snapshot(RemotePtr rptr, const byte_t* ptr, NodeSnapshot& snapshot) {
  snapshot = NodeSnapshot{};
  snapshot.rptr = rptr;
  snapshot.header = *reinterpret_cast<const u64*>(ptr);
  snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
  snapshot.generation =
    *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_generation());
  snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
  snapshot.vector_data.resize(VamanaNode::vector_bytes());
  std::memcpy(snapshot.vector_data.data(), ptr + VamanaNode::offset_vector(),
              VamanaNode::vector_bytes());
}

}  // namespace

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
    snapshot.generation =
      *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_generation());
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
  snapshot.generation =
    *reinterpret_cast<const u32*>(prefix + VamanaNode::offset_generation());
  snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
  return true;
}

vec<RemotePtr> MemoryNode::read_neighbor_list(RemotePtr rptr) {
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
      parse_remote_snapshot(rptr, buffer, snapshot);
      return std::move(snapshot);
    }
  };

  if (local_shard(rptr.memory_node())) {
    NodeSnapshot snapshot;
    read_node_snapshot(rptr, snapshot);
    return Awaitable{true, rptr, nullptr, std::move(snapshot), this, &thread};
  }

  byte_t* buffer = thread.coroutine_scratch();
  post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                       VamanaNode::size_until_vector_end());
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
        parse_remote_snapshot(read.rptr, read.buffer, snapshot);
        snapshots.push_back(std::move(snapshot));
      }
      return std::move(snapshots);
    }
  };

  Awaitable awaitable;
  awaitable.snapshots.reserve(rptrs.size());
  awaitable.pending.reserve(rptrs.size());
  awaitable.thread = &thread;

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
    post_peer_read_async(thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                         VamanaNode::size_until_vector_end());
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
      post_peer_read_async(*thread, rptr.memory_node(), rptr.byte_offset(), buffer,
                           VamanaNode::size_until_vector_end());
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
      vec<byte_t> decoded(VamanaNode::neighbor_read_size());
      if (!VamanaNode::decode_hot_graph_entry(buffer, decoded.data())) {
        return node->read_neighbor_list(rptr);
      }
      const byte_t* parse_buffer = decoded.data();
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
  };

  if (local_shard(rptr.memory_node())) {
    vec<RemotePtr> neighbors = read_neighbor_list(rptr);
    return Awaitable{true, rptr, nullptr, std::move(neighbors), this, &thread};
  }

  byte_t* buffer = thread.coroutine_scratch();
  const auto neighbor_read = vamana::StorageLayoutResolver::neighbor_read(rptr);
  post_peer_read_async(thread,
                       rptr.memory_node(),
                       neighbor_read.address.offset,
                       buffer,
                       neighbor_read.address.size);
  return Awaitable{false, rptr, buffer, {}, this, &thread};
}

void MemoryNode::write_hot_graph_entry(RemotePtr rptr, const vec<RemotePtr>& neighbors) {
  if (!VamanaNode::hot_graph_entry_available(rptr)) {
    return;
  }
  const size_t entry_size = VamanaNode::hot_graph_entry_size();
  vec<byte_t> entry(entry_size, 0);
  const u8 edge_count = static_cast<u8>(std::min<size_t>(neighbors.size(), VamanaNode::R));
  VamanaNode::encode_hot_graph_entry(entry.data(), edge_count,
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
  write_hot_graph_entry(rptr, neighbors);
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
  *reinterpret_cast<u32*>(ptr + VamanaNode::offset_generation()) = generation;
  encode_float_vector_to_storage(components.data(), VamanaNode::DIM, VamanaNode::vector_dtype(),
                                 ptr + VamanaNode::offset_vector());
  write_hot_graph_entry(rptr, neighbors);
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

vec<RemotePtr> MemoryNode::beam_search_candidates(const span<const element_t> query,
                                                  RemotePtr medoid,
                                                  const Configuration& config,
                                                  InsertBreakdownCounters* breakdown) {
  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;

  auto t_snapshot = std::chrono::steady_clock::now();
  NodeSnapshot medoid_snapshot;
  read_node_snapshot(medoid, medoid_snapshot);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
  }
  auto t_distance = std::chrono::steady_clock::now();
  const distance_t medoid_dist = distance_to_stored_vector(query, medoid_snapshot.vector_data.data(), config);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
  }

  beam.push_back({medoid, medoid_dist, false});
  visited.insert(medoid);

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert shard locality summary
  static std::atomic<u32> insert_seq{0};
  u32 this_insert = insert_seq.fetch_add(1, std::memory_order_relaxed);
  bool should_log = (this_insert < 5) || (this_insert % 500 == 0);
  u32 iter_count = 0;
  u32 local_unvisited_sum = 0, remote_unvisited_sum = 0;
#endif

  for (;;) {
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    ++iter_count;
#endif
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
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    u32 iter_local = 0, iter_remote = 0;
#endif
    for (const RemotePtr& neighbor : neighbors) {
      if (neighbor.is_null() || visited.contains(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      unvisited_neighbors.push_back(neighbor);
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
      if (local_shard(neighbor.memory_node())) ++iter_local; else ++iter_remote;
#endif
    }
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    local_unvisited_sum += iter_local;
    remote_unvisited_sum += iter_remote;
#endif

    const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);
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
        if (snapshot.deleted) {
          continue;
        }
        t_distance = std::chrono::steady_clock::now();
        const distance_t dist = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
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

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert summary
  if (should_log) {
    u32 total = local_unvisited_sum + remote_unvisited_sum;
    float local_pct = total > 0 ? 100.0f * local_unvisited_sum / total : 0;
    std::cerr << "[beam_search] insert=" << this_insert
              << " shard=" << storage_id_
              << " iters=" << iter_count
              << " local=" << local_unvisited_sum
              << " remote=" << remote_unvisited_sum
              << " local_pct=" << local_pct << "%"
              << std::endl;
  }
#endif

  vec<RemotePtr> candidates;
  candidates.reserve(beam.size());
  auto t_sort = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
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
  StorageOwnerCoroutineScratch& scratch = thread.coroutine_scratch_state();
  scratch.clear_search();
  hashset_t<RemotePtr>& visited = scratch.visited;
  vec<BeamEntry>& beam = scratch.beam;
  vec<RemotePtr>& unvisited_neighbors = scratch.unvisited;
  vec<RemotePtr>& batch = scratch.batch;
  const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, &thread);
  const u32 construction_width = storage_owner_construction_width(config);
  visited.reserve(static_cast<size_t>(construction_width) * std::max<u32>(1, config.R));
  beam.reserve(construction_width);
  unvisited_neighbors.reserve(config.R);
  batch.reserve(snapshot_batch);

  auto t_snapshot = std::chrono::steady_clock::now();
  NodeSnapshot medoid_snapshot = co_await async_read_node_snapshot(medoid, thread);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
  }
  auto t_distance = std::chrono::steady_clock::now();
  const distance_t medoid_dist = distance_to_stored_vector(query, medoid_snapshot.vector_data.data(), config);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
  }

  beam.push_back({medoid, medoid_dist, false});
  visited.insert(medoid);

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert per-shard distribution
  static std::atomic<u32> async_insert_seq{0};
  u32 this_insert_a = async_insert_seq.fetch_add(1, std::memory_order_relaxed);
  bool should_log_a = (this_insert_a < 5) || (this_insert_a % 500 == 0);
  u32 iter_count_a = 0;
  u32 shard_hist[6] = {0};  // [0..3]=remote by shard, [4]=local(self), [5]=total expanded
#endif

  for (;;) {
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    ++iter_count_a;
#endif
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
    unvisited_neighbors.clear();
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    u32 expanded_shard = beam[best_idx].rptr.memory_node();
#endif
    for (const RemotePtr& neighbor : neighbors) {
      if (neighbor.is_null() || visited.contains(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      unvisited_neighbors.push_back(neighbor);
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
      u32 ns = neighbor.memory_node();
      if (ns == storage_id_) ++shard_hist[4]; else ++shard_hist[ns];
#endif
    }
#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
    ++shard_hist[5];
#endif

    for (size_t begin = 0; begin < unvisited_neighbors.size(); begin += snapshot_batch) {
      const size_t end = std::min(unvisited_neighbors.size(), begin + snapshot_batch);
      batch.clear();
      batch.insert(batch.end(), unvisited_neighbors.begin() + begin, unvisited_neighbors.begin() + end);
      t_snapshot = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = co_await async_read_node_snapshots(batch, config, thread);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted) {
          continue;
        }
        t_distance = std::chrono::steady_clock::now();
        const distance_t dist = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
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

#ifdef DVSTOR_DEBUG_SHARD_LOCALITY
  // DEBUG: per-insert per-shard summary
  if (should_log_a) {
    u32 total_neighbors = 0;
    for (u32 s = 0; s < 5; ++s) total_neighbors += shard_hist[s];
    std::cerr << "[beam_search_async] insert=" << this_insert_a
              << " self=" << storage_id_
              << " iters=" << iter_count_a
              << " expanded=" << shard_hist[5]
              << " local=" << shard_hist[4];
    for (u32 s = 0; s < 5; ++s) {
      if (s == storage_id_) continue;
      float pct = total_neighbors > 0 ? 100.0f * shard_hist[s] / total_neighbors : 0;
      std::cerr << " sh" << s << "=" << shard_hist[s] << "(" << int(pct) << "%)";
    }
    std::cerr << std::endl;
  }
#endif

  auto& out = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  out.clear();
  out.reserve(beam.size());
  auto t_sort = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
  }
  for (const auto& entry : beam) {
    out.push_back(entry.rptr);
  }
}

auto MemoryNode::anchor_search_candidates_async(const span<const element_t> query,
                                                const vec<RemotePtr>& anchor_hints,
                                                const Configuration& config,
                                                StorageOwnerThread& thread,
                                                InsertBreakdownCounters* breakdown,
                                                bool local_only)
  -> StorageOwnerInsertCoroutine {
  StorageOwnerCoroutineScratch& scratch = thread.coroutine_scratch_state();
  scratch.clear_search();
  hashset_t<RemotePtr>& visited = scratch.visited;
  vec<BeamEntry>& beam = scratch.beam;
  vec<RemotePtr>& batch = scratch.batch;
  vec<RemotePtr>& unvisited = scratch.unvisited;
  const u32 beam_width = std::max<u32>(config.R, config.storage_owner_anchor_beam_width);
  const u32 batch_limit = storage_owner_snapshot_batch_size(config, &thread);
  visited.reserve(anchor_hints.size() +
                  static_cast<size_t>(config.storage_owner_anchor_expand_cap) *
                    std::max<u32>(1, config.R));
  beam.reserve(beam_width);
  batch.reserve(batch_limit);
  unvisited.reserve(config.R);

  if (breakdown != nullptr) {
    breakdown->storage_owner_anchor_hints += anchor_hints.size();
  }
  for (size_t begin = 0; begin < anchor_hints.size(); begin += batch_limit) {
    const size_t end = std::min(anchor_hints.size(), begin + batch_limit);
    batch.clear();
    for (size_t i = begin; i < end; ++i) {
      const RemotePtr hint = anchor_hints[i];
      if (!hint.is_null() && hint.memory_node() < num_storage_nodes_ &&
          (!local_only || local_shard(hint.memory_node())) &&
          visited.insert(hint).second) {
        batch.push_back(hint);
      }
    }
    auto started = std::chrono::steady_clock::now();
    vec<NodeSnapshot> snapshots = co_await async_read_node_snapshots(batch, config, thread);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(started);
    }
    for (const NodeSnapshot& snapshot : snapshots) {
      if (snapshot.deleted) continue;
      started = std::chrono::steady_clock::now();
      const distance_t distance = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_distance_ns += elapsed_ns_since(started);
        ++breakdown->storage_owner_anchor_valid_hints;
      }
      insert_into_beam(beam, snapshot.rptr, distance, beam_width);
    }
  }

  u32 expansions = 0;
  u32 remote_expansions = 0;
  while (expansions < config.storage_owner_anchor_expand_cap) {
    auto started = std::chrono::steady_clock::now();
    i32 best = -1;
    distance_t best_distance = std::numeric_limits<distance_t>::max();
    for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
      if (!beam[i].expanded && beam[i].distance < best_distance) {
        best = i;
        best_distance = beam[i].distance;
      }
    }
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_select_ns += elapsed_ns_since(started);
    }
    if (best < 0) break;

    BeamEntry& entry = beam[best];
    entry.expanded = true;
    const bool remote = !local_shard(entry.rptr.memory_node());
    if (local_only && remote) {
      continue;
    }
    if (remote && remote_expansions >= config.storage_owner_anchor_remote_rescue_cap) {
      continue;
    }
    ++expansions;
    if (remote) ++remote_expansions;

    started = std::chrono::steady_clock::now();
    const vec<RemotePtr> neighbors = co_await async_read_neighbor_list(entry.rptr, thread);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_neighbor_read_ns += elapsed_ns_since(started);
    }
    unvisited.clear();
    for (const RemotePtr neighbor : neighbors) {
      if (!neighbor.is_null() && neighbor.memory_node() < num_storage_nodes_ &&
          (!local_only || local_shard(neighbor.memory_node())) &&
          visited.insert(neighbor).second) {
        unvisited.push_back(neighbor);
      }
    }

    for (size_t begin = 0; begin < unvisited.size(); begin += batch_limit) {
      const size_t end = std::min(unvisited.size(), begin + batch_limit);
      batch.clear();
      batch.insert(batch.end(), unvisited.begin() + begin, unvisited.begin() + end);
      started = std::chrono::steady_clock::now();
      vec<NodeSnapshot> snapshots = co_await async_read_node_snapshots(batch, config, thread);
      if (breakdown != nullptr) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(started);
      }
      for (const NodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted) continue;
        started = std::chrono::steady_clock::now();
        const distance_t distance = distance_to_stored_vector(query, snapshot.vector_data.data(), config);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_distance_ns += elapsed_ns_since(started);
        }
        insert_into_beam(beam, snapshot.rptr, distance, beam_width);
      }
    }
  }

  if (breakdown != nullptr) {
    breakdown->storage_owner_anchor_expansions += expansions;
    breakdown->storage_owner_anchor_remote_expansions += remote_expansions;
  }
  auto& out = storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  out.clear();
  out.reserve(beam.size());
  auto started = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(started);
  }
  for (const BeamEntry& entry : beam) out.push_back(entry.rptr);
}

vec<RemotePtr> MemoryNode::robust_prune_cpu(const byte_t* source,
                                            VectorDType source_dtype,
                                            const vec<RemotePtr>& candidates,
                                            const hashset_t<RemotePtr>& skip,
                                            const Configuration& config,
                                            InsertBreakdownCounters* breakdown,
                                            u32 candidate_limit_override) {
  StorageOwnerCoroutineScratch* scratch = current_storage_owner_thread_ != nullptr
                                            ? &current_storage_owner_thread_->coroutine_scratch_state()
                                            : nullptr;
  vec<StorageOwnerPruneCandidateInfo> local_infos;
  vec<RemotePtr> local_filtered;
  vec<RemotePtr> local_batch;
  vec<RemotePtr> local_selected;
  vec<const byte_t*> local_selected_vectors;
  if (scratch != nullptr) {
    scratch->clear_prune();
  }
  vec<StorageOwnerPruneCandidateInfo>& infos = scratch != nullptr ? scratch->prune_infos : local_infos;
  vec<RemotePtr>& filtered = scratch != nullptr ? scratch->filtered : local_filtered;
  vec<RemotePtr>& batch = scratch != nullptr ? scratch->batch : local_batch;
  vec<RemotePtr>& selected = scratch != nullptr ? scratch->selected : local_selected;
  vec<const byte_t*>& selected_vectors = scratch != nullptr ? scratch->selected_vectors : local_selected_vectors;
  const u32 prune_candidate_limit = candidate_limit_override == 0
                                      ? storage_owner_prune_candidate_limit(config)
                                      : std::max(config.R, candidate_limit_override);
  infos.reserve(candidates.size());
  filtered.reserve(std::min<size_t>(candidates.size(), prune_candidate_limit));
  batch.reserve(storage_owner_snapshot_batch_size(config, current_storage_owner_thread_));
  selected.reserve(config.R);
  selected_vectors.reserve(config.R);

  for (const RemotePtr& candidate : candidates) {
    if (candidate.is_null() || skip.contains(candidate)) {
      continue;
    }
    filtered.push_back(candidate);
    if (filtered.size() >= prune_candidate_limit) {
      break;
    }
  }

  const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);
  for (size_t begin = 0; begin < filtered.size(); begin += snapshot_batch) {
    const size_t end = std::min(filtered.size(), begin + snapshot_batch);
    batch.clear();
    batch.insert(batch.end(), filtered.begin() + begin, filtered.begin() + end);
    auto t_snapshot = std::chrono::steady_clock::now();
    vec<NodeSnapshot> snapshots = read_node_snapshots_batched(batch, config);
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_snapshot);
    }
    for (NodeSnapshot& snapshot : snapshots) {
      if (snapshot.deleted) {
        continue;
      }
      auto t_distance = std::chrono::steady_clock::now();
      const distance_t dist = distance_between_vectors(source, source_dtype,
                                                       snapshot.vector_data.data(), VamanaNode::vector_dtype(), config);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_distance_ns += elapsed_ns_since(t_distance);
      }
      infos.push_back({snapshot.rptr, dist, std::move(snapshot.vector_data)});
    }
  }

  auto t_sort = std::chrono::steady_clock::now();
  std::sort(infos.begin(), infos.end(), [](const StorageOwnerPruneCandidateInfo& lhs,
                                           const StorageOwnerPruneCandidateInfo& rhs) {
    return lhs.dist < rhs.dist;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_prune_sort_ns += elapsed_ns_since(t_sort);
  }

  for (const auto& candidate : infos) {
    if (selected.size() >= config.R) {
      break;
    }

    bool pruned = false;
    for (idx_t i = 0; i < selected_vectors.size(); ++i) {
      auto t_pair_distance = std::chrono::steady_clock::now();
      const distance_t pair_dist = distance_between_vectors(candidate.vector_data.data(), VamanaNode::vector_dtype(),
                                                           selected_vectors[i], VamanaNode::vector_dtype(), config);
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
      selected_vectors.push_back(candidate.vector_data.data());
    }
  }

  return selected;
}

auto MemoryNode::execute_storage_owner_insert_job_async(StorageOwnerThread& thread,
                                            StorageOwnerInsertJob& job,
                                            dense_hashmap_t<u64, vec<RemotePtr>>& local_updates,
                                            dense_hashmap_t<u32, vec<service::storage_owner::ReverseUpdateOp>>& remote_updates,
                                            InsertBreakdownCounters& breakdown,
                                            const Configuration& config) -> StorageOwnerInsertCoroutine {
  const auto components = span<const element_t>{reinterpret_cast<const element_t*>(job.vector_data.data()),
                                                 VamanaNode::DIM};
  FreshnessEntry old_entry{};
  u32 generation = 0;
  const auto status = prepare_mutation(job.id, job.kind, &old_entry, &generation);
  job.old_ptr = old_entry.current;
  job.generation = generation;
  const bool maintenance_enabled = storage_owner_maintenance_enabled(config);
  if (status != service::storage_owner::MutationStatus::ok) {
    job.status = status;
    job.ok = false;
    co_return;
  }
  if (job.kind == service::storage_owner::MutationKind::erase) {
    job.ok = mark_node_deleted(old_entry.current, old_entry.generation);
    job.status = job.ok ? service::storage_owner::MutationStatus::ok
                        : service::storage_owner::MutationStatus::failed;
    if (job.ok) {
      publish_mutation(job.id, old_entry.current, old_entry.generation, true);
      if (maintenance_enabled) {
        (void)enqueue_deleted_node_cleanup(old_entry.current, config);
      }
    }
    co_return;
  }
  const bool local_stitch = local_stitch_enabled(config);
  const bool use_anchors = anchor_update_enabled(config, job.anchor_hints);
  RemotePtr medoid_ptr{};
  bool medoid_loaded = false;
  const vec<RemotePtr>* candidates = nullptr;

  if (use_anchors) {
    auto t_search = std::chrono::steady_clock::now();
    auto search = anchor_search_candidates_async(components, job.anchor_hints, config, thread,
                                                 &breakdown, local_stitch);
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
    candidates = &storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  }

  if (!use_anchors) {
    auto t_medoid = std::chrono::steady_clock::now();
    medoid_ptr = co_await async_read_global_medoid(thread);
    medoid_loaded = true;
    breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
  }
  if (medoid_loaded && medoid_ptr.is_null()) {
    const RemotePtr new_ptr = allocate_local_node();
    job.new_ptr = new_ptr;
    auto t_write = std::chrono::steady_clock::now();
    write_new_node(new_ptr, job.id, components, {}, generation);
    breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
    RemotePtr observed;
    if (try_set_global_medoid(RemotePtr{}, new_ptr, observed) || observed.is_null()) {
      job.ok = true;
      job.status = service::storage_owner::MutationStatus::ok;
      if (job.kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
        mark_node_deleted(old_entry.current, old_entry.generation);
      }
      publish_mutation(job.id, new_ptr, generation, false);
      if (maintenance_enabled) {
        (void)enqueue_insert_stitch(job.id, generation, new_ptr, config);
        (void)enqueue_deleted_node_cleanup(old_entry.current, config);
      }
      co_return;
    }
    medoid_ptr = observed;
  }

  if (!use_anchors) {
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
    candidates = &storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  }

  lib_assert(candidates != nullptr, "storage-owner insert search produced no candidate set");
  StorageOwnerCoroutineScratch& scratch = thread.coroutine_scratch_state();
  scratch.empty_skip.clear();
  auto t_prune = std::chrono::steady_clock::now();
  vec<RemotePtr> selected_neighbors = robust_prune_cpu(reinterpret_cast<const byte_t*>(components.data()),
                                                       VectorDType::float32, *candidates, scratch.empty_skip, config, &breakdown);
  breakdown.storage_owner_prune_ns += elapsed_ns_since(t_prune);
  const RemotePtr new_ptr = allocate_local_node();
  job.new_ptr = new_ptr;
  auto t_write = std::chrono::steady_clock::now();
  write_new_node(new_ptr, job.id, components, selected_neighbors, generation);
  breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
  if (job.kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
    mark_node_deleted(old_entry.current, old_entry.generation);
  }
  publish_mutation(job.id, new_ptr, generation, false);
  if (maintenance_enabled) {
    (void)enqueue_insert_stitch(job.id, generation, new_ptr, config);
    (void)enqueue_deleted_node_cleanup(old_entry.current, config);
  }

  if (!maintenance_enabled || local_stitch) {
    for (const RemotePtr& neighbor_ptr : selected_neighbors) {
      if (local_shard(neighbor_ptr.memory_node())) {
        local_updates[neighbor_ptr.raw_address].push_back(new_ptr);
      } else if (!local_stitch) {
        remote_updates[neighbor_ptr.memory_node()].push_back(
          service::storage_owner::ReverseUpdateOp{neighbor_ptr.raw_address, new_ptr.raw_address});
      }
    }
  }
  job.ok = true;
  job.status = service::storage_owner::MutationStatus::ok;
}

bool MemoryNode::apply_local_reverse_update(RemotePtr target_ptr,
                                const vec<RemotePtr>& candidate_ptrs,
                                const Configuration& config,
                                bool enqueue_maintenance) {
  lib_assert(local_shard(target_ptr.memory_node()), "target reverse update must be local");
  if (candidate_ptrs.empty()) {
    return true;
  }

  const auto update_started = std::chrono::steady_clock::now();
  StorageOwnerCoroutineScratch* scratch = current_storage_owner_thread_ != nullptr
                                            ? &current_storage_owner_thread_->coroutine_scratch_state()
                                            : nullptr;
  vec<RemotePtr> local_unique_candidates;
  vec<RemotePtr> local_current_neighbors;
  vec<RemotePtr> local_filtered_candidates;
  vec<RemotePtr> local_updated_neighbors;
  vec<RemotePtr> local_remote_neighbors;
  vec<RemotePtr> local_remote_candidates;
  vec<distance_t> local_neighbor_dists;
  if (scratch != nullptr) {
    scratch->clear_reverse_update();
  }
  vec<RemotePtr>& unique_candidates = scratch != nullptr ? scratch->reverse_unique_candidates : local_unique_candidates;
  vec<RemotePtr>& current_neighbors = scratch != nullptr ? scratch->reverse_current_neighbors : local_current_neighbors;
  vec<RemotePtr>& filtered_candidates =
    scratch != nullptr ? scratch->reverse_filtered_candidates : local_filtered_candidates;
  vec<RemotePtr>& updated_neighbors = scratch != nullptr ? scratch->reverse_updated_neighbors : local_updated_neighbors;
  vec<RemotePtr>& remote_neighbors = scratch != nullptr ? scratch->reverse_remote_neighbors : local_remote_neighbors;
  vec<RemotePtr>& remote_candidates = scratch != nullptr ? scratch->reverse_remote_candidates : local_remote_candidates;
  vec<distance_t>& neighbor_dists = scratch != nullptr ? scratch->reverse_neighbor_dists : local_neighbor_dists;

  bool changed = false;
  bool pruned = false;
  size_t current_count = 0;
  size_t filtered_count = 0;
  u64 lock_wait_ns = 0;
  u64 snapshot_ns = 0;
  u64 neighbor_read_ns = 0;
  u64 filter_ns = 0;
  u64 prune_ns = 0;
  u64 write_ns = 0;

  auto step_started = std::chrono::steady_clock::now();
  const auto target_vector_addr = vamana::StorageLayoutResolver::vector(target_ptr);
  lib_assert(target_vector_addr.offset + target_vector_addr.size <= mn_memory_bytes_,
             "local reverse-update target vector exceeds shard bounds");
  const byte_t* target_node = local_node_ptr(target_ptr);
  const byte_t* target_vector = index_buffer_.get_full_buffer() + target_vector_addr.offset;
  if ((*reinterpret_cast<const u64*>(target_node) & VamanaNode::HEADER_DELETED) != 0) {
    return true;
  }
  snapshot_ns = elapsed_ns_since(step_started);

  for (const RemotePtr& candidate_ptr : candidate_ptrs) {
    if (!candidate_ptr.is_null() &&
        std::find(unique_candidates.begin(), unique_candidates.end(), candidate_ptr) == unique_candidates.end()) {
      unique_candidates.push_back(candidate_ptr);
    }
  }
  if (unique_candidates.empty()) {
    return true;
  }

  auto target_deleted = [&]() {
    return (*reinterpret_cast<const u64*>(local_node_ptr(target_ptr)) &
            VamanaNode::HEADER_DELETED) != 0;
  };

  auto vector_ptr = [&](const RemotePtr& rptr) {
    const auto addr = vamana::StorageLayoutResolver::vector(rptr);
    lib_assert(addr.offset + addr.size <= mn_memory_bytes_,
               "local reverse-update vector read exceeds shard bounds");
    return index_buffer_.get_full_buffer() + addr.offset;
  };

  auto push_candidate = [&](const RemotePtr& candidate, distance_t candidate_dist) {
    if (updated_neighbors.size() < config.R) {
      updated_neighbors.push_back(candidate);
      neighbor_dists.push_back(candidate_dist);
      return;
    }
    lib_assert(!neighbor_dists.empty(), "reverse-update neighbor distances are unexpectedly empty");
    size_t farthest_idx = 0;
    distance_t farthest_dist = neighbor_dists[0];
    for (size_t i = 1; i < neighbor_dists.size(); ++i) {
      if (neighbor_dists[i] > farthest_dist) {
        farthest_dist = neighbor_dists[i];
        farthest_idx = i;
      }
    }
    if (candidate_dist < farthest_dist) {
      updated_neighbors[farthest_idx] = candidate;
      neighbor_dists[farthest_idx] = candidate_dist;
    }
  };

  auto build_pruned_neighbors = [&](const vec<RemotePtr>& source_neighbors,
                                    const vec<RemotePtr>& source_candidates) {
    updated_neighbors.clear();
    neighbor_dists.clear();
    remote_neighbors.clear();
    remote_candidates.clear();
    updated_neighbors.reserve(config.R);
    neighbor_dists.reserve(config.R);
    remote_neighbors.reserve(source_neighbors.size());
    remote_candidates.reserve(source_candidates.size());

    for (const RemotePtr& neighbor : source_neighbors) {
      if (neighbor.is_null()) {
        continue;
      }
      if (local_shard(neighbor.memory_node())) {
        updated_neighbors.push_back(neighbor);
        neighbor_dists.push_back(distance_between_vectors(target_vector, VamanaNode::vector_dtype(),
                                                          vector_ptr(neighbor), VamanaNode::vector_dtype(), config));
      } else {
        remote_neighbors.push_back(neighbor);
      }
    }
    if (!remote_neighbors.empty()) {
      vec<NodeSnapshot> snapshots = read_node_snapshots_batched(remote_neighbors, config);
      for (const NodeSnapshot& snapshot : snapshots) {
        updated_neighbors.push_back(snapshot.rptr);
        neighbor_dists.push_back(distance_between_vectors(target_vector, VamanaNode::vector_dtype(),
                                                          snapshot.vector_data.data(), VamanaNode::vector_dtype(),
                                                          config));
      }
    }

    for (const RemotePtr& candidate : source_candidates) {
      if (candidate.is_null()) {
        continue;
      }
      if (local_shard(candidate.memory_node())) {
        const distance_t candidate_dist = distance_between_vectors(target_vector, VamanaNode::vector_dtype(),
                                                                   vector_ptr(candidate), VamanaNode::vector_dtype(),
                                                                   config);
        push_candidate(candidate, candidate_dist);
      } else {
        remote_candidates.push_back(candidate);
      }
    }
    if (!remote_candidates.empty()) {
      vec<NodeSnapshot> candidate_snapshots = read_node_snapshots_batched(remote_candidates, config);
      for (const NodeSnapshot& snapshot : candidate_snapshots) {
        const distance_t candidate_dist = distance_between_vectors(target_vector, VamanaNode::vector_dtype(),
                                                                   snapshot.vector_data.data(),
                                                                   VamanaNode::vector_dtype(), config);
        push_candidate(snapshot.rptr, candidate_dist);
      }
    }
  };

  const auto lock_started = std::chrono::steady_clock::now();
  lock_node(target_ptr);
  lock_wait_ns += elapsed_ns_since(lock_started);
  if (target_deleted()) {
    unlock_node(target_ptr);
    return true;
  }

  step_started = std::chrono::steady_clock::now();
  current_neighbors = read_neighbor_list(target_ptr);
  neighbor_read_ns += elapsed_ns_since(step_started);
  current_count = current_neighbors.size();

  step_started = std::chrono::steady_clock::now();
  filtered_candidates.clear();
  filtered_candidates.reserve(unique_candidates.size());
  for (const RemotePtr& candidate_ptr : unique_candidates) {
    bool already_present = false;
    for (const RemotePtr& current : current_neighbors) {
      if (current == candidate_ptr) {
        already_present = true;
        break;
      }
    }
    if (!already_present) {
      filtered_candidates.push_back(candidate_ptr);
    }
  }
  filter_ns += elapsed_ns_since(step_started);
  filtered_count = filtered_candidates.size();
  if (filtered_candidates.empty()) {
    unlock_node(target_ptr);
    return true;
  }

  changed = true;
  if (current_neighbors.size() + filtered_candidates.size() <= config.R) {
    updated_neighbors = current_neighbors;
    updated_neighbors.insert(updated_neighbors.end(), filtered_candidates.begin(), filtered_candidates.end());
  } else {
    pruned = true;
    step_started = std::chrono::steady_clock::now();
    build_pruned_neighbors(current_neighbors, filtered_candidates);
    prune_ns += elapsed_ns_since(step_started);
  }

  step_started = std::chrono::steady_clock::now();
  write_neighbor_list(target_ptr, updated_neighbors);
  write_ns += elapsed_ns_since(step_started);
  unlock_node(target_ptr);

  (void)enqueue_maintenance;

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
