#include "memory_node/memory_node.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include "common/index_path.hh"
#include "vamana/idmap.hh"
#include "vamana/rabitq_cache.hh"
#include "vamana/storage_layout_resolver.hh"

namespace {

using Configuration = configuration::IndexConfiguration;
using BeamEntry = memory_node_detail::BeamEntry;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using QirCodeSnapshot = memory_node_detail::QirCodeSnapshot;
using QirDistanceInterval = memory_node_detail::QirDistanceInterval;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;
using InsertBreakdownCounters = service::storage_owner::InsertBreakdownCounters;

constexpr size_t kSnapshotPrefixBytes =
  VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;
constexpr u32 kHeaderGenerationShift = 32;

u32 snapshot_generation(const byte_t* prefix, u64 header) {
  return VamanaNode::compact_storage()
    ? *reinterpret_cast<const u32*>(prefix + VamanaNode::offset_generation())
    : static_cast<u32>(header >> kHeaderGenerationShift);
}

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

void parse_remote_snapshot(RemotePtr rptr, const byte_t* ptr, NodeSnapshot& snapshot) {
  snapshot = NodeSnapshot{};
  snapshot.rptr = rptr;
  snapshot.header = *reinterpret_cast<const u64*>(ptr);
  snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
  snapshot.generation = snapshot_generation(ptr, snapshot.header);
  snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
  snapshot.vector_data.resize(VamanaNode::vector_bytes());
  const size_t vector_offset = VamanaNode::compact_storage()
    ? VamanaNode::offset_vector() : kSnapshotPrefixBytes;
  std::memcpy(snapshot.vector_data.data(), ptr + vector_offset, VamanaNode::vector_bytes());
}

void parse_qir_prefix(RemotePtr rptr, const byte_t* prefix, QirCodeSnapshot& snapshot) {
  snapshot.rptr = rptr;
  const u64 header = *reinterpret_cast<const u64*>(prefix);
  snapshot.deleted = (header & VamanaNode::HEADER_DELETED) != 0;
  snapshot.generation = snapshot_generation(prefix, header);
  snapshot.prefix_validated = true;
}

void qir_read_local_code(const byte_t* node_ptr, RemotePtr rptr, QirCodeSnapshot& snapshot) {
  parse_qir_prefix(rptr, node_ptr, snapshot);
  snapshot.entry.resize(VamanaNode::rabitq_entry_size());
  std::memcpy(snapshot.entry.data(),
              node_ptr + VamanaNode::offset_rabitq_code(),
              snapshot.entry.size());
}

float qir_load_float(const byte_t* ptr) {
  float value = 0.0f;
  std::memcpy(&value, ptr, sizeof(value));
  return value;
}

float qir_entry_norm(const byte_t* entry) {
  return qir_load_float(entry + VamanaNode::rabitq_code_storage_size());
}

float qir_entry_error(const byte_t* entry) {
  return qir_load_float(entry + VamanaNode::rabitq_code_storage_size() + sizeof(float));
}

float qir_source_interval_slack(float query_norm2, const byte_t* entry) {
  const float norm = qir_entry_norm(entry);
  const float error = std::clamp(qir_entry_error(entry), 1e-6f, 1.0f);
  const float residual = std::sqrt(std::max(1.0f / (error * error) - 1.0f, 0.0f));
  return 2.0f * std::sqrt(std::max(query_norm2, 0.0f)) * norm * residual;
}

float qir_pair_signed_dot(const byte_t* lhs, const byte_t* rhs) {
  i32 same_minus_diff = 0;
  for (u32 bit = 0; bit < VamanaNode::rabitq_code_bits(); ++bit) {
    const bool lhs_positive = (lhs[bit >> 3] & (1u << (7u - (bit & 7u)))) != 0;
    const bool rhs_positive = (rhs[bit >> 3] & (1u << (7u - (bit & 7u)))) != 0;
    same_minus_diff += lhs_positive == rhs_positive ? 1 : -1;
  }
  return static_cast<float>(same_minus_diff);
}

bool same_neighbor_set_prefix(const vec<RemotePtr>& lhs, const vec<RemotePtr>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  vec<u64> lhs_raw;
  vec<u64> rhs_raw;
  lhs_raw.reserve(lhs.size());
  rhs_raw.reserve(rhs.size());
  for (const auto& ptr : lhs) lhs_raw.push_back(ptr.raw_address);
  for (const auto& ptr : rhs) rhs_raw.push_back(ptr.raw_address);
  std::sort(lhs_raw.begin(), lhs_raw.end());
  std::sort(rhs_raw.begin(), rhs_raw.end());
  return lhs_raw == rhs_raw;
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
  const bool remote = !local_shard(rptr.memory_node());
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + header_addr.offset);
    std::atomic_ref<u64> ref(*header_ptr);
    ref.fetch_or(static_cast<u64>(VamanaNode::HEADER_DELETED), std::memory_order_acq_rel);
  } else {
    lock_node(rptr);
    u64 header = 0;
    remote_read_bytes(rptr.memory_node(), header_addr.offset, &header, sizeof(header), 0);
    header |= static_cast<u64>(VamanaNode::HEADER_DELETED);
    remote_write_bytes(rptr.memory_node(), header_addr.offset, &header, sizeof(header), 0);
  }
  if (VamanaNode::compact_storage()) {
    vec<byte_t> entry(VamanaNode::hot_graph_entry_size(), 0);
    VamanaNode::encode_hot_graph_entry(entry.data(), 0, 0, nullptr, 0,
      VamanaNode::HOT_GRAPH_SHARD_BITS, generation, 2, true);
    const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
    if (local_shard(rptr.memory_node())) {
      std::memcpy(index_buffer_.get_full_buffer() + hot_offset, entry.data(), entry.size());
    } else {
      remote_write_bytes(rptr.memory_node(), hot_offset, entry.data(), entry.size(), 0);
    }
  }
  if (remote) {
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
    snapshot.generation = snapshot_generation(ptr, snapshot.header);
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
  snapshot.generation = snapshot_generation(prefix, snapshot.header);
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

auto MemoryNode::async_read_neighbor_list(RemotePtr rptr, StorageOwnerThread& thread) {
  struct Awaitable {
    bool ready{};
    RemotePtr rptr;
    byte_t* buffer{};
    vec<RemotePtr> neighbors;
    MemoryNode* node{};
    StorageOwnerThread* thread{};
    bool hot_graph{};

    bool await_ready() const { return ready; }
    static void await_suspend(std::coroutine_handle<>) {}
    vec<RemotePtr> await_resume() {
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
  };

  if (local_shard(rptr.memory_node())) {
    vec<RemotePtr> neighbors = read_neighbor_list(rptr);
    return Awaitable{true, rptr, nullptr, std::move(neighbors), this, &thread, false};
  }

  byte_t* buffer = thread.coroutine_scratch();
  const auto neighbor_read = vamana::StorageLayoutResolver::neighbor_read(rptr);
  const bool use_hot_graph = neighbor_read.compact;
  post_peer_read_async(thread,
                       rptr.memory_node(),
                       neighbor_read.address.offset,
                       buffer,
                       neighbor_read.address.size);
  return Awaitable{false, rptr, buffer, {}, this, &thread, use_hot_graph};
}

void MemoryNode::init_qir_runtime(const Configuration& config) {
  qir_qcode_cache_capacity_bytes_ =
    static_cast<size_t>(config.qir_cache_mb) * 1024ull * 1024ull;
  qir_effective_exact_budget_.store(std::max<u32>(1, config.qir_exact_budget),
                                    std::memory_order_release);
  const size_t shard_capacity = qir_qcode_cache_capacity_bytes_ / kQirCacheShards;
  for (QirCacheShard& shard : qir_qcode_cache_) {
    std::lock_guard<std::mutex> lock(shard.mutex);
    shard.entries.clear();
    shard.lru.clear();
    shard.bytes = 0;
    shard.capacity = shard_capacity;
  }
}

bool MemoryNode::qir_enabled(const Configuration& config) const {
  return config.use_storage_owner_qir_search() && VamanaNode::HAS_RABITQ_CODE && !config.ip_distance;
}

bool MemoryNode::qir_cache_lookup(RemotePtr rptr,
                                  QirCodeSnapshot& snapshot,
                                  InsertBreakdownCounters* breakdown) {
  if (qir_qcode_cache_capacity_bytes_ == 0 || rptr.is_null()) {
    if (breakdown != nullptr) ++breakdown->qir_qcode_cache_misses;
    return false;
  }
  QirCacheShard& shard = qir_qcode_cache_[(rptr.raw_address >> 6) % kQirCacheShards];
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto it = shard.entries.find(rptr.raw_address);
  if (it == shard.entries.end()) {
    if (breakdown != nullptr) ++breakdown->qir_qcode_cache_misses;
    return false;
  }
  if (std::chrono::steady_clock::now() >= it->second.expires_at) {
    shard.bytes -= it->second.entry.size() + sizeof(QirCacheEntry) + sizeof(u64);
    shard.lru.erase(it->second.lru_it);
    shard.entries.erase(it);
    if (breakdown != nullptr) ++breakdown->qir_qcode_cache_misses;
    return false;
  }
  shard.lru.splice(shard.lru.begin(), shard.lru, it->second.lru_it);
  snapshot.rptr = rptr;
  snapshot.generation = it->second.generation;
  snapshot.deleted = false;
  snapshot.prefix_validated = std::chrono::steady_clock::now() < it->second.prefix_valid_until;
  snapshot.entry = it->second.entry;
  if (breakdown != nullptr) ++breakdown->qir_qcode_cache_hits;
  return true;
}

void MemoryNode::qir_cache_store(const QirCodeSnapshot& snapshot) {
  if (qir_qcode_cache_capacity_bytes_ == 0 || snapshot.rptr.is_null() || snapshot.deleted ||
      snapshot.entry.empty()) {
    return;
  }
  const size_t charge = snapshot.entry.size() + sizeof(QirCacheEntry) + sizeof(u64);
  QirCacheShard& shard = qir_qcode_cache_[(snapshot.rptr.raw_address >> 6) % kQirCacheShards];
  if (charge > shard.capacity) {
    return;
  }
  std::lock_guard<std::mutex> lock(shard.mutex);
  const u64 raw = snapshot.rptr.raw_address;
  const auto now = std::chrono::steady_clock::now();
  const auto expires_at = now + std::chrono::seconds(1);
  const auto prefix_valid_until = snapshot.prefix_validated
    ? now + std::chrono::milliseconds(10)
    : now;
  const auto existing = shard.entries.find(raw);
  if (existing != shard.entries.end()) {
    shard.bytes -= existing->second.entry.size() + sizeof(QirCacheEntry) + sizeof(u64);
    existing->second.generation = snapshot.generation;
    existing->second.entry = snapshot.entry;
    existing->second.expires_at = expires_at;
    existing->second.prefix_valid_until = prefix_valid_until;
    shard.lru.splice(shard.lru.begin(), shard.lru, existing->second.lru_it);
    existing->second.lru_it = shard.lru.begin();
    shard.bytes += charge;
  } else {
    shard.lru.push_front(raw);
    shard.entries.emplace(raw, QirCacheEntry{
      snapshot.generation,
      snapshot.entry,
      expires_at,
      prefix_valid_until,
      shard.lru.begin()});
    shard.bytes += charge;
  }

  while (shard.bytes > shard.capacity && !shard.lru.empty()) {
    const u64 evict = shard.lru.back();
    shard.lru.pop_back();
    const auto evict_it = shard.entries.find(evict);
    if (evict_it == shard.entries.end()) {
      continue;
    }
    shard.bytes -= evict_it->second.entry.size() + sizeof(QirCacheEntry) + sizeof(u64);
    shard.entries.erase(evict_it);
  }
}

void MemoryNode::qir_cache_erase(RemotePtr rptr) {
  if (rptr.is_null()) {
    return;
  }
  QirCacheShard& shard = qir_qcode_cache_[(rptr.raw_address >> 6) % kQirCacheShards];
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto it = shard.entries.find(rptr.raw_address);
  if (it == shard.entries.end()) {
    return;
  }
  shard.bytes -= it->second.entry.size() + sizeof(QirCacheEntry) + sizeof(u64);
  shard.lru.erase(it->second.lru_it);
  shard.entries.erase(it);
}

bool MemoryNode::read_qir_code_snapshot(RemotePtr rptr,
                                        QirCodeSnapshot& snapshot,
                                        const Configuration&,
                                        InsertBreakdownCounters* breakdown,
                                        bool search_phase) {
  if (!VamanaNode::HAS_RABITQ_CODE || rptr.is_null()) {
    return false;
  }
  snapshot = QirCodeSnapshot{};

  if (local_shard(rptr.memory_node())) {
    qir_read_local_code(local_node_ptr(rptr), rptr, snapshot);
    return true;
  }

  if (qir_cache_lookup(rptr, snapshot, breakdown)) {
    return true;
  }

  byte_t prefix[kSnapshotPrefixBytes]{};
  vec<byte_t> entry(VamanaNode::rabitq_entry_size(), 0);
  auto t_read = std::chrono::steady_clock::now();
  remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), prefix, sizeof(prefix), 0);
  remote_read_bytes(rptr.memory_node(),
                    rptr.byte_offset() + VamanaNode::offset_rabitq_code(),
                    entry.data(),
                    entry.size(),
                    align_up(sizeof(prefix)));
  if (breakdown != nullptr) {
    if (search_phase) {
      breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_read);
    } else {
      breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_read);
    }
    breakdown->qir_qcode_rdma_ops += 2;
    breakdown->qir_qcode_rdma_bytes += sizeof(prefix) + entry.size();
  }
  parse_qir_prefix(rptr, prefix, snapshot);
  snapshot.entry = std::move(entry);
  qir_cache_store(snapshot);
  return true;
}

vec<MemoryNode::QirCodeSnapshot> MemoryNode::read_qir_code_snapshots_batched(
    const vec<RemotePtr>& rptrs,
    const Configuration& config,
    InsertBreakdownCounters* breakdown,
    bool search_phase) {
  vec<QirCodeSnapshot> snapshots;
  snapshots.reserve(rptrs.size());
  if (!VamanaNode::HAS_RABITQ_CODE || rptrs.empty()) {
    return snapshots;
  }

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    for (const RemotePtr& rptr : rptrs) {
      QirCodeSnapshot snapshot;
      if (read_qir_code_snapshot(rptr, snapshot, config, breakdown, search_phase)) {
        snapshots.push_back(std::move(snapshot));
      }
    }
    return snapshots;
  }

  struct PendingRead {
    RemotePtr rptr;
    byte_t* buffer{};
  };

  const size_t prefix_size = kSnapshotPrefixBytes;
  const size_t entry_size = VamanaNode::rabitq_entry_size();
  const size_t read_stride = align_up(prefix_size + entry_size);
  const size_t max_batch = std::max<size_t>(1, thread->scratch_stride / read_stride);
  vec<RemotePtr> ordered_rptrs = rptrs;
  std::stable_sort(ordered_rptrs.begin(), ordered_rptrs.end(),
                   [](const RemotePtr& lhs, const RemotePtr& rhs) {
                     return lhs.memory_node() < rhs.memory_node();
                   });

  for (size_t begin = 0; begin < ordered_rptrs.size(); begin += max_batch) {
    const size_t end = std::min(ordered_rptrs.size(), begin + max_batch);
    vec<PendingRead> pending;
    pending.reserve(end - begin);
    u32 remote_slot = 0;
    auto t_read = std::chrono::steady_clock::now();

    for (size_t idx = begin; idx < end; ++idx) {
      const RemotePtr& rptr = ordered_rptrs[idx];
      if (rptr.is_null()) {
        continue;
      }
      if (local_shard(rptr.memory_node())) {
        QirCodeSnapshot snapshot;
        qir_read_local_code(local_node_ptr(rptr), rptr, snapshot);
        snapshots.push_back(std::move(snapshot));
        continue;
      }
      QirCodeSnapshot cached;
      if (qir_cache_lookup(rptr, cached, breakdown)) {
        snapshots.push_back(std::move(cached));
        continue;
      }

      const size_t scratch_offset = static_cast<size_t>(remote_slot) * read_stride;
      lib_assert(scratch_offset + prefix_size + entry_size <= thread->scratch_stride,
                 "storage-owner QIR scratch stride is too small for qcode batch");
      byte_t* buffer = thread->coroutine_scratch(scratch_offset);
      post_peer_read_async(*thread, rptr.memory_node(), rptr.byte_offset(), buffer, prefix_size);
      post_peer_read_async(*thread,
                           rptr.memory_node(),
                           rptr.byte_offset() + VamanaNode::offset_rabitq_code(),
                           buffer + prefix_size,
                           entry_size);
      pending.push_back(PendingRead{rptr, buffer});
      ++remote_slot;
    }

    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }
    if (breakdown != nullptr && !pending.empty()) {
      if (search_phase) {
        breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_read);
      } else {
        breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_read);
      }
      breakdown->qir_qcode_rdma_ops += pending.size() * 2;
      breakdown->qir_qcode_rdma_bytes += pending.size() * (prefix_size + entry_size);
    }
    for (const PendingRead& read : pending) {
      QirCodeSnapshot snapshot;
      parse_qir_prefix(read.rptr, read.buffer, snapshot);
      snapshot.entry.resize(entry_size);
      std::memcpy(snapshot.entry.data(), read.buffer + prefix_size, entry_size);
      qir_cache_store(snapshot);
      snapshots.push_back(std::move(snapshot));
    }
  }

  return snapshots;
}

MemoryNode::QirDistanceInterval MemoryNode::qir_estimate_source_interval(
    const byte_t* source,
    VectorDType source_dtype,
    const QirCodeSnapshot& candidate,
    const Configuration&) const {
  thread_local vec<float> rotated;
  rotated.assign(VamanaNode::rabitq_code_bits(), 0.0f);
  float query_norm2 = 0.0f;
  VamanaNode::compute_rotated_query(source, source_dtype, rotated.data(), &query_norm2);
  const float estimate = vamana::rabitq::estimate_full_entry(
    rotated.data(), query_norm2, candidate.entry.data());
  const float slack = qir_source_interval_slack(query_norm2, candidate.entry.data());
  const float query_norm = std::sqrt(std::max(query_norm2, 0.0f));
  const float candidate_norm = qir_entry_norm(candidate.entry.data());
  const float natural_lower = (query_norm - candidate_norm) * (query_norm - candidate_norm);
  const float natural_upper = (query_norm + candidate_norm) * (query_norm + candidate_norm);
  const float lower = std::min(std::max(estimate - slack, natural_lower), natural_upper);
  return QirDistanceInterval{
    estimate,
    lower,
    std::max(lower, std::min(estimate + slack, natural_upper))};
}

MemoryNode::QirDistanceInterval MemoryNode::qir_estimate_pair_interval(
    const QirCodeSnapshot& lhs,
    const QirCodeSnapshot& rhs) const {
  const float lhs_norm = qir_entry_norm(lhs.entry.data());
  const float rhs_norm = qir_entry_norm(rhs.entry.data());
  const float lhs_error = std::clamp(qir_entry_error(lhs.entry.data()), 1e-6f, 1.0f);
  const float rhs_error = std::clamp(qir_entry_error(rhs.entry.data()), 1e-6f, 1.0f);
  const float signed_dot = qir_pair_signed_dot(lhs.entry.data(), rhs.entry.data());
  const float denom = static_cast<float>(std::max<u32>(1, VamanaNode::rabitq_code_bits())) *
                      lhs_error * rhs_error;
  const float inner_product = lhs_norm * rhs_norm * signed_dot / denom;
  const float estimate = std::max(lhs_norm * lhs_norm + rhs_norm * rhs_norm -
                                  2.0f * inner_product, 0.0f);
  const float lhs_residual =
    std::sqrt(std::max(1.0f / (lhs_error * lhs_error) - 1.0f, 0.0f));
  const float rhs_residual =
    std::sqrt(std::max(1.0f / (rhs_error * rhs_error) - 1.0f, 0.0f));
  const float normalized_slack = std::min(
    lhs_residual / rhs_error + rhs_residual,
    rhs_residual / lhs_error + lhs_residual);
  const float slack = 2.0f * lhs_norm * rhs_norm * normalized_slack;
  const float natural_lower = (lhs_norm - rhs_norm) * (lhs_norm - rhs_norm);
  const float natural_upper = (lhs_norm + rhs_norm) * (lhs_norm + rhs_norm);
  const float lower = std::min(std::max(estimate - slack, natural_lower), natural_upper);
  return QirDistanceInterval{
    estimate,
    lower,
    std::max(lower, std::min(estimate + slack, natural_upper))};
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
  *reinterpret_cast<u64*>(ptr) =
    VamanaNode::compact_storage() ? 0 : static_cast<u64>(generation) << kHeaderGenerationShift;
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

vec<RemotePtr> MemoryNode::beam_search_candidates_qir(const span<const element_t> query,
                                                      RemotePtr medoid,
                                                      const Configuration& config,
                                                      InsertBreakdownCounters* breakdown) {
  if (!qir_enabled(config)) {
    return beam_search_candidates(query, medoid, config, breakdown);
  }

  const u32 construction_width = storage_owner_construction_width(config);
  hashset_t<RemotePtr> visited;
  vec<BeamEntry> beam;
  beam.reserve(construction_width);
  visited.reserve(static_cast<size_t>(construction_width) * std::max<u32>(1, config.R));

  vec<float> rotated_query(VamanaNode::rabitq_code_bits(), 0.0f);
  float query_norm2 = 0.0f;
  VamanaNode::compute_rotated_query(reinterpret_cast<const byte_t*>(query.data()),
                                    VectorDType::float32,
                                    rotated_query.data(),
                                    &query_norm2);
  const auto estimate_from_entry = [&](const byte_t* entry) {
    auto t_distance = std::chrono::steady_clock::now();
    const distance_t distance =
      vamana::rabitq::estimate_full_entry(rotated_query.data(), query_norm2, entry);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
    }
    return distance;
  };

  NodeSnapshot medoid_snapshot;
  auto t_snapshot = std::chrono::steady_clock::now();
  read_node_snapshot(medoid, medoid_snapshot);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
  }
  auto t_distance = std::chrono::steady_clock::now();
  const distance_t medoid_distance =
    distance_to_stored_vector(query, medoid_snapshot.vector_data.data(), config);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
  }
  beam.push_back({medoid, medoid_distance, false});
  visited.insert(medoid);

  for (;;) {
    i32 best_idx = -1;
    distance_t best_distance = std::numeric_limits<distance_t>::max();
    auto t_select = std::chrono::steady_clock::now();
    for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
      if (!beam[i].expanded && beam[i].distance < best_distance) {
        best_distance = beam[i].distance;
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

    vec<RemotePtr> unvisited;
    unvisited.reserve(neighbors.size());
    for (const RemotePtr& neighbor : neighbors) {
      if (neighbor.is_null() || visited.contains(neighbor)) {
        continue;
      }
      visited.insert(neighbor);
      unvisited.push_back(neighbor);
    }

    const u32 qcode_batch = std::max<u32>(1, config.storage_owner_search_snapshot_batch);
    for (size_t begin = 0; begin < unvisited.size(); begin += qcode_batch) {
      const size_t end = std::min(unvisited.size(), begin + qcode_batch);
      vec<RemotePtr> batch(unvisited.begin() + begin, unvisited.begin() + end);
      vec<QirCodeSnapshot> snapshots =
        read_qir_code_snapshots_batched(batch, config, breakdown, true);
      for (const QirCodeSnapshot& snapshot : snapshots) {
        if (snapshot.deleted || snapshot.entry.empty()) {
          continue;
        }
        const distance_t distance = estimate_from_entry(snapshot.entry.data());
        auto t_beam_update = std::chrono::steady_clock::now();
        insert_into_beam(beam, snapshot.rptr, distance, construction_width);
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
        }
      }
    }
  }

  auto t_sort = std::chrono::steady_clock::now();
  std::sort(beam.begin(), beam.end(), [](const BeamEntry& lhs, const BeamEntry& rhs) {
    return lhs.distance < rhs.distance;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_result_sort_ns += elapsed_ns_since(t_sort);
  }
  vec<RemotePtr> candidates;
  candidates.reserve(beam.size());
  for (const BeamEntry& entry : beam) {
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
    vec<RemotePtr> unvisited_neighbors;
    unvisited_neighbors.reserve(neighbors.size());
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

    const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, &thread);
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

vec<RemotePtr> MemoryNode::robust_prune_cpu(const byte_t* source,
                                            VectorDType source_dtype,
                                            const vec<RemotePtr>& candidates,
                                            const hashset_t<RemotePtr>& skip,
                                            const Configuration& config,
                                            InsertBreakdownCounters* breakdown) {
  struct CandidateInfo {
    RemotePtr rptr;
    distance_t dist{};
    vec<byte_t> vector_data;
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

  const u32 snapshot_batch = storage_owner_snapshot_batch_size(config, current_storage_owner_thread_);
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
  std::sort(infos.begin(), infos.end(), [](const CandidateInfo& lhs, const CandidateInfo& rhs) {
    return lhs.dist < rhs.dist;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_prune_sort_ns += elapsed_ns_since(t_sort);
  }

  vec<RemotePtr> selected;
  selected.reserve(config.R);
  vec<const byte_t*> selected_vectors;
  selected_vectors.reserve(config.R);

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

vec<RemotePtr> MemoryNode::robust_prune_qir_cpu(const byte_t* source,
                                                VectorDType source_dtype,
                                                const vec<RemotePtr>& candidates,
                                                const hashset_t<RemotePtr>& skip,
                                                const Configuration& config,
                                                InsertBreakdownCounters* breakdown) {
  if (!qir_enabled(config)) {
    return robust_prune_cpu(source, source_dtype, candidates, skip, config, breakdown);
  }

  struct CandidateInfo {
    QirCodeSnapshot qcode;
    QirDistanceInterval source_interval;
    distance_t source_distance{};
    vec<byte_t> vector_data;
    bool exact{false};
    bool deleted{false};
  };

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
  if (filtered.empty()) {
    return {};
  }

  vec<QirCodeSnapshot> qcodes = read_qir_code_snapshots_batched(filtered, config, breakdown);
  vec<CandidateInfo> infos;
  infos.reserve(qcodes.size());

  vec<float> rotated_source(VamanaNode::rabitq_code_bits(), 0.0f);
  float source_norm2 = 0.0f;
  VamanaNode::compute_rotated_query(source, source_dtype, rotated_source.data(), &source_norm2);
  const auto source_interval_from_entry = [&](const byte_t* entry) {
    const distance_t estimate = vamana::rabitq::estimate_full_entry(
      rotated_source.data(), source_norm2, entry);
    const distance_t slack = qir_source_interval_slack(source_norm2, entry);
    const distance_t source_norm = std::sqrt(std::max(source_norm2, 0.0f));
    const distance_t candidate_norm = qir_entry_norm(entry);
    const distance_t natural_lower =
      (source_norm - candidate_norm) * (source_norm - candidate_norm);
    const distance_t natural_upper =
      (source_norm + candidate_norm) * (source_norm + candidate_norm);
    const distance_t lower =
      std::min(std::max(estimate - slack, natural_lower), natural_upper);
    return QirDistanceInterval{
      estimate,
      lower,
      std::max(lower, std::min(estimate + slack, natural_upper))};
  };

  for (QirCodeSnapshot& snapshot : qcodes) {
    if (snapshot.deleted || snapshot.entry.empty()) {
      continue;
    }
    QirDistanceInterval interval = source_interval_from_entry(snapshot.entry.data());
    infos.push_back(CandidateInfo{
      std::move(snapshot),
      interval,
      interval.estimate,
      {},
      false,
      false});
  }
  if (infos.empty()) {
    return {};
  }

  const u32 exact_budget = std::max<u32>(
    1, qir_effective_exact_budget_.load(std::memory_order_acquire));
  u32 exact_reads = 0;
  bool fallback_required = false;
  const auto request_fallback = [&]() {
    fallback_required = true;
    if (breakdown != nullptr) {
      ++breakdown->qir_prune_fallbacks;
    }
  };
  auto exactify = [&](CandidateInfo& info) -> bool {
    if (info.exact || info.deleted) {
      return !info.deleted;
    }
    if (exact_reads >= exact_budget) {
      return false;
    }
    NodeSnapshot snapshot;
    auto t_snapshot = std::chrono::steady_clock::now();
    const bool read_ok = read_node_snapshot(info.qcode.rptr, snapshot);
    ++exact_reads;
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_snapshot);
      ++breakdown->qir_exact_reads;
    }
    if (!read_ok || snapshot.deleted ||
        snapshot.generation != info.qcode.generation) {
      qir_cache_erase(info.qcode.rptr);
      info.deleted = true;
      return false;
    }
    auto t_distance = std::chrono::steady_clock::now();
    info.source_distance = distance_between_vectors(source, source_dtype,
                                                    snapshot.vector_data.data(),
                                                    VamanaNode::vector_dtype(),
                                                    config);
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_distance_ns += elapsed_ns_since(t_distance);
    }
    info.source_interval = QirDistanceInterval{
      info.source_distance,
      info.source_distance,
      info.source_distance};
    info.vector_data = std::move(snapshot.vector_data);
    info.exact = true;
    return true;
  };

  const auto is_fresh = [&](CandidateInfo& info) {
    if (info.exact || info.qcode.prefix_validated) {
      return true;
    }
    byte_t prefix[kSnapshotPrefixBytes]{};
    if (local_shard(info.qcode.rptr.memory_node())) {
      std::memcpy(prefix, local_node_ptr(info.qcode.rptr), sizeof(prefix));
    } else {
      auto t_snapshot = std::chrono::steady_clock::now();
      remote_read_bytes(info.qcode.rptr.memory_node(),
                        info.qcode.rptr.byte_offset(),
                        prefix,
                        sizeof(prefix),
                        0);
      if (breakdown != nullptr) {
        breakdown->storage_owner_prune_snapshot_read_ns += elapsed_ns_since(t_snapshot);
        ++breakdown->qir_qcode_rdma_ops;
        breakdown->qir_qcode_rdma_bytes += sizeof(prefix);
      }
    }
    QirCodeSnapshot current;
    parse_qir_prefix(info.qcode.rptr, prefix, current);
    if (current.deleted || current.generation != info.qcode.generation) {
      qir_cache_erase(info.qcode.rptr);
      info.deleted = true;
      return false;
    }
    return true;
  };

  auto t_sort = std::chrono::steady_clock::now();
  std::sort(infos.begin(), infos.end(), [](const CandidateInfo& lhs, const CandidateInfo& rhs) {
    if (lhs.source_distance != rhs.source_distance) {
      return lhs.source_distance < rhs.source_distance;
    }
    return lhs.qcode.rptr.raw_address < rhs.qcode.rptr.raw_address;
  });
  if (breakdown != nullptr) {
    breakdown->storage_owner_prune_sort_ns += elapsed_ns_since(t_sort);
  }

  // Exactify candidates whose source intervals overlap the top-R boundary.
  // If the overlap set does not fit in the budget, exact prune is safer than
  // mixing incomparable approximate and exact ordering keys.
  const size_t boundary_index = std::min<size_t>(config.R, infos.size()) - 1;
  const QirDistanceInterval boundary = infos[boundary_index].source_interval;
  vec<size_t> boundary_candidates;
  boundary_candidates.reserve(infos.size());
  for (size_t i = 0; i < infos.size(); ++i) {
    const auto& interval = infos[i].source_interval;
    if (interval.lower <= boundary.upper && interval.upper >= boundary.lower) {
      boundary_candidates.push_back(i);
    }
  }
  const u32 top_exact = std::min<u32>(std::max<u32>(1, exact_budget / 4), infos.size());
  for (u32 i = 0; i < top_exact; ++i) {
    if (!exactify(infos[i]) && !infos[i].deleted) {
      request_fallback();
      break;
    }
  }
  for (const size_t i : boundary_candidates) {
    if (fallback_required || infos[i].exact) {
      continue;
    }
    if (!exactify(infos[i]) && !infos[i].deleted) {
      request_fallback();
      break;
    }
  }
  if (fallback_required) {
    return robust_prune_cpu(source, source_dtype, candidates, skip, config, breakdown);
  }
  if (exact_reads > 0) {
    t_sort = std::chrono::steady_clock::now();
    std::sort(infos.begin(), infos.end(), [](const CandidateInfo& lhs, const CandidateInfo& rhs) {
      if (lhs.deleted != rhs.deleted) return !lhs.deleted;
      if (lhs.source_distance != rhs.source_distance) {
        return lhs.source_distance < rhs.source_distance;
      }
      return lhs.qcode.rptr.raw_address < rhs.qcode.rptr.raw_address;
    });
    if (breakdown != nullptr) {
      breakdown->storage_owner_prune_sort_ns += elapsed_ns_since(t_sort);
    }
  }

  vec<size_t> selected_indices;
  selected_indices.reserve(config.R);
  u64 uncertain_count = 0;
  u64 evaluated_count = 0;

  for (size_t idx = 0; idx < infos.size(); ++idx) {
    CandidateInfo& candidate = infos[idx];
    if (selected_indices.size() >= config.R) {
      break;
    }
    if (candidate.deleted) {
      continue;
    }
    ++evaluated_count;

    const auto classify = [&](vec<size_t>* uncertain_selected) {
      bool pruned = false;
      bool uncertain = false;
      for (const size_t selected_idx : selected_indices) {
        CandidateInfo& selected = infos[selected_idx];
        auto t_pair_distance = std::chrono::steady_clock::now();
        if (candidate.exact && selected.exact) {
          const distance_t pair_distance =
            distance_between_vectors(candidate.vector_data.data(),
                                     VamanaNode::vector_dtype(),
                                     selected.vector_data.data(),
                                     VamanaNode::vector_dtype(),
                                     config);
          if (breakdown != nullptr) {
            breakdown->storage_owner_prune_pair_distance_ns +=
              elapsed_ns_since(t_pair_distance);
          }
          if (config.alpha * pair_distance <= candidate.source_distance) {
            pruned = true;
            break;
          }
          continue;
        }

        const QirDistanceInterval pair_interval =
          qir_estimate_pair_interval(selected.qcode, candidate.qcode);
        if (breakdown != nullptr) {
          breakdown->storage_owner_prune_pair_distance_ns +=
            elapsed_ns_since(t_pair_distance);
        }
        if (config.alpha * pair_interval.upper <= candidate.source_interval.lower) {
          pruned = true;
          break;
        }
        if (config.alpha * pair_interval.lower <= candidate.source_interval.upper) {
          uncertain = true;
          if (uncertain_selected != nullptr) {
            uncertain_selected->push_back(selected_idx);
          }
        }
      }
      return std::pair{pruned, uncertain};
    };

    vec<size_t> uncertain_selected;
    auto [pruned, uncertain] = classify(&uncertain_selected);
    if (pruned) {
      continue;
    }

    if (uncertain) {
      ++uncertain_count;
      if (!candidate.exact && exact_reads < exact_budget) {
        exactify(candidate);
      }
      for (const size_t selected_idx : uncertain_selected) {
        if (exact_reads >= exact_budget) {
          break;
        }
        exactify(infos[selected_idx]);
      }
      if (candidate.deleted) {
        continue;
      }
      std::tie(pruned, uncertain) = classify(nullptr);
      if (pruned) {
        continue;
      }
      if (uncertain) {
        request_fallback();
        break;
      }
    }
    if (!is_fresh(candidate)) {
      continue;
    }
    selected_indices.push_back(idx);
  }

  const double uncertain_ratio = evaluated_count == 0
    ? 0.0
    : static_cast<double>(uncertain_count) / static_cast<double>(evaluated_count);
  if (breakdown != nullptr) {
    breakdown->qir_uncertain_candidates += uncertain_count;
  }
  if (fallback_required || uncertain_ratio > config.qir_uncertain_ratio_threshold) {
    if (!fallback_required && breakdown != nullptr) {
      ++breakdown->qir_prune_fallbacks;
    }
    return robust_prune_cpu(source, source_dtype, candidates, skip, config, breakdown);
  }
  if (breakdown != nullptr && infos.size() > exact_reads) {
    breakdown->qir_exact_reads_avoided += infos.size() - exact_reads;
  }
  vec<RemotePtr> selected;
  selected.reserve(config.R);
  for (const size_t idx : selected_indices) {
    selected.push_back(infos[idx].qcode.rptr);
    if (selected.size() >= config.R) {
      break;
    }
  }
  return selected;
}

bool MemoryNode::maybe_audit_qir_prune(const byte_t* source,
                                       VectorDType source_dtype,
                                       const vec<RemotePtr>& candidates,
                                       const hashset_t<RemotePtr>& skip,
                                       const vec<RemotePtr>& selected,
                                       const Configuration& config,
                                       InsertBreakdownCounters& breakdown) {
  if (!qir_enabled(config)) {
    return false;
  }

  ++breakdown.qir_audit_samples;
  const vec<RemotePtr> exact = robust_prune_cpu(source, source_dtype, candidates, skip, config, nullptr);
  if (same_neighbor_set_prefix(selected, exact)) {
    return false;
  }
  ++breakdown.qir_audit_disagreements;
  u32 current = qir_effective_exact_budget_.load(std::memory_order_acquire);
  const u32 max_budget = std::max<u32>(config.R, storage_owner_prune_candidate_limit(config));
  while (current < max_budget) {
    const u32 raised = std::min<u32>(
      std::max<u32>(current + 1, current * 2), max_budget);
    if (qir_effective_exact_budget_.compare_exchange_weak(
          current, raised, std::memory_order_acq_rel, std::memory_order_acquire)) {
      break;
    }
  }
  return true;
}


auto MemoryNode::beam_search_candidates_qir_async(const span<const element_t> query,
                                                  RemotePtr medoid,
                                                  const Configuration& config,
                                                  StorageOwnerThread& thread,
                                                  InsertBreakdownCounters* breakdown)
  -> StorageOwnerInsertCoroutine {
  if (!qir_enabled(config)) {
    auto fallback = beam_search_candidates_async(query, medoid, config, thread, breakdown);
    co_await std::suspend_always{};
    while (!fallback.handle.done()) {
      if (thread.is_ready(thread.running_coroutine)) {
        fallback.handle.resume();
      } else {
        co_await std::suspend_always{};
      }
    }
    fallback.handle.destroy();
    co_return;
  }

  const u32 construction_width = storage_owner_construction_width(config);
  vec<BeamEntry> beam;
  beam.reserve(construction_width);
  hashset_t<RemotePtr> visited;
  visited.reserve(static_cast<size_t>(construction_width) * std::max<u32>(1, config.R));

  vec<float> rotated_query(VamanaNode::rabitq_code_bits(), 0.0f);
  float query_norm2 = 0.0f;
  VamanaNode::compute_rotated_query(reinterpret_cast<const byte_t*>(query.data()),
                                    VectorDType::float32,
                                    rotated_query.data(),
                                    &query_norm2);

  const auto estimate_from_entry = [&](const byte_t* entry) {
    auto t_distance = std::chrono::steady_clock::now();
    const distance_t dist = vamana::rabitq::estimate_full_entry(
      rotated_query.data(), query_norm2, entry);
    if (breakdown != nullptr) {
      breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
    }
    return dist;
  };

  auto t_snapshot = std::chrono::steady_clock::now();
  NodeSnapshot medoid_snapshot = co_await async_read_node_snapshot(medoid, thread);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_snapshot);
  }
  auto t_distance = std::chrono::steady_clock::now();
  const distance_t medoid_dist = distance_to_stored_vector(
    query, medoid_snapshot.vector_data.data(), config);
  if (breakdown != nullptr) {
    breakdown->storage_owner_search_distance_ns += elapsed_ns_since(t_distance);
  }
  beam.push_back({medoid, medoid_dist, false});
  visited.insert(medoid);

  const size_t prefix_size = kSnapshotPrefixBytes;
  const size_t qir_entry_size = VamanaNode::rabitq_entry_size();
  const size_t qir_entry_stride = align_up(prefix_size + qir_entry_size);
  const u32 qir_batch = std::max<u32>(1, static_cast<u32>(std::min<size_t>(
    std::max<u32>(1, config.storage_owner_search_snapshot_batch),
    std::max<size_t>(1, thread.scratch_stride / qir_entry_stride))));

  struct PendingQirRead {
    RemotePtr rptr;
    byte_t* buffer{};
  };

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
    std::stable_sort(unvisited_neighbors.begin(), unvisited_neighbors.end(),
                     [](const RemotePtr& lhs, const RemotePtr& rhs) {
                       return lhs.memory_node() < rhs.memory_node();
                     });

    for (size_t begin = 0; begin < unvisited_neighbors.size(); begin += qir_batch) {
      const size_t end = std::min(unvisited_neighbors.size(), begin + qir_batch);
      vec<PendingQirRead> pending;
      pending.reserve(end - begin);
      auto t_qcode_read = std::chrono::steady_clock::now();
      u32 remote_slot = 0;

      for (size_t idx = begin; idx < end; ++idx) {
        const RemotePtr& candidate = unvisited_neighbors[idx];
        QirCodeSnapshot cached;
        if (local_shard(candidate.memory_node())) {
          qir_read_local_code(local_node_ptr(candidate), candidate, cached);
          if (!cached.deleted) {
            const distance_t dist = estimate_from_entry(cached.entry.data());
            auto t_beam_update = std::chrono::steady_clock::now();
            insert_into_beam(beam, candidate, dist, construction_width);
            if (breakdown != nullptr) {
              breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
            }
          }
          continue;
        }
        if (qir_cache_lookup(candidate, cached, breakdown)) {
          const distance_t dist = estimate_from_entry(cached.entry.data());
          auto t_beam_update = std::chrono::steady_clock::now();
          insert_into_beam(beam, candidate, dist, construction_width);
          if (breakdown != nullptr) {
            breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
          }
          continue;
        }

        const size_t scratch_offset = static_cast<size_t>(remote_slot) * qir_entry_stride;
        lib_assert(scratch_offset + prefix_size + qir_entry_size <= thread.scratch_stride,
                   "storage-owner QIR scratch stride is too small: offset=" +
                   std::to_string(scratch_offset) +
                   " entry=" + std::to_string(qir_entry_size) +
                   " stride=" + std::to_string(thread.scratch_stride));
        byte_t* buffer = thread.coroutine_scratch(scratch_offset);
        post_peer_read_async(thread, candidate.memory_node(), candidate.byte_offset(), buffer, prefix_size);
        post_peer_read_async(thread,
                             candidate.memory_node(),
                             candidate.byte_offset() + VamanaNode::offset_rabitq_code(),
                             buffer + prefix_size,
                             qir_entry_size);
        pending.push_back(PendingQirRead{candidate, buffer});
        ++remote_slot;
      }

      if (!pending.empty()) {
        while (!thread.is_ready(thread.running_coroutine)) {
          co_await std::suspend_always{};
        }
        if (breakdown != nullptr) {
          breakdown->storage_owner_search_snapshot_read_ns += elapsed_ns_since(t_qcode_read);
          breakdown->qir_qcode_rdma_ops += pending.size() * 2;
          breakdown->qir_qcode_rdma_bytes += pending.size() * (prefix_size + qir_entry_size);
        }
        for (const PendingQirRead& read : pending) {
          QirCodeSnapshot snapshot;
          parse_qir_prefix(read.rptr, read.buffer, snapshot);
          snapshot.entry.resize(qir_entry_size);
          std::memcpy(snapshot.entry.data(), read.buffer + prefix_size, qir_entry_size);
          qir_cache_store(snapshot);
          if (snapshot.deleted) {
            continue;
          }
          const distance_t dist = estimate_from_entry(snapshot.entry.data());
          auto t_beam_update = std::chrono::steady_clock::now();
          insert_into_beam(beam, snapshot.rptr, dist, construction_width);
          if (breakdown != nullptr) {
            breakdown->storage_owner_search_beam_update_ns += elapsed_ns_since(t_beam_update);
          }
        }
      }
    }
  }

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
  co_return;
}

auto MemoryNode::execute_storage_owner_insert_job_async(StorageOwnerThread& thread,
                                            StorageOwnerInsertJob& job,
                                            std::unordered_map<u64, vec<service::storage_owner::ReverseUpdateOp>>& local_updates,
                                            std::unordered_map<u32, vec<service::storage_owner::ReverseUpdateOp>>& remote_updates,
                                            InsertBreakdownCounters& breakdown,
                                            const Configuration& config) -> StorageOwnerInsertCoroutine {
  const auto components = span<const element_t>{reinterpret_cast<const element_t*>(job.vector_data.data()),
                                                 VamanaNode::DIM};
  FreshnessEntry old_entry{};
  u32 generation = 0;
  const auto status = prepare_mutation(job.id, job.kind, &old_entry, &generation);
  job.old_ptr = old_entry.current;
  job.generation = generation;
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
      qir_cache_erase(old_entry.current);
      publish_mutation(job.id, old_entry.current, old_entry.generation, true);
    }
    co_return;
  }
  auto t_medoid = std::chrono::steady_clock::now();
  RemotePtr medoid_ptr = co_await async_read_global_medoid(thread);
  breakdown.storage_owner_medoid_ns += elapsed_ns_since(t_medoid);
  if (medoid_ptr.is_null()) {
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
        qir_cache_erase(old_entry.current);
      }
      publish_mutation(job.id, new_ptr, generation, false);
      co_return;
    }
    medoid_ptr = observed;
  }

  bool audit_qir = false;
  if (qir_enabled(config) && config.use_storage_owner_qir_prune()) {
    const u64 sequence = qir_insert_sequence_.fetch_add(1, std::memory_order_acq_rel) + 1;
    audit_qir = config.qir_audit_rate != 0 && sequence % config.qir_audit_rate == 0;
  }

  auto t_search = std::chrono::steady_clock::now();
  if (config.use_storage_owner_qir_search()) {
    auto search = beam_search_candidates_qir_async(
      components, medoid_ptr, config, thread, &breakdown);
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
  } else {
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
  }

  const vec<RemotePtr> candidates =
    storage_owner_async_candidates_[thread.id][thread.running_coroutine];
  hashset_t<RemotePtr> empty_skip;
  auto t_prune = std::chrono::steady_clock::now();
  vec<RemotePtr> selected_neighbors = config.use_storage_owner_qir_prune()
    ? robust_prune_qir_cpu(reinterpret_cast<const byte_t*>(components.data()),
                           VectorDType::float32, candidates, empty_skip, config, &breakdown)
    : robust_prune_cpu(reinterpret_cast<const byte_t*>(components.data()),
                       VectorDType::float32, candidates, empty_skip, config, &breakdown);
  if (selected_neighbors.empty() && !medoid_ptr.is_null()) {
    selected_neighbors.push_back(medoid_ptr);
  }
  breakdown.storage_owner_prune_ns += elapsed_ns_since(t_prune);
  const u64 source_insert_id = qir_enabled(config)
    ? qir_next_insert_id_.fetch_add(1, std::memory_order_acq_rel)
    : 0;
  const RemotePtr new_ptr = allocate_local_node();
  job.new_ptr = new_ptr;
  auto t_write = std::chrono::steady_clock::now();
  write_new_node(new_ptr, job.id, components, selected_neighbors, generation);
  breakdown.storage_owner_write_node_ns += elapsed_ns_since(t_write);
  RemotePtr anchor_target;
  if (qir_enabled(config) && config.storage_owner_reverse_mode == "async" &&
      !install_qir_reachability_anchor(
        new_ptr, generation, selected_neighbors, config, &anchor_target)) {
    mark_node_deleted(new_ptr, generation);
    job.status = service::storage_owner::MutationStatus::failed;
    job.ok = false;
    co_return;
  }
  if (job.kind == service::storage_owner::MutationKind::upsert && !old_entry.deleted) {
    mark_node_deleted(old_entry.current, old_entry.generation);
    qir_cache_erase(old_entry.current);
  }
  publish_mutation(job.id, new_ptr, generation, false);
  if (audit_qir) {
    QirAuditTask audit;
    audit.source.assign(components.begin(), components.end());
    audit.medoid = medoid_ptr;
    audit.selected = selected_neighbors;
    (void)enqueue_qir_audit(std::move(audit));
  }

  for (const RemotePtr& neighbor_ptr : selected_neighbors) {
    if (neighbor_ptr == anchor_target) {
      continue;
    }
    service::storage_owner::ReverseUpdateOp op{
      neighbor_ptr.raw_address,
      new_ptr.raw_address,
      generation,
      0,
      source_insert_id};
    if (local_shard(neighbor_ptr.memory_node())) {
      local_updates[neighbor_ptr.raw_address].push_back(op);
    } else {
      remote_updates[neighbor_ptr.memory_node()].push_back(op);
    }
    if (qir_enabled(config)) {
      ++breakdown.qir_repair_intents;
    }
  }
  job.ok = true;
  job.status = service::storage_owner::MutationStatus::ok;
}

bool MemoryNode::apply_local_reverse_update(RemotePtr target_ptr,
                                const vec<service::storage_owner::ReverseUpdateOp>& ops,
                                const Configuration& config) {
  lib_assert(local_shard(target_ptr.memory_node()), "target reverse update must be local");
  if (ops.empty()) {
    return true;
  }
  const bool requires_reachability = std::any_of(ops.begin(), ops.end(), [](const auto& op) {
    return (op.reserved & service::storage_owner::kReverseUpdateReachability) != 0;
  });

  const auto update_started = std::chrono::steady_clock::now();
  const auto lock_started = std::chrono::steady_clock::now();
  lock_node(target_ptr);
  const u64 lock_wait_ns = elapsed_ns_since(lock_started);
  vec<RemotePtr> updated_neighbors;
  bool changed = false;
  bool pruned = false;
  size_t current_count = 0;
  size_t filtered_count = 0;
  size_t stale_count = 0;
  RemotePtr reachability_anchor;
  u64 snapshot_ns = 0;
  u64 neighbor_read_ns = 0;
  u64 filter_ns = 0;
  u64 prune_ns = 0;
  u64 write_ns = 0;

  NodeSnapshot target_snapshot;
  auto step_started = std::chrono::steady_clock::now();
  read_node_snapshot(target_ptr, target_snapshot);
  snapshot_ns = elapsed_ns_since(step_started);
  if (target_snapshot.deleted) {
    unlock_node(target_ptr);
    return !requires_reachability;
  }

  step_started = std::chrono::steady_clock::now();
  vec<RemotePtr> current_neighbors = read_neighbor_list(target_ptr);
  neighbor_read_ns = elapsed_ns_since(step_started);
  current_count = current_neighbors.size();
  const u32 target_generation = target_snapshot.generation;
  unlock_node(target_ptr);

  step_started = std::chrono::steady_clock::now();
  vec<RemotePtr> filtered_candidates;
  filtered_candidates.reserve(ops.size());
  vec<NodeSnapshot> filtered_candidate_snapshots;
  filtered_candidate_snapshots.reserve(ops.size());
  hashset_t<RemotePtr> qir_candidates;
  qir_candidates.reserve(ops.size());
  for (const auto& op : ops) {
    const RemotePtr candidate_ptr{op.candidate_raw};
    if (candidate_ptr.is_null()) {
      continue;
    }
    NodeSnapshot candidate_snapshot;
    if (!read_node_snapshot(candidate_ptr, candidate_snapshot) ||
        candidate_snapshot.deleted ||
        candidate_snapshot.generation != op.candidate_generation) {
      if (op.source_insert_id != 0) {
        ++stale_count;
      }
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
      filtered_candidate_snapshots.push_back(std::move(candidate_snapshot));
    }
    if ((op.reserved & service::storage_owner::kReverseUpdateReachability) != 0) {
      reachability_anchor = candidate_ptr;
    }
    if (!already_present && op.source_insert_id != 0) {
      qir_candidates.insert(candidate_ptr);
    }
  }
  filter_ns = elapsed_ns_since(step_started);
  filtered_count = filtered_candidates.size();
  if (requires_reachability && reachability_anchor.is_null()) {
    return false;
  }

  const auto merge_reverse_candidates = [&](const NodeSnapshot& target,
                                            const vec<RemotePtr>& base_neighbors) {
    vec<RemotePtr> merged = base_neighbors;
    for (const RemotePtr candidate : filtered_candidates) {
      if (std::find(merged.begin(), merged.end(), candidate) == merged.end()) {
        merged.push_back(candidate);
      }
    }
    if (merged.size() <= config.R) {
      return merged;
    }
    if (qir_enabled(config)) {
      hashset_t<RemotePtr> empty_skip;
      return robust_prune_cpu(target.vector_data.data(),
                              VamanaNode::vector_dtype(),
                              merged,
                              empty_skip,
                              config,
                              nullptr);
    }

    // Preserve the original storage-owner O(R) reverse-update policy for the
    // exact baseline. QIR alone pays for full exact RobustPrune repair.
    vec<RemotePtr> result;
    result.reserve(config.R);
    for (const RemotePtr neighbor : base_neighbors) {
      if (!neighbor.is_null() && result.size() < config.R) {
        result.push_back(neighbor);
      }
    }
    vec<distance_t> distances;
    distances.reserve(result.size());
    const vec<NodeSnapshot> base_snapshots = read_node_snapshots_batched(result, config);
    result.clear();
    for (const NodeSnapshot& snapshot : base_snapshots) {
      if (snapshot.deleted) {
        continue;
      }
      result.push_back(snapshot.rptr);
      distances.push_back(distance_between_vectors(target.vector_data.data(),
                                                   VamanaNode::vector_dtype(),
                                                   snapshot.vector_data.data(),
                                                   VamanaNode::vector_dtype(),
                                                   config));
    }
    for (const NodeSnapshot& snapshot : filtered_candidate_snapshots) {
      if (snapshot.deleted) {
        continue;
      }
      const distance_t candidate_distance = distance_between_vectors(
        target.vector_data.data(),
        VamanaNode::vector_dtype(),
        snapshot.vector_data.data(),
        VamanaNode::vector_dtype(),
        config);
      if (result.size() < config.R) {
        result.push_back(snapshot.rptr);
        distances.push_back(candidate_distance);
        continue;
      }
      lib_assert(!distances.empty(), "reverse-update distances must be non-empty");
      const auto farthest = std::max_element(distances.begin(), distances.end());
      if (candidate_distance < *farthest) {
        const size_t index = static_cast<size_t>(std::distance(distances.begin(), farthest));
        result[index] = snapshot.rptr;
        distances[index] = candidate_distance;
      }
    }
    return result;
  };

  if (!filtered_candidates.empty()) {
    changed = true;
    pruned = current_neighbors.size() + filtered_candidates.size() > config.R;
    step_started = std::chrono::steady_clock::now();
    updated_neighbors = merge_reverse_candidates(target_snapshot, current_neighbors);
    prune_ns = elapsed_ns_since(step_started);
    if (!reachability_anchor.is_null() &&
        std::find(updated_neighbors.begin(), updated_neighbors.end(), reachability_anchor) ==
          updated_neighbors.end()) {
      if (updated_neighbors.size() < config.R) {
        updated_neighbors.push_back(reachability_anchor);
      } else {
        updated_neighbors.back() = reachability_anchor;
      }
    }
  }

  if (changed) {
    lock_node(target_ptr);
    NodeSnapshot commit_snapshot;
    read_node_snapshot(target_ptr, commit_snapshot);
    if (commit_snapshot.deleted || commit_snapshot.generation != target_generation) {
      unlock_node(target_ptr);
      return !requires_reachability;
    }
    const vec<RemotePtr> latest_neighbors = read_neighbor_list(target_ptr);
    if (latest_neighbors != current_neighbors) {
      updated_neighbors = merge_reverse_candidates(commit_snapshot, latest_neighbors);
      if (!reachability_anchor.is_null() &&
          std::find(updated_neighbors.begin(), updated_neighbors.end(), reachability_anchor) ==
            updated_neighbors.end()) {
        if (updated_neighbors.size() < config.R) {
          updated_neighbors.push_back(reachability_anchor);
        } else {
          updated_neighbors.back() = reachability_anchor;
        }
      }
    }
    step_started = std::chrono::steady_clock::now();
    write_neighbor_list(target_ptr, updated_neighbors);
    write_ns = elapsed_ns_since(step_started);
    unlock_node(target_ptr);
  }

  if (stale_count > 0) {
    qir_repair_stale_skips_total_.fetch_add(stale_count, std::memory_order_relaxed);
  }
  if (changed) {
    u64 applied = 0;
    for (const RemotePtr& candidate : filtered_candidates) {
      if (qir_candidates.contains(candidate) &&
          std::find(updated_neighbors.begin(), updated_neighbors.end(), candidate) !=
          updated_neighbors.end()) {
        ++applied;
      }
    }
    qir_repair_applied_edges_total_.fetch_add(applied, std::memory_order_relaxed);
  }

  const u64 update_ns = elapsed_ns_since(update_started);
  if (update_ns > 1000ull * 1000ull * 1000ull) {
    static std::atomic<u32> slow_update_logs{0};
    const u32 log_index = slow_update_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_index < 16) {
      std::cerr << "[storage-owner] slow reverse-update target"
                << " self_shard=" << storage_id_
                << " target_raw=" << target_ptr.raw_address
                << " candidates=" << ops.size()
                << " current_neighbors=" << current_count
                << " filtered_candidates=" << filtered_count
                << " stale_candidates=" << stale_count
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

bool MemoryNode::install_qir_reachability_anchor(
    RemotePtr new_ptr,
    u32 generation,
    const vec<RemotePtr>& selected_neighbors,
    const Configuration& config,
    RemotePtr* anchor_target) {
  if (anchor_target != nullptr) {
    *anchor_target = RemotePtr{};
  }
  if (selected_neighbors.empty()) {
    return false;
  }

  vec<RemotePtr> ordered = selected_neighbors;
  std::stable_partition(ordered.begin(), ordered.end(), [&](RemotePtr candidate) {
    return local_shard(candidate.memory_node());
  });
  for (const RemotePtr anchor : ordered) {
    const service::storage_owner::ReverseUpdateOp op{
      anchor.raw_address,
      new_ptr.raw_address,
      generation,
      service::storage_owner::kReverseUpdatePriority |
        service::storage_owner::kReverseUpdateReachability,
      qir_next_insert_id_.fetch_add(1, std::memory_order_acq_rel)};
    const vec<service::storage_owner::ReverseUpdateOp> ops{op};
    const bool success = local_shard(anchor.memory_node())
      ? apply_local_reverse_update(anchor, ops, config)
      : send_reverse_update_batch_direct(anchor.memory_node(), ops, true, config);
    if (!success) {
      continue;
    }
    if (anchor_target != nullptr) {
      *anchor_target = anchor;
    }
    return true;
  }
  return false;
}
