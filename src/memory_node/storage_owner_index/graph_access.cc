#include "memory_node/storage_owner_index/detail.hh"
#include "memory_node/storage_owner_index/locked_node_publication.hh"
#include "memory_node/storage_owner_index/vector_snapshot_policy.hh"

using namespace memory_node_storage_owner_index_detail;

void MemoryNode::report_rejected_graph_pointer(
    const char* boundary,
    RemotePtr pointer,
    RemotePtr parent,
    u64 context) const {
  static std::atomic<u64> rejected{0};
  const u64 count = rejected.fetch_add(1, std::memory_order_relaxed) + 1;
  // Malformed pointers are never a normal churn outcome. Keep diagnostics
  // useful without turning a damaged edge into an unbounded logging attack.
  if (count <= 16 || (count & (count - 1)) == 0) {
    std::cerr << "[storage-owner] rejected malformed graph pointer"
              << " boundary=" << boundary
              << " raw=0x" << std::hex << pointer.raw_address << std::dec
              << " shard=" << pointer.memory_node()
              << " offset=" << pointer.byte_offset()
              << " incarnation=" << pointer.incarnation()
              << " configured_shards=" << num_storage_nodes_
              << " shard_bytes=" << mn_memory_bytes_;
    if (!parent.is_null()) {
      std::cerr << " parent_raw=0x" << std::hex << parent.raw_address
                << std::dec
                << " parent_shard=" << parent.memory_node()
                << " parent_offset=" << parent.byte_offset()
                << " parent_incarnation=" << parent.incarnation();
    }
    if (context != std::numeric_limits<u64>::max()) {
      std::cerr << " context=" << context;
    }
    std::cerr
              << " count=" << count << '\n';
  }
}

IncarnationLockResult MemoryNode::try_lock_node(RemotePtr rptr) {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_ ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return IncarnationLockResult::stale;
  }
  const auto header_address = vamana::StorageLayoutResolver::header(rptr);
  if (header_address.offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_address.offset) {
    return IncarnationLockResult::stale;
  }

  // A local validity check rejects unallocated dynamic addresses and malformed
  // static/dynamic tag combinations.  It is only a fast prefilter: the CAS
  // below is the actual race-closing identity boundary.
  if (local_shard(rptr.memory_node()) &&
      !valid_local_storage_node_pointer(rptr)) {
    return IncarnationLockResult::stale;
  }

  constexpr u32 kMaxAttempts = 8;
  for (u32 attempt = 0; attempt < kMaxAttempts; ++attempt) {
    IncarnationLockResult result = IncarnationLockResult::busy;
    if (local_shard(rptr.memory_node())) {
      auto* header = reinterpret_cast<u64*>(
        index_buffer_.get_full_buffer() + header_address.offset);
      result = try_lock_header_once(*header, rptr.incarnation());
    } else {
      const auto [locked, observed] = try_lock_remote_header(rptr);
      if (locked) {
        result = IncarnationLockResult::locked;
      } else if (VamanaNode::header_incarnation(observed) !=
                 rptr.incarnation()) {
        result = IncarnationLockResult::stale;
      }
    }
    if (result != IncarnationLockResult::busy) return result;
    std::this_thread::yield();
  }
  return IncarnationLockResult::busy;
}

bool MemoryNode::read_locked_node_identity(RemotePtr rptr,
                                           u64& header,
                                           node_t& id,
                                           u32& generation) {
  header = 0;
  id = 0;
  generation = 0;
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_ ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  constexpr size_t identity_bytes =
    memory_node_storage_owner_index_detail::kLockedNodeIdentityBytes;
  if (rptr.byte_offset() > mn_memory_bytes_ ||
      identity_bytes > mn_memory_bytes_ - rptr.byte_offset()) {
    return false;
  }

  memory_node_storage_owner_index_detail::LockedNodeIdentity identity;
  const bool valid = memory_node_storage_owner_index_detail::
    read_and_validate_locked_node_identity(
      rptr,
      [&](byte_t* destination, size_t bytes) {
        if (local_shard(rptr.memory_node())) {
          const u64 local_header = load_local_node_header_acquire(rptr);
          std::memcpy(destination, &local_header, sizeof(local_header));
          std::memcpy(destination + VamanaNode::HEADER_SIZE,
                      index_buffer_.get_full_buffer() +
                        rptr.byte_offset() + VamanaNode::HEADER_SIZE,
                      VamanaNode::COMPACT_META_SIZE);
          std::atomic_thread_fence(std::memory_order_acquire);
        } else {
          remote_read_bytes(rptr.memory_node(), rptr.byte_offset(),
                            destination, bytes, 0);
        }
        return true;
      },
      identity);
  header = identity.header;
  id = identity.id;
  generation = identity.generation;
  return valid;
}

bool MemoryNode::publish_locked_node_header(RemotePtr rptr,
                                            u64 observed_header,
                                            u64 set_flags,
                                            u64 clear_flags) {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_ ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  const u64 header_offset =
    vamana::StorageLayoutResolver::header(rptr).offset;
  if (header_offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_offset) {
    return false;
  }
  if (local_shard(rptr.memory_node())) {
    auto* storage = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + header_offset);
    std::atomic_ref<u64> header_ref(*storage);
    return memory_node_storage_owner_index_detail::
      publish_locked_node_header_transition(
        rptr, observed_header, set_flags, clear_flags,
        [&](u64 expected, u64 desired) {
          const u64 original = expected;
          if (header_ref.compare_exchange_strong(
                expected, desired, std::memory_order_release,
                std::memory_order_acquire)) {
            return original;
          }
          return expected;
        });
  }
  return memory_node_storage_owner_index_detail::
    publish_locked_node_header_transition(
      rptr, observed_header, set_flags, clear_flags,
      [&](u64 expected, u64 desired) {
        return remote_compare_and_swap(
          rptr.memory_node(), header_offset, expected, desired,
          align_up(sizeof(expected)));
      });
}

bool MemoryNode::storage_node_pointer_addressable(RemotePtr rptr) const {
  return memory_node_storage_owner_index_detail::storage_pointer_addressable(
    rptr, num_storage_nodes_, mn_memory_bytes_);
}

bool MemoryNode::read_node_snapshot(RemotePtr rptr, NodeSnapshot& snapshot) {
  const auto clear_snapshot = [&]() {
    snapshot.rptr.reset();
    snapshot.header = 0;
    snapshot.id = 0;
    snapshot.generation = 0;
    snapshot.slot_incarnation = 0;
    snapshot.deleted = false;
    snapshot.vector_data.clear();
  };
  if (!storage_node_pointer_addressable(rptr)) {
    if (!rptr.is_null()) {
      report_rejected_graph_pointer("read_node_snapshot", rptr);
    }
    clear_snapshot();
    return false;
  }
  const auto vector_addr = vamana::StorageLayoutResolver::vector(rptr);
  clear_snapshot();
  snapshot.rptr = rptr;
  snapshot.vector_data.resize(VamanaNode::vector_bytes());

  constexpr u32 kMaxReadAttempts = 3;
  if (local_shard(rptr.memory_node())) {
    const byte_t* base = index_buffer_.get_full_buffer();
    const byte_t* ptr = base + rptr.byte_offset();
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(rptr);
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0 ||
          VamanaNode::header_incarnation(before) != rptr.incarnation()) {
        std::this_thread::yield();
        continue;
      }
      snapshot.id = *reinterpret_cast<const u32*>(
        ptr + VamanaNode::offset_id());
      snapshot.generation = *reinterpret_cast<const u32*>(
        ptr + VamanaNode::offset_generation());
      snapshot.slot_incarnation = *reinterpret_cast<const u32*>(
        ptr + VamanaNode::offset_slot_incarnation());
      std::memcpy(snapshot.vector_data.data(), base + vector_addr.offset,
                  VamanaNode::vector_bytes());
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(rptr);
      if (before == after &&
          snapshot.slot_incarnation == rptr.incarnation()) {
        snapshot.header = after;
        snapshot.deleted =
          (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
        return true;
      }
      std::this_thread::yield();
    }
    clear_snapshot();
    return false;
  }

  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                          ? owner_thread->coroutine_scratch()
                          : peer_scratch_buffer_.get_full_buffer();
  for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
    remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), read_buffer,
                      VamanaNode::size_until_vector_end(), 0);
    if (!parse_remote_snapshot(rptr, read_buffer, snapshot)) {
      std::this_thread::yield();
      continue;
    }
    u64 after = 0;
    remote_read_bytes(rptr.memory_node(), rptr.byte_offset(), &after,
                      sizeof(after), 0);
    if (snapshot.header == after &&
        (after & VamanaNode::HEADER_NODE_LOCK) == 0 &&
        VamanaNode::header_incarnation(after) == rptr.incarnation()) {
      return true;
    }
    std::this_thread::yield();
  }
  clear_snapshot();
  return false;
}

bool MemoryNode::valid_local_storage_node_pointer(RemotePtr rptr) const {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      !local_shard(rptr.memory_node()) ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  const auto header_address = vamana::StorageLayoutResolver::header(rptr);
  if (header_address.offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_address.offset) {
    return false;
  }
  if (rptr.byte_offset() < gpu_dynamic_node_base_) {
    return rptr.incarnation() == 0 &&
      VamanaNode::header_incarnation(
        load_local_node_header_acquire(rptr)) == 0 &&
      *reinterpret_cast<const u32*>(
        index_buffer_.get_full_buffer() + rptr.byte_offset() +
          VamanaNode::offset_slot_incarnation()) == 0;
  }
  if (rptr.incarnation() == 0) return false;
  const auto* control = reinterpret_cast<const
    gpu_search::format::StorageControlBlock*>(
      index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const u64 high_watermark = std::atomic_ref<const u64>(
    control->dynamic_high_watermark).load(std::memory_order_acquire);
  const u64 node_bytes = VamanaNode::allocation_size();
  if (rptr.byte_offset() > high_watermark ||
      node_bytes > high_watermark - rptr.byte_offset()) {
    return false;
  }
  const byte_t* node = index_buffer_.get_full_buffer() +
    rptr.byte_offset();
  return VamanaNode::header_incarnation(
           load_local_node_header_acquire(rptr)) == rptr.incarnation() &&
    *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_slot_incarnation()) == rptr.incarnation();
}

bool MemoryNode::storage_owner_node_live(RemotePtr rptr) {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_) {
    return false;
  }
  if (!VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  if (local_shard(rptr.memory_node()) &&
      !valid_local_storage_node_pointer(rptr)) {
    return false;
  }
  const auto header_address = vamana::StorageLayoutResolver::header(rptr);
  if (header_address.offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_address.offset) {
    return false;
  }

  byte_t identity[VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE]{};
  u64 after = 0;
  if (local_shard(rptr.memory_node())) {
    const u64 before = load_local_node_header_acquire(rptr);
    std::memcpy(identity, &before, sizeof(before));
    std::memcpy(identity + VamanaNode::HEADER_SIZE,
                index_buffer_.get_full_buffer() + rptr.byte_offset() +
                  VamanaNode::HEADER_SIZE,
                VamanaNode::COMPACT_META_SIZE);
    std::atomic_thread_fence(std::memory_order_acquire);
    after = load_local_node_header_acquire(rptr);
  } else {
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      identity, sizeof(identity), 0);
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      &after, sizeof(after), 0);
  }
  const u64 before = *reinterpret_cast<const u64*>(identity);
  const u32 incarnation = *reinterpret_cast<const u32*>(
    identity + VamanaNode::offset_slot_incarnation());
  return before == after &&
    (after & (VamanaNode::HEADER_NODE_LOCK |
              VamanaNode::HEADER_DELETED)) == 0 &&
    VamanaNode::header_incarnation(after) == rptr.incarnation() &&
    incarnation == rptr.incarnation();
}

bool MemoryNode::storage_owner_node_stable(RemotePtr rptr) {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_ ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  if (local_shard(rptr.memory_node()) &&
      !valid_local_storage_node_pointer(rptr)) {
    return false;
  }
  const auto header_address = vamana::StorageLayoutResolver::header(rptr);
  if (header_address.offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_address.offset) {
    return false;
  }

  byte_t identity[VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE]{};
  u64 after = 0;
  if (local_shard(rptr.memory_node())) {
    const u64 before = load_local_node_header_acquire(rptr);
    std::memcpy(identity, &before, sizeof(before));
    std::memcpy(identity + VamanaNode::HEADER_SIZE,
                index_buffer_.get_full_buffer() + rptr.byte_offset() +
                  VamanaNode::HEADER_SIZE,
                VamanaNode::COMPACT_META_SIZE);
    std::atomic_thread_fence(std::memory_order_acquire);
    after = load_local_node_header_acquire(rptr);
  } else {
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      identity, sizeof(identity), 0);
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      &after, sizeof(after), 0);
  }
  const u64 before = *reinterpret_cast<const u64*>(identity);
  const u32 incarnation = *reinterpret_cast<const u32*>(
    identity + VamanaNode::offset_slot_incarnation());
  return before == after &&
    (after & (VamanaNode::HEADER_NODE_LOCK |
              VamanaNode::HEADER_DELETED |
              VamanaNode::HEADER_PROVISIONAL)) == 0 &&
    VamanaNode::header_incarnation(after) == rptr.incarnation() &&
    incarnation == rptr.incarnation();
}

bool MemoryNode::read_stable_node_identity(RemotePtr rptr) {
  if (rptr.is_null() || !rptr.is_well_formed() ||
      rptr.memory_node() >= num_storage_nodes_ ||
      !VamanaNode::hot_graph_entry_available(rptr)) {
    return false;
  }
  const auto header_address = vamana::StorageLayoutResolver::header(rptr);
  if (header_address.offset > mn_memory_bytes_ ||
      VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE >
        mn_memory_bytes_ - header_address.offset) {
    return false;
  }

  constexpr u32 kMaxReadAttempts = 3;
  if (local_shard(rptr.memory_node())) {
    const byte_t* record = index_buffer_.get_full_buffer() +
      rptr.byte_offset();
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(rptr);
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0 ||
          VamanaNode::header_incarnation(before) != rptr.incarnation()) {
        std::this_thread::yield();
        continue;
      }
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        record + VamanaNode::offset_slot_incarnation());
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(rptr);
      if (before == after && slot_incarnation == rptr.incarnation()) {
        return stable_vector_snapshot_valid(
          rptr, before, after, slot_incarnation);
      }
      std::this_thread::yield();
    }
    return false;
  }

  byte_t identity[VamanaNode::HEADER_SIZE +
                  VamanaNode::COMPACT_META_SIZE]{};
  for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      identity, sizeof(identity), 0);
    const u64 before = *reinterpret_cast<const u64*>(identity);
    const u32 slot_incarnation = *reinterpret_cast<const u32*>(
      identity + VamanaNode::offset_slot_incarnation());
    if ((before & VamanaNode::HEADER_NODE_LOCK) != 0 ||
        VamanaNode::header_incarnation(before) != rptr.incarnation() ||
        slot_incarnation != rptr.incarnation()) {
      std::this_thread::yield();
      continue;
    }
    u64 after = 0;
    remote_read_bytes(rptr.memory_node(), header_address.offset,
                      &after, sizeof(after), 0);
    if (before == after) {
      return stable_vector_snapshot_valid(
        rptr, before, after, slot_incarnation);
    }
    std::this_thread::yield();
  }
  return false;
}

size_t MemoryNode::read_node_identity_headers_batched_into(
    span<const RemotePtr> rptrs,
    const Configuration& config,
    vec<std::pair<RemotePtr, u64>>& identities,
    vec<StableNodeSnapshotState>* states) {
  identities.reserve(rptrs.size());
  size_t identity_count = 0;
  if (states != nullptr) {
    states->assign(rptrs.size(), StableNodeSnapshotState::terminal);
  }
  if (rptrs.empty()) return identity_count;

  const auto set_state = [&](size_t input_index,
                             StableNodeSnapshotState state) {
    if (states != nullptr) (*states)[input_index] = state;
  };
  const auto next_identity = [&](size_t input_index,
                                 RemotePtr pointer,
                                 u64 header) {
    if (identity_count == identities.size()) {
      identities.emplace_back(pointer, header);
    } else {
      identities[identity_count] = {pointer, header};
    }
    set_state(input_index, StableNodeSnapshotState::stable);
    ++identity_count;
  };
  const auto pointer_valid = [&](RemotePtr pointer) {
    if (pointer.is_null() || !pointer.is_well_formed() ||
        pointer.memory_node() >= num_storage_nodes_ ||
        !VamanaNode::hot_graph_entry_available(pointer)) {
      return false;
    }
    const auto header = vamana::StorageLayoutResolver::header(pointer);
    constexpr size_t kIdentityBytes =
      VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;
    return header.offset <= mn_memory_bytes_ &&
      kIdentityBytes <= mn_memory_bytes_ - header.offset;
  };
  const auto read_local = [&](size_t input_index, RemotePtr pointer) {
    const byte_t* record = index_buffer_.get_full_buffer() +
      pointer.byte_offset();
    constexpr u32 kMaxReadAttempts = 3;
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(pointer);
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        record + VamanaNode::offset_slot_incarnation());
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(pointer);
      const StableNodeSnapshotState disposition =
        classify_physical_node_snapshot(
          pointer, before, after, slot_incarnation);
      if (disposition == StableNodeSnapshotState::stable) {
        next_identity(input_index, pointer, before);
        return;
      }
      if (disposition == StableNodeSnapshotState::terminal) {
        return;
      }
      std::this_thread::yield();
    }
    set_state(input_index, StableNodeSnapshotState::retryable);
  };

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    constexpr size_t kIdentityBytes =
      VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;
    byte_t identity[kIdentityBytes]{};
    for (size_t input_index = 0; input_index < rptrs.size(); ++input_index) {
      const RemotePtr pointer = rptrs[input_index];
      if (!pointer_valid(pointer)) continue;
      if (local_shard(pointer.memory_node())) {
        read_local(input_index, pointer);
        continue;
      }
      constexpr u32 kMaxReadAttempts = 3;
      bool resolved = false;
      for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
        remote_read_bytes(pointer.memory_node(), pointer.byte_offset(),
                          identity, sizeof(identity), 0);
        const u64 before = *reinterpret_cast<const u64*>(identity);
        const u32 slot_incarnation = *reinterpret_cast<const u32*>(
          identity + VamanaNode::offset_slot_incarnation());
        u64 after = 0;
        remote_read_bytes(pointer.memory_node(), pointer.byte_offset(),
                          &after, sizeof(after), 0);
        const StableNodeSnapshotState disposition =
          classify_physical_node_snapshot(
            pointer, before, after, slot_incarnation);
        if (disposition == StableNodeSnapshotState::stable) {
          next_identity(input_index, pointer, before);
          resolved = true;
          break;
        }
        if (disposition == StableNodeSnapshotState::terminal) {
          resolved = true;
          break;
        }
        std::this_thread::yield();
      }
      if (!resolved) {
        set_state(input_index, StableNodeSnapshotState::retryable);
      }
    }
    return identity_count;
  }

  struct PendingIdentityRead {
    size_t input_index{};
    RemotePtr pointer;
    byte_t* buffer{};
    u64 before{};
    u32 slot_incarnation{};
  };
  constexpr size_t kIdentityBytes =
    VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;
  const size_t identity_stride = align_up(kIdentityBytes);
  // A compact header permits far more records to fit in scratch than a full
  // vector snapshot, but scratch capacity is not a transport credit. Keep the
  // wave inside the configured/tested request window; the linked-read batch
  // still stripes bounded chains over the available data QPs.
  const size_t max_batch = std::max<size_t>(1, std::min<size_t>(
    thread->scratch_stride / identity_stride,
    storage_owner_snapshot_batch_size(config, thread)));
  thread_local vec<PendingIdentityRead> pending;
  thread_local vec<PeerReadRequest> read_requests;
  pending.reserve(max_batch);

  for (size_t begin = 0; begin < rptrs.size(); begin += max_batch) {
    const size_t end = std::min(rptrs.size(), begin + max_batch);
    pending.clear();
    read_requests.clear();
    size_t remote_slot = 0;
    for (size_t index = begin; index < end; ++index) {
      const RemotePtr pointer = rptrs[index];
      if (!pointer_valid(pointer)) continue;
      if (local_shard(pointer.memory_node())) {
        read_local(index, pointer);
        continue;
      }
      const size_t scratch_offset = remote_slot++ * identity_stride;
      lib_assert(scratch_offset + kIdentityBytes <= thread->scratch_stride,
                 "storage-owner scratch cannot hold identity wave");
      byte_t* buffer = thread->coroutine_scratch(scratch_offset);
      read_requests.push_back(PeerReadRequest{
        .shard_id = pointer.memory_node(),
        .remote_offset = pointer.byte_offset(),
        .destination = buffer,
        .bytes = kIdentityBytes,
      });
      pending.push_back({index, pointer, buffer});
    }
    post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }

    read_requests.clear();
    for (PendingIdentityRead& read : pending) {
      read.before = *reinterpret_cast<const u64*>(read.buffer);
      read.slot_incarnation = *reinterpret_cast<const u32*>(
        read.buffer + VamanaNode::offset_slot_incarnation());
      read_requests.push_back(PeerReadRequest{
        .shard_id = read.pointer.memory_node(),
        .remote_offset = read.pointer.byte_offset(),
        .destination = read.buffer,
        .bytes = VamanaNode::HEADER_SIZE,
      });
    }
    post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }
    for (const PendingIdentityRead& read : pending) {
      const u64 after = *reinterpret_cast<const u64*>(read.buffer);
      const StableNodeSnapshotState disposition =
        classify_physical_node_snapshot(
          read.pointer, read.before, after, read.slot_incarnation);
      if (disposition == StableNodeSnapshotState::stable) {
        next_identity(read.input_index, read.pointer, read.before);
      } else {
        set_state(read.input_index, disposition);
      }
    }
  }
  return identity_count;
}

bool MemoryNode::read_graph_adjacency(RemotePtr rptr,
                                      GraphAdjacency& adjacency) {
  adjacency.stable.clear();
  adjacency.provisional.clear();
  adjacency.generation = 0;
  adjacency.deleted = false;
  if (!storage_node_pointer_addressable(rptr)) {
    if (!rptr.is_null()) {
      report_rejected_graph_pointer("read_graph_adjacency/input", rptr);
    }
    return false;
  }
  thread_local vec<byte_t> local_entry;
  StorageOwnerThread* owner_thread = current_storage_owner_thread_;
  byte_t* read_buffer = nullptr;
  if (local_shard(rptr.memory_node())) {
    local_entry.resize(VamanaNode::hot_graph_entry_size());
    read_buffer = local_entry.data();
  } else {
    read_buffer = owner_thread != nullptr && owner_thread->has_peer_scratch()
                    ? owner_thread->coroutine_scratch()
                    : peer_scratch_buffer_.get_full_buffer();
  }
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
    const u8 stable_count = read_buffer[0];
    const u8 provisional_count =
      vamana::hot_graph::provisional_count(read_buffer);
    const u16 expected = vamana::hot_graph::load_u16_le(read_buffer + 2);
    const u16 actual = vamana::hot_graph::checksum16(
      read_buffer, VamanaNode::hot_graph_entry_size());
    const bool entry_ok = stable_count <= VamanaNode::R &&
      provisional_count <= VamanaNode::provisional_slots() &&
      static_cast<u32>(stable_count) + provisional_count <=
        VamanaNode::graph_entry_capacity() &&
      (read_buffer[1] & 0x0eu) == 0 &&
      vamana::hot_graph::load_u32_le(read_buffer + 8) ==
        rptr.incarnation() &&
      vamana::hot_graph::load_u32_le(read_buffer + 12) == 0 &&
      expected == actual;
    if (!entry_ok) {
      std::this_thread::yield();
      continue;
    }

    adjacency.stable.clear();
    adjacency.provisional.clear();
    adjacency.generation = vamana::hot_graph::load_u32_le(read_buffer + 4);
    adjacency.deleted =
      (read_buffer[1] & VamanaNode::HOT_GRAPH_DELETED) != 0;
    adjacency.stable.reserve(stable_count);
    adjacency.provisional.reserve(provisional_count);
    bool malformed_neighbor = false;
    for (u32 index = 0; index < stable_count; ++index) {
      const RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
        read_buffer + vamana::hot_graph::neighbor_offset(index),
        VamanaNode::HOT_GRAPH_SHARD_BITS);
      if (neighbor.is_null()) {
        // The count is part of the checksummed publication. A null inside
        // that counted prefix is structural corruption, not unused capacity.
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacency/stable_null", neighbor, rptr, index);
        continue;
      }
      if (!storage_node_pointer_addressable(neighbor)) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacency/stable", neighbor, rptr, index);
        continue;
      }
      adjacency.stable.push_back(neighbor);
    }
    for (u32 index = 0; index < provisional_count; ++index) {
      const RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
        read_buffer + vamana::hot_graph::neighbor_offset(stable_count + index),
        VamanaNode::HOT_GRAPH_SHARD_BITS);
      if (neighbor.is_null()) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacency/provisional_null", neighbor, rptr, index);
        continue;
      }
      if (!storage_node_pointer_addressable(neighbor)) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacency/provisional", neighbor, rptr, index);
        continue;
      }
      adjacency.provisional.push_back(neighbor);
    }
    // A malformed pointer can be a transient torn graph snapshot, so retry
    // optimistic reads. It must never be accepted after filtering: mutation
    // maintenance could otherwise acknowledge a graph with silently missing
    // authoritative edges. Repeated checksum-consistent damage is terminal.
    if (!malformed_neighbor) return true;
    if (attempt + 1 == kMaxReadAttempts) {
      lib_failure(
        "persistent malformed counted graph edge: parent_raw=" +
        std::to_string(rptr.raw_address));
    }
    std::this_thread::yield();
  }
  adjacency.stable.clear();
  adjacency.provisional.clear();
  adjacency.generation = 0;
  adjacency.deleted = false;
  return false;
}

vec<std::pair<RemotePtr, MemoryNode::GraphAdjacency>>
MemoryNode::read_graph_adjacencies_batched(
    span<const RemotePtr> rptrs,
    const Configuration& config) {
  vec<std::pair<RemotePtr, GraphAdjacency>> results;
  const size_t result_count =
    read_graph_adjacencies_batched_into(rptrs, config, results);
  results.resize(result_count);
  return results;
}

size_t MemoryNode::read_graph_adjacencies_batched_into(
    span<const RemotePtr> rptrs,
    const Configuration& config,
    vec<std::pair<RemotePtr, GraphAdjacency>>& results) {
  results.reserve(rptrs.size());
  size_t result_count = 0;
  if (rptrs.empty()) return result_count;

  const auto next_result = [&]() -> std::pair<RemotePtr, GraphAdjacency>& {
    if (result_count == results.size()) {
      results.emplace_back();
    }
    return results[result_count];
  };

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    for (const RemotePtr rptr : rptrs) {
      auto& slot = next_result();
      if (read_graph_adjacency(rptr, slot.second)) {
        slot.first = rptr;
        ++result_count;
      }
    }
    return result_count;
  }

  struct PendingRead {
    RemotePtr rptr;
    byte_t* buffer{};
  };

  enum class GraphDecodeResult : u8 {
    invalid_snapshot,
    valid,
    malformed_pointer,
  };
  const auto decode = [&](RemotePtr rptr, const byte_t* entry,
                          GraphAdjacency& adjacency) {
    adjacency.stable.clear();
    adjacency.provisional.clear();
    adjacency.generation = 0;
    adjacency.deleted = false;
    const u8 stable_count = entry[0];
    const u8 provisional_count =
      vamana::hot_graph::provisional_count(entry);
    const u16 expected = vamana::hot_graph::load_u16_le(entry + 2);
    const u16 actual = vamana::hot_graph::checksum16(
      entry, VamanaNode::hot_graph_entry_size());
    if (stable_count > VamanaNode::R ||
        provisional_count > VamanaNode::provisional_slots() ||
        static_cast<u32>(stable_count) + provisional_count >
          VamanaNode::graph_entry_capacity() ||
        (entry[1] & 0x0eu) != 0 ||
        vamana::hot_graph::load_u32_le(entry + 8) !=
          rptr.incarnation() ||
        vamana::hot_graph::load_u32_le(entry + 12) != 0 ||
        expected != actual) {
      return GraphDecodeResult::invalid_snapshot;
    }
    adjacency.generation = vamana::hot_graph::load_u32_le(entry + 4);
    adjacency.deleted =
      (entry[1] & VamanaNode::HOT_GRAPH_DELETED) != 0;
    adjacency.stable.reserve(stable_count);
    adjacency.provisional.reserve(provisional_count);
    bool malformed_neighbor = false;
    for (u32 index = 0; index < stable_count; ++index) {
      const RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
        entry + vamana::hot_graph::neighbor_offset(index),
        VamanaNode::HOT_GRAPH_SHARD_BITS);
      if (neighbor.is_null()) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacencies_batched/stable_null", neighbor, rptr,
          index);
        continue;
      }
      if (!storage_node_pointer_addressable(neighbor)) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacencies_batched/stable", neighbor, rptr, index);
        continue;
      }
      adjacency.stable.push_back(neighbor);
    }
    for (u32 index = 0; index < provisional_count; ++index) {
      const RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
        entry + vamana::hot_graph::neighbor_offset(stable_count + index),
        VamanaNode::HOT_GRAPH_SHARD_BITS);
      if (neighbor.is_null()) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacencies_batched/provisional_null", neighbor,
          rptr, index);
        continue;
      }
      if (!storage_node_pointer_addressable(neighbor)) {
        malformed_neighbor = true;
        report_rejected_graph_pointer(
          "read_graph_adjacencies_batched/provisional", neighbor, rptr,
          index);
        continue;
      }
      adjacency.provisional.push_back(neighbor);
    }
    return malformed_neighbor ? GraphDecodeResult::malformed_pointer
                              : GraphDecodeResult::valid;
  };

  const size_t scratch_stride = aligned_graph_entry_bytes();
  const size_t max_batch = storage_owner_graph_batch_size(config, thread);
  constexpr u32 kMaxReadAttempts = 3;
  thread_local vec<PendingRead> pending;
  thread_local vec<PendingRead> retry;
  thread_local vec<PeerReadRequest> read_requests;
  pending.reserve(max_batch);
  retry.reserve(max_batch);
  for (size_t begin = 0; begin < rptrs.size(); begin += max_batch) {
    const size_t end = std::min(rptrs.size(), begin + max_batch);
    pending.clear();
    u32 remote_slot = 0;
    for (size_t index = begin; index < end; ++index) {
      const RemotePtr rptr = rptrs[index];
      if (!storage_node_pointer_addressable(rptr)) {
        if (!rptr.is_null()) {
          report_rejected_graph_pointer(
            "read_graph_adjacencies_batched/input", rptr);
        }
        continue;
      }
      if (local_shard(rptr.memory_node())) {
        auto& slot = next_result();
        if (read_graph_adjacency(rptr, slot.second)) {
          slot.first = rptr;
          ++result_count;
        }
        continue;
      }
      const size_t scratch_offset =
        static_cast<size_t>(remote_slot++) * scratch_stride;
      lib_assert(scratch_offset + VamanaNode::hot_graph_entry_size() <=
                   thread->scratch_stride,
                 "storage-owner scratch cannot hold batched graph reads");
      pending.push_back({rptr, thread->coroutine_scratch(scratch_offset)});
    }

    for (u32 attempt = 0;
         attempt < kMaxReadAttempts && !pending.empty(); ++attempt) {
      read_requests.clear();
      for (const PendingRead& read : pending) {
        read_requests.push_back(PeerReadRequest{
          .shard_id = read.rptr.memory_node(),
          .remote_offset = VamanaNode::hot_graph_entry_offset(read.rptr),
          .destination = read.buffer,
          .bytes = VamanaNode::hot_graph_entry_size(),
        });
      }
      post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
      while (!thread->is_ready(thread->running_coroutine)) {
        poll_peer_send_cq();
        std::this_thread::yield();
      }

      retry.clear();
      for (const PendingRead& read : pending) {
        auto& slot = next_result();
        const GraphDecodeResult decoded =
          decode(read.rptr, read.buffer, slot.second);
        if (decoded == GraphDecodeResult::valid) {
          slot.first = read.rptr;
          ++result_count;
        } else if (decoded == GraphDecodeResult::malformed_pointer &&
                   attempt + 1 == kMaxReadAttempts) {
          lib_failure(
            "persistent malformed counted graph edge in batched read: "
            "parent_raw=" + std::to_string(read.rptr.raw_address));
        } else if (attempt + 1 < kMaxReadAttempts) {
          retry.push_back(read);
        }
      }
      pending.swap(retry);
    }
  }
  return result_count;
}

vec<RemotePtr> MemoryNode::read_neighbor_list(RemotePtr rptr) {
  GraphAdjacency adjacency;
  if (!read_graph_adjacency(rptr, adjacency) || adjacency.deleted) {
    return {};
  }
  vec<RemotePtr> neighbors;
  neighbors.reserve(adjacency.stable.size() + adjacency.provisional.size());
  neighbors.insert(neighbors.end(), adjacency.stable.begin(),
                   adjacency.stable.end());
  neighbors.insert(neighbors.end(), adjacency.provisional.begin(),
                   adjacency.provisional.end());
  return neighbors;
}

vec<RemotePtr> MemoryNode::read_stable_neighbor_list(RemotePtr rptr) {
  GraphAdjacency adjacency;
  if (!read_graph_adjacency(rptr, adjacency) || adjacency.deleted) {
    return {};
  }
  return std::move(adjacency.stable);
}

bool MemoryNode::read_local_neighbor_list(RemotePtr rptr,
                                          vec<RemotePtr>& neighbors,
                                          vec<byte_t>& entry,
                                          vec<byte_t>& decoded) const {
  lib_assert(local_shard(rptr.memory_node()),
             "local neighbor lookup received a remote pointer");
  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
  const size_t entry_size = VamanaNode::hot_graph_entry_size();
  lib_assert(hot_offset + entry_size <= mn_memory_bytes_,
             "local neighbor lookup exceeds shard bounds");

  entry.resize(entry_size);
  decoded.resize(VamanaNode::neighbor_read_size());
  neighbors.clear();
  constexpr u32 kMaxReadAttempts = 3;
  bool decoded_ok = false;
  for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
    std::memcpy(entry.data(), index_buffer_.get_full_buffer() + hot_offset,
                entry_size);
    decoded_ok = VamanaNode::decode_hot_graph_entry(
      entry.data(), decoded.data(), rptr.incarnation());
    if (decoded_ok) {
      break;
    }
    std::this_thread::yield();
  }
  if (!decoded_ok) {
    return false;
  }

  const u32 edge_count = VamanaNode::decoded_neighbor_count(decoded.data());
  const auto* slots = reinterpret_cast<const RemotePtr*>(
    decoded.data() + VamanaNode::neighbor_payload_offset_in_read());
  neighbors.reserve(edge_count);
  for (u32 index = 0; index < edge_count &&
                       index < VamanaNode::graph_entry_capacity(); ++index) {
    const RemotePtr neighbor = slots[index];
    if (neighbor.is_null()) continue;
    if (!storage_node_pointer_addressable(neighbor)) {
      report_rejected_graph_pointer(
        "read_local_neighbor_list", neighbor, rptr, index);
      continue;
    }
    neighbors.push_back(neighbor);
  }
  return true;
}

vec<MemoryNode::NodeSnapshot> MemoryNode::read_node_snapshots_batched(
    const vec<RemotePtr>& rptrs,
    const Configuration& config,
    const char* boundary,
    vec<StableNodeSnapshotState>* states) {
  vec<NodeSnapshot> snapshots;
  const size_t snapshot_count = read_node_snapshots_batched_into(
    span<const RemotePtr>{rptrs}, config, snapshots, boundary, states);
  snapshots.resize(snapshot_count);
  return snapshots;
}

size_t MemoryNode::read_node_snapshots_batched_into(
    span<const RemotePtr> rptrs,
    const Configuration& config,
    vec<NodeSnapshot>& snapshots,
    const char* boundary,
    vec<StableNodeSnapshotState>* states) {
  snapshots.reserve(rptrs.size());
  size_t snapshot_count = 0;
  if (states != nullptr) {
    states->assign(rptrs.size(), StableNodeSnapshotState::terminal);
  }
  if (rptrs.empty()) {
    return snapshot_count;
  }

  const auto set_state = [&](size_t input_index,
                             StableNodeSnapshotState state) {
    if (states != nullptr) (*states)[input_index] = state;
  };

  // If a complete D-byte optimistic snapshot did not stabilize, a compact
  // identity observation distinguishes a stable replacement from transient
  // lock/torn contention. A current identity still means retryable here: the
  // caller requested the vector payload and must never receive a fabricated
  // or partially observed one.
  const auto classify_failed_snapshot = [&](size_t input_index,
                                            RemotePtr pointer) {
    vec<std::pair<RemotePtr, u64>> identity;
    vec<StableNodeSnapshotState> identity_state;
    read_node_identity_headers_batched_into(
      span<const RemotePtr>{&pointer, 1}, config, identity, &identity_state);
    lib_assert(identity_state.size() == 1,
               "single snapshot retry classification lost input alignment");
    set_state(
      input_index,
      identity_state.front() == StableNodeSnapshotState::stable
        ? StableNodeSnapshotState::retryable
        : identity_state.front());
  };

  const auto next_snapshot = [&]() -> NodeSnapshot& {
    if (snapshot_count == snapshots.size()) snapshots.emplace_back();
    return snapshots[snapshot_count];
  };

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    for (size_t input_index = 0; input_index < rptrs.size(); ++input_index) {
      const RemotePtr rptr = rptrs[input_index];
      NodeSnapshot& snapshot = next_snapshot();
      if (!rptr.is_null() && read_node_snapshot(rptr, snapshot)) {
        set_state(input_index, StableNodeSnapshotState::stable);
        ++snapshot_count;
      } else if (!rptr.is_null() && storage_node_pointer_addressable(rptr)) {
        classify_failed_snapshot(input_index, rptr);
      }
    }
    return snapshot_count;
  }

  struct PendingRead {
    size_t input_index{};
    RemotePtr rptr;
    byte_t* buffer{};
    u64 before{};
    u32 slot_incarnation{};
  };

  const size_t snapshot_size = snapshot_buffer_bytes();
  const size_t snapshot_stride = aligned_snapshot_bytes();
  const size_t max_batch = storage_owner_snapshot_batch_size(config, thread);
  thread_local vec<PeerReadRequest> read_requests;

  for (size_t begin = 0; begin < rptrs.size(); begin += max_batch) {
    const size_t end = std::min(rptrs.size(), begin + max_batch);
    thread_local vec<PendingRead> pending;
    pending.reserve(end - begin);
    pending.clear();
    read_requests.clear();
    u32 remote_slot = 0;

    for (size_t idx = begin; idx < end; ++idx) {
      const RemotePtr& rptr = rptrs[idx];
      if (rptr.is_null()) {
        continue;
      }

      if (!storage_node_pointer_addressable(rptr)) {
        report_rejected_graph_pointer(boundary, rptr);
        continue;
      }

      if (local_shard(rptr.memory_node())) {
        NodeSnapshot& snapshot = next_snapshot();
        if (read_node_snapshot(rptr, snapshot)) {
          set_state(idx, StableNodeSnapshotState::stable);
          ++snapshot_count;
        } else {
          classify_failed_snapshot(idx, rptr);
        }
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
      read_requests.push_back(PeerReadRequest{
        .shard_id = rptr.memory_node(),
        .remote_offset = rptr.byte_offset(),
        .destination = buffer,
        .bytes = VamanaNode::size_until_vector_end(),
      });
      pending.push_back(PendingRead{idx, rptr, buffer});
      ++remote_slot;
    }

    post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }

    // A structurally verified base record has an immutable identity/vector
    // payload, so its first unlocked body header is already a linearization
    // point. Dynamic slots can be recycled while this READ crosses the body;
    // retain their second header read after saving the first identity fields.
    // Equality closes that overwrite window without another scratch plane.
    read_requests.clear();
    size_t dynamic_count = 0;
    for (PendingRead& read : pending) {
      read.before = *reinterpret_cast<const u64*>(read.buffer);
      read.slot_incarnation = *reinterpret_cast<const u32*>(
        read.buffer + VamanaNode::offset_slot_incarnation());
      if (VamanaNode::immutable_base_record(read.rptr)) {
        const StableNodeSnapshotState disposition =
          classify_physical_node_snapshot(
            read.rptr, read.before, read.before,
            read.slot_incarnation);
        if (disposition == StableNodeSnapshotState::stable) {
          NodeSnapshot& snapshot = next_snapshot();
          lib_assert(parse_remote_snapshot(
                       read.rptr, read.buffer, snapshot),
                     "immutable base snapshot failed to parse");
          set_state(read.input_index, StableNodeSnapshotState::stable);
          ++snapshot_count;
        } else {
          set_state(read.input_index, disposition);
        }
        continue;
      }
      if (dynamic_count !=
          static_cast<size_t>(&read - pending.data())) {
        pending[dynamic_count] = read;
      }
      PendingRead& dynamic = pending[dynamic_count++];
      read_requests.push_back(PeerReadRequest{
        .shard_id = dynamic.rptr.memory_node(),
        .remote_offset = dynamic.rptr.byte_offset(),
        .destination = dynamic.buffer,
        .bytes = VamanaNode::HEADER_SIZE,
      });
    }
    pending.resize(dynamic_count);
    if (!read_requests.empty()) {
      post_peer_reads_async(
        *thread, span<const PeerReadRequest>{read_requests});
      while (!thread->is_ready(thread->running_coroutine)) {
        poll_peer_send_cq();
        std::this_thread::yield();
      }
    }
    for (const PendingRead& read : pending) {
      const u64 after = *reinterpret_cast<const u64*>(read.buffer);
      const StableNodeSnapshotState disposition =
        classify_physical_node_snapshot(
          read.rptr, read.before, after, read.slot_incarnation);
      if (disposition == StableNodeSnapshotState::stable) {
        NodeSnapshot& snapshot = next_snapshot();
        lib_assert(parse_remote_snapshot(read.rptr, read.buffer, snapshot),
                   "coherent remote snapshot failed to parse");
        set_state(read.input_index, StableNodeSnapshotState::stable);
        ++snapshot_count;
      } else {
        set_state(read.input_index, disposition);
      }
    }
  }

  return snapshot_count;
}

const vec<MemoryNode::BeamEntry>&
MemoryNode::score_stable_node_vectors_batched(
    span<const RemotePtr> rptrs,
    const byte_t* stored_query,
    span<const element_t> decoded_query,
    const Configuration& config) {
  struct PendingVectorRead {
    RemotePtr rptr;
    byte_t* buffer{};
    byte_t* after_header{};
    u64 before{};
    u32 slot_incarnation{};
  };

  // Stage2 is synchronous on one storage-owner OS worker, just like its
  // continuation beam. Keeping these containers thread-local preserves their
  // capacity across high-frequency insertions without sharing mutable state
  // across workers or allocating once per L*R candidate batch.
  thread_local vec<BeamEntry> scored;
  thread_local vec<PendingVectorRead> pending;
  scored.clear();
  pending.clear();
  if (rptrs.empty()) return scored;

  const VectorDType dtype = VamanaNode::vector_dtype();
  const bool integral =
    dtype == VectorDType::uint8 || dtype == VectorDType::int8;
  lib_assert(stored_query != nullptr &&
               (integral || decoded_query.size() >= VamanaNode::DIM),
             "stage2 vector scoring query is incomplete");
  scored.reserve(rptrs.size());

  const auto score_vector = [&](const byte_t* vector) {
    return integral
      ? typed_l2_distance(stored_query, dtype, vector, dtype, config.dim)
      : distance_to_stored_vector(decoded_query, vector, config);
  };
  const auto validate_pointer = [&](RemotePtr rptr) {
    if (storage_node_pointer_addressable(rptr)) return true;
    if (!rptr.is_null()) {
      report_rejected_graph_pointer("stage2_vector_scoring", rptr);
    }
    return false;
  };
  const auto score_local = [&](RemotePtr rptr) {
    const byte_t* record = index_buffer_.get_full_buffer() +
      rptr.byte_offset();
    constexpr u32 kMaxReadAttempts = 3;
    for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const u64 before = load_local_node_header_acquire(rptr);
      if (VamanaNode::header_incarnation(before) != rptr.incarnation()) {
        return;
      }
      if ((before & VamanaNode::HEADER_NODE_LOCK) != 0) {
        std::this_thread::yield();
        continue;
      }
      const u32 slot_incarnation = *reinterpret_cast<const u32*>(
        record + VamanaNode::offset_slot_incarnation());
      const distance_t distance = score_vector(
        record + VamanaNode::offset_vector());
      std::atomic_thread_fence(std::memory_order_acquire);
      const u64 after = load_local_node_header_acquire(rptr);
      if (stable_vector_snapshot_valid(
            rptr, before, after, slot_incarnation)) {
        scored.push_back(BeamEntry{rptr, distance, false});
        return;
      }
      if (before == after ||
          VamanaNode::header_incarnation(after) != rptr.incarnation()) {
        return;
      }
      std::this_thread::yield();
    }
  };

  StorageOwnerThread* thread = current_storage_owner_thread_;
  if (thread == nullptr || !thread->has_peer_scratch()) {
    byte_t* read_buffer = peer_scratch_buffer_.get_full_buffer();
    constexpr u32 kMaxReadAttempts = 3;
    for (const RemotePtr rptr : rptrs) {
      if (!validate_pointer(rptr)) continue;
      if (local_shard(rptr.memory_node())) {
        score_local(rptr);
        continue;
      }
      for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
        remote_read_bytes(
          rptr.memory_node(), rptr.byte_offset(), read_buffer,
          VamanaNode::size_until_vector_end(), 0);
        const u64 before = *reinterpret_cast<const u64*>(read_buffer);
        const u32 slot_incarnation = *reinterpret_cast<const u32*>(
          read_buffer + VamanaNode::offset_slot_incarnation());
        if (VamanaNode::header_incarnation(before) !=
              rptr.incarnation() ||
            slot_incarnation != rptr.incarnation()) {
          break;
        }
        if ((before & VamanaNode::HEADER_NODE_LOCK) != 0) {
          std::this_thread::yield();
          continue;
        }
        u64 after = 0;
        remote_read_bytes(rptr.memory_node(), rptr.byte_offset(),
                          &after, sizeof(after), 0);
        if (stable_vector_snapshot_valid(
              rptr, before, after, slot_incarnation)) {
          scored.push_back(BeamEntry{
            rptr,
            score_vector(read_buffer + VamanaNode::offset_vector()),
            false});
          break;
        }
        if (before == after ||
            VamanaNode::header_incarnation(after) !=
              rptr.incarnation()) {
          break;
        }
        std::this_thread::yield();
      }
    }
    return scored;
  }

  const size_t snapshot_stride = aligned_snapshot_bytes();
  const size_t validation_offset =
    memory_node_detail::storage_owner_snapshot_validation_offset();
  const size_t max_batch = storage_owner_snapshot_batch_size(config, thread);
  thread_local vec<PeerReadRequest> read_requests;
  thread_local vec<PeerReadPairRequest> read_pairs;
  const bool ordered_pair_supported =
    memory_node_detail::peer_rdma_read_pair_group_limit(
      peer_rdma_read_credit_plan()) != 0;
  lib_assert(validation_offset + VamanaNode::HEADER_SIZE <=
               snapshot_stride,
             "storage-owner snapshot slot lost after-header scratch");
  pending.reserve(max_batch);
  for (size_t begin = 0; begin < rptrs.size(); begin += max_batch) {
    const size_t end = std::min(rptrs.size(), begin + max_batch);
    pending.clear();
    read_requests.clear();
    read_pairs.clear();
    u32 remote_slot = 0;
    for (size_t index = begin; index < end; ++index) {
      const RemotePtr rptr = rptrs[index];
      if (!validate_pointer(rptr)) continue;
      if (local_shard(rptr.memory_node())) {
        score_local(rptr);
        continue;
      }
      const size_t scratch_offset =
        static_cast<size_t>(remote_slot) * snapshot_stride;
      lib_assert(scratch_offset + validation_offset +
                   VamanaNode::HEADER_SIZE <= thread->scratch_stride,
                 "storage-owner coroutine scratch stride is too small for "
                 "stage2 vector scoring");
      byte_t* buffer = thread->coroutine_scratch(scratch_offset);
      const PeerReadRequest full_snapshot{
        .shard_id = rptr.memory_node(),
        .remote_offset = rptr.byte_offset(),
        .destination = buffer,
        .bytes = VamanaNode::size_until_vector_end(),
      };
      byte_t* after_header = buffer + validation_offset;
      if (ordered_pair_supported) {
        read_pairs.push_back(PeerReadPairRequest{
          .full_snapshot = full_snapshot,
          .after_header = PeerReadRequest{
            .shard_id = rptr.memory_node(),
            .remote_offset = rptr.byte_offset(),
            .destination = after_header,
            .bytes = VamanaNode::HEADER_SIZE,
          },
        });
      } else {
        read_requests.push_back(full_snapshot);
      }
      pending.push_back(PendingVectorRead{rptr, buffer, after_header});
      ++remote_slot;
    }

    if (ordered_pair_supported) {
      post_peer_read_pairs_async(
        *thread, span<const PeerReadPairRequest>{read_pairs});
    } else {
      post_peer_reads_async(
        *thread, span<const PeerReadRequest>{read_requests});
    }
    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }

    size_t valid_count = 0;
    read_requests.clear();
    for (PendingVectorRead& read : pending) {
      read.before = *reinterpret_cast<const u64*>(read.buffer);
      read.slot_incarnation = *reinterpret_cast<const u32*>(
        read.buffer + VamanaNode::offset_slot_incarnation());
      if (!stable_vector_snapshot_valid(
            read.rptr, read.before, read.before,
            read.slot_incarnation)) {
        continue;
      }
      if (valid_count != static_cast<size_t>(&read - pending.data())) {
        pending[valid_count] = read;
      }
      PendingVectorRead& accepted = pending[valid_count++];
      if (!ordered_pair_supported) {
        read_requests.push_back(PeerReadRequest{
          .shard_id = accepted.rptr.memory_node(),
          .remote_offset = accepted.rptr.byte_offset(),
          .destination = accepted.buffer,
          .bytes = VamanaNode::HEADER_SIZE,
        });
      }
    }
    pending.resize(valid_count);
    if (ordered_pair_supported) {
      for (const PendingVectorRead& read : pending) {
        const u64 after = *reinterpret_cast<const u64*>(
          read.after_header);
        if (!stable_vector_snapshot_valid(
              read.rptr, read.before, after,
              read.slot_incarnation)) {
          continue;
        }
        scored.push_back(BeamEntry{
          read.rptr,
          score_vector(read.buffer + VamanaNode::offset_vector()),
          false});
      }
      continue;
    }

    // Keep the original two-wave path when the transport cannot reserve two
    // ordered READ credits. A stable vector snapshot is never reduced to a
    // single body read.
    post_peer_reads_async(*thread, span<const PeerReadRequest>{read_requests});
    while (!thread->is_ready(thread->running_coroutine)) {
      poll_peer_send_cq();
      std::this_thread::yield();
    }

    for (const PendingVectorRead& read : pending) {
      const u64 after = *reinterpret_cast<const u64*>(read.buffer);
      if (!stable_vector_snapshot_valid(
            read.rptr, read.before, after,
            read.slot_incarnation)) {
        continue;
      }
      scored.push_back(BeamEntry{
        read.rptr,
        score_vector(read.buffer + VamanaNode::offset_vector()),
        false});
    }
  }
  return scored;
}

void MemoryNode::write_hot_graph_entry(
    RemotePtr rptr,
    const vec<RemotePtr>& neighbors,
    std::optional<u32> generation_override,
    std::optional<bool> deleted_override) {
  GraphAdjacency previous;
  const bool previous_valid = read_graph_adjacency(rptr, previous);
  write_graph_adjacency(
    rptr, neighbors,
    previous_valid ? previous.provisional : vec<RemotePtr>{},
    generation_override.has_value()
      ? generation_override
      : (previous_valid
           ? std::optional<u32>{previous.generation}
           : std::nullopt),
    deleted_override.has_value()
      ? deleted_override
      : (previous_valid
           ? std::optional<bool>{previous.deleted}
           : std::nullopt));
}

void MemoryNode::write_graph_adjacency(
    RemotePtr rptr,
    const vec<RemotePtr>& stable,
    const vec<RemotePtr>& provisional,
    std::optional<u32> generation_override,
    std::optional<bool> deleted_override) {
  if (!storage_node_pointer_addressable(rptr)) return;

  const size_t entry_size = VamanaNode::hot_graph_entry_size();
  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
  lib_assert(hot_offset <= mn_memory_bytes_ &&
               entry_size <= mn_memory_bytes_ - hot_offset,
             "hot graph write exceeds shard bounds");

  u32 generation = generation_override.value_or(0);
  bool deleted = deleted_override.value_or(false);
  if (!generation_override.has_value() || !deleted_override.has_value()) {
    vec<byte_t> previous(entry_size, 0);
    if (local_shard(rptr.memory_node())) {
      std::memcpy(previous.data(),
                  index_buffer_.get_full_buffer() + hot_offset, entry_size);
    } else {
      remote_read_bytes(
        rptr.memory_node(), hot_offset, previous.data(), previous.size(), 0);
    }
    const bool previous_valid =
      vamana::hot_graph::load_u16_le(previous.data() + 2) ==
        vamana::hot_graph::checksum16(previous.data(), previous.size());
    if (previous_valid) {
      if (!generation_override.has_value()) {
        generation = vamana::hot_graph::load_u32_le(previous.data() + 4);
      }
      if (!deleted_override.has_value()) {
        deleted =
          (previous[1] & VamanaNode::HOT_GRAPH_DELETED) != 0;
      }
    } else {
      byte_t prefix[VamanaNode::HEADER_SIZE +
                    VamanaNode::COMPACT_META_SIZE]{};
      if (local_shard(rptr.memory_node())) {
        const u64 header = load_local_node_header_acquire(rptr);
        std::memcpy(prefix, &header, sizeof(header));
        std::memcpy(prefix + VamanaNode::HEADER_SIZE,
                    index_buffer_.get_full_buffer() + rptr.byte_offset() +
                      VamanaNode::HEADER_SIZE,
                    VamanaNode::COMPACT_META_SIZE);
      } else {
        remote_read_bytes(
          rptr.memory_node(), rptr.byte_offset(), prefix, sizeof(prefix), 0);
      }
      if (!deleted_override.has_value()) {
        deleted = (*reinterpret_cast<const u64*>(prefix) &
                   VamanaNode::HEADER_DELETED) != 0;
      }
      if (!generation_override.has_value()) {
        generation = *reinterpret_cast<const u32*>(
          prefix + VamanaNode::offset_generation());
      }
    }
  }

  vec<RemotePtr> bounded_stable;
  bounded_stable.reserve(std::min<size_t>(stable.size(), VamanaNode::R));
  for (const RemotePtr candidate : stable) {
    if (candidate.is_null() || candidate == rptr ||
        std::find(bounded_stable.begin(), bounded_stable.end(), candidate) !=
          bounded_stable.end()) {
      continue;
    }
    if (!storage_node_pointer_addressable(candidate)) {
      report_rejected_graph_pointer(
        "write_graph_adjacency/stable", candidate, rptr);
      continue;
    }
    bounded_stable.push_back(candidate);
    if (bounded_stable.size() == VamanaNode::R) break;
  }
  vec<RemotePtr> bounded_provisional;
  bounded_provisional.reserve(std::min<size_t>(
    provisional.size(), VamanaNode::provisional_slots()));
  for (const RemotePtr candidate : provisional) {
    if (candidate.is_null() || candidate == rptr ||
        std::find(bounded_stable.begin(), bounded_stable.end(), candidate) !=
          bounded_stable.end() ||
        std::find(bounded_provisional.begin(), bounded_provisional.end(),
                  candidate) != bounded_provisional.end()) {
      continue;
    }
    if (!storage_node_pointer_addressable(candidate)) {
      report_rejected_graph_pointer(
        "write_graph_adjacency/provisional", candidate, rptr);
      continue;
    }
    bounded_provisional.push_back(candidate);
    if (bounded_provisional.size() == VamanaNode::provisional_slots()) break;
  }

  vec<byte_t> entry(entry_size, 0);
  const u8 edge_count = static_cast<u8>(bounded_stable.size());
  VamanaNode::encode_hot_graph_entry(entry.data(), edge_count,
                                     bounded_stable.data(), edge_count,
                                     VamanaNode::HOT_GRAPH_SHARD_BITS,
                                     generation, false,
                                     bounded_provisional.data(),
                                     bounded_provisional.size(),
                                     rptr.incarnation());
  if (deleted) {
    // Deleted nodes retain their preserved adjacency for cleanup, while GPU
    // readers still observe the tombstone and ignore the payload.
    entry[1] |= VamanaNode::HOT_GRAPH_DELETED;
    const u16 checksum =
      vamana::hot_graph::checksum16(entry.data(), entry.size());
    vamana::hot_graph::store_u16_le(entry.data() + 2, checksum);
  }
  if (local_shard(rptr.memory_node())) {
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

void MemoryNode::write_dynamic_navigation_code(
    RemotePtr rptr, const span<const element_t> components) {
  lib_assert(local_shard(rptr.memory_node()) &&
               rptr.byte_offset() >= gpu_dynamic_node_base_,
             "dynamic PQ code must be written to a local dynamic node");
  thread_local vec<f32> transformed;
  transformed.resize(gpu_navigation_model_.dim);
  byte_t* destination = index_buffer_.get_full_buffer() +
    VamanaNode::dynamic_navigation_code_offset(rptr);
  *reinterpret_cast<u32*>(destination) = rptr.incarnation();
  gpu_search::pq::encode(
    gpu_navigation_model_,
    std::span<const f32>{components.data(), components.size()},
    std::span<u8>{destination + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES,
                  gpu_navigation_model_.code_bytes()}, transformed);
}

void MemoryNode::write_new_node(RemotePtr rptr,
                    node_t id,
                    const span<const element_t> components,
                    const vec<RemotePtr>& neighbors,
                    u32 generation,
                    bool provisional) {
  byte_t* ptr = local_node_ptr(rptr);
  const u64 publishing_header = VamanaNode::make_header(
    rptr.incarnation(), VamanaNode::HEADER_NODE_LOCK);
  const u64 initial_header = VamanaNode::make_header(
    rptr.incarnation(), provisional ? VamanaNode::HEADER_PROVISIONAL : 0);
  auto* header_storage = reinterpret_cast<u64*>(ptr);
  std::atomic_ref<u64> header_ref(*header_storage);
  lib_assert(header_ref.load(std::memory_order_acquire) == publishing_header &&
               *reinterpret_cast<const u32*>(
                 ptr + VamanaNode::offset_slot_incarnation()) ==
                 rptr.incarnation(),
             "stale local allocation handle cannot materialize a reused slot");
  std::memset(ptr + VamanaNode::HEADER_SIZE, 0,
              VamanaNode::allocation_size() - VamanaNode::HEADER_SIZE);
  *reinterpret_cast<u32*>(ptr + VamanaNode::offset_id()) = id;
  *reinterpret_cast<u32*>(ptr + VamanaNode::offset_generation()) = generation;
  *reinterpret_cast<u32*>(ptr + VamanaNode::offset_slot_incarnation()) =
    rptr.incarnation();
  encode_float_vector_to_storage(components.data(), VamanaNode::DIM, VamanaNode::vector_dtype(),
                                 ptr + VamanaNode::offset_vector());
  write_dynamic_navigation_code(rptr, components);
  write_hot_graph_entry(rptr, neighbors, generation, false);
  header_ref.store(initial_header, std::memory_order_release);
}

void MemoryNode::write_new_node_on_shard(
    RemotePtr rptr,
    node_t id,
    const span<const element_t> components,
    const vec<RemotePtr>& neighbors,
    u32 generation,
    bool provisional) {
  lib_assert(rptr.memory_node() < num_storage_nodes_ &&
               components.size() == VamanaNode::DIM,
             "invalid remote-node materialization request");
  if (local_shard(rptr.memory_node())) {
    write_new_node(rptr, id, components, neighbors,
                   generation, provisional);
    return;
  }

  vec<byte_t> record(VamanaNode::allocation_size(), 0);
  const u64 publishing_header = VamanaNode::make_header(
    rptr.incarnation(), VamanaNode::HEADER_NODE_LOCK);
  const u64 final_header = VamanaNode::make_header(
    rptr.incarnation(), provisional ? VamanaNode::HEADER_PROVISIONAL : 0);
  *reinterpret_cast<u64*>(record.data()) = publishing_header;
  *reinterpret_cast<u32*>(record.data() + VamanaNode::offset_id()) = id;
  *reinterpret_cast<u32*>(
    record.data() + VamanaNode::offset_generation()) = generation;
  *reinterpret_cast<u32*>(
    record.data() + VamanaNode::offset_slot_incarnation()) =
      rptr.incarnation();
  encode_float_vector_to_storage(
    components.data(), VamanaNode::DIM, VamanaNode::vector_dtype(),
    record.data() + VamanaNode::offset_vector());

  vec<RemotePtr> bounded_neighbors;
  bounded_neighbors.reserve(std::min<size_t>(neighbors.size(), VamanaNode::R));
  for (const RemotePtr neighbor : neighbors) {
    if (neighbor.is_null() || neighbor == rptr ||
        std::find(bounded_neighbors.begin(), bounded_neighbors.end(),
                  neighbor) != bounded_neighbors.end()) {
      continue;
    }
    if (!storage_node_pointer_addressable(neighbor)) {
      report_rejected_graph_pointer(
        "write_new_node_on_shard", neighbor, rptr);
      continue;
    }
    bounded_neighbors.push_back(neighbor);
    if (bounded_neighbors.size() == VamanaNode::R) break;
  }
  const u8 stable_count = static_cast<u8>(bounded_neighbors.size());
  VamanaNode::encode_hot_graph_entry(
    record.data() + VamanaNode::HOT_GRAPH_DYNAMIC_HOT_OFFSET,
    stable_count, bounded_neighbors.data(), stable_count,
    VamanaNode::HOT_GRAPH_SHARD_BITS, generation, false,
    nullptr, 0, rptr.incarnation());

  thread_local vec<f32> transformed;
  transformed.resize(gpu_navigation_model_.dim);
  gpu_search::pq::encode(
    gpu_navigation_model_,
    std::span<const f32>{components.data(), components.size()},
    std::span<u8>{
      record.data() + VamanaNode::HOT_GRAPH_DYNAMIC_CODE_OFFSET +
        VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES,
      gpu_navigation_model_.code_bytes()},
    transformed);
  *reinterpret_cast<u32*>(
    record.data() + VamanaNode::HOT_GRAPH_DYNAMIC_CODE_OFFSET) =
      rptr.incarnation();
  // Publish remote records header-last. A reader either observes the old
  // tombstone, the publishing lock, or a complete new incarnation.
  const u64 observed = remote_compare_and_swap(
    rptr.memory_node(), rptr.byte_offset(), publishing_header,
    publishing_header, align_up(sizeof(publishing_header)));
  lib_assert(observed == publishing_header,
             "stale remote allocation handle cannot materialize a reused slot");
  remote_write_bytes(
    rptr.memory_node(), rptr.byte_offset() + VamanaNode::HEADER_SIZE,
    record.data() + VamanaNode::HEADER_SIZE,
    record.size() - VamanaNode::HEADER_SIZE, 0);
  remote_write_bytes(rptr.memory_node(), rptr.byte_offset(),
                     &final_header, sizeof(final_header), 0);
}

bool MemoryNode::set_node_provisional(RemotePtr rptr, bool provisional) {
  if (rptr.is_null() || rptr.memory_node() >= num_storage_nodes_) {
    return false;
  }
  const auto header_addr = vamana::StorageLayoutResolver::header(rptr);
  if (header_addr.offset > mn_memory_bytes_ ||
      sizeof(u64) > mn_memory_bytes_ - header_addr.offset) {
    return false;
  }
  if (local_shard(rptr.memory_node())) {
    auto* header = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + header_addr.offset);
    std::atomic_ref<u64> ref(*header);
    for (u32 attempt = 0; attempt < 8; ++attempt) {
      u64 observed = ref.load(std::memory_order_acquire);
      if (VamanaNode::header_incarnation(observed) !=
          rptr.incarnation()) {
        return false;
      }
      if ((observed & VamanaNode::HEADER_NODE_LOCK) != 0) {
        std::this_thread::yield();
        continue;
      }
      const u64 desired = provisional
        ? observed | static_cast<u64>(VamanaNode::HEADER_PROVISIONAL)
        : observed & ~static_cast<u64>(VamanaNode::HEADER_PROVISIONAL);
      if (ref.compare_exchange_weak(
            observed, desired, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return true;
      }
    }
    return false;
  }

  if (try_lock_node(rptr) != IncarnationLockResult::locked) {
    return false;
  }
  u64 header = 0;
  remote_read_bytes(rptr.memory_node(), header_addr.offset,
                    &header, sizeof(header), 0);
  if (provisional) {
    header |= static_cast<u64>(VamanaNode::HEADER_PROVISIONAL);
  } else {
    header &= ~static_cast<u64>(VamanaNode::HEADER_PROVISIONAL);
  }
  remote_write_bytes(rptr.memory_node(), header_addr.offset,
                     &header, sizeof(header), 0);
  unlock_node(rptr);
  return true;
}

void MemoryNode::lock_node(RemotePtr rptr) {
  if (local_shard(rptr.memory_node())) {
    auto* header_ptr = reinterpret_cast<u64*>(
      index_buffer_.get_full_buffer() + vamana::StorageLayoutResolver::header(rptr).offset);
    std::atomic_ref<u64> ref(*header_ptr);
    for (;;) {
      u64 header = ref.load(std::memory_order_acquire);
      lib_assert(VamanaNode::header_incarnation(header) ==
                   rptr.incarnation(),
                 "stale tagged handle attempted to lock a reused local slot");
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
    lib_assert(VamanaNode::header_incarnation(header) ==
                 rptr.incarnation(),
               "stale tagged handle attempted to lock a reused remote slot");
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
