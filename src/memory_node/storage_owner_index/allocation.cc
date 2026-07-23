#include "memory_node/storage_owner_index/detail.hh"

#include <cstring>
#include <filesystem>

using namespace memory_node_storage_owner_index_detail;

namespace {

u64 dynamic_node_allocation_stride() {
  const u64 bytes = VamanaNode::allocation_size();
  return (bytes + alignof(u64) - 1) & ~(alignof(u64) - 1);
}

}  // namespace

RemotePtr MemoryNode::allocate_local_node() {
  const u64 node_size = dynamic_node_allocation_stride();

  const auto reserve_slot = [&](RemotePtr pointer) {
    byte_t* slot = index_buffer_.get_full_buffer() + pointer.byte_offset();
    auto* header = reinterpret_cast<u64*>(slot);
    std::atomic_ref<u64>(*header).store(
      VamanaNode::make_header(
        pointer.incarnation(), VamanaNode::HEADER_NODE_LOCK),
      std::memory_order_release);
    *reinterpret_cast<u32*>(
      slot + VamanaNode::offset_slot_incarnation()) =
        pointer.incarnation();
    std::atomic_thread_fence(std::memory_order_release);
    return pointer;
  };

  const auto reserve_reclaimed_slot = [&](RemotePtr reclaimed) {
    lib_assert(reclaimed.incarnation() < RemotePtr::MAX_INCARNATION,
               "cannot reuse a slot after incarnation exhaustion");
    const RemotePtr replacement = reclaimed.with_incarnation(
      reclaimed.incarnation() + 1);
    byte_t* slot = index_buffer_.get_full_buffer() +
      reclaimed.byte_offset();
    auto* header_storage = reinterpret_cast<u64*>(slot);
    std::atomic_ref<u64> header_ref(*header_storage);
    for (;;) {
      u64 observed = header_ref.load(std::memory_order_acquire);
      lib_assert(VamanaNode::header_incarnation(observed) ==
                   reclaimed.incarnation() &&
                   (observed & VamanaNode::HEADER_DELETED) != 0,
                 "reclaim queue identity no longer matches its physical slot");
      if ((observed & VamanaNode::HEADER_NODE_LOCK) != 0) {
        // A delayed old-incarnation data-plane operation won the lock before
        // reuse. Let it observe the tombstone and finish; overwriting its lock
        // would allow it to mutate or unlock the replacement occupant.
        std::this_thread::yield();
        continue;
      }
      const u64 publishing_header = VamanaNode::make_header(
        replacement.incarnation(), VamanaNode::HEADER_NODE_LOCK);
      if (header_ref.compare_exchange_weak(
            observed, publishing_header, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        break;
      }
    }
    *reinterpret_cast<u32*>(
      slot + VamanaNode::offset_slot_incarnation()) =
        replacement.incarnation();
    std::atomic_thread_fence(std::memory_order_release);
    return replacement;
  };

  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  const u64 durable = std::atomic_ref<u64>(
    control->durable_maintenance_sequence).load(std::memory_order_acquire);
  {
    std::lock_guard<std::mutex> lock(storage_owner_reclaim_mutex_);
    // Dynamic queries use read-committed, incarnation-tagged dereferences.
    // Durable maintenance is therefore the sole reuse fence: a stale handle
    // cannot match the incremented slot incarnation, and the replacement
    // record is exposed only by its final header publication.
    for (std::optional<RemotePtr> reclaimed =
           storage_owner_reclaim_queue_.acquire(durable);
         reclaimed.has_value();
         reclaimed = storage_owner_reclaim_queue_.acquire(durable)) {
      lib_assert(reclaimed->memory_node() == storage_id_ &&
                   reclaimed->byte_offset() >= gpu_dynamic_node_base_ &&
                   (reclaimed->byte_offset() - gpu_dynamic_node_base_) %
                       node_size == 0,
                 "storage reclaim queue returned a non-dynamic slot");
      const u64 pending = storage_owner_reclaim_queue_.size();
      storage_owner_reclaim_candidates_.store(
        pending, std::memory_order_release);
      std::atomic_ref<u64>(control->reclaim_pending_nodes).store(
        pending, std::memory_order_release);
      std::atomic_ref<u64>(control->reclaim_reused_nodes).store(
        storage_owner_reclaim_queue_.reused(), std::memory_order_release);
      if (reclaimed->incarnation() >= RemotePtr::MAX_INCARNATION) {
        // Never wrap a tag: a handle may persist in an arbitrarily old graph
        // record. Permanently retire this physical slot and grow the dynamic
        // region instead.
        continue;
      }
      return reserve_reclaimed_slot(*reclaimed);
    }
  }

  auto* free_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer());
  std::atomic_ref<u64> alloc_ref(*free_ptr);
  const u64 offset = alloc_ref.fetch_add(node_size, std::memory_order_acq_rel);
  const u64 allocation_limit = dynamic_allocation_limit_ == 0
    ? mn_memory_bytes_ : dynamic_allocation_limit_;
  lib_assert(offset <= allocation_limit &&
               node_size <= allocation_limit - offset,
             "storage node dynamic region is out of memory");
  std::atomic_ref<u64> high_watermark(control->dynamic_high_watermark);
  u64 observed = high_watermark.load(std::memory_order_relaxed);
  while (observed < offset + node_size &&
         !high_watermark.compare_exchange_weak(
           observed, offset + node_size,
           std::memory_order_release, std::memory_order_relaxed)) {}
  return reserve_slot(RemotePtr{storage_id_, offset, 1});
}

void MemoryNode::retire_local_dynamic_node(RemotePtr pointer,
                                           u64 maintenance_sequence) {
  if (pointer.is_null() || pointer.memory_node() != storage_id_ ||
      pointer.byte_offset() < gpu_dynamic_node_base_ || maintenance_sequence == 0) {
    return;
  }
  const u64 node_size = dynamic_node_allocation_stride();
  lib_assert((pointer.byte_offset() - gpu_dynamic_node_base_) % node_size == 0,
             "cannot retire a misaligned storage-owner dynamic node");
  const u64 allocation_limit = dynamic_allocation_limit_ == 0
    ? mn_memory_bytes_ : dynamic_allocation_limit_;
  lib_assert(pointer.byte_offset() <= allocation_limit &&
               node_size <= allocation_limit - pointer.byte_offset(),
             "cannot retire a dynamic node outside the allocation region");
  const byte_t* node = index_buffer_.get_full_buffer() + pointer.byte_offset();
  const u64 header = std::atomic_ref<const u64>(
    *reinterpret_cast<const u64*>(node)).load(std::memory_order_acquire);
  const u32 stored_incarnation = *reinterpret_cast<const u32*>(
    node + VamanaNode::offset_slot_incarnation());
  if ((header & VamanaNode::HEADER_DELETED) == 0 ||
      VamanaNode::header_incarnation(header) != pointer.incarnation() ||
      stored_incarnation != pointer.incarnation()) {
    // A delayed duplicate cleanup must never retire a newer occupant of the
    // same physical slot.
    return;
  }
  u64 pending = 0;
  u64 reused = 0;
  {
    std::lock_guard<std::mutex> lock(storage_owner_reclaim_mutex_);
    (void)storage_owner_reclaim_queue_.retire(
      pointer, maintenance_sequence);
    pending = storage_owner_reclaim_queue_.size();
    reused = storage_owner_reclaim_queue_.reused();
  }
  storage_owner_reclaim_candidates_.store(
    pending, std::memory_order_release);
  auto* control = reinterpret_cast<gpu_search::format::StorageControlBlock*>(
    index_buffer_.get_full_buffer() + gpu_storage_control_offset_);
  std::atomic_ref<u64>(control->reclaim_pending_nodes).store(
    pending, std::memory_order_release);
  std::atomic_ref<u64>(control->reclaim_reused_nodes).store(
    reused, std::memory_order_release);
}

bool MemoryNode::load_owner_idmap(const filepath_t& index_prefix) {
  base_idmap_.clear();
  for (DynamicFreshnessShard& shard : dynamic_freshness_shards_) {
    shard.entries.clear();
    shard.mutation_leases.clear();
  }
  if (index_prefix.empty()) {
    return true;
  }
  const filepath_t path = index_path::owner_idmap_file(index_prefix, storage_id_ + 1, num_storage_nodes_);
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    std::cerr << "[storage-owner] missing idmap sidecar: " << path << std::endl;
    return false;
  }
  std::error_code file_error;
  const uintmax_t raw_file_bytes = std::filesystem::file_size(path, file_error);
  if (file_error || raw_file_bytes > std::numeric_limits<u64>::max()) {
    std::cerr << "[storage-owner] cannot size idmap sidecar: " << path
              << std::endl;
    return false;
  }
  vamana::idmap::Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  const vamana::idmap::ValidationContext validation{
    .build_fingerprint = gpu_index_build_fingerprint_,
    .owner_shard_fingerprint = gpu_shard_build_fingerprint_,
    .node_base_offset = vamana::hot_graph::kNodeBaseOffset,
    .owner_shard = storage_id_,
    .shard_count = num_storage_nodes_,
    .node_size = static_cast<u32>(VamanaNode::total_size()),
    .id_offset = static_cast<u32>(VamanaNode::offset_id()),
    .generation_offset =
      static_cast<u32>(VamanaNode::offset_generation()),
    .slot_incarnation_offset =
      static_cast<u32>(VamanaNode::offset_slot_incarnation()),
    .static_entry_counts =
      span<const u64>{VamanaNode::HOT_GRAPH_ENTRY_COUNTS},
  };
  if (input.gcount() != static_cast<std::streamsize>(sizeof(header)) ||
      !vamana::idmap::valid_header(
        header, static_cast<u64>(raw_file_bytes), validation) ||
      header.entry_count > std::numeric_limits<size_t>::max()) {
    std::cerr << "[storage-owner] invalid or obsolete bound-v2 idmap "
              << "sidecar: " << path << std::endl;
    return false;
  }
  // Reserve a small bounded fraction of this authority's immutable base.
  // A fixed million-entry reserve per shard wastes gigabytes across many
  // small shards; unbounded proportional reserve is equally unsuitable for a
  // billion-vector base. The map still grows normally beyond this hint.
  const u64 dynamic_headroom = std::clamp<u64>(
    header.entry_count / 100, 64, 64ull * 1024ull);
  base_idmap_.reserve(static_cast<size_t>(header.entry_count));
  const size_t dynamic_reserve_per_shard = static_cast<size_t>(
    (dynamic_headroom + kDynamicFreshnessShardCount - 1) /
    kDynamicFreshnessShardCount);
  for (DynamicFreshnessShard& shard : dynamic_freshness_shards_) {
    shard.entries.reserve(dynamic_reserve_per_shard);
    shard.mutation_leases.reserve(64);
  }
  const bool valid_payload = vamana::idmap::read_validated_payload(
    input, header, validation, [&](const vamana::idmap::Entry& entry) {
      if (entry.id >= vector_id_namespace_size_) {
        return false;
      }
      const RemotePtr pointer{entry.rptr_raw};
      if (pointer.memory_node() == storage_id_) {
        if (pointer.byte_offset() > mn_memory_bytes_ ||
            VamanaNode::total_size() >
              mn_memory_bytes_ - pointer.byte_offset()) {
          return false;
        }
        const byte_t* node =
          index_buffer_.get_full_buffer() + pointer.byte_offset();
        const u64 node_header = std::atomic_ref<const u64>(
          *reinterpret_cast<const u64*>(node)).load(
            std::memory_order_acquire);
        const u64 disallowed = VamanaNode::HEADER_NODE_LOCK |
          VamanaNode::HEADER_DELETED | VamanaNode::HEADER_PROVISIONAL |
          VamanaNode::HEADER_RETIRING | VamanaNode::HEADER_STAGE2_FROZEN;
        const u64 compact_offset =
          VamanaNode::hot_graph_entry_offset(pointer);
        if ((node_header & disallowed) != 0 ||
            (node_header & VamanaNode::HEADER_CENTROID_ACCOUNTED) == 0 ||
            VamanaNode::header_incarnation(node_header) != 0 ||
            *reinterpret_cast<const node_t*>(
              node + VamanaNode::offset_id()) != entry.id ||
            *reinterpret_cast<const u32*>(
              node + VamanaNode::offset_generation()) != 0 ||
            *reinterpret_cast<const u32*>(
              node + VamanaNode::offset_slot_incarnation()) != 0 ||
            !VamanaNode::hot_graph_entry_available(pointer) ||
            compact_offset > mn_memory_bytes_ ||
            VamanaNode::hot_graph_entry_size() >
              mn_memory_bytes_ - compact_offset) {
          return false;
        }
        const byte_t* compact =
          index_buffer_.get_full_buffer() + compact_offset;
        if ((compact[1] & VamanaNode::HOT_GRAPH_DELETED) != 0 ||
            compact[0] > VamanaNode::R ||
            vamana::hot_graph::provisional_count(compact) >
              VamanaNode::provisional_slots() ||
            vamana::hot_graph::load_u32_le(compact + 4) != 0 ||
            vamana::hot_graph::load_u32_le(compact + 8) != 0 ||
            vamana::hot_graph::load_u32_le(compact + 12) != 0 ||
            vamana::hot_graph::load_u16_le(compact + 2) !=
              vamana::hot_graph::checksum16(
                compact, VamanaNode::hot_graph_entry_size())) {
          return false;
        }
      }
      return base_idmap_.emplace(entry.id, pointer).second;
    });
  if (!valid_payload || base_idmap_.size() != header.entry_count) {
    base_idmap_.clear();
    std::cerr << "[storage-owner] idmap payload checksum, authority, "
                 "duplicate, or static-record validation failed: "
              << path << std::endl;
    return false;
  }
  print_status("storage-owner idmap loaded entries=" +
               std::to_string(base_idmap_.size()) +
               " format=owner_sharded_v2_bound immutable_base=true "
               "dynamic_shards=" +
               std::to_string(kDynamicFreshnessShardCount) +
               " dynamic_reserved=" + std::to_string(dynamic_headroom));
  return true;
}

MemoryNode::AuthorityDirectoryState
MemoryNode::load_authority_directory_state_locked(
    const DynamicFreshnessShard& shard,
    node_t id) const {
  AuthorityDirectoryState state;
  const auto dynamic = shard.entries.find(id);
  if (dynamic != shard.entries.end()) {
    state.exists = true;
    state.entry = dynamic->second;
  } else {
    const auto base = base_idmap_.find(id);
    if (base != base_idmap_.end()) {
      state.exists = true;
      state.entry = FreshnessEntry{
        .current = base->second,
        .generation = 0,
        .deleted = false,
        .placement_version = 0,
        .last_committed_operation = {},
        .last_committed_kind =
          service::storage_owner::MutationKind::insert,
        .last_committed_stage1_home = 0,
        .last_committed_result = {},
        .last_relocation_operation = {},
        .last_relocation_generation = 0,
        .last_relocation_expected = {},
        .last_relocation_desired = {},
        .last_relocation_expected_version = 0,
      };
    }
  }
  const auto lease = shard.mutation_leases.find(id);
  if (lease != shard.mutation_leases.end()) {
    state.lease = lease->second;
  }
  return state;
}

void MemoryNode::store_authority_directory_state_locked(
    DynamicFreshnessShard& shard,
    node_t id,
    const AuthorityDirectoryState& state) {
  if (state.exists) {
    shard.entries[id] = state.entry;
  } else {
    shard.entries.erase(id);
  }
  if (state.lease.has_value()) {
    shard.mutation_leases[id] = *state.lease;
  } else {
    shard.mutation_leases.erase(id);
  }
}

MemoryNode::AuthorityBeginResult MemoryNode::begin_authority_mutation(
    node_t id,
    service::storage_owner::MutationKind kind,
    AuthorityOperationToken operation,
    u32 stage1_home) {
  DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
  for (;;) {
    std::unique_lock<std::mutex> lock(shard.mutex);
    AuthorityDirectoryState state =
      load_authority_directory_state_locked(shard, id);
    const AuthorityBeginResult result =
      memory_node_storage_owner_index_detail::begin_authority_mutation(
        state, kind, operation, stage1_home);
    if (result.state !=
        memory_node_storage_owner_index_detail::AuthorityBeginState::replay) {
      if (result.acquired()) {
        store_authority_directory_state_locked(shard, id, state);
      }
      return result;
    }

    // Exactly one foreground executor may perform physical Stage1 work for a
    // semantic token. A duplicate request that arrived before the original
    // commit waits on this ID shard and then observes committed_replay (or
    // legitimately re-acquires after abort). This closes the otherwise
    // unbounded "future replay after receipt release" race without retaining
    // per-client deltas or duplicating graph work.
    shard.changed.wait(lock, [&]() {
      if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
        return true;
      }
      const auto active = shard.mutation_leases.find(id);
      return active == shard.mutation_leases.end() ||
        !memory_node_storage_owner_index_detail::same_authority_operation(
          active->second.operation, operation);
    });
    if (storage_insert_shutdown_.load(std::memory_order_acquire)) {
      return {
        .state = memory_node_storage_owner_index_detail::
          AuthorityBeginState::busy,
        .previous = {},
        .generation = 0,
        .replay_result = {},
      };
    }
  }
}

MemoryNode::AuthorityCommitState MemoryNode::commit_authority_mutation(
    node_t id,
    AuthorityOperationToken operation,
    RemotePtr desired,
    u32 generation,
    bool deleted,
    u64 maintenance_sequence) {
  DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
  AuthorityCommitState result;
  {
    std::lock_guard<std::mutex> lock(shard.mutex);
    AuthorityDirectoryState state =
      load_authority_directory_state_locked(shard, id);
    result = memory_node_storage_owner_index_detail::
      commit_authority_mutation(
        state, operation, desired, generation, deleted,
        maintenance_sequence);
    if (result == AuthorityCommitState::committed ||
        result == AuthorityCommitState::replay) {
      store_authority_directory_state_locked(shard, id, state);
    }
  }
  if (result == AuthorityCommitState::committed ||
      result == AuthorityCommitState::replay) {
    shard.changed.notify_all();
  }
  return result;
}

MemoryNode::AuthorityAbortState MemoryNode::abort_authority_mutation(
    node_t id,
    AuthorityOperationToken operation) {
  DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
  AuthorityAbortState result;
  {
    std::lock_guard<std::mutex> lock(shard.mutex);
    AuthorityDirectoryState state =
      load_authority_directory_state_locked(shard, id);
    result = memory_node_storage_owner_index_detail::
      abort_authority_mutation(state, operation);
    if (result == AuthorityAbortState::aborted) {
      store_authority_directory_state_locked(shard, id, state);
    }
  }
  if (result == AuthorityAbortState::aborted) shard.changed.notify_all();
  return result;
}

MemoryNode::AuthorityCheckState MemoryNode::check_authority_current(
    node_t id,
    AuthorityOperationToken operation,
    u32 generation,
    RemotePtr expected,
    u64 expected_placement_version) {
  DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
  std::lock_guard<std::mutex> lock(shard.mutex);
  const AuthorityDirectoryState state =
    load_authority_directory_state_locked(shard, id);
  return memory_node_storage_owner_index_detail::check_authority_current(
    state, operation, generation, expected, expected_placement_version);
}

MemoryNode::AuthorityRelocateState MemoryNode::relocate_authority_if_current(
    node_t id,
    AuthorityOperationToken operation,
    u32 generation,
    RemotePtr expected,
    RemotePtr desired,
    u64 expected_placement_version,
    u64* resulting_placement_version) {
  DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
  std::lock_guard<std::mutex> lock(shard.mutex);
  AuthorityDirectoryState state =
    load_authority_directory_state_locked(shard, id);
  const AuthorityRelocateState result =
    memory_node_storage_owner_index_detail::relocate_authority_if_current(
      state, operation, generation, expected, desired,
      expected_placement_version);
  if (result == AuthorityRelocateState::committed ||
      result == AuthorityRelocateState::replay) {
    store_authority_directory_state_locked(shard, id, state);
  }
  if (resulting_placement_version != nullptr) {
    *resulting_placement_version =
      state.exists ? state.entry.placement_version : 0;
  }
  return result;
}

bool MemoryNode::mark_node_deleted(RemotePtr rptr, u32 generation) {
  if (rptr.is_null()) return true;
  const auto header_addr = vamana::StorageLayoutResolver::header(rptr);
  const bool local = local_shard(rptr.memory_node());
  bool locked = false;
  if (local) {
    auto* header_ptr = reinterpret_cast<u64*>(index_buffer_.get_full_buffer() + header_addr.offset);
    std::atomic_ref<u64> ref(*header_ptr);
    if (try_lock_node(rptr) != IncarnationLockResult::locked) {
      return false;
    }
    locked = true;
    const u64 header = ref.load(std::memory_order_acquire);
    const byte_t* node = index_buffer_.get_full_buffer() +
      rptr.byte_offset();
    const u32 stored_generation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_generation());
    const u32 stored_incarnation = *reinterpret_cast<const u32*>(
      node + VamanaNode::offset_slot_incarnation());
    if (VamanaNode::header_incarnation(header) != rptr.incarnation() ||
        stored_incarnation != rptr.incarnation() ||
        stored_generation != generation ||
        (header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0) {
      unlock_node(rptr);
      return false;
    }
    ref.fetch_or(static_cast<u64>(VamanaNode::HEADER_DELETED),
                 std::memory_order_acq_rel);
  } else {
    if (try_lock_node(rptr) != IncarnationLockResult::locked) {
      return false;
    }
    locked = true;
    constexpr size_t identity_bytes =
      VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;
    byte_t identity[identity_bytes]{};
    remote_read_bytes(rptr.memory_node(), header_addr.offset,
                      identity, sizeof(identity), 0);
    u64 header = 0;
    std::memcpy(&header, identity, sizeof(header));
    u32 stored_generation = 0;
    u32 stored_incarnation = 0;
    std::memcpy(&stored_generation,
                identity + VamanaNode::offset_generation(),
                sizeof(stored_generation));
    std::memcpy(&stored_incarnation,
                identity + VamanaNode::offset_slot_incarnation(),
                sizeof(stored_incarnation));
    if (VamanaNode::header_incarnation(header) != rptr.incarnation() ||
        stored_incarnation != rptr.incarnation() ||
        stored_generation != generation ||
        (header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0) {
      unlock_node(rptr);
      return false;
    }
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
