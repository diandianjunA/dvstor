#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>

#include <library/context.hh>
#include <library/hugepage.hh>
#include <library/memory_region.hh>
#include <library/utils.hh>

#include "common/constants.hh"
#include "common/types.hh"
#include "memory_node/storage_owner_index/authority_directory_policy.hh"
#include "remote_pointer.hh"
#include "service/storage_owner_protocol.hh"
#include "vamana/vamana_node.hh"

namespace memory_node_detail {

inline size_t align_to_cacheline(size_t value) {
  while (value % kCacheLineBytes != 0) {
    ++value;
  }
  return value;
}

inline size_t storage_owner_snapshot_bytes() {
  return VamanaNode::size_until_vector_end();
}

inline size_t storage_owner_snapshot_stride() {
  return align_to_cacheline(storage_owner_snapshot_bytes());
}

struct BeamEntry {
  RemotePtr rptr;
  distance_t distance{};
  bool expanded{false};
};

struct NodeSnapshot {
  RemotePtr rptr;
  u64 header{};
  node_t id{};
  u32 generation{};
  u32 slot_incarnation{};
  bool deleted{};
  vec<byte_t> vector_data;
};

struct StorageOwnerPruneCandidateInfo {
  RemotePtr rptr;
  distance_t dist{};
  vec<byte_t> vector_data;
};

struct StorageOwnerScoredSnapshot {
  const NodeSnapshot* snapshot{};
  distance_t distance{};
};

struct StorageOwnerCoroutineScratch {
  hashset_t<RemotePtr> visited;
  hashset_t<RemotePtr> empty_skip;
  vec<BeamEntry> beam;
  vec<RemotePtr> neighbors;
  vec<RemotePtr> unvisited;
  vec<RemotePtr> batch;
  vec<byte_t> neighbor_entry;
  vec<byte_t> neighbor_decoded;
  vec<RemotePtr> filtered;
  vec<RemotePtr> selected;
  vec<const byte_t*> selected_vectors;
  vec<size_t> prune_selected_indices;
  vec<StorageOwnerPruneCandidateInfo> prune_infos;
  hashset_t<RemotePtr> prune_seen;
  vec<StorageOwnerScoredSnapshot> scored_snapshots;
  vec<RemotePtr> reverse_unique_candidates;
  vec<RemotePtr> reverse_current_neighbors;
  vec<RemotePtr> reverse_filtered_candidates;
  vec<RemotePtr> reverse_updated_neighbors;
  vec<RemotePtr> reverse_remote_neighbors;
  vec<RemotePtr> reverse_remote_candidates;
  vec<distance_t> reverse_neighbor_dists;

  void clear_search() {
    visited.clear();
    beam.clear();
    neighbors.clear();
    unvisited.clear();
    batch.clear();
  }

  void clear_prune() {
    filtered.clear();
    batch.clear();
    selected.clear();
    selected_vectors.clear();
    prune_selected_indices.clear();
    prune_infos.clear();
    prune_seen.clear();
    scored_snapshots.clear();
  }

  void clear_reverse_update() {
    reverse_unique_candidates.clear();
    reverse_current_neighbors.clear();
    reverse_filtered_candidates.clear();
    reverse_updated_neighbors.clear();
    reverse_remote_neighbors.clear();
    reverse_remote_candidates.clear();
    reverse_neighbor_dists.clear();
  }
};

struct InsertRuntimeState {
  HugePage<byte_t> buffer;
  std::unique_ptr<LocalMemoryRegion> region;
  size_t request_bytes{};
  size_t response_offset{};
  u32 request_slot_count{1};
};

struct PeerRpcRuntimeState {
  HugePage<byte_t> buffer;
  std::unique_ptr<LocalMemoryRegion> region;
  size_t message_bytes{};
  size_t recv_region_bytes{};
  size_t sync_send_offset{};
  size_t async_send_offset{};
  u32 recv_slots_per_peer{1};
  u32 send_slots_per_peer{1};
};

struct StorageOwnerThread;

struct PeerPendingSend {
  u32 target_shard{};
  u32 target_qp_idx{};
  u32 thread_id{};
  u32 coroutine_id{};
  StorageOwnerThread* thread{};
  bool async{};
  bool rdma_read_credit{};
  bool release_rpc_slot{};
  u32 rpc_slot_id{};
  // A linked RC READ chain produces one successful CQE at its signaled tail.
  // Credit counters still account for every WR in that chain.
  u32 rdma_read_count{1};
};

struct StorageOwnerInsertTask {
  u32 client_id{};
  u32 slot_id{};
  u32 item_count{};
  u64 batch_id{};
  size_t byte_len{};
  std::chrono::steady_clock::time_point received_at{};
};

struct StorageOwnerResponseReady {
  u32 client_id{};
  u32 slot_id{};
  u32 byte_len{};
  std::chrono::steady_clock::time_point queued_at{};
};

struct StorageOwnerRequestScratch {
  vec<service::storage_owner::MutationKind> kinds;
  vec<element_t> decoded_vectors;
  vec<vec<u64>> invalidated_neighbors;
  vec<u32> statuses;
  vec<service::storage_owner::MutationResult> results;
  vec<u64> response_invalidations;

  void clear() {
    kinds.clear();
    decoded_vectors.clear();
    invalidated_neighbors.clear();
    statuses.clear();
    results.clear();
    response_invalidations.clear();
  }
};

struct StorageOwnerThread {
  explicit StorageOwnerThread(u32 id, u32 num_coroutines, i32 max_send_queue_wr)
      : id(id),
        send_wcs(std::max<i32>(1, max_send_queue_wr)),
        post_balances(num_coroutines),
        coroutine_scratch_states(num_coroutines) {
    for (auto& balance : post_balances) {
      balance.store(0, std::memory_order_relaxed);
    }
  }

  void init_peer_scratch(Context& peer_context, size_t bytes, size_t per_coroutine_stride = 0) {
    scratch_stride = per_coroutine_stride == 0
                       ? align_to_cacheline(VamanaNode::total_size())
                       : align_to_cacheline(per_coroutine_stride);
    const size_t required_bytes = static_cast<size_t>(std::max<size_t>(1, post_balances.size())) * scratch_stride;
    scratch_buffer.allocate(std::max(bytes, required_bytes));
    scratch_buffer.touch_memory();
    scratch_region = std::make_unique<LocalMemoryRegion>(
      peer_context, scratch_buffer.get_full_buffer(), scratch_buffer.buffer_size);
  }

  bool has_peer_scratch() const { return scratch_region != nullptr; }

  void set_current_coroutine(u32 coroutine_id) {
    lib_assert(coroutine_id < post_balances.size(), "invalid storage-owner coroutine id");
    running_coroutine = coroutine_id;
  }
  void track_post() { ++post_balances[running_coroutine]; }
  bool is_ready(u32 coroutine_id) const { return post_balances[coroutine_id] == 0; }
  StorageOwnerCoroutineScratch& coroutine_scratch_state() {
    lib_assert(running_coroutine < coroutine_scratch_states.size(),
               "storage-owner coroutine container scratch is not initialized");
    return coroutine_scratch_states[running_coroutine];
  }
  byte_t* coroutine_scratch(size_t extra_offset = 0) {
    const size_t offset = static_cast<size_t>(running_coroutine) * scratch_stride + extra_offset;
    lib_assert(offset < scratch_buffer.buffer_size, "storage-owner coroutine scratch buffer exhausted");
    return scratch_buffer.get_full_buffer() + offset;
  }
  size_t coroutine_scratch_offset(size_t extra_offset = 0) const {
    return static_cast<size_t>(running_coroutine) * scratch_stride + extra_offset;
  }

  u32 id{};
  vec<ibv_wc> send_wcs;
  vec<std::atomic<i32>> post_balances;
  vec<StorageOwnerCoroutineScratch> coroutine_scratch_states;
  StorageOwnerRequestScratch request_scratch;
  HugePage<byte_t> scratch_buffer;
  std::unique_ptr<LocalMemoryRegion> scratch_region;
  u32 running_coroutine{};
  // Used only by the owning OS thread. Async reads rotate across every data
  // QP instead of pinning a low worker count to a strict subset of the lanes.
  u32 next_peer_data_qp_ticket{};
  size_t scratch_stride{};
};

using FreshnessEntry =
  memory_node_storage_owner_index_detail::AuthorityDirectoryEntry;

}  // namespace memory_node_detail
