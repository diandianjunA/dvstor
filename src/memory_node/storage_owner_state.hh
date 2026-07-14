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
#include "coroutine.hh"
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
  bool deleted{};
  vec<byte_t> vector_data;
};

struct StorageOwnerPruneCandidateInfo {
  RemotePtr rptr;
  distance_t dist{};
  vec<byte_t> vector_data;
};

struct StorageOwnerCoroutineScratch {
  hashset_t<RemotePtr> visited;
  hashset_t<RemotePtr> empty_skip;
  vec<BeamEntry> beam;
  vec<RemotePtr> unvisited;
  vec<RemotePtr> batch;
  vec<RemotePtr> filtered;
  vec<RemotePtr> selected;
  vec<const byte_t*> selected_vectors;
  vec<StorageOwnerPruneCandidateInfo> prune_infos;
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
    unvisited.clear();
    batch.clear();
  }

  void clear_prune() {
    filtered.clear();
    batch.clear();
    selected.clear();
    selected_vectors.clear();
    prune_infos.clear();
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
};

struct PeerRpcMessage {
  u32 source_shard{};
  vec<byte_t> payload;
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
  vec<u_ptr<StorageOwnerInsertCoroutine>> coroutines;
  HugePage<byte_t> scratch_buffer;
  std::unique_ptr<LocalMemoryRegion> scratch_region;
  u32 running_coroutine{};
  size_t scratch_stride{};
};

struct StorageOwnerInsertJob {
  node_t id{};
  service::storage_owner::MutationKind kind{service::storage_owner::MutationKind::insert};
  vec<byte_t> vector_data;
  service::storage_owner::MutationStatus status{service::storage_owner::MutationStatus::failed};
  bool ok{false};
  RemotePtr new_ptr{};
  RemotePtr old_ptr{};
  u32 generation{};
  u64 maintenance_sequence{};
  vec<RemotePtr> anchor_hints;
  vec<u64> invalidated_neighbors;
};

struct FreshnessEntry {
  RemotePtr current;
  u32 generation{};
  bool deleted{};
};

}  // namespace memory_node_detail
