#pragma once

#include "memory_node/memory_node.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "vamana/idmap.hh"
#include "vamana/storage_layout_resolver.hh"

namespace memory_node_storage_owner_index_detail {

using Configuration = configuration::IndexConfiguration;
using BeamEntry = memory_node_detail::BeamEntry;
using NodeSnapshot = memory_node_detail::NodeSnapshot;
using StorageOwnerCoroutineScratch =
  memory_node_detail::StorageOwnerCoroutineScratch;
using StorageOwnerPruneCandidateInfo =
  memory_node_detail::StorageOwnerPruneCandidateInfo;
using StorageOwnerScoredSnapshot =
  memory_node_detail::StorageOwnerScoredSnapshot;
using StorageOwnerThread = memory_node_detail::StorageOwnerThread;

inline size_t snapshot_buffer_bytes() {
  return memory_node_detail::storage_owner_snapshot_bytes();
}

inline size_t aligned_snapshot_bytes() {
  return memory_node_detail::storage_owner_snapshot_stride();
}

inline u32 storage_owner_construction_width(const Configuration& config) {
  const u32 configured = config.storage_owner_construction_beam_width == 0
                           ? config.beam_width_construction
                           : config.storage_owner_construction_beam_width;
  return std::max<u32>(1, std::min(config.beam_width_construction, configured));
}

inline u32 storage_owner_snapshot_batch_size(
    const Configuration& config, const StorageOwnerThread* thread = nullptr) {
  const u32 configured =
    std::max<u32>(1, config.storage_owner_search_snapshot_batch);
  if (thread == nullptr || !thread->has_peer_scratch()) {
    return configured;
  }
  const size_t stride = aligned_snapshot_bytes();
  const size_t capacity = stride == 0 ? 0 : thread->scratch_stride / stride;
  lib_assert(capacity > 0,
             "storage-owner coroutine scratch cannot hold one snapshot: "
             "snapshot_stride=" +
               std::to_string(stride) + " scratch_stride=" +
               std::to_string(thread->scratch_stride));
  return static_cast<u32>(std::min<size_t>(configured, capacity));
}

inline u32 storage_owner_prune_candidate_limit(const Configuration& config) {
  if (config.storage_owner_prune_max_candidates == 0) {
    return std::numeric_limits<u32>::max();
  }
  return std::max(config.R, config.storage_owner_prune_max_candidates);
}

inline bool anchor_update_enabled(const Configuration& config,
                                  const vec<RemotePtr>& hints) {
  return config.storage_owner_update_mode == "local_stitch" && !hints.empty();
}

inline bool local_stitch_enabled(const Configuration& config) {
  return config.storage_owner_update_mode == "local_stitch";
}

inline void parse_remote_snapshot(RemotePtr remote_pointer, const byte_t* data,
                                  NodeSnapshot& snapshot) {
  snapshot = NodeSnapshot{};
  snapshot.rptr = remote_pointer;
  snapshot.header = *reinterpret_cast<const u64*>(data);
  snapshot.id = *reinterpret_cast<const u32*>(data + VamanaNode::offset_id());
  snapshot.generation =
    *reinterpret_cast<const u32*>(data + VamanaNode::offset_generation());
  snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
  snapshot.vector_data.resize(VamanaNode::vector_bytes());
  std::memcpy(snapshot.vector_data.data(), data + VamanaNode::offset_vector(),
              VamanaNode::vector_bytes());
}

}  // namespace memory_node_storage_owner_index_detail

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

struct MemoryNode::NodeSnapshotReadAwaitable {
  bool ready{};
  RemotePtr remote_pointer;
  byte_t* buffer{};
  memory_node_detail::NodeSnapshot snapshot;

  bool await_ready() const { return ready; }
  static void await_suspend(std::coroutine_handle<>) {}
  memory_node_detail::NodeSnapshot await_resume() {
    if (!ready) {
      memory_node_storage_owner_index_detail::parse_remote_snapshot(
        remote_pointer, buffer, snapshot);
    }
    return std::move(snapshot);
  }
};

struct MemoryNode::NodeSnapshotsReadAwaitable {
  struct PendingRead {
    RemotePtr remote_pointer;
    byte_t* buffer{};
  };

  bool ready{true};
  vec<memory_node_detail::NodeSnapshot> snapshots;
  vec<PendingRead> pending;

  bool await_ready() const { return ready; }
  static void await_suspend(std::coroutine_handle<>) {}
  vec<memory_node_detail::NodeSnapshot> await_resume() {
    for (const PendingRead& read : pending) {
      memory_node_detail::NodeSnapshot snapshot;
      memory_node_storage_owner_index_detail::parse_remote_snapshot(
        read.remote_pointer, read.buffer, snapshot);
      snapshots.push_back(std::move(snapshot));
    }
    return std::move(snapshots);
  }
};

struct MemoryNode::NeighborListReadAwaitable {
  bool ready{};
  RemotePtr remote_pointer;
  byte_t* buffer{};
  vec<RemotePtr> neighbors;
  MemoryNode* node{};

  bool await_ready() const { return ready; }
  static void await_suspend(std::coroutine_handle<>) {}
  vec<RemotePtr> await_resume() {
    if (ready) return std::move(neighbors);
    vec<byte_t> decoded(VamanaNode::neighbor_read_size());
    if (!VamanaNode::decode_hot_graph_entry(buffer, decoded.data())) {
      return node->read_neighbor_list(remote_pointer);
    }
    const byte_t* parse_buffer = decoded.data();
    const u8 edge_count = *reinterpret_cast<const u8*>(
      parse_buffer + VamanaNode::neighbor_count_offset_in_read());
    const auto* slots = reinterpret_cast<const RemotePtr*>(
      parse_buffer + VamanaNode::neighbor_payload_offset_in_read());
    neighbors.reserve(edge_count);
    for (u32 index = 0; index < edge_count && index < VamanaNode::R; ++index) {
      if (!slots[index].is_null()) neighbors.push_back(slots[index]);
    }
    return std::move(neighbors);
  }
};
