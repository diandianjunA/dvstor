#pragma once

#include "memory_node/memory_node.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

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
  return config.resolved_storage_owner_construction_width();
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

inline bool parse_remote_snapshot(RemotePtr remote_pointer, const byte_t* data,
                                  NodeSnapshot& snapshot) {
  snapshot = NodeSnapshot{};
  snapshot.rptr = remote_pointer;
  snapshot.header = *reinterpret_cast<const u64*>(data);
  snapshot.id = *reinterpret_cast<const u32*>(data + VamanaNode::offset_id());
  snapshot.generation =
    *reinterpret_cast<const u32*>(data + VamanaNode::offset_generation());
  snapshot.slot_incarnation = *reinterpret_cast<const u32*>(
    data + VamanaNode::offset_slot_incarnation());
  snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
  if ((snapshot.header & VamanaNode::HEADER_NODE_LOCK) != 0 ||
      VamanaNode::header_incarnation(snapshot.header) !=
        remote_pointer.incarnation() ||
      snapshot.slot_incarnation != remote_pointer.incarnation()) {
    snapshot = NodeSnapshot{};
    return false;
  }
  snapshot.vector_data.resize(VamanaNode::vector_bytes());
  std::memcpy(snapshot.vector_data.data(), data + VamanaNode::offset_vector(),
              VamanaNode::vector_bytes());
  return true;
}

}  // namespace memory_node_storage_owner_index_detail
