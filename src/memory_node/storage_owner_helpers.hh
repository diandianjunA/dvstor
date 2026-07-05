#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include "common/configuration.hh"
#include "memory_node/storage_owner_state.hh"

namespace memory_node_detail {

inline constexpr size_t kSnapshotPrefixBytes =
  VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;

inline size_t snapshot_buffer_bytes() {
  return storage_owner_snapshot_bytes();
}

inline size_t aligned_snapshot_bytes() {
  return storage_owner_snapshot_stride();
}

inline u32 storage_owner_construction_width(const configuration::IndexConfiguration& config) {
  const u32 configured = config.storage_owner_construction_beam_width == 0
                           ? config.beam_width_construction
                           : config.storage_owner_construction_beam_width;
  return std::max<u32>(1, std::min(config.beam_width_construction, configured));
}

inline u32 storage_owner_snapshot_batch_size(const configuration::IndexConfiguration& config,
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

inline u32 storage_owner_prune_candidate_limit(const configuration::IndexConfiguration& config) {
  if (config.storage_owner_prune_max_candidates == 0) {
    return std::numeric_limits<u32>::max();
  }
  return std::max(config.R, config.storage_owner_prune_max_candidates);
}

inline bool anchored_update_enabled(const configuration::IndexConfiguration& config,
                                    const vec<RemotePtr>& hints) {
  return config.storage_owner_update_mode == "anchored" && !hints.empty();
}

inline void parse_remote_snapshot(RemotePtr rptr, const byte_t* ptr, NodeSnapshot& snapshot) {
  snapshot = NodeSnapshot{};
  snapshot.rptr = rptr;
  snapshot.header = *reinterpret_cast<const u64*>(ptr);
  snapshot.id = *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_id());
  snapshot.generation = VamanaNode::compact_storage()
    ? *reinterpret_cast<const u32*>(ptr + VamanaNode::offset_generation()) : 0;
  snapshot.deleted = (snapshot.header & VamanaNode::HEADER_DELETED) != 0;
  snapshot.vector_data.resize(VamanaNode::vector_bytes());
  const size_t vector_offset = VamanaNode::compact_storage()
    ? VamanaNode::offset_vector() : kSnapshotPrefixBytes;
  std::memcpy(snapshot.vector_data.data(), ptr + vector_offset, VamanaNode::vector_bytes());
}

}  // namespace memory_node_detail
