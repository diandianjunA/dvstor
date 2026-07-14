#include "memory_node/memory_node.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
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

#include "memory_node/storage_owner_index/allocation.ipp"
#include "memory_node/storage_owner_index/graph_access.ipp"
#include "memory_node/storage_owner_index/candidate_search.ipp"
#include "memory_node/storage_owner_index/graph_mutation.ipp"
