#pragma once

#include <algorithm>

#include "remote_pointer.hh"
#include "vamana/storage_layout_resolver.hh"

namespace memory_node_storage_owner_index_detail {

inline size_t graph_read_slot_stride() {
  const size_t bytes = VamanaNode::hot_graph_entry_size();
  return (bytes + kCacheLineBytes - 1) & ~(kCacheLineBytes - 1);
}

inline size_t batched_read_slot_stride(size_t snapshot_stride) {
  return std::max(snapshot_stride, graph_read_slot_stride());
}

// RemotePtr::is_well_formed() only validates the tagged-handle encoding.  A
// well-formed handle can still name any aligned byte offset in the 256-GiB
// representation range, while a storage shard normally exports a much smaller
// MR.  Every graph-read and Stage2 admission boundary must therefore validate
// both the vector and hot-graph ranges before issuing an RDMA operation.
inline bool storage_pointer_addressable(RemotePtr pointer,
                                        u32 shard_count,
                                        u64 shard_bytes) {
  if (pointer.is_null() || !pointer.is_well_formed() ||
      pointer.memory_node() >= shard_count ||
      !VamanaNode::hot_graph_entry_available(pointer)) {
    return false;
  }

  const auto vector = vamana::StorageLayoutResolver::vector(pointer);
  if (vector.offset > shard_bytes ||
      vector.size > shard_bytes - vector.offset) {
    return false;
  }

  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(pointer);
  return hot_offset <= shard_bytes &&
    VamanaNode::hot_graph_entry_size() <= shard_bytes - hot_offset;
}

}  // namespace memory_node_storage_owner_index_detail
