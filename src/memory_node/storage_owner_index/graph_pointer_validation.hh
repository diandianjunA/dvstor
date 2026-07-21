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

// Validate the immutable part of a storage handle at a local control-plane
// trust boundary.  This intentionally says nothing about the slot's *current*
// incarnation: idempotent control operations can legitimately arrive after
// the named incarnation was retired and the slot was reused.  Callers that
// may mutate or dereference the logical record must perform a separate
// identity snapshot and act only when that exact incarnation still matches.
inline bool local_storage_pointer_addressable(RemotePtr pointer,
                                              u32 owning_shard,
                                              u32 shard_count,
                                              u64 shard_bytes,
                                              bool allow_null = false) {
  if (pointer.is_null()) return allow_null;
  return pointer.memory_node() == owning_shard &&
    storage_pointer_addressable(pointer, shard_count, shard_bytes);
}

// A receipt release names the physical identity that was admitted earlier,
// but it never dereferences or mutates that node.  By the time the authority
// sends the ordered release marker, Stage2/cleanup may already have migrated,
// retired, or even recycled the original slot.  Requiring the pointer's
// incarnation to still be live would therefore turn a successful commit into
// an unreleasable receipt and make the authority retry forever.  Keep the
// wire trust boundary (encoding, owning shard, and exported-MR bounds) while
// deliberately avoiding any current-slot-incarnation check.
inline bool receipt_release_pointer_addressable(RemotePtr pointer,
                                                u32 owning_shard,
                                                u32 shard_count,
                                                u64 shard_bytes,
                                                bool allow_null) {
  return local_storage_pointer_addressable(
    pointer, owning_shard, shard_count, shard_bytes, allow_null);
}

}  // namespace memory_node_storage_owner_index_detail
