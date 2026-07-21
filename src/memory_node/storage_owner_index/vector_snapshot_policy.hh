#pragma once

#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace memory_node_storage_owner_index_detail {

// Stage2 may consume a vector directly from local storage or registered RDMA
// scratch only when one immutable physical incarnation spans the complete
// observation. Deleted and provisional records are not members of the stable
// continuation graph. RETIRING/FROZEN are intentionally not rejected here:
// the existing snapshot path also admits them for search while excluding only
// mutation/query-invalid flags.
inline bool stable_vector_snapshot_valid(RemotePtr pointer,
                                         u64 before,
                                         u64 after,
                                         u32 slot_incarnation) {
  return before == after &&
    (after & (VamanaNode::HEADER_NODE_LOCK |
              VamanaNode::HEADER_DELETED |
              VamanaNode::HEADER_PROVISIONAL)) == 0 &&
    VamanaNode::header_incarnation(after) == pointer.incarnation() &&
    slot_incarnation == pointer.incarnation();
}

}  // namespace memory_node_storage_owner_index_detail
