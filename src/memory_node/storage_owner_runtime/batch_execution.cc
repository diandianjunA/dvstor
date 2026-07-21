#include "memory_node/storage_owner_runtime/detail.hh"

using namespace memory_node_storage_owner_runtime_detail;

size_t MemoryNode::insert_request_slot_offset(u32 client_id, u32 slot_id) const {
  const size_t slot_index =
    static_cast<size_t>(client_id) * insert_runtime_.request_slot_count + slot_id;
  return slot_index * insert_runtime_.request_bytes;
}

size_t MemoryNode::insert_response_slot_offset(const Configuration& config, u32 client_id, u32 slot_id) const {
  const size_t slot_index =
    static_cast<size_t>(client_id) * insert_runtime_.request_slot_count + slot_id;
  return insert_runtime_.response_offset + slot_index * response_slot_bytes(config);
}
