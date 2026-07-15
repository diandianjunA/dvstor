#include "memory_node/storage_owner_maintenance/detail.hh"

using namespace memory_node_storage_owner_maintenance_detail;

bool MemoryNode::try_lock_node(RemotePtr rptr) {
  if (rptr.is_null() || !local_shard(rptr.memory_node())) {
    return false;
  }

  auto* header_ptr = reinterpret_cast<u64*>(
    index_buffer_.get_full_buffer() + vamana::StorageLayoutResolver::header(rptr).offset);
  std::atomic_ref<u64> ref(*header_ptr);
  for (u32 attempt = 0; attempt < 8; ++attempt) {
    u64 header = ref.load(std::memory_order_acquire);
    if ((header & VamanaNode::HEADER_NODE_LOCK) != 0) {
      return false;
    }
    const u64 desired = header | VamanaNode::HEADER_NODE_LOCK;
    if (ref.compare_exchange_weak(header, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

bool MemoryNode::storage_owner_task_current(node_t id, u32 generation, RemotePtr target) {
  DynamicFreshnessShard& shard = dynamic_freshness_shard(id);
  std::lock_guard<std::mutex> lock(shard.mutex);
  const auto dynamic = shard.entries.find(id);
  if (dynamic != shard.entries.end()) {
    return !dynamic->second.deleted &&
           dynamic->second.generation == generation &&
           dynamic->second.current == target;
  }
  const auto& immutable_base = base_idmap_;
  const auto base = immutable_base.find(id);
  return base != immutable_base.end() &&
         !base->second.deleted &&
         base->second.generation == generation &&
         base->second.current == target;
}

vec<RemotePtr> MemoryNode::read_preserved_neighbor_list(RemotePtr rptr) {
  if (rptr.is_null()) {
    return {};
  }
  vec<byte_t> entry(VamanaNode::hot_graph_entry_size());
  const u64 hot_offset = VamanaNode::hot_graph_entry_offset(rptr);
  if (local_shard(rptr.memory_node())) {
    std::memcpy(entry.data(),
                index_buffer_.get_full_buffer() + hot_offset,
                entry.size());
  } else {
    remote_read_bytes(rptr.memory_node(), hot_offset, entry.data(), entry.size(), 0);
  }

  const u8 edge_count = entry[0];
  if (edge_count > VamanaNode::R) {
    return {};
  }
  const u16 expected = vamana::hot_graph::load_u16_le(entry.data() + 2);
  const u16 actual = vamana::hot_graph::checksum16(entry.data(), entry.size());
  if (expected != actual) {
    return {};
  }
  vec<RemotePtr> neighbors;
  neighbors.reserve(edge_count);
  for (u32 i = 0; i < edge_count; ++i) {
    RemotePtr neighbor = vamana::hot_graph::decode_remote_ptr(
      entry.data() + vamana::hot_graph::neighbor_offset(i),
      VamanaNode::HOT_GRAPH_SHARD_BITS);
    if (!neighbor.is_null()) {
      neighbors.push_back(neighbor);
    }
  }
  return neighbors;
}

bool MemoryNode::remove_local_neighbor(RemotePtr target_ptr,
                                       RemotePtr deleted_ptr,
                                       const Configuration&) {
  if (target_ptr.is_null() || deleted_ptr.is_null() || !local_shard(target_ptr.memory_node())) {
    return false;
  }

  lock_node(target_ptr);
  const bool target_deleted =
    (load_local_node_header_acquire(target_ptr) &
     VamanaNode::HEADER_DELETED) != 0;
  if (target_deleted) {
    unlock_node(target_ptr);
    return true;
  }

  vec<RemotePtr> neighbors = read_neighbor_list(target_ptr);
  const auto old_size = neighbors.size();
  neighbors.erase(
    std::remove(neighbors.begin(), neighbors.end(), deleted_ptr),
    neighbors.end());
  if (neighbors.size() != old_size) {
    write_neighbor_list(target_ptr, neighbors);
  }
  unlock_node(target_ptr);
  return true;
}

bool MemoryNode::remove_local_neighbors_batched(
    const dense_hashmap_t<u64, vec<RemotePtr>>& removals,
    const Configuration&) {
  bool success = true;
  for (const auto& [target_raw, deleted_ptrs] : removals) {
    const RemotePtr target_ptr{target_raw};
    if (target_ptr.is_null() || !local_shard(target_ptr.memory_node())) {
      success = false;
      continue;
    }

    lock_node(target_ptr);
    const bool target_deleted =
      (load_local_node_header_acquire(target_ptr) &
       VamanaNode::HEADER_DELETED) != 0;
    if (target_deleted) {
      unlock_node(target_ptr);
      continue;
    }

    vec<RemotePtr> neighbors = read_neighbor_list(target_ptr);
    const auto old_size = neighbors.size();
    neighbors.erase(
      std::remove_if(neighbors.begin(), neighbors.end(), [&](const RemotePtr& neighbor) {
        return std::find(deleted_ptrs.begin(), deleted_ptrs.end(), neighbor) !=
               deleted_ptrs.end();
      }),
      neighbors.end());
    if (neighbors.size() != old_size) {
      write_neighbor_list(target_ptr, neighbors);
    }
    unlock_node(target_ptr);
  }
  return success;
}
