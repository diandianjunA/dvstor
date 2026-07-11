#pragma once

#include <cstddef>

#include "common/types.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace vamana {

class StorageLayoutResolver {
public:
  struct Address {
    u32 memory_node{};
    u64 offset{};
    size_t size{};
  };

  struct NeighborRead {
    Address address;
  };

  static Address header(RemotePtr ptr) {
    return {ptr.memory_node(), ptr.byte_offset(), VamanaNode::HEADER_SIZE};
  }

  static Address id(RemotePtr ptr) {
    return {ptr.memory_node(), ptr.byte_offset() + VamanaNode::offset_id(), VamanaNode::ID_SIZE};
  }

  static Address generation(RemotePtr ptr) {
    return {ptr.memory_node(), ptr.byte_offset() + VamanaNode::offset_generation(),
            VamanaNode::GENERATION_SIZE};
  }

  static Address vector(RemotePtr ptr) {
    return {ptr.memory_node(), ptr.byte_offset() + VamanaNode::offset_vector(),
            VamanaNode::vector_bytes()};
  }

  static NeighborRead neighbor_read(RemotePtr ptr) {
    return {{ptr.memory_node(), VamanaNode::hot_graph_entry_offset(ptr),
             VamanaNode::hot_graph_entry_size()}};
  }

  static Address neighbor_slots(RemotePtr ptr) {
    return {ptr.memory_node(), VamanaNode::hot_graph_entry_offset(ptr),
            VamanaNode::hot_graph_entry_size()};
  }

  static u64 allocation_size() { return VamanaNode::allocation_size(); }

  static bool ptr_in_bounds(RemotePtr ptr, u64 shard_cap) {
    return !ptr.is_null() && ptr.byte_offset() + VamanaNode::total_size() <= shard_cap;
  }
};

}  // namespace vamana
