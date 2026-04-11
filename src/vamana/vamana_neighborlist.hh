#pragma once

#include <library/utils.hh>

#include "common/types.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

// forward declaration
class ComputeThread;

/**
 * Vamana Neighborlist: a view of the neighbor list portion of a remote node.
 * Fixed-size buffer: edge_count(1B) + R * RemotePtr(8B).
 *
 * Unlike HNSW's Neighborlist which varies by level, all Vamana nodes have
 * the same maximum degree R.
 */
class VamanaNeighborlist {
public:
  VamanaNeighborlist() = default;
  VamanaNeighborlist(byte_t* buffer_ptr, size_t buffer_size, ComputeThread* owner)
      : buffer_slice_(buffer_ptr), buffer_size_(buffer_size), owner_(owner) {}

  ~VamanaNeighborlist();

  VamanaNeighborlist(const VamanaNeighborlist&) = delete;
  VamanaNeighborlist(VamanaNeighborlist&&) noexcept = delete;
  VamanaNeighborlist& operator=(const VamanaNeighborlist&) = delete;
  VamanaNeighborlist& operator=(VamanaNeighborlist&&) noexcept = delete;

  // Number of active neighbors
  u8 num_neighbors() const {
    return *reinterpret_cast<u8*>(buffer_slice_);
  }

  // Set number of active neighbors
  void set_num_neighbors(u8 count) {
    *reinterpret_cast<u8*>(buffer_slice_) = count;
  }

  // View of active neighbors (only the first edge_count entries)
  span<RemotePtr> view() const {
    return {reinterpret_cast<RemotePtr*>(buffer_slice_ + sizeof(u8)),
            static_cast<size_t>(num_neighbors())};
  }

  // View of all R neighbor slots
  span<RemotePtr> all_slots() const {
    return {reinterpret_cast<RemotePtr*>(buffer_slice_ + sizeof(u8)),
            static_cast<size_t>(VamanaNode::R)};
  }

  // Add a neighbor (appends to the end)
  void add(const RemotePtr& rptr) {
    u8 count = num_neighbors();
    lib_assert(count < VamanaNode::R, "neighbor list is full");
    reinterpret_cast<RemotePtr*>(buffer_slice_ + sizeof(u8))[count] = rptr;
    set_num_neighbors(count + 1);
  }

  // Reset the neighbor list
  void reset() {
    set_num_neighbors(0);
  }

  // Total buffer size needed
  static size_t buffer_size() {
    return sizeof(u8) + VamanaNode::R * sizeof(u64);
  }

  byte_t* get_underlying_buffer() const { return buffer_slice_; }
  ComputeThread* get_owner() const { return owner_; }

private:
  byte_t* buffer_slice_{};
  size_t buffer_size_{};
  ComputeThread* owner_{};
};
