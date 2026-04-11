#include "vamana_neighborlist.hh"

#include "compute_thread.hh"

VamanaNeighborlist::~VamanaNeighborlist() {
  if (buffer_slice_ != nullptr) {
    owner_->buffer_allocator.free_buffer(buffer_slice_, buffer_size_);
  }
}
