#include "vamana/vamana_node.hh"

#include "compute_thread.hh"

VamanaNode::~VamanaNode() {
  if (buffer_slice_ != nullptr) {
    owner_->buffer_allocator.free_buffer(buffer_slice_, buffer_size_);
  }
}
