#pragma once
#include <library/utils.hh>

#include "common/constants.hh"
#include "common/types.hh"

namespace io {

template <typename T>
class Database {
public:
  Database() = default;

  ~Database() { deallocate(); }
  Database(Database&) = delete;
  Database& operator=(Database&) = delete;

  size_t offset() const { return dim * sizeof(T) + sizeof(node_t); }
  void set_id(idx_t slot, node_t id) { *reinterpret_cast<node_t*>(buffer + slot * offset() + dim * sizeof(T)) = id; }

  span<T> get_components(idx_t slot) { return {reinterpret_cast<T*>(buffer + slot * offset()), dim}; }
  node_t get_id(idx_t slot) const { return *reinterpret_cast<node_t*>(buffer + slot * offset() + dim * sizeof(T)); }

  void allocate(u32 dimension, size_t capacity) {
    dim = dimension;
    buffer_capacity = capacity;
    num_vectors_total = capacity;
    num_vectors_read = 0;
    max_slot = 0;
    allocate();
  }

  void allocate() {
    const size_t capacity = buffer_capacity == 0 ? num_vectors_total : buffer_capacity;
    lib_assert(dim > 0 && capacity > 0, "unable to allocate buffer");

    // we allocate space for all queries (such that routing can copy queries into that buffer)
    size_t buffer_size = capacity * offset();
    while (buffer_size % CACHELINE_SIZE != 0) {
      ++buffer_size;
    }

    buffer = static_cast<byte_t*>(std::aligned_alloc(CACHELINE_SIZE, buffer_size));
    lib_assert(buffer != nullptr, "memory allocation failed");
  }

  void deallocate() {
    if (buffer != nullptr) {
      std::free(buffer);
      buffer = nullptr;
    }
  }

private:
  byte_t* buffer{nullptr};  // [ components | id | ... ]

public:
  u32 dim{0};
  size_t buffer_capacity{0};
  size_t num_vectors_total{0};
  size_t num_vectors_read{0};
  idx_t max_slot{0};
};

using GroundTruth = Database<node_t>;

}  // namespace io
