#pragma once

#include <ostream>

#include "remote_pointer.hh"

// forward declaration
class ComputeThread;

/**
 *  Vamana Node layout (single-layer, fixed-size):
 *  [
 *     header: 8B                        | ... | is_medoid(1b) | ... | medoid_lock(1b) | ... | lock(1b) |
 *                                              ^----- 1B -----^ ^--------- 1B ---------^ ^----- 1B -----^
 *     id: 4B                            | uid(4) |
 *     edge_count: 1B                    | count(1) |
 *     padding: 3B                       | pad(3) |
 *     vector: d * 4B                    | v_1(4) | ... | v_d(4) |
 *     rabitq: rabitq_size B             | quantized_data | add(4) | rescale(4) |
 *     neighbors: R * 8B                | n_1(8) | ... | n_R(8) |
 *  ]
 *
 *  All nodes have the same fixed size (no per-level variability as in HNSW).
 */
class VamanaNode {
public:  // static storage
  // Header bit positions (little endian)
  static constexpr size_t HEADER_NODE_LOCK = 0b01;
  static constexpr size_t HEADER_MEDOID_LOCK = 0b100000000;  // was HEADER_NEW_LEVEL_LOCK
  static constexpr size_t HEADER_IS_MEDOID = 0b10000000000000000;  // was HEADER_ENTRY_NODE
  static constexpr size_t HEADER_SIZE = sizeof(u64);
  static constexpr size_t ID_SIZE = sizeof(u32);
  static constexpr size_t EDGE_COUNT_SIZE = sizeof(u8);
  static constexpr size_t PADDING_SIZE = 3;
  static constexpr size_t META_SIZE = ID_SIZE + EDGE_COUNT_SIZE + PADDING_SIZE;  // 8B total

  // Header byte positions (little endian)
  static constexpr size_t HEADER_UNTIL_LOCK = 0;
  static constexpr size_t HEADER_UNTIL_MEDOID_LOCK = 1;
  static constexpr size_t HEADER_UNTIL_IS_MEDOID = 2;

  // Initialized by the Vamana instance
  inline static u32 DIM;
  inline static u32 R;           // max out-degree
  inline static u32 RABITQ_BITS; // bits per dimension
  inline static u32 RABITQ_SIZE; // total bytes for RaBitQ data
  inline static u32 NEIGHBORS_SIZE; // R * sizeof(RemotePtr)

  static void init_static_storage(u32 dim, u32 max_degree, u32 rabitq_bits_per_dim) {
    DIM = dim;
    R = max_degree;
    RABITQ_BITS = rabitq_bits_per_dim;
    // RaBitQ data: packed bits + add(float) + rescale(float)
    RABITQ_SIZE = (rabitq_bits_per_dim * dim + 7) / 8 + 2 * sizeof(f32);
    NEIGHBORS_SIZE = max_degree * sizeof(u64);  // RemotePtr is u64
  }

  // Offsets within the node buffer
  static size_t offset_id() { return HEADER_SIZE; }
  static size_t offset_edge_count() { return HEADER_SIZE + ID_SIZE; }
  static size_t offset_vector() { return HEADER_SIZE + META_SIZE; }
  static size_t offset_rabitq() { return HEADER_SIZE + META_SIZE + DIM * sizeof(element_t); }
  static size_t offset_neighbors() { return HEADER_SIZE + META_SIZE + DIM * sizeof(element_t) + RABITQ_SIZE; }

  // Size calculations
  static size_t size_until_vector_end() { return HEADER_SIZE + META_SIZE + DIM * sizeof(element_t); }
  static size_t total_size() {
    return HEADER_SIZE + META_SIZE + DIM * sizeof(element_t) + RABITQ_SIZE + NEIGHBORS_SIZE;
  }

public:  // member storage
  VamanaNode() = default;
  VamanaNode(byte_t* buffer_ptr, size_t buffer_size, const RemotePtr& rptr, ComputeThread* owner)
      : owner_(owner), buffer_slice_(buffer_ptr), buffer_size_(buffer_size), rptr(rptr) {}

  // Rule of 5: use shared_ptr for ownership, prevent copies/moves
  VamanaNode(const VamanaNode&) = delete;
  VamanaNode(VamanaNode&&) noexcept = delete;
  VamanaNode& operator=(const VamanaNode&) = delete;
  VamanaNode& operator=(VamanaNode&&) noexcept = delete;

  ~VamanaNode();

  // Equality
  bool operator==(const VamanaNode& other) const { return id() == other.id(); }

  // Accessors
  u32 id() const { return *reinterpret_cast<u32*>(buffer_slice_ + offset_id()); }
  u8 edge_count() const { return *reinterpret_cast<u8*>(buffer_slice_ + offset_edge_count()); }
  u64& header() const { return *reinterpret_cast<u64*>(buffer_slice_); }

  span<element_t> components() const {
    return {reinterpret_cast<element_t*>(buffer_slice_ + offset_vector()), DIM};
  }

  byte_t* rabitq_data() const {
    return buffer_slice_ + offset_rabitq();
  }

  span<RemotePtr> neighbors() const {
    return {reinterpret_cast<RemotePtr*>(buffer_slice_ + offset_neighbors()),
            static_cast<size_t>(edge_count())};
  }

  span<RemotePtr> all_neighbor_slots() const {
    return {reinterpret_cast<RemotePtr*>(buffer_slice_ + offset_neighbors()),
            static_cast<size_t>(R)};
  }

  // Mutators
  void set_edge_count(u8 count) {
    *reinterpret_cast<u8*>(buffer_slice_ + offset_edge_count()) = count;
  }

  void set_id(u32 uid) {
    *reinterpret_cast<u32*>(buffer_slice_ + offset_id()) = uid;
  }

  // Header bit operations
  bool is_locked() const { return header() & HEADER_NODE_LOCK; }
  bool is_medoid_locked() const { return header() & HEADER_MEDOID_LOCK; }
  bool is_medoid() const { return header() & HEADER_IS_MEDOID; }

  void set_lock() { header() |= HEADER_NODE_LOCK; }
  void reset_lock() { header() &= ~HEADER_NODE_LOCK; }
  void set_medoid_lock() { header() |= HEADER_MEDOID_LOCK; }
  void reset_medoid_lock() { header() &= ~HEADER_MEDOID_LOCK; }
  void set_is_medoid() { header() |= HEADER_IS_MEDOID; }
  void reset_is_medoid() { header() &= ~HEADER_IS_MEDOID; }

  // Buffer access
  ComputeThread* get_owner() const { return owner_; }
  byte_t* get_underlying_buffer() const { return buffer_slice_; }

  // Remote offset for the neighbor list portion of this node
  u64 compute_remote_neighbors_offset() const {
    return rptr.byte_offset() + offset_neighbors();
  }

  // Remote offset for the rabitq portion
  u64 compute_remote_rabitq_offset() const {
    return rptr.byte_offset() + offset_rabitq();
  }

  // Remote offset for edge_count (1 byte)
  u64 compute_remote_edge_count_offset() const {
    return rptr.byte_offset() + offset_edge_count();
  }

  friend std::ostream& operator<<(std::ostream& os, const VamanaNode& n) {
    os << "VamanaNode{id=" << n.id() << ", edges=" << static_cast<int>(n.edge_count())
       << ", rptr=" << n.rptr << "}";
    return os;
  }

private:
  ComputeThread* owner_{};
  byte_t* buffer_slice_{};  // points to local buffer
  size_t buffer_size_{};

public:
  RemotePtr rptr;  // points to remote memory
};
