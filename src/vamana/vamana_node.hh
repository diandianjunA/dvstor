#pragma once

#include <ostream>

#include <library/utils.hh>

#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "remote_pointer.hh"

class ComputeThread;

/**
 * Vamana node layout (single-layer, fixed-size):
 * [
 *   header: 8B
 *   id: 4B
 *   edge_count: 1B
 *   padding: 3B
 *   vector: vector_bytes
 *   neighbors: R * 8B
 *   rabitq_code: 8B
 * ]
 */
class VamanaNode {
public:
  static constexpr size_t HEADER_NODE_LOCK = 0b01;
  static constexpr size_t HEADER_MEDOID_LOCK = 0b100000000;
  static constexpr size_t HEADER_IS_MEDOID = 0b10000000000000000;
  static constexpr size_t HEADER_SIZE = sizeof(u64);
  static constexpr size_t ID_SIZE = sizeof(u32);
  static constexpr size_t EDGE_COUNT_SIZE = sizeof(u8);
  static constexpr size_t PADDING_SIZE = 3;
  static constexpr size_t META_SIZE = ID_SIZE + EDGE_COUNT_SIZE + PADDING_SIZE;

  static constexpr size_t HEADER_UNTIL_LOCK = 0;
  static constexpr size_t HEADER_UNTIL_MEDOID_LOCK = 1;
  static constexpr size_t HEADER_UNTIL_IS_MEDOID = 2;

  inline static u32 DIM{};
  inline static u32 R{};
  inline static u32 NEIGHBORS_SIZE{};
  inline static VectorDType VECTOR_DTYPE{VectorDType::float32};
  inline static u32 VECTOR_COMPONENT_SIZE{sizeof(element_t)};
  inline static u32 VECTOR_BYTES{0};

  static void init_static_storage(u32 dim,
                                  u32 max_degree,
                                  VectorDType vector_dtype = VectorDType::float32) {
    DIM = dim;
    R = max_degree;
    VECTOR_DTYPE = vector_dtype;
    VECTOR_COMPONENT_SIZE = static_cast<u32>(vector_dtype_component_size(vector_dtype));
    VECTOR_BYTES = static_cast<u32>(vector_dtype_bytes(vector_dtype, dim));
    NEIGHBORS_SIZE = max_degree * sizeof(u64);
  }

  static VectorDType vector_dtype() { return VECTOR_DTYPE; }
  static str vector_dtype_name() { return ::vector_dtype_name(VECTOR_DTYPE); }
  static size_t vector_component_size() { return VECTOR_COMPONENT_SIZE; }
  static str layout_name() { return HAS_RABITQ_CODE ? "rabitq" : "standard"; }

  static size_t offset_id() { return HEADER_SIZE; }
  static size_t offset_edge_count() { return HEADER_SIZE + ID_SIZE; }
  static size_t offset_vector() { return HEADER_SIZE + META_SIZE; }
  static constexpr size_t RABITQ_CODE_SIZE = sizeof(u64);
  static size_t offset_neighbors() { return HEADER_SIZE + META_SIZE + vector_bytes(); }
  static size_t offset_rabitq_code() { return offset_neighbors() + NEIGHBORS_SIZE; }

  static size_t vector_bytes() { return VECTOR_BYTES; }
  static size_t size_until_vector_end() { return offset_vector() + vector_bytes(); }
  static size_t total_size() { return offset_neighbors() + NEIGHBORS_SIZE + (HAS_RABITQ_CODE ? RABITQ_CODE_SIZE : 0); }

  // RaBitQ projection matrix + code computation
  inline static bool HAS_RABITQ_CODE = false;
  inline static vec<int8_t> rabitq_proj_matrix;
  inline static float rabitq_scaling = 1.0f;

  static void enable_rabitq() {
      if (HAS_RABITQ_CODE) return;
      HAS_RABITQ_CODE = true;
      init_rabitq_matrix();
  }

  static void init_rabitq_matrix() {
      if (!rabitq_proj_matrix.empty()) return;
      rabitq_proj_matrix.resize(static_cast<size_t>(DIM) * 64);
      uint32_t seed = 42;
      for (size_t i = 0; i < rabitq_proj_matrix.size(); ++i) {
          seed = seed * 1103515245 + 12345;
          rabitq_proj_matrix[i] = (seed & 1) ? 1 : -1;
      }
      rabitq_scaling = static_cast<float>(DIM) / 32.0f;
  }

public:
  static uint64_t compute_rabitq_code(const byte_t* vec, VectorDType dtype) {
      switch (dtype) {
          case VectorDType::float32: {
              const auto* fv = reinterpret_cast<const float*>(vec);
              float mean = 0.0f;
              for (u32 d = 0; d < DIM; ++d) mean += fv[d];
              mean /= static_cast<float>(DIM);
              uint64_t code = 0;
              for (int b = 0; b < 64; ++b) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - mean) * static_cast<float>(rabitq_proj_matrix[d * 64 + b]);
                  if (sum > 0.0f) code |= (1ULL << (63 - b));
              }
              return code;
          }
          case VectorDType::uint8: {
              const auto* uv = reinterpret_cast<const u8*>(vec);
              int mean = 0;
              for (u32 d = 0; d < DIM; ++d) mean += static_cast<int>(uv[d]);
              mean /= static_cast<int>(DIM);
              uint64_t code = 0;
              for (int b = 0; b < 64; ++b) {
                  int sum = 0;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (static_cast<int>(uv[d]) - mean) * rabitq_proj_matrix[d * 64 + b];
                  if (sum > 0) code |= (1ULL << (63 - b));
              }
              return code;
          }
          case VectorDType::int8: {
              const auto* sv = reinterpret_cast<const i8*>(vec);
              int mean = 0;
              for (u32 d = 0; d < DIM; ++d) mean += static_cast<int>(sv[d]);
              mean /= static_cast<int>(DIM);
              uint64_t code = 0;
              for (int b = 0; b < 64; ++b) {
                  int sum = 0;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (static_cast<int>(sv[d]) - mean) * rabitq_proj_matrix[d * 64 + b];
                  if (sum > 0) code |= (1ULL << (63 - b));
              }
              return code;
          }
      }
      return 0;
  }

private:

public:
  VamanaNode() = default;
  VamanaNode(byte_t* buffer_ptr, size_t buffer_size, const RemotePtr& rptr, ComputeThread* owner)
      : owner_(owner), buffer_slice_(buffer_ptr), buffer_size_(buffer_size), rptr(rptr) {}

  VamanaNode(const VamanaNode&) = delete;
  VamanaNode(VamanaNode&&) noexcept = delete;
  VamanaNode& operator=(const VamanaNode&) = delete;
  VamanaNode& operator=(VamanaNode&&) noexcept = delete;

  ~VamanaNode();

  bool operator==(const VamanaNode& other) const { return id() == other.id(); }

  u32 id() const { return *reinterpret_cast<u32*>(buffer_slice_ + offset_id()); }
  u8 edge_count() const { return *reinterpret_cast<u8*>(buffer_slice_ + offset_edge_count()); }
  u64& header() const { return *reinterpret_cast<u64*>(buffer_slice_); }

  span<element_t> components() const {
    return {reinterpret_cast<element_t*>(buffer_slice_ + offset_vector()), DIM};
  }

  byte_t* vector_data() const { return buffer_slice_ + offset_vector(); }

  float component_as_float(size_t index) const {
    return vector_component_as_float(vector_data(), VECTOR_DTYPE, index);
  }

  vec<float> components_as_float() const {
    return decode_storage_vector_to_float(vector_data(), VECTOR_DTYPE, DIM);
  }

  span<RemotePtr> neighbors() const {
    return {reinterpret_cast<RemotePtr*>(buffer_slice_ + offset_neighbors()), static_cast<size_t>(edge_count())};
  }

  span<RemotePtr> all_neighbor_slots() const {
    return {reinterpret_cast<RemotePtr*>(buffer_slice_ + offset_neighbors()), static_cast<size_t>(R)};
  }

  void set_edge_count(u8 count) {
    *reinterpret_cast<u8*>(buffer_slice_ + offset_edge_count()) = count;
  }

  void set_id(u32 uid) {
    *reinterpret_cast<u32*>(buffer_slice_ + offset_id()) = uid;
  }

  bool is_locked() const { return header() & HEADER_NODE_LOCK; }
  bool is_medoid_locked() const { return header() & HEADER_MEDOID_LOCK; }
  bool is_medoid() const { return header() & HEADER_IS_MEDOID; }

  void set_lock() { header() |= HEADER_NODE_LOCK; }
  void reset_lock() { header() &= ~HEADER_NODE_LOCK; }
  void set_medoid_lock() { header() |= HEADER_MEDOID_LOCK; }
  void reset_medoid_lock() { header() &= ~HEADER_MEDOID_LOCK; }
  void set_is_medoid() { header() |= HEADER_IS_MEDOID; }
  void reset_is_medoid() { header() &= ~HEADER_IS_MEDOID; }

  ComputeThread* get_owner() const { return owner_; }
  byte_t* get_underlying_buffer() const { return buffer_slice_; }

  u64 compute_remote_neighbors_offset() const {
    return rptr.byte_offset() + offset_neighbors();
  }

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
  byte_t* buffer_slice_{};
  size_t buffer_size_{};

public:
  RemotePtr rptr;
};
