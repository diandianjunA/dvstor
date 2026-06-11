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
 *   rabitq_code: next_power_of_two(dim) bits
 *   rabitq_norm: 4B    (float, ||x - centroid||)
 *   rabitq_error: 4B   (float, quantization error correction factor)
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
  static size_t offset_neighbors() { return HEADER_SIZE + META_SIZE + vector_bytes(); }
  static size_t offset_rabitq_code() { return offset_neighbors() + NEIGHBORS_SIZE; }
  static u32 rabitq_code_bits() {
      lib_assert(DIM > 0 && DIM <= (1u << 30), "RaBitQ dimension must be in [1, 2^30]");
      u32 value = 1;
      while (value < DIM) value <<= 1;
      return std::max<u32>(value, 8);
  }
  static size_t rabitq_code_size() { return rabitq_code_bits() / 8; }
  static size_t rabitq_entry_size() { return (rabitq_code_size() + 8 + 7) & ~size_t{7}; }
  static size_t offset_rabitq_norm() { return offset_rabitq_code() + rabitq_code_size(); }
  static size_t offset_rabitq_error() { return offset_rabitq_norm() + sizeof(float); }

  static size_t vector_bytes() { return VECTOR_BYTES; }
  static size_t size_until_vector_end() { return offset_vector() + vector_bytes(); }
  static size_t total_size() {
      return offset_neighbors() + NEIGHBORS_SIZE
             + (HAS_RABITQ_CODE ? rabitq_entry_size() : 0);
  }

  // RaBitQ: asymmetric binary quantization after a deterministic signed Hadamard rotation.
  inline static bool HAS_RABITQ_CODE = false;
  inline static vec<float> rabitq_centroid;     // DIM floats (global dataset centroid)

  static void enable_rabitq() {
      HAS_RABITQ_CODE = true;
  }

  static void disable_rabitq() {
      HAS_RABITQ_CODE = false;
      rabitq_centroid.clear();
  }

  static void set_rabitq_centroid(const vec<float>& c) {
      if (c.size() == DIM) rabitq_centroid = c;
      else rabitq_centroid.assign(DIM, 0.0f);
  }

public:
  using RabitqCode = vec<byte_t>;

  static bool rabitq_bit(const RabitqCode& code, u32 bit) {
      return (code[bit >> 3] & static_cast<byte_t>(1u << (7u - (bit & 7u)))) != 0;
  }

  static void compute_rotated_query(const byte_t* vector, VectorDType dtype,
                                    float* rotated_out, float* norm2_out) {
      const u32 bits = rabitq_code_bits();
      const float* centroid = rabitq_centroid.empty() ? nullptr : rabitq_centroid.data();
      float norm2 = 0.0f;
      for (u32 d = 0; d < bits; ++d) {
          float value = 0.0f;
          if (d < DIM) {
              value = vector_component_as_float(vector, dtype, d) - (centroid ? centroid[d] : 0.0f);
              norm2 += value * value;
              u32 hash = d + 0x9e3779b9u;
              hash ^= hash >> 16;
              hash *= 0x7feb352du;
              hash ^= hash >> 15;
              value = (hash & 1u) ? value : -value;
          }
          rotated_out[d] = value;
      }
      for (u32 width = 1; width < bits; width <<= 1) {
          for (u32 base = 0; base < bits; base += width << 1) {
              for (u32 offset = 0; offset < width; ++offset) {
                  const float lhs = rotated_out[base + offset];
                  const float rhs = rotated_out[base + width + offset];
                  rotated_out[base + offset] = lhs + rhs;
                  rotated_out[base + width + offset] = lhs - rhs;
              }
          }
      }
      const float scale = 1.0f / std::sqrt(static_cast<float>(bits));
      for (u32 d = 0; d < bits; ++d) rotated_out[d] *= scale;
      *norm2_out = norm2;
  }

  // Compute ||x - centroid|| (L2 norm, not squared).
  static float compute_rabitq_norm(const byte_t* vec, VectorDType dtype) {
      vec<float> rotated(rabitq_code_bits());
      float norm2 = 0.0f;
      compute_rotated_query(vec, dtype, rotated.data(), &norm2);
      return std::sqrt(norm2);
  }

  // Compute binary code from the signs of the rotated centered vector.
  static RabitqCode compute_rabitq_code(const byte_t* vec, VectorDType dtype) {
      const u32 bits = rabitq_code_bits();
      vec<float> rotated(bits);
      float norm2 = 0.0f;
      compute_rotated_query(vec, dtype, rotated.data(), &norm2);
      RabitqCode code(rabitq_code_size(), 0);
      for (u32 bit = 0; bit < bits; ++bit) {
          if (rotated[bit] > 0.0f) {
              code[bit >> 3] |= static_cast<byte_t>(1u << (7u - (bit & 7u)));
          }
      }
      return code;
  }

  static void compute_rabitq_entry(const byte_t* vec, VectorDType dtype,
                                   RabitqCode& code, float& norm, float& error) {
      const u32 bits = rabitq_code_bits();
      vec<float> rotated(bits);
      float norm2 = 0.0f;
      compute_rotated_query(vec, dtype, rotated.data(), &norm2);

      code.assign(rabitq_code_size(), 0);
      float signed_dot = 0.0f;
      for (u32 bit = 0; bit < bits; ++bit) {
          const bool positive = rotated[bit] > 0.0f;
          if (positive) {
              code[bit >> 3] |= static_cast<byte_t>(1u << (7u - (bit & 7u)));
          }
          signed_dot += positive ? rotated[bit] : -rotated[bit];
      }

      norm = std::sqrt(norm2);
      if (norm <= 1e-15f) {
          error = 1.0f;
      } else {
          error = std::max(signed_dot / (norm * std::sqrt(static_cast<float>(bits))), 1e-15f);
      }
  }

  // Compute error correction factor: e = (1/√D) * <R*x̄, b>
  // where x̄ = (x - centroid) / ||x - centroid|| and b is the dynamic binary code.
  static float compute_rabitq_error_factor(const byte_t* vec, VectorDType dtype, const RabitqCode& code) {
      const u32 bits = rabitq_code_bits();
      vec<float> rotated(bits);
      float norm2 = 0.0f;
      compute_rotated_query(vec, dtype, rotated.data(), &norm2);
      const float inv_norm = 1.0f / std::max(std::sqrt(norm2), 1e-15f);
      float dot_sum = 0.0f;
      for (u32 bit = 0; bit < bits; ++bit) {
          dot_sum += rotated[bit] * inv_norm * (rabitq_bit(code, bit) ? 1.0f : -1.0f);
      }
      float e = dot_sum / std::sqrt(static_cast<float>(bits));
      return std::max(e, 0.0f);
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
