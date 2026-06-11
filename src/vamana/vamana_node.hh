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
 *   rabitq_code: 16B   (128-bit binary code, two u64: lo then hi)
 *   rabitq_norm: 4B    (float, ||x - centroid||²)
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
  static constexpr size_t RABITQ_CODE_BITS = 128;
  static constexpr size_t RABITQ_CODE_SIZE = RABITQ_CODE_BITS / 8;  // 16 bytes
  // Entry: code(16B) + x_norm(4B) + error_factor(4B) + reserved(8B) = 32B
  static constexpr size_t RABITQ_ENTRY_SIZE = 32;
  static size_t offset_neighbors() { return HEADER_SIZE + META_SIZE + vector_bytes(); }
  static size_t offset_rabitq_code() { return offset_neighbors() + NEIGHBORS_SIZE; }
  static size_t offset_rabitq_norm() { return offset_rabitq_code() + RABITQ_CODE_SIZE; }
  static size_t offset_rabitq_error() { return offset_rabitq_norm() + sizeof(float); }

  static size_t vector_bytes() { return VECTOR_BYTES; }
  static size_t size_until_vector_end() { return offset_vector() + vector_bytes(); }
  static size_t total_size() {
      return offset_neighbors() + NEIGHBORS_SIZE
             + (HAS_RABITQ_CODE ? RABITQ_ENTRY_SIZE : 0);
  }

  // RaBitQ: asymmetric 128-bit quantization with error correction.
  inline static bool HAS_RABITQ_CODE = false;
  inline static vec<float> rabitq_proj_matrix;  // DIM × 128 (column-major)
  inline static vec<float> rabitq_centroid;     // DIM floats (global dataset centroid)

  static void enable_rabitq() {
      if (HAS_RABITQ_CODE) return;
      HAS_RABITQ_CODE = true;
      init_rabitq_matrix();
  }

  static void set_rabitq_centroid(const vec<float>& c) {
      if (c.size() == DIM) rabitq_centroid = c;
      else rabitq_centroid.assign(DIM, 0.0f);
  }

  static void init_rabitq_matrix() {
      if (!rabitq_proj_matrix.empty() || DIM == 0) return;
      const u32 b = RABITQ_CODE_BITS;
      rabitq_proj_matrix.resize(static_cast<size_t>(DIM) * b);
      uint32_t seed = 42;
      auto rand_gaussian = [&]() -> float {
          seed = seed * 1103515245 + 12345;
          float u1 = static_cast<float>(seed & 0x7FFFFFFF) / 2147483648.0f;
          seed = seed * 1103515245 + 12345;
          float u2 = static_cast<float>(seed & 0x7FFFFFFF) / 2147483648.0f;
          return std::sqrt(-2.0f * std::log(std::max(u1, 1e-9f)))
                 * std::cos(6.2831853f * u2);
      };
      for (size_t i = 0; i < rabitq_proj_matrix.size(); ++i)
          rabitq_proj_matrix[i] = rand_gaussian();
      // Gram-Schmidt orthonormalization of columns
      for (u32 j = 0; j < b; ++j) {
          for (u32 i = 0; i < j; ++i) {
              float dot = 0.0f;
              for (u32 d = 0; d < DIM; ++d)
                  dot += rabitq_proj_matrix[d * b + i] * rabitq_proj_matrix[d * b + j];
              for (u32 d = 0; d < DIM; ++d)
                  rabitq_proj_matrix[d * b + j] -= dot * rabitq_proj_matrix[d * b + i];
          }
          float norm = 0.0f;
          for (u32 d = 0; d < DIM; ++d)
              norm += rabitq_proj_matrix[d * b + j] * rabitq_proj_matrix[d * b + j];
          norm = std::sqrt(norm);
          if (norm > 1e-9f)
              for (u32 d = 0; d < DIM; ++d)
                  rabitq_proj_matrix[d * b + j] /= norm;
      }
  }

public:
  struct RabitqCode { u64 hi; u64 lo; };

  // Compute ||x - centroid|| (L2 norm, not squared).
  static float compute_rabitq_norm(const byte_t* vec, VectorDType dtype) {
      const float* c = rabitq_centroid.empty() ? nullptr : rabitq_centroid.data();
      switch (dtype) {
          case VectorDType::float32: {
              const auto* fv = reinterpret_cast<const float*>(vec);
              float nsq = 0.0f;
              for (u32 d = 0; d < DIM; ++d) {
                  float diff = fv[d] - (c ? c[d] : 0.0f);
                  nsq += diff * diff;
              }
              return std::sqrt(nsq);
          }
          default: {
              float nsq = 0.0f;
              for (u32 d = 0; d < DIM; ++d) {
                  float diff = static_cast<float>(static_cast<int>(vec[d])) - (c ? c[d] : 0.0f);
                  nsq += diff * diff;
              }
              return std::sqrt(nsq);
          }
      }
  }

  // Compute binary code: b = sign(R * (x - centroid)).
  static RabitqCode compute_rabitq_code(const byte_t* vec, VectorDType dtype) {
      const u32 b = RABITQ_CODE_BITS;
      const float* c = rabitq_centroid.empty() ? nullptr : rabitq_centroid.data();
      u64 hi = 0, lo = 0;
      switch (dtype) {
          case VectorDType::float32: {
              const auto* fv = reinterpret_cast<const float*>(vec);
              for (u32 j = 0; j < 64; ++j) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - (c ? c[d] : 0.0f)) * rabitq_proj_matrix[d * b + j];
                  if (sum > 0.0f) lo |= (1ULL << (63 - j));
              }
              for (u32 j = 64; j < b; ++j) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - (c ? c[d] : 0.0f)) * rabitq_proj_matrix[d * b + j];
                  if (sum > 0.0f) hi |= (1ULL << (127 - j));
              }
              break;
          }
          default: {
              float fv[DIM];
              for (u32 d = 0; d < DIM; ++d)
                  fv[d] = static_cast<float>(static_cast<int>(vec[d]));
              for (u32 j = 0; j < 64; ++j) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - (c ? c[d] : 0.0f)) * rabitq_proj_matrix[d * b + j];
                  if (sum > 0.0f) lo |= (1ULL << (63 - j));
              }
              for (u32 j = 64; j < b; ++j) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - (c ? c[d] : 0.0f)) * rabitq_proj_matrix[d * b + j];
                  if (sum > 0.0f) hi |= (1ULL << (127 - j));
              }
              break;
          }
      }
      return {hi, lo};
  }

  // Compute error correction factor: e = (1/√D) * <R*x̄, b>
  // where x̄ = (x - centroid) / ||x - centroid||, b ∈ {-1,+1}^128.
  static float compute_rabitq_error_factor(const byte_t* vec, VectorDType dtype, RabitqCode code) {
      const u32 b = RABITQ_CODE_BITS;
      const float* c = rabitq_centroid.empty() ? nullptr : rabitq_centroid.data();
      float x_norm = compute_rabitq_norm(vec, dtype);
      float inv_norm = 1.0f / std::max(x_norm, 1e-15f);

      float dot_sum = 0.0f;
      switch (dtype) {
          case VectorDType::float32: {
              const auto* fv = reinterpret_cast<const float*>(vec);
              for (u32 j = 0; j < 64; ++j) {
                  float proj = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      proj += (fv[d] - (c ? c[d] : 0.0f)) * inv_norm * rabitq_proj_matrix[d * b + j];
                  float sign = (code.lo & (1ULL << (63 - j))) ? 1.0f : -1.0f;
                  dot_sum += proj * sign;
              }
              for (u32 j = 64; j < b; ++j) {
                  float proj = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      proj += (fv[d] - (c ? c[d] : 0.0f)) * inv_norm * rabitq_proj_matrix[d * b + j];
                  float sign = (code.hi & (1ULL << (127 - j))) ? 1.0f : -1.0f;
                  dot_sum += proj * sign;
              }
              break;
          }
          default: {
              float fv[DIM];
              for (u32 d = 0; d < DIM; ++d)
                  fv[d] = static_cast<float>(static_cast<int>(vec[d]));
              for (u32 j = 0; j < 64; ++j) {
                  float proj = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      proj += (fv[d] - (c ? c[d] : 0.0f)) * inv_norm * rabitq_proj_matrix[d * b + j];
                  float sign = (code.lo & (1ULL << (63 - j))) ? 1.0f : -1.0f;
                  dot_sum += proj * sign;
              }
              for (u32 j = 64; j < b; ++j) {
                  float proj = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      proj += (fv[d] - (c ? c[d] : 0.0f)) * inv_norm * rabitq_proj_matrix[d * b + j];
                  float sign = (code.hi & (1ULL << (127 - j))) ? 1.0f : -1.0f;
                  dot_sum += proj * sign;
              }
              break;
          }
      }
      float e = dot_sum / std::sqrt(static_cast<float>(b));
      return std::max(e, 0.0f);
  }

  // Compute the full-precision rotated query for asymmetric distance.
  // rotated_out[j] = Σ_d (q[d] - centroid[d]) * proj_matrix[d*128 + j]
  static void compute_rotated_query(const byte_t* query, VectorDType dtype,
                                     float* rotated_out, float* norm2_out) {
      const u32 b = RABITQ_CODE_BITS;
      const float* c = rabitq_centroid.empty() ? nullptr : rabitq_centroid.data();
      float norm2 = 0.0f;

      switch (dtype) {
          case VectorDType::float32: {
              const auto* fv = reinterpret_cast<const float*>(query);
              for (u32 j = 0; j < b; ++j) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - (c ? c[d] : 0.0f)) * rabitq_proj_matrix[d * b + j];
                  rotated_out[j] = sum;
              }
              for (u32 d = 0; d < DIM; ++d) {
                  float diff = fv[d] - (c ? c[d] : 0.0f);
                  norm2 += diff * diff;
              }
              break;
          }
          default: {
              float fv[DIM];
              for (u32 d = 0; d < DIM; ++d)
                  fv[d] = static_cast<float>(static_cast<int>(query[d]));
              for (u32 j = 0; j < b; ++j) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - (c ? c[d] : 0.0f)) * rabitq_proj_matrix[d * b + j];
                  rotated_out[j] = sum;
              }
              for (u32 d = 0; d < DIM; ++d) {
                  float diff = fv[d] - (c ? c[d] : 0.0f);
                  norm2 += diff * diff;
              }
              break;
          }
      }
      *norm2_out = norm2;
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
