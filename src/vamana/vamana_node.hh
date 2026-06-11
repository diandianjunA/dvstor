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
 *   rabitq_code: 16B   (128-bit, two u64 halves)
 *   rabitq_norm: 4B    (float, centred L2 norm)
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
  static constexpr size_t RABITQ_NORM_SIZE = sizeof(float);          // 4 bytes
  static size_t offset_neighbors() { return HEADER_SIZE + META_SIZE + vector_bytes(); }
  static size_t offset_rabitq_code() { return offset_neighbors() + NEIGHBORS_SIZE; }
  static size_t offset_rabitq_norm() { return offset_rabitq_code() + RABITQ_CODE_SIZE; }

  static size_t vector_bytes() { return VECTOR_BYTES; }
  static size_t size_until_vector_end() { return offset_vector() + vector_bytes(); }
  static size_t total_size() {
      return offset_neighbors() + NEIGHBORS_SIZE
             + (HAS_RABITQ_CODE ? RABITQ_CODE_SIZE + RABITQ_NORM_SIZE : 0);
  }

  // RaBitQ: 128-bit codes via random orthogonal projection.
  inline static bool HAS_RABITQ_CODE = false;
  inline static vec<float> rabitq_proj_matrix;  // DIM × 128
  inline static float rabitq_scale_coarse = 1.0f;

  static void enable_rabitq() {
      if (HAS_RABITQ_CODE) return;
      HAS_RABITQ_CODE = true;
      init_rabitq_matrix();
  }

  static void init_rabitq_matrix() {
      if (!rabitq_proj_matrix.empty() || DIM == 0) return;
      const u32 b = RABITQ_CODE_BITS;
      rabitq_proj_matrix.resize(static_cast<size_t>(DIM) * b);
      // Random Gaussian matrix
      uint32_t seed = 42;
      auto rand_gaussian = [&]() -> float {
          // Box-Muller transform
          seed = seed * 1103515245 + 12345;
          float u1 = static_cast<float>(seed & 0x7FFFFFFF) / 2147483648.0f;
          seed = seed * 1103515245 + 12345;
          float u2 = static_cast<float>(seed & 0x7FFFFFFF) / 2147483648.0f;
          return std::sqrt(-2.0f * std::log(std::max(u1, 1e-9f)))
                 * std::cos(6.2831853f * u2);
      };
      for (size_t i = 0; i < rabitq_proj_matrix.size(); ++i)
          rabitq_proj_matrix[i] = rand_gaussian();
      // Gram-Schmidt orthogonalization of columns
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
      // Coarse quantizer scale: average absolute projection value
      float avg_abs = 0.0f;
      for (size_t i = 0; i < rabitq_proj_matrix.size(); ++i)
          avg_abs += std::abs(rabitq_proj_matrix[i]);
      rabitq_scale_coarse = avg_abs / static_cast<float>(rabitq_proj_matrix.size());
  }

public:
  // RaBitQ 128-bit code: two u64 halves [hi, lo].
  struct RabitqCode { u64 hi; u64 lo; };

  // Compute centred L2 norm (for distance estimation).
  static float compute_rabitq_norm(const byte_t* vec, VectorDType dtype) {
      switch (dtype) {
          case VectorDType::float32: {
              const auto* fv = reinterpret_cast<const float*>(vec);
              float mean = 0.0f;
              for (u32 d = 0; d < DIM; ++d) mean += fv[d];
              mean /= static_cast<float>(DIM);
              float nsq = 0.0f;
              for (u32 d = 0; d < DIM; ++d) { float c = fv[d] - mean; nsq += c * c; }
              return nsq;
          }
          default: {
              // Use float centering, consistent with compute_rabitq_code.
              float mean = 0.0f;
              for (u32 d = 0; d < DIM; ++d) mean += static_cast<float>(static_cast<int>(vec[d]));
              mean /= static_cast<float>(DIM);
              float nsq = 0.0f;
              for (u32 d = 0; d < DIM; ++d) {
                  float c = static_cast<float>(static_cast<int>(vec[d])) - mean;
                  nsq += c * c;
              }
              return nsq;
          }
      }
  }

  static RabitqCode compute_rabitq_code(const byte_t* vec, VectorDType dtype) {
      const u32 b = RABITQ_CODE_BITS;
      u64 hi = 0, lo = 0;
      switch (dtype) {
          case VectorDType::float32: {
              const auto* fv = reinterpret_cast<const float*>(vec);
              float mean = 0.0f;
              for (u32 d = 0; d < DIM; ++d) mean += fv[d];
              mean /= static_cast<float>(DIM);
              for (u32 j = 0; j < 64; ++j) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - mean) * rabitq_proj_matrix[d * b + j];
                  if (sum > 0.0f) lo |= (1ULL << (63 - j));
              }
              for (u32 j = 64; j < b; ++j) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - mean) * rabitq_proj_matrix[d * b + j];
                  if (sum > 0.0f) hi |= (1ULL << (127 - j));
              }
              break;
          }
          default: {
              // Convert integer vector to float for projection onto the
              // orthonormal float matrix (cast-to-int would truncate all
              // |values|<1 to zero).
              vec<float> fv(DIM);
              float mean = 0.0f;
              for (u32 d = 0; d < DIM; ++d) {
                  fv[d] = static_cast<float>(static_cast<int>(vec[d]));
                  mean += fv[d];
              }
              mean /= static_cast<float>(DIM);
              for (u32 j = 0; j < 64; ++j) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - mean) * rabitq_proj_matrix[d * b + j];
                  if (sum > 0.0f) lo |= (1ULL << (63 - j));
              }
              for (u32 j = 64; j < b; ++j) {
                  float sum = 0.0f;
                  for (u32 d = 0; d < DIM; ++d)
                      sum += (fv[d] - mean) * rabitq_proj_matrix[d * b + j];
                  if (sum > 0.0f) hi |= (1ULL << (127 - j));
              }
              break;
          }
      }
      return {hi, lo};
  }

  // Approximate L2 distance from RaBitQ codes using arcsin formula:
  //   ||q-v||² ≈ qn2 + vn2 - 2*√(qn2)*√(vn2)*cos(π*popcount/128)
  static float rabitq_approx_l2(float q_norm2, float v_norm2, u32 popcount) {
      float qn = std::sqrt(std::max(q_norm2, 0.0f));
      float vn = std::sqrt(std::max(v_norm2, 0.0f));
      float angle = 3.14159265f * static_cast<float>(popcount) / static_cast<float>(RABITQ_CODE_BITS);
      float cos_angle = std::cos(angle);
      float d2 = q_norm2 + v_norm2 - 2.0f * qn * vn * cos_angle;
      return std::max(d2, 0.0f);
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
