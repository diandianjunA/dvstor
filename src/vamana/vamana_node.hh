#pragma once

#include <ostream>

#include <library/utils.hh>

#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "remote_pointer.hh"
#include "vamana/hot_graph.hh"
#include "vamana/storage_format.hh"

class ComputeThread;

/**
 * AoS format:
 * [
 *   header: 8B
 *   id: 4B
 *   edge_count: 1B
 *   padding: 3B
 *   neighbors: R * 8B
 *   graph padding: to 64B
 *   vector: vector_bytes, padded to 64B
 *   rabitq_code: next_power_of_two(dim) bits
 *   rabitq_norm: 4B    (float, ||x - centroid||)
 *   rabitq_error: 4B   (float, quantization error correction factor)
 *   rabitq padding: to 64B
 * ]
 *
 * Compact format keeps header, id, generation, vector, and optional RaBitQ
 * data in the fixed node. Its authoritative neighbor list is stored in the
 * compact graph plane and addressed deterministically from RemotePtr.
 */
class VamanaNode {
public:
  static constexpr size_t HEADER_NODE_LOCK = 0b01;
  static constexpr size_t HEADER_MEDOID_LOCK = 0b100000000;
  static constexpr size_t HEADER_IS_MEDOID = 0b10000000000000000;
  static constexpr size_t HEADER_DELETED = 0b1000000000000000000000000;
  static constexpr u8 HOT_GRAPH_DELETED = 1u << 0;
  static constexpr size_t HEADER_SIZE = sizeof(u64);
  static constexpr size_t ID_SIZE = sizeof(u32);
  static constexpr size_t EDGE_COUNT_SIZE = sizeof(u8);
  static constexpr size_t PADDING_SIZE = 3;
  static constexpr size_t META_SIZE = ID_SIZE + EDGE_COUNT_SIZE + PADDING_SIZE;
  static constexpr size_t GENERATION_SIZE = sizeof(u32);
  static constexpr size_t COMPACT_META_SIZE = ID_SIZE + GENERATION_SIZE;

  static constexpr size_t HEADER_UNTIL_LOCK = 0;
  static constexpr size_t HEADER_UNTIL_MEDOID_LOCK = 1;
  static constexpr size_t HEADER_UNTIL_IS_MEDOID = 2;

  inline static u32 DIM{};
  inline static u32 R{};
  inline static u32 NEIGHBORS_SIZE{};
  inline static VectorDType VECTOR_DTYPE{VectorDType::float32};
  inline static u32 VECTOR_COMPONENT_SIZE{sizeof(element_t)};
  inline static u32 VECTOR_BYTES{0};
  inline static vamana::StorageFormat STORAGE_FORMAT{vamana::StorageFormat::aos_v1};

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
  static str storage_format_name() { return vamana::storage_format_name(STORAGE_FORMAT); }
  static bool supports_storage_format(const str& name) {
    return vamana::parse_storage_format(name).has_value();
  }
  static void set_storage_format(vamana::StorageFormat format) { STORAGE_FORMAT = format; }
  static bool compact_storage() { return STORAGE_FORMAT == vamana::StorageFormat::compact_v1; }

  static constexpr size_t STORAGE_ALIGNMENT = 64;
  static constexpr size_t COMPACT_ALIGNMENT = 16;
  static constexpr size_t NODE_PREFIX_SIZE = HEADER_SIZE + META_SIZE;

  static size_t align_storage(size_t value) {
    return (value + STORAGE_ALIGNMENT - 1) & ~(STORAGE_ALIGNMENT - 1);
  }

  static size_t align_compact(size_t value) {
    return (value + COMPACT_ALIGNMENT - 1) & ~(COMPACT_ALIGNMENT - 1);
  }

  static size_t align8(size_t value) {
    return (value + 7) & ~size_t{7};
  }

  static size_t offset_id() { return HEADER_SIZE; }
  static size_t offset_generation() { return HEADER_SIZE + ID_SIZE; }
  static size_t offset_edge_count() { return HEADER_SIZE + ID_SIZE; }
  static size_t offset_neighbors() { return NODE_PREFIX_SIZE; }
  static size_t graph_hot_bytes() {
    return compact_storage()
      ? HEADER_SIZE + COMPACT_META_SIZE
      : align_storage(offset_neighbors() + NEIGHBORS_SIZE);
  }
  static size_t offset_vector() { return graph_hot_bytes(); }
  static size_t vector_storage_bytes() {
    return compact_storage() ? align8(vector_bytes()) : align_storage(vector_bytes());
  }
  static size_t offset_rabitq_code() { return offset_vector() + vector_storage_bytes(); }
  static u32 rabitq_code_bits() {
      lib_assert(DIM > 0 && DIM <= (1u << 30), "RaBitQ dimension must be in [1, 2^30]");
      u32 value = 1;
      while (value < DIM) value <<= 1;
      return std::max<u32>(value, 8);
  }
  static size_t rabitq_code_size() { return rabitq_code_bits() / 8; }
  static size_t rabitq_code_storage_size() { return (rabitq_code_size() + 3) & ~size_t{3}; }
  static size_t rabitq_entry_size() { return (rabitq_code_storage_size() + 8 + 7) & ~size_t{7}; }
  static size_t rabitq_entry_storage_size() {
    return compact_storage() ? align8(rabitq_entry_size()) : align_storage(rabitq_entry_size());
  }
  static size_t offset_rabitq_norm() { return offset_rabitq_code() + rabitq_code_storage_size(); }
  static size_t offset_rabitq_error() { return offset_rabitq_norm() + sizeof(float); }

  static size_t vector_bytes() { return VECTOR_BYTES; }
  static size_t size_until_vector_end() { return offset_vector() + vector_bytes(); }
  static size_t neighbor_read_offset() { return offset_id(); }
  static size_t neighbor_read_size() { return 8 + NEIGHBORS_SIZE; }
  static constexpr size_t neighbor_count_offset_in_read() { return ID_SIZE; }
  static constexpr size_t neighbor_payload_offset_in_read() { return 8; }
  static size_t total_size() {
    const size_t end = HAS_RABITQ_CODE
          ? offset_rabitq_code() + rabitq_entry_storage_size()
          : offset_vector() + vector_storage_bytes();
    return compact_storage() ? align_compact(end) : align_storage(end);
  }

  // Compact graph plane appended after fixed nodes in the same RDMA region.
  // RemotePtr points to the fixed node; the graph entry is derived from its slot.
  inline static bool HAS_HOT_GRAPH = false;
  inline static u32 HOT_GRAPH_FORMAT_VERSION = 0;
  inline static u32 HOT_GRAPH_ENTRY_BYTES = 0;
  inline static u32 HOT_GRAPH_SHARD_BITS = 0;
  inline static vec<u64> HOT_GRAPH_ENTRY_OFFSETS;
  inline static vec<u64> HOT_GRAPH_ENTRY_COUNTS;
  inline static vec<u64> HOT_GRAPH_DYNAMIC_BASE_OFFSETS;
  inline static u32 HOT_GRAPH_DYNAMIC_RECORD_BYTES = 0;
  inline static u32 HOT_GRAPH_DYNAMIC_HOT_OFFSET = 0;

  static size_t hot_graph_entry_size() { return vamana::hot_graph::entry_bytes(R); }
  static size_t dynamic_record_size() {
    return compact_storage()
      ? align_compact(total_size() + hot_graph_entry_size())
      : align_storage(total_size() + hot_graph_entry_size());
  }
  static size_t allocation_size() {
    return HAS_HOT_GRAPH && HOT_GRAPH_FORMAT_VERSION >= 2
      ? HOT_GRAPH_DYNAMIC_RECORD_BYTES
      : total_size();
  }

  static void disable_hot_graph() {
    HAS_HOT_GRAPH = false;
    HOT_GRAPH_FORMAT_VERSION = 0;
    HOT_GRAPH_ENTRY_BYTES = 0;
    HOT_GRAPH_SHARD_BITS = 0;
    HOT_GRAPH_ENTRY_OFFSETS.clear();
    HOT_GRAPH_ENTRY_COUNTS.clear();
    HOT_GRAPH_DYNAMIC_BASE_OFFSETS.clear();
    HOT_GRAPH_DYNAMIC_RECORD_BYTES = 0;
    HOT_GRAPH_DYNAMIC_HOT_OFFSET = 0;
  }

  static void configure_hot_graph(const vec<u64>& entry_offsets,
                                  const vec<u64>& entry_counts,
                                  u32 entry_bytes,
                                  u32 shard_bits,
                                  u32 format_version = 1,
                                  const vec<u64>& dynamic_base_offsets = {},
                                  u32 dynamic_record_bytes = 0,
                                  u32 dynamic_hot_offset = 0) {
    if (entry_offsets.empty() || entry_offsets.size() != entry_counts.size()) {
      disable_hot_graph();
      return;
    }
    HOT_GRAPH_ENTRY_OFFSETS = entry_offsets;
    HOT_GRAPH_ENTRY_COUNTS = entry_counts;
    HOT_GRAPH_ENTRY_BYTES = entry_bytes;
    HOT_GRAPH_SHARD_BITS = shard_bits;
    HOT_GRAPH_FORMAT_VERSION = format_version;
    HOT_GRAPH_DYNAMIC_BASE_OFFSETS = dynamic_base_offsets;
    HOT_GRAPH_DYNAMIC_RECORD_BYTES = dynamic_record_bytes == 0
      ? static_cast<u32>(dynamic_record_size())
      : dynamic_record_bytes;
    HOT_GRAPH_DYNAMIC_HOT_OFFSET = dynamic_hot_offset == 0
      ? static_cast<u32>(total_size())
      : dynamic_hot_offset;
    HAS_HOT_GRAPH = entry_bytes >= hot_graph_entry_size();
    if (format_version >= 2) {
      HAS_HOT_GRAPH = HAS_HOT_GRAPH &&
        HOT_GRAPH_DYNAMIC_BASE_OFFSETS.size() == HOT_GRAPH_ENTRY_OFFSETS.size() &&
        HOT_GRAPH_DYNAMIC_RECORD_BYTES >= HOT_GRAPH_DYNAMIC_HOT_OFFSET + HOT_GRAPH_ENTRY_BYTES &&
        HOT_GRAPH_DYNAMIC_HOT_OFFSET >= total_size();
    }
    if (!HAS_HOT_GRAPH) disable_hot_graph();
  }

  static bool hot_graph_entry_available(RemotePtr ptr) {
    if (!HAS_HOT_GRAPH || ptr.memory_node() >= HOT_GRAPH_ENTRY_OFFSETS.size() ||
        ptr.byte_offset() < vamana::hot_graph::kNodeBaseOffset) {
      return false;
    }
    const u64 relative = ptr.byte_offset() - vamana::hot_graph::kNodeBaseOffset;
    const u64 node_size = total_size();
    if (node_size != 0 && relative % node_size == 0) {
      const u64 slot = relative / node_size;
      if (slot < HOT_GRAPH_ENTRY_COUNTS[ptr.memory_node()]) return true;
    }
    if (HOT_GRAPH_FORMAT_VERSION < 2 ||
        ptr.memory_node() >= HOT_GRAPH_DYNAMIC_BASE_OFFSETS.size() ||
        HOT_GRAPH_DYNAMIC_RECORD_BYTES == 0 ||
        ptr.byte_offset() < HOT_GRAPH_DYNAMIC_BASE_OFFSETS[ptr.memory_node()]) {
      return false;
    }
    const u64 dynamic_relative = ptr.byte_offset() - HOT_GRAPH_DYNAMIC_BASE_OFFSETS[ptr.memory_node()];
    return dynamic_relative % HOT_GRAPH_DYNAMIC_RECORD_BYTES == 0;
  }

  static u64 hot_graph_entry_offset(RemotePtr ptr) {
    const u64 relative = ptr.byte_offset() - vamana::hot_graph::kNodeBaseOffset;
    const u64 node_size = total_size();
    if (node_size != 0 && relative % node_size == 0) {
      const u64 slot = relative / node_size;
      if (slot < HOT_GRAPH_ENTRY_COUNTS[ptr.memory_node()]) {
        return HOT_GRAPH_ENTRY_OFFSETS[ptr.memory_node()] + slot * HOT_GRAPH_ENTRY_BYTES;
      }
    }
    return ptr.byte_offset() + HOT_GRAPH_DYNAMIC_HOT_OFFSET;
  }

  static void encode_hot_graph_entry(byte_t* out,
                                     u32 id,
                                     u8 edge_count,
                                     const RemotePtr* neighbors,
                                     size_t neighbor_count,
                                     u32 shard_bits = HOT_GRAPH_SHARD_BITS,
                                     u32 generation = 0,
                                     u32 format_version = HOT_GRAPH_FORMAT_VERSION,
                                     bool deleted = false) {
    std::memset(out, 0, hot_graph_entry_size());
    const bool v2 = format_version >= 2;
    if (v2) {
      out[0] = deleted ? 0 : static_cast<u8>(std::min<size_t>(edge_count, R));
      out[1] = deleted ? HOT_GRAPH_DELETED : 0;
      vamana::hot_graph::store_u32_le(out + 4, generation);
    } else {
      *reinterpret_cast<u32*>(out) = id;
      out[sizeof(u32)] = static_cast<u8>(std::min<size_t>(edge_count, R));
    }
    for (u32 i = 0; i < R; ++i) {
      byte_t* encoded = out + vamana::hot_graph::neighbor_offset(i);
      if (!deleted && i < neighbor_count) {
        (void)vamana::hot_graph::encode_remote_ptr(neighbors[i], shard_bits, encoded);
      } else {
        (void)vamana::hot_graph::encode_remote_ptr(RemotePtr{}, shard_bits, encoded);
      }
    }
    if (v2) {
      const u16 checksum = vamana::hot_graph::checksum16(out, hot_graph_entry_size());
      vamana::hot_graph::store_u16_le(out + 2, checksum);
    }
  }

  static bool decode_hot_graph_entry(const byte_t* compact, byte_t* neighbor_read_buffer) {
    std::memset(neighbor_read_buffer, 0, neighbor_read_size());
    u8 edge_count = 0;
    if (HOT_GRAPH_FORMAT_VERSION >= 2) {
      edge_count = compact[0];
      if (edge_count > R) return false;
      const u16 expected = vamana::hot_graph::load_u16_le(compact + 2);
      const u16 actual = vamana::hot_graph::checksum16(compact, hot_graph_entry_size());
      if (expected != actual) return false;
      if ((compact[1] & HOT_GRAPH_DELETED) != 0) {
        *reinterpret_cast<u8*>(
          neighbor_read_buffer + neighbor_count_offset_in_read()) = 0;
        return true;
      }
    } else {
      *reinterpret_cast<u32*>(neighbor_read_buffer) = *reinterpret_cast<const u32*>(compact);
      edge_count = compact[sizeof(u32)];
      if (edge_count > R) return false;
    }
    *reinterpret_cast<u8*>(neighbor_read_buffer + neighbor_count_offset_in_read()) = edge_count;
    auto* out = reinterpret_cast<RemotePtr*>(neighbor_read_buffer + neighbor_payload_offset_in_read());
    for (u32 i = 0; i < edge_count; ++i) {
      out[i] = vamana::hot_graph::decode_remote_ptr(
        compact + vamana::hot_graph::neighbor_offset(i), HOT_GRAPH_SHARD_BITS);
    }
    return true;
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
  static float compute_rabitq_norm(const byte_t* vector, VectorDType dtype) {
      vec<float> rotated(rabitq_code_bits());
      float norm2 = 0.0f;
      compute_rotated_query(vector, dtype, rotated.data(), &norm2);
      return std::sqrt(norm2);
  }

  // Compute binary code from the signs of the rotated centered vector.
  static RabitqCode compute_rabitq_code(const byte_t* vector, VectorDType dtype) {
      const u32 bits = rabitq_code_bits();
      vec<float> rotated(bits);
      float norm2 = 0.0f;
      compute_rotated_query(vector, dtype, rotated.data(), &norm2);
      RabitqCode code(rabitq_code_size(), 0);
      for (u32 bit = 0; bit < bits; ++bit) {
          if (rotated[bit] > 0.0f) {
              code[bit >> 3] |= static_cast<byte_t>(1u << (7u - (bit & 7u)));
          }
      }
      return code;
  }

  static void compute_rabitq_entry(const byte_t* vector, VectorDType dtype,
                                   RabitqCode& code, float& norm, float& error) {
      const u32 bits = rabitq_code_bits();
      vec<float> rotated(bits);
      float norm2 = 0.0f;
      compute_rotated_query(vector, dtype, rotated.data(), &norm2);

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
  static float compute_rabitq_error_factor(const byte_t* vector, VectorDType dtype, const RabitqCode& code) {
      const u32 bits = rabitq_code_bits();
      vec<float> rotated(bits);
      float norm2 = 0.0f;
      compute_rotated_query(vector, dtype, rotated.data(), &norm2);
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
  u8 edge_count() const {
    return compact_storage() ? 0 : *reinterpret_cast<u8*>(buffer_slice_ + offset_edge_count());
  }
  u32 generation() const {
    return compact_storage() ? *reinterpret_cast<u32*>(buffer_slice_ + offset_generation()) : 0;
  }
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
    if (compact_storage()) return {};
    return {reinterpret_cast<RemotePtr*>(buffer_slice_ + offset_neighbors()), static_cast<size_t>(edge_count())};
  }

  span<RemotePtr> all_neighbor_slots() const {
    if (compact_storage()) return {};
    return {reinterpret_cast<RemotePtr*>(buffer_slice_ + offset_neighbors()), static_cast<size_t>(R)};
  }

  void set_edge_count(u8 count) {
    if (compact_storage()) return;
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
