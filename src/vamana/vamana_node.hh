#pragma once

#include <limits>

#include <library/utils.hh>

#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "remote_pointer.hh"
#include "vamana/hot_graph.hh"

/**
 * Fixed records contain header, id, generation, and exact vector. The compact
 * graph plane stores authoritative neighbors and is addressed deterministically
 * from the record's RemotePtr.
 */
class VamanaNode {
public:
  static constexpr size_t HEADER_NODE_LOCK = 0b01;
  static constexpr size_t HEADER_IS_MEDOID = 0b10000000000000000;
  static constexpr size_t HEADER_DELETED = 0b1000000000000000000000000;
  static constexpr u8 HOT_GRAPH_DELETED = 1u << 0;
  static constexpr size_t HEADER_SIZE = sizeof(u64);
  static constexpr size_t ID_SIZE = sizeof(u32);
  static constexpr size_t GENERATION_SIZE = sizeof(u32);
  static constexpr size_t COMPACT_META_SIZE = ID_SIZE + GENERATION_SIZE;

  static constexpr size_t HEADER_UNTIL_LOCK = 0;

  inline static u32 DIM{};
  inline static u32 R{};
  inline static VectorDType VECTOR_DTYPE{VectorDType::float32};
  inline static u32 VECTOR_COMPONENT_SIZE{sizeof(element_t)};
  inline static u32 VECTOR_BYTES{0};

  static void init_static_storage(u32 dim,
                                  u32 max_degree,
                                  VectorDType vector_dtype = VectorDType::float32) {
    lib_assert(dim > 0, "Vamana dimension must be > 0");
    lib_assert(max_degree > 0, "Vamana max degree R must be > 0");
    lib_assert(max_degree <= std::numeric_limits<u8>::max(),
               "Vamana max degree R must be <= 255 because edge_count is stored in one byte");
    const size_t bytes = vector_dtype_bytes(vector_dtype, dim);
    lib_assert(bytes <= std::numeric_limits<u32>::max(),
               "Vamana vector byte width exceeds the runtime layout limit");
    DIM = dim;
    R = max_degree;
    VECTOR_DTYPE = vector_dtype;
    VECTOR_COMPONENT_SIZE = static_cast<u32>(vector_dtype_component_size(vector_dtype));
    VECTOR_BYTES = static_cast<u32>(bytes);
  }

  static VectorDType vector_dtype() { return VECTOR_DTYPE; }
  static str vector_dtype_name() { return ::vector_dtype_name(VECTOR_DTYPE); }
  static size_t vector_component_size() { return VECTOR_COMPONENT_SIZE; }
  static str layout_name() { return "plain"; }
  static str storage_format_name() { return "vamana_compact_v1"; }
  static constexpr size_t STORAGE_ALIGNMENT = 64;
  static constexpr size_t COMPACT_ALIGNMENT = 16;

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
  static size_t graph_hot_bytes() { return HEADER_SIZE + COMPACT_META_SIZE; }
  static size_t offset_vector() { return graph_hot_bytes(); }
  static size_t vector_storage_bytes() { return align8(vector_bytes()); }
  static size_t vector_bytes() { return VECTOR_BYTES; }
  static size_t size_until_vector_end() { return offset_vector() + vector_bytes(); }
  static size_t neighbor_read_size() { return 8 + static_cast<size_t>(R) * sizeof(RemotePtr); }
  static constexpr size_t neighbor_count_offset_in_read() { return ID_SIZE; }
  static constexpr size_t neighbor_payload_offset_in_read() { return 8; }
  static size_t total_size() {
    const size_t end = offset_vector() + vector_storage_bytes();
    return align_compact(end);
  }

  // Compact graph plane appended after fixed nodes in the same RDMA region.
  // RemotePtr points to the fixed node; the graph entry is derived from its slot.
  inline static bool HAS_HOT_GRAPH = false;
  inline static u32 HOT_GRAPH_ENTRY_BYTES = 0;
  inline static u32 HOT_GRAPH_SHARD_BITS = 0;
  inline static vec<u64> HOT_GRAPH_ENTRY_OFFSETS;
  inline static vec<u64> HOT_GRAPH_ENTRY_COUNTS;
  inline static vec<u64> HOT_GRAPH_DYNAMIC_BASE_OFFSETS;
  inline static u32 HOT_GRAPH_DYNAMIC_RECORD_BYTES = 0;
  inline static u32 HOT_GRAPH_DYNAMIC_HOT_OFFSET = 0;

  static size_t hot_graph_entry_size() { return vamana::hot_graph::entry_bytes(R); }
  static size_t dynamic_record_size() {
    return align_compact(total_size() + hot_graph_entry_size());
  }
  static size_t allocation_size() {
    return HAS_HOT_GRAPH ? HOT_GRAPH_DYNAMIC_RECORD_BYTES : total_size();
  }

  static void disable_hot_graph() {
    HAS_HOT_GRAPH = false;
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
    HOT_GRAPH_DYNAMIC_BASE_OFFSETS = dynamic_base_offsets;
    HOT_GRAPH_DYNAMIC_RECORD_BYTES = dynamic_record_bytes == 0
      ? static_cast<u32>(dynamic_record_size())
      : dynamic_record_bytes;
    HOT_GRAPH_DYNAMIC_HOT_OFFSET = dynamic_hot_offset == 0
      ? static_cast<u32>(total_size())
      : dynamic_hot_offset;
    HAS_HOT_GRAPH = entry_bytes >= hot_graph_entry_size() &&
      HOT_GRAPH_DYNAMIC_BASE_OFFSETS.size() == HOT_GRAPH_ENTRY_OFFSETS.size() &&
      HOT_GRAPH_DYNAMIC_RECORD_BYTES >= HOT_GRAPH_DYNAMIC_HOT_OFFSET + HOT_GRAPH_ENTRY_BYTES &&
      HOT_GRAPH_DYNAMIC_HOT_OFFSET >= total_size();
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
    if (ptr.memory_node() >= HOT_GRAPH_DYNAMIC_BASE_OFFSETS.size() ||
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
                                     u8 edge_count,
                                     const RemotePtr* neighbors,
                                     size_t neighbor_count,
                                     u32 shard_bits = HOT_GRAPH_SHARD_BITS,
                                     u32 generation = 0,
                                     bool deleted = false) {
    std::memset(out, 0, hot_graph_entry_size());
    out[0] = deleted ? 0 : static_cast<u8>(std::min<size_t>(edge_count, R));
    out[1] = deleted ? HOT_GRAPH_DELETED : 0;
    vamana::hot_graph::store_u32_le(out + 4, generation);
    for (u32 i = 0; i < R; ++i) {
      byte_t* encoded = out + vamana::hot_graph::neighbor_offset(i);
      if (!deleted && i < neighbor_count) {
        (void)vamana::hot_graph::encode_remote_ptr(neighbors[i], shard_bits, encoded);
      } else {
        (void)vamana::hot_graph::encode_remote_ptr(RemotePtr{}, shard_bits, encoded);
      }
    }
    const u16 checksum = vamana::hot_graph::checksum16(out, hot_graph_entry_size());
    vamana::hot_graph::store_u16_le(out + 2, checksum);
  }

  static bool decode_hot_graph_entry(const byte_t* compact, byte_t* neighbor_read_buffer) {
    std::memset(neighbor_read_buffer, 0, neighbor_read_size());
    const u8 edge_count = compact[0];
    if (edge_count > R) return false;
    const u16 expected = vamana::hot_graph::load_u16_le(compact + 2);
    const u16 actual = vamana::hot_graph::checksum16(compact, hot_graph_entry_size());
    if (expected != actual) return false;
    if ((compact[1] & HOT_GRAPH_DELETED) != 0) {
      *reinterpret_cast<u8*>(
        neighbor_read_buffer + neighbor_count_offset_in_read()) = 0;
      return true;
    }
    *reinterpret_cast<u8*>(neighbor_read_buffer + neighbor_count_offset_in_read()) = edge_count;
    auto* out = reinterpret_cast<RemotePtr*>(neighbor_read_buffer + neighbor_payload_offset_in_read());
    for (u32 i = 0; i < edge_count; ++i) {
      out[i] = vamana::hot_graph::decode_remote_ptr(
        compact + vamana::hot_graph::neighbor_offset(i), HOT_GRAPH_SHARD_BITS);
    }
    return true;
  }

};
