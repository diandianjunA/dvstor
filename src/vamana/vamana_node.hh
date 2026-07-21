#pragma once

#include <limits>

#include <library/utils.hh>

#include "common/constants.hh"
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
  static constexpr size_t HEADER_DELETED = 0b1000000000000000000000000;
  // A provisional node is visible to ordinary queries but is ineligible as a
  // durable construction neighbor until Stage2 commits its final placement.
  static constexpr size_t HEADER_PROVISIONAL = 0b10000000000000000000000000;
  // Exact physical-centroid membership is recorded on the node itself. This
  // makes insert/erase centroid RPC retries idempotent without an unbounded
  // operation-ID table and without coupling membership to logical ID owner.
  static constexpr size_t HEADER_CENTROID_ACCOUNTED =
    0b100000000000000000000000000;
  // Cleanup sets RETIRING before its final incoming-edge snapshot. Queries
  // still treat the node as live; graph mutations reject it so no new
  // protected child can appear behind the snapshot.
  static constexpr size_t HEADER_RETIRING = u64{1} << 27;
  // Stage2 freezes only the graph mutation plane while it rebases and
  // publishes the final adjacency.  Queries continue to traverse the source
  // record.  Unlike RETIRING, an in-place finalization clears this bit after
  // the final graph entry is published; a migrated source keeps it until the
  // source tombstone is visible.
  static constexpr size_t HEADER_STAGE2_FROZEN = u64{1} << 28;
  static constexpr u8 HOT_GRAPH_DELETED = vamana::hot_graph::kDeletedFlag;
  static constexpr size_t HEADER_SIZE = sizeof(u64);
  static constexpr size_t ID_SIZE = sizeof(u32);
  static constexpr size_t GENERATION_SIZE = sizeof(u32);
  static constexpr size_t SLOT_INCARNATION_SIZE = sizeof(u32);
  static constexpr size_t COMPACT_META_SIZE =
    ID_SIZE + GENERATION_SIZE + SLOT_INCARNATION_SIZE;
  static constexpr u32 HEADER_INCARNATION_SHIFT = 32;
  static constexpr u64 HEADER_FLAG_MASK = (u64{1} << HEADER_INCARNATION_SHIFT) - 1;

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
    lib_assert(max_degree <= kMaxSupportedGraphDegree,
               "Vamana max degree R exceeds the system-wide CPU/GPU limit");
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
  static str storage_format_name() { return "vamana_tagged_v2"; }
  static constexpr size_t STORAGE_ALIGNMENT = 64;
  static constexpr size_t COMPACT_ALIGNMENT = 16;
  static constexpr u32 DYNAMIC_CODE_INCARNATION_BYTES = sizeof(u32);

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
  static size_t offset_slot_incarnation() {
    return HEADER_SIZE + ID_SIZE + GENERATION_SIZE;
  }
  static size_t graph_hot_bytes() {
    return align8(HEADER_SIZE + COMPACT_META_SIZE);
  }
  static size_t offset_vector() { return graph_hot_bytes(); }
  static size_t vector_storage_bytes() { return align8(vector_bytes()); }
  static size_t vector_bytes() { return VECTOR_BYTES; }
  static size_t size_until_vector_end() { return offset_vector() + vector_bytes(); }
  static u32 provisional_slots() {
    // Reserve 1/16 of the stable degree, bounded by the four-bit on-wire
    // counter. This scales with graph degree without exposing a benchmark
    // tuning knob or allowing transient edges to grow without bound.
    return std::min<u32>(15, std::max<u32>(2, (R + 15) / 16));
  }
  static u32 graph_entry_capacity() { return R + provisional_slots(); }
  static size_t neighbor_read_size() {
    return 8 + static_cast<size_t>(graph_entry_capacity()) *
      sizeof(RemotePtr);
  }
  static constexpr size_t stable_neighbor_count_offset_in_read() { return ID_SIZE; }
  static constexpr size_t provisional_neighbor_count_offset_in_read() { return ID_SIZE + 1; }
  static constexpr size_t neighbor_count_offset_in_read() {
    return stable_neighbor_count_offset_in_read();
  }
  static constexpr size_t neighbor_payload_offset_in_read() { return 8; }
  static u32 decoded_neighbor_count(const byte_t* neighbor_read_buffer) {
    return static_cast<u32>(neighbor_read_buffer[
      stable_neighbor_count_offset_in_read()]) +
      static_cast<u32>(neighbor_read_buffer[
        provisional_neighbor_count_offset_in_read()]);
  }
  static size_t total_size() {
    const size_t end = offset_vector() + vector_storage_bytes();
    return align_compact(end);
  }

  static constexpr u64 make_header(u32 incarnation, u64 flags = 0) {
    return (static_cast<u64>(incarnation) << HEADER_INCARNATION_SHIFT) |
      (flags & HEADER_FLAG_MASK);
  }

  static constexpr u32 header_incarnation(u64 header) {
    return static_cast<u32>(header >> HEADER_INCARNATION_SHIFT);
  }

  static constexpr bool graph_mutation_quiesced(u64 header) {
    return (header & (HEADER_RETIRING | HEADER_STAGE2_FROZEN)) != 0;
  }

  // Admission predicate for an ordinary stable-adjacency mutation while the
  // caller owns NODE_LOCK.  NODE_LOCK is deliberately ignored: the lock is
  // the serialization boundary, whereas DELETED/PROVISIONAL and the two
  // quiescence states are lifecycle fences that make a mutation retryable.
  static constexpr bool stable_graph_mutation_allowed(u64 header) {
    return (header & (HEADER_DELETED | HEADER_PROVISIONAL)) == 0 &&
      !graph_mutation_quiesced(header);
  }

  static constexpr u64 complete_in_place_stage2_header(u64 header) {
    return header &
      ~(static_cast<u64>(HEADER_PROVISIONAL) |
        static_cast<u64>(HEADER_STAGE2_FROZEN));
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
  inline static u32 HOT_GRAPH_DYNAMIC_CODE_OFFSET = 0;
  inline static u32 HOT_GRAPH_DYNAMIC_CODE_BYTES = 0;

  static size_t hot_graph_entry_size() {
    return vamana::hot_graph::entry_bytes(R, provisional_slots());
  }
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
    HOT_GRAPH_DYNAMIC_CODE_OFFSET = 0;
    HOT_GRAPH_DYNAMIC_CODE_BYTES = 0;
  }

  static void configure_hot_graph(const vec<u64>& entry_offsets,
                                  const vec<u64>& entry_counts,
                                  u32 entry_bytes,
                                  u32 shard_bits,
                                  const vec<u64>& dynamic_base_offsets = {},
                                  u32 dynamic_record_bytes = 0,
                                  u32 dynamic_hot_offset = 0,
                                  u32 dynamic_code_offset = 0,
                                  u32 dynamic_code_bytes = 0) {
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
    HOT_GRAPH_DYNAMIC_CODE_OFFSET = dynamic_code_offset;
    HOT_GRAPH_DYNAMIC_CODE_BYTES = dynamic_code_bytes == 0
      ? 0 : dynamic_code_bytes + DYNAMIC_CODE_INCARNATION_BYTES;
    HAS_HOT_GRAPH = entry_bytes >= hot_graph_entry_size() &&
      HOT_GRAPH_DYNAMIC_BASE_OFFSETS.size() == HOT_GRAPH_ENTRY_OFFSETS.size() &&
      HOT_GRAPH_DYNAMIC_RECORD_BYTES >= HOT_GRAPH_DYNAMIC_HOT_OFFSET + HOT_GRAPH_ENTRY_BYTES &&
      HOT_GRAPH_DYNAMIC_HOT_OFFSET >= total_size() &&
      (HOT_GRAPH_DYNAMIC_CODE_BYTES == 0 ||
       (HOT_GRAPH_DYNAMIC_CODE_OFFSET >= HOT_GRAPH_DYNAMIC_HOT_OFFSET + HOT_GRAPH_ENTRY_BYTES &&
        HOT_GRAPH_DYNAMIC_RECORD_BYTES >=
          HOT_GRAPH_DYNAMIC_CODE_OFFSET + HOT_GRAPH_DYNAMIC_CODE_BYTES));
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

  static u64 dynamic_navigation_code_offset(RemotePtr ptr) {
    lib_assert(HAS_HOT_GRAPH && HOT_GRAPH_DYNAMIC_CODE_BYTES != 0 &&
                 hot_graph_entry_available(ptr),
               "dynamic navigation code requested for an invalid node");
    return ptr.byte_offset() + HOT_GRAPH_DYNAMIC_CODE_OFFSET;
  }

  static u64 dynamic_navigation_code_data_offset(RemotePtr ptr) {
    return dynamic_navigation_code_offset(ptr) +
      DYNAMIC_CODE_INCARNATION_BYTES;
  }

  static void encode_hot_graph_entry(byte_t* out,
                                     u8 edge_count,
                                     const RemotePtr* neighbors,
                                     size_t neighbor_count,
                                     u32 shard_bits = HOT_GRAPH_SHARD_BITS,
                                     u32 generation = 0,
                                     bool deleted = false,
                                     const RemotePtr* provisional_neighbors = nullptr,
                                     size_t provisional_count = 0,
                                     u32 slot_incarnation = 0) {
    std::memset(out, 0, hot_graph_entry_size());
    out[0] = deleted ? 0 : static_cast<u8>(std::min<size_t>(edge_count, R));
    out[1] = deleted ? HOT_GRAPH_DELETED : 0;
    const u8 encoded_provisional_count = deleted ? 0 : static_cast<u8>(
      std::min<size_t>(provisional_count, provisional_slots()));
    vamana::hot_graph::store_provisional_count(
      out, encoded_provisional_count);
    vamana::hot_graph::store_u32_le(out + 4, generation);
    vamana::hot_graph::store_u32_le(out + 8, slot_incarnation);
    vamana::hot_graph::store_u32_le(out + 12, 0);
    const u32 stable_count = out[0];
    for (u32 i = 0; i < graph_entry_capacity(); ++i) {
      byte_t* encoded = out + vamana::hot_graph::neighbor_offset(i);
      if (!deleted && i < stable_count && i < neighbor_count) {
        (void)vamana::hot_graph::encode_remote_ptr(neighbors[i], shard_bits, encoded);
      } else if (!deleted && i >= stable_count &&
                 i - stable_count < encoded_provisional_count &&
                 provisional_neighbors != nullptr) {
        (void)vamana::hot_graph::encode_remote_ptr(
          provisional_neighbors[i - stable_count], shard_bits, encoded);
      } else {
        (void)vamana::hot_graph::encode_remote_ptr(RemotePtr{}, shard_bits, encoded);
      }
    }
    const u16 checksum = vamana::hot_graph::checksum16(out, hot_graph_entry_size());
    vamana::hot_graph::store_u16_le(out + 2, checksum);
  }

  static bool decode_hot_graph_entry(const byte_t* compact,
                                     byte_t* neighbor_read_buffer,
                                     u32 expected_incarnation) {
    std::memset(neighbor_read_buffer, 0, neighbor_read_size());
    const u8 stable_count = compact[0];
    const u8 provisional_count =
      vamana::hot_graph::provisional_count(compact);
    if (stable_count > R || provisional_count > provisional_slots() ||
        static_cast<u32>(stable_count) + provisional_count >
          graph_entry_capacity() ||
        vamana::hot_graph::load_u32_le(compact + 8) !=
          expected_incarnation ||
        vamana::hot_graph::load_u32_le(compact + 12) != 0) {
      return false;
    }
    const u16 expected = vamana::hot_graph::load_u16_le(compact + 2);
    const u16 actual = vamana::hot_graph::checksum16(compact, hot_graph_entry_size());
    if (expected != actual) return false;
    if ((compact[1] & HOT_GRAPH_DELETED) != 0) {
      *reinterpret_cast<u8*>(
        neighbor_read_buffer + stable_neighbor_count_offset_in_read()) = 0;
      *reinterpret_cast<u8*>(
        neighbor_read_buffer + provisional_neighbor_count_offset_in_read()) = 0;
      return true;
    }
    *reinterpret_cast<u8*>(
      neighbor_read_buffer + stable_neighbor_count_offset_in_read()) =
        stable_count;
    *reinterpret_cast<u8*>(
      neighbor_read_buffer + provisional_neighbor_count_offset_in_read()) =
        provisional_count;
    auto* out = reinterpret_cast<RemotePtr*>(neighbor_read_buffer + neighbor_payload_offset_in_read());
    for (u32 i = 0;
         i < static_cast<u32>(stable_count) + provisional_count; ++i) {
      out[i] = vamana::hot_graph::decode_remote_ptr(
        compact + vamana::hot_graph::neighbor_offset(i), HOT_GRAPH_SHARD_BITS);
      if (!out[i].is_well_formed()) return false;
    }
    return true;
  }

};
