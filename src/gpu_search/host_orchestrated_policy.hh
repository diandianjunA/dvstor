#pragma once

#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>

#include "gpu_search/index_format.hh"
#include "remote_pointer.hh"
#include "vamana/dynamic_navigation_code.hh"
#include "vamana/vamana_node.hh"

namespace gpu_search::host_orchestrated_policy {

struct ResolvedRecord {
  u32 shard{};
  u64 node_offset{};
  u64 graph_offset{};
  u64 dynamic_code_offset{};
  u32 static_ordinal{std::numeric_limits<u32>::max()};
  bool immutable_base{};
};

// Validate the complete tagged physical handle before using any derived RDMA
// address. Static and dynamic records retain the exact schema-v16 layout used
// by the persistent engine; no baseline-only pointer or index representation
// is admitted here.
inline bool resolve_record(const format::View& view,
                           RemotePtr pointer,
                           u64 storage_region_bytes,
                           ResolvedRecord& result) {
  result = {};
  result.static_ordinal = std::numeric_limits<u32>::max();
  if (pointer.is_null() || !pointer.is_well_formed() ||
      pointer.memory_node() >= view.shards.size()) {
    return false;
  }
  const format::ShardRegion& shard = view.shards[pointer.memory_node()];
  u32 ordinal = 0;
  if (format::remote_to_ordinal(view, pointer, ordinal)) {
    const u64 slot = static_cast<u64>(ordinal) - shard.ordinal_base;
    if (slot >= shard.node_count) return false;
    result = {
      .shard = pointer.memory_node(),
      .node_offset = pointer.byte_offset(),
      .graph_offset = shard.graph_base_offset +
        slot * view.layout.graph_entry_bytes,
      .dynamic_code_offset = 0,
      .static_ordinal = ordinal,
      .immutable_base = true,
    };
    return result.graph_offset <= storage_region_bytes &&
      view.layout.graph_entry_bytes <=
        storage_region_bytes - result.graph_offset &&
      result.node_offset <= storage_region_bytes &&
      VamanaNode::size_until_vector_end() <=
        storage_region_bytes - result.node_offset;
  }

  if (!pointer.is_dynamic() || shard.dynamic_record_bytes == 0 ||
      pointer.byte_offset() < shard.dynamic_base_offset) {
    return false;
  }
  const u64 relative = pointer.byte_offset() - shard.dynamic_base_offset;
  if (relative % shard.dynamic_record_bytes != 0 ||
      pointer.byte_offset() > storage_region_bytes ||
      shard.dynamic_record_bytes >
        storage_region_bytes - pointer.byte_offset() ||
      shard.dynamic_hot_offset > shard.dynamic_record_bytes ||
      view.layout.graph_entry_bytes >
        shard.dynamic_record_bytes - shard.dynamic_hot_offset ||
      shard.dynamic_code_offset > shard.dynamic_record_bytes) {
    return false;
  }
  const u64 dynamic_code_bytes =
    VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
    view.layout.code_bytes + VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES;
  if (dynamic_code_bytes >
      shard.dynamic_record_bytes - shard.dynamic_code_offset) {
    return false;
  }
  result = {
    .shard = pointer.memory_node(),
    .node_offset = pointer.byte_offset(),
    .graph_offset = pointer.byte_offset() + shard.dynamic_hot_offset,
    .dynamic_code_offset = pointer.byte_offset() +
      shard.dynamic_code_offset,
    .static_ordinal = std::numeric_limits<u32>::max(),
    .immutable_base = false,
  };
  return true;
}

inline bool exact_snapshot_visible(const byte_t* record,
                                   size_t record_bytes,
                                   u64 header_after,
                                   RemotePtr pointer) {
  if (record == nullptr ||
      record_bytes < VamanaNode::size_until_vector_end()) {
    return false;
  }
  u64 header_before = 0;
  u32 stored_incarnation = 0;
  std::memcpy(&header_before, record, sizeof(header_before));
  std::memcpy(&stored_incarnation,
              record + VamanaNode::offset_slot_incarnation(),
              sizeof(stored_incarnation));
  return header_before == header_after &&
    (header_before & (VamanaNode::HEADER_NODE_LOCK |
                      VamanaNode::HEADER_DELETED)) == 0 &&
    VamanaNode::header_incarnation(header_before) ==
      pointer.incarnation() &&
    stored_incarnation == pointer.incarnation();
}

inline bool dynamic_code_snapshot_visible(const byte_t* record,
                                          u32 code_bytes,
                                          RemotePtr pointer,
                                          u8* extent_class = nullptr) {
  if (record == nullptr || !pointer.is_dynamic() || code_bytes == 0) {
    return false;
  }
  u32 tag = 0;
  std::memcpy(&tag, record, sizeof(tag));
  if ((tag & 0x80000000u) != 0 ||
      VamanaNode::dynamic_navigation_tag_incarnation(tag) !=
        pointer.incarnation() ||
      !vamana::dynamic_navigation_code::validate(
        tag, record + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES,
        code_bytes,
        record + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES + code_bytes)) {
    return false;
  }
  if (extent_class != nullptr) {
    *extent_class = VamanaNode::dynamic_navigation_tag_extent_class(tag);
  }
  return true;
}

inline bool distance_handle_less(f32 lhs_distance, RemotePtr lhs,
                                 f32 rhs_distance, RemotePtr rhs) {
  const bool lhs_valid = floating_value_is_finite(lhs_distance) &&
    lhs_distance < FLT_MAX;
  const bool rhs_valid = floating_value_is_finite(rhs_distance) &&
    rhs_distance < FLT_MAX;
  if (lhs_valid != rhs_valid) return lhs_valid;
  if (lhs_distance != rhs_distance) return lhs_distance < rhs_distance;
  return lhs.raw_address < rhs.raw_address;
}

// A lane owns QPs, CQ state, registered RDMA scratch, and CUDA scratch as one
// reuse unit.  Once either transport or CUDA execution becomes uncertain the
// lane must never re-enter the free pool: a late CQE/RDMA write could otherwise
// be mistaken for work issued by the next query.
inline constexpr bool lane_reusable(bool poisoned, bool stopping,
                                    bool healthy) {
  return !poisoned && !stopping && healthy;
}

// Unresolved/out-of-range handles never had an RDMA snapshot, and a recycled
// dynamic handle is an expected read-committed miss. Only an explicitly
// validated graph record may reach the decoder.
inline constexpr bool graph_snapshot_decodable(bool validation_succeeded,
                                                bool stale) {
  return validation_succeeded && !stale;
}

}  // namespace gpu_search::host_orchestrated_policy
