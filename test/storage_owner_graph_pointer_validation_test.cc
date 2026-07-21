#include <cassert>

#include "memory_node/storage_owner_state.hh"
#include "memory_node/storage_owner_index/graph_pointer_validation.hh"
#include "vamana/hot_graph.hh"

int main() {
  constexpr u32 kShardCount = 2;
  constexpr u64 kShardBytes = 1ull << 20;
  constexpr u64 kDynamicBase = 0x10000;

  VamanaNode::disable_hot_graph();
  VamanaNode::init_static_storage(128, 96, VectorDType::uint8);
  const u32 graph_bytes =
    static_cast<u32>(VamanaNode::hot_graph_entry_size());
  const u32 dynamic_hot_offset =
    static_cast<u32>(VamanaNode::total_size());
  const u32 dynamic_code_offset = dynamic_hot_offset + graph_bytes;
  const u32 dynamic_record_bytes = static_cast<u32>(
    VamanaNode::align_compact(
      dynamic_code_offset + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
      32));
  VamanaNode::configure_hot_graph(
    {0x2000, 0x4000}, {1, 1}, graph_bytes,
    vamana::hot_graph::shard_bits_for(kShardCount),
    {kDynamicBase, kDynamicBase}, dynamic_record_bytes,
    dynamic_hot_offset, dynamic_code_offset, 32);
  assert(VamanaNode::HAS_HOT_GRAPH);

  // Regression for the Stage2 corruption: a graph RDMA read is much larger
  // than a uint8 D128 vector snapshot. Consecutive graph slots must be spaced
  // by the graph size, never by the snapshot stride.
  const size_t snapshot_stride =
    memory_node_detail::storage_owner_snapshot_stride();
  const size_t validation_offset =
    memory_node_detail::storage_owner_snapshot_validation_offset();
  assert(validation_offset >= VamanaNode::size_until_vector_end());
  assert(validation_offset % alignof(u64) == 0);
  assert(validation_offset + VamanaNode::HEADER_SIZE <= snapshot_stride);
  const size_t graph_stride =
    memory_node_storage_owner_index_detail::graph_read_slot_stride();
  const size_t batch_slot_stride =
    memory_node_storage_owner_index_detail::batched_read_slot_stride(
      snapshot_stride);
  assert(VamanaNode::hot_graph_entry_size() > snapshot_stride);
  assert(graph_stride >= VamanaNode::hot_graph_entry_size());
  assert(batch_slot_stride == graph_stride);
  assert(graph_stride + VamanaNode::hot_graph_entry_size() <=
         2 * graph_stride);

  const RemotePtr valid{1, kDynamicBase, 7};
  assert(valid.is_well_formed());
  assert(VamanaNode::hot_graph_entry_available(valid));
  assert(memory_node_storage_owner_index_detail::
           storage_pointer_addressable(valid, kShardCount, kShardBytes));

  // Receipt release is an identity/control operation, not a graph access.
  // An arbitrary old incarnation at an addressable slot remains a valid
  // release target after Stage2 has retired or reused that slot.  The helper
  // must still reject a wrong physical home, an out-of-range address, and a
  // null target unless the Stage1-abort protocol explicitly permits it.
  assert(memory_node_storage_owner_index_detail::
           receipt_release_pointer_addressable(
             valid, 1, kShardCount, kShardBytes, false));
  assert(!memory_node_storage_owner_index_detail::
           receipt_release_pointer_addressable(
             valid, 0, kShardCount, kShardBytes, false));
  assert(memory_node_storage_owner_index_detail::
           receipt_release_pointer_addressable(
             RemotePtr{}, 1, kShardCount, kShardBytes, true));
  assert(!memory_node_storage_owner_index_detail::
           receipt_release_pointer_addressable(
             RemotePtr{}, 1, kShardCount, kShardBytes, false));

  // Model the production race directly: Stage2/cleanup sends a delayed
  // control request for incarnation 7 after the allocator has already reused
  // the same physical slot as incarnation 8.  Both handles are structurally
  // addressable; the control handler must inspect the slot identity and ACK
  // the old one as stale instead of rejecting the RPC before that check.
  const RemotePtr recycled{1, kDynamicBase, 8};
  assert(recycled.byte_offset() == valid.byte_offset());
  assert(recycled.incarnation() != valid.incarnation());
  assert(memory_node_storage_owner_index_detail::
           local_storage_pointer_addressable(
             valid, 1, kShardCount, kShardBytes));
  assert(memory_node_storage_owner_index_detail::
           local_storage_pointer_addressable(
             recycled, 1, kShardCount, kShardBytes));
  assert(!memory_node_storage_owner_index_detail::
           local_storage_pointer_addressable(
             valid, 0, kShardCount, kShardBytes));

  const u64 records_to_cap =
    (kShardBytes - kDynamicBase + dynamic_record_bytes - 1) /
    dynamic_record_bytes;
  const u64 bad_offset =
    kDynamicBase + records_to_cap * dynamic_record_bytes;
  const RemotePtr out_of_bounds{1, bad_offset, 7};
  // This is the exact class of handle that caused the crash: its tag, shard,
  // alignment and dynamic-record stride are all valid, but its byte range is
  // outside the registered shard MR.
  assert(out_of_bounds.is_well_formed());
  assert(out_of_bounds.memory_node() < kShardCount);
  assert(VamanaNode::hot_graph_entry_available(out_of_bounds));
  assert(!memory_node_storage_owner_index_detail::
           storage_pointer_addressable(
             out_of_bounds, kShardCount, kShardBytes));
  assert(!memory_node_storage_owner_index_detail::
           receipt_release_pointer_addressable(
             out_of_bounds, 1, kShardCount, kShardBytes, false));
  assert(!memory_node_storage_owner_index_detail::
           local_storage_pointer_addressable(
             out_of_bounds, 1, kShardCount, kShardBytes));

  // Validation scratch must stay naturally aligned for arbitrary integral
  // dimensions, not only the common D128 shape.  The paired RDMA path reads
  // this location as a u64 after-header.
  for (const auto& [dim, dtype] : {
         std::pair<u32, VectorDType>{127, VectorDType::uint8},
         std::pair<u32, VectorDType>{129, VectorDType::int8}}) {
    VamanaNode::init_static_storage(dim, 96, dtype);
    const size_t odd_validation_offset =
      memory_node_detail::storage_owner_snapshot_validation_offset();
    const size_t odd_snapshot_stride =
      memory_node_detail::storage_owner_snapshot_stride();
    assert(odd_validation_offset >= VamanaNode::size_until_vector_end());
    assert(odd_validation_offset % alignof(u64) == 0);
    assert(odd_validation_offset + VamanaNode::HEADER_SIZE <=
           odd_snapshot_stride);
  }

  VamanaNode::disable_hot_graph();
  return 0;
}
