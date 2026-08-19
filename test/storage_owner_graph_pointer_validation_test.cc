#include <algorithm>
#include <cassert>
#include <vector>

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
      32 + VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES));
  VamanaNode::configure_hot_graph(
    {0x2000, 0x4000}, {1, 1}, graph_bytes,
    vamana::hot_graph::shard_bits_for(kShardCount),
    {kDynamicBase, kDynamicBase}, dynamic_record_bytes,
    dynamic_hot_offset, dynamic_code_offset, 32);
  assert(VamanaNode::HAS_HOT_GRAPH);
  assert(VamanaNode::HOT_GRAPH_DYNAMIC_CODE_OFFSET ==
         VamanaNode::HOT_GRAPH_DYNAMIC_HOT_OFFSET + graph_bytes);

  // Publication size is selected from the handle incarnation, so the handle
  // kind must agree with the physical record plane.  In particular, a tagged
  // handle forged over an immutable slot must never append a dynamic tag to
  // the following compact graph record, and incarnation zero cannot name a
  // recyclable dynamic slot.
  const RemotePtr valid_base{0, vamana::hot_graph::kNodeBaseOffset, 0};
  const RemotePtr tagged_base{0, vamana::hot_graph::kNodeBaseOffset, 1};
  const RemotePtr valid_dynamic{0, kDynamicBase, 1};
  const RemotePtr untagged_dynamic{0, kDynamicBase, 0};
  assert(VamanaNode::hot_graph_record_kind_matches(valid_base));
  assert(!VamanaNode::hot_graph_record_kind_matches(tagged_base));
  assert(VamanaNode::hot_graph_record_kind_matches(valid_dynamic));
  assert(!VamanaNode::hot_graph_record_kind_matches(untagged_dynamic));

  // DynaExtent reuses the existing four-byte PQ validation prefix for the
  // extent hint.  A four-byte incarnation-bound trailer closes torn PQ reads;
  // in the production R96/PQ32 layout it consumes existing alignment padding,
  // so neither the physical dynamic-record stride nor update WQE count grows.
  static_assert(VamanaNode::DYNAMIC_CODE_TAG_BYTES == sizeof(u32));
  static_assert(VamanaNode::DYNAMIC_CODE_INCARNATION_MASK == 0x00ffffffu);
  assert(VamanaNode::graph_extent_class(0, 0) == 0);
  assert(VamanaNode::graph_extent_class(8, 0) == 1);
  assert(VamanaNode::graph_extent_class(8, 1) == 2);
  assert(VamanaNode::graph_extent_class(96, 6) == 13);
  constexpr u32 dynamic_tag = VamanaNode::pack_dynamic_navigation_tag(
    0x00abc123u, 13);
  static_assert(VamanaNode::dynamic_navigation_tag_incarnation(dynamic_tag) ==
                0x00abc123u);
  static_assert(VamanaNode::dynamic_navigation_tag_extent_class(dynamic_tag) ==
                13);
  static_assert((dynamic_tag & 0x80000000u) == 0);
  constexpr u32 unknown_tag = VamanaNode::pack_dynamic_navigation_tag(
    RemotePtr::MAX_INCARNATION,
    VamanaNode::DYNAMIC_CODE_EXTENT_CLASS_UNKNOWN);
  static_assert((unknown_tag & 0x80000000u) == 0);

  // Exercise the production serializer used by dynamic graph updates and
  // remote inserts. Reusing one dirty buffer is intentional: shrink must erase
  // neighbors left by a previous larger publication, while the following PQ
  // payload must remain untouched in every lifecycle state.
  assert(VamanaNode::dynamic_graph_publication_layout_valid());
  const size_t publication_bytes =
    VamanaNode::dynamic_graph_publication_size();
  assert(publication_bytes ==
         graph_bytes + VamanaNode::DYNAMIC_CODE_TAG_BYTES);
  constexpr byte_t kPqGuard = 0xa5u;
  std::vector<byte_t> dynamic_tail(publication_bytes + 32, kPqGuard);
  std::vector<byte_t> dynamic_decoded_graph(
    VamanaNode::neighbor_read_size());
  const std::vector<RemotePtr> stable_neighbors{
    RemotePtr{0, 0x2000}, RemotePtr{0, 0x2010},
    RemotePtr{0, 0x2020}, RemotePtr{0, 0x2030},
    RemotePtr{0, 0x2040}, RemotePtr{0, 0x2050},
    RemotePtr{0, 0x2060}, RemotePtr{0, 0x2070},
    RemotePtr{0, 0x2080}};
  const std::vector<RemotePtr> provisional_neighbors{
    RemotePtr{1, 0x3000}, RemotePtr{1, 0x3010}};

  const auto verify_publication = [&](size_t stable_count,
                                      size_t provisional_count,
                                      u32 incarnation,
                                      bool deleted) {
    assert(dynamic_tail[0] == stable_count);
    assert(vamana::hot_graph::provisional_count(dynamic_tail.data()) ==
           provisional_count);
    assert(((dynamic_tail[1] & VamanaNode::HOT_GRAPH_DELETED) != 0) ==
           deleted);
    assert(vamana::hot_graph::load_u32_le(dynamic_tail.data() + 8) ==
           incarnation);
    assert(vamana::hot_graph::load_u16_le(dynamic_tail.data() + 2) ==
           vamana::hot_graph::checksum16(dynamic_tail.data(), graph_bytes));
    const u32 tag = vamana::hot_graph::load_u32_le(
      dynamic_tail.data() + graph_bytes);
    assert(VamanaNode::dynamic_navigation_tag_incarnation(tag) ==
           incarnation);
    assert(VamanaNode::dynamic_navigation_tag_extent_class(tag) ==
           VamanaNode::graph_extent_class(
             static_cast<u32>(stable_count),
             static_cast<u32>(provisional_count)));
    const size_t live_count = stable_count + provisional_count;
    for (size_t byte = vamana::hot_graph::neighbor_offset(
           static_cast<u32>(live_count));
         byte < graph_bytes; ++byte) {
      assert(dynamic_tail[byte] == 0);
    }
    for (size_t byte = publication_bytes;
         byte < dynamic_tail.size(); ++byte) {
      assert(dynamic_tail[byte] == kPqGuard);
    }
    assert(VamanaNode::decode_hot_graph_entry(
      dynamic_tail.data(), dynamic_decoded_graph.data(), incarnation));
    assert(VamanaNode::decoded_neighbor_count(dynamic_decoded_graph.data()) ==
           (deleted ? 0 : live_count));
  };

  // Grow across an eight-edge class boundary.
  assert(VamanaNode::encode_dynamic_graph_publication(
    dynamic_tail.data(), dynamic_tail.size(), stable_neighbors.data(), 3,
    provisional_neighbors.data(), 1, 11, false, 7));
  verify_publication(3, 1, 7, false);
  assert(VamanaNode::encode_dynamic_graph_publication(
    dynamic_tail.data(), dynamic_tail.size(), stable_neighbors.data(), 9,
    provisional_neighbors.data(), 2, 12, false, 7));
  verify_publication(9, 2, 7, false);

  // Shrink must clear the previously published counted suffix.
  assert(VamanaNode::encode_dynamic_graph_publication(
    dynamic_tail.data(), dynamic_tail.size(), stable_neighbors.data(), 1,
    nullptr, 0, 13, false, 7));
  verify_publication(1, 0, 7, false);

  // A tombstone retains its preserved adjacency and class for cleanup, while
  // readers decode the deleted record as an empty adjacency. Re-incarnation
  // must replace the low-24-bit identity in the same publication.
  assert(VamanaNode::encode_dynamic_graph_publication(
    dynamic_tail.data(), dynamic_tail.size(), stable_neighbors.data(), 3,
    provisional_neighbors.data(), 2, 14, true, 8));
  verify_publication(3, 2, 8, true);

  // Fixed graph access keeps the identical graph+tag publication layout but
  // makes the advisory byte inert. The incarnation remains authoritative and
  // the following PQ payload is still outside the write range.
  std::fill(dynamic_tail.begin(), dynamic_tail.end(), kPqGuard);
  assert(VamanaNode::encode_dynamic_graph_publication(
    dynamic_tail.data(), dynamic_tail.size(), stable_neighbors.data(), 3,
    provisional_neighbors.data(), 1, 15, false, 9,
    VamanaNode::HOT_GRAPH_SHARD_BITS, false));
  const u32 fixed_access_tag = vamana::hot_graph::load_u32_le(
    dynamic_tail.data() + graph_bytes);
  assert(VamanaNode::dynamic_navigation_tag_incarnation(fixed_access_tag) ==
         9);
  assert(VamanaNode::dynamic_navigation_tag_extent_class(fixed_access_tag) ==
         VamanaNode::DYNAMIC_CODE_EXTENT_CLASS_UNKNOWN);
  assert(VamanaNode::decode_hot_graph_entry(
    dynamic_tail.data(), dynamic_decoded_graph.data(), 9));
  assert(VamanaNode::decoded_neighbor_count(dynamic_decoded_graph.data()) ==
         4);
  for (size_t byte = publication_bytes;
       byte < dynamic_tail.size(); ++byte) {
    assert(dynamic_tail[byte] == kPqGuard);
  }

  const RemotePtr immutable_base{
    0, vamana::hot_graph::kNodeBaseOffset, 0};
  const RemotePtr tagged_static_address{
    0, vamana::hot_graph::kNodeBaseOffset, 1};
  const RemotePtr misaligned_static_slot{
    0, vamana::hot_graph::kNodeBaseOffset + RemotePtr::OFFSET_ALIGNMENT, 0};
  const RemotePtr past_static_count{
    0, vamana::hot_graph::kNodeBaseOffset + VamanaNode::total_size(), 0};
  const RemotePtr zero_tag_dynamic{0, kDynamicBase, 0};
  assert(VamanaNode::immutable_base_record(immutable_base));
  assert(!VamanaNode::immutable_base_record(tagged_static_address));
  assert(!VamanaNode::immutable_base_record(misaligned_static_slot));
  assert(!VamanaNode::immutable_base_record(past_static_count));
  assert(!VamanaNode::immutable_base_record(zero_tag_dynamic));

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

  // A checksummed counted prefix cannot contain an empty slot. Accepting it
  // as a shorter list would let maintenance silently discard an authoritative
  // edge after a durable malformed publication.
  const RemotePtr encoded_neighbors[] = {
    RemotePtr{0, 0x2000}, RemotePtr{1, 0x4000}};
  std::vector<byte_t> compact_graph(VamanaNode::hot_graph_entry_size());
  std::vector<byte_t> decoded_graph(VamanaNode::neighbor_read_size());
  VamanaNode::encode_hot_graph_entry(
    compact_graph.data(), 2, encoded_neighbors, 2,
    VamanaNode::HOT_GRAPH_SHARD_BITS);
  assert(VamanaNode::decode_hot_graph_entry(
    compact_graph.data(), decoded_graph.data(), 0));

  std::vector<byte_t> reserved_flag_graph = compact_graph;
  reserved_flag_graph[1] |= 0x02u;
  vamana::hot_graph::store_u16_le(
    reserved_flag_graph.data() + 2,
    vamana::hot_graph::checksum16(
      reserved_flag_graph.data(), reserved_flag_graph.size()));
  assert(!VamanaNode::decode_hot_graph_entry(
    reserved_flag_graph.data(), decoded_graph.data(), 0));

  (void)vamana::hot_graph::encode_remote_ptr(
    RemotePtr{}, VamanaNode::HOT_GRAPH_SHARD_BITS,
    compact_graph.data() + vamana::hot_graph::neighbor_offset(0));
  vamana::hot_graph::store_u16_le(
    compact_graph.data() + 2,
    vamana::hot_graph::checksum16(
      compact_graph.data(), compact_graph.size()));
  assert(!VamanaNode::decode_hot_graph_entry(
    compact_graph.data(), decoded_graph.data(), 0));

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

  // Production regression from the five-shard SIFT100M run. This handle was
  // incorrectly reported as malformed after scheduler failure even though it
  // names a properly aligned immutable shard-0 record and its compact graph
  // entry is fully inside the 24-GiB MR.
  constexpr u64 kProductionShardBytes = 24ull << 30;
  constexpr u64 kReportedRaw = 0x8088b29ull;
  VamanaNode::init_static_storage(128, 96, VectorDType::uint8);
  VamanaNode::configure_hot_graph(
    {3194389888ull, 3296159808ull, 3106645568ull,
     3296159808ull, 3106645568ull},
    {19964936ull, 20600998ull, 19416534ull, 20600998ull, 19416534ull},
    832, 3,
    {20444099040ull, 21095426384ull, 19882535296ull,
     21095426384ull, 19882535296ull},
    1040, 160, 992, 32);
  const RemotePtr reported{kReportedRaw};
  assert(reported.memory_node() == 0);
  assert(reported.byte_offset() == 2156442256ull);
  assert(reported.incarnation() == 0);
  assert(VamanaNode::immutable_base_record(reported));
  assert(VamanaNode::hot_graph_entry_offset(reported) == 14407889536ull);
  assert(memory_node_storage_owner_index_detail::storage_pointer_addressable(
    reported, 5, kProductionShardBytes));

  VamanaNode::disable_hot_graph();
  return 0;
}
