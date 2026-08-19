#include <cassert>
#include <cstring>
#include <vector>

#include "gpu_search/host_orchestrated_policy.hh"
#include "vamana/dynamic_navigation_code.hh"
#include "vamana/hot_graph.hh"

int main() {
  using gpu_search::host_orchestrated_policy::lane_reusable;
  static_assert(lane_reusable(false, false, true));
  static_assert(!lane_reusable(true, false, true));
  static_assert(!lane_reusable(false, true, true));
  static_assert(!lane_reusable(false, false, false));
  using gpu_search::host_orchestrated_policy::graph_snapshot_decodable;
  static_assert(graph_snapshot_decodable(true, false));
  static_assert(!graph_snapshot_decodable(false, false));  // unresolved
  static_assert(!graph_snapshot_decodable(false, true));   // stale handle
  static_assert(!graph_snapshot_decodable(true, true));

  constexpr u32 kDim = 8;
  constexpr u32 kDegree = 8;
  constexpr u32 kCodeBytes = 4;
  constexpr u64 kNodeBase = 0x1000;
  constexpr u64 kGraphBase = 0x8000;
  constexpr u64 kDynamicBase = 0x10000;
  constexpr u64 kStorageBytes = 1ull << 20;

  VamanaNode::disable_hot_graph();
  VamanaNode::init_static_storage(kDim, kDegree, VectorDType::uint8);
  const u32 graph_bytes = VamanaNode::hot_graph_entry_size();
  const u32 dynamic_hot = VamanaNode::total_size();
  const u32 dynamic_code = dynamic_hot + graph_bytes;
  const u32 dynamic_record = VamanaNode::align_compact(
    dynamic_code + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
      kCodeBytes + VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES);
  VamanaNode::configure_hot_graph(
    {kGraphBase}, {2}, graph_bytes,
    vamana::hot_graph::shard_bits_for(1), {kDynamicBase},
    dynamic_record, dynamic_hot, dynamic_code, kCodeBytes);

  gpu_search::format::View view;
  view.layout.dim = kDim;
  view.layout.graph_degree = kDegree;
  view.layout.code_bytes = kCodeBytes;
  view.layout.num_nodes = 2;
  view.layout.num_shards = 1;
  view.layout.graph_entry_bytes = graph_bytes;
  view.shards.push_back({
    .ordinal_base = 0,
    .node_count = 2,
    .node_base_offset = kNodeBase,
    .node_stride = VamanaNode::total_size(),
    .graph_base_offset = kGraphBase,
    .dynamic_base_offset = kDynamicBase,
    .memory_node = 0,
    .dynamic_record_bytes = dynamic_record,
    .dynamic_hot_offset = dynamic_hot,
    .dynamic_code_offset = dynamic_code,
  });

  using gpu_search::host_orchestrated_policy::ResolvedRecord;
  ResolvedRecord resolved;
  const RemotePtr immutable{0, kNodeBase + VamanaNode::total_size(), 0};
  assert(gpu_search::host_orchestrated_policy::resolve_record(
    view, immutable, kStorageBytes, resolved));
  assert(resolved.immutable_base && resolved.static_ordinal == 1);
  assert(resolved.graph_offset == kGraphBase + graph_bytes);

  const RemotePtr dynamic{0, kDynamicBase + dynamic_record, 7};
  assert(gpu_search::host_orchestrated_policy::resolve_record(
    view, dynamic, kStorageBytes, resolved));
  assert(!resolved.immutable_base);
  assert(resolved.graph_offset == dynamic.byte_offset() + dynamic_hot);
  assert(resolved.dynamic_code_offset ==
         dynamic.byte_offset() + dynamic_code);
  assert(!gpu_search::host_orchestrated_policy::resolve_record(
    view, RemotePtr{0, kDynamicBase + 16, 7}, kStorageBytes, resolved));

  std::vector<byte_t> exact(VamanaNode::size_until_vector_end());
  const u64 header = VamanaNode::make_header(dynamic.incarnation());
  std::memcpy(exact.data(), &header, sizeof(header));
  const u32 incarnation = dynamic.incarnation();
  std::memcpy(exact.data() + VamanaNode::offset_slot_incarnation(),
              &incarnation, sizeof(incarnation));
  assert(gpu_search::host_orchestrated_policy::exact_snapshot_visible(
    exact.data(), exact.size(), header, dynamic));
  const u64 locked = header | VamanaNode::HEADER_NODE_LOCK;
  std::memcpy(exact.data(), &locked, sizeof(locked));
  assert(!gpu_search::host_orchestrated_policy::exact_snapshot_visible(
    exact.data(), exact.size(), locked, dynamic));

  std::vector<byte_t> code(
    VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES + kCodeBytes +
      VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES);
  const u32 tag = VamanaNode::pack_dynamic_navigation_tag(
    incarnation, 3);
  vamana::dynamic_navigation_code::store_u32_le(code.data(), tag);
  for (u32 index = 0; index < kCodeBytes; ++index) {
    code[VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES + index] =
      static_cast<byte_t>(index + 1);
  }
  const u32 checksum = vamana::dynamic_navigation_code::checksum(
    tag, code.data() + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES,
    kCodeBytes);
  vamana::dynamic_navigation_code::store_u32_le(
    code.data() + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES + kCodeBytes,
    checksum);
  u8 extent = 0;
  assert(gpu_search::host_orchestrated_policy::
    dynamic_code_snapshot_visible(code.data(), kCodeBytes, dynamic,
                                  &extent));
  assert(extent == 3);
  code.back() ^= 1u;
  assert(!gpu_search::host_orchestrated_policy::
    dynamic_code_snapshot_visible(code.data(), kCodeBytes, dynamic,
                                  &extent));
  return 0;
}
