#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/constants.hh"
#include "gpu_search/graph_record_validation.hh"
#include "gpu_search/index_format.hh"
#include "gpu_search/persistent_kernel.hh"
#include "nlohmann/json.hh"
#include "vamana/hot_graph.hh"
#include "vamana/vamana_node.hh"

namespace {

void test_dynamic_pq_arena_mapping_and_incarnation_order() {
  const gpu_search::DeviceShardRegion shard{
    .dynamic_base_offset = 4096,
    .dynamic_record_bytes = 1040,
    .dynamic_arena_base_slot = 700,
    .dynamic_arena_slot_count = 3,
  };
  u64 slot = 0;
  assert(
    gpu_search::dynamic_code_arena_slot_from_offset(shard, 4096, 703, slot));
  assert(slot == 700);
  assert(gpu_search::dynamic_code_arena_slot_from_offset(shard, 4096 + 2 * 1040,
                                                         703, slot));
  assert(slot == 702);
  assert(
    !gpu_search::dynamic_code_arena_slot_from_offset(shard, 4095, 703, slot));
  assert(
    !gpu_search::dynamic_code_arena_slot_from_offset(shard, 4097, 703, slot));
  assert(!gpu_search::dynamic_code_arena_slot_from_offset(
    shard, 4096 + 3 * 1040, 703, slot));

  constexpr u32 incarnation = 19;
  const u32 tag = gpu_search::make_dynamic_code_tag(incarnation, 7);
  assert(gpu_search::dynamic_code_tag_incarnation(tag) == incarnation);
  assert(gpu_search::dynamic_code_tag_extent_class(tag) == 7);
  assert(gpu_search::dynamic_code_arena_state_matches(tag, incarnation));
  assert(!gpu_search::dynamic_code_arena_state_matches(
    gpu_search::kPersistentDynamicCodeArenaBusy | tag, incarnation));
  assert(gpu_search::dynamic_code_arena_read_stable(
    tag, gpu_search::make_dynamic_code_tag(incarnation, 8), incarnation));
  assert(!gpu_search::dynamic_code_arena_read_stable(
    tag, gpu_search::kPersistentDynamicCodeArenaBusy | tag, incarnation));
  assert(!gpu_search::dynamic_code_arena_read_stable(
    tag, gpu_search::make_dynamic_code_tag(incarnation + 1, 1), incarnation));
  assert(gpu_search::dynamic_code_tag_extent_class(0xff000000u | incarnation) ==
         gpu_search::kPersistentDynamicCodeArenaUnknownExtent);

  assert(gpu_search::dynamic_code_arena_can_publish(0, 1));
  assert(gpu_search::dynamic_code_arena_can_publish(
    gpu_search::make_dynamic_code_tag(1, 9), 2));
  assert(!gpu_search::dynamic_code_arena_can_publish(
    gpu_search::make_dynamic_code_tag(2, 1), 2));
  assert(!gpu_search::dynamic_code_arena_can_publish(
    gpu_search::make_dynamic_code_tag(3, 1), 2));
  assert(!gpu_search::dynamic_code_arena_can_publish(
    gpu_search::kPersistentDynamicCodeArenaBusy | 1u, 2));
  assert(gpu_search::dynamic_code_arena_first_occupancy(0));
  assert(!gpu_search::dynamic_code_arena_first_occupancy(tag));
  assert(!gpu_search::dynamic_code_arena_first_occupancy(
    gpu_search::kPersistentDynamicCodeArenaBusy | tag));

  u32 promoted = 0;
  assert(gpu_search::dynamic_code_arena_promoted_extent_state(tag, incarnation,
                                                              11, promoted));
  assert(gpu_search::dynamic_code_tag_incarnation(promoted) == incarnation);
  assert(gpu_search::dynamic_code_tag_extent_class(promoted) == 11);
  assert(!gpu_search::dynamic_code_arena_promoted_extent_state(
    promoted, incarnation, 10, promoted));
  // A delayed repair for incarnation 19 must not change a recycled slot 20.
  const u32 recycled = gpu_search::make_dynamic_code_tag(incarnation + 1, 2);
  assert(!gpu_search::dynamic_code_arena_promoted_extent_state(
    recycled, incarnation, 12, promoted));
  assert(promoted == recycled);

  const u32 unknown = gpu_search::make_dynamic_code_tag(
    incarnation, gpu_search::kPersistentDynamicCodeArenaUnknownExtent);
  u32 refined = 0;
  assert(gpu_search::dynamic_code_arena_refined_unknown_extent_state(
    unknown, incarnation, 6, refined));
  assert(gpu_search::dynamic_code_tag_incarnation(refined) == incarnation);
  assert(gpu_search::dynamic_code_tag_extent_class(refined) == 6);
  assert(!gpu_search::dynamic_code_arena_refined_unknown_extent_state(
    refined, incarnation, 5, refined));
  assert(!gpu_search::dynamic_code_arena_refined_unknown_extent_state(
    unknown, incarnation + 1, 6, refined));
  assert(refined == unknown);
  assert(!gpu_search::dynamic_code_arena_refined_unknown_extent_state(
    0, incarnation, 6, refined));
  assert(refined == 0);
  const u32 busy_unknown =
    gpu_search::kPersistentDynamicCodeArenaBusy | unknown;
  assert(!gpu_search::dynamic_code_arena_refined_unknown_extent_state(
    busy_unknown, incarnation, 6, refined));
  assert(refined == busy_unknown);

  u32 demoted = 0;
  assert(gpu_search::dynamic_code_arena_guarded_demoted_extent_state(
    promoted = gpu_search::make_dynamic_code_tag(incarnation, 11), incarnation,
    7, demoted));
  assert(gpu_search::dynamic_code_tag_extent_class(demoted) == 8);
  assert(!gpu_search::dynamic_code_arena_guarded_demoted_extent_state(
    demoted, incarnation, 7, demoted));
  assert(!gpu_search::dynamic_code_arena_guarded_demoted_extent_state(
    recycled, incarnation, 0, demoted));
  assert(demoted == recycled);
}

void test_dynamic_navigation_code_width_semantics() {
  namespace format = gpu_search::format;
  constexpr u32 dim = 128;
  constexpr u32 degree = 96;
  VamanaNode::init_static_storage(dim, degree, VectorDType::uint8);

  for (u32 payload_bytes : {16u, 32u, 64u}) {
    const u32 dynamic_hot_offset = static_cast<u32>(VamanaNode::total_size());
    const u32 graph_entry_bytes =
      static_cast<u32>(VamanaNode::hot_graph_entry_size());
    const u32 dynamic_code_offset = dynamic_hot_offset + graph_entry_bytes;
    const u32 dynamic_record_bytes = static_cast<u32>(VamanaNode::align_compact(
      dynamic_code_offset + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
      payload_bytes + VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES));
    if (payload_bytes == 32) {
      const u32 old_stride = static_cast<u32>(VamanaNode::align_compact(
        dynamic_code_offset + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
        payload_bytes));
      assert(dynamic_code_offset == 992);
      assert(old_stride == 1040);
      assert(dynamic_record_bytes == old_stride);
      assert(VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES + payload_bytes +
               VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES ==
             40);
    }

    VamanaNode::configure_hot_graph({4096}, {1}, graph_entry_bytes, 0, {8192},
                                    dynamic_record_bytes, dynamic_hot_offset,
                                    dynamic_code_offset, payload_bytes);
    assert(VamanaNode::HAS_HOT_GRAPH);
    assert(VamanaNode::HOT_GRAPH_DYNAMIC_CODE_BYTES == payload_bytes);
    assert(VamanaNode::dynamic_navigation_code_payload_bytes() ==
           payload_bytes);
    assert(dynamic_code_offset + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
             payload_bytes + VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES <=
           VamanaNode::allocation_size());

    std::vector<u8> payload(payload_bytes);
    for (u32 byte = 0; byte < payload_bytes; ++byte) {
      payload[byte] = static_cast<u8>(byte * 17u + 3u);
    }
    const u32 tag = VamanaNode::pack_dynamic_navigation_tag(19, 2);
    std::array<u8, VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES> checksum{};
    vamana::dynamic_navigation_code::store_u32_le(
      checksum.data(), vamana::dynamic_navigation_code::checksum(
                         tag, payload.data(), payload_bytes));
    assert(vamana::dynamic_navigation_code::validate(
      tag, payload.data(), payload_bytes, checksum.data()));
    // Extent is advisory and may change independently of the immutable PQ.
    assert(vamana::dynamic_navigation_code::validate(
      VamanaNode::pack_dynamic_navigation_tag(19, 9), payload.data(),
      payload_bytes, checksum.data()));
    assert(!vamana::dynamic_navigation_code::validate(
      VamanaNode::pack_dynamic_navigation_tag(20, 2), payload.data(),
      payload_bytes, checksum.data()));
    payload[payload_bytes / 2] ^= 1u;
    assert(!vamana::dynamic_navigation_code::validate(
      tag, payload.data(), payload_bytes, checksum.data()));

    format::StorageControlBlock control{
      .dynamic_record_bytes = dynamic_record_bytes,
      .dynamic_hot_offset = dynamic_hot_offset,
      .dynamic_code_offset = dynamic_code_offset,
      .code_bytes = VamanaNode::dynamic_navigation_code_payload_bytes(),
    };
    assert(control.code_bytes == payload_bytes);

    // The graph publication serializer writes the four-byte dynamic tag
    // immediately after the graph entry. A metadata padding gap would make the
    // GPU read a different address, so reject it even when the padded record
    // is otherwise large enough for the complete PQ payload.
    const u32 padded_code_offset = dynamic_code_offset + 16;
    const u32 padded_record_bytes = static_cast<u32>(VamanaNode::align_compact(
      padded_code_offset + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
      payload_bytes + VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES));
    VamanaNode::configure_hot_graph({4096}, {1}, graph_entry_bytes, 0, {8192},
                                    padded_record_bytes, dynamic_hot_offset,
                                    padded_code_offset, payload_bytes);
    assert(!VamanaNode::HAS_HOT_GRAPH);

    // A record that fits the prefix and PQ payload but omits the checksum must
    // be rejected even though code_bytes itself remains payload-only.
    VamanaNode::configure_hot_graph(
      {4096}, {1}, graph_entry_bytes, 0, {8192},
      dynamic_code_offset + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
        payload_bytes,
      dynamic_hot_offset, dynamic_code_offset, payload_bytes);
    assert(!VamanaNode::HAS_HOT_GRAPH);
  }
  VamanaNode::disable_hot_graph();
}

void test_dynamic_navigation_code_torn_snapshot_detection() {
  constexpr u32 payload_bytes = 32;
  constexpr u32 record_bytes = VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
                               payload_bytes +
                               VamanaNode::DYNAMIC_CODE_CHECKSUM_BYTES;
  const auto make_record = [](u32 incarnation, u8 extent, u8 seed) {
    std::array<u8, record_bytes> record{};
    const u32 tag =
      VamanaNode::pack_dynamic_navigation_tag(incarnation, extent);
    vamana::dynamic_navigation_code::store_u32_le(record.data(), tag);
    for (u32 byte = 0; byte < payload_bytes; ++byte) {
      record[VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES + byte] =
        static_cast<u8>(seed + byte * 29u);
    }
    const u8* payload =
      record.data() + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES;
    vamana::dynamic_navigation_code::store_u32_le(
      record.data() + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
        payload_bytes,
      vamana::dynamic_navigation_code::checksum(tag, payload, payload_bytes));
    return record;
  };
  const auto accepted = [](const std::array<u8, record_bytes>& record,
                           u32 incarnation) {
    const u32 tag = vamana::dynamic_navigation_code::load_u32_le(record.data());
    const u8* payload =
      record.data() + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES;
    const u8* checksum = payload + payload_bytes;
    return VamanaNode::dynamic_navigation_tag_incarnation(tag) == incarnation &&
           vamana::dynamic_navigation_code::validate(tag, payload,
                                                     payload_bytes, checksum);
  };

  const auto old_record = make_record(41, 2, 7);
  const auto new_record = make_record(42, 9, 131);
  assert(accepted(old_record, 41));
  assert(accepted(new_record, 42));

  // Model every prefix/suffix visibility boundary of an in-place remote body
  // WRITE in both directions.  Apart from the documented 32-bit collision
  // boundary, a mixed record must not be accepted as either incarnation.
  for (u32 cut = 0; cut <= record_bytes; ++cut) {
    for (bool new_prefix : {false, true}) {
      std::array<u8, record_bytes> mixed{};
      const auto& prefix = new_prefix ? new_record : old_record;
      const auto& suffix = new_prefix ? old_record : new_record;
      std::copy_n(prefix.begin(), cut, mixed.begin());
      std::copy(suffix.begin() + cut, suffix.end(), mixed.begin() + cut);
      if (accepted(mixed, 41)) assert(mixed == old_record);
      if (accepted(mixed, 42)) assert(mixed == new_record);
    }
  }
}

void test_supported_gpu_layout_limits() {
  namespace format = gpu_search::format;
  format::NavigationLayout layout{
    .dim = 128,
    .graph_degree = kMaxSupportedGraphDegree,
    .vector_dtype = static_cast<u32>(VectorDType::float32),
    .pq_subquantizers = 16,
    .pq_bits = 8,
    .code_bytes = 16,
    .num_shards = 64,
    .graph_entry_bytes = 1088,
    .graph_pointer_bytes = format::kCompactPointerBytes,
    .graph_shard_bits = RemotePtr::MEMORY_NODE_BITS,
    .num_nodes = 64,
    .base_generation = 1,
    .model_checksum = 1,
  };
  std::string error;
  assert(format::validate_layout(layout, &error));
  format::View view;
  view.layout = layout;
  view.shards.reserve(layout.num_shards);
  for (u32 shard = 0; shard < layout.num_shards; ++shard) {
    view.shards.push_back(format::ShardRegion{
      .ordinal_base = shard,
      .node_count = 1,
      .node_base_offset = format::kNodeBaseOffset,
      .node_stride = 576,
      .graph_base_offset = 4096,
      .dynamic_base_offset = 16384,
      .control_remote_offset = 8192,
      .code_remote_offset = 12288,
      .code_bytes = 16,
      .memory_node = shard,
      .dynamic_record_bytes = 1728,
      .dynamic_hot_offset = 576,
      .dynamic_code_offset = 1664,
    });
  }
  assert(format::validate_view(view, &error));

  layout.num_shards = 65;
  layout.graph_shard_bits = RemotePtr::MEMORY_NODE_BITS + 1;
  assert(!format::validate_layout(layout, &error));

  layout.num_shards = 64;
  layout.graph_shard_bits = RemotePtr::MEMORY_NODE_BITS;
  layout.graph_degree = kMaxSupportedGraphDegree + 1;
  assert(!format::validate_layout(layout, &error));
}

void test_tagged_remote_pointer() {
  const RemotePtr static_node{63, RemotePtr::BYTE_OFFSET_CAPACITY - 16};
  const RemotePtr first_dynamic{63, RemotePtr::BYTE_OFFSET_CAPACITY - 16, 1};
  const RemotePtr last_dynamic{63, RemotePtr::BYTE_OFFSET_CAPACITY - 16,
                               RemotePtr::MAX_INCARNATION};
  assert(static_node.memory_node() == 63);
  assert(static_node.byte_offset() == RemotePtr::BYTE_OFFSET_CAPACITY - 16);
  assert(static_node.incarnation() == 0);
  assert(first_dynamic.physical_address_raw() ==
         static_node.physical_address_raw());
  assert(first_dynamic != static_node);
  assert(last_dynamic.incarnation() == RemotePtr::MAX_INCARNATION);
  assert(RemotePtr{last_dynamic.raw_address} == last_dynamic);
  std::array<byte_t, vamana::hot_graph::kTaggedPointerBytes> encoded{};
  assert(vamana::hot_graph::encode_remote_ptr(last_dynamic, 0, encoded.data()));
  assert(vamana::hot_graph::decode_remote_ptr(encoded.data(), 0) ==
         last_dynamic);

  bool rejected = false;
  try {
    (void)RemotePtr{0, 17};
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  assert(rejected);
  rejected = false;
  try {
    (void)RemotePtr{0, 16, RemotePtr::MAX_INCARNATION + 1};
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  assert(rejected);
}

void test_graph_record_stale_incarnation_is_not_transport_failure() {
  namespace validation = gpu_search::graph_record_validation;
  constexpr u32 graph_degree = 3;
  constexpr u32 graph_capacity = 5;
  constexpr u32 record_bytes = 56;
  std::array<byte_t, record_bytes> record{};
  record[0] = 1;
  record[1] = 0;
  const auto store_u32 = [&](size_t offset, u32 value) {
    record[offset + 0] = static_cast<byte_t>(value);
    record[offset + 1] = static_cast<byte_t>(value >> 8);
    record[offset + 2] = static_cast<byte_t>(value >> 16);
    record[offset + 3] = static_cast<byte_t>(value >> 24);
  };
  store_u32(8, 12);
  store_u32(12, 0);
  const auto seal = [&]() {
    const u16 checksum =
      validation::checksum16(record.data(), static_cast<u32>(record.size()));
    record[2] = static_cast<byte_t>(checksum);
    record[3] = static_cast<byte_t>(checksum >> 8);
  };
  seal();

  const auto current = validation::classify_snapshot(
    record.data(), record.size(), graph_degree, graph_capacity, 12);
  assert(current == validation::SnapshotState::valid);
  assert(validation::decide_read_action(true, current, false) ==
         validation::ReadAction::accept);

  // An older dynamic handle can outlive durable cleanup in a read-committed
  // query. A complete record for the replacement incarnation is stale for that
  // handle, not evidence of transport corruption.
  const auto stale = validation::classify_snapshot(
    record.data(), record.size(), graph_degree, graph_capacity, 11);
  assert(stale == validation::SnapshotState::stale_incarnation);
  assert(validation::decide_read_action(true, stale, false) ==
         validation::ReadAction::discard_stale);

  // Static slots never recycle, and a torn record must not be mistaken for a
  // benign stale dynamic handle merely because its incarnation bytes differ.
  assert(validation::classify_snapshot(record.data(), record.size(),
                                       graph_degree, graph_capacity, 0) ==
         validation::SnapshotState::invalid);
  record[16] ^= 1;
  assert(validation::classify_snapshot(record.data(), record.size(),
                                       graph_degree, graph_capacity, 11) ==
         validation::SnapshotState::invalid);
  assert(validation::decide_read_action(true,
                                        validation::SnapshotState::invalid,
                                        true) == validation::ReadAction::retry);
  assert(validation::decide_read_action(true,
                                        validation::SnapshotState::invalid,
                                        false) == validation::ReadAction::fail);
  assert(validation::decide_read_action(false, validation::SnapshotState::valid,
                                        true) == validation::ReadAction::fail);
}

void test_graph_live_extent_reconstructs_canonical_record() {
  namespace validation = gpu_search::graph_record_validation;
  constexpr u32 graph_degree = 96;
  constexpr u32 graph_capacity = 102;
  constexpr u32 record_bytes = 16 + graph_capacity * sizeof(u64);
  static_assert(record_bytes == 832);

  for (const u32 exponent : {0u, 1u, 2u, 7u, 64u, 368u, 816u}) {
    u32 repeated = 1;
    for (u32 index = 0; index < exponent; ++index) {
      repeated *= 16777619u;
    }
    assert(validation::fnv1a_prime_power(exponent) == repeated);
  }

  assert(validation::graph_extent_bytes_for_class(0, record_bytes,
                                                  graph_capacity) == 16);
  assert(validation::graph_extent_bytes_for_class(1, record_bytes,
                                                  graph_capacity) == 80);
  assert(validation::graph_extent_bytes_for_class(7, record_bytes,
                                                  graph_capacity) == 464);
  assert(validation::graph_extent_bytes_for_class(
           13, record_bytes, graph_capacity) == record_bytes);
  assert(validation::graph_extent_bytes_for_class(
           validation::kGraphExtentClassUnknown, record_bytes,
           graph_capacity) == record_bytes);
  assert(
    validation::graph_extent_class_for_required_bytes(16, graph_capacity) == 0);
  assert(validation::graph_extent_class_for_required_bytes(
           16 + 8 * sizeof(u64), graph_capacity) == 1);
  assert(validation::graph_extent_class_for_required_bytes(
           16 + 9 * sizeof(u64), graph_capacity) == 2);
  assert(validation::graph_extent_class_for_required_bytes(
           17, graph_capacity) == validation::kGraphExtentClassUnknown);
  assert(validation::graph_extent_class_for_required_bytes(
           record_bytes + sizeof(u64), graph_capacity) ==
         validation::kGraphExtentClassUnknown);

  const u32 packed_classes =
    3u | (7u << 8) | (validation::kGraphExtentClassUnknown << 16) | (9u << 24);
  assert(validation::packed_graph_extent_class(packed_classes, 0) == 3);
  assert(validation::packed_graph_extent_class(packed_classes, 1) == 7);
  assert(validation::packed_graph_extent_class(packed_classes, 2) ==
         validation::kGraphExtentClassUnknown);
  assert(validation::packed_graph_extent_class(packed_classes, 3) == 9);
  u32 promoted_word = 0;
  assert(validation::promoted_graph_extent_word(packed_classes, 1, 8,
                                                promoted_word));
  assert(validation::packed_graph_extent_class(promoted_word, 0) == 3);
  assert(validation::packed_graph_extent_class(promoted_word, 1) == 8);
  assert(validation::packed_graph_extent_class(promoted_word, 2) ==
         validation::kGraphExtentClassUnknown);
  assert(validation::packed_graph_extent_class(promoted_word, 3) == 9);
  assert(!validation::promoted_graph_extent_word(promoted_word, 1, 7,
                                                 promoted_word));
  assert(!validation::promoted_graph_extent_word(promoted_word, 2, 12,
                                                 promoted_word));
  // The optimistic hint is not charged against the legacy three full-record
  // snapshot attempts. Fixed/full: attempts 0,1 may retry and 2 is final.
  assert(validation::snapshot_retry_available(0, 0, false, 3, 3));
  assert(validation::snapshot_retry_available(1, 0, false, 3, 3));
  assert(!validation::snapshot_retry_available(2, 0, false, 3, 3));
  // Live: attempt 0 is short; attempts 1,2,3 remain three independent full
  // opportunities, with the third full attempt final.
  assert(validation::snapshot_retry_available(0, 1, true, 4, 3));
  assert(validation::snapshot_retry_available(1, 1, false, 4, 3));
  assert(validation::snapshot_retry_available(2, 1, false, 4, 3));
  assert(!validation::snapshot_retry_available(3, 1, false, 4, 3));
  // Header->Neighbor consumes two optimistic partial reads. A checksum
  // conflict between them must still leave all three authoritative full
  // attempts available: attempts 2,3 may retry and attempt 4 is final.
  assert(validation::snapshot_retry_available(0, 2, true, 5, 3));
  assert(validation::snapshot_retry_available(1, 2, true, 5, 3));
  assert(validation::snapshot_retry_available(2, 2, false, 5, 3));
  assert(validation::snapshot_retry_available(3, 2, false, 5, 3));
  assert(!validation::snapshot_retry_available(4, 2, false, 5, 3));
  // R=128 plus eight provisional slots requires 17 classes; a four-bit
  // encoding would silently truncate this supported layout.
  assert(validation::graph_extent_bytes_for_class(
           17, 16 + 136 * sizeof(u64), 136) == 16 + 136 * sizeof(u64));

  std::array<byte_t, record_bytes> authoritative{};
  authoritative[0] = 47;
  authoritative[1] = static_cast<byte_t>(2u << 4);
  for (u32 edge = 0; edge < 49; ++edge) {
    const u64 handle =
      RemotePtr{edge % 4, 16 + static_cast<u64>(edge) * 64}.raw_address;
    std::memcpy(authoritative.data() + validation::kGraphRecordHeaderBytes +
                  static_cast<size_t>(edge) * sizeof(handle),
                &handle, sizeof(handle));
  }
  const u16 checksum =
    validation::checksum16(authoritative.data(), authoritative.size());
  authoritative[2] = static_cast<byte_t>(checksum);
  authoritative[3] = static_cast<byte_t>(checksum >> 8);
  assert(validation::classify_snapshot(
           authoritative.data(), authoritative.size(), graph_degree,
           graph_capacity, 0) == validation::SnapshotState::valid);

  const u32 extent_bytes =
    validation::graph_extent_bytes_for_class(7, record_bytes, graph_capacity);
  assert(validation::checksum16_zero_extended_prefix(
           authoritative.data(), extent_bytes, authoritative.size()) ==
         checksum);
  std::array<byte_t, record_bytes> reconstructed;
  // The unread scratch suffix is deliberately poisoned. Short validation must
  // neither read nor clear it: it hashes the transferred prefix and advances
  // the canonical zero suffix algebraically.
  reconstructed.fill(0xa5);
  std::memcpy(reconstructed.data(), authoritative.data(), extent_bytes);
  u32 required_bytes = 0;
  assert(validation::required_live_extent_bytes(
    reconstructed.data(), extent_bytes, graph_degree, graph_capacity,
    required_bytes));
  assert(required_bytes == 16 + 49 * sizeof(u64));
  assert(required_bytes <= extent_bytes);
  // Class rounding fetched seven inactive slots. They are outside the logical
  // prefix and outside the decoder's count, so checksum validation need not
  // read them. Even a concurrently written unpublished slot can be ignored as
  // part of the still-valid old logical snapshot.
  reconstructed[required_bytes + 3] = 0x4b;
  assert(validation::classify_zero_extended_snapshot(
           reconstructed.data(), extent_bytes, reconstructed.size(),
           graph_degree, graph_capacity,
           0) == validation::SnapshotState::valid);

  // Canonical records in several extent classes must produce exactly the
  // existing full-record checksum without touching the unread scratch bytes.
  for (const u32 live_edges : {0u, 1u, 8u, 9u, 32u, 80u, 95u}) {
    std::array<byte_t, record_bytes> record{};
    record[0] = static_cast<byte_t>(live_edges);
    for (u32 edge = 0; edge < live_edges; ++edge) {
      const u64 handle =
        RemotePtr{edge % 4, 16 + static_cast<u64>(edge) * 64}.raw_address;
      std::memcpy(record.data() + validation::kGraphRecordHeaderBytes +
                    static_cast<size_t>(edge) * sizeof(handle),
                  &handle, sizeof(handle));
    }
    const u16 full_checksum =
      validation::checksum16(record.data(), record.size());
    record[2] = static_cast<byte_t>(full_checksum);
    record[3] = static_cast<byte_t>(full_checksum >> 8);
    const u8 extent_class =
      static_cast<u8>((live_edges + validation::kGraphExtentEdgesPerClass - 1) /
                      validation::kGraphExtentEdgesPerClass);
    const u32 transferred = validation::graph_extent_bytes_for_class(
      extent_class, record_bytes, graph_capacity);
    assert(transferred < record_bytes);
    const u32 logical_bytes =
      validation::kGraphRecordHeaderBytes + live_edges * sizeof(u64);
    assert(validation::checksum16_zero_extended_prefix(
             record.data(), logical_bytes, record.size()) == full_checksum);

    std::array<byte_t, record_bytes> poisoned_scratch;
    poisoned_scratch.fill(0x5a);
    std::memcpy(poisoned_scratch.data(), record.data(), transferred);
    for (u32 byte = logical_bytes; byte < transferred; ++byte) {
      poisoned_scratch[byte] = 0xc3;
    }
    assert(validation::classify_zero_extended_snapshot(
             poisoned_scratch.data(), transferred, poisoned_scratch.size(),
             graph_degree, graph_capacity,
             0) == validation::SnapshotState::valid);
  }

  // A build-time hint that is now one class too small must be detected from
  // the returned header before checksum acceptance or neighbor decoding.
  const u32 stale_extent_bytes =
    validation::graph_extent_bytes_for_class(6, record_bytes, graph_capacity);
  assert(stale_extent_bytes < required_bytes);
  assert(validation::required_live_extent_bytes(
    reconstructed.data(), stale_extent_bytes, graph_degree, graph_capacity,
    required_bytes));
  assert(required_bytes > stale_extent_bytes);

  // If the authoritative record contains data outside the hinted extent, its
  // stored full checksum cannot be reconstructed with a logical-zero suffix.
  // The short attempt is rejected and the existing retry policy safely
  // promotes it to a full read; the authoritative full snapshot then passes.
  std::array<byte_t, record_bytes> nonzero_unread_suffix = authoritative;
  nonzero_unread_suffix[extent_bytes + 17] = 0x6d;
  const u16 nonzero_checksum = validation::checksum16(
    nonzero_unread_suffix.data(), nonzero_unread_suffix.size());
  nonzero_unread_suffix[2] = static_cast<byte_t>(nonzero_checksum);
  nonzero_unread_suffix[3] = static_cast<byte_t>(nonzero_checksum >> 8);
  reconstructed.fill(0xa5);
  std::memcpy(reconstructed.data(), nonzero_unread_suffix.data(), extent_bytes);
  const auto short_nonzero = validation::classify_zero_extended_snapshot(
    reconstructed.data(), extent_bytes, reconstructed.size(), graph_degree,
    graph_capacity, 0);
  assert(short_nonzero == validation::SnapshotState::invalid);
  assert(validation::decide_read_action(true, short_nonzero, true) ==
         validation::ReadAction::retry);
  assert(validation::classify_snapshot(nonzero_unread_suffix.data(),
                                       nonzero_unread_suffix.size(),
                                       graph_degree, graph_capacity,
                                       0) == validation::SnapshotState::valid);

  reconstructed.fill(0xa5);
  std::memcpy(reconstructed.data(), authoritative.data(), extent_bytes);
  reconstructed[32] ^= 1;
  assert(validation::classify_zero_extended_snapshot(
           reconstructed.data(), extent_bytes, reconstructed.size(),
           graph_degree, graph_capacity,
           0) == validation::SnapshotState::invalid);
}

void test_centroid_route_publication(
  gpu_search::format::CentroidScalarType scalar_type, u32 dim) {
  namespace format = gpu_search::format;
  constexpr u32 shard = 1;
  constexpr u32 shard_count = 3;
  constexpr u32 entry_capacity = format::kStorageCentroidRouteMaxLiveEntries;
  const u64 publication_bytes =
    format::storage_centroid_route_publication_bytes(dim, scalar_type,
                                                     entry_capacity);
  assert(publication_bytes != 0);
  assert(publication_bytes % 64 == 0);

  void* allocation =
    std::aligned_alloc(64, static_cast<size_t>(publication_bytes));
  assert(allocation != nullptr);
  span<byte_t> publication{static_cast<byte_t*>(allocation),
                           static_cast<size_t>(publication_bytes)};

  format::StorageCentroidRouteDescriptor descriptor{
    .remote_offset = 64 * 1024,
    .publication_bytes = publication_bytes,
    .layout_version = 9,
    .dim = dim,
    .centroid_scalar_type = static_cast<u32>(scalar_type),
    .shard_count = shard_count,
    .live_entry_capacity = entry_capacity,
  };
  std::string error;
  assert(format::validate_storage_centroid_route_descriptor(
    descriptor, dim, shard_count, &error));

  std::vector<f32> centroid32;
  std::vector<f64> centroid64;
  const void* centroid_data = nullptr;
  if (scalar_type == format::CentroidScalarType::float32) {
    centroid32.resize(dim);
    for (u32 index = 0; index < dim; ++index) {
      centroid32[index] = static_cast<f32>(index) * 0.125f - 7.0f;
    }
    centroid_data = centroid32.data();
  } else {
    centroid64.resize(dim);
    for (u32 index = 0; index < dim; ++index) {
      centroid64[index] = static_cast<f64>(index) * 0.0625 - 11.0;
    }
    centroid_data = centroid64.data();
  }
  const std::array<format::StorageCentroidRouteEntry, entry_capacity>
    entry_storage{{
      {.remote_node = RemotePtr{shard, 64}.raw_address,
       .generation = 7,
       .flags = format::kStorageCentroidRouteLive},
      {.remote_node = RemotePtr{shard, 128}.raw_address,
       .generation = 8,
       .flags = format::kStorageCentroidRouteLive},
      // Poison immediately follows the logical span. A capacity-sized
      // memcpy would copy these records and is caught even without an ASan
      // build.
      {.remote_node = ~u64{0}, .generation = ~u32{0}, .flags = ~u32{0}},
      {.remote_node = ~u64{0}, .generation = ~u32{0}, .flags = ~u32{0}},
    }};
  const span<const format::StorageCentroidRouteEntry> entries{
    entry_storage.data(), 2};
  assert(format::prepare_storage_centroid_route_publication(
    publication, shard, dim, scalar_type, entry_capacity, 17, 1234,
    centroid_data, entries, &error));
  if (!format::validate_storage_centroid_route_publication(
        publication, descriptor, shard, &error)) {
    throw std::runtime_error(error);
  }

  auto* header =
    reinterpret_cast<format::StorageCentroidRoutePublicationHeader*>(
      publication.data());
  assert(header->sequence == 2);
  assert(header->total_bytes == publication_bytes);
  assert(header->shard_version == 17);
  assert(header->vector_count == 1234);
  assert(header->live_entry_count == entries.size());
  const auto decoded_entries = format::storage_centroid_route_entries(
    span<const byte_t>{publication.data(), publication.size()});
  assert(decoded_entries.size() == entries.size());
  assert(std::memcmp(decoded_entries.data(), entries.data(),
                     entries.size() * sizeof(entries[0])) == 0);
  const auto* capacity_entries =
    reinterpret_cast<const format::StorageCentroidRouteEntry*>(
      publication.data() + header->entries_offset);
  for (size_t index = entries.size(); index < entry_capacity; ++index) {
    assert(capacity_entries[index].remote_node == 0);
    assert(capacity_entries[index].generation == 0);
    assert(capacity_entries[index].flags == 0);
  }
  const void* decoded_centroid = format::storage_centroid_route_centroid_data(
    span<const byte_t>{publication.data(), publication.size()});
  const size_t centroid_bytes =
    static_cast<size_t>(dim) * format::centroid_scalar_bytes(scalar_type);
  assert(decoded_centroid != nullptr);
  assert(std::memcmp(decoded_centroid, centroid_data, centroid_bytes) == 0);

  // An odd sequence is a publication in progress and must never be consumed.
  header->sequence = 3;
  assert(!format::validate_storage_centroid_route_publication(
    publication, descriptor, shard, &error));
  header->sequence = 2;

  // The seqlock does not replace body integrity: a stable-looking torn body is
  // rejected by its checksum.
  publication[header->centroid_offset + centroid_bytes / 2] ^= 1;
  assert(!format::validate_storage_centroid_route_publication(
    publication, descriptor, shard, &error));
  publication[header->centroid_offset + centroid_bytes / 2] ^= 1;
  assert(format::validate_storage_centroid_route_publication(
    publication, descriptor, shard, &error));

  auto malformed_descriptor = descriptor;
  malformed_descriptor.publication_bytes -= 64;
  assert(!format::validate_storage_centroid_route_descriptor(
    malformed_descriptor, dim, shard_count, &error));
  std::free(allocation);
}

void test_exact_record_trailer_alignment() {
  constexpr u32 spacev_record_bytes = 24u + 100u;
  constexpr u32 deep_record_bytes = 24u + 96u * sizeof(float);
  constexpr u32 sift_record_bytes = 24u + 128u;
  static_assert(gpu_search::exact_record_trailer_offset(spacev_record_bytes) ==
                128u);
  static_assert(gpu_search::exact_record_trailer_offset(deep_record_bytes) ==
                deep_record_bytes);
  static_assert(gpu_search::exact_record_trailer_offset(sift_record_bytes) ==
                sift_record_bytes);
  static_assert(gpu_search::exact_record_trailer_offset(spacev_record_bytes) %
                  alignof(u64) ==
                0u);
  static_assert(gpu_search::exact_record_trailer_offset(spacev_record_bytes) +
                  sizeof(u64) ==
                136u);

  static_assert(gpu_search::exact_snapshot_local_layout_matches(
    4096u, 4096u + 128u, spacev_record_bytes));
  static_assert(gpu_search::exact_snapshot_local_layout_matches(
    4096u, 4096u + 408u, deep_record_bytes));
  static_assert(gpu_search::exact_snapshot_local_layout_matches(
    4096u, 4096u + 152u, sift_record_bytes));
  static_assert(
    gpu_search::exact_snapshot_local_layout_matches(4096u, 4096u + 128u, 125u));
  static_assert(!gpu_search::exact_snapshot_local_layout_matches(
    4096u, 4096u + spacev_record_bytes, spacev_record_bytes));

  static_assert(gpu_search::dynamic_code_scratch_stride(23u) == 24u);
  static_assert(gpu_search::dynamic_code_scratch_stride(28u) == 28u);
  static_assert(gpu_search::dynamic_code_scratch_stride(38u) == 40u);
  static_assert(gpu_search::dynamic_code_scratch_stride(40u) == 40u);
  static_assert((4096u + gpu_search::dynamic_code_scratch_stride(23u)) %
                  alignof(u32) ==
                0u);
}

}  // namespace

int main() {
  namespace format = gpu_search::format;
  test_dynamic_pq_arena_mapping_and_incarnation_order();
  test_dynamic_navigation_code_width_semantics();
  test_dynamic_navigation_code_torn_snapshot_detection();
  test_exact_record_trailer_alignment();
  test_supported_gpu_layout_limits();
  test_tagged_remote_pointer();
  test_graph_record_stale_incarnation_is_not_transport_failure();
  test_graph_live_extent_reconstructs_canonical_record();
  format::View view;
  view.layout.dim = 16;
  view.layout.graph_degree = 3;
  view.layout.vector_dtype = static_cast<u32>(VectorDType::uint8);
  view.layout.pq_subquantizers = 16;
  view.layout.pq_bits = 8;
  view.layout.code_bytes = 16;
  view.layout.model_checksum = 0x12345678ULL;
  view.layout.num_shards = 2;
  view.layout.graph_entry_bytes = 40;
  view.layout.graph_pointer_bytes = 8;
  view.layout.graph_shard_bits = 1;
  view.layout.num_nodes = 4;
  view.shards = {
    {.ordinal_base = 0,
     .node_count = 2,
     .node_base_offset = 16,
     .node_stride = 64,
     .graph_base_offset = 4096,
     .dynamic_base_offset = 16384,
     .control_remote_offset = 8192,
     .code_remote_offset = 12288,
     .code_bytes = 32,
     .memory_node = 0,
     .dynamic_record_bytes = 128,
     .dynamic_hot_offset = 64,
     .dynamic_code_offset = 104},
    {.ordinal_base = 2,
     .node_count = 2,
     .node_base_offset = 16,
     .node_stride = 64,
     .graph_base_offset = 4096,
     .dynamic_base_offset = 16384,
     .control_remote_offset = 8192,
     .code_remote_offset = 12288,
     .code_bytes = 32,
     .memory_node = 1,
     .dynamic_record_bytes = 128,
     .dynamic_hot_offset = 64,
     .dynamic_code_offset = 104},
  };
  std::string error;
  assert(format::validate_view(view, &error));
  static_assert(format::kStorageControlVersion == 4);
  format::StorageControlBlock control;
  assert(control.version == format::kStorageControlVersion);
  assert(control.header_bytes == sizeof(format::StorageControlBlock));
  static_assert(sizeof(format::StorageControlBlock) == 192);
  static_assert(offsetof(format::StorageControlBlock, centroid_route) == 128);
  assert(sizeof(format::StorageControlBlock) <= format::kStorageControlBytes);
  assert(control.reserved0 == 0);
  assert(control.reserved1 == 0);
  test_centroid_route_publication(format::CentroidScalarType::float32, 257);
  test_centroid_route_publication(format::CentroidScalarType::float64, 1024);
  assert(format::storage_centroid_route_publication_bytes(
           1024, format::CentroidScalarType::float64,
           format::kStorageCentroidRouteMaxLiveEntries) >
         format::kStorageControlBytes);

  RemotePtr pointer;
  assert(format::ordinal_to_remote(view, 0, pointer));
  assert(pointer == RemotePtr(0, 16));
  assert(format::ordinal_to_remote(view, 3, pointer));
  assert(pointer == RemotePtr(1, 80));
  u32 ordinal = 0;
  assert(format::remote_to_ordinal(view, RemotePtr(1, 16), ordinal));
  assert(ordinal == 2);
  assert(!format::remote_to_ordinal(view, RemotePtr(1, 16384), ordinal));

  format::View malformed = view;
  malformed.shards[1].ordinal_base = 3;
  assert(!format::validate_view(malformed, &error));
  malformed = view;
  malformed.shards[0].code_remote_offset += 64;
  assert(!format::validate_view(malformed, &error));
  malformed = view;
  malformed.shards[0].dynamic_code_offset += 8;
  malformed.shards[0].dynamic_record_bytes += 16;
  assert(!format::validate_view(malformed, &error));
  malformed = view;
  malformed.shards[0].dynamic_base_offset =
    RemotePtr::BYTE_OFFSET_CAPACITY - 64;
  assert(!format::validate_view(malformed, &error));

  const auto code_path =
    std::filesystem::temp_directory_path() / "dvstor-pq16-codes.bin";
  std::vector<byte_t> payload(32, 0x5a);
  format::CodeHeader code_header;
  code_header.memory_node = 0;
  code_header.code_bytes = 16;
  code_header.node_size = 64;
  code_header.vector_dtype = static_cast<u32>(VectorDType::uint8);
  code_header.entry_count = 2;
  code_header.remote_offset = 12288;
  code_header.payload_bytes = payload.size();
  code_header.model_checksum = view.layout.model_checksum;
  code_header.payload_checksum =
    format::checksum64(payload.data(), payload.size());
  code_header.build_fingerprint = 0x123456789abcdef0ULL;
  code_header.shard_fingerprint = 0x0fedcba987654321ULL;
  {
    std::ofstream output(code_path, std::ios::binary | std::ios::trunc);
    format::CodeHeader placeholder;
    output.write(reinterpret_cast<const char*>(&placeholder),
                 sizeof(placeholder));
    output.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    assert(format::write_code_header(output, code_header, &error));
  }
  format::CodeHeader loaded_code;
  assert(format::read_code_header(code_path, loaded_code, &error));
  assert(loaded_code.payload_checksum == code_header.payload_checksum);
  assert(loaded_code.vector_dtype == static_cast<u32>(VectorDType::uint8));
  assert(loaded_code.build_fingerprint == code_header.build_fingerprint);
  assert(loaded_code.shard_fingerprint == code_header.shard_fingerprint);

  auto invalid_code_header = loaded_code;
  invalid_code_header.vector_dtype = static_cast<u32>(VectorDType::int8) + 1;
  assert(!format::validate_code_header(invalid_code_header, &error));
  invalid_code_header = loaded_code;
  invalid_code_header.build_fingerprint = 0;
  assert(!format::validate_code_header(invalid_code_header, &error));
  invalid_code_header = loaded_code;
  invalid_code_header.shard_fingerprint = 0;
  assert(!format::validate_code_header(invalid_code_header, &error));
  invalid_code_header = loaded_code;
  invalid_code_header.reserved[0] = 1;
  assert(!format::validate_code_header(invalid_code_header, &error));

  // Header identity is checksum-covered. A sidecar renamed from another build
  // cannot have its build fingerprint edited to look local.
  {
    std::fstream codes(code_path,
                       std::ios::binary | std::ios::in | std::ios::out);
    codes.seekg(static_cast<std::streamoff>(
      offsetof(format::CodeHeader, build_fingerprint)));
    char byte = 0;
    codes.read(&byte, 1);
    byte ^= 1;
    codes.seekp(static_cast<std::streamoff>(
      offsetof(format::CodeHeader, build_fingerprint)));
    codes.write(&byte, 1);
    assert(codes.good());
  }
  format::CodeHeader tampered_code;
  assert(!format::read_code_header(code_path, tampered_code, &error));
  {
    std::fstream codes(code_path,
                       std::ios::binary | std::ios::in | std::ios::out);
    codes.seekg(static_cast<std::streamoff>(
      offsetof(format::CodeHeader, build_fingerprint)));
    char byte = 0;
    codes.read(&byte, 1);
    byte ^= 1;
    codes.seekp(static_cast<std::streamoff>(
      offsetof(format::CodeHeader, build_fingerprint)));
    codes.write(&byte, 1);
    assert(codes.good());
  }
  assert(format::read_code_header(code_path, tampered_code, &error));

  {
    std::fstream codes(code_path,
                       std::ios::binary | std::ios::in | std::ios::out);
    codes.seekg(static_cast<std::streamoff>(sizeof(format::CodeHeader)));
    char byte = 0;
    codes.read(&byte, 1);
    byte ^= 1;
    codes.seekp(static_cast<std::streamoff>(sizeof(format::CodeHeader)));
    codes.write(&byte, 1);
    assert(codes.good());
  }
  format::CodeHeader unchanged_header;
  assert(format::read_code_header(code_path, unchanged_header, &error));
  assert(unchanged_header.payload_checksum == code_header.payload_checksum);

  // Runtime layout synthesis is deliberately independent of an offline
  // medoid or a sampled static entry table. Query seeds arrive only through
  // validated storage-canonical centroid publications.
  const auto metadata_prefix =
    std::filesystem::temp_directory_path() / "dvstor-layout-metadata-test";
  const auto metadata_path =
    std::filesystem::path{metadata_prefix.string() + ".meta.json"};
  const std::vector<u64> counts{100, 100, 100, 100, 100};
  const std::vector<u64> dynamic_offsets{196608, 196608, 196608, 196608,
                                         196608};
  const std::vector<u64> control_offsets{196608, 196608, 196608, 196608,
                                         196608};
  const std::vector<u64> code_offsets{
    196608 + format::kStorageControlBytes,
    196608 + format::kStorageControlBytes,
    196608 + format::kStorageControlBytes,
    196608 + format::kStorageControlBytes,
    196608 + format::kStorageControlBytes,
  };
  const std::vector<u64> code_sizes{3200, 3200, 3200, 3200, 3200};
  const std::vector<u64> dynamic_node_offsets{
    204960, 204960, 204960, 204960, 204960,
  };
  nlohmann::json metadata{
    {"schema_version", format::kMetadataSchemaVersion},
    {"distance", "l2"},
    {"node_layout", "plain"},
    {"storage_format", "vamana_tagged_v2"},
    {"remote_ptr_format", "tagged_inc24_shard6_off34x16_v1"},
    {"centroid_state_format", "physical_shard_centroid_v2_bound"},
    {"index_build_fingerprint", 0x123456789abcdef0ull},
    {"shard_build_fingerprints", std::vector<u64>{11, 12, 13, 14, 15}},
    {"slot_incarnation_offset", 16},
    {"navigation_quantizer", "opq_pq"},
    {"navigation_format", "opq_pq_graph_v1"},
    {"num_memory_nodes", 5},
    {"hot_graph_entry_counts", counts},
    {"hot_graph_offsets", std::vector<u64>{65536, 65536, 65536, 65536, 65536}},
    {"hot_graph_dynamic_base_offsets", dynamic_offsets},
    {"storage_control_remote_offsets", control_offsets},
    {"dynamic_node_base_offsets", dynamic_node_offsets},
    {"dynamic_navigation_code_validation_bytes", 4},
    {"dynamic_navigation_code_checksum_bytes", 4},
    {"navigation_code_remote_offsets", code_offsets},
    {"navigation_code_region_bytes", code_sizes},
    {"dim", 128},
    {"R", 96},
    {"vector_data_type", "uint8"},
    {"vector_bytes", 128},
    {"pq_subquantizers", 32},
    {"pq_bits", 8},
    {"navigation_code_bytes", 32},
    {"navigation_model_checksum", 1},
    {"hot_graph_entry_size", 832},
    {"hot_graph_pointer_bytes", 8},
    {"hot_graph_shard_bits", 3},
    {"node_size", 512},
    {"hot_graph_dynamic_record_bytes", 1392},
    {"hot_graph_dynamic_hot_offset", 512},
    {"dynamic_navigation_code_offset", 1344},
    {"num_vectors", 500},
  };
  assert(!metadata.contains("medoid"));
  assert(!metadata.contains("navigation_entry_points"));
  {
    std::ofstream output(metadata_path, std::ios::trunc);
    output << metadata;
  }
  format::View synthesized;
  if (!format::synthesize_distributed_view(metadata_prefix, synthesized,
                                           &error)) {
    throw std::runtime_error(error);
  }
  assert(synthesized.layout.num_nodes == 500);
  assert(synthesized.layout.num_shards == 5);
  assert(synthesized.shards.size() == 5);
  assert(format::validate_view(synthesized, &error));

  metadata["R"] = kMaxSupportedGraphDegree + 1;
  {
    std::ofstream output(metadata_path, std::ios::trunc);
    output << metadata;
  }
  assert(
    !format::synthesize_distributed_view(metadata_prefix, synthesized, &error));

  std::filesystem::remove(code_path);
  std::filesystem::remove(metadata_path);
  return 0;
}
