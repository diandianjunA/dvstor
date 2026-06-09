#pragma once

#include "common/types.hh"
#include "vamana/vamana_node.hh"

namespace service::storage_owner {

constexpr u32 kInsertMagic = 0x53494e54;  // "SINT"
constexpr u32 kPeerRpcMagic = 0x53505250;  // "SPRP"

enum class InsertStatus : u32 {
  ok = 0,
  failed = 1,
};

enum class PeerRpcType : u32 {
  reverse_update_request = 1,
  reverse_update_response = 2,
  search_handoff_request = 3,
  search_handoff_response = 4,
};

struct InsertBatchRequestHeader {
  u32 magic{kInsertMagic};
  u32 dim{};
  u32 owner_storage{};
  u32 source_client{};
  u32 item_count{};
  u32 vector_dtype{};
  u32 vector_bytes{};
  u32 reserved{};
  u64 batch_id{};
};

struct InsertBatchResponseHeader {
  u32 magic{kInsertMagic};
  u32 owner_storage{};
  u32 item_count{};
  u32 reserved{};
  u64 batch_id{};
};

struct InsertBreakdownCounters {
  u64 storage_owner_queue_wait_ns{};
  u64 storage_owner_medoid_ns{};
  u64 storage_owner_search_ns{};
  u64 storage_owner_prune_ns{};
  u64 storage_owner_write_node_ns{};
  u64 storage_owner_local_reverse_ns{};
  u64 storage_owner_remote_reverse_ns{};
  u64 storage_owner_peer_reverse_apply_ns{};
  u64 storage_owner_response_send_ns{};

  u64 storage_owner_search_select_ns{};
  u64 storage_owner_search_neighbor_read_ns{};
  u64 storage_owner_search_snapshot_read_ns{};
  u64 storage_owner_search_distance_ns{};
  u64 storage_owner_search_beam_update_ns{};
  u64 storage_owner_search_result_sort_ns{};
  u64 storage_owner_prune_snapshot_read_ns{};
  u64 storage_owner_prune_distance_ns{};
  u64 storage_owner_prune_sort_ns{};
  u64 storage_owner_prune_pair_distance_ns{};

  u64 total() const {
    return storage_owner_queue_wait_ns +
           storage_owner_medoid_ns +
           storage_owner_search_ns +
           storage_owner_prune_ns +
           storage_owner_write_node_ns +
           storage_owner_local_reverse_ns +
           storage_owner_remote_reverse_ns +
           storage_owner_peer_reverse_apply_ns +
           storage_owner_response_send_ns;
  }
};

struct PeerRpcHeader {
  u32 magic{kPeerRpcMagic};
  u32 type{};
  u32 source_shard{};
  u32 item_count{};
  u64 request_id{};
  u32 status{static_cast<u32>(InsertStatus::failed)};
  u32 reserved{};
};

struct ReverseUpdateOp {
  u64 target_raw{};
  u64 candidate_raw{};
};

inline size_t insert_batch_request_bytes(u32 item_count, u32 dim) {
  (void)dim;
  return sizeof(InsertBatchRequestHeader) +
         static_cast<size_t>(item_count) * sizeof(node_t) +
         static_cast<size_t>(item_count) * VamanaNode::vector_bytes();
}

inline size_t insert_batch_response_bytes(u32 item_count) {
  return sizeof(InsertBatchResponseHeader) +
         static_cast<size_t>(item_count) * sizeof(u32) +
         sizeof(InsertBreakdownCounters) +
         sizeof(u32) +
         static_cast<size_t>(item_count) * VamanaNode::R * sizeof(u64);
}

inline node_t* request_ids(void* payload) {
  return reinterpret_cast<node_t*>(reinterpret_cast<byte_t*>(payload) + sizeof(InsertBatchRequestHeader));
}

inline const node_t* request_ids(const void* payload) {
  return reinterpret_cast<const node_t*>(reinterpret_cast<const byte_t*>(payload) + sizeof(InsertBatchRequestHeader));
}

inline byte_t* request_vectors(void* payload, u32 item_count) {
  return reinterpret_cast<byte_t*>(request_ids(payload) + item_count);
}

inline const byte_t* request_vectors(const void* payload, u32 item_count) {
  return reinterpret_cast<const byte_t*>(request_ids(payload) + item_count);
}

inline byte_t* request_vector(void* payload, u32 item_count, u32 index) {
  return request_vectors(payload, item_count) + static_cast<size_t>(index) * VamanaNode::vector_bytes();
}

inline const byte_t* request_vector(const void* payload, u32 item_count, u32 index) {
  return request_vectors(payload, item_count) + static_cast<size_t>(index) * VamanaNode::vector_bytes();
}

inline u32* response_statuses(void* payload) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(payload) + sizeof(InsertBatchResponseHeader));
}

inline const u32* response_statuses(const void* payload) {
  return reinterpret_cast<const u32*>(reinterpret_cast<const byte_t*>(payload) + sizeof(InsertBatchResponseHeader));
}

inline InsertBreakdownCounters* response_breakdown(void* payload, u32 item_count) {
  return reinterpret_cast<InsertBreakdownCounters*>(
    reinterpret_cast<byte_t*>(response_statuses(payload) + item_count));
}

inline const InsertBreakdownCounters* response_breakdown(const void* payload, u32 item_count) {
  return reinterpret_cast<const InsertBreakdownCounters*>(
    reinterpret_cast<const byte_t*>(response_statuses(payload) + item_count));
}

inline u32* response_invalidation_count(void* payload, u32 item_count) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(response_breakdown(payload, item_count) + 1));
}

inline const u32* response_invalidation_count(const void* payload, u32 item_count) {
  return reinterpret_cast<const u32*>(reinterpret_cast<const byte_t*>(response_breakdown(payload, item_count) + 1));
}

inline u64* response_invalidated_raws(void* payload, u32 item_count) {
  return reinterpret_cast<u64*>(response_invalidation_count(payload, item_count) + 1);
}

inline const u64* response_invalidated_raws(const void* payload, u32 item_count) {
  return reinterpret_cast<const u64*>(response_invalidation_count(payload, item_count) + 1);
}

inline u32 response_invalidation_capacity(u32 item_count) {
  return item_count * VamanaNode::R;
}

inline size_t reverse_update_request_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) * sizeof(ReverseUpdateOp);
}

inline size_t reverse_update_response_bytes() {
  return sizeof(PeerRpcHeader);
}

inline ReverseUpdateOp* reverse_update_ops(void* payload) {
  return reinterpret_cast<ReverseUpdateOp*>(reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const ReverseUpdateOp* reverse_update_ops(const void* payload) {
  return reinterpret_cast<const ReverseUpdateOp*>(reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

struct __attribute__((packed)) BeamEntrySerialized {
  u64 rptr_raw;
  float distance;
};
static_assert(sizeof(BeamEntrySerialized) == 12, "BeamEntrySerialized must be 12 bytes");

struct SearchHandoffRequestHeader {
  PeerRpcHeader rpc;
  u32 beam_width;
  u32 snapshot_batch;
  u32 originator_shard;
  u32 visited_count;
  u32 vector_bytes;
  u32 reserved;
};

struct SearchHandoffResponseHeader {
  PeerRpcHeader rpc;
  u32 updated_beam_count;
  u32 new_visited_count;
  u32 total_visited_count;
  u32 reserved;
};

inline size_t search_handoff_request_bytes(u32 beam_count, u32 visited_count, u32 vector_bytes) {
  return sizeof(SearchHandoffRequestHeader) +
         static_cast<size_t>(vector_bytes) +
         static_cast<size_t>(beam_count) * sizeof(BeamEntrySerialized) +
         static_cast<size_t>(visited_count) * sizeof(u64);
}

inline size_t search_handoff_response_bytes(u32 beam_count, u32 visited_count) {
  return sizeof(SearchHandoffResponseHeader) +
         static_cast<size_t>(beam_count) * sizeof(BeamEntrySerialized) +
         static_cast<size_t>(visited_count) * sizeof(u64);
}

inline byte_t* handoff_query_vector(void* payload) {
  return reinterpret_cast<byte_t*>(reinterpret_cast<SearchHandoffRequestHeader*>(payload) + 1);
}

inline const byte_t* handoff_query_vector(const void* payload) {
  return reinterpret_cast<const byte_t*>(reinterpret_cast<const SearchHandoffRequestHeader*>(payload) + 1);
}

inline BeamEntrySerialized* handoff_request_beam(void* payload, u32 vector_bytes) {
  return reinterpret_cast<BeamEntrySerialized*>(handoff_query_vector(payload) + vector_bytes);
}

inline const BeamEntrySerialized* handoff_request_beam(const void* payload, u32 vector_bytes) {
  return reinterpret_cast<const BeamEntrySerialized*>(handoff_query_vector(payload) + vector_bytes);
}

inline u64* handoff_request_visited(void* payload, u32 vector_bytes, u32 beam_count) {
  return reinterpret_cast<u64*>(handoff_request_beam(payload, vector_bytes) + beam_count);
}

inline const u64* handoff_request_visited(const void* payload, u32 vector_bytes, u32 beam_count) {
  return reinterpret_cast<const u64*>(handoff_request_beam(payload, vector_bytes) + beam_count);
}

inline BeamEntrySerialized* handoff_response_beam(void* payload) {
  return reinterpret_cast<BeamEntrySerialized*>(reinterpret_cast<SearchHandoffResponseHeader*>(payload) + 1);
}

inline const BeamEntrySerialized* handoff_response_beam(const void* payload) {
  return reinterpret_cast<const BeamEntrySerialized*>(reinterpret_cast<const SearchHandoffResponseHeader*>(payload) + 1);
}

inline u64* handoff_response_visited(void* payload, u32 beam_count) {
  return reinterpret_cast<u64*>(handoff_response_beam(payload) + beam_count);
}

inline const u64* handoff_response_visited(const void* payload, u32 beam_count) {
  return reinterpret_cast<const u64*>(handoff_response_beam(payload) + beam_count);
}

}  // namespace service::storage_owner
