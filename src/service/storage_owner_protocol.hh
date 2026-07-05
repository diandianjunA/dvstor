#pragma once

#include "common/types.hh"
#include "vamana/vamana_node.hh"

namespace service::storage_owner {

constexpr u32 kInsertMagic = 0x53494e54;  // "SINT"
constexpr u32 kMutationMagic = 0x4d555444;  // D T U M / "DUTM"
constexpr u32 kPeerRpcMagic = 0x53505250;  // "SPRP"

enum class InsertStatus : u32 {
  ok = 0,
  failed = 1,
  overloaded = 2,
};

enum class MutationKind : u32 {
  insert = 1,
  upsert = 2,
  erase = 3,
};

enum class MutationStatus : u32 {
  ok = 0,
  not_found = 1,
  already_exists = 2,
  already_deleted = 3,
  failed = 4,
};

enum class PeerRpcType : u32 {
  reverse_update_request = 1,
  reverse_update_response = 2,
  cleanup_deleted_request = 3,
};

struct InsertBatchRequestHeader {
  u32 magic{kInsertMagic};
  u32 dim{};
  u32 owner_storage{};
  u32 source_client{};
  u32 item_count{};
  u32 vector_dtype{};
  u32 vector_bytes{};
  u32 anchor_hint_count{};
  u64 batch_id{};
};

struct MutationBatchRequestHeader {
  u32 magic{kMutationMagic};
  u32 dim{};
  u32 owner_storage{};
  u32 source_client{};
  u32 item_count{};
  u32 vector_dtype{};
  u32 vector_bytes{};
  u32 anchor_hint_count{};
  u64 batch_id{};
};

struct InsertBatchResponseHeader {
  u32 magic{kInsertMagic};
  u32 owner_storage{};
  u32 item_count{};
  u32 reserved{};
  u64 batch_id{};
};

struct MutationResult {
  u64 new_rptr_raw{};
  u64 old_rptr_raw{};
  u32 generation{};
  u32 reserved{};
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

  u64 storage_owner_anchor_hints{};
  u64 storage_owner_anchor_valid_hints{};
  u64 storage_owner_anchor_expansions{};
  u64 storage_owner_anchor_remote_expansions{};
  u64 storage_owner_anchor_fallbacks{};
  u64 storage_owner_anchor_audits{};
  u64 storage_owner_anchor_audit_failures{};

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

constexpr size_t align_wire_u64(size_t value) {
  return (value + alignof(u64) - 1) & ~(alignof(u64) - 1);
}

static_assert(align_wire_u64(1) == 8);
static_assert(align_wire_u64(8) == 8);

inline size_t insert_anchor_offset(u32 item_count) {
  return align_wire_u64(sizeof(InsertBatchRequestHeader) +
                        static_cast<size_t>(item_count) * sizeof(node_t) +
                        static_cast<size_t>(item_count) * VamanaNode::vector_bytes());
}

inline size_t mutation_anchor_offset(u32 item_count) {
  return align_wire_u64(sizeof(MutationBatchRequestHeader) +
                        static_cast<size_t>(item_count) * sizeof(u32) +
                        static_cast<size_t>(item_count) * sizeof(node_t) +
                        static_cast<size_t>(item_count) * VamanaNode::vector_bytes());
}

inline size_t insert_batch_request_bytes(u32 item_count, u32 dim, u32 anchor_hint_count = 0) {
  (void)dim;
  const size_t vector_end = sizeof(InsertBatchRequestHeader) +
                            static_cast<size_t>(item_count) * sizeof(node_t) +
                            static_cast<size_t>(item_count) * VamanaNode::vector_bytes();
  return anchor_hint_count == 0
    ? vector_end
    : insert_anchor_offset(item_count) +
        static_cast<size_t>(item_count) * anchor_hint_count * sizeof(u64);
}

inline size_t mutation_batch_request_bytes(u32 item_count, u32 dim, u32 anchor_hint_count = 0) {
  (void)dim;
  const size_t vector_end = sizeof(MutationBatchRequestHeader) +
                            static_cast<size_t>(item_count) * sizeof(u32) +
                            static_cast<size_t>(item_count) * sizeof(node_t) +
                            static_cast<size_t>(item_count) * VamanaNode::vector_bytes();
  return anchor_hint_count == 0
    ? vector_end
    : mutation_anchor_offset(item_count) +
        static_cast<size_t>(item_count) * anchor_hint_count * sizeof(u64);
}

inline size_t insert_batch_response_bytes(u32 item_count) {
  return sizeof(InsertBatchResponseHeader) +
         static_cast<size_t>(item_count) * sizeof(u32) +
         static_cast<size_t>(item_count) * sizeof(MutationResult) +
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

inline u32* mutation_request_kinds(void* payload) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(payload) + sizeof(MutationBatchRequestHeader));
}

inline const u32* mutation_request_kinds(const void* payload) {
  return reinterpret_cast<const u32*>(reinterpret_cast<const byte_t*>(payload) + sizeof(MutationBatchRequestHeader));
}

inline node_t* mutation_request_ids(void* payload) {
  return reinterpret_cast<node_t*>(mutation_request_kinds(payload) +
                                   reinterpret_cast<MutationBatchRequestHeader*>(payload)->item_count);
}

inline const node_t* mutation_request_ids(const void* payload) {
  return reinterpret_cast<const node_t*>(mutation_request_kinds(payload) +
                                         reinterpret_cast<const MutationBatchRequestHeader*>(payload)->item_count);
}

inline byte_t* mutation_request_vectors(void* payload, u32 item_count) {
  return reinterpret_cast<byte_t*>(mutation_request_ids(payload) + item_count);
}

inline const byte_t* mutation_request_vectors(const void* payload, u32 item_count) {
  return reinterpret_cast<const byte_t*>(mutation_request_ids(payload) + item_count);
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

inline u64* request_anchor_hints(void* payload, u32 item_count) {
  if (reinterpret_cast<InsertBatchRequestHeader*>(payload)->anchor_hint_count == 0) return nullptr;
  return reinterpret_cast<u64*>(reinterpret_cast<byte_t*>(payload) + insert_anchor_offset(item_count));
}

inline const u64* request_anchor_hints(const void* payload, u32 item_count) {
  if (reinterpret_cast<const InsertBatchRequestHeader*>(payload)->anchor_hint_count == 0) return nullptr;
  return reinterpret_cast<const u64*>(reinterpret_cast<const byte_t*>(payload) +
                                      insert_anchor_offset(item_count));
}

inline u64* mutation_request_anchor_hints(void* payload, u32 item_count) {
  if (reinterpret_cast<MutationBatchRequestHeader*>(payload)->anchor_hint_count == 0) return nullptr;
  return reinterpret_cast<u64*>(reinterpret_cast<byte_t*>(payload) + mutation_anchor_offset(item_count));
}

inline const u64* mutation_request_anchor_hints(const void* payload, u32 item_count) {
  if (reinterpret_cast<const MutationBatchRequestHeader*>(payload)->anchor_hint_count == 0) return nullptr;
  return reinterpret_cast<const u64*>(reinterpret_cast<const byte_t*>(payload) +
                                      mutation_anchor_offset(item_count));
}

inline u32* response_statuses(void* payload) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(payload) + sizeof(InsertBatchResponseHeader));
}

inline const u32* response_statuses(const void* payload) {
  return reinterpret_cast<const u32*>(reinterpret_cast<const byte_t*>(payload) + sizeof(InsertBatchResponseHeader));
}

inline MutationResult* response_mutation_results(void* payload, u32 item_count) {
  return reinterpret_cast<MutationResult*>(response_statuses(payload) + item_count);
}

inline const MutationResult* response_mutation_results(const void* payload, u32 item_count) {
  return reinterpret_cast<const MutationResult*>(response_statuses(payload) + item_count);
}

inline InsertBreakdownCounters* response_breakdown(void* payload, u32 item_count) {
  return reinterpret_cast<InsertBreakdownCounters*>(
    reinterpret_cast<byte_t*>(response_mutation_results(payload, item_count) + item_count));
}

inline const InsertBreakdownCounters* response_breakdown(const void* payload, u32 item_count) {
  return reinterpret_cast<const InsertBreakdownCounters*>(
    reinterpret_cast<const byte_t*>(response_mutation_results(payload, item_count) + item_count));
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

}  // namespace service::storage_owner
