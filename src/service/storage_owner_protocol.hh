#pragma once

#include "common/types.hh"

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
};

struct InsertBatchRequestHeader {
  u32 magic{kInsertMagic};
  u32 dim{};
  u32 owner_storage{};
  u32 source_client{};
  u32 item_count{};
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

struct InsertBatchStats {
  u64 cpu_quantize_ns{};
  u64 cpu_search_ns{};
  u64 cpu_prune_ns{};
  u64 cpu_write_ns{};
  u64 cpu_local_reverse_update_ns{};
  u64 cpu_remote_reverse_update_ns{};
  u64 rdma_peer_read_ns{};
  u64 rdma_peer_write_ns{};
  u64 rdma_peer_cas_ns{};
  u64 snapshot_cache_hits{};
  u64 snapshot_cache_misses{};
  u64 neighbor_cache_hits{};
  u64 neighbor_cache_misses{};
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
  return sizeof(InsertBatchRequestHeader) +
         static_cast<size_t>(item_count) * sizeof(node_t) +
         static_cast<size_t>(item_count) * static_cast<size_t>(dim) * sizeof(element_t);
}

inline size_t insert_batch_response_bytes(u32 item_count) {
  return sizeof(InsertBatchResponseHeader) + sizeof(InsertBatchStats) +
         static_cast<size_t>(item_count) * sizeof(u32);
}

inline InsertBatchStats* response_stats(void* payload) {
  return reinterpret_cast<InsertBatchStats*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(InsertBatchResponseHeader));
}

inline const InsertBatchStats* response_stats(const void* payload) {
  return reinterpret_cast<const InsertBatchStats*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(InsertBatchResponseHeader));
}

inline node_t* request_ids(void* payload) {
  return reinterpret_cast<node_t*>(reinterpret_cast<byte_t*>(payload) + sizeof(InsertBatchRequestHeader));
}

inline const node_t* request_ids(const void* payload) {
  return reinterpret_cast<const node_t*>(reinterpret_cast<const byte_t*>(payload) + sizeof(InsertBatchRequestHeader));
}

inline element_t* request_vectors(void* payload, u32 item_count) {
  return reinterpret_cast<element_t*>(reinterpret_cast<byte_t*>(request_ids(payload) + item_count));
}

inline const element_t* request_vectors(const void* payload, u32 item_count) {
  return reinterpret_cast<const element_t*>(reinterpret_cast<const byte_t*>(request_ids(payload) + item_count));
}

inline u32* response_statuses(void* payload) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(response_stats(payload) + 1));
}

inline const u32* response_statuses(const void* payload) {
  return reinterpret_cast<const u32*>(reinterpret_cast<const byte_t*>(response_stats(payload) + 1));
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
