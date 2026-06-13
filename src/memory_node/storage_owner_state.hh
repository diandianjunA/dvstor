#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>

#include <library/context.hh>
#include <library/hugepage.hh>
#include <library/memory_region.hh>
#include <library/utils.hh>

#include "common/constants.hh"
#include "common/types.hh"
#include "coroutine.hh"
#include "remote_pointer.hh"
#include "service/storage_owner_protocol.hh"
#include "vamana/vamana_node.hh"

namespace memory_node_detail {

inline size_t align_to_cacheline(size_t value) {
  while (value % CACHELINE_SIZE != 0) {
    ++value;
  }
  return value;
}

struct BeamEntry {
  RemotePtr rptr;
  distance_t distance{};
  bool expanded{false};
};

struct NodeSnapshot {
  RemotePtr rptr;
  u64 header{};
  node_t id{};
  u32 generation{};
  u8 edge_count{};
  bool deleted{};
  vec<byte_t> vector_data;
};

struct InsertRuntimeState {
  HugePage<byte_t> buffer;
  std::unique_ptr<LocalMemoryRegion> region;
  size_t request_bytes{};
  size_t response_offset{};
  u32 request_slot_count{1};
};

struct PeerRpcRuntimeState {
  HugePage<byte_t> buffer;
  std::unique_ptr<LocalMemoryRegion> region;
  size_t message_bytes{};
  size_t recv_region_bytes{};
  size_t sync_send_offset{};
  size_t async_send_offset{};
  u32 recv_slots_per_peer{1};
  u32 send_slots_per_peer{1};
};

struct StorageOwnerThread;

enum class HandoffResultStatus : u8 {
  pending = 0,
  ok,
  overloaded,
  queue_full,
  timeout,
  shutdown,
  failed,
};

struct HandoffResult {
  HandoffResultStatus status{HandoffResultStatus::failed};
  vec<byte_t> response;
  u64 queue_wait_ns{};
  u64 send_ns{};
  u64 response_wait_ns{};

  bool ok() const { return status == HandoffResultStatus::ok; }
};

struct HandoffRequestState {
  u64 request_id{};
  u32 target_shard{};
  StorageOwnerThread* thread{};
  u32 coroutine_id{};
  vec<byte_t> request;
  vec<byte_t> response;
  std::chrono::steady_clock::time_point queued_at{};
  std::chrono::steady_clock::time_point deadline{};
  std::chrono::steady_clock::time_point send_posted_at{};
  std::chrono::steady_clock::time_point send_completed_at{};
  std::chrono::steady_clock::time_point response_completed_at{};
  HandoffResultStatus status{HandoffResultStatus::pending};
  bool sent{};
  std::atomic<bool> completed{false};
};

struct HandoffResponseTask {
  u32 target_shard{};
  vec<byte_t> payload;
};

struct HandoffSendSlot {
  bool in_use{};
  bool response_only{};
  u32 peer_id{};
  u32 slot_id{};
  u64 wr_id{};
  std::shared_ptr<HandoffRequestState> request;
};

struct PeerHandoffState {
  std::deque<std::shared_ptr<HandoffRequestState>> request_queue;
  std::deque<HandoffResponseTask> response_queue;
  vec<HandoffSendSlot> send_slots;
  std::deque<u32> free_slots;
  u32 inflight_requests{};
  u32 max_queue_depth{};
  u32 max_inflight_requests{};
};

struct PeerPendingSend {
  u32 target_shard{};
  u32 target_qp_idx{};
  u32 thread_id{};
  u32 coroutine_id{};
  StorageOwnerThread* thread{};
  bool async{};
  bool rdma_read_credit{};
};

struct PeerRpcMessage {
  u32 source_shard{};
  vec<byte_t> payload;
};

struct StorageOwnerInsertTask {
  u32 client_id{};
  u32 item_count{};
  u64 batch_id{};
  std::chrono::steady_clock::time_point received_at{};
  vec<byte_t> payload;
};

struct StorageOwnerThread {
  explicit StorageOwnerThread(u32 id, u32 num_coroutines, i32 max_send_queue_wr)
      : id(id), send_wcs(std::max<i32>(1, max_send_queue_wr)), post_balances(num_coroutines) {
    for (auto& balance : post_balances) {
      balance.store(0, std::memory_order_relaxed);
    }
  }

  void init_peer_scratch(Context& peer_context, size_t bytes, size_t per_coroutine_stride = 0) {
    scratch_stride = per_coroutine_stride == 0
                       ? align_to_cacheline(VamanaNode::total_size())
                       : align_to_cacheline(per_coroutine_stride);
    const size_t required_bytes = static_cast<size_t>(std::max<size_t>(1, post_balances.size())) * scratch_stride;
    scratch_buffer.allocate(std::max(bytes, required_bytes));
    scratch_buffer.touch_memory();
    scratch_region = std::make_unique<LocalMemoryRegion>(
      peer_context, scratch_buffer.get_full_buffer(), scratch_buffer.buffer_size);
  }

  bool has_peer_scratch() const { return scratch_region != nullptr; }

  void set_current_coroutine(u32 coroutine_id) { running_coroutine = coroutine_id; }
  void track_post() { ++post_balances[running_coroutine]; }
  bool is_ready(u32 coroutine_id) const { return post_balances[coroutine_id] == 0; }
  byte_t* coroutine_scratch(size_t extra_offset = 0) {
    const size_t offset = static_cast<size_t>(running_coroutine) * scratch_stride + extra_offset;
    lib_assert(offset < scratch_buffer.buffer_size, "storage-owner coroutine scratch buffer exhausted");
    return scratch_buffer.get_full_buffer() + offset;
  }
  size_t coroutine_scratch_offset(size_t extra_offset = 0) const {
    return static_cast<size_t>(running_coroutine) * scratch_stride + extra_offset;
  }

  u32 id{};
  vec<ibv_wc> send_wcs;
  vec<std::atomic<i32>> post_balances;
  vec<u_ptr<StorageOwnerInsertCoroutine>> coroutines;
  HugePage<byte_t> scratch_buffer;
  std::unique_ptr<LocalMemoryRegion> scratch_region;
  u32 running_coroutine{};
  size_t scratch_stride{};
};

struct StorageOwnerInsertJob {
  node_t id{};
  service::storage_owner::MutationKind kind{service::storage_owner::MutationKind::insert};
  vec<byte_t> vector_data;
  service::storage_owner::MutationStatus status{service::storage_owner::MutationStatus::failed};
  bool ok{false};
};

struct FreshnessEntry {
  RemotePtr current;
  u32 generation{};
  bool deleted{};
};

}  // namespace memory_node_detail
