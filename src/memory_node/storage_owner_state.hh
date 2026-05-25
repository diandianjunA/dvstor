#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <library/context.hh>
#include <library/hugepage.hh>
#include <library/memory_region.hh>
#include <library/utils.hh>

#include "common/constants.hh"
#include "common/types.hh"
#include "coroutine.hh"
#include "remote_pointer.hh"
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
  u8 edge_count{};
  vec<element_t> components;
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
  u32 recv_slots_per_peer{1};
};

struct StorageOwnerThread;

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

class StorageOwnerLocalCache {
public:
  void init(size_t bytes) {
    if (bytes == 0) {
      return;
    }
    enabled_ = true;
    const size_t snapshot_bytes = VamanaNode::size_until_vector_end() + sizeof(NodeSnapshot) + 64;
    const size_t neighbor_bytes = VamanaNode::NEIGHBORS_SIZE + sizeof(RemotePtr) + 64;
    snapshot_capacity_ = std::max<size_t>(1, bytes / 2 / std::max<size_t>(1, snapshot_bytes));
    neighbor_capacity_ = std::max<size_t>(1, bytes / 2 / std::max<size_t>(1, neighbor_bytes));
  }

  bool enabled() const { return enabled_; }

  bool lookup_snapshot(RemotePtr key, NodeSnapshot& snapshot) {
    if (!enabled_) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = snapshots_.find(key.raw_address);
    if (it == snapshots_.end()) {
      return false;
    }
    snapshot = it->second;
    return true;
  }

  void insert_snapshot(const NodeSnapshot& snapshot) {
    if (!enabled_ || snapshot.rptr.is_null()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (snapshots_.contains(snapshot.rptr.raw_address)) {
      snapshots_[snapshot.rptr.raw_address] = snapshot;
      return;
    }
    evict_fifo(snapshot_order_, snapshots_, snapshot_capacity_);
    snapshot_order_.push_back(snapshot.rptr.raw_address);
    snapshots_[snapshot.rptr.raw_address] = snapshot;
  }

  bool lookup_neighbors(RemotePtr key, vec<RemotePtr>& neighbors) {
    if (!enabled_) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = neighbors_.find(key.raw_address);
    if (it == neighbors_.end()) {
      return false;
    }
    neighbors = it->second;
    return true;
  }

  void insert_neighbors(RemotePtr key, const vec<RemotePtr>& values) {
    if (!enabled_ || key.is_null()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (neighbors_.contains(key.raw_address)) {
      neighbors_[key.raw_address] = values;
      return;
    }
    evict_fifo(neighbor_order_, neighbors_, neighbor_capacity_);
    neighbor_order_.push_back(key.raw_address);
    neighbors_[key.raw_address] = values;
  }

  void invalidate(RemotePtr key) {
    if (!enabled_ || key.is_null()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    snapshots_.erase(key.raw_address);
    neighbors_.erase(key.raw_address);
  }

private:
  template <class Value>
  static void evict_fifo(std::deque<u64>& order, std::unordered_map<u64, Value>& map, size_t capacity) {
    while (map.size() >= capacity && !order.empty()) {
      const u64 victim = order.front();
      order.pop_front();
      map.erase(victim);
    }
  }

  bool enabled_{false};
  size_t snapshot_capacity_{0};
  size_t neighbor_capacity_{0};
  std::mutex mutex_;
  std::deque<u64> snapshot_order_;
  std::deque<u64> neighbor_order_;
  std::unordered_map<u64, NodeSnapshot> snapshots_;
  std::unordered_map<u64, vec<RemotePtr>> neighbors_;
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
  StorageOwnerLocalCache cache;
  u32 running_coroutine{};
  size_t scratch_stride{};
};

struct StorageOwnerInsertJob {
  node_t id{};
  vec<element_t> components;
  bool ok{false};
};

}  // namespace memory_node_detail
