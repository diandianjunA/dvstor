#pragma once

#include <algorithm>
#include <atomic>
#include <limits>

#include <library/connection_manager.hh>
#include <library/detached_qp.hh>
#include <library/utils.hh>

struct RdmaReadBatchOptions {
  bool adaptive{true};
  u32 chain_size{};
  u32 max_inflight_wrs{};
};

template <typename T>
class SharedContext {
public:
  struct QpRuntimeState {
    u32 send_wr_capacity{};
    std::atomic<u32> outstanding_wrs{0};
    std::atomic<u32> outstanding_chunks{0};
    std::atomic<u32> high_water_wrs{0};
  };

  struct CompletionSlot {
    std::atomic<bool> in_use{false};
    u32 thread_index{};
    u32 coroutine_id{};
    u32 memory_node{};
    u32 qp_index{};
    u32 wr_count{};
  };

  SharedContext(Context& channel_context,
                ClientConnectionManager& cm,
                HugePage<byte_t>& buffer,
                const MemoryRegionTokens& remote_mrts,
                u32 qp_pool_size = 1,
                RdmaReadBatchOptions batch_options = {})
      : context(channel_context.get_config()),
        remote_mrts(remote_mrts),
        batch_options_(batch_options) {
    const u32 num_nodes = cm.server_qps.size();
    qps.resize(num_nodes);
    qp_runtime.resize(num_nodes);
    qp_tie_breakers.resize(num_nodes);
    for (u32 node = 0; node < num_nodes; ++node) {
      auto& per_node = qps[node];
      auto& per_node_runtime = qp_runtime[node];
      per_node.reserve(qp_pool_size);
      per_node_runtime.reserve(qp_pool_size);
      for (u32 p = 0; p < qp_pool_size; ++p) {
        auto& dqp = per_node.emplace_back(
            std::make_unique<DetachedQP>(context, context.get_send_cq(), context.get_receive_cq()));
        dqp->connect(channel_context, context.get_lid(), cm.server_qps[node]);
        auto state = std::make_unique<QpRuntimeState>();
        state->send_wr_capacity = dqp->qp->max_send_wr();
        per_node_runtime.push_back(std::move(state));
      }
      qp_tie_breakers[node] = std::make_unique<std::atomic<u32>>(0);
    }
    const u32 completion_slots = std::max<u32>(
        64, static_cast<u32>(context.get_config().max_send_queue_wr));
    completion_slots_.reserve(completion_slots);
    for (u32 i = 0; i < completion_slots; ++i) {
      completion_slots_.push_back(std::make_unique<CompletionSlot>());
    }
    memory_region = std::make_unique<LocalMemoryRegion>(context, buffer.get_full_buffer(), buffer.buffer_size);
  }

  void register_thread(T* thread) {
    registered_threads.push_back(thread);
    thread->ctx = this;
    thread->ctx_tid = registered_threads.size() - 1;
  }

  ibv_cq* get_cq() { return context.get_send_cq(); }
  u32 get_lkey() const { return memory_region->get_lkey(); }
  MemoryRegionToken* get_remote_mrt(u32 memory_node) { return remote_mrts[memory_node].get(); }
  const RdmaReadBatchOptions& batch_options() const { return batch_options_; }
  u32 effective_chain_size() const {
    if (batch_options_.chain_size != 0) return batch_options_.chain_size;
    return std::max<u32>(1, 2 * context.max_qp_read_atomic());
  }

  u32 qp_credit_limit(u32 node, u32 qp) const {
    const u32 capacity = qp_runtime[node][qp]->send_wr_capacity;
    if (batch_options_.max_inflight_wrs != 0) {
      return std::max<u32>(1, std::min(capacity, batch_options_.max_inflight_wrs));
    }
    const u32 reserve = std::min<u32>(64, std::max<u32>(1, capacity / 8));
    return std::max<u32>(1, capacity - reserve);
  }

  bool try_reserve_qp_wrs(u32 node, u32 qp, u32 wr_count) {
    auto& state = *qp_runtime[node][qp];
    const u32 limit = qp_credit_limit(node, qp);
    u32 current = state.outstanding_wrs.load(std::memory_order_acquire);
    while (current + wr_count <= limit) {
      if (state.outstanding_wrs.compare_exchange_weak(
              current, current + wr_count,
              std::memory_order_acq_rel, std::memory_order_acquire)) {
        state.outstanding_chunks.fetch_add(1, std::memory_order_relaxed);
        u32 high = state.high_water_wrs.load(std::memory_order_relaxed);
        while (high < current + wr_count &&
               !state.high_water_wrs.compare_exchange_weak(
                   high, current + wr_count, std::memory_order_relaxed)) {}
        return true;
      }
    }
    return false;
  }

  bool try_reserve_bulk_qp_wrs(u32 node,
                               u32 preferred_qp,
                               u32 wr_count,
                               u32& selected_qp) {
    const u32 pool_size = static_cast<u32>(qp_runtime[node].size());
    const u32 first_bulk_qp = pool_size > 1 ? 1 : 0;
    u32 best_qp = pool_size;
    u32 best_load = std::numeric_limits<u32>::max();
    for (u32 qp = first_bulk_qp; qp < pool_size; ++qp) {
      const u32 load = qp_runtime[node][qp]->outstanding_wrs.load(std::memory_order_acquire);
      if (load < best_load || (load == best_load && qp == preferred_qp)) {
        best_load = load;
        best_qp = qp;
      }
    }
    if (best_qp < pool_size && try_reserve_qp_wrs(node, best_qp, wr_count)) {
      selected_qp = best_qp;
      return true;
    }
    for (u32 qp = first_bulk_qp; qp < pool_size; ++qp) {
      if (qp == best_qp) continue;
      if (!try_reserve_qp_wrs(node, qp, wr_count)) continue;
      selected_qp = qp;
      return true;
    }
    return false;
  }

  void qp_outstanding_snapshot(vec<vec<u32>>& snapshot) const {
    snapshot.resize(qp_runtime.size());
    for (u32 node = 0; node < qp_runtime.size(); ++node) {
      snapshot[node].resize(qp_runtime[node].size());
      for (u32 qp = 0; qp < qp_runtime[node].size(); ++qp) {
        snapshot[node][qp] =
            qp_runtime[node][qp]->outstanding_wrs.load(std::memory_order_acquire);
      }
    }
  }

  void next_qp_tie_breakers(vec<u32>& values) {
    values.resize(qp_tie_breakers.size());
    for (u32 node = 0; node < qp_tie_breakers.size(); ++node) {
      values[node] = qp_tie_breakers[node]->fetch_add(1, std::memory_order_relaxed);
    }
  }

  u64 try_create_batch_completion(u32 thread_index,
                                  u32 coroutine_id,
                                  u32 node,
                                  u32 qp,
                                  u32 wr_count) {
    if (completion_slots_.empty()) return 0;
    const u32 start = next_completion_slot_.fetch_add(1, std::memory_order_relaxed);
    for (u32 attempt = 0; attempt < completion_slots_.size(); ++attempt) {
      const u32 index = (start + attempt) % completion_slots_.size();
      auto& slot = *completion_slots_[index];
      bool expected = false;
      if (!slot.in_use.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel)) {
        continue;
      }
      slot.thread_index = thread_index;
      slot.coroutine_id = coroutine_id;
      slot.memory_node = node;
      slot.qp_index = qp;
      slot.wr_count = wr_count;
      return kBatchCompletionFlag | (static_cast<u64>(index) + 1);
    }
    return 0;
  }

  void complete_send(u64 wr_id) {
    if ((wr_id & kBatchCompletionFlag) == 0) {
      auto [ctx_offset, coroutine_id] = decode_64bit(wr_id);
      --registered_threads[ctx_offset]->post_balances[coroutine_id];
      return;
    }

    const u64 encoded_index = wr_id & ~kBatchCompletionFlag;
    lib_assert(encoded_index > 0 && encoded_index <= completion_slots_.size(),
               "invalid RDMA batch completion token");
    auto& slot = *completion_slots_[encoded_index - 1];
    lib_assert(slot.in_use.load(std::memory_order_acquire),
               "RDMA batch completion token already released");

    auto& state = *qp_runtime[slot.memory_node][slot.qp_index];
    state.outstanding_wrs.fetch_sub(slot.wr_count, std::memory_order_acq_rel);
    state.outstanding_chunks.fetch_sub(1, std::memory_order_relaxed);
    --registered_threads[slot.thread_index]->post_balances[slot.coroutine_id];
    slot.in_use.store(false, std::memory_order_release);
  }

public:
  Context context;
  vec<vec<u_ptr<DetachedQP>>> qps;  // [memory_node][pool_index]
  vec<vec<u_ptr<QpRuntimeState>>> qp_runtime;
  vec<T*> registered_threads;

private:
  static constexpr u64 kBatchCompletionFlag = 1ull << 63;
  u_ptr<LocalMemoryRegion> memory_region;
  const MemoryRegionTokens& remote_mrts;
  RdmaReadBatchOptions batch_options_;
  vec<u_ptr<std::atomic<u32>>> qp_tie_breakers;
  vec<u_ptr<CompletionSlot>> completion_slots_;
  std::atomic<u32> next_completion_slot_{0};
};
