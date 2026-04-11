#pragma once

#include <library/hugepage.hh>
#include <library/thread.hh>
#include <random>

#include "buffer_allocator.hh"
#include "common/statistics.hh"
#include "coroutine.hh"
#include "gpu/gpu_buffer_manager.hh"
#include "service/breakdown.hh"
#include "shared_context.hh"

// forward declaration
namespace cache {
class Cache;
}

enum class ServiceWorkerRole { none, insert, query };

class ComputeThread : public Thread {
public:
  ComputeThread(u32 id,
                u32 compute_node_id,
                i32 max_send_queue_wr,
                BufferAllocator& buffer_allocator,
                cache::Cache& cache,
                u32 num_memory_nodes,
                u32 num_coroutines)
      : Thread(id),
        node_id(compute_node_id),
        send_wcs(max_send_queue_wr),
        buffer_allocator(buffer_allocator),
        cache(cache),
        post_balances(num_coroutines),
        gpu_post_balances(num_coroutines),
        max_send_queue_wr_(max_send_queue_wr),
        active_samples_(num_coroutines, nullptr) {
    for (auto& balance : post_balances) {
      balance.store(0, std::memory_order_relaxed);
    }
    for (auto& balance : gpu_post_balances) {
      balance.store(0, std::memory_order_relaxed);
    }

    // allocate single pointer slot (for RDMA requests) per coroutine
    for (idx_t i = 0; i < num_coroutines; ++i) {
      pointer_slots_.push_back(buffer_allocator.allocate_pointer());
    }

    // initialize PRNG
    dist_ = std::uniform_int_distribution<u32>(0, num_memory_nodes - 1);
  }

  void poll_cq() {
    Context::poll_send_cq(send_wcs.data(), max_send_queue_wr_, ctx->get_cq(), [&](u64 wr_id) {
      auto [ctx_offset, coroutine_id] = decode_64bit(wr_id);
      --ctx->registered_threads[ctx_offset]->post_balances[coroutine_id];
    });
  }

  void reset() {
    stats = statistics::ThreadStatistics{};
    query_results.clear();

    running_coroutine_ = 0;
    vamana_coroutines.clear();
    for (auto& balance : post_balances) {
      balance.store(0, std::memory_order_relaxed);
    }
    for (auto& balance : gpu_post_balances) {
      balance.store(0, std::memory_order_relaxed);
    }
    std::fill(active_samples_.begin(), active_samples_.end(), nullptr);
  }

  u32 get_random_memory_node() { return dist_(generator_); }
  u64 create_wr_id() const { return encode_64bit(ctx_tid, running_coroutine_); }
  bool is_ready(u32 coroutine_id) const {
    return post_balances[coroutine_id] == 0 && gpu_post_balances[coroutine_id] == 0;
  }

  void track_post() { ++post_balances[running_coroutine_]; }
  void track_gpu_post() { ++gpu_post_balances[running_coroutine_]; }
  void set_current_coroutine(u32 id) { running_coroutine_ = id; }
  u32 current_coroutine_id() const { return running_coroutine_; }
  VamanaCoroutine& current_vamana_coroutine() const { return *vamana_coroutines[running_coroutine_]; }
  u64* coros_pointer_slot() const { return pointer_slots_[running_coroutine_]; }
  void set_active_sample(u32 coroutine_id, service::breakdown::Sample* sample) { active_samples_[coroutine_id] = sample; }
  service::breakdown::Sample* active_sample(u32 coroutine_id) const { return active_samples_[coroutine_id]; }
  service::breakdown::Sample* current_breakdown_sample() const { return active_samples_[running_coroutine_]; }
  void set_service_role(ServiceWorkerRole role) { service_role_ = role; }
  ServiceWorkerRole service_role() const { return service_role_; }
  bool is_query_worker() const { return service_role_ == ServiceWorkerRole::query; }
  bool is_insert_worker() const { return service_role_ == ServiceWorkerRole::insert; }

  /**
   * Poll GPU events for all coroutines.
   * Decrements gpu_post_balances when a coroutine's GPU work completes.
   */
  void poll_gpu_events();

public:
  const u32 node_id;
  vec<ibv_wc> send_wcs;

  BufferAllocator& buffer_allocator;  // global per compute node
  cache::Cache& cache;  // global per compute node

  SharedContext<ComputeThread>* ctx{nullptr};  // initialized by WorkerPool
  u32 ctx_tid{};

  // stores the k-NNs (node ids) of this thread's processed queries (for recall computation)
  hashmap_t<node_t, vec<node_t>> query_results;

  vec<u_ptr<VamanaCoroutine>> vamana_coroutines;
  vec<std::atomic<i32>> post_balances;  // per coroutine (RDMA)
  vec<std::atomic<i32>> gpu_post_balances;  // per coroutine (GPU)

  gpu::GpuBufferManager gpu_buffers;  // CUDA streams, events, staging buffers

  statistics::ThreadStatistics stats{};

private:
  const i32 max_send_queue_wr_;
  u32 running_coroutine_{};  // tracks the id of the currently running coroutine
  vec<u64*> pointer_slots_;  // memory region for a single pointer per coroutine
  vec<service::breakdown::Sample*> active_samples_;
  ServiceWorkerRole service_role_{ServiceWorkerRole::none};

  std::mt19937 generator_{std::random_device{}()};
  std::uniform_int_distribution<u32> dist_;
};
