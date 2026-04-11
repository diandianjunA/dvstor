#pragma once

/**
 * Service-mode Vamana schedulers: continuously pull insert/query requests
 * from concurrent queues and schedule VamanaCoroutines.
 *
 * Mirrors service_scheduler.hh but uses VamanaCoroutine with GPU polling.
 */

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include "common/types.hh"
#include "coroutine.hh"
#include "http/service_types.hh"
#include "io/database.hh"
#include "vamana/vamana.hh"

namespace service {

static VamanaCoroutine dummy_vamana_service_coroutine() {
  co_return;
}

inline void reset_vamana_coroutine_state(VamanaCoroutine& coroutine) {
  coroutine.beam.clear();
  coroutine.visited_nodes.clear();
  coroutine.gpu_pending = false;
}

/**
 * Service-mode Vamana insert scheduler.
 * Runs continuously, pulling InsertRequests from the queue.
 * Polls both RDMA CQ and GPU events.
 */
template <class Distance>
void vamana_service_schedule_inserts(vamana::Vamana<Distance>& vamana_idx,
                                     InsertQueue& queue,
                                     std::atomic<bool>& shutdown,
                                     u32 num_coroutines,
                                     const u_ptr<ComputeThread>& thread,
                                     u32 dim,
                                     std::atomic<bool>& paused,
                                     std::atomic<u32>& idle_count) {
  thread->reset();
  thread->set_service_role(ServiceWorkerRole::insert);
  io::Database<element_t> staging;
  staging.allocate(dim, num_coroutines);

  vec<InsertRequest*> active_requests(num_coroutines, nullptr);

  thread->vamana_coroutines.reserve(num_coroutines);
  for (u32 i = 0; i < num_coroutines; ++i) {
    thread->vamana_coroutines.emplace_back(
        std::make_unique<VamanaCoroutine>(dummy_vamana_service_coroutine()));
    thread->set_current_coroutine(i);
    thread->vamana_coroutines.back()->handle.resume();
  }

  for (;;) {
    bool all_idle = true;

    for (u32 cid = 0; cid < thread->vamana_coroutines.size(); ++cid) {
      auto& coroutine = *thread->vamana_coroutines[cid];

      thread->poll_cq();
      thread->poll_gpu_events();

      if (coroutine.handle.done()) {
        if (active_requests[cid]) {
          if (active_requests[cid]->breakdown_sample) {
            active_requests[cid]->breakdown_sample->mark_finished(std::chrono::steady_clock::now(), thread->stats);
            thread->set_active_sample(cid, nullptr);
          }
          active_requests[cid]->result.set_value(true);
          active_requests[cid] = nullptr;
        }

        InsertRequest* req = nullptr;
        if (queue.try_dequeue(req)) {
          all_idle = false;
          const auto queue_wait = std::chrono::steady_clock::now() - req->enqueued_at;
          thread->stats.insert_queue_wait_ns +=
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(queue_wait).count());

          auto slot_components = staging.get_components(cid);
          std::copy(req->components.begin(), req->components.end(), slot_components.begin());
          staging.set_id(cid, req->id);
          active_requests[cid] = req;
          thread->set_active_sample(cid, req->breakdown_sample.get());
          if (req->breakdown_sample) {
            const auto now = std::chrono::steady_clock::now();
            req->breakdown_sample->enqueued_at = req->enqueued_at;
            req->breakdown_sample->mark_started(now, now, thread->stats);
          }

          coroutine.handle.destroy();
          reset_vamana_coroutine_state(coroutine);
          thread->set_current_coroutine(cid);
          coroutine.handle = vamana_idx.insert(req->id, slot_components, thread).handle;
        }

      } else if (thread->is_ready(cid)) {
        all_idle = false;
        thread->set_current_coroutine(cid);
        coroutine.handle.resume();
      } else {
        all_idle = false;
      }
    }

    if (all_idle) {
      if (shutdown.load(std::memory_order_relaxed)) {
        break;
      }
      if (paused.load(std::memory_order_relaxed)) {
        idle_count.fetch_add(1, std::memory_order_release);
        while (paused.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
        idle_count.fetch_sub(1, std::memory_order_release);
        continue;
      }
      std::this_thread::yield();
    }
  }

  for (const auto& coroutine : thread->vamana_coroutines) {
    coroutine->handle.destroy();
  }
}

/**
 * Service-mode Vamana query scheduler.
 * Runs continuously, pulling QueryRequests from the queue.
 * Polls both RDMA CQ and GPU events.
 */
template <class Distance>
void vamana_service_schedule_queries(vamana::Vamana<Distance>& vamana_idx,
                                     QueryQueue& queue,
                                     std::atomic<bool>& shutdown,
                                     u32 num_coroutines,
                                     const u_ptr<ComputeThread>& thread,
                                     u32 dim,
                                     std::atomic<bool>& paused,
                                     std::atomic<u32>& idle_count) {
  thread->reset();
  thread->set_service_role(ServiceWorkerRole::query);
  io::Database<element_t> staging;
  staging.allocate(dim, num_coroutines);

  vec<QueryRequest*> active_requests(num_coroutines, nullptr);
  vec<node_t> slot_ids(num_coroutines);
  for (u32 i = 0; i < num_coroutines; ++i) {
    slot_ids[i] = i;
  }

  thread->vamana_coroutines.reserve(num_coroutines);
  for (u32 i = 0; i < num_coroutines; ++i) {
    thread->vamana_coroutines.emplace_back(
        std::make_unique<VamanaCoroutine>(dummy_vamana_service_coroutine()));
    thread->set_current_coroutine(i);
    thread->vamana_coroutines.back()->handle.resume();
  }

  for (;;) {
    bool all_idle = true;

    for (u32 cid = 0; cid < thread->vamana_coroutines.size(); ++cid) {
      auto& coroutine = *thread->vamana_coroutines[cid];

      thread->poll_cq();
      thread->poll_gpu_events();

      if (coroutine.handle.done()) {
        if (active_requests[cid]) {
          if (active_requests[cid]->breakdown_sample) {
            active_requests[cid]->breakdown_sample->mark_finished(std::chrono::steady_clock::now(), thread->stats);
            thread->set_active_sample(cid, nullptr);
          }
          node_t q_id = slot_ids[cid];
          auto it = thread->query_results.find(q_id);
          if (it != thread->query_results.end()) {
            active_requests[cid]->result.set_value(std::move(it->second));
            thread->query_results.erase(it);
          } else {
            active_requests[cid]->result.set_value(vec<node_t>{});
          }
          active_requests[cid] = nullptr;
        }

        QueryRequest* req = nullptr;
        if (queue.try_dequeue(req)) {
          all_idle = false;
          const auto queue_wait = std::chrono::steady_clock::now() - req->enqueued_at;
          thread->stats.query_queue_wait_ns +=
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(queue_wait).count());

          auto slot_components = staging.get_components(cid);
          std::copy(req->components.begin(), req->components.end(), slot_components.begin());
          active_requests[cid] = req;
          thread->set_active_sample(cid, req->breakdown_sample.get());
          if (req->breakdown_sample) {
            const auto now = std::chrono::steady_clock::now();
            req->breakdown_sample->enqueued_at = req->enqueued_at;
            req->breakdown_sample->mark_started(now, now, thread->stats);
          }

          coroutine.handle.destroy();
          reset_vamana_coroutine_state(coroutine);
          thread->set_current_coroutine(cid);
          coroutine.handle = vamana_idx.knn(slot_ids[cid], slot_components, thread).handle;
        }

      } else if (thread->is_ready(cid)) {
        all_idle = false;
        thread->set_current_coroutine(cid);
        coroutine.handle.resume();
      } else {
        all_idle = false;
      }
    }

    if (all_idle) {
      if (shutdown.load(std::memory_order_relaxed)) {
        break;
      }
      if (paused.load(std::memory_order_relaxed)) {
        idle_count.fetch_add(1, std::memory_order_release);
        while (paused.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
        idle_count.fetch_sub(1, std::memory_order_release);
        continue;
      }
      std::this_thread::yield();
    }
  }

  for (const auto& coroutine : thread->vamana_coroutines) {
    coroutine->handle.destroy();
  }
}

}  // namespace service
