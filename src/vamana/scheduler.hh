#pragma once

/**
 * Vamana scheduler: extends the HNSW scheduler pattern with GPU event polling.
 *
 * The main loop polls both:
 *  1. RDMA completion queue (thread->poll_cq())
 *  2. GPU events (thread->poll_gpu_events())
 *
 * A coroutine is ready to resume when both RDMA and GPU post_balances are zero.
 */

#include "coroutine.hh"
#include "router/query_router.hh"

// Forward declare Vamana class
namespace vamana {
template <class Distance>
class Vamana;
}

namespace vamana {

static VamanaCoroutine dummy_vamana_coroutine() {
    co_return;
}

/**
 * Schedule Vamana coroutines (search or insert).
 *
 * @tparam Distance     L2Distance or IPDistance
 * @tparam insert       If true, calls vamana.insert(); otherwise vamana.knn()
 * @param vamana_idx    Reference to the Vamana index
 * @param next_idx      Shared atomic counter for next unprocessed slot (inserts)
 * @param db            Database of vectors to insert/query
 * @param num_coroutines Number of coroutines per thread
 * @param thread        The compute thread running this scheduler
 * @param query_router  Optional query router for search mode
 */
template <class Distance, bool insert>
void schedule(Vamana<Distance>& vamana_idx,
              std::atomic<idx_t>& next_idx,
              io::Database<element_t>& db,
              u32 num_coroutines,
              const u_ptr<ComputeThread>& thread,
              query_router::QueryRouter<Distance>* query_router = nullptr) {
    const auto print_status = [&db](idx_t slot) {
        if (slot % (db.num_vectors_total / 10) == 0) {
            std::cerr << (insert ? "insert " : "query ")
                      << db.get_id(slot) << "/" << db.num_vectors_total << std::endl;
        }
    };

    if constexpr (not insert) {
        lib_assert(query_router, "invalid query_router");
    }

    // Initialize coroutines
    thread->vamana_coroutines.reserve(num_coroutines);
    for (u32 i = 0; i < num_coroutines; ++i) {
        thread->vamana_coroutines.emplace_back(
            std::make_unique<VamanaCoroutine>(dummy_vamana_coroutine()));
    }

    for (;;) {
        bool all_done = true;
        for (u32 coroutine_id = 0; coroutine_id < thread->vamana_coroutines.size(); ++coroutine_id) {
            auto& coroutine = *thread->vamana_coroutines[coroutine_id];

            // Poll both RDMA and GPU completions
            thread->poll_cq();
            thread->poll_gpu_events();

            // Recycle coroutine (assign new work)
            if (coroutine.handle.done()) {
                if constexpr (insert) {
                    const idx_t slot = next_idx.fetch_add(1);

                    if (slot < db.num_vectors_read) {
                        print_status(slot);
                        all_done = false;

                        coroutine.handle.destroy();
                        thread->set_current_coroutine(coroutine_id);

                        coroutine.handle = vamana_idx.insert(
                            db.get_id(slot), db.get_components(slot), thread).handle;
                    }
                } else {
                    if (not query_router->done || query_router->queue_size > 0) {
                        idx_t slot;
                        all_done = false;

                        if (query_router->query_queue.try_dequeue(slot)) {
                            query_router->queue_size.fetch_sub(1);
                            print_status(slot);

                            coroutine.handle.destroy();
                            thread->set_current_coroutine(coroutine_id);

                            coroutine.handle = vamana_idx.knn(
                                db.get_id(slot), db.get_components(slot), thread).handle;
                        }
                    }
                }

            // Resume coroutine if both RDMA and GPU are ready
            } else if (thread->is_ready(coroutine_id)) {
                all_done = false;

                thread->set_current_coroutine(coroutine_id);
                coroutine.handle.resume();

            // Keep polling
            } else {
                all_done = false;
            }
        }

        if (all_done) {
            break;
        }
    }

    for (const auto& coroutine : thread->vamana_coroutines) {
        lib_assert(coroutine->handle.done(), "vamana coroutine not done yet");
        coroutine->handle.destroy();
    }
}

}  // namespace vamana
