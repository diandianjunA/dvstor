#pragma once

/**
 * Vamana Index: GPU-accelerated beam-search graph index with RDMA disaggregated memory.
 *
 * Replaces HNSW with a single-layer directed graph using:
 *  - Beam search (instead of multi-layer greedy descent)
 *  - RobustPrune (alpha-based diversity pruning instead of HNSW heuristic)
 *  - Exact L2 distances for search (GPU-accelerated)
 *  - Full-precision L2 distances for insert/prune (GPU-accelerated)
 *  - Coroutine-based RDMA overlap (from DVSTOR baseline)
 */

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#include "cache/cache.hh"
#include "common/constants.hh"
#include "common/debug.hh"
#include "common/types.hh"
#include "compute_thread.hh"
#include "coroutine.hh"
#include "gpu/gpu_awaitable.hh"
#include "gpu/gpu_kernel_launcher.hh"
#include "rdma/vamana_rdma_operations.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_neighborlist.hh"
#include "vamana/vamana_node.hh"

namespace vamana {

constexpr u32 kRabitqSearchBeamSlack = 64;

template <class Distance>
class Vamana {
public:
    Vamana(u32 R, u32 beam_width, u32 beam_width_construction, f64 alpha,
           u32 k, u32 rabitq_bits, u32 dim, bool use_cache, bool use_rabitq_search)
        : R_(R),
          beam_width_(beam_width),
          beam_width_construction_(beam_width_construction),
          alpha_(static_cast<f32>(alpha)),
          k_(k),
          rabitq_bits_(rabitq_bits),
          dim_(dim),
          use_cache_(use_cache),
          use_rabitq_search_(use_rabitq_search) {
        lib_assert(beam_width_ >= k_, "beam_width must be >= k");
        VamanaNode::init_static_storage(dim, R, rabitq_bits);
    }

    // =========================================================================
    // Search (knn)
    // =========================================================================

    VamanaCoroutine knn(node_t q_id, const span<element_t> components,
                        const u_ptr<ComputeThread>& thread) const {
        dbg::print(dbg::stream{} << "T" << thread->get_id() << " queries " << q_id << "\n");
        ++thread->stats.processed;
        ++thread->stats.processed_queries;

        auto& coro_state = thread->current_vamana_coroutine();
        auto& beam = coro_state.beam;
        auto& visited = coro_state.visited_nodes;
        auto& gpu = thread->gpu_buffers;
        const u32 coro_id = thread->current_coroutine_id();  // current coroutine id managed by scheduler
        auto& gs = gpu.state(coro_id);
        const bool use_gpudirect_candidate_rdma =
            gpu.gpudirect_candidate_ready() && gs.d_candidate_vecs_rdma_registered;
        const bool use_gpudirect_rabitq_rdma =
            gpu.gpudirect_rabitq_ready() && gs.d_rabitq_vecs_rdma_registered;

        lib_assert(!use_rabitq_search_ || gpu.rabitq_ready(),
                   "rabitq_gpu search requested before RaBitQ artifacts were loaded");

        // Read medoid
        const auto t_medoid_ptr_start = std::chrono::steady_clock::now();
        RemotePtr medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_medoid_ptr, t_medoid_ptr_start);

        s_ptr<VamanaNode> medoid_node;
        {
            const auto t_cache_start = std::chrono::steady_clock::now();
            auto coro = cache_lookup(medoid_ptr, medoid_node, thread, true);
            while (!coro.handle.done()) {
                co_await std::suspend_always{};
                coro.handle.resume();
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_cache_lookup, t_cache_start);
        }

        // Upload query vector to GPU once.
        const auto t_query_h2d = std::chrono::steady_clock::now();
        std::memcpy(gs.h_query, components.data(), dim_ * sizeof(float));
        cudaMemcpyAsync(gs.d_query, gs.h_query, dim_ * sizeof(float),
                        cudaMemcpyHostToDevice, gs.stream);
        track_query_h2d(thread, dim_ * sizeof(float));
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_query_h2d, t_query_h2d);

        if (use_rabitq_search_) {
            const auto t_gpu_prepare = std::chrono::steady_clock::now();
            gpu::launch_rabitq_query_prepare(
                gs.stream, gs.event,
                gpu.cublas_handle(),
                gs.d_query,
                gpu.d_rotation_matrix(),
                gpu.d_centroid(),
                gs.d_rot_query,
                gs.d_query_factor,
                dim_, rabitq_bits_);
            ++thread->stats.query_rabitq_kernels;
            co_await gpu::GpuAwaitable{thread.get()};
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_query_prepare, t_gpu_prepare);
        }

        // Initialize beam with medoid (exact L2 distance)
        distance_t medoid_dist = Distance::dist(components, medoid_node->components(), VamanaNode::DIM);
        ++thread->stats.distcomps;
        ++thread->stats.query_distcomps;

        beam.clear();
        beam.push_back({medoid_ptr, medoid_dist, false});
        visited.clear();
        visited.insert(medoid_ptr);

        const u32 search_beam_capacity =
            use_rabitq_search_ ? (beam_width_ + kRabitqSearchBeamSlack) : beam_width_;

        // Beam search loop: Jasper-style RaBitQ search if enabled, exact GPU otherwise.
        while (true) {
            // Find closest unexpanded candidate
            const auto t_select = std::chrono::steady_clock::now();
            i32 best_idx = -1;
            distance_t best_dist = std::numeric_limits<distance_t>::max();
            for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
                if (!beam[i].expanded && beam[i].distance < best_dist) {
                    best_dist = beam[i].distance;
                    best_idx = i;
                }
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_select, t_select);
            if (best_idx < 0) break;  // all expanded

            beam[best_idx].expanded = true;

            // Read neighbor list of best candidate, using the query-side neighbor cache when enabled.
            u8 neighbor_count = 0;
            const RemotePtr* neighbor_ptrs = nullptr;
            s_ptr<VamanaNeighborlist> nlist;
            {
                const auto t_neighbor_lookup = std::chrono::steady_clock::now();
                if (thread->neighbor_cache.lookup(beam[best_idx].rptr, neighbor_count, neighbor_ptrs)) {
                    ++thread->stats.neighbor_cache_hits;
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_cache_lookup,
                                              t_neighbor_lookup);
                } else {
                    ++thread->stats.neighbor_cache_misses;
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_cache_lookup,
                                              t_neighbor_lookup);
                    const auto t_neighbor_fetch = std::chrono::steady_clock::now();
                    nlist = co_await rdma::vamana::read_vamana_neighbors(beam[best_idx].rptr, thread);
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_neighbor_fetch,
                                              t_neighbor_fetch);
                    thread->neighbor_cache.insert(beam[best_idx].rptr, nlist->view());
                    neighbor_count = nlist->num_neighbors();
                    neighbor_ptrs = nlist->view().data();
                }
            }
            ++thread->stats.visited_neighborlists;

            // Filter unvisited neighbors
            const auto t_filter = std::chrono::steady_clock::now();
            vec<RemotePtr> unvisited;
            for (u32 neighbor_idx = 0; neighbor_idx < neighbor_count; ++neighbor_idx) {
                const RemotePtr& n_ptr = neighbor_ptrs[neighbor_idx];
                if (n_ptr.is_null()) continue;
                if (!visited.contains(n_ptr)) {
                    visited.insert(n_ptr);
                    unvisited.push_back(n_ptr);
                }
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_filter, t_filter);

            if (unvisited.empty()) continue;

            const u32 n_batch = unvisited.size();
            if (use_rabitq_search_) {
                auto& rabitq_cache = gpu.rabitq_cache();
                bool used_gpu_cache = false;
                if (rabitq_cache.enabled()) {
                    std::vector<u32> miss_indices;
                    std::vector<u32> miss_slots;
                    std::vector<u64> miss_addrs;
                    const auto resolved = rabitq_cache.resolve_batch(
                        unvisited.data(), n_batch, gs.h_cache_slot_ids, miss_indices, miss_slots, miss_addrs);

                    if (resolved.ok) {
                        used_gpu_cache = true;
                        thread->stats.gpu_rabitq_cache_hits += resolved.hit_count;
                        thread->stats.gpu_rabitq_cache_misses += n_batch - resolved.hit_count;
                        thread->stats.gpu_rabitq_cache_duplicate_fills += resolved.duplicate_loading_count;

                        if (!miss_indices.empty()) {
                            std::vector<RemotePtr> miss_ptrs;
                            miss_ptrs.reserve(miss_indices.size());
                            for (u32 idx : miss_indices) {
                                miss_ptrs.push_back(unvisited[idx]);
                            }
                            vec<rdma::vamana::BatchReadDestination> destinations;
                            destinations.reserve(miss_ptrs.size());
                            for (u32 i = 0; i < miss_ptrs.size(); ++i) {
                                destinations.push_back({miss_addrs[i], rabitq_cache.lkey(), nullptr, true});
                            }
                            const auto t_rabitq_fetch = std::chrono::steady_clock::now();
                            co_await rdma::vamana::batch_read_rabitq(miss_ptrs, thread, &destinations);
                            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_rabitq_fetch,
                                                      t_rabitq_fetch);
                            rabitq_cache.publish_batch(miss_slots);
                            thread->stats.gpu_rabitq_cache_fills += miss_ptrs.size();
                            thread->stats.gpu_rabitq_cache_fill_bytes +=
                                static_cast<u64>(miss_ptrs.size()) * VamanaNode::RABITQ_SIZE;
                        }

                        rabitq_cache.acquire_slots(gs.h_cache_slot_ids, n_batch);
                        gpu::launch_gather_cached_rabitq(
                            gs.stream,
                            rabitq_cache.base(),
                            gs.d_cache_slot_ids,
                            gs.d_rabitq_vecs,
                            n_batch,
                            rabitq_cache.stride());
                        const auto t_gpu_distance = std::chrono::steady_clock::now();
                        gpu::launch_batch_rabitq_distances(
                            gs.stream, gs.event,
                            gs.d_rot_query,
                            gs.d_query_factor,
                            gs.d_rabitq_vecs,
                            gs.d_distances,
                            n_batch, dim_, rabitq_bits_);
                        ++thread->stats.query_rabitq_kernels;
                        co_await gpu::GpuAwaitable{thread.get()};
                        rabitq_cache.release_slots(gs.h_cache_slot_ids, n_batch);
                        add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_query_distance,
                                                  t_gpu_distance);
                    } else {
                        ++thread->stats.gpu_rabitq_cache_fallback_batches;
                    }
                }

                if (!used_gpu_cache) {
                    const auto t_rabitq_fetch = std::chrono::steady_clock::now();
                    auto rabitq_read = co_await rdma::vamana::batch_read_rabitq(
                        unvisited, thread,
                        use_gpudirect_rabitq_rdma ? gs.d_rabitq_vecs : nullptr,
                        use_gpudirect_rabitq_rdma ? gs.d_rabitq_vecs_lkey : 0);
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_rabitq_fetch,
                                              t_rabitq_fetch);

                    if (rabitq_read.direct_to_gpu) {
                        thread->stats.query_rdma_to_staging_bytes +=
                            static_cast<u64>(n_batch) * VamanaNode::RABITQ_SIZE;
                    } else {
                        thread->stats.query_host_staging_fallback_bytes +=
                            static_cast<u64>(n_batch) * VamanaNode::RABITQ_SIZE;
                        const auto t_stage_candidates = std::chrono::steady_clock::now();
                        for (u32 i = 0; i < n_batch; ++i) {
                            std::memcpy(gs.h_rabitq_vecs + i * VamanaNode::RABITQ_SIZE,
                                        rabitq_read.host_buffers[i],
                                        VamanaNode::RABITQ_SIZE);
                            thread->buffer_allocator.free_buffer(rabitq_read.host_buffers[i], VamanaNode::RABITQ_SIZE);
                        }
                        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_stage_candidates,
                                                  t_stage_candidates);

                        const auto t_rabitq_h2d = std::chrono::steady_clock::now();
                        cudaMemcpyAsync(gs.d_rabitq_vecs, gs.h_rabitq_vecs,
                                        n_batch * VamanaNode::RABITQ_SIZE,
                                        cudaMemcpyHostToDevice, gs.stream);
                        track_query_h2d(thread, static_cast<u64>(n_batch) * VamanaNode::RABITQ_SIZE);
                        add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_rabitq_h2d,
                                                  t_rabitq_h2d);
                    }

                    const auto t_gpu_distance = std::chrono::steady_clock::now();
                    gpu::launch_batch_rabitq_distances(
                        gs.stream, gs.event,
                        gs.d_rot_query,
                        gs.d_query_factor,
                        gs.d_rabitq_vecs,
                        gs.d_distances,
                        n_batch, dim_, rabitq_bits_);
                    ++thread->stats.query_rabitq_kernels;
                    co_await gpu::GpuAwaitable{thread.get()};
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_query_distance,
                                              t_gpu_distance);
                }
            } else {
                const auto t_vector_fetch = std::chrono::steady_clock::now();
                auto vec_read = co_await rdma::vamana::batch_read_vectors(
                    unvisited, thread,
                    use_gpudirect_candidate_rdma ? gs.d_candidate_vecs : nullptr,
                    use_gpudirect_candidate_rdma ? gs.d_candidate_vecs_lkey : 0);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_vector_fetch, t_vector_fetch);

                if (vec_read.direct_to_gpu) {
                    thread->stats.query_rdma_to_staging_bytes += static_cast<u64>(n_batch) * dim_ * sizeof(float);
                } else {
                    thread->stats.query_host_staging_fallback_bytes += static_cast<u64>(n_batch) * dim_ * sizeof(float);
                    const auto t_stage_candidates = std::chrono::steady_clock::now();
                    for (u32 i = 0; i < n_batch; ++i) {
                        std::memcpy(gs.h_candidate_vecs + i * dim_,
                                    reinterpret_cast<float*>(vec_read.host_buffers[i]),
                                    dim_ * sizeof(float));
                        thread->buffer_allocator.free_buffer(vec_read.host_buffers[i], dim_ * sizeof(element_t));
                    }
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_stage_candidates,
                                              t_stage_candidates);

                    const auto t_candidate_h2d = std::chrono::steady_clock::now();
                    cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                                    n_batch * dim_ * sizeof(float),
                                    cudaMemcpyHostToDevice, gs.stream);
                    track_query_h2d(thread, static_cast<u64>(n_batch) * dim_ * sizeof(float));
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_candidate_h2d,
                                              t_candidate_h2d);
                }

                const auto t_gpu_distance = std::chrono::steady_clock::now();
                gpu::launch_batch_l2_distances(
                    gs.stream, gs.event,
                    gs.d_query, gs.d_candidate_vecs,
                    gs.d_distances, n_batch, dim_);
                co_await gpu::GpuAwaitable{thread.get()};
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_query_distance, t_gpu_distance);
            }

            const auto t_distance_d2h = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                            n_batch * sizeof(float),
                            cudaMemcpyDeviceToHost, gs.stream);
            track_query_d2h(thread, n_batch * sizeof(float));
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_distance_d2h, t_distance_d2h);

            const auto t_beam_update = std::chrono::steady_clock::now();
            for (u32 i = 0; i < n_batch; ++i) {
                distance_t dist = gs.h_distances[i];
                ++thread->stats.distcomps;
                ++thread->stats.query_distcomps;
                insert_into_beam(beam, unvisited[i], dist, search_beam_capacity);
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_beam_update, t_beam_update);
            continue;

        }

        if (use_rabitq_search_ && !beam.empty()) {
            vec<RemotePtr> rerank_ptrs;
            rerank_ptrs.reserve(beam.size());
            const auto t_rerank_collect = std::chrono::steady_clock::now();
            for (const auto& entry : beam) {
                rerank_ptrs.push_back(entry.rptr);
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_rerank_collect, t_rerank_collect);

            const u32 n_rerank = static_cast<u32>(rerank_ptrs.size());
            const auto t_rerank_fetch = std::chrono::steady_clock::now();
            auto rerank_read = co_await rdma::vamana::batch_read_vectors(
                rerank_ptrs, thread,
                use_gpudirect_candidate_rdma ? gs.d_candidate_vecs : nullptr,
                use_gpudirect_candidate_rdma ? gs.d_candidate_vecs_lkey : 0);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_rerank_fetch, t_rerank_fetch);

            if (rerank_read.direct_to_gpu) {
                thread->stats.query_rdma_to_staging_bytes += static_cast<u64>(n_rerank) * dim_ * sizeof(float);
            } else {
                thread->stats.query_host_staging_fallback_bytes += static_cast<u64>(n_rerank) * dim_ * sizeof(float);
                const auto t_rerank_prepare = std::chrono::steady_clock::now();
                for (u32 i = 0; i < n_rerank; ++i) {
                    std::memcpy(gs.h_candidate_vecs + i * dim_,
                                reinterpret_cast<float*>(rerank_read.host_buffers[i]),
                                dim_ * sizeof(float));
                    thread->buffer_allocator.free_buffer(rerank_read.host_buffers[i], dim_ * sizeof(element_t));
                }
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_rerank_prepare,
                                          t_rerank_prepare);

                const auto t_rerank_h2d = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                                n_rerank * dim_ * sizeof(float),
                                cudaMemcpyHostToDevice, gs.stream);
                track_query_h2d(thread, static_cast<u64>(n_rerank) * dim_ * sizeof(float));
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_rerank_h2d,
                                          t_rerank_h2d);
            }

            const auto t_gpu_rerank = std::chrono::steady_clock::now();
            gpu::launch_batch_l2_distances(
                gs.stream, gs.event,
                gs.d_query, gs.d_candidate_vecs,
                gs.d_distances, n_rerank, dim_);
            ++thread->stats.query_exact_reranks;
            co_await gpu::GpuAwaitable{thread.get()};
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_query_rerank, t_gpu_rerank);

            const auto t_rerank_d2h = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                            n_rerank * sizeof(float),
                            cudaMemcpyDeviceToHost, gs.stream);
            track_query_d2h(thread, n_rerank * sizeof(float));
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_rerank_d2h, t_rerank_d2h);

            const auto t_rerank_update = std::chrono::steady_clock::now();
            for (u32 i = 0; i < n_rerank; ++i) {
                beam[i].distance = gs.h_distances[i];
                ++thread->stats.distcomps;
                ++thread->stats.query_distcomps;
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_rerank_update, t_rerank_update);
        }

        const auto t_beam_sort = std::chrono::steady_clock::now();
        std::sort(beam.begin(), beam.end(),
                  [](const auto& a, const auto& b) { return a.distance < b.distance; });
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_beam_sort, t_beam_sort);

        const auto t_finalize = std::chrono::steady_clock::now();
        auto& results = thread->query_results[q_id];
        results.clear();
        u32 count = std::min(k_, static_cast<u32>(beam.size()));

        // We need to resolve node IDs — read the nodes for top-k
        const auto t_result_ids = std::chrono::steady_clock::now();
        for (u32 i = 0; i < count; ++i) {
            s_ptr<VamanaNode> node;
            const auto t_cache_lookup = std::chrono::steady_clock::now();
            auto coro = cache_lookup(beam[i].rptr, node, thread, true);
            while (!coro.handle.done()) {
                co_await std::suspend_always{};
                coro.handle.resume();
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_cache_lookup, t_cache_lookup);
            results.push_back(node->id());
        }
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_result_ids, t_result_ids);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_finalize, t_finalize);

        beam.clear();
        visited.clear();
    }

    // =========================================================================
    // Insert
    // =========================================================================

    VamanaCoroutine insert(node_t id, const span<element_t> components,
                           const u_ptr<ComputeThread>& thread) {
        dbg::print(dbg::stream{} << "T" << thread->get_id() << " inserts " << id << "\n");
        ++thread->stats.processed;
        ++thread->stats.processed_inserts;

        auto& coro_state = thread->current_vamana_coroutine();
        auto& beam = coro_state.beam;
        auto& visited = coro_state.visited_nodes;
        auto& gpu = thread->gpu_buffers;
        const u32 coro_id = thread->current_coroutine_id();
        auto& gs = gpu.state(coro_id);
        const bool use_gpudirect_candidate_rdma =
            gpu.gpudirect_candidate_ready() && gs.d_candidate_vecs_rdma_registered;

        lib_assert(!use_rabitq_search_ || gpu.rabitq_ready(),
                   "rabitq_gpu insert requested before RaBitQ artifacts were loaded");

        // Read medoid
        const auto t_medoid_ptr = std::chrono::steady_clock::now();
        RemotePtr medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_medoid_ptr, t_medoid_ptr);

        // Handle first insert (empty index)
        if (medoid_ptr.is_null()) {
            const auto t_alloc = std::chrono::steady_clock::now();
            RemotePtr new_ptr = co_await rdma::vamana::allocate_vamana_node(thread);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_alloc, t_alloc);

            // Quantize vector with GPU
            const auto t_first_query_h2d = std::chrono::steady_clock::now();
            std::memcpy(gs.h_query, components.data(), dim_ * sizeof(float));
            cudaMemcpyAsync(gs.d_query, gs.h_query, dim_ * sizeof(float),
                           cudaMemcpyHostToDevice, gs.stream);
            track_build_h2d(thread, dim_ * sizeof(float));
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_insert_query_h2d, t_first_query_h2d);

            // Allocate temp device buffer for RaBitQ output
            u32 rabitq_data_size = VamanaNode::RABITQ_SIZE;
            uint8_t* d_rabitq_out = reinterpret_cast<uint8_t*>(gs.d_rabitq_vecs);  // reuse buffer

            const auto t_quantize = std::chrono::steady_clock::now();
            gpu::launch_rabitq_quantize_single(
                gs.stream, gs.event,
                gpu.cublas_handle(),
                gs.d_query,
                gpu.d_rotation_matrix(),
                gpu.d_centroid(),
                d_rabitq_out,
                dim_, rabitq_bits_, gpu.t_const());
            co_await gpu::GpuAwaitable{thread.get()};
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_insert_quantize, t_quantize);

            // Copy RaBitQ data back
            const auto t_quantize_d2h = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gs.h_rabitq_vecs, d_rabitq_out, rabitq_data_size,
                           cudaMemcpyDeviceToHost, gs.stream);
            track_build_d2h(thread, rabitq_data_size);
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_quantize_d2h, t_quantize_d2h);

            // Write node with no neighbors
            vec<RemotePtr> empty_neighbors;
            const auto t_write = std::chrono::steady_clock::now();
            s_ptr<VamanaNode> new_node = co_await rdma::vamana::write_vamana_node(
                new_ptr, id, components, gs.h_rabitq_vecs, span<RemotePtr>{},
                0, false, false, thread);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_new_node_write, t_write);

            // Try to set as medoid
            const auto t_medoid_update = std::chrono::steady_clock::now();
            RemotePtr old_medoid = co_await rdma::vamana::swap_medoid_ptr(
                RemotePtr{}, new_ptr, thread);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_medoid_update, t_medoid_update);
            if (old_medoid.is_null()) {
                // Success — we set the medoid
                const auto t_header_write = std::chrono::steady_clock::now();
                co_await rdma::vamana::write_vamana_header(new_ptr, true, false, false, thread);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_header_write, t_header_write);
                co_return;
            }
            // Another thread won the race — fall through to normal insert
            medoid_ptr = old_medoid;
        }

        // Phase 1: Beam search for candidate neighbors
        s_ptr<VamanaNode> medoid_node;
        {
            const auto t_cache_start = std::chrono::steady_clock::now();
            auto coro = cache_lookup(medoid_ptr, medoid_node, thread, true);
            while (!coro.handle.done()) {
                co_await std::suspend_always{};
                coro.handle.resume();
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_cache_lookup, t_cache_start);
        }

        const auto t_insert_init = std::chrono::steady_clock::now();
        distance_t medoid_dist = Distance::dist(components, medoid_node->components(), VamanaNode::DIM);
        ++thread->stats.distcomps;
        ++thread->stats.build_distcomps;

        beam.clear();
        beam.push_back({medoid_ptr, medoid_dist, false});
        visited.clear();
        visited.insert(medoid_ptr);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_init, t_insert_init);

        // Beam search using full L2 distances (exact, not RaBitQ)
        // Upload query vector once
        const auto t_insert_query_h2d = std::chrono::steady_clock::now();
        std::memcpy(gs.h_query, components.data(), dim_ * sizeof(float));
        cudaMemcpyAsync(gs.d_query, gs.h_query, dim_ * sizeof(float),
                       cudaMemcpyHostToDevice, gs.stream);
        track_build_h2d(thread, dim_ * sizeof(float));
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_insert_query_h2d, t_insert_query_h2d);

        while (true) {
            const auto t_select = std::chrono::steady_clock::now();
            i32 best_idx = -1;
            distance_t best_dist = std::numeric_limits<distance_t>::max();
            for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
                if (!beam[i].expanded && beam[i].distance < best_dist) {
                    best_dist = beam[i].distance;
                    best_idx = i;
                }
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_select, t_select);
            if (best_idx < 0) break;

            beam[best_idx].expanded = true;

            const auto t_neighbor_fetch = std::chrono::steady_clock::now();
            s_ptr<VamanaNeighborlist> nlist =
                co_await rdma::vamana::read_vamana_neighbors(beam[best_idx].rptr, thread);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_neighbor_fetch, t_neighbor_fetch);
            ++thread->stats.visited_neighborlists;

            const auto t_filter = std::chrono::steady_clock::now();
            vec<RemotePtr> unvisited;
            for (const RemotePtr& n_ptr : nlist->view()) {
                if (n_ptr.is_null()) continue;
                if (!visited.contains(n_ptr)) {
                    visited.insert(n_ptr);
                    unvisited.push_back(n_ptr);
                }
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_filter, t_filter);

            if (unvisited.empty()) continue;

            // Batch read full vectors for exact distance
            const u32 n_batch = unvisited.size();
            const auto t_vector_fetch = std::chrono::steady_clock::now();
            auto vec_read = co_await rdma::vamana::batch_read_vectors(
                unvisited, thread,
                use_gpudirect_candidate_rdma ? gs.d_candidate_vecs : nullptr,
                use_gpudirect_candidate_rdma ? gs.d_candidate_vecs_lkey : 0);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_vector_fetch, t_vector_fetch);

            if (!vec_read.direct_to_gpu) {
                const auto t_stage_candidates = std::chrono::steady_clock::now();
                for (u32 i = 0; i < n_batch; ++i) {
                    std::memcpy(gs.h_candidate_vecs + i * dim_,
                                reinterpret_cast<float*>(vec_read.host_buffers[i]),
                                dim_ * sizeof(float));
                    thread->buffer_allocator.free_buffer(vec_read.host_buffers[i], dim_ * sizeof(element_t));
                }
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_stage_candidates, t_stage_candidates);

                const auto t_candidate_h2d = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                                n_batch * dim_ * sizeof(float),
                                cudaMemcpyHostToDevice, gs.stream);
                track_build_h2d(thread, n_batch * dim_ * sizeof(float));
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_candidate_h2d, t_candidate_h2d);
            }

            const auto t_gpu_distance = std::chrono::steady_clock::now();
            gpu::launch_batch_l2_distances(
                gs.stream, gs.event,
                gs.d_query, gs.d_candidate_vecs,
                gs.d_distances, n_batch, dim_);
            ++thread->stats.build_l2_kernels;
            co_await gpu::GpuAwaitable{thread.get()};
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_insert_distance, t_gpu_distance);

            const auto t_dist_d2h = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                           n_batch * sizeof(float),
                           cudaMemcpyDeviceToHost, gs.stream);
            track_build_d2h(thread, n_batch * sizeof(float));
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_distance_d2h, t_dist_d2h);

            const auto t_beam_update = std::chrono::steady_clock::now();
            for (u32 i = 0; i < n_batch; ++i) {
                ++thread->stats.distcomps;
                ++thread->stats.build_distcomps;
                insert_into_beam(beam, unvisited[i], gs.h_distances[i], beam_width_construction_);
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_beam_update, t_beam_update);

            // (bump allocator; no individual free)
        }

        // Phase 2: GPU RobustPrune to select R neighbors
        const auto t_preprune_sort = std::chrono::steady_clock::now();
        std::sort(beam.begin(), beam.end(),
                  [](const auto& a, const auto& b) { return a.distance < b.distance; });
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_preprune_sort, t_preprune_sort);

        const u32 n_candidates = beam.size();

        // Collect candidate RemotePtrs and distances (already sorted)
        const auto t_candidate_collect = std::chrono::steady_clock::now();
        vec<RemotePtr> candidate_rptrs;
        candidate_rptrs.reserve(n_candidates);
        for (auto& entry : beam) {
            candidate_rptrs.push_back(entry.rptr);
        }
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_candidate_collect, t_candidate_collect);

        // Batch read full vectors for all candidates (for RobustPrune)
        const auto t_candidate_fetch = std::chrono::steady_clock::now();
        auto candidate_read = co_await rdma::vamana::batch_read_vectors(
            candidate_rptrs,
            thread,
            use_gpudirect_candidate_rdma ? gs.d_candidate_vecs : nullptr,
            use_gpudirect_candidate_rdma ? gs.d_candidate_vecs_lkey : 0);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_candidate_fetch, t_candidate_fetch);

        // Stage to GPU: candidate vectors + distances
        const auto t_candidate_sort = std::chrono::steady_clock::now();
        if (!candidate_read.direct_to_gpu) {
            for (u32 i = 0; i < n_candidates; ++i) {
                std::memcpy(gs.h_candidate_vecs + i * dim_,
                           reinterpret_cast<float*>(candidate_read.host_buffers[i]),
                           dim_ * sizeof(float));
                thread->buffer_allocator.free_buffer(candidate_read.host_buffers[i], dim_ * sizeof(element_t));
            }
        }
        for (u32 i = 0; i < n_candidates; ++i) {
            gs.h_candidate_dists[i] = beam[i].distance;
        }
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_candidate_sort, t_candidate_sort);

        const auto t_prune_prepare = std::chrono::steady_clock::now();
        const auto t_prune_h2d = std::chrono::steady_clock::now();
        if (!candidate_read.direct_to_gpu) {
            cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                           n_candidates * dim_ * sizeof(float),
                           cudaMemcpyHostToDevice, gs.stream);
        }
        cudaMemcpyAsync(gs.d_candidate_dists, gs.h_candidate_dists,
                       n_candidates * sizeof(float),
                       cudaMemcpyHostToDevice, gs.stream);
        const size_t prune_h2d_bytes =
            (candidate_read.direct_to_gpu ? 0 : (n_candidates * dim_ * sizeof(float))) + n_candidates * sizeof(float);
        track_build_h2d(thread, prune_h2d_bytes);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_prune_prepare, t_prune_prepare);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_prune_h2d, t_prune_h2d);

        const auto t_gpu_prune = std::chrono::steady_clock::now();
        gpu::launch_robust_prune(
            gs.stream, gs.event,
            gs.d_query,  // source vector (the new node being inserted)
            gs.d_candidate_vecs,
            gs.d_candidate_dists,
            nullptr,
            n_candidates, dim_, alpha_, R_,
            gs.d_pruned_indices, gs.d_pruned_count);
        ++thread->stats.build_prune_kernels;
        co_await gpu::GpuAwaitable{thread.get()};
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_insert_prune, t_gpu_prune);

        const auto t_prune_d2h = std::chrono::steady_clock::now();
        cudaMemcpyAsync(gs.h_pruned_indices, gs.d_pruned_indices,
                       R_ * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost, gs.stream);
        cudaMemcpyAsync(gs.h_pruned_count, gs.d_pruned_count,
                       sizeof(uint32_t),
                       cudaMemcpyDeviceToHost, gs.stream);
        track_build_d2h(thread, R_ * sizeof(uint32_t) + sizeof(uint32_t));
        cudaStreamSynchronize(gs.stream);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_prune_d2h, t_prune_d2h);

        const u32 pruned_count = *gs.h_pruned_count;

        // Map pruned indices back to RemotePtrs
        const auto t_pruned_collect = std::chrono::steady_clock::now();
        vec<RemotePtr> selected_neighbors;
        selected_neighbors.reserve(pruned_count);
        for (u32 i = 0; i < pruned_count; ++i) {
            u32 idx = gs.h_pruned_indices[i];
            selected_neighbors.push_back(candidate_rptrs[idx]);
        }
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_pruned_neighbor_collect, t_pruned_collect);

        // Free candidate vector buffers (bump allocator; no individual free)

        // Phase 3: Quantize new vector with RaBitQ
        const auto t_quantize_prepare = std::chrono::steady_clock::now();
        uint8_t* d_rabitq_out = reinterpret_cast<uint8_t*>(gs.d_rabitq_vecs);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_quantize_prepare, t_quantize_prepare);
        const auto t_quantize = std::chrono::steady_clock::now();
        gpu::launch_rabitq_quantize_single(
            gs.stream, gs.event,
            gpu.cublas_handle(),
            gs.d_query,
            gpu.d_rotation_matrix(),
            gpu.d_centroid(),
            d_rabitq_out,
            dim_, rabitq_bits_, gpu.t_const());
        co_await gpu::GpuAwaitable{thread.get()};
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_insert_quantize, t_quantize);

        u32 rabitq_data_size = VamanaNode::RABITQ_SIZE;
        const auto t_insert_quantize_d2h = std::chrono::steady_clock::now();
        cudaMemcpyAsync(gs.h_rabitq_vecs, d_rabitq_out, rabitq_data_size,
                       cudaMemcpyDeviceToHost, gs.stream);
        track_build_d2h(thread, rabitq_data_size);
        cudaStreamSynchronize(gs.stream);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_quantize_d2h, t_insert_quantize_d2h);

        // Phase 4: Allocate and write new node
        const auto t_alloc = std::chrono::steady_clock::now();
        RemotePtr new_ptr = co_await rdma::vamana::allocate_vamana_node(thread);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_alloc, t_alloc);

        const auto t_new_write = std::chrono::steady_clock::now();
        s_ptr<VamanaNode> new_node = co_await rdma::vamana::write_vamana_node(
            new_ptr, id, components, gs.h_rabitq_vecs,
            span<RemotePtr>{selected_neighbors.data(), selected_neighbors.size()},
            static_cast<u8>(pruned_count), false, false, thread);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_new_node_write, t_new_write);

        // Phase 5: Update reverse edges (bidirectional connectivity)
        for (const RemotePtr& neighbor_ptr : selected_neighbors) {
            // Lock the neighbor
            const auto t_neighbor_node_read = std::chrono::steady_clock::now();
            s_ptr<VamanaNode> neighbor_node =
                co_await rdma::vamana::read_vamana_node(neighbor_ptr, thread);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_neighbor_node_read, t_neighbor_node_read);
            {
                const auto t_neighbor_lock = std::chrono::steady_clock::now();
                auto coro = rdma::vamana::spinlock_vamana_node(neighbor_node, thread);
                while (!coro.handle.done()) {
                    co_await std::suspend_always{};
                    coro.handle.resume();
                }
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_neighbor_lock, t_neighbor_lock);
            }

            // Read neighbor's neighbor list
            const auto t_neighbor_list_read = std::chrono::steady_clock::now();
            s_ptr<VamanaNeighborlist> neighbor_nlist =
                co_await rdma::vamana::read_vamana_neighbors(neighbor_ptr, thread);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_neighbor_list_read, t_neighbor_list_read);

            if (neighbor_nlist->num_neighbors() < R_) {
                // Room available — just append
                const auto t_neighbor_prepare = std::chrono::steady_clock::now();
                neighbor_nlist->add(new_ptr);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_neighbor_prepare, t_neighbor_prepare);

                // Write updated neighbor list
                const auto t_neighbor_list_write = std::chrono::steady_clock::now();
                co_await rdma::vamana::write_vamana_neighbors(
                    neighbor_node,
                    neighbor_nlist->view(),
                    neighbor_nlist->num_neighbors(),
                    thread);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_neighbor_list_write, t_neighbor_list_write);
            } else {
                // Need to prune: gather all candidate neighbors + new node
                ++thread->stats.build_overflow_prunes;
                const auto t_overflow_prepare = std::chrono::steady_clock::now();
                const auto t_neighbor_collect = std::chrono::steady_clock::now();
                vec<RemotePtr> all_candidate_ptrs;
                all_candidate_ptrs.reserve(neighbor_nlist->num_neighbors() + 1);
                for (const auto& n : neighbor_nlist->view()) {
                    all_candidate_ptrs.push_back(n);
                }
                all_candidate_ptrs.push_back(new_ptr);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_neighbor_collect, t_neighbor_collect);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_overflow_prepare, t_overflow_prepare);

                u32 n_all = all_candidate_ptrs.size();
                vec<byte_t*> all_vec_bufs(n_all, nullptr);

                // Read remote vectors for the existing neighbors only.
                vec<RemotePtr> remote_candidate_ptrs;
                remote_candidate_ptrs.reserve(n_all - 1);
                for (u32 i = 0; i + 1 < n_all; ++i) {
                    remote_candidate_ptrs.push_back(all_candidate_ptrs[i]);
                }
                const auto t_overflow_vec_fetch = std::chrono::steady_clock::now();
                auto remote_vec_read = co_await rdma::vamana::batch_read_vectors(remote_candidate_ptrs, thread);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_overflow_vec_fetch, t_overflow_vec_fetch);
                for (u32 i = 0; i + 1 < n_all; ++i) {
                    all_vec_bufs[i] = remote_vec_read.host_buffers[i];
                }

                // Upload neighbor's vector as query for prune
                const auto t_overflow_query_h2d = std::chrono::steady_clock::now();
                std::memcpy(gs.h_query, neighbor_node->components().data(), dim_ * sizeof(float));
                cudaMemcpyAsync(gs.d_query, gs.h_query, dim_ * sizeof(float),
                               cudaMemcpyHostToDevice, gs.stream);
                track_build_h2d(thread, dim_ * sizeof(float));
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_overflow_query_h2d, t_overflow_query_h2d);

                // Compute distances from neighbor to all candidates
                const auto t_stage_overflow = std::chrono::steady_clock::now();
                for (u32 i = 0; i < n_all; ++i) {
                    if (i + 1 == n_all) {
                        std::memcpy(gs.h_candidate_vecs + i * dim_,
                                   components.data(),
                                   dim_ * sizeof(float));
                    } else {
                        std::memcpy(gs.h_candidate_vecs + i * dim_,
                                   reinterpret_cast<float*>(all_vec_bufs[i]),
                                   dim_ * sizeof(float));
                    }
                }
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_stage_candidates, t_stage_overflow);

                const auto t_overflow_candidate_h2d = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                               n_all * dim_ * sizeof(float),
                               cudaMemcpyHostToDevice, gs.stream);
                track_build_h2d(thread, n_all * dim_ * sizeof(float));
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_overflow_candidate_h2d, t_overflow_candidate_h2d);

                const auto t_overflow_gpu_distance = std::chrono::steady_clock::now();
                gpu::launch_batch_l2_distances(
                    gs.stream, gs.event,
                    gs.d_query, gs.d_candidate_vecs,
                    gs.d_distances, n_all, dim_);
                ++thread->stats.build_l2_kernels;
                co_await gpu::GpuAwaitable{thread.get()};
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_insert_overflow_distance, t_overflow_gpu_distance);

                const auto t_overflow_dist_d2h = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                               n_all * sizeof(float),
                               cudaMemcpyDeviceToHost, gs.stream);
                track_build_d2h(thread, n_all * sizeof(float));
                cudaStreamSynchronize(gs.stream);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_overflow_dist_d2h, t_overflow_dist_d2h);
                thread->stats.distcomps += n_all;
                thread->stats.build_distcomps += n_all;

                // Sort candidates by distance
                const auto t_overflow_cpu = std::chrono::steady_clock::now();
                vec<std::pair<u32, float>> idx_dist;
                idx_dist.reserve(n_all);
                for (u32 i = 0; i < n_all; ++i) {
                    idx_dist.push_back({i, gs.h_distances[i]});
                }
                std::sort(idx_dist.begin(), idx_dist.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });

                // Reorder candidate vecs and distances for prune kernel
                for (u32 i = 0; i < n_all; ++i) {
                    // Copy sorted distances
                    // We can reuse h_candidate_dists as temp
                    gs.h_candidate_dists[i] = idx_dist[i].second;
                    gs.h_candidate_order[i] = idx_dist[i].first;
                }
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_overflow_prepare, t_overflow_cpu);

                // Upload sorted distances and the corresponding original candidate order.
                const auto t_overflow_prune_inputs_h2d = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.d_candidate_dists, gs.h_candidate_dists,
                               n_all * sizeof(float),
                               cudaMemcpyHostToDevice, gs.stream);
                cudaMemcpyAsync(gs.d_candidate_order, gs.h_candidate_order,
                               n_all * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, gs.stream);
                track_build_h2d(thread, n_all * sizeof(float) + n_all * sizeof(uint32_t));
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_overflow_prune_inputs_h2d, t_overflow_prune_inputs_h2d);

                for (u32 i = 0; i + 1 < n_all; ++i) {
                    thread->buffer_allocator.free_buffer(all_vec_bufs[i], dim_ * sizeof(element_t));
                }

                const auto t_overflow_gpu_prune = std::chrono::steady_clock::now();
                gpu::launch_robust_prune(
                    gs.stream, gs.event,
                    gs.d_query,
                    gs.d_candidate_vecs,
                    gs.d_candidate_dists,
                    gs.d_candidate_order,
                    n_all, dim_, alpha_, R_,
                    gs.d_pruned_indices, gs.d_pruned_count);
                ++thread->stats.build_prune_kernels;
                co_await gpu::GpuAwaitable{thread.get()};
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_insert_overflow_prune, t_overflow_gpu_prune);

                const auto t_overflow_prune_d2h = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.h_pruned_indices, gs.d_pruned_indices,
                               R_ * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, gs.stream);
                cudaMemcpyAsync(gs.h_pruned_count, gs.d_pruned_count,
                               sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, gs.stream);
                track_build_d2h(thread, R_ * sizeof(uint32_t) + sizeof(uint32_t));
                cudaStreamSynchronize(gs.stream);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_overflow_prune_d2h, t_overflow_prune_d2h);

                const auto t_pruned_collect = std::chrono::steady_clock::now();
                u32 new_count = *gs.h_pruned_count;
                vec<RemotePtr> pruned_neighbors;
                pruned_neighbors.reserve(new_count);
                for (u32 i = 0; i < new_count; ++i) {
                    pruned_neighbors.push_back(all_candidate_ptrs[gs.h_pruned_indices[i]]);
                }
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_pruned_neighbor_collect, t_pruned_collect);

                const auto t_pruned_neighbor_write = std::chrono::steady_clock::now();
                co_await rdma::vamana::write_vamana_neighbors(
                    neighbor_node,
                    span<RemotePtr>{pruned_neighbors.data(), pruned_neighbors.size()},
                    static_cast<u8>(new_count),
                    thread);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_pruned_neighbor_write, t_pruned_neighbor_write);

                // (bump allocator; no individual free)
            }

            // Unlock neighbor
            const auto t_neighbor_unlock = std::chrono::steady_clock::now();
            co_await rdma::vamana::unlock_vamana_node(neighbor_node, thread);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_neighbor_unlock, t_neighbor_unlock);
        }

        const auto t_insert_finalize = std::chrono::steady_clock::now();
        beam.clear();
        visited.clear();
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_finalize, t_insert_finalize);
    }

    // =========================================================================
    // Index size estimation
    // =========================================================================

    size_t estimate_index_size(size_t num_nodes) const {
        return num_nodes * VamanaNode::total_size();
    }

private:
    // =========================================================================
    // Beam management
    // =========================================================================

    static void insert_into_beam(vec<VamanaCoroutine::BeamEntry>& beam,
                                 const RemotePtr& rptr, distance_t dist,
                                 u32 max_beam_width) {
        // Find insertion position (beam is maintained sorted by distance ascending)
        auto it = std::lower_bound(beam.begin(), beam.end(), dist,
            [](const VamanaCoroutine::BeamEntry& e, distance_t d) {
                return e.distance < d;
            });

        // Insert
        beam.insert(it, {rptr, dist, false});

        // Trim if over capacity
        if (beam.size() > max_beam_width) {
            beam.resize(max_beam_width);
        }
    }

    static void track_query_h2d(const u_ptr<ComputeThread>& thread, size_t bytes) {
        thread->stats.query_h2d_bytes += bytes;
    }

    static void track_query_d2h(const u_ptr<ComputeThread>& thread, size_t bytes) {
        thread->stats.query_d2h_bytes += bytes;
    }

    static void track_build_h2d(const u_ptr<ComputeThread>& thread, size_t bytes) {
        thread->stats.build_h2d_bytes += bytes;
    }

    static void track_build_d2h(const u_ptr<ComputeThread>& thread, size_t bytes) {
        thread->stats.build_d2h_bytes += bytes;
    }

    static void add_breakdown_subcategory(const u_ptr<ComputeThread>& thread,
                                    const service::breakdown::Subcategory subcategory,
                                    const std::chrono::steady_clock::time_point start) {
        if (auto* sample = thread->current_breakdown_sample()) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
            sample->add_subcategory(subcategory, static_cast<u64>(elapsed));
        }
    }

    // =========================================================================
    // Cache lookup (currently bypasses cache; VamanaNode caching TBD)
    // =========================================================================

    MinorCoroutine cache_lookup(RemotePtr rptr,
                                s_ptr<VamanaNode>& value,
                                const u_ptr<ComputeThread>& thread,
                                bool admit) const {
        if (!use_cache_) {
            value = co_await rdma::vamana::read_vamana_node(rptr, thread);
            co_return;
        }

        auto cache_entry = thread->cache.get<VamanaNode>(rptr);
        if (cache_entry.has_value()) {
            value = *cache_entry;
            ++thread->stats.cache_hits;
        } else {
            value = co_await rdma::vamana::read_vamana_node(rptr, thread);
            if (admit) {
                thread->cache.insert(rptr, value, thread->get_id());
            }
            ++thread->stats.cache_misses;
        }
    }

private:
    const u32 R_;
    const u32 beam_width_;
    const u32 beam_width_construction_;
    const f32 alpha_;
    const u32 k_;
    const u32 rabitq_bits_;
    const u32 dim_;
    const bool use_cache_;
    const bool use_rabitq_search_;
};

}  // namespace vamana
