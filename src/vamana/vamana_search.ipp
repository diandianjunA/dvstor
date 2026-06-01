    VamanaCoroutine knn(node_t q_id, const span<element_t> components,
                        const u_ptr<ComputeThread>& thread) const {
        dbg::print(dbg::stream{} << "T" << thread->get_id() << " queries " << q_id << "\n");
        ++thread->stats.processed;
        ++thread->stats.processed_queries;
        thread->refresh_neighbor_cache_if_stale();

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
                auto& cached_neighbors = coro_state.scratch_cached_neighbors;
                if (thread->neighbor_cache_enabled() &&
                    thread->neighbor_cache->lookup_copy(beam[best_idx].rptr, cached_neighbors)) {
                    ++thread->stats.neighbor_cache_hits;
                    neighbor_count = static_cast<u8>(cached_neighbors.size());
                    neighbor_ptrs = cached_neighbors.data();
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
                    if (thread->neighbor_cache_enabled()) {
                        const bool pin_entry = beam[best_idx].rptr == medoid_ptr;
                        thread->neighbor_cache->insert(beam[best_idx].rptr, nlist->view(), pin_entry);
                    }
                    neighbor_count = nlist->num_neighbors();
                    neighbor_ptrs = nlist->view().data();
                }
            }
            ++thread->stats.visited_neighborlists;

            // Filter unvisited neighbors
            const auto t_filter = std::chrono::steady_clock::now();
            auto& unvisited = coro_state.scratch_unvisited;
            unvisited.clear();
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
                    auto& fill_indices = gs.scratch_fill_indices;
                    auto& fill_slots = gs.scratch_fill_slots;
                    auto& fill_addrs = gs.scratch_fill_addrs;
                    auto& inflight_indices = gs.scratch_inflight_indices;
                    const auto resolved = rabitq_cache.resolve_batch(
                        unvisited.data(), n_batch, gs.h_cache_slot_ids, fill_indices, fill_slots, fill_addrs,
                        inflight_indices);

                    if (resolved.ok && inflight_indices.empty()) {
                        used_gpu_cache = true;
                        thread->stats.gpu_rabitq_cache_hits += resolved.hit_count;
                        thread->stats.gpu_rabitq_cache_misses += resolved.fill_count;
                        thread->stats.gpu_rabitq_cache_duplicate_fills += resolved.duplicate_loading_count;

                        if (!fill_indices.empty()) {
                            auto& fill_ptrs = coro_state.scratch_cache_ptrs;
                            fill_ptrs.clear();
                            fill_ptrs.reserve(fill_indices.size());
                            for (u32 idx : fill_indices) {
                                fill_ptrs.push_back(unvisited[idx]);
                            }
                            vec<rdma::vamana::BatchReadDestination> destinations;
                            destinations.reserve(fill_ptrs.size());
                            for (u32 i = 0; i < fill_ptrs.size(); ++i) {
                                destinations.push_back({fill_addrs[i], rabitq_cache.lkey(), nullptr, true});
                            }
                            const auto t_rabitq_fetch = std::chrono::steady_clock::now();
                            co_await rdma::vamana::batch_read_rabitq(fill_ptrs, thread, &destinations);
                            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_rabitq_fetch,
                                                      t_rabitq_fetch);
                            rabitq_cache.publish_batch(fill_slots);
                            thread->stats.gpu_rabitq_cache_fills += fill_ptrs.size();
                            thread->stats.gpu_rabitq_cache_fill_bytes +=
                                static_cast<u64>(fill_ptrs.size()) * VamanaNode::RABITQ_SIZE;
                        }

                        rabitq_cache.acquire_slots(gs.h_cache_slot_ids, n_batch);
                        const auto t_gpu_distance = std::chrono::steady_clock::now();
                        gpu::launch_batch_cached_rabitq_distances(
                            gs.stream, gs.event,
                            gs.d_rot_query,
                            gs.d_query_factor,
                            rabitq_cache.base(),
                            gs.d_cache_slot_ids,
                            gs.d_distances,
                            n_batch, dim_, rabitq_bits_,
                            rabitq_cache.stride());
                        ++thread->stats.query_rabitq_kernels;
                        co_await gpu::GpuAwaitable{thread.get()};
                        rabitq_cache.release_slots(gs.h_cache_slot_ids, n_batch);
                        add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_query_distance,
                                                  t_gpu_distance);
                    } else {
                        thread->stats.gpu_rabitq_cache_hits += resolved.hit_count;
                        thread->stats.gpu_rabitq_cache_misses += resolved.fill_count + resolved.inflight_fallback_count;
                        thread->stats.gpu_rabitq_cache_loading_fallbacks += resolved.inflight_fallback_count;
                        thread->stats.gpu_rabitq_cache_duplicate_fills += resolved.duplicate_loading_count;
                        ++thread->stats.gpu_rabitq_cache_fallback_batches;
                        rabitq_cache.rollback_loading(fill_slots);
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

            thread->stats.distcomps += n_batch;
            thread->stats.query_distcomps += n_batch;

            const auto t_distance_d2h = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                            n_batch * sizeof(float),
                            cudaMemcpyDeviceToHost, gs.stream);
            track_query_d2h(thread, n_batch * sizeof(float));
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_distance_d2h, t_distance_d2h);

            const auto t_beam_update = std::chrono::steady_clock::now();
            for (u32 i = 0; i < n_batch; ++i) {
                insert_into_beam(beam, unvisited[i], gs.h_distances[i], search_beam_capacity);
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
            results.push_back({node->id(), beam[i].distance});
        }
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_result_ids, t_result_ids);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_finalize, t_finalize);

        beam.clear();
        visited.clear();
    }

    // =========================================================================
    // Insert
    // =========================================================================
