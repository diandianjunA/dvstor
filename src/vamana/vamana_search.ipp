    VamanaCoroutine knn(node_t q_id, const span<element_t> components,
                        const u_ptr<ComputeThread>& thread) const {
        return knn_raw(q_id, reinterpret_cast<const byte_t*>(components.data()), VectorDType::float32, thread);
    }

    VamanaCoroutine knn_raw(node_t q_id, const byte_t* query_data, VectorDType query_dtype,
                            const u_ptr<ComputeThread>& thread) const {
        dbg::print(dbg::stream{} << "T" << thread->get_id() << " queries " << q_id << "\n");
        ++thread->stats.processed;
        ++thread->stats.processed_queries;

        auto& coro_state = thread->current_vamana_coroutine();
        auto& beam = coro_state.beam;
        auto& visited = coro_state.visited_nodes;
        auto& gpu = thread->gpu_buffers;
        const u32 coro_id = thread->current_coroutine_id();
        auto& gs = gpu.state(coro_id);
        const bool use_gpudirect_candidate_rdma =
            gpu.gpudirect_candidate_ready() && gs.d_candidate_vecs_rdma_registered;

        const auto t_medoid_ptr_start = std::chrono::steady_clock::now();
        RemotePtr medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_medoid_ptr, t_medoid_ptr_start);

        s_ptr<VamanaNode> medoid_node;
        {
            const auto t_node_read = std::chrono::steady_clock::now();
            auto coro = read_node(medoid_ptr, medoid_node, thread, true);
            while (!coro.handle.done()) {
                co_await std::suspend_always{};
                coro.handle.resume();
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_node_read, t_node_read);
        }

        const size_t query_bytes = vector_dtype_bytes(query_dtype, dim_);
        const auto t_query_h2d = std::chrono::steady_clock::now();
        std::memcpy(reinterpret_cast<byte_t*>(gs.h_query), query_data, query_bytes);
        cudaMemcpyAsync(gs.d_query, gs.h_query, query_bytes, cudaMemcpyHostToDevice, gs.stream);
        track_query_h2d(thread, query_bytes);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_query_h2d, t_query_h2d);

        distance_t medoid_dist = distance_to_stored_vector<Distance>(query_data, query_dtype, medoid_node->vector_data());
        ++thread->stats.distcomps;
        ++thread->stats.query_distcomps;

        beam.clear();
        beam.push_back({medoid_ptr, medoid_dist, false});
        visited.clear();
        visited.insert(medoid_ptr);

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
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_select, t_select);
            if (best_idx < 0) break;

            beam[best_idx].expanded = true;

            const auto t_neighbor_fetch = std::chrono::steady_clock::now();
            auto nlist = co_await rdma::vamana::read_vamana_neighbors_cached(
                beam[best_idx].rptr, thread, neighbor_cache_);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_neighbor_fetch,
                                      t_neighbor_fetch);
            const u8 neighbor_count = nlist->num_neighbors();
            const RemotePtr* neighbor_ptrs = nlist->view().data();
            ++thread->stats.visited_neighborlists;

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
            const bool use_gpu_vector_cache_path =
                gpu_vector_cache_ != nullptr && gpu_vector_cache_->gpu_buffer_registered();
            const bool use_indirect_candidate_path =
                use_gpudirect_candidate_rdma && thread->reserved_query_state[1] != nullptr;
            uint8_t* query_staging_vecs = nullptr;
            auto& miss_ptrs = coro_state.indirect_candidate_ptrs;
            auto& miss_indices = coro_state.indirect_candidate_indices;
            miss_ptrs.clear();
            miss_indices.clear();

            if (use_gpu_vector_cache_path) {
                // ── GPU Vector Cache path ───────────────────────────────
                // NOTE: The cache is safe under the current single-threaded
                // coroutine model where all coroutines on one ComputeThread
                // execute sequentially.  If the cache is shared across
                // threads, there is a TOCTOU window between find() returning
                // a slot_id and the GPU kernel reading d_candidate_ptrs:
                // another thread could evict the slot via allocate_slot.
                // Cross-thread safety would require a read-side epoch or RCU.
                gs.flip_query_candidate_buffer();
                query_staging_vecs = gs.current_query_candidate_vecs();
                const u32 query_staging_lkey = gs.current_query_candidate_vecs_lkey();

                struct CacheMissEntry {
                    RemotePtr rptr;
                    int32_t slot;
                    u32 batch_idx;
                };
                vec<CacheMissEntry> cache_slot_misses;  // RDMA → cache slot
                vec<CacheMissEntry> staging_misses;      // RDMA → staging (fallback)
                cache_slot_misses.reserve(n_batch);
                staging_misses.reserve(n_batch);

                // Step 1: Check cache for all unvisited nodes
                for (u32 i = 0; i < n_batch; ++i) {
                    int32_t slot = gpu_vector_cache_->find(unvisited[i]);
                    if (slot >= 0) {
                        // Cache hit — vector already on GPU
                        gs.h_candidate_ptrs[i] = gpu_vector_cache_->gpu_slot_ptr(slot);
                        thread->stats.gpu_vector_cache_hits++;
                    } else {
                        thread->stats.gpu_vector_cache_misses++;
                        // Try to reserve a cache slot for direct RDMA
                        int32_t new_slot = gpu_vector_cache_->allocate_slot(unvisited[i]);
                        if (new_slot >= 0) {
                            cache_slot_misses.push_back({unvisited[i], new_slot, i});
                        } else {
                            staging_misses.push_back({unvisited[i], -1, i});
                        }
                    }
                }

                // Step 2: Build RDMA destinations (single batch)
                vec<rdma::vamana::BatchReadDestination> destinations;
                destinations.reserve(cache_slot_misses.size() + staging_misses.size());

                for (auto& m : cache_slot_misses) {
                    miss_ptrs.push_back(m.rptr);
                    miss_indices.push_back(m.batch_idx);
                    destinations.push_back(rdma::vamana::BatchReadDestination{
                        gpu_vector_cache_->gpu_slot_addr(m.slot),
                        gpu_vector_cache_->gpu_buffer_lkey(thread->ctx->context.get_protection_domain()),
                        nullptr,
                        true});
                }
                for (auto& m : staging_misses) {
                    miss_ptrs.push_back(m.rptr);
                    miss_indices.push_back(m.batch_idx);
                    auto* staging = query_staging_vecs
                                  + static_cast<size_t>(m.batch_idx) * VamanaNode::vector_bytes();
                    gs.h_candidate_ptrs[m.batch_idx] = staging;
                    destinations.push_back(rdma::vamana::BatchReadDestination{
                        reinterpret_cast<u64>(staging),
                        query_staging_lkey,
                        nullptr,
                        true});
                }

                // Step 3: Batch RDMA read
                if (!miss_ptrs.empty()) {
                    const auto t_vector_fetch = std::chrono::steady_clock::now();
                    auto vec_read = co_await rdma::vamana::batch_read_vectors(
                        miss_ptrs, thread, &destinations);
                    (void)vec_read;
                    add_breakdown_subcategory(thread,
                        service::breakdown::Subcategory::rdma_vector_fetch, t_vector_fetch);
                    thread->stats.query_rdma_to_staging_bytes +=
                        static_cast<u64>(miss_ptrs.size()) * VamanaNode::vector_bytes();

                    // Step 4: Commit cache slots (vectors now in GPU via GPUDirect RDMA)
                    for (auto& m : cache_slot_misses) {
                        gpu_vector_cache_->commit_slot(m.slot, m.rptr);
                        gs.h_candidate_ptrs[m.batch_idx] =
                            gpu_vector_cache_->gpu_slot_ptr(m.slot);
                    }
                }

                // Step 5: H2D copy of d_candidate_ptrs
                cudaMemcpyAsync(const_cast<void**>(gs.d_candidate_ptrs), gs.h_candidate_ptrs,
                                static_cast<size_t>(n_batch) * sizeof(void*),
                                cudaMemcpyHostToDevice, gs.stream);
                track_query_h2d(thread, static_cast<u64>(n_batch) * sizeof(void*));

            } else if (use_indirect_candidate_path) {
                gs.flip_query_candidate_buffer();
                query_staging_vecs = gs.current_query_candidate_vecs();
                const u32 query_staging_lkey = gs.current_query_candidate_vecs_lkey();
                vec<rdma::vamana::BatchReadDestination> destinations;
                destinations.reserve(n_batch);
                miss_ptrs.reserve(n_batch);
                miss_indices.reserve(n_batch);

                for (u32 i = 0; i < n_batch; ++i) {
                    const auto staging_ptr = query_staging_vecs + static_cast<size_t>(i) * VamanaNode::vector_bytes();
                    gs.h_candidate_ptrs[i] = staging_ptr;
                    miss_ptrs.push_back(unvisited[i]);
                    miss_indices.push_back(i);
                    destinations.push_back(rdma::vamana::BatchReadDestination{
                        reinterpret_cast<u64>(staging_ptr),
                        query_staging_lkey,
                        nullptr,
                        true});
                }

                if (!miss_ptrs.empty()) {
                    const auto t_vector_fetch = std::chrono::steady_clock::now();
                    auto vec_read = co_await rdma::vamana::batch_read_vectors(miss_ptrs, thread, &destinations);
                    (void)vec_read;
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_vector_fetch,
                                              t_vector_fetch);
                    thread->stats.query_rdma_to_staging_bytes +=
                        static_cast<u64>(miss_ptrs.size()) * VamanaNode::vector_bytes();
                }

                cudaMemcpyAsync(const_cast<void**>(gs.d_candidate_ptrs), gs.h_candidate_ptrs,
                                static_cast<size_t>(n_batch) * sizeof(void*),
                                cudaMemcpyHostToDevice, gs.stream);
                track_query_h2d(thread, static_cast<u64>(n_batch) * sizeof(void*));
            } else {
                const auto t_vector_fetch = std::chrono::steady_clock::now();
                auto vec_read = co_await rdma::vamana::batch_read_vectors(
                    unvisited, thread,
                    use_gpudirect_candidate_rdma ? gs.d_candidate_vecs : nullptr,
                    use_gpudirect_candidate_rdma ? gs.d_candidate_vecs_lkey : 0);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_vector_fetch, t_vector_fetch);

                if (vec_read.direct_to_gpu) {
                    thread->stats.query_rdma_to_staging_bytes += static_cast<u64>(n_batch) * VamanaNode::vector_bytes();
                } else {
                    thread->stats.query_host_staging_fallback_bytes += static_cast<u64>(n_batch) * VamanaNode::vector_bytes();
                    const auto t_stage_candidates = std::chrono::steady_clock::now();
                    for (u32 i = 0; i < n_batch; ++i) {
                        std::memcpy(gs.h_candidate_vecs + static_cast<size_t>(i) * VamanaNode::vector_bytes(),
                                    vec_read.host_buffers[i],
                                    VamanaNode::vector_bytes());
                        thread->buffer_allocator.free_buffer(vec_read.host_buffers[i], VamanaNode::vector_bytes());
                    }
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_stage_candidates,
                                              t_stage_candidates);

                    const auto t_candidate_h2d = std::chrono::steady_clock::now();
                    cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                                    static_cast<size_t>(n_batch) * VamanaNode::vector_bytes(),
                                    cudaMemcpyHostToDevice, gs.stream);
                    track_query_h2d(thread, static_cast<u64>(n_batch) * VamanaNode::vector_bytes());
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_candidate_h2d,
                                              t_candidate_h2d);
                }
            }

            const auto t_gpu_distance = std::chrono::steady_clock::now();
            if (use_gpu_vector_cache_path || use_indirect_candidate_path) {
                gpu::launch_batch_typed_query_l2_distances_indirect(
                    gs.stream, gs.event,
                    gs.d_query, static_cast<u32>(query_dtype),
                    gs.d_candidate_ptrs, static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances, n_batch, dim_);
            } else {
                gpu::launch_batch_typed_query_l2_distances(
                    gs.stream, gs.event,
                    gs.d_query, static_cast<u32>(query_dtype),
                    gs.d_candidate_vecs, static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances, n_batch, dim_);
            }
            co_await gpu::GpuAwaitable{thread.get()};
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_query_distance, t_gpu_distance);

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
                insert_into_beam(beam, unvisited[i], gs.h_distances[i], beam_width_);
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_beam_update, t_beam_update);
        }

        const auto t_beam_sort = std::chrono::steady_clock::now();
        std::sort(beam.begin(), beam.end(), [](const auto& a, const auto& b) { return a.distance < b.distance; });
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_beam_sort, t_beam_sort);

        const auto t_finalize = std::chrono::steady_clock::now();
        auto& results = thread->query_results[q_id];
        results.clear();
        u32 count = std::min(k_, static_cast<u32>(beam.size()));

        const auto t_result_ids = std::chrono::steady_clock::now();
        for (u32 i = 0; i < count; ++i) {
            if (direct_node_reads_) {
                const node_t id = co_await rdma::vamana::read_vamana_id(beam[i].rptr, thread);
                results.push_back({id, beam[i].distance});
                continue;
            }
            s_ptr<VamanaNode> node;
            const auto t_node_read = std::chrono::steady_clock::now();
            auto coro = read_node(beam[i].rptr, node, thread, true);
            while (!coro.handle.done()) {
                co_await std::suspend_always{};
                coro.handle.resume();
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_node_read, t_node_read);
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
