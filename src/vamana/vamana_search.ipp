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

        // ── Two-stage RDMA prefetch pipeline ───────────────────────────
        // Prefetch state uses a SHADOW visited set (pf_visited) so the
        // real search state is never corrupted.
        //
        // Each iteration N:
        //   if prefetch ready: consume pf_unvisited → skip filter + vector RDMA
        //   else:             await neighbour → filter → vector RDMA normally
        //   select best(N+1) → start pf_neighbor for N+1
        //   during vector co_await: pf_neighbor(N+1) completes
        //   consume pf_neighbor(N+1) → filter into pf_unvisited (shadow)
        //   start pf_vectors into INACTIVE GPU buffer
        //   during GPU co_await: pf_vectors completes
        //   D2H → beam update
        //   select best(N+2) → start pf_neighbor for N+2 (for next iteration)
        //
        const bool use_indirect_candidate_path =
            use_gpudirect_candidate_rdma && thread->reserved_query_state[1] != nullptr;
        const bool use_prefetch = prefetch_pipeline_;
        auto select_best = [&beam]() -> i32 {
            i32 best = -1;
            distance_t best_d = std::numeric_limits<distance_t>::max();
            for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
                if (!beam[i].expanded && beam[i].distance < best_d) {
                    best_d = beam[i].distance;
                    best = i;
                }
            }
            return best;
        };
        auto inactive_staging = [&gs]() -> uint8_t* {
            return gs.query_candidate_buffer_index == 0
                       ? gs.d_candidate_vecs_alt : gs.d_candidate_vecs;
        };
        auto inactive_staging_lkey = [&gs]() -> u32 {
            return gs.query_candidate_buffer_index == 0
                       ? gs.d_candidate_vecs_alt_lkey : gs.d_candidate_vecs_lkey;
        };

        // Prefetch shadow state.
        rdma::vamana::NeighborReadAwaitable pf_neighbor;
        rdma::vamana::VectorBatchReadAwaitable pf_vectors;
        vec<RemotePtr>  pf_unvisited;
        hashset_t<RemotePtr> pf_visited;
        bool pf_ready = false;   // true when pf_unvisited + pf_vectors are ready

        // Cold start: fetch neighbour for the first expanded node.
        i32 best_idx = select_best();
        if (best_idx < 0) co_return;
        beam[best_idx].expanded = true;
        pf_neighbor = rdma::vamana::read_vamana_neighbors(beam[best_idx].rptr, &thread);

        while (true) {
            vec<RemotePtr> unvisited_storage;
            const RemotePtr* unvisited_ptr = nullptr;
            u32 n_batch = 0;

            if (pf_ready && use_prefetch) {
                // ── Prefetch hit: consume shadow unvisited ──────────────
                // pf_ready stays true so the vector-RDMA stage below
                // also skips RDMA (vectors are already in GPU staging).
                unvisited_storage = std::move(pf_unvisited);
                pf_unvisited.clear();
                pf_visited.clear();
                unvisited_ptr = unvisited_storage.data();
                n_batch = static_cast<u32>(unvisited_storage.size());
                for (u32 i = 0; i < n_batch; ++i)
                    visited.insert(unvisited_ptr[i]);
            } else {
                // ── Normal path: await neighbour → filter ───────────────
                const auto t_neighbor_fetch = std::chrono::steady_clock::now();
                auto nlist = co_await pf_neighbor;
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_neighbor_fetch,
                                          t_neighbor_fetch);
                const u8 neighbor_count = nlist->num_neighbors();
                const RemotePtr* neighbor_ptrs = nlist->view().data();
                ++thread->stats.visited_neighborlists;

                const auto t_filter = std::chrono::steady_clock::now();
                auto& unvisited = coro_state.scratch_unvisited;
                unvisited.clear();
                for (u32 i = 0; i < neighbor_count; ++i) {
                    const RemotePtr& np = neighbor_ptrs[i];
                    if (np.is_null()) continue;
                    if (!visited.contains(np)) {
                        visited.insert(np);
                        unvisited.push_back(np);
                    }
                }
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_filter, t_filter);
                unvisited_ptr = unvisited.data();
                n_batch = static_cast<u32>(unvisited.size());
            }

            if (n_batch == 0) {
                best_idx = select_best();
                if (best_idx < 0) break;
                beam[best_idx].expanded = true;
                pf_neighbor = rdma::vamana::read_vamana_neighbors(beam[best_idx].rptr, &thread);
                pf_ready = false;
                continue;
            }

            // ── Issue neighbour read for next iteration ─────────────────
            // In the prefetch pipeline, this issues pf_neighbor for N+1,
            // which overlaps with the vector RDMA below and is consumed
            // at the "Consume neighbour prefetch" stage later in this
            // iteration.  In the normal (non-prefetch) path, this is
            // simply the next iteration's pf_neighbor — it ensures
            // pf_neighbor is always valid when we loop back.
            best_idx = select_best();
            if (best_idx >= 0) {
                beam[best_idx].expanded = true;
                pf_neighbor = rdma::vamana::read_vamana_neighbors(beam[best_idx].rptr, &thread);
            }

            // ── Vector RDMA ────────────────────────────────────────────
            gs.flip_query_candidate_buffer();
            uint8_t* staging = gs.current_query_candidate_vecs();
            const u32 staging_lkey = gs.current_query_candidate_vecs_lkey();

            if (pf_ready && use_prefetch) {
                // Ensure prefetched vector RDMA is complete before the GPU
                // kernel reads the staging buffer.  Normally already done
                // (completed during previous iteration's GPU co_await).
                co_await pf_vectors;
                for (u32 i = 0; i < n_batch; ++i)
                    gs.h_candidate_ptrs[i] = staging + i * VamanaNode::vector_bytes();
                thread->stats.query_rdma_to_staging_bytes += n_batch * VamanaNode::vector_bytes();
                pf_ready = false;
            } else if (use_indirect_candidate_path) {
                vec<rdma::vamana::BatchReadDestination> dests; dests.reserve(n_batch);
                auto& mps = coro_state.indirect_candidate_ptrs; mps.clear(); mps.reserve(n_batch);
                auto& mis = coro_state.indirect_candidate_indices; mis.clear(); mis.reserve(n_batch);
                for (u32 i = 0; i < n_batch; ++i) {
                    auto* sp = staging + i * VamanaNode::vector_bytes();
                    gs.h_candidate_ptrs[i] = sp;
                    mps.push_back(unvisited_ptr[i]); mis.push_back(i);
                    dests.push_back({reinterpret_cast<u64>(sp), staging_lkey, nullptr, true});
                }
                const auto tvf = std::chrono::steady_clock::now();
                co_await rdma::vamana::batch_read_vectors(mps, thread, &dests);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_vector_fetch, tvf);
                thread->stats.query_rdma_to_staging_bytes += n_batch * VamanaNode::vector_bytes();
            } else {
                // Use the scratch unvisited (populated in filter step above).
                auto& uv = coro_state.scratch_unvisited;
                const auto tvf = std::chrono::steady_clock::now();
                auto vr = co_await rdma::vamana::batch_read_vectors(
                    uv, thread,
                    use_gpudirect_candidate_rdma ? gs.d_candidate_vecs : nullptr,
                    use_gpudirect_candidate_rdma ? gs.d_candidate_vecs_lkey : 0);
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_vector_fetch, tvf);
                if (vr.direct_to_gpu) {
                    thread->stats.query_rdma_to_staging_bytes += n_batch * VamanaNode::vector_bytes();
                } else {
                    thread->stats.query_host_staging_fallback_bytes += n_batch * VamanaNode::vector_bytes();
                    const auto tsc = std::chrono::steady_clock::now();
                    for (u32 i = 0; i < n_batch; ++i) {
                        std::memcpy(gs.h_candidate_vecs + i * VamanaNode::vector_bytes(),
                                    vr.host_buffers[i], VamanaNode::vector_bytes());
                        thread->buffer_allocator.free_buffer(vr.host_buffers[i], VamanaNode::vector_bytes());
                    }
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_stage_candidates, tsc);
                    const auto th2d = std::chrono::steady_clock::now();
                    cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                                    n_batch * VamanaNode::vector_bytes(), cudaMemcpyHostToDevice, gs.stream);
                    track_query_h2d(thread, n_batch * VamanaNode::vector_bytes());
                    add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_candidate_h2d, th2d);
                }
            }

            // ── Consume neighbour prefetch → issue vector prefetch ─────
            if (best_idx >= 0 && use_prefetch) {
                auto nlist_pf = co_await pf_neighbor;
                pf_unvisited.clear(); pf_visited.clear();
                const RemotePtr* pfp = nlist_pf->view().data();
                for (u32 i = 0; i < nlist_pf->num_neighbors(); ++i) {
                    if (pfp[i].is_null()) continue;
                    if (!visited.contains(pfp[i]) && !pf_visited.contains(pfp[i])) {
                        pf_visited.insert(pfp[i]); pf_unvisited.push_back(pfp[i]);
                    }
                }
                if (!pf_unvisited.empty()) {
                    auto* ib = inactive_staging(); u32 ilk = inactive_staging_lkey();
                    vec<rdma::vamana::BatchReadDestination> pd; pd.reserve(pf_unvisited.size());
                    for (size_t pi = 0; pi < pf_unvisited.size(); ++pi)
                        pd.push_back({reinterpret_cast<u64>(ib + pi * VamanaNode::vector_bytes()), ilk, nullptr, true});
                    pf_vectors = rdma::vamana::batch_read_vectors(pf_unvisited, thread, &pd);
                    pf_ready = true;
                    thread->stats.query_rdma_to_staging_bytes += pf_unvisited.size() * VamanaNode::vector_bytes();
                } else {
                    // All prefetched neighbours were already visited.
                    // pf_neighbor was consumed above → local_buffer is nullptr.
                    // Select the next best and issue a new pf_neighbor so the
                    // next (normal) iteration has a valid read to consume.
                    // If no unexpanded nodes remain, the line-278 fallback
                    // after the beam update will issue pf_neighbor or break.
                    best_idx = select_best();
                    if (best_idx >= 0) {
                        beam[best_idx].expanded = true;
                        pf_neighbor = rdma::vamana::read_vamana_neighbors(
                            beam[best_idx].rptr, &thread);
                    }
                }
            }

            cudaMemcpyAsync(const_cast<void**>(gs.d_candidate_ptrs), gs.h_candidate_ptrs,
                            n_batch * sizeof(void*), cudaMemcpyHostToDevice, gs.stream);
            track_query_h2d(thread, n_batch * sizeof(void*));

            const auto t_gpu = std::chrono::steady_clock::now();
            if (use_indirect_candidate_path)
                gpu::launch_batch_typed_query_l2_distances_indirect(
                    gs.stream, gs.event, gs.d_query, static_cast<u32>(query_dtype),
                    gs.d_candidate_ptrs, static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances, n_batch, dim_);
            else
                gpu::launch_batch_typed_query_l2_distances(
                    gs.stream, gs.event, gs.d_query, static_cast<u32>(query_dtype),
                    gs.d_candidate_vecs, static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances, n_batch, dim_);
            co_await gpu::GpuAwaitable{thread.get()};
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::gpu_query_distance, t_gpu);
            thread->stats.distcomps += n_batch;
            thread->stats.query_distcomps += n_batch;

            const auto t_d2h = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gs.h_distances, gs.d_distances, n_batch * sizeof(float),
                            cudaMemcpyDeviceToHost, gs.stream);
            track_query_d2h(thread, n_batch * sizeof(float));
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_distance_d2h, t_d2h);

            const auto t_bu = std::chrono::steady_clock::now();
            for (u32 i = 0; i < n_batch; ++i)
                insert_into_beam(beam, unvisited_ptr[i], gs.h_distances[i], beam_width_);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_beam_update, t_bu);

            // ── Start neighbour prefetch for iteration N+2 ─────────────
            if (best_idx < 0) {
                best_idx = select_best();
                if (best_idx < 0) break;
                beam[best_idx].expanded = true;
                pf_neighbor = rdma::vamana::read_vamana_neighbors(beam[best_idx].rptr, &thread);
                pf_ready = false;
            }
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
