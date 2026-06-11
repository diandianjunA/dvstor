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

        if (!use_rabitq_) {
            const size_t query_bytes = vector_dtype_bytes(query_dtype, dim_);
            const auto t_query_h2d = std::chrono::steady_clock::now();
            std::memcpy(reinterpret_cast<byte_t*>(gs.h_query), query_data, query_bytes);
            cudaMemcpyAsync(gs.d_query, gs.h_query, query_bytes, cudaMemcpyHostToDevice, gs.stream);
            track_query_h2d(thread, query_bytes);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_query_h2d, t_query_h2d);
        }

        distance_t medoid_dist = distance_to_stored_vector<Distance>(query_data, query_dtype, medoid_node->vector_data());
        ++thread->stats.distcomps;
        ++thread->stats.query_distcomps;

        beam.clear();
        beam.push_back({medoid_ptr, medoid_dist, false});
        visited.clear();
        visited.insert(medoid_ptr);

        // ── K-way Batched Beam Expansion ──────────────────────────────
        // Instead of expanding one node per iteration (serial beam search),
        // expand the top-K unexpanded nodes from the beam in each iteration.
        // This batches K neighbour RDMA reads, K vector batches, and K
        // GPU distance computations into one GPU kernel launch and one D2H
        // transfer, reducing per-iteration overhead by K×.
        //
        const u32 K = expansion_batch_;
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
        const bool use_indirect_candidate_path =
            use_gpudirect_candidate_rdma && thread->reserved_query_state[1] != nullptr;

        vec<rdma::vamana::NeighborReadAwaitable> pf_neighbors(K);
        u32 pending_K = 0;

        // ── Cold start: read the first neighbour ───────────────────────
        i32 best_idx = select_best();
        if (best_idx < 0) co_return;
        beam[best_idx].expanded = true;
        pf_neighbors[0] = rdma::vamana::read_vamana_neighbors(
            beam[best_idx].rptr, &thread);
        pending_K = 1;

        // Pre-compute asymmetric RaBitQ rotated query + norm once.
        float rabitq_query_norm2 = 0.0f;
        if (use_rabitq_) {
            const auto t_query_h2d = std::chrono::steady_clock::now();
            auto* rabitq_rotated_query = static_cast<float*>(gs.h_query);
            VamanaNode::compute_rotated_query(query_data, query_dtype,
                                              rabitq_rotated_query, &rabitq_query_norm2);
            const size_t rotated_query_bytes =
                static_cast<size_t>(VamanaNode::rabitq_code_bits()) * sizeof(float);
            cudaMemcpyAsync(gs.d_query, gs.h_query, rotated_query_bytes,
                            cudaMemcpyHostToDevice, gs.stream);
            track_query_h2d(thread, rotated_query_bytes);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_query_h2d, t_query_h2d);
        }

        while (true) {
            // ── Phase 1: consume neighbour reads → filter ──────────────
            auto& all_unvisited = coro_state.scratch_unvisited;
            all_unvisited.clear();
            for (u32 k = 0; k < pending_K; ++k) {
                thread->poll_cq();
                if (thread->post_balances[thread->current_coroutine_id()].load(
                        std::memory_order_acquire) == 0) {
                    pf_neighbors[k].mark_ready();
                }
                const auto t_nf = std::chrono::steady_clock::now();
                auto nlist = co_await pf_neighbors[k];
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::rdma_neighbor_fetch, t_nf);
                ++thread->stats.visited_neighborlists;

                const u8 nc = nlist->num_neighbors();
                const RemotePtr* np = nlist->view().data();
                for (u32 i = 0; i < nc; ++i) {
                    if (np[i].is_null()) continue;
                    if (!visited.contains(np[i])) {
                        visited.insert(np[i]);
                        all_unvisited.push_back(np[i]);
                    }
                }
            }
            pending_K = 0;

            if (all_unvisited.empty()) {
                best_idx = select_best();
                if (best_idx < 0) break;
                beam[best_idx].expanded = true;
                pf_neighbors[0] = rdma::vamana::read_vamana_neighbors(
                    beam[best_idx].rptr, &thread);
                pending_K = 1;
                continue;
            }

            const u32 n_batch = static_cast<u32>(all_unvisited.size());

            // ── Phase 2: vector / RaBitQ RDMA ──────────────────────────
            gs.flip_query_candidate_buffer();
            uint8_t* staging = gs.current_query_candidate_vecs();
            const u32 staging_lkey = gs.current_query_candidate_vecs_lkey();

            const auto tvf = std::chrono::steady_clock::now();
            const size_t rabitq_entry_size = VamanaNode::rabitq_entry_size();
            if (use_rabitq_) {
                // Read the aligned dynamic entry: code, norm, error, and reserved padding.
                if (use_gpudirect_candidate_rdma
                    && gs.current_query_candidate_vecs_registered()) {
                    co_await rdma::vamana::batch_read_at_offset(
                        all_unvisited, thread,
                        VamanaNode::offset_rabitq_code(), rabitq_entry_size,
                        staging, staging_lkey);
                } else {
                    auto rvr = co_await rdma::vamana::batch_read_at_offset_to_host(
                        all_unvisited, thread,
                        VamanaNode::offset_rabitq_code(), rabitq_entry_size);
                    thread->stats.query_host_staging_fallback_bytes += n_batch * rabitq_entry_size;
                    for (u32 i = 0; i < n_batch; ++i) {
                        std::memcpy(gs.h_candidate_vecs + i * rabitq_entry_size,
                                    rvr.host_buffers[i], rabitq_entry_size);
                        thread->buffer_allocator.free_buffer(rvr.host_buffers[i], rabitq_entry_size);
                    }
                    cudaMemcpyAsync(staging, gs.h_candidate_vecs,
                        n_batch * rabitq_entry_size, cudaMemcpyHostToDevice, gs.stream);
                    track_query_h2d(thread, n_batch * rabitq_entry_size);
                }
                thread->stats.query_rdma_to_staging_bytes += n_batch * rabitq_entry_size;
            } else {
                auto vr = co_await rdma::vamana::batch_read_vectors(
                    all_unvisited, thread,
                    use_gpudirect_candidate_rdma ? staging : nullptr,
                    use_gpudirect_candidate_rdma ? staging_lkey : 0);
                if (vr.direct_to_gpu) {
                    thread->stats.query_rdma_to_staging_bytes +=
                        n_batch * VamanaNode::vector_bytes();
                } else {
                    thread->stats.query_host_staging_fallback_bytes +=
                        n_batch * VamanaNode::vector_bytes();
                    const auto tsc = std::chrono::steady_clock::now();
                    for (u32 i = 0; i < n_batch; ++i) {
                        std::memcpy(gs.h_candidate_vecs + i * VamanaNode::vector_bytes(),
                                    vr.host_buffers[i], VamanaNode::vector_bytes());
                        thread->buffer_allocator.free_buffer(
                            vr.host_buffers[i], VamanaNode::vector_bytes());
                    }
                    add_breakdown_subcategory(thread,
                        service::breakdown::Subcategory::cpu_query_stage_candidates, tsc);
                    const auto th2d = std::chrono::steady_clock::now();
                    cudaMemcpyAsync(staging, gs.h_candidate_vecs,
                        n_batch * VamanaNode::vector_bytes(),
                        cudaMemcpyHostToDevice, gs.stream);
                    track_query_h2d(thread, n_batch * VamanaNode::vector_bytes());
                    add_breakdown_subcategory(thread,
                        service::breakdown::Subcategory::transfer_candidate_h2d, th2d);
                }
            }
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::rdma_vector_fetch, tvf);

            // ── Phase 3: GPU ───────────────────────────────────────────
            const auto t_gpu = std::chrono::steady_clock::now();
            if (use_rabitq_) {
                gpu::launch_batch_rabitq_asymmetric_distances(
                    gs.stream, gs.event,
                    reinterpret_cast<const float*>(gs.d_query),
                    staging,
                    gs.d_distances, rabitq_query_norm2,
                    n_batch, VamanaNode::rabitq_code_bits(),
                    static_cast<u32>(VamanaNode::rabitq_code_size()),
                    static_cast<u32>(rabitq_entry_size));
            } else if (use_indirect_candidate_path) {
                for (u32 i = 0; i < n_batch; ++i)
                    gs.h_candidate_ptrs[i] = staging + i * VamanaNode::vector_bytes();
                cudaMemcpyAsync(const_cast<void**>(gs.d_candidate_ptrs),
                    gs.h_candidate_ptrs, n_batch * sizeof(void*),
                    cudaMemcpyHostToDevice, gs.stream);
                track_query_h2d(thread, n_batch * sizeof(void*));
                gpu::launch_batch_typed_query_l2_distances_indirect(
                    gs.stream, gs.event, gs.d_query,
                    static_cast<u32>(query_dtype),
                    gs.d_candidate_ptrs,
                    static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances, n_batch, dim_);
            } else {
                gpu::launch_batch_typed_query_l2_distances(
                    gs.stream, gs.event, gs.d_query,
                    static_cast<u32>(query_dtype),
                    staging, static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances, n_batch, dim_);
            }
            co_await gpu::GpuAwaitable{thread.get()};
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::gpu_query_distance, t_gpu);
            thread->stats.distcomps += n_batch;
            thread->stats.query_distcomps += n_batch;

            // ── Phase 3: issue neighbour reads for next iteration ────
            // Issued before D2H so the RDMA overlaps with
            // cudaMemcpyAsync + cudaStreamSynchronize (8-10μs).
            for (u32 k = 0; k < K; ++k) {
                best_idx = select_best();
                if (best_idx < 0) break;
                beam[best_idx].expanded = true;
                pf_neighbors[k] = rdma::vamana::read_vamana_neighbors(
                    beam[best_idx].rptr, &thread);
                pending_K = k + 1;
            }

            const auto t_d2h = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                            n_batch * sizeof(float),
                            cudaMemcpyDeviceToHost, gs.stream);
            track_query_d2h(thread, n_batch * sizeof(float));
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::transfer_distance_d2h, t_d2h);

            const auto t_bu = std::chrono::steady_clock::now();
            for (u32 i = 0; i < n_batch; ++i)
                insert_into_beam(beam, all_unvisited[i], gs.h_distances[i], beam_width_);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::cpu_query_beam_update, t_bu);

            if (pending_K == 0) {
                best_idx = select_best();
                if (best_idx < 0) break;
                beam[best_idx].expanded = true;
                pf_neighbors[0] = rdma::vamana::read_vamana_neighbors(
                    beam[best_idx].rptr, &thread);
                pending_K = 1;
            }
        }

        const auto t_beam_sort = std::chrono::steady_clock::now();
        std::sort(beam.begin(), beam.end(), [](const auto& a, const auto& b) { return a.distance < b.distance; });
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_beam_sort, t_beam_sort);

        // RaBitQ: re-rank top candidates with exact L2 distances
        if (use_rabitq_ && !beam.empty()) {
            const u32 rerank_target = std::max(k_ * 4, k_ + 64);
            const u32 rerank_n = std::min(rerank_target, static_cast<u32>(beam.size()));
            vec<RemotePtr> rerank_ptrs(rerank_n);
            for (u32 i = 0; i < rerank_n; ++i)
                rerank_ptrs[i] = beam[i].rptr;
            // Read full vectors to host buffers for exact L2 on CPU.
            auto rvr = co_await rdma::vamana::batch_read_vectors(
                rerank_ptrs, thread,
                static_cast<void*>(nullptr), static_cast<u32>(0));
            for (u32 i = 0; i < rerank_n; ++i) {
                distance_t exact = distance_to_stored_vector<Distance>(
                    query_data, query_dtype,
                    reinterpret_cast<const byte_t*>(rvr.host_buffers[i]));
                beam[i].distance = exact;
                thread->buffer_allocator.free_buffer(
                    rvr.host_buffers[i], VamanaNode::vector_bytes());
            }
            beam.resize(rerank_n);
            std::sort(beam.begin(), beam.end(),
                [](const auto& a, const auto& b) { return a.distance < b.distance; });
            ++thread->stats.query_exact_reranks;
        }

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

    VamanaCoroutine knn_batch(const vec<node_t>& q_ids,
                               const vec<const byte_t*>& query_datas,
                               VectorDType query_dtype,
                               const u_ptr<ComputeThread>& thread) const {
        const u32 Q = static_cast<u32>(q_ids.size());
        const u32 coro_id = thread->current_coroutine_id();
        auto& gpu = thread->gpu_buffers;
        auto& gs = gpu.state(coro_id);

        // Per-query state arrays
        using BeamEntry = VamanaCoroutine::BeamEntry;
        struct QState {
            vec<BeamEntry> beam;
            hashset_t<RemotePtr> visited;
            rdma::vamana::NeighborReadAwaitable pf_neighbor;
            i32 best_idx = -1;
            bool active = true;
            node_t q_id;
        };
        vec<QState> qs(Q);
        const size_t q_bytes = vector_dtype_bytes(query_dtype, dim_);
        vec<u32> batch_offsets(Q + 1, 0);  // candidate offsets per query
        vec<RemotePtr> all_unvisited;
        all_unvisited.reserve(Q * R_);

        // Init: read medoid + first neighbor for each query
        for (u32 q = 0; q < Q; ++q) {
            auto& s = qs[q];
            s.q_id = q_ids[q];
            ++thread->stats.processed;
            ++thread->stats.processed_queries;
            RemotePtr medoid = co_await rdma::vamana::read_medoid_ptr(thread);
            s_ptr<VamanaNode> medoid_node;
            auto coro = read_node(medoid, medoid_node, thread, true);
            while (!coro.handle.done()) { co_await std::suspend_always{}; coro.handle.resume(); }
            distance_t medoid_dist = distance_to_stored_vector<Distance>(
                query_datas[q], query_dtype, medoid_node->vector_data());
            ++thread->stats.distcomps; ++thread->stats.query_distcomps;
            s.beam.clear(); s.beam.push_back({medoid, medoid_dist, false});
            s.visited.clear(); s.visited.insert(medoid);
            s.best_idx = 0; s.beam[0].expanded = true;
            s.pf_neighbor = rdma::vamana::read_vamana_neighbors(
                s.beam[0].rptr, &thread);
        }

        while (true) {
            // Phase 1: consume neighbor reads, collect all unvisited
            all_unvisited.clear();
            batch_offsets[0] = 0;
            u32 active_count = 0;
            for (u32 q = 0; q < Q; ++q) {
                if (!qs[q].active) { batch_offsets[q+1] = batch_offsets[q]; continue; }
                active_count++;
                auto& s = qs[q];
                thread->poll_cq();
                if (thread->post_balances[thread->current_coroutine_id()].load(
                        std::memory_order_acquire) == 0)
                    s.pf_neighbor.mark_ready();
                auto nlist = co_await s.pf_neighbor;
                ++thread->stats.visited_neighborlists;
                const u8 nc = nlist->num_neighbors();
                const RemotePtr* np = nlist->view().data();
                for (u32 i = 0; i < nc; ++i) {
                    if (np[i].is_null()) continue;
                    if (!s.visited.contains(np[i])) {
                        s.visited.insert(np[i]);
                        all_unvisited.push_back(np[i]);
                    }
                }
                batch_offsets[q+1] = static_cast<u32>(all_unvisited.size());
            }
            if (active_count == 0) break;

            const u32 n_batch = static_cast<u32>(all_unvisited.size());
            if (n_batch == 0) {
                // All neighbors visited for every query; pick new bests
                for (u32 q = 0; q < Q; ++q) {
                    if (!qs[q].active) continue;
                    auto& s = qs[q];
                    i32 best = -1; distance_t best_d = std::numeric_limits<distance_t>::max();
                    for (i32 i = 0; i < static_cast<i32>(s.beam.size()); ++i)
                        if (!s.beam[i].expanded && s.beam[i].distance < best_d)
                            { best_d = s.beam[i].distance; best = i; }
                    if (best < 0) { s.active = false; continue; }
                    s.beam[best].expanded = true;
                    s.pf_neighbor = rdma::vamana::read_vamana_neighbors(
                        s.beam[best].rptr, &thread);
                }
                continue;
            }

            // Phase 2: select best for next iteration
            for (u32 q = 0; q < Q; ++q) {
                if (!qs[q].active) continue;
                auto& s = qs[q];
                i32 best = -1; distance_t best_d = std::numeric_limits<distance_t>::max();
                for (i32 i = 0; i < static_cast<i32>(s.beam.size()); ++i)
                    if (!s.beam[i].expanded && s.beam[i].distance < best_d)
                        { best_d = s.beam[i].distance; best = i; }
                if (best >= 0) {
                    s.beam[best].expanded = true;
                    s.pf_neighbor = rdma::vamana::read_vamana_neighbors(
                        s.beam[best].rptr, &thread);
                }
                s.best_idx = best;
            }

            // Phase 3: vector RDMA for all_unvisited (one batch)
            gs.flip_query_candidate_buffer();
            uint8_t* staging = gs.current_query_candidate_vecs();
            const u32 staging_lkey = gs.current_query_candidate_vecs_lkey();
            const bool use_gdr = gpu.gpudirect_candidate_ready() &&
                                 gs.d_candidate_vecs_rdma_registered;
            const auto tvf = std::chrono::steady_clock::now();
            auto vr = co_await rdma::vamana::batch_read_vectors(
                all_unvisited, thread,
                use_gdr ? staging : nullptr, use_gdr ? staging_lkey : 0);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::rdma_vector_fetch, tvf);
            if (vr.direct_to_gpu) {
                thread->stats.query_rdma_to_staging_bytes += n_batch * VamanaNode::vector_bytes();
            }

            // Phase 4: Q GPU kernels (one per query) + ONE D2H for all
            const size_t vec_sz = VamanaNode::vector_bytes();
            const auto t_gpu = std::chrono::steady_clock::now();
            for (u32 q = 0; q < Q; ++q) {
                u32 nq = batch_offsets[q+1] - batch_offsets[q];
                if (nq == 0) continue;
                cudaMemcpyAsync(gs.d_query, query_datas[q], q_bytes,
                                cudaMemcpyHostToDevice, gs.stream);
                gpu::launch_batch_typed_query_l2_distances(
                    gs.stream, nullptr,
                    gs.d_query, static_cast<u32>(query_dtype),
                    staging + batch_offsets[q] * vec_sz,
                    static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances + batch_offsets[q], nq, dim_);
            }
            cudaEventRecord(gs.event, gs.stream);
            co_await gpu::GpuAwaitable{thread.get()};
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::gpu_query_distance, t_gpu);
            thread->stats.distcomps += n_batch;
            thread->stats.query_distcomps += n_batch;

            // Phase 5: D2H + per-query beam update
            const auto t_d2h = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                            n_batch * sizeof(float),
                            cudaMemcpyDeviceToHost, gs.stream);
            track_query_d2h(thread, n_batch * sizeof(float));
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::transfer_distance_d2h, t_d2h);

            for (u32 q = 0; q < Q; ++q) {
                if (!qs[q].active) continue;
                auto& s = qs[q];
                for (u32 i = batch_offsets[q]; i < batch_offsets[q+1]; ++i)
                    insert_into_beam(s.beam, all_unvisited[i],
                                     gs.h_distances[i], beam_width_);
                if (s.best_idx < 0) {
                    i32 best = -1; distance_t best_d = std::numeric_limits<distance_t>::max();
                    for (i32 i = 0; i < static_cast<i32>(s.beam.size()); ++i)
                        if (!s.beam[i].expanded && s.beam[i].distance < best_d)
                            { best_d = s.beam[i].distance; best = i; }
                    if (best < 0) { s.active = false; continue; }
                    s.beam[best].expanded = true;
                    s.pf_neighbor = rdma::vamana::read_vamana_neighbors(
                        s.beam[best].rptr, &thread);
                }
            }
        }

        // Finalize: sort beams, collect results
        for (u32 q = 0; q < Q; ++q) {
            auto& s = qs[q];
            std::sort(s.beam.begin(), s.beam.end(),
                [](const auto& a, const auto& b) { return a.distance < b.distance; });
            auto& results = thread->query_results[s.q_id];
            results.clear();
            u32 count = std::min(k_, static_cast<u32>(s.beam.size()));
            for (u32 i = 0; i < count; ++i) {
                if (direct_node_reads_) {
                    const node_t id = co_await rdma::vamana::read_vamana_id(
                        s.beam[i].rptr, thread);
                    results.push_back({id, s.beam[i].distance});
                    continue;
                }
                s_ptr<VamanaNode> node;
                auto coro = read_node(s.beam[i].rptr, node, thread, true);
                while (!coro.handle.done()) { co_await std::suspend_always{}; coro.handle.resume(); }
                results.push_back({node->id(), s.beam[i].distance});
            }
        }
    }

    // =========================================================================
    // Insert
    // =========================================================================
