    VamanaCoroutine knn(node_t q_id, const span<element_t> components,
                        const u_ptr<ComputeThread>& thread) const {
        return knn_raw(q_id, reinterpret_cast<const byte_t*>(components.data()), VectorDType::float32, thread);
    }

    VamanaCoroutine knn_raw(node_t q_id, const byte_t* query_data, VectorDType query_dtype,
                            const u_ptr<ComputeThread>& thread) const {
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
        bool exact_query_uploaded = false;
        u32 exact_budget_remaining = rabitq_exact_budget_;
        const bool use_local_rabitq_cache = use_rabitq_ && rabitq_cache_ != nullptr;
        rabitq::QueryLut rabitq_query_lut{};
        if (use_rabitq_) {
            const auto t_query_h2d = std::chrono::steady_clock::now();
            auto* rabitq_rotated_query = static_cast<float*>(gs.h_query);
            VamanaNode::compute_rotated_query(query_data, query_dtype,
                                              rabitq_rotated_query, &rabitq_query_norm2);
            const size_t rotated_query_bytes =
                static_cast<size_t>(VamanaNode::rabitq_code_bits()) * sizeof(float);
            if (!use_local_rabitq_cache) {
                cudaMemcpyAsync(gs.d_query, gs.h_query, rotated_query_bytes,
                                cudaMemcpyHostToDevice, gs.stream);
                track_query_h2d(thread, rotated_query_bytes);
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::transfer_query_h2d, t_query_h2d);
            } else {
                rabitq_query_lut = rabitq::build_query_lut(rabitq_rotated_query);
            }
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

            if (use_rabitq_) {
                lib_assert(rabitq_cache_ != nullptr,
                           "RaBitQ gate requires a loaded v2 sidecar");
                const auto t_gate = std::chrono::steady_clock::now();
                vec<f32> approximate_distances(n_batch,
                    std::numeric_limits<f32>::infinity());
                vec<u32> cache_miss_indices;
                const auto& quantization = rabitq_cache_->quantization();
                for (u32 i = 0; i < n_batch; ++i) {
                    const auto* entry = rabitq_cache_->find(all_unvisited[i]);
                    if (entry == nullptr) {
                        cache_miss_indices.push_back(i);
                    } else {
                        approximate_distances[i] = rabitq::estimate_distance_lut(
                            rabitq_query_lut, rabitq_query_norm2, *entry, quantization);
                    }
                }
                thread->stats.query_rabitq_l0_candidates += n_batch;
                thread->stats.query_rabitq_cache_misses += cache_miss_indices.size();
                const vec<u32> gate_indices = rabitq::select_gate(
                    approximate_distances, cache_miss_indices,
                    rabitq_gate_width_, rabitq_gate_max_width_, rabitq_gate_margin_);
                thread->stats.query_rabitq_l1_candidates += gate_indices.size();
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::cpu_query_rabitq_gate, t_gate);

                vec<RemotePtr> exact_ptrs;
                exact_ptrs.reserve(gate_indices.size());
                for (u32 index : gate_indices) exact_ptrs.push_back(all_unvisited[index]);

                const auto t_exact_fetch = std::chrono::steady_clock::now();
                if (!exact_query_uploaded) {
                    const size_t exact_query_bytes = vector_dtype_bytes(query_dtype, dim_);
                    const auto t_exact_query_h2d = std::chrono::steady_clock::now();
                    cudaMemcpyAsync(gs.d_query, query_data, exact_query_bytes,
                                    cudaMemcpyHostToDevice, gs.stream);
                    track_query_h2d(thread, exact_query_bytes);
                    add_breakdown_subcategory(thread,
                        service::breakdown::Subcategory::transfer_query_h2d,
                        t_exact_query_h2d);
                    exact_query_uploaded = true;
                }
                gs.flip_query_candidate_buffer();
                uint8_t* exact_staging = gs.current_query_candidate_vecs();
                auto exact_vectors = co_await rdma::vamana::batch_read_vectors(
                    exact_ptrs, thread,
                    use_gpudirect_candidate_rdma ? exact_staging : nullptr,
                    use_gpudirect_candidate_rdma
                      ? gs.current_query_candidate_vecs_lkey() : 0);
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::rdma_vector_fetch, t_exact_fetch);
                if (exact_vectors.direct_to_gpu) {
                    thread->stats.query_rdma_to_staging_bytes +=
                        gate_indices.size() * VamanaNode::vector_bytes();
                } else {
                    thread->stats.query_host_staging_fallback_bytes +=
                        gate_indices.size() * VamanaNode::vector_bytes();
                    const auto t_stage = std::chrono::steady_clock::now();
                    for (u32 i = 0; i < gate_indices.size(); ++i) {
                        std::memcpy(gs.h_candidate_vecs + i * VamanaNode::vector_bytes(),
                                    exact_vectors.host_buffers[i], VamanaNode::vector_bytes());
                        thread->buffer_allocator.free_buffer(
                            exact_vectors.host_buffers[i], VamanaNode::vector_bytes());
                    }
                    add_breakdown_subcategory(thread,
                        service::breakdown::Subcategory::cpu_query_stage_candidates, t_stage);
                    const auto t_h2d = std::chrono::steady_clock::now();
                    cudaMemcpyAsync(exact_staging, gs.h_candidate_vecs,
                                    gate_indices.size() * VamanaNode::vector_bytes(),
                                    cudaMemcpyHostToDevice, gs.stream);
                    track_query_h2d(thread, gate_indices.size() * VamanaNode::vector_bytes());
                    add_breakdown_subcategory(thread,
                        service::breakdown::Subcategory::transfer_candidate_h2d, t_h2d);
                }

                const auto t_exact = std::chrono::steady_clock::now();
                gpu::launch_batch_typed_query_l2_distances(
                    gs.stream, gs.event, gs.d_query,
                    static_cast<u32>(query_dtype), exact_staging,
                    static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances, static_cast<u32>(gate_indices.size()), dim_);
                co_await gpu::GpuAwaitable{thread.get()};
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::gpu_query_distance, t_exact);
                const auto t_d2h = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                                gate_indices.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, gs.stream);
                track_query_d2h(thread, gate_indices.size() * sizeof(float));
                cudaStreamSynchronize(gs.stream);
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::transfer_distance_d2h, t_d2h);
                const auto t_beam_update = std::chrono::steady_clock::now();
                for (u32 i = 0; i < gate_indices.size(); ++i) {
                    insert_into_beam(beam, exact_ptrs[i], gs.h_distances[i], beam_width_);
                }
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::cpu_query_beam_update, t_beam_update);
                thread->stats.distcomps += gate_indices.size();
                thread->stats.query_distcomps += gate_indices.size();
                thread->stats.query_exact_reranks += gate_indices.size();
                thread->stats.query_rabitq_l2_candidates += gate_indices.size();

                for (u32 k = 0; k < K; ++k) {
                    best_idx = select_best();
                    if (best_idx < 0) break;
                    beam[best_idx].expanded = true;
                    pf_neighbors[k] = rdma::vamana::read_vamana_neighbors(
                        beam[best_idx].rptr, &thread);
                    pending_K = k + 1;
                }
                if (pending_K == 0) break;
                continue;
            }

            // ── Phase 2: vector / RaBitQ RDMA ──────────────────────────
            gs.flip_query_candidate_buffer();
            uint8_t* staging = gs.current_query_candidate_vecs();
            const u32 staging_lkey = gs.current_query_candidate_vecs_lkey();

            const auto tvf = std::chrono::steady_clock::now();
            const size_t rabitq_entry_size = VamanaNode::rabitq_entry_size();
            if (use_rabitq_ && !use_local_rabitq_cache) {
                thread->stats.query_rabitq_l1_candidates += n_batch;
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
            } else if (!use_local_rabitq_cache) {
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
            if (!use_local_rabitq_cache) {
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::rdma_vector_fetch, tvf);
            }

            // ── Phase 3: GPU ───────────────────────────────────────────
            const auto t_gpu = std::chrono::steady_clock::now();
            if (use_local_rabitq_cache) {
                thread->stats.query_rabitq_l0_candidates += n_batch;
                const auto* rotated_query = static_cast<const f32*>(gs.h_query);
                const auto& quantization = rabitq_cache_->quantization();
                vec<RemotePtr> cache_misses;
                vec<u32> cache_miss_indices;
                for (u32 i = 0; i < n_batch; ++i) {
                    const auto* entry = rabitq_cache_->find(all_unvisited[i]);
                    if (entry == nullptr) {
                        cache_misses.push_back(all_unvisited[i]);
                        cache_miss_indices.push_back(i);
                    } else {
                        const auto estimate = rabitq::estimate_interval_lut(
                            rabitq_query_lut, rabitq_query_norm2, *entry, quantization,
                            rabitq_confidence_epsilon_);
                        gs.h_distances[i] = estimate.distance;
                        gs.h_candidate_dists[i] = estimate.lower_bound;
                    }
                }
                if (!cache_misses.empty()) {
                    thread->stats.query_rabitq_cache_misses += cache_misses.size();
                    thread->stats.query_rabitq_l1_candidates += cache_misses.size();
                    auto misses = co_await rdma::vamana::batch_read_at_offset_to_host(
                        cache_misses, thread, VamanaNode::offset_rabitq_code(), rabitq_entry_size);
                    for (u32 i = 0; i < cache_misses.size(); ++i) {
                        gs.h_distances[cache_miss_indices[i]] = rabitq::estimate_full_entry(
                            rotated_query, rabitq_query_norm2, misses.host_buffers[i]);
                        gs.h_candidate_dists[cache_miss_indices[i]] =
                            std::numeric_limits<distance_t>::quiet_NaN();
                        thread->buffer_allocator.free_buffer(
                            misses.host_buffers[i], rabitq_entry_size);
                    }
                }
            } else if (use_rabitq_) {
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
            if (!use_local_rabitq_cache) {
                co_await gpu::GpuAwaitable{thread.get()};
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::gpu_query_distance, t_gpu);
            }
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

            if (!use_local_rabitq_cache) {
                const auto t_d2h = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                                n_batch * sizeof(float),
                                cudaMemcpyDeviceToHost, gs.stream);
                track_query_d2h(thread, n_batch * sizeof(float));
                cudaStreamSynchronize(gs.stream);
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::transfer_distance_d2h, t_d2h);
                if (use_rabitq_) {
                    // Full-dimensional entries provide an estimate but no conservative
                    // interval. Mark that explicitly so exactification is ranked by the
                    // estimate instead of treating every candidate as lower_bound=0.
                    std::fill(gs.h_candidate_dists, gs.h_candidate_dists + n_batch,
                              std::numeric_limits<distance_t>::quiet_NaN());
                }
            }

            if (use_rabitq_ && rabitq_exact_batch_ > 0 && exact_budget_remaining > 0) {
                const bool beam_full = beam.size() >= beam_width_;
                const distance_t cutoff = beam_full
                    ? beam.back().distance
                    : std::numeric_limits<distance_t>::max();
                u32 eligible_n = 0;
                for (u32 i = 0; i < n_batch; ++i) {
                    const distance_t lower = gs.h_candidate_dists[i];
                    const bool has_interval = std::isfinite(lower);
                    const distance_t upper = has_interval
                        ? 2.0f * gs.h_distances[i] - lower
                        : gs.h_distances[i];
                    const bool competitive = has_interval
                        ? (lower <= cutoff && upper >= cutoff)
                        : gs.h_distances[i] <= cutoff;
                    if (!beam_full || competitive) {
                        gs.h_candidate_order[eligible_n++] = i;
                    }
                }

                const u32 refine_n = std::min({rabitq_exact_batch_, eligible_n,
                                               exact_budget_remaining});
                if (refine_n > 0) {
                    std::partial_sort(gs.h_candidate_order,
                                      gs.h_candidate_order + refine_n,
                                      gs.h_candidate_order + eligible_n,
                                      [&](u32 lhs, u32 rhs) {
                                          const distance_t lhs_lower = gs.h_candidate_dists[lhs];
                                          const distance_t rhs_lower = gs.h_candidate_dists[rhs];
                                          const distance_t lhs_rank = std::isfinite(lhs_lower)
                                              ? lhs_lower : gs.h_distances[lhs];
                                          const distance_t rhs_rank = std::isfinite(rhs_lower)
                                              ? rhs_lower : gs.h_distances[rhs];
                                          return lhs_rank < rhs_rank;
                                      });

                    vec<RemotePtr> refine_ptrs(refine_n);
                    for (u32 i = 0; i < refine_n; ++i) {
                        refine_ptrs[i] = all_unvisited[gs.h_candidate_order[i]];
                    }

                    const auto t_rerank_fetch = std::chrono::steady_clock::now();
                    rdma::vamana::VectorBatchReadResult exact_vectors;
                    uint8_t* exact_staging = nullptr;
                    if (use_local_rabitq_cache) {
                        if (!exact_query_uploaded) {
                            const size_t exact_query_bytes = vector_dtype_bytes(query_dtype, dim_);
                            const auto t_exact_query_h2d = std::chrono::steady_clock::now();
                            cudaMemcpyAsync(gs.d_query, query_data, exact_query_bytes,
                                            cudaMemcpyHostToDevice, gs.stream);
                            track_query_h2d(thread, exact_query_bytes);
                            add_breakdown_subcategory(thread,
                                service::breakdown::Subcategory::transfer_query_h2d,
                                t_exact_query_h2d);
                            exact_query_uploaded = true;
                        }
                        gs.flip_query_candidate_buffer();
                        exact_staging = gs.current_query_candidate_vecs();
                        exact_vectors = co_await rdma::vamana::batch_read_vectors(
                            refine_ptrs, thread,
                            use_gpudirect_candidate_rdma ? exact_staging : nullptr,
                            use_gpudirect_candidate_rdma
                              ? gs.current_query_candidate_vecs_lkey() : 0);
                    } else {
                        exact_vectors = co_await rdma::vamana::batch_read_vectors(
                            refine_ptrs, thread,
                            static_cast<void*>(nullptr), static_cast<u32>(0));
                    }
                    add_breakdown_subcategory(thread,
                        service::breakdown::Subcategory::rdma_rerank_fetch,
                        t_rerank_fetch);

                    if (use_local_rabitq_cache) {
                        if (exact_vectors.direct_to_gpu) {
                            thread->stats.query_rdma_to_staging_bytes +=
                                refine_n * VamanaNode::vector_bytes();
                        } else {
                            thread->stats.query_host_staging_fallback_bytes +=
                                refine_n * VamanaNode::vector_bytes();
                        }
                        if (!exact_vectors.direct_to_gpu) {
                            for (u32 i = 0; i < refine_n; ++i) {
                                std::memcpy(gs.h_candidate_vecs + i * VamanaNode::vector_bytes(),
                                            exact_vectors.host_buffers[i], VamanaNode::vector_bytes());
                                thread->buffer_allocator.free_buffer(
                                    exact_vectors.host_buffers[i], VamanaNode::vector_bytes());
                            }
                            cudaMemcpyAsync(exact_staging, gs.h_candidate_vecs,
                                            refine_n * VamanaNode::vector_bytes(),
                                            cudaMemcpyHostToDevice, gs.stream);
                            track_query_h2d(thread, refine_n * VamanaNode::vector_bytes());
                        }
                        const auto t_rerank_gpu = std::chrono::steady_clock::now();
                        gpu::launch_batch_typed_query_l2_distances(
                            gs.stream, gs.event, gs.d_query,
                            static_cast<u32>(query_dtype), exact_staging,
                            static_cast<u32>(VamanaNode::vector_dtype()),
                            gs.d_distances, refine_n, dim_);
                        co_await gpu::GpuAwaitable{thread.get()};
                        add_breakdown_subcategory(thread,
                            service::breakdown::Subcategory::gpu_query_rerank,
                            t_rerank_gpu);
                        const auto t_rerank_d2h = std::chrono::steady_clock::now();
                        cudaMemcpyAsync(gs.h_candidate_dists, gs.d_distances,
                                        refine_n * sizeof(float),
                                        cudaMemcpyDeviceToHost, gs.stream);
                        track_query_d2h(thread, refine_n * sizeof(float));
                        cudaStreamSynchronize(gs.stream);
                        add_breakdown_subcategory(thread,
                            service::breakdown::Subcategory::transfer_rerank_d2h,
                            t_rerank_d2h);
                        const auto t_rerank_update = std::chrono::steady_clock::now();
                        for (u32 i = 0; i < refine_n; ++i) {
                            gs.h_distances[gs.h_candidate_order[i]] = gs.h_candidate_dists[i];
                        }
                        add_breakdown_subcategory(thread,
                            service::breakdown::Subcategory::cpu_query_rerank_update,
                            t_rerank_update);
                    } else {
                        const auto t_rerank_update = std::chrono::steady_clock::now();
                        for (u32 i = 0; i < refine_n; ++i) {
                            const u32 candidate_index = gs.h_candidate_order[i];
                            gs.h_distances[candidate_index] = distance_to_stored_vector<Distance>(
                                query_data, query_dtype,
                                reinterpret_cast<const byte_t*>(exact_vectors.host_buffers[i]));
                            thread->buffer_allocator.free_buffer(
                                exact_vectors.host_buffers[i], VamanaNode::vector_bytes());
                        }
                        add_breakdown_subcategory(thread,
                            service::breakdown::Subcategory::cpu_query_rerank_update,
                            t_rerank_update);
                    }
                    thread->stats.query_exact_reranks += refine_n;
                    thread->stats.query_rabitq_l2_candidates += refine_n;
                    exact_budget_remaining -= refine_n;
                }
            }

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
        if (false && use_rabitq_ && !beam.empty()) {
            const u32 rerank_target = std::max(k_ * 4, k_ + 64);
            const u32 rerank_n = std::min(rerank_target, static_cast<u32>(beam.size()));
            vec<RemotePtr> rerank_ptrs(rerank_n);
            for (u32 i = 0; i < rerank_n; ++i)
                rerank_ptrs[i] = beam[i].rptr;
            if (!exact_query_uploaded) {
                const size_t exact_query_bytes = vector_dtype_bytes(query_dtype, dim_);
                const auto t_exact_query_h2d = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.d_query, query_data, exact_query_bytes,
                                cudaMemcpyHostToDevice, gs.stream);
                track_query_h2d(thread, exact_query_bytes);
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::transfer_query_h2d,
                    t_exact_query_h2d);
                exact_query_uploaded = true;
            }
            gs.flip_query_candidate_buffer();
            uint8_t* rerank_staging = gs.current_query_candidate_vecs();
            const auto t_final_rerank_fetch = std::chrono::steady_clock::now();
            auto rvr = co_await rdma::vamana::batch_read_vectors(
                rerank_ptrs, thread,
                use_gpudirect_candidate_rdma ? rerank_staging : nullptr,
                use_gpudirect_candidate_rdma
                  ? gs.current_query_candidate_vecs_lkey() : 0);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::rdma_rerank_fetch,
                t_final_rerank_fetch);
            if (rvr.direct_to_gpu) {
                thread->stats.query_rdma_to_staging_bytes +=
                    rerank_n * VamanaNode::vector_bytes();
            } else {
                thread->stats.query_host_staging_fallback_bytes +=
                    rerank_n * VamanaNode::vector_bytes();
                for (u32 i = 0; i < rerank_n; ++i) {
                    std::memcpy(gs.h_candidate_vecs + i * VamanaNode::vector_bytes(),
                                rvr.host_buffers[i], VamanaNode::vector_bytes());
                    thread->buffer_allocator.free_buffer(
                        rvr.host_buffers[i], VamanaNode::vector_bytes());
                }
                cudaMemcpyAsync(rerank_staging, gs.h_candidate_vecs,
                                rerank_n * VamanaNode::vector_bytes(),
                                cudaMemcpyHostToDevice, gs.stream);
                track_query_h2d(thread, rerank_n * VamanaNode::vector_bytes());
            }
            const auto t_final_rerank_gpu = std::chrono::steady_clock::now();
            gpu::launch_batch_typed_query_l2_distances(
                gs.stream, gs.event, gs.d_query,
                static_cast<u32>(query_dtype), rerank_staging,
                static_cast<u32>(VamanaNode::vector_dtype()),
                gs.d_distances, rerank_n, dim_);
            co_await gpu::GpuAwaitable{thread.get()};
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::gpu_query_rerank,
                t_final_rerank_gpu);
            const auto t_final_rerank_d2h = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gs.h_candidate_dists, gs.d_distances,
                            rerank_n * sizeof(float),
                            cudaMemcpyDeviceToHost, gs.stream);
            track_query_d2h(thread, rerank_n * sizeof(float));
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::transfer_rerank_d2h,
                t_final_rerank_d2h);
            const auto t_final_rerank_update = std::chrono::steady_clock::now();
            for (u32 i = 0; i < rerank_n; ++i) {
                beam[i].distance = gs.h_candidate_dists[i];
            }
            beam.resize(rerank_n);
            std::sort(beam.begin(), beam.end(),
                [](const auto& a, const auto& b) { return a.distance < b.distance; });
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::cpu_query_rerank_update,
                t_final_rerank_update);
            thread->stats.query_exact_reranks += rerank_n;
            thread->stats.query_rabitq_l2_candidates += rerank_n;
        }

        const auto t_finalize = std::chrono::steady_clock::now();
        auto& results = thread->query_results[q_id];
        results.clear();
        const auto t_result_ids = std::chrono::steady_clock::now();
        for (u32 i = 0; i < beam.size() && results.size() < k_; ++i) {
            if (direct_node_reads_) {
                const node_t id = co_await rdma::vamana::read_vamana_id(beam[i].rptr, thread);
                if (id == std::numeric_limits<node_t>::max()) {
                    continue;
                }
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
            if ((node->header() & VamanaNode::HEADER_DELETED) != 0) {
                continue;
            }
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
        vec<u32> gate_offsets(Q + 1, 0);
        vec<RemotePtr> gate_unvisited;
        gate_unvisited.reserve(Q * rabitq_gate_max_width_);
        vec<rabitq::QueryLut> rabitq_luts;
        vec<f32> rabitq_query_norm2s;
        if (use_rabitq_) {
            lib_assert(rabitq_cache_ != nullptr,
                       "RaBitQ batch gate requires a loaded budget sidecar");
            rabitq_luts.resize(Q);
            rabitq_query_norm2s.resize(Q, 0.0f);
            vec<f32> rotated(VamanaNode::rabitq_code_bits());
            for (u32 q = 0; q < Q; ++q) {
                VamanaNode::compute_rotated_query(query_datas[q], query_dtype,
                                                  rotated.data(), &rabitq_query_norm2s[q]);
                rabitq_luts[q] = rabitq::build_query_lut(rotated.data());
            }
        }

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

            const vec<RemotePtr>* distance_ptrs = &all_unvisited;
            const vec<u32>* distance_offsets = &batch_offsets;
            u32 distance_n_batch = n_batch;
            if (use_rabitq_) {
                const auto t_gate = std::chrono::steady_clock::now();
                gate_unvisited.clear();
                gate_offsets[0] = 0;
                const auto& quantization = rabitq_cache_->quantization();
                vec<f32> approximate_distances;
                vec<u32> cache_miss_indices;
                for (u32 q = 0; q < Q; ++q) {
                    const u32 begin = batch_offsets[q];
                    const u32 end = batch_offsets[q + 1];
                    const u32 count = end - begin;
                    approximate_distances.assign(count,
                        std::numeric_limits<f32>::infinity());
                    cache_miss_indices.clear();
                    for (u32 i = 0; i < count; ++i) {
                        const auto* entry = rabitq_cache_->find(all_unvisited[begin + i]);
                        if (entry == nullptr) {
                            cache_miss_indices.push_back(i);
                        } else {
                            approximate_distances[i] = rabitq::estimate_distance_lut(
                                rabitq_luts[q], rabitq_query_norm2s[q], *entry, quantization);
                        }
                    }
                    thread->stats.query_rabitq_l0_candidates += count;
                    thread->stats.query_rabitq_cache_misses += cache_miss_indices.size();
                    const vec<u32> selected = rabitq::select_gate(
                        approximate_distances, cache_miss_indices,
                        rabitq_gate_width_, rabitq_gate_max_width_, rabitq_gate_margin_);
                    for (u32 local_index : selected) {
                        gate_unvisited.push_back(all_unvisited[begin + local_index]);
                    }
                    gate_offsets[q + 1] = static_cast<u32>(gate_unvisited.size());
                }
                thread->stats.query_rabitq_l1_candidates += gate_unvisited.size();
                thread->stats.query_rabitq_l2_candidates += gate_unvisited.size();
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::cpu_query_rabitq_gate, t_gate);
                distance_ptrs = &gate_unvisited;
                distance_offsets = &gate_offsets;
                distance_n_batch = static_cast<u32>(gate_unvisited.size());
                if (distance_n_batch == 0) continue;
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
                *distance_ptrs, thread,
                use_gdr ? staging : nullptr, use_gdr ? staging_lkey : 0);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::rdma_vector_fetch, tvf);
            if (vr.direct_to_gpu) {
                thread->stats.query_rdma_to_staging_bytes +=
                    distance_n_batch * VamanaNode::vector_bytes();
            }

            // Phase 4: Q GPU kernels (one per query) + ONE D2H for all
            const size_t vec_sz = VamanaNode::vector_bytes();
            const auto t_gpu = std::chrono::steady_clock::now();
            byte_t* batched_queries = static_cast<byte_t*>(gs.h_query);
            for (u32 q = 0; q < Q; ++q) {
                std::memcpy(batched_queries + static_cast<size_t>(q) * q_bytes,
                            query_datas[q], q_bytes);
                for (u32 i = (*distance_offsets)[q]; i < (*distance_offsets)[q + 1]; ++i) {
                    gs.h_candidate_order[i] = q;
                }
            }
            cudaMemcpyAsync(gs.d_query, gs.h_query, static_cast<size_t>(Q) * q_bytes,
                            cudaMemcpyHostToDevice, gs.stream);
            track_query_h2d(thread, static_cast<size_t>(Q) * q_bytes);
            cudaMemcpyAsync(gs.d_candidate_order, gs.h_candidate_order,
                            distance_n_batch * sizeof(u32),
                            cudaMemcpyHostToDevice, gs.stream);
            track_query_h2d(thread, distance_n_batch * sizeof(u32));
            gpu::launch_batch_typed_multi_query_l2_distances(
                gs.stream, gs.event,
                gs.d_query, static_cast<u32>(query_dtype),
                gs.d_candidate_order,
                staging, static_cast<u32>(VamanaNode::vector_dtype()),
                gs.d_distances, distance_n_batch, dim_);
            co_await gpu::GpuAwaitable{thread.get()};
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::gpu_query_distance, t_gpu);
            thread->stats.distcomps += distance_n_batch;
            thread->stats.query_distcomps += distance_n_batch;

            // Phase 5: D2H + per-query beam update
            const auto t_d2h = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                            distance_n_batch * sizeof(float),
                            cudaMemcpyDeviceToHost, gs.stream);
            track_query_d2h(thread, distance_n_batch * sizeof(float));
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::transfer_distance_d2h, t_d2h);

            for (u32 q = 0; q < Q; ++q) {
                if (!qs[q].active) continue;
                auto& s = qs[q];
                for (u32 i = (*distance_offsets)[q]; i < (*distance_offsets)[q+1]; ++i)
                    insert_into_beam(s.beam, (*distance_ptrs)[i],
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
            for (u32 i = 0; i < s.beam.size() && results.size() < k_; ++i) {
                if (direct_node_reads_) {
                    const node_t id = co_await rdma::vamana::read_vamana_id(
                        s.beam[i].rptr, thread);
                    if (id == std::numeric_limits<node_t>::max()) {
                        continue;
                    }
                    results.push_back({id, s.beam[i].distance});
                    continue;
                }
                s_ptr<VamanaNode> node;
                auto coro = read_node(s.beam[i].rptr, node, thread, true);
                while (!coro.handle.done()) { co_await std::suspend_always{}; coro.handle.resume(); }
                if ((node->header() & VamanaNode::HEADER_DELETED) != 0) {
                    continue;
                }
                results.push_back({node->id(), s.beam[i].distance});
            }
        }
    }

    // =========================================================================
    // Insert
    // =========================================================================
