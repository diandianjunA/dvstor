    VamanaCoroutine knn(node_t q_id, const span<element_t> components,
                        const u_ptr<ComputeThread>& thread,
                        const vec<RemotePtr>* entry_points = nullptr) const {
        return knn_raw(q_id, reinterpret_cast<const byte_t*>(components.data()),
                       VectorDType::float32, thread, entry_points);
    }

    VamanaCoroutine knn_raw(node_t q_id, const byte_t* query_data, VectorDType query_dtype,
                            const u_ptr<ComputeThread>& thread,
                            const vec<RemotePtr>* entry_points = nullptr) const {
        ++thread->stats.processed;
        ++thread->stats.processed_queries;

        auto& coro_state = thread->current_vamana_coroutine();
        auto& beam = coro_state.beam;
        auto& visited = coro_state.visited_nodes;
        auto& gpu = thread->gpu_buffers;
        const u32 coro_id = thread->current_coroutine_id();
        auto& gs = gpu.state(coro_id);
        if (observe_device_utilization_) {
            if (auto* sample = thread->current_breakdown_sample()) {
                sample->set_device_utilization_observed();
            }
        }
        const bool use_gpudirect_candidate_rdma =
            gpu.gpudirect_candidate_ready() && gs.d_candidate_vecs_rdma_registered;

        if (!use_rabitq_) {
            const size_t query_bytes = vector_dtype_bytes(query_dtype, dim_);
            const auto t_query_h2d = breakdown_start(thread);
            std::memcpy(reinterpret_cast<byte_t*>(gs.h_query), query_data, query_bytes);
            cudaMemcpyAsync(gs.d_query, gs.h_query, query_bytes, cudaMemcpyHostToDevice, gs.stream);
            track_query_h2d(thread, query_bytes);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_query_h2d, t_query_h2d);
        }

        beam.clear();
        visited.clear();

        auto add_start_point = [&](RemotePtr ptr) -> VamanaCoroutine {
            if (ptr.is_null() || visited.contains(ptr)) {
                co_return;
            }
            s_ptr<VamanaNode> node;
            const auto t_node_read = breakdown_start(thread);
            auto coro = read_node(ptr, node, thread, true);
            while (!coro.handle.done()) {
                co_await std::suspend_always{};
                coro.handle.resume();
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_node_read, t_node_read);
            const distance_t dist =
                distance_to_stored_vector<Distance>(query_data, query_dtype, node->vector_data());
            ++thread->stats.distcomps;
            ++thread->stats.query_distcomps;
            beam.push_back({ptr, dist, false});
            visited.insert(ptr);
        };

        if (entry_points != nullptr) {
            for (const RemotePtr& entry : *entry_points) {
                auto add = add_start_point(entry);
                while (!add.handle.done()) {
                    co_await std::suspend_always{};
                    add.handle.resume();
                }
                add.handle.destroy();
            }
        }
        if (beam.empty()) {
            const auto t_medoid_ptr_start = breakdown_start(thread);
            RemotePtr medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_medoid_ptr, t_medoid_ptr_start);
            auto add = add_start_point(medoid_ptr);
            while (!add.handle.done()) {
                co_await std::suspend_always{};
                add.handle.resume();
            }
            add.handle.destroy();
        }
        std::sort(beam.begin(), beam.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.distance < rhs.distance;
        });

        // ── K-way Batched Beam Expansion ──────────────────────────────
        // Instead of expanding one node per iteration (serial beam search),
        // expand the top-K unexpanded nodes from the beam in each iteration.
        // This batches K neighbour RDMA reads, K vector batches, and K
        // GPU distance computations into one GPU kernel launch and one D2H
        // transfer, reducing per-iteration overhead by K×.
        //
        const u32 K = std::max<u32>(1, expansion_batch_);
        struct CreditExpansionController {
            bool enabled{};
            u32 min_k{1};
            u32 max_k{1};
            u32 issue_k{1};
            u32 max_lookahead{};
            u32 lookahead_k{};
            u32 target_candidates{};
            u32 graph_degree{};
            u32 no_progress_streak{};
            bool cost_guard{};
            f32 cost_max_extra_ratio{1.05f};
            u32 cost_probe_rounds{4};
            f32 baseline_cost_per_expansion{};
            f32 ewma_cost_per_expansion{};
            u32 baseline_samples{};

            f32 round_cost_per_expansion(const u32 issued_expansions,
                                         const u32 frontier_candidates,
                                         const u32 exact_candidates) const {
                if (issued_expansions == 0) return 0.0f;
                const f32 expansion_cost =
                    static_cast<f32>(issued_expansions) * static_cast<f32>(graph_degree);
                const f32 frontier_cost = static_cast<f32>(frontier_candidates);
                const f32 exact_cost = 2.0f * static_cast<f32>(exact_candidates);
                return (expansion_cost + frontier_cost + exact_cost) /
                    static_cast<f32>(issued_expansions);
            }

            u32 issue_width() const {
                return enabled ? std::clamp(issue_k, min_k, max_k) : max_k;
            }

            u32 precommit_width() const {
                if (!enabled) return max_k;
                return std::min(lookahead_k, issue_width());
            }

            void record_round(const u32 issued_expansions,
                              const u32 frontier_candidates,
                              const u32 exact_candidates,
                              const bool credit_stall,
                              const bool progressed,
                              statistics::ThreadStatistics& stats) {
                if (!enabled) return;
                ++stats.query_credit_rounds;
                if (credit_stall) ++stats.query_credit_credit_stalls;
                const f32 cost_per_expansion = round_cost_per_expansion(
                    issued_expansions, frontier_candidates, exact_candidates);
                if (cost_guard && issued_expansions > 0 && cost_per_expansion > 0.0f) {
                    if (issued_expansions == min_k) {
                        const f32 alpha = baseline_samples == 0 ? 1.0f : 0.25f;
                        baseline_cost_per_expansion =
                            baseline_cost_per_expansion == 0.0f
                                ? cost_per_expansion
                                : baseline_cost_per_expansion * (1.0f - alpha) +
                                    cost_per_expansion * alpha;
                        ++baseline_samples;
                        ++stats.query_credit_cost_baseline_samples;
                    }
                    const f32 alpha = ewma_cost_per_expansion == 0.0f ? 1.0f : 0.125f;
                    ewma_cost_per_expansion =
                        ewma_cost_per_expansion == 0.0f
                            ? cost_per_expansion
                            : ewma_cost_per_expansion * (1.0f - alpha) +
                                cost_per_expansion * alpha;
                }
                const bool underfilled = target_candidates > 0 &&
                    frontier_candidates * 4u <= target_candidates * 3u;
                const bool overfilled = target_candidates > 0 &&
                    frontier_candidates * 4u > target_candidates * 5u;
                const bool cost_guard_ready = cost_guard &&
                    baseline_samples >= cost_probe_rounds &&
                    baseline_cost_per_expansion > 0.0f &&
                    ewma_cost_per_expansion > 0.0f;
                const bool cost_too_high = cost_guard_ready &&
                    ewma_cost_per_expansion >
                        baseline_cost_per_expansion * cost_max_extra_ratio;
                if (underfilled) ++stats.query_credit_underfilled_rounds;
                if (overfilled) ++stats.query_credit_overfilled_rounds;
                if (!progressed) {
                    ++no_progress_streak;
                    ++stats.query_credit_no_progress_rounds;
                } else {
                    no_progress_streak = 0;
                }

                if (cost_too_high) {
                    ++stats.query_credit_cost_guard_events;
                }

                if ((credit_stall || overfilled || cost_too_high ||
                     no_progress_streak >= 2) &&
                    issue_k > min_k) {
                    --issue_k;
                    ++stats.query_credit_shrink_events;
                } else if (!credit_stall && !overfilled && !cost_too_high &&
                           underfilled && progressed && issue_k < max_k) {
                    ++issue_k;
                    ++stats.query_credit_grow_events;
                } else if (cost_too_high && underfilled && progressed && issue_k < max_k) {
                    ++stats.query_credit_cost_growth_blocked;
                }

                const bool shrink_lookahead =
                    credit_stall || cost_too_high || no_progress_streak >= 1;
                if (shrink_lookahead && lookahead_k > 0) {
                    --lookahead_k;
                } else if (!shrink_lookahead && progressed &&
                           lookahead_k < std::min(max_lookahead, issue_width())) {
                    ++lookahead_k;
                }
            }
        };
        CreditExpansionController credit{};
        credit.enabled = credit_aware_expansion_;
        credit.min_k = std::min(std::max<u32>(1, credit_aware_min_k_), K);
        credit.max_k = credit_aware_max_k_ == 0 ? K : std::min(credit_aware_max_k_, K);
        credit.max_k = std::max(credit.max_k, credit.min_k);
        credit.issue_k = credit.enabled
            ? std::clamp<u32>(std::max<u32>(1, credit.max_k / 2u), credit.min_k, credit.max_k)
            : credit.max_k;
        credit.target_candidates = credit_aware_target_candidates_ == 0
            ? std::max<u32>(R_, (R_ * credit.max_k) / 2u)
            : credit_aware_target_candidates_;
        credit.graph_degree = R_;
        credit.max_lookahead = credit_aware_max_lookahead_ == 0
            ? std::min<u32>(credit.max_k, std::max<u32>(1, credit.max_k / 2u))
            : std::min(credit_aware_max_lookahead_, credit.max_k);
        credit.lookahead_k = credit.enabled
            ? std::min(credit.max_lookahead, credit.issue_k)
            : credit.max_k;
        credit.cost_guard = credit_aware_cost_guard_;
        credit.cost_max_extra_ratio = credit_aware_cost_max_extra_ratio_;
        credit.cost_probe_rounds = credit_aware_cost_probe_rounds_;

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
        auto best_beam_distance = [&beam]() -> distance_t {
            return beam.empty()
                ? std::numeric_limits<distance_t>::max()
                : beam.front().distance;
        };
        const bool use_indirect_candidate_path =
            use_gpudirect_candidate_rdma && thread->reserved_query_state[1] != nullptr;

        vec<rdma::vamana::NeighborReadAwaitable> pf_neighbors(K);
        u32 pending_K = 0;
        u32 rabitq_warmup_remaining = rabitq_warmup_exact_expansions_;
        u32 rabitq_expansions_seen = 0;
        u32 rabitq_next_audit_expansion = rabitq_audit_period_ == 0
            ? 0 : rabitq_warmup_exact_expansions_ + rabitq_audit_period_;
        i32 best_idx = -1;
        auto issue_plain_neighbor_reads = [&](u32 desired, u32 start_slot,
                                             bool precommit) -> u32 {
            desired = std::min(desired, K);
            u32 slot = std::min(start_slot, desired);
            for (; slot < desired; ++slot) {
                best_idx = select_best();
                if (best_idx < 0) break;
                beam[best_idx].expanded = true;
                pf_neighbors[slot] = rdma::vamana::read_vamana_neighbors(
                    beam[best_idx].rptr, &thread);
            }
            const u32 issued = slot > start_slot ? slot - start_slot : 0;
            if (credit.enabled && issued > 0) {
                thread->stats.query_credit_expansions_issued += issued;
                if (precommit) {
                    thread->stats.query_credit_precommit_expansions += issued;
                } else {
                    thread->stats.query_credit_postcommit_expansions += issued;
                }
            }
            return slot;
        };

        // ── Cold start: read the first neighbour ───────────────────────
        best_idx = select_best();
        if (best_idx < 0) co_return;
        beam[best_idx].expanded = true;
        pf_neighbors[0] = rdma::vamana::read_vamana_neighbors(
            beam[best_idx].rptr, &thread);
        pending_K = 1;

        // Pre-compute asymmetric RaBitQ rotated query + norm once.
        float rabitq_query_norm2 = 0.0f;
        bool exact_query_uploaded = false;
        rabitq::QueryLut rabitq_query_lut{};
        if (use_rabitq_) {
            lib_assert(rabitq_cache_ != nullptr,
                       "RaBitQ CPU gate requires a loaded RFQ5 sidecar");
            const auto t_query_h2d = breakdown_start(thread);
            auto* rabitq_rotated_query = static_cast<float*>(gs.h_query);
            VamanaNode::compute_rotated_query(query_data, query_dtype,
                                              rabitq_rotated_query, &rabitq_query_norm2);
            const u32 cache_code_bits = static_cast<u32>(rabitq_cache_->code_bits());
            rabitq_query_lut = rabitq::build_query_lut(rabitq_rotated_query, cache_code_bits);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::cpu_query_rabitq_gate, t_query_h2d);
        }

        while (true) {
            // ── Phase 1: consume neighbour reads → filter ──────────────
            auto& all_unvisited = coro_state.scratch_unvisited;
            all_unvisited.clear();
            const u32 consumed_K = pending_K;
            for (u32 k = 0; k < pending_K; ++k) {
                thread->poll_cq();
                if (thread->post_balances[thread->current_coroutine_id()].load(
                        std::memory_order_acquire) == 0) {
                    pf_neighbors[k].mark_ready();
                }
                const auto t_nf = breakdown_start(thread);
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
            rabitq_expansions_seen += consumed_K;

            if (all_unvisited.empty()) {
                credit.record_round(consumed_K, 0, 0, false, false, thread->stats);
                if (credit.enabled) {
                    pending_K = issue_plain_neighbor_reads(credit.issue_width(), 0, false);
                } else {
                    best_idx = select_best();
                    if (best_idx >= 0) {
                        beam[best_idx].expanded = true;
                        pf_neighbors[0] = rdma::vamana::read_vamana_neighbors(
                            beam[best_idx].rptr, &thread);
                        pending_K = 1;
                    }
                }
                if (pending_K == 0) break;
                continue;
            }

            const u32 n_batch = static_cast<u32>(all_unvisited.size());

            if (use_rabitq_) {
                lib_assert(rabitq_cache_ != nullptr,
                           "RaBitQ gate requires a loaded RFQ5 sidecar");
                auto& exact_ptrs = coro_state.reserved_ptrs_a;
                auto& gate_indices = coro_state.scratch_indices_b;
                const bool warmup_exact = rabitq_warmup_remaining > 0;
                rabitq_warmup_remaining = consumed_K >= rabitq_warmup_remaining
                    ? 0 : rabitq_warmup_remaining - consumed_K;
                const bool audit_exact = !warmup_exact && rabitq_audit_period_ > 0 &&
                    rabitq_expansions_seen >= rabitq_next_audit_expansion;
                if (audit_exact) {
                    while (rabitq_expansions_seen >= rabitq_next_audit_expansion) {
                        rabitq_next_audit_expansion += rabitq_audit_period_;
                    }
                }
                const u32 gate_scale = std::max<u32>(1, consumed_K);
                const u32 effective_gate_width =
                    std::min<u32>(n_batch, rabitq_gate_width_ * gate_scale);
                const u32 effective_gate_max_width =
                    std::min<u32>(n_batch, std::max(rabitq_gate_max_width_ * gate_scale,
                                                    effective_gate_width));
                if (warmup_exact || audit_exact) {
                    gate_indices.clear();
                    gate_indices.reserve(n_batch);
                    exact_ptrs.clear();
                    exact_ptrs.reserve(n_batch);
                    for (u32 i = 0; i < n_batch; ++i) {
                        gate_indices.push_back(i);
                        exact_ptrs.push_back(all_unvisited[i]);
                    }
                    thread->stats.query_rabitq_l1_candidates += gate_indices.size();
                    if (warmup_exact) ++thread->stats.query_rabitq_forced_widen;
                    if (audit_exact) {
                        thread->stats.query_rabitq_audit_expansions += consumed_K;
                        thread->stats.query_rabitq_audit_candidates += n_batch;
                    }
                } else {
                    const auto t_gate = breakdown_start(thread);
                    auto& approximate_distances = coro_state.scratch_distances;
                    auto& cache_miss_indices = coro_state.scratch_indices_a;
                    auto& cached_order = coro_state.indirect_candidate_indices;
                    auto& gate_flags = coro_state.scratch_flags;
                    rabitq_cache_->estimate_batch_lut(
                        rabitq_query_lut, rabitq_query_norm2,
                        all_unvisited, 0, n_batch,
                        approximate_distances, cache_miss_indices,
                        coro_state.scratch_entry_ptrs);
                    thread->stats.query_rabitq_l0_candidates += n_batch;
                    thread->stats.query_rabitq_cache_misses += cache_miss_indices.size();
                    rabitq::select_gate_into(approximate_distances, cache_miss_indices,
                        effective_gate_width, effective_gate_max_width, rabitq_gate_margin_,
                        gate_indices, cached_order, gate_flags);
                    if (rabitq_strict_recall_ && gate_indices.size() < n_batch &&
                        gate_indices.size() < effective_gate_width) {
                        gate_flags.assign(n_batch, 0);
                        for (u32 index : gate_indices) gate_flags[index] = 1;
                        const u32 target = std::min<u32>(n_batch, effective_gate_width);
                        for (u32 index : cached_order) {
                            if (gate_indices.size() >= target) break;
                            if (!gate_flags[index]) {
                                gate_flags[index] = 1;
                                gate_indices.push_back(index);
                            }
                        }
                        ++thread->stats.query_rabitq_forced_widen;
                    }
                    thread->stats.query_rabitq_l1_candidates += gate_indices.size();
                    add_breakdown_subcategory(thread,
                        service::breakdown::Subcategory::cpu_query_rabitq_gate, t_gate);

                    exact_ptrs.clear();
                    exact_ptrs.reserve(gate_indices.size());
                    for (u32 index : gate_indices) exact_ptrs.push_back(all_unvisited[index]);
                }

                if (gate_indices.empty()) {
                    pending_K = issue_plain_neighbor_reads(credit.issue_width(), 0, false);
                    if (pending_K == 0) break;
                    continue;
                }

                const auto t_exact_fetch = breakdown_start(thread);
                const auto credit_waits_before = thread->stats.vector_rdma_credit_waits;
                const auto credit_completion_before =
                    thread->stats.vector_rdma_completion_token_waits;
                const auto credit_retries_before =
                    thread->stats.vector_rdma_post_send_retries;
                if (!exact_query_uploaded) {
                    const size_t exact_query_bytes = vector_dtype_bytes(query_dtype, dim_);
                    const auto t_exact_query_h2d = breakdown_start(thread);
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
                const bool exact_credit_stall =
                    thread->stats.vector_rdma_credit_waits != credit_waits_before ||
                    thread->stats.vector_rdma_completion_token_waits != credit_completion_before ||
                    thread->stats.vector_rdma_post_send_retries != credit_retries_before;
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::rdma_vector_fetch, t_exact_fetch);
                if (exact_vectors.direct_to_gpu) {
                    thread->stats.query_rdma_to_staging_bytes +=
                        gate_indices.size() * VamanaNode::vector_bytes();
                } else {
                    thread->stats.query_host_staging_fallback_bytes +=
                        gate_indices.size() * VamanaNode::vector_bytes();
                    const auto t_stage = breakdown_start(thread);
                    for (u32 i = 0; i < gate_indices.size(); ++i) {
                        std::memcpy(gs.h_candidate_vecs + i * VamanaNode::vector_bytes(),
                                    exact_vectors.host_buffers[i], VamanaNode::vector_bytes());
                        thread->buffer_allocator.free_buffer(
                            exact_vectors.host_buffers[i], VamanaNode::vector_bytes());
                    }
                    add_breakdown_subcategory(thread,
                        service::breakdown::Subcategory::cpu_query_stage_candidates, t_stage);
                    const auto t_h2d = breakdown_start(thread);
                    cudaMemcpyAsync(exact_staging, gs.h_candidate_vecs,
                                    gate_indices.size() * VamanaNode::vector_bytes(),
                                    cudaMemcpyHostToDevice, gs.stream);
                    track_query_h2d(thread, gate_indices.size() * VamanaNode::vector_bytes());
                    add_breakdown_subcategory(thread,
                        service::breakdown::Subcategory::transfer_candidate_h2d, t_h2d);
                }

                const auto t_exact = breakdown_start(thread);
                if (observe_device_utilization_) begin_query_gpu_kernel_timing(gs);
                gpu::launch_batch_typed_query_l2_distances(
                    gs.stream, gs.event, gs.d_query,
                    static_cast<u32>(query_dtype), exact_staging,
                    static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances, static_cast<u32>(gate_indices.size()), dim_);

                co_await gpu::GpuAwaitable{thread.get()};
                if (observe_device_utilization_) finish_query_gpu_kernel_timing(thread, gs);
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::gpu_query_distance, t_exact);

                const auto t_d2h = breakdown_start(thread);
                cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                                gate_indices.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, gs.stream);
                track_query_d2h(thread, gate_indices.size() * sizeof(float));
                cudaStreamSynchronize(gs.stream);
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::transfer_distance_d2h, t_d2h);
                const auto t_beam_update = breakdown_start(thread);
                const size_t beam_size_before_update = beam.size();
                const distance_t best_before_update = best_beam_distance();
                const distance_t cutoff_before_update = beam.size() >= beam_width_
                    ? beam.back().distance
                    : std::numeric_limits<distance_t>::max();
                for (u32 i = 0; i < gate_indices.size(); ++i) {
                    insert_into_beam(beam, exact_ptrs[i], gs.h_distances[i], beam_width_);
                }
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::cpu_query_beam_update, t_beam_update);
                thread->stats.distcomps += gate_indices.size();
                thread->stats.query_distcomps += gate_indices.size();
                thread->stats.query_exact_reranks += gate_indices.size();
                thread->stats.query_rabitq_l2_candidates += gate_indices.size();
                const distance_t cutoff_after_update = beam.size() >= beam_width_
                    ? beam.back().distance
                    : std::numeric_limits<distance_t>::max();
                const bool progressed =
                    beam.size() > beam_size_before_update ||
                    best_beam_distance() + std::numeric_limits<distance_t>::epsilon() <
                        best_before_update ||
                    cutoff_after_update + std::numeric_limits<distance_t>::epsilon() <
                        cutoff_before_update;
                credit.record_round(consumed_K, n_batch,
                                    static_cast<u32>(gate_indices.size()),
                                    exact_credit_stall, progressed, thread->stats);

                const u32 desired_issue = credit.issue_width();
                u32 issued_next = 0;
                for (u32 k = 0; k < desired_issue; ++k) {
                    best_idx = select_best();
                    if (best_idx < 0) break;
                    beam[best_idx].expanded = true;
                    const RemotePtr selected = beam[best_idx].rptr;
                    pf_neighbors[k] = rdma::vamana::read_vamana_neighbors(
                        selected, &thread);
                    pending_K = k + 1;
                    ++issued_next;
                }
                if (credit.enabled && issued_next > 0) {
                    thread->stats.query_credit_expansions_issued += issued_next;
                    thread->stats.query_credit_postcommit_expansions += issued_next;
                }
                if (pending_K == 0) break;
                continue;
            }

            // ── Phase 2: vector / RaBitQ RDMA ──────────────────────────
            gs.flip_query_candidate_buffer();
            uint8_t* staging = gs.current_query_candidate_vecs();
            const u32 staging_lkey = gs.current_query_candidate_vecs_lkey();

            const auto tvf = breakdown_start(thread);
            const auto vector_credit_waits_before = thread->stats.vector_rdma_credit_waits;
            const auto vector_completion_before =
                thread->stats.vector_rdma_completion_token_waits;
            const auto vector_retries_before =
                thread->stats.vector_rdma_post_send_retries;
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
                const auto tsc = breakdown_start(thread);
                for (u32 i = 0; i < n_batch; ++i) {
                    std::memcpy(gs.h_candidate_vecs + i * VamanaNode::vector_bytes(),
                                vr.host_buffers[i], VamanaNode::vector_bytes());
                    thread->buffer_allocator.free_buffer(
                        vr.host_buffers[i], VamanaNode::vector_bytes());
                }
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::cpu_query_stage_candidates, tsc);
                const auto th2d = breakdown_start(thread);
                cudaMemcpyAsync(staging, gs.h_candidate_vecs,
                    n_batch * VamanaNode::vector_bytes(),
                    cudaMemcpyHostToDevice, gs.stream);
                track_query_h2d(thread, n_batch * VamanaNode::vector_bytes());
                add_breakdown_subcategory(thread,
                    service::breakdown::Subcategory::transfer_candidate_h2d, th2d);
            }
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::rdma_vector_fetch, tvf);
            const bool vector_credit_stall =
                thread->stats.vector_rdma_credit_waits != vector_credit_waits_before ||
                thread->stats.vector_rdma_completion_token_waits != vector_completion_before ||
                thread->stats.vector_rdma_post_send_retries != vector_retries_before;

            // ── Phase 3: GPU ───────────────────────────────────────────
            const auto t_gpu = breakdown_start(thread);
            if (use_indirect_candidate_path) {
                for (u32 i = 0; i < n_batch; ++i)
                    gs.h_candidate_ptrs[i] = staging + i * VamanaNode::vector_bytes();
                cudaMemcpyAsync(const_cast<void**>(gs.d_candidate_ptrs),
                    gs.h_candidate_ptrs, n_batch * sizeof(void*),
                    cudaMemcpyHostToDevice, gs.stream);
                track_query_h2d(thread, n_batch * sizeof(void*));
                if (observe_device_utilization_) begin_query_gpu_kernel_timing(gs);
                gpu::launch_batch_typed_query_l2_distances_indirect(
                    gs.stream, gs.event, gs.d_query,
                    static_cast<u32>(query_dtype),
                    gs.d_candidate_ptrs,
                    static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances, n_batch, dim_);
            } else {
                if (observe_device_utilization_) begin_query_gpu_kernel_timing(gs);
                gpu::launch_batch_typed_query_l2_distances(
                    gs.stream, gs.event, gs.d_query,
                    static_cast<u32>(query_dtype),
                    staging, static_cast<u32>(VamanaNode::vector_dtype()),
                    gs.d_distances, n_batch, dim_);
            }
            co_await gpu::GpuAwaitable{thread.get()};
            if (observe_device_utilization_) finish_query_gpu_kernel_timing(thread, gs);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::gpu_query_distance, t_gpu);
            thread->stats.distcomps += n_batch;
            thread->stats.query_distcomps += n_batch;

            // ── Phase 3: issue neighbour reads for next iteration ────
            // Issued before D2H so the RDMA overlaps with
            // cudaMemcpyAsync + cudaStreamSynchronize (8-10μs).
            pending_K = issue_plain_neighbor_reads(credit.precommit_width(), 0, true);

            const auto t_d2h = breakdown_start(thread);
            cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                            n_batch * sizeof(float),
                            cudaMemcpyDeviceToHost, gs.stream);
            track_query_d2h(thread, n_batch * sizeof(float));
            cudaStreamSynchronize(gs.stream);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::transfer_distance_d2h, t_d2h);

            const auto t_bu = breakdown_start(thread);
            const size_t beam_size_before_update = beam.size();
            const distance_t best_before_update = best_beam_distance();
            const distance_t cutoff_before_update = beam.size() >= beam_width_
                ? beam.back().distance
                : std::numeric_limits<distance_t>::max();
            for (u32 i = 0; i < n_batch; ++i)
                insert_into_beam(beam, all_unvisited[i], gs.h_distances[i], beam_width_);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::cpu_query_beam_update, t_bu);
            const distance_t cutoff_after_update = beam.size() >= beam_width_
                ? beam.back().distance
                : std::numeric_limits<distance_t>::max();
            const bool progressed =
                beam.size() > beam_size_before_update ||
                best_beam_distance() + std::numeric_limits<distance_t>::epsilon() <
                    best_before_update ||
                cutoff_after_update + std::numeric_limits<distance_t>::epsilon() <
                    cutoff_before_update;
            credit.record_round(consumed_K, n_batch, n_batch,
                                vector_credit_stall, progressed, thread->stats);

            if (credit.enabled && pending_K < credit.issue_width()) {
                pending_K = issue_plain_neighbor_reads(credit.issue_width(), pending_K, false);
            }

            if (pending_K == 0) {
                if (credit.enabled) {
                    pending_K = issue_plain_neighbor_reads(credit.issue_width(), 0, false);
                    if (pending_K == 0) break;
                } else {
                    best_idx = select_best();
                    if (best_idx < 0) break;
                    beam[best_idx].expanded = true;
                    pf_neighbors[0] = rdma::vamana::read_vamana_neighbors(
                        beam[best_idx].rptr, &thread);
                    pending_K = 1;
                }
            }
        }

        const auto t_beam_sort = breakdown_start(thread);
        std::sort(beam.begin(), beam.end(), [](const auto& a, const auto& b) { return a.distance < b.distance; });
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_query_beam_sort, t_beam_sort);

        const auto t_finalize = breakdown_start(thread);
        auto& results = thread->query_results[q_id];
        results.clear();
        const auto t_result_ids = breakdown_start(thread);
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
            const auto t_node_read = breakdown_start(thread);
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
                rabitq_luts[q] = rabitq::build_query_lut(
                    rotated.data(), static_cast<u32>(rabitq_cache_->code_bits()));
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
                const auto t_gate = breakdown_start(thread);
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
                                rabitq_luts[q], rabitq_query_norm2s[q], entry, quantization);
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
            const auto tvf = breakdown_start(thread);
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
            const auto t_gpu = breakdown_start(thread);
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
            if (observe_device_utilization_) begin_query_gpu_kernel_timing(gs);
            gpu::launch_batch_typed_multi_query_l2_distances(
                gs.stream, gs.event,
                gs.d_query, static_cast<u32>(query_dtype),
                gs.d_candidate_order,
                staging, static_cast<u32>(VamanaNode::vector_dtype()),
                gs.d_distances, distance_n_batch, dim_);
            co_await gpu::GpuAwaitable{thread.get()};
            if (observe_device_utilization_) finish_query_gpu_kernel_timing(thread, gs);
            add_breakdown_subcategory(thread,
                service::breakdown::Subcategory::gpu_query_distance, t_gpu);
            thread->stats.distcomps += distance_n_batch;
            thread->stats.query_distcomps += distance_n_batch;

            // Phase 5: D2H + per-query beam update
            const auto t_d2h = breakdown_start(thread);
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
