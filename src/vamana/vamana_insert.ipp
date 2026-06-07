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

        // Read medoid
        const auto t_medoid_ptr = std::chrono::steady_clock::now();
        RemotePtr medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_medoid_ptr, t_medoid_ptr);

        // Handle first insert (empty index)
        if (medoid_ptr.is_null()) {
            const auto t_alloc = std::chrono::steady_clock::now();
            RemotePtr new_ptr = co_await rdma::vamana::allocate_vamana_node(thread);
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_alloc, t_alloc);

            // Write node with no neighbors.
            const auto t_write = std::chrono::steady_clock::now();
            s_ptr<VamanaNode> new_node = co_await rdma::vamana::write_vamana_node(
                new_ptr, id, components, span<RemotePtr>{},
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
            const auto t_node_read = std::chrono::steady_clock::now();
            auto coro = read_node(medoid_ptr, medoid_node, thread, true);
            while (!coro.handle.done()) {
                co_await std::suspend_always{};
                coro.handle.resume();
            }
            add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_node_read, t_node_read);
        }

        const auto t_insert_init = std::chrono::steady_clock::now();
        distance_t medoid_dist = distance_to_stored_vector<Distance>(components, medoid_node->vector_data());
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
                    std::memcpy(gs.h_candidate_vecs + static_cast<size_t>(i) * VamanaNode::vector_bytes(),
                                vec_read.host_buffers[i],
                                VamanaNode::vector_bytes());
                    thread->buffer_allocator.free_buffer(vec_read.host_buffers[i], VamanaNode::vector_bytes());
                }
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_stage_candidates, t_stage_candidates);

                const auto t_candidate_h2d = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                                static_cast<size_t>(n_batch) * VamanaNode::vector_bytes(),
                                cudaMemcpyHostToDevice, gs.stream);
                track_build_h2d(thread, static_cast<u64>(n_batch) * VamanaNode::vector_bytes());
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_candidate_h2d, t_candidate_h2d);
            }

            const auto t_gpu_distance = std::chrono::steady_clock::now();
            gpu::launch_batch_typed_l2_distances(
                gs.stream, gs.event,
                static_cast<const float*>(gs.d_query), gs.d_candidate_vecs,
                static_cast<u32>(VamanaNode::vector_dtype()),
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
                std::memcpy(gs.h_candidate_vecs + static_cast<size_t>(i) * VamanaNode::vector_bytes(),
                           candidate_read.host_buffers[i],
                           VamanaNode::vector_bytes());
                thread->buffer_allocator.free_buffer(candidate_read.host_buffers[i], VamanaNode::vector_bytes());
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
                           static_cast<size_t>(n_candidates) * VamanaNode::vector_bytes(),
                           cudaMemcpyHostToDevice, gs.stream);
        }
        cudaMemcpyAsync(gs.d_candidate_dists, gs.h_candidate_dists,
                       n_candidates * sizeof(float),
                       cudaMemcpyHostToDevice, gs.stream);
        const size_t prune_h2d_bytes =
            (candidate_read.direct_to_gpu ? 0 : (static_cast<size_t>(n_candidates) * VamanaNode::vector_bytes())) + n_candidates * sizeof(float);
        track_build_h2d(thread, prune_h2d_bytes);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_prune_prepare, t_prune_prepare);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_prune_h2d, t_prune_h2d);

        const auto t_gpu_prune = std::chrono::steady_clock::now();
        gpu::launch_robust_prune_typed(
            gs.stream, gs.event,
            gs.d_candidate_vecs,
            static_cast<u32>(VamanaNode::vector_dtype()),
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

        // Phase 3: Allocate and write new node
        const auto t_alloc = std::chrono::steady_clock::now();
        RemotePtr new_ptr = co_await rdma::vamana::allocate_vamana_node(thread);
        add_breakdown_subcategory(thread, service::breakdown::Subcategory::rdma_alloc, t_alloc);

        const auto t_new_write = std::chrono::steady_clock::now();
        s_ptr<VamanaNode> new_node = co_await rdma::vamana::write_vamana_node(
            new_ptr, id, components,
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
                const u64 selected_upper_bound = std::min<u64>(R_, n_all);
                const u64 pair_checks_upper_bound =
                    selected_upper_bound * static_cast<u64>(n_all) -
                    selected_upper_bound * (selected_upper_bound + 1) / 2;
                const u64 bytes_per_pair_upper_bound =
                    2ull * static_cast<u64>(VamanaNode::vector_bytes()) + sizeof(float) + 2ull * sizeof(u32);
                thread->stats.build_overflow_prune_candidates += n_all;
                thread->stats.build_overflow_prune_max_candidates =
                    std::max(thread->stats.build_overflow_prune_max_candidates, static_cast<size_t>(n_all));
                thread->stats.build_overflow_prune_pair_checks_upper_bound += pair_checks_upper_bound;
                thread->stats.build_overflow_prune_global_load_bytes_upper_bound +=
                    pair_checks_upper_bound * bytes_per_pair_upper_bound;
                const size_t prune_threads = std::min<size_t>(GPU_BLOCK_SIZE, n_all);
                thread->stats.build_overflow_prune_kernel_blocks += 1;
                thread->stats.build_overflow_prune_kernel_threads += prune_threads;
                thread->stats.build_overflow_prune_max_kernel_threads =
                    std::max(thread->stats.build_overflow_prune_max_kernel_threads, prune_threads);
                if (auto* sample = thread->current_breakdown_sample()) {
                    sample->overflow_prune_max_candidates =
                        std::max<u64>(sample->overflow_prune_max_candidates, n_all);
                    sample->overflow_prune_max_kernel_threads =
                        std::max<u64>(sample->overflow_prune_max_kernel_threads, prune_threads);
                }
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
                decode_storage_vector_to_float(neighbor_node->vector_data(), VamanaNode::vector_dtype(), dim_,
                                               static_cast<float*>(gs.h_query));
                cudaMemcpyAsync(gs.d_query, gs.h_query, dim_ * sizeof(float),
                               cudaMemcpyHostToDevice, gs.stream);
                track_build_h2d(thread, dim_ * sizeof(float));
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_overflow_query_h2d, t_overflow_query_h2d);

                // Compute distances from neighbor to all candidates
                const auto t_stage_overflow = std::chrono::steady_clock::now();
                for (u32 i = 0; i < n_all; ++i) {
                    if (i + 1 == n_all) {
                        encode_float_vector_to_storage(components.data(), dim_, VamanaNode::vector_dtype(),
                                                       gs.h_candidate_vecs + static_cast<size_t>(i) * VamanaNode::vector_bytes());
                    } else {
                        std::memcpy(gs.h_candidate_vecs + static_cast<size_t>(i) * VamanaNode::vector_bytes(),
                                   all_vec_bufs[i],
                                   VamanaNode::vector_bytes());
                    }
                }
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::cpu_insert_stage_candidates, t_stage_overflow);

                const auto t_overflow_candidate_h2d = std::chrono::steady_clock::now();
                cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                               static_cast<size_t>(n_all) * VamanaNode::vector_bytes(),
                               cudaMemcpyHostToDevice, gs.stream);
                track_build_h2d(thread, static_cast<u64>(n_all) * VamanaNode::vector_bytes());
                add_breakdown_subcategory(thread, service::breakdown::Subcategory::transfer_overflow_candidate_h2d, t_overflow_candidate_h2d);

                const auto t_overflow_gpu_distance = std::chrono::steady_clock::now();
                gpu::launch_batch_typed_l2_distances(
                    gs.stream, gs.event,
                    static_cast<const float*>(gs.d_query), gs.d_candidate_vecs,
                    static_cast<u32>(VamanaNode::vector_dtype()),
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
                    thread->buffer_allocator.free_buffer(all_vec_bufs[i], VamanaNode::vector_bytes());
                }

                const auto t_overflow_gpu_prune = std::chrono::steady_clock::now();
                gpu::launch_robust_prune_typed(
                    gs.stream, gs.event,
                    gs.d_candidate_vecs,
                    static_cast<u32>(VamanaNode::vector_dtype()),
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
