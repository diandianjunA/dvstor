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

