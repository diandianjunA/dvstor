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
        auto it = std::lower_bound(beam.begin(), beam.end(), dist,
            [](const VamanaCoroutine::BeamEntry& e, distance_t d) {
                return e.distance < d;
            });
        beam.insert(it, {rptr, dist, false});
        if (beam.size() > max_beam_width) beam.resize(max_beam_width);
    }

    // Insert or update: if rptr already in beam, update its distance
    // (only if new distance is smaller).  Otherwise insert normally.
    // Used for eager beam updates with estimated distances.
    static void upsert_beam(vec<VamanaCoroutine::BeamEntry>& beam,
                            const RemotePtr& rptr, distance_t dist,
                            u32 max_beam_width) {
        // Linear scan to find existing entry (beam ≤ 128, cheap)
        for (auto& e : beam) {
            if (e.rptr == rptr) {
                if (dist < e.distance) e.distance = dist;
                return;
            }
        }
        insert_into_beam(beam, rptr, dist, max_beam_width);
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

    static std::chrono::steady_clock::time_point breakdown_start(
                                    const u_ptr<ComputeThread>& thread) {
        const auto* sample = thread->current_breakdown_sample();
        return sample == nullptr || !sample->collects_breakdown()
            ? std::chrono::steady_clock::time_point{}
            : std::chrono::steady_clock::now();
    }

    static void add_breakdown_subcategory(const u_ptr<ComputeThread>& thread,
                                    const service::breakdown::Subcategory subcategory,
                                    const std::chrono::steady_clock::time_point start) {
        if (start == std::chrono::steady_clock::time_point{}) return;
        if (auto* sample = thread->current_breakdown_sample()) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
            sample->add_subcategory(subcategory, static_cast<u64>(elapsed));
        }
    }

    static void begin_query_gpu_kernel_timing(gpu::CoroutineGpuState& state) {
        lib_assert(cudaEventRecord(state.kernel_start_event, state.stream) == cudaSuccess,
                   "failed to record GPU kernel start event");
    }

    static void finish_query_gpu_kernel_timing(const u_ptr<ComputeThread>& thread,
                                                gpu::CoroutineGpuState& state) {
        float elapsed_ms = 0.0f;
        // GpuAwaitable resumes only after state.event completes, so this is a
        // non-blocking measurement of the interval recorded around the kernel.
        lib_assert(cudaEventElapsedTime(&elapsed_ms, state.kernel_start_event, state.event) == cudaSuccess,
                   "failed to measure GPU kernel time");
        if (auto* sample = thread->current_breakdown_sample()) {
            sample->add_gpu_kernel_time(static_cast<u64>(elapsed_ms * 1'000'000.0f));
        }
    }

    // =========================================================================
    // Full-node RDMA read kept as a nested coroutine so the search coroutine
    // retains its established suspension and frame layout.
    // =========================================================================

    MinorCoroutine read_node(RemotePtr rptr,
                             s_ptr<VamanaNode>& value,
                             const u_ptr<ComputeThread>& thread,
                             bool) const {
        value = co_await rdma::vamana::read_vamana_node(rptr, thread);
    }
