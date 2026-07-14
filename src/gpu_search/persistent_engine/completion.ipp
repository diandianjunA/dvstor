  void report_direct_path_failure() {
    if (direct_disabled_host == nullptr || direct_disabled_device == nullptr ||
        direct_error_host == nullptr || direct_error_device == nullptr) return;
    check_cuda(cudaMemcpyAsync(direct_disabled_host, direct_disabled_device,
                               sizeof(u32), cudaMemcpyDeviceToHost, rdma_stream),
               "cudaMemcpyAsync(GPUNetIO failure flag)");
    check_cuda(cudaMemcpyAsync(direct_error_host, direct_error_device,
                               sizeof(i32), cudaMemcpyDeviceToHost, rdma_stream),
               "cudaMemcpyAsync(GPUNetIO failure status)");
    check_cuda(cudaStreamSynchronize(rdma_stream),
               "cudaStreamSynchronize(GPUNetIO failure status)");
    if (*direct_disabled_host == 0) return;
    bool expected = false;
    if (!direct_failure_logged.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) return;
    const i32 direct_error = *direct_error_host;
    std::cerr << "[gpu-search] GPUNetIO direct read failed with status=" << direct_error
              << "; strict GPUNetIO mode rejects the query\n";
    engine.telemetry_.direct_path_failures.fetch_add(1, std::memory_order_relaxed);
    mark_unhealthy("GPUNetIO direct read failed with status " +
                   std::to_string(direct_error));
  }

  void completion_loop() {
    while (!shutdown.load(std::memory_order_acquire) ||
           pending_count.load(std::memory_order_acquire) != 0) {
      CompletionDescriptor completion;
      if (!completions.try_pop(completion)) {
        std::this_thread::yield();
        continue;
      }
      if (completion.status != 0) report_direct_path_failure();
      std::shared_ptr<PendingQuery> pending;
      {
        std::lock_guard<std::mutex> lock(pending_mutex);
        const auto it = pending_queries.find(completion.request_id);
        if (it != pending_queries.end()) {
          pending = std::move(it->second);
          pending_queries.erase(it);
        }
      }
      if (!pending) {
        if (completion.query_slot < query_slots) {
          active_query_tickets[completion.query_slot].store(
            0, std::memory_order_release);
          active_query_snapshots[completion.query_slot].store(
            0, std::memory_order_release);
        }
        active_gpu_queries.fetch_sub(1, std::memory_order_release);
        maintenance_cv.notify_all();
        continue;
      }
      const auto completed_at = std::chrono::steady_clock::now();
      const u64 gpu_ns = completion.gpu_cycles * 1000000ULL / gpu_clock_khz;
      const auto phase_ns = [&](u64 cycles) {
        return cycles * 1000000ULL / gpu_clock_khz;
      };
      const u64 end_to_end_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          completed_at - pending->submitted_at).count());
      if (end_to_end_ns >= 10000000ULL &&
          slow_query_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
        std::cerr << "[gpu-search] slow query e2e_us=" << end_to_end_ns / 1000
                  << " gpu_us=" << gpu_ns / 1000
                  << " prepare_us=" << completion.prepare_cycles * 1000ULL / gpu_clock_khz
                  << " graph_us=" << completion.graph_cycles * 1000ULL / gpu_clock_khz
                  << " score_us=" << completion.score_cycles * 1000ULL / gpu_clock_khz
                  << " beam_us=" << completion.beam_cycles * 1000ULL / gpu_clock_khz
                  << " exact_us=" << completion.exact_cycles * 1000ULL / gpu_clock_khz
                  << " graph_reads=" << completion.remote_pages
                  << " graph_batches=" << completion.remote_batches
                  << " graph_rounds=" << completion.graph_rounds
                  << " graph_hits=" << completion.cache_hits
                  << " route_hits=" << completion.route_hits
                  << " exact_reads=" << completion.exact_vectors
                  << " exact_hits=" << completion.exact_cache_hits << '\n';
      }
      try {
        if (completion.status != 0) {
          const std::string message = "persistent GPU query failed with status " +
            std::to_string(completion.status);
          mark_unhealthy(message);
          throw std::runtime_error(message);
        }
        const size_t offset = static_cast<size_t>(pending->slot) * result_capacity;
        service::QueryResult result;
        result.reserve(completion.result_count);
        for (u32 index = 0; index < completion.result_count; ++index) {
          result.push_back({result_ids_host[offset + index],
                            result_distances_host[offset + index]});
        }
        pending->promise.set_value(std::move(result));
      } catch (...) {
        pending->promise.set_exception(std::current_exception());
      }
      {
        active_query_tickets[pending->slot].store(0, std::memory_order_release);
        active_query_snapshots[pending->slot].store(0, std::memory_order_release);
        std::lock_guard<std::mutex> lock(slot_mutex);
        free_slots.push_back(pending->slot);
      }
      slot_cv.notify_one();
      pending_count.fetch_sub(1, std::memory_order_release);
      active_gpu_queries.fetch_sub(1, std::memory_order_release);
      maintenance_cv.notify_all();
      engine.telemetry_.queries_completed.fetch_add(1, std::memory_order_relaxed);
      engine.telemetry_.gpu_active_ns.fetch_add(gpu_ns, std::memory_order_relaxed);
      engine.telemetry_.gpu_prepare_ns.fetch_add(
        phase_ns(completion.prepare_cycles), std::memory_order_relaxed);
      engine.telemetry_.gpu_graph_ns.fetch_add(
        phase_ns(completion.graph_cycles), std::memory_order_relaxed);
      engine.telemetry_.gpu_score_ns.fetch_add(
        phase_ns(completion.score_cycles), std::memory_order_relaxed);
      engine.telemetry_.gpu_beam_ns.fetch_add(
        phase_ns(completion.beam_cycles), std::memory_order_relaxed);
      engine.telemetry_.gpu_exact_ns.fetch_add(
        phase_ns(completion.exact_cycles), std::memory_order_relaxed);
      engine.telemetry_.completion_wait_ns.fetch_add(end_to_end_ns,
                                                     std::memory_order_relaxed);
      if (completion.snapshot_epoch != 0) {
        engine.telemetry_.delta_queries.fetch_add(1, std::memory_order_relaxed);
      }
      engine.telemetry_.rdma_read_ops.fetch_add(
        static_cast<u64>(completion.exact_vectors) + completion.remote_pages,
        std::memory_order_relaxed);
      engine.telemetry_.rdma_read_bytes.fetch_add(
        static_cast<u64>(completion.exact_vectors) * node_record_bytes +
        static_cast<u64>(completion.remote_pages) * index.layout.graph_entry_bytes,
        std::memory_order_relaxed);
      if (completion.remote_pages > completion.remote_batches) {
        engine.telemetry_.rdma_merged_requests.fetch_add(
          completion.remote_pages - completion.remote_batches,
          std::memory_order_relaxed);
      }
      engine.telemetry_.exact_vector_reads.fetch_add(completion.exact_vectors,
                                                     std::memory_order_relaxed);
      engine.telemetry_.graph_page_requests.fetch_add(completion.remote_pages,
                                                      std::memory_order_relaxed);
      engine.telemetry_.graph_dependency_rounds.fetch_add(
        completion.graph_rounds, std::memory_order_relaxed);
      engine.telemetry_.graph_page_cache_hits.fetch_add(completion.cache_hits,
                                                        std::memory_order_relaxed);
      engine.telemetry_.graph_route_hits.fetch_add(completion.route_hits,
                                                   std::memory_order_relaxed);
      engine.telemetry_.exact_vector_cache_hits.fetch_add(completion.exact_cache_hits,
                                                          std::memory_order_relaxed);
    }
  }

