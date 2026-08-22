#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

#include <cerrno>

namespace gpu_search {

using namespace persistent_engine_detail;

void PersistentSearchEngine::Impl::write_query_rdma_trace(
    const CompletionDescriptor& completion) {
  if (!query_rdma_trace_stream.is_open()) {
    return;
  }
  const u32 rdma_event_count = std::min(
    completion.trace_event_count, config.query_rdma_trace_events_per_query);
  const bool have_rdma_events =
    rdma_event_count != 0 && d_query_rdma_trace_events != nullptr;
  if (!have_rdma_events) return;

  std::vector<QueryRdmaTraceEvent> trace_events(rdma_event_count);
  if (have_rdma_events) {
    check_cuda(cudaMemcpy(
      trace_events.data(),
      d_query_rdma_trace_events +
        static_cast<size_t>(completion.query_slot) *
          config.query_rdma_trace_events_per_query,
      trace_events.size() * sizeof(QueryRdmaTraceEvent),
      cudaMemcpyDeviceToHost), "cudaMemcpy(query RDMA trace events)");
  }
  std::lock_guard<std::mutex> trace_lock(query_rdma_trace_mutex);
  query_rdma_trace_stream
    << "{\"type\":\"query\",\"request_id\":"
    << completion.request_id << ",\"query_slot\":"
    << completion.query_slot << ",\"status\":" << completion.status
    << ",\"event_count\":" << rdma_event_count
    << ",\"overflow\":" << completion.trace_overflow
    << ",\"gpu_cycles\":" << completion.gpu_cycles
    << ",\"gpu_clock_khz\":" << gpu_clock_khz
    << ",\"graph_rounds\":" << completion.graph_rounds
    << ",\"graph_reads\":" << completion.remote_pages
    << ",\"graph_batches\":" << completion.remote_batches
    << ",\"graph_read_retries\":" << completion.graph_read_retries
    << ",\"graph_read_bytes\":" << completion.graph_read_bytes
    << ",\"graph_live_extent_reads\":"
    << completion.graph_live_extent_reads
    << ",\"graph_full_record_reads\":"
    << completion.graph_full_record_reads
    << ",\"graph_extent_fallback_reads\":"
    << completion.graph_extent_fallback_reads
    << ",\"graph_extent_underhint_reads\":"
    << completion.graph_extent_underhint_reads
    << ",\"graph_extent_hint_promotions\":"
    << completion.graph_extent_hint_promotions
    << ",\"dynamic_graph_short_reads\":"
    << completion.dynamic_graph_short_reads
    << ",\"dynamic_graph_full_reads\":"
    << completion.dynamic_graph_full_reads
    << ",\"dynamic_graph_read_bytes\":"
    << completion.dynamic_graph_read_bytes
    << ",\"dynamic_graph_fallback_reads\":"
    << completion.dynamic_graph_fallback_reads
    << ",\"dynamic_graph_hint_promotions\":"
    << completion.dynamic_graph_hint_promotions
    << ",\"dynamic_graph_hint_demotions\":"
    << completion.dynamic_graph_hint_demotions
    << ",\"logical_expansions\":" << completion.logical_expansions
    << ",\"critical_graph_reads\":" << completion.critical_graph_reads
    << ",\"critical_graph_bytes\":" << completion.critical_graph_bytes
    << ",\"speculative_graph_reads\":"
    << completion.speculative_graph_reads
    << ",\"speculative_graph_bytes\":"
    << completion.speculative_graph_bytes
    << ",\"speculative_wasted_bytes\":"
    << completion.speculative_wasted_bytes
    << ",\"rdma_completion_latency_ns\":"
    << completion.rdma_completion_latency_ns
    << ",\"speculative_completion_latency_ns\":"
    << completion.speculative_completion_latency_ns
    << ",\"rdma_completion_groups\":"
    << completion.rdma_completion_groups
    << ",\"speculative_completion_groups\":"
    << completion.speculative_completion_groups
    << ",\"speculative_arrived\":" << completion.speculative_arrived
    << ",\"speculative_promoted\":" << completion.speculative_promoted
    << ",\"speculative_stale\":" << completion.speculative_stale
    << ",\"speculative_queue_rejects\":"
    << completion.speculative_queue_rejects
    << ",\"core_prefetch_reads\":" << completion.core_prefetch_reads
    << ",\"core_prefetch_bytes\":" << completion.core_prefetch_bytes
    << ",\"core_prefetch_arrived\":" << completion.core_prefetch_arrived
    << ",\"core_prefetch_promoted\":"
    << completion.core_prefetch_promoted
    << ",\"core_prefetch_stale\":" << completion.core_prefetch_stale
    << ",\"core_prefetch_queue_rejects\":"
    << completion.core_prefetch_queue_rejects
    << ",\"core_prefetch_waves\":" << completion.core_prefetch_waves
    << ",\"core_ready_waves\":" << completion.core_ready_waves
    << ",\"terminal_exact_cache_attempted_queries\":"
    << completion.terminal_exact_cache_attempted_queries
    << ",\"terminal_exact_cache_issued_records\":"
    << completion.terminal_exact_cache_issued_records
    << ",\"terminal_exact_cache_promoted_records\":"
    << completion.terminal_exact_cache_promoted_records
    << ",\"terminal_exact_cache_wasted_bytes\":"
    << completion.terminal_exact_cache_wasted_bytes
    << ",\"terminal_exact_cache_queue_rejects\":"
    << completion.terminal_exact_cache_queue_rejects
    << ",\"terminal_exact_cache_miss_records\":"
    << completion.terminal_exact_cache_miss_records
    << ",\"completion_score_batches\":"
    << completion.completion_score_batches
    << ",\"completion_score_candidates\":"
    << completion.completion_score_candidates
    << ",\"frontier_reusable_certificates\":"
    << completion.frontier_reusable_certificates
    << ",\"frontier_streamed_candidate_runs\":"
    << completion.frontier_streamed_candidate_runs
    << ",\"ordered_score_batches\":"
    << completion.ordered_score_batches
    << ",\"ordered_score_candidates\":"
    << completion.ordered_score_candidates
    << ",\"ooo_bypassed_parents\":"
    << (completion.status == 0
          ? completion.frontier_telemetry_reserved1 : 0u)
    << ",\"frontier_reusable_prefix_ranks\":"
    << completion.frontier_reusable_prefix_ranks
    << ",\"frontier_reusable_full_prefix_certificates\":"
    << completion.frontier_reusable_full_prefix_certificates
    << ",\"frontier_reusable_issued_certificates\":"
    << completion.frontier_reusable_issued_certificates
    << ",\"frontier_certificate_rejects\":"
    << (completion.status == 0
          ? completion.frontier_telemetry_reserved0 : 0u)
    << ",\"issue_epochs\":" << completion.issue_epochs
    << ",\"commit_epochs\":" << completion.commit_epochs
    << ",\"issue_width_sum\":" << completion.issue_width_sum
    << ",\"issue_width_capacity_sum\":"
    << completion.issue_width_capacity_sum
    << ",\"commit_width_sum\":" << completion.commit_width_sum
    << ",\"max_issue_width\":" << completion.max_issue_width
    << ",\"max_commit_width\":" << completion.max_commit_width
    << ",\"critical_rob_hits\":" << completion.critical_rob_hits
    << ",\"critical_misses\":" << completion.critical_misses
    << ",\"speculative_wait_cycles\":"
    << completion.speculative_wait_cycles
    << ",\"beam_selection_cycles\":" << completion.beam_selection_cycles
    << ",\"rdma_issue_cycles\":" << completion.rdma_issue_cycles
    << ",\"rdma_wait_cycles\":" << completion.rdma_wait_cycles
    << ",\"graph_validation_cycles\":"
    << completion.graph_validation_cycles
    << ",\"neighbor_decode_cycles\":" << completion.neighbor_decode_cycles
    << ",\"pq_score_cycles\":" << completion.pq_score_cycles
    << ",\"visited_cycles\":" << completion.visited_cycles
    << ",\"beam_merge_cycles\":" << completion.beam_merge_cycles
    << ",\"beam_merge_prepare_cycles\":"
    << completion.beam_merge_prepare_cycles
    << ",\"beam_merge_sort_cycles\":"
    << completion.beam_merge_sort_cycles
    << ",\"beam_merge_materialize_cycles\":"
    << completion.beam_merge_materialize_cycles
    << ",\"exact_cycles\":" << completion.exact_cycles
    << ",\"exact_snapshot_train_batches\":"
    << completion.exact_snapshot_train_batches
    << ",\"exact_snapshot_train_fallbacks\":"
    << completion.exact_snapshot_train_fallbacks
    << ",\"dynamic_code_cycles\":" << completion.dynamic_code_cycles
    << "}\n";
  for (const QueryRdmaTraceEvent& event : trace_events) {
    query_rdma_trace_stream
      << "{\"type\":\"shard_batch\",\"request_id\":" << event.request_id
      << ",\"route_attempt\":" << event.route_attempt
      << ",\"search_round\":" << event.search_round
      << ",\"snapshot_attempt\":" << event.snapshot_attempt
      << ",\"target_shard\":" << event.target_shard
      << ",\"parent_count\":" << event.parent_count
      << ",\"payload_bytes\":" << event.payload_bytes
      << ",\"minimum_bytes_per_parent\":"
      << event.minimum_bytes_per_parent
      << ",\"maximum_bytes_per_parent\":"
      << event.maximum_bytes_per_parent
      << ",\"issue_timestamp_ns\":" << event.issue_timestamp_ns
      << ",\"wait_phase_start_timestamp_ns\":"
      << event.wait_phase_start_timestamp_ns
      << ",\"completion_timestamp_ns\":" << event.completion_timestamp_ns
      << ",\"batch_process_start_timestamp_ns\":"
      << event.batch_process_start_timestamp_ns << "}\n";
  }
  query_rdma_trace_stream.flush();
}

namespace {

u64 steady_now_ns() {
  return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count());
}

owner_watchdog::Observation sample_owner_progress(
    const DirectOwnerProgress& progress) {
  return {
    .announced = static_cast<u64>(progress.announced),
    .dequeued = static_cast<u64>(progress.dequeued),
    .completed = static_cast<u64>(progress.completed),
    .heartbeat = static_cast<u64>(progress.heartbeat),
  };
}

const char* query_failure_reason_name(QueryFailureReason reason) {
  switch (reason) {
    case QueryFailureReason::none: return "none";
    case QueryFailureReason::invalid_descriptor: return "invalid_descriptor";
    case QueryFailureReason::route_snapshot_timeout:
      return "route_snapshot_timeout";
    case QueryFailureReason::route_no_seed: return "route_no_seed";
    case QueryFailureReason::graph_fetch: return "graph_fetch";
    case QueryFailureReason::dynamic_code_fetch: return "dynamic_code_fetch";
    case QueryFailureReason::exact_rerank_empty: return "exact_rerank_empty";
    case QueryFailureReason::exact_fetch: return "exact_fetch";
  }
  return "unknown";
}

}  // namespace

void PersistentSearchEngine::Impl::report_direct_path_failure() {
  if (direct_disabled_host == nullptr || direct_disabled_device == nullptr ||
      direct_error_host == nullptr || direct_error_device == nullptr) return;
  cudaError_t status = cudaMemcpyAsync(
    direct_disabled_host, direct_disabled_device, sizeof(u32),
    cudaMemcpyDeviceToHost, rdma_stream);
  if (status == cudaSuccess) {
    status = cudaMemcpyAsync(direct_error_host, direct_error_device,
                             sizeof(i32), cudaMemcpyDeviceToHost, rdma_stream);
  }
  if (status == cudaSuccess) status = cudaStreamSynchronize(rdma_stream);
  if (status != cudaSuccess) {
    mark_unhealthy(std::string("failed to inspect GPUNetIO failure state: ") +
                   cudaGetErrorString(status));
    return;
  }
  if (*direct_disabled_host == 0) return;
  bool expected = false;
  if (!direct_failure_logged.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel)) return;
  const i32 direct_error = *direct_error_host;
  const bool graph_snapshot_error = direct_error == -EBADMSG;
  std::cerr << "[gpu-search] "
            << (graph_snapshot_error
                  ? "graph snapshot validation failed after bounded rereads"
                  : "GPUNetIO direct read failed")
            << " with status=" << direct_error;
  if (direct_error == -ETIMEDOUT) {
    std::cerr << " after cq_timeout_ms="
              << kernel_params.direct_timeout_ns / 1'000'000ULL;
  }
  std::cerr
            << "; strict query mode rejects the query\n";
  engine.telemetry_.direct_path_failures.fetch_add(1, std::memory_order_relaxed);
  mark_unhealthy(std::string(graph_snapshot_error
                 ? "graph snapshot validation failed with status "
                 : "GPUNetIO direct read failed with status ") +
                 std::to_string(direct_error));
}

void PersistentSearchEngine::Impl::completion_loop() {
  try {
    bind_cuda_device("cudaSetDevice(GPU completion/watchdog)");
  } catch (const std::exception& exception) {
    mark_unhealthy(exception.what());
    return;
  }

  std::vector<owner_watchdog::Tracker> owner_trackers(
    direct_batch_queue_count);
  const u64 watchdog_timeout_ns = owner_watchdog::stall_timeout_ns(
    kernel_params.direct_timeout_ns);
  const u64 watchdog_poll_ns = std::max<u64>(
    1'000'000ULL, std::min<u64>(10'000'000ULL, watchdog_timeout_ns / 8));
  u64 next_watchdog_poll_ns = steady_now_ns() + watchdog_poll_ns;

  while (!shutdown.load(std::memory_order_acquire)) {
    const u64 now_ns = steady_now_ns();
    if (now_ns >= next_watchdog_poll_ns &&
        accepting.load(std::memory_order_acquire) &&
        healthy.load(std::memory_order_acquire) &&
        !query_stop.load(std::memory_order_acquire)) {
      next_watchdog_poll_ns = now_ns + watchdog_poll_ns;

      const cudaError_t stream_status = cudaStreamQuery(kernel_stream);
      if (stream_status == cudaSuccess) {
        mark_unhealthy(
          "persistent CUDA kernel terminated while query admission was active");
      } else if (stream_status != cudaErrorNotReady) {
        mark_unhealthy(std::string("persistent CUDA kernel failed: ") +
                       cudaGetErrorString(stream_status));
      } else if (direct_owner_progress_host != nullptr &&
                 d_direct_owner_progress != nullptr) {
        bool has_pending_query = false;
        for (u32 slot = 0; slot < query_slots; ++slot) {
          if (query_slot_states[slot].phase.load(std::memory_order_acquire) ==
              static_cast<u32>(QuerySlotPhase::pending)) {
            has_pending_query = true;
            break;
          }
        }
        if (!has_pending_query) continue;

        cudaError_t progress_status = cudaMemcpyAsync(
          direct_owner_progress_host, d_direct_owner_progress,
          static_cast<size_t>(direct_batch_queue_count) *
            sizeof(DirectOwnerProgress),
          cudaMemcpyDeviceToHost, rdma_stream);
        if (progress_status == cudaSuccess) {
          progress_status = cudaStreamSynchronize(rdma_stream);
        }
        if (progress_status != cudaSuccess) {
          mark_unhealthy(std::string("failed to sample GPUNetIO owner progress: ") +
                         cudaGetErrorString(progress_status));
          continue;
        }

        for (u32 owner = 0; owner < direct_batch_queue_count; ++owner) {
          const owner_watchdog::Observation observation =
            sample_owner_progress(direct_owner_progress_host[owner]);
          if (!owner_trackers[owner].observe(
                observation, now_ns, watchdog_timeout_ns)) {
            continue;
          }

          // Wake every GPU waiter before rejecting host slots. This is a
          // transport-wide fail-stop, but unlike the removed query-side timer
          // it is reached only after an owner with outstanding work made no
          // dequeue/completion progress for a transport-derived grace period.
          *direct_disabled_host = 1;
          *direct_error_host = -ETIMEDOUT;
          cudaError_t publish_status = cudaMemcpyAsync(
            direct_disabled_device, direct_disabled_host, sizeof(u32),
            cudaMemcpyHostToDevice, rdma_stream);
          if (publish_status == cudaSuccess) {
            publish_status = cudaMemcpyAsync(
              direct_error_device, direct_error_host, sizeof(i32),
              cudaMemcpyHostToDevice, rdma_stream);
          }
          if (publish_status == cudaSuccess) {
            publish_status = cudaStreamSynchronize(rdma_stream);
          }

          const u32 phase = direct_owner_phases_host == nullptr ? 0u :
            std::atomic_ref<u32>(direct_owner_phases_host[owner]).load(
              std::memory_order_acquire);
          const u64 outstanding = observation.announced >= observation.completed
            ? observation.announced - observation.completed : 0;
          std::ostringstream message;
          message << "GPUNetIO owner watchdog stalled owner=" << owner
                  << " outstanding=" << outstanding
                  << " announced=" << observation.announced
                  << " dequeued=" << observation.dequeued
                  << " completed=" << observation.completed
                  << " heartbeat=" << observation.heartbeat
                  << " phase=" << phase
                  << " stalled_ms="
                  << owner_trackers[owner].stalled_for_ns(now_ns) / 1'000'000
                  << " transport_timeout_ms="
                  << kernel_params.direct_timeout_ns / 1'000'000;
          if (publish_status != cudaSuccess) {
            message << " failure_publish_error="
                    << cudaGetErrorString(publish_status);
          }
          bool expected = false;
          if (direct_failure_logged.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            engine.telemetry_.direct_path_failures.fetch_add(
              1, std::memory_order_relaxed);
          }
          mark_unhealthy(message.str());
          break;
        }
      }
    }

    CompletionDescriptor completion;
    if (!completions.try_pop(completion)) {
      std::this_thread::yield();
      continue;
    }
    if (completion.query_slot >= query_slots) {
      continue;
    }
    QuerySlotState& state = query_slot_states[completion.query_slot];
    if (state.phase.load(std::memory_order_acquire) !=
          static_cast<u32>(QuerySlotPhase::pending) ||
        state.request_id != completion.request_id) {
      // A fail-stop/shutdown rejection can race a descriptor already running
      // on the GPU. Slot+request identity prevents that stale completion from
      // publishing into a later slot generation.
      continue;
    }
    if (completion.status == 0 &&
        completion.result_count > result_capacity) {
      completion.status = -EOVERFLOW;
    }
    const QueryFailureReason failure_reason =
      query_failure_reason(completion.diagnostic);
    const u32 route_snapshot_retries =
      query_route_snapshot_retries(completion.diagnostic);
    engine.telemetry_.centroid_route_query_retries.fetch_add(
      route_snapshot_retries, std::memory_order_relaxed);
    if (failure_reason == QueryFailureReason::route_snapshot_timeout) {
      engine.telemetry_.centroid_route_query_timeouts.fetch_add(
        1, std::memory_order_relaxed);
    }
    const auto submitted_at = state.submitted_at;
    const auto completed_at = std::chrono::steady_clock::now();
    const u64 gpu_ns = completion.gpu_cycles * 1000000ULL / gpu_clock_khz;
    const auto phase_ns = [&](u64 cycles) {
      return cycles * 1000000ULL / gpu_clock_khz;
    };
    const u64 end_to_end_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        completed_at - submitted_at).count());
    if (end_to_end_ns >= 10000000ULL &&
        slow_query_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
      std::cerr << "[gpu-search] slow query e2e_us=" << end_to_end_ns / 1000
                << " gpu_us=" << gpu_ns / 1000
                << " prepare_us=" << completion.prepare_cycles * 1000ULL / gpu_clock_khz
                << " graph_us=" << completion.graph_cycles * 1000ULL / gpu_clock_khz
                << " score_us=" << completion.score_cycles * 1000ULL / gpu_clock_khz
                << " beam_us=" << completion.beam_cycles * 1000ULL / gpu_clock_khz
                << " exact_us=" << completion.exact_cycles * 1000ULL / gpu_clock_khz
                << " beam_select_us="
                << completion.beam_selection_cycles * 1000ULL / gpu_clock_khz
                << " rdma_issue_us="
                << completion.rdma_issue_cycles * 1000ULL / gpu_clock_khz
                << " rdma_wait_us="
                << completion.rdma_wait_cycles * 1000ULL / gpu_clock_khz
                << " graph_validation_us="
                << completion.graph_validation_cycles * 1000ULL / gpu_clock_khz
                << " neighbor_decode_us="
                << completion.neighbor_decode_cycles * 1000ULL / gpu_clock_khz
                << " pq_score_us="
                << completion.pq_score_cycles * 1000ULL / gpu_clock_khz
                << " visited_us="
                << completion.visited_cycles * 1000ULL / gpu_clock_khz
                << " beam_merge_us="
                << completion.beam_merge_cycles * 1000ULL / gpu_clock_khz
                << " graph_reads=" << completion.remote_pages
                << " graph_rereads=" << completion.graph_read_retries
                << " graph_bytes=" << completion.graph_read_bytes
                << " graph_extent_reads="
                << completion.graph_live_extent_reads
                << " graph_full_reads="
                << completion.graph_full_record_reads
                << " graph_extent_fallbacks="
                << completion.graph_extent_fallback_reads
                << " graph_extent_underhints="
                << completion.graph_extent_underhint_reads
                << " graph_extent_promotions="
                << completion.graph_extent_hint_promotions
                << " dynamic_graph_short_reads="
                << completion.dynamic_graph_short_reads
                << " dynamic_graph_full_reads="
                << completion.dynamic_graph_full_reads
                << " dynamic_graph_bytes="
                << completion.dynamic_graph_read_bytes
                << " dynamic_graph_fallbacks="
                << completion.dynamic_graph_fallback_reads
                << " dynamic_graph_promotions="
                << completion.dynamic_graph_hint_promotions
                << " dynamic_graph_demotions="
                << completion.dynamic_graph_hint_demotions
                << " logical_expansions=" << completion.logical_expansions
                << " critical_graph_reads="
                << completion.critical_graph_reads
                << " speculative_graph_reads="
                << completion.speculative_graph_reads
                << " speculative_promoted="
                << completion.speculative_promoted
                << " speculative_stale="
                << completion.speculative_stale
                << " speculative_wasted_bytes="
                << completion.speculative_wasted_bytes
                << " rdma_completion_latency_ns="
                << completion.rdma_completion_latency_ns
                << " rdma_completion_groups="
                << completion.rdma_completion_groups
                << " issue_width_max=" << completion.max_issue_width
                << " commit_width_max=" << completion.max_commit_width
                << " critical_misses=" << completion.critical_misses
                << " certificate_rejects="
                << (completion.status == 0
                      ? completion.frontier_telemetry_reserved0 : 0u)
                << " graph_batches=" << completion.remote_batches
                << " graph_rounds=" << completion.graph_rounds
                << " route_hits=" << completion.route_hits
                << " exact_reads=" << completion.exact_vectors
                << " exact_snapshot_trains="
                << completion.exact_snapshot_train_batches
                << " exact_snapshot_fallbacks="
                << completion.exact_snapshot_train_fallbacks
                << " terminal_exact_cache_attempted="
                << completion.terminal_exact_cache_attempted_queries
                << " terminal_exact_cache_issued="
                << completion.terminal_exact_cache_issued_records
                << " terminal_exact_cache_promoted="
                << completion.terminal_exact_cache_promoted_records
                << " terminal_exact_cache_misses="
                << completion.terminal_exact_cache_miss_records
                << " terminal_exact_cache_wasted_bytes="
                << completion.terminal_exact_cache_wasted_bytes
                << " terminal_exact_cache_queue_rejects="
                << completion.terminal_exact_cache_queue_rejects
                << " dynamic_pq_reads=" << completion.dynamic_code_reads
                << " dynamic_pq_us="
                << completion.dynamic_code_cycles * 1000ULL / gpu_clock_khz
                << " dynamic_pq_incarnation_rejects="
                << completion.dynamic_code_incarnation_rejects
                << " dynamic_pq_arena_hits="
                << completion.dynamic_code_cache_hits
                << " dynamic_pq_batch_deduplicated="
                << completion.dynamic_code_batch_deduplicated << '\n';
    }
    write_query_rdma_trace(completion);
    if (completion.status != 0) {
      report_direct_path_failure();
      if (healthy.load(std::memory_order_acquire)) {
        std::ostringstream message;
        message << "persistent GPU query failed with status "
                << completion.status
                << " reason=" << query_failure_reason_name(failure_reason)
                << " route_snapshot_retries=" << route_snapshot_retries;
        if (failure_reason == QueryFailureReason::graph_fetch ||
            failure_reason == QueryFailureReason::dynamic_code_fetch) {
          const u32 failure_code = completion.result_count;
          const u32 failure_stage = failure_code & 0xffu;
          message << " failure_stage=" << failure_stage;
          if (failure_stage == 6u) {
            const u32 detail = failure_code >> 8;
            message << " authoritative_fetch_detail=" << detail;
            if ((detail & (u32{1} << 16)) != 0) {
              message << " authoritative_fetch_status=-"
                      << (detail & 0xffffu);
            } else {
              const u32 prepare_reason = detail & 0xfu;
              if (prepare_reason == 1u || prepare_reason == 2u ||
                  prepare_reason == 3u || prepare_reason == 4u ||
                  prepare_reason == 6u || prepare_reason == 7u ||
                  prepare_reason == 8u) {
                message << " prepare_reason=" << prepare_reason
                        << " prepare_index=" << ((detail >> 4) & 0x1fu)
                        << " prepare_scratch_slot="
                        << ((detail >> 9) & 0x3fu)
                        << " selection_from_certificate="
                        << ((detail >> 15) & 1u)
                        << " rejected_handle=0x" << std::hex
                        << (static_cast<u64>(
                              completion.frontier_telemetry_reserved1)
                              << 32 |
                            completion.frontier_telemetry_reserved0)
                        << std::dec;
              }
            }
          } else {
            message << " dependency_status=-"
                    << ((failure_code >> 8) & 0xffffu);
          }
        } else if (failure_reason ==
                   QueryFailureReason::exact_rerank_empty) {
          const u32 detail = completion.result_count;
          const u64 rejected_handle =
            (static_cast<u64>(
               completion.frontier_telemetry_reserved1) << 32) |
            completion.frontier_telemetry_reserved0;
          message
            << " exact_candidates=" << (detail & 0xffu)
            << " exact_resolved=" << ((detail >> 8) & 0xffu)
            << " exact_equal_headers=" << ((detail >> 16) & 0xffu)
            << " exact_visible=" << ((detail >> 24) & 0xffu)
            << " first_rejected_handle=0x" << std::hex
            << rejected_handle
            << " first_header_before=0x"
            << completion.rdma_completion_latency_ns
            << " first_header_after=0x"
            << completion.speculative_completion_latency_ns
            << std::dec
            << " expected_incarnation="
            << static_cast<u32>(completion.issue_width_sum >> 32)
            << " stored_incarnation="
            << static_cast<u32>(completion.issue_width_sum)
            << " beam_empty_detail=0x" << std::hex
            << completion.commit_width_sum << std::dec;
        }
        mark_unhealthy(message.str());
      }
      reject_query_slot(completion.query_slot);
    } else {
      state.completion = completion;
      u32 expected = static_cast<u32>(QuerySlotPhase::pending);
      if (!state.phase.compare_exchange_strong(
            expected, static_cast<u32>(QuerySlotPhase::completed),
            std::memory_order_release, std::memory_order_acquire)) {
        continue;
      }
      state.phase.notify_all();
    }
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
    engine.telemetry_.gpu_beam_selection_ns.fetch_add(
      phase_ns(completion.beam_selection_cycles), std::memory_order_relaxed);
    engine.telemetry_.gpu_rdma_issue_ns.fetch_add(
      phase_ns(completion.rdma_issue_cycles), std::memory_order_relaxed);
    engine.telemetry_.gpu_frontier_preview_ns.fetch_add(
      phase_ns(completion.frontier_preview_cycles),
      std::memory_order_relaxed);
    engine.telemetry_.gpu_frontier_prepare_ns.fetch_add(
      phase_ns(completion.frontier_prepare_cycles),
      std::memory_order_relaxed);
    engine.telemetry_.gpu_frontier_enqueue_ns.fetch_add(
      phase_ns(completion.frontier_enqueue_cycles),
      std::memory_order_relaxed);
    engine.telemetry_.gpu_rdma_wait_ns.fetch_add(
      phase_ns(completion.rdma_wait_cycles), std::memory_order_relaxed);
    engine.telemetry_.gpu_graph_validation_ns.fetch_add(
      phase_ns(completion.graph_validation_cycles), std::memory_order_relaxed);
    engine.telemetry_.gpu_neighbor_decode_ns.fetch_add(
      phase_ns(completion.neighbor_decode_cycles), std::memory_order_relaxed);
    engine.telemetry_.gpu_pq_score_ns.fetch_add(
      phase_ns(completion.pq_score_cycles), std::memory_order_relaxed);
    engine.telemetry_.gpu_visited_ns.fetch_add(
      phase_ns(completion.visited_cycles), std::memory_order_relaxed);
    engine.telemetry_.gpu_beam_merge_ns.fetch_add(
      phase_ns(completion.beam_merge_cycles), std::memory_order_relaxed);
    engine.telemetry_.gpu_beam_merge_prepare_ns.fetch_add(
      phase_ns(completion.beam_merge_prepare_cycles),
      std::memory_order_relaxed);
    engine.telemetry_.gpu_beam_merge_sort_ns.fetch_add(
      phase_ns(completion.beam_merge_sort_cycles),
      std::memory_order_relaxed);
    engine.telemetry_.gpu_beam_merge_materialize_ns.fetch_add(
      phase_ns(completion.beam_merge_materialize_cycles),
      std::memory_order_relaxed);
    engine.telemetry_.completion_wait_ns.fetch_add(end_to_end_ns,
                                                   std::memory_order_relaxed);
    const u64 physical_graph_reads =
      static_cast<u64>(completion.graph_live_extent_reads) +
      completion.graph_full_record_reads;
    engine.telemetry_.graph_shard_batches.fetch_add(
      completion.remote_batches, std::memory_order_relaxed);
    engine.telemetry_.rdma_read_ops.fetch_add(
      static_cast<u64>(completion.exact_vectors) + physical_graph_reads +
        completion.dynamic_code_reads,
      std::memory_order_relaxed);
    engine.telemetry_.rdma_read_bytes.fetch_add(
      static_cast<u64>(completion.exact_vectors) * node_record_bytes +
      completion.graph_read_bytes +
      static_cast<u64>(completion.dynamic_code_reads) *
        dynamic_code_record_bytes,
      std::memory_order_relaxed);
    if (physical_graph_reads > completion.remote_batches) {
      engine.telemetry_.rdma_merged_requests.fetch_add(
        physical_graph_reads - completion.remote_batches,
        std::memory_order_relaxed);
    }
    engine.telemetry_.exact_vector_reads.fetch_add(completion.exact_vectors,
                                                   std::memory_order_relaxed);
    engine.telemetry_.exact_snapshot_train_batches.fetch_add(
      completion.exact_snapshot_train_batches,
      std::memory_order_relaxed);
    engine.telemetry_.exact_snapshot_train_fallbacks.fetch_add(
      completion.exact_snapshot_train_fallbacks,
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_candidates.fetch_add(
      completion.dynamic_code_candidates, std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_reads.fetch_add(
      completion.dynamic_code_reads, std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_read_bytes.fetch_add(
      static_cast<u64>(completion.dynamic_code_reads) *
        dynamic_code_record_bytes,
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_incarnation_rejects.fetch_add(
      completion.dynamic_code_incarnation_rejects,
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_wait_ns.fetch_add(
      phase_ns(completion.dynamic_code_cycles),
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_cache_hits.fetch_add(
      completion.dynamic_code_cache_hits, std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_batch_deduplicated.fetch_add(
      completion.dynamic_code_batch_deduplicated,
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_cache_publish_successes.fetch_add(
      completion.dynamic_code_cache_publish_successes,
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_cache_publish_races.fetch_add(
      completion.dynamic_code_cache_publish_races,
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_cache_lookup_probe_exhaustions.fetch_add(
      completion.dynamic_code_cache_lookup_probe_exhaustions,
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_cache_publish_probe_exhaustions.fetch_add(
      completion.dynamic_code_cache_publish_probe_exhaustions,
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_cache_lookup_probes.fetch_add(
      completion.dynamic_code_cache_lookup_probes,
      std::memory_order_relaxed);
    engine.telemetry_.dynamic_code_cache_occupied.fetch_add(
      completion.dynamic_code_cache_first_occupancies,
      std::memory_order_relaxed);
    u64 observed_max =
      engine.telemetry_.dynamic_code_cache_max_lookup_probes.load(
        std::memory_order_relaxed);
    while (observed_max < completion.dynamic_code_cache_max_lookup_probes &&
           !engine.telemetry_.dynamic_code_cache_max_lookup_probes
              .compare_exchange_weak(
                observed_max, completion.dynamic_code_cache_max_lookup_probes,
                std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
    engine.telemetry_.graph_page_requests.fetch_add(completion.remote_pages,
                                                    std::memory_order_relaxed);
    engine.telemetry_.graph_read_retries.fetch_add(
      completion.graph_read_retries, std::memory_order_relaxed);
    engine.telemetry_.graph_read_bytes.fetch_add(
      completion.graph_read_bytes, std::memory_order_relaxed);
    engine.telemetry_.graph_live_extent_reads.fetch_add(
      completion.graph_live_extent_reads, std::memory_order_relaxed);
    engine.telemetry_.graph_full_record_reads.fetch_add(
      completion.graph_full_record_reads, std::memory_order_relaxed);
    engine.telemetry_.graph_extent_fallback_reads.fetch_add(
      completion.graph_extent_fallback_reads, std::memory_order_relaxed);
    engine.telemetry_.graph_extent_underhint_reads.fetch_add(
      completion.graph_extent_underhint_reads, std::memory_order_relaxed);
    engine.telemetry_.graph_extent_hint_promotions.fetch_add(
      completion.graph_extent_hint_promotions, std::memory_order_relaxed);
    engine.telemetry_.expanded_parent_count.fetch_add(
      completion.expanded_parent_count, std::memory_order_relaxed);
    engine.telemetry_.expanded_neighbor_count_sum.fetch_add(
      completion.expanded_neighbor_count_sum, std::memory_order_relaxed);
    for (u32 bucket = 0; bucket < kGraphDegreeHistogramBuckets; ++bucket) {
      engine.telemetry_.expanded_degree_histogram[bucket].fetch_add(
        completion.expanded_degree_histogram[bucket],
        std::memory_order_relaxed);
    }
    engine.telemetry_.dynamic_graph_short_reads.fetch_add(
      completion.dynamic_graph_short_reads, std::memory_order_relaxed);
    engine.telemetry_.dynamic_graph_full_reads.fetch_add(
      completion.dynamic_graph_full_reads, std::memory_order_relaxed);
    engine.telemetry_.dynamic_graph_read_bytes.fetch_add(
      completion.dynamic_graph_read_bytes, std::memory_order_relaxed);
    engine.telemetry_.dynamic_graph_fallback_reads.fetch_add(
      completion.dynamic_graph_fallback_reads, std::memory_order_relaxed);
    engine.telemetry_.dynamic_graph_hint_promotions.fetch_add(
      completion.dynamic_graph_hint_promotions, std::memory_order_relaxed);
    engine.telemetry_.dynamic_graph_hint_demotions.fetch_add(
      completion.dynamic_graph_hint_demotions, std::memory_order_relaxed);
    engine.telemetry_.logical_expansions.fetch_add(
      completion.logical_expansions, std::memory_order_relaxed);
    engine.telemetry_.critical_graph_reads.fetch_add(
      completion.critical_graph_reads, std::memory_order_relaxed);
    engine.telemetry_.critical_graph_bytes.fetch_add(
      completion.critical_graph_bytes, std::memory_order_relaxed);
    engine.telemetry_.speculative_graph_reads.fetch_add(
      completion.speculative_graph_reads, std::memory_order_relaxed);
    engine.telemetry_.speculative_graph_bytes.fetch_add(
      completion.speculative_graph_bytes, std::memory_order_relaxed);
    engine.telemetry_.speculative_wasted_bytes.fetch_add(
      completion.speculative_wasted_bytes, std::memory_order_relaxed);
    engine.telemetry_.terminal_exact_cache_wasted_bytes.fetch_add(
      completion.terminal_exact_cache_wasted_bytes,
      std::memory_order_relaxed);
    engine.telemetry_.rdma_completion_latency_ns.fetch_add(
      completion.rdma_completion_latency_ns, std::memory_order_relaxed);
    engine.telemetry_.speculative_completion_latency_ns.fetch_add(
      completion.speculative_completion_latency_ns,
      std::memory_order_relaxed);
    engine.telemetry_.rdma_completion_groups.fetch_add(
      completion.rdma_completion_groups, std::memory_order_relaxed);
    engine.telemetry_.speculative_completion_groups.fetch_add(
      completion.speculative_completion_groups,
      std::memory_order_relaxed);
    engine.telemetry_.speculative_arrived.fetch_add(
      completion.speculative_arrived, std::memory_order_relaxed);
    engine.telemetry_.speculative_promoted.fetch_add(
      completion.speculative_promoted, std::memory_order_relaxed);
    engine.telemetry_.speculative_stale.fetch_add(
      completion.speculative_stale, std::memory_order_relaxed);
    engine.telemetry_.speculative_queue_rejects.fetch_add(
      completion.speculative_queue_rejects, std::memory_order_relaxed);
    engine.telemetry_.core_prefetch_reads.fetch_add(
      completion.core_prefetch_reads, std::memory_order_relaxed);
    engine.telemetry_.core_prefetch_bytes.fetch_add(
      completion.core_prefetch_bytes, std::memory_order_relaxed);
    engine.telemetry_.core_prefetch_arrived.fetch_add(
      completion.core_prefetch_arrived, std::memory_order_relaxed);
    engine.telemetry_.core_prefetch_promoted.fetch_add(
      completion.core_prefetch_promoted, std::memory_order_relaxed);
    engine.telemetry_.core_prefetch_stale.fetch_add(
      completion.core_prefetch_stale, std::memory_order_relaxed);
    engine.telemetry_.core_prefetch_queue_rejects.fetch_add(
      completion.core_prefetch_queue_rejects, std::memory_order_relaxed);
    engine.telemetry_.core_prefetch_waves.fetch_add(
      completion.core_prefetch_waves, std::memory_order_relaxed);
    engine.telemetry_.core_ready_waves.fetch_add(
      completion.core_ready_waves, std::memory_order_relaxed);
    engine.telemetry_.terminal_exact_cache_attempted_queries.fetch_add(
      completion.terminal_exact_cache_attempted_queries,
      std::memory_order_relaxed);
    engine.telemetry_.terminal_exact_cache_issued_records.fetch_add(
      completion.terminal_exact_cache_issued_records,
      std::memory_order_relaxed);
    engine.telemetry_.terminal_exact_cache_promoted_records.fetch_add(
      completion.terminal_exact_cache_promoted_records,
      std::memory_order_relaxed);
    engine.telemetry_.terminal_exact_cache_queue_rejects.fetch_add(
      completion.terminal_exact_cache_queue_rejects,
      std::memory_order_relaxed);
    engine.telemetry_.terminal_exact_cache_miss_records.fetch_add(
      completion.terminal_exact_cache_miss_records,
      std::memory_order_relaxed);
    engine.telemetry_.completion_score_batches.fetch_add(
      completion.completion_score_batches, std::memory_order_relaxed);
    engine.telemetry_.completion_score_candidates.fetch_add(
      completion.completion_score_candidates, std::memory_order_relaxed);
    engine.telemetry_.frontier_reusable_certificates.fetch_add(
      completion.frontier_reusable_certificates,
      std::memory_order_relaxed);
    engine.telemetry_.frontier_streamed_candidate_runs.fetch_add(
      completion.frontier_streamed_candidate_runs,
      std::memory_order_relaxed);
    engine.telemetry_.ordered_score_batches.fetch_add(
      completion.ordered_score_batches, std::memory_order_relaxed);
    engine.telemetry_.ordered_score_candidates.fetch_add(
      completion.ordered_score_candidates, std::memory_order_relaxed);
    if (completion.status == 0) {
      engine.telemetry_.ooo_bypassed_parents.fetch_add(
        completion.frontier_telemetry_reserved1,
        std::memory_order_relaxed);
    }
    engine.telemetry_.frontier_reusable_prefix_ranks.fetch_add(
      completion.frontier_reusable_prefix_ranks,
      std::memory_order_relaxed);
    engine.telemetry_.frontier_reusable_full_prefix_certificates.fetch_add(
      completion.frontier_reusable_full_prefix_certificates,
      std::memory_order_relaxed);
    engine.telemetry_.frontier_reusable_issued_certificates.fetch_add(
      completion.frontier_reusable_issued_certificates,
      std::memory_order_relaxed);
    if (completion.status == 0) {
      engine.telemetry_.frontier_certificate_rejects.fetch_add(
        completion.frontier_telemetry_reserved0,
        std::memory_order_relaxed);
    }
    engine.telemetry_.issue_epochs.fetch_add(
      completion.issue_epochs, std::memory_order_relaxed);
    engine.telemetry_.commit_epochs.fetch_add(
      completion.commit_epochs, std::memory_order_relaxed);
    engine.telemetry_.issue_width_sum.fetch_add(
      completion.issue_width_sum, std::memory_order_relaxed);
    engine.telemetry_.issue_width_capacity_sum.fetch_add(
      completion.issue_width_capacity_sum, std::memory_order_relaxed);
    engine.telemetry_.commit_width_sum.fetch_add(
      completion.commit_width_sum, std::memory_order_relaxed);
    engine.telemetry_.critical_rob_hits.fetch_add(
      completion.critical_rob_hits, std::memory_order_relaxed);
    engine.telemetry_.critical_misses.fetch_add(
      completion.critical_misses, std::memory_order_relaxed);
    engine.telemetry_.speculative_wait_ns.fetch_add(
      phase_ns(completion.speculative_wait_cycles),
      std::memory_order_relaxed);
    u64 observed_max_issue_width =
      engine.telemetry_.max_issue_width.load(std::memory_order_relaxed);
    while (observed_max_issue_width < completion.max_issue_width &&
           !engine.telemetry_.max_issue_width.compare_exchange_weak(
             observed_max_issue_width, completion.max_issue_width,
             std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
    u64 observed_max_commit_width =
      engine.telemetry_.max_commit_width.load(std::memory_order_relaxed);
    while (observed_max_commit_width < completion.max_commit_width &&
           !engine.telemetry_.max_commit_width.compare_exchange_weak(
             observed_max_commit_width, completion.max_commit_width,
             std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
    engine.telemetry_.graph_dependency_rounds.fetch_add(
      completion.graph_rounds, std::memory_order_relaxed);
    engine.telemetry_.graph_route_hits.fetch_add(completion.route_hits,
                                                 std::memory_order_relaxed);
  }
}

}  // namespace gpu_search
