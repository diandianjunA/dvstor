#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

#include <cerrno>

namespace gpu_search {

using namespace persistent_engine_detail;

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
                << " graph_reads=" << completion.remote_pages
                << " graph_rereads=" << completion.graph_read_retries
                << " graph_batches=" << completion.remote_batches
                << " graph_rounds=" << completion.graph_rounds
                << " route_hits=" << completion.route_hits
                << " exact_reads=" << completion.exact_vectors
                << " dynamic_pq_reads=" << completion.dynamic_code_reads
                << " dynamic_pq_us="
                << completion.dynamic_code_cycles * 1000ULL / gpu_clock_khz
                << " dynamic_pq_incarnation_rejects="
                << completion.dynamic_code_incarnation_rejects << '\n';
    }
    if (completion.status != 0) {
      report_direct_path_failure();
      if (healthy.load(std::memory_order_acquire)) {
        std::ostringstream message;
        message << "persistent GPU query failed with status "
                << completion.status
                << " reason=" << query_failure_reason_name(failure_reason)
                << " route_snapshot_retries=" << route_snapshot_retries;
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
    engine.telemetry_.completion_wait_ns.fetch_add(end_to_end_ns,
                                                   std::memory_order_relaxed);
    const u64 physical_graph_reads =
      static_cast<u64>(completion.remote_pages) + completion.graph_read_retries;
    engine.telemetry_.rdma_read_ops.fetch_add(
      static_cast<u64>(completion.exact_vectors) + physical_graph_reads +
        completion.dynamic_code_reads,
      std::memory_order_relaxed);
    engine.telemetry_.rdma_read_bytes.fetch_add(
      static_cast<u64>(completion.exact_vectors) * node_record_bytes +
      physical_graph_reads * index.layout.graph_entry_bytes +
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
    engine.telemetry_.graph_page_requests.fetch_add(completion.remote_pages,
                                                    std::memory_order_relaxed);
    engine.telemetry_.graph_read_retries.fetch_add(
      completion.graph_read_retries, std::memory_order_relaxed);
    engine.telemetry_.graph_dependency_rounds.fetch_add(
      completion.graph_rounds, std::memory_order_relaxed);
    engine.telemetry_.graph_route_hits.fetch_add(completion.route_hits,
                                                 std::memory_order_relaxed);
  }
}

}  // namespace gpu_search
