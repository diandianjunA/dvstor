#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

namespace gpu_search {

using namespace persistent_engine_detail;

service::QueryResult PersistentSearchEngine::Impl::search(
    VectorDType query_dtype, const byte_t* query_data, u32 k) {
  if (!accepting.load(std::memory_order_acquire)) {
    throw std::runtime_error("persistent GPU search engine is stopping");
  }
  if (!healthy.load(std::memory_order_acquire)) {
    throw std::runtime_error(unhealthy_message());
  }
  if (query_data == nullptr || static_cast<u32>(query_dtype) > 2 ||
      k == 0 || k > result_capacity) {
    throw std::invalid_argument("invalid persistent GPU query");
  }
  if (query_dtype == VectorDType::float32) {
    const auto* components = reinterpret_cast<const f32*>(query_data);
    for (u32 dimension = 0; dimension < config.dim; ++dimension) {
      if (!floating_value_is_finite(components[dimension])) {
        throw std::invalid_argument(
          "persistent GPU query components must be finite");
      }
    }
  }

  u32 slot = 0;
  if (!free_slots->pop_wait(slot, query_stop)) {
    if (!healthy.load(std::memory_order_acquire)) {
      throw std::runtime_error(unhealthy_message());
    }
    throw std::runtime_error("persistent GPU search engine stopped");
  }

  QuerySlotState& state = query_slot_states[slot];
  u32 expected = static_cast<u32>(QuerySlotPhase::free);
  if (!state.phase.compare_exchange_strong(
        expected, static_cast<u32>(QuerySlotPhase::preparing),
        std::memory_order_acq_rel, std::memory_order_acquire)) {
    mark_unhealthy("bounded GPU query slot was reused before release");
    throw std::runtime_error(unhealthy_message());
  }
  try {
    // stop-aware queue operations are deliberately work-conserving and may
    // win a ready item concurrently with stop. Recheck after publishing the
    // preparing phase so fail-stop can no longer miss this slot generation.
    if (query_stop.load(std::memory_order_acquire)) {
      reject_query_slot(slot);
      throw std::runtime_error(
        healthy.load(std::memory_order_acquire)
          ? "persistent GPU search engine stopped"
          : unhealthy_message());
    }
    const u64 request_id =
      next_request_id.fetch_add(1, std::memory_order_relaxed);
    const auto submitted_at = std::chrono::steady_clock::now();
    state.request_id = request_id;
    state.submitted_at = submitted_at;

    const size_t query_bytes = vector_dtype_bytes(query_dtype, config.dim);
    byte_t* query_slot =
      query_input_host + static_cast<size_t>(slot) * query_input_stride;
    std::memcpy(query_slot, query_data, query_bytes);

    QueryDescriptor descriptor{
      .request_id = request_id,
      .query_device_address = reinterpret_cast<u64>(
        d_query_input + static_cast<size_t>(slot) * query_input_stride),
      .result_device_address = reinterpret_cast<u64>(
        d_result_ids + static_cast<size_t>(slot) * result_capacity),
      .query_slot = slot,
      .result_capacity = result_capacity,
      .dim = config.dim,
      .k = static_cast<u16>(k),
      .query_dtype = static_cast<u8>(query_dtype),
    };

    expected = static_cast<u32>(QuerySlotPhase::preparing);
    if (!state.phase.compare_exchange_strong(
          expected, static_cast<u32>(QuerySlotPhase::pending),
          std::memory_order_release, std::memory_order_acquire)) {
      throw std::runtime_error(
        healthy.load(std::memory_order_acquire)
          ? "persistent GPU search engine stopped"
          : unhealthy_message());
    }

    const PendingSubmission submission{
      .descriptor = descriptor,
      .enqueued_at = submitted_at,
    };
    if (!admission_queue->push_wait(submission, query_stop)) {
      reject_query_slot(slot);
    } else {
      engine.telemetry_.queries_submitted.fetch_add(
        1, std::memory_order_relaxed);
    }

    u32 phase = state.phase.load(std::memory_order_acquire);
    while (phase == static_cast<u32>(QuerySlotPhase::preparing) ||
           phase == static_cast<u32>(QuerySlotPhase::pending)) {
      state.phase.wait(phase, std::memory_order_relaxed);
      phase = state.phase.load(std::memory_order_acquire);
    }

    if (phase != static_cast<u32>(QuerySlotPhase::completed)) {
      throw std::runtime_error(
        healthy.load(std::memory_order_acquire)
          ? "persistent GPU search engine stopped before query completion"
          : unhealthy_message());
    }

    const CompletionDescriptor completion = state.completion;
    if (completion.request_id != request_id || completion.query_slot != slot ||
        completion.status != 0 || completion.result_count > result_capacity) {
      throw std::runtime_error(
        "persistent GPU query completion identity mismatch");
    }
    const size_t offset = static_cast<size_t>(slot) * result_capacity;
    service::QueryResult result;
    result.reserve(completion.result_count);
    for (u32 index = 0; index < completion.result_count; ++index) {
      result.push_back({result_ids_host[offset + index],
                        result_distances_host[offset + index]});
    }
    release_query_slot(slot);
    return result;
  } catch (...) {
    const u32 phase = state.phase.load(std::memory_order_acquire);
    if (phase == static_cast<u32>(QuerySlotPhase::preparing) ||
        phase == static_cast<u32>(QuerySlotPhase::pending)) {
      reject_query_slot(slot);
    }
    release_query_slot(slot);
    throw;
  }
}

void PersistentSearchEngine::Impl::admission_loop() {
  std::vector<PendingSubmission> batch;
  batch.reserve(config.gpu_query_slots);
  try {
    bind_cuda_device("cudaSetDevice(GPU navigation admission)");
    for (;;) {
      PendingSubmission first;
      if (!admission_queue->pop_wait(first, query_stop)) return;
      batch.clear();
      batch.push_back(first);
      PendingSubmission next;
      while (batch.size() < config.gpu_query_slots &&
             admission_queue->try_pop(next)) {
        batch.push_back(next);
      }

      const auto admitted_at = std::chrono::steady_clock::now();
      u64 wait_ns = 0;
      size_t submitted = 0;
      for (const PendingSubmission& submission : batch) {
        if (query_stop.load(std::memory_order_acquire)) return;
        const QueryDescriptor& descriptor = submission.descriptor;
        if (descriptor.query_slot >= query_slots) {
          throw std::runtime_error("GPU admission received an invalid query slot");
        }
        QuerySlotState& state = query_slot_states[descriptor.query_slot];
        if (state.phase.load(std::memory_order_acquire) !=
              static_cast<u32>(QuerySlotPhase::pending) ||
            state.request_id != descriptor.request_id) {
          continue;
        }
        while (!submissions.try_push(descriptor)) {
          if (query_stop.load(std::memory_order_acquire)) return;
          std::this_thread::yield();
        }
        ++submitted;
        wait_ns += static_cast<u64>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            admitted_at - submission.enqueued_at).count());
      }
      if (submitted != 0) {
        engine.telemetry_.batches.fetch_add(1, std::memory_order_relaxed);
        engine.telemetry_.batch_queries.fetch_add(
          submitted, std::memory_order_relaxed);
        engine.telemetry_.submission_wait_ns.fetch_add(
          wait_ns, std::memory_order_relaxed);
      }
    }
  } catch (const std::exception& error) {
    mark_unhealthy(std::string{"GPU admission failed: "} + error.what());
  } catch (...) {
    mark_unhealthy("unknown GPU admission failure");
  }
}

}  // namespace gpu_search
