#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

namespace gpu_search {

using namespace persistent_engine_detail;
service::QueryResult PersistentSearchEngine::Impl::search(VectorDType query_dtype, const byte_t* query_data, u32 k) {
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
  u32 slot = 0;
  {
    std::unique_lock<std::mutex> lock(slot_mutex);
    slot_cv.wait(lock, [&] {
      return !free_slots.empty() || !accepting.load() || !healthy.load();
    });
    if (!accepting.load()) throw std::runtime_error("persistent GPU search engine stopped");
    if (!healthy.load()) {
      lock.unlock();
      throw std::runtime_error(unhealthy_message());
    }
    slot = free_slots.back();
    free_slots.pop_back();
  }
  const size_t query_bytes = vector_dtype_bytes(query_dtype, config.dim);
  byte_t* query_slot = query_input_host + static_cast<size_t>(slot) * query_input_stride;
  std::memcpy(query_slot, query_data, query_bytes);
  const u64 request_id = next_request_id.fetch_add(1, std::memory_order_relaxed);
  const auto submitted_at = std::chrono::steady_clock::now();
  auto pending = std::make_shared<PendingQuery>();
  pending->slot = slot;
  pending->submitted_at = submitted_at;
  auto future = pending->promise.get_future();
  {
    std::lock_guard<std::mutex> lock(pending_mutex);
    pending_queries.emplace(request_id, pending);
    pending_count.fetch_add(1, std::memory_order_relaxed);
  }
  QueryDescriptor descriptor{
    .request_id = request_id,
    .snapshot_epoch = engine.delta_.published_epoch(),
    .query_device_address = reinterpret_cast<u64>(
      d_query_input + static_cast<size_t>(slot) * query_input_stride),
    .result_device_address = reinterpret_cast<u64>(
      d_result_ids + static_cast<size_t>(slot) * result_capacity),
    .query_slot = slot,
    .result_capacity = result_capacity,
    .dim = static_cast<u16>(config.dim),
    .k = static_cast<u16>(k),
    .query_dtype = static_cast<u8>(query_dtype),
  };
  bool rejected = false;
  std::string rejection_message;
  {
    std::lock_guard<std::mutex> lock(admission_mutex);
    if (!healthy.load(std::memory_order_relaxed)) {
      rejected = true;
      rejection_message = health_error;
    } else {
      admission_queue.push_back({.descriptor = descriptor, .enqueued_at = submitted_at});
    }
  }
  if (rejected) {
    reject_submission({.descriptor = descriptor, .enqueued_at = submitted_at},
                      rejection_message);
  } else {
    admission_cv.notify_one();
  }
  engine.telemetry_.queries_submitted.fetch_add(1, std::memory_order_relaxed);
  return future.get();
}

void PersistentSearchEngine::Impl::admission_loop() {
  std::vector<PendingSubmission> batch;
  batch.reserve(config.gpu_query_slots);
  size_t submitted_count = 0;
  try {
    bind_cuda_device("cudaSetDevice(GPU navigation admission)");
    while (!shutdown.load(std::memory_order_acquire)) {
      batch.clear();
      submitted_count = 0;
      {
        std::unique_lock<std::mutex> lock(admission_mutex);
        admission_cv.wait(lock, [&] {
          return !admission_queue.empty() ||
                 !healthy.load() || shutdown.load();
        });
        if (!healthy.load(std::memory_order_acquire) || shutdown.load()) return;
        if (admission_queue.empty()) continue;
        const size_t count = std::min<size_t>(
          admission_queue.size(), config.gpu_query_slots);
        for (size_t index = 0; index < count; ++index) {
          batch.push_back(admission_queue.front());
          admission_queue.pop_front();
        }
        active_gpu_queries.fetch_add(count, std::memory_order_release);
      }
      if (batch.empty()) continue;
      const auto admitted_at = std::chrono::steady_clock::now();
      u64 wait_ns = 0;
      {
        std::lock_guard<std::mutex> snapshot_lock(query_snapshot_mutex);
        for (PendingSubmission& submission : batch) {
          submission.descriptor.snapshot_epoch = engine.delta_.published_epoch();
          const u64 query_ticket =
            next_query_ticket.fetch_add(1, std::memory_order_acq_rel);
          const u32 slot = submission.descriptor.query_slot;
          active_query_snapshots[slot].store(
            submission.descriptor.snapshot_epoch + 1,
            std::memory_order_release);
          active_query_tickets[slot].store(query_ticket, std::memory_order_release);
        }
      }
      for (PendingSubmission& submission : batch) {
        while (!submissions.try_push(submission.descriptor)) {
          if (shutdown.load(std::memory_order_acquire) ||
              !healthy.load(std::memory_order_acquire)) {
            throw std::runtime_error("GPU submission ring stopped making progress");
          }
          std::this_thread::yield();
        }
        ++submitted_count;
        wait_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
          admitted_at - submission.enqueued_at).count());
      }
      engine.telemetry_.batches.fetch_add(1, std::memory_order_relaxed);
      engine.telemetry_.batch_queries.fetch_add(batch.size(), std::memory_order_relaxed);
      engine.telemetry_.submission_wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
    }
  } catch (const std::exception& error) {
    for (size_t index = submitted_count; index < batch.size(); ++index) {
      reject_submission(batch[index], error.what());
    }
    const size_t rejected_count = batch.size() - submitted_count;
    if (rejected_count != 0) {
      active_gpu_queries.fetch_sub(rejected_count, std::memory_order_release);
      maintenance_cv.notify_all();
    }
    mark_unhealthy(std::string{"GPU admission failed: "} + error.what());
  } catch (...) {
    for (size_t index = submitted_count; index < batch.size(); ++index) {
      reject_submission(batch[index], "unknown GPU admission failure");
    }
    const size_t rejected_count = batch.size() - submitted_count;
    if (rejected_count != 0) {
      active_gpu_queries.fetch_sub(rejected_count, std::memory_order_release);
      maintenance_cv.notify_all();
    }
    mark_unhealthy("unknown GPU admission failure");
  }
}

}  // namespace gpu_search
