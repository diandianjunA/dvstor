#include "gpu_search/persistent_engine.hh"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "gpu_search/persistent_engine/cuda_helpers.hh"
#include "gpu_search/persistent_engine/impl.hh"

namespace gpu_search {

using persistent_engine_detail::check_cuda;

PersistentSearchEngine::PersistentSearchEngine(
    configuration::IndexConfiguration& config,
    Context& channel_context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions)
    : delta_() {
  check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)),
             "cudaSetDevice(GPU navigation engine)");
  impl_ = std::make_unique<Impl>(*this, config, channel_context,
                                 connection_manager, remote_regions);
}

PersistentSearchEngine::~PersistentSearchEngine() {
  impl_.reset();
}

service::QueryResult PersistentSearchEngine::search(
    VectorDType query_dtype, const byte_t* query_data, u32 k) {
  return impl_->search(query_dtype, query_data, k);
}

service::QueryResult PersistentSearchEngine::search(std::span<const element_t> query, u32 k) {
  return search(VectorDType::float32,
                reinterpret_cast<const byte_t*>(query.data()), k);
}

bool PersistentSearchEngine::publish_mutations(
    std::vector<DeltaMutation> mutations, u64 epoch,
    std::span<const u64> invalidated_graph_nodes) {
  std::lock_guard<std::mutex> publish_lock(mutation_publish_mutex_);
  if (mutations.empty() || epoch == 0) {
    throw std::invalid_argument("GPU mutation publication requires a non-empty epoch batch");
  }
  std::unordered_map<node_t, u32> accepted_generations;
  mutations.erase(
    std::remove_if(mutations.begin(), mutations.end(), [&](DeltaMutation& mutation) {
      auto [entry, inserted] = accepted_generations.emplace(mutation.id, 0);
      if (inserted) {
        const auto current = delta_.version(mutation.id);
        entry->second = current ? current->generation : 0;
      }
      if (mutation.generation == 0) {
        mutation.generation = entry->second + 1;
      } else if (mutation.generation <= entry->second) {
        return true;
      }
      entry->second = mutation.generation;
      return false;
    }),
    mutations.end());
  if (mutations.empty()) {
    return true;
  }
  const size_t mutation_count = mutations.size();
  const auto publication_started = std::chrono::steady_clock::now();
  u64 publication_queue_ns = 0;
  for (const DeltaMutation& mutation : mutations) {
    if (mutation.enqueued_at == std::chrono::steady_clock::time_point{}) continue;
    publication_queue_ns += static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        publication_started - mutation.enqueued_at).count());
  }
  telemetry_.publication_queue_ns_total.fetch_add(publication_queue_ns,
                                                  std::memory_order_relaxed);
  size_t graph_cache_invalidations = 0;
  try {
    graph_cache_invalidations =
      impl_->upload_mutations(mutations, epoch, invalidated_graph_nodes);
  } catch (const MutationCapacityError&) {
    telemetry_.mutation_capacity_rejections.fetch_add(1, std::memory_order_relaxed);
    throw;
  } catch (const std::exception& error) {
    impl_->mark_unhealthy(std::string{"GPU mutation publication failed: "} + error.what());
    throw;
  }
  const auto gpu_upload_completed_at = std::chrono::steady_clock::now();
  u64 visibility_ns_total = 0;
  u64 visibility_ns_max = 0;
  u64 visibility_sample_count = 0;
  for (const DeltaMutation& mutation : mutations) {
    if (mutation.enqueued_at == std::chrono::steady_clock::time_point{}) continue;
    const u64 visibility_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        gpu_upload_completed_at - mutation.enqueued_at).count());
    visibility_ns_total += visibility_ns;
    visibility_ns_max = std::max(visibility_ns_max, visibility_ns);
    ++visibility_sample_count;
  }
  try {
    if (!delta_.publish(std::move(mutations), epoch)) {
      impl_->mark_unhealthy("GPU mutation publication lost its coordinator epoch");
      return false;
    }
  } catch (const std::exception& error) {
    impl_->mark_unhealthy(std::string{"GPU epoch publication failed: "} + error.what());
    throw;
  }
  // Queries cannot select this epoch until the coordinator publish above.
  // Include that final handoff in the stage1-response-to-visible SLO.
  const u64 coordinator_publish_ns = static_cast<u64>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - gpu_upload_completed_at).count());
  visibility_ns_total += coordinator_publish_ns * visibility_sample_count;
  if (visibility_sample_count != 0) {
    visibility_ns_max += coordinator_publish_ns;
  }
  telemetry_.mutations_published.fetch_add(mutation_count, std::memory_order_relaxed);
  telemetry_.delta_publications.fetch_add(1, std::memory_order_relaxed);
  telemetry_.graph_cache_invalidations.fetch_add(
    graph_cache_invalidations, std::memory_order_relaxed);
  telemetry_.visibility_ns_total.fetch_add(visibility_ns_total,
                                           std::memory_order_relaxed);
  telemetry_.delta_live_entries.store(delta_.delta_size(), std::memory_order_relaxed);
  u64 current_max = telemetry_.visibility_ns_max.load(std::memory_order_relaxed);
  while (current_max < visibility_ns_max &&
         !telemetry_.visibility_ns_max.compare_exchange_weak(
           current_max, visibility_ns_max, std::memory_order_relaxed)) {}
  return true;
}

bool PersistentSearchEngine::try_reserve_mutation_capacity(size_t mutation_count) {
  if (mutation_count == 0) return true;
  std::lock_guard<std::mutex> lock(impl_->delta_mutex);
  impl_->reclaim_retired_delta_slots_locked();
  const size_t active_slots = impl_->active_delta_slots_locked();
  const size_t hard_watermark = static_cast<size_t>(impl_->delta_capacity) * 9 / 10;
  const size_t active_resident_pq = impl_->active_resident_pq_slots_locked();
  const size_t resident_pq_hard_watermark =
    std::max<size_t>(1, static_cast<size_t>(impl_->resident_pq_capacity) * 95 / 100);
  if (mutation_count > hard_watermark ||
      active_slots > hard_watermark - mutation_count ||
      impl_->reserved_mutation_capacity >
        hard_watermark - mutation_count - active_slots ||
      mutation_count > resident_pq_hard_watermark ||
      active_resident_pq > resident_pq_hard_watermark - mutation_count ||
      impl_->reserved_mutation_capacity >
        resident_pq_hard_watermark - mutation_count - active_resident_pq) {
    telemetry_.mutation_capacity_rejections.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  impl_->reserved_mutation_capacity += mutation_count;
  const u64 reserved = static_cast<u64>(impl_->reserved_mutation_capacity);
  telemetry_.mutation_capacity_reserved.store(reserved, std::memory_order_relaxed);
  u64 current_max = telemetry_.mutation_capacity_reserved_max.load(
    std::memory_order_relaxed);
  while (current_max < reserved &&
         !telemetry_.mutation_capacity_reserved_max.compare_exchange_weak(
           current_max, reserved, std::memory_order_relaxed)) {}
  return true;
}

void PersistentSearchEngine::reserve_mutation_capacity(size_t mutation_count) {
  if (mutation_count == 0) return;
  bool observed_pressure = false;
  std::chrono::steady_clock::time_point pressure_started{};
  std::unique_lock<std::mutex> lock(impl_->delta_mutex);
  for (;;) {
    impl_->reclaim_retired_delta_slots_locked();
    const size_t active_slots = impl_->active_delta_slots_locked();
    const size_t hard_watermark =
      static_cast<size_t>(impl_->delta_capacity) * 9 / 10;
    const size_t active_resident_pq =
      impl_->active_resident_pq_slots_locked();
    const size_t resident_pq_hard_watermark = std::max<size_t>(
      1, static_cast<size_t>(impl_->resident_pq_capacity) * 95 / 100);
    const bool available =
      mutation_count <= hard_watermark &&
      active_slots <= hard_watermark - mutation_count &&
      impl_->reserved_mutation_capacity <=
        hard_watermark - mutation_count - active_slots &&
      mutation_count <= resident_pq_hard_watermark &&
      active_resident_pq <= resident_pq_hard_watermark - mutation_count &&
      impl_->reserved_mutation_capacity <=
        resident_pq_hard_watermark - mutation_count - active_resident_pq;
    if (available) {
      if (observed_pressure) {
        telemetry_.mutation_capacity_wait_ns.fetch_add(
          static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - pressure_started).count()),
          std::memory_order_relaxed);
      }
      impl_->reserved_mutation_capacity += mutation_count;
      const u64 reserved = static_cast<u64>(
        impl_->reserved_mutation_capacity);
      telemetry_.mutation_capacity_reserved.store(
        reserved, std::memory_order_relaxed);
      u64 current_max = telemetry_.mutation_capacity_reserved_max.load(
        std::memory_order_relaxed);
      while (current_max < reserved &&
             !telemetry_.mutation_capacity_reserved_max.compare_exchange_weak(
               current_max, reserved, std::memory_order_relaxed)) {}
      return;
    }
    if (!observed_pressure) {
      pressure_started = std::chrono::steady_clock::now();
      telemetry_.mutation_capacity_wait_events.fetch_add(
        1, std::memory_order_relaxed);
      observed_pressure = true;
    }
    // Publication releases reservations and notifies directly. The bounded
    // wait also rechecks capacity reclaimed by the independent maintenance
    // thread, which does not need to know about submitters.
    impl_->delta_capacity_cv.wait_for(lock, std::chrono::milliseconds(1));
  }
}

void PersistentSearchEngine::release_mutation_capacity(size_t mutation_count) {
  if (mutation_count == 0) return;
  std::lock_guard<std::mutex> lock(impl_->delta_mutex);
  if (mutation_count > impl_->reserved_mutation_capacity) {
    impl_->mark_unhealthy("GPU mutation capacity reservation accounting underflow");
    impl_->reserved_mutation_capacity = 0;
  } else {
    impl_->reserved_mutation_capacity -= mutation_count;
  }
  telemetry_.mutation_capacity_reserved.store(
    static_cast<u64>(impl_->reserved_mutation_capacity),
    std::memory_order_release);
  impl_->delta_capacity_cv.notify_all();
}

void PersistentSearchEngine::mark_committed_mutation_gap(
    const std::string& reason) {
  impl_->mark_unhealthy(
    "storage committed a mutation that is not GPU-visible: " + reason);
}

void PersistentSearchEngine::reset_telemetry() {
  telemetry_.reset();
  telemetry_.delta_live_entries.store(delta_.delta_size(), std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(impl_->delta_mutex);
  telemetry_.delta_physical_entries.store(
    impl_->active_delta_slots_locked(), std::memory_order_relaxed);
  telemetry_.delta_mutable_entries.store(
    impl_->mutable_delta_entries, std::memory_order_relaxed);
  telemetry_.delta_durable_entries.store(
    impl_->durable_delta_entries, std::memory_order_relaxed);
  telemetry_.resident_pq_capacity.store(
    impl_->resident_pq_capacity, std::memory_order_relaxed);
  telemetry_.resident_pq_entries.store(
    impl_->active_resident_pq_slots_locked(), std::memory_order_relaxed);
  telemetry_.resident_pq_peak_entries.store(
    impl_->active_resident_pq_slots_locked(), std::memory_order_relaxed);
  telemetry_.mutation_capacity_reserved.store(
    impl_->reserved_mutation_capacity, std::memory_order_relaxed);
}

}  // namespace gpu_search
