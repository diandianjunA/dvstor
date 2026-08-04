#include "gpu_search/persistent_engine.hh"

#include <memory>

#include "gpu_search/persistent_engine/cuda_helpers.hh"
#include "gpu_search/persistent_engine/impl.hh"

namespace gpu_search {

using persistent_engine_detail::check_cuda;

PersistentSearchEngine::PersistentSearchEngine(
    configuration::IndexConfiguration& config,
    Context& channel_context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions) {
  check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)),
             "cudaSetDevice(GPU navigation engine)");
  impl_ = std::make_unique<Impl>(*this, config, channel_context,
                                 connection_manager, remote_regions);
  telemetry_.set_gpu_occupancy(
    impl_->kernel_threads,
    impl_->persistent_kernel_occupancy.registers_per_thread,
    impl_->persistent_kernel_occupancy.static_shared_bytes,
    impl_->persistent_kernel_occupancy.active_blocks_per_sm,
    impl_->persistent_grid_plan.selected.effective_blocks_per_sm,
    impl_->persistent_grid_plan.selected.query_blocks,
    impl_->persistent_grid_plan.selected.owner_blocks,
    impl_->persistent_grid_plan.selected.total_blocks);
}

PersistentSearchEngine::~PersistentSearchEngine() {
  impl_.reset();
}

service::QueryResult PersistentSearchEngine::search(
    VectorDType query_dtype, const byte_t* query_data, u32 k) {
  return impl_->search(query_dtype, query_data, k);
}

service::QueryResult PersistentSearchEngine::search(
    std::span<const element_t> query, u32 k) {
  return search(VectorDType::float32,
                reinterpret_cast<const byte_t*>(query.data()), k);
}

std::optional<u32> PersistentSearchEngine::select_centroid_home(
    std::span<const f32> vector) const {
  return impl_->select_centroid_home(vector);
}

bool PersistentSearchEngine::wait_for_maintenance(
    std::span<const u64> target_sequences,
    std::chrono::milliseconds timeout,
    std::vector<u64>* durable_sequences,
    std::vector<u64>* effective_target_sequences) {
  return impl_->wait_for_maintenance(
    target_sequences, timeout, durable_sequences,
    effective_target_sequences);
}

std::vector<std::optional<maintenance_telemetry::Snapshot>>
PersistentSearchEngine::read_maintenance_telemetry() {
  return impl_->read_maintenance_telemetry();
}

TelemetrySnapshot PersistentSearchEngine::telemetry() const {
  TelemetrySnapshot result = telemetry_.snapshot();
  if (impl_ == nullptr || impl_->d_direct_owner_progress == nullptr ||
      impl_->direct_batch_queue_count == 0) {
    return result;
  }
  impl_->bind_cuda_device("telemetry");
  std::vector<DirectOwnerProgress> progress(
    impl_->direct_batch_queue_count);
  check_cuda(cudaMemcpy(
    progress.data(), impl_->d_direct_owner_progress,
    progress.size() * sizeof(DirectOwnerProgress),
    cudaMemcpyDeviceToHost), "cudaMemcpy(owner utilization telemetry)");
  for (const DirectOwnerProgress& owner : progress) {
    result.owner_submitted_wqes += owner.submitted_wqes;
    result.owner_submission_wqe_capacity += owner.submission_wqe_capacity;
    result.owner_critical_batches += owner.critical_batches;
    result.owner_speculative_batches += owner.speculative_batches;
  }
  const auto subtract_baseline = [](u64 total, u64 baseline) {
    return total >= baseline ? total - baseline : 0;
  };
  result.owner_submitted_wqes = subtract_baseline(
    result.owner_submitted_wqes,
    impl_->owner_submitted_wqes_baseline.load(std::memory_order_relaxed));
  result.owner_submission_wqe_capacity = subtract_baseline(
    result.owner_submission_wqe_capacity,
    impl_->owner_submission_wqe_capacity_baseline.load(
      std::memory_order_relaxed));
  result.owner_critical_batches = subtract_baseline(
    result.owner_critical_batches,
    impl_->owner_critical_batches_baseline.load(std::memory_order_relaxed));
  result.owner_speculative_batches = subtract_baseline(
    result.owner_speculative_batches,
    impl_->owner_speculative_batches_baseline.load(
      std::memory_order_relaxed));
  return result;
}

void PersistentSearchEngine::reset_telemetry() {
  const TelemetrySnapshot owner_window = telemetry();
  impl_->owner_submitted_wqes_baseline.fetch_add(
    owner_window.owner_submitted_wqes, std::memory_order_relaxed);
  impl_->owner_submission_wqe_capacity_baseline.fetch_add(
    owner_window.owner_submission_wqe_capacity, std::memory_order_relaxed);
  impl_->owner_critical_batches_baseline.fetch_add(
    owner_window.owner_critical_batches, std::memory_order_relaxed);
  impl_->owner_speculative_batches_baseline.fetch_add(
    owner_window.owner_speculative_batches, std::memory_order_relaxed);
  telemetry_.reset();
}

}  // namespace gpu_search
