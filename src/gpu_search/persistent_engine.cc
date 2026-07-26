#include "gpu_search/persistent_engine.hh"

#include <memory>

#include "gpu_search/persistent_engine/cuda_helpers.hh"
#include "gpu_search/persistent_engine/impl.hh"

namespace gpu_search {

using persistent_engine_detail::check_cuda;

namespace {

u64 counter_delta(u64 current, u64 baseline) {
  return current >= baseline ? current - baseline : 0;
}

}  // namespace

PersistentSearchEngine::PersistentSearchEngine(
    configuration::IndexConfiguration& config,
    Context& channel_context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions) {
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
  TelemetrySnapshot snapshot = telemetry_.snapshot();
  if (impl_ != nullptr) {
    impl_->augment_expansion_pressure_telemetry(snapshot);
  }
  return snapshot;
}

void PersistentSearchEngine::reset_telemetry() {
  telemetry_.reset();
  if (impl_ != nullptr) impl_->reset_expansion_pressure_telemetry();
}

void PersistentSearchEngine::Impl::augment_expansion_pressure_telemetry(
    TelemetrySnapshot& snapshot) const {
  if (d_expansion_pressure == nullptr) return;
  bind_cuda_device("augment expansion pressure telemetry");
  ExpansionPressureState current{};
  check_cuda(cudaMemcpy(
               &current, d_expansion_pressure, sizeof(current),
               cudaMemcpyDeviceToHost),
             "cudaMemcpy(expansion pressure telemetry)");
  snapshot.expansion_pressure_active_queries =
    expansion_pressure_active(current.control);
  snapshot.expansion_pressure_active_queries_peak =
    expansion_pressure_active_peak(current.control);
  snapshot.expansion_pressure_credit_current =
    expansion_pressure_credit(current.control);
  snapshot.expansion_pressure_credit_max_observed =
    expansion_pressure_credit_peak(current.control);
  snapshot.expansion_pressure_maximum_credit_tiles =
    current.maximum_credit_tiles;
  snapshot.expansion_pressure_hunger_grants = counter_delta(
    current.hunger_grants, expansion_pressure_baseline.hunger_grants);
  snapshot.expansion_pressure_idle_owner_episodes = counter_delta(
    current.idle_owner_episodes,
    expansion_pressure_baseline.idle_owner_episodes);
  snapshot.expansion_pressure_congestion_clears = counter_delta(
    current.congestion_clears,
    expansion_pressure_baseline.congestion_clears);
  snapshot.expansion_pressure_ring_backpressure_events = counter_delta(
    current.ring_backpressure_events,
    expansion_pressure_baseline.ring_backpressure_events);
  snapshot.expansion_pressure_sq_defer_events = counter_delta(
    current.sq_defer_events, expansion_pressure_baseline.sq_defer_events);
  if (d_expansion_qp_leases != nullptr && direct_batch_queue_count != 0) {
    std::vector<QpExpansionLeaseState> leases(direct_batch_queue_count);
    check_cuda(cudaMemcpy(
                 leases.data(), d_expansion_qp_leases,
                 leases.size() * sizeof(QpExpansionLeaseState),
                 cudaMemcpyDeviceToHost),
               "cudaMemcpy(expansion QP lease telemetry)");
    for (u32 qp = 0; qp < direct_batch_queue_count; ++qp) {
      const QpExpansionLeaseState& lease = leases[qp];
      const QpExpansionLeaseState baseline =
        qp < expansion_qp_lease_baseline.size()
          ? expansion_qp_lease_baseline[qp] : QpExpansionLeaseState{};
      snapshot.expansion_qp_lease_available_wqes +=
        qp_expansion_lease_available(lease.control);
      snapshot.expansion_qp_lease_offers +=
        counter_delta(lease.offers, baseline.offers);
      snapshot.expansion_qp_lease_claimed_wqes +=
        counter_delta(lease.claims, baseline.claims);
      snapshot.expansion_qp_lease_owner_rejects +=
        counter_delta(lease.rejects, baseline.rejects);
      snapshot.expansion_qp_lease_returns +=
        counter_delta(lease.returns, baseline.returns);
      snapshot.expansion_qp_lease_revocations +=
        counter_delta(lease.revocations, baseline.revocations);
      snapshot.expansion_qp_lease_stale_returns +=
        counter_delta(lease.stale_returns, baseline.stale_returns);
    }
  }
}

void PersistentSearchEngine::Impl::reset_expansion_pressure_telemetry() {
  if (d_expansion_pressure == nullptr) return;
  bind_cuda_device("reset expansion pressure telemetry");
  check_cuda(cudaMemcpy(
               &expansion_pressure_baseline, d_expansion_pressure,
               sizeof(expansion_pressure_baseline), cudaMemcpyDeviceToHost),
             "cudaMemcpy(expansion pressure telemetry baseline)");
  if (d_expansion_qp_leases != nullptr && direct_batch_queue_count != 0) {
    expansion_qp_lease_baseline.resize(direct_batch_queue_count);
    check_cuda(cudaMemcpy(
                 expansion_qp_lease_baseline.data(),
                 d_expansion_qp_leases,
                 expansion_qp_lease_baseline.size() *
                   sizeof(QpExpansionLeaseState),
                 cudaMemcpyDeviceToHost),
               "cudaMemcpy(expansion QP lease telemetry baseline)");
  }
}

}  // namespace gpu_search
