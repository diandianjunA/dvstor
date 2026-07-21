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

void PersistentSearchEngine::reset_telemetry() {
  telemetry_.reset();
}

}  // namespace gpu_search
