#pragma once

#include <memory>

#include <library/connection_manager.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "gpu_search/search_engine.hh"

namespace gpu_search {

// Baseline query backend for the coupled search-progression mode. The CPU
// owns Beam/visited state and posts ordinary one-sided reads in strict commit
// waves; CUDA is used only for bounded, finite distance batches.
class HostOrchestratedSearchEngine final : public SearchEngine {
public:
  HostOrchestratedSearchEngine(
    configuration::IndexConfiguration& config,
    Context& channel_context,
    ClientConnectionManager& connection_manager,
    const MemoryRegionTokens& remote_regions);
  ~HostOrchestratedSearchEngine() override;

  HostOrchestratedSearchEngine(const HostOrchestratedSearchEngine&) = delete;
  HostOrchestratedSearchEngine& operator=(
    const HostOrchestratedSearchEngine&) = delete;

  service::QueryResult search(
    VectorDType query_dtype, const byte_t* query_data, u32 k) override;
  using SearchEngine::search;
  std::optional<u32> select_centroid_home(
    std::span<const f32> vector) const override;
  bool wait_for_maintenance(
    std::span<const u64> target_sequences,
    std::chrono::milliseconds timeout,
    std::vector<u64>* durable_sequences = nullptr,
    std::vector<u64>* effective_target_sequences = nullptr) override;
  std::vector<std::optional<maintenance_telemetry::Snapshot>>
    read_maintenance_telemetry() override;

  TelemetrySnapshot telemetry() const override;
  void reset_telemetry() override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Telemetry telemetry_;
};

}  // namespace gpu_search
