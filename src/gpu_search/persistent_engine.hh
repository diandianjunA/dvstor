#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include <library/connection_manager.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "common/vector_dtype.hh"
#include "gpu_search/maintenance_telemetry.hh"
#include "gpu_search/search_engine.hh"
#include "gpu_search/types.hh"
#include "service/query_result.hh"

namespace gpu_search {

class PersistentSearchEngine final : public SearchEngine {
public:
  PersistentSearchEngine(configuration::IndexConfiguration& config,
                         Context& channel_context,
                         ClientConnectionManager& connection_manager,
                         const MemoryRegionTokens& remote_regions);
  ~PersistentSearchEngine() override;

  PersistentSearchEngine(const PersistentSearchEngine&) = delete;
  PersistentSearchEngine& operator=(const PersistentSearchEngine&) = delete;

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
