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
#include "gpu_search/types.hh"
#include "service/query_result.hh"

namespace gpu_search {

class PersistentSearchEngine {
public:
  PersistentSearchEngine(configuration::IndexConfiguration& config,
                         Context& channel_context,
                         ClientConnectionManager& connection_manager,
                         const MemoryRegionTokens& remote_regions);
  ~PersistentSearchEngine();

  PersistentSearchEngine(const PersistentSearchEngine&) = delete;
  PersistentSearchEngine& operator=(const PersistentSearchEngine&) = delete;

  service::QueryResult search(VectorDType query_dtype, const byte_t* query_data, u32 k);
  service::QueryResult search(std::span<const element_t> query, u32 k);
  std::optional<u32> select_centroid_home(
    std::span<const f32> vector) const;
  bool wait_for_maintenance(
    std::span<const u64> target_sequences,
    std::chrono::milliseconds timeout,
    std::vector<u64>* durable_sequences = nullptr,
    std::vector<u64>* effective_target_sequences = nullptr);
  std::vector<std::optional<maintenance_telemetry::Snapshot>>
    read_maintenance_telemetry();

  TelemetrySnapshot telemetry() const;
  void reset_telemetry();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Telemetry telemetry_;
};

}  // namespace gpu_search
