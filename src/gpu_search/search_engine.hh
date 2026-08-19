#pragma once

#include <chrono>
#include <optional>
#include <span>
#include <vector>

#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "gpu_search/maintenance_telemetry.hh"
#include "gpu_search/types.hh"
#include "service/query_result.hh"

namespace gpu_search {

// Query execution and storage-control observation share one service-facing
// contract.  The persistent GPU implementation and the CPU-orchestrated
// baseline deliberately differ below this boundary while consuming the same
// schema-v16 index, route publications, and maintenance control pages.
class SearchEngine {
public:
  virtual ~SearchEngine() = default;

  SearchEngine(const SearchEngine&) = delete;
  SearchEngine& operator=(const SearchEngine&) = delete;

  virtual service::QueryResult search(
    VectorDType query_dtype, const byte_t* query_data, u32 k) = 0;

  service::QueryResult search(std::span<const element_t> query, u32 k) {
    return search(VectorDType::float32,
                  reinterpret_cast<const byte_t*>(query.data()), k);
  }

  virtual std::optional<u32> select_centroid_home(
    std::span<const f32> vector) const = 0;
  virtual bool wait_for_maintenance(
    std::span<const u64> target_sequences,
    std::chrono::milliseconds timeout,
    std::vector<u64>* durable_sequences = nullptr,
    std::vector<u64>* effective_target_sequences = nullptr) = 0;
  virtual std::vector<std::optional<maintenance_telemetry::Snapshot>>
    read_maintenance_telemetry() = 0;

  virtual TelemetrySnapshot telemetry() const = 0;
  virtual void reset_telemetry() = 0;

protected:
  SearchEngine() = default;
};

}  // namespace gpu_search
