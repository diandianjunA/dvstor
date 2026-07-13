#pragma once

#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>

#include <library/connection_manager.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "common/vector_dtype.hh"
#include "gpu_search/delta_index.hh"
#include "gpu_search/types.hh"
#include "service/query_result.hh"

namespace gpu_search {

class MutationCapacityError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

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

  bool publish_mutations(std::vector<DeltaMutation> mutations, u64 epoch,
                         std::span<const u64> invalidated_graph_nodes = {});
  bool try_reserve_mutation_capacity(size_t mutation_count);
  void release_mutation_capacity(size_t mutation_count);
  void mark_committed_mutation_gap(const std::string& reason);
  DeltaCoordinator& delta() { return delta_; }
  const DeltaCoordinator& delta() const { return delta_; }
  TelemetrySnapshot telemetry() const { return telemetry_.snapshot(); }
  void reset_telemetry();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  DeltaCoordinator delta_;
  Telemetry telemetry_;
  std::mutex mutation_publish_mutex_;
};

}  // namespace gpu_search
