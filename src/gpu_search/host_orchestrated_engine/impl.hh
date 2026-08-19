#pragma once

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <library/detached_qp.hh>
#include <library/memory_region.hh>

#include "common/bounded_queue.hh"
#include "gpu_search/centroid_home_selector.hh"
#include "gpu_search/host_orchestrated_engine.hh"
#include "gpu_search/index_format.hh"
#include "gpu_search/pq_index.hh"

namespace gpu_search {

struct HostOrchestratedSearchEngine::Impl {
  struct Candidate {
    RemotePtr pointer{};
    f32 distance{};
    bool expanded{};
    u8 extent_class{0xffu};
  };

  struct RouteShard {
    u64 publication_sequence{};
    u64 body_checksum{};
    u64 version{};
    u64 vector_count{};
    std::vector<f32> centroid;
    std::array<format::StorageCentroidRouteEntry,
               format::kStorageCentroidRouteMaxLiveEntries> entries{};
    u32 live_entry_count{};
  };

  struct RouteSnapshot {
    centroid_home::Snapshot home;
    std::vector<RouteShard> shards;
  };

  struct ReadRequest {
    u32 shard{};
    u64 remote_offset{};
    size_t local_offset{};
    u32 bytes{};
  };

  struct Lane {
    Lane(Impl& engine, u32 lane_id);
    ~Lane();

    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;

    Impl& engine;
    u32 id{};
    std::vector<std::unique_ptr<DetachedQP>> qps;
    byte_t* scratch{};
    size_t scratch_bytes{};
    std::unique_ptr<LocalMemoryRegion> scratch_region;
    cudaStream_t stream{};
    f32* d_query{};
    f32* d_lut{};
    u32* d_ordinals{};
    u8* d_dynamic_codes{};
    f32* d_distances{};
    byte_t* d_exact_records{};
    f32* d_exact_distances{};
    std::vector<f32> query;
    std::vector<f32> transformed;
    std::vector<f32> lut;
    std::vector<u32> ordinals;
    std::vector<u8> packed_dynamic_codes;
    std::vector<f32> distances;
    std::vector<Candidate> beam;
    std::vector<Candidate> pending;
    std::unordered_set<u64> visited;
    bool poisoned{};
  };

  struct LaneGuard {
    Impl* engine{};
    u32 lane{};
    LaneGuard() = default;
    LaneGuard(Impl* owner, u32 index) : engine(owner), lane(index) {}
    LaneGuard(const LaneGuard&) = delete;
    LaneGuard& operator=(const LaneGuard&) = delete;
    LaneGuard(LaneGuard&& other) noexcept
        : engine(other.engine), lane(other.lane) { other.engine = nullptr; }
    ~LaneGuard();
    Lane& get() const { return *engine->lanes[lane]; }
  };

  Impl(HostOrchestratedSearchEngine& owner,
       configuration::IndexConfiguration& config,
       Context& channel_context,
       ClientConnectionManager& connection_manager,
       const MemoryRegionTokens& remote_regions);
  ~Impl();

  service::QueryResult search(VectorDType query_dtype,
                              const byte_t* query_data, u32 k);
  std::optional<u32> select_centroid_home(
    std::span<const f32> vector) const;
  bool wait_for_maintenance(
    std::span<const u64> target_sequences,
    std::chrono::milliseconds timeout,
    std::vector<u64>* durable_sequences,
    std::vector<u64>* effective_target_sequences);
  std::vector<std::optional<maintenance_telemetry::Snapshot>>
    read_maintenance_telemetry();

  LaneGuard acquire_lane(bool account_query_wait = false);
  void release_lane(u32 lane);
  std::string unhealthy_message() const;
  void mark_unhealthy(Lane* lane, const std::string& message);
  void check_lane_cuda(Lane& lane, cudaError_t status,
                       const char* operation);
  void read_batch(Lane& lane, std::span<const ReadRequest> requests,
                  bool account_query_io);
  void stream_codes_to_gpu();
  std::vector<format::StorageControlBlock> read_storage_controls(Lane& lane);
  void validate_storage_control(
    const format::StorageControlBlock& control, size_t shard) const;
  // Returns true only when at least one shard publication changed.  A stable
  // poll must remain distinguishable so the background refresher can back off
  // instead of imposing a permanent high-frequency control-plane load.
  bool synchronize_storage_routes(Lane& lane);
  void initialize_storage_routes(Lane& lane);
  void maintenance_loop();
  std::vector<std::optional<maintenance_telemetry::Snapshot>>
    read_maintenance_telemetry(Lane& lane);

  std::vector<Candidate> route_seeds(
    const RouteSnapshot& routes, std::span<const f32> query) const;
  void score_candidates(Lane& lane, std::vector<Candidate>& candidates);
  void fetch_graph_wave(
    Lane& lane, std::span<Candidate*> wave,
    std::vector<std::vector<RemotePtr>>& neighbors);
  service::QueryResult exact_rerank(
    Lane& lane, std::span<const Candidate> beam, u32 k);
  service::QueryResult execute_query(
    Lane& lane, VectorDType query_dtype, const byte_t* query_data, u32 k,
    const std::shared_ptr<const RouteSnapshot>& routes);

  HostOrchestratedSearchEngine& engine;
  configuration::IndexConfiguration& config;
  Context& channel_context;
  ClientConnectionManager& connection_manager;
  const MemoryRegionTokens& remote_regions;
  Context data_context;
  format::View index;
  pq::Model pq_model;
  std::vector<u8> graph_extent_classes;
  std::vector<format::StorageCentroidRouteDescriptor> route_descriptors;
  std::vector<RouteShard> route_cache;
  std::shared_ptr<const RouteSnapshot> route_snapshot;
  std::vector<std::unique_ptr<Lane>> lanes;
  std::unique_ptr<bounded::Queue<u32>> free_lanes;
  std::atomic<bool> stopping{false};
  std::atomic<bool> healthy{true};
  std::atomic<bool> maintenance_shutdown{false};
  mutable std::mutex health_mutex;
  std::string health_error;
  std::mutex maintenance_mutex;
  std::condition_variable maintenance_cv;
  std::mutex route_refresh_mutex;
  std::thread maintenance_thread;
  u32 route_poll_salt{};
  u32 code_bytes{};
  u32 dynamic_code_record_bytes{};
  u32 graph_entry_bytes{};
  u32 graph_entry_capacity{};
  u32 score_capacity{};
  u32 exact_capacity{};
  u32 exact_record_bytes{};
  u32 exact_record_stride{};
  u64 storage_region_bytes{};
  u64 dynamic_allocation_limit{};
  size_t graph_scratch_offset{};
  size_t dynamic_code_scratch_offset{};
  size_t exact_scratch_offset{};
  size_t exact_header_scratch_offset{};
  size_t control_scratch_offset{};
  size_t route_scratch_offset{};
  size_t route_sequence_scratch_offset{};
  size_t lane_scratch_bytes{};
  size_t route_snapshot_stride{};
  u8* d_base_codes{};
};

}  // namespace gpu_search
