#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/bounded_queue.hh"
#include "common/constants.hh"
#include "common/index_path.hh"
#include "gpu_search/centroid_route_poll_policy.hh"
#include "gpu_search/persistent_engine.hh"
#include "gpu_search/centroid_home_selector.hh"
#include "gpu_search/navigation_bootstrapper.hh"
#ifdef DVSTOR_HAVE_GPUNETIO
#include "gpu/gpunetio_transport.hh"
#endif
#include "gpu_search/index_format.hh"
#include "gpu_search/mapped_ring.hh"
#include "gpu_search/memory_budget.hh"
#include "gpu_search/pq_index.hh"
#include "gpu_search/persistent_grid_plan.hh"
#include "gpu_search/persistent_kernel.hh"
#include "gpu_search/persistent_owner_watchdog.hh"
#include "vamana/vamana_node.hh"

namespace gpu_search {

struct PersistentSearchEngine::Impl {
  enum class QuerySlotPhase : u32 {
    free,
    preparing,
    pending,
    completed,
    rejected,
  };

  // One cache-line-isolated rendezvous per device query slot. A caller owns a
  // slot from free_slots.pop until release_query_slot(); the completion thread
  // publishes only after checking both slot and request_id, which fences stale
  // GPU completions from a later reuse of the same slot.
  struct alignas(kCacheLineBytes) QuerySlotState {
    std::atomic<u32> phase{static_cast<u32>(QuerySlotPhase::free)};
    u64 request_id{};
    std::chrono::steady_clock::time_point submitted_at{};
    CompletionDescriptor completion{};
  };

  struct PendingSubmission {
    QueryDescriptor descriptor{};
    std::chrono::steady_clock::time_point enqueued_at{};
  };

  struct CentroidRouteSnapshot {
    u64 publication_sequence{};
    u64 body_checksum{};
    u64 version{};
    u64 vector_count{};
    // Canonical route representation shared by CPU update routing and GPU
    // query routing. Storage-side centroid maintenance still uses FP64 sums.
    std::vector<f32> centroid;
    std::array<DeviceCentroidRouteEntry,
               kCentroidRouteMaxLiveEntries> entries{};
    u32 live_entry_count{};
  };

  struct CentroidRouteReadResult {
    std::vector<u32> shards;
    std::vector<CentroidRouteSnapshot> snapshots;
    bool transient{};
  };

  enum class StorageRouteSyncResult : u8 {
    unchanged,
    changed,
    transient,
  };

  Impl(PersistentSearchEngine& owner,
       configuration::IndexConfiguration& config_in,
       Context& channel_context,
       ClientConnectionManager& connection_manager,
       const MemoryRegionTokens& remote_regions);
  ~Impl();

  std::string unhealthy_message();
  void mark_unhealthy(const std::string& message);
  void reject_query_slot(u32 slot);
  void reject_all_pending(const std::string& message);
  void release_query_slot(u32 slot);
  void bind_cuda_device(const char* operation) const;

  void stream_codes_to_gpu(NavigationBootstrapper& source);
  void start_persistent_kernel();
  void stop_persistent_kernel();

  service::QueryResult search(VectorDType query_dtype,
                              const byte_t* query_data, u32 k);
  std::optional<u32> select_centroid_home(
    std::span<const f32> vector) const;
  void admission_loop();
  void report_direct_path_failure();
  void completion_loop();
  void write_query_rdma_trace(const CompletionDescriptor& completion);

  void submit_centroid_route_publication(
    const CentroidRoutePublishDescriptor& descriptor);

  void validate_storage_control(const format::StorageControlBlock& control,
                                size_t shard) const;
  std::vector<format::StorageControlBlock> read_storage_controls();
  std::vector<std::optional<maintenance_telemetry::Snapshot>>
    read_maintenance_telemetry();
  CentroidRouteReadResult
    read_storage_centroid_route_publications();
  StorageRouteSyncResult synchronize_storage_routes();
  bool wait_for_maintenance(
    std::span<const u64> target_sequences,
    std::chrono::milliseconds timeout,
    std::vector<u64>* durable_sequences,
    std::vector<u64>* effective_target_sequences);
  void initialize_storage_route_descriptors();
  void maintenance_loop();

  PersistentSearchEngine& engine;
  configuration::IndexConfiguration& config;
  format::View index;
  pq::Model pq_model;
  std::vector<u64> centroid_route_versions;
  std::vector<CentroidRouteSnapshot> centroid_route_snapshots;
  // Use the C++11 atomic shared_ptr free functions.  std::atomic<shared_ptr<T>>
  // is a C++20 library specialization that is not provided by every supported
  // host toolchain (notably the GCC 11 libstdc++ used by the target system).
  std::shared_ptr<const centroid_home::Snapshot> centroid_home_snapshot;
  std::vector<format::StorageCentroidRouteDescriptor>
    storage_centroid_route_descriptors;
#ifdef DVSTOR_HAVE_GPUNETIO
  std::unique_ptr<gpu::GpuNetioPersistentTransport> direct_transport;
  gpu::GpuNetioPersistentView direct_view{};
#else
  struct EmptyDirectView {
    void** qp_array{};
    void* remote_regions{};
    u32 remote_region_count{};
    u32 qps_per_node{};
    u32 local_mkey{};
    u64 local_iova_base{};
    byte_t* data{};
    size_t data_bytes{};
    byte_t* dump{};
  } direct_view;
#endif
  std::unique_ptr<NavigationBootstrapper> control_bootstrapper;
  MappedRing<QueryDescriptor> submissions;
  MappedRing<CompletionDescriptor> completions;
  MappedRing<CentroidRoutePublishDescriptor> route_submissions;
  MappedRing<CentroidRoutePublishCompletion> route_completions;
  u32 query_slots{};
  u32 result_capacity{};
  u32 exact_width{};
  u32 code_bytes{};
  u32 dynamic_code_record_bytes{};
  u64 dynamic_code_arena_capacity{};
  u32 visited_capacity{};
  u32 node_record_bytes{};
  u32 node_record_stride{};
  u32 centroid_route_shard_capacity{};
  u32 centroid_route_entry_capacity{kCentroidRouteMaxLiveEntries};
  u32 query_dispatch_capacity{};
  u32 direct_batch_queue_count{};
  size_t dynamic_code_region_offset{};
  size_t exact_region_offset{};
  size_t graph_scratch_offset{};
  size_t control_region_offset{};
  u64 route_graph_bytes{};
  u64 explicit_gpu_bytes{};
  u64 gpu_clock_khz{1};
  DeviceShardRegion* d_shards{};
  byte_t* d_pq_codes{};
  f32* d_opq_matrix{};
  f32* d_pq_centroids{};
  f32* d_shard_centroids{};
  size_t query_input_stride{};
  f32* d_queries{};
  byte_t* query_input_host{};
  byte_t* d_query_input{};
  f32* d_transformed_queries{};
  f32* d_query_luts{};
  u64* d_navigation_candidate_handles{};
  f32* d_navigation_candidate_distances{};
  u64* d_visited{};
  byte_t* d_dynamic_code_records{};
  u32* d_dynamic_code_arena_states{};
  byte_t* d_dynamic_code_arena_records{};
  u32* d_dynamic_code_request_shards{};
  u64* d_dynamic_code_request_offsets{};
  u64* d_dynamic_code_request_local_iovas{};
  // Optional packed per-base-node extent classes. The at-rest sidecar remains
  // u8/node; the device copy is u32-aligned so query CTAs can monotonically
  // repair stale bytes with CAS. Dynamic records always use the full record.
  u32* d_graph_extent_class_words{};
  u32* d_graph_request_bytes{};
  u32* d_speculative_graph_request_shards{};
  u64* d_speculative_graph_request_offsets{};
  u64* d_speculative_graph_request_local_iovas{};
  u32* d_speculative_graph_request_bytes{};
  u64* d_speculative_graph_request_handles{};
  u8* d_speculative_graph_validation_states{};
  u64 graph_extent_sidecar_bytes{};
  u64* d_query_dispatch_enqueue{};
  u64* d_query_dispatch_dequeue{};
  u64* d_query_dispatch_sequences{};
  QueryDescriptor* d_query_dispatch_entries{};
  u64* d_direct_batch_enqueue{};
  u64* d_direct_batch_dequeue{};
  u64* d_direct_batch_sequences{};
  DirectBatchDescriptor* d_direct_batch_entries{};
  DeviceRingView<DirectBatchDescriptor>* d_direct_batch_queues{};
  // Standalone shadow-frontier reads use a disjoint per-QP ring.  Keeping
  // them out of the critical producer ring lets the persistent owner enforce
  // critical-first admission without a global scheduler or producer atomics.
  u64* d_direct_speculative_batch_enqueue{};
  u64* d_direct_speculative_batch_dequeue{};
  u64* d_direct_speculative_batch_sequences{};
  DirectBatchDescriptor* d_direct_speculative_batch_entries{};
  DeviceRingView<DirectBatchDescriptor>*
    d_direct_speculative_batch_queues{};
  i32* d_direct_batch_statuses{};
  u64* d_direct_batch_completion_timestamps_ns{};
  i32* d_core_batch_statuses{};
  u64* d_core_batch_completion_timestamps_ns{};
  i32* d_tail_batch_statuses{};
  u64* d_tail_batch_completion_timestamps_ns{};
  QueryRdmaTraceHeader* d_query_rdma_trace_headers{};
  QueryRdmaTraceEvent* d_query_rdma_trace_events{};
  std::ofstream query_rdma_trace_stream;
  std::mutex query_rdma_trace_mutex;
  u32* direct_owner_phases_host{};
  u32* d_direct_owner_phases{};
  DirectOwnerProgress* direct_owner_progress_host{};
  DirectOwnerProgress* d_direct_owner_progress{};
  std::atomic<u64> owner_submitted_wqes_baseline{0};
  std::atomic<u64> owner_submission_wqe_capacity_baseline{0};
  std::atomic<u64> owner_critical_batches_baseline{0};
  std::atomic<u64> owner_speculative_batches_baseline{0};
  u32* query_kernel_ready_host{};
  u32* d_query_kernel_ready{};
  u32* dispatcher_kernel_ready_host{};
  u32* d_dispatcher_kernel_ready{};
  u32* control_kernel_ready_host{};
  u32* d_control_kernel_ready{};
  byte_t* d_exact_records{};
  byte_t* d_remote_buffer{};
  byte_t* d_graph_scratch{};
  format::StorageControlBlock* d_control_snapshots{};
  maintenance_telemetry::Snapshot* d_maintenance_snapshots{};
  u64* d_maintenance_sequence_after{};
  byte_t* d_storage_route_snapshots{};
  size_t storage_route_snapshot_stride{};
  u64* d_storage_route_sequence_after{};
  bool owns_remote_buffer{};
  u32* result_ids_host{};
  f32* result_distances_host{};
  u32* d_result_ids{};
  f32* d_result_distances{};
  CentroidRouteUpdate* centroid_route_updates_host{};
  CentroidRouteUpdate* d_centroid_route_updates{};
  f32* centroid_route_centroid_updates_host{};
  f32* d_centroid_route_centroid_updates{};
  DeviceCentroidRouteShard* d_centroid_route_shards{};
  DeviceCentroidRouteEntry* d_centroid_route_entries{};
  u64* d_centroid_route_epoch{};
  u32 route_poll_salt{};
  u32* stop_host{};
  u32* stop_device{};
  u32* direct_disabled_host{};
  u32* direct_disabled_device{};
  i32* direct_error_host{};
  i32* direct_error_device{};
  cudaStream_t kernel_stream{};
  cudaStream_t route_stream{};
  cudaStream_t rdma_stream{};
  PersistentKernelParams kernel_params{};
  PersistentGridPlan persistent_grid_plan{};
  PersistentKernelOccupancy persistent_kernel_occupancy{};
  u32 kernel_threads{};
  u32 owner_kernel_blocks{};
  u32 kernel_blocks{};
  bool kernel_running{};
  std::atomic<bool> direct_failure_logged{false};
  std::atomic<u32> slow_query_logs{0};
  std::atomic<bool> accepting{true};
  std::atomic<bool> healthy{true};
  std::atomic<bool> shutdown{false};
  std::atomic<bool> query_stop{false};
  std::atomic<bool> maintenance_shutdown{false};
  std::atomic<u64> next_request_id{1};
  std::atomic<u64> next_route_command_id{1};
  std::string health_error;
  std::mutex health_mutex;
  std::unique_ptr<QuerySlotState[]> query_slot_states;
  std::unique_ptr<bounded::Queue<u32>> free_slots;
  std::unique_ptr<bounded::Queue<PendingSubmission>> admission_queue;
  std::thread admission_thread;
  std::thread completion_thread;
  std::mutex maintenance_mutex;
  std::mutex storage_control_read_mutex;
  std::condition_variable maintenance_cv;
  std::thread maintenance_thread;

};

}  // namespace gpu_search
