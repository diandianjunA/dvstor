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
#include <deque>
#include <exception>
#include <fstream>
#include <future>
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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/index_path.hh"
#include "gpu_search/persistent_engine.hh"
#include "gpu_search/navigation_bootstrapper.hh"
#ifdef DVSTOR_HAVE_GPUNETIO
#include "gpu/gpunetio_transport.hh"
#endif
#include "gpu_search/index_format.hh"
#include "gpu_search/mapped_ring.hh"
#include "gpu_search/memory_budget.hh"
#include "gpu_search/pq_index.hh"
#include "gpu_search/persistent_kernel.hh"
#include "vamana/anchor_index.hh"
#include "vamana/vamana_node.hh"

namespace gpu_search {

namespace persistent_engine_detail {

struct AnchorTable {
  u32 dim{};
  std::vector<f32> vectors;
  std::vector<u32> handles;
  std::vector<u64> raw_pointers;
  std::vector<u32> shard_offsets;

  u32 count() const {
    return dim == 0 ? 0 : static_cast<u32>(vectors.size() / dim);
  }
};

}  // namespace persistent_engine_detail

struct PersistentSearchEngine::Impl {
  struct PendingQuery {
    u32 slot{};
    std::chrono::steady_clock::time_point submitted_at{};
    std::promise<service::QueryResult> promise;
  };

  struct PendingSubmission {
    QueryDescriptor descriptor{};
    std::chrono::steady_clock::time_point enqueued_at{};
  };

  struct RetiredDeltaBatch {
    u64 query_ticket_barrier{};
    std::vector<u32> slots;
  };

  struct RetiredResidentPqBatch {
    u64 query_ticket_barrier{};
    std::vector<ResidentPqEraseUpdate> entries;
  };

  struct PendingStorageReclaimAck {
    u64 maintenance_sequence{};
    u64 query_ticket_barrier{};
  };

  struct DurableRetirement {
    node_t id{};
    service::storage_owner::MutationKind kind{
      service::storage_owner::MutationKind::insert};
    u64 epoch{};
    u64 remote_node{};
    u64 old_remote_node{};
  };

  Impl(PersistentSearchEngine& owner,
       configuration::IndexConfiguration& config_in,
       Context& channel_context,
       ClientConnectionManager& connection_manager,
       const MemoryRegionTokens& remote_regions);
  ~Impl();

  std::string unhealthy_message();
  void reject_submission(const PendingSubmission& submission,
                         const std::string& message);
  void mark_unhealthy(const std::string& message);
  void reject_queued_submissions(const std::string& message);
  void reject_all_pending(const std::string& message);
  void bind_cuda_device(const char* operation) const;

  void stream_codes_to_gpu(NavigationBootstrapper& source);
  void stream_anchor_graph_to_gpu(NavigationBootstrapper& source);
  void clear_delta_device_state(cudaStream_t stream = nullptr);
  void start_persistent_kernel();
  void stop_persistent_kernel();

  service::QueryResult search(VectorDType query_dtype,
                              const byte_t* query_data, u32 k);
  void admission_loop();
  void report_direct_path_failure();
  void completion_loop();

  void decode_mutation_payload(const DeltaMutation& mutation,
                               std::vector<f32>& decoded) const;
  u32 nearest_anchor(const std::vector<f32>& vector, u64 remote_node) const;
  u64 graph_cache_key(u64 raw) const;
  std::vector<u64> graph_cache_keys(std::span<const u64> raw_nodes) const;
  void refresh_anchor_graph_records(std::span<const u64> invalidation_keys);

  void submit_delta_publication(const DeltaPublishDescriptor& descriptor);
  size_t active_resident_pq_slots_locked() const;
  u32 allocate_resident_pq_slot_locked(u64 remote_node);
  void upload_records_locked(std::vector<DeltaMutation>& mutations,
                             std::span<const u64> invalidation_keys = {});
  size_t upload_mutations(std::vector<DeltaMutation>& mutations, u64 epoch,
                          std::span<const u64> invalidated_graph_nodes);
  size_t active_delta_slots_locked() const;
  bool query_ticket_barrier_passed(u64 barrier) const;
  bool durable_snapshot_safe(u64 durable_epoch) const;
  void reclaim_retired_delta_slots_locked();

  void validate_storage_control(const format::StorageControlBlock& control,
                                size_t shard) const;
  std::vector<format::StorageControlBlock> read_storage_controls();
  void write_storage_reclaim_acks(std::span<const u64> sequences);
  void initialize_storage_reclaim_ack();
  void enqueue_storage_reclaim_barriers();
  void publish_ready_storage_reclaim_acks();
  std::vector<DeltaMutation> retire_durable_delta();
  void mark_durable_delta_records_locked(
    std::span<const DurableRetirement> retired);
  void maintenance_loop();

  PersistentSearchEngine& engine;
  configuration::IndexConfiguration& config;
  format::View index;
  pq::Model pq_model;
  persistent_engine_detail::AnchorTable anchor_table;
  std::vector<u32> entry_handles;
  std::unordered_map<u64, u32> anchor_buckets_by_raw;
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
  MappedRing<DeltaPublishDescriptor> delta_submissions;
  MappedRing<DeltaPublishCompletion> delta_completions;
  u32 query_slots{};
  u32 result_capacity{};
  u32 exact_width{};
  u32 code_bytes{};
  u32 visited_capacity{};
  u32 node_record_bytes{};
  u32 delta_capacity{};
  u32 delta_table_capacity{};
  u32 resident_pq_capacity{};
  u32 resident_pq_table_capacity{};
  u32 permanent_override_words{};
  u32 graph_cache_sets{};
  u32 graph_cache_slots{};
  u32 exact_cache_sets{};
  u32 exact_cache_slots{};
  u32 exact_cache_stride{};
  u32 graph_admission_sets{};
  u32 exact_admission_sets{};
  u32 graph_invalidation_capacity{};
  u32 delta_command_capacity{};
  u32 query_dispatch_capacity{};
  u32 direct_batch_queue_count{};
  size_t graph_cache_bytes{};
  size_t exact_cache_bytes{};
  size_t anchor_graph_region_offset{};
  size_t dynamic_code_region_offset{};
  size_t exact_region_offset{};
  size_t exact_cache_offset{};
  size_t graph_cache_offset{};
  size_t graph_scratch_offset{};
  size_t control_region_offset{};
  u64 route_graph_bytes{};
  u64 resident_pq_bytes{};
  u64 explicit_gpu_bytes{};
  u64 gpu_clock_khz{1};
  DeviceShardRegion* d_shards{};
  byte_t* d_pq_codes{};
  f32* d_opq_matrix{};
  f32* d_pq_centroids{};
  u32* d_entry_points{};
  f32* d_anchor_vectors{};
  u32* d_anchor_handles{};
  u8* d_anchor_pq_codes{};
  std::vector<u64> anchor_graph_keys_host;
  std::vector<u32> anchor_graph_ready_states_host;
  u32* anchor_graph_readers_host{};
  byte_t* anchor_graph_validation_host{};
  u64* d_anchor_graph_keys{};
  u32* d_anchor_graph_states{};
  u32* d_anchor_graph_readers{};
  u32* d_delta_bucket_heads{};
  size_t query_input_stride{};
  f32* d_queries{};
  byte_t* query_input_host{};
  byte_t* d_query_input{};
  f32* d_transformed_queries{};
  f32* d_query_luts{};
  u32* d_navigation_candidate_handles{};
  f32* d_navigation_candidate_distances{};
  u32* d_visited{};
  byte_t* d_dynamic_code_records{};
  u32* d_dynamic_code_request_shards{};
  u64* d_dynamic_code_request_offsets{};
  u64* d_dynamic_code_request_local_iovas{};
  u64* d_query_dispatch_enqueue{};
  u64* d_query_dispatch_dequeue{};
  u64* d_query_dispatch_sequences{};
  QueryDescriptor* d_query_dispatch_entries{};
  u64* d_direct_batch_enqueue{};
  u64* d_direct_batch_dequeue{};
  u64* d_direct_batch_sequences{};
  DirectBatchDescriptor* d_direct_batch_entries{};
  DeviceRingView<DirectBatchDescriptor>* d_direct_batch_queues{};
  i32* d_direct_batch_statuses{};
  u32* direct_owner_phases_host{};
  u32* d_direct_owner_phases{};
  u32* query_kernel_ready_host{};
  u32* d_query_kernel_ready{};
  u32* dispatcher_kernel_ready_host{};
  u32* d_dispatcher_kernel_ready{};
  u32* control_kernel_ready_host{};
  u32* d_control_kernel_ready{};
  byte_t* d_exact_records{};
  byte_t* d_exact_cache{};
  byte_t* d_remote_buffer{};
  byte_t* d_anchor_graph_records{};
  byte_t* d_graph_cache{};
  byte_t* d_graph_scratch{};
  format::StorageControlBlock* d_control_snapshots{};
  u64* d_graph_cache_keys{};
  u64* d_graph_cache_generations{};
  u64* d_graph_cache_timestamps{};
  u32* d_graph_cache_states{};
  u32* d_graph_cache_readers{};
  u32* d_graph_cache_victims{};
  u64* d_graph_admission_keys{};
  u32* d_graph_admission_victims{};
  u64* d_graph_cache_generation{};
  u64* graph_invalidation_keys_host{};
  u64* d_graph_invalidation_keys{};
  u32* d_exact_cache_keys{};
  u32* d_exact_cache_states{};
  u32* d_exact_cache_readers{};
  u32* d_exact_cache_victims{};
  u32* d_exact_admission_keys{};
  u32* d_exact_admission_victims{};
  bool owns_remote_buffer{};
  u32* result_ids_host{};
  f32* result_distances_host{};
  u32* d_result_ids{};
  f32* d_result_distances{};
  DeviceDeltaRecord* d_delta_records{};
  byte_t* d_delta_vectors{};
  byte_t* d_delta_pq_codes{};
  f32* d_delta_encode_scratch{};
  u32* delta_staging_slots_host{};
  u32* d_delta_staging_slots{};
  DeviceDeltaRecord* delta_staging_records_host{};
  DeviceDeltaRecord* d_delta_staging_records{};
  byte_t* delta_staging_vectors_host{};
  byte_t* d_delta_staging_vectors{};
  u32* d_delta_next{};
  u32* d_delta_prev{};
  u32* d_delta_remote_positions{};
  u32* d_base_override_keys{};
  u64* d_base_override_epochs{};
  u32* d_permanent_override_bits{};
  u64* d_delta_remote_keys{};
  u32* d_delta_remote_slots{};
  byte_t* d_resident_pq_codes{};
  u64* d_resident_pq_keys{};
  u32* d_resident_pq_slots{};
  u32* d_resident_pq_positions{};
  DeltaSupersedeUpdate* delta_supersede_updates_host{};
  DeltaSupersedeUpdate* d_delta_supersede_updates{};
  DeltaOverrideUpdate* delta_override_updates_host{};
  DeltaOverrideUpdate* d_delta_override_updates{};
  DeltaDurableUpdate* delta_durable_updates_host{};
  DeltaDurableUpdate* d_delta_durable_updates{};
  ResidentPqEraseUpdate* resident_pq_erase_updates_host{};
  ResidentPqEraseUpdate* d_resident_pq_erase_updates{};
  u32* d_delta_count{};
  std::vector<DeviceDeltaRecord> delta_records_host;
  std::vector<u32> free_delta_slots;
  std::unordered_map<node_t, std::vector<u32>> superseded_delta_slots;
  std::deque<RetiredDeltaBatch> retired_delta_batches;
  std::deque<RetiredResidentPqBatch> retired_resident_pq_batches;
  std::multimap<u64, DurableRetirement> pending_durable_retirements;
  std::unordered_map<node_t, u32> latest_delta_slot;
  std::unordered_map<u64, u32> resident_pq_slots_by_remote;
  std::vector<u32> free_resident_pq_slots;
  u32 resident_pq_high_watermark{};
  std::unordered_map<u32, u64> base_override_epochs;
  std::vector<std::deque<std::pair<u64, std::chrono::steady_clock::time_point>>>
    durable_sequence_history;
  std::vector<u64> observed_durable_sequences;
  std::vector<u64> safe_durable_sequences;
  std::vector<std::deque<PendingStorageReclaimAck>> pending_storage_reclaim_acks;
  std::vector<u64> enqueued_reclaim_ack_sequences;
  std::vector<u64> published_reclaim_ack_sequences;
  std::mutex delta_mutex;
  size_t reserved_mutation_capacity{};
  u64 mutable_delta_entries{};
  u64 durable_delta_entries{};
  u32 compute_client_id{};
  u32 compute_client_count{};
  u32* stop_host{};
  u32* stop_device{};
  u32* direct_disabled_host{};
  u32* direct_disabled_device{};
  i32* direct_error_host{};
  i32* direct_error_device{};
  cudaStream_t kernel_stream{};
  cudaStream_t delta_stream{};
  cudaStream_t rdma_stream{};
  cudaStream_t route_refresh_stream{};
  PersistentKernelParams kernel_params{};
  u32 owner_kernel_blocks{};
  u32 kernel_blocks{};
  bool kernel_running{};
  std::atomic<bool> direct_failure_logged{false};
  std::atomic<u32> slow_query_logs{0};
  std::atomic<bool> accepting{true};
  std::atomic<bool> healthy{true};
  std::atomic<bool> shutdown{false};
  std::atomic<bool> maintenance_shutdown{false};
  std::atomic<u64> active_gpu_queries{0};
  std::atomic<u64> next_query_ticket{1};
  std::atomic<u64> next_request_id{1};
  std::atomic<u64> next_delta_command_id{1};
  std::atomic<u64> pending_count{0};
  std::mutex admission_mutex;
  std::condition_variable admission_cv;
  std::deque<PendingSubmission> admission_queue;
  std::string health_error;
  std::mutex slot_mutex;
  std::condition_variable slot_cv;
  std::vector<u32> free_slots;
  std::unique_ptr<std::atomic<u64>[]> active_query_tickets;
  std::unique_ptr<std::atomic<u64>[]> active_query_snapshots;
  std::mutex query_snapshot_mutex;
  std::mutex pending_mutex;
  std::unordered_map<u64, std::shared_ptr<PendingQuery>> pending_queries;
  std::thread admission_thread;
  std::thread completion_thread;
  std::mutex maintenance_mutex;
  std::condition_variable maintenance_cv;
  std::thread maintenance_thread;

};

}  // namespace gpu_search
