#pragma once

#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <library/connection_manager.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "common/bounded_queue.hh"
#include "common/completion_pool.hh"
#include "common/core_assignment.hh"
#include "gpu_search/search_engine.hh"
#include "memory_node/startup_protocol.hh"
#include "service/breakdown.hh"
#include "service/storage_owner_protocol.hh"
#include "service/index_metadata.hh"
#include "service/query_result.hh"

class ComputeService {
private:
  using Configuration = configuration::IndexConfiguration;
  using Assignment = CoreAssignment<interleaved>;

public:
  struct StorageOwnerSenderTelemetry {
    u64 submitted_batches{};
    u64 submitted_items{};
    u64 completed_batches{};
    u64 completed_items{};
    u64 completed_rpc_wall_ns{};
    u64 max_rpc_wall_ns{};
  };

  struct InsertItem {
    node_t id;
    vec<element_t> values;
  };

  struct Status {
    str state{"running"};
    size_t vectors_inserted{};
    u32 dimension{};
    u32 threads{};
  };

  struct LocalMainSearchOutput {
    service::QueryResult results;
    std::shared_ptr<service::breakdown::Sample> sample;
  };

public:
  explicit ComputeService(const Configuration& config);
  ~ComputeService();

  ComputeService(const ComputeService&) = delete;
  ComputeService& operator=(const ComputeService&) = delete;

  size_t insert(const vec<InsertItem>& batch);
  size_t upsert(const vec<InsertItem>& batch);
  size_t erase(const vec<node_t>& ids);
  bool wait_for_storage_maintenance(
    std::chrono::milliseconds timeout,
    vec<u64>* target_sequences = nullptr,
    vec<u64>* durable_sequences = nullptr);
  vec<node_t> search(const vec<element_t>& query, u32 k);
  vec<node_t> search_raw(VectorDType query_dtype, const byte_t* query_data, u32 dim, u32 k);
  Status status() const;
  void reset_breakdown_state();
  void clear_thread_statistics();
  service::breakdown::Report collect_breakdown_report() const;
  gpu_search::TelemetrySnapshot gpu_search_telemetry() const {
    return search_engine_ == nullptr
      ? gpu_search::TelemetrySnapshot{} : search_engine_->telemetry();
  }
  std::vector<std::optional<gpu_search::maintenance_telemetry::Snapshot>>
    storage_maintenance_telemetry() {
    return search_engine_ == nullptr
      ? std::vector<std::optional<
          gpu_search::maintenance_telemetry::Snapshot>>{}
      : search_engine_->read_maintenance_telemetry();
  }
  u64 late_storage_owner_rpc_completions() const {
    return storage_insert_late_rpc_completions_.load(
      std::memory_order_relaxed);
  }
  StorageOwnerSenderTelemetry storage_owner_sender_telemetry() const {
    return {
      .submitted_batches = storage_owner_submitted_batches_.load(
        std::memory_order_relaxed),
      .submitted_items = storage_owner_submitted_items_.load(
        std::memory_order_relaxed),
      .completed_batches = storage_owner_completed_batches_.load(
        std::memory_order_relaxed),
      .completed_items = storage_owner_completed_items_.load(
        std::memory_order_relaxed),
      .completed_rpc_wall_ns = storage_owner_completed_rpc_wall_ns_.load(
        std::memory_order_relaxed),
      .max_rpc_wall_ns = storage_owner_max_rpc_wall_ns_.load(
        std::memory_order_relaxed),
    };
  }

  const Configuration& config() const { return config_; }
private:
  struct StorageInsertTask {
    node_t id{};
    // Canonical storage bytes are produced before centroid routing. Keeping
    // the exact bytes here prevents uint8/int8 home selection from happening
    // in a different vector space and avoids a second quantization in sender.
    vec<byte_t> encoded_vector;
    service::storage_owner::MutationKind kind{service::storage_owner::MutationKind::insert};
    u32 completion_id{std::numeric_limits<u32>::max()};
    u32 stage1_home{};
    // Stable logical identity.  Transport batches may be rebuilt without
    // changing this value, so authority replay never depends on batch ordinal.
    u64 operation_id{};
    u32 operation_generation{};
    std::chrono::steady_clock::time_point enqueued_at{};
    std::chrono::steady_clock::time_point sender_dequeued_at{};
  };

  struct StorageOwnerRpcSlot {
    u32 owner_storage{};
    u32 slot_id{};
    bool in_use{false};
    bool send_done{false};
    bool response_done{false};
    bool results_completed{false};
    bool completion_claimed{false};
    bool response_valid{false};
    service::storage_owner::MutationBatchAckV2 response_ack{};
    // Kept only for the unreachable protocol-v3 parser below the v4 return;
    // no v4 receive slot is retained by an RPC batch.
    u32 response_slot_id{std::numeric_limits<u32>::max()};
    u32 item_count{};
    u64 batch_id{};
    u64 request_prepare_ns{};
    u64 cq_progress_gap_ns{};
    size_t request_size{};
    size_t response_size{};
    std::chrono::steady_clock::time_point send_posted_at{};
    std::chrono::steady_clock::time_point send_completed_at{};
    std::chrono::steady_clock::time_point response_completed_at{};
    vec<byte_t> request_buffer;
    std::unique_ptr<LocalMemoryRegion> request_region;
    vec<u32> tasks;
  };

  struct StorageOwnerResponseSlot {
    u32 owner_storage{};
    u32 slot_id{};
    vec<byte_t> buffer;
    std::unique_ptr<LocalMemoryRegion> region;
  };

  struct StorageOwnerSenderState {
    u32 task_capacity{};
    // Producers announce before canonicalization/centroid routing and retire
    // the announcement only after publishing their task into queue. This lets
    // the sender distinguish real concurrent demand from an isolated write
    // without a time-based batching delay.
    std::atomic<u32> pending_producers{0};
    // Accounting credit for cells whose Queue::push_wait has returned.
    // Multiple producers can publish out of FIFO-reservation order, so this
    // is an exact total credit but only a batching hint for the immediately
    // dequeue-visible prefix. The single progress consumer subtracts only the
    // prefix it actually popped.
    std::atomic<u32> published_tasks{0};
    // Admission backpressure is authority-wide. A busy ACK schedules a short
    // retry deadline; any real token completion clears it because the remote
    // batch-context guard is about to return capacity. Both fields are atomic
    // because ACKs are committed by the response thread while token
    // completions are consumed by the CQ/progress thread.
    std::atomic<u32> consecutive_busy_batches{0};
    std::atomic<u64> retry_not_before_ns{0};
    // One assembler per logical authority. Physical-home fanout happens at
    // the authority after acceptance; splitting this queue by home produces
    // 25 sparse flows and destroys useful foreground batching.
    std::unique_ptr<bounded::Queue<u32>> queue;
    u64 oldest_published_observed_ns{};
    std::unique_ptr<bounded::Queue<u32>> free_tasks;
    std::unique_ptr<StorageInsertTask[]> tasks;
    vec<StorageOwnerRpcSlot> slots;
    vec<StorageOwnerResponseSlot> response_slots;
    u32 completion_slot_count{};
    vec<byte_t> completion_buffer;
    std::unique_ptr<LocalMemoryRegion> completion_region;
    vec<u32> free_slots;
    // Set and cleared only by the CQ/progress thread. published_tasks is an
    // exact total but may include cells behind an invisible MPMC head, so this
    // is an observation time rather than proof that the FIFO head is visible.
    // Written only by the progress thread. Power-of-two snapshots are logged
    // so a benchmark can verify that batching is real rather than inferred
    // from submitted operation counts.
    u64 rpc_batches{};
    u64 rpc_items{};
    u64 full_batches{};
    u64 tail_escape_batches{};
    u64 max_wait_flush_batches{};
    // A later producer may publish behind a reserved-but-invisible FIFO head.
    // These counters distinguish that transient MPMC condition from remote
    // RPC or storage-maintenance stalls.
    u64 queue_visibility_stalls{};
    u64 partial_visible_batches{};
    // Raw batch critical-path telemetry. These values deliberately are not
    // divided across logical items: an item experiences the complete batch
    // response latency even though CPU work counters are per-item shares.
    u64 completed_rpc_batches{};
    u64 completed_rpc_items{};
    u64 completed_rpc_wall_ns{};
    u64 max_rpc_wall_ns{};
    u32 max_active_rpcs{};
    u32 max_published_tasks{};
    u64 busy_batches{};
    u64 busy_items{};
    u32 max_consecutive_busy_batches{};
  };

  struct StorageOwnerReadySlot {
    u32 owner_storage{};
    u32 slot_id{};
  };

  struct StorageOwnerReleasedSlot {
    u32 owner_storage{};
    u32 slot_id{};
  };

  void init_remote_tokens();
  void receive_remote_access_tokens();
  void start_storage_nodes();
  bool validate_index_metadata(const filepath_t& index_prefix, str* error_message = nullptr);
  void synchronize_clients_after_startup();
  LocalMainSearchOutput search_local_result(const vec<element_t>& query, u32 k);
  LocalMainSearchOutput search_local_raw_result(VectorDType query_dtype, const byte_t* query_data, u32 k);
  vec<node_t> search_local(const vec<element_t>& query, u32 k);
  vec<node_t> search_local_raw(VectorDType query_dtype, const byte_t* query_data, u32 k);
  void start_storage_insert_runtime();
  void stop_storage_insert_runtime();
  void release_storage_insert_runtime();
  void run_storage_insert_progress_loop();
  void run_storage_insert_completion_loop();
  bool drain_storage_owner_submissions(u32& first_owner);
  void reclaim_storage_owner_slots();
  void post_storage_owner_batch(u32 owner_storage,
                                u32 slot_id);
  void handle_storage_owner_send_completion(u32 owner_storage, u32 slot_id);
  void handle_storage_owner_response(
    u32 owner_storage,
    const service::storage_owner::MutationBatchAckV2& response,
    u32 received_bytes);
  void post_storage_owner_response_receive(u32 owner_storage, u32 response_slot_id);
  void post_storage_owner_completion_receive(u32 owner_storage,
                                             u32 completion_slot_id);
  void handle_storage_owner_token_completion(
    u32 owner_storage,
    const service::storage_owner::MutationCompletionV2& completion,
    u32 received_bytes);
  bool queue_storage_owner_completion(StorageOwnerRpcSlot& slot);
  void commit_storage_owner_slot(u32 owner_storage, u32 slot_id);
  void release_storage_owner_slot(u32 owner_storage, u32 slot_id);
  void complete_storage_owner_task(u32 owner_storage, u32 task_id, bool success);
  void fail_storage_owner_tasks(u32 owner_storage, vec<u32>& tasks);
  size_t submit_storage_owner_mutations(
    const vec<InsertItem>& items,
    service::storage_owner::MutationKind kind);

private:
  Configuration config_;
  Context context_;
  ClientConnectionManager cm_;
  const u32 num_servers_;

  MemoryRegionTokens remote_access_tokens_;
  Assignment core_assignment_;

  std::atomic<size_t> vectors_inserted_{0};

  // Both implementations expose identical query, route, and maintenance-
  // control semantics through this backend-neutral boundary.
  std::unique_ptr<gpu_search::SearchEngine> search_engine_;
  std::thread storage_insert_progress_thread_;
  std::thread storage_insert_completion_thread_;
  std::atomic<bool> storage_insert_shutdown_{false};
  std::atomic<bool> storage_insert_progress_done_{false};
  std::atomic<u32> storage_insert_inflight_{0};
  std::atomic<u64> storage_insert_late_rpc_completions_{0};
  std::atomic<u64> storage_owner_submitted_batches_{0};
  std::atomic<u64> storage_owner_submitted_items_{0};
  std::atomic<u64> storage_owner_completed_batches_{0};
  std::atomic<u64> storage_owner_completed_items_{0};
  std::atomic<u64> storage_owner_completed_rpc_wall_ns_{0};
  std::atomic<u64> storage_owner_max_rpc_wall_ns_{0};
  u64 storage_insert_current_cq_gap_ns_{};
  std::unique_ptr<bounded::Queue<StorageOwnerReadySlot>> storage_ready_slots_;
  std::unique_ptr<bounded::Queue<StorageOwnerReleasedSlot>> storage_released_slots_;
  std::unique_ptr<bounded::CompletionPool> storage_completion_pool_;
  std::unique_ptr<service::breakdown::Sample[]> storage_completion_samples_;
  std::unique_ptr<std::atomic<u64>[]> storage_maintenance_targets_;
  vec<std::unique_ptr<StorageOwnerSenderState>> storage_insert_owners_;
  // Authority replay identity includes a per-process incarnation, not the
  // reusable connection-manager client ordinal. This prevents a restarted
  // compute process from replaying an old operation token accidentally.
  u32 storage_operation_source_{};
  std::atomic<u64> next_request_id_{1};

  std::atomic<bool> breakdown_enabled_{false};
  service::breakdown::ConcurrentReport completed_breakdown_report_;
};
