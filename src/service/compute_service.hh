#pragma once

#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <library/connection_manager.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "common/bounded_queue.hh"
#include "common/completion_pool.hh"
#include "common/core_assignment.hh"
#include "gpu_search/persistent_engine.hh"
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
  vec<node_t> search(const vec<element_t>& query, u32 k);
  vec<node_t> search_raw(VectorDType query_dtype, const byte_t* query_data, u32 dim, u32 k);
  Status status() const;
  void reset_breakdown_state();
  void clear_thread_statistics();
  service::breakdown::Report collect_breakdown_report() const;
  gpu_search::TelemetrySnapshot gpu_search_telemetry() const {
    return persistent_search_ == nullptr
      ? gpu_search::TelemetrySnapshot{} : persistent_search_->telemetry();
  }
  u64 late_storage_owner_rpc_completions() const {
    return storage_insert_late_rpc_completions_.load(
      std::memory_order_relaxed);
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
    std::unique_ptr<bounded::Queue<u32>> queue;
    std::unique_ptr<bounded::Queue<u32>> free_tasks;
    std::unique_ptr<StorageInsertTask[]> tasks;
    vec<StorageOwnerRpcSlot> slots;
    vec<StorageOwnerResponseSlot> response_slots;
    vec<u32> free_slots;
  };

  struct StorageOwnerReadySlot {
    u32 owner_storage{};
    u32 slot_id{};
  };

  struct StorageOwnerReleasedSlot {
    u32 owner_storage{};
    u32 slot_id{};
    u32 response_slot_id{};
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
  void handle_storage_owner_response(u32 owner_storage,
                                     u32 response_slot_id,
                                     u32 received_bytes);
  void post_storage_owner_response_receive(u32 owner_storage, u32 response_slot_id);
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

  std::unique_ptr<gpu_search::PersistentSearchEngine> persistent_search_;
  std::thread storage_insert_progress_thread_;
  std::thread storage_insert_completion_thread_;
  std::atomic<bool> storage_insert_shutdown_{false};
  std::atomic<bool> storage_insert_progress_done_{false};
  std::atomic<u32> storage_insert_inflight_{0};
  std::atomic<u64> storage_insert_late_rpc_completions_{0};
  u64 storage_insert_current_cq_gap_ns_{};
  std::unique_ptr<bounded::Queue<StorageOwnerReadySlot>> storage_ready_slots_;
  std::unique_ptr<bounded::Queue<StorageOwnerReleasedSlot>> storage_released_slots_;
  std::unique_ptr<bounded::CompletionPool> storage_completion_pool_;
  std::unique_ptr<service::breakdown::Sample[]> storage_completion_samples_;
  vec<std::unique_ptr<StorageOwnerSenderState>> storage_insert_owners_;
  std::atomic<u64> next_request_id_{1};

  mutable std::mutex breakdown_mutex_;
  std::atomic<bool> breakdown_enabled_{false};
  service::breakdown::Report completed_breakdown_report_;
};
