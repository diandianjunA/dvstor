#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <library/connection_manager.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "common/core_assignment.hh"
#include "gpu_search/persistent_engine.hh"
#include "memory_node/startup_protocol.hh"
#include "service/breakdown.hh"
#include "service/storage_owner_protocol.hh"
#include "service/index_metadata.hh"
#include "service/query_result.hh"
#include "vamana/anchor_index.hh"

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

  const Configuration& config() const { return config_; }
private:
  struct StorageInsertTask {
    InsertItem item;
    service::storage_owner::MutationKind kind{service::storage_owner::MutationKind::insert};
    std::shared_ptr<service::breakdown::Sample> sample;
    std::promise<bool> result;
    std::chrono::steady_clock::time_point enqueued_at{};
    std::chrono::steady_clock::time_point sender_dequeued_at{};
    vec<RemotePtr> anchor_hints;
    RemotePtr anchor_bucket_hint;
  };

  struct StorageOwnerRpcSlot {
    u32 owner_storage{};
    u32 slot_id{};
    bool in_use{false};
    bool send_done{false};
    bool response_done{false};
    bool results_completed{false};
    bool completion_claimed{false};
    u32 gpu_reserved_items{};
    u32 item_count{};
    u64 batch_id{};
    u64 batch_wait_ns{};
    u64 request_prepare_ns{};
    size_t request_size{};
    size_t response_size{};
    std::chrono::steady_clock::time_point send_posted_at{};
    std::chrono::steady_clock::time_point send_completed_at{};
    std::chrono::steady_clock::time_point response_completed_at{};
    vec<byte_t> request_buffer;
    vec<byte_t> response_buffer;
    std::unique_ptr<LocalMemoryRegion> request_region;
    std::unique_ptr<LocalMemoryRegion> response_region;
    vec<std::unique_ptr<StorageInsertTask>> tasks;
    vec<std::shared_ptr<service::breakdown::Sample>> samples;
  };

  struct StorageOwnerResponseSlot {
    u32 owner_storage{};
    u32 slot_id{};
    vec<byte_t> buffer;
    std::unique_ptr<LocalMemoryRegion> region;
  };

  struct StorageOwnerSenderState {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::unique_ptr<StorageInsertTask>> queue;
    vec<StorageOwnerRpcSlot> slots;
    vec<StorageOwnerResponseSlot> response_slots;
    std::deque<u32> free_slots;
    dense_hashmap_t<u64, u32> batch_to_slot;
    std::thread thread;
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
  void run_storage_insert_sender(u32 owner_storage);
  void run_storage_insert_completion_loop();
  void post_storage_owner_batch(u32 owner_storage,
                                u32 slot_id,
                                vec<std::unique_ptr<StorageInsertTask>>&& tasks,
                                u64 batch_wait_ns);
  void handle_storage_owner_send_completion(u32 owner_storage, u32 slot_id);
  void handle_storage_owner_response(u32 owner_storage, u32 response_slot_id);
  void post_storage_owner_response_receive(u32 owner_storage, u32 response_slot_id);
  void complete_ready_storage_owner_slots();
  void maybe_release_storage_owner_slot_locked(StorageOwnerSenderState& state,
                                               StorageOwnerRpcSlot& slot,
                                               bool gpu_visible);
  void fail_storage_owner_tasks(vec<std::unique_ptr<StorageInsertTask>>& tasks);
  void publish_compute_side_id(node_t id, RemotePtr ptr, bool deleted, u32 owner_storage);
  bool lookup_compute_side_id(node_t id, RemotePtr* ptr, bool* deleted = nullptr) const;
  u32 storage_owner_for_id(node_t id) const;
  vamana::anchor::Route route_storage_owner_update(const InsertItem& item,
                                                    std::optional<u32> owner_override = std::nullopt) const;

private:
  Configuration config_;
  Context context_;
  ClientConnectionManager cm_;
  const u32 num_servers_;

  MemoryRegionTokens remote_access_tokens_;
  Assignment core_assignment_;

  std::atomic<size_t> vectors_inserted_{0};

  std::unique_ptr<vamana::anchor::Index> anchor_index_;
  std::unique_ptr<gpu_search::PersistentSearchEngine> persistent_search_;
  std::thread storage_insert_completion_thread_;
  std::atomic<bool> storage_insert_shutdown_{false};
  std::atomic<bool> storage_insert_senders_done_{false};
  std::atomic<u32> storage_insert_inflight_{0};
  std::atomic<u32> storage_insert_timeout_logs_{0};
  vec<std::unique_ptr<StorageOwnerSenderState>> storage_insert_owners_;
  struct ComputeSideIdEntry {
    RemotePtr ptr;
    bool deleted{};
    u32 owner_storage{};
  };
  mutable std::mutex compute_side_idmap_mutex_;
  hashmap_t<node_t, ComputeSideIdEntry> compute_side_idmap_;

  std::atomic<u64> next_request_id_{1};

  mutable std::mutex breakdown_mutex_;
  bool breakdown_enabled_{false};
  std::vector<service::breakdown::Sample> completed_query_samples_;
  std::vector<service::breakdown::Sample> completed_insert_samples_;
};
