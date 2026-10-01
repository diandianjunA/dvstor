#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

// Refresh per-shard Stage2 watermarks for the sender's early flow control.
// The read goes through the query engine's registered control-block path on
// this dedicated thread, so the sender's decision path only touches shared
// atomics and never waits on RDMA or a query lane.
void ComputeService::run_storage_flow_control_poll_loop() {
  const auto poll_period = std::chrono::milliseconds(
    std::max<u32>(1, config_.storage_owner_flow_control_poll_ms));
  while (!storage_insert_shutdown_.load(std::memory_order_acquire)) {
    try {
      if (search_engine_ != nullptr &&
          storage_flow_control_next_sequences_ != nullptr) {
        const std::vector<gpu_search::MaintenanceWatermark> watermarks =
          search_engine_->read_maintenance_watermarks();
        const size_t limit = std::min(
          watermarks.size(), storage_insert_owners_.size());
        for (size_t shard = 0; shard < limit; ++shard) {
          // Publication order next-then-durable pairs with the sender's
          // durable-then-next read order: an interleaved refresh can only
          // overestimate the backlog, never hide submitted work.
          storage_flow_control_next_sequences_[shard].store(
            watermarks[shard].next_sequence, std::memory_order_release);
          storage_flow_control_durable_sequences_[shard].store(
            watermarks[shard].durable_sequence, std::memory_order_release);
        }
        storage_flow_control_refreshes_.fetch_add(
          1, std::memory_order_relaxed);
      }
    } catch (const std::exception& error) {
      // A failed poll retains the previous watermarks. A stale durable
      // watermark can only overestimate the backlog, so the failure mode is
      // bounded extra throttling rather than hidden Stage2 debt.
      const u64 failures =
        storage_flow_control_refresh_failures_.fetch_add(
          1, std::memory_order_relaxed) + 1;
      if (failures <= 8 || (failures & (failures - 1)) == 0) {
        std::cerr << "[storage-owner] flow-control watermark refresh failed: "
                  << error.what() << " count=" << failures << std::endl;
      }
    }
    // Poll in short slices so shutdown joins promptly even with a large
    // configured period.
    const auto deadline = std::chrono::steady_clock::now() + poll_period;
    while (!storage_insert_shutdown_.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}
