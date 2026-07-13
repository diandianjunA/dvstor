#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/types.hh"
#include "service/storage_owner_protocol.hh"

namespace gpu_search {

struct DeltaMutation {
  node_t id{};
  service::storage_owner::MutationKind kind{service::storage_owner::MutationKind::insert};
  u32 generation{};
  u64 epoch{};
  u64 remote_node{};
  u64 old_remote_node{};
  u64 anchor_hint{};
  u64 maintenance_sequence{};
  u32 owner_storage{};
  bool durable{};
  std::vector<byte_t> vector;
  std::vector<node_t> neighbors;
  std::chrono::steady_clock::time_point enqueued_at{};
  std::chrono::steady_clock::time_point published_at{};
};

struct VersionEntry {
  u32 generation{};
  u64 epoch{};
  bool deleted{};
  bool in_delta{};
};

struct DeltaSnapshot {
  u64 epoch{};
  u64 base_generation{};
  std::vector<DeltaMutation> mutations;
};

class DeltaCoordinator {
public:
  explicit DeltaCoordinator(u64 base_generation = 1);

  u64 reserve_epoch();
  void enqueue(DeltaMutation mutation);
  std::vector<DeltaMutation> take_pending(size_t max_items,
                                         std::chrono::microseconds max_wait);
  bool publish(std::vector<DeltaMutation> mutations, u64 epoch,
               std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

  u64 published_epoch() const;
  u64 base_generation() const;
  size_t delta_size() const;
  size_t pending_size() const;
  std::optional<VersionEntry> version(node_t id) const;
  DeltaSnapshot snapshot(u64 epoch = 0) const;
  std::vector<DeltaMutation> retire_durable(
    std::span<const u64> durable_sequences,
    size_t max_items = std::numeric_limits<size_t>::max());

  bool should_consolidate(u64 base_nodes, size_t delta_budget_bytes,
                          f64 max_ratio, f64 budget_high_watermark,
                          std::chrono::milliseconds max_age) const;
  DeltaSnapshot begin_consolidation();
  void complete_consolidation(u64 new_base_generation, u64 through_epoch);
  void complete_partial_consolidation(const std::vector<node_t>& merged_ids,
                                      u64 new_base_generation, u64 through_epoch);
  void mark_compacted();

private:
  mutable std::shared_mutex state_mutex_;
  std::unordered_map<node_t, DeltaMutation> delta_;
  std::unordered_map<node_t, VersionEntry> versions_;
  u64 base_generation_{1};
  std::atomic<u64> next_epoch_{1};
  std::atomic<u64> published_epoch_{0};
  size_t delta_bytes_{0};
  std::chrono::steady_clock::time_point last_consolidation_;

  mutable std::mutex pending_mutex_;
  std::condition_variable pending_cv_;
  std::deque<DeltaMutation> pending_;
};

}  // namespace gpu_search
