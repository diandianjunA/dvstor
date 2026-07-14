#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
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
  std::chrono::steady_clock::time_point enqueued_at{};
};

struct VersionEntry {
  u32 generation{};
  u64 epoch{};
  bool deleted{};
  bool in_delta{};
};

class DeltaCoordinator {
public:
  u64 reserve_epoch();
  bool publish(std::vector<DeltaMutation> mutations, u64 epoch);

  u64 published_epoch() const;
  size_t delta_size() const;
  std::optional<VersionEntry> version(node_t id) const;
  std::vector<DeltaMutation> retire_durable(
    std::span<const u64> durable_sequences,
    size_t max_items = std::numeric_limits<size_t>::max());

private:
  struct DurableCandidate {
    u64 maintenance_sequence{};
    u64 epoch{};
    node_t id{};
    u32 generation{};
  };

  struct DurableCandidateGreater {
    bool operator()(const DurableCandidate& lhs,
                    const DurableCandidate& rhs) const {
      if (lhs.maintenance_sequence != rhs.maintenance_sequence) {
        return lhs.maintenance_sequence > rhs.maintenance_sequence;
      }
      if (lhs.epoch != rhs.epoch) return lhs.epoch > rhs.epoch;
      return lhs.id > rhs.id;
    }
  };

  using DurableQueue = std::priority_queue<
    DurableCandidate, std::vector<DurableCandidate>, DurableCandidateGreater>;

  mutable std::shared_mutex state_mutex_;
  std::unordered_map<node_t, DeltaMutation> delta_;
  std::unordered_map<node_t, VersionEntry> versions_;
  std::vector<DurableQueue> durable_candidates_;
  size_t durable_owner_cursor_{};
  std::atomic<u64> next_epoch_{1};
  std::atomic<u64> published_epoch_{0};
};

}  // namespace gpu_search
