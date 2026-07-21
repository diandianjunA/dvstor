#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "memory_node/storage_owner_maintenance/stage2_tracker.hh"
#include "service/storage_owner_protocol.hh"

namespace memory_node_storage_owner_maintenance_detail {

// A logical context request. The op span is published through the outbox
// mutex and copied into fixed aggregate storage before the entry leaves the
// peer FIFO. Its owning context does not modify the span before completion.
struct Stage2ReverseDispatch {
  std::uint64_t logical_request_id{};
  Stage2ContextHandle context{};
  std::uint32_t worker_id{};
  std::uint32_t peer_index{};
  service::storage_owner::PeerRpcType request_type{
    service::storage_owner::PeerRpcType::reverse_update_request};
  std::uint32_t item_count{};
  const service::storage_owner::ReverseUpdateOp* ops{};
  std::uint64_t ready_at_ns{};

  [[nodiscard]] bool same_request(
      const Stage2ReverseDispatch& other) const noexcept {
    return logical_request_id == other.logical_request_id &&
           context == other.context && worker_id == other.worker_id &&
           peer_index == other.peer_index &&
           request_type == other.request_type &&
           item_count == other.item_count && ops == other.ops;
  }
};

struct Stage2ReverseCompletion {
  std::uint64_t logical_request_id{};
  Stage2ContextHandle context{};
  std::uint32_t worker_id{};
  std::uint32_t peer_index{};
};

// One bounded peer request. A locally retained member list maps its single wire
// request_id back to every contributing stage2 context when the aggregate ACK
// arrives. Retries resend this exact ID and exact copied payload.
struct Stage2ReverseAggregate {
  std::uint64_t wire_request_id{};
  std::uint32_t owner_worker_id{};
  std::uint32_t peer_index{};
  service::storage_owner::PeerRpcType request_type{
    service::storage_owner::PeerRpcType::reverse_update_request};
  std::uint32_t item_count{};
  std::uint32_t logical_count{};
  std::uint64_t ready_at_ns{};
  std::uint64_t deadline_ns{};
};

enum class Stage2ReverseEnqueueResult : std::uint8_t {
  enqueued,
  duplicate,
  conflict,
  full,
  invalid,
};

// Fixed-capacity MPMC logical outbox plus fixed-capacity aggregate storage.
// All vectors reserve their maximum in the constructor and never grow beyond
// it. Queueing is work-conserving: form_aggregate consumes the currently
// available prefix for one peer and RPC type up to the unchanged wire bound;
// it never waits for a batching timer.
class Stage2ReverseOutbox {
 public:
  Stage2ReverseOutbox(std::size_t logical_capacity,
                      std::size_t aggregate_capacity,
                      std::uint32_t peer_count,
                      std::uint32_t wire_max_ops)
      : entries_(logical_capacity),
        aggregates_(aggregate_capacity),
        peers_(peer_count),
        wire_max_ops_(wire_max_ops) {
    if (logical_capacity == 0 || aggregate_capacity == 0 ||
        logical_capacity > std::numeric_limits<std::uint32_t>::max() ||
        aggregate_capacity > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("stage2 reverse outbox capacity is out of range");
    }
    if (peer_count == 0 || peer_count > 64 || wire_max_ops == 0) {
      throw std::invalid_argument("stage2 reverse outbox geometry is invalid");
    }

    free_entries_.reserve(logical_capacity);
    for (std::size_t index = logical_capacity; index != 0; --index) {
      free_entries_.push_back(static_cast<std::uint32_t>(index - 1));
    }
    free_aggregates_.reserve(aggregate_capacity);
    for (std::size_t index = aggregate_capacity; index != 0; --index) {
      free_aggregates_.push_back(static_cast<std::uint32_t>(index - 1));
    }
    for (Aggregate& aggregate : aggregates_) {
      aggregate.ops.reserve(wire_max_ops_);
    }
  }

  Stage2ReverseOutbox(const Stage2ReverseOutbox&) = delete;
  Stage2ReverseOutbox& operator=(const Stage2ReverseOutbox&) = delete;

  [[nodiscard]] Stage2ReverseEnqueueResult try_enqueue(
      Stage2ReverseDispatch dispatch) {
    if (!valid(dispatch)) return Stage2ReverseEnqueueResult::invalid;

    std::lock_guard<std::mutex> lock(mutex_);
    for (const Entry& entry : entries_) {
      if (!entry.in_use ||
          entry.dispatch.logical_request_id != dispatch.logical_request_id) {
        continue;
      }
      return entry.dispatch.same_request(dispatch)
               ? Stage2ReverseEnqueueResult::duplicate
               : Stage2ReverseEnqueueResult::conflict;
    }
    if (free_entries_.empty()) return Stage2ReverseEnqueueResult::full;

    const std::uint32_t index = free_entries_.back();
    free_entries_.pop_back();
    Entry& entry = entries_[index];
    entry.in_use = true;
    entry.queued = true;
    entry.dispatch = dispatch;
    entry.previous = peers_[dispatch.peer_index].tail;
    entry.next = npos;
    entry.aggregate_next = npos;

    PeerQueue& peer = peers_[dispatch.peer_index];
    if (peer.tail == npos) {
      peer.head = index;
    } else {
      entries_[peer.tail].next = index;
    }
    peer.tail = index;
    ++peer.size;
    ++logical_size_;
    return Stage2ReverseEnqueueResult::enqueued;
  }

  [[nodiscard]] std::optional<Stage2ReverseAggregate> form_aggregate(
      std::uint32_t peer_index,
      std::uint32_t owner_worker_id,
      std::uint64_t wire_request_id,
      std::uint64_t now_ns) {
    if (wire_request_id == 0) return std::nullopt;
    std::lock_guard<std::mutex> lock(mutex_);
    if (peer_index >= peers_.size() || peers_[peer_index].head == npos ||
        free_aggregates_.empty()) {
      return std::nullopt;
    }

    const Entry& first = entries_[peers_[peer_index].head];
    if (first.dispatch.ready_at_ns > now_ns) return std::nullopt;

    const std::uint32_t aggregate_index = free_aggregates_.back();
    free_aggregates_.pop_back();
    Aggregate& aggregate = aggregates_[aggregate_index];
    aggregate.in_use = true;
    aggregate.leased = false;
    aggregate.state = AggregateState::ready_to_post;
    aggregate.snapshot = Stage2ReverseAggregate{
      .wire_request_id = wire_request_id,
      .owner_worker_id = owner_worker_id,
      .peer_index = peer_index,
      .request_type = first.dispatch.request_type,
      .ready_at_ns = now_ns,
    };
    aggregate.member_head = npos;
    aggregate.member_tail = npos;
    aggregate.ops.clear();

    PeerQueue& peer = peers_[peer_index];
    while (peer.head != npos) {
      const std::uint32_t entry_index = peer.head;
      Entry& entry = entries_[entry_index];
      const Stage2ReverseDispatch& dispatch = entry.dispatch;
      if (dispatch.ready_at_ns > now_ns ||
          dispatch.request_type != aggregate.snapshot.request_type ||
          aggregate.ops.size() + dispatch.item_count > wire_max_ops_) {
        break;
      }

      unlink_queued(entry_index);
      entry.aggregate_next = npos;
      if (aggregate.member_tail == npos) {
        aggregate.member_head = entry_index;
      } else {
        entries_[aggregate.member_tail].aggregate_next = entry_index;
      }
      aggregate.member_tail = entry_index;
      aggregate.ops.insert(aggregate.ops.end(), dispatch.ops,
                           dispatch.ops + dispatch.item_count);
      ++aggregate.snapshot.logical_count;
    }

    if (aggregate.snapshot.logical_count == 0) {
      release_aggregate(aggregate_index);
      return std::nullopt;
    }
    aggregate.snapshot.item_count =
      static_cast<std::uint32_t>(aggregate.ops.size());
    ++aggregate_size_;
    return aggregate.snapshot;
  }

  [[nodiscard]] bool can_form_aggregate(std::uint32_t peer_index,
                                        std::uint64_t now_ns) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peer_index < peers_.size() && peers_[peer_index].head != npos &&
           !free_aggregates_.empty() &&
           entries_[peers_[peer_index].head].dispatch.ready_at_ns <= now_ns;
  }

  [[nodiscard]] std::optional<Stage2ReverseAggregate> claim_ready_to_post(
      std::uint32_t owner_worker_id,
      std::uint64_t now_ns,
      std::size_t& cursor) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (cursor < aggregates_.size()) {
      Aggregate& aggregate = aggregates_[cursor++];
      if (aggregate.in_use && !aggregate.leased &&
          aggregate.state == AggregateState::ready_to_post &&
          aggregate.snapshot.ready_at_ns <= now_ns) {
        aggregate.leased = true;
        aggregate.snapshot.owner_worker_id = owner_worker_id;
        return aggregate.snapshot;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<Stage2ReverseAggregate> claim_awaiting_response(
      std::uint32_t owner_worker_id,
      std::size_t& cursor) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (cursor < aggregates_.size()) {
      Aggregate& aggregate = aggregates_[cursor++];
      if (aggregate.in_use && !aggregate.leased &&
          aggregate.state == AggregateState::awaiting_response &&
          aggregate.snapshot.owner_worker_id == owner_worker_id) {
        aggregate.leased = true;
        return aggregate.snapshot;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] bool copy_ops(
      std::uint32_t owner_worker_id,
      std::uint64_t wire_request_id,
      std::span<service::storage_owner::ReverseUpdateOp> destination) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const Aggregate* aggregate = find_aggregate(wire_request_id);
    if (aggregate == nullptr || !aggregate->leased ||
        aggregate->snapshot.owner_worker_id != owner_worker_id ||
        destination.size() < aggregate->ops.size()) {
      return false;
    }
    std::copy(aggregate->ops.begin(), aggregate->ops.end(),
              destination.begin());
    return true;
  }

  [[nodiscard]] bool finish_post(std::uint32_t owner_worker_id,
                                 std::uint64_t wire_request_id,
                                 bool sent,
                                 std::uint64_t ready_or_deadline_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    Aggregate* aggregate = find_aggregate(wire_request_id);
    if (!owned_lease(aggregate, owner_worker_id) ||
        aggregate->state != AggregateState::ready_to_post) {
      return false;
    }
    aggregate->leased = false;
    if (sent) {
      aggregate->state = AggregateState::awaiting_response;
      aggregate->snapshot.deadline_ns = ready_or_deadline_ns;
    } else {
      aggregate->snapshot.ready_at_ns = ready_or_deadline_ns;
    }
    return true;
  }

  [[nodiscard]] bool release_poll(std::uint32_t owner_worker_id,
                                  std::uint64_t wire_request_id,
                                  bool retry,
                                  std::uint64_t ready_at_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    Aggregate* aggregate = find_aggregate(wire_request_id);
    if (!owned_lease(aggregate, owner_worker_id) ||
        aggregate->state != AggregateState::awaiting_response) {
      return false;
    }
    aggregate->leased = false;
    if (retry) {
      aggregate->state = AggregateState::ready_to_post;
      aggregate->snapshot.ready_at_ns = ready_at_ns;
    }
    return true;
  }

  [[nodiscard]] std::optional<std::size_t> copy_completions(
      std::uint32_t owner_worker_id,
      std::uint64_t wire_request_id,
      std::span<Stage2ReverseCompletion> destination) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const Aggregate* aggregate = find_aggregate(wire_request_id);
    if (!owned_lease(aggregate, owner_worker_id) ||
        aggregate->state != AggregateState::awaiting_response ||
        destination.size() < aggregate->snapshot.logical_count) {
      return std::nullopt;
    }
    std::size_t count = 0;
    for (std::uint32_t entry_index = aggregate->member_head;
         entry_index != npos;
         entry_index = entries_[entry_index].aggregate_next) {
      const Stage2ReverseDispatch& dispatch = entries_[entry_index].dispatch;
      destination[count++] = Stage2ReverseCompletion{
        .logical_request_id = dispatch.logical_request_id,
        .context = dispatch.context,
        .worker_id = dispatch.worker_id,
        .peer_index = dispatch.peer_index,
      };
    }
    return count;
  }

  [[nodiscard]] bool finish_success(std::uint32_t owner_worker_id,
                                    std::uint64_t wire_request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::optional<std::uint32_t> index =
      find_aggregate_index(wire_request_id);
    if (!index.has_value()) return false;
    Aggregate& aggregate = aggregates_[*index];
    if (!owned_lease(&aggregate, owner_worker_id) ||
        aggregate.state != AggregateState::awaiting_response) {
      return false;
    }
    release_aggregate_members(aggregate);
    release_aggregate(*index);
    --aggregate_size_;
    return true;
  }

  // Shutdown helpers. Removing queued entries takes the same mutex used while
  // aggregate formation copies their context-owned spans, so context teardown
  // cannot race that copy. Aggregates already own fixed payload copies.
  std::size_t erase_queued_worker(std::uint32_t worker_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t erased = 0;
    for (std::uint32_t index = 0; index < entries_.size(); ++index) {
      Entry& entry = entries_[index];
      if (entry.in_use && entry.queued &&
          entry.dispatch.worker_id == worker_id) {
        unlink_queued(index);
        release_entry(index);
        ++erased;
      }
    }
    return erased;
  }

  [[nodiscard]] std::optional<std::uint64_t> discard_owned_aggregate(
      std::uint32_t owner_worker_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::uint32_t index = 0; index < aggregates_.size(); ++index) {
      Aggregate& aggregate = aggregates_[index];
      if (!aggregate.in_use ||
          aggregate.snapshot.owner_worker_id != owner_worker_id) {
        continue;
      }
      const std::uint64_t request_id = aggregate.snapshot.wire_request_id;
      release_aggregate_members(aggregate);
      release_aggregate(index);
      --aggregate_size_;
      return request_id;
    }
    return std::nullopt;
  }

  [[nodiscard]] std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return logical_size_;
  }

  [[nodiscard]] std::size_t queued_size(std::uint32_t peer_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peer_index < peers_.size() ? peers_[peer_index].size : 0;
  }

  [[nodiscard]] std::size_t aggregate_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return aggregate_size_;
  }

  [[nodiscard]] std::size_t capacity() const noexcept {
    return entries_.size();
  }

  [[nodiscard]] std::size_t aggregate_capacity() const noexcept {
    return aggregates_.size();
  }

  [[nodiscard]] std::uint32_t wire_max_ops() const noexcept {
    return wire_max_ops_;
  }

 private:
  static constexpr std::uint32_t npos =
    std::numeric_limits<std::uint32_t>::max();

  enum class AggregateState : std::uint8_t {
    ready_to_post,
    awaiting_response,
  };

  struct Entry {
    bool in_use{};
    bool queued{};
    Stage2ReverseDispatch dispatch{};
    std::uint32_t previous{npos};
    std::uint32_t next{npos};
    std::uint32_t aggregate_next{npos};
  };

  struct PeerQueue {
    std::uint32_t head{npos};
    std::uint32_t tail{npos};
    std::size_t size{};
  };

  struct Aggregate {
    bool in_use{};
    bool leased{};
    AggregateState state{AggregateState::ready_to_post};
    Stage2ReverseAggregate snapshot{};
    std::uint32_t member_head{npos};
    std::uint32_t member_tail{npos};
    std::vector<service::storage_owner::ReverseUpdateOp> ops;
  };

  [[nodiscard]] bool valid(const Stage2ReverseDispatch& dispatch) const {
    const bool reverse = dispatch.request_type ==
      service::storage_owner::PeerRpcType::reverse_update_request;
    const bool cleanup = dispatch.request_type ==
      service::storage_owner::PeerRpcType::cleanup_deleted_request;
    return dispatch.logical_request_id != 0 &&
           dispatch.context.generation != 0 &&
           dispatch.peer_index < peers_.size() &&
           dispatch.item_count != 0 && dispatch.item_count <= wire_max_ops_ &&
           dispatch.ops != nullptr && (reverse || cleanup);
  }

  void unlink_queued(std::uint32_t index) {
    Entry& entry = entries_[index];
    PeerQueue& peer = peers_[entry.dispatch.peer_index];
    if (entry.previous == npos) {
      peer.head = entry.next;
    } else {
      entries_[entry.previous].next = entry.next;
    }
    if (entry.next == npos) {
      peer.tail = entry.previous;
    } else {
      entries_[entry.next].previous = entry.previous;
    }
    --peer.size;
    entry.queued = false;
    entry.previous = npos;
    entry.next = npos;
  }

  void release_entry(std::uint32_t index) {
    entries_[index] = {};
    entries_[index].previous = npos;
    entries_[index].next = npos;
    entries_[index].aggregate_next = npos;
    free_entries_.push_back(index);
    --logical_size_;
  }

  void release_aggregate_members(Aggregate& aggregate) {
    std::uint32_t entry_index = aggregate.member_head;
    while (entry_index != npos) {
      const std::uint32_t next = entries_[entry_index].aggregate_next;
      release_entry(entry_index);
      entry_index = next;
    }
    aggregate.member_head = npos;
    aggregate.member_tail = npos;
  }

  void release_aggregate(std::uint32_t index) {
    Aggregate& aggregate = aggregates_[index];
    aggregate.in_use = false;
    aggregate.leased = false;
    aggregate.state = AggregateState::ready_to_post;
    aggregate.snapshot = {};
    aggregate.member_head = npos;
    aggregate.member_tail = npos;
    aggregate.ops.clear();
    free_aggregates_.push_back(index);
  }

  [[nodiscard]] std::optional<std::uint32_t> find_aggregate_index(
      std::uint64_t wire_request_id) const {
    for (std::uint32_t index = 0; index < aggregates_.size(); ++index) {
      if (aggregates_[index].in_use &&
          aggregates_[index].snapshot.wire_request_id == wire_request_id) {
        return index;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] Aggregate* find_aggregate(std::uint64_t wire_request_id) {
    const auto index = find_aggregate_index(wire_request_id);
    return index.has_value() ? &aggregates_[*index] : nullptr;
  }

  [[nodiscard]] const Aggregate* find_aggregate(
      std::uint64_t wire_request_id) const {
    const auto index = find_aggregate_index(wire_request_id);
    return index.has_value() ? &aggregates_[*index] : nullptr;
  }

  [[nodiscard]] static bool owned_lease(
      const Aggregate* aggregate,
      std::uint32_t owner_worker_id) {
    return aggregate != nullptr && aggregate->leased &&
           aggregate->snapshot.owner_worker_id == owner_worker_id;
  }

  mutable std::mutex mutex_;
  std::vector<Entry> entries_;
  std::vector<std::uint32_t> free_entries_;
  std::vector<Aggregate> aggregates_;
  std::vector<std::uint32_t> free_aggregates_;
  std::vector<PeerQueue> peers_;
  std::uint32_t wire_max_ops_{};
  std::size_t logical_size_{};
  std::size_t aggregate_size_{};
};

}  // namespace memory_node_storage_owner_maintenance_detail
