#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "memory_node/storage_owner_maintenance/ready_context_queue.hh"
#include "service/storage_owner_protocol.hh"

namespace memory_node_storage_owner_maintenance_detail {

// One already-runnable logical Stage2-home request. The outbox copies the
// complete protocol image during enqueue, so a context may release/reuse its
// local build buffer immediately. The logical request ID is never placed on
// the combined wire request; it remains the correlation key seen by the
// context after response demultiplexing.
struct Stage2HomeRpcDispatch {
  std::uint64_t logical_request_id{};
  std::uint32_t peer_index{};
  service::storage_owner::PeerRpcType request_type{
    service::storage_owner::PeerRpcType::stage2_expand_score_request};
  std::uint32_t item_count{};
  bool speculative{};
  Stage2ContextOwnerKey completion_owner{};
  std::span<const byte_t> request;
};

struct Stage2HomeRpcAggregate {
  std::uint64_t wire_request_id{};
  std::uint32_t peer_index{};
  service::storage_owner::PeerRpcType request_type{
    service::storage_owner::PeerRpcType::stage2_expand_score_request};
  service::storage_owner::PeerRpcType response_type{
    service::storage_owner::PeerRpcType::stage2_expand_score_response};
  std::uint32_t item_count{};
  std::uint32_t logical_count{};
  std::size_t request_bytes{};
  bool speculative{};
  // A singleton keeps the original logical wire ID and protocol image.  It
  // still occupies an aggregate slot solely for exact retry/deadline
  // ownership, but its response must use the original borrowed receive-slot
  // registry path rather than owned-payload fan-out.
  bool direct{};
};

// A complete protocol image suitable for publishing into a logical response
// registry. Its header carries the original logical request ID and original
// item count, exactly as if that request had travelled alone on the wire.
struct Stage2HomeRpcLogicalResponse {
  std::uint64_t logical_request_id{};
  std::uint32_t peer_index{};
  service::storage_owner::PeerRpcType response_type{
    service::storage_owner::PeerRpcType::stage2_expand_score_response};
  std::uint32_t item_count{};
  Stage2ContextOwnerKey completion_owner{};
  std::vector<byte_t> response;
};

struct Stage2HomeRpcPostLease {
  std::size_t slot{std::numeric_limits<std::size_t>::max()};
  std::uint64_t generation{};
  std::uint64_t wire_request_id{};

  [[nodiscard]] bool valid() const noexcept {
    return slot != std::numeric_limits<std::size_t>::max() &&
      generation != 0 && wire_request_id != 0;
  }
};

struct Stage2HomeRpcAggregateLease {
  std::size_t slot{std::numeric_limits<std::size_t>::max()};
  std::uint64_t generation{};
  std::uint64_t wire_request_id{};

  [[nodiscard]] bool valid() const noexcept {
    return slot != std::numeric_limits<std::size_t>::max() &&
      generation != 0 && wire_request_id != 0;
  }
};

struct Stage2HomeRpcDemuxResult {
  Stage2HomeRpcAggregateLease lease{};
  std::vector<Stage2HomeRpcLogicalResponse> logical_responses;
};

enum class Stage2HomeRpcEnqueueResult : std::uint8_t {
  enqueued,
  duplicate,
  conflict,
  full,
  invalid,
};

enum class Stage2HomeRpcDirectResponseResult : std::uint8_t {
  not_direct,
  invalid,
  finished,
};

// Bounded, process-wide transport combiner for authoritative Stage2-home RPCs.
// Enqueue never starts a timer. form_aggregate() consumes only descriptors
// already present in one per-(peer,type,traffic-class) FIFO. Even a one-entry
// prefix is formed immediately; aggregation never creates a hidden wait.
//
// The remote protocol is unchanged. expand_score concatenates its item/query
// arrays. score_many additionally rebases every query_index into the combined
// query table. A successful outer response is sliced back into complete
// logical response images; compact expand_score neighbor offsets are rebased
// to each logical response. Invalid responses retain the exact aggregate and
// byte-identical request for retry under the same wire request ID.
class Stage2HomeRpcOutbox {
 public:
  Stage2HomeRpcOutbox(std::size_t logical_capacity,
                      std::size_t aggregate_capacity,
                      std::uint32_t peer_count,
                      std::uint32_t expand_wire_max_items,
                      std::uint32_t score_wire_max_items,
                      std::uint32_t score_wire_max_queries,
                      std::size_t byte_capacity)
      : entries_(logical_capacity),
        aggregates_(aggregate_capacity),
        queues_(peer_count),
        aggregate_ready_queues_(peer_count),
        logical_buckets_(hash_capacity(logical_capacity)),
        aggregate_buckets_(hash_capacity(aggregate_capacity)),
        logical_bucket_mask_(logical_buckets_.size() - 1),
        aggregate_bucket_mask_(aggregate_buckets_.size() - 1),
        expand_wire_max_items_(expand_wire_max_items),
        score_wire_max_items_(score_wire_max_items),
        score_wire_max_queries_(score_wire_max_queries),
        byte_capacity_(byte_capacity) {
    if (logical_capacity == 0 || aggregate_capacity == 0 ||
        logical_capacity > std::numeric_limits<std::uint32_t>::max() ||
        aggregate_capacity > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument(
        "stage2 home RPC outbox capacity is out of range");
    }
    if (peer_count == 0 || peer_count > 64 ||
        expand_wire_max_items == 0 || score_wire_max_items == 0 ||
        score_wire_max_queries == 0 || byte_capacity == 0) {
      throw std::invalid_argument(
        "stage2 home RPC outbox geometry is invalid");
    }
    free_entries_.reserve(logical_capacity);
    for (std::size_t index = logical_capacity; index != 0; --index) {
      free_entries_.push_back(static_cast<std::uint32_t>(index - 1));
    }
    free_aggregates_.reserve(aggregate_capacity);
    for (std::size_t index = aggregate_capacity; index != 0; --index) {
      free_aggregates_.push_back(static_cast<std::uint32_t>(index - 1));
    }
    max_request_bytes_ = std::max(
      service::storage_owner::stage2_expand_score_request_bytes(
        expand_wire_max_items_),
      service::storage_owner::stage2_score_many_request_bytes(
        score_wire_max_items_, score_wire_max_queries_));
    if (max_request_bytes_ == std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(
        "stage2 home RPC outbox wire request size overflows");
    }
    if (byte_capacity_ <= max_request_bytes_) {
      throw std::invalid_argument(
        "stage2 home RPC outbox byte budget has no build headroom");
    }
    resident_limit_ = byte_capacity_ - max_request_bytes_;
  }

  Stage2HomeRpcOutbox(const Stage2HomeRpcOutbox&) = delete;
  Stage2HomeRpcOutbox& operator=(const Stage2HomeRpcOutbox&) = delete;

  [[nodiscard]] Stage2HomeRpcEnqueueResult try_enqueue(
      const Stage2HomeRpcDispatch& dispatch) {
    const auto metadata = validate_dispatch(dispatch);
    if (!metadata.has_value()) return Stage2HomeRpcEnqueueResult::invalid;
    std::vector<byte_t> request_copy(
      dispatch.request.begin(), dispatch.request.end());
    const RequestDigest request_digest = digest_request(request_copy);
    const std::size_t request_capacity = request_copy.capacity();

    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing_index = find_entry_index(
      dispatch.logical_request_id);
    if (existing_index.has_value()) {
      const Entry& entry = entries_[*existing_index];
      return same_request(entry, dispatch, *metadata, request_digest)
        ? Stage2HomeRpcEnqueueResult::duplicate
        : Stage2HomeRpcEnqueueResult::conflict;
    }
    if (find_aggregate_index(dispatch.logical_request_id).has_value()) {
      return Stage2HomeRpcEnqueueResult::conflict;
    }
    if (free_entries_.empty() || resident_bytes_ > resident_limit_ ||
        request_capacity > resident_limit_ - resident_bytes_) {
      return Stage2HomeRpcEnqueueResult::full;
    }

    const std::uint32_t index = free_entries_.back();
    free_entries_.pop_back();
    Entry& entry = entries_[index];
    entry.in_use = true;
    entry.queued = true;
    entry.cancelled = false;
    entry.logical_request_id = dispatch.logical_request_id;
    entry.peer_index = dispatch.peer_index;
    entry.request_type = dispatch.request_type;
    entry.response_type = response_type(dispatch.request_type);
    entry.item_count = dispatch.item_count;
    entry.query_count = metadata->query_count;
    entry.score_flags = metadata->score_flags;
    entry.queue_class = metadata->queue_class;
    entry.completion_owner = dispatch.completion_owner;
    entry.request_digest = request_digest;
    entry.request = std::move(request_copy);
    resident_bytes_ += entry.request.capacity();
    PeerQueue& queue = queues_[entry.peer_index][entry.queue_class];
    entry.previous = queue.tail;
    entry.next = npos;
    if (queue.tail == npos) {
      queue.head = index;
    } else {
      entries_[queue.tail].next = index;
    }
    queue.tail = index;
    ++queue.size;
    ready_peer_masks_[entry.queue_class].fetch_or(
      std::uint64_t{1} << entry.peer_index, std::memory_order_release);
    insert_logical_bucket(entry.logical_request_id, index);
    ++logical_size_;
    return Stage2HomeRpcEnqueueResult::enqueued;
  }

  [[nodiscard]] std::optional<Stage2HomeRpcAggregate> form_aggregate(
      std::uint32_t peer_index,
      service::storage_owner::PeerRpcType request_type,
      std::uint64_t wire_request_id,
      std::uint32_t source_shard,
      bool speculative = false) {
    if (wire_request_id == 0) return std::nullopt;
    const auto requested_queue_class = queue_class(
      request_type,
      speculative
        ? service::storage_owner::kStage2ScoreManyFlagSpeculative
        : 0);
    if (!requested_queue_class.has_value()) return std::nullopt;

    std::lock_guard<std::mutex> lock(mutex_);
    if (peer_index >= queues_.size() || source_shard >= queues_.size() ||
        source_shard == peer_index || free_aggregates_.empty() ||
        find_entry_index(wire_request_id).has_value() ||
        find_aggregate_index(wire_request_id).has_value()) {
      return std::nullopt;
    }
    PeerQueue& queue = queues_[peer_index][*requested_queue_class];
    if (queue.head == npos) return std::nullopt;

    const bool score_many = request_type ==
      service::storage_owner::PeerRpcType::stage2_score_many_request;
    const std::uint32_t max_items = score_many
      ? score_wire_max_items_ : expand_wire_max_items_;
    std::uint32_t total_items = 0;
    std::uint32_t total_queries = 0;
    std::size_t selected_count = 0;
    std::uint32_t entry_index = queue.head;
    while (entry_index != npos) {
      const Entry& entry = entries_[entry_index];
      if (!entry.in_use || !entry.queued ||
          entry.request_type != request_type ||
          entry.peer_index != peer_index ||
          entry.queue_class != *requested_queue_class) {
        return std::nullopt;
      }
      if (entry.item_count > max_items - total_items) break;
      if (score_many &&
          entry.query_count > score_wire_max_queries_ - total_queries) {
        break;
      }
      total_items += entry.item_count;
      total_queries += entry.query_count;
      ++selected_count;
      entry_index = entry.next;
    }
    if (selected_count == 0) return std::nullopt;

    const std::uint32_t aggregate_index = free_aggregates_.back();
    free_aggregates_.pop_back();
    Aggregate& aggregate = aggregates_[aggregate_index];
    aggregate.in_use = true;
    aggregate.state = AggregateState::ready;
    aggregate.queue_class = *requested_queue_class;
    advance_generation(aggregate.generation);
    aggregate.snapshot = Stage2HomeRpcAggregate{
      .wire_request_id = wire_request_id,
      .peer_index = peer_index,
      .request_type = request_type,
      .response_type = response_type(request_type),
      .item_count = total_items,
      .logical_count = static_cast<std::uint32_t>(selected_count),
      .request_bytes = 0,
      .speculative = speculative,
      .direct = false,
    };
    aggregate.members.clear();
    aggregate.request.clear();
    std::uint32_t item_offset = 0;
    std::uint32_t query_offset = 0;
    entry_index = queue.head;
    for (std::size_t selected = 0; selected < selected_count; ++selected) {
      const Entry& entry = entries_[entry_index];
      aggregate.members.push_back(Member{
        .entry_index = entry_index,
        .item_offset = item_offset,
        .query_offset = query_offset,
      });
      item_offset += entry.item_count;
      query_offset += entry.query_count;
      entry_index = entry.next;
    }
    std::size_t member_request_capacity = 0;
    for (const Member& member : aggregate.members) {
      const std::size_t capacity =
        entries_[member.entry_index].request.capacity();
      if (capacity > resident_bytes_ - member_request_capacity) {
        release_aggregate(aggregate_index);
        return std::nullopt;
      }
      member_request_capacity += capacity;
    }
    std::vector<byte_t> wire_request;
    if (!build_wire_request(
          aggregate.members, wire_request_id, source_shard, request_type,
          total_items, total_queries, wire_request) ||
        wire_request.capacity() > byte_capacity_ - resident_bytes_ ||
        wire_request.capacity() >
          resident_limit_ - (resident_bytes_ - member_request_capacity)) {
      release_aggregate(aggregate_index);
      return std::nullopt;
    }
    resident_bytes_ += wire_request.capacity();
    aggregate.request = std::move(wire_request);
    aggregate.snapshot.request_bytes = aggregate.request.size();
    for (const Member& member : aggregate.members) {
      Entry& entry = entries_[member.entry_index];
      unlink_queued(member.entry_index);
      entry.aggregate_slot = aggregate_index;
      entry.aggregate_generation = aggregate.generation;
      release_entry_request(entry);
    }
    insert_aggregate_bucket(wire_request_id, aggregate_index);
    ++aggregate_size_;
    return aggregate.snapshot;
  }

  // Claim a queue only when it contains exactly one logical request.  The
  // retained protocol image is moved (not rebuilt/copied), and the original
  // logical request ID remains the wire ID.  Keeping a normal aggregate
  // lifecycle around the entry provides the same bounded timeout/retry and
  // cancellation semantics as a combined request without imposing aggregate
  // request/response construction on the overwhelmingly common singleton.
  [[nodiscard]] std::optional<Stage2HomeRpcAggregate>
  form_singleton_direct(
      std::uint32_t peer_index,
      service::storage_owner::PeerRpcType request_type,
      std::uint32_t source_shard,
      bool speculative = false) {
    const auto requested_queue_class = queue_class(
      request_type,
      speculative
        ? service::storage_owner::kStage2ScoreManyFlagSpeculative
        : 0);
    if (!requested_queue_class.has_value()) return std::nullopt;

    std::lock_guard<std::mutex> lock(mutex_);
    if (peer_index >= queues_.size() || source_shard >= queues_.size() ||
        source_shard == peer_index || free_aggregates_.empty()) {
      return std::nullopt;
    }
    PeerQueue& queue = queues_[peer_index][*requested_queue_class];
    if (queue.size != 1 || queue.head == npos || queue.tail != queue.head) {
      return std::nullopt;
    }
    const std::uint32_t entry_index = queue.head;
    Entry& entry = entries_[entry_index];
    if (!entry.in_use || !entry.queued || entry.cancelled ||
        entry.request_type != request_type ||
        entry.peer_index != peer_index ||
        entry.queue_class != *requested_queue_class ||
        entry.request.size() <
          sizeof(service::storage_owner::PeerRpcHeader) ||
        find_aggregate_index(entry.logical_request_id).has_value()) {
      return std::nullopt;
    }

    const std::uint32_t aggregate_index = free_aggregates_.back();
    free_aggregates_.pop_back();
    Aggregate& aggregate = aggregates_[aggregate_index];
    aggregate.in_use = true;
    aggregate.state = AggregateState::ready;
    aggregate.queue_class = *requested_queue_class;
    advance_generation(aggregate.generation);
    aggregate.snapshot = Stage2HomeRpcAggregate{
      .wire_request_id = entry.logical_request_id,
      .peer_index = peer_index,
      .request_type = request_type,
      .response_type = response_type(request_type),
      .item_count = entry.item_count,
      .logical_count = 1,
      .request_bytes = entry.request.size(),
      .speculative = speculative,
      .direct = true,
    };
    aggregate.members.clear();
    aggregate.members.push_back(Member{
      .entry_index = entry_index,
      .item_offset = 0,
      .query_offset = 0,
    });
    aggregate.request.clear();
    aggregate.request.swap(entry.request);

    // The logical builder already supplied this header, but normalize every
    // transport-owned field so retries always repost one exact valid image.
    auto* header = reinterpret_cast<
      service::storage_owner::PeerRpcHeader*>(aggregate.request.data());
    header->magic = service::storage_owner::kPeerRpcMagic;
    header->version = service::storage_owner::kPeerRpcVersion;
    header->type = static_cast<std::uint32_t>(request_type);
    header->source_shard = source_shard;
    header->item_count = entry.item_count;
    header->request_id = entry.logical_request_id;
    header->status = static_cast<std::uint32_t>(
      service::storage_owner::InsertStatus::ok);
    header->reserved = 0;

    unlink_queued(entry_index);
    entry.aggregate_slot = aggregate_index;
    entry.aggregate_generation = aggregate.generation;
    // Capacity ownership moved to aggregate.request, so resident byte
    // accounting is intentionally unchanged here.
    release_entry_request(entry);
    insert_aggregate_bucket(
      aggregate.snapshot.wire_request_id, aggregate_index);
    ++aggregate_size_;
    return aggregate.snapshot;
  }

  [[nodiscard]] std::optional<Stage2HomeRpcPostLease>
  claim_ready_for_post(
      std::uint64_t wire_request_id,
      std::span<byte_t> destination,
      std::size_t& request_bytes) {
    request_bytes = 0;
    std::lock_guard<std::mutex> lock(mutex_);
    Aggregate* aggregate = find_aggregate(wire_request_id);
    if (aggregate == nullptr || aggregate->state != AggregateState::ready ||
        destination.size() < aggregate->request.size()) {
      return std::nullopt;
    }
    std::copy(aggregate->request.begin(), aggregate->request.end(),
              destination.begin());
    if (aggregate->retry_queued) {
      unlink_ready_aggregate(static_cast<std::uint32_t>(
        aggregate - aggregates_.data()));
    }
    aggregate->state = AggregateState::posted;
    aggregate->deadline_ns = 0;
    request_bytes = aggregate->request.size();
    return Stage2HomeRpcPostLease{
      .slot = static_cast<std::size_t>(aggregate - aggregates_.data()),
      .generation = aggregate->generation,
      .wire_request_id = wire_request_id,
    };
  }

  [[nodiscard]] bool mark_awaiting_response(
      Stage2HomeRpcPostLease lease,
      std::uint64_t deadline_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (deadline_ns == 0 ||
        !valid_post_lease(lease, AggregateState::posted)) {
      return false;
    }
    aggregates_[lease.slot].state = AggregateState::await_response;
    aggregates_[lease.slot].deadline_ns = deadline_ns;
    lower_earliest_deadline(deadline_ns);
    return true;
  }

  [[nodiscard]] bool release_post_claim(Stage2HomeRpcPostLease lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_post_lease(lease, AggregateState::posted)) return false;
    Aggregate& aggregate = aggregates_[lease.slot];
    aggregate.state = AggregateState::ready;
    enqueue_ready_aggregate(static_cast<std::uint32_t>(lease.slot));
    return true;
  }

  [[nodiscard]] bool retry_after_timeout(std::uint64_t wire_request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    Aggregate* aggregate = find_aggregate(wire_request_id);
    if (aggregate == nullptr ||
        (aggregate->state != AggregateState::posted &&
         aggregate->state != AggregateState::await_response)) {
      return false;
    }
    aggregate->state = AggregateState::ready;
    aggregate->deadline_ns = 0;
    enqueue_ready_aggregate(static_cast<std::uint32_t>(
      aggregate - aggregates_.data()));
    return true;
  }

  // The aggregate remains live after demultiplexing. The caller must publish
  // every returned logical response first, then call finish_success(). This
  // ordering prevents exact-capacity reuse from racing a delayed fan-out.
  [[nodiscard]] std::optional<Stage2HomeRpcDemuxResult>
  demultiplex_response(
      std::uint64_t wire_request_id,
      std::span<const byte_t> response) {
    std::lock_guard<std::mutex> lock(mutex_);
    Aggregate* aggregate = find_aggregate(wire_request_id);
    if (aggregate == nullptr ||
        aggregate->snapshot.direct ||
        (aggregate->state != AggregateState::posted &&
         aggregate->state != AggregateState::await_response) ||
        !valid_outer_response(*aggregate, response)) {
      return std::nullopt;
    }
    std::optional<std::vector<Stage2HomeRpcLogicalResponse>> logical;
    if (aggregate->snapshot.request_type ==
        service::storage_owner::PeerRpcType::stage2_expand_score_request) {
      logical = demultiplex_expand_response(*aggregate, response);
    } else {
      logical = demultiplex_score_response(*aggregate, response);
    }
    if (!logical.has_value()) return std::nullopt;
    aggregate->state = AggregateState::leased;
    return Stage2HomeRpcDemuxResult{
      .lease = Stage2HomeRpcAggregateLease{
        .slot = static_cast<std::size_t>(aggregate - aggregates_.data()),
        .generation = aggregate->generation,
        .wire_request_id = wire_request_id,
      },
      .logical_responses = std::move(*logical),
    };
  }

  [[nodiscard]] bool finish_success(Stage2HomeRpcAggregateLease lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_lease(lease)) return false;
    const auto aggregate_index = static_cast<std::uint32_t>(lease.slot);
    Aggregate& aggregate = aggregates_[aggregate_index];
    for (const Member& member : aggregate.members) {
      release_entry(member.entry_index);
    }
    erase_aggregate_bucket(lease.wire_request_id);
    release_aggregate(aggregate_index);
    --aggregate_size_;
    return true;
  }

  // Validate and retire a singleton before its response becomes visible in
  // the original logical registry.  Publishing first would allow an active
  // context to consume/rearm the registry cell and enqueue the same logical
  // ID while the old outbox entry still exists; deleting that old entry
  // afterwards would strand the new pending registry cell.  The peer/header
  // validation and retirement therefore share this one outbox critical
  // section.  A late response remains valid after deadline promotion
  // (ready), while posting (posted), or while awaiting a response.
  [[nodiscard]] Stage2HomeRpcDirectResponseResult finish_direct_response(
      std::uint32_t peer_index,
      std::span<const byte_t> response) {
    using service::storage_owner::PeerRpcHeader;
    if (response.size() < sizeof(PeerRpcHeader)) {
      return Stage2HomeRpcDirectResponseResult::not_direct;
    }
    PeerRpcHeader header{};
    std::memcpy(&header, response.data(), sizeof(header));
    std::lock_guard<std::mutex> lock(mutex_);
    const auto aggregate_index = find_aggregate_index(header.request_id);
    if (!aggregate_index.has_value()) {
      return Stage2HomeRpcDirectResponseResult::not_direct;
    }
    Aggregate& aggregate = aggregates_[*aggregate_index];
    if (!aggregate.snapshot.direct) {
      return Stage2HomeRpcDirectResponseResult::not_direct;
    }
    if (aggregate.snapshot.peer_index != peer_index ||
        (aggregate.state != AggregateState::ready &&
         aggregate.state != AggregateState::posted &&
         aggregate.state != AggregateState::await_response) ||
        !valid_outer_response(aggregate, response)) {
      return Stage2HomeRpcDirectResponseResult::invalid;
    }
    for (const Member& member : aggregate.members) {
      release_entry(member.entry_index);
    }
    erase_aggregate_bucket(header.request_id);
    release_aggregate(*aggregate_index);
    --aggregate_size_;
    return Stage2HomeRpcDirectResponseResult::finished;
  }

  // Publication failed before any logical completion became visible. Release
  // only the generation-fenced demux lease; the aggregate, wire ID, and exact
  // request bytes remain available for an outer retry.
  [[nodiscard]] bool release_demux(Stage2HomeRpcAggregateLease lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_lease(lease)) return false;
    Aggregate& aggregate = aggregates_[lease.slot];
    const bool all_cancelled = std::all_of(
      aggregate.members.begin(), aggregate.members.end(),
      [&](const Member& member) {
        return entries_[member.entry_index].cancelled;
      });
    if (all_cancelled) {
      const auto aggregate_index = static_cast<std::uint32_t>(lease.slot);
      for (const Member& member : aggregate.members) {
        release_entry(member.entry_index);
      }
      erase_aggregate_bucket(lease.wire_request_id);
      release_aggregate(aggregate_index);
      --aggregate_size_;
      return true;
    }
    aggregate.state = AggregateState::ready;
    aggregate.deadline_ns = 0;
    enqueue_ready_aggregate(static_cast<std::uint32_t>(lease.slot));
    return true;
  }

  // A queued logical request can disappear immediately. Once included in an
  // outer request, its copied item range must remain until that aggregate is
  // completed/discarded; cancellation merely suppresses logical fan-out.
  [[nodiscard]] bool cancel_logical(std::uint64_t logical_request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto entry_index = find_entry_index(logical_request_id);
    if (!entry_index.has_value()) return false;
    Entry& entry = entries_[*entry_index];
    if (entry.queued) {
      unlink_queued(*entry_index);
      release_entry(*entry_index);
      return true;
    }
    if (entry.aggregate_slot >= aggregates_.size()) return false;
    Aggregate& aggregate = aggregates_[entry.aggregate_slot];
    if (!aggregate.in_use ||
        aggregate.generation != entry.aggregate_generation) {
      return false;
    }
    entry.cancelled = true;
    const bool all_cancelled = std::all_of(
      aggregate.members.begin(), aggregate.members.end(),
      [&](const Member& member) {
        return entries_[member.entry_index].cancelled;
      });
    if (all_cancelled && aggregate.state != AggregateState::leased) {
      const std::uint32_t aggregate_index = entry.aggregate_slot;
      for (const Member& member : aggregate.members) {
        release_entry(member.entry_index);
      }
      erase_aggregate_bucket(aggregate.snapshot.wire_request_id);
      release_aggregate(aggregate_index);
      --aggregate_size_;
    }
    return true;
  }

  [[nodiscard]] bool discard_aggregate(std::uint64_t wire_request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto aggregate_index = find_aggregate_index(wire_request_id);
    if (!aggregate_index.has_value()) return false;
    Aggregate& aggregate = aggregates_[*aggregate_index];
    if (aggregate.state != AggregateState::ready) {
      return false;
    }
    for (const Member& member : aggregate.members) {
      release_entry(member.entry_index);
    }
    erase_aggregate_bucket(wire_request_id);
    release_aggregate(*aggregate_index);
    --aggregate_size_;
    return true;
  }

  [[nodiscard]] std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return logical_size_;
  }

  [[nodiscard]] std::size_t aggregate_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return aggregate_size_;
  }

  [[nodiscard]] bool owns_wire_request(
      std::uint64_t wire_request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return find_aggregate_index(wire_request_id).has_value();
  }

  [[nodiscard]] bool is_direct_wire_request(
      std::uint64_t wire_request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const Aggregate* aggregate = find_aggregate(wire_request_id);
    return aggregate != nullptr && aggregate->snapshot.direct;
  }

  [[nodiscard]] std::size_t resident_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return resident_bytes_;
  }

  [[nodiscard]] std::size_t byte_capacity() const noexcept {
    return byte_capacity_;
  }

  [[nodiscard]] std::size_t queued_size(
      std::uint32_t peer_index,
      service::storage_owner::PeerRpcType request_type,
      bool speculative = false) const {
    const auto requested_queue_class = queue_class(
      request_type,
      speculative
        ? service::storage_owner::kStage2ScoreManyFlagSpeculative
        : 0);
    if (!requested_queue_class.has_value()) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    return peer_index < queues_.size()
      ? queues_[peer_index][*requested_queue_class].size : 0;
  }

  [[nodiscard]] bool has_ready(
      std::uint32_t peer_index,
      service::storage_owner::PeerRpcType request_type,
      bool speculative = false) const {
    if (peer_index >= queues_.size()) return false;
    return (ready_peer_mask(request_type, speculative) &
            (std::uint64_t{1} << peer_index)) != 0;
  }

  [[nodiscard]] std::uint64_t ready_peer_mask(
      service::storage_owner::PeerRpcType request_type,
      bool speculative = false) const {
    const auto requested_queue_class = queue_class(
      request_type,
      speculative
        ? service::storage_owner::kStage2ScoreManyFlagSpeculative
        : 0);
    if (!requested_queue_class.has_value()) return 0;
    return ready_peer_masks_[*requested_queue_class].load(
             std::memory_order_acquire) |
      retry_ready_peer_masks_[*requested_queue_class].load(
        std::memory_order_acquire);
  }

  [[nodiscard]] std::optional<std::uint64_t> next_retry_wire_request(
      std::uint32_t peer_index,
      service::storage_owner::PeerRpcType request_type,
      bool speculative = false) const {
    if (peer_index >= aggregate_ready_queues_.size()) return std::nullopt;
    const auto requested_queue_class = queue_class(
      request_type,
      speculative
        ? service::storage_owner::kStage2ScoreManyFlagSpeculative
        : 0);
    if (!requested_queue_class.has_value()) return std::nullopt;
    std::lock_guard<std::mutex> lock(mutex_);
    const AggregateQueue& queue =
      aggregate_ready_queues_[peer_index][*requested_queue_class];
    if (queue.head == npos) return std::nullopt;
    const Aggregate& aggregate = aggregates_[queue.head];
    if (!aggregate.in_use || aggregate.state != AggregateState::ready ||
        !aggregate.retry_queued) {
      return std::nullopt;
    }
    return aggregate.snapshot.wire_request_id;
  }

  std::size_t promote_expired(std::uint64_t now_ns) {
    if (now_ns == 0) return 0;
    // The progress loop calls this every millisecond.  In the normal case
    // RPCs finish in microseconds and the configured deadline is tens of
    // seconds away, so avoid both the global mutex and a full aggregate-slab
    // scan until an observed deadline can actually be due.  A completed or
    // cancelled earliest owner may leave a stale lower bound; that costs one
    // scan at its former deadline and is repaired below without a removal
    // heap or another contended data structure.
    if (now_ns < earliest_deadline_ns_.load(std::memory_order_acquire)) {
      return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    // Another progress attempt may have repaired the lower bound while this
    // caller waited for a producer/response critical section.
    if (now_ns < earliest_deadline_ns_.load(std::memory_order_relaxed)) {
      return 0;
    }
    std::size_t promoted = 0;
    std::uint64_t next_deadline =
      std::numeric_limits<std::uint64_t>::max();
    for (std::uint32_t index = 0; index < aggregates_.size(); ++index) {
      Aggregate& aggregate = aggregates_[index];
      if (!aggregate.in_use ||
          aggregate.state != AggregateState::await_response ||
          aggregate.deadline_ns == 0) {
        continue;
      }
      if (aggregate.deadline_ns <= now_ns) {
        aggregate.state = AggregateState::ready;
        aggregate.deadline_ns = 0;
        enqueue_ready_aggregate(index);
        ++promoted;
      } else {
        next_deadline = std::min(next_deadline, aggregate.deadline_ns);
      }
    }
    earliest_deadline_ns_.store(next_deadline, std::memory_order_release);
    return promoted;
  }

 private:
  static constexpr std::size_t kQueueClassCount = 4;
  static constexpr std::size_t kExpandAuthoritativeQueue = 0;
  static constexpr std::size_t kExpandSpeculativeQueue = 1;
  static constexpr std::size_t kScoreAuthoritativeQueue = 2;
  static constexpr std::size_t kScoreSpeculativeQueue = 3;
  static constexpr std::uint32_t npos =
    std::numeric_limits<std::uint32_t>::max();

  struct DispatchMetadata {
    std::uint32_t query_count{};
    std::uint32_t score_flags{};
    std::size_t queue_class{};
  };

  struct RequestDigest {
    std::uint64_t first{};
    std::uint64_t second{};

    bool operator==(const RequestDigest&) const = default;
  };

  struct Entry {
    bool in_use{};
    bool queued{};
    bool cancelled{};
    std::uint64_t logical_request_id{};
    std::uint32_t peer_index{};
    service::storage_owner::PeerRpcType request_type{
      service::storage_owner::PeerRpcType::stage2_expand_score_request};
    service::storage_owner::PeerRpcType response_type{
      service::storage_owner::PeerRpcType::stage2_expand_score_response};
    std::uint32_t item_count{};
    std::uint32_t query_count{};
    std::uint32_t score_flags{};
    std::size_t queue_class{};
    Stage2ContextOwnerKey completion_owner{};
    RequestDigest request_digest{};
    std::uint32_t previous{npos};
    std::uint32_t next{npos};
    std::uint32_t aggregate_slot{npos};
    std::uint64_t aggregate_generation{};
    std::vector<byte_t> request;
  };

  struct Member {
    std::uint32_t entry_index{};
    std::uint32_t item_offset{};
    std::uint32_t query_offset{};
  };

  enum class AggregateState : std::uint8_t {
    free,
    ready,
    posted,
    await_response,
    leased,
  };

  struct Aggregate {
    bool in_use{};
    AggregateState state{AggregateState::free};
    bool retry_queued{};
    std::size_t queue_class{};
    std::uint32_t retry_previous{npos};
    std::uint32_t retry_next{npos};
    std::uint64_t deadline_ns{};
    std::uint64_t generation{};
    Stage2HomeRpcAggregate snapshot{};
    std::vector<Member> members;
    std::vector<byte_t> request;
  };

  struct PeerQueue {
    std::uint32_t head{npos};
    std::uint32_t tail{npos};
    std::size_t size{};
  };

  struct AggregateQueue {
    std::uint32_t head{npos};
    std::uint32_t tail{npos};
    std::size_t size{};
  };

  using PeerQueues = std::array<PeerQueue, kQueueClassCount>;
  using PeerAggregateQueues =
    std::array<AggregateQueue, kQueueClassCount>;

  struct IdBucket {
    std::uint64_t request_id{};
    std::uint32_t slot{npos};
    bool occupied{};
  };

  [[nodiscard]] std::optional<DispatchMetadata> validate_dispatch(
      const Stage2HomeRpcDispatch& dispatch) const {
    using namespace service::storage_owner;
    if (dispatch.logical_request_id == 0 ||
        dispatch.peer_index >= queues_.size() ||
        dispatch.item_count == 0 || dispatch.request.empty() ||
        !valid_completion_owner(dispatch.completion_owner)) {
      return std::nullopt;
    }
    if (dispatch.request_type ==
        PeerRpcType::stage2_expand_score_request) {
      if (dispatch.item_count > expand_wire_max_items_ ||
          dispatch.request.size() !=
            stage2_expand_score_request_bytes(dispatch.item_count)) {
        return std::nullopt;
      }
      const auto* items = stage2_expand_score_items(
        dispatch.request.data());
      for (std::uint32_t index = 0; index < dispatch.item_count; ++index) {
        if (items[index].operation > static_cast<std::uint32_t>(
              Stage2HomeOperation::score_only)) {
          return std::nullopt;
        }
      }
      return DispatchMetadata{
        .queue_class = dispatch.speculative
          ? kExpandSpeculativeQueue : kExpandAuthoritativeQueue,
      };
    }
    if (dispatch.request_type !=
        PeerRpcType::stage2_score_many_request ||
        dispatch.item_count > score_wire_max_items_ ||
        dispatch.request.size() < stage2_score_many_items_offset()) {
      return std::nullopt;
    }
    const auto* own_header = stage2_score_many_header(
      dispatch.request.data());
    if (own_header->query_count == 0 ||
        own_header->query_count > dispatch.item_count ||
        own_header->query_count > score_wire_max_queries_ ||
        !stage2_score_many_flags_valid(own_header->flags) ||
        dispatch.request.size() != stage2_score_many_request_bytes(
          dispatch.item_count, own_header->query_count)) {
      return std::nullopt;
    }
    const auto* items = stage2_score_many_items(dispatch.request.data());
    for (std::uint32_t index = 0; index < dispatch.item_count; ++index) {
      if (items[index].query_index >= own_header->query_count) {
        return std::nullopt;
      }
    }
    if (dispatch.speculative !=
        stage2_score_many_is_speculative(own_header->flags)) {
      return std::nullopt;
    }
    const auto selected_queue = queue_class(
      dispatch.request_type, own_header->flags);
    if (!selected_queue.has_value()) return std::nullopt;
    return DispatchMetadata{
      .query_count = own_header->query_count,
      .score_flags = own_header->flags,
      .queue_class = *selected_queue,
    };
  }

  [[nodiscard]] static std::optional<std::size_t> queue_class(
      service::storage_owner::PeerRpcType request_type,
      std::uint32_t score_flags) {
    using namespace service::storage_owner;
    if (request_type == PeerRpcType::stage2_expand_score_request) {
      if (score_flags != 0 &&
          score_flags != kStage2ScoreManyFlagSpeculative) {
        return std::nullopt;
      }
      return score_flags == 0
        ? std::optional<std::size_t>{kExpandAuthoritativeQueue}
        : std::optional<std::size_t>{kExpandSpeculativeQueue};
    }
    if (request_type != PeerRpcType::stage2_score_many_request ||
        !stage2_score_many_flags_valid(score_flags)) {
      return std::nullopt;
    }
    return stage2_score_many_is_speculative(score_flags)
      ? kScoreSpeculativeQueue : kScoreAuthoritativeQueue;
  }

  [[nodiscard]] static service::storage_owner::PeerRpcType response_type(
      service::storage_owner::PeerRpcType request_type) {
    using service::storage_owner::PeerRpcType;
    return request_type == PeerRpcType::stage2_score_many_request
      ? PeerRpcType::stage2_score_many_response
      : PeerRpcType::stage2_expand_score_response;
  }

  [[nodiscard]] static bool same_request(
      const Entry& entry,
      const Stage2HomeRpcDispatch& dispatch,
      const DispatchMetadata& metadata,
      RequestDigest request_digest) {
    if (entry.peer_index != dispatch.peer_index ||
        entry.request_type != dispatch.request_type ||
        entry.item_count != dispatch.item_count ||
        entry.query_count != metadata.query_count ||
        entry.score_flags != metadata.score_flags ||
        entry.queue_class != metadata.queue_class ||
        entry.completion_owner != dispatch.completion_owner ||
        entry.request_digest != request_digest) {
      return false;
    }
    if (entry.request.empty()) return true;
    if (entry.request.size() != dispatch.request.size()) return false;
    const std::size_t header_bytes =
      sizeof(service::storage_owner::PeerRpcHeader);
    return std::equal(
      entry.request.begin() + header_bytes, entry.request.end(),
      dispatch.request.begin() + header_bytes);
  }

  [[nodiscard]] static bool valid_completion_owner(
      const Stage2ContextOwnerKey& owner) {
    const bool empty = owner.runtime_epoch == 0 && owner.worker_id == 0 &&
      owner.slot == 0 && owner.token == 0;
    return empty || (owner.runtime_epoch != 0 && owner.token != 0);
  }

  [[nodiscard]] static RequestDigest digest_request(
      std::span<const byte_t> request) {
    constexpr std::uint64_t kOffsetFirst = 1469598103934665603ULL;
    constexpr std::uint64_t kOffsetSecond = 1099511628211ULL;
    constexpr std::uint64_t kPrimeFirst = 1099511628211ULL;
    constexpr std::uint64_t kPrimeSecond = 14029467366897019727ULL;
    const std::size_t begin = std::min(
      request.size(), sizeof(service::storage_owner::PeerRpcHeader));
    RequestDigest digest{kOffsetFirst, kOffsetSecond};
    for (std::size_t index = begin; index < request.size(); ++index) {
      const std::uint64_t value = static_cast<std::uint8_t>(request[index]);
      digest.first = (digest.first ^ value) * kPrimeFirst;
      digest.second ^= value + 0x9e3779b97f4a7c15ULL +
        (digest.second << 6) + (digest.second >> 2);
      digest.second *= kPrimeSecond;
    }
    digest.first ^= request.size();
    digest.second ^= std::rotl(
      static_cast<std::uint64_t>(request.size()), 29);
    return digest;
  }

  [[nodiscard]] bool build_wire_request(
      std::span<const Member> members,
      std::uint64_t wire_request_id,
      std::uint32_t source_shard,
      service::storage_owner::PeerRpcType request_type,
      std::uint32_t total_items,
      std::uint32_t total_queries,
      std::vector<byte_t>& request) const {
    using namespace service::storage_owner;
    const bool score_many = request_type ==
      PeerRpcType::stage2_score_many_request;
    const std::size_t bytes = score_many
      ? stage2_score_many_request_bytes(total_items, total_queries)
      : stage2_expand_score_request_bytes(total_items);
    if (bytes == std::numeric_limits<std::size_t>::max() ||
        bytes > max_request_bytes_) {
      return false;
    }
    request.reserve(bytes);
    request.resize(bytes);
    std::fill(request.begin(), request.end(), byte_t{0});
    auto* header = reinterpret_cast<PeerRpcHeader*>(request.data());
    header->magic = kPeerRpcMagic;
    header->version = kPeerRpcVersion;
    header->type = static_cast<std::uint32_t>(request_type);
    header->source_shard = source_shard;
    header->item_count = total_items;
    header->request_id = wire_request_id;
    header->status = static_cast<std::uint32_t>(InsertStatus::ok);
    header->reserved = 0;

    if (!score_many) {
      auto* output_items = stage2_expand_score_items(request.data());
      byte_t* output_queries = stage2_expand_score_queries(
        request.data(), total_items);
      std::uint32_t item_offset = 0;
      for (const Member& member : members) {
        const Entry& entry = entries_[member.entry_index];
        const auto* input_items = stage2_expand_score_items(
          entry.request.data());
        const byte_t* input_queries = stage2_expand_score_queries(
          entry.request.data(), entry.item_count);
        std::copy(input_items, input_items + entry.item_count,
                  output_items + item_offset);
        std::memcpy(
          output_queries + static_cast<std::size_t>(item_offset) *
            VamanaNode::vector_bytes(),
          input_queries,
          static_cast<std::size_t>(entry.item_count) *
            VamanaNode::vector_bytes());
        item_offset += entry.item_count;
      }
      return true;
    }

    const Entry& first = entries_[members.front().entry_index];
    auto* output_own_header = stage2_score_many_header(request.data());
    output_own_header->query_count = total_queries;
    output_own_header->flags = first.score_flags;
    auto* output_items = stage2_score_many_items(request.data());
    byte_t* output_queries = stage2_score_many_queries(
      request.data(), total_items);
    std::uint32_t item_offset = 0;
    std::uint32_t query_offset = 0;
    for (const Member& member : members) {
      const Entry& entry = entries_[member.entry_index];
      if (entry.score_flags != first.score_flags) return false;
      const auto* input_items = stage2_score_many_items(
        entry.request.data());
      for (std::uint32_t item = 0; item < entry.item_count; ++item) {
        output_items[item_offset + item] = input_items[item];
        output_items[item_offset + item].query_index += query_offset;
      }
      const byte_t* input_queries = stage2_score_many_queries(
        entry.request.data(), entry.item_count);
      std::memcpy(
        output_queries + static_cast<std::size_t>(query_offset) *
          VamanaNode::vector_bytes(),
        input_queries,
        static_cast<std::size_t>(entry.query_count) *
          VamanaNode::vector_bytes());
      item_offset += entry.item_count;
      query_offset += entry.query_count;
    }
    return true;
  }

  [[nodiscard]] static bool valid_outer_response(
      const Aggregate& aggregate,
      std::span<const byte_t> response) {
    using namespace service::storage_owner;
    if (response.size() < sizeof(PeerRpcHeader)) return false;
    PeerRpcHeader header{};
    std::memcpy(&header, response.data(), sizeof(header));
    if (header.magic != kPeerRpcMagic ||
        header.version != kPeerRpcVersion ||
        header.type != static_cast<std::uint32_t>(
          aggregate.snapshot.response_type) ||
        header.source_shard != aggregate.snapshot.peer_index ||
        header.item_count != aggregate.snapshot.item_count ||
        header.request_id != aggregate.snapshot.wire_request_id ||
        header.status != static_cast<std::uint32_t>(InsertStatus::ok) ||
        header.reserved != 0) {
      return false;
    }
    if (aggregate.snapshot.request_type ==
        PeerRpcType::stage2_score_many_request) {
      return response.size() == stage2_score_many_response_bytes(
        aggregate.snapshot.item_count);
    }
    const std::size_t minimum = stage2_expand_score_response_bytes(
      aggregate.snapshot.item_count, 0);
    const std::size_t maximum = stage2_expand_score_response_bytes(
      aggregate.snapshot.item_count);
    return response.size() >= minimum && response.size() <= maximum &&
      (response.size() - minimum) % sizeof(Stage2ExpandScoreNeighbor) == 0;
  }

  [[nodiscard]] std::optional<std::vector<Stage2HomeRpcLogicalResponse>>
  demultiplex_expand_response(
      const Aggregate& aggregate,
      std::span<const byte_t> response) const {
    using namespace service::storage_owner;
    const std::size_t minimum = stage2_expand_score_response_bytes(
      aggregate.snapshot.item_count, 0);
    const std::size_t compact_neighbor_count =
      (response.size() - minimum) / sizeof(Stage2ExpandScoreNeighbor);
    const auto* results = stage2_expand_score_results(response.data());
    const auto* neighbors = stage2_expand_score_neighbors(
      response.data(), aggregate.snapshot.item_count);
    const auto* request_items = stage2_expand_score_items(
      aggregate.request.data());
    const std::size_t neighbor_stride = VamanaNode::graph_entry_capacity();
    std::size_t running_neighbor = 0;
    for (std::uint32_t item = 0;
         item < aggregate.snapshot.item_count; ++item) {
      const Stage2ExpandScoreItem& request = request_items[item];
      const Stage2ExpandScoreResult& result = results[item];
      if (result.pointer_raw != request.pointer_raw ||
          result.generation != request.generation ||
          result.search_index != request.search_index ||
          result.operation != request.operation ||
          result.disposition > static_cast<std::uint32_t>(
            Stage2HomeDisposition::terminal) ||
          result.neighbor_count > neighbor_stride ||
          (request.operation == static_cast<std::uint32_t>(
             Stage2HomeOperation::score_only) &&
           result.neighbor_count != 0) ||
          result.neighbor_offset != running_neighbor ||
          result.neighbor_count > compact_neighbor_count - running_neighbor ||
          result.operation > static_cast<std::uint32_t>(
            Stage2HomeOperation::score_only)) {
        return std::nullopt;
      }
      for (std::uint32_t neighbor = 0;
           neighbor < result.neighbor_count; ++neighbor) {
        if (neighbors[running_neighbor + neighbor].disposition >
            static_cast<std::uint32_t>(
              Stage2HomeDisposition::unscored)) {
          return std::nullopt;
        }
      }
      running_neighbor += result.neighbor_count;
    }
    if (running_neighbor != compact_neighbor_count) return std::nullopt;

    std::vector<Stage2HomeRpcLogicalResponse> logical;
    logical.reserve(aggregate.members.size());
    for (const Member& member : aggregate.members) {
      const Entry& entry = entries_[member.entry_index];
      if (entry.cancelled) continue;
      const std::size_t neighbor_begin =
        results[member.item_offset].neighbor_offset;
      std::size_t member_neighbor_count = 0;
      for (std::uint32_t item = 0; item < entry.item_count; ++item) {
        member_neighbor_count +=
          results[member.item_offset + item].neighbor_count;
      }
      std::vector<byte_t> output(stage2_expand_score_response_bytes(
        entry.item_count, static_cast<std::uint32_t>(
          member_neighbor_count)), byte_t{0});
      write_logical_header(output, entry);
      auto* output_results = stage2_expand_score_results(output.data());
      for (std::uint32_t item = 0; item < entry.item_count; ++item) {
        output_results[item] = results[member.item_offset + item];
        output_results[item].neighbor_offset -=
          static_cast<std::uint32_t>(neighbor_begin);
      }
      auto* output_neighbors = stage2_expand_score_neighbors(
        output.data(), entry.item_count);
      std::copy(neighbors + neighbor_begin,
                neighbors + neighbor_begin + member_neighbor_count,
                output_neighbors);
      logical.push_back(Stage2HomeRpcLogicalResponse{
        .logical_request_id = entry.logical_request_id,
        .peer_index = entry.peer_index,
        .response_type = entry.response_type,
        .item_count = entry.item_count,
        .completion_owner = entry.completion_owner,
        .response = std::move(output),
      });
    }
    return logical;
  }

  [[nodiscard]] std::optional<std::vector<Stage2HomeRpcLogicalResponse>>
  demultiplex_score_response(
      const Aggregate& aggregate,
      std::span<const byte_t> response) const {
    using namespace service::storage_owner;
    const auto* results = stage2_score_many_results(response.data());
    const auto* request_items = stage2_score_many_items(
      aggregate.request.data());
    for (std::uint32_t item = 0;
         item < aggregate.snapshot.item_count; ++item) {
      const Stage2ScoreManyItem& request = request_items[item];
      const Stage2ScoreManyResult& result = results[item];
      if (result.pointer_raw != request.pointer_raw ||
          result.generation != request.generation ||
          result.search_index != request.search_index ||
          result.reserved != 0 ||
          result.disposition > static_cast<std::uint32_t>(
            Stage2HomeDisposition::terminal)) {
        return std::nullopt;
      }
    }
    std::vector<Stage2HomeRpcLogicalResponse> logical;
    logical.reserve(aggregate.members.size());
    for (const Member& member : aggregate.members) {
      const Entry& entry = entries_[member.entry_index];
      if (entry.cancelled) continue;
      std::vector<byte_t> output(
        stage2_score_many_response_bytes(entry.item_count), byte_t{0});
      write_logical_header(output, entry);
      auto* output_results = stage2_score_many_results(output.data());
      std::copy(results + member.item_offset,
                results + member.item_offset + entry.item_count,
                output_results);
      logical.push_back(Stage2HomeRpcLogicalResponse{
        .logical_request_id = entry.logical_request_id,
        .peer_index = entry.peer_index,
        .response_type = entry.response_type,
        .item_count = entry.item_count,
        .completion_owner = entry.completion_owner,
        .response = std::move(output),
      });
    }
    return logical;
  }

  static void write_logical_header(std::vector<byte_t>& output,
                                   const Entry& entry) {
    using namespace service::storage_owner;
    auto* header = reinterpret_cast<PeerRpcHeader*>(output.data());
    header->magic = kPeerRpcMagic;
    header->version = kPeerRpcVersion;
    header->type = static_cast<std::uint32_t>(entry.response_type);
    header->source_shard = entry.peer_index;
    header->item_count = entry.item_count;
    header->request_id = entry.logical_request_id;
    header->status = static_cast<std::uint32_t>(InsertStatus::ok);
    header->reserved = 0;
  }

  [[nodiscard]] std::optional<std::uint32_t> find_entry_index(
      std::uint64_t logical_request_id) const {
    const auto slot = find_bucket(
      logical_buckets_, logical_bucket_mask_, logical_request_id);
    if (!slot.has_value() || *slot >= entries_.size() ||
        !entries_[*slot].in_use ||
        entries_[*slot].logical_request_id != logical_request_id) {
      return std::nullopt;
    }
    return *slot;
  }

  [[nodiscard]] std::optional<std::uint32_t> find_aggregate_index(
      std::uint64_t wire_request_id) const {
    const auto slot = find_bucket(
      aggregate_buckets_, aggregate_bucket_mask_, wire_request_id);
    if (!slot.has_value() || *slot >= aggregates_.size() ||
        !aggregates_[*slot].in_use ||
        aggregates_[*slot].snapshot.wire_request_id != wire_request_id) {
      return std::nullopt;
    }
    return *slot;
  }

  [[nodiscard]] Aggregate* find_aggregate(
      std::uint64_t wire_request_id) {
    const auto index = find_aggregate_index(wire_request_id);
    return index.has_value() ? &aggregates_[*index] : nullptr;
  }

  [[nodiscard]] const Aggregate* find_aggregate(
      std::uint64_t wire_request_id) const {
    const auto index = find_aggregate_index(wire_request_id);
    return index.has_value() ? &aggregates_[*index] : nullptr;
  }

  void unlink_queued(std::uint32_t index) {
    Entry& entry = entries_[index];
    PeerQueue& queue = queues_[entry.peer_index][entry.queue_class];
    if (!entry.queued || queue.size == 0) {
      throw std::logic_error("stage2 home RPC queue unlink is inconsistent");
    }
    if (entry.previous == npos) {
      queue.head = entry.next;
    } else {
      entries_[entry.previous].next = entry.next;
    }
    if (entry.next == npos) {
      queue.tail = entry.previous;
    } else {
      entries_[entry.next].previous = entry.previous;
    }
    --queue.size;
    if (queue.size == 0) {
      ready_peer_masks_[entry.queue_class].fetch_and(
        ~(std::uint64_t{1} << entry.peer_index),
        std::memory_order_acq_rel);
    }
    entry.queued = false;
    entry.previous = npos;
    entry.next = npos;
  }

  void enqueue_ready_aggregate(std::uint32_t index) {
    if (index >= aggregates_.size()) {
      throw std::logic_error("stage2 home retry aggregate is out of range");
    }
    Aggregate& aggregate = aggregates_[index];
    if (!aggregate.in_use || aggregate.state != AggregateState::ready ||
        aggregate.retry_queued ||
        aggregate.snapshot.peer_index >= aggregate_ready_queues_.size() ||
        aggregate.queue_class >= kQueueClassCount) {
      throw std::logic_error("stage2 home retry aggregate is inconsistent");
    }
    AggregateQueue& queue = aggregate_ready_queues_[
      aggregate.snapshot.peer_index][aggregate.queue_class];
    aggregate.retry_previous = queue.tail;
    aggregate.retry_next = npos;
    if (queue.tail == npos) {
      queue.head = index;
    } else {
      aggregates_[queue.tail].retry_next = index;
    }
    queue.tail = index;
    ++queue.size;
    aggregate.retry_queued = true;
    retry_ready_peer_masks_[aggregate.queue_class].fetch_or(
      std::uint64_t{1} << aggregate.snapshot.peer_index,
      std::memory_order_release);
  }

  void unlink_ready_aggregate(std::uint32_t index) {
    if (index >= aggregates_.size()) {
      throw std::logic_error("stage2 home retry unlink is out of range");
    }
    Aggregate& aggregate = aggregates_[index];
    if (!aggregate.retry_queued ||
        aggregate.snapshot.peer_index >= aggregate_ready_queues_.size() ||
        aggregate.queue_class >= kQueueClassCount) {
      throw std::logic_error("stage2 home retry unlink is inconsistent");
    }
    AggregateQueue& queue = aggregate_ready_queues_[
      aggregate.snapshot.peer_index][aggregate.queue_class];
    if (queue.size == 0) {
      throw std::logic_error("stage2 home retry queue underflow");
    }
    if (aggregate.retry_previous == npos) {
      queue.head = aggregate.retry_next;
    } else {
      aggregates_[aggregate.retry_previous].retry_next =
        aggregate.retry_next;
    }
    if (aggregate.retry_next == npos) {
      queue.tail = aggregate.retry_previous;
    } else {
      aggregates_[aggregate.retry_next].retry_previous =
        aggregate.retry_previous;
    }
    --queue.size;
    if (queue.size == 0) {
      retry_ready_peer_masks_[aggregate.queue_class].fetch_and(
        ~(std::uint64_t{1} << aggregate.snapshot.peer_index),
        std::memory_order_acq_rel);
    }
    aggregate.retry_queued = false;
    aggregate.retry_previous = npos;
    aggregate.retry_next = npos;
  }

  void release_entry(std::uint32_t index) {
    Entry& entry = entries_[index];
    erase_logical_bucket(entry.logical_request_id);
    release_entry_request(entry);
    entry.in_use = false;
    entry.queued = false;
    entry.cancelled = false;
    entry.logical_request_id = 0;
    entry.peer_index = 0;
    entry.item_count = 0;
    entry.query_count = 0;
    entry.score_flags = 0;
    entry.queue_class = 0;
    entry.completion_owner = {};
    entry.request_digest = {};
    entry.previous = npos;
    entry.next = npos;
    entry.aggregate_slot = npos;
    entry.aggregate_generation = 0;
    free_entries_.push_back(index);
    --logical_size_;
  }

  void release_entry_request(Entry& entry) {
    const std::size_t capacity = entry.request.capacity();
    if (capacity != 0) {
      if (capacity > resident_bytes_) {
        throw std::logic_error(
          "stage2 home RPC entry byte accounting underflow");
      }
      resident_bytes_ -= capacity;
      std::vector<byte_t>{}.swap(entry.request);
    }
  }

  void release_aggregate_request(Aggregate& aggregate) {
    const std::size_t capacity = aggregate.request.capacity();
    if (capacity != 0) {
      if (capacity > resident_bytes_) {
        throw std::logic_error(
          "stage2 home RPC aggregate byte accounting underflow");
      }
      resident_bytes_ -= capacity;
      std::vector<byte_t>{}.swap(aggregate.request);
    }
  }

  void release_aggregate(std::uint32_t index) {
    Aggregate& aggregate = aggregates_[index];
    if (aggregate.retry_queued) unlink_ready_aggregate(index);
    release_aggregate_request(aggregate);
    aggregate.in_use = false;
    aggregate.state = AggregateState::free;
    aggregate.retry_queued = false;
    aggregate.queue_class = 0;
    aggregate.retry_previous = npos;
    aggregate.retry_next = npos;
    aggregate.deadline_ns = 0;
    aggregate.snapshot = {};
    aggregate.members.clear();
    free_aggregates_.push_back(index);
  }

  [[nodiscard]] bool valid_lease(
      Stage2HomeRpcAggregateLease lease) const {
    return lease.valid() && lease.slot < aggregates_.size() &&
      aggregates_[lease.slot].in_use &&
      aggregates_[lease.slot].state == AggregateState::leased &&
      aggregates_[lease.slot].generation == lease.generation &&
      aggregates_[lease.slot].snapshot.wire_request_id ==
        lease.wire_request_id;
  }

  [[nodiscard]] bool valid_post_lease(
      Stage2HomeRpcPostLease lease,
      AggregateState state) const {
    return lease.valid() && lease.slot < aggregates_.size() &&
      aggregates_[lease.slot].in_use &&
      aggregates_[lease.slot].state == state &&
      aggregates_[lease.slot].generation == lease.generation &&
      aggregates_[lease.slot].snapshot.wire_request_id ==
        lease.wire_request_id;
  }

  static void advance_generation(std::uint64_t& generation) {
    ++generation;
    if (generation == 0) ++generation;
  }

  void lower_earliest_deadline(std::uint64_t deadline_ns) noexcept {
    std::uint64_t observed = earliest_deadline_ns_.load(
      std::memory_order_relaxed);
    while (deadline_ns < observed &&
           !earliest_deadline_ns_.compare_exchange_weak(
             observed, deadline_ns,
             std::memory_order_release, std::memory_order_relaxed)) {
    }
  }

  [[nodiscard]] static std::size_t hash_capacity(std::size_t capacity) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::invalid_argument("stage2 home RPC hash capacity overflows");
    }
    return std::bit_ceil(std::max<std::size_t>(4, capacity * 2));
  }

  [[nodiscard]] static std::size_t hash_request_id(std::uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<std::size_t>(value);
  }

  [[nodiscard]] static std::optional<std::uint32_t> find_bucket(
      const std::vector<IdBucket>& buckets,
      std::size_t mask,
      std::uint64_t request_id) {
    std::size_t bucket = hash_request_id(request_id) & mask;
    for (;;) {
      if (!buckets[bucket].occupied) return std::nullopt;
      if (buckets[bucket].request_id == request_id) {
        return buckets[bucket].slot;
      }
      bucket = (bucket + 1) & mask;
    }
  }

  static void insert_bucket(std::vector<IdBucket>& buckets,
                            std::size_t mask,
                            std::uint64_t request_id,
                            std::uint32_t slot) {
    std::size_t bucket = hash_request_id(request_id) & mask;
    for (;;) {
      if (!buckets[bucket].occupied) {
        buckets[bucket] = IdBucket{request_id, slot, true};
        return;
      }
      if (buckets[bucket].request_id == request_id) {
        throw std::logic_error("duplicate stage2 home RPC hash key");
      }
      bucket = (bucket + 1) & mask;
    }
  }

  static void erase_bucket(std::vector<IdBucket>& buckets,
                           std::size_t mask,
                           std::uint64_t request_id) {
    std::size_t hole = hash_request_id(request_id) & mask;
    while (buckets[hole].occupied &&
           buckets[hole].request_id != request_id) {
      hole = (hole + 1) & mask;
    }
    if (!buckets[hole].occupied) return;
    std::size_t scan = (hole + 1) & mask;
    while (buckets[scan].occupied) {
      const std::size_t home = hash_request_id(
        buckets[scan].request_id) & mask;
      const std::size_t scan_distance = (scan - home) & mask;
      const std::size_t hole_distance = (hole - home) & mask;
      if (scan_distance > hole_distance) {
        buckets[hole] = buckets[scan];
        hole = scan;
      }
      scan = (scan + 1) & mask;
    }
    buckets[hole] = {};
  }

  void insert_logical_bucket(std::uint64_t request_id,
                             std::uint32_t slot) {
    insert_bucket(logical_buckets_, logical_bucket_mask_, request_id, slot);
  }

  void erase_logical_bucket(std::uint64_t request_id) {
    if (request_id != 0) {
      erase_bucket(logical_buckets_, logical_bucket_mask_, request_id);
    }
  }

  void insert_aggregate_bucket(std::uint64_t request_id,
                               std::uint32_t slot) {
    insert_bucket(
      aggregate_buckets_, aggregate_bucket_mask_, request_id, slot);
  }

  void erase_aggregate_bucket(std::uint64_t request_id) {
    if (request_id != 0) {
      erase_bucket(
        aggregate_buckets_, aggregate_bucket_mask_, request_id);
    }
  }

  mutable std::mutex mutex_;
  std::vector<Entry> entries_;
  std::vector<std::uint32_t> free_entries_;
  std::vector<Aggregate> aggregates_;
  std::vector<std::uint32_t> free_aggregates_;
  std::vector<PeerQueues> queues_;
  std::vector<PeerAggregateQueues> aggregate_ready_queues_;
  std::array<std::atomic<std::uint64_t>, kQueueClassCount>
    ready_peer_masks_{};
  std::array<std::atomic<std::uint64_t>, kQueueClassCount>
    retry_ready_peer_masks_{};
  std::vector<IdBucket> logical_buckets_;
  std::vector<IdBucket> aggregate_buckets_;
  std::size_t logical_bucket_mask_{};
  std::size_t aggregate_bucket_mask_{};
  std::uint32_t expand_wire_max_items_{};
  std::uint32_t score_wire_max_items_{};
  std::uint32_t score_wire_max_queries_{};
  std::size_t max_request_bytes_{};
  std::size_t byte_capacity_{};
  std::size_t resident_limit_{};
  std::size_t resident_bytes_{};
  std::size_t logical_size_{};
  std::size_t aggregate_size_{};
  std::atomic<std::uint64_t> earliest_deadline_ns_{
    std::numeric_limits<std::uint64_t>::max()};
};

}  // namespace memory_node_storage_owner_maintenance_detail
