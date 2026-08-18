#pragma once

#include <algorithm>
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

#include "common/types.hh"
#include "memory_node/storage_owner_maintenance/ready_context_queue.hh"
#include "service/storage_owner_protocol.hh"

namespace memory_node_detail {

inline constexpr u32 kNoMaintenanceWakeOwner =
  std::numeric_limits<u32>::max();

enum class TryPeerResponse : std::uint8_t {
  pending,
  success,
  failure,
  stale,
};

enum class PeerResponseRegistration : std::uint8_t {
  registered,
  retry,
  already_complete,
  conflict,
  full,
};

enum class PeerRequestAction : std::uint8_t {
  execute,
  duplicate_inflight,
  replay,
  conflict,
  full,
};

// A delayed consumer must prove that the fixed slab slot still belongs to the
// operation it observed. The generation also changes when a malformed
// response is rearmed, closing the same-slot/same-request retry ABA window.
struct PeerResponseLease {
  size_t slot{std::numeric_limits<size_t>::max()};
  u64 generation{};

  [[nodiscard]] bool valid() const noexcept {
    return slot != std::numeric_limits<size_t>::max() && generation != 0;
  }
};

struct PeerRequestLease {
  size_t slot{std::numeric_limits<size_t>::max()};
  u64 generation{};

  [[nodiscard]] bool valid() const noexcept {
    return slot != std::numeric_limits<size_t>::max() && generation != 0;
  }
};

struct PeerRequestDecision {
  PeerRequestAction action{PeerRequestAction::conflict};
  service::storage_owner::PeerRpcHeader response{};
  PeerRequestLease lease{};
};

struct PeerResponseDescriptor {
  u32 peer_id{};
  u32 receive_slot{};
  size_t bytes{};
  service::storage_owner::PeerRpcHeader header{};
  bool owned_payload{};
};

struct PeerResponseCompletionTarget {
  u32 maintenance_wake_owner{kNoMaintenanceWakeOwner};
  memory_node_storage_owner_maintenance_detail::Stage2ContextOwnerKey
    context_owner{};

  [[nodiscard]] bool has_context_owner() const noexcept {
    return context_owner.runtime_epoch != 0 && context_owner.token != 0;
  }

  bool operator==(const PeerResponseCompletionTarget&) const = default;
};

struct PeerOwnedResponse {
  u32 peer_id{};
  service::storage_owner::PeerRpcHeader header{};
  PeerResponseCompletionTarget completion_target{};
  std::vector<byte_t> payload;
};

struct PeerHashProbeTelemetry {
  u64 lookups{};
  u64 probes{};
  size_t max_probe{};
};

// Fixed-capacity request/response correlation for peer RPCs. Payloads remain
// in registered RDMA receive slots until the requesting executor copies them,
// so CQ progress performs neither allocation nor payload copying.
//
// Entries live in a preallocated slab. A separate hash index has twice as many
// buckets and uses backward-shift deletion, so cumulative request churn never
// leaves tombstones and the index load is bounded by 0.5. try_take() leases a
// descriptor instead of deleting it: the operation-specific parser must call
// ack_consumed(), retry(), or await_late_delivery() with that exact
// generation.
class PeerAsyncResponseRegistry {
public:
  explicit PeerAsyncResponseRegistry(size_t requested_capacity)
      : capacity_(normalize_capacity(requested_capacity)),
        bucket_capacity_(capacity_ * 2),
        bucket_mask_(bucket_capacity_ - 1),
        slots_(capacity_),
        buckets_(bucket_capacity_) {
    initialize_free_list();
  }

  PeerAsyncResponseRegistry(const PeerAsyncResponseRegistry&) = delete;
  PeerAsyncResponseRegistry& operator=(const PeerAsyncResponseRegistry&) = delete;

  [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] size_t bucket_capacity() const noexcept {
    return bucket_capacity_;
  }

  [[nodiscard]] size_t size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_;
  }

  [[nodiscard]] PeerHashProbeTelemetry probe_telemetry() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return probe_telemetry_;
  }

  PeerResponseRegistration register_request(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count,
      u32 maintenance_wake_owner = kNoMaintenanceWakeOwner) {
    std::lock_guard<std::mutex> lock(mutex_);
    return register_request_locked(request_id, expected_shard, expected_type,
                                   expected_item_count,
                                   PeerResponseCompletionTarget{
                                     .maintenance_wake_owner =
                                       maintenance_wake_owner});
  }

  PeerResponseRegistration register_request_with_target(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count,
      PeerResponseCompletionTarget completion_target) {
    std::lock_guard<std::mutex> lock(mutex_);
    return register_request_locked(request_id, expected_shard, expected_type,
                                   expected_item_count, completion_target);
  }

  PeerResponseRegistration register_send_attempt(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count,
      u32 maintenance_wake_owner = kNoMaintenanceWakeOwner) {
    std::lock_guard<std::mutex> lock(mutex_);
    return register_request_locked(request_id, expected_shard, expected_type,
                                   expected_item_count,
                                   PeerResponseCompletionTarget{
                                     .maintenance_wake_owner =
                                       maintenance_wake_owner});
  }

  PeerResponseRegistration register_send_attempt_with_target(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count,
      PeerResponseCompletionTarget completion_target) {
    std::lock_guard<std::mutex> lock(mutex_);
    return register_request_locked(request_id, expected_shard, expected_type,
                                   expected_item_count, completion_target);
  }

  // Returns true only when ownership of the receive descriptor transfers to
  // the registry. Unknown, malformed, duplicate, late, and currently leased
  // responses remain owned by the CQ caller and are reposted immediately.
  bool try_deliver(u32 peer_id,
                   u32 receive_slot,
                   size_t bytes,
                   const service::storage_owner::PeerRpcHeader& header,
                   u32* maintenance_wake_owner = nullptr) {
    if (maintenance_wake_owner != nullptr) {
      *maintenance_wake_owner = kNoMaintenanceWakeOwner;
    }
    PeerResponseCompletionTarget target;
    const bool delivered = try_deliver_with_target(
      peer_id, receive_slot, bytes, header, &target);
    if (delivered && maintenance_wake_owner != nullptr) {
      *maintenance_wake_owner = target.maintenance_wake_owner;
    }
    return delivered;
  }

  bool try_deliver_with_target(
      u32 peer_id,
      u32 receive_slot,
      size_t bytes,
      const service::storage_owner::PeerRpcHeader& header,
      PeerResponseCompletionTarget* completion_target = nullptr) {
    if (completion_target != nullptr) *completion_target = {};
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t slot_index = find_slot_locked(header.request_id);
    if (slot_index == npos) return false;
    Slot& slot = slots_[slot_index];
    if (!delivery_matches(slot, peer_id, header)) return false;
    slot.response = PeerResponseDescriptor{
      .peer_id = peer_id,
      .receive_slot = receive_slot,
      .bytes = bytes,
      .header = header,
      .owned_payload = false,
    };
    slot.receive_descriptor_held = true;
    slot.owned_payload.clear();
    slot.state = State::complete;
    if (completion_target != nullptr) {
      *completion_target = slot.completion_target;
    }
    return true;
  }

  // Atomically publishes a demultiplexed aggregate. Validation of every
  // logical registry cell precedes the first move, so callers never expose a
  // partial fan-out that could make an exact outer retry unsafe.
  bool try_deliver_owned_batch(
      std::span<PeerOwnedResponse> responses,
      std::vector<PeerResponseCompletionTarget>& completion_targets) {
    completion_targets.clear();
    completion_targets.reserve(responses.size());
    std::lock_guard<std::mutex> lock(mutex_);
    ++validation_epoch_;
    if (validation_epoch_ == 0) ++validation_epoch_;
    for (const PeerOwnedResponse& response : responses) {
      if (!valid_owned_response(response)) return false;
      const std::size_t slot_index = find_slot_locked(
        response.header.request_id);
      if (slot_index == npos ||
          !delivery_matches(slots_[slot_index], response.peer_id,
                            response.header) ||
          slots_[slot_index].completion_target !=
            response.completion_target) {
        return false;
      }
      Slot& slot = slots_[slot_index];
      if (slot.validation_epoch == validation_epoch_) return false;
      slot.validation_epoch = validation_epoch_;
    }
    for (PeerOwnedResponse& response : responses) {
      const std::size_t slot_index = find_slot_locked(
        response.header.request_id);
      Slot& slot = slots_[slot_index];
      slot.response = PeerResponseDescriptor{
        .peer_id = response.peer_id,
        .receive_slot = std::numeric_limits<u32>::max(),
        .bytes = response.payload.size(),
        .header = response.header,
        .owned_payload = true,
      };
      slot.receive_descriptor_held = false;
      slot.owned_payload = std::move(response.payload);
      slot.state = State::complete;
      completion_targets.push_back(slot.completion_target);
    }
    return true;
  }

  TryPeerResponse try_take(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count,
      PeerResponseDescriptor& response,
      PeerResponseLease& lease) {
    response = {};
    lease = {};
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t slot_index = find_slot_locked(request_id);
    if (slot_index == npos) return TryPeerResponse::stale;

    Slot& slot = slots_[slot_index];
    if (!metadata_matches(slot, expected_shard, expected_type,
                          expected_item_count)) {
      return TryPeerResponse::stale;
    }
    if (slot.state == State::pending || slot.state == State::retryable ||
        slot.state == State::consuming) {
      return TryPeerResponse::pending;
    }
    if (slot.state != State::complete) return TryPeerResponse::stale;

    response = slot.response;
    lease = PeerResponseLease{slot_index, slot.generation};
    slot.state = State::consuming;
    const bool success = response.header.status == static_cast<u32>(
      service::storage_owner::InsertStatus::ok);
    return success ? TryPeerResponse::success : TryPeerResponse::failure;
  }

  bool take_owned_payload(PeerResponseLease lease,
                          std::vector<byte_t>& payload) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_lease_locked(lease, State::consuming)) return false;
    Slot& slot = slots_[lease.slot];
    if (!slot.response.owned_payload || slot.receive_descriptor_held ||
        slot.owned_payload.size() != slot.response.bytes) {
      return false;
    }
    payload = std::move(slot.owned_payload);
    return true;
  }

  // The consumer calls this immediately after copying the payload and
  // reposting the receive WR. Semantic parsing may continue under the lease,
  // but shutdown must no longer treat this descriptor as registry-owned.
  bool mark_receive_reposted(PeerResponseLease lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_lease_locked(lease, State::consuming)) return false;
    Slot& slot = slots_[lease.slot];
    if (!slot.receive_descriptor_held) return false;
    slot.receive_descriptor_held = false;
    return true;
  }

  // Commit parsing/copying of a leased response. Deletion is generation
  // checked, removes the hash bucket without a tombstone, and returns the slab
  // entry to the fixed free list.
  bool ack_consumed(PeerResponseLease lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_lease_locked(lease, State::consuming) ||
        slots_[lease.slot].receive_descriptor_held) {
      return false;
    }
    release_slot_locked(lease.slot);
    return true;
  }

  // Reject a structurally delivered response after operation-specific
  // parsing. Incrementing the generation immediately invalidates all delayed
  // actions associated with the rejected descriptor.
  bool retry(PeerResponseLease lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_lease_locked(lease, State::consuming) ||
        slots_[lease.slot].receive_descriptor_held) {
      return false;
    }
    Slot& slot = slots_[lease.slot];
    slot.response = {};
    std::vector<byte_t>{}.swap(slot.owned_payload);
    slot.state = State::retryable;
    advance_generation(slot);
    return true;
  }

  // Consume a valid transient response while continuing to wait for another
  // response from work that may already be in flight under the same request
  // ID and metadata. Unlike retry(), this transition becomes pending
  // immediately: no subsequent register_send_attempt() is required before a
  // late response can be delivered. The generation change invalidates the
  // consumed descriptor's lease without changing the request identity.
  bool await_late_delivery(PeerResponseLease lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_lease_locked(lease, State::consuming) ||
        slots_[lease.slot].receive_descriptor_held) {
      return false;
    }
    Slot& slot = slots_[lease.slot];
    slot.response = {};
    std::vector<byte_t>{}.swap(slot.owned_payload);
    slot.state = State::pending;
    advance_generation(slot);
    return true;
  }

  // Cancelling a completed request returns its held receive descriptor so the
  // caller can repost it. A consuming entry belongs to its generation-checked
  // lease and therefore cannot be cancelled by key behind that consumer.
  std::optional<PeerResponseDescriptor> cancel(u64 request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t slot_index = find_slot_locked(request_id);
    if (slot_index == npos) return std::nullopt;
    Slot& slot = slots_[slot_index];
    if (slot.state == State::consuming) return std::nullopt;

    std::optional<PeerResponseDescriptor> response;
    if (slot.state == State::complete && slot.receive_descriptor_held) {
      response = slot.response;
    }
    release_slot_locked(slot_index);
    return response;
  }

  std::vector<PeerResponseDescriptor> drain_completed() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<PeerResponseDescriptor> responses;
    responses.reserve(size_);
    for (size_t index = 0; index < capacity_; ++index) {
      const Slot& slot = slots_[index];
      if ((slot.state == State::complete || slot.state == State::consuming) &&
          slot.receive_descriptor_held) {
        responses.push_back(slot.response);
      }
    }
    for (Bucket& bucket : buckets_) bucket = {};
    for (Slot& slot : slots_) {
      advance_generation(slot);
      const u64 generation = slot.generation;
      slot = Slot{};
      slot.generation = generation;
    }
    size_ = 0;
    initialize_free_list();
    return responses;
  }

private:
  enum class State : std::uint8_t {
    free,
    pending,
    complete,
    consuming,
    retryable,
  };

  struct Slot {
    u64 request_id{};
    u32 expected_shard{};
    service::storage_owner::PeerRpcType expected_type{
      service::storage_owner::PeerRpcType::reverse_update_response};
    u32 expected_item_count{};
    PeerResponseCompletionTarget completion_target{};
    PeerResponseDescriptor response{};
    std::vector<byte_t> owned_payload;
    bool receive_descriptor_held{};
    u64 generation{};
    u64 validation_epoch{};
    size_t next_free{npos};
    State state{State::free};
  };

  struct Bucket {
    u64 request_id{};
    size_t slot{npos};
    bool occupied{};
  };

  static constexpr size_t npos = std::numeric_limits<size_t>::max();

  PeerResponseRegistration register_request_locked(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count,
      PeerResponseCompletionTarget completion_target) {
    if (request_id == 0 ||
        !valid_completion_target(completion_target)) {
      return PeerResponseRegistration::conflict;
    }

    const size_t existing = find_slot_locked(request_id);
    if (existing != npos) {
      Slot& slot = slots_[existing];
      if (!metadata_matches(slot, expected_shard, expected_type,
                            expected_item_count) ||
          slot.completion_target != completion_target) {
        return PeerResponseRegistration::conflict;
      }
      if (slot.state == State::pending) {
        return PeerResponseRegistration::retry;
      }
      if (slot.state == State::complete || slot.state == State::consuming) {
        return PeerResponseRegistration::already_complete;
      }
      if (slot.state == State::retryable) {
        slot.state = State::pending;
        return PeerResponseRegistration::retry;
      }
      return PeerResponseRegistration::conflict;
    }

    if (free_head_ == npos) return PeerResponseRegistration::full;
    const size_t slot_index = allocate_slot_locked();
    Slot& slot = slots_[slot_index];
    slot.request_id = request_id;
    slot.expected_shard = expected_shard;
    slot.expected_type = expected_type;
    slot.expected_item_count = expected_item_count;
    slot.completion_target = completion_target;
    slot.response = {};
    slot.owned_payload.clear();
    slot.receive_descriptor_held = false;
    slot.state = State::pending;
    insert_bucket_locked(request_id, slot_index);
    ++size_;
    return PeerResponseRegistration::registered;
  }

  [[nodiscard]] size_t find_slot_locked(u64 request_id) {
    size_t index = hash_request_id(request_id) & bucket_mask_;
    size_t probes = 0;
    for (;;) {
      ++probes;
      const Bucket& bucket = buckets_[index];
      if (!bucket.occupied) {
        record_probe_locked(probes);
        return npos;
      }
      if (bucket.request_id == request_id) {
        record_probe_locked(probes);
        return bucket.slot;
      }
      index = (index + 1) & bucket_mask_;
    }
  }

  void insert_bucket_locked(u64 request_id, size_t slot_index) {
    size_t index = hash_request_id(request_id) & bucket_mask_;
    size_t probes = 0;
    for (;;) {
      ++probes;
      Bucket& bucket = buckets_[index];
      if (!bucket.occupied) {
        bucket = Bucket{request_id, slot_index, true};
        record_probe_locked(probes);
        return;
      }
      index = (index + 1) & bucket_mask_;
    }
  }

  void erase_bucket_locked(u64 request_id) {
    size_t hole = hash_request_id(request_id) & bucket_mask_;
    while (buckets_[hole].occupied &&
           buckets_[hole].request_id != request_id) {
      hole = (hole + 1) & bucket_mask_;
    }
    if (!buckets_[hole].occupied) return;

    size_t scan = (hole + 1) & bucket_mask_;
    while (buckets_[scan].occupied) {
      const size_t home = hash_request_id(buckets_[scan].request_id) &
                          bucket_mask_;
      const size_t scan_distance = (scan - home) & bucket_mask_;
      const size_t hole_distance = (hole - home) & bucket_mask_;
      if (scan_distance > hole_distance) {
        buckets_[hole] = buckets_[scan];
        hole = scan;
      }
      scan = (scan + 1) & bucket_mask_;
    }
    buckets_[hole] = {};
  }

  [[nodiscard]] bool valid_lease_locked(PeerResponseLease lease,
                                         State state) const {
    return lease.valid() && lease.slot < capacity_ &&
           slots_[lease.slot].generation == lease.generation &&
           slots_[lease.slot].state == state;
  }

  static bool metadata_matches(
      const Slot& slot,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count) {
    return slot.expected_shard == expected_shard &&
           slot.expected_type == expected_type &&
           slot.expected_item_count == expected_item_count;
  }

  static bool valid_completion_target(
      const PeerResponseCompletionTarget& target) {
    const auto& owner = target.context_owner;
    const bool empty_owner = owner.runtime_epoch == 0 &&
      owner.worker_id == 0 && owner.slot == 0 && owner.token == 0;
    const bool valid_owner = owner.runtime_epoch != 0 && owner.token != 0;
    if (!empty_owner && !valid_owner) return false;
    return !valid_owner ||
      target.maintenance_wake_owner == kNoMaintenanceWakeOwner;
  }

  static bool delivery_matches(
      const Slot& slot,
      u32 peer_id,
      const service::storage_owner::PeerRpcHeader& header) {
    return slot.state == State::pending &&
      metadata_matches(
        slot, peer_id,
        static_cast<service::storage_owner::PeerRpcType>(header.type),
        header.item_count) &&
      header.source_shard == peer_id;
  }

  static bool valid_owned_response(const PeerOwnedResponse& response) {
    using namespace service::storage_owner;
    if (response.peer_id == std::numeric_limits<u32>::max() ||
        response.payload.size() < sizeof(PeerRpcHeader)) {
      return false;
    }
    PeerRpcHeader embedded{};
    std::memcpy(&embedded, response.payload.data(), sizeof(embedded));
    return embedded.magic == kPeerRpcMagic &&
      embedded.version == kPeerRpcVersion &&
      embedded.type == response.header.type &&
      embedded.source_shard == response.header.source_shard &&
      embedded.item_count == response.header.item_count &&
      embedded.request_id == response.header.request_id &&
      embedded.status == response.header.status &&
      embedded.reserved == response.header.reserved &&
      embedded.source_shard == response.peer_id;
  }

  size_t allocate_slot_locked() {
    const size_t slot_index = free_head_;
    Slot& slot = slots_[slot_index];
    free_head_ = slot.next_free;
    slot.next_free = npos;
    advance_generation(slot);
    return slot_index;
  }

  void release_slot_locked(size_t slot_index) {
    Slot& slot = slots_[slot_index];
    erase_bucket_locked(slot.request_id);
    const u64 generation = slot.generation;
    slot = Slot{};
    slot.generation = generation;
    slot.next_free = free_head_;
    free_head_ = slot_index;
    --size_;
  }

  void initialize_free_list() {
    free_head_ = capacity_ == 0 ? npos : 0;
    for (size_t index = 0; index < capacity_; ++index) {
      slots_[index].next_free = index + 1 < capacity_ ? index + 1 : npos;
      slots_[index].state = State::free;
    }
  }

  static void advance_generation(Slot& slot) {
    ++slot.generation;
    if (slot.generation == 0) ++slot.generation;
  }

  void record_probe_locked(size_t probes) {
    ++probe_telemetry_.lookups;
    probe_telemetry_.probes += probes;
    probe_telemetry_.max_probe = std::max(
      probe_telemetry_.max_probe, probes);
  }

  static size_t normalize_capacity(size_t requested) {
    requested = std::max<size_t>(2, requested);
    if (requested > (size_t{1} << 61)) {
      throw std::invalid_argument("peer response registry capacity is too large");
    }
    return std::bit_ceil(requested);
  }

  static size_t hash_request_id(u64 value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<size_t>(value);
  }

  const size_t capacity_;
  const size_t bucket_capacity_;
  const size_t bucket_mask_;
  mutable std::mutex mutex_;
  std::vector<Slot> slots_;
  std::vector<Bucket> buckets_;
  size_t free_head_{npos};
  size_t size_{};
  PeerHashProbeTelemetry probe_telemetry_{};
  u64 validation_epoch_{};
};

// Bounded receiver-side de-duplication for retries that reuse a request ID.
// Successful fixed-header responses remain in a completed FIFO. Eviction is
// O(1), and the stable slab index lets the FIFO remain intrusive while hash
// buckets move during backward-shift deletion. Inflight work is never evicted.
//
// Bounded replay is valid only for RPCs whose successful response denotes an
// idempotent postcondition. The current replayable callers are reverse-edge,
// deletion-edge, and centroid-membership operations; each carries generation
// or membership semantics that make an already-applied success replay-safe.
class PeerRequestDeduplicator {
public:
  explicit PeerRequestDeduplicator(size_t requested_capacity)
      : capacity_(normalize_capacity(requested_capacity)),
        bucket_capacity_(capacity_ * 2),
        bucket_mask_(bucket_capacity_ - 1),
        slots_(capacity_),
        buckets_(bucket_capacity_) {
    initialize_free_list();
  }

  [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] size_t bucket_capacity() const noexcept {
    return bucket_capacity_;
  }

  [[nodiscard]] size_t size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_;
  }

  [[nodiscard]] PeerHashProbeTelemetry probe_telemetry() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return probe_telemetry_;
  }

  PeerRequestDecision begin(
      u32 source_shard,
      const service::storage_owner::PeerRpcHeader& request,
      bool response_replayable) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (request.request_id == 0) {
      return {.action = PeerRequestAction::conflict};
    }

    size_t slot_index = find_slot_locked(source_shard, request.request_id);
    if (slot_index != npos) {
      Slot& slot = slots_[slot_index];
      if (!metadata_matches(slot, source_shard, request)) {
        return {.action = PeerRequestAction::conflict};
      }
      if (slot.state == State::inflight) {
        return {.action = PeerRequestAction::duplicate_inflight};
      }
      if (slot.state == State::complete && response_replayable) {
        return {
          .action = PeerRequestAction::replay,
          .response = slot.response,
        };
      }
      if (slot.state == State::complete) {
        remove_completed_locked(slot_index);
        slot.response = {};
        slot.state = State::inflight;
        advance_generation(slot);
        return {
          .action = PeerRequestAction::execute,
          .lease = PeerRequestLease{slot_index, slot.generation},
        };
      }
      return {.action = PeerRequestAction::conflict};
    }

    if (free_head_ == npos) evict_oldest_complete_locked();
    if (free_head_ == npos) return {.action = PeerRequestAction::full};

    slot_index = allocate_slot_locked();
    Slot& slot = slots_[slot_index];
    slot.source_shard = source_shard;
    slot.request_id = request.request_id;
    slot.type = request.type;
    slot.item_count = request.item_count;
    slot.reserved = request.reserved;
    slot.response = {};
    slot.state = State::inflight;
    insert_bucket_locked(source_shard, request.request_id, slot_index);
    ++size_;
    return {
      .action = PeerRequestAction::execute,
      .lease = PeerRequestLease{slot_index, slot.generation},
    };
  }

  bool complete(PeerRequestLease lease,
                u32 source_shard,
                const service::storage_owner::PeerRpcHeader& request,
                const service::storage_owner::PeerRpcHeader& response) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_inflight_lease_locked(lease) ||
        !metadata_matches(slots_[lease.slot], source_shard, request)) {
      return false;
    }

    // Failed/overloaded operations are retryable with the same wire ID and
    // therefore leave no cached entry. Successful operations enter the
    // bounded completed FIFO so a lost ACK never causes a second apply.
    if (response.status != static_cast<u32>(
          service::storage_owner::InsertStatus::ok)) {
      release_slot_locked(lease.slot);
      return true;
    }
    Slot& slot = slots_[lease.slot];
    slot.response = response;
    slot.state = State::complete;
    append_completed_locked(lease.slot);
    return true;
  }

  bool abandon(PeerRequestLease lease,
               u32 source_shard,
               const service::storage_owner::PeerRpcHeader& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!valid_inflight_lease_locked(lease) ||
        !metadata_matches(slots_[lease.slot], source_shard, request)) {
      return false;
    }
    release_slot_locked(lease.slot);
    return true;
  }

private:
  enum class State : std::uint8_t {
    free,
    inflight,
    complete,
  };

  struct Slot {
    u32 source_shard{};
    u64 request_id{};
    u32 type{};
    u32 item_count{};
    u32 reserved{};
    service::storage_owner::PeerRpcHeader response{};
    u64 generation{};
    size_t next_free{npos};
    size_t completed_prev{npos};
    size_t completed_next{npos};
    State state{State::free};
  };

  struct Bucket {
    u32 source_shard{};
    u64 request_id{};
    size_t slot{npos};
    bool occupied{};
  };

  static constexpr size_t npos = std::numeric_limits<size_t>::max();

  size_t find_slot_locked(u32 source_shard, u64 request_id) {
    size_t index = hash_key(source_shard, request_id) & bucket_mask_;
    size_t probes = 0;
    for (;;) {
      ++probes;
      const Bucket& bucket = buckets_[index];
      if (!bucket.occupied) {
        record_probe_locked(probes);
        return npos;
      }
      if (bucket.source_shard == source_shard &&
          bucket.request_id == request_id) {
        record_probe_locked(probes);
        return bucket.slot;
      }
      index = (index + 1) & bucket_mask_;
    }
  }

  void insert_bucket_locked(u32 source_shard,
                            u64 request_id,
                            size_t slot_index) {
    size_t index = hash_key(source_shard, request_id) & bucket_mask_;
    size_t probes = 0;
    for (;;) {
      ++probes;
      Bucket& bucket = buckets_[index];
      if (!bucket.occupied) {
        bucket = Bucket{source_shard, request_id, slot_index, true};
        record_probe_locked(probes);
        return;
      }
      index = (index + 1) & bucket_mask_;
    }
  }

  void erase_bucket_locked(u32 source_shard, u64 request_id) {
    size_t hole = hash_key(source_shard, request_id) & bucket_mask_;
    while (buckets_[hole].occupied &&
           (buckets_[hole].source_shard != source_shard ||
            buckets_[hole].request_id != request_id)) {
      hole = (hole + 1) & bucket_mask_;
    }
    if (!buckets_[hole].occupied) return;

    size_t scan = (hole + 1) & bucket_mask_;
    while (buckets_[scan].occupied) {
      const size_t home = hash_key(buckets_[scan].source_shard,
                                   buckets_[scan].request_id) & bucket_mask_;
      const size_t scan_distance = (scan - home) & bucket_mask_;
      const size_t hole_distance = (hole - home) & bucket_mask_;
      if (scan_distance > hole_distance) {
        buckets_[hole] = buckets_[scan];
        hole = scan;
      }
      scan = (scan + 1) & bucket_mask_;
    }
    buckets_[hole] = {};
  }

  static bool metadata_matches(
      const Slot& slot,
      u32 source_shard,
      const service::storage_owner::PeerRpcHeader& request) {
    return slot.source_shard == source_shard &&
           slot.request_id == request.request_id &&
           slot.type == request.type &&
           slot.item_count == request.item_count &&
           slot.reserved == request.reserved;
  }

  [[nodiscard]] bool valid_inflight_lease_locked(
      PeerRequestLease lease) const {
    return lease.valid() && lease.slot < capacity_ &&
           slots_[lease.slot].generation == lease.generation &&
           slots_[lease.slot].state == State::inflight;
  }

  size_t allocate_slot_locked() {
    const size_t slot_index = free_head_;
    Slot& slot = slots_[slot_index];
    free_head_ = slot.next_free;
    slot.next_free = npos;
    advance_generation(slot);
    return slot_index;
  }

  void release_slot_locked(size_t slot_index) {
    Slot& slot = slots_[slot_index];
    if (slot.state == State::complete) remove_completed_locked(slot_index);
    erase_bucket_locked(slot.source_shard, slot.request_id);
    const u64 generation = slot.generation;
    slot = Slot{};
    slot.generation = generation;
    slot.next_free = free_head_;
    free_head_ = slot_index;
    --size_;
  }

  void append_completed_locked(size_t slot_index) {
    Slot& slot = slots_[slot_index];
    slot.completed_prev = completed_tail_;
    slot.completed_next = npos;
    if (completed_tail_ == npos) {
      completed_head_ = slot_index;
    } else {
      slots_[completed_tail_].completed_next = slot_index;
    }
    completed_tail_ = slot_index;
  }

  void remove_completed_locked(size_t slot_index) {
    Slot& slot = slots_[slot_index];
    if (slot.completed_prev == npos) {
      completed_head_ = slot.completed_next;
    } else {
      slots_[slot.completed_prev].completed_next = slot.completed_next;
    }
    if (slot.completed_next == npos) {
      completed_tail_ = slot.completed_prev;
    } else {
      slots_[slot.completed_next].completed_prev = slot.completed_prev;
    }
    slot.completed_prev = npos;
    slot.completed_next = npos;
  }

  void evict_oldest_complete_locked() {
    if (completed_head_ != npos) release_slot_locked(completed_head_);
  }

  void initialize_free_list() {
    free_head_ = capacity_ == 0 ? npos : 0;
    for (size_t index = 0; index < capacity_; ++index) {
      slots_[index].next_free = index + 1 < capacity_ ? index + 1 : npos;
      slots_[index].state = State::free;
    }
  }

  static void advance_generation(Slot& slot) {
    ++slot.generation;
    if (slot.generation == 0) ++slot.generation;
  }

  void record_probe_locked(size_t probes) {
    ++probe_telemetry_.lookups;
    probe_telemetry_.probes += probes;
    probe_telemetry_.max_probe = std::max(
      probe_telemetry_.max_probe, probes);
  }

  static size_t normalize_capacity(size_t requested) {
    requested = std::max<size_t>(2, requested);
    if (requested > (size_t{1} << 61)) {
      throw std::invalid_argument("peer request dedup capacity is too large");
    }
    return std::bit_ceil(requested);
  }

  static size_t hash_key(u32 source_shard, u64 request_id) {
    u64 value = request_id ^ (static_cast<u64>(source_shard) << 32);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<size_t>(value);
  }

  const size_t capacity_;
  const size_t bucket_capacity_;
  const size_t bucket_mask_;
  mutable std::mutex mutex_;
  std::vector<Slot> slots_;
  std::vector<Bucket> buckets_;
  size_t free_head_{npos};
  size_t completed_head_{npos};
  size_t completed_tail_{npos};
  size_t size_{};
  PeerHashProbeTelemetry probe_telemetry_{};
};

}  // namespace memory_node_detail
