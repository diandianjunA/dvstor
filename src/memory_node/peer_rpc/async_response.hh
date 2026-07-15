#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "common/types.hh"
#include "service/storage_owner_protocol.hh"

namespace memory_node_detail {

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
  retired,
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

struct PeerRequestDecision {
  PeerRequestAction action{PeerRequestAction::conflict};
  service::storage_owner::PeerRpcHeader response{};
};

struct PeerResponseDescriptor {
  u32 peer_id{};
  u32 receive_slot{};
  size_t bytes{};
  service::storage_owner::PeerRpcHeader header{};
};

// Fixed-capacity request/response correlation for stage2 peer RPCs. Payloads
// remain in their registered RDMA receive slots until the stage2 executor
// consumes them, so CQ progress performs neither allocation nor payload copy.
// All methods are short critical sections; callers perform copies and reposts
// after the registry lock has been released.
class PeerAsyncResponseRegistry {
public:
  explicit PeerAsyncResponseRegistry(size_t requested_capacity)
      : capacity_(normalize_capacity(requested_capacity)),
        mask_(capacity_ - 1),
        slots_(capacity_) {}

  PeerAsyncResponseRegistry(const PeerAsyncResponseRegistry&) = delete;
  PeerAsyncResponseRegistry& operator=(const PeerAsyncResponseRegistry&) = delete;

  [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

  [[nodiscard]] size_t size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_;
  }

  PeerResponseRegistration register_request(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    return register_request_locked(request_id, expected_shard, expected_type,
                                   expected_item_count, false);
  }

  // Register an actual send attempt owned by an already-live logical stage2
  // request. Unlike register_request(), this may atomically revive a matching
  // retired tombstone after a consumed response was rejected by the payload
  // parser. If that tombstone has meanwhile been reused, the same call either
  // installs the missing ID or reports bounded-capacity pressure for retry.
  PeerResponseRegistration register_send_attempt(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    return register_request_locked(request_id, expected_shard, expected_type,
                                   expected_item_count, true);
  }

  // Returns true only when ownership of the receive descriptor transfers to
  // the registry. Unknown, stale, malformed, and duplicate responses remain
  // owned by the caller and must be reposted immediately.
  bool try_deliver(u32 peer_id,
                   u32 receive_slot,
                   size_t bytes,
                   const service::storage_owner::PeerRpcHeader& header) {
    std::lock_guard<std::mutex> lock(mutex_);
    const Lookup lookup = find_locked(header.request_id);
    if (lookup.found == npos) return false;

    Slot& slot = slots_[lookup.found];
    if (slot.state != State::pending ||
        !metadata_matches(slot,
                          peer_id,
                          static_cast<service::storage_owner::PeerRpcType>(header.type),
                          header.item_count) ||
        header.source_shard != peer_id) {
      return false;
    }

    slot.response = PeerResponseDescriptor{
      .peer_id = peer_id,
      .receive_slot = receive_slot,
      .bytes = bytes,
      .header = header,
    };
    slot.state = State::complete;
    return true;
  }

  TryPeerResponse try_take(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count,
      PeerResponseDescriptor& response) {
    std::lock_guard<std::mutex> lock(mutex_);
    const Lookup lookup = find_locked(request_id);
    if (lookup.found == npos) return TryPeerResponse::stale;

    Slot& slot = slots_[lookup.found];
    if (!metadata_matches(slot, expected_shard, expected_type,
                          expected_item_count) ||
        slot.state == State::retired) {
      return TryPeerResponse::stale;
    }
    // retryable means the logical request is still live but is waiting for
    // its identical-ID resend deadline.  Reporting it as stale makes the
    // stage2 poller cancel the entry before it can call register_request()
    // again, permanently poisoning every explicit failure/malformed retry.
    if (slot.state == State::pending || slot.state == State::retryable) {
      return TryPeerResponse::pending;
    }
    if (slot.state != State::complete) return TryPeerResponse::stale;

    response = slot.response;
    const bool success = response.header.status == static_cast<u32>(
      service::storage_owner::InsertStatus::ok);
    if (success) {
      retire_locked(slot);
    } else {
      slot.response = {};
      slot.state = State::retryable;
    }
    return success ? TryPeerResponse::success : TryPeerResponse::failure;
  }

  // Cancelling a completed request returns its held receive descriptor so the
  // caller can repost the slot. Pending requests have no receive ownership.
  std::optional<PeerResponseDescriptor> cancel(u64 request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const Lookup lookup = find_locked(request_id);
    if (lookup.found == npos) return std::nullopt;

    Slot& slot = slots_[lookup.found];
    std::optional<PeerResponseDescriptor> response;
    if (slot.state == State::complete) response = slot.response;
    if (slot.state == State::pending || slot.state == State::complete) {
      retire_locked(slot);
    } else if (slot.state == State::retryable) {
      slot.state = State::retired;
      --size_;
    }
    return response;
  }

  // Rearm a response that was structurally valid at CQ ingress but rejected
  // by the stage2 payload parser (for example an out-of-range candidate count).
  // Exact metadata prevents an old context from reviving a reused ID.
  bool mark_retryable(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    const Lookup lookup = find_locked(request_id);
    if (lookup.found == npos) return false;
    Slot& slot = slots_[lookup.found];
    if (!metadata_matches(slot, expected_shard, expected_type,
                          expected_item_count)) {
      return false;
    }
    if (slot.state == State::retryable || slot.state == State::pending) {
      return true;
    }
    if (slot.state != State::retired || size_ == capacity_) return false;
    slot.response = {};
    slot.state = State::retryable;
    ++size_;
    return true;
  }

  std::vector<PeerResponseDescriptor> drain_completed() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<PeerResponseDescriptor> responses;
    responses.reserve(size_);
    for (Slot& slot : slots_) {
      if (slot.state == State::complete) responses.push_back(slot.response);
      if (slot.state == State::pending || slot.state == State::complete ||
          slot.state == State::retryable) {
        retire_locked(slot);
      }
    }
    return responses;
  }

private:
  enum class State : std::uint8_t {
    empty,
    pending,
    complete,
    retryable,
    retired,
  };

  struct Slot {
    u64 request_id{};
    u32 expected_shard{};
    service::storage_owner::PeerRpcType expected_type{
      service::storage_owner::PeerRpcType::reverse_update_response};
    u32 expected_item_count{};
    PeerResponseDescriptor response{};
    State state{State::empty};
  };

  struct Lookup {
    size_t found{static_cast<size_t>(-1)};
    size_t insertion{static_cast<size_t>(-1)};
  };

  static constexpr size_t npos = static_cast<size_t>(-1);

  PeerResponseRegistration register_request_locked(
      u64 request_id,
      u32 expected_shard,
      service::storage_owner::PeerRpcType expected_type,
      u32 expected_item_count,
      bool revive_retired) {
    if (request_id == 0) return PeerResponseRegistration::conflict;

    const Lookup lookup = find_locked(request_id);
    if (lookup.found != npos) {
      Slot& slot = slots_[lookup.found];
      if (!metadata_matches(slot, expected_shard, expected_type,
                            expected_item_count)) {
        return PeerResponseRegistration::conflict;
      }
      if (slot.state == State::pending) {
        return PeerResponseRegistration::retry;
      }
      if (slot.state == State::complete) {
        return PeerResponseRegistration::already_complete;
      }
      if (slot.state == State::retryable) {
        slot.response = {};
        slot.state = State::pending;
        return PeerResponseRegistration::retry;
      }
      if (!revive_retired) return PeerResponseRegistration::retired;
      slot.response = {};
      slot.state = State::pending;
      ++size_;
      return PeerResponseRegistration::retry;
    }

    if (size_ == capacity_ || lookup.insertion == npos) {
      return PeerResponseRegistration::full;
    }

    Slot& slot = slots_[lookup.insertion];
    slot.request_id = request_id;
    slot.expected_shard = expected_shard;
    slot.expected_type = expected_type;
    slot.expected_item_count = expected_item_count;
    slot.response = {};
    slot.state = State::pending;
    ++size_;
    return PeerResponseRegistration::registered;
  }

  [[nodiscard]] Lookup find_locked(u64 request_id) const {
    Lookup result;
    size_t first_retired = npos;
    size_t index = hash_request_id(request_id) & mask_;
    for (size_t probe = 0; probe < capacity_; ++probe) {
      const Slot& slot = slots_[index];
      if (slot.state == State::empty) {
        result.insertion = first_retired == npos ? index : first_retired;
        return result;
      }
      if (slot.request_id == request_id) {
        result.found = index;
        return result;
      }
      if (slot.state == State::retired && first_retired == npos) {
        first_retired = index;
      }
      index = (index + 1) & mask_;
    }
    result.insertion = first_retired;
    return result;
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

  void retire_locked(Slot& slot) {
    slot.response = {};
    slot.state = State::retired;
    --size_;
  }

  static size_t normalize_capacity(size_t requested) {
    requested = std::max<size_t>(2, requested);
    if (requested > (size_t{1} << 62)) {
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
  const size_t mask_;
  mutable std::mutex mutex_;
  std::vector<Slot> slots_;
  size_t size_{};
};

// Bounded receiver-side de-duplication for retries that reuse a request ID.
// Reverse/cleanup completions replay their cached fixed-size response. Stitch
// search completions are read-only and may be recomputed after completion;
// concurrent duplicates are still coalesced while the first search runs.
class PeerRequestDeduplicator {
public:
  explicit PeerRequestDeduplicator(size_t requested_capacity)
      : capacity_(normalize_capacity(requested_capacity)),
        mask_(capacity_ - 1),
        slots_(capacity_) {}

  PeerRequestDecision begin(
      u32 source_shard,
      const service::storage_owner::PeerRpcHeader& request,
      bool response_replayable) {
    std::lock_guard<std::mutex> lock(mutex_);
    Lookup lookup = find_locked(source_shard, request.request_id);
    if (lookup.found != npos) {
      Slot& slot = slots_[lookup.found];
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
        slot.state = State::inflight;
        slot.last_used = ++clock_;
        return {.action = PeerRequestAction::execute};
      }
      if (slot.state == State::retired) {
        slot.response = {};
        slot.last_used = ++clock_;
        slot.state = State::inflight;
        ++size_;
        return {.action = PeerRequestAction::execute};
      }
    }

    if (lookup.insertion == npos || size_ == capacity_) {
      evict_oldest_complete_locked();
      lookup = find_locked(source_shard, request.request_id);
    }
    if (lookup.insertion == npos || size_ == capacity_) {
      return {.action = PeerRequestAction::full};
    }

    Slot& slot = slots_[lookup.insertion];
    slot.source_shard = source_shard;
    slot.request_id = request.request_id;
    slot.type = request.type;
    slot.item_count = request.item_count;
    slot.reserved = request.reserved;
    slot.response = {};
    slot.last_used = ++clock_;
    slot.state = State::inflight;
    ++size_;
    return {.action = PeerRequestAction::execute};
  }

  void complete(u32 source_shard,
                const service::storage_owner::PeerRpcHeader& request,
                const service::storage_owner::PeerRpcHeader& response) {
    std::lock_guard<std::mutex> lock(mutex_);
    const Lookup lookup = find_locked(source_shard, request.request_id);
    if (lookup.found == npos) return;
    Slot& slot = slots_[lookup.found];
    if (slot.state != State::inflight ||
        !metadata_matches(slot, source_shard, request)) {
      return;
    }

    // A failed/overloaded operation is retryable with the same ID. Successful
    // operations remain cached so a lost ACK never causes a second apply.
    if (response.status != static_cast<u32>(
          service::storage_owner::InsertStatus::ok)) {
      retire_locked(slot);
      return;
    }
    slot.response = response;
    slot.last_used = ++clock_;
    slot.state = State::complete;
  }

  void abandon(u32 source_shard,
               const service::storage_owner::PeerRpcHeader& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    const Lookup lookup = find_locked(source_shard, request.request_id);
    if (lookup.found == npos) return;
    Slot& slot = slots_[lookup.found];
    if (slot.state == State::inflight &&
        metadata_matches(slot, source_shard, request)) {
      retire_locked(slot);
    }
  }

private:
  enum class State : std::uint8_t {
    empty,
    inflight,
    complete,
    retired,
  };

  struct Slot {
    u32 source_shard{};
    u64 request_id{};
    u32 type{};
    u32 item_count{};
    u32 reserved{};
    service::storage_owner::PeerRpcHeader response{};
    u64 last_used{};
    State state{State::empty};
  };

  struct Lookup {
    size_t found{static_cast<size_t>(-1)};
    size_t insertion{static_cast<size_t>(-1)};
  };

  static constexpr size_t npos = static_cast<size_t>(-1);

  Lookup find_locked(u32 source_shard, u64 request_id) const {
    Lookup result;
    size_t first_retired = npos;
    size_t index = hash_key(source_shard, request_id) & mask_;
    for (size_t probe = 0; probe < capacity_; ++probe) {
      const Slot& slot = slots_[index];
      if (slot.state == State::empty) {
        result.insertion = first_retired == npos ? index : first_retired;
        return result;
      }
      if (slot.source_shard == source_shard &&
          slot.request_id == request_id) {
        result.found = index;
        return result;
      }
      if (slot.state == State::retired && first_retired == npos) {
        first_retired = index;
      }
      index = (index + 1) & mask_;
    }
    result.insertion = first_retired;
    return result;
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

  void evict_oldest_complete_locked() {
    Slot* oldest = nullptr;
    for (Slot& slot : slots_) {
      if (slot.state == State::complete &&
          (oldest == nullptr || slot.last_used < oldest->last_used)) {
        oldest = &slot;
      }
    }
    if (oldest != nullptr) retire_locked(*oldest);
  }

  void retire_locked(Slot& slot) {
    slot.response = {};
    slot.state = State::retired;
    --size_;
  }

  static size_t normalize_capacity(size_t requested) {
    requested = std::max<size_t>(2, requested);
    if (requested > (size_t{1} << 62)) {
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
  const size_t mask_;
  std::mutex mutex_;
  std::vector<Slot> slots_;
  size_t size_{};
  u64 clock_{};
};

}  // namespace memory_node_detail
