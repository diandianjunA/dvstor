#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace memory_node_storage_owner_maintenance_detail {

enum class Stage2Phase : std::uint8_t {
  local_ready,
  remote_search_pending,
  prune_ready,
  reverse_pending,
  finalized,
};

struct Stage2ContextHandle {
  std::uint32_t slot{};
  std::uint32_t generation{};

  bool operator==(const Stage2ContextHandle&) const = default;
};

enum class Stage2EventResult : std::uint8_t {
  accepted,
  phase_advanced,
  ready_to_finalize,
  duplicate,
  stale_context,
  invalid_phase,
  invalid_peer_mask,
  unexpected_peer,
  incomplete,
  unknown_request,
};

enum class Stage2RequestKind : std::uint8_t {
  remote_search,
  reverse_update,
};

struct Stage2ContextSnapshot {
  Stage2Phase phase{Stage2Phase::local_ready};
  std::uint64_t expected_search_mask{};
  std::uint64_t completed_search_mask{};
  std::uint64_t expected_reverse_mask{};
  std::uint64_t completed_reverse_mask{};
};

// Fixed-capacity stage2 state storage. The owner must serialize calls; the
// tracker deliberately performs no allocation after construction and no
// internal locking.
class Stage2StateTracker {
 public:
  Stage2StateTracker(std::size_t capacity, std::uint32_t peer_count)
      : slots_(capacity), peer_count_(peer_count) {
    if (capacity == 0 ||
        capacity > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("stage2 context capacity is out of range");
    }
    if (peer_count == 0 || peer_count > 64) {
      throw std::invalid_argument("stage2 peer count must be in [1, 64]");
    }

    free_slots_.reserve(capacity);
    for (std::size_t index = capacity; index != 0; --index) {
      free_slots_.push_back(static_cast<std::uint32_t>(index - 1));
    }
  }

  [[nodiscard]] std::optional<Stage2ContextHandle> try_acquire() {
    if (free_slots_.empty()) return std::nullopt;

    const std::uint32_t slot_index = free_slots_.back();
    free_slots_.pop_back();
    Slot& slot = slots_[slot_index];
    ++slot.generation;
    if (slot.generation == 0) ++slot.generation;
    slot.in_use = true;
    slot.state = {};
    ++size_;
    return Stage2ContextHandle{slot_index, slot.generation};
  }

  [[nodiscard]] bool release(Stage2ContextHandle handle) {
    Slot* slot = resolve(handle);
    if (slot == nullptr || slot->state.phase != Stage2Phase::finalized) {
      return false;
    }

    slot->in_use = false;
    slot->state = {};
    free_slots_.push_back(handle.slot);
    --size_;
    return true;
  }

  [[nodiscard]] bool is_current(Stage2ContextHandle handle) const {
    return resolve(handle) != nullptr;
  }

  [[nodiscard]] std::optional<Stage2ContextSnapshot> snapshot(
      Stage2ContextHandle handle) const {
    const Slot* slot = resolve(handle);
    if (slot == nullptr) return std::nullopt;
    return slot->state;
  }

  [[nodiscard]] Stage2EventResult begin_remote_search(
      Stage2ContextHandle handle, std::uint64_t expected_peer_mask) {
    Slot* slot = resolve(handle);
    if (slot == nullptr) return Stage2EventResult::stale_context;
    if (!valid_mask(expected_peer_mask)) {
      return Stage2EventResult::invalid_peer_mask;
    }
    if (slot->state.phase != Stage2Phase::local_ready) {
      return Stage2EventResult::invalid_phase;
    }

    slot->state.expected_search_mask = expected_peer_mask;
    slot->state.completed_search_mask = 0;
    slot->state.phase = expected_peer_mask == 0
                          ? Stage2Phase::prune_ready
                          : Stage2Phase::remote_search_pending;
    return Stage2EventResult::phase_advanced;
  }

  [[nodiscard]] Stage2EventResult record_remote_search_response(
      Stage2ContextHandle handle, std::uint32_t peer_index) {
    Slot* slot = resolve(handle);
    if (slot == nullptr) return Stage2EventResult::stale_context;
    if (peer_index >= peer_count_) {
      return Stage2EventResult::unexpected_peer;
    }

    const std::uint64_t bit = std::uint64_t{1} << peer_index;
    if ((slot->state.expected_search_mask & bit) == 0) {
      return Stage2EventResult::unexpected_peer;
    }
    if ((slot->state.completed_search_mask & bit) != 0) {
      return Stage2EventResult::duplicate;
    }
    if (slot->state.phase != Stage2Phase::remote_search_pending) {
      return Stage2EventResult::invalid_phase;
    }

    slot->state.completed_search_mask |= bit;
    if (slot->state.completed_search_mask ==
        slot->state.expected_search_mask) {
      slot->state.phase = Stage2Phase::prune_ready;
      return Stage2EventResult::phase_advanced;
    }
    return Stage2EventResult::accepted;
  }

  [[nodiscard]] Stage2EventResult begin_reverse(
      Stage2ContextHandle handle, std::uint64_t expected_peer_mask) {
    Slot* slot = resolve(handle);
    if (slot == nullptr) return Stage2EventResult::stale_context;
    if (!valid_mask(expected_peer_mask)) {
      return Stage2EventResult::invalid_peer_mask;
    }
    if (slot->state.phase != Stage2Phase::prune_ready) {
      return Stage2EventResult::invalid_phase;
    }

    slot->state.expected_reverse_mask = expected_peer_mask;
    slot->state.completed_reverse_mask = 0;
    slot->state.phase = Stage2Phase::reverse_pending;
    return expected_peer_mask == 0
             ? Stage2EventResult::ready_to_finalize
             : Stage2EventResult::phase_advanced;
  }

  [[nodiscard]] Stage2EventResult record_reverse_ack(
      Stage2ContextHandle handle, std::uint32_t peer_index) {
    Slot* slot = resolve(handle);
    if (slot == nullptr) return Stage2EventResult::stale_context;
    if (peer_index >= peer_count_) {
      return Stage2EventResult::unexpected_peer;
    }

    const std::uint64_t bit = std::uint64_t{1} << peer_index;
    if ((slot->state.expected_reverse_mask & bit) == 0) {
      return Stage2EventResult::unexpected_peer;
    }
    if ((slot->state.completed_reverse_mask & bit) != 0) {
      return Stage2EventResult::duplicate;
    }
    if (slot->state.phase != Stage2Phase::reverse_pending) {
      return Stage2EventResult::invalid_phase;
    }

    slot->state.completed_reverse_mask |= bit;
    return slot->state.completed_reverse_mask ==
               slot->state.expected_reverse_mask
             ? Stage2EventResult::ready_to_finalize
             : Stage2EventResult::accepted;
  }

  [[nodiscard]] Stage2EventResult finalize(Stage2ContextHandle handle) {
    Slot* slot = resolve(handle);
    if (slot == nullptr) return Stage2EventResult::stale_context;
    if (slot->state.phase != Stage2Phase::reverse_pending) {
      return Stage2EventResult::invalid_phase;
    }
    if (slot->state.completed_reverse_mask !=
        slot->state.expected_reverse_mask) {
      return Stage2EventResult::incomplete;
    }

    slot->state.phase = Stage2Phase::finalized;
    return Stage2EventResult::phase_advanced;
  }

  [[nodiscard]] bool awaits(Stage2ContextHandle handle,
                            Stage2RequestKind kind,
                            std::uint32_t peer_index) const {
    const Slot* slot = resolve(handle);
    if (slot == nullptr || peer_index >= peer_count_) return false;
    const std::uint64_t bit = std::uint64_t{1} << peer_index;

    if (kind == Stage2RequestKind::remote_search) {
      return slot->state.phase == Stage2Phase::remote_search_pending &&
             (slot->state.expected_search_mask & bit) != 0 &&
             (slot->state.completed_search_mask & bit) == 0;
    }
    return slot->state.phase == Stage2Phase::reverse_pending &&
           (slot->state.expected_reverse_mask & bit) != 0 &&
           (slot->state.completed_reverse_mask & bit) == 0;
  }

  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] std::size_t capacity() const { return slots_.size(); }
  [[nodiscard]] bool full() const { return size_ == slots_.size(); }
  [[nodiscard]] std::uint32_t peer_count() const { return peer_count_; }

 private:
  struct Slot {
    std::uint32_t generation{};
    bool in_use{};
    Stage2ContextSnapshot state{};
  };

  [[nodiscard]] Slot* resolve(Stage2ContextHandle handle) {
    if (handle.slot >= slots_.size()) return nullptr;
    Slot& slot = slots_[handle.slot];
    return slot.in_use && slot.generation == handle.generation ? &slot
                                                               : nullptr;
  }

  [[nodiscard]] const Slot* resolve(Stage2ContextHandle handle) const {
    if (handle.slot >= slots_.size()) return nullptr;
    const Slot& slot = slots_[handle.slot];
    return slot.in_use && slot.generation == handle.generation ? &slot
                                                               : nullptr;
  }

  [[nodiscard]] bool valid_mask(std::uint64_t mask) const {
    if (peer_count_ == 64) return true;
    const std::uint64_t valid = (std::uint64_t{1} << peer_count_) - 1;
    return (mask & ~valid) == 0;
  }

  std::vector<Slot> slots_;
  std::vector<std::uint32_t> free_slots_;
  std::size_t size_{};
  std::uint32_t peer_count_{};
};

struct Stage2RequestMetadata {
  std::uint64_t request_id{};
  Stage2ContextHandle context{};
  Stage2RequestKind kind{Stage2RequestKind::remote_search};
  std::uint32_t peer_index{};
  std::uint32_t attempt_count{};
  std::uint64_t last_send_time{};
  std::uint64_t deadline{};
  bool response_seen{};
};

enum class Stage2RequestRegisterResult : std::uint8_t {
  registered,
  capacity_exhausted,
  duplicate_request_id,
  stale_context,
  invalid_phase_or_peer,
};

// Fixed-capacity request lookup and retry metadata. Open addressing keeps the
// response path O(1) without allocating per request.
class Stage2RequestTracker {
 public:
  explicit Stage2RequestTracker(std::size_t capacity)
      : records_(capacity), buckets_(hash_capacity(capacity)) {
    if (capacity == 0 ||
        capacity > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("stage2 request capacity is out of range");
    }

    free_records_.reserve(capacity);
    for (std::size_t index = capacity; index != 0; --index) {
      free_records_.push_back(static_cast<std::uint32_t>(index - 1));
    }
  }

  [[nodiscard]] Stage2RequestRegisterResult try_register(
      std::uint64_t request_id, Stage2ContextHandle context,
      Stage2RequestKind kind, std::uint32_t peer_index,
      std::uint64_t sent_at, std::uint64_t deadline,
      const Stage2StateTracker& states) {
    if (!states.is_current(context)) {
      return Stage2RequestRegisterResult::stale_context;
    }
    if (!states.awaits(context, kind, peer_index)) {
      return Stage2RequestRegisterResult::invalid_phase_or_peer;
    }
    if (find_bucket(request_id).has_value()) {
      return Stage2RequestRegisterResult::duplicate_request_id;
    }
    if (free_records_.empty()) {
      return Stage2RequestRegisterResult::capacity_exhausted;
    }

    const std::uint32_t record_index = free_records_.back();
    free_records_.pop_back();
    Record& record = records_[record_index];
    record.in_use = true;
    record.metadata = Stage2RequestMetadata{
      request_id, context, kind, peer_index, 1, sent_at, deadline, false};

    Bucket& bucket = bucket_for_insert(request_id);
    bucket.state = BucketState::occupied;
    bucket.request_id = request_id;
    bucket.record_index = record_index;
    ++size_;
    return Stage2RequestRegisterResult::registered;
  }

  [[nodiscard]] std::optional<Stage2RequestMetadata> find(
      std::uint64_t request_id) const {
    const std::optional<std::size_t> bucket_index = find_bucket(request_id);
    if (!bucket_index.has_value()) return std::nullopt;
    return records_[buckets_[*bucket_index].record_index].metadata;
  }

  [[nodiscard]] bool retry_due(std::uint64_t request_id,
                               std::uint64_t now) const {
    const std::optional<Stage2RequestMetadata> metadata = find(request_id);
    return metadata.has_value() && !metadata->response_seen &&
           now >= metadata->deadline;
  }

  [[nodiscard]] std::optional<Stage2RequestMetadata> mark_retry(
      std::uint64_t request_id, std::uint64_t sent_at,
      std::uint64_t deadline) {
    const std::optional<std::size_t> bucket_index = find_bucket(request_id);
    if (!bucket_index.has_value()) return std::nullopt;
    Stage2RequestMetadata& metadata =
      records_[buckets_[*bucket_index].record_index].metadata;
    if (metadata.response_seen) return std::nullopt;
    ++metadata.attempt_count;
    metadata.last_send_time = sent_at;
    metadata.deadline = deadline;
    return metadata;
  }

  [[nodiscard]] Stage2EventResult record_response(
      std::uint64_t request_id, Stage2StateTracker& states) {
    const std::optional<std::size_t> bucket_index = find_bucket(request_id);
    if (!bucket_index.has_value()) return Stage2EventResult::unknown_request;
    Stage2RequestMetadata& metadata =
      records_[buckets_[*bucket_index].record_index].metadata;

    // Generation validation precedes duplicate detection so a late response
    // cannot be mistaken for a response to a context that reused the slot.
    if (!states.is_current(metadata.context)) {
      return Stage2EventResult::stale_context;
    }
    if (metadata.response_seen) return Stage2EventResult::duplicate;

    const Stage2EventResult result =
      metadata.kind == Stage2RequestKind::remote_search
        ? states.record_remote_search_response(metadata.context,
                                               metadata.peer_index)
        : states.record_reverse_ack(metadata.context, metadata.peer_index);
    if (result == Stage2EventResult::accepted ||
        result == Stage2EventResult::phase_advanced ||
        result == Stage2EventResult::ready_to_finalize ||
        result == Stage2EventResult::duplicate) {
      metadata.response_seen = true;
    }
    return result;
  }

  [[nodiscard]] bool erase(std::uint64_t request_id) {
    const std::optional<std::size_t> bucket_index = find_bucket(request_id);
    if (!bucket_index.has_value()) return false;

    Bucket& bucket = buckets_[*bucket_index];
    Record& record = records_[bucket.record_index];
    record.in_use = false;
    record.metadata = {};
    free_records_.push_back(bucket.record_index);
    bucket.state = BucketState::tombstone;
    --size_;
    return true;
  }

  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] std::size_t capacity() const { return records_.size(); }
  [[nodiscard]] bool full() const { return size_ == records_.size(); }

 private:
  enum class BucketState : std::uint8_t { empty, occupied, tombstone };

  struct Record {
    bool in_use{};
    Stage2RequestMetadata metadata{};
  };

  struct Bucket {
    BucketState state{BucketState::empty};
    std::uint64_t request_id{};
    std::uint32_t record_index{};
  };

  [[nodiscard]] static std::size_t hash_capacity(std::size_t capacity) {
    if (capacity == 0 ||
        capacity > std::numeric_limits<std::uint32_t>::max()) {
      return 1;
    }
    if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::length_error("stage2 request hash capacity overflow");
    }
    const std::size_t target = capacity * 2;
    std::size_t result = 1;
    while (result < target) {
      if (result > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::length_error("stage2 request hash capacity overflow");
      }
      result *= 2;
    }
    return result;
  }

  [[nodiscard]] static std::uint64_t mix(std::uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
  }

  [[nodiscard]] std::optional<std::size_t> find_bucket(
      std::uint64_t request_id) const {
    const std::size_t mask = buckets_.size() - 1;
    std::size_t index = static_cast<std::size_t>(mix(request_id)) & mask;
    for (std::size_t probe = 0; probe < buckets_.size(); ++probe) {
      const Bucket& bucket = buckets_[index];
      if (bucket.state == BucketState::empty) return std::nullopt;
      if (bucket.state == BucketState::occupied &&
          bucket.request_id == request_id) {
        return index;
      }
      index = (index + 1) & mask;
    }
    return std::nullopt;
  }

  [[nodiscard]] Bucket& bucket_for_insert(std::uint64_t request_id) {
    const std::size_t mask = buckets_.size() - 1;
    std::size_t index = static_cast<std::size_t>(mix(request_id)) & mask;
    std::optional<std::size_t> first_tombstone;
    for (std::size_t probe = 0; probe < buckets_.size(); ++probe) {
      Bucket& bucket = buckets_[index];
      if (bucket.state == BucketState::tombstone &&
          !first_tombstone.has_value()) {
        first_tombstone = index;
      } else if (bucket.state == BucketState::empty) {
        return buckets_[first_tombstone.value_or(index)];
      }
      index = (index + 1) & mask;
    }
    if (first_tombstone.has_value()) return buckets_[*first_tombstone];
    throw std::logic_error("stage2 request hash table is unexpectedly full");
  }

  std::vector<Record> records_;
  std::vector<std::uint32_t> free_records_;
  std::vector<Bucket> buckets_;
  std::size_t size_{};
};

}  // namespace memory_node_storage_owner_maintenance_detail
