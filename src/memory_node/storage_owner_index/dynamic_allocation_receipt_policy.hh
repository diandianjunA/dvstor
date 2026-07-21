#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>

#include "service/storage_owner_protocol.hh"

namespace memory_node_storage_owner_index_detail {

namespace protocol = service::storage_owner;

struct DynamicAllocationReceiptKey {
  u32 authority_shard{};
  u32 source_client{};
  u32 item_index{};
  u64 client_batch_id{};

  bool operator==(const DynamicAllocationReceiptKey&) const = default;
};

struct DynamicAllocationReceiptKeyHash {
  size_t operator()(const DynamicAllocationReceiptKey& key) const noexcept {
    size_t value = std::hash<u64>{}(key.client_batch_id);
    value ^= std::hash<u32>{}(key.authority_shard) +
      0x9e3779b9u + (value << 6) + (value >> 2);
    value ^= std::hash<u32>{}(key.source_client) +
      0x9e3779b9u + (value << 6) + (value >> 2);
    value ^= std::hash<u32>{}(key.item_index) +
      0x9e3779b9u + (value << 6) + (value >> 2);
    return value;
  }
};

inline DynamicAllocationReceiptKey dynamic_allocation_receipt_key(
    const protocol::DynamicNodeControlItem& item) {
  return {
    .authority_shard = item.authority_shard,
    .source_client = item.token.source_client,
    .item_index = item.token.item_index,
    .client_batch_id = item.token.client_batch_id,
  };
}

inline bool same_dynamic_allocation(
    const protocol::DynamicNodeControlItem& lhs,
    const protocol::DynamicNodeControlItem& rhs) {
  return lhs.token.source_client == rhs.token.source_client &&
    lhs.token.item_index == rhs.token.item_index &&
    lhs.token.client_batch_id == rhs.token.client_batch_id &&
    lhs.node_raw == rhs.node_raw && lhs.id == rhs.id &&
    lhs.generation == rhs.generation &&
    lhs.authority_shard == rhs.authority_shard;
}

// Allocation receipts are not a timeout cache. They are ownership records
// for reserved physical slots. A receipt remains pinned until the caller
// proves both sides of the handoff terminal: the source incarnation can no
// longer authorize a fresh migration and the destination has either been
// materialized or reclaimed. Consequently an arbitrarily late retry either
// replays the one reserved handle or is rejected before allocating a slot.
class DynamicAllocationReceiptLedger {
public:
  enum class SourceState : u8 {
    live,
    terminal,
    indeterminate,
  };

  enum class BeginState : u8 {
    claimed,
    replay,
    pending,
    stale_source,
    indeterminate_source,
    conflict,
    pressure,
  };

  struct BeginResult {
    BeginState state{BeginState::conflict};
    protocol::DynamicNodeControlResult result{};
  };

  // begin() may consume an identity observation made before a concurrent
  // settlement erased the preceding ready receipt.  A new owner must
  // therefore validate the source again after installing its pending claim
  // and before reserving physical storage.  Keeping this transition in the
  // ledger makes validation and cancellation one atomic receipt operation.
  enum class ClaimValidationState : u8 {
    validated,
    stale_source,
    indeterminate_source,
    conflict,
  };

  enum class SettleState : u8 {
    settled,
    replay,
    pending,
    unsafe,
    conflict,
  };

  void reset(size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.clear();
    records_.reserve(capacity);
    capacity_ = capacity;
  }

  BeginResult begin(const protocol::DynamicNodeControlItem& item,
                    SourceState source_state) {
    std::lock_guard<std::mutex> lock(mutex_);
    const DynamicAllocationReceiptKey key =
      dynamic_allocation_receipt_key(item);
    const auto existing = records_.find(key);
    if (existing != records_.end()) {
      if (!same_dynamic_allocation(existing->second.item, item)) {
        return {.state = BeginState::conflict};
      }
      if (!existing->second.ready) {
        return {.state = BeginState::pending};
      }
      return {
        .state = BeginState::replay,
        .result = existing->second.result,
      };
    }
    if (source_state == SourceState::terminal) {
      return {.state = BeginState::stale_source};
    }
    if (source_state == SourceState::indeterminate) {
      return {.state = BeginState::indeterminate_source};
    }
    if (capacity_ == 0 || records_.size() >= capacity_) {
      return {.state = BeginState::pressure};
    }
    const auto [position, inserted] = records_.emplace(
      key, Record{.item = item});
    (void)position;
    if (!inserted) return {.state = BeginState::conflict};
    return {.state = BeginState::claimed};
  }

  ClaimValidationState validate_claim_source(
      const protocol::DynamicNodeControlItem& item,
      SourceState source_state) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto position = records_.find(
      dynamic_allocation_receipt_key(item));
    if (position == records_.end() || position->second.ready ||
        !same_dynamic_allocation(position->second.item, item)) {
      return ClaimValidationState::conflict;
    }
    if (source_state == SourceState::live) {
      position->second.validated = true;
      return ClaimValidationState::validated;
    }

    // No physical slot has been reserved while a receipt is pending and
    // unvalidated, so erasing it is both necessary and sufficient to make a
    // later retry safe.  In particular, an indeterminate read is never
    // promoted to live merely to preserve throughput.
    records_.erase(position);
    return source_state == SourceState::terminal
      ? ClaimValidationState::stale_source
      : ClaimValidationState::indeterminate_source;
  }

  bool publish(const protocol::DynamicNodeControlItem& item,
               const protocol::DynamicNodeControlResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto position = records_.find(
      dynamic_allocation_receipt_key(item));
    if (position == records_.end() || position->second.ready ||
        !position->second.validated ||
        !same_dynamic_allocation(position->second.item, item)) {
      return false;
    }
    position->second.result = result;
    position->second.ready = true;
    return true;
  }

  bool cancel_claim(const protocol::DynamicNodeControlItem& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto position = records_.find(
      dynamic_allocation_receipt_key(item));
    if (position == records_.end() || position->second.ready ||
        !same_dynamic_allocation(position->second.item, item)) {
      return false;
    }
    records_.erase(position);
    return true;
  }

  SettleState settle(const protocol::DynamicNodeControlItem& item,
                     bool source_is_terminal,
                     bool destination_is_terminal) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto position = records_.find(
      dynamic_allocation_receipt_key(item));
    if (position == records_.end()) {
      return source_is_terminal && destination_is_terminal
        ? SettleState::replay : SettleState::unsafe;
    }
    if (!same_dynamic_allocation(position->second.item, item) ||
        (position->second.ready &&
         position->second.result.node_raw != item.allocated_raw)) {
      return SettleState::conflict;
    }
    if (!position->second.ready) return SettleState::pending;
    if (!source_is_terminal || !destination_is_terminal) {
      return SettleState::unsafe;
    }
    records_.erase(position);
    return SettleState::settled;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_.size();
  }

private:
  struct Record {
    protocol::DynamicNodeControlItem item{};
    protocol::DynamicNodeControlResult result{};
    bool validated{};
    bool ready{};
  };

  mutable std::mutex mutex_;
  std::unordered_map<DynamicAllocationReceiptKey, Record,
                     DynamicAllocationReceiptKeyHash> records_;
  size_t capacity_{};
};

}  // namespace memory_node_storage_owner_index_detail
