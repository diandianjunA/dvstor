#include <cassert>

#include "memory_node/storage_owner_index/dynamic_allocation_receipt_policy.hh"

namespace {

namespace detail = memory_node_storage_owner_index_detail;
namespace protocol = service::storage_owner;

protocol::DynamicNodeControlItem allocation(
    u64 batch, u32 item_index, u64 source_raw) {
  return {
    .token = {
      .source_client = 7,
      .item_index = item_index,
      .client_batch_id = batch,
    },
    .node_raw = source_raw,
    .allocated_raw = 0,
    .id = 91,
    .generation = 4,
    .authority_shard = 2,
    .action = static_cast<u32>(
      protocol::DynamicNodeControlAction::allocate),
  };
}

protocol::DynamicNodeControlItem settlement(
    protocol::DynamicNodeControlItem item, u64 allocated_raw) {
  item.allocated_raw = allocated_raw;
  item.action = static_cast<u32>(
    protocol::DynamicNodeControlAction::settle_allocation);
  return item;
}

void test_lost_response_replays_one_reservation() {
  using Ledger = detail::DynamicAllocationReceiptLedger;
  Ledger ledger;
  ledger.reset(2);
  const auto item = allocation(100, 3, 0x1001);

  assert(ledger.begin(item, Ledger::SourceState::live).state ==
         Ledger::BeginState::claimed);
  assert(!ledger.publish(item, {
    .node_raw = 0x1fff,
    .status = static_cast<u32>(protocol::DynamicNodeControlStatus::ok),
  }));
  assert(ledger.validate_claim_source(item, Ledger::SourceState::live) ==
         Ledger::ClaimValidationState::validated);
  assert(ledger.begin(item, Ledger::SourceState::live).state ==
         Ledger::BeginState::pending);

  const protocol::DynamicNodeControlResult result{
    .node_raw = 0x2001,
    .status = static_cast<u32>(protocol::DynamicNodeControlStatus::ok),
  };
  assert(ledger.publish(item, result));
  const auto replay = ledger.begin(item, Ledger::SourceState::terminal);
  assert(replay.state == Ledger::BeginState::replay);
  assert(replay.result.node_raw == result.node_raw);
  assert(ledger.size() == 1);
}

void test_settlement_is_semantic_not_time_based() {
  using Ledger = detail::DynamicAllocationReceiptLedger;
  Ledger ledger;
  ledger.reset(1);
  const auto item = allocation(101, 0, 0x3001);
  assert(ledger.begin(item, Ledger::SourceState::live).state ==
         Ledger::BeginState::claimed);
  assert(ledger.validate_claim_source(item, Ledger::SourceState::live) ==
         Ledger::ClaimValidationState::validated);
  assert(ledger.publish(item, {
    .node_raw = 0x4001,
    .status = static_cast<u32>(protocol::DynamicNodeControlStatus::ok),
  }));
  const auto settle = settlement(item, 0x4001);

  assert(ledger.settle(settle, false, true) ==
         Ledger::SettleState::unsafe);
  assert(ledger.settle(settle, true, false) ==
         Ledger::SettleState::unsafe);
  assert(ledger.size() == 1);

  assert(ledger.settle(settle, true, true) ==
         Ledger::SettleState::settled);
  assert(ledger.size() == 0);
  assert(ledger.settle(settle, true, true) ==
         Ledger::SettleState::replay);
  assert(ledger.begin(item, Ledger::SourceState::terminal).state ==
         Ledger::BeginState::stale_source);
}

void test_conflict_and_capacity_backpressure() {
  using Ledger = detail::DynamicAllocationReceiptLedger;
  Ledger ledger;
  ledger.reset(1);
  const auto first = allocation(102, 0, 0x5001);
  assert(ledger.begin(first, Ledger::SourceState::live).state ==
         Ledger::BeginState::claimed);
  assert(ledger.validate_claim_source(first, Ledger::SourceState::live) ==
         Ledger::ClaimValidationState::validated);

  auto conflict = first;
  conflict.node_raw = 0x6001;
  assert(ledger.begin(conflict, Ledger::SourceState::live).state ==
         Ledger::BeginState::conflict);

  const auto second = allocation(103, 0, 0x7001);
  assert(ledger.begin(second, Ledger::SourceState::live).state ==
         Ledger::BeginState::pressure);
  assert(ledger.size() == 1);
  assert(ledger.cancel_claim(first));
  assert(ledger.begin(second, Ledger::SourceState::indeterminate).state ==
         Ledger::BeginState::indeterminate_source);
  assert(ledger.begin(second, Ledger::SourceState::live).state ==
         Ledger::BeginState::claimed);
  assert(ledger.validate_claim_source(
           second, Ledger::SourceState::indeterminate) ==
         Ledger::ClaimValidationState::indeterminate_source);
  assert(ledger.size() == 0);
}

void test_wrong_destination_cannot_release_receipt() {
  using Ledger = detail::DynamicAllocationReceiptLedger;
  Ledger ledger;
  ledger.reset(1);
  const auto item = allocation(104, 1, 0x8001);
  assert(ledger.begin(item, Ledger::SourceState::live).state ==
         Ledger::BeginState::claimed);
  assert(ledger.validate_claim_source(item, Ledger::SourceState::live) ==
         Ledger::ClaimValidationState::validated);
  assert(ledger.publish(item, {
    .node_raw = 0x9001,
    .status = static_cast<u32>(protocol::DynamicNodeControlStatus::ok),
  }));
  assert(ledger.settle(settlement(item, 0xa001), true, true) ==
         Ledger::SettleState::conflict);
  assert(ledger.size() == 1);
}

void test_sustained_updates_reuse_bounded_receipt_capacity() {
  using Ledger = detail::DynamicAllocationReceiptLedger;
  Ledger ledger;
  ledger.reset(1);
  for (u64 batch = 1; batch <= 10000; ++batch) {
    const auto item = allocation(
      1000 + batch, 0, 0x100000 + batch);
    assert(ledger.begin(item, Ledger::SourceState::live).state ==
           Ledger::BeginState::claimed);
    assert(ledger.validate_claim_source(item, Ledger::SourceState::live) ==
           Ledger::ClaimValidationState::validated);
    const u64 destination = 0x200000 + batch;
    assert(ledger.publish(item, {
      .node_raw = destination,
      .status = static_cast<u32>(protocol::DynamicNodeControlStatus::ok),
    }));
    assert(ledger.settle(settlement(item, destination), true, true) ==
           Ledger::SettleState::settled);
    assert(ledger.size() == 0);
    assert(ledger.begin(item, Ledger::SourceState::terminal).state ==
           Ledger::BeginState::stale_source);
  }
}

// Integration-style model of the physical-control ordering. The late RPC
// reads the source while the original receipt still exists, then pauses. The
// original handoff settles and erases its receipt before the late RPC calls
// begin() with that stale live observation. No allocator invocation is
// permitted after the post-claim source observation sees terminal.
void test_stale_preobservation_after_settlement_cannot_allocate_twice() {
  using Ledger = detail::DynamicAllocationReceiptLedger;
  Ledger ledger;
  ledger.reset(1);
  const auto item = allocation(105, 2, 0xb001);

  assert(ledger.begin(item, Ledger::SourceState::live).state ==
         Ledger::BeginState::claimed);
  assert(ledger.validate_claim_source(item, Ledger::SourceState::live) ==
         Ledger::ClaimValidationState::validated);
  constexpr u64 first_destination = 0xc001;
  assert(ledger.publish(item, {
    .node_raw = first_destination,
    .status = static_cast<u32>(protocol::DynamicNodeControlStatus::ok),
  }));

  const Ledger::SourceState stale_preobservation =
    Ledger::SourceState::live;
  assert(ledger.settle(
           settlement(item, first_destination), true, true) ==
         Ledger::SettleState::settled);
  assert(ledger.size() == 0);

  size_t allocator_invocations = 1;
  assert(ledger.begin(item, stale_preobservation).state ==
         Ledger::BeginState::claimed);
  const auto validation = ledger.validate_claim_source(
    item, Ledger::SourceState::terminal);
  if (validation == Ledger::ClaimValidationState::validated) {
    ++allocator_invocations;
  }
  assert(validation == Ledger::ClaimValidationState::stale_source);
  assert(allocator_invocations == 1);
  assert(ledger.size() == 0);
  assert(!ledger.publish(item, {
    .node_raw = 0xd001,
    .status = static_cast<u32>(protocol::DynamicNodeControlStatus::ok),
  }));

  // A retry performs a fresh pre-observation and is rejected before claiming.
  assert(ledger.begin(item, Ledger::SourceState::terminal).state ==
         Ledger::BeginState::stale_source);
}

void test_claim_cancel_is_exception_safe_and_retryable() {
  using Ledger = detail::DynamicAllocationReceiptLedger;
  Ledger ledger;
  ledger.reset(1);
  const auto item = allocation(106, 0, 0xe001);

  assert(ledger.begin(item, Ledger::SourceState::live).state ==
         Ledger::BeginState::claimed);
  assert(ledger.validate_claim_source(item, Ledger::SourceState::live) ==
         Ledger::ClaimValidationState::validated);
  // Model an allocator exception before it returns a reserved slot.
  assert(ledger.cancel_claim(item));
  assert(ledger.size() == 0);

  assert(ledger.begin(item, Ledger::SourceState::live).state ==
         Ledger::BeginState::claimed);
  assert(ledger.validate_claim_source(item, Ledger::SourceState::live) ==
         Ledger::ClaimValidationState::validated);
  assert(ledger.publish(item, {
    .node_raw = 0xf001,
    .status = static_cast<u32>(protocol::DynamicNodeControlStatus::ok),
  }));
  assert(ledger.size() == 1);
}

}  // namespace

int main() {
  test_lost_response_replays_one_reservation();
  test_settlement_is_semantic_not_time_based();
  test_conflict_and_capacity_backpressure();
  test_wrong_destination_cannot_release_receipt();
  test_sustained_updates_reuse_bounded_receipt_capacity();
  test_stale_preobservation_after_settlement_cannot_allocate_twice();
  test_claim_cancel_is_exception_safe_and_retryable();
  return 0;
}
