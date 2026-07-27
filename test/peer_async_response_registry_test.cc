#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "memory_node/peer_rpc/async_response.hh"

namespace detail = memory_node_detail;
namespace protocol = service::storage_owner;

namespace {

protocol::PeerRpcHeader response(u64 request_id,
                                 u32 shard,
                                 protocol::PeerRpcType type,
                                 u32 item_count,
                                 protocol::InsertStatus status =
                                   protocol::InsertStatus::ok) {
  protocol::PeerRpcHeader header{};
  header.magic = protocol::kPeerRpcMagic;
  header.version = protocol::kPeerRpcVersion;
  header.type = static_cast<u32>(type);
  header.source_shard = shard;
  header.item_count = item_count;
  header.request_id = request_id;
  header.status = static_cast<u32>(status);
  return header;
}

protocol::PeerRpcHeader request(u64 request_id,
                                u32 shard,
                                protocol::PeerRpcType type,
                                u32 item_count,
                                u32 reserved = 0) {
  protocol::PeerRpcHeader header{};
  header.magic = protocol::kPeerRpcMagic;
  header.version = protocol::kPeerRpcVersion;
  header.type = static_cast<u32>(type);
  header.source_shard = shard;
  header.item_count = item_count;
  header.request_id = request_id;
  header.reserved = reserved;
  return header;
}

void test_registration_delivery_and_explicit_consumption() {
  detail::PeerAsyncResponseRegistry registry(2);
  constexpr auto type = protocol::PeerRpcType::stage1_execute_response;

  assert(registry.register_request(11, 2, type, 3) ==
         detail::PeerResponseRegistration::registered);
  assert(registry.register_request(11, 2, type, 3) ==
         detail::PeerResponseRegistration::retry);
  assert(registry.register_request(11, 1, type, 3) ==
         detail::PeerResponseRegistration::conflict);
  assert(registry.register_request(12, 3, type, 1) ==
         detail::PeerResponseRegistration::registered);
  assert(registry.register_request(13, 4, type, 1) ==
         detail::PeerResponseRegistration::full);

  auto ok = response(11, 2, type, 3);
  auto wrong_peer = ok;
  wrong_peer.source_shard = 4;
  assert(!registry.try_deliver(4, 1, sizeof(ok), wrong_peer));
  auto wrong_count = ok;
  wrong_count.item_count = 4;
  assert(!registry.try_deliver(2, 1, sizeof(ok), wrong_count));
  assert(!registry.try_deliver(
    2, 1, sizeof(ok), response(999, 2, type, 3)));

  assert(registry.try_deliver(2, 7, 128, ok));
  assert(!registry.try_deliver(2, 8, 128, ok));
  detail::PeerResponseDescriptor descriptor;
  detail::PeerResponseLease lease;
  assert(registry.try_take(11, 3, type, 3, descriptor, lease) ==
         detail::TryPeerResponse::stale);
  assert(registry.try_take(11, 2, type, 3, descriptor, lease) ==
         detail::TryPeerResponse::success);
  assert(descriptor.peer_id == 2);
  assert(descriptor.receive_slot == 7);
  assert(descriptor.bytes == 128);
  assert(lease.valid());

  // A second poll cannot take ownership while the parser holds the lease.
  detail::PeerResponseLease duplicate_lease;
  assert(registry.try_take(
           11, 2, type, 3, descriptor, duplicate_lease) ==
         detail::TryPeerResponse::pending);
  assert(!duplicate_lease.valid());
  assert(!registry.ack_consumed(lease));
  assert(registry.mark_receive_reposted(lease));
  assert(registry.ack_consumed(lease));
  assert(registry.size() == 1);

  // Deletion removes the key entirely. A late packet cannot find a tombstone
  // or resurrect a completed request.
  assert(!registry.try_deliver(2, 9, 128, ok));
  assert(registry.try_take(11, 2, type, 3, descriptor, lease) ==
         detail::TryPeerResponse::stale);
}

void test_shutdown_drain_respects_receive_ownership() {
  constexpr auto type = protocol::PeerRpcType::reverse_update_response;
  detail::PeerAsyncResponseRegistry registry(2);
  detail::PeerResponseDescriptor descriptor;
  detail::PeerResponseLease lease;

  assert(registry.register_request(91, 1, type, 1) ==
         detail::PeerResponseRegistration::registered);
  const auto consumed = response(91, 1, type, 1);
  assert(registry.try_deliver(1, 5, sizeof(consumed), consumed));
  assert(registry.try_take(91, 1, type, 1, descriptor, lease) ==
         detail::TryPeerResponse::success);
  assert(registry.mark_receive_reposted(lease));
  // The receive WR is already back on the QP even though semantic parsing is
  // still leased. Shutdown must not repost it a second time.
  assert(registry.drain_completed().empty());
  assert(!registry.ack_consumed(lease));

  assert(registry.register_request(92, 1, type, 1) ==
         detail::PeerResponseRegistration::registered);
  const auto held = response(92, 1, type, 1);
  assert(registry.try_deliver(1, 6, sizeof(held), held));
  const auto drain = registry.drain_completed();
  assert(drain.size() == 1);
  assert(drain.front().receive_slot == 6);
}

void test_retry_generation_and_cancelled_descriptor() {
  detail::PeerAsyncResponseRegistry registry(4);
  constexpr auto type = protocol::PeerRpcType::reverse_update_response;
  assert(registry.register_request(21, 1, type, 5) ==
         detail::PeerResponseRegistration::registered);

  auto failed = response(
    21, 1, type, 5, protocol::InsertStatus::overloaded);
  assert(registry.try_deliver(1, 2, sizeof(failed), failed));
  detail::PeerResponseDescriptor descriptor;
  detail::PeerResponseLease rejected_lease;
  assert(registry.try_take(21, 1, type, 5, descriptor, rejected_lease) ==
         detail::TryPeerResponse::failure);
  assert(registry.mark_receive_reposted(rejected_lease));
  assert(registry.retry(rejected_lease));
  assert(!registry.ack_consumed(rejected_lease));
  // Until the sender explicitly rearms, a late duplicate is dropped.
  assert(!registry.try_deliver(1, 3, sizeof(failed), failed));
  assert(registry.try_take(21, 1, type, 5, descriptor, rejected_lease) ==
         detail::TryPeerResponse::pending);
  assert(registry.register_send_attempt(21, 1, type, 5) ==
         detail::PeerResponseRegistration::retry);

  auto ok = response(21, 1, type, 5);
  assert(registry.try_deliver(1, 4, sizeof(ok), ok));
  detail::PeerResponseLease accepted_lease;
  assert(registry.try_take(21, 1, type, 5, descriptor, accepted_lease) ==
         detail::TryPeerResponse::success);
  assert(accepted_lease.generation != rejected_lease.generation);
  assert(!registry.retry(rejected_lease));
  assert(registry.mark_receive_reposted(accepted_lease));
  assert(registry.ack_consumed(accepted_lease));

  assert(registry.register_request(22, 1, type, 1) ==
         detail::PeerResponseRegistration::registered);
  auto held = response(22, 1, type, 1);
  assert(registry.try_deliver(1, 6, sizeof(held), held));
  const auto cancelled = registry.cancel(22);
  assert(cancelled.has_value());
  assert(cancelled->receive_slot == 6);
  assert(!registry.try_deliver(1, 7, sizeof(held), held));

  assert(registry.register_request(23, 1, type, 1) ==
         detail::PeerResponseRegistration::registered);
  assert(!registry.cancel(23).has_value());
}

void test_transient_response_keeps_late_same_attempt_delivery_open() {
  detail::PeerAsyncResponseRegistry registry(2);
  constexpr auto type = protocol::PeerRpcType::stage1_execute_response;
  constexpr u64 request_id = 24;
  constexpr u32 shard = 3;
  constexpr u32 item_count = 5;
  assert(registry.register_request(request_id, shard, type, item_count) ==
         detail::PeerResponseRegistration::registered);

  // A synthetic all-retry response can arrive while the original Stage1
  // handler is still running. Its descriptor is consumed and reposted, but
  // no second send is posted during the sender's retry backoff.
  const auto transient = response(
    request_id, shard, type, item_count, protocol::InsertStatus::overloaded);
  assert(registry.try_deliver(shard, 2, sizeof(transient), transient));
  detail::PeerResponseDescriptor descriptor;
  detail::PeerResponseLease transient_lease;
  assert(registry.try_take(
           request_id, shard, type, item_count, descriptor,
           transient_lease) == detail::TryPeerResponse::failure);
  assert(!registry.await_late_delivery(transient_lease));
  assert(registry.mark_receive_reposted(transient_lease));
  assert(registry.await_late_delivery(transient_lease));
  assert(!registry.ack_consumed(transient_lease));

  // The original success is accepted immediately in the pending backoff
  // interval. In particular, this does not call register_send_attempt().
  const auto success = response(request_id, shard, type, item_count);
  assert(registry.try_deliver(shard, 3, sizeof(success), success));
  detail::PeerResponseLease success_lease;
  assert(registry.try_take(
           request_id, shard, type, item_count, descriptor,
           success_lease) == detail::TryPeerResponse::success);
  assert(success_lease.generation != transient_lease.generation);
  assert(!registry.await_late_delivery(transient_lease));
  assert(registry.mark_receive_reposted(success_lease));
  assert(registry.ack_consumed(success_lease));
}

void test_response_slab_aba_and_late_response() {
  detail::PeerAsyncResponseRegistry registry(2);
  constexpr auto type = protocol::PeerRpcType::cleanup_deleted_response;
  detail::PeerResponseDescriptor descriptor;

  assert(registry.register_request(101, 3, type, 1) ==
         detail::PeerResponseRegistration::registered);
  auto first = response(101, 3, type, 1);
  assert(registry.try_deliver(3, 1, sizeof(first), first));
  detail::PeerResponseLease stale_lease;
  assert(registry.try_take(101, 3, type, 1, descriptor, stale_lease) ==
         detail::TryPeerResponse::success);
  assert(registry.mark_receive_reposted(stale_lease));
  assert(registry.ack_consumed(stale_lease));

  // The LIFO free list intentionally reuses the slab slot immediately. The
  // old delayed lease must not acknowledge or rearm the new owner.
  assert(registry.register_request(102, 3, type, 1) ==
         detail::PeerResponseRegistration::registered);
  auto second = response(102, 3, type, 1);
  assert(registry.try_deliver(3, 2, sizeof(second), second));
  detail::PeerResponseLease current_lease;
  assert(registry.try_take(102, 3, type, 1, descriptor, current_lease) ==
         detail::TryPeerResponse::success);
  assert(stale_lease.slot == current_lease.slot);
  assert(stale_lease.generation != current_lease.generation);
  assert(!registry.ack_consumed(stale_lease));
  assert(!registry.retry(stale_lease));
  assert(registry.mark_receive_reposted(current_lease));
  assert(registry.ack_consumed(current_lease));

  // The first request ID is absent even after its former slab slot was reused.
  assert(!registry.try_deliver(3, 3, sizeof(first), first));
}

void test_response_high_churn_has_no_probe_cliff() {
  constexpr size_t capacity = 64;
  detail::PeerAsyncResponseRegistry registry(capacity);
  constexpr auto type = protocol::PeerRpcType::cleanup_deleted_response;
  detail::PeerResponseDescriptor descriptor;

  u64 next_request = 1'000;
  constexpr size_t rounds = 24;  // 24C unique IDs (>10C).
  for (size_t round = 0; round < rounds; ++round) {
    std::vector<u64> ids;
    ids.reserve(capacity);
    for (size_t index = 0; index < capacity; ++index) {
      const u64 id = next_request++;
      ids.push_back(id);
      assert(registry.register_request(id, 3, type, 1) ==
             detail::PeerResponseRegistration::registered);
    }
    assert(registry.size() == capacity);
    for (size_t index = 0; index < capacity; ++index) {
      const u64 id = ids[(index * 17) & (capacity - 1)];
      auto ok = response(id, 3, type, 1);
      assert(registry.try_deliver(
        3, static_cast<u32>(id & 7), sizeof(ok), ok));
      detail::PeerResponseLease lease;
      assert(registry.try_take(id, 3, type, 1, descriptor, lease) ==
             detail::TryPeerResponse::success);
      assert(registry.mark_receive_reposted(lease));
      assert(registry.ack_consumed(lease));
    }
    assert(registry.size() == 0);
  }

  const auto probes = registry.probe_telemetry();
  assert(probes.lookups != 0);
  assert(probes.probes / probes.lookups < 8);
  // At most half of the separate bucket table is occupied. This deterministic
  // churn must remain far below the old capacity-sized miss scan.
  assert(probes.max_probe < capacity / 2);
}

void test_receiver_request_dedup_and_generation() {
  detail::PeerRequestDeduplicator dedup(2);
  auto reverse = request(
    31, 2, protocol::PeerRpcType::reverse_update_request, 4);
  auto execute = dedup.begin(2, reverse, true);
  assert(execute.action == detail::PeerRequestAction::execute);
  assert(execute.lease.valid());
  assert(dedup.begin(2, reverse, true).action ==
         detail::PeerRequestAction::duplicate_inflight);

  auto reverse_ok = response(
    31, 0, protocol::PeerRpcType::reverse_update_response, 4);
  assert(dedup.complete(execute.lease, 2, reverse, reverse_ok));
  const auto replay = dedup.begin(2, reverse, true);
  assert(replay.action == detail::PeerRequestAction::replay);
  assert(replay.response.request_id == 31);

  // A non-replayable payload executes the same idempotent semantic operation
  // again, but with a fresh lease generation.
  auto second_execute = dedup.begin(2, reverse, false);
  assert(second_execute.action == detail::PeerRequestAction::execute);
  assert(second_execute.lease.generation != execute.lease.generation);
  assert(!dedup.complete(execute.lease, 2, reverse, reverse_ok));
  assert(dedup.complete(second_execute.lease, 2, reverse, reverse_ok));

  // Failed completion is removed and the identical request may execute again.
  auto cleanup = request(
    32, 3, protocol::PeerRpcType::cleanup_deleted_request, 2);
  auto cleanup_execute = dedup.begin(3, cleanup, true);
  assert(cleanup_execute.action == detail::PeerRequestAction::execute);
  auto cleanup_failed = response(
    32, 0, protocol::PeerRpcType::cleanup_deleted_response, 2,
    protocol::InsertStatus::failed);
  assert(dedup.complete(
    cleanup_execute.lease, 3, cleanup, cleanup_failed));
  cleanup_execute = dedup.begin(3, cleanup, true);
  assert(cleanup_execute.action == detail::PeerRequestAction::execute);
  assert(dedup.abandon(cleanup_execute.lease, 3, cleanup));

  // A stale delayed lease cannot affect the next occupant of the same slab.
  auto newer = request(
    33, 3, protocol::PeerRpcType::cleanup_deleted_request, 2);
  const auto newer_execute = dedup.begin(3, newer, true);
  assert(newer_execute.action == detail::PeerRequestAction::execute);
  assert(!dedup.abandon(cleanup_execute.lease, 3, cleanup));
  assert(dedup.begin(3, newer, true).action ==
         detail::PeerRequestAction::duplicate_inflight);
  assert(dedup.abandon(newer_execute.lease, 3, newer));
}

void test_dedup_inflight_capacity_and_high_churn_fifo() {
  {
    detail::PeerRequestDeduplicator inflight(2);
    const auto first = request(
      1, 1, protocol::PeerRpcType::reverse_update_request, 1);
    const auto second = request(
      2, 1, protocol::PeerRpcType::reverse_update_request, 1);
    const auto third = request(
      3, 1, protocol::PeerRpcType::reverse_update_request, 1);
    assert(inflight.begin(1, first, true).action ==
           detail::PeerRequestAction::execute);
    assert(inflight.begin(1, second, true).action ==
           detail::PeerRequestAction::execute);
    // Inflight work is never evicted to manufacture apparent capacity.
    assert(inflight.begin(1, third, true).action ==
           detail::PeerRequestAction::full);
  }

  constexpr size_t capacity = 64;
  detail::PeerRequestDeduplicator dedup(capacity);
  constexpr size_t operations = capacity * 24;  // >10C unique requests.
  for (size_t index = 0; index < operations; ++index) {
    const u64 id = 10'000 + index;
    const u32 shard = static_cast<u32>(index & 7);
    const auto req = request(
      id, shard, protocol::PeerRpcType::reverse_update_request, 1);
    const auto decision = dedup.begin(shard, req, true);
    assert(decision.action == detail::PeerRequestAction::execute);
    const auto ok = response(
      id, 0, protocol::PeerRpcType::reverse_update_response, 1);
    assert(dedup.complete(decision.lease, shard, req, ok));
    assert(dedup.size() <= capacity);
    const auto replay = dedup.begin(shard, req, true);
    assert(replay.action == detail::PeerRequestAction::replay);
  }
  assert(dedup.size() == capacity);

  const auto probes = dedup.probe_telemetry();
  assert(probes.lookups != 0);
  assert(probes.probes / probes.lookups < 8);
  assert(probes.max_probe < capacity / 2);
}

}  // namespace

int main() {
  test_registration_delivery_and_explicit_consumption();
  test_retry_generation_and_cancelled_descriptor();
  test_transient_response_keeps_late_same_attempt_delivery_open();
  test_response_slab_aba_and_late_response();
  test_shutdown_drain_respects_receive_ownership();
  test_response_high_churn_has_no_probe_cliff();
  test_receiver_request_dedup_and_generation();
  test_dedup_inflight_capacity_and_high_churn_fifo();
  return 0;
}
