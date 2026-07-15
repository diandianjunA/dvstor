#include <cassert>
#include <cstddef>
#include <cstdint>

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
  assert(header.magic == protocol::kPeerRpcMagic);
  assert(header.version == protocol::kPeerRpcVersion);
  header.type = static_cast<u32>(type);
  header.source_shard = shard;
  header.item_count = item_count;
  header.request_id = request_id;
  header.reserved = reserved;
  return header;
}

void test_registration_delivery_and_stale_suppression() {
  detail::PeerAsyncResponseRegistry registry(2);
  constexpr auto type = protocol::PeerRpcType::stitch_search_response;

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
  auto unknown = response(999, 2, type, 3);
  assert(!registry.try_deliver(2, 1, sizeof(unknown), unknown));

  assert(registry.try_deliver(2, 7, 128, ok));
  assert(!registry.try_deliver(2, 8, 128, ok));
  detail::PeerResponseDescriptor descriptor;
  assert(registry.try_take(11, 3, type, 3, descriptor) ==
         detail::TryPeerResponse::stale);
  assert(registry.try_take(11, 2, type, 3, descriptor) ==
         detail::TryPeerResponse::success);
  assert(descriptor.peer_id == 2);
  assert(descriptor.receive_slot == 7);
  assert(descriptor.bytes == 128);
  assert(descriptor.header.request_id == 11);

  // A late duplicate cannot resurrect a successfully consumed request, and a
  // completed ID cannot accidentally be reused as a new logical operation.
  assert(!registry.try_deliver(2, 9, 128, ok));
  assert(registry.register_request(11, 2, type, 3) ==
         detail::PeerResponseRegistration::retired);
  assert(registry.mark_retryable(11, 2, type, 3));
  assert(registry.register_request(11, 2, type, 3) ==
         detail::PeerResponseRegistration::retry);
  assert(registry.try_deliver(2, 10, 128, ok));
  assert(registry.try_take(11, 2, type, 3, descriptor) ==
         detail::TryPeerResponse::success);
}

void test_retryable_failure_and_cancelled_descriptor() {
  detail::PeerAsyncResponseRegistry registry(4);
  constexpr auto type = protocol::PeerRpcType::reverse_update_response;
  assert(registry.register_request(21, 1, type, 5) ==
         detail::PeerResponseRegistration::registered);

  auto failed = response(
    21, 1, type, 5, protocol::InsertStatus::overloaded);
  assert(registry.try_deliver(1, 2, sizeof(failed), failed));
  detail::PeerResponseDescriptor descriptor;
  assert(registry.try_take(21, 1, type, 5, descriptor) ==
         detail::TryPeerResponse::failure);
  // Until the sender explicitly rearms, a late duplicate failure is dropped.
  assert(!registry.try_deliver(1, 3, sizeof(failed), failed));
  // Polling between the failure and its retry deadline must not classify the
  // still-live logical request as stale and cancel its same-ID retry state.
  assert(registry.try_take(21, 1, type, 5, descriptor) ==
         detail::TryPeerResponse::pending);
  assert(registry.register_request(21, 1, type, 5) ==
         detail::PeerResponseRegistration::retry);
  auto ok = response(21, 1, type, 5);
  assert(registry.try_deliver(1, 4, sizeof(ok), ok));
  assert(registry.try_take(21, 1, type, 5, descriptor) ==
         detail::TryPeerResponse::success);
  assert(descriptor.receive_slot == 4);

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

void test_capacity_churn_and_slot_wrap() {
  detail::PeerAsyncResponseRegistry registry(2);
  constexpr auto type = protocol::PeerRpcType::cleanup_deleted_response;
  detail::PeerResponseDescriptor descriptor;
  for (u64 request_id = 100; request_id < 10'000; ++request_id) {
    assert(registry.register_request(request_id, 3, type, 1) ==
           detail::PeerResponseRegistration::registered);
    auto ok = response(request_id, 3, type, 1);
    assert(registry.try_deliver(
      3, static_cast<u32>(request_id & 7), sizeof(ok), ok));
    assert(registry.try_take(request_id, 3, type, 1, descriptor) ==
           detail::TryPeerResponse::success);
    assert(registry.size() == 0);
  }
}

void test_send_attempt_recovers_reused_retired_tombstone() {
  constexpr auto type = protocol::PeerRpcType::reverse_update_response;
  detail::PeerResponseDescriptor descriptor;

  // A live logical request may explicitly retry the same ID after consuming
  // and rejecting a structurally valid response.
  detail::PeerAsyncResponseRegistry direct(2);
  assert(direct.register_request(201, 1, type, 2) ==
         detail::PeerResponseRegistration::registered);
  auto first_ok = response(201, 1, type, 2);
  assert(direct.try_deliver(1, 1, sizeof(first_ok), first_ok));
  assert(direct.try_take(201, 1, type, 2, descriptor) ==
         detail::TryPeerResponse::success);
  assert(direct.register_request(201, 1, type, 2) ==
         detail::PeerResponseRegistration::retired);
  assert(direct.register_send_attempt(201, 1, type, 2) ==
         detail::PeerResponseRegistration::retry);

  // Under concurrency the retired slot may be reused before a best-effort
  // rearm. Capacity pressure is transient: the same send attempt installs the
  // missing ID once one bounded slot becomes available.
  detail::PeerAsyncResponseRegistry reused(2);
  assert(reused.register_request(211, 1, type, 2) ==
         detail::PeerResponseRegistration::registered);
  auto reused_ok = response(211, 1, type, 2);
  assert(reused.try_deliver(1, 2, sizeof(reused_ok), reused_ok));
  assert(reused.try_take(211, 1, type, 2, descriptor) ==
         detail::TryPeerResponse::success);
  assert(reused.register_request(212, 1, type, 1) ==
         detail::PeerResponseRegistration::registered);
  assert(reused.register_request(213, 1, type, 1) ==
         detail::PeerResponseRegistration::registered);
  assert(!reused.mark_retryable(211, 1, type, 2));
  assert(reused.register_send_attempt(211, 1, type, 2) ==
         detail::PeerResponseRegistration::full);
  assert(!reused.cancel(212).has_value());
  assert(reused.register_send_attempt(211, 1, type, 2) ==
         detail::PeerResponseRegistration::registered);
  assert(reused.try_deliver(1, 3, sizeof(reused_ok), reused_ok));
  assert(reused.try_take(211, 1, type, 2, descriptor) ==
         detail::TryPeerResponse::success);
}

void test_receiver_request_dedup_and_replay() {
  detail::PeerRequestDeduplicator dedup(2);
  auto reverse = request(
    31, 2, protocol::PeerRpcType::reverse_update_request, 4);
  assert(dedup.begin(2, reverse, true).action ==
         detail::PeerRequestAction::execute);
  assert(dedup.begin(2, reverse, true).action ==
         detail::PeerRequestAction::duplicate_inflight);

  auto reverse_ok = response(
    31, 0, protocol::PeerRpcType::reverse_update_response, 4);
  dedup.complete(2, reverse, reverse_ok);
  const auto replay = dedup.begin(2, reverse, true);
  assert(replay.action == detail::PeerRequestAction::replay);
  assert(replay.response.request_id == 31);
  assert(replay.response.status ==
         static_cast<u32>(protocol::InsertStatus::ok));

  // A retryable failure/queue rejection may execute again with the exact ID.
  auto cleanup = request(
    32, 3, protocol::PeerRpcType::cleanup_deleted_request, 2);
  assert(dedup.begin(3, cleanup, true).action ==
         detail::PeerRequestAction::execute);
  auto cleanup_failed = response(
    32, 0, protocol::PeerRpcType::cleanup_deleted_response, 2,
    protocol::InsertStatus::failed);
  dedup.complete(3, cleanup, cleanup_failed);
  assert(dedup.begin(3, cleanup, true).action ==
         detail::PeerRequestAction::execute);
  dedup.abandon(3, cleanup);
  assert(dedup.begin(3, cleanup, true).action ==
         detail::PeerRequestAction::execute);

  // Stitch payloads are not cached: concurrent duplicates coalesce, while a
  // completed read-only search can be recomputed with the same ID.
  detail::PeerRequestDeduplicator stitch_dedup(2);
  auto stitch = request(
    41, 1, protocol::PeerRpcType::stitch_search_request, 3, 8);
  assert(stitch_dedup.begin(1, stitch, false).action ==
         detail::PeerRequestAction::execute);
  assert(stitch_dedup.begin(1, stitch, false).action ==
         detail::PeerRequestAction::duplicate_inflight);
  auto stitch_ok = response(
    41, 0, protocol::PeerRpcType::stitch_search_response, 3);
  stitch_dedup.complete(1, stitch, stitch_ok);
  assert(stitch_dedup.begin(1, stitch, false).action ==
         detail::PeerRequestAction::execute);

  auto conflicting = stitch;
  conflicting.item_count = 4;
  assert(stitch_dedup.begin(1, conflicting, false).action ==
         detail::PeerRequestAction::conflict);
}

}  // namespace

int main() {
  test_registration_delivery_and_stale_suppression();
  test_retryable_failure_and_cancelled_descriptor();
  test_capacity_churn_and_slot_wrap();
  test_send_attempt_recovers_reused_retired_tombstone();
  test_receiver_request_dedup_and_replay();
  return 0;
}
