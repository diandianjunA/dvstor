#include <array>
#include <cassert>
#include <cstdint>
#include <span>

#include "memory_node/storage_owner_maintenance/reverse_outbox.hh"

namespace detail = memory_node_storage_owner_maintenance_detail;
namespace protocol = service::storage_owner;

namespace {

using Op = protocol::ReverseUpdateOp;

detail::Stage2ReverseDispatch dispatch(
    std::uint64_t logical_request_id,
    std::uint32_t worker_id,
    std::uint32_t peer_index,
    std::uint32_t context_slot,
    const Op* ops,
    std::uint32_t item_count,
    protocol::PeerRpcType request_type =
      protocol::PeerRpcType::reverse_update_request) {
  return detail::Stage2ReverseDispatch{
    .logical_request_id = logical_request_id,
    .context = detail::Stage2ContextHandle{context_slot, 1},
    .worker_id = worker_id,
    .peer_index = peer_index,
    .request_type = request_type,
    .item_count = item_count,
    .ops = ops,
    .ready_at_ns = 0,
  };
}

void test_two_contexts_share_one_wire_request_and_fan_out_ack() {
  detail::Stage2ReverseOutbox outbox(8, 8, 5, 8);
  const std::array<Op, 2> first_ops{{{1, 101}, {2, 102}}};
  const std::array<Op, 3> second_ops{{{3, 103}, {4, 104}, {5, 105}}};
  assert(outbox.try_enqueue(dispatch(
           11, 0, 3, 0, first_ops.data(), first_ops.size())) ==
         detail::Stage2ReverseEnqueueResult::enqueued);
  assert(outbox.try_enqueue(dispatch(
           12, 1, 3, 1, second_ops.data(), second_ops.size())) ==
         detail::Stage2ReverseEnqueueResult::enqueued);

  const auto formed = outbox.form_aggregate(3, 2, 1001, 10);
  assert(formed.has_value());
  assert(formed->wire_request_id == 1001);
  assert(formed->logical_count == 2);
  assert(formed->item_count == 5);
  assert(outbox.queued_size(3) == 0);
  assert(outbox.aggregate_size() == 1);

  size_t cursor = 0;
  // The wire request keeps the worker that formed it for its full lifecycle.
  // This makes the async response-registry owner exact even when a send-slot
  // miss or timeout returns the aggregate to ready_to_post.
  assert(!outbox.claim_ready_to_post(3, 10, cursor).has_value());
  cursor = 0;
  const auto ready = outbox.claim_ready_to_post(2, 10, cursor);
  assert(ready.has_value());
  assert(ready->owner_worker_id == 2);
  std::array<Op, 8> wire_ops{};
  assert(outbox.copy_ops(2, 1001, std::span<Op>{wire_ops}));
  assert(wire_ops[0].target_raw == 1);
  assert(wire_ops[1].target_raw == 2);
  assert(wire_ops[2].target_raw == 3);
  assert(wire_ops[4].candidate_raw == 105);
  assert(outbox.finish_post(2, 1001, true, 500));

  cursor = 0;
  const auto awaiting = outbox.claim_awaiting_response(2, cursor);
  assert(awaiting.has_value());
  assert(awaiting->owner_worker_id == 2);
  std::array<detail::Stage2ReverseCompletion, 8> completions{};
  const auto completion_count = outbox.copy_completions(
    2, 1001, std::span<detail::Stage2ReverseCompletion>{completions});
  assert(completion_count == 2);
  assert(completions[0].logical_request_id == 11);
  assert(completions[0].worker_id == 0);
  assert(completions[1].logical_request_id == 12);
  assert(completions[1].worker_id == 1);
  assert(outbox.finish_success(2, 1001));
  assert(outbox.size() == 0);
  assert(outbox.aggregate_size() == 0);
}

void test_wire_bound_and_rpc_type_partition() {
  detail::Stage2ReverseOutbox outbox(8, 8, 4, 5);
  const std::array<Op, 3> first{{{1, 1}, {2, 2}, {3, 3}}};
  const std::array<Op, 3> second{{{4, 4}, {5, 5}, {6, 6}}};
  const std::array<Op, 1> cleanup{{{7, 7}}};
  assert(outbox.try_enqueue(dispatch(
           20, 0, 1, 0, first.data(), first.size())) ==
         detail::Stage2ReverseEnqueueResult::enqueued);
  assert(outbox.try_enqueue(dispatch(
           21, 0, 1, 1, second.data(), second.size())) ==
         detail::Stage2ReverseEnqueueResult::enqueued);
  assert(outbox.try_enqueue(dispatch(
           22, 0, 1, 2, cleanup.data(), cleanup.size(),
           protocol::PeerRpcType::cleanup_deleted_request)) ==
         detail::Stage2ReverseEnqueueResult::enqueued);

  const auto first_aggregate = outbox.form_aggregate(1, 0, 2001, 0);
  assert(first_aggregate->logical_count == 1);
  assert(first_aggregate->item_count == 3);
  const auto second_aggregate = outbox.form_aggregate(1, 0, 2002, 0);
  assert(second_aggregate->logical_count == 1);
  assert(second_aggregate->item_count == 3);
  assert(second_aggregate->request_type ==
         protocol::PeerRpcType::reverse_update_request);
  const auto cleanup_aggregate = outbox.form_aggregate(1, 0, 2003, 0);
  assert(cleanup_aggregate->logical_count == 1);
  assert(cleanup_aggregate->request_type ==
         protocol::PeerRpcType::cleanup_deleted_request);
  assert(outbox.queued_size(1) == 0);
}

void test_retry_reuses_wire_id_and_payload() {
  detail::Stage2ReverseOutbox outbox(2, 2, 3, 4);
  const std::array<Op, 2> ops{{{41, 141}, {42, 142}}};
  assert(outbox.try_enqueue(dispatch(
           31, 0, 2, 0, ops.data(), ops.size())) ==
         detail::Stage2ReverseEnqueueResult::enqueued);
  assert(outbox.form_aggregate(2, 0, 3001, 0).has_value());

  size_t cursor = 0;
  assert(outbox.claim_ready_to_post(0, 0, cursor)->wire_request_id == 3001);
  assert(outbox.finish_post(0, 3001, false, 100));
  cursor = 0;
  assert(!outbox.claim_ready_to_post(1, 100, cursor).has_value());
  cursor = 0;
  assert(!outbox.claim_ready_to_post(0, 99, cursor).has_value());
  cursor = 0;
  assert(outbox.claim_ready_to_post(0, 100, cursor)->wire_request_id == 3001);
  std::array<Op, 4> copied{};
  assert(outbox.copy_ops(0, 3001, std::span<Op>{copied}));
  assert(copied[0].target_raw == 41 && copied[1].target_raw == 42);
  assert(outbox.finish_post(0, 3001, true, 500));

  cursor = 0;
  assert(outbox.claim_awaiting_response(0, cursor)->wire_request_id == 3001);
  assert(outbox.release_poll(0, 3001, true, 600));
  cursor = 0;
  assert(outbox.claim_ready_to_post(0, 600, cursor)->wire_request_id == 3001);
  assert(outbox.copy_ops(0, 3001, std::span<Op>{copied}));
  assert(copied[0].candidate_raw == 141 && copied[1].candidate_raw == 142);
}

void test_release_precedes_fanout_at_exact_capacity() {
  detail::Stage2ReverseOutbox outbox(1, 1, 2, 1);
  const std::array<Op, 1> old_ops{{{61, 161}}};
  const std::array<Op, 1> replacement_ops{{{62, 162}}};
  assert(outbox.try_enqueue(dispatch(
           51, 0, 1, 0, old_ops.data(), old_ops.size())) ==
         detail::Stage2ReverseEnqueueResult::enqueued);
  assert(outbox.form_aggregate(1, 0, 5001, 0).has_value());
  size_t cursor = 0;
  assert(outbox.claim_ready_to_post(0, 0, cursor).has_value());
  assert(outbox.finish_post(0, 5001, true, 100));
  cursor = 0;
  assert(outbox.claim_awaiting_response(0, cursor).has_value());

  std::array<detail::Stage2ReverseCompletion, 1> completion{};
  assert(outbox.copy_completions(
           0, 5001,
           std::span<detail::Stage2ReverseCompletion>{completion}) == 1);
  assert(outbox.finish_success(0, 5001));
  assert(!outbox.finish_success(0, 5001));
  cursor = 0;
  assert(!outbox.claim_awaiting_response(0, cursor).has_value());

  // The destination may reuse its context as soon as this value snapshot is
  // fanned out. The old descriptor must already be free, while the snapshot
  // remains valid and identifies exactly the old logical request.
  assert(outbox.try_enqueue(dispatch(
           52, 0, 1, 0, replacement_ops.data(), replacement_ops.size())) ==
         detail::Stage2ReverseEnqueueResult::enqueued);
  assert(completion[0].logical_request_id == 51);
  assert((completion[0].context == detail::Stage2ContextHandle{0, 1}));
  assert(completion[0].worker_id == 0);
  assert(completion[0].peer_index == 1);
}

void test_duplicate_capacity_and_shutdown_cleanup() {
  detail::Stage2ReverseOutbox outbox(2, 2, 3, 4);
  const std::array<Op, 1> ops{{{1, 2}}};
  const auto first = dispatch(41, 0, 1, 0, ops.data(), ops.size());
  assert(outbox.try_enqueue(first) ==
         detail::Stage2ReverseEnqueueResult::enqueued);
  assert(outbox.try_enqueue(first) ==
         detail::Stage2ReverseEnqueueResult::duplicate);
  auto conflict = first;
  conflict.item_count = 2;
  assert(outbox.try_enqueue(conflict) ==
         detail::Stage2ReverseEnqueueResult::conflict);
  assert(outbox.try_enqueue(dispatch(
           42, 1, 2, 0, ops.data(), ops.size())) ==
         detail::Stage2ReverseEnqueueResult::enqueued);
  assert(outbox.try_enqueue(dispatch(
           43, 1, 2, 1, ops.data(), ops.size())) ==
         detail::Stage2ReverseEnqueueResult::full);

  assert(outbox.erase_queued_worker(0) == 1);
  assert(outbox.form_aggregate(2, 1, 4001, 0).has_value());
  assert(outbox.discard_owned_aggregate(1) == 4001);
  assert(!outbox.discard_owned_aggregate(1).has_value());
  assert(outbox.size() == 0);
}

}  // namespace

int main() {
  test_two_contexts_share_one_wire_request_and_fan_out_ack();
  test_wire_bound_and_rpc_type_partition();
  test_retry_reuses_wire_id_and_payload();
  test_release_precedes_fanout_at_exact_capacity();
  test_duplicate_capacity_and_shutdown_cleanup();
  return 0;
}
