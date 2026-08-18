#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

#include "memory_node/storage_owner_maintenance/home_rpc_outbox.hh"

namespace detail = memory_node_storage_owner_maintenance_detail;
namespace protocol = service::storage_owner;

namespace {

std::vector<byte_t> expand_request(
    std::uint32_t item_count,
    std::uint64_t pointer_base,
    byte_t query_base,
    protocol::Stage2HomeOperation operation =
      protocol::Stage2HomeOperation::expand_score) {
  std::vector<byte_t> request(
    protocol::stage2_expand_score_request_bytes(item_count), byte_t{0});
  auto* items = protocol::stage2_expand_score_items(request.data());
  byte_t* queries = protocol::stage2_expand_score_queries(
    request.data(), item_count);
  for (std::uint32_t item = 0; item < item_count; ++item) {
    items[item] = protocol::Stage2ExpandScoreItem{
      .pointer_raw = pointer_base + item,
      .generation = 100 + item,
      .search_index = 10 + item,
      .operation = static_cast<std::uint32_t>(operation),
    };
    for (std::size_t byte = 0; byte < VamanaNode::vector_bytes(); ++byte) {
      queries[static_cast<std::size_t>(item) * VamanaNode::vector_bytes() +
              byte] = static_cast<byte_t>(query_base + item);
    }
  }
  return request;
}

std::vector<byte_t> score_request(
    std::span<const std::uint32_t> query_indexes,
    std::uint32_t query_count,
    std::uint64_t pointer_base,
    byte_t query_base,
    std::uint32_t flags = 0) {
  const auto item_count = static_cast<std::uint32_t>(query_indexes.size());
  std::vector<byte_t> request(
    protocol::stage2_score_many_request_bytes(item_count, query_count),
    byte_t{0});
  auto* own_header = protocol::stage2_score_many_header(request.data());
  own_header->query_count = query_count;
  own_header->flags = flags;
  auto* items = protocol::stage2_score_many_items(request.data());
  for (std::uint32_t item = 0; item < item_count; ++item) {
    items[item] = protocol::Stage2ScoreManyItem{
      .pointer_raw = pointer_base + item,
      .generation = 200 + item,
      .search_index = 20 + item,
      .query_index = query_indexes[item],
    };
  }
  byte_t* queries = protocol::stage2_score_many_queries(
    request.data(), item_count);
  for (std::uint32_t query = 0; query < query_count; ++query) {
    for (std::size_t byte = 0; byte < VamanaNode::vector_bytes(); ++byte) {
      queries[static_cast<std::size_t>(query) * VamanaNode::vector_bytes() +
              byte] = static_cast<byte_t>(query_base + query);
    }
  }
  return request;
}

detail::Stage2HomeRpcDispatch dispatch(
    std::uint64_t logical_request_id,
    std::uint32_t peer_index,
    protocol::PeerRpcType request_type,
    std::uint32_t item_count,
    const std::vector<byte_t>& request,
    bool speculative = false) {
  return detail::Stage2HomeRpcDispatch{
    .logical_request_id = logical_request_id,
    .peer_index = peer_index,
    .request_type = request_type,
    .item_count = item_count,
    .speculative = speculative,
    .request = std::span<const byte_t>{request},
  };
}

std::vector<byte_t> claim_for_response(
    detail::Stage2HomeRpcOutbox& outbox,
    std::uint64_t wire_request_id,
    std::size_t request_bytes) {
  std::vector<byte_t> wire(request_bytes);
  std::size_t copied = 0;
  const auto lease = outbox.claim_ready_for_post(
    wire_request_id, wire, copied);
  assert(lease.has_value() && copied == request_bytes);
  std::size_t duplicate_bytes = 0;
  assert(!outbox.claim_ready_for_post(
            wire_request_id, wire, duplicate_bytes).has_value());
  assert(outbox.mark_awaiting_response(*lease, 100));
  return wire;
}

void test_expand_score_exact_wire_combine_and_compact_demux() {
  detail::Stage2HomeRpcOutbox outbox(
    8, 4, 5, 32, 256, 256, 1u << 20);
  auto first = expand_request(2, 1000, 11);
  auto second = expand_request(1, 2000, 22);
  auto* first_items = protocol::stage2_expand_score_items(first.data());
  first_items[1].operation = static_cast<std::uint32_t>(
    protocol::Stage2HomeOperation::score_only);

  assert(outbox.try_enqueue(dispatch(
           11, 3, protocol::PeerRpcType::stage2_expand_score_request,
           2, first)) == detail::Stage2HomeRpcEnqueueResult::enqueued);
  assert(outbox.try_enqueue(dispatch(
           12, 3, protocol::PeerRpcType::stage2_expand_score_request,
           1, second)) == detail::Stage2HomeRpcEnqueueResult::enqueued);
  const auto aggregate = outbox.form_aggregate(
    3, protocol::PeerRpcType::stage2_expand_score_request,
    9001, 0);
  assert(aggregate.has_value());
  assert(aggregate->logical_count == 2);
  assert(aggregate->item_count == 3);

  std::vector<byte_t> wire = claim_for_response(
    outbox, 9001, aggregate->request_bytes);
  const auto* header = reinterpret_cast<const protocol::PeerRpcHeader*>(
    wire.data());
  assert(header->request_id == 9001);
  assert(header->source_shard == 0);
  assert(header->item_count == 3);
  const auto* wire_items = protocol::stage2_expand_score_items(wire.data());
  assert(wire_items[0].pointer_raw == 1000);
  assert(wire_items[1].pointer_raw == 1001);
  assert(wire_items[2].pointer_raw == 2000);
  const byte_t* wire_queries = protocol::stage2_expand_score_queries(
    wire.data(), 3);
  assert(wire_queries[0] == 11);
  assert(wire_queries[VamanaNode::vector_bytes()] == 12);
  assert(wire_queries[2 * VamanaNode::vector_bytes()] == 22);

  std::vector<byte_t> response(
    protocol::stage2_expand_score_response_bytes(3, 3), byte_t{0});
  auto* response_header = reinterpret_cast<protocol::PeerRpcHeader*>(
    response.data());
  *response_header = {
    .magic = protocol::kPeerRpcMagic,
    .version = protocol::kPeerRpcVersion,
    .type = static_cast<std::uint32_t>(
      protocol::PeerRpcType::stage2_expand_score_response),
    .source_shard = 3,
    .item_count = 3,
    .request_id = 9001,
    .status = static_cast<std::uint32_t>(protocol::InsertStatus::ok),
  };
  auto* results = protocol::stage2_expand_score_results(response.data());
  results[0] = {
    .pointer_raw = 1000,
    .generation = 100,
    .search_index = 10,
    .neighbor_count = 2,
    .disposition = static_cast<std::uint32_t>(
      protocol::Stage2HomeDisposition::stable),
    .neighbor_offset = 0,
  };
  results[1] = {
    .pointer_raw = 1001,
    .generation = 101,
    .search_index = 11,
    .neighbor_count = 0,
    .disposition = static_cast<std::uint32_t>(
      protocol::Stage2HomeDisposition::stable),
    .neighbor_offset = 2,
    .operation = static_cast<std::uint32_t>(
      protocol::Stage2HomeOperation::score_only),
  };
  results[2] = {
    .pointer_raw = 2000,
    .generation = 100,
    .search_index = 10,
    .neighbor_count = 1,
    .disposition = static_cast<std::uint32_t>(
      protocol::Stage2HomeDisposition::stable),
    .neighbor_offset = 2,
  };
  auto* neighbors = protocol::stage2_expand_score_neighbors(
    response.data(), 3);
  neighbors[0].pointer_raw = 3000;
  neighbors[1].pointer_raw = 3001;
  neighbors[2].pointer_raw = 4000;

  const auto logical = outbox.demultiplex_response(9001, response);
  assert(logical.has_value());
  assert(logical->logical_responses.size() == 2);
  const auto& logical_responses = logical->logical_responses;
  assert(logical_responses[0].logical_request_id == 11);
  assert(logical_responses[1].logical_request_id == 12);
  const auto* first_header = reinterpret_cast<const protocol::PeerRpcHeader*>(
    logical_responses[0].response.data());
  const auto* second_header = reinterpret_cast<const protocol::PeerRpcHeader*>(
    logical_responses[1].response.data());
  assert(first_header->request_id == 11 && first_header->item_count == 2);
  assert(second_header->request_id == 12 && second_header->item_count == 1);
  const auto* first_results = protocol::stage2_expand_score_results(
    logical_responses[0].response.data());
  const auto* second_results = protocol::stage2_expand_score_results(
    logical_responses[1].response.data());
  assert(first_results[0].neighbor_offset == 0);
  assert(first_results[1].neighbor_offset == 2);
  assert(second_results[0].neighbor_offset == 0);
  const auto* first_neighbors = protocol::stage2_expand_score_neighbors(
    logical_responses[0].response.data(), 2);
  const auto* second_neighbors = protocol::stage2_expand_score_neighbors(
    logical_responses[1].response.data(), 1);
  assert(first_neighbors[0].pointer_raw == 3000);
  assert(first_neighbors[1].pointer_raw == 3001);
  assert(second_neighbors[0].pointer_raw == 4000);
  assert(outbox.finish_success(logical->lease));
  assert(!outbox.finish_success(logical->lease));
  assert(outbox.size() == 0 && outbox.aggregate_size() == 0);
}

void test_score_many_query_rebase_and_exact_demux() {
  detail::Stage2HomeRpcOutbox outbox(
    8, 4, 5, 32, 256, 256, 1u << 20);
  const std::vector<std::uint32_t> first_query_indexes{0, 0};
  const std::vector<std::uint32_t> second_query_indexes{0, 1};
  auto first = score_request(first_query_indexes, 1, 5000, 31);
  auto second = score_request(second_query_indexes, 2, 6000, 41);
  assert(outbox.try_enqueue(dispatch(
           21, 4, protocol::PeerRpcType::stage2_score_many_request,
           2, first)) == detail::Stage2HomeRpcEnqueueResult::enqueued);
  assert(outbox.try_enqueue(dispatch(
           22, 4, protocol::PeerRpcType::stage2_score_many_request,
           2, second)) == detail::Stage2HomeRpcEnqueueResult::enqueued);
  const auto aggregate = outbox.form_aggregate(
    4, protocol::PeerRpcType::stage2_score_many_request,
    9002, 0);
  assert(aggregate.has_value());
  assert(aggregate->logical_count == 2 && aggregate->item_count == 4);

  std::vector<byte_t> wire = claim_for_response(
    outbox, 9002, aggregate->request_bytes);
  const auto* own_header = protocol::stage2_score_many_header(wire.data());
  assert(own_header->query_count == 3 && own_header->flags == 0);
  const auto* items = protocol::stage2_score_many_items(wire.data());
  assert(items[0].query_index == 0 && items[1].query_index == 0);
  assert(items[2].query_index == 1 && items[3].query_index == 2);
  const byte_t* queries = protocol::stage2_score_many_queries(wire.data(), 4);
  assert(queries[0] == 31);
  assert(queries[VamanaNode::vector_bytes()] == 41);
  assert(queries[2 * VamanaNode::vector_bytes()] == 42);

  std::vector<byte_t> response(
    protocol::stage2_score_many_response_bytes(4), byte_t{0});
  auto* response_header = reinterpret_cast<protocol::PeerRpcHeader*>(
    response.data());
  *response_header = {
    .magic = protocol::kPeerRpcMagic,
    .version = protocol::kPeerRpcVersion,
    .type = static_cast<std::uint32_t>(
      protocol::PeerRpcType::stage2_score_many_response),
    .source_shard = 4,
    .item_count = 4,
    .request_id = 9002,
    .status = static_cast<std::uint32_t>(protocol::InsertStatus::ok),
  };
  auto* results = protocol::stage2_score_many_results(response.data());
  for (std::uint32_t item = 0; item < 4; ++item) {
    results[item].pointer_raw = item < 2 ? 5000 + item : 5998 + item;
    results[item].generation = 200 + (item % 2);
    results[item].search_index = 20 + (item % 2);
    results[item].disposition = static_cast<std::uint32_t>(
      protocol::Stage2HomeDisposition::stable);
    results[item].distance = static_cast<distance_t>(100 + item);
  }
  results[3].pointer_raw ^= 1;
  assert(!outbox.demultiplex_response(9002, response).has_value());
  results[3].pointer_raw ^= 1;
  const auto logical = outbox.demultiplex_response(9002, response);
  assert(logical.has_value() && logical->logical_responses.size() == 2);
  const auto& logical_responses = logical->logical_responses;
  const auto* first_results = protocol::stage2_score_many_results(
    logical_responses[0].response.data());
  const auto* second_results = protocol::stage2_score_many_results(
    logical_responses[1].response.data());
  assert(first_results[0].pointer_raw == 5000);
  assert(first_results[1].distance == static_cast<distance_t>(101));
  assert(second_results[0].pointer_raw == 6000);
  assert(second_results[1].distance == static_cast<distance_t>(103));
  assert(reinterpret_cast<const protocol::PeerRpcHeader*>(
           logical_responses[1].response.data())->request_id == 22);
  assert(outbox.finish_success(logical->lease));
}

void test_partition_bounds_retry_and_cancellation() {
  detail::Stage2HomeRpcOutbox outbox(
    8, 4, 5, 3, 8, 4, 1u << 20);
  std::vector<byte_t> truncated_score(
    protocol::stage2_score_many_items_offset() - 1, byte_t{0});
  assert(outbox.try_enqueue(dispatch(
           30, 2, protocol::PeerRpcType::stage2_score_many_request,
           1, truncated_score)) == detail::Stage2HomeRpcEnqueueResult::invalid);
  auto first = expand_request(2, 7000, 51);
  auto second = expand_request(4, 8000, 61);
  auto third = expand_request(1, 9000, 71);
  const auto first_dispatch = dispatch(
    31, 2, protocol::PeerRpcType::stage2_expand_score_request, 2, first);
  assert(outbox.try_enqueue(first_dispatch) ==
         detail::Stage2HomeRpcEnqueueResult::enqueued);
  assert(outbox.try_enqueue(first_dispatch) ==
         detail::Stage2HomeRpcEnqueueResult::duplicate);
  auto conflict = first_dispatch;
  conflict.peer_index = 1;
  assert(outbox.try_enqueue(conflict) ==
         detail::Stage2HomeRpcEnqueueResult::conflict);
  assert(outbox.try_enqueue(dispatch(
           32, 2, protocol::PeerRpcType::stage2_expand_score_request,
           4, second)) == detail::Stage2HomeRpcEnqueueResult::invalid);
  assert(outbox.try_enqueue(dispatch(
           33, 2, protocol::PeerRpcType::stage2_expand_score_request,
           1, third)) == detail::Stage2HomeRpcEnqueueResult::enqueued);
  const auto aggregate = outbox.form_aggregate(
    2, protocol::PeerRpcType::stage2_expand_score_request,
    9003, 0);
  assert(aggregate.has_value() && aggregate->item_count == 3);

  std::vector<byte_t> original_wire = claim_for_response(
    outbox, 9003, aggregate->request_bytes);
  std::vector<byte_t> malformed(
    protocol::stage2_expand_score_response_bytes(3, 1), byte_t{0});
  auto* malformed_header = reinterpret_cast<protocol::PeerRpcHeader*>(
    malformed.data());
  *malformed_header = {
    .magic = protocol::kPeerRpcMagic,
    .version = protocol::kPeerRpcVersion,
    .type = static_cast<std::uint32_t>(
      protocol::PeerRpcType::stage2_expand_score_response),
    .source_shard = 2,
    .item_count = 3,
    .request_id = 9003,
    .status = static_cast<std::uint32_t>(protocol::InsertStatus::ok),
  };
  auto* malformed_results = protocol::stage2_expand_score_results(
    malformed.data());
  const auto* original_items = protocol::stage2_expand_score_items(
    original_wire.data());
  for (std::uint32_t item = 0; item < 3; ++item) {
    malformed_results[item].pointer_raw = original_items[item].pointer_raw;
    malformed_results[item].generation = original_items[item].generation;
    malformed_results[item].search_index = original_items[item].search_index;
    malformed_results[item].operation = original_items[item].operation;
  }
  malformed_results[0].neighbor_offset = 1;
  malformed_results[0].neighbor_count = 1;
  malformed_results[1].neighbor_offset = 1;
  malformed_results[2].neighbor_offset = 1;
  assert(!outbox.demultiplex_response(9003, malformed).has_value());
  assert(outbox.retry_after_timeout(9003));
  std::vector<byte_t> retry_wire = claim_for_response(
    outbox, 9003, aggregate->request_bytes);
  assert(retry_wire == original_wire);

  assert(outbox.cancel_logical(33));
  malformed_results[0].neighbor_offset = 0;
  const auto logical = outbox.demultiplex_response(9003, malformed);
  assert(logical.has_value() &&
         logical->logical_responses.size() == 1);
  assert(logical->logical_responses[0].logical_request_id == 31);
  // Cancellation may race after demux leased the aggregate but before the
  // transactional registry fan-out. Marking the final live member causes the
  // failed fan-out release to retire the aggregate instead of retrying a
  // response for a context that no longer exists.
  assert(outbox.cancel_logical(31));
  assert(outbox.release_demux(logical->lease));
  assert(outbox.size() == 0);
  assert(outbox.aggregate_size() == 0);

  const std::vector<std::uint32_t> query_index{0};
  auto authoritative = score_request(query_index, 1, 10000, 81);
  auto speculative = score_request(
    query_index, 1, 11000, 91,
    protocol::kStage2ScoreManyFlagSpeculative);
  assert(outbox.try_enqueue(dispatch(
           41, 1, protocol::PeerRpcType::stage2_score_many_request,
           1, authoritative)) == detail::Stage2HomeRpcEnqueueResult::enqueued);
  assert(outbox.try_enqueue(dispatch(
           42, 1, protocol::PeerRpcType::stage2_score_many_request,
           1, speculative, true)) ==
         detail::Stage2HomeRpcEnqueueResult::enqueued);
  assert(outbox.queued_size(
           1, protocol::PeerRpcType::stage2_score_many_request, false) == 1);
  assert(outbox.queued_size(
           1, protocol::PeerRpcType::stage2_score_many_request, true) == 1);
  const auto auth_aggregate = outbox.form_aggregate(
    1, protocol::PeerRpcType::stage2_score_many_request,
    9004, 0, false);
  assert(auth_aggregate.has_value() && auth_aggregate->logical_count == 1);
  const auto spec_aggregate = outbox.form_aggregate(
    1, protocol::PeerRpcType::stage2_score_many_request,
    9005, 0, true);
  assert(spec_aggregate.has_value() && spec_aggregate->logical_count == 1);
  assert(outbox.discard_aggregate(9004));
  assert(outbox.discard_aggregate(9005));

  auto auth_expand = expand_request(1, 12000, 101);
  auto spec_expand = expand_request(1, 13000, 111);
  assert(outbox.try_enqueue(dispatch(
           51, 1, protocol::PeerRpcType::stage2_expand_score_request,
           1, auth_expand, false)) ==
         detail::Stage2HomeRpcEnqueueResult::enqueued);
  assert(outbox.try_enqueue(dispatch(
           52, 1, protocol::PeerRpcType::stage2_expand_score_request,
           1, spec_expand, true)) ==
         detail::Stage2HomeRpcEnqueueResult::enqueued);
  const auto auth_expand_aggregate = outbox.form_aggregate(
    1, protocol::PeerRpcType::stage2_expand_score_request,
    9006, 0, false);
  const auto spec_expand_aggregate = outbox.form_aggregate(
    1, protocol::PeerRpcType::stage2_expand_score_request,
    9007, 0, true);
  assert(auth_expand_aggregate.has_value() &&
         auth_expand_aggregate->logical_count == 1);
  assert(spec_expand_aggregate.has_value() &&
         spec_expand_aggregate->logical_count == 1);
  assert(outbox.discard_aggregate(9006));
  assert(outbox.discard_aggregate(9007));
  assert(outbox.size() == 0);
}

void test_leased_partial_cancel_retries_only_live_members() {
  detail::Stage2HomeRpcOutbox outbox(
    8, 4, 5, 32, 256, 256, 1u << 20);
  const std::vector<std::uint32_t> query_index{0};
  auto first = score_request(query_index, 1, 14000, 121);
  auto second = score_request(query_index, 1, 15000, 131);
  assert(outbox.try_enqueue(dispatch(
           61, 3, protocol::PeerRpcType::stage2_score_many_request,
           1, first)) == detail::Stage2HomeRpcEnqueueResult::enqueued);
  assert(outbox.try_enqueue(dispatch(
           62, 3, protocol::PeerRpcType::stage2_score_many_request,
           1, second)) == detail::Stage2HomeRpcEnqueueResult::enqueued);
  const auto aggregate = outbox.form_aggregate(
    3, protocol::PeerRpcType::stage2_score_many_request,
    9010, 0, false);
  assert(aggregate.has_value() && aggregate->logical_count == 2);
  const std::vector<byte_t> original_wire = claim_for_response(
    outbox, 9010, aggregate->request_bytes);

  std::vector<byte_t> response(
    protocol::stage2_score_many_response_bytes(2), byte_t{0});
  auto* header = reinterpret_cast<protocol::PeerRpcHeader*>(response.data());
  *header = {
    .magic = protocol::kPeerRpcMagic,
    .version = protocol::kPeerRpcVersion,
    .type = static_cast<std::uint32_t>(
      protocol::PeerRpcType::stage2_score_many_response),
    .source_shard = 3,
    .item_count = 2,
    .request_id = 9010,
    .status = static_cast<std::uint32_t>(protocol::InsertStatus::ok),
  };
  auto* results = protocol::stage2_score_many_results(response.data());
  results[0] = {
    .pointer_raw = 14000,
    .generation = 200,
    .search_index = 20,
    .disposition = static_cast<std::uint32_t>(
      protocol::Stage2HomeDisposition::stable),
    .distance = static_cast<distance_t>(1),
  };
  results[1] = {
    .pointer_raw = 15000,
    .generation = 200,
    .search_index = 20,
    .disposition = static_cast<std::uint32_t>(
      protocol::Stage2HomeDisposition::stable),
    .distance = static_cast<distance_t>(2),
  };

  const auto first_demux = outbox.demultiplex_response(9010, response);
  assert(first_demux.has_value() &&
         first_demux->logical_responses.size() == 2);
  // Model a context reset between response validation and the transactional
  // registry publication. The immutable outer request is retried, but the
  // cancelled logical member must be suppressed on the next demultiplex.
  assert(outbox.cancel_logical(62));
  assert(outbox.release_demux(first_demux->lease));
  const auto retry = outbox.next_retry_wire_request(
    3, protocol::PeerRpcType::stage2_score_many_request, false);
  assert(retry.has_value() && *retry == 9010);
  const std::vector<byte_t> retry_wire = claim_for_response(
    outbox, 9010, aggregate->request_bytes);
  assert(retry_wire == original_wire);
  const auto second_demux = outbox.demultiplex_response(9010, response);
  assert(second_demux.has_value() &&
         second_demux->logical_responses.size() == 1);
  assert(second_demux->logical_responses[0].logical_request_id == 61);
  assert(outbox.finish_success(second_demux->lease));
  assert(outbox.size() == 0 && outbox.aggregate_size() == 0);
}

}  // namespace

int main() {
  VamanaNode::init_static_storage(128, 96, VectorDType::uint8);
  test_expand_score_exact_wire_combine_and_compact_demux();
  test_score_many_query_rebase_and_exact_demux();
  test_partition_bounds_retry_and_cancellation();
  test_leased_partial_cancel_retries_only_live_members();
  return 0;
}
