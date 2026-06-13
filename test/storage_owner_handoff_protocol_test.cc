#include <cassert>
#include <cstddef>
#include <vector>

#include "service/storage_owner_protocol.hh"

int main() {
  using namespace service::storage_owner;

  constexpr u32 beam_count = 3;
  constexpr u32 visited_count = 5;
  constexpr u32 vector_bytes = 16;
  std::vector<byte_t> request(
    search_handoff_request_bytes(beam_count, visited_count, vector_bytes));
  auto* request_header = reinterpret_cast<SearchHandoffRequestHeader*>(request.data());
  request_header->rpc.item_count = beam_count;
  request_header->visited_count = visited_count;
  request_header->vector_bytes = vector_bytes;

  assert(handoff_query_vector(request_header) ==
         request.data() + sizeof(SearchHandoffRequestHeader));
  assert(reinterpret_cast<byte_t*>(handoff_request_beam(request_header, vector_bytes)) ==
         handoff_query_vector(request_header) + vector_bytes);
  assert(handoff_request_visited(request_header, vector_bytes, beam_count) +
           visited_count * sizeof(u64) == request.data() + request.size());

  constexpr u32 uint8_stored_vector_bytes = 128;
  constexpr u32 float_query_wire_bytes = 128 * sizeof(float);
  static_assert(float_query_wire_bytes > uint8_stored_vector_bytes);
  std::vector<byte_t> uint8_request(
    search_handoff_request_bytes(beam_count, visited_count, float_query_wire_bytes));
  auto* uint8_request_header = reinterpret_cast<SearchHandoffRequestHeader*>(uint8_request.data());
  uint8_request_header->rpc.item_count = beam_count;
  uint8_request_header->visited_count = visited_count;
  uint8_request_header->vector_bytes = float_query_wire_bytes;
  assert(reinterpret_cast<byte_t*>(handoff_request_beam(uint8_request_header, float_query_wire_bytes)) ==
         handoff_query_vector(uint8_request_header) + float_query_wire_bytes);
  assert(handoff_request_visited(uint8_request_header, float_query_wire_bytes, beam_count) +
           visited_count * sizeof(u64) == uint8_request.data() + uint8_request.size());

  std::vector<byte_t> response(search_handoff_response_bytes(beam_count, visited_count));
  auto* response_header = reinterpret_cast<SearchHandoffResponseHeader*>(response.data());
  response_header->updated_beam_count = beam_count;
  response_header->new_visited_count = visited_count;
  assert(reinterpret_cast<byte_t*>(handoff_response_beam(response_header)) ==
         response.data() + sizeof(SearchHandoffResponseHeader));
  assert(handoff_response_visited(response_header, beam_count) +
           visited_count * sizeof(u64) == response.data() + response.size());
  constexpr u32 item_count = 4;
  std::vector<byte_t> insert_response(insert_batch_response_bytes(item_count));
  auto* insert_header = reinterpret_cast<InsertBatchResponseHeader*>(insert_response.data());
  insert_header->item_count = item_count;
  u32* statuses = response_statuses(insert_response.data());
  MutationResult* mutation_results = response_mutation_results(insert_response.data(), item_count);
  auto* breakdown = response_breakdown(insert_response.data(), item_count);
  u32* invalidation_count = response_invalidation_count(insert_response.data(), item_count);
  u64* invalidated = response_invalidated_raws(insert_response.data(), item_count);
  statuses[0] = static_cast<u32>(MutationStatus::ok);
  mutation_results[0].new_rptr_raw = 0x1234;
  mutation_results[0].old_rptr_raw = 0x5678;
  mutation_results[0].generation = 9;
  breakdown->storage_owner_write_node_ns = 11;
  *invalidation_count = 1;
  invalidated[0] = 0x9abc;
  assert(reinterpret_cast<byte_t*>(statuses) ==
         insert_response.data() + sizeof(InsertBatchResponseHeader));
  assert(reinterpret_cast<byte_t*>(mutation_results) ==
         reinterpret_cast<byte_t*>(statuses + item_count));
  assert(reinterpret_cast<byte_t*>(breakdown) ==
         reinterpret_cast<byte_t*>(mutation_results + item_count));
  assert(reinterpret_cast<byte_t*>(invalidated + response_invalidation_capacity(item_count)) ==
         insert_response.data() + insert_response.size());
  assert(response_mutation_results(insert_response.data(), item_count)[0].generation == 9);
  assert(static_cast<u32>(InsertStatus::overloaded) == 2);
}
