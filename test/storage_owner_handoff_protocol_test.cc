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

  std::vector<byte_t> response(search_handoff_response_bytes(beam_count, visited_count));
  auto* response_header = reinterpret_cast<SearchHandoffResponseHeader*>(response.data());
  response_header->updated_beam_count = beam_count;
  response_header->new_visited_count = visited_count;
  assert(reinterpret_cast<byte_t*>(handoff_response_beam(response_header)) ==
         response.data() + sizeof(SearchHandoffResponseHeader));
  assert(handoff_response_visited(response_header, beam_count) +
           visited_count * sizeof(u64) == response.data() + response.size());
  assert(static_cast<u32>(InsertStatus::overloaded) == 2);
}
