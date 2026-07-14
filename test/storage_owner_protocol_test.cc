#include <cassert>
#include <cstring>
#include <vector>

#include "service/storage_owner_protocol.hh"

int main() {
  namespace protocol = service::storage_owner;
  VamanaNode::init_static_storage(128, 96, VectorDType::uint8);

  constexpr u32 item_count = 2;
  const size_t response_bytes = protocol::stitch_search_response_bytes(item_count);
  std::vector<byte_t> response(response_bytes, 0);
  auto* header = reinterpret_cast<protocol::PeerRpcHeader*>(response.data());
  header->magic = protocol::kPeerRpcMagic;
  header->version = protocol::kPeerRpcVersion;
  header->item_count = item_count;

  u32* counts = protocol::stitch_search_response_counts(response.data());
  auto* candidates = protocol::stitch_search_response_candidates(
    response.data(), item_count);
  byte_t* vectors = protocol::stitch_search_response_candidate_vectors(
    response.data(), item_count);

  counts[0] = 1;
  candidates[0].raw = 0x1234;
  candidates[0].generation = 7;
  std::memset(vectors, 0x5a, VamanaNode::vector_bytes());

  assert(header->version == protocol::kPeerRpcVersion);
  assert(reinterpret_cast<byte_t*>(candidates) >=
         reinterpret_cast<byte_t*>(counts + item_count));
  assert(vectors >= reinterpret_cast<byte_t*>(
                      candidates + static_cast<size_t>(item_count) * VamanaNode::R));
  assert(vectors + static_cast<size_t>(item_count) * VamanaNode::R *
                     VamanaNode::vector_bytes() ==
         response.data() + response.size());
  assert(candidates[0].raw == 0x1234);
  assert(candidates[0].generation == 7);
  assert(vectors[0] == 0x5a);
  assert(vectors[VamanaNode::vector_bytes() - 1] == 0x5a);
  return 0;
}
