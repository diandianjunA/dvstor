#include <cassert>
#include <limits>
#include <vector>

#include "service/storage_owner_protocol.hh"

int main() {
  using namespace service::storage_owner;
  assert(align_wire_u64(1) == 8);
  assert(align_wire_u64(std::numeric_limits<size_t>::max()) ==
         std::numeric_limits<size_t>::max());

  VamanaNode::init_static_storage(
    std::numeric_limits<u32>::max(), 1, VectorDType::uint8);
  assert(insert_batch_request_bytes(std::numeric_limits<u32>::max()) ==
         std::numeric_limits<size_t>::max());
  assert(mutation_batch_request_bytes(std::numeric_limits<u32>::max()) ==
         std::numeric_limits<size_t>::max());
  assert(stage1_execute_request_bytes(std::numeric_limits<u32>::max()) ==
         std::numeric_limits<size_t>::max());
  assert(stage2_expand_score_request_bytes(
           std::numeric_limits<u32>::max()) ==
         std::numeric_limits<size_t>::max());

  VamanaNode::init_static_storage(128, 96, VectorDType::uint8);
  const size_t expected = sizeof(InsertBatchRequestHeader) +
    4 * sizeof(node_t) + 4 * sizeof(u64) + 4 * sizeof(u32) + 4 * 128;
  assert(insert_batch_request_bytes(4) == expected);
  std::vector<byte_t> request(insert_batch_request_bytes(3), 0);
  auto* header = reinterpret_cast<InsertBatchRequestHeader*>(request.data());
  header->item_count = 3;
  auto* operations = request_operation_ids(request.data(), 3);
  operations[0] = 11;
  operations[1] = 12;
  operations[2] = 13;
  assert(reinterpret_cast<uintptr_t>(operations) % alignof(u64) == 0);
  assert(request_stage1_homes(request.data(), 3) ==
         reinterpret_cast<u32*>(operations + 3));
  const size_t stage2_request_expected = align_wire_u64(
      sizeof(PeerRpcHeader) + 4 * sizeof(Stage2ExpandScoreItem)) +
    4 * VamanaNode::vector_bytes();
  assert(stage2_expand_score_request_bytes(4) ==
         stage2_request_expected);
  const size_t stage2_response_max_expected = sizeof(PeerRpcHeader) +
    4 * sizeof(Stage2ExpandScoreResult) +
    4 * VamanaNode::graph_entry_capacity() *
      sizeof(Stage2ExpandScoreNeighbor);
  assert(stage2_expand_score_response_bytes(4) ==
         stage2_response_max_expected);
  const size_t stage2_response_compact_expected = sizeof(PeerRpcHeader) +
    4 * sizeof(Stage2ExpandScoreResult) +
    17 * sizeof(Stage2ExpandScoreNeighbor);
  assert(stage2_expand_score_response_bytes(4, 17) ==
         stage2_response_compact_expected);
  return 0;
}
