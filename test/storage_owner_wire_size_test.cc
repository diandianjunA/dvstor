#include <cassert>
#include <limits>

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

  VamanaNode::init_static_storage(128, 96, VectorDType::uint8);
  const size_t expected = sizeof(InsertBatchRequestHeader) +
    4 * sizeof(node_t) + 4 * sizeof(u32) + 4 * 128;
  assert(insert_batch_request_bytes(4) == expected);
  return 0;
}
