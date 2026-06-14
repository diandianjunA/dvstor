#include <cassert>
#include <cstddef>
#include <vector>

#include "service/storage_owner_protocol.hh"

int main() {
  using namespace service::storage_owner;

  constexpr u32 item_count = 4;
  std::vector<byte_t> insert_response(insert_batch_response_bytes(item_count));
  auto* insert_header = reinterpret_cast<InsertBatchResponseHeader*>(insert_response.data());
  insert_header->item_count = item_count;
  u32* statuses = response_statuses(insert_response.data());
  MutationResult* mutation_results = response_mutation_results(insert_response.data(), item_count);
  auto* breakdown = response_breakdown(insert_response.data(), item_count);
  statuses[0] = static_cast<u32>(MutationStatus::ok);
  mutation_results[0].new_rptr_raw = 0x1234;
  mutation_results[0].old_rptr_raw = 0x5678;
  mutation_results[0].generation = 9;
  breakdown->storage_owner_write_node_ns = 11;
  assert(reinterpret_cast<byte_t*>(statuses) ==
         insert_response.data() + sizeof(InsertBatchResponseHeader));
  assert(reinterpret_cast<byte_t*>(mutation_results) ==
         reinterpret_cast<byte_t*>(statuses + item_count));
  assert(reinterpret_cast<byte_t*>(breakdown) ==
         reinterpret_cast<byte_t*>(mutation_results + item_count));
  assert(reinterpret_cast<byte_t*>(breakdown + 1) ==
         insert_response.data() + insert_response.size());
  assert(response_mutation_results(insert_response.data(), item_count)[0].generation == 9);
  assert(static_cast<u32>(InsertStatus::overloaded) == 2);

  std::vector<byte_t> repair(reverse_update_request_bytes(item_count));
  auto* repair_header = reinterpret_cast<PeerRpcHeader*>(repair.data());
  repair_header->item_count = item_count;
  ReverseUpdateOp* ops = reverse_update_ops(repair.data());
  ops[0].target_raw = 0x1111;
  ops[0].candidate_raw = 0x2222;
  ops[0].candidate_generation = 7;
  ops[0].reserved = kReverseUpdatePriority | kReverseUpdateReachability;
  ops[0].source_insert_id = 42;
  assert(reinterpret_cast<byte_t*>(ops) == repair.data() + sizeof(PeerRpcHeader));
  assert(reinterpret_cast<byte_t*>(ops + item_count) == repair.data() + repair.size());
  assert(reverse_update_ops(repair.data())[0].candidate_generation == 7);
  assert(reverse_update_ops(repair.data())[0].reserved ==
         (kReverseUpdatePriority | kReverseUpdateReachability));
  assert(reverse_update_ops(repair.data())[0].source_insert_id == 42);
}
