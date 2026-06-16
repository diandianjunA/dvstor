#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "rdma/rdma_send_chain.hh"
#include "rdma/vector_batch_planner.hh"

namespace {

void check(bool condition) {
  if (!condition) throw std::runtime_error("vector batch planner check failed");
}

void assert_all_requests_once(const rdma::vamana::VectorReadBatchPlan& plan,
                              u32 request_count) {
  vec<u32> seen(request_count, 0);
  for (const auto& chunk : plan.chunks) {
    for (u32 i = 0; i < chunk.request_count; ++i) {
      const u32 index = plan.request_order[chunk.request_offset + i];
      check(index < request_count);
      ++seen[index];
    }
  }
  check(std::all_of(seen.begin(), seen.end(), [](u32 count) { return count == 1; }));
}

void test_adaptive_hot_shard() {
  vec<u32> nodes(100, 0);
  const auto plan = rdma::vamana::plan_vector_read_batch(
      nodes, {4}, {{0, 0, 0, 0}}, {0}, 32, true);
  check(plan.active_nodes == 1);
  check(plan.active_qps == 3);
  check(plan.chunks.size() == 4);
  check(plan.max_chain_wrs == 32);
  for (const auto& chunk : plan.chunks) {
    check(chunk.qp_index >= 1 && chunk.qp_index <= 3);
    check(chunk.request_count <= 32);
  }
  assert_all_requests_once(plan, nodes.size());
}

void test_adaptive_uses_least_loaded_qp() {
  vec<u32> nodes(32, 0);
  const auto plan = rdma::vamana::plan_vector_read_batch(
      nodes, {4}, {{0, 64, 0, 0}}, {0}, 32, true);
  check(plan.chunks.size() == 1);
  check(plan.chunks.front().qp_index == 2);
}

void test_single_qp_chunks_at_limit() {
  vec<u32> nodes(41, 0);
  const auto plan = rdma::vamana::plan_vector_read_batch(
      nodes, {1}, {{0}}, {0}, 16, true);
  check(plan.active_qps == 1);
  check(plan.chunks.size() == 3);
  check(plan.chunks[0].request_count == 16);
  check(plan.chunks[1].request_count == 16);
  check(plan.chunks[2].request_count == 9);
  assert_all_requests_once(plan, nodes.size());
}

void test_legacy_preserves_per_qp_chains() {
  vec<u32> nodes(100, 0);
  const auto plan = rdma::vamana::plan_vector_read_batch(
      nodes, {4}, {{0, 0, 0, 0}}, {0}, 8, false);
  check(plan.active_qps == 4);
  check(plan.chunks.size() == 4);
  check(plan.max_chain_wrs == 25);
  for (const auto& chunk : plan.chunks) check(chunk.request_count == 25);
  assert_all_requests_once(plan, nodes.size());
}

void test_multiple_nodes() {
  vec<u32> nodes;
  for (u32 node = 0; node < 5; ++node) {
    nodes.insert(nodes.end(), 20, node);
  }
  const auto plan = rdma::vamana::plan_vector_read_batch(
      nodes, {4, 4, 4, 4, 4},
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
       {0, 0, 0, 0}, {0, 0, 0, 0}},
      {0, 1, 2, 0, 1}, 32, true);
  check(plan.active_nodes == 5);
  check(plan.active_qps == 5);
  check(plan.chunks.size() == 5);
  assert_all_requests_once(plan, nodes.size());
}

void test_send_chain_partial_post_retry() {
  ibv_send_wr wrs[3]{};
  wrs[0].next = &wrs[1];
  wrs[1].next = &wrs[2];
  u32 calls = 0;
  u32 polls = 0;
  const auto result = rdma::post_send_chain_with_retry(
      wrs,
      [&](ibv_send_wr* first, ibv_send_wr** bad) {
        ++calls;
        if (calls == 1) {
          check(first == &wrs[0]);
          *bad = &wrs[1];
          return ENOMEM;
        }
        check(first == &wrs[1]);
        return 0;
      },
      [&] { ++polls; });
  check(result.success);
  check(result.post_calls == 2);
  check(result.retries == 1);
  check(result.error == 0);
  check(polls == 1);
}

void test_send_chain_fatal_error() {
  ibv_send_wr wr{};
  u32 polls = 0;
  const auto result = rdma::post_send_chain_with_retry(
      &wr,
      [](ibv_send_wr*, ibv_send_wr**) { return EINVAL; },
      [&] { ++polls; });
  check(!result.success);
  check(result.post_calls == 1);
  check(result.retries == 0);
  check(result.error == EINVAL);
  check(polls == 0);
}

}  // namespace

int main() {
  test_adaptive_hot_shard();
  test_adaptive_uses_least_loaded_qp();
  test_single_qp_chunks_at_limit();
  test_legacy_preserves_per_qp_chains();
  test_multiple_nodes();
  test_send_chain_partial_post_retry();
  test_send_chain_fatal_error();
  std::cout << "vector_batch_planner_test passed\n";
  return 0;
}
