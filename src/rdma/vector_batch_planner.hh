#pragma once

#include <algorithm>
#include <limits>

#include <library/types.hh>

namespace rdma::vamana {

struct VectorReadChunkPlan {
  u32 memory_node{};
  u32 qp_index{};
  u32 request_offset{};
  u32 request_count{};
};

struct VectorReadBatchPlan {
  vec<VectorReadChunkPlan> chunks;
  vec<u32> request_order;
  u32 active_nodes{};
  u32 active_qps{};
  u32 max_chain_wrs{};
};

struct VectorReadPlannerScratch {
  vec<vec<u32>> requests_by_node;
  vec<vec<u32>> legacy_per_qp;
  vec<vec<bool>> used_qps;
  vec<u64> projected_load;
};

// Pure planning helper kept independent from verbs so the balancing policy can
// be tested without RDMA hardware.
inline void plan_vector_read_batch(
    const vec<u32>& request_nodes,
    const vec<u32>& qp_counts,
    const vec<vec<u32>>& outstanding_wrs,
    const vec<u32>& tie_breakers,
    u32 max_chain_wrs,
    bool adaptive,
    VectorReadBatchPlan& plan,
    VectorReadPlannerScratch& scratch) {
  plan.chunks.clear();
  plan.request_order.clear();
  plan.active_nodes = 0;
  plan.active_qps = 0;
  plan.max_chain_wrs = 0;
  if (request_nodes.empty()) return;

  const u32 num_nodes = static_cast<u32>(qp_counts.size());
  scratch.requests_by_node.resize(num_nodes);
  scratch.used_qps.resize(num_nodes);
  for (u32 node = 0; node < num_nodes; ++node) {
    scratch.requests_by_node[node].clear();
    scratch.used_qps[node].assign(qp_counts[node], false);
  }
  for (u32 i = 0; i < request_nodes.size(); ++i) {
    const u32 node = request_nodes[i];
    if (node < num_nodes) scratch.requests_by_node[node].push_back(i);
  }

  for (u32 node = 0; node < num_nodes; ++node) {
    const auto& indices = scratch.requests_by_node[node];
    if (indices.empty() || qp_counts[node] == 0) continue;
    ++plan.active_nodes;

    if (!adaptive) {
      scratch.legacy_per_qp.resize(qp_counts[node]);
      for (u32 qp = 0; qp < qp_counts[node]; ++qp) {
        scratch.legacy_per_qp[qp].clear();
      }
      for (u32 i = 0; i < indices.size(); ++i) {
        scratch.legacy_per_qp[i % qp_counts[node]].push_back(indices[i]);
      }
      for (u32 qp = 0; qp < qp_counts[node]; ++qp) {
        const auto& per_qp = scratch.legacy_per_qp[qp];
        if (per_qp.empty()) continue;
        scratch.used_qps[node][qp] = true;
        plan.max_chain_wrs = std::max<u32>(
            plan.max_chain_wrs, static_cast<u32>(per_qp.size()));
        const u32 offset = static_cast<u32>(plan.request_order.size());
        plan.request_order.insert(plan.request_order.end(), per_qp.begin(), per_qp.end());
        plan.chunks.push_back(
            {node, qp, offset, static_cast<u32>(per_qp.size())});
      }
      continue;
    }

    // QP0 remains the low-latency control lane when a bulk lane exists.
    const u32 first_bulk_qp = qp_counts[node] > 1 ? 1 : 0;
    const u32 bulk_qps = qp_counts[node] - first_bulk_qp;
    scratch.projected_load.assign(qp_counts[node], 0);
    for (u32 qp = first_bulk_qp; qp < qp_counts[node]; ++qp) {
      if (node < outstanding_wrs.size() && qp < outstanding_wrs[node].size()) {
        scratch.projected_load[qp] = outstanding_wrs[node][qp];
      }
    }

    const u32 chain_limit = std::max<u32>(1, max_chain_wrs);
    const u32 tie = node < tie_breakers.size() ? tie_breakers[node] : 0;
    for (u32 begin = 0; begin < indices.size(); begin += chain_limit) {
      const u32 count = std::min<u32>(chain_limit,
                                      static_cast<u32>(indices.size()) - begin);
      u32 best_qp = first_bulk_qp;
      u64 best_load = std::numeric_limits<u64>::max();
      u32 best_rank = std::numeric_limits<u32>::max();
      for (u32 qp = first_bulk_qp; qp < qp_counts[node]; ++qp) {
        const u32 local = qp - first_bulk_qp;
        const u32 rank = (local + bulk_qps - (tie % bulk_qps)) % bulk_qps;
        if (scratch.projected_load[qp] < best_load ||
            (scratch.projected_load[qp] == best_load && rank < best_rank)) {
          best_qp = qp;
          best_load = scratch.projected_load[qp];
          best_rank = rank;
        }
      }

      const u32 offset = static_cast<u32>(plan.request_order.size());
      plan.request_order.insert(plan.request_order.end(),
                                indices.begin() + begin,
                                indices.begin() + begin + count);
      plan.chunks.push_back({node, best_qp, offset, count});
      scratch.projected_load[best_qp] += count;
      scratch.used_qps[node][best_qp] = true;
      plan.max_chain_wrs = std::max(plan.max_chain_wrs, count);
    }
  }

  for (const auto& node_qps : scratch.used_qps) {
    plan.active_qps += static_cast<u32>(
        std::count(node_qps.begin(), node_qps.end(), true));
  }
}

inline VectorReadBatchPlan plan_vector_read_batch(
    const vec<u32>& request_nodes,
    const vec<u32>& qp_counts,
    const vec<vec<u32>>& outstanding_wrs,
    const vec<u32>& tie_breakers,
    u32 max_chain_wrs,
    bool adaptive) {
  VectorReadBatchPlan plan;
  VectorReadPlannerScratch scratch;
  plan_vector_read_batch(request_nodes, qp_counts, outstanding_wrs, tie_breakers,
                         max_chain_wrs, adaptive, plan, scratch);
  return plan;
}

}  // namespace rdma::vamana
