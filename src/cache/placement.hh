#pragma once

#include <queue>

#include "cache/kmeans.hh"
#include "common/timing.hh"
#include "coroutine.hh"
#include "rdma/vamana_rdma_operations.hh"

template <class Distance>
class Placement {
public:
  using Centroids = Kmeans<Distance>::Centroids;
  using Nodes = Kmeans<Distance>::Nodes;

  struct MinHeapPlacementCompare {
    bool operator()(const auto& lhs, const auto& rhs) const { return lhs.second > rhs.second; }
  };

  using MinPlacement = std::priority_queue<std::pair<idx_t, distance_t>,
                                          vec<std::pair<idx_t, distance_t>>,
                                          MinHeapPlacementCompare>;

  Placement() = default;
  Placement(u32 k, const u_ptr<ComputeThread>& thread, timing::Timing& timing) {
    const auto t_placement_fetch = timing.create_enroll("placement_fetch");
    const auto t_placement_kmeans = timing.create_enroll("placement_kmeans");
    Nodes nodes;

    t_placement_fetch->start();
    {
      const auto coro = fetch_level(nodes, 500, thread);  // TODO
      while (!coro.handle.done()) {
        thread->poll_cq();
        if (thread->is_ready(0)) {
          coro.handle.resume();
        }
      }

      thread->reset();
    }
    t_placement_fetch->stop();

    t_placement_kmeans->start();
    if constexpr (query_router::BALANCED_KMEANS_WITH_HEURISTIC) {
      std::tie(centroids_, mapping_) = Kmeans<Distance>::run_and_optimize(nodes, k);  // optimized balanced k-means

    } else {  // default k-means
      auto [c, _, cluster_sizes] = Kmeans<Distance>::run_kmeans(nodes, k);

      centroids_ = c;
      mapping_.resize(k);

      for (idx_t i = 0; i < k; ++i) {
        mapping_[i] = i;
      }

      for (idx_t i = 0; i < k; ++i) {
        std::cerr << "  cluster " << i << ": " << cluster_sizes[i] << std::endl;
      }
    }

    t_placement_kmeans->stop();
  }

  MinPlacement closest_centroids(const span<element_t> query_components) const {
    MinPlacement placement;

    for (idx_t i = 0; i < centroids_.size(); ++i) {
      const f32 distance = Distance::dist(query_components, centroids_[i], VamanaNode::DIM);
      placement.push({mapping_[i], distance});
    }

    return placement;
  }

private:
  /**
   * @brief Fetches nodes via BFS from the medoid until we have at least k nodes.
   */
  static MinorCoroutine fetch_level(Nodes& nodes, u32 k, const u_ptr<ComputeThread>& thread) {
    RemotePtr medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread);
    if (medoid_ptr.is_null()) {
      co_return;
    }

    auto medoid_node = co_await rdma::vamana::read_vamana_node(medoid_ptr, thread);
    nodes.emplace_back(medoid_node);

    hashset_t<RemotePtr> visited_nodes;
    visited_nodes.insert(medoid_ptr);

    // BFS expansion from medoid
    for (idx_t iter = 0; iter < nodes.size() && nodes.size() < k; ++iter) {
      const auto& node = nodes[iter];
      const auto nlist = co_await rdma::vamana::read_vamana_neighbors(node->rptr, thread);

      for (const RemotePtr& r_ptr : nlist->view()) {
        if (!r_ptr.is_null() && !visited_nodes.contains(r_ptr)) {
          auto fetched = co_await rdma::vamana::read_vamana_node(r_ptr, thread);
          nodes.emplace_back(fetched);
          visited_nodes.insert(r_ptr);

          if (nodes.size() >= k) break;
        }
      }
    }

    std::cerr << "  fetched " << nodes.size() << " nodes via BFS from medoid" << std::endl;
  }

private:
  Centroids centroids_;
  vec<idx_t> mapping_;
};