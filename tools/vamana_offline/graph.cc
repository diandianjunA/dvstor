#include "tools/vamana_offline/graph.hh"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_set>

#include <library/utils.hh>

#include "tools/vamana_offline/progress.hh"

namespace tools::vamana_offline {

namespace {

class LocalIdSet {
public:
  explicit LocalIdSet(size_t expected_items) {
    size_t capacity = 1;
    while (capacity < expected_items * 2) capacity <<= 1;
    table_.assign(capacity, kEmpty);
    mask_ = capacity - 1;
  }

  bool contains(u32 value) const {
    size_t pos = hash(value) & mask_;
    for (;;) {
      const u32 current = table_[pos];
      if (current == kEmpty) return false;
      if (current == value) return true;
      pos = (pos + 1) & mask_;
    }
  }

  bool insert(u32 value) {
    size_t pos = hash(value) & mask_;
    for (;;) {
      const u32 current = table_[pos];
      if (current == value) return false;
      if (current == kEmpty) {
        table_[pos] = value;
        return true;
      }
      pos = (pos + 1) & mask_;
    }
  }

private:
  static constexpr u32 kEmpty = std::numeric_limits<u32>::max();

  static size_t hash(u32 value) {
    uint64_t x = value;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return static_cast<size_t>(x);
  }

  vec<u32> table_;
  size_t mask_{0};
};

bool candidate_id_less(const std::pair<float, u32>& a, const std::pair<float, u32>& b) {
  if (a.first != b.first) return a.first < b.first;
  return a.second < b.second;
}

u64 mix_seed(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

void sort_and_unique_candidates(vec<std::pair<float, u32>>& candidates) {
  std::sort(candidates.begin(), candidates.end(), candidate_id_less);
  vec<std::pair<float, u32>> unique;
  unique.reserve(candidates.size());
  LocalIdSet seen(std::max<size_t>(1024, candidates.size()));
  for (const auto& item : candidates) {
    if (seen.insert(item.second)) {
      unique.push_back(item);
    }
  }
  candidates.swap(unique);
}

void insert_sorted_beam(vec<std::pair<float, u32>>& beam,
                        std::pair<float, u32> candidate,
                        size_t beam_width) {
  if (beam.size() == beam_width && !candidate_id_less(candidate, beam.back())) return;
  const auto pos = std::lower_bound(beam.begin(), beam.end(), candidate, candidate_id_less);
  beam.insert(pos, candidate);
  if (beam.size() > beam_width) beam.pop_back();
}

}  // namespace

void VamanaGraph::init(size_t n, u32 d, u32 max_degree, size_t requested_lock_stripes) {
  lib_assert(max_degree <= std::numeric_limits<u8>::max(), "offline graph degree must fit in u8");
  num_nodes = n;
  dim = d;
  R = max_degree;
  medoid = 0;
  neighbors.assign(n * static_cast<size_t>(R), kEmptyNeighbor);
  degrees.assign(n, 0);
  lock_stripe_count = 1;
  const size_t target = std::max<size_t>(1, std::min(requested_lock_stripes, n));
  while (lock_stripe_count < target) lock_stripe_count <<= 1;
  lock_stripes.reset(new std::atomic_flag[lock_stripe_count]);
  for (size_t i = 0; i < lock_stripe_count; ++i) {
    lock_stripes[i].clear(std::memory_order_relaxed);
  }
  std::cerr << "offline graph memory: neighbors=" << neighbors.size() * sizeof(u32)
            << " bytes, degrees=" << degrees.size() * sizeof(u8)
            << " bytes, lock_stripes=" << lock_stripe_count << "\n";
}

void VamanaGraph::copy_neighbors(size_t node, vec<u32>& out) const {
  out.clear();
  const size_t base = offset(node);
  const u8 count = degrees[node];
  out.reserve(count);
  for (u8 i = 0; i < count; ++i) {
    const u32 nbr = neighbors[base + i];
    if (nbr != kEmptyNeighbor) out.push_back(nbr);
  }
}

bool VamanaGraph::contains_neighbor_unlocked(size_t node, u32 neighbor) const {
  const size_t base = offset(node);
  const u8 count = degrees[node];
  for (u8 i = 0; i < count; ++i) {
    if (neighbors[base + i] == neighbor) return true;
  }
  return false;
}

bool VamanaGraph::try_append_neighbor_unlocked(size_t node, u32 neighbor) {
  const u8 count = degrees[node];
  if (count >= R) return false;
  const size_t base = offset(node);
  for (u8 i = 0; i < count; ++i) {
    if (neighbors[base + i] == neighbor) return false;
  }
  neighbors[base + count] = neighbor;
  degrees[node] = static_cast<u8>(count + 1);
  return true;
}

void VamanaGraph::set_neighbors(size_t node, const vec<u32>& new_neighbors) {
  const size_t base = offset(node);
  const u8 count = static_cast<u8>(std::min<size_t>(new_neighbors.size(), R));
  for (u8 i = 0; i < count; ++i) {
    neighbors[base + i] = new_neighbors[i];
  }
  for (u32 i = count; i < R; ++i) {
    neighbors[base + i] = kEmptyNeighbor;
  }
  degrees[node] = count;
}

void VamanaGraph::lock_node(size_t node) {
  auto& flag = lock_stripes[node & (lock_stripe_count - 1)];
  while (flag.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
  }
}

void VamanaGraph::unlock_node(size_t node) {
  lock_stripes[node & (lock_stripe_count - 1)].clear(std::memory_order_release);
}

size_t compute_medoid(const Dataset& dataset, bool ip_distance) {
  const size_t n = dataset.size();
  const u32 dim = dataset.dim;
  const size_t sample_size = std::min<size_t>(n, 10000);
  vec<u32> sample_indices(sample_size);
  for (size_t i = 0; i < sample_size; ++i) sample_indices[i] = static_cast<u32>(i);
  if (sample_size < n) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<u32> dist(0, static_cast<u32>(n - 1));
    for (size_t i = 0; i < sample_size; ++i) sample_indices[i] = dist(rng);
  }

  vec<float> centroid(dim, 0.0f);
  vec<float> decoded(dim);
  for (u32 idx : sample_indices) {
    dataset_decode_vector(dataset, idx, decoded.data());
    for (u32 d = 0; d < dim; ++d) centroid[d] += decoded[d];
  }
  for (u32 d = 0; d < dim; ++d) centroid[d] /= static_cast<float>(sample_size);

  size_t best = 0;
  float best_dist = std::numeric_limits<float>::max();
  for (size_t i = 0; i < n; ++i) {
    const float dist = dataset_distance_float_query(dataset, centroid.data(), i, ip_distance);
    if (dist < best_dist) {
      best_dist = dist;
      best = i;
    }
  }
  return best;
}

vec<std::pair<float, u32>> beam_search(VamanaGraph& graph,
                                       const Dataset& dataset,
                                       u32 query_id,
                                       u32 beam_width,
                                       bool ip_distance) {
  vec<std::pair<float, u32>> all_visited;
  vec<std::pair<float, u32>> beam;
  const size_t expected_seen = std::max<size_t>(1024, static_cast<size_t>(beam_width) * graph.R + graph.R + 1);
  LocalIdSet visited(expected_seen);
  LocalIdSet expanded(std::max<size_t>(1024, beam_width * 2));

  const float medoid_dist = dataset_distance(dataset, query_id, graph.medoid, ip_distance);
  beam.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  all_visited.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  visited.insert(static_cast<u32>(graph.medoid));

  vec<u32> nbrs;
  vec<u32> unvisited;
  while (true) {
    ssize_t best_pos = -1;
    for (size_t i = 0; i < beam.size(); ++i) {
      if (!expanded.contains(beam[i].second)) {
        best_pos = static_cast<ssize_t>(i);
        break;
      }
    }
    if (best_pos < 0) break;

    const u32 best_node = beam[best_pos].second;
    expanded.insert(best_node);

    {
      NodeLockGuard lock(graph, best_node);
      graph.copy_neighbors(best_node, nbrs);
    }

    unvisited.clear();
    for (u32 nbr : nbrs) {
      if (!visited.insert(nbr)) continue;
      unvisited.push_back(nbr);
    }

    for (u32 nbr : unvisited) {
      const float d = dataset_distance(dataset, query_id, nbr, ip_distance);
      insert_sorted_beam(beam, {d, nbr}, beam_width);
      all_visited.push_back({d, nbr});
    }

  }

  std::sort(all_visited.begin(), all_visited.end(), candidate_id_less);
  return all_visited;
}

vec<std::pair<float, u32>> beam_search_float_query(VamanaGraph& graph,
                                                   const Dataset& dataset,
                                                   const float* query,
                                                   u32 beam_width,
                                                   bool ip_distance) {
  vec<std::pair<float, u32>> all_visited;
  vec<std::pair<float, u32>> beam;
  const size_t expected_seen = std::max<size_t>(1024, static_cast<size_t>(beam_width) * graph.R + graph.R + 1);
  LocalIdSet visited(expected_seen);
  LocalIdSet expanded(std::max<size_t>(1024, beam_width * 2));

  const float medoid_dist = dataset_distance_float_query(dataset, query, graph.medoid, ip_distance);
  beam.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  all_visited.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  visited.insert(static_cast<u32>(graph.medoid));

  vec<u32> nbrs;
  while (true) {
    ssize_t best_pos = -1;
    for (size_t i = 0; i < beam.size(); ++i) {
      if (!expanded.contains(beam[i].second)) {
        best_pos = static_cast<ssize_t>(i);
        break;
      }
    }
    if (best_pos < 0) break;
    const u32 best_node = beam[best_pos].second;
    expanded.insert(best_node);
    {
      NodeLockGuard lock(graph, best_node);
      graph.copy_neighbors(best_node, nbrs);
    }
    for (u32 nbr : nbrs) {
      if (!visited.insert(nbr)) continue;
      const float d = dataset_distance_float_query(dataset, query, nbr, ip_distance);
      insert_sorted_beam(beam, {d, nbr}, beam_width);
      all_visited.push_back({d, nbr});
    }
  }
  std::sort(all_visited.begin(), all_visited.end(), candidate_id_less);
  return all_visited;
}

vec<u32> robust_prune(const Dataset& dataset,
                      u32 source,
                      const vec<std::pair<float, u32>>& sorted_candidates,
                      float alpha,
                      u32 R,
                      bool ip_distance) {
  vec<u32> selected;
  selected.reserve(R);
  for (const auto& [cand_dist, cand_id] : sorted_candidates) {
    if (cand_id == source) continue;
    if (selected.size() >= R) break;

    bool pruned = false;
    for (u32 sel_id : selected) {
      const float d_sel_cand = dataset_distance(dataset, sel_id, cand_id, ip_distance);
      if (alpha * d_sel_cand <= cand_dist) {
        pruned = true;
        break;
      }
    }
    if (!pruned) selected.push_back(cand_id);
  }
  return selected;
}

void consolidate_reverse_edges(VamanaGraph& graph,
                               const Dataset& dataset,
                               const VamanaBuildConfig& config,
                               size_t num_threads,
                               float build_alpha) {
  const size_t n = graph.num_nodes;
  std::cerr << "building reverse-edge CSR for bulk consolidation...\n";

  std::unique_ptr<std::atomic<u32>[]> incoming_counts(new std::atomic<u32>[n]);
  parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t node, size_t) {
    incoming_counts[node].store(0, std::memory_order_relaxed);
  });

  parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t source, size_t) {
    const size_t base = graph.offset(source);
    const u8 degree = graph.degree(source);
    for (u8 i = 0; i < degree; ++i) {
      const u32 target = graph.neighbors[base + i];
      if (target < n && target != source) {
        incoming_counts[target].fetch_add(1, std::memory_order_relaxed);
      }
    }
  });

  vec<u64> incoming_offsets(n + 1, 0);
  for (size_t node = 0; node < n; ++node) {
    incoming_offsets[node + 1] = incoming_offsets[node] +
        incoming_counts[node].load(std::memory_order_relaxed);
  }
  const u64 total_incoming = incoming_offsets.back();
  std::cerr << "bulk reverse consolidation memory: incoming_edges=" << total_incoming
            << " edge_bytes=" << total_incoming * sizeof(u32)
            << " offset_bytes=" << incoming_offsets.size() * sizeof(u64)
            << " count_bytes=" << n * sizeof(std::atomic<u32>) << "\n";

  vec<u32> incoming_edges(static_cast<size_t>(total_incoming));
  parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t node, size_t) {
    incoming_counts[node].store(0, std::memory_order_relaxed);
  });
  parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t source, size_t) {
    const size_t base = graph.offset(source);
    const u8 degree = graph.degree(source);
    for (u8 i = 0; i < degree; ++i) {
      const u32 target = graph.neighbors[base + i];
      if (target >= n || target == source) continue;
      const u32 slot = incoming_counts[target].fetch_add(1, std::memory_order_relaxed);
      incoming_edges[static_cast<size_t>(incoming_offsets[target] + slot)] = static_cast<u32>(source);
    }
  });
  incoming_counts.reset();

  ProgressReporter reverse_progress{"Bulk reverse-edge consolidation", n};
  parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t node, size_t) {
    vec<u32> candidate_ids;
    const u8 outgoing_degree = graph.degree(node);
    const u64 incoming_begin = incoming_offsets[node];
    const u64 incoming_end = incoming_offsets[node + 1];
    candidate_ids.reserve(static_cast<size_t>(outgoing_degree) +
                          static_cast<size_t>(incoming_end - incoming_begin));

    const size_t base = graph.offset(node);
    for (u8 i = 0; i < outgoing_degree; ++i) {
      const u32 candidate = graph.neighbors[base + i];
      if (candidate != node && candidate < n) candidate_ids.push_back(candidate);
    }
    for (u64 pos = incoming_begin; pos < incoming_end; ++pos) {
      const u32 candidate = incoming_edges[static_cast<size_t>(pos)];
      if (candidate != node) candidate_ids.push_back(candidate);
    }

    std::sort(candidate_ids.begin(), candidate_ids.end());
    candidate_ids.erase(std::unique(candidate_ids.begin(), candidate_ids.end()), candidate_ids.end());

    vec<std::pair<float, u32>> prune_candidates;
    prune_candidates.reserve(candidate_ids.size());
    for (u32 candidate : candidate_ids) {
      prune_candidates.push_back({dataset_distance(dataset, node, candidate, config.ip_distance), candidate});
    }
    std::sort(prune_candidates.begin(), prune_candidates.end(), candidate_id_less);
    graph.set_neighbors(node, robust_prune(dataset, static_cast<u32>(node), prune_candidates,
                                           build_alpha, graph.R, config.ip_distance));
    reverse_progress.increment();
  });
  reverse_progress.finish();
}

void build_vamana_graph(VamanaGraph& graph,
                        const Dataset& dataset,
                        const VamanaBuildConfig& config) {
  const size_t n = dataset.size();
  const u32 R = config.R;
  const float alpha = static_cast<float>(config.alpha);
  const u32 beam_width = config.beam_width;
  const size_t num_threads = effective_thread_count(config.threads);

  graph.init(n, dataset.dim, R);

  std::cerr << "computing medoid...\n";
  graph.medoid = compute_medoid(dataset, config.ip_distance);
  std::cerr << "medoid: node " << graph.medoid << "\n";

  vec<u32> order(n);
  std::iota(order.begin(), order.end(), 0);
  const size_t seed = config.seed == -1 ? std::random_device{}() : static_cast<size_t>(config.seed);
  std::mt19937 rng(seed);
  std::shuffle(order.begin(), order.end(), rng);

  std::cerr << "initializing random R-regular graph (degree=" << R << ")...\n";
  parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t i, size_t) {
    const u64 node_seed = mix_seed(static_cast<u64>(seed) ^ static_cast<u64>(i));
    std::mt19937 init_rng(static_cast<u32>(node_seed ^ (node_seed >> 32)));
    std::uniform_int_distribution<u32> dist(0, static_cast<u32>(n - 1));
    vec<u32> initial;
    initial.reserve(R);
    while (initial.size() < R) {
      const u32 j = dist(init_rng);
      if (j == i) continue;
      bool dup = false;
      for (u32 existing : initial) {
        if (existing == j) { dup = true; break; }
      }
      if (!dup) initial.push_back(j);
    }
    graph.set_neighbors(i, initial);
  });

  const size_t signature_sample = std::min<size_t>(n, 4096);
  std::unordered_set<u64> neighbor_signatures;
  neighbor_signatures.reserve(signature_sample * 2);
  vec<u32> signature_neighbors;
  for (size_t i = 0; i < signature_sample; ++i) {
    graph.copy_neighbors(i, signature_neighbors);
    u64 signature = 1469598103934665603ULL;
    for (u32 neighbor : signature_neighbors) {
      signature ^= neighbor;
      signature *= 1099511628211ULL;
    }
    neighbor_signatures.insert(signature);
  }
  const double unique_signature_ratio = signature_sample == 0
      ? 1.0
      : static_cast<double>(neighbor_signatures.size()) / static_cast<double>(signature_sample);
  std::cerr << "random graph unique neighbor signature ratio=" << unique_signature_ratio
            << " sample=" << signature_sample << "\n";
  lib_assert(unique_signature_ratio >= 0.99,
             "random graph initialization produced too many duplicate neighbor lists");

  const float build_alpha = 1.0f;
  if (alpha > 1.0f + 1e-6f) {
    std::cerr << "note: using alpha=1.0 for construction (config alpha="
              << alpha << " stored in metadata)\n";
  }
  std::cerr << "reverse edge maintenance: cheap append plus bulk consolidation\n";

  ProgressReporter progress{"Building Vamana graph", n};
  parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t step, size_t) {
    const u32 node_idx = order[step];

    auto candidates = beam_search(graph, dataset, node_idx, beam_width, config.ip_distance);

    vec<u32> existing_neighbors;
    {
      NodeLockGuard lock(graph, node_idx);
      graph.copy_neighbors(node_idx, existing_neighbors);
    }
    for (u32 existing : existing_neighbors) {
      candidates.push_back({dataset_distance(dataset, node_idx, existing, config.ip_distance), existing});
    }
    sort_and_unique_candidates(candidates);

    vec<u32> new_neighbors = robust_prune(dataset, node_idx, candidates, build_alpha, R, config.ip_distance);
    {
      NodeLockGuard lock(graph, node_idx);
      graph.set_neighbors(node_idx, new_neighbors);
    }

    for (u32 nbr : new_neighbors) {
      NodeLockGuard lock(graph, nbr);
      graph.try_append_neighbor_unlocked(nbr, node_idx);
    }
    progress.increment();
  });
  progress.finish();

  consolidate_reverse_edges(graph, dataset, config, num_threads, build_alpha);

  size_t total_edges = 0;
  size_t max_edges = 0;
  size_t min_edges = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < n; ++i) {
    const size_t count = graph.degree(i);
    total_edges += count;
    max_edges = std::max(max_edges, count);
    min_edges = std::min(min_edges, count);
  }
  std::cerr << "graph stats: avg_degree=" << (static_cast<double>(total_edges) / n)
            << " max=" << max_edges << " min=" << min_edges << "\n";

  if (!config.skip_sanity_check) {
    const size_t n_queries = std::min<size_t>(200, n);
    const u32 topk = 10;
    std::mt19937 sample_rng(42);
    vec<u32> query_indices(n_queries);
    std::uniform_int_distribution<u32> qdist(0, static_cast<u32>(n - 1));
    for (u32& q : query_indices) q = qdist(sample_rng);

    size_t total_hits = 0;
    for (u32 qid : query_indices) {
      vec<std::pair<float, u32>> all_dists;
      all_dists.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        if (i == qid) continue;
        all_dists.push_back({dataset_distance(dataset, qid, i, config.ip_distance), static_cast<u32>(i)});
      }
      std::partial_sort(all_dists.begin(),
                        all_dists.begin() + std::min<size_t>(topk, all_dists.size()),
                        all_dists.end(),
                        candidate_id_less);
      std::unordered_set<u32> gt;
      for (size_t i = 0; i < topk && i < all_dists.size(); ++i) gt.insert(all_dists[i].second);

      auto results = beam_search(graph, dataset, qid, beam_width, config.ip_distance);
      size_t hits = 0;
      for (size_t i = 0; i < topk && i < results.size(); ++i) {
        if (gt.count(results[i].second)) ++hits;
      }
      total_hits += hits;
    }
    const double recall = static_cast<double>(total_hits) / static_cast<double>(n_queries * topk);
    std::cerr << "in-memory recall@" << topk << " (sample " << n_queries << "): " << recall << "\n";
  }
}

}  // namespace tools::vamana_offline
