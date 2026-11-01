#include "tools/vamana_offline/graph.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_set>

#include <library/utils.hh>

#include "common/constants.hh"
#include "tools/vamana_offline/checkpoint.hh"
#include "tools/vamana_offline/local_id_set.hh"
#include "tools/vamana_offline/progress.hh"

namespace tools::vamana_offline {

namespace {

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

void VamanaGraph::init(size_t n, u32 d, u32 max_degree, size_t requested_lock_stripes,
                       size_t num_threads, u32 build_capacity) {
  lib_assert(n > 0, "offline graph requires at least one node");
  lib_assert(max_degree > 0 && max_degree <= kMaxSupportedGraphDegree,
             "offline graph degree exceeds the system-wide limit");
  const u32 resolved_capacity = build_capacity == 0 ? max_degree : build_capacity;
  lib_assert(resolved_capacity >= max_degree &&
                 resolved_capacity <= std::numeric_limits<u8>::max(),
             "offline graph build capacity must be between R and 255");
  lib_assert(n <= std::numeric_limits<size_t>::max() / static_cast<size_t>(resolved_capacity),
             "offline graph neighbor allocation size overflows");
  num_nodes = n;
  dim = d;
  R = max_degree;
  capacity = resolved_capacity;
  medoid = 0;
  // LargeBuffer mmap-reserves zeroed pages; the parallel fill below both
  // installs kEmptyNeighbor and first-touches the pages from all sockets
  // instead of leaving the whole array on one NUMA node.
  const size_t neighbor_slots = n * static_cast<size_t>(capacity);
  neighbors.reset(neighbor_slots);
  degrees.reset(n);  // mmap-zeroed: every degree starts at 0
  {
    constexpr size_t kFillBlock = size_t{1} << 20;
    const size_t blocks = (neighbor_slots + kFillBlock - 1) / kFillBlock;
    parallel_for(static_cast<size_t>(0), blocks, num_threads, [&](size_t block, size_t) {
      const size_t begin = block * kFillBlock;
      const size_t end = std::min(neighbor_slots, begin + kFillBlock);
      std::fill_n(neighbors.data() + begin, end - begin, kEmptyNeighbor);
    });
  }
  lock_stripe_count = 1;
  const size_t target = std::max<size_t>(1, std::min(requested_lock_stripes, n));
  while (lock_stripe_count < target) lock_stripe_count <<= 1;
  lock_stripes.reset(new std::atomic_flag[lock_stripe_count]);
  for (size_t i = 0; i < lock_stripe_count; ++i) {
    lock_stripes[i].clear(std::memory_order_relaxed);
  }
  std::cerr << "offline graph memory: neighbors=" << neighbors.size() * sizeof(u32)
            << " bytes, degrees=" << degrees.size() * sizeof(u8)
            << " bytes, lock_stripes=" << lock_stripe_count
            << " final_degree=" << R << " build_capacity=" << capacity << "\n";
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
  if (count >= capacity) return false;
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
  const u8 count = static_cast<u8>(std::min<size_t>(new_neighbors.size(), capacity));
  for (u8 i = 0; i < count; ++i) {
    neighbors[base + i] = new_neighbors[i];
  }
  for (u32 i = count; i < capacity; ++i) {
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

size_t compute_medoid(const Dataset& dataset, size_t num_threads) {
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

  const size_t workers =
    std::min(effective_thread_count(static_cast<u32>(num_threads)), n);
  vec<float> worker_best_dist(workers, std::numeric_limits<float>::max());
  vec<size_t> worker_best(workers, 0);
  parallel_for(static_cast<size_t>(0), n, workers, [&](size_t i, size_t tid) {
    const float dist = dataset_distance_float_query(dataset, centroid.data(), i);
    if (dist < worker_best_dist[tid]) {
      worker_best_dist[tid] = dist;
      worker_best[tid] = i;
    }
  });

  // Deterministic reduction: smallest distance wins, ties resolve to the
  // lowest node id exactly as the serial scan did.
  size_t best = 0;
  float best_dist = std::numeric_limits<float>::max();
  for (size_t tid = 0; tid < workers; ++tid) {
    if (worker_best_dist[tid] == std::numeric_limits<float>::max()) continue;
    if (worker_best_dist[tid] < best_dist ||
        (worker_best_dist[tid] == best_dist && worker_best[tid] < best)) {
      best_dist = worker_best_dist[tid];
      best = worker_best[tid];
    }
  }
  return best;
}

vec<std::pair<float, u32>> beam_search(VamanaGraph& graph,
                                       const Dataset& dataset,
                                       u32 query_id,
                                       u32 beam_width) {
  vec<std::pair<float, u32>> all_visited;
  thread_local vec<std::pair<float, u32>> beam;
  beam.clear();
  const size_t expected_seen = std::max<size_t>(1024, static_cast<size_t>(beam_width) * graph.capacity + graph.capacity + 1);
  LocalIdSet visited(expected_seen);
  LocalIdSet expanded(std::max<size_t>(
    1024, static_cast<size_t>(beam_width) * 2));

  const float medoid_dist = dataset_distance(dataset, query_id, graph.medoid);
  beam.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  visited.insert(static_cast<u32>(graph.medoid));

  thread_local vec<u32> nbrs;
  nbrs.clear();
  thread_local vec<u32> unvisited;
  unvisited.clear();
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
    all_visited.push_back(beam[best_pos]);

    {
      NodeLockGuard lock(graph, best_node);
      graph.copy_neighbors(best_node, nbrs);
    }

    unvisited.clear();
    for (u32 nbr : nbrs) {
      if (!visited.insert(nbr)) continue;
      unvisited.push_back(nbr);
    }

    for (size_t i = 0; i < unvisited.size(); ++i) {
      const u32 nbr = unvisited[i];
      if (i + 1 < unvisited.size()) {
        __builtin_prefetch(dataset.raw_vector(unvisited[i + 1]), 0, 1);
      }
      const float d = dataset_distance(dataset, query_id, nbr);
      insert_sorted_beam(beam, {d, nbr}, beam_width);
    }

  }

  std::sort(all_visited.begin(), all_visited.end(), candidate_id_less);
  return all_visited;
}

vec<std::pair<float, u32>> beam_search_float_query(VamanaGraph& graph,
                                                   const Dataset& dataset,
                                                   const float* query,
                                                   u32 beam_width) {
  vec<std::pair<float, u32>> all_visited;
  thread_local vec<std::pair<float, u32>> beam;
  beam.clear();
  const size_t expected_seen = std::max<size_t>(1024, static_cast<size_t>(beam_width) * graph.capacity + graph.capacity + 1);
  LocalIdSet visited(expected_seen);
  LocalIdSet expanded(std::max<size_t>(
    1024, static_cast<size_t>(beam_width) * 2));

  const float medoid_dist = dataset_distance_float_query(dataset, query, graph.medoid);
  beam.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  all_visited.push_back({medoid_dist, static_cast<u32>(graph.medoid)});
  visited.insert(static_cast<u32>(graph.medoid));

  thread_local vec<u32> nbrs;
  nbrs.clear();
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
    for (size_t i = 0; i < nbrs.size(); ++i) {
      const u32 nbr = nbrs[i];
      if (!visited.insert(nbr)) continue;
      if (i + 1 < nbrs.size()) {
        __builtin_prefetch(dataset.raw_vector(nbrs[i + 1]), 0, 1);
      }
      const float d = dataset_distance_float_query(dataset, query, nbr);
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
                      u32 R) {
  vec<u32> selected;
  selected.reserve(R);
  LocalIdSet selected_ids(R);
  for (float cur_alpha = 1.0F;
       cur_alpha <= alpha && selected.size() < R; cur_alpha *= 1.2F) {
    for (const auto& [cand_dist, cand_id] : sorted_candidates) {
      if (selected.size() >= R) break;
      if (cand_id == source || selected_ids.contains(cand_id)) continue;

      bool pruned = false;
      for (u32 sel_id : selected) {
        const L2PruneEarlyDecision decision = typed_l2_prune_predicate_early(
            dataset.raw_vector(sel_id), dataset.dtype,
            dataset.raw_vector(cand_id), dataset.dtype,
            dataset.dim, static_cast<f64>(cur_alpha), cand_dist);
        bool pair_prunes;
        if (decision == L2PruneEarlyDecision::pruned) {
          pair_prunes = true;
        } else if (decision == L2PruneEarlyDecision::not_pruned) {
          pair_prunes = false;
        } else {
          const float d_sel_cand = dataset_distance(dataset, sel_id, cand_id);
          pair_prunes = cur_alpha * d_sel_cand < cand_dist;
        }
        if (pair_prunes) {
          pruned = true;
          break;
        }
      }
      if (!pruned) {
        selected.push_back(cand_id);
        selected_ids.insert(cand_id);
      }
    }
  }
  return selected;
}

bool append_reverse_edge_with_prune_unlocked(VamanaGraph& graph,
                                             const Dataset& dataset,
                                             u32 target, u32 source,
                                             float alpha) {
  if (graph.try_append_neighbor_unlocked(target, source)) return false;
  if (graph.degree(target) < graph.capacity) return false;

  thread_local vec<u32> existing;
  existing.clear();
  graph.copy_neighbors(target, existing);
  if (std::find(existing.begin(), existing.end(), source) != existing.end()) {
    return false;
  }
  thread_local vec<std::pair<float, u32>> candidates;
  candidates.clear();
  candidates.reserve(existing.size() + 1);
  for (const u32 candidate : existing) {
    candidates.push_back({dataset_distance(dataset, target, candidate), candidate});
  }
  candidates.push_back({dataset_distance(dataset, target, source), source});
  std::sort(candidates.begin(), candidates.end(), candidate_id_less);
  graph.set_neighbors(target, robust_prune(dataset, target, candidates,
                                           alpha, graph.R));
  return true;
}

void consolidate_reverse_edges(VamanaGraph& graph,
                               const Dataset& dataset,
                               size_t num_threads,
                               float build_alpha,
                               size_t chunk_target_bytes,
                               size_t start_chunk,
                               const std::function<void(size_t next_chunk)>&
                                   between_chunks) {
  const size_t n = graph.num_nodes;
  std::cerr << "building reverse-edge CSR for bulk consolidation...\n";

  // Upper bound for the reverse CSR size: every directed edge contributes at
  // most one incoming slot.
  const size_t workers = effective_thread_count(static_cast<u32>(num_threads));
  vec<size_t> edge_partials(workers, 0);
  parallel_for(static_cast<size_t>(0), n, workers, [&](size_t node, size_t tid) {
    edge_partials[tid] += graph.degree(node);
  });
  size_t total_edges = 0;
  for (size_t partial : edge_partials) total_edges += partial;

  constexpr size_t kDefaultChunkTargetBytes = size_t{32} << 30;
  const size_t target_bytes =
      chunk_target_bytes == 0 ? kDefaultChunkTargetBytes : chunk_target_bytes;
  // Each chunk re-scans the full neighbor array twice (count + fill), so keep
  // chunks as large as the memory budget allows.
  size_t num_chunks =
      (total_edges * sizeof(u32) + target_bytes - 1) / target_bytes;
  num_chunks = std::max<size_t>(1, num_chunks);
  const size_t nodes_per_chunk = (n + num_chunks - 1) / num_chunks;
  num_chunks = (n + nodes_per_chunk - 1) / nodes_per_chunk;
  std::cerr << "bulk reverse consolidation: total_edges=" << total_edges
            << " chunks=" << num_chunks
            << " chunk_target_bytes=" << target_bytes << "\n";

  for (size_t chunk = start_chunk; chunk < num_chunks; ++chunk) {
    const size_t node_begin = chunk * nodes_per_chunk;
    const size_t node_end = std::min(n, node_begin + nodes_per_chunk);
    const size_t range = node_end - node_begin;

    // mmap-zeroed: incoming counts start at 0 without a clearing pass.
    LargeBuffer<std::atomic<u32>> incoming_counts(range);

    parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t source, size_t) {
      const size_t base = graph.offset(source);
      const u8 degree = graph.degree(source);
      for (u8 i = 0; i < degree; ++i) {
        const u32 target = graph.neighbors[base + i];
        if (target >= node_begin && target < node_end && target != source) {
          incoming_counts[target - node_begin].fetch_add(1, std::memory_order_relaxed);
        }
      }
    });

    LargeBuffer<u64> incoming_offsets(range + 1);  // mmap-zeroed
    for (size_t node = 0; node < range; ++node) {
      incoming_offsets[node + 1] = incoming_offsets[node] +
          incoming_counts[node].load(std::memory_order_relaxed);
    }
    const u64 total_incoming = incoming_offsets[range];
    std::cerr << "bulk reverse consolidation chunk " << (chunk + 1) << "/" << num_chunks
              << " nodes=[" << node_begin << ", " << node_end << ")"
              << " incoming_edges=" << total_incoming
              << " edge_bytes=" << total_incoming * sizeof(u32)
              << " offset_bytes=" << incoming_offsets.size() * sizeof(u64)
              << " count_bytes=" << range * sizeof(std::atomic<u32>) << "\n";

    LargeBuffer<u32> incoming_edges(static_cast<size_t>(total_incoming));
    {
      constexpr size_t kFillBlock = size_t{1} << 20;
      const size_t blocks = (range + kFillBlock - 1) / kFillBlock;
      parallel_for(static_cast<size_t>(0), blocks, num_threads, [&](size_t block, size_t) {
        const size_t begin = block * kFillBlock;
        const size_t end = std::min(range, begin + kFillBlock);
        for (size_t node = begin; node < end; ++node) {
          incoming_counts[node].store(0, std::memory_order_relaxed);
        }
      });
    }
    parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t source, size_t) {
      const size_t base = graph.offset(source);
      const u8 degree = graph.degree(source);
      for (u8 i = 0; i < degree; ++i) {
        const u32 target = graph.neighbors[base + i];
        if (target < node_begin || target >= node_end || target == source) continue;
        const u32 slot =
            incoming_counts[target - node_begin].fetch_add(1, std::memory_order_relaxed);
        incoming_edges[static_cast<size_t>(incoming_offsets[target - node_begin] + slot)] =
            static_cast<u32>(source);
      }
    });
    incoming_counts.release();

    ProgressReporter reverse_progress{
        "Bulk reverse-edge consolidation chunk " + std::to_string(chunk + 1) + "/" +
            std::to_string(num_chunks),
        range};
    parallel_for(node_begin, node_end, num_threads, [&](size_t node, size_t) {
      vec<u32> candidate_ids;
      const u8 outgoing_degree = graph.degree(node);
      const u64 incoming_begin = incoming_offsets[node - node_begin];
      const u64 incoming_end = incoming_offsets[node - node_begin + 1];
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
        prune_candidates.push_back({dataset_distance(dataset, node, candidate), candidate});
      }
      std::sort(prune_candidates.begin(), prune_candidates.end(), candidate_id_less);
      graph.set_neighbors(node, robust_prune(dataset, static_cast<u32>(node), prune_candidates,
                                             build_alpha, graph.R));
      reverse_progress.increment();
    });
    reverse_progress.finish();
    if (between_chunks) between_chunks(chunk + 1);
  }
}

void build_vamana_graph(VamanaGraph& graph,
                        const Dataset& dataset,
                        const VamanaBuildConfig& config,
                        const filepath_t& checkpoint_prefix) {
  const size_t n = dataset.size();
  const u32 R = config.R;
  const float alpha = static_cast<float>(config.alpha);
  const u32 beam_width = config.beam_width;
  const size_t num_threads = effective_thread_count(config.threads);
  const u32 build_capacity = static_cast<u32>(std::ceil(
      static_cast<f64>(R) * config.build_slack_factor));

  lib_assert(beam_width > 0, "offline graph beam width must be > 0");
  lib_assert(n > static_cast<size_t>(R),
             "dataset must contain at least R + 1 vectors");
  lib_assert(build_capacity >= R && build_capacity <= std::numeric_limits<u8>::max(),
             "invalid offline graph build capacity");

  const bool ckpt_enabled = !checkpoint_prefix.empty();
  const size_t segment_size =
      std::max<size_t>(1, config.checkpoint_segment_nodes);
  if (ckpt_enabled && config.seed == -1)
    lib_failure("checkpoint builds require a fixed --seed so the insertion "
                "order can be reproduced on resume");

  graph.init(n, dataset.dim, R, 1 << 24, num_threads, build_capacity);

  size_t resume_step = 0;
  size_t resume_chunk = 0;
  bool resumed = false;
  str resume_phase{"main_loop"};
  if (ckpt_enabled && graph_checkpoint_exists(checkpoint_prefix)) {
    const GraphCheckpointMeta meta =
        read_graph_checkpoint_meta(checkpoint_prefix);
    lib_assert(meta.version == 2,
               "legacy checkpoints cannot be resumed by the reference builder");
    lib_assert(meta.num_nodes == n && meta.R == R &&
                   meta.build_capacity == build_capacity &&
                   meta.dim == dataset.dim && meta.seed == config.seed &&
                   meta.beam_width == config.beam_width &&
                   meta.alpha == config.alpha &&
                   meta.build_slack_factor == config.build_slack_factor &&
                   meta.legacy_reverse_drop == config.legacy_reverse_drop &&
                   meta.random_initial_graph == config.random_initial_graph &&
                   meta.final_reverse_consolidation == config.final_reverse_consolidation &&
                   meta.vector_data_type == vector_dtype_name(dataset.dtype),
               "checkpoint does not match this build configuration");
    lib_assert(meta.segment_size == segment_size,
               "checkpoint segment size changed between pause and resume");
    load_graph_checkpoint_arrays(checkpoint_prefix, graph, meta, num_threads);
    resumed = true;
    resume_phase = meta.phase;
    if (meta.phase == "consolidate") {
      resume_step = n;
      resume_chunk = meta.next_chunk;
    } else {
      resume_step = meta.phase == "final_prune" ? n : meta.next_step;
      resume_chunk = 0;
    }
    std::cerr << "resuming from checkpoint: phase=" << meta.phase
              << " resume_step=" << resume_step
              << " resume_chunk=" << resume_chunk << "\n";
  }

  const size_t seed =
      config.seed == -1 ? std::random_device{}() : static_cast<size_t>(config.seed);
  vec<u32> order(n);
  std::iota(order.begin(), order.end(), 0);
  {
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
  }

  if (!resumed) {
    std::cerr << "computing medoid...\n";
    graph.medoid = compute_medoid(dataset, num_threads);
    std::cerr << "medoid: node " << graph.medoid << "\n";

    if (config.random_initial_graph) {
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
      const double unique_signature_ratio =
          static_cast<double>(neighbor_signatures.size()) /
          static_cast<double>(signature_sample);
      std::cerr << "random graph unique neighbor signature ratio="
                << unique_signature_ratio << " sample=" << signature_sample << "\n";
      lib_assert(unique_signature_ratio >= 0.99,
                 "random graph initialization produced too many duplicate neighbor lists");
    } else {
      std::cerr << "initializing empty graph for incremental linking\n";
    }
  }

  const float build_alpha = alpha;
  std::atomic<size_t> reverse_appends{0};
  std::atomic<size_t> reverse_duplicates{0};
  std::atomic<size_t> reverse_immediate_prunes{0};
  std::atomic<size_t> reverse_silent_drops{0};
  std::cerr << "graph build semantics: "
            << (config.legacy_reverse_drop ? "legacy-drop" : "immediate-prune")
            << " initial=" << (config.random_initial_graph ? "random" : "empty")
            << " final=" << (config.final_reverse_consolidation
                                 ? "global-consolidation" : "over-capacity-prune")
            << " alpha=" << build_alpha << " R=" << R
            << " capacity=" << build_capacity << " candidate_pool=expanded\n";

  auto commit_checkpoint = [&](const str& phase, size_t next_step,
                               size_t next_chunk) {
    GraphCheckpointMeta meta;
    meta.num_nodes = n;
    meta.dim = dataset.dim;
    meta.R = R;
    meta.build_capacity = build_capacity;
    meta.beam_width = config.beam_width;
    meta.alpha = config.alpha;
    meta.build_slack_factor = config.build_slack_factor;
    meta.legacy_reverse_drop = config.legacy_reverse_drop;
    meta.random_initial_graph = config.random_initial_graph;
    meta.final_reverse_consolidation = config.final_reverse_consolidation;
    meta.vector_data_type = vector_dtype_name(dataset.dtype);
    meta.medoid = graph.medoid;
    meta.seed = config.seed;
    meta.phase = phase;
    meta.next_step = next_step;
    meta.next_chunk = next_chunk;
    meta.segment_size = segment_size;
    save_graph_checkpoint(checkpoint_prefix, graph, meta, num_threads);
  };
  auto pause_checkpoint = [&](const str& phase, size_t next_step,
                              size_t next_chunk) {
    if (!ckpt_enabled || !checkpoint_pause_requested()) return;
    commit_checkpoint(phase, next_step, next_chunk);
    throw CheckpointPauseException();
  };
  auto log_graph_health = [&](size_t completed_steps) {
    const size_t sample = std::min<size_t>(n, 1024);
    const size_t stride = n / sample;
    size_t edges = 0;
    size_t reciprocal = 0;
    thread_local vec<u32> nbrs;
    nbrs.clear();
    for (size_t i = 0; i < sample; ++i) {
      const size_t node = i * stride;
      graph.copy_neighbors(node, nbrs);
      edges += nbrs.size();
      for (const u32 nbr : nbrs) {
        if (nbr < n && graph.contains_neighbor_unlocked(nbr, node)) ++reciprocal;
      }
    }
    const size_t probes = std::min<size_t>(completed_steps, 32);
    const size_t probe_stride = completed_steps / probes;
    size_t reachable = 0;
    for (size_t i = 0; i < probes; ++i) {
      const u32 target = order[i * probe_stride];
      const auto visited = beam_search(graph, dataset, target, beam_width);
      if (std::any_of(visited.begin(), visited.end(), [&](const auto& item) {
            return item.second == target;
          })) {
        ++reachable;
      }
    }
    std::cerr << "graph health: completed_steps=" << completed_steps
              << " sample=" << sample
              << " avg_degree=" << (edges == 0 ? 0.0 : static_cast<double>(edges) / sample)
              << " reciprocal_ratio=" << (edges == 0 ? 0.0 : static_cast<double>(reciprocal) / edges)
              << " self_reachable_ratio=" << (static_cast<double>(reachable) / probes)
              << " reverse_appends=" << reverse_appends.load(std::memory_order_relaxed)
              << " reverse_duplicates=" << reverse_duplicates.load(std::memory_order_relaxed)
              << " reverse_immediate_prunes=" << reverse_immediate_prunes.load(std::memory_order_relaxed)
              << " reverse_silent_drops=" << reverse_silent_drops.load(std::memory_order_relaxed)
              << "\n";
  };

  pause_checkpoint(resume_phase, resume_step, resume_chunk);

  if (resume_phase == "main_loop") {
    struct alignas(64) PhaseTiming {
      u64 beam_ns{0};
      u64 prune_ns{0};
      u64 reverse_ns{0};
      u64 candidate_count{0};
      u64 max_candidates{0};
    };
    vec<PhaseTiming> phase_times(num_threads);
    auto log_phase_times = [&](size_t completed_steps) {
      u64 beam_ns = 0;
      u64 prune_ns = 0;
      u64 reverse_ns = 0;
      u64 candidate_count = 0;
      u64 max_candidates = 0;
      for (const auto& timing : phase_times) {
        beam_ns += timing.beam_ns;
        prune_ns += timing.prune_ns;
        reverse_ns += timing.reverse_ns;
        candidate_count += timing.candidate_count;
        max_candidates = std::max(max_candidates, timing.max_candidates);
      }
      const size_t processed = completed_steps - resume_step;
      std::cerr << "build phase cpu_seconds: completed_steps=" << completed_steps
                << " beam=" << beam_ns * 1e-9
                << " prune=" << prune_ns * 1e-9
                << " reverse=" << reverse_ns * 1e-9
                << " candidate_avg=" << (processed == 0 ? 0.0 :
                    static_cast<double>(candidate_count) / processed)
                << " candidate_max=" << max_candidates << "\n";
    };
    ProgressReporter progress{"Building Vamana graph", n - resume_step};
    for (size_t seg_begin = resume_step; seg_begin < n;
         seg_begin += segment_size) {
      const size_t seg_end = std::min(n, seg_begin + segment_size);
      parallel_for(seg_begin, seg_end, num_threads, [&](size_t step, size_t tid) {
        const u32 node_idx = order[step];

        auto phase_start = std::chrono::steady_clock::now();
        auto candidates = beam_search(graph, dataset, node_idx, beam_width);
        phase_times[tid].beam_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - phase_start).count();

        phase_start = std::chrono::steady_clock::now();
        thread_local vec<u32> existing_neighbors;
        existing_neighbors.clear();
        {
          NodeLockGuard lock(graph, node_idx);
          graph.copy_neighbors(node_idx, existing_neighbors);
        }
        for (u32 existing : existing_neighbors) {
          candidates.push_back({dataset_distance(dataset, node_idx, existing), existing});
        }
        sort_and_unique_candidates(candidates);
        phase_times[tid].candidate_count += candidates.size();
        phase_times[tid].max_candidates = std::max<u64>(
            phase_times[tid].max_candidates, candidates.size());

        vec<u32> new_neighbors = robust_prune(dataset, node_idx, candidates, build_alpha, R);
        {
          NodeLockGuard lock(graph, node_idx);
          graph.set_neighbors(node_idx, new_neighbors);
        }
        phase_times[tid].prune_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - phase_start).count();

        phase_start = std::chrono::steady_clock::now();
        for (u32 nbr : new_neighbors) {
          NodeLockGuard lock(graph, nbr);
          const u8 before = graph.degree(nbr);
          if (config.legacy_reverse_drop) {
            if (graph.try_append_neighbor_unlocked(nbr, node_idx)) {
              reverse_appends.fetch_add(1, std::memory_order_relaxed);
            } else if (graph.contains_neighbor_unlocked(nbr, node_idx)) {
              reverse_duplicates.fetch_add(1, std::memory_order_relaxed);
            } else {
              reverse_silent_drops.fetch_add(1, std::memory_order_relaxed);
            }
          } else if (append_reverse_edge_with_prune_unlocked(
                         graph, dataset, nbr, node_idx, build_alpha)) {
            reverse_immediate_prunes.fetch_add(1, std::memory_order_relaxed);
          } else if (graph.degree(nbr) == static_cast<u8>(before + 1)) {
            reverse_appends.fetch_add(1, std::memory_order_relaxed);
          } else {
            reverse_duplicates.fetch_add(1, std::memory_order_relaxed);
          }
        }
        phase_times[tid].reverse_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - phase_start).count();
        progress.increment();
      });
      log_phase_times(seg_end);
      log_graph_health(seg_end);
      pause_checkpoint("main_loop", seg_end, 0);
      if (ckpt_enabled && config.checkpoint_interval_segments != 0 &&
          seg_end % segment_size == 0 &&
          (seg_end / segment_size) % config.checkpoint_interval_segments == 0) {
        commit_checkpoint("main_loop", seg_end, 0);
      }
    }
    progress.finish();
    log_phase_times(n);
  }

  if (config.final_reverse_consolidation) {
    pause_checkpoint("consolidate", n, resume_chunk);
    consolidate_reverse_edges(graph, dataset, num_threads, build_alpha,
                              config.consolidation_chunk_bytes, resume_chunk,
                              [&](size_t next_chunk) {
                                pause_checkpoint("consolidate", n, next_chunk);
                              });
  } else {
    pause_checkpoint("final_prune", n, 0);
    std::atomic<size_t> final_pruned{0};
    parallel_for(static_cast<size_t>(0), n, num_threads, [&](size_t node, size_t) {
      if (graph.degree(node) <= R) return;
      thread_local vec<u32> existing;
  existing.clear();
      graph.copy_neighbors(node, existing);
      thread_local vec<std::pair<float, u32>> candidates;
  candidates.clear();
      candidates.reserve(existing.size());
      for (const u32 candidate : existing) {
        candidates.push_back({dataset_distance(dataset, node, candidate), candidate});
      }
      std::sort(candidates.begin(), candidates.end(), candidate_id_less);
      graph.set_neighbors(node, robust_prune(dataset, node, candidates,
                                             build_alpha, R));
      final_pruned.fetch_add(1, std::memory_order_relaxed);
    });
    std::cerr << "reference final cleanup pruned_nodes="
              << final_pruned.load(std::memory_order_relaxed) << "\n";
  }

  if (ckpt_enabled) {
    commit_checkpoint(config.final_reverse_consolidation ? "consolidate" : "final_prune",
                      n, 0);
  }

  size_t total_edges = 0;
  size_t max_edges = 0;
  size_t min_edges = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < n; ++i) {
    const size_t count = graph.degree(i);
    total_edges += count;
    max_edges = std::max(max_edges, count);
    min_edges = std::min(min_edges, count);
  }
  lib_assert(max_edges <= R, "final graph contains a node above final R");
  std::cerr << "graph stats: avg_degree=" << (static_cast<double>(total_edges) / n)
            << " max=" << max_edges << " min=" << min_edges << "\n";
  log_graph_health(n);

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
        all_dists.push_back({dataset_distance(dataset, qid, i), static_cast<u32>(i)});
      }
      std::partial_sort(all_dists.begin(),
                        all_dists.begin() + std::min<size_t>(topk, all_dists.size()),
                        all_dists.end(),
                        candidate_id_less);
      std::unordered_set<u32> gt;
      for (size_t i = 0; i < topk && i < all_dists.size(); ++i) gt.insert(all_dists[i].second);

      auto results = beam_search(graph, dataset, qid, beam_width);
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
