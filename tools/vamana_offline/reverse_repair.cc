#include "tools/vamana_offline/reverse_repair.hh"

#include <algorithm>
#include <iostream>

#include "tools/vamana_offline/progress.hh"

namespace tools::vamana_offline {
namespace {

bool candidate_id_less(const std::pair<float, u32>& lhs,
                       const std::pair<float, u32>& rhs) {
  if (lhs.first != rhs.first) return lhs.first < rhs.first;
  return lhs.second < rhs.second;
}

}  // namespace

void repair_reverse_edges(VamanaGraph& graph, const Dataset& dataset,
                          size_t num_threads, u32 reverse_slots,
                          size_t chunk_target_bytes) {
  const size_t n = graph.num_nodes;
  const u32 degree = graph.R;
  lib_assert(reverse_slots < degree,
             "reverse repair slots must be smaller than R");
  const size_t original_slots = degree - reverse_slots;
  std::cerr << "reserved reverse repair: original_slots=" << original_slots
            << " reverse_slots=" << reverse_slots << "\n";

  const size_t workers = effective_thread_count(
    static_cast<u32>(num_threads));
  vec<size_t> edge_partials(workers, 0);
  parallel_for(size_t{0}, n, workers, [&](size_t node, size_t tid) {
    edge_partials[tid] += graph.degree(node);
  });
  size_t total_edges = 0;
  for (const size_t partial : edge_partials) total_edges += partial;

  constexpr size_t kDefaultChunkTargetBytes = size_t{32} << 30;
  const size_t target_bytes = chunk_target_bytes == 0
    ? kDefaultChunkTargetBytes : chunk_target_bytes;
  size_t num_chunks =
    (total_edges * sizeof(u32) + target_bytes - 1) / target_bytes;
  num_chunks = std::max<size_t>(1, num_chunks);
  const size_t nodes_per_chunk = (n + num_chunks - 1) / num_chunks;
  num_chunks = (n + nodes_per_chunk - 1) / nodes_per_chunk;
  std::cerr << "reserved reverse repair: total_edges=" << total_edges
            << " chunks=" << num_chunks
            << " chunk_target_bytes=" << target_bytes << "\n";

  for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
    const size_t node_begin = chunk * nodes_per_chunk;
    const size_t node_end = std::min(n, node_begin + nodes_per_chunk);
    const size_t range = node_end - node_begin;
    LargeBuffer<std::atomic<u32>> incoming_counts(range);
    parallel_for(size_t{0}, n, num_threads, [&](size_t source, size_t) {
      const size_t base = graph.offset(source);
      const u8 count = graph.degree(source);
      for (u8 index = 0; index < count; ++index) {
        const u32 target = graph.neighbors[base + index];
        if (target >= node_begin && target < node_end && target != source) {
          incoming_counts[target - node_begin].fetch_add(
            1, std::memory_order_relaxed);
        }
      }
    });

    LargeBuffer<u64> incoming_offsets(range + 1);
    for (size_t node = 0; node < range; ++node) {
      incoming_offsets[node + 1] = incoming_offsets[node] +
        incoming_counts[node].load(std::memory_order_relaxed);
    }
    const u64 total_incoming = incoming_offsets[range];
    LargeBuffer<u32> incoming_edges(static_cast<size_t>(total_incoming));
    constexpr size_t kFillBlock = size_t{1} << 20;
    const size_t blocks = (range + kFillBlock - 1) / kFillBlock;
    parallel_for(size_t{0}, blocks, num_threads, [&](size_t block, size_t) {
      const size_t begin = block * kFillBlock;
      const size_t end = std::min(range, begin + kFillBlock);
      for (size_t node = begin; node < end; ++node) {
        incoming_counts[node].store(0, std::memory_order_relaxed);
      }
    });
    parallel_for(size_t{0}, n, num_threads, [&](size_t source, size_t) {
      const size_t base = graph.offset(source);
      const u8 count = graph.degree(source);
      for (u8 index = 0; index < count; ++index) {
        const u32 target = graph.neighbors[base + index];
        if (target < node_begin || target >= node_end || target == source) {
          continue;
        }
        const u32 slot = incoming_counts[target - node_begin].fetch_add(
          1, std::memory_order_relaxed);
        incoming_edges[static_cast<size_t>(
          incoming_offsets[target - node_begin] + slot)] =
          static_cast<u32>(source);
      }
    });
    incoming_counts.release();

    ProgressReporter progress{"Reserved reverse repair", range};
    parallel_for(node_begin, node_end, num_threads,
                 [&](size_t node, size_t) {
      const size_t base = graph.offset(node);
      const u8 outgoing_degree = graph.degree(node);
      vec<u32> selected;
      selected.reserve(degree);
      const size_t keep = std::min<size_t>(outgoing_degree, original_slots);
      for (size_t index = 0; index < keep; ++index) {
        const u32 candidate = graph.neighbors[base + index];
        if (candidate != node && candidate < n) selected.push_back(candidate);
      }

      vec<std::pair<float, u32>> reverse_candidates;
      const u64 incoming_begin = incoming_offsets[node - node_begin];
      const u64 incoming_end = incoming_offsets[node - node_begin + 1];
      reverse_candidates.reserve(incoming_end - incoming_begin);
      for (u64 position = incoming_begin; position < incoming_end; ++position) {
        const u32 candidate = incoming_edges[static_cast<size_t>(position)];
        if (candidate == node || candidate >= n) continue;
        bool duplicate = false;
        for (const u32 existing : selected) {
          if (existing == candidate) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          reverse_candidates.push_back(
            {dataset_distance(dataset, node, candidate), candidate});
        }
      }
      std::sort(reverse_candidates.begin(), reverse_candidates.end(),
                candidate_id_less);
      for (const auto& [distance, candidate] : reverse_candidates) {
        if (selected.size() >= degree) break;
        selected.push_back(candidate);
      }
      for (size_t index = keep; index < outgoing_degree &&
                                  selected.size() < degree; ++index) {
        const u32 candidate = graph.neighbors[base + index];
        if (candidate == node || candidate >= n) continue;
        bool duplicate = false;
        for (const u32 existing : selected) {
          if (existing == candidate) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) selected.push_back(candidate);
      }
      graph.set_neighbors(node, selected);
      progress.increment();
    });
    progress.finish();
  }
}

}  // namespace tools::vamana_offline
