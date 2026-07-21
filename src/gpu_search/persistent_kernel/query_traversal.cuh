#pragma once

#include "gpu_search/centroid_route_ranking.hh"
#include "gpu_search/persistent_kernel/rdma_read.cuh"

#include <cuda/atomic>

namespace gpu_search::persistent_kernel_detail {

__device__ u64 decode_tagged_raw(const u8* source) {
  return *reinterpret_cast<const u64*>(source);
}

struct CentroidRouteShardSnapshot {
  u64 version{};
  u64 vector_count{};
  f32 distance{FLT_MAX};
  u32 live_entry_count{};
  u64 remote_nodes[kCentroidRouteMaxLiveEntries]{};
};

template <typename T>
__device__ T centroid_route_atomic_load(const T& value) {
  cuda::atomic_ref<T, cuda::thread_scope_device> reference(
    const_cast<T&>(value));
  return reference.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ void centroid_route_atomic_store(T& destination, T value) {
  cuda::atomic_ref<T, cuda::thread_scope_device> reference(destination);
  reference.store(value, cuda::memory_order_relaxed);
}

__device__ bool snapshot_centroid_route_shard(
    const PersistentKernelParams& params, u32 shard,
    const f32* query, CentroidRouteShardSnapshot& result) {
  if (params.centroid_route_shards == nullptr ||
      params.centroid_route_entries == nullptr ||
      params.shard_centroids == nullptr ||
      params.centroid_route_entry_capacity == 0 ||
      params.centroid_route_entry_capacity > kCentroidRouteMaxLiveEntries ||
      shard >= params.num_shards ||
      shard >= params.centroid_route_shard_capacity) {
    return false;
  }
  const DeviceCentroidRouteShard& source =
    params.centroid_route_shards[shard];
  cuda::atomic_ref<u64, cuda::thread_scope_device> sequence(
    const_cast<u64&>(source.sequence));
  // A writer window is very short.  Two attempts recover from the common
  // boundary race without ever making a query wait on route publication.
  for (u32 attempt = 0; attempt < 2; ++attempt) {
    const u64 before = sequence.load(cuda::memory_order_acquire);
    if ((before & 1u) != 0) continue;
    CentroidRouteShardSnapshot candidate{
      .version = centroid_route_atomic_load(source.version),
      .vector_count = centroid_route_atomic_load(source.vector_count),
      .live_entry_count =
        centroid_route_atomic_load(source.live_entry_count),
    };
    if (candidate.version == 0 ||
        candidate.vector_count == 0 || candidate.live_entry_count == 0 ||
        candidate.live_entry_count > params.centroid_route_entry_capacity) {
      return false;
    }
    f32 distance = 0.0f;
    const f32* centroid = params.shard_centroids +
      static_cast<size_t>(shard) * params.dim;
    for (u32 dimension = 0; dimension < params.dim; ++dimension) {
      const f32 difference = query[dimension] - centroid[dimension];
      // Match CPU update-home routing exactly: canonical FP32 centroids,
      // left-to-right FP32 fused accumulation, then physical-shard tie break.
      distance = fmaf(difference, difference, distance);
    }
    if (!finite_f32_bits(distance) || distance == FLT_MAX) {
      double wide_distance = 0.0;
      for (u32 dimension = 0; dimension < params.dim; ++dimension) {
        const double difference = static_cast<double>(query[dimension]) -
          static_cast<double>(centroid[dimension]);
        wide_distance = fma(difference, difference, wide_distance);
      }
      distance = saturate_device_squared_l2(wide_distance);
    }
    candidate.distance = distance;
    const DeviceCentroidRouteEntry* entries =
      params.centroid_route_entries +
      static_cast<size_t>(shard) * params.centroid_route_entry_capacity;
    for (u32 index = 0; index < candidate.live_entry_count; ++index) {
      const u64 remote_node = centroid_route_atomic_load(
        entries[index].remote_node);
      const u32 flags = centroid_route_atomic_load(entries[index].flags);
      if (remote_node == 0 || flags != kCentroidRouteLive ||
          remote_shard(remote_node) != shard) {
        return false;
      }
      candidate.remote_nodes[index] = remote_node;
    }
    const u64 after = sequence.load(cuda::memory_order_acquire);
    if (before != after || (after & 1u) != 0) continue;
    result = candidate;
    return true;
  }
  return false;
}

// All 64 RemotePtr-addressable shards participate in the same fixed sorting
// network. Invalid/unpublished shards sort after every valid shard, including
// a valid distance saturated near FLT_MAX. Every compare-exchange is owned by
// exactly one CUDA thread, and the barrier preserves the bitonic stage order.
__device__ void sort_centroid_route_shards(
    centroid_route_ranking::RankedShard* routes) {
  static_assert(kPersistentMaxShards == 64);
  for (u32 sequence = 2; sequence <= kPersistentMaxShards; sequence <<= 1) {
    for (u32 stride = sequence >> 1; stride != 0; stride >>= 1) {
      for (u32 index = threadIdx.x; index < kPersistentMaxShards;
           index += blockDim.x) {
        const u32 partner = index ^ stride;
        if (partner <= index) continue;
        const bool ascending = (index & sequence) == 0;
        if (!centroid_route_ranking::should_exchange(
              routes[index], routes[partner], ascending)) {
          continue;
        }
        const centroid_route_ranking::RankedShard temporary = routes[index];
        routes[index] = routes[partner];
        routes[partner] = temporary;
      }
      __syncthreads();
    }
  }
}

__device__ void process_query(const PersistentKernelParams& params,
                              const QueryDescriptor& descriptor) {
  const u32 query_slot = descriptor.query_slot;
  __shared__ u64 query_started_cycles;
  if (threadIdx.x == 0) query_started_cycles = clock64();
  __syncthreads();
  CompletionDescriptor completion{
    .request_id = descriptor.request_id,
    .query_slot = query_slot,
  };
  if (query_slot >= params.query_slots || descriptor.dim != params.dim ||
      descriptor.query_dtype > 2 || params.decoded_queries == nullptr ||
      params.navigation_candidate_handles == nullptr ||
      params.navigation_candidate_distances == nullptr ||
      descriptor.k == 0 || descriptor.k > descriptor.result_capacity) {
    if (threadIdx.x == 0) {
      completion.status = -EINVAL;
      completion.gpu_cycles = clock64() - query_started_cycles;
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }

  const u8* query_input = reinterpret_cast<const u8*>(descriptor.query_device_address);
  __shared__ u64 prepare_started_cycles;
  __shared__ u64 graph_phase_cycles;
  __shared__ u64 score_phase_cycles;
  __shared__ u64 beam_phase_cycles;
  __shared__ u64 exact_phase_cycles;
  __shared__ u64 phase_started_cycles;
  if (threadIdx.x == 0) {
    prepare_started_cycles = clock64();
    graph_phase_cycles = 0;
    score_phase_cycles = 0;
    beam_phase_cycles = 0;
    exact_phase_cycles = 0;
  }
  __syncthreads();
  f32* query = params.decoded_queries + static_cast<size_t>(query_slot) * params.dim;
  for (u32 dimension = threadIdx.x; dimension < params.dim; dimension += blockDim.x) {
    query[dimension] = query_component(query_input, descriptor.query_dtype, dimension);
  }
  __syncthreads();
  f32* transformed = params.transformed_queries +
    static_cast<size_t>(query_slot) * params.dim;
  for (u32 row = threadIdx.x; row < params.dim; row += blockDim.x) {
    if (params.opq_matrix == nullptr) {
      transformed[row] = query[row];
      continue;
    }
    f32 value = 0.0f;
    const f32* matrix_row = params.opq_matrix + static_cast<size_t>(row) * params.dim;
    for (u32 column = 0; column < params.dim; ++column) {
      value += matrix_row[column] * query[column];
    }
    if (!finite_f32_bits(value)) {
      double wide_value = 0.0;
      for (u32 column = 0; column < params.dim; ++column) {
        wide_value = fma(static_cast<double>(matrix_row[column]),
                         static_cast<double>(query[column]), wide_value);
      }
      value = saturate_device_component(wide_value);
    }
    transformed[row] = value;
  }
  __syncthreads();
  f32* query_lut = params.query_luts +
    static_cast<size_t>(query_slot) * params.pq_subquantizers * 256;
  const u32 table_entries = params.pq_subquantizers * 256;
  for (u32 index = threadIdx.x; index < table_entries; index += blockDim.x) {
    const u32 subquantizer = index / 256;
    const f32* query_subvector = transformed +
      static_cast<size_t>(subquantizer) * params.pq_subvector_dim;
    const f32* centroid_subvector = params.pq_centroids +
      static_cast<size_t>(index) * params.pq_subvector_dim;
    f32 distance = 0.0f;
    for (u32 dimension = 0; dimension < params.pq_subvector_dim; ++dimension) {
      const f32 difference = query_subvector[dimension] - centroid_subvector[dimension];
      distance += difference * difference;
    }
    if (!finite_f32_bits(distance) || distance == FLT_MAX) {
      double wide_distance = 0.0;
      for (u32 dimension = 0; dimension < params.pq_subvector_dim;
           ++dimension) {
        const double difference =
          static_cast<double>(query_subvector[dimension]) -
          static_cast<double>(centroid_subvector[dimension]);
        wide_distance += difference * difference;
      }
      distance = saturate_device_squared_l2(wide_distance);
    }
    query_lut[index] = distance;
  }
  __syncthreads();

  __shared__ u64 shared_beam_handles[kPersistentMaxBeam];
  __shared__ u32 shared_beam_ids[kPersistentMaxBeam];
  __shared__ f32 shared_beam_distances[kPersistentMaxBeam];
  __shared__ u8 shared_beam_expanded[kPersistentMaxBeam];
  __shared__ CandidateWorkspace candidate_workspace;
  u64* merge_handles = candidate_workspace.arrays.handles;
  u32* merge_ids = candidate_workspace.arrays.ids;
  f32* merge_distances = candidate_workspace.arrays.distances;
  u8* merge_expanded = candidate_workspace.arrays.expanded;
  u64* navigation_handles = params.navigation_candidate_handles +
    static_cast<size_t>(query_slot) * kPersistentMaxMergeCandidates;
  f32* navigation_distances = params.navigation_candidate_distances +
    static_cast<size_t>(query_slot) * kPersistentMaxMergeCandidates;
  u64* beam_handles = shared_beam_handles;
  u32* beam_ids = shared_beam_ids;
  f32* beam_distances = shared_beam_distances;
  u8* beam_expanded = shared_beam_expanded;
  const u32 traversal_capacity = min(kPersistentMaxBeam, params.traversal_beam_width);
  u64* visited = params.visited_hash +
    static_cast<size_t>(query_slot) * params.visited_capacity;
  for (u32 index = threadIdx.x; index < traversal_capacity; index += blockDim.x) {
    beam_handles[index] = kInvalidDeviceHandle;
    beam_ids[index] = UINT32_MAX;
    beam_distances[index] = FLT_MAX;
    beam_expanded[index] = 0;
  }
  for (u32 index = threadIdx.x; index < params.visited_capacity; index += blockDim.x) {
    visited[index] = kInvalidDeviceHandle;
  }
  __syncthreads();

  __shared__ u32 beam_count;
  __shared__ u64 rerank_handles[kPersistentMaxExact];
  __shared__ u32 rerank_ids[kPersistentMaxExact];
  __shared__ f32 rerank_distances[kPersistentMaxExact];
  __shared__ u32 rerank_count;
  __shared__ u32 total_exact_reads;
  __shared__ centroid_route_ranking::RankedShard
    ranked_routes[kPersistentMaxShards];
  __shared__ u32 route_snapshot_entry_counts[kPersistentMaxShards];
  __shared__ u64 route_snapshot_remote_nodes[
    kPersistentMaxShards * kCentroidRouteMaxLiveEntries];
  __shared__ u32 route_entry_count;
  __shared__ u32 ranked_route_count;
  __shared__ u32 route_seed_failed;
  __shared__ u64 route_epoch_before;
  if (threadIdx.x == 0) {
    rerank_count = 0;
    total_exact_reads = 0;
  }
  __syncthreads();

  // A route publication may replace every seed after this CTA snapshots it.
  // Re-snapshot once in the same query when no seed remains routable. This is
  // deliberately bounded and never falls back to an immutable entry table.
  for (u32 route_attempt = 0; route_attempt < 2; ++route_attempt) {
    if (threadIdx.x == 0) {
      route_entry_count = 0;
      ranked_route_count = 0;
      route_seed_failed = 0;
      if (params.centroid_route_epoch == nullptr) {
        route_epoch_before = 1;
        route_seed_failed = 1;
      } else {
        cuda::atomic_ref<u64, cuda::thread_scope_device> route_epoch(
          *params.centroid_route_epoch);
        route_epoch_before = route_epoch.load(cuda::memory_order_acquire);
        if ((route_epoch_before & 1u) != 0) route_seed_failed = 1;
      }
      beam_count = 0;
      rerank_count = 0;
    }
    // Parallelize across physical shards while preserving the exact scalar
    // fmaf recurrence within a shard. A dimension-wise tree reduction would
    // change FP32 rounding and could send inserts and queries to different
    // homes. With at most 64 shards, one lane per shard removes the previous
    // thread-0 shards*dim serialization without weakening that invariant.
    for (u32 shard = threadIdx.x; shard < kPersistentMaxShards;
         shard += blockDim.x) {
      route_snapshot_entry_counts[shard] = 0;
      ranked_routes[shard] = centroid_route_ranking::RankedShard{
        .distance = FLT_MAX,
        .shard = shard,
        .valid = 0,
      };
      if (shard >= params.num_shards) continue;
      CentroidRouteShardSnapshot snapshot;
      if (!snapshot_centroid_route_shard(params, shard, query, snapshot)) {
        continue;
      }
      route_snapshot_entry_counts[shard] = snapshot.live_entry_count;
      for (u32 entry = 0; entry < snapshot.live_entry_count; ++entry) {
        route_snapshot_remote_nodes[
          static_cast<size_t>(shard) * kCentroidRouteMaxLiveEntries + entry] =
          snapshot.remote_nodes[entry];
      }
      ranked_routes[shard] = centroid_route_ranking::RankedShard{
        .distance = snapshot.distance,
        .shard = shard,
        .valid = 1,
      };
    }
    __syncthreads();
    sort_centroid_route_shards(ranked_routes);
    if (threadIdx.x == 0) {
      u64 route_epoch_after = 1;
      if (params.centroid_route_epoch != nullptr) {
        cuda::atomic_ref<u64, cuda::thread_scope_device> route_epoch(
          *params.centroid_route_epoch);
        route_epoch_after = route_epoch.load(cuda::memory_order_acquire);
      }
      if (params.centroid_route_epoch == nullptr ||
          !centroid_route_ranking::stable_publication_epoch(
            route_epoch_before, route_epoch_after)) {
        route_seed_failed = 1;
        route_entry_count = 0;
      } else {
        while (ranked_route_count < kPersistentMaxShards &&
               ranked_routes[ranked_route_count].valid != 0) {
          ++ranked_route_count;
        }
      }
      // Query and Stage1 share one locality decision: begin only at the
      // nearest routable centroid shard. Cross-shard work is introduced by
      // graph edges, never by an eager multi-shard seed fanout.
      if (route_seed_failed == 0 && ranked_route_count != 0) {
        const u32 shard = ranked_routes[0].shard;
        for (u32 local = 0; local < route_snapshot_entry_counts[shard];
             ++local) {
          const u64 remote_node = route_snapshot_remote_nodes[
            static_cast<size_t>(shard) * kCentroidRouteMaxLiveEntries + local];
          const u64 handle = handle_from_raw(params, remote_node);
          if (handle != kInvalidDeviceHandle) {
            navigation_handles[route_entry_count++] = handle;
          }
        }
      }
    }
    __syncthreads();
    if (route_entry_count != 0 &&
        !approximate_handles_batch(params, descriptor, query_lut,
                                   navigation_handles, route_entry_count,
                                   navigation_distances)) {
      if (threadIdx.x == 0) route_seed_failed = 1;
    }
    __syncthreads();
    if (route_seed_failed == 0) {
      for (u32 index = threadIdx.x; index < route_entry_count;
           index += blockDim.x) {
        merge_handles[index] = navigation_handles[index];
        merge_ids[index] = UINT32_MAX;
        merge_distances[index] = navigation_distances[index];
        merge_expanded[index] = 0;
      }
      __syncthreads();
      if (route_entry_count != 0) {
        sort_candidates(merge_handles, nullptr, merge_distances,
                        merge_expanded, route_entry_count);
      }
      if (threadIdx.x == 0) {
        const u32 initial_seed_capacity = min(
          route_entry_count, traversal_capacity);
        u32 unique_count = 0;
        for (u32 input = 0;
             input < route_entry_count &&
               unique_count < initial_seed_capacity;
             ++input) {
          const u64 handle = merge_handles[input];
          if (handle == kInvalidDeviceHandle ||
              !isfinite(merge_distances[input]) ||
              merge_distances[input] == FLT_MAX) {
            continue;
          }
          bool duplicate = false;
          for (u32 prior = 0; prior < unique_count; ++prior) {
            if (merge_handles[prior] == handle) {
              duplicate = true;
              break;
            }
          }
          if (duplicate) continue;
          if (unique_count != input) {
            merge_handles[unique_count] = handle;
            merge_distances[unique_count] = merge_distances[input];
            merge_expanded[unique_count] = 0;
          }
          ++unique_count;
        }
        beam_count = unique_count;
        for (u32 index = 0; index < beam_count; ++index) {
          beam_handles[index] = merge_handles[index];
          beam_ids[index] = UINT32_MAX;
          beam_distances[index] = merge_distances[index];
          beam_expanded[index] = 0;
          insert_visited(visited, params.visited_capacity,
                         beam_handles[index]);
        }
      }
      __syncthreads();
    }
    if (beam_count == 0) {
      if (route_attempt == 0) {
        if (threadIdx.x == 0) device_ring_relax(64);
        __syncthreads();
        continue;
      }
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.gpu_cycles = clock64() - query_started_cycles;
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }

  __shared__ u64 selected_handles[kPersistentMaxPrefetch];
  __shared__ u32 selected_count;
  __shared__ u32 neighbor_counts[kPersistentMaxPrefetch];
  __shared__ u32 neighbor_offsets[kPersistentMaxPrefetch + 1];
  __shared__ u32 flattened_neighbors;
  __shared__ u32 remote_reads_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 graph_record_slots[kPersistentMaxPrefetch];
  __shared__ u32 total_remote_reads;
  __shared__ u32 total_remote_batches;
  __shared__ u32 total_graph_read_retries;
  __shared__ u32 total_graph_rounds;
  __shared__ u32 graph_failed;
  if (threadIdx.x == 0) {
    total_remote_reads = 0;
    total_remote_batches = 0;
    total_graph_read_retries = 0;
    total_graph_rounds = 0;
    graph_failed = 0;
  }
  __syncthreads();

  __shared__ u32 expansions;
  if (threadIdx.x == 0) expansions = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    completion.prepare_cycles = clock64() - prepare_started_cycles;
  }
  __syncthreads();
  while (expansions < params.max_expansions) {
    if (threadIdx.x == 0) phase_started_cycles = clock64();
    __syncthreads();
    if (threadIdx.x == 0) {
      selected_count = 0;
      graph_failed = 0;
      const u32 target = min(params.prefetch_depth, params.max_expansions - expansions);
      for (u32 index = 0; index < beam_count && selected_count < target; ++index) {
        if (beam_expanded[index] != 0) continue;
        beam_expanded[index] = 1;
        selected_handles[selected_count++] = beam_handles[index];
      }
    }
    __syncthreads();
    if (selected_count == 0) break;
    if (threadIdx.x == 0) ++total_graph_rounds;
    __syncthreads();
    constexpr u32 warp_width = 32;
    const u32 warp = threadIdx.x / warp_width;
    const u32 lane_in_warp = threadIdx.x % warp_width;
    if (!fetch_graph_records_batch(
          params, descriptor, selected_handles, selected_count,
          graph_record_slots, remote_reads_by_lane,
          &total_remote_batches, &total_graph_read_retries)) {
      if (threadIdx.x == 0) graph_failed = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      graph_phase_cycles += clock64() - phase_started_cycles;
      for (u32 selected = 0; selected < selected_count; ++selected) {
        total_remote_reads += remote_reads_by_lane[selected];
      }
    }
    __syncthreads();
    if (graph_failed != 0) {
      for (u32 selected = warp; selected < selected_count;
           selected += blockDim.x / warp_width) {
        if (lane_in_warp == 0) graph_record_slots[selected] = UINT32_MAX;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.remote_batches = total_remote_batches;
        completion.graph_read_retries = total_graph_read_retries;
        completion.graph_rounds = total_graph_rounds;
        completion.exact_vectors = total_exact_reads;
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }

    const u32 score_chunk_capacity = persistent_score_chunk_capacity(
      params.graph_entry_capacity, traversal_capacity);
    if (score_chunk_capacity == 0) {
      if (threadIdx.x == 0) graph_failed = 1;
      __syncthreads();
    }
    for (u32 chunk_begin = 0;
         graph_failed == 0 && chunk_begin < selected_count;
         chunk_begin += score_chunk_capacity) {
      const u32 chunk_count = min(score_chunk_capacity,
                                  selected_count - chunk_begin);
      for (u32 local = warp; local < chunk_count;
           local += blockDim.x / warp_width) {
        const u32 selected = chunk_begin + local;
        const u32 slot = graph_record_slots[selected];
        const u8* record = slot == UINT32_MAX ? nullptr :
          graph_record_pointer(params, descriptor.query_slot, slot);
        if (lane_in_warp == 0) {
          const u32 stable_count = record == nullptr ? 0 : record[0];
          const u32 provisional_count = record == nullptr
            ? 0 : (record[1] >> 4) & 0xfu;
          neighbor_counts[local] = record != nullptr && (record[1] & 1u) == 0
            ? min(stable_count + provisional_count,
                  params.graph_entry_capacity)
            : 0;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        neighbor_offsets[0] = 0;
        for (u32 local = 0; local < chunk_count; ++local) {
          neighbor_offsets[local + 1] =
            neighbor_offsets[local] + neighbor_counts[local];
        }
        flattened_neighbors = neighbor_offsets[chunk_count];
        phase_started_cycles = clock64();
      }
      __syncthreads();
      for (u32 local = warp; local < chunk_count;
           local += blockDim.x / warp_width) {
        const u32 selected = chunk_begin + local;
        const u32 slot = graph_record_slots[selected];
        const u8* record = slot == UINT32_MAX ? nullptr :
          graph_record_pointer(params, descriptor.query_slot, slot);
        __syncwarp();
        const u32 count = neighbor_counts[local];
        for (u32 neighbor = lane_in_warp; neighbor < count; neighbor += warp_width) {
          const u64 raw = decode_tagged_raw(
            record + 16 + neighbor * sizeof(u64));
          navigation_handles[neighbor_offsets[local] + neighbor] =
            handle_from_raw(params, raw);
        }
        __syncwarp();
        if (lane_in_warp == 0) graph_record_slots[selected] = UINT32_MAX;
      }
      __syncthreads();
      const u32 candidate_count = flattened_neighbors;
      for (u32 flat = threadIdx.x; flat < candidate_count; flat += blockDim.x) {
        const u64 handle = navigation_handles[flat];
        if (handle == kInvalidDeviceHandle ||
            !insert_visited(visited, params.visited_capacity, handle)) {
          navigation_handles[flat] = kInvalidDeviceHandle;
        }
      }
      __syncthreads();
      if (!approximate_handles_batch(params, descriptor, query_lut,
                                     navigation_handles,
                                     candidate_count,
                                     navigation_distances)) {
        if (threadIdx.x == 0) graph_failed = 1;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        score_phase_cycles += clock64() - phase_started_cycles;
        phase_started_cycles = clock64();
      }
      __syncthreads();
      if (graph_failed != 0) break;
      merge_approximate_into_beam(
        navigation_handles, navigation_distances,
        candidate_count, beam_handles, beam_ids, beam_distances,
        beam_expanded, beam_count, traversal_capacity,
        merge_handles, merge_ids, merge_distances, merge_expanded,
        rerank_handles, rerank_ids, rerank_distances,
        candidate_workspace);
      if (threadIdx.x == 0) {
        beam_phase_cycles += clock64() - phase_started_cycles;
      }
      __syncthreads();
    }
    if (graph_failed != 0) {
      __syncthreads();
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.remote_batches = total_remote_batches;
        completion.graph_read_retries = total_graph_read_retries;
        completion.graph_rounds = total_graph_rounds;
        completion.exact_vectors = total_exact_reads;
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }
    if (threadIdx.x == 0) {
      expansions += selected_count;
    }
    __syncthreads();
  }

  for (u32 index = threadIdx.x; index < beam_count; index += blockDim.x) {
    merge_handles[index] = beam_handles[index];
    merge_distances[index] = beam_distances[index];
    merge_expanded[index] = 0;
  }
  __syncthreads();
  sort_candidates(merge_handles, nullptr, merge_distances, merge_expanded, beam_count);
  if (threadIdx.x == 0) {
    // Exactify the entire traversal beam. Remote fixed-record headers are the
    // source of truth for delete/upsert visibility; overfetching the full beam
    // lets tombstoned prefix candidates be replaced instead of returning < k.
    rerank_count = min(beam_count, kPersistentMaxExact);
    for (u32 index = 0; index < rerank_count; ++index) {
      rerank_handles[index] = merge_handles[index];
      rerank_ids[index] = UINT32_MAX;
      rerank_distances[index] = merge_distances[index];
    }
    phase_started_cycles = clock64();
  }
  __syncthreads();
  exactify_into_beam(params, descriptor, query, rerank_handles, rerank_ids, rerank_distances,
                     rerank_count, beam_handles, beam_ids, beam_distances, beam_expanded,
                     beam_count, &total_exact_reads,
                     min(kPersistentMaxExact,
                         max(params.final_rerank_width,
                             static_cast<u32>(descriptor.k))),
                     true, merge_handles, merge_ids,
                     merge_distances, merge_expanded);
  if (threadIdx.x == 0) {
    exact_phase_cycles += clock64() - phase_started_cycles;
  }
  __syncthreads();

  if (beam_count == 0) {
    if (route_attempt == 0) {
      for (u32 index = threadIdx.x; index < params.visited_capacity;
           index += blockDim.x) {
        visited[index] = kInvalidDeviceHandle;
      }
      if (threadIdx.x == 0) device_ring_relax(64);
      __syncthreads();
      continue;
    }
    if (threadIdx.x == 0) {
      completion.status = -EIO;
      completion.gpu_cycles = clock64() - query_started_cycles;
      completion.remote_pages = total_remote_reads;
      completion.remote_batches = total_remote_batches;
      completion.graph_read_retries = total_graph_read_retries;
      completion.graph_rounds = total_graph_rounds;
      completion.exact_vectors = total_exact_reads;
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }
  sort_candidates(beam_handles, beam_ids, beam_distances, beam_expanded,
                  beam_count);
  if (threadIdx.x == 0) {
    u32 valid = 0;
    while (valid < beam_count && beam_ids[valid] != UINT32_MAX &&
           isfinite(beam_distances[valid])) ++valid;
    const u32 result_count = min(static_cast<u32>(descriptor.k), valid);
    u32* output_ids = reinterpret_cast<u32*>(descriptor.result_device_address);
    f32* output_distances = params.result_distances +
      static_cast<size_t>(query_slot) * descriptor.result_capacity;
    for (u32 index = 0; index < result_count; ++index) {
      output_ids[index] = beam_ids[index];
      output_distances[index] = beam_distances[index];
    }
    completion.result_count = result_count;
    completion.status = 0;
    completion.gpu_cycles = clock64() - query_started_cycles;
    completion.graph_cycles = graph_phase_cycles;
    completion.score_cycles = score_phase_cycles;
    completion.beam_cycles = beam_phase_cycles;
    completion.exact_cycles = exact_phase_cycles;
    completion.remote_pages = total_remote_reads;
    completion.remote_batches = total_remote_batches;
    completion.graph_read_retries = total_graph_read_retries;
    completion.graph_rounds = total_graph_rounds;
    completion.exact_vectors = total_exact_reads;
    device_ring_push(params.completions, completion);
  }
  __syncthreads();
  return;
  }
}

}  // namespace gpu_search::persistent_kernel_detail
