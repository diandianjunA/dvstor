#pragma once

#include "gpu_search/centroid_route_ranking.hh"
#include "gpu_search/persistent_kernel/adjacency_oracle_trace.cuh"
#include "gpu_search/persistent_kernel/rdma_read.cuh"

#include <cuda/atomic>

namespace gpu_search::persistent_kernel_detail {

template <bool EnableAdjacencyOracle>
__device__ void set_query_trace_completion(
    const PersistentKernelParams& params, u32 query_slot,
    CompletionDescriptor& completion) {
  if (params.query_rdma_trace_headers == nullptr ||
      query_slot >= params.query_slots) return;
  const QueryRdmaTraceHeader& header =
    params.query_rdma_trace_headers[query_slot];
  if (header.request_id != completion.request_id || header.enabled == 0) return;
  completion.trace_event_count = min(
    header.event_count, params.query_rdma_trace_events_per_query);
  completion.trace_overflow = header.overflow;
  if constexpr (EnableAdjacencyOracle) {
    if (params.query_adjacency_oracle_trace_headers == nullptr) return;
    const QueryAdjacencyOracleTraceHeader& adjacency_header =
      params.query_adjacency_oracle_trace_headers[query_slot];
    if (adjacency_header.request_id != completion.request_id ||
        adjacency_header.enabled == 0) {
      return;
    }
    completion.adjacency_oracle_event_count = min(
      adjacency_header.event_count, params.query_rdma_trace_events_per_query);
    completion.adjacency_oracle_overflow = adjacency_header.overflow;
  }
}

__device__ void set_dynamic_code_cache_completion(
    CompletionDescriptor& completion, u32 cache_hits,
    u32 batch_deduplicated, u32 publish_successes, u32 publish_races,
    u32 lookup_probe_exhaustions, u32 publish_probe_exhaustions,
    u32 lookup_probes, u32 max_lookup_probes) {
  completion.dynamic_code_cache_hits = cache_hits;
  completion.dynamic_code_batch_deduplicated = batch_deduplicated;
  completion.dynamic_code_cache_publish_successes = publish_successes;
  completion.dynamic_code_cache_publish_races = publish_races;
  completion.dynamic_code_cache_lookup_probe_exhaustions =
    lookup_probe_exhaustions;
  completion.dynamic_code_cache_publish_probe_exhaustions =
    publish_probe_exhaustions;
  completion.dynamic_code_cache_lookup_probes = lookup_probes;
  completion.dynamic_code_cache_max_lookup_probes = max_lookup_probes;
}

__device__ void set_expansion_completion(
    CompletionDescriptor& completion, const PersistentKernelParams& params,
    u32 sum_selected_parents, u32 sum_feedback_horizon,
    u32 sum_hardware_credit_tiles, u32 minimum_selected_batch,
    u32 maximum_selected_batch, u32 minimum_feedback_horizon,
    u32 maximum_feedback_horizon) {
  completion.expansion_policy = params.query_expansion_policy;
  completion.sum_selected_parents = sum_selected_parents;
  completion.sum_feedback_horizon = sum_feedback_horizon;
  completion.sum_hardware_credit_tiles = sum_hardware_credit_tiles;
  completion.minimum_selected_batch =
    minimum_selected_batch == UINT32_MAX ? 0 : minimum_selected_batch;
  completion.maximum_selected_batch = maximum_selected_batch;
  completion.minimum_feedback_horizon =
    minimum_feedback_horizon == UINT32_MAX ? 0 : minimum_feedback_horizon;
  completion.maximum_feedback_horizon = maximum_feedback_horizon;
}

__device__ void set_beam_merge_completion(
    CompletionDescriptor& completion,
    const BeamMergeCycleBreakdown& breakdown) {
  completion.beam_merge_prepare_cycles = breakdown.prepare;
  completion.beam_merge_sort_cycles = breakdown.sort;
  completion.beam_merge_materialize_cycles = breakdown.materialize;
}

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

struct ExpansionRoundCycleBaseline {
  u64 beam_selection{};
  u64 rdma_issue{};
  u64 neighbor_decode{};
  u64 visited{};
  u64 pq_score{};
  u64 beam_merge{};
};

__device__ bool try_claim_expansion_tile(
    const PersistentKernelParams& params, u32 query_slot,
    const u64* handles, u32 begin, u32 count,
    u32* claim_qps, u32* claim_epochs, u32* claim_wqes,
    u32& claim_count,
    u32& rollback_count) {
  claim_count = 0;
  if (params.expansion_qp_leases == nullptr ||
      params.expansion_qp_lease_count == 0 ||
      params.direct_qps_per_node == 0 || params.direct_region_count == 0) {
    return false;
  }
  for (u32 item = 0; item < count; ++item) {
    u64 raw = 0;
    u64 graph_offset = 0;
    u32 shard = 0;
    if (!resolve_handle(
          params, handles[begin + item], raw, shard, graph_offset)) {
      return false;
    }
    const u32 qp =
      ((query_slot + shard) % params.direct_qps_per_node) *
        params.direct_region_count + shard;
    if (qp >= params.expansion_qp_lease_count) return false;
    u32 demand = 0;
    for (; demand < claim_count; ++demand) {
      if (claim_qps[demand] == qp) {
        ++claim_wqes[demand];
        break;
      }
    }
    if (demand == claim_count) {
      if (claim_count >= kPersistentMaxPrefetch) return false;
      claim_qps[claim_count] = qp;
      claim_epochs[claim_count] = 0;
      claim_wqes[claim_count] = 1;
      ++claim_count;
    }
  }

  // A deterministic QP order avoids lock-order cycles when one tile spans
  // multiple shards.  Claims are non-blocking; failure returns the complete
  // partial vector before the authoritative expanded bits are changed.
  for (u32 left = 1; left < claim_count; ++left) {
    const u32 value_qp = claim_qps[left];
    const u32 value_wqes = claim_wqes[left];
    u32 right = left;
    while (right != 0 && claim_qps[right - 1] > value_qp) {
      claim_qps[right] = claim_qps[right - 1];
      claim_wqes[right] = claim_wqes[right - 1];
      --right;
    }
    claim_qps[right] = value_qp;
    claim_wqes[right] = value_wqes;
  }
  u32 claimed = 0;
  for (; claimed < claim_count; ++claimed) {
    const u32 qp = claim_qps[claimed];
    const u32 wqes = claim_wqes[claimed];
    QpExpansionLeaseClaim acquired{};
    if (!qp_expansion_lease_try_claim(
          params.expansion_qp_leases, params.expansion_qp_lease_count,
          qp, wqes, acquired)) {
      break;
    }
    claim_epochs[claimed] = acquired.epoch;
  }
  if (claimed == claim_count) return true;
  for (u32 index = 0; index < claimed; ++index) {
    qp_expansion_lease_return(
      params.expansion_qp_leases, params.expansion_qp_lease_count,
      QpExpansionLeaseClaim{
        .qp = claim_qps[index],
        .epoch = claim_epochs[index],
        .wqes = claim_wqes[index],
      });
  }
  rollback_count += claimed;
  claim_count = 0;
  return false;
}

__device__ void return_unissued_expansion_leases(
    const PersistentKernelParams& params,
    const QpExpansionLeaseClaim* claims, u32 claim_count,
    const u32* issued_qps, u32 issued_count) {
  for (u32 claim = 0; claim < claim_count; ++claim) {
    bool issued = false;
    for (u32 shard = 0; shard < issued_count; ++shard) {
      issued |= issued_qps[shard] == claims[claim].qp;
    }
    if (!issued) {
      qp_expansion_lease_return(
        params.expansion_qp_leases, params.expansion_qp_lease_count,
        claims[claim]);
    }
  }
}

template <bool EnableAdjacencyOracle>
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
      completion.diagnostic = make_query_diagnostic(
        QueryFailureReason::invalid_descriptor);
      completion.gpu_cycles = clock64() - query_started_cycles;
      set_query_trace_completion<EnableAdjacencyOracle>(
        params, query_slot, completion);
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }
  if (threadIdx.x == 0) {
    expansion_pressure_query_enter(params.expansion_pressure);
  }
  __syncthreads();

  const u8* query_input = reinterpret_cast<const u8*>(descriptor.query_device_address);
  __shared__ u64 prepare_started_cycles;
  __shared__ u64 graph_phase_cycles;
  __shared__ u64 score_phase_cycles;
  __shared__ u64 beam_phase_cycles;
  __shared__ u64 exact_phase_cycles;
  __shared__ u64 dynamic_code_cycles;
  __shared__ u64 beam_selection_cycles;
  __shared__ u64 rdma_issue_cycles;
  __shared__ u64 rdma_wait_cycles;
  __shared__ u64 graph_validation_cycles;
  __shared__ u64 neighbor_decode_cycles;
  __shared__ u64 pq_score_cycles;
  __shared__ u64 visited_cycles;
  __shared__ u64 beam_merge_cycles;
  __shared__ BeamMergeCycleBreakdown beam_merge_breakdown;
  __shared__ BeamMergeCycleBreakdown beam_merge_round_breakdown;
  __shared__ u32 feedback_horizon;
  __shared__ u32 sum_selected_parents;
  __shared__ u32 sum_feedback_horizon;
  __shared__ u32 sum_hardware_credit_tiles;
  __shared__ u32 minimum_selected_batch;
  __shared__ u32 maximum_selected_batch;
  __shared__ u32 minimum_feedback_horizon;
  __shared__ u32 maximum_feedback_horizon;
  __shared__ u32 compute_extra_tile_allowance;
  __shared__ u32 round_extra_tiles;
  __shared__ ExpansionRoundCycleBaseline expansion_cycle_baseline;
  __shared__ FeedbackHorizonResult merge_feedback;
  __shared__ u32 rdma_trace_enabled;
  __shared__ GraphFetchCycleBreakdown graph_fetch_breakdown;
  __shared__ u32 dynamic_code_candidates;
  __shared__ u32 dynamic_code_reads;
  __shared__ u32 dynamic_code_incarnation_rejects;
  __shared__ u32 dynamic_code_cache_hits;
  __shared__ u32 dynamic_code_batch_deduplicated;
  __shared__ u32 dynamic_code_cache_publish_successes;
  __shared__ u32 dynamic_code_cache_publish_races;
  __shared__ u32 dynamic_code_cache_lookup_probe_exhaustions;
  __shared__ u32 dynamic_code_cache_publish_probe_exhaustions;
  __shared__ u32 dynamic_code_cache_lookup_probes;
  __shared__ u32 dynamic_code_cache_max_lookup_probes;
  __shared__ u64 phase_started_cycles;
  if (threadIdx.x == 0) {
    prepare_started_cycles = clock64();
    graph_phase_cycles = 0;
    score_phase_cycles = 0;
    beam_phase_cycles = 0;
    exact_phase_cycles = 0;
    dynamic_code_cycles = 0;
    beam_selection_cycles = 0;
    rdma_issue_cycles = 0;
    rdma_wait_cycles = 0;
    graph_validation_cycles = 0;
    neighbor_decode_cycles = 0;
    pq_score_cycles = 0;
    visited_cycles = 0;
    beam_merge_cycles = 0;
    beam_merge_breakdown = {};
    beam_merge_round_breakdown = {};
    feedback_horizon = params.efficient_batch_cap;
    sum_selected_parents = 0;
    sum_feedback_horizon = 0;
    sum_hardware_credit_tiles = 0;
    minimum_selected_batch = UINT32_MAX;
    maximum_selected_batch = 0;
    minimum_feedback_horizon = UINT32_MAX;
    maximum_feedback_horizon = 0;
    compute_extra_tile_allowance =
      (params.efficient_batch_cap +
       max(1u, blockDim.x / 32u) - 1u) /
      max(1u, blockDim.x / 32u);
    round_extra_tiles = 0;
    rdma_trace_enabled =
      params.query_rdma_trace_mode ==
        static_cast<u32>(QueryRdmaTraceMode::full) ||
      (params.query_rdma_trace_mode ==
         static_cast<u32>(QueryRdmaTraceMode::sampled) &&
       params.query_rdma_trace_sample_rate != 0 &&
       descriptor.request_id % params.query_rdma_trace_sample_rate == 0);
    if (params.query_rdma_trace_headers != nullptr) {
      params.query_rdma_trace_headers[query_slot] = {
        .request_id = descriptor.request_id,
        .event_count = 0,
        .overflow = 0,
        .enabled = rdma_trace_enabled,
      };
    }
    if constexpr (EnableAdjacencyOracle) {
      if (params.query_adjacency_oracle_trace_headers != nullptr) {
        params.query_adjacency_oracle_trace_headers[query_slot] = {
          .request_id = descriptor.request_id,
          .event_count = 0,
          .overflow = 0,
          .enabled = rdma_trace_enabled,
        };
      }
    }
    dynamic_code_candidates = 0;
    dynamic_code_reads = 0;
    dynamic_code_incarnation_rejects = 0;
    dynamic_code_cache_hits = 0;
    dynamic_code_batch_deduplicated = 0;
    dynamic_code_cache_publish_successes = 0;
    dynamic_code_cache_publish_races = 0;
    dynamic_code_cache_lookup_probe_exhaustions = 0;
    dynamic_code_cache_publish_probe_exhaustions = 0;
    dynamic_code_cache_lookup_probes = 0;
    dynamic_code_cache_max_lookup_probes = 0;
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
  __shared__ u32 route_snapshot_conflicted;
  __shared__ u32 route_snapshot_timed_out;
  __shared__ u32 route_snapshot_cancelled;
  __shared__ u32 route_snapshot_retries;
  __shared__ u32 route_snapshot_backoff_ns;
  __shared__ u32 route_failure_reason;
  __shared__ u64 route_epoch_before;
  __shared__ u64 route_wait_started_ns;
  if (threadIdx.x == 0) {
    rerank_count = 0;
    total_exact_reads = 0;
    route_snapshot_retries = 0;
  }
  __syncthreads();

  // A route publication may replace every seed after this CTA snapshots it.
  // Epoch contention is not an I/O error: wait for one complete authoritative
  // table transaction. The occupancy planner reserves a resident control CTA,
  // so the writer can always close the short odd-epoch window. A genuinely
  // stuck writer is bounded by the independent route-control deadline below.
  // Separately, re-snapshot once when a once-valid seed becomes stale after the
  // transaction; no immutable entry table is used as a fallback.
  for (u32 route_attempt = 0; route_attempt < 2; ++route_attempt) {
    if (threadIdx.x == 0) {
      feedback_horizon = params.efficient_batch_cap;
      beam_count = 0;
      rerank_count = 0;
      route_wait_started_ns = global_time_ns();
      route_snapshot_backoff_ns = 128u +
        ((descriptor.query_slot * 97u + route_attempt * 53u) & 255u);
    }
    __syncthreads();
    for (;;) {
      if (threadIdx.x == 0) {
        route_entry_count = 0;
        ranked_route_count = 0;
        route_seed_failed = 0;
        route_snapshot_conflicted = 0;
        route_snapshot_timed_out = 0;
        route_snapshot_cancelled = 0;
        route_failure_reason = static_cast<u32>(
          QueryFailureReason::route_no_seed);
        if (params.centroid_route_epoch == nullptr) {
          route_epoch_before = 1;
          route_seed_failed = 1;
        } else {
          cuda::atomic_ref<u64, cuda::thread_scope_device> route_epoch(
            *params.centroid_route_epoch);
          route_epoch_before = route_epoch.load(cuda::memory_order_acquire);
          if ((route_epoch_before & 1u) != 0) {
            route_snapshot_conflicted = 1;
          }
        }
      }
      __syncthreads();

      if (route_snapshot_conflicted == 0) {
        // Parallelize across physical shards while preserving the exact scalar
        // fmaf recurrence within a shard. A dimension-wise tree reduction
        // would change FP32 rounding and could send inserts and queries to
        // different homes.
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
            route_snapshot_conflicted = 1;
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
          if (route_snapshot_conflicted == 0 && route_seed_failed == 0 &&
              ranked_route_count != 0) {
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
      }

      if (route_snapshot_conflicted == 0) break;
      if (threadIdx.x == 0) {
        ++route_snapshot_retries;
        if (params.stop != nullptr &&
            *reinterpret_cast<const volatile u32*>(params.stop) != 0) {
          route_snapshot_cancelled = 1;
        } else {
          const u64 timeout_ns = params.route_snapshot_timeout_ns == 0
            ? u64{100'000'000} : params.route_snapshot_timeout_ns;
          if (global_time_ns() - route_wait_started_ns >= timeout_ns) {
            route_snapshot_timed_out = 1;
          } else {
            device_ring_relax(route_snapshot_backoff_ns);
            route_snapshot_backoff_ns = min(
              route_snapshot_backoff_ns * 2u, 8192u);
          }
        }
      }
      __syncthreads();
      if (route_snapshot_cancelled != 0) {
        if (threadIdx.x == 0) {
          expansion_pressure_query_exit(params.expansion_pressure);
        }
        __syncthreads();
        return;
      }
      if (route_snapshot_timed_out != 0) {
        if (threadIdx.x == 0) {
          completion.status = -ETIMEDOUT;
          completion.diagnostic = make_query_diagnostic(
            QueryFailureReason::route_snapshot_timeout,
            route_snapshot_retries);
          completion.gpu_cycles = clock64() - query_started_cycles;
          set_expansion_completion(
            completion, params, sum_selected_parents, sum_feedback_horizon,
            sum_hardware_credit_tiles, minimum_selected_batch,
            maximum_selected_batch, minimum_feedback_horizon,
            maximum_feedback_horizon);
          set_beam_merge_completion(completion, beam_merge_breakdown);
          expansion_pressure_query_exit(params.expansion_pressure);
          set_query_trace_completion<EnableAdjacencyOracle>(
            params, query_slot, completion);
          device_ring_push(params.completions, completion);
        }
        __syncthreads();
        return;
      }
    }
    __syncthreads();
    if (route_entry_count != 0 &&
        !approximate_handles_batch(
          params, descriptor, query_lut, navigation_handles,
          route_entry_count, navigation_distances, &dynamic_code_cycles,
          &dynamic_code_candidates, &dynamic_code_reads,
          &dynamic_code_incarnation_rejects, &dynamic_code_cache_hits,
          &dynamic_code_batch_deduplicated,
          &dynamic_code_cache_publish_successes,
          &dynamic_code_cache_publish_races,
          &dynamic_code_cache_lookup_probe_exhaustions,
          &dynamic_code_cache_publish_probe_exhaustions,
          &dynamic_code_cache_lookup_probes,
          &dynamic_code_cache_max_lookup_probes)) {
      if (threadIdx.x == 0) {
        route_seed_failed = 1;
        route_failure_reason = static_cast<u32>(
          QueryFailureReason::dynamic_code_fetch);
      }
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
        completion.diagnostic = make_query_diagnostic(
          static_cast<QueryFailureReason>(route_failure_reason),
          route_snapshot_retries);
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.dynamic_code_cycles = dynamic_code_cycles;
        completion.dynamic_code_candidates = dynamic_code_candidates;
        completion.dynamic_code_reads = dynamic_code_reads;
        completion.dynamic_code_incarnation_rejects =
          dynamic_code_incarnation_rejects;
        set_dynamic_code_cache_completion(
          completion, dynamic_code_cache_hits,
          dynamic_code_batch_deduplicated,
          dynamic_code_cache_publish_successes, dynamic_code_cache_publish_races,
          dynamic_code_cache_lookup_probe_exhaustions,
          dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
        set_expansion_completion(
          completion, params, sum_selected_parents, sum_feedback_horizon,
          sum_hardware_credit_tiles, minimum_selected_batch,
          maximum_selected_batch, minimum_feedback_horizon,
          maximum_feedback_horizon);
        set_beam_merge_completion(completion, beam_merge_breakdown);
        expansion_pressure_query_exit(params.expansion_pressure);
        set_query_trace_completion<EnableAdjacencyOracle>(
          params, query_slot, completion);
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }

  __shared__ u64 selected_handles[kPersistentMaxPrefetch];
  __shared__ u32 selected_count;
  __shared__ QpExpansionLeaseClaim expansion_lease_claims[kPersistentMaxPrefetch];
  __shared__ u32 round_lease_claim_count;
  __shared__ u32 neighbor_counts[kPersistentMaxPrefetch];
  __shared__ u32 neighbor_offsets[kPersistentMaxPrefetch + 1];
  __shared__ u32 flattened_neighbors;
  __shared__ u32 remote_reads_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 graph_record_slots[kPersistentMaxPrefetch];
  __shared__ u32 issued_qps[kPersistentMaxShards];
  __shared__ u32 total_remote_reads;
  __shared__ u32 total_remote_batches;
  __shared__ u32 total_graph_read_retries;
  __shared__ u64 total_graph_read_bytes;
  __shared__ u32 total_graph_live_extent_reads;
  __shared__ u32 total_graph_full_record_reads;
  __shared__ u32 total_graph_extent_fallback_reads;
  __shared__ u32 total_graph_extent_underhint_reads;
  __shared__ u32 total_graph_extent_hint_promotions;
  __shared__ u32 total_graph_rounds;
  __shared__ u32 graph_failed;
  __shared__ u32 adjacency_oracle_trace_event_index;
  if (threadIdx.x == 0) {
    total_remote_reads = 0;
    total_remote_batches = 0;
    total_graph_read_retries = 0;
    total_graph_read_bytes = 0;
    total_graph_live_extent_reads = 0;
    total_graph_full_record_reads = 0;
    total_graph_extent_fallback_reads = 0;
    total_graph_extent_underhint_reads = 0;
    total_graph_extent_hint_promotions = 0;
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
    if (threadIdx.x == 0) {
      phase_started_cycles = clock64();
      expansion_cycle_baseline = {
        .beam_selection = beam_selection_cycles,
        .rdma_issue = rdma_issue_cycles,
        .neighbor_decode = neighbor_decode_cycles,
        .visited = visited_cycles,
        .pq_score = pq_score_cycles,
        .beam_merge = beam_merge_cycles,
      };
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      selected_count = 0;
      round_lease_claim_count = 0;
      graph_failed = 0;
      round_extra_tiles = 0;
      const bool feedback_policy =
        params.query_expansion_policy ==
          static_cast<u32>(QueryExpansionPolicy::feedback_hunger);
      const u32 expansion_tile = max(1u, blockDim.x / 32u);
      const u32 remaining = params.max_expansions - expansions;
      if (!feedback_policy) {
        const u32 target = min(params.prefetch_depth, remaining);
        for (u32 index = 0;
             index < beam_count && selected_count < target; ++index) {
          if (beam_expanded[index] != 0) continue;
          beam_expanded[index] = 1;
          selected_handles[selected_count++] = beam_handles[index];
        }
      } else {
        const u32 selection_limit = min(params.efficient_batch_cap, remaining);
        u32 eligible = 0;
        for (u32 index = 0;
             index < beam_count && eligible < selection_limit; ++index) {
          if (beam_expanded[index] != 0) continue;
          selected_handles[eligible] = beam_handles[index];
          // neighbor_counts is not live until graph records have arrived.
          neighbor_counts[eligible] = index;
          ++eligible;
        }
        const u32 base_count = min(feedback_horizon, eligible);
        for (u32 index = 0; index < base_count; ++index) {
          beam_expanded[neighbor_counts[index]] = 1;
        }
        selected_count = base_count;

        const u32 structural_extra_tiles =
          (eligible > selected_count
             ? eligible - selected_count + expansion_tile - 1u : 0u) /
            expansion_tile;
        // The ledger is diagnostic only.  It must not become a hidden,
        // dataset-specific width knob: the admissible width is determined by
        // the natural CTA tile and the actual per-QP leases.
        const u32 maximum_extra_tiles = structural_extra_tiles;
        while (selected_count < eligible &&
               round_extra_tiles < maximum_extra_tiles) {
          const u32 tile_count = min(
            expansion_tile, eligible - selected_count);
          u32 claim_count = 0;
          u32 rollback_count = 0;
          if (!try_claim_expansion_tile(
                params, descriptor.query_slot, selected_handles,
                selected_count, tile_count,
                remote_reads_by_lane, graph_record_slots, neighbor_offsets,
                claim_count, rollback_count)) {
            ++completion.qp_lease_reject_count;
            completion.qp_lease_rollback_count += rollback_count;
            break;
          }
          for (u32 index = 0; index < tile_count; ++index) {
            beam_expanded[
              neighbor_counts[selected_count + index]] = 1;
          }
          selected_count += tile_count;
          ++round_extra_tiles;
          completion.extra_parent_count += tile_count;
          completion.qp_lease_claim_count += claim_count;
          for (u32 claim = 0; claim < claim_count; ++claim) {
            expansion_lease_claims[round_lease_claim_count++] = {
              .qp = remote_reads_by_lane[claim],
              .epoch = graph_record_slots[claim],
              .wqes = neighbor_offsets[claim],
            };
          }
        }
        completion.compute_allowance_tile_sum +=
          compute_extra_tile_allowance;
      }
      if (selected_count != 0) {
        sum_selected_parents += selected_count;
        sum_feedback_horizon += feedback_policy ? feedback_horizon : 0u;
        sum_hardware_credit_tiles += round_extra_tiles;
        minimum_selected_batch = min(
          minimum_selected_batch, selected_count);
        maximum_selected_batch = max(
          maximum_selected_batch, selected_count);
        if (feedback_policy) {
          minimum_feedback_horizon = min(
            minimum_feedback_horizon, feedback_horizon);
          maximum_feedback_horizon = max(
            maximum_feedback_horizon, feedback_horizon);
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      beam_selection_cycles += clock64() - phase_started_cycles;
      phase_started_cycles = clock64();
    }
    if (selected_count == 0) break;
    if (threadIdx.x == 0) ++total_graph_rounds;
    __syncthreads();
    constexpr u32 warp_width = 32;
    const u32 warp = threadIdx.x / warp_width;
    const u32 lane_in_warp = threadIdx.x % warp_width;
    if (threadIdx.x == 0) graph_fetch_breakdown = {};
    __syncthreads();
    if (!fetch_graph_records_batch(
          params, descriptor, selected_handles, selected_count,
          graph_record_slots, remote_reads_by_lane,
          &total_remote_batches, &total_graph_read_retries,
          &total_graph_read_bytes,
          &total_graph_live_extent_reads,
          &total_graph_full_record_reads,
          &total_graph_extent_fallback_reads,
          &total_graph_extent_underhint_reads,
          &total_graph_extent_hint_promotions,
          issued_qps,
          route_attempt,
          total_graph_rounds - 1, rdma_trace_enabled != 0,
          &graph_fetch_breakdown)) {
      if (threadIdx.x == 0) graph_failed = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rdma_issue_cycles += graph_fetch_breakdown.issue;
      rdma_wait_cycles += graph_fetch_breakdown.wait;
      graph_validation_cycles += graph_fetch_breakdown.validation;
      graph_phase_cycles += clock64() - phase_started_cycles;
      for (u32 selected = 0; selected < selected_count; ++selected) {
        // fetch_graph_records_batch may use upper bits as private per-parent
        // retry metadata. Bit zero alone denotes the one logical graph read
        // selected by the search algorithm.
        total_remote_reads += remote_reads_by_lane[selected] & 1u;
      }
    }
    __syncthreads();
    if (graph_failed != 0) {
      if (threadIdx.x == 0) {
        return_unissued_expansion_leases(
          params, expansion_lease_claims, round_lease_claim_count,
          issued_qps, params.num_shards);
      }
      for (u32 selected = warp; selected < selected_count;
           selected += blockDim.x / warp_width) {
        if (lane_in_warp == 0) graph_record_slots[selected] = UINT32_MAX;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.diagnostic = make_query_diagnostic(
          QueryFailureReason::graph_fetch, route_snapshot_retries);
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.remote_batches = total_remote_batches;
        completion.graph_read_retries = total_graph_read_retries;
        completion.graph_read_bytes = total_graph_read_bytes;
        completion.graph_live_extent_reads =
          total_graph_live_extent_reads;
        completion.graph_full_record_reads =
          total_graph_full_record_reads;
        completion.graph_extent_fallback_reads =
          total_graph_extent_fallback_reads;
        completion.graph_extent_underhint_reads =
          total_graph_extent_underhint_reads;
        completion.graph_extent_hint_promotions =
          total_graph_extent_hint_promotions;
        completion.graph_rounds = total_graph_rounds;
        completion.exact_vectors = total_exact_reads;
        completion.dynamic_code_cycles = dynamic_code_cycles;
        completion.dynamic_code_candidates = dynamic_code_candidates;
        completion.dynamic_code_reads = dynamic_code_reads;
        completion.dynamic_code_incarnation_rejects =
          dynamic_code_incarnation_rejects;
        set_dynamic_code_cache_completion(
          completion, dynamic_code_cache_hits,
          dynamic_code_batch_deduplicated,
          dynamic_code_cache_publish_successes, dynamic_code_cache_publish_races,
          dynamic_code_cache_lookup_probe_exhaustions,
          dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
        set_expansion_completion(
          completion, params, sum_selected_parents, sum_feedback_horizon,
          sum_hardware_credit_tiles, minimum_selected_batch,
          maximum_selected_batch, minimum_feedback_horizon,
          maximum_feedback_horizon);
        set_beam_merge_completion(completion, beam_merge_breakdown);
        expansion_pressure_query_exit(params.expansion_pressure);
        set_query_trace_completion<EnableAdjacencyOracle>(
          params, query_slot, completion);
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }

    const u32 score_chunk_capacity = persistent_score_chunk_capacity(
      params.graph_entry_capacity, traversal_capacity);
    if (score_chunk_capacity == 0 ||
        (params.query_expansion_policy ==
           static_cast<u32>(QueryExpansionPolicy::feedback_hunger) &&
         selected_count > score_chunk_capacity)) {
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
      if (threadIdx.x == 0) {
        neighbor_decode_cycles += clock64() - phase_started_cycles;
        phase_started_cycles = clock64();
      }
      if constexpr (EnableAdjacencyOracle) {
        if (rdma_trace_enabled != 0) {
          begin_adjacency_oracle_trace(
            params, descriptor, true,
            total_graph_rounds - 1u, chunk_begin, selected_handles,
            chunk_count, neighbor_counts, neighbor_offsets,
            navigation_handles, navigation_distances, query_lut,
            beam_distances, beam_count, traversal_capacity,
            adjacency_oracle_trace_event_index);
          if (threadIdx.x == 0) {
            // Keep the deliberately expensive probe out of the production
            // phase breakdown. Trace runs are never throughput measurements.
            phase_started_cycles = clock64();
          }
        }
      }
      const u32 candidate_count = flattened_neighbors;
      for (u32 flat = threadIdx.x; flat < candidate_count; flat += blockDim.x) {
        const u64 handle = navigation_handles[flat];
        if (handle == kInvalidDeviceHandle ||
            !insert_visited(visited, params.visited_capacity, handle)) {
          navigation_handles[flat] = kInvalidDeviceHandle;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        visited_cycles += clock64() - phase_started_cycles;
        phase_started_cycles = clock64();
      }
      if (!approximate_handles_batch(
            params, descriptor, query_lut, navigation_handles,
            candidate_count, navigation_distances, &dynamic_code_cycles,
            &dynamic_code_candidates, &dynamic_code_reads,
            &dynamic_code_incarnation_rejects, &dynamic_code_cache_hits,
            &dynamic_code_batch_deduplicated,
            &dynamic_code_cache_publish_successes,
            &dynamic_code_cache_publish_races,
            &dynamic_code_cache_lookup_probe_exhaustions,
            &dynamic_code_cache_publish_probe_exhaustions,
            &dynamic_code_cache_lookup_probes,
            &dynamic_code_cache_max_lookup_probes)) {
        if (threadIdx.x == 0) graph_failed = 1;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        pq_score_cycles += clock64() - phase_started_cycles;
        score_phase_cycles = neighbor_decode_cycles +
          visited_cycles + pq_score_cycles;
        phase_started_cycles = clock64();
      }
      __syncthreads();
      if (graph_failed != 0) break;
      if constexpr (EnableAdjacencyOracle) {
        if (rdma_trace_enabled != 0) {
          record_adjacency_oracle_post_visited(
            params, descriptor, adjacency_oracle_trace_event_index,
            chunk_count, neighbor_counts, neighbor_offsets,
            navigation_handles, navigation_distances);
          if (threadIdx.x == 0) {
            phase_started_cycles = clock64();
          }
          __syncthreads();
        }
      }
      merge_approximate_into_beam(
        navigation_handles, navigation_distances,
        candidate_count, beam_handles, beam_ids, beam_distances,
        beam_expanded, beam_count, traversal_capacity,
        merge_handles, merge_ids, merge_distances, merge_expanded,
        rerank_handles, rerank_ids, rerank_distances,
        candidate_workspace, params.efficient_batch_cap,
        (params.query_expansion_policy ==
            static_cast<u32>(QueryExpansionPolicy::feedback_hunger) ||
         (EnableAdjacencyOracle && rdma_trace_enabled != 0))
          ? &merge_feedback : nullptr,
        static_cast<BeamMergePolicy>(params.beam_merge_policy),
        params.beam_merge_policy ==
            static_cast<u32>(BeamMergePolicy::stable_run)
          ? &beam_merge_round_breakdown : nullptr);
      if (threadIdx.x == 0) {
        const u64 merge_cycles = clock64() - phase_started_cycles;
        beam_merge_cycles += merge_cycles;
        beam_phase_cycles += merge_cycles;
        if (params.beam_merge_policy ==
            static_cast<u32>(BeamMergePolicy::stable_run)) {
          beam_merge_breakdown.prepare +=
            beam_merge_round_breakdown.prepare;
          beam_merge_breakdown.sort += beam_merge_round_breakdown.sort;
          beam_merge_breakdown.materialize +=
            beam_merge_round_breakdown.materialize;
        }
        if (params.query_expansion_policy ==
            static_cast<u32>(QueryExpansionPolicy::feedback_hunger)) {
          feedback_horizon = merge_feedback.horizon;
        }
      }
      __syncthreads();
      if constexpr (EnableAdjacencyOracle) {
        if (rdma_trace_enabled != 0) {
          finish_adjacency_oracle_trace(
            params, descriptor, adjacency_oracle_trace_event_index,
            chunk_count, neighbor_counts, neighbor_offsets,
            navigation_handles, navigation_distances,
            beam_handles, beam_distances, beam_expanded, beam_count,
            merge_feedback.new_candidates_in_beam,
            graph_fetch_breakdown.issue + graph_fetch_breakdown.wait +
              graph_fetch_breakdown.validation +
              (neighbor_decode_cycles -
                expansion_cycle_baseline.neighbor_decode),
            (visited_cycles - expansion_cycle_baseline.visited) +
              (pq_score_cycles - expansion_cycle_baseline.pq_score),
            beam_merge_cycles - expansion_cycle_baseline.beam_merge);
        }
      }
    }
    if (graph_failed != 0) {
      __syncthreads();
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.diagnostic = make_query_diagnostic(
          QueryFailureReason::dynamic_code_fetch,
          route_snapshot_retries);
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.remote_batches = total_remote_batches;
        completion.graph_read_retries = total_graph_read_retries;
        completion.graph_read_bytes = total_graph_read_bytes;
        completion.graph_live_extent_reads =
          total_graph_live_extent_reads;
        completion.graph_full_record_reads =
          total_graph_full_record_reads;
        completion.graph_extent_fallback_reads =
          total_graph_extent_fallback_reads;
        completion.graph_extent_underhint_reads =
          total_graph_extent_underhint_reads;
        completion.graph_extent_hint_promotions =
          total_graph_extent_hint_promotions;
        completion.graph_rounds = total_graph_rounds;
        completion.exact_vectors = total_exact_reads;
        completion.dynamic_code_cycles = dynamic_code_cycles;
        completion.dynamic_code_candidates = dynamic_code_candidates;
        completion.dynamic_code_reads = dynamic_code_reads;
        completion.dynamic_code_incarnation_rejects =
          dynamic_code_incarnation_rejects;
        set_dynamic_code_cache_completion(
          completion, dynamic_code_cache_hits,
          dynamic_code_batch_deduplicated,
          dynamic_code_cache_publish_successes, dynamic_code_cache_publish_races,
          dynamic_code_cache_lookup_probe_exhaustions,
          dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
        set_expansion_completion(
          completion, params, sum_selected_parents, sum_feedback_horizon,
          sum_hardware_credit_tiles, minimum_selected_batch,
          maximum_selected_batch, minimum_feedback_horizon,
          maximum_feedback_horizon);
        set_beam_merge_completion(completion, beam_merge_breakdown);
        expansion_pressure_query_exit(params.expansion_pressure);
        set_query_trace_completion<EnableAdjacencyOracle>(
          params, query_slot, completion);
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }
    if (threadIdx.x == 0) {
      if (params.query_expansion_policy ==
          static_cast<u32>(QueryExpansionPolicy::feedback_hunger)) {
        const u64 fixed_cycles =
          (beam_selection_cycles - expansion_cycle_baseline.beam_selection) +
          (rdma_issue_cycles - expansion_cycle_baseline.rdma_issue) +
          (beam_merge_cycles - expansion_cycle_baseline.beam_merge);
        const u64 parent_cycles =
          (neighbor_decode_cycles -
             expansion_cycle_baseline.neighbor_decode) +
          (visited_cycles - expansion_cycle_baseline.visited) +
          (pq_score_cycles - expansion_cycle_baseline.pq_score);
        const u32 expansion_tile = max(1u, blockDim.x / 32u);
        const u32 processed_tiles =
          (selected_count + expansion_tile - 1u) / expansion_tile;
        const u64 tile_cycles = processed_tiles == 0 ? 0 :
          (parent_cycles + processed_tiles - 1u) / processed_tiles;
        const u32 structural_max =
          (params.efficient_batch_cap + expansion_tile - 1u) /
          expansion_tile;
        compute_extra_tile_allowance = tile_cycles == 0
          ? structural_max
          : min(structural_max,
                static_cast<u32>(fixed_cycles / tile_cycles));
        if (round_extra_tiles != 0) {
          if (compute_extra_tile_allowance >= round_extra_tiles) {
            ++completion.marginal_probe_pass_count;
          } else {
            ++completion.marginal_probe_fail_count;
          }
        }
      }
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
      completion.diagnostic = make_query_diagnostic(
        QueryFailureReason::exact_rerank_empty,
        route_snapshot_retries);
      completion.gpu_cycles = clock64() - query_started_cycles;
      completion.remote_pages = total_remote_reads;
      completion.remote_batches = total_remote_batches;
      completion.graph_read_retries = total_graph_read_retries;
      completion.graph_read_bytes = total_graph_read_bytes;
      completion.graph_live_extent_reads =
        total_graph_live_extent_reads;
      completion.graph_full_record_reads =
        total_graph_full_record_reads;
      completion.graph_extent_fallback_reads =
        total_graph_extent_fallback_reads;
      completion.graph_extent_underhint_reads =
        total_graph_extent_underhint_reads;
      completion.graph_extent_hint_promotions =
        total_graph_extent_hint_promotions;
      completion.graph_rounds = total_graph_rounds;
      completion.exact_vectors = total_exact_reads;
      completion.dynamic_code_cycles = dynamic_code_cycles;
      completion.dynamic_code_candidates = dynamic_code_candidates;
      completion.dynamic_code_reads = dynamic_code_reads;
      completion.dynamic_code_incarnation_rejects =
        dynamic_code_incarnation_rejects;
      set_dynamic_code_cache_completion(
        completion, dynamic_code_cache_hits,
        dynamic_code_batch_deduplicated,
        dynamic_code_cache_publish_successes, dynamic_code_cache_publish_races,
        dynamic_code_cache_lookup_probe_exhaustions,
        dynamic_code_cache_publish_probe_exhaustions,
        dynamic_code_cache_lookup_probes,
        dynamic_code_cache_max_lookup_probes);
      set_expansion_completion(
        completion, params, sum_selected_parents, sum_feedback_horizon,
        sum_hardware_credit_tiles, minimum_selected_batch,
        maximum_selected_batch, minimum_feedback_horizon,
        maximum_feedback_horizon);
      set_beam_merge_completion(completion, beam_merge_breakdown);
      expansion_pressure_query_exit(params.expansion_pressure);
      set_query_trace_completion<EnableAdjacencyOracle>(
        params, query_slot, completion);
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
    completion.diagnostic = make_query_diagnostic(
      QueryFailureReason::none, route_snapshot_retries);
    completion.gpu_cycles = clock64() - query_started_cycles;
    completion.graph_cycles = graph_phase_cycles;
    completion.score_cycles = score_phase_cycles;
    completion.beam_cycles = beam_phase_cycles;
    completion.exact_cycles = exact_phase_cycles;
    completion.beam_selection_cycles = beam_selection_cycles;
    completion.rdma_issue_cycles = rdma_issue_cycles;
    completion.rdma_wait_cycles = rdma_wait_cycles;
    completion.graph_validation_cycles = graph_validation_cycles;
    completion.neighbor_decode_cycles = neighbor_decode_cycles;
    completion.pq_score_cycles = pq_score_cycles;
    completion.visited_cycles = visited_cycles;
    completion.beam_merge_cycles = beam_merge_cycles;
    completion.remote_pages = total_remote_reads;
    completion.remote_batches = total_remote_batches;
    completion.graph_read_retries = total_graph_read_retries;
    completion.graph_read_bytes = total_graph_read_bytes;
    completion.graph_live_extent_reads =
      total_graph_live_extent_reads;
    completion.graph_full_record_reads =
      total_graph_full_record_reads;
    completion.graph_extent_fallback_reads =
      total_graph_extent_fallback_reads;
    completion.graph_extent_underhint_reads =
      total_graph_extent_underhint_reads;
    completion.graph_extent_hint_promotions =
      total_graph_extent_hint_promotions;
    completion.graph_rounds = total_graph_rounds;
    completion.exact_vectors = total_exact_reads;
    completion.dynamic_code_cycles = dynamic_code_cycles;
    completion.dynamic_code_candidates = dynamic_code_candidates;
    completion.dynamic_code_reads = dynamic_code_reads;
    completion.dynamic_code_incarnation_rejects =
      dynamic_code_incarnation_rejects;
    set_dynamic_code_cache_completion(
      completion, dynamic_code_cache_hits,
      dynamic_code_batch_deduplicated,
      dynamic_code_cache_publish_successes, dynamic_code_cache_publish_races,
      dynamic_code_cache_lookup_probe_exhaustions,
      dynamic_code_cache_publish_probe_exhaustions,
      dynamic_code_cache_lookup_probes,
      dynamic_code_cache_max_lookup_probes);
    set_expansion_completion(
      completion, params, sum_selected_parents, sum_feedback_horizon,
      sum_hardware_credit_tiles, minimum_selected_batch,
      maximum_selected_batch, minimum_feedback_horizon,
      maximum_feedback_horizon);
    set_beam_merge_completion(completion, beam_merge_breakdown);
    expansion_pressure_query_exit(params.expansion_pressure);
    set_query_trace_completion<EnableAdjacencyOracle>(
      params, query_slot, completion);
    device_ring_push(params.completions, completion);
  }
  __syncthreads();
  return;
  }
}

// Unit probes that call the traversal helper directly exercise the production,
// trace-free instantiation.
__device__ void process_query(const PersistentKernelParams& params,
                              const QueryDescriptor& descriptor) {
  process_query<false>(params, descriptor);
}

}  // namespace gpu_search::persistent_kernel_detail
