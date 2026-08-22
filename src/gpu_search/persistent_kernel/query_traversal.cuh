#pragma once

#include "gpu_search/adaptive_frontier.hh"
#include "gpu_search/centroid_route_ranking.hh"
#include "gpu_search/persistent_kernel/rdma_read.cuh"

#include <cuda/atomic>

namespace gpu_search::persistent_kernel_detail {

struct TailFrontierFeedback {
  u32 promoted{};
  u32 retained{};
  u32 stale{};
  u32 queue_rejects{};
  u32 core_hits{};
  u32 core_misses{};
  u32 tail_admitted{};
  u32 commit_waves_observed{};
  u32 commit_waves_covered{};
};

__device__ __forceinline__ void apply_tail_admission_correction(
    TailAdmissionCorrection& correction, TailFrontierFeedback& feedback,
    u32& total_remote_batches, u32& total_remote_reads,
    u64& total_graph_read_bytes, u32& total_graph_live_extent_reads,
    u32& total_graph_full_record_reads, u32& speculative_graph_reads,
    u64& speculative_graph_bytes, u32& speculative_queue_rejects) {
  if (threadIdx.x != 0 || (correction.rejected_batches == 0 &&
                           correction.rejected_reads == 0)) {
    return;
  }
  const u32 rejected_batches = correction.rejected_batches;
  const u32 rejected_reads = correction.rejected_reads;
  const u64 rejected_bytes = correction.rejected_bytes;
  const u32 rejected_live = correction.rejected_live_extent_reads;
  const u32 rejected_full = correction.rejected_full_record_reads;

  // Split issue accounting is intentionally optimistic because the producer
  // cannot know owner SQ slack. Reconcile it exactly once when -EAGAIN proves
  // that no tail WQE was posted. Saturation guards keep diagnostics bounded
  // if a future caller violates the one-shot issue/completion contract.
  total_remote_batches -= min(total_remote_batches, rejected_batches);
  total_remote_reads -= min(total_remote_reads, rejected_reads);
  total_graph_read_bytes -= min(total_graph_read_bytes, rejected_bytes);
  total_graph_live_extent_reads -=
    min(total_graph_live_extent_reads, rejected_live);
  total_graph_full_record_reads -=
    min(total_graph_full_record_reads, rejected_full);
  speculative_graph_reads -=
    min(speculative_graph_reads, rejected_reads);
  speculative_graph_bytes -=
    min(speculative_graph_bytes, rejected_bytes);
  speculative_queue_rejects += rejected_reads;
  feedback.queue_rejects += rejected_reads;
  feedback.tail_admitted -= min(feedback.tail_admitted, rejected_reads);
  correction = {};
}

// The exact Stable-Run issue certificate already maps each next-commit handle
// to its query-local ROB slot.  Preserve that map across the communication
// overlap window instead of repeating selected×ROB and ROB×Beam searches at
// the next epoch.  All fields point to CTA-private shared storage.
struct CertifiedCommitReconcileContext {
  const u64* selected_handles{};
  const u32* certified_rob_slots{};
  FrontierRobEntry* frontier_rob{};
  u32* commit_rob_slots{};
  u32* graph_record_slots{};
  u64* critical_fetch_handles{};
  u32* critical_fetch_to_commit{};
  u32* critical_fetch_count{};
  u32* critical_rob_hits{};
  u32* critical_misses{};
  u32* speculative_promoted{};
  u32* core_prefetch_promoted{};
  u32* core_prefetch_stale{};
  TailFrontierFeedback* feedback{};
};

enum class UnderhintLookupMode : u8 {
  positional = 0,
  certified = 1,
  associative = 2,
};

__device__ __forceinline__ bool is_exact_underhint_evidence(
    const FrontierRobEntry& entry, u64 node_handle) {
  return entry.node_handle == node_handle &&
    entry.state == static_cast<u8>(FrontierRequestState::stale) &&
    entry.validation ==
      static_cast<u8>(FrontierValidationState::extent_underhint);
}

// Snapshot under-hint evidence before reconciliation clears stale ROB entries.
// Positional core and reusable-certificate paths already own exact position to
// ROB maps, so each lane checks only that one slot. Only the general/shadow
// path pays the bounded associative lookup. Exact tagged-handle equality keeps
// evidence query-local and prevents it from being mapped to a different
// incarnation. A warp-wide precheck skips all handle matching in the common
// case where this ROB contains no underhint evidence.
__device__ __forceinline__ void identify_selected_underhint_force_full(
    const u64* selected_handles,
    u32 selected_count,
    const FrontierRobEntry* frontier_rob,
    const u32* certified_rob_slots,
    UnderhintLookupMode lookup_mode,
    u8* selected_force_full,
    u32* any_force_full) {
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 position = threadIdx.x;
    const FrontierRobEntry& lane_entry = frontier_rob[position];
    const bool lane_has_underhint =
      lane_entry.state == static_cast<u8>(FrontierRequestState::stale) &&
      lane_entry.validation ==
        static_cast<u8>(FrontierValidationState::extent_underhint);
    const u32 underhint_mask = __ballot_sync(
      0xffffffffu, lane_has_underhint);
    if (position == 0) *any_force_full = underhint_mask != 0 ? 1u : 0u;
    bool force_full = false;
    if (underhint_mask != 0 && position < selected_count) {
      const u64 handle = selected_handles[position];
      if (lookup_mode == UnderhintLookupMode::positional) {
        force_full =
          is_exact_underhint_evidence(frontier_rob[position], handle);
      } else if (lookup_mode == UnderhintLookupMode::certified) {
        const u32 slot = certified_rob_slots[position];
        force_full = slot < kPersistentFrontierRobCapacity &&
          is_exact_underhint_evidence(frontier_rob[slot], handle);
      } else {
        for (u32 slot = 0; slot < kPersistentFrontierRobCapacity; ++slot) {
          if (is_exact_underhint_evidence(frontier_rob[slot], handle)) {
            force_full = true;
            break;
          }
        }
      }
    }
    if (underhint_mask != 0) {
      selected_force_full[position] = force_full ? 1u : 0u;
    }
  }
  __syncthreads();
}

// Every reconcile path publishes critical_fetch_to_commit. Re-indexing the
// position-local evidence only after compaction avoids threading another flag
// through each path and guarantees baseline misses are explicitly zeroed.
__device__ __forceinline__ void remap_critical_underhint_force_full(
    const u8* selected_force_full,
    u32 selected_count,
    const u32* critical_fetch_to_commit,
    u32 critical_fetch_count,
    u8* critical_force_full) {
  for (u32 fetch = threadIdx.x; fetch < critical_fetch_count;
       fetch += blockDim.x) {
    const u32 position = critical_fetch_to_commit[fetch];
    critical_force_full[fetch] =
      position < selected_count && selected_force_full[position] != 0
        ? 1u : 0u;
  }
  __syncthreads();
}

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
}

__device__ void set_dynamic_code_cache_completion(
    CompletionDescriptor& completion, u32 cache_hits,
    u32 batch_deduplicated, u32 publish_successes,
    u32 first_occupancies, u32 publish_races,
    u32 lookup_probe_exhaustions, u32 publish_probe_exhaustions,
    u32 lookup_probes, u32 max_lookup_probes) {
  completion.dynamic_code_cache_hits = cache_hits;
  completion.dynamic_code_batch_deduplicated = batch_deduplicated;
  completion.dynamic_code_cache_publish_successes = publish_successes;
  completion.dynamic_code_cache_first_occupancies = first_occupancies;
  completion.dynamic_code_cache_publish_races = publish_races;
  completion.dynamic_code_cache_lookup_probe_exhaustions =
    lookup_probe_exhaustions;
  completion.dynamic_code_cache_publish_probe_exhaustions =
    publish_probe_exhaustions;
  completion.dynamic_code_cache_lookup_probes = lookup_probes;
  completion.dynamic_code_cache_max_lookup_probes = max_lookup_probes;
}

__device__ __forceinline__ void set_dynamic_graph_completion(
    CompletionDescriptor& completion,
    const DynamicGraphTelemetry& telemetry) {
  completion.dynamic_graph_read_bytes = telemetry.read_bytes;
  completion.dynamic_graph_short_reads = telemetry.short_reads;
  completion.dynamic_graph_full_reads = telemetry.full_reads;
  completion.dynamic_graph_fallback_reads = telemetry.fallback_reads;
  completion.dynamic_graph_hint_promotions = telemetry.hint_promotions;
  completion.dynamic_graph_hint_demotions = telemetry.hint_demotions;
}

__device__ void set_beam_merge_completion(
    CompletionDescriptor& completion,
    const BeamMergeCycleBreakdown& breakdown) {
  completion.beam_merge_prepare_cycles = breakdown.prepare;
  completion.beam_merge_sort_cycles = breakdown.sort;
  completion.beam_merge_materialize_cycles = breakdown.materialize;
}

template <bool EnableAsfe>
__device__ __forceinline__ void set_terminal_exact_cache_completion(
    CompletionDescriptor& completion,
    const TerminalExactCacheState& cache) {
  if constexpr (EnableAsfe) {
    completion.terminal_exact_cache_attempted_queries =
      cache.attempted != 0 ? 1u : 0u;
    completion.terminal_exact_cache_issued_records = cache.issued_records;
    completion.terminal_exact_cache_promoted_records =
      cache.promoted_records;
    completion.terminal_exact_cache_wasted_bytes = cache.wasted_bytes;
    completion.terminal_exact_cache_queue_rejects = cache.queue_rejects;
    completion.terminal_exact_cache_miss_records = cache.miss_count;
  } else {
    completion.terminal_exact_cache_attempted_queries = 0;
    completion.terminal_exact_cache_issued_records = 0;
    completion.terminal_exact_cache_promoted_records = 0;
    completion.terminal_exact_cache_wasted_bytes = 0;
    completion.terminal_exact_cache_queue_rejects = 0;
    completion.terminal_exact_cache_miss_records = 0;
  }
}

__device__ __noinline__ void set_frontier_completion_full(
    CompletionDescriptor& completion,
    u32 logical_expansions,
    u32 critical_graph_reads,
    u64 critical_graph_bytes,
    u32 speculative_graph_reads,
    u64 speculative_graph_bytes,
    u32 speculative_arrived,
    u32 speculative_promoted,
    u32 speculative_stale,
    u64 speculative_wasted_bytes,
    u32 speculative_queue_rejects,
    u32 issue_epochs,
    u32 commit_epochs,
    u64 issue_width_sum,
    u64 issue_width_capacity_sum,
    u64 commit_width_sum,
    u32 max_issue_width,
    u32 max_commit_width,
    u32 critical_rob_hits,
    u32 critical_misses,
    u64 speculative_wait_cycles,
    u64 rdma_completion_latency_ns,
    u64 speculative_completion_latency_ns,
    u64 rdma_completion_groups,
    u64 speculative_completion_groups,
    u64 core_prefetch_bytes,
    u32 core_prefetch_reads,
    u32 core_prefetch_arrived,
    u32 core_prefetch_promoted,
    u32 core_prefetch_stale,
    u32 core_prefetch_queue_rejects,
    u32 core_prefetch_waves,
    u32 core_ready_waves) {
  completion.logical_expansions = logical_expansions;
  completion.critical_graph_reads = critical_graph_reads;
  completion.critical_graph_bytes = critical_graph_bytes;
  completion.speculative_graph_reads = speculative_graph_reads;
  completion.speculative_graph_bytes = speculative_graph_bytes;
  completion.speculative_arrived = speculative_arrived;
  completion.speculative_promoted = speculative_promoted;
  completion.speculative_stale = speculative_stale;
  completion.speculative_wasted_bytes = speculative_wasted_bytes;
  completion.speculative_queue_rejects = speculative_queue_rejects;
  completion.issue_epochs = issue_epochs;
  completion.commit_epochs = commit_epochs;
  completion.issue_width_sum = issue_width_sum;
  completion.issue_width_capacity_sum = issue_width_capacity_sum;
  completion.commit_width_sum = commit_width_sum;
  completion.max_issue_width = max_issue_width;
  completion.max_commit_width = max_commit_width;
  completion.critical_rob_hits = critical_rob_hits;
  completion.critical_misses = critical_misses;
  completion.speculative_wait_cycles = speculative_wait_cycles;
  completion.rdma_completion_latency_ns = rdma_completion_latency_ns;
  completion.speculative_completion_latency_ns =
    speculative_completion_latency_ns;
  completion.rdma_completion_groups = rdma_completion_groups;
  completion.speculative_completion_groups = speculative_completion_groups;
  completion.core_prefetch_bytes = core_prefetch_bytes;
  completion.core_prefetch_reads = core_prefetch_reads;
  completion.core_prefetch_arrived = core_prefetch_arrived;
  completion.core_prefetch_promoted = core_prefetch_promoted;
  completion.core_prefetch_stale = core_prefetch_stale;
  completion.core_prefetch_queue_rejects = core_prefetch_queue_rejects;
  completion.core_prefetch_waves = core_prefetch_waves;
  completion.core_ready_waves = core_ready_waves;
}

template <bool EnableAsfe>
__device__ __forceinline__ void set_frontier_completion(
    CompletionDescriptor& completion,
    u32 logical_expansions,
    u32 critical_graph_reads,
    u64 critical_graph_bytes,
    u32 speculative_graph_reads,
    u64 speculative_graph_bytes,
    u32 speculative_arrived,
    u32 speculative_promoted,
    u32 speculative_stale,
    u64 speculative_wasted_bytes,
    u32 speculative_queue_rejects,
    u32 issue_epochs,
    u32 commit_epochs,
    u64 issue_width_sum,
    u64 issue_width_capacity_sum,
    u64 commit_width_sum,
    u32 max_issue_width,
    u32 max_commit_width,
    u32 critical_rob_hits,
    u32 critical_misses,
    u64 speculative_wait_cycles,
    u64 rdma_completion_latency_ns,
    u64 speculative_completion_latency_ns,
    u64 rdma_completion_groups,
    u64 speculative_completion_groups,
    u64 core_prefetch_bytes,
    u32 core_prefetch_reads,
    u32 core_prefetch_arrived,
    u32 core_prefetch_promoted,
    u32 core_prefetch_stale,
    u32 core_prefetch_queue_rejects,
    u32 core_prefetch_waves,
    u32 core_ready_waves) {
  if constexpr (EnableAsfe) {
    set_frontier_completion_full(
      completion, logical_expansions, critical_graph_reads,
      critical_graph_bytes, speculative_graph_reads,
      speculative_graph_bytes, speculative_arrived,
      speculative_promoted, speculative_stale,
      speculative_wasted_bytes, speculative_queue_rejects,
      issue_epochs, commit_epochs, issue_width_sum,
      issue_width_capacity_sum, commit_width_sum, max_issue_width,
      max_commit_width, critical_rob_hits, critical_misses,
      speculative_wait_cycles, rdma_completion_latency_ns,
      speculative_completion_latency_ns, rdma_completion_groups,
      speculative_completion_groups, core_prefetch_bytes,
      core_prefetch_reads, core_prefetch_arrived, core_prefetch_promoted,
      core_prefetch_stale, core_prefetch_queue_rejects,
      core_prefetch_waves, core_ready_waves);
  } else {
    completion.logical_expansions = logical_expansions;
    completion.critical_graph_reads = critical_graph_reads;
    completion.critical_graph_bytes = critical_graph_bytes;
    completion.speculative_graph_reads = 0;
    completion.speculative_graph_bytes = 0;
    completion.speculative_arrived = 0;
    completion.speculative_promoted = 0;
    completion.speculative_stale = 0;
    completion.speculative_wasted_bytes = 0;
    completion.speculative_queue_rejects = 0;
    completion.issue_epochs = 0;
    completion.commit_epochs = commit_epochs;
    completion.issue_width_sum = 0;
    completion.issue_width_capacity_sum = 0;
    completion.commit_width_sum = commit_width_sum;
    completion.max_issue_width = 0;
    completion.max_commit_width = max_commit_width;
    completion.critical_rob_hits = 0;
    completion.critical_misses = critical_misses;
    completion.speculative_wait_cycles = 0;
    completion.rdma_completion_latency_ns = rdma_completion_latency_ns;
    completion.speculative_completion_latency_ns = 0;
    completion.rdma_completion_groups = rdma_completion_groups;
    completion.speculative_completion_groups = 0;
    completion.core_prefetch_bytes = 0;
    completion.core_prefetch_reads = 0;
    completion.core_prefetch_arrived = 0;
    completion.core_prefetch_promoted = 0;
    completion.core_prefetch_stale = 0;
    completion.core_prefetch_queue_rejects = 0;
    completion.core_prefetch_waves = 0;
    completion.core_ready_waves = 0;
  }
}

template <bool EnableAsfe>
__device__ __forceinline__ void set_frontier_certificate_completion(
    CompletionDescriptor& completion, u32 reusable_certificates,
    u32 streamed_candidate_runs, u32 ordered_score_batches,
    u32 ordered_score_candidates, u32 reusable_prefix_ranks,
    u32 reusable_full_prefix_certificates,
    u32 reusable_issued_certificates,
    u32 ooo_bypassed_parents,
    u32 certificate_rejects) {
  if constexpr (EnableAsfe) {
    // DEEC remains available as a focused proof/test primitive; production
    // uses the sort-once reusable PFEC certificate below.
    completion.frontier_telemetry_reserved0 = certificate_rejects;
    // The second retired DEEC slot is a success-only counter. Failure paths
    // keep it zero until they optionally overwrite both reserved words with
    // a rejected-handle diagnostic.
    completion.frontier_telemetry_reserved1 =
      completion.status == 0 ? ooo_bypassed_parents : 0u;
    completion.frontier_reusable_certificates = reusable_certificates;
    completion.frontier_streamed_candidate_runs =
      streamed_candidate_runs;
    completion.ordered_score_batches = ordered_score_batches;
    completion.ordered_score_candidates = ordered_score_candidates;
    completion.frontier_reusable_prefix_ranks =
      reusable_prefix_ranks;
    completion.frontier_reusable_full_prefix_certificates =
      reusable_full_prefix_certificates;
    completion.frontier_reusable_issued_certificates =
      reusable_issued_certificates;
  } else {
    completion.completion_score_batches = 0;
    completion.completion_score_candidates = 0;
    completion.frontier_telemetry_reserved0 = 0;
    completion.frontier_telemetry_reserved1 = 0;
    completion.frontier_reusable_certificates = 0;
    completion.frontier_streamed_candidate_runs = 0;
    completion.ordered_score_batches = 0;
    completion.ordered_score_candidates = 0;
    completion.frontier_reusable_prefix_ranks = 0;
    completion.frontier_reusable_full_prefix_certificates = 0;
    completion.frontier_reusable_issued_certificates = 0;
  }
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


// Reconcile one completed issue wave with the next exact Stable-Run frontier.
// The preview reads the already sorted runs, so the core prefix is
// authoritative for the Beam that finish_approximate_stable_runs will publish.
// Only validated payloads are retained; COMMITTED payloads have already been
// decoded, and every unretained shadow payload is charged as waste exactly
// once. One warp performs matching, reclamation and allocation with ballot
// compaction, so no global queue or per-entry atomics are needed.
__device__ __noinline__ void prepare_issue_frontier_entries(
    const u64* issue_handles, const u16* issue_ranks,
    u32& issue_count, FrontierRobEntry* frontier_rob,
    u32& issue_epoch, adaptive_frontier::ControllerState& controller,
    u32 core_slot_count, TailFrontierFeedback& feedback,
    u32& speculative_stale,
    u64& speculative_wasted_bytes, u32& core_prefetch_stale,
    u32& issue_epochs, u64& issue_width_sum,
    u64& issue_width_capacity_sum, u32& observed_max_issue_width,
    u32* issue_rob_slots, u32& physical_issue_span,
    bool allow_new_tail = true,
    bool apply_controller_feedback = true,
    bool start_new_issue_epoch = true) {
  constexpr u32 full_warp = 0xffffffffu;
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    if (threadIdx.x == 0) {
      // The caller normally materializes exactly this many preview entries.
      // Keep the helper defensive without letting the certificate update
      // below retroactively resize an already-generated wave.
      issue_count = min(issue_count, controller.current_issue_width);
    }
    __syncwarp(full_warp);
  }
  // Attribute benefit only at an exact Stable-Run certificate, never at the
  // round in which a shadow request happened to be admitted.  Early shadow
  // predicts this preview and split tail predicts a later one, so checking
  // here gives both paths the same unambiguous wave identity.  ARRIVED and
  // INFLIGHT entries deliberately do not count: the entire actual Commit
  // prefix must already be validated when authority is established. Fold
  // this certificate into the mandatory handle-to-ROB mapping below rather
  // than adding a second Commit-by-ROB scan.
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 issue_position = threadIdx.x;
    u32 matching_slot = UINT32_MAX;
    if (issue_position < issue_count) {
      for (u32 slot = 0; slot < kPersistentFrontierRobCapacity; ++slot) {
        const FrontierRobEntry& entry = frontier_rob[slot];
        const bool resident_tail =
          !allow_new_tail && slot >= core_slot_count &&
          (entry.state ==
               static_cast<u8>(FrontierRequestState::inflight) ||
           entry.state ==
               static_cast<u8>(FrontierRequestState::arrived));
        if ((entry.state ==
               static_cast<u8>(FrontierRequestState::validated) ||
             resident_tail) &&
            entry.node_handle == issue_handles[issue_position]) {
          matching_slot = slot;
          break;
        }
      }
    }
    issue_rob_slots[issue_position] = matching_slot;

    const bool observe_wave =
      apply_controller_feedback && feedback.tail_admitted != 0 &&
      feedback.commit_waves_observed == 0;
    const u32 commit_prefix_count = min(issue_count, core_slot_count);
    const bool exact_speculative_match =
      observe_wave && issue_position < commit_prefix_count &&
      matching_slot != UINT32_MAX &&
      frontier_rob[matching_slot].state ==
        static_cast<u8>(FrontierRequestState::validated) &&
      frontier_rob[matching_slot].priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const u32 covered_mask =
      __ballot_sync(full_warp, exact_speculative_match);
    if (issue_position == 0 && observe_wave) {
      feedback.commit_waves_observed = 1;
      const u32 required_mask =
        commit_prefix_count == kPersistentFrontierRobCapacity
          ? full_warp
          : ((u32{1} << commit_prefix_count) - 1u);
      feedback.commit_waves_covered =
        commit_prefix_count != 0 &&
        (covered_mask & required_mask) == required_mask
          ? 1u : 0u;
    }
  }
  if (threadIdx.x == 0) {
    if (start_new_issue_epoch) ++issue_epoch;
  }
  __syncthreads();

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 slot = threadIdx.x;
    FrontierRobEntry& entry = frontier_rob[slot];
    bool discarded_speculative = false;
    bool discarded_core = false;
    bool retained_speculative = false;
    u64 discarded_bytes = 0;
    if (entry.state ==
          static_cast<u8>(FrontierRequestState::committed) ||
        entry.state == static_cast<u8>(FrontierRequestState::stale)) {
      entry = {};
    } else if (entry.state ==
               static_cast<u8>(FrontierRequestState::validated)) {
      bool retained = false;
      for (u32 issue_position = 0; issue_position < issue_count;
           ++issue_position) {
        if (issue_rob_slots[issue_position] != slot) continue;
        retained = true;
        entry.issue_epoch = issue_epoch;
        entry.beam_rank = issue_ranks[issue_position];
        break;
      }
      if (!retained) {
        discarded_speculative = entry.priority ==
          static_cast<u8>(DirectBatchPriority::speculative);
        discarded_core = !discarded_speculative;
        discarded_bytes =
          discarded_speculative ? entry.transfer_bytes : 0;
        entry = {};
      } else {
        const bool newly_accounted =
          entry.priority ==
            static_cast<u8>(DirectBatchPriority::speculative) &&
          (entry.flags & kFrontierRobFlagUtilityAccounted) == 0;
        // The entry has now survived the exact post-merge certificate.  It
        // is no longer merely a pre-merge prediction, so clear the provenance
        // bit and account this physical read at most once before feeding
        // frontier-stability evidence to the controller.
        entry.flags &= static_cast<u8>(~kFrontierRobFlagEarlyShadow);
        if (newly_accounted) {
          entry.flags |= kFrontierRobFlagUtilityAccounted;
        }
        retained_speculative = newly_accounted;
      }
    }
    const u32 discarded_speculative_mask =
      __ballot_sync(full_warp, discarded_speculative);
    const u32 discarded_core_mask =
      __ballot_sync(full_warp, discarded_core);
    const u32 retained_speculative_mask =
      __ballot_sync(full_warp, retained_speculative);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      discarded_bytes +=
        __shfl_down_sync(full_warp, discarded_bytes, offset);
    }
    if (slot == 0) {
      const u32 speculative_count =
        __popc(discarded_speculative_mask);
      speculative_stale += speculative_count;
      core_prefetch_stale += __popc(discarded_core_mask);
      speculative_wasted_bytes += discarded_bytes;
      feedback.stale += speculative_count;
      // A validated tail record that survives the exact next Stable-Run
      // preview is frontier-stability evidence, but it has not yet eliminated
      // a critical read. The controller may use retention to preserve a width
      // or strengthen whole-wave coverage; retention alone cannot bootstrap
      // another shadow slot. Stale/rejected bytes still drive contraction.
      feedback.retained += __popc(retained_speculative_mask);
    }
  }
  __syncthreads();

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 issue_position = threadIdx.x;
    const bool missing =
      issue_position < issue_count &&
      issue_rob_slots[issue_position] == UINT32_MAX &&
      (issue_position < core_slot_count || allow_new_tail);
    const bool core_issue = issue_position < core_slot_count;
    const bool free_core =
      issue_position < core_slot_count &&
      frontier_rob[issue_position].state ==
        static_cast<u8>(FrontierRequestState::init);
    const bool free_tail =
      issue_position >= core_slot_count &&
      frontier_rob[issue_position].state ==
        static_cast<u8>(FrontierRequestState::init);
    const u32 missing_core_mask = __ballot_sync(
      full_warp, missing && core_issue);
    const u32 missing_tail_mask = __ballot_sync(
      full_warp, missing && !core_issue);
    const u32 free_core_mask = __ballot_sync(full_warp, free_core);
    const u32 free_tail_mask = __ballot_sync(full_warp, free_tail);
    u32 destination = issue_rob_slots[issue_position];
    if (missing) {
      const u32 lower_issue_positions =
        issue_position == 0
          ? 0u : ((u32{1} << issue_position) - 1u);
      u32 free_ordinal = __popc(
        (core_issue ? missing_core_mask : missing_tail_mask) &
        lower_issue_positions);
      const u32 begin = core_issue ? 0u : core_slot_count;
      const u32 end = core_issue
        ? core_slot_count
        : static_cast<u32>(kPersistentFrontierRobCapacity);
      const u32 free_mask =
        core_issue ? free_core_mask : free_tail_mask;
      for (u32 slot = begin; slot < end; ++slot) {
        if ((free_mask & (u32{1} << slot)) == 0) continue;
        if (free_ordinal == 0) {
          destination = slot;
          break;
        }
        --free_ordinal;
      }
      if (destination != UINT32_MAX) {
        FrontierRobEntry& entry = frontier_rob[destination];
        entry = {};
        entry.node_handle = issue_handles[issue_position];
        entry.issue_epoch = issue_epoch;
        entry.beam_rank = issue_ranks[issue_position];
        entry.scratch_slot = static_cast<u8>(destination);
        entry.state = static_cast<u8>(FrontierRequestState::issued);
        entry.priority = static_cast<u8>(
          core_issue
            ? DirectBatchPriority::critical
            : DirectBatchPriority::speculative);
      }
    }
    issue_rob_slots[issue_position] = destination;
    const u32 admitted_mask = __ballot_sync(
      full_warp,
      issue_position < issue_count && destination != UINT32_MAX);
    u32 mapped_span =
      issue_position < issue_count && destination != UINT32_MAX
        ? destination + 1u : 0u;
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      mapped_span = max(
        mapped_span,
        __shfl_down_sync(full_warp, mapped_span, offset));
    }
    if (issue_position == 0) {
      const u32 admitted_issue_count = __popc(admitted_mask);
      // Logical certificate positions and physical ROB slots cease to be a
      // dense identity map as soon as a validated tail record is promoted,
      // or when a resident tail leaves a logical hole.  Preserve issue_count
      // as the immutable certificate span: every later lookup is explicitly
      // through issue_rob_slots[position].  Only accounting uses the admitted
      // popcount, while the descriptor uses the physical high-water mark.
      physical_issue_span = mapped_span;
      if (start_new_issue_epoch) {
        ++issue_epochs;
        issue_width_sum += admitted_issue_count;
        issue_width_capacity_sum += controller.current_issue_width;
      }
      observed_max_issue_width =
        max(observed_max_issue_width, admitted_issue_count);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && apply_controller_feedback) {
    // The exact certificate, completion state, and retention/staleness
    // classification above all describe the same physical wave. Train once
    // on that complete observation, then reset before the newly admitted
    // wave can contribute feedback. The preview was generated at the old
    // width, so the updated width deliberately takes effect on the next
    // preview; both growth and contraction are one-wave delayed and policy
    // shrink cannot relabel a valid retained record as stale.
    adaptive_frontier::update_issue_width(
      controller,
      adaptive_frontier::Feedback{
        .promoted = feedback.promoted,
        .retained = feedback.retained,
        .stale = feedback.stale,
        .queue_rejects = feedback.queue_rejects,
        .critical_misses = feedback.core_misses,
        .tail_admitted = feedback.tail_admitted,
        .commit_waves_observed = feedback.commit_waves_observed,
        .commit_waves_covered = feedback.commit_waves_covered,
      });
    feedback = {};
  }
  __syncthreads();
}

struct BeamShadowFrontierContext {
  FrontierRobEntry* frontier_rob{};
  adaptive_frontier::ControllerState* controller{};
  TailFrontierFeedback* feedback{};
  u32* issue_epoch{};
  u64* shadow_handles{};
  u16* shadow_ranks{};
  u32* shadow_count{};
  u32* shadow_rob_slots{};
  u32* new_shadow_count{};
  u32* speculative_stale{};
  u64* speculative_wasted_bytes{};
  u32* issue_epochs{};
  u64* issue_width_sum{};
  u64* issue_width_capacity_sum{};
  u32* observed_max_issue_width{};
};

// Admit one predicted issue wave proposed by the partial Stable-Run merge.
// RDMA issue overlaps the unchanged authoritative materialization; the next
// epoch still validates every entry against the newly merged Beam before it
// can become authoritative. The whole query-local ROB is available because
// exact misses are discovered only after this wave has been reconciled.
//
// A query starts from the current commit demand plus a bounded online probe.
// Promotion/retention and stale/rejected bytes adjust the following wave
// proportionally, without a dataset-specific predictor width.
__device__ __noinline__ void prepare_candidate_shadow_frontier(
    u32 proposed_count, u32 remaining_after_commit,
    const BeamShadowFrontierContext& context) {
  constexpr u32 full_warp = 0xffffffffu;
  adaptive_frontier::ControllerState& controller = *context.controller;
  TailFrontierFeedback& feedback = *context.feedback;
  u32& issue_epoch = *context.issue_epoch;
  u32& shadow_count = *context.shadow_count;
  u32& new_shadow_count = *context.new_shadow_count;
  if (threadIdx.x == 0) {
    adaptive_frontier::update_issue_width(
      controller,
      adaptive_frontier::Feedback{
        .promoted = feedback.promoted,
        .retained = feedback.retained,
        .stale = feedback.stale,
        .queue_rejects = feedback.queue_rejects,
        .critical_misses = feedback.core_misses,
        .tail_admitted = feedback.tail_admitted,
        .commit_waves_observed = feedback.commit_waves_observed,
        .commit_waves_covered = feedback.commit_waves_covered,
      });
    u32 shadow_budget = min(
      controller.current_issue_width,
      static_cast<u32>(kPersistentFrontierRobCapacity));
    shadow_budget = min(shadow_budget, remaining_after_commit);

    shadow_count = min(proposed_count, shadow_budget);
    ++issue_epoch;
    feedback = {};
    new_shadow_count = 0;
  }
  __syncthreads();

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 position = threadIdx.x;
    u32 matching_slot = UINT32_MAX;
    if (position < shadow_count) {
      for (u32 slot = 0;
           slot < kPersistentFrontierRobCapacity; ++slot) {
        const FrontierRobEntry& entry = context.frontier_rob[slot];
        if (entry.state ==
              static_cast<u8>(FrontierRequestState::validated) &&
            entry.node_handle == context.shadow_handles[position]) {
          matching_slot = slot;
          break;
        }
      }
    }
    context.shadow_rob_slots[position] = matching_slot;
  }
  __syncthreads();

  // Reclaim validated tail payloads that fell outside the new shadow prefix.
  // Committed slots still feed the current expansion and cannot be reused
  // until graph decode has consumed them.
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 slot = threadIdx.x;
    FrontierRobEntry& entry = context.frontier_rob[slot];
    bool discarded = false;
    bool retained = false;
    u64 discarded_bytes = 0;
    if (entry.state ==
        static_cast<u8>(FrontierRequestState::validated)) {
      for (u32 position = 0; position < shadow_count; ++position) {
        if (context.shadow_rob_slots[position] != slot) continue;
        retained = true;
        entry.issue_epoch = issue_epoch;
        entry.beam_rank = context.shadow_ranks[position];
        break;
      }
      if (!retained) {
        discarded = true;
        discarded_bytes = entry.transfer_bytes;
        entry = {};
      }
    } else if (entry.state ==
               static_cast<u8>(FrontierRequestState::stale)) {
      entry = {};
    }
    const u32 discarded_mask = __ballot_sync(full_warp, discarded);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      discarded_bytes +=
        __shfl_down_sync(full_warp, discarded_bytes, offset);
    }
    if (slot == 0) {
      const u32 discarded_count = __popc(discarded_mask);
      *context.speculative_stale += discarded_count;
      *context.speculative_wasted_bytes += discarded_bytes;
      feedback.stale += discarded_count;
    }
  }
  __syncthreads();

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 position = threadIdx.x;
    const bool missing =
      position < shadow_count &&
      context.shadow_rob_slots[position] == UINT32_MAX;
    const bool free_tail =
      context.frontier_rob[threadIdx.x].state ==
        static_cast<u8>(FrontierRequestState::init);
    const u32 missing_mask = __ballot_sync(full_warp, missing);
    const u32 free_mask = __ballot_sync(full_warp, free_tail);
    u32 destination = context.shadow_rob_slots[position];
    if (missing) {
      u32 free_ordinal = __popc(
        missing_mask &
        (position == 0 ? 0u : ((u32{1} << position) - 1u)));
      for (u32 slot = 0;
           slot < kPersistentFrontierRobCapacity; ++slot) {
        if ((free_mask & (u32{1} << slot)) == 0) continue;
        if (free_ordinal == 0) {
          destination = slot;
          break;
        }
        --free_ordinal;
      }
      if (destination != UINT32_MAX) {
        FrontierRobEntry& entry = context.frontier_rob[destination];
        entry = {};
        entry.node_handle = context.shadow_handles[position];
        entry.issue_epoch = issue_epoch;
        entry.beam_rank = context.shadow_ranks[position];
        entry.scratch_slot = static_cast<u8>(destination);
        entry.state = static_cast<u8>(FrontierRequestState::issued);
        entry.priority =
          static_cast<u8>(DirectBatchPriority::speculative);
      }
    }
    context.shadow_rob_slots[position] = destination;
    const bool admitted =
      position < shadow_count && destination != UINT32_MAX;
    const u32 admitted_mask = __ballot_sync(full_warp, admitted);
    const u32 new_mask =
      __ballot_sync(full_warp, admitted && missing);
    if (position == 0) {
      shadow_count = __popc(admitted_mask);
      new_shadow_count = __popc(new_mask);
      ++*context.issue_epochs;
      const u32 logical_issue_width = shadow_count;
      *context.issue_width_sum += logical_issue_width;
      *context.issue_width_capacity_sum += controller.current_issue_width;
      *context.observed_max_issue_width =
        max(*context.observed_max_issue_width, logical_issue_width);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ bool finish_query_core_frontier_batch(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    FrontierRobEntry* frontier_rob,
    FrontierGraphBatchState& core_batch,
    u32 slot_count,
    const CoreFrontierTelemetry& telemetry) {
  return finish_core_frontier_graph_batch(
    params, descriptor, frontier_rob, core_batch, slot_count,
    params.core_batch_statuses,
    params.core_batch_completion_timestamps_ns,
    telemetry.wait_cycles, telemetry.completion_latency_ns,
    telemetry.completion_groups, telemetry.arrived, telemetry.stale,
    telemetry.ready_waves, telemetry.dynamic_graph);
}

// Failure completions normally collapse every query-local graph dependency
// error into QueryFailureReason::graph_fetch.  Preserve a compact, zero-
// allocation postmortem in result_count (which is otherwise meaningless for
// a failed query): bits [7:0] identify the call site and bits [23:8] contain
// the magnitude of the first terminal core completion status.  The scan runs
// only after a fatal transition, so the successful query path is unchanged.
__device__ __forceinline__ u32 frontier_graph_failure_code(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    u32 stage) {
  i32 first_status = 0;
  if (params.core_batch_statuses != nullptr) {
    const i32* statuses = params.core_batch_statuses +
      static_cast<size_t>(descriptor.query_slot) * params.num_shards;
    for (u32 shard = 0; shard < params.num_shards; ++shard) {
      const i32 status =
        *reinterpret_cast<const volatile i32*>(statuses + shard);
      if (status == 0 || status == -EINPROGRESS || status == -EAGAIN) {
        continue;
      }
      first_status = status;
      break;
    }
  }
  const u32 magnitude = first_status < 0
    ? min(static_cast<u32>(-(first_status + 1)) + 1u, 0xffffu)
    : min(static_cast<u32>(first_status), 0xffffu);
  return (stage & 0xffu) | (magnitude << 8);
}

// Keep frontier selection and ROB reconciliation out of process_query's long
// live range. The metadata is query-CTA local; lane zero freezes the exact
// authoritative prefix and the first warp performs ROB matching, stale
// reconciliation, and miss compaction without a global queue.
__device__ __forceinline__ void select_commit_frontier(
    const u64* beam_handles, const u8* beam_expanded, u32 beam_count,
    u32 target, u64* selected_handles, u32* selected_beam_ranks,
    u32* commit_rob_slots, u32& selected_count, u32& shadow_frontier_count,
    u32& commit_epochs, u64& commit_width_sum, u32& max_commit_width) {
  if (threadIdx.x == 0) {
    selected_count = 0;
    for (u32 index = 0;
         index < beam_count && selected_count < target; ++index) {
      if (beam_expanded[index] != 0) continue;
      selected_handles[selected_count] = beam_handles[index];
      selected_beam_ranks[selected_count] = index;
      commit_rob_slots[selected_count] = UINT32_MAX;
      ++selected_count;
    }
    shadow_frontier_count = 0;
    if (selected_count != 0) {
      ++commit_epochs;
      commit_width_sum += selected_count;
      max_commit_width = max(max_commit_width, selected_count);
    }
  }
  __syncthreads();
}

__device__ __noinline__ void plan_commit_frontier(
    const u64* beam_handles, const u8* beam_expanded, u32 beam_count,
    u32 target, FrontierRobEntry* frontier_rob,
    adaptive_frontier::ControllerState& controller,
    bool issue_wave, u32 feedback_stale,
    u32 feedback_queue_rejects,
    u64* selected_handles, u32* selected_beam_ranks,
    u32* commit_rob_slots, u32& selected_count,
    u64* critical_fetch_handles, u32* critical_fetch_to_commit,
    u32& critical_fetch_count, u32* graph_record_slots,
    u32& shadow_frontier_count, u32& speculative_stale,
    u64& speculative_wasted_bytes, u32& speculative_promoted,
    u32& core_prefetch_stale, u32& core_prefetch_promoted,
    u32& critical_rob_hits, u32& critical_misses,
    u32& commit_epochs, u64& commit_width_sum, u32& max_commit_width,
    TailFrontierFeedback& tail_feedback,
    bool selection_prepared = false) {
  if (!selection_prepared) {
    select_commit_frontier(
      beam_handles, beam_expanded, beam_count, target,
      selected_handles, selected_beam_ranks, commit_rob_slots,
      selected_count, shadow_frontier_count, commit_epochs,
      commit_width_sum, max_commit_width);
  }
  if (threadIdx.x == 0) critical_fetch_count = 0;
  __syncthreads();

  // Coupled Stable-Run + Live-Extent baseline: no ROB state exists and every
  // authoritative commit item is a critical fetch. Keep this path to one
  // ballot/compaction warp so ASFE instrumentation cannot tax the reference
  // execution.
  if (controller.max_issue_width == controller.commit_width) {
    if (threadIdx.x < kPersistentFrontierRobCapacity) {
      constexpr u32 full_warp = 0xffffffffu;
      const u32 position = threadIdx.x;
      const bool selected = position < selected_count;
      const u32 selected_mask = __ballot_sync(full_warp, selected);
      const u32 lower_lanes =
        position == 0 ? 0u : ((u32{1} << position) - 1u);
      if (selected) {
        const u32 fetch = __popc(selected_mask & lower_lanes);
        critical_fetch_to_commit[fetch] = position;
        critical_fetch_handles[fetch] = selected_handles[position];
      }
      if (position == 0) {
        critical_fetch_count = __popc(selected_mask);
        critical_misses += critical_fetch_count;
      }
    }
    __syncthreads();
    return;
  }

  // The overwhelmingly common steady state has no resident tail: the exact
  // preview issued commit-width records into positional core slots. Commit
  // rank therefore maps directly to scratch/ROB rank. Avoid all associative
  // ROB searches and retention scans in that state; the general path remains
  // for a real shadow tail or promoted tail record.
  if (controller.max_issue_width > controller.commit_width &&
      threadIdx.x < kPersistentFrontierRobCapacity) {
    constexpr u32 full_warp = 0xffffffffu;
    const u32 lane = threadIdx.x;
    const bool live_tail =
      lane >= controller.commit_width &&
      frontier_rob[lane].state !=
        static_cast<u8>(FrontierRequestState::init);
    const u32 tail_mask = __ballot_sync(full_warp, live_tail);
    const bool speculative_core =
      lane < controller.commit_width &&
      frontier_rob[lane].state !=
        static_cast<u8>(FrontierRequestState::init) &&
      frontier_rob[lane].priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const u32 speculative_core_mask =
      __ballot_sync(full_warp, speculative_core);
    if (lane == 0) {
      shadow_frontier_count =
        __popc(tail_mask) + (speculative_core_mask == 0 ? 0u : 1u);
    }
  }
  __syncthreads();
  if (controller.max_issue_width > controller.commit_width &&
      shadow_frontier_count == 0) {
    if (threadIdx.x < kPersistentFrontierRobCapacity) {
      constexpr u32 full_warp = 0xffffffffu;
      const u32 position = threadIdx.x;
      const bool selected = position < selected_count;
      FrontierRobEntry& entry = frontier_rob[position];
      const bool hit =
        selected && position < controller.commit_width &&
        entry.state == static_cast<u8>(FrontierRequestState::validated) &&
        entry.node_handle == selected_handles[position];
      const bool stale_core =
        position < controller.commit_width &&
        entry.state != static_cast<u8>(FrontierRequestState::init) &&
        !hit;
      if (hit) {
        commit_rob_slots[position] = position;
        entry.state = static_cast<u8>(FrontierRequestState::committed);
        graph_record_slots[position] =
          kGraphScratchBit | static_cast<u32>(entry.scratch_slot);
      } else if (stale_core) {
        entry = {};
      }
      const u32 miss_mask =
        __ballot_sync(full_warp, selected && !hit);
      const u32 lower_lanes =
        position == 0 ? 0u : ((u32{1} << position) - 1u);
      if (selected && !hit) {
        const u32 fetch = __popc(miss_mask & lower_lanes);
        critical_fetch_to_commit[fetch] = position;
        critical_fetch_handles[fetch] = selected_handles[position];
      }
      const u32 hit_mask = __ballot_sync(full_warp, hit);
      const u32 stale_mask = __ballot_sync(full_warp, stale_core);
      if (position == 0) {
        const u32 hits = __popc(hit_mask);
        const u32 misses = __popc(miss_mask);
        critical_fetch_count = __popc(miss_mask);
        critical_misses += critical_fetch_count;
        critical_rob_hits += hits;
        core_prefetch_promoted += hits;
        core_prefetch_stale += __popc(stale_mask);
        tail_feedback.stale += feedback_stale;
        tail_feedback.queue_rejects += feedback_queue_rejects;
        if (issue_wave) {
          tail_feedback.core_hits += hits;
          tail_feedback.core_misses += misses;
        }
      }
    }
    __syncthreads();
    return;
  }

  // The coupled baseline never creates ROB entries. Avoid making it pay the
  // candidate's bounded reconciliation work so the A/B comparison isolates
  // issue/commit decoupling. In the candidate, one warp owns the complete
  // 32-entry ROB: one lane per commit position/slot, with no scalar 32x32
  // nesting on the query's control lane.
  if (controller.max_issue_width > controller.commit_width &&
      threadIdx.x < kPersistentFrontierRobCapacity) {
    constexpr u32 full_warp = 0xffffffffu;
    const u32 lane = threadIdx.x;
    if (lane < selected_count) {
      u32 matching_slot = UINT32_MAX;
      for (u32 slot = 0; slot < kPersistentFrontierRobCapacity; ++slot) {
        const FrontierRobEntry& entry = frontier_rob[slot];
        if (entry.state ==
              static_cast<u8>(FrontierRequestState::validated) &&
            entry.node_handle == selected_handles[lane]) {
          matching_slot = slot;
          break;
        }
      }
      commit_rob_slots[lane] = matching_slot;
    }
    __syncwarp(full_warp);

    FrontierRobEntry& entry = frontier_rob[lane];
    bool stale_speculative = false;
    bool stale_core = false;
    bool retained_speculative = false;
    u64 stale_bytes = 0;
    if (entry.state == static_cast<u8>(FrontierRequestState::stale)) {
      entry = {};
    } else if (entry.state ==
               static_cast<u8>(FrontierRequestState::validated)) {
      bool selected = false;
      for (u32 position = 0; position < selected_count; ++position) {
        if (commit_rob_slots[position] == lane) {
          selected = true;
          break;
        }
      }
      if (!selected) {
        bool still_unexpanded = false;
        u32 frontier_ordinal = 0;
        for (u32 rank = 0; rank < beam_count; ++rank) {
          if (beam_expanded[rank] != 0) continue;
          if (frontier_ordinal >= controller.current_issue_width) break;
          ++frontier_ordinal;
          if (beam_handles[rank] == entry.node_handle) {
            still_unexpanded = true;
            break;
          }
        }
        if (!still_unexpanded) {
          stale_speculative =
            entry.priority ==
            static_cast<u8>(DirectBatchPriority::speculative);
          stale_core = !stale_speculative;
          stale_bytes = stale_speculative ? entry.transfer_bytes : 0;
          entry = {};
        } else {
          const bool newly_accounted =
            entry.priority ==
              static_cast<u8>(DirectBatchPriority::speculative) &&
            (entry.flags & kFrontierRobFlagUtilityAccounted) == 0;
          // Exact Beam inspection has confirmed this previously early-issued
          // handle is still in the authoritative unexpanded frontier.
          // Promote its metadata from prediction to ordinary retained
          // evidence only after that validation point.
          entry.flags &= static_cast<u8>(~kFrontierRobFlagEarlyShadow);
          if (newly_accounted) {
            entry.flags |= kFrontierRobFlagUtilityAccounted;
          }
          retained_speculative = newly_accounted;
        }
      }
    }
    const u32 speculative_mask =
      __ballot_sync(full_warp, stale_speculative);
    const u32 core_mask = __ballot_sync(full_warp, stale_core);
    const u32 retained_mask =
      __ballot_sync(full_warp, retained_speculative);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      stale_bytes += __shfl_down_sync(full_warp, stale_bytes, offset);
    }
    if (lane == 0) {
      const u32 newly_stale = __popc(speculative_mask);
      speculative_stale += newly_stale;
      speculative_wasted_bytes += stale_bytes;
      core_prefetch_stale += __popc(core_mask);
      tail_feedback.stale += feedback_stale + newly_stale;
      tail_feedback.retained += __popc(retained_mask);
      tail_feedback.queue_rejects += feedback_queue_rejects;
    }
  } else if (threadIdx.x == 0) {
    tail_feedback.stale += feedback_stale;
    tail_feedback.queue_rejects += feedback_queue_rejects;
  }
  __syncthreads();

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    constexpr u32 full_warp = 0xffffffffu;
    const u32 position = threadIdx.x;
    const bool selected = position < selected_count;
    const u32 slot = selected ? commit_rob_slots[position] : UINT32_MAX;
    const bool hit = selected && slot != UINT32_MAX;
    const bool speculative_hit =
      hit && frontier_rob[slot].priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const bool controller_utility_hit =
      speculative_hit &&
      (frontier_rob[slot].flags &
         kFrontierRobFlagUtilityAccounted) == 0;
    if (hit) {
      if (controller_utility_hit) {
        frontier_rob[slot].flags |=
          kFrontierRobFlagUtilityAccounted;
      }
      frontier_rob[slot].state =
        static_cast<u8>(FrontierRequestState::committed);
      graph_record_slots[position] =
        kGraphScratchBit |
        static_cast<u32>(frontier_rob[slot].scratch_slot);
    }
    const u32 miss_mask = __ballot_sync(full_warp, selected && !hit);
    const u32 lower_lanes =
      position == 0 ? 0u : ((u32{1} << position) - 1u);
    if (selected && !hit) {
      const u32 fetch = __popc(miss_mask & lower_lanes);
      critical_fetch_to_commit[fetch] = position;
      critical_fetch_handles[fetch] = selected_handles[position];
    }
    const u32 hit_mask = __ballot_sync(full_warp, hit);
    const u32 speculative_hit_mask =
      __ballot_sync(full_warp, speculative_hit);
    const u32 controller_utility_hit_mask =
      __ballot_sync(full_warp, controller_utility_hit);
    if (position == 0) {
      const u32 hits = __popc(hit_mask);
      const u32 misses = __popc(miss_mask);
      critical_fetch_count = __popc(miss_mask);
      critical_misses += critical_fetch_count;
      critical_rob_hits += hits;
      const u32 promoted = __popc(speculative_hit_mask);
      speculative_promoted += promoted;
      core_prefetch_promoted += hits - promoted;
      // Promotion remains a physical/usefulness telemetry event, but a
      // request already credited when it survived an exact certificate must
      // not train the width controller a second time.
      tail_feedback.promoted +=
        __popc(controller_utility_hit_mask);
      if (issue_wave) {
        tail_feedback.core_hits += hits;
        tail_feedback.core_misses += misses;
      }
    }
  }
  __syncthreads();
}

// Publish the exact next commit prefix into positional core ROB slots. The
// current round has already consumed every COMMITTED payload, so those slots
// can be recycled without an associative ROB pass. One lane owns one slot;
// no atomics or global queue are involved in this metadata transition.
__device__ __forceinline__ void prepare_exact_core_frontier(
    const u64* issue_handles, const u16* issue_ranks, u32& issue_count,
    FrontierRobEntry* frontier_rob, u32 core_slot_count, u32& issue_epoch,
    u32& issue_epochs, u64& issue_width_sum,
    u64& issue_width_capacity_sum, u32 issue_width_capacity,
    u32& observed_max_issue_width) {
  constexpr u32 full_warp = 0xffffffffu;
  core_slot_count = min(
    core_slot_count,
    static_cast<u32>(kPersistentFrontierRobCapacity));
  if (threadIdx.x < 32) {
    const u32 lane = threadIdx.x;
    const u32 admitted = min(issue_count, core_slot_count);
    if (lane < core_slot_count) {
      FrontierRobEntry& entry = frontier_rob[lane];
      entry = {};
      if (lane < admitted) {
        entry.node_handle = issue_handles[lane];
        entry.issue_epoch = issue_epoch + 1u;
        entry.beam_rank = issue_ranks[lane];
        entry.scratch_slot = static_cast<u8>(lane);
        entry.state = static_cast<u8>(FrontierRequestState::issued);
        entry.priority =
          static_cast<u8>(DirectBatchPriority::critical);
      }
    }
    __syncwarp(full_warp);
    if (lane == 0) {
      ++issue_epoch;
      ++issue_epochs;
      issue_count = admitted;
      issue_width_sum += admitted;
      issue_width_capacity_sum += issue_width_capacity;
      observed_max_issue_width =
        max(observed_max_issue_width, admitted);
    }
  }
  __syncthreads();
}

// Build a bounded look-ahead suffix from the *pre-merge* authoritative Beam.
// The first commit_count unexpanded entries are being expanded by the current
// epoch, so the following entries are the only legal candidates for an early
// shadow read.  Their handles/ranks are merely ROB metadata: this helper never
// touches Beam distances, expanded bits, or visited state.  The exact
// Stable-Run merge later reconciles these entries by handle; a displaced
// prediction is therefore harmless and is charged as speculative waste.
//
// This path is deliberately one warp and one ROB-tail allocation.  It runs
// after the current critical records have been admitted, so critical requests
// retain queue priority while the shadow suffix is in flight during
// neighbor/PQ scoring and authoritative merge.
__device__ __noinline__ void prepare_early_shadow_frontier(
    const u64* beam_handles, const u8* beam_expanded, u32 beam_count,
    u32 commit_count, u32 requested_count, FrontierRobEntry* frontier_rob,
    u32 core_slot_count, u32& issue_count, u32& issue_epoch) {
  core_slot_count = min(
    core_slot_count,
    static_cast<u32>(kPersistentFrontierRobCapacity));
  requested_count = min(
    requested_count,
    static_cast<u32>(kPersistentFrontierRobCapacity) - core_slot_count);

  // Reclaim terminal metadata before calculating the free tail mask.  No
  // payload can still be owned by a terminal slot at this point: the caller
  // has drained the preceding tail batch before entering the commit epoch.
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    FrontierRobEntry& entry = frontier_rob[threadIdx.x];
    if (threadIdx.x >= core_slot_count &&
        (entry.state ==
           static_cast<u8>(FrontierRequestState::committed) ||
         entry.state ==
           static_cast<u8>(FrontierRequestState::stale))) {
      entry = {};
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    issue_count = 0;
    bool tail_empty = true;
    for (u32 slot = core_slot_count;
         slot < kPersistentFrontierRobCapacity; ++slot) {
      if (frontier_rob[slot].state !=
          static_cast<u8>(FrontierRequestState::init)) {
        tail_empty = false;
        break;
      }
    }
    if (tail_empty) {
      u32 skipped = 0;
      for (u32 rank = 0;
           rank < beam_count && issue_count < requested_count; ++rank) {
        if (beam_expanded[rank] != 0) continue;
        if (skipped < commit_count) {
          ++skipped;
          continue;
        }
        const u32 destination = core_slot_count + issue_count;
        FrontierRobEntry& entry = frontier_rob[destination];
        entry = {};
        entry.node_handle = beam_handles[rank];
        entry.issue_epoch = issue_epoch + 1u;
        entry.beam_rank =
          static_cast<u16>(min(rank, static_cast<u32>(UINT16_MAX)));
        entry.scratch_slot = static_cast<u8>(destination);
        entry.state = static_cast<u8>(FrontierRequestState::issued);
        entry.priority =
          static_cast<u8>(DirectBatchPriority::speculative);
        entry.flags = kFrontierRobFlagEarlyShadow;
        ++issue_count;
      }
      if (issue_count != 0) ++issue_epoch;
    }
  }
  __syncthreads();
}

// Admit a candidate-only shadow frontier into the ROB tail.  Unlike
// prepare_early_shadow_frontier(), the input is produced from a sorted,
// distance-ready Stable-Run leaf.  The entries are still predictions: only a
// later exact Stable-Run certificate may promote them.  Keeping admission in
// the contiguous ROB suffix preserves coalesced scratch addressing and lets
// the existing split critical/speculative descriptor path consume it without
// a global queue or another mapping structure.
__device__ __noinline__ void prepare_candidate_shadow_frontier(
    const u64* candidate_handles, const u16* candidate_ranks,
    u32 candidate_count, u32 requested_count,
    FrontierRobEntry* frontier_rob, u32 core_slot_count,
    u32& issue_count, u32& issue_epoch) {
  core_slot_count = min(
    core_slot_count,
    static_cast<u32>(kPersistentFrontierRobCapacity));
  requested_count = min(
    min(requested_count, candidate_count),
    static_cast<u32>(kPersistentFrontierRobCapacity) - core_slot_count);

  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    FrontierRobEntry& entry = frontier_rob[threadIdx.x];
    if (threadIdx.x >= core_slot_count &&
        (entry.state ==
           static_cast<u8>(FrontierRequestState::committed) ||
         entry.state ==
           static_cast<u8>(FrontierRequestState::stale))) {
      entry = {};
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    issue_count = 0;
    bool tail_empty = true;
    for (u32 slot = core_slot_count;
         slot < kPersistentFrontierRobCapacity; ++slot) {
      if (frontier_rob[slot].state !=
          static_cast<u8>(FrontierRequestState::init)) {
        tail_empty = false;
        break;
      }
    }
    if (tail_empty) {
      for (u32 position = 0; position < requested_count; ++position) {
        const u64 handle = candidate_handles[position];
        if (handle == kInvalidDeviceHandle) continue;
        const u32 destination = core_slot_count + issue_count;
        FrontierRobEntry& entry = frontier_rob[destination];
        entry = {};
        entry.node_handle = handle;
        entry.issue_epoch = issue_epoch + 1u;
        entry.beam_rank = candidate_ranks[position];
        entry.scratch_slot = static_cast<u8>(destination);
        entry.state = static_cast<u8>(FrontierRequestState::issued);
        entry.priority =
          static_cast<u8>(DirectBatchPriority::speculative);
        entry.flags = kFrontierRobFlagEarlyShadow;
        ++issue_count;
      }
      if (issue_count != 0) ++issue_epoch;
    }
  }
  __syncthreads();
}

// Close the utility epoch of the preceding pre-merge shadow wave before
// admitting another one.  A speculative payload that was selected by the
// exact certificate has already transitioned to COMMITTED and is released
// only after its neighbors have been copied.  Consequently every remaining
// VALIDATED tail entry at this point is a real prediction miss and can be
// charged/reclaimed without inspecting Beam or touching authoritative state.
//
// The controller update is deliberately query-CTA local.  It consumes one
// complete issue -> exact-certificate -> commit observation and then resets
// the feedback before the next wave is formed; no global queue, atomics, or
// CPU scheduler participate.
__device__ __forceinline__ void close_early_shadow_epoch(
    FrontierRobEntry* frontier_rob, u32 core_slot_count,
    adaptive_frontier::ControllerState& controller,
    TailFrontierFeedback& feedback, u32& speculative_stale,
    u64& speculative_wasted_bytes, u32& core_prefetch_stale) {
  constexpr u32 full_warp = 0xffffffffu;
  core_slot_count = min(
    core_slot_count,
    static_cast<u32>(kPersistentFrontierRobCapacity));
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 slot = threadIdx.x;
    FrontierRobEntry& entry = frontier_rob[slot];
    const bool tail_slot = slot >= core_slot_count;
    const bool unconsumed =
      tail_slot &&
      entry.state ==
        static_cast<u8>(FrontierRequestState::validated);
    const bool discarded_speculative =
      unconsumed &&
      entry.priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const bool discarded_core = unconsumed && !discarded_speculative;
    u64 discarded_bytes =
      discarded_speculative ? entry.transfer_bytes : 0;
    if (tail_slot &&
        (unconsumed ||
         entry.state ==
           static_cast<u8>(FrontierRequestState::committed) ||
         entry.state ==
           static_cast<u8>(FrontierRequestState::stale))) {
      entry = {};
    }
    const u32 speculative_mask =
      __ballot_sync(full_warp, discarded_speculative);
    const u32 core_mask = __ballot_sync(full_warp, discarded_core);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      discarded_bytes +=
        __shfl_down_sync(full_warp, discarded_bytes, offset);
    }
    if (slot == 0) {
      const u32 newly_stale = __popc(speculative_mask);
      speculative_stale += newly_stale;
      speculative_wasted_bytes += discarded_bytes;
      core_prefetch_stale += __popc(core_mask);
      feedback.stale += newly_stale;
      adaptive_frontier::update_issue_width(
        controller,
        adaptive_frontier::Feedback{
          .promoted = feedback.promoted,
          .retained = feedback.retained,
          .stale = feedback.stale,
          .queue_rejects = feedback.queue_rejects,
          .critical_misses = feedback.core_misses,
          .tail_admitted = feedback.tail_admitted,
          .commit_waves_observed = feedback.commit_waves_observed,
          .commit_waves_covered = feedback.commit_waves_covered,
        });
      feedback = {};
    }
  }
  __syncthreads();
}

// Reconcile the exact post-merge preview with the next authoritative commit.
// Core entries are positional: preview rank i owns ROB/scratch slot i. A
// rejected or invalid preview therefore releases exactly its own scratch slot
// for the critical fallback. Keeping this transition out of process_query
// shortens the persistent kernel's live range and avoids an associative ROB
// scan on the steady-state path.
__device__ __forceinline__ void reconcile_exact_core_frontier(
    const u64* selected_handles, u32 selected_count,
    FrontierRobEntry* frontier_rob, u32 core_slot_count,
    u32* commit_rob_slots, u32* graph_record_slots,
    u64* critical_fetch_handles, u32* critical_fetch_to_commit,
    u32* critical_fetch_destination_slots, u32& critical_fetch_count,
    u32& critical_rob_hits, u32& critical_misses,
    u32& core_prefetch_promoted, u32& core_prefetch_stale) {
  constexpr u32 full_warp = 0xffffffffu;
  if (threadIdx.x < 32) {
    const u32 position = threadIdx.x;
    const bool selected = position < selected_count;
    FrontierRobEntry* entry =
      position < core_slot_count ? frontier_rob + position : nullptr;
    const bool hit =
      selected && entry != nullptr &&
      (entry->state ==
         static_cast<u8>(FrontierRequestState::validated) ||
       entry->state ==
         static_cast<u8>(FrontierRequestState::committed)) &&
      entry->node_handle == selected_handles[position];
    const bool stale =
      entry != nullptr &&
      entry->state != static_cast<u8>(FrontierRequestState::init) &&
      !hit;
    commit_rob_slots[position] = hit ? position : UINT32_MAX;
    if (selected) {
      graph_record_slots[position] = hit
        ? kGraphScratchBit | static_cast<u32>(entry->scratch_slot)
        : UINT32_MAX;
    }
    if (hit) {
      entry->state = static_cast<u8>(FrontierRequestState::committed);
    } else if (stale) {
      *entry = {};
    }

    const u32 miss_mask =
      __ballot_sync(full_warp, selected && !hit);
    const u32 lower_lanes =
      position == 0 ? 0u : ((u32{1} << position) - 1u);
    if (selected && !hit) {
      const u32 fetch = __popc(miss_mask & lower_lanes);
      critical_fetch_handles[fetch] = selected_handles[position];
      critical_fetch_to_commit[fetch] = position;
      critical_fetch_destination_slots[fetch] = position;
    }
    const u32 hit_mask = __ballot_sync(full_warp, hit);
    const u32 stale_mask = __ballot_sync(full_warp, stale);
    if (position == 0) {
      const u32 hits = __popc(hit_mask);
      critical_fetch_count = __popc(miss_mask);
      critical_rob_hits += hits;
      critical_misses += critical_fetch_count;
      core_prefetch_promoted += hits;
      core_prefetch_stale += __popc(stale_mask);
    }
  }
  __syncthreads();
}

// Consume an arbitrary-slot mapping produced by
// prepare_issue_frontier_entries().  A mapping is evidence, not authority:
// only a completed, validated payload whose handle still equals the frozen
// Commit Frontier may transition to COMMITTED.  Every other case is compacted
// into the mandatory critical retry path without waiting on speculative I/O.
__device__ __noinline__ void reconcile_certified_commit_frontier(
    const CertifiedCommitReconcileContext& context,
    u32 selected_count, bool issue_wave) {
  constexpr u32 full_warp = 0xffffffffu;
  if (threadIdx.x < 32) {
    const u32 position = threadIdx.x;
    const bool selected = position < selected_count;
    const u32 slot = selected
      ? context.certified_rob_slots[position] : UINT32_MAX;
    FrontierRobEntry* entry =
      slot < kPersistentFrontierRobCapacity
        ? context.frontier_rob + slot : nullptr;
    const bool hit =
      selected && entry != nullptr &&
      (entry->state == static_cast<u8>(
         FrontierRequestState::validated) ||
       entry->state == static_cast<u8>(
         FrontierRequestState::committed)) &&
      entry->node_handle == context.selected_handles[position];
    const bool speculative_hit =
      hit && entry->priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const bool controller_utility_hit =
      speculative_hit &&
      (entry->flags & kFrontierRobFlagUtilityAccounted) == 0;
    const bool stale_core =
      selected && entry != nullptr && !hit &&
      entry->state != static_cast<u8>(FrontierRequestState::init) &&
      entry->state != static_cast<u8>(FrontierRequestState::inflight) &&
      entry->priority ==
        static_cast<u8>(DirectBatchPriority::critical);

    context.commit_rob_slots[position] = hit ? slot : UINT32_MAX;
    if (selected) {
      context.graph_record_slots[position] = hit
        ? kGraphScratchBit | static_cast<u32>(entry->scratch_slot)
        : UINT32_MAX;
    }
    if (hit) {
      if (controller_utility_hit) {
        entry->flags |= kFrontierRobFlagUtilityAccounted;
      }
      entry->state =
        static_cast<u8>(FrontierRequestState::committed);
    }

    const u32 miss_mask =
      __ballot_sync(full_warp, selected && !hit);
    const u32 lower_lanes =
      position == 0 ? 0u : ((u32{1} << position) - 1u);
    if (selected && !hit) {
      const u32 fetch = __popc(miss_mask & lower_lanes);
      context.critical_fetch_handles[fetch] =
        context.selected_handles[position];
      context.critical_fetch_to_commit[fetch] = position;
    }
    const u32 hit_mask = __ballot_sync(full_warp, hit);
    const u32 speculative_hit_mask =
      __ballot_sync(full_warp, speculative_hit);
    const u32 controller_utility_hit_mask =
      __ballot_sync(full_warp, controller_utility_hit);
    const u32 stale_core_mask =
      __ballot_sync(full_warp, stale_core);
    if (position == 0) {
      const u32 hits = __popc(hit_mask);
      const u32 misses = __popc(miss_mask);
      const u32 promoted = __popc(speculative_hit_mask);
      *context.critical_fetch_count = misses;
      *context.critical_rob_hits += hits;
      *context.critical_misses += misses;
      *context.speculative_promoted += promoted;
      *context.core_prefetch_promoted += hits - promoted;
      *context.core_prefetch_stale += __popc(stale_core_mask);
      context.feedback->promoted +=
        __popc(controller_utility_hit_mask);
      if (issue_wave) {
        context.feedback->core_hits += hits;
        context.feedback->core_misses += misses;
      }
    }
  }
  __syncthreads();
}

// Reserve scratch destinations for authoritative misses after every
// asynchronous wave has been drained. Critical reads must never wait behind
// validated shadow records merely because those records still occupy ROB
// slots. One warp allocates in strict priority order:
//
//   empty/stale -> speculative validated -> non-selected critical validated
//
// COMMITTED records remain protected until their neighbors have been copied,
// and any unexpectedly live request remains protected against a late DMA.
// Thus critical-first scheduling is enforced without a global queue, an
// atomic allocator, or an additional query-level synchronization point.
__device__ __noinline__ void reserve_critical_fetch_destinations(
    FrontierRobEntry* frontier_rob, u32 critical_fetch_count,
    u32* critical_fetch_destination_slots, u32& graph_failed,
    u32& speculative_stale, u64& speculative_wasted_bytes,
    u32& core_prefetch_stale, TailFrontierFeedback& feedback) {
  constexpr u32 full_warp = 0xffffffffu;
  if (threadIdx.x < kPersistentFrontierRobCapacity) {
    const u32 lane = threadIdx.x;
    const FrontierRobEntry& resident = frontier_rob[lane];
    const u8 state = resident.state;
    const bool immediately_free =
      state == static_cast<u8>(FrontierRequestState::init) ||
      state == static_cast<u8>(FrontierRequestState::stale);
    const bool validated =
      state == static_cast<u8>(FrontierRequestState::validated);
    const bool speculative_victim =
      validated &&
      resident.priority ==
        static_cast<u8>(DirectBatchPriority::speculative);
    const bool critical_victim = validated && !speculative_victim;

    const u32 free_mask =
      __ballot_sync(full_warp, immediately_free);
    const u32 speculative_mask =
      __ballot_sync(full_warp, speculative_victim);
    const u32 critical_mask =
      __ballot_sync(full_warp, critical_victim);
    const u32 free_count = __popc(free_mask);
    const u32 speculative_count = __popc(speculative_mask);

    u32 destination = UINT32_MAX;
    if (lane < critical_fetch_count) {
      u32 candidate_mask = 0;
      u32 ordinal = lane;
      if (ordinal < free_count) {
        candidate_mask = free_mask;
      } else {
        ordinal -= free_count;
        if (ordinal < speculative_count) {
          candidate_mask = speculative_mask;
        } else {
          ordinal -= speculative_count;
          candidate_mask = critical_mask;
        }
      }
      for (u32 prior = 0;
           prior < ordinal && candidate_mask != 0; ++prior) {
        candidate_mask &= candidate_mask - 1u;
      }
      if (candidate_mask != 0) {
        destination =
          static_cast<u32>(__ffs(static_cast<int>(candidate_mask)) - 1);
      }
      critical_fetch_destination_slots[lane] = destination;
    }

    const bool allocation_failed =
      lane < critical_fetch_count && destination == UINT32_MAX;
    const u32 allocation_failed_mask =
      __ballot_sync(full_warp, allocation_failed);
    if (lane == 0 && allocation_failed_mask != 0) {
      graph_failed = 5u;
    }

    bool evicted_speculative = false;
    bool evicted_critical = false;
    u64 evicted_bytes = 0;
    if (destination != UINT32_MAX) {
      FrontierRobEntry& victim = frontier_rob[destination];
      evicted_speculative =
        victim.state ==
          static_cast<u8>(FrontierRequestState::validated) &&
        victim.priority ==
          static_cast<u8>(DirectBatchPriority::speculative);
      evicted_critical =
        victim.state ==
          static_cast<u8>(FrontierRequestState::validated) &&
        !evicted_speculative;
      evicted_bytes =
        evicted_speculative ? victim.transfer_bytes : 0;
      victim = {};
    }
    const u32 evicted_speculative_mask =
      __ballot_sync(full_warp, evicted_speculative);
    const u32 evicted_critical_mask =
      __ballot_sync(full_warp, evicted_critical);
    for (u32 offset = 16; offset != 0; offset >>= 1) {
      evicted_bytes +=
        __shfl_down_sync(full_warp, evicted_bytes, offset);
    }
    if (lane == 0) {
      const u32 evicted_speculative_count =
        __popc(evicted_speculative_mask);
      speculative_stale += evicted_speculative_count;
      speculative_wasted_bytes += evicted_bytes;
      core_prefetch_stale += __popc(evicted_critical_mask);
      feedback.stale += evicted_speculative_count;
    }
  }
  __syncthreads();
}

template <bool EnableAsfe>
__device__ __forceinline__ void process_query(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    adaptive_frontier::ControllerState& frontier_controller) {
  // query_slot is consumed throughout the entire traversal.  Keeping the
  // descriptor field in a per-thread scalar extends one register (and, in
  // the 128-thread ASFE specialization, one spill value) across every RDMA,
  // scoring, and merge phase.  A single CTA-resident copy is broadcast by
  // shared memory and piggybacks on the completion-initialization barrier.
  __shared__ u32 query_slot;
  __shared__ u64 query_started_cycles;
  // Completion is produced by the CTA control lane only.  Keeping one copy
  // in shared memory prevents the large telemetry object from extending every
  // thread's local stack across the full ASFE query state machine.  That
  // local lifetime previously forced the 128-thread ASFE specialization to
  // spill even though all frontier metadata itself was CTA-resident.
  __shared__ CompletionDescriptor shared_completion;
  CompletionDescriptor& completion = shared_completion;
  if (threadIdx.x == 0) {
    query_slot = descriptor.query_slot;
    query_started_cycles = clock64();
    completion = CompletionDescriptor{
      .request_id = descriptor.request_id,
      .query_slot = query_slot,
    };
  }
  __syncthreads();
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
      set_query_trace_completion(params, query_slot, completion);
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
  __shared__ u64 dynamic_code_cycles;
  __shared__ u64 beam_selection_cycles;
  __shared__ u64 rdma_issue_cycles;
  __shared__ u64 frontier_preview_cycles;
  __shared__ u64 frontier_prepare_cycles;
  __shared__ u64 frontier_enqueue_cycles;
  __shared__ u64 frontier_subphase_started_cycles;
  __shared__ u64 rdma_wait_cycles;
  __shared__ u64 graph_validation_cycles;
  __shared__ u64 neighbor_decode_cycles;
  __shared__ u64 pq_score_cycles;
  __shared__ u64 visited_cycles;
  __shared__ u64 beam_merge_cycles;
  __shared__ BeamMergeCycleBreakdown beam_merge_breakdown;
  __shared__ BeamMergeCycleBreakdown beam_merge_round_breakdown;
  __shared__ u32 rdma_trace_enabled;
  __shared__ GraphFetchCycleBreakdown graph_fetch_breakdown;
  __shared__ u32 dynamic_code_candidates;
  __shared__ u32 dynamic_code_reads;
  __shared__ u32 dynamic_code_incarnation_rejects;
  __shared__ u32 dynamic_code_cache_hits;
  __shared__ u32 dynamic_code_batch_deduplicated;
  __shared__ u32 dynamic_code_cache_publish_successes;
  __shared__ u32 dynamic_code_cache_first_occupancies;
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
    frontier_preview_cycles = 0;
    frontier_prepare_cycles = 0;
    frontier_enqueue_cycles = 0;
    frontier_subphase_started_cycles = 0;
    rdma_wait_cycles = 0;
    graph_validation_cycles = 0;
    neighbor_decode_cycles = 0;
    pq_score_cycles = 0;
    visited_cycles = 0;
    beam_merge_cycles = 0;
    beam_merge_breakdown = {};
    beam_merge_round_breakdown = {};
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
    dynamic_code_candidates = 0;
    dynamic_code_reads = 0;
    dynamic_code_incarnation_rejects = 0;
    dynamic_code_cache_hits = 0;
    dynamic_code_batch_deduplicated = 0;
    dynamic_code_cache_publish_successes = 0;
    dynamic_code_cache_first_occupancies = 0;
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
  f32* query_lut = params.query_luts +
    static_cast<size_t>(query_slot) * params.pq_subquantizers * 256;
  const u32 table_entries = params.pq_subquantizers * 256;
  for (u32 row = threadIdx.x; row < params.dim; row += blockDim.x) {
    if (params.opq_matrix == nullptr) {
      transformed[row] = query[row];
      continue;
    }
    f32 value = 0.0f;
    const f32* matrix_row =
      params.opq_matrix + static_cast<size_t>(row) * params.dim;
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
  for (u32 index = threadIdx.x; index < table_entries;
       index += blockDim.x) {
    const u32 subquantizer = index / 256;
    const f32* query_subvector = transformed +
      static_cast<size_t>(subquantizer) * params.pq_subvector_dim;
    const f32* centroid_subvector = params.pq_centroids +
      static_cast<size_t>(index) * params.pq_subvector_dim;
    f32 distance = 0.0f;
    for (u32 dimension = 0; dimension < params.pq_subvector_dim;
         ++dimension) {
      const f32 difference =
        query_subvector[dimension] - centroid_subvector[dimension];
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
  __shared__ StableMergePreparedState stable_merge_state;
  u64* merge_handles = candidate_workspace.arrays.handles;
  // Stable-Run no longer needs its origin/candidate handle tail after graph
  // traversal. Reuse that dead lifetime for final exact-rerank IDs instead of
  // reserving a disjoint 512-entry shared array throughout the query.
  u32* merge_ids = reinterpret_cast<u32*>(
    candidate_workspace.arrays.handles + kPersistentMaxBeam);
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
  // The compact 128-thread Stable-Run path retains four independently sorted
  // candidate prefixes until its exact preview has issued RDMA. Flags are
  // binary, so overlay their byte array with the wider IDs needed only by the
  // later exact-rerank phase. This keeps the four-run overlap below the 48-KiB
  // static shared-memory limit without adding global traffic.
  union RerankMetadataScratch {
    u8 flags[kPersistentStableRunScratch];
    u32 ids[kPersistentMaxExact];
  };
  __shared__ u64 rerank_handles[kPersistentStableRunScratch];
  __shared__ RerankMetadataScratch rerank_metadata;
  __shared__ f32 rerank_distances[kPersistentStableRunScratch];
  u8* rerank_flags = rerank_metadata.flags;
  u32* rerank_ids = rerank_metadata.ids;
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
          route_snapshot_entry_counts[shard] =
            snapshot.live_entry_count;
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
            for (u32 local = 0;
                 local < route_snapshot_entry_counts[shard];
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
        return;
      }
      if (route_snapshot_timed_out != 0) {
        if (threadIdx.x == 0) {
          completion.status = -ETIMEDOUT;
          completion.diagnostic = make_query_diagnostic(
            QueryFailureReason::route_snapshot_timeout,
            route_snapshot_retries);
          completion.gpu_cycles = clock64() - query_started_cycles;
          set_beam_merge_completion(completion, beam_merge_breakdown);
          set_query_trace_completion(params, query_slot, completion);
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
          &dynamic_code_cache_first_occupancies,
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
          dynamic_code_cache_publish_successes,
          dynamic_code_cache_first_occupancies,
          dynamic_code_cache_publish_races,
          dynamic_code_cache_lookup_probe_exhaustions,
          dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
        set_beam_merge_completion(completion, beam_merge_breakdown);
        set_query_trace_completion(params, query_slot, completion);
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }

  __shared__ u64 selected_handles[kPersistentMaxPrefetch];
  __shared__ u32 selected_beam_ranks[kPersistentMaxPrefetch];
  __shared__ u32 selected_count;
  // The exact Stable-Run preview at the end of a round already identifies
  // the next authoritative commit prefix.  Carry that certificate across
  // the round boundary so the next iteration does not rescan the materialized
  // Beam on lane zero.  The certificate is query-local and is consumed only
  // after the preceding merge has passed its CTA barrier.
  __shared__ u32 next_commit_ready;
  __shared__ u32 next_commit_count;
  __shared__ u32 selection_from_certificate;
  __shared__ u64 critical_fetch_handles[kPersistentMaxPrefetch];
  __shared__ u32 critical_fetch_slots[kPersistentMaxPrefetch];
  __shared__ u32 critical_fetch_to_commit[kPersistentMaxPrefetch];
  __shared__ u32 critical_fetch_destination_slots[kPersistentMaxPrefetch];
  __shared__ u8 selected_underhint_force_full[kPersistentMaxPrefetch];
  __shared__ u8 critical_fetch_force_full[kPersistentMaxPrefetch];
  // Tail completion has finished before underhint mapping starts. Reuse its
  // phase-local scalar for the uniform fast-path gate instead of growing the
  // persistent CTA's already occupancy-sensitive shared-memory footprint.
  union TailUnderhintPhaseScratch {
    u32 tail_stale_before;
    u32 selected_underhint_any;
  };
  __shared__ TailUnderhintPhaseScratch tail_underhint_scratch;
  __shared__ u32 commit_rob_slots[kPersistentMaxPrefetch];
  __shared__ u32 critical_fetch_count;
  __shared__ u64 shadow_frontier_handles[kPersistentFrontierRobCapacity];
  __shared__ u16 shadow_frontier_ranks[kPersistentFrontierRobCapacity];
  __shared__ u32 shadow_frontier_count;
  // 0: no reusable leaf state; 2: complete immutable leaves may be sealed as
  // ordered CQ/PQ progresses; 1: all four compact leaf slots (including a
  // final partial leaf and absent-leaf padding) are sorted; 3: only the first
  // compact leaf is ready and its remaining leaves are deferred past issue;
  // 5: PFEC produced all four reusable leaves and an exact frontier
  // certificate in one warp-parallel pass.
  __shared__ u32 stable_runs_prepared_before_issue;
  __shared__ u32 frontier_reusable_certificates;
  __shared__ u32 frontier_streamed_candidate_runs;
  __shared__ u32 ordered_score_batches;
  __shared__ u32 ordered_score_candidates;
  __shared__ u32 ooo_bypassed_parents;
  __shared__ u32 frontier_reusable_prefix_ranks;
  __shared__ u32 frontier_reusable_full_prefix_certificates;
  __shared__ u32 frontier_reusable_issued_certificates;
  __shared__ u32 frontier_certificate_rejects;
  __shared__ u32 issue_rob_slots[kPersistentFrontierRobCapacity];
  // Number of physical ROB slots covered by the current issue descriptor.
  // This is intentionally independent of the logical certificate width:
  // retained/promoted records can make the physical mapping sparse.
  __shared__ u32 physical_issue_span;
  __shared__ u32 neighbor_counts[kPersistentMaxPrefetch];
  __shared__ u32 neighbor_offsets[kPersistentMaxPrefetch + 1];
  __shared__ u32 flattened_neighbors;
  __shared__ u32 remote_reads_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 graph_record_slots[kPersistentMaxPrefetch];
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
  __shared__ FrontierRobEntry frontier_rob[kPersistentFrontierRobCapacity];
  __shared__ FrontierGraphBatchState core_batch;
  __shared__ FrontierGraphBatchState tail_batch;
  __shared__ TerminalExactCacheState terminal_exact_cache;
  __shared__ u32 terminal_exact_horizon_ready;
  __shared__ TailFrontierFeedback tail_feedback;
  __shared__ u32 frontier_issue_epoch;
  __shared__ u32 logical_expansions;
  __shared__ u32 critical_graph_reads;
  __shared__ u64 critical_graph_bytes;
  __shared__ u32 speculative_graph_reads;
  __shared__ u64 speculative_graph_bytes;
  __shared__ u32 speculative_arrived;
  __shared__ u32 speculative_promoted;
  __shared__ u32 speculative_stale;
  __shared__ u64 speculative_wasted_bytes;
  __shared__ u32 speculative_queue_rejects;
  __shared__ u32 issue_epochs;
  __shared__ u32 commit_epochs;
  __shared__ u64 issue_width_sum;
  __shared__ u64 issue_width_capacity_sum;
  __shared__ u64 commit_width_sum;
  __shared__ u32 max_issue_width;
  __shared__ u32 max_commit_width;
  __shared__ u32 critical_rob_hits;
  __shared__ u32 critical_misses;
  __shared__ u64 speculative_wait_cycles;
  __shared__ u64 rdma_completion_latency_ns;
  __shared__ u64 speculative_completion_latency_ns;
  __shared__ u64 rdma_completion_groups;
  __shared__ u64 speculative_completion_groups;
  __shared__ u64 core_prefetch_bytes;
  __shared__ u32 core_prefetch_reads;
  __shared__ u32 core_prefetch_arrived;
  __shared__ u32 core_prefetch_promoted;
  __shared__ u32 core_prefetch_stale;
  __shared__ u32 core_prefetch_queue_rejects;
  __shared__ u32 core_prefetch_waves;
  __shared__ u32 core_ready_waves;
  __shared__ DynamicGraphTelemetry dynamic_graph_telemetry;
  __shared__ CoreFrontierTelemetry core_telemetry;
  __shared__ TailFrontierTelemetry tail_telemetry;
  __shared__ TailAdmissionCorrection tail_admission_correction;
  __shared__ u64 shadow_issue_started_cycles;
  // Completion harvesting is independent of Beam rank. Candidate decode and
  // scoring deliberately remain one canonical full-width batch after every
  // mandatory parent has arrived; fragmenting that GPU work per CQ group is
  // substantially more expensive than the network wait it can hide.
  __shared__ u32 ooo_completed_parent_mask;
  __shared__ u32 early_queue_rejects_before;
  __shared__ u32 core_batch_positional;
  __shared__ u32 certified_mapping_ready;
  __shared__ u32 reconciled_positional_core;
  __shared__ u32 core_only_issue_epoch;
  __shared__ CertifiedCommitReconcileContext
    certified_commit_context;
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
    core_batch.active = 0;
    core_batch.fatal = 0;
    core_batch.rejected = 0;
    core_batch.finish_had_pending = 0;
    tail_batch.active = 0;
    tail_batch.fatal = 0;
    tail_batch.rejected = 0;
    tail_batch.finish_had_pending = 0;
    terminal_exact_cache = {};
    terminal_exact_horizon_ready = 0;
    next_commit_ready = 0;
    next_commit_count = 0;
    selection_from_certificate = 0;
    tail_underhint_scratch.selected_underhint_any = 0;
    stable_runs_prepared_before_issue = 0;
    ooo_completed_parent_mask = 0;
    frontier_reusable_certificates = 0;
    ooo_bypassed_parents = 0;
    if constexpr (EnableAsfe) {
      frontier_streamed_candidate_runs = 0;
      ordered_score_batches = 0;
      ordered_score_candidates = 0;
      frontier_reusable_prefix_ranks = 0;
      frontier_reusable_full_prefix_certificates = 0;
      frontier_reusable_issued_certificates = 0;
      frontier_certificate_rejects = 0;
      // The controller is owned by the persistent CTA and deliberately
      // survives this function. Advance its collapsed re-probe cadence once
      // per valid query instead of restarting at commit+1 per submission.
      adaptive_frontier::begin_query(frontier_controller);
    }
    tail_feedback = {};
    frontier_issue_epoch = 0;
    logical_expansions = 0;
    critical_graph_reads = 0;
    critical_graph_bytes = 0;
    speculative_graph_reads = 0;
    speculative_graph_bytes = 0;
    speculative_arrived = 0;
    speculative_promoted = 0;
    speculative_stale = 0;
    speculative_wasted_bytes = 0;
    speculative_queue_rejects = 0;
    issue_epochs = 0;
    commit_epochs = 0;
    issue_width_sum = 0;
    issue_width_capacity_sum = 0;
    commit_width_sum = 0;
    max_issue_width = 0;
    max_commit_width = 0;
    critical_rob_hits = 0;
    critical_misses = 0;
    speculative_wait_cycles = 0;
    rdma_completion_latency_ns = 0;
    speculative_completion_latency_ns = 0;
    rdma_completion_groups = 0;
    speculative_completion_groups = 0;
    core_prefetch_bytes = 0;
    core_prefetch_reads = 0;
    core_prefetch_arrived = 0;
    core_prefetch_promoted = 0;
    core_prefetch_stale = 0;
    core_prefetch_queue_rejects = 0;
    core_prefetch_waves = 0;
    core_ready_waves = 0;
    dynamic_graph_telemetry = {};
    core_batch_positional = 0;
    certified_mapping_ready = 0;
    reconciled_positional_core = 0;
    core_only_issue_epoch = 0;
    physical_issue_span = 0;
    core_telemetry = CoreFrontierTelemetry{
      .wait_cycles = &speculative_wait_cycles,
      .completion_latency_ns = &rdma_completion_latency_ns,
      .completion_groups = &rdma_completion_groups,
      .arrived = &core_prefetch_arrived,
      .stale = &core_prefetch_stale,
      .ready_waves = &core_ready_waves,
      .dynamic_graph = &dynamic_graph_telemetry,
    };
    tail_telemetry = TailFrontierTelemetry{
      .arrived = &speculative_arrived,
      .stale = &speculative_stale,
      .wait_cycles = &speculative_wait_cycles,
      .completion_latency_ns = &speculative_completion_latency_ns,
      .completion_groups = &speculative_completion_groups,
      .wasted_bytes = &speculative_wasted_bytes,
      .admission_correction = &tail_admission_correction,
      .dynamic_graph = &dynamic_graph_telemetry,
    };
    tail_admission_correction = {};
    certified_commit_context = CertifiedCommitReconcileContext{
      .selected_handles = selected_handles,
      .certified_rob_slots = issue_rob_slots,
      .frontier_rob = frontier_rob,
      .commit_rob_slots = commit_rob_slots,
      .graph_record_slots = graph_record_slots,
      .critical_fetch_handles = critical_fetch_handles,
      .critical_fetch_to_commit = critical_fetch_to_commit,
      .critical_fetch_count = &critical_fetch_count,
      .critical_rob_hits = &critical_rob_hits,
      .critical_misses = &critical_misses,
      .speculative_promoted = &speculative_promoted,
      .core_prefetch_promoted = &core_prefetch_promoted,
      .core_prefetch_stale = &core_prefetch_stale,
      .feedback = &tail_feedback,
    };
  }
  if constexpr (EnableAsfe) {
    for (u32 shard = threadIdx.x; shard < kPersistentMaxShards;
         shard += blockDim.x) {
      core_batch.issue_timestamp_ns[shard] = 0;
      tail_batch.issue_timestamp_ns[shard] = 0;
    }
    for (u32 slot = threadIdx.x;
         slot < kPersistentFrontierRobCapacity;
         slot += blockDim.x) {
      frontier_rob[slot] = {};
    }
  }
  // Fixed-record policy leaves graph_request_bytes null, so no asynchronous
  // short read can produce extent-underhint evidence. Avoid even the one-time
  // flag stores for that baseline; the fixed path also skips both per-round
  // helper barriers below and passes no force-full vector to the fetcher.
  if (params.graph_request_bytes != nullptr) {
    for (u32 slot = threadIdx.x; slot < kPersistentMaxPrefetch;
         slot += blockDim.x) {
      selected_underhint_force_full[slot] = 0;
      critical_fetch_force_full[slot] = 0;
    }
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
    __shared__ u64 speculative_wait_before;
    if constexpr (EnableAsfe) {
      // Freeze the authoritative commit prefix before consuming any payload.
      // Completed shard groups may retire ROB entries in any order, but
      // candidate decode, visited, PQ, Beam, and expanded bits remain
      // unchanged until the complete dependency set has been resolved.
      if (threadIdx.x == 0) {
        phase_started_cycles = clock64();
        graph_failed = 0;
        // A candidate is valid for exactly one preview/materialization pair.
        // Clear it even on rounds that cannot construct an exact preview.
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        selection_from_certificate = 0;
        const bool certificate_candidate =
          next_commit_ready != 0 &&
          next_commit_count != 0 &&
          next_commit_count <=
            min(params.commit_width, params.max_expansions - expansions);
        // The certificate and Beam publication are the two outputs of the
        // same exact Stable-Run inputs inside this CTA. finish() completes
        // before the loop-back barrier, and no other actor can mutate this
        // query-local Beam. Re-scanning up to K ranks here duplicated the
        // merge proof serially on lane zero. Bitwise equivalence is enforced
        // by the focused DEEC/PBEC/PFEC and publish tests instead.
        const bool certificate_valid = certificate_candidate;
        if (certificate_valid) {
          // The preview is an exact stable merge certificate.  Its first
          // commit-width entries are therefore the same handles/ranks that a
          // fresh scan of the just-materialized authoritative Beam would
          // select.  Keep the rank mapping and reset ROB destinations here;
          // plan_commit_frontier() will reconcile any retained tail below.
          selected_count = next_commit_count;
          shadow_frontier_count = 0;
          for (u32 position = 0; position < selected_count; ++position) {
            commit_rob_slots[position] = UINT32_MAX;
          }
          ++commit_epochs;
          commit_width_sum += selected_count;
          max_commit_width = max(max_commit_width, selected_count);
          selection_from_certificate = 1;
          next_commit_ready = 0;
          next_commit_count = 0;
        } else {
          if (certificate_candidate) {
            ++frontier_certificate_rejects;
          }
          next_commit_ready = 0;
          next_commit_count = 0;
        }
      }
      __syncthreads();
      if (selection_from_certificate == 0) {
        select_commit_frontier(
          beam_handles, beam_expanded, beam_count,
          min(params.commit_width, params.max_expansions - expansions),
          selected_handles, selected_beam_ranks, commit_rob_slots,
          selected_count, shadow_frontier_count, commit_epochs,
          commit_width_sum, max_commit_width);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        beam_selection_cycles += clock64() - phase_started_cycles;
        core_batch.rejected = 0;
        speculative_wait_before = speculative_wait_cycles;
        ooo_completed_parent_mask = 0;
        stable_runs_prepared_before_issue = 0;
      }
      __syncthreads();
      const u32 single_chunk_parent_capacity =
        persistent_score_chunk_capacity(
          params.graph_entry_capacity, traversal_capacity);
      const bool canonical_core_mapping_ready =
        core_batch_positional != 0 ||
        (selection_from_certificate != 0 &&
         certified_mapping_ready != 0);
      const bool ooo_pipeline_enabled =
        core_batch.active != 0 && canonical_core_mapping_ready &&
        selected_count != 0 &&
        selected_count <= single_chunk_parent_capacity &&
        blockDim.x == kApproximateSortThreadsCompact;
      const bool frontier_stable_run_enabled =
        ooo_pipeline_enabled &&
        params.issue_width > params.commit_width &&
        params.beam_merge_policy ==
          static_cast<u32>(BeamMergePolicy::stable_run) &&
        expansions + selected_count < params.max_expansions;
      if (frontier_stable_run_enabled) {
        if (threadIdx.x == 0) {
          stable_merge_state = {};
          stable_merge_state.original_count = beam_count;
          stable_merge_state.compact = 1;
          stable_merge_state.origin_copied = 0;
          stable_merge_state.phase_started = clock64();
          beam_merge_round_breakdown = {};
          stable_runs_prepared_before_issue = 2;
        }
        __syncthreads();
      }
      if (ooo_pipeline_enabled) {
        constexpr u32 completion_warp_width = 32;
        const u32 completion_lane =
          threadIdx.x % completion_warp_width;

        while (core_batch.active != 0 && graph_failed == 0) {
          const i32 completion_progress =
            finish_next_core_frontier_group(
              params, descriptor, frontier_rob, core_batch,
              min(params.commit_width,
                  static_cast<u32>(kPersistentFrontierRobCapacity)),
              core_telemetry, true);
          if (completion_progress < 0) {
            if (threadIdx.x == 0) {
              graph_failed = frontier_graph_failure_code(
                params, descriptor, 1u);
            }
            __syncthreads();
            break;
          }

          // The completion helper returns at a CTA barrier. One warp can now
          // retire every newly VALIDATED logical parent without another
          // block-wide readiness scan or publication barrier. Other warps
          // rendezvous at the helper's first barrier in the next iteration.
          if (threadIdx.x < completion_warp_width) {
            const u32 position = completion_lane;
            const u32 rob_slot = position < selected_count
              ? (core_batch_positional != 0
                   ? position : issue_rob_slots[position])
              : UINT32_MAX;
            FrontierRobEntry* entry =
              rob_slot < kPersistentFrontierRobCapacity
                ? frontier_rob + rob_slot : nullptr;
            const bool ready =
              position < selected_count &&
              (ooo_completed_parent_mask &
                 (u32{1} << position)) == 0 &&
              entry != nullptr &&
              entry->state ==
                static_cast<u8>(FrontierRequestState::validated) &&
              entry->node_handle == selected_handles[position];
            const u32 ready_parent_mask =
              __ballot_sync(0xffffffffu, ready);
            if (ready) {
              entry->state =
                static_cast<u8>(FrontierRequestState::committed);
            }
            if (completion_lane == 0) {
              const u32 selected_mask =
                selected_count == completion_warp_width
                  ? 0xffffffffu : (u32{1} << selected_count) - 1u;
              const u32 not_ready_mask =
                selected_mask & ~ooo_completed_parent_mask &
                ~ready_parent_mask;
              if (not_ready_mask != 0) {
                const u32 first_hole =
                  static_cast<u32>(__ffs(not_ready_mask) - 1);
                const u32 prefix_through_hole =
                  first_hole + 1u == completion_warp_width
                    ? 0xffffffffu
                    : (u32{1} << (first_hole + 1u)) - 1u;
                ooo_bypassed_parents += __popc(
                  ready_parent_mask & ~prefix_through_hole);
              }
              ooo_completed_parent_mask |= ready_parent_mask;
            }
          }
        }
        // The final completion clears core_batch.active, so there is no next
        // helper-entry barrier at which non-owner warps can rendezvous.
        __syncthreads();
      }

      if (!ooo_pipeline_enabled &&
          !finish_query_core_frontier_batch(
                   params, descriptor, frontier_rob, core_batch,
                   min(params.commit_width,
                       static_cast<u32>(
                         kPersistentFrontierRobCapacity)),
                   core_telemetry)) {
        if (threadIdx.x == 0) {
          graph_failed = frontier_graph_failure_code(
            params, descriptor, 3u);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        rdma_wait_cycles +=
          speculative_wait_cycles - speculative_wait_before;
        speculative_wait_before = speculative_wait_cycles;
      }
      __syncthreads();
      // The tail state is CTA-resident and therefore a uniform predicate.
      // Skip the out-of-line completion frame entirely when no speculative
      // descriptor is active; this is the steady state after the controller
      // collapses an unprofitable suffix.
      if (tail_batch.active != 0) {
        if (threadIdx.x == 0) {
          tail_underhint_scratch.tail_stale_before = speculative_stale;
        }
        __syncthreads();
        if (!finish_query_tail_frontier_batch<false>(
              params, descriptor, frontier_rob, tail_batch,
              tail_telemetry)) {
          if (threadIdx.x == 0) graph_failed = 4u;
        }
        apply_tail_admission_correction(
          tail_admission_correction, tail_feedback,
          total_remote_batches, total_remote_reads,
          total_graph_read_bytes, total_graph_live_extent_reads,
          total_graph_full_record_reads, speculative_graph_reads,
          speculative_graph_bytes, speculative_queue_rejects);
        // Every active path through finish_query_tail_frontier_batch ends at
        // a CTA barrier, so lane zero can consume its counters immediately.
        if (threadIdx.x == 0) {
          const u64 tail_wait_delta =
            speculative_wait_cycles - speculative_wait_before;
          rdma_wait_cycles += tail_wait_delta;
          speculative_wait_before = speculative_wait_cycles;
          tail_feedback.stale +=
            speculative_stale -
              tail_underhint_scratch.tail_stale_before;
        }
      }
      __syncthreads();

      if (params.graph_request_bytes != nullptr) {
        const UnderhintLookupMode underhint_lookup_mode =
          core_batch_positional != 0
            ? UnderhintLookupMode::positional
            : ((selection_from_certificate != 0 &&
                certified_mapping_ready != 0)
                 ? UnderhintLookupMode::certified
                 : UnderhintLookupMode::associative);
        identify_selected_underhint_force_full(
          selected_handles, selected_count, frontier_rob, issue_rob_slots,
          underhint_lookup_mode, selected_underhint_force_full,
          &tail_underhint_scratch.selected_underhint_any);
      }

      if (core_batch_positional != 0) {
        reconcile_exact_core_frontier(
          selected_handles, selected_count, frontier_rob,
          min(params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity)),
          commit_rob_slots, graph_record_slots,
          critical_fetch_handles, critical_fetch_to_commit,
          critical_fetch_destination_slots, critical_fetch_count,
          critical_rob_hits, critical_misses,
          core_prefetch_promoted, core_prefetch_stale);
        if (threadIdx.x == 0) {
          reconciled_positional_core = 1;
          core_batch_positional = 0;
          certified_mapping_ready = 0;
        }
      } else if (selection_from_certificate != 0 &&
                 certified_mapping_ready != 0) {
        reconcile_certified_commit_frontier(
          certified_commit_context, selected_count,
          issue_epochs != 0);
        if (threadIdx.x == 0) {
          // Arbitrary retained slots still use the ordinary free-slot
          // allocator below for any mandatory critical retry.
          reconciled_positional_core = 0;
          certified_mapping_ready = 0;
        }
      } else {
        plan_commit_frontier(
          beam_handles, beam_expanded, beam_count,
          min(params.commit_width, params.max_expansions - expansions),
          frontier_rob, frontier_controller, issue_epochs != 0,
          0, 0, selected_handles, selected_beam_ranks,
          commit_rob_slots, selected_count,
          critical_fetch_handles, critical_fetch_to_commit,
          critical_fetch_count, graph_record_slots,
          shadow_frontier_count, speculative_stale,
          speculative_wasted_bytes, speculative_promoted,
          core_prefetch_stale, core_prefetch_promoted,
          critical_rob_hits, critical_misses,
          commit_epochs, commit_width_sum, max_commit_width,
          tail_feedback, true);
        if (threadIdx.x == 0) {
          reconciled_positional_core = 0;
          certified_mapping_ready = 0;
        }
      }
      __syncthreads();

      if (reconciled_positional_core == 0) {
        reserve_critical_fetch_destinations(
          frontier_rob, critical_fetch_count,
          critical_fetch_destination_slots, graph_failed,
          speculative_stale, speculative_wasted_bytes,
          core_prefetch_stale, tail_feedback);
      }
      __syncthreads();
    } else if (threadIdx.x == 0) {
      graph_failed = 0;
      selected_count = 0;
      const u32 target =
        min(params.commit_width, params.max_expansions - expansions);
      for (u32 index = 0;
           index < beam_count && selected_count < target; ++index) {
        if (beam_expanded[index] != 0) continue;
        const u32 position = selected_count++;
        selected_handles[position] = beam_handles[index];
        selected_beam_ranks[position] = index;
        critical_fetch_handles[position] = beam_handles[index];
        critical_fetch_to_commit[position] = position;
      }
      critical_fetch_count = selected_count;
      if (selected_count != 0) {
        ++commit_epochs;
        commit_width_sum += selected_count;
        max_commit_width = max(max_commit_width, selected_count);
        critical_misses += selected_count;
      }
    }
    if constexpr (!EnableAsfe) {
      for (u32 fetch = threadIdx.x; fetch < critical_fetch_count;
           fetch += blockDim.x) {
        critical_fetch_destination_slots[fetch] =
          critical_fetch_to_commit[fetch];
      }
    }
    __syncthreads();
    if (params.graph_request_bytes != nullptr &&
        tail_underhint_scratch.selected_underhint_any != 0) {
      remap_critical_underhint_force_full(
        selected_underhint_force_full, selected_count,
        critical_fetch_to_commit, critical_fetch_count,
        critical_fetch_force_full);
    }
    if (threadIdx.x == 0) {
      phase_started_cycles = clock64();
    }
    if (selected_count == 0) break;
    if (threadIdx.x == 0) ++total_graph_rounds;
    __syncthreads();
    constexpr u32 warp_width = 32;
    const u32 warp = threadIdx.x / warp_width;
    const u32 lane_in_warp = threadIdx.x % warp_width;
    __shared__ u64 critical_bytes_before;
    if (threadIdx.x == 0) {
      graph_fetch_breakdown = {};
      critical_bytes_before = total_graph_read_bytes;
    }
    __syncthreads();
    if (critical_fetch_count != 0 && graph_failed == 0) {
      const u32 fetch_failure = fetch_graph_records_batch(
            params, descriptor, critical_fetch_handles,
            critical_fetch_count, critical_fetch_slots,
            critical_fetch_destination_slots,
            remote_reads_by_lane, &total_remote_batches,
            &total_graph_read_retries, &total_graph_read_bytes,
            &total_graph_live_extent_reads,
            &total_graph_full_record_reads,
            &total_graph_extent_fallback_reads,
            &total_graph_extent_underhint_reads,
            &total_graph_extent_hint_promotions,
            route_attempt, total_graph_rounds - 1,
            rdma_trace_enabled != 0, &graph_fetch_breakdown,
            &dynamic_graph_telemetry,
            params.graph_request_bytes != nullptr &&
                tail_underhint_scratch.selected_underhint_any != 0
              ? critical_fetch_force_full : nullptr);
      if (fetch_failure != 0) {
        if (threadIdx.x == 0) {
          const u32 contextual_failure =
            fetch_failure |
            (selection_from_certificate != 0 &&
                 (fetch_failure & (u32{1} << 16)) == 0
               ? u32{1} << 15 : 0u);
          graph_failed = 6u | (contextual_failure << 8);
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rdma_issue_cycles += graph_fetch_breakdown.issue;
      rdma_wait_cycles += graph_fetch_breakdown.wait;
      graph_validation_cycles += graph_fetch_breakdown.validation;
      rdma_completion_latency_ns +=
        graph_fetch_breakdown.completion_latency_ns;
      rdma_completion_groups += graph_fetch_breakdown.completion_groups;
      graph_phase_cycles += clock64() - phase_started_cycles;
      critical_graph_bytes +=
        total_graph_read_bytes - critical_bytes_before;
      for (u32 selected = 0; selected < critical_fetch_count; ++selected) {
        // fetch_graph_records_batch may use upper bits as private per-parent
        // retry metadata. Bit zero alone denotes the one logical graph read
        // selected by the search algorithm.
        total_remote_reads += remote_reads_by_lane[selected] & 1u;
        critical_graph_reads += remote_reads_by_lane[selected] & 1u;
        graph_record_slots[critical_fetch_to_commit[selected]] =
          critical_fetch_slots[selected];
      }
    }
    __syncthreads();
    if (graph_failed != 0) {
      for (u32 selected = warp; selected < selected_count;
           selected += blockDim.x / warp_width) {
        if (lane_in_warp == 0) graph_record_slots[selected] = UINT32_MAX;
      }
      __syncthreads();
      // A failed critical fetch may race with an already-admitted ASFE wave.
      // The query slot (and its graph scratch) cannot be reused until both
      // owner descriptors have been synchronously drained.  This mirrors the
      // normal end-of-query cleanup and closes the late-DMA overwrite window.
      if constexpr (EnableAsfe) {
        (void)finish_query_core_frontier_batch(
          params, descriptor, frontier_rob, core_batch,
          min(params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity)),
          core_telemetry);
        if (tail_batch.active != 0) {
          (void)finish_query_tail_frontier_batch<true>(
            params, descriptor, frontier_rob, tail_batch, tail_telemetry);
          apply_tail_admission_correction(
            tail_admission_correction, tail_feedback,
            total_remote_batches, total_remote_reads,
            total_graph_read_bytes, total_graph_live_extent_reads,
            total_graph_full_record_reads, speculative_graph_reads,
            speculative_graph_bytes, speculative_queue_rejects);
        }
        __syncthreads();
        // A terminal exact train is launched at the end of the preceding
        // epoch and intentionally remains active while this epoch advances.
        // Therefore this early authoritative-fetch failure can observe an
        // older terminal train even though it occurs before this epoch's
        // terminal-horizon code below. The completion releases query_slot;
        // synchronously retire that train first so its full-record/header
        // DMA cannot land in the next incarnation of the slot.
        drain_terminal_exact_cache_prefetch(
          params, descriptor, terminal_exact_cache);
      }
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.result_count = graph_failed;
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
        set_dynamic_graph_completion(completion, dynamic_graph_telemetry);
        set_frontier_completion<EnableAsfe>(
          completion, logical_expansions, critical_graph_reads,
          critical_graph_bytes, speculative_graph_reads,
          speculative_graph_bytes, speculative_arrived,
          speculative_promoted, speculative_stale,
          speculative_wasted_bytes, speculative_queue_rejects,
          issue_epochs, commit_epochs, issue_width_sum,
          issue_width_capacity_sum, commit_width_sum, max_issue_width,
          max_commit_width, critical_rob_hits, critical_misses,
          speculative_wait_cycles,
          rdma_completion_latency_ns, speculative_completion_latency_ns,
          rdma_completion_groups, speculative_completion_groups,
          core_prefetch_bytes, core_prefetch_reads, core_prefetch_arrived,
          core_prefetch_promoted, core_prefetch_stale,
          core_prefetch_queue_rejects, core_prefetch_waves,
          core_ready_waves);
        set_frontier_certificate_completion<EnableAsfe>(
          completion, frontier_reusable_certificates,
          frontier_streamed_candidate_runs, ordered_score_batches,
          ordered_score_candidates, frontier_reusable_prefix_ranks,
          frontier_reusable_full_prefix_certificates,
          frontier_reusable_issued_certificates,
          ooo_bypassed_parents,
          frontier_certificate_rejects);
        // Failure-only provenance: preserve the exact handle rejected by the
        // authoritative fetch preparer.  The two reserved words normally
        // carry no ABI-visible result data; on a failed query they let the
        // host distinguish a malformed Beam value from a bad destination or
        // transport error without adding any work to successful queries.
        const u32 fetch_detail = graph_failed >> 8;
        if ((fetch_detail & (u32{1} << 16)) == 0) {
          const u32 failed_index = (fetch_detail >> 4) & 0x1fu;
          if (failed_index < critical_fetch_count) {
            const u64 failed_handle = critical_fetch_handles[failed_index];
            completion.frontier_telemetry_reserved0 =
              static_cast<u32>(failed_handle);
            completion.frontier_telemetry_reserved1 =
              static_cast<u32>(failed_handle >> 32);
          }
        }
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
          dynamic_code_cache_publish_successes,
          dynamic_code_cache_first_occupancies,
          dynamic_code_cache_publish_races,
          dynamic_code_cache_lookup_probe_exhaustions,
          dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
        set_beam_merge_completion(completion, beam_merge_breakdown);
        set_terminal_exact_cache_completion<EnableAsfe>(
          completion, terminal_exact_cache);
        set_query_trace_completion(params, query_slot, completion);
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }

    // Commit becomes authoritative only after every missing dependency has
    // completed the full critical Live-Extent/retry path. The frozen Beam
    // ranks preserve the legacy whole-batch expansion order.
    if (threadIdx.x == 0 && graph_failed == 0) {
      for (u32 position = 0; position < selected_count; ++position) {
        beam_expanded[selected_beam_ranks[position]] = 1;
      }
      phase_started_cycles = clock64();
    }
    __syncthreads();
    const u32 score_chunk_capacity = persistent_score_chunk_capacity(
      params.graph_entry_capacity, traversal_capacity);
    if (score_chunk_capacity == 0) {
      if (threadIdx.x == 0) graph_failed = 7u;
      __syncthreads();
    }
    for (u32 chunk_begin = 0;
         graph_failed == 0 && chunk_begin < selected_count;
         chunk_begin += score_chunk_capacity) {
      const u32 chunk_count = min(score_chunk_capacity,
                                  selected_count - chunk_begin);
      const bool final_parent_chunk =
        chunk_begin + chunk_count >= selected_count;
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
          neighbor_counts[local] =
            record != nullptr && (record[1] & 1u) == 0
              ? min(stable_count + provisional_count,
                    params.graph_entry_capacity)
              : 0;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        neighbor_offsets[0] = 0;
        for (u32 local = 0; local < chunk_count; ++local) {
          const u32 count = neighbor_counts[local];
          const u32 extent_class = min(
            (count + 7u) / 8u, kGraphDegreeHistogramBuckets - 1u);
          ++completion.expanded_parent_count;
          completion.expanded_neighbor_count_sum += count;
          ++completion.expanded_degree_histogram[extent_class];
          // selected_handles names the exact, already validated version that
          // is about to be expanded.  Record its true degree after the read;
          // this is observational Oracle telemetry and is never consulted by
          // request preparation.
          if (dynamic_graph_telemetry_handle(
                selected_handles[chunk_begin + local])) {
            ++completion.dynamic_expanded_parent_count;
            completion.dynamic_expanded_neighbor_count_sum += count;
            ++completion.dynamic_expanded_degree_histogram[extent_class];
          }
          neighbor_offsets[local + 1] =
            neighbor_offsets[local] + count;
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
        for (u32 neighbor = lane_in_warp; neighbor < count;
             neighbor += warp_width) {
          const u64 raw = decode_tagged_raw(
            record + 16 + neighbor * sizeof(u64));
          navigation_handles[neighbor_offsets[local] + neighbor] =
            handle_from_raw(params, raw);
        }
        __syncwarp();
        if (lane_in_warp == 0) {
          graph_record_slots[selected] = UINT32_MAX;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        neighbor_decode_cycles += clock64() - phase_started_cycles;
      }
      if constexpr (EnableAsfe) {
        if (final_parent_chunk) {
          // Every committed payload has now been copied into the flat
          // navigation stream. Release its ROB/scratch ownership before the
          // candidate Stable-Run launches the next shadow wave; the RDMA
          // destination may then be overwritten because visited/PQ consume
          // only the copied handles.
          for (u32 slot = threadIdx.x;
               slot < kPersistentFrontierRobCapacity;
               slot += blockDim.x) {
            if (frontier_rob[slot].state ==
                static_cast<u8>(FrontierRequestState::committed)) {
              frontier_rob[slot] = {};
            }
          }
        }
        __syncthreads();
      }
      const u32 candidate_count = flattened_neighbors;
      if (candidate_count != 0) {
        if (threadIdx.x == 0) phase_started_cycles = clock64();
        __syncthreads();
        for (u32 flat = threadIdx.x;
             flat < candidate_count; flat += blockDim.x) {
          const u64 handle = navigation_handles[flat];
          if (handle == kInvalidDeviceHandle ||
              !insert_visited(visited, params.visited_capacity, handle)) {
            navigation_handles[flat] = kInvalidDeviceHandle;
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          visited_cycles += clock64() - phase_started_cycles;
        }
      }
      if constexpr (EnableAsfe) {
        if (threadIdx.x == 0 && candidate_count != 0) {
          // All speculative/critical graph dependencies have been reconciled
          // and every mandatory miss has completed before this point.
          ++completion.completion_score_batches;
          completion.completion_score_candidates += candidate_count;
        }
      }
      if (threadIdx.x == 0) phase_started_cycles = clock64();
      if (candidate_count != 0 &&
          !approximate_handles_batch(
            params, descriptor, query_lut,
            navigation_handles, candidate_count,
            navigation_distances,
            &dynamic_code_cycles,
            &dynamic_code_candidates, &dynamic_code_reads,
            &dynamic_code_incarnation_rejects, &dynamic_code_cache_hits,
            &dynamic_code_batch_deduplicated,
            &dynamic_code_cache_publish_successes,
            &dynamic_code_cache_first_occupancies,
            &dynamic_code_cache_publish_races,
            &dynamic_code_cache_lookup_probe_exhaustions,
            &dynamic_code_cache_publish_probe_exhaustions,
            &dynamic_code_cache_lookup_probes,
            &dynamic_code_cache_max_lookup_probes)) {
        if (threadIdx.x == 0) graph_failed = 8u;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        if (candidate_count != 0) {
          pq_score_cycles += clock64() - phase_started_cycles;
        }
        score_phase_cycles = neighbor_decode_cycles +
          visited_cycles + pq_score_cycles;
        phase_started_cycles = clock64();
        // rerank_count is otherwise dead throughout traversal. Preserve the
        // non-empty old Beam cardinality across this merge so an impossible
        // topK(old Beam U candidates) -> empty transition can be attributed
        // to the exact Stable-Run publication path that produced it.
        rerank_count = beam_count;
      }
      __syncthreads();
      if (graph_failed != 0) break;
      u64* candidate_input_handles = navigation_handles;
      f32* candidate_input_distances = navigation_distances;
      const bool asfe_stable_run =
        EnableAsfe &&
        blockDim.x == kApproximateSortThreadsCompact &&
        params.beam_merge_policy ==
          static_cast<u32>(BeamMergePolicy::stable_run);
      if (stable_runs_prepared_before_issue == 3) {
        if (threadIdx.x == 0) {
          // Exclude descriptor publication from Stable-Run accounting; only
          // the mandatory remaining leaf work belongs to Beam merge.
          stable_merge_state.phase_started = clock64();
        }
        __syncthreads();
        complete_compact_approximate_stable_runs(
          candidate_input_handles, candidate_input_distances,
          candidate_count, traversal_capacity,
          rerank_handles, rerank_flags, rerank_distances,
          candidate_workspace, stable_merge_state,
          &beam_merge_round_breakdown, false);
        if (threadIdx.x == 0) {
          stable_runs_prepared_before_issue = 1;
        }
        __syncthreads();
      }
      if (stable_runs_prepared_before_issue == 2) {
        constexpr u32 stable_leaf_capacity =
          kApproximateSortThreadsCompact *
          kApproximateSortItemsCompactFinal256;
        // Sort every complete/partial canonical leaf after visited and
        // dynamic PQ have resolved. An absent leaf is the identity
        // (+infinity) run, so retain the logical arity instead of writing and
        // later merging its 128 sentinel entries.
        const u32 final_candidate_run_count = min(
          4u,
          (candidate_count + stable_leaf_capacity - 1u) /
            stable_leaf_capacity);
        while (stable_merge_state.candidate_run_count <
               final_candidate_run_count) {
          const u32 pass =
            stable_merge_state.candidate_run_count;
          const u32 input_offset =
            pass * stable_leaf_capacity;
          const u32 output_offset =
            pass * traversal_capacity;
          if (threadIdx.x == 0) {
            stable_merge_state.phase_started = clock64();
          }
          __syncthreads();
          stable_sort_candidate_run<
              ApproximateBlockSortCompactFinal256,
              kApproximateSortItemsCompactFinal256>(
            candidate_workspace.sort.radix_sort_compact_final_256,
            candidate_input_handles, candidate_input_distances,
            candidate_count, input_offset,
            rerank_handles, rerank_flags, rerank_distances,
            output_offset, traversal_capacity,
            &beam_merge_round_breakdown,
            &stable_merge_state.phase_started);
          if (threadIdx.x == 0) {
            ++stable_merge_state.candidate_run_count;
            ++frontier_streamed_candidate_runs;
          }
          __syncthreads();
        }
        if (threadIdx.x == 0) {
          stable_merge_state.prepared = 1;
          stable_runs_prepared_before_issue = 1;
        }
        __syncthreads();
      }
      const bool exact_frontier_issue_enabled =
        asfe_stable_run && final_parent_chunk &&
        params.issue_width > params.commit_width &&
        expansions + selected_count < params.max_expansions;
      if (exact_frontier_issue_enabled) {
        const u32 remaining_search_budget =
          params.max_expansions - (expansions + selected_count);
        if (remaining_search_budget != 0 &&
            core_batch.active == 0) {
          // There is no future Commit Frontier when this round consumes the
          // remaining expansion budget. Do not issue a tail that would only
          // be synchronously drained on query completion.
          const u32 issue_capacity_limit =
            remaining_search_budget > params.commit_width
              ? frontier_controller.current_issue_width
              : params.commit_width;
          const u32 preview_capacity = min(
            issue_capacity_limit, remaining_search_budget);
          const u32 certificate_target = min(
            params.commit_width, remaining_search_budget);

          if (threadIdx.x == 0) {
            shadow_issue_started_cycles = clock64();
          }
          __syncthreads();
          if (stable_runs_prepared_before_issue == 0) {
            // PFEC sorts the four immutable leaves once, emits an exact
            // certificate, and retains the reusable private merge prefix.
            // Beam/expanded/visited remain untouched until finish below.
            auto& warp_leaf_storage =
              *reinterpret_cast<ApproximateWarpLeafSortStorage*>(
                &candidate_workspace);
            prepare_warp_leaf_fused_frontier_certificate(
              candidate_input_handles, candidate_input_distances,
              candidate_count, beam_handles, beam_distances,
              beam_expanded, beam_count, traversal_capacity,
              rerank_handles, rerank_flags, rerank_distances,
              warp_leaf_storage, candidate_workspace.arrays,
              preview_capacity, shadow_frontier_handles,
              shadow_frontier_ranks, shadow_frontier_count,
              stable_merge_state, &beam_merge_round_breakdown);
            if (threadIdx.x == 0) {
              stable_runs_prepared_before_issue = 5;
              frontier_streamed_candidate_runs += 4;
            }
            __syncthreads();
          } else {
            prepare_deferred_fused_frontier_certificate(
              beam_handles, beam_distances, beam_expanded,
              traversal_capacity, stable_merge_state.original_count,
              rerank_handles, rerank_flags, rerank_distances,
              stable_merge_state.candidate_run_count,
              candidate_workspace.arrays, preview_capacity,
              shadow_frontier_handles, shadow_frontier_ranks,
              shadow_frontier_count, stable_merge_state, nullptr);
          }
          if (threadIdx.x == 0) {
            const u64 now = clock64();
            frontier_preview_cycles +=
              now - shadow_issue_started_cycles;
            shadow_issue_started_cycles = now;
            ++frontier_reusable_certificates;
            frontier_reusable_prefix_ranks +=
              shadow_frontier_count;
            // Certificate construction is complete. Start communication
            // accounting only after its exact frontier exists.
            frontier_subphase_started_cycles = now;
          }
          // Cache the next authoritative commit prefix before ROB admission
          // compacts or rejects the speculative suffix.  The preview is over
          // the exact four-run merge and therefore remains valid even when
          // the tail is throttled; it is consumed only after this round's
          // materialization barrier at the top of the next round.
          if (threadIdx.x == 0) {
            if (shadow_frontier_count >= certificate_target &&
                certificate_target != 0) {
              for (u32 position = 0; position < certificate_target;
                   ++position) {
                selected_handles[position] =
                  shadow_frontier_handles[position];
                selected_beam_ranks[position] =
                  static_cast<u32>(shadow_frontier_ranks[position]);
                commit_rob_slots[position] = UINT32_MAX;
              }
              next_commit_count = certificate_target;
              next_commit_ready = 1;
            } else {
              next_commit_count = 0;
              next_commit_ready = 0;
            }
          }
          // Tail completion is consumed once, immediately before the next
          // Commit Frontier is reconciled.  A second end-of-round poll used
          // to scan the same per-shard words solely to enable immediate tail
          // refill.  Leaving an unfinished tail resident is both safe (core
          // metadata/scratch are disjoint) and a direct, latency-derived
          // admission throttle: the next exact critical prefix can still be
          // issued, while no further speculative wave is admitted until the
          // current one has actually completed.
          if (shadow_frontier_count != 0) {
            // Do not round a valid shadow suffix down to a full Commit
            // Frontier.  A one-entry probe is intentional: it gives the
            // online controller a query-local promotion/staleness sample and
            // lets the owner pipeline hide even a small amount of latency.
            // The controller, rather than a width heuristic, decides whether
            // subsequent suffixes grow or shrink.
            __syncthreads();
            const u32 positional_core_width = min(
              params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity));
            if (threadIdx.x < 32) {
              const u32 lane = threadIdx.x;
              const bool live_tail =
                lane >= positional_core_width &&
                frontier_rob[lane].state !=
                  static_cast<u8>(FrontierRequestState::init);
              const u32 live_tail_mask =
                __ballot_sync(0xffffffffu, live_tail);
              if (lane == 0) {
                const bool controller_feedback_empty =
                  tail_feedback.promoted == 0 &&
                  tail_feedback.retained == 0 &&
                  tail_feedback.stale == 0 &&
                  tail_feedback.queue_rejects == 0 &&
                  tail_feedback.tail_admitted == 0 &&
                  tail_feedback.commit_waves_observed == 0 &&
                  tail_feedback.commit_waves_covered == 0;
                core_only_issue_epoch =
                  tail_batch.active == 0 &&
                  live_tail_mask == 0 &&
                  controller_feedback_empty &&
                  shadow_frontier_count <= positional_core_width
                    ? 1u : 0u;
              }
            }
            __syncthreads();
            if (threadIdx.x == 0) physical_issue_span = 0;
            __syncthreads();
            if (core_only_issue_epoch != 0) {
              prepare_exact_core_frontier(
                shadow_frontier_handles, shadow_frontier_ranks,
                shadow_frontier_count, frontier_rob,
                positional_core_width, frontier_issue_epoch,
                issue_epochs, issue_width_sum,
                issue_width_capacity_sum,
                frontier_controller.current_issue_width, max_issue_width);
            } else {
              // Only an empty physical tail may accept a fresh speculative
              // suffix.  A resident tail is still eligible for exact
              // promotion/mapping above, but it must not be overwritten or
              // allowed to delay this round's critical prefix.
              const bool may_admit_exact_tail = tail_batch.active == 0;
              prepare_issue_frontier_entries(
                shadow_frontier_handles, shadow_frontier_ranks,
                shadow_frontier_count, frontier_rob,
                frontier_issue_epoch, frontier_controller,
                positional_core_width,
                tail_feedback, speculative_stale,
                speculative_wasted_bytes, core_prefetch_stale,
                issue_epochs, issue_width_sum,
                issue_width_capacity_sum, max_issue_width,
                issue_rob_slots, physical_issue_span,
                // PFEC has exposed the exact next frontier before the
                // authoritative merge.  Its suffix therefore has the full
                // merge + next-round overlap window and is safe to issue
                // whenever no older tail occupies the speculative WQEs.
                may_admit_exact_tail,
                // Reconcile and train on the preceding wave exactly once
                // before admitting this wave.  The updated width applies to
                // the following certificate, avoiding self-training.
                true);
            }
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            const u64 now = clock64();
            frontier_prepare_cycles +=
              now - frontier_subphase_started_cycles;
            frontier_subphase_started_cycles = now;
            early_queue_rejects_before =
              speculative_queue_rejects;
          }
          __syncthreads();
          if (shadow_frontier_count != 0) {
            const u32 core_slot_count = min(
              params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity - 1));
            const bool split_issue =
              physical_issue_span > core_slot_count;
            bool issue_ok = true;
            if (tail_batch.active == 0 && split_issue) {
              issue_ok = issue_split_frontier_graph_batch(
                params, descriptor, frontier_rob,
                core_batch, tail_batch, physical_issue_span,
                core_slot_count,
                params.core_batch_statuses,
                params.core_batch_completion_timestamps_ns,
                params.tail_batch_statuses,
                params.tail_batch_completion_timestamps_ns,
                &total_remote_batches, &total_remote_reads,
                &total_graph_read_bytes,
                &total_graph_live_extent_reads,
                &total_graph_full_record_reads,
                &critical_graph_reads, &critical_graph_bytes,
                &speculative_graph_reads, &speculative_graph_bytes,
                &core_prefetch_queue_rejects,
                &speculative_queue_rejects,
                &core_prefetch_reads, &core_prefetch_bytes,
                &tail_feedback.tail_admitted,
                &dynamic_graph_telemetry);
            } else {
              // The narrow core-only path preserves the critical-first
              // publication cost of the coupled engine. A resident tail does
              // not block this independent critical prefix.
              issue_ok = issue_frontier_graph_batch(
                params, descriptor, frontier_rob, core_batch,
                0, core_slot_count,
                params.core_batch_statuses,
                params.core_batch_completion_timestamps_ns,
                DirectBatchPriority::critical, true,
                &total_remote_batches, &total_remote_reads,
                &total_graph_read_bytes,
                &total_graph_live_extent_reads,
                &total_graph_full_record_reads,
                &critical_graph_reads, &critical_graph_bytes,
                &core_prefetch_queue_rejects,
                &core_prefetch_reads, &core_prefetch_bytes,
                &dynamic_graph_telemetry);
            }
            if (!issue_ok && threadIdx.x == 0) {
              graph_failed = frontier_graph_failure_code(
                params, descriptor, 9u);
            }
            if (threadIdx.x == 0) {
              core_batch_positional =
                core_only_issue_epoch != 0 ? 1u : 0u;
              certified_mapping_ready =
                core_only_issue_epoch == 0 &&
                next_commit_ready != 0 ? 1u : 0u;
            }
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            const u64 now = clock64();
            frontier_enqueue_cycles +=
              now - frontier_subphase_started_cycles;
            if (core_batch.active != 0) {
              ++core_prefetch_waves;
              ++frontier_reusable_issued_certificates;
            }
            tail_feedback.queue_rejects +=
              speculative_queue_rejects -
                early_queue_rejects_before;
            const u64 frontier_issue_cycles =
              now - shadow_issue_started_cycles;
            rdma_issue_cycles += frontier_issue_cycles;
            graph_phase_cycles += frontier_issue_cycles;
            phase_started_cycles = clock64();
          }
          __syncthreads();
        }
        if (graph_failed != 0) break;
        if (threadIdx.x == 0) {
          // Do not charge certificate preview/enqueue/RDMA issue time to
          // Stable-Run. Prepared leaves remain immutable across issue.
          stable_merge_state.phase_started = clock64();
        }
        __syncthreads();
        finish_approximate_stable_runs(
          beam_handles, beam_ids, beam_distances,
          beam_expanded, beam_count, traversal_capacity,
          rerank_handles, rerank_flags, rerank_distances,
          candidate_workspace, stable_merge_state,
          &beam_merge_round_breakdown, true);
      } else {
        if (asfe_stable_run) {
          if (stable_runs_prepared_before_issue == 0) {
            prepare_approximate_stable_runs(
              candidate_input_handles, candidate_input_distances,
              candidate_count,
              beam_handles, beam_ids, beam_distances, beam_expanded,
              beam_count, traversal_capacity,
              rerank_handles, rerank_flags, rerank_distances,
              candidate_workspace, stable_merge_state,
              &beam_merge_round_breakdown, false);
          } else {
            if (threadIdx.x == 0) {
              stable_merge_state.phase_started = clock64();
            }
            __syncthreads();
          }
          finish_approximate_stable_runs(
            beam_handles, beam_ids, beam_distances,
            beam_expanded, beam_count, traversal_capacity,
            rerank_handles, rerank_flags, rerank_distances,
            candidate_workspace, stable_merge_state,
            &beam_merge_round_breakdown, true);
        } else {
          merge_approximate_into_beam(
            candidate_input_handles, candidate_input_distances,
            candidate_count, beam_handles, beam_ids, beam_distances,
            beam_expanded, beam_count, traversal_capacity,
            rerank_handles, rerank_flags, rerank_distances,
            candidate_workspace,
            static_cast<BeamMergePolicy>(params.beam_merge_policy),
            params.beam_merge_policy ==
                static_cast<u32>(BeamMergePolicy::stable_run)
              ? &beam_merge_round_breakdown : nullptr);
        }
      }
      if (threadIdx.x == 0) {
        if (rerank_count != 0 && beam_count == 0) {
          route_failure_reason =
            0x80000000u |
            ((stable_runs_prepared_before_issue & 0x3u) << 28) |
            ((stable_merge_state.prepared & 0x1u) << 27) |
            ((stable_merge_state.original_count & 0xffu) << 19) |
            ((candidate_count & 0x7ffu) << 8) |
            (total_graph_rounds & 0xffu);
        }
        const u64 merge_cycles = exact_frontier_issue_enabled
          ? beam_merge_round_breakdown.prepare +
              beam_merge_round_breakdown.sort +
              beam_merge_round_breakdown.materialize
          : clock64() - phase_started_cycles;
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
      }
      __syncthreads();
      __syncthreads();
    }
    if constexpr (EnableAsfe) {
      // The complete Stable-Run publication above is the only authority for
      // traversal state.  Once its remaining unexpanded frontier fits in one
      // Commit wave, the following epoch is the earliest observable terminal
      // horizon: either that wave exhausts the frontier or it publishes the
      // final replacement candidates.  Enqueue a read-only exact snapshot
      // now, behind the already-issued critical graph core, and overlap it
      // with that last graph epoch.  The terminal exactifier revalidates every
      // cached record against the final Beam/header and falls back to the
      // established fenced path for all churn, deletes, and updates.
      if (threadIdx.x == 0) {
        terminal_exact_horizon_ready = 0;
        if (terminal_exact_cache.attempted == 0) {
          u32 remaining = 0;
          for (u32 rank = 0; rank < beam_count; ++rank) {
            remaining += beam_expanded[rank] == 0 ? 1u : 0u;
          }
          // graph_failed is known before this point.  Never publish a new
          // terminal RDMA descriptor for a query that is about to take the
          // failure exit below: its exact-record scratch is query-slot-owned
          // and must not outlive the completion that releases that slot.
          terminal_exact_horizon_ready = graph_failed == 0 &&
            (remaining <= params.commit_width ||
             expansions + selected_count >= params.max_expansions);
        }
      }
      __syncthreads();
      if (terminal_exact_horizon_ready != 0) {
        begin_terminal_exact_cache_prefetch(
          params, descriptor, beam_handles, beam_count,
          terminal_exact_cache);
      }
      __syncthreads();
    }
    if (graph_failed != 0) {
      __shared__ u64 failed_speculative_wait_before;
      if constexpr (EnableAsfe) {
        if (threadIdx.x == 0) {
          failed_speculative_wait_before = speculative_wait_cycles;
        }
        __syncthreads();
        (void)finish_query_core_frontier_batch(
          params, descriptor, frontier_rob, core_batch,
          min(params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity)),
          core_telemetry);
        if (tail_batch.active != 0) {
          (void)finish_query_tail_frontier_batch<true>(
            params, descriptor, frontier_rob, tail_batch, tail_telemetry);
          apply_tail_admission_correction(
            tail_admission_correction, tail_feedback,
            total_remote_batches, total_remote_reads,
            total_graph_read_bytes, total_graph_live_extent_reads,
            total_graph_full_record_reads, speculative_graph_reads,
            speculative_graph_bytes, speculative_queue_rejects);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          rdma_wait_cycles +=
            speculative_wait_cycles - failed_speculative_wait_before;
        }
        __syncthreads();
        // A terminal exact wave may have been admitted by an earlier epoch.
        // The graph-failure completion releases descriptor.query_slot to the
        // host, so every DMA targeting that slot must be quiescent first.
        // Treat its result as unused; the authoritative query is failing for
        // the graph/dynamic-code reason recorded below.
        drain_terminal_exact_cache_prefetch(
          params, descriptor, terminal_exact_cache);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.result_count = graph_failed;
        completion.diagnostic = make_query_diagnostic(
          core_batch.fatal != 0
            ? QueryFailureReason::graph_fetch
            : QueryFailureReason::dynamic_code_fetch,
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
        set_dynamic_graph_completion(completion, dynamic_graph_telemetry);
        set_frontier_completion<EnableAsfe>(
          completion, logical_expansions, critical_graph_reads,
          critical_graph_bytes, speculative_graph_reads,
          speculative_graph_bytes, speculative_arrived,
          speculative_promoted, speculative_stale,
          speculative_wasted_bytes, speculative_queue_rejects,
          issue_epochs, commit_epochs, issue_width_sum,
          issue_width_capacity_sum, commit_width_sum, max_issue_width,
          max_commit_width, critical_rob_hits, critical_misses,
          speculative_wait_cycles,
          rdma_completion_latency_ns, speculative_completion_latency_ns,
          rdma_completion_groups, speculative_completion_groups,
          core_prefetch_bytes, core_prefetch_reads, core_prefetch_arrived,
          core_prefetch_promoted, core_prefetch_stale,
          core_prefetch_queue_rejects, core_prefetch_waves,
          core_ready_waves);
        set_frontier_certificate_completion<EnableAsfe>(
          completion, frontier_reusable_certificates,
          frontier_streamed_candidate_runs, ordered_score_batches,
          ordered_score_candidates, frontier_reusable_prefix_ranks,
          frontier_reusable_full_prefix_certificates,
          frontier_reusable_issued_certificates,
          ooo_bypassed_parents,
          frontier_certificate_rejects);
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
          dynamic_code_cache_publish_successes,
          dynamic_code_cache_first_occupancies,
          dynamic_code_cache_publish_races,
          dynamic_code_cache_lookup_probe_exhaustions,
          dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
        set_beam_merge_completion(completion, beam_merge_breakdown);
        set_terminal_exact_cache_completion<EnableAsfe>(
          completion, terminal_exact_cache);
        set_query_trace_completion(params, query_slot, completion);
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }
    if constexpr (EnableAsfe) {
      for (u32 slot = threadIdx.x; slot < kPersistentFrontierRobCapacity;
           slot += blockDim.x) {
        if (frontier_rob[slot].state ==
            static_cast<u8>(FrontierRequestState::committed)) {
          frontier_rob[slot] = {};
        }
      }
    }
    if (threadIdx.x == 0) {
      expansions += selected_count;
      logical_expansions += selected_count;
    }
    __syncthreads();
  }

  // The last commit epoch may have launched a shadow wave immediately before
  // max_expansions was reached. Drain it before exact rerank or query-slot
  // reuse; otherwise a late DMA could overwrite the next query's scratch.
  __shared__ u32 final_speculative_drain_failed;
  __shared__ u64 final_speculative_wait_before;
  if constexpr (EnableAsfe) {
    if (threadIdx.x == 0) {
      final_speculative_drain_failed = 0;
      final_speculative_wait_before = speculative_wait_cycles;
    }
    __syncthreads();
    if (!finish_query_core_frontier_batch(
          params, descriptor, frontier_rob, core_batch,
          min(params.commit_width,
              static_cast<u32>(kPersistentFrontierRobCapacity)),
          core_telemetry)) {
      if (threadIdx.x == 0) final_speculative_drain_failed = 1;
    }
    if (tail_batch.active != 0) {
      if (!finish_query_tail_frontier_batch<true>(
            params, descriptor, frontier_rob, tail_batch,
            tail_telemetry)) {
        if (threadIdx.x == 0) final_speculative_drain_failed = 1;
      }
      apply_tail_admission_correction(
        tail_admission_correction, tail_feedback,
        total_remote_batches, total_remote_reads,
        total_graph_read_bytes, total_graph_live_extent_reads,
        total_graph_full_record_reads, speculative_graph_reads,
        speculative_graph_bytes, speculative_queue_rejects);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rdma_wait_cycles +=
        speculative_wait_cycles - final_speculative_wait_before;
      // No shadow payload is needed after the last authoritative commit.
      for (u32 slot = 0; slot < kPersistentFrontierRobCapacity; ++slot) {
        FrontierRobEntry& entry = frontier_rob[slot];
        if (entry.state ==
            static_cast<u8>(FrontierRequestState::validated)) {
          const bool speculative =
            entry.priority ==
            static_cast<u8>(DirectBatchPriority::speculative);
          if (speculative && entry.transfer_bytes != 0) {
            speculative_wasted_bytes += entry.transfer_bytes;
          }
          if (speculative) {
            ++speculative_stale;
            // Query termination right-censors this record: no future Commit
            // Frontier exists in which it could either promote or become a
            // true turnover miss. Charge its physical bytes to telemetry,
            // but do not train the CTA-persistent controller with an outcome
            // created solely by the configured search budget. Width learning
            // therefore transfers across queries and datasets using only
            // observable frontier utility/pressure, not query length.
          } else {
            ++core_prefetch_stale;
          }
        }
        if (entry.state !=
            static_cast<u8>(FrontierRequestState::init)) {
          entry = {};
        }
      }
      adaptive_frontier::update_issue_width(
        frontier_controller,
        adaptive_frontier::Feedback{
          .promoted = tail_feedback.promoted,
          .retained = tail_feedback.retained,
          .stale = tail_feedback.stale,
          .queue_rejects = tail_feedback.queue_rejects,
          .critical_misses = tail_feedback.core_misses,
          .tail_admitted = tail_feedback.tail_admitted,
          .commit_waves_observed =
            tail_feedback.commit_waves_observed,
          .commit_waves_covered =
            tail_feedback.commit_waves_covered,
      });
      tail_feedback = {};
    }
    __syncthreads();
  } else if (threadIdx.x == 0) {
    final_speculative_drain_failed = 0;
  }
  __syncthreads();
  if (final_speculative_drain_failed != 0) {
    if constexpr (EnableAsfe) {
      // Preserve the same query-slot lifetime rule on this second exceptional
      // exit.  Without the drain, an admitted terminal exact train can write
      // into a later query generation after this completion is observed.
      drain_terminal_exact_cache_prefetch(
        params, descriptor, terminal_exact_cache);
    }
    if (threadIdx.x == 0) {
      completion.status = -EIO;
      completion.result_count = frontier_graph_failure_code(
        params, descriptor, 10u);
      completion.diagnostic = make_query_diagnostic(
        QueryFailureReason::graph_fetch, route_snapshot_retries);
      completion.gpu_cycles = clock64() - query_started_cycles;
      completion.remote_pages = total_remote_reads;
      completion.remote_batches = total_remote_batches;
      completion.graph_read_retries = total_graph_read_retries;
      completion.graph_read_bytes = total_graph_read_bytes;
      completion.graph_live_extent_reads = total_graph_live_extent_reads;
      completion.graph_full_record_reads = total_graph_full_record_reads;
      completion.graph_extent_fallback_reads =
        total_graph_extent_fallback_reads;
      completion.graph_extent_underhint_reads =
        total_graph_extent_underhint_reads;
      completion.graph_extent_hint_promotions =
        total_graph_extent_hint_promotions;
      set_dynamic_graph_completion(completion, dynamic_graph_telemetry);
      set_frontier_completion<EnableAsfe>(
        completion, logical_expansions, critical_graph_reads,
        critical_graph_bytes, speculative_graph_reads,
        speculative_graph_bytes, speculative_arrived,
        speculative_promoted, speculative_stale,
        speculative_wasted_bytes, speculative_queue_rejects,
        issue_epochs, commit_epochs, issue_width_sum,
        issue_width_capacity_sum, commit_width_sum, max_issue_width,
        max_commit_width, critical_rob_hits, critical_misses,
        speculative_wait_cycles,
        rdma_completion_latency_ns, speculative_completion_latency_ns,
        rdma_completion_groups, speculative_completion_groups,
        core_prefetch_bytes, core_prefetch_reads, core_prefetch_arrived,
        core_prefetch_promoted, core_prefetch_stale,
        core_prefetch_queue_rejects, core_prefetch_waves,
        core_ready_waves);
      set_frontier_certificate_completion<EnableAsfe>(
        completion, frontier_reusable_certificates,
        frontier_streamed_candidate_runs, ordered_score_batches,
        ordered_score_candidates, frontier_reusable_prefix_ranks,
        frontier_reusable_full_prefix_certificates,
        frontier_reusable_issued_certificates,
        ooo_bypassed_parents,
        frontier_certificate_rejects);
      completion.graph_rounds = total_graph_rounds;
      set_terminal_exact_cache_completion<EnableAsfe>(
        completion, terminal_exact_cache);
      set_query_trace_completion(params, query_slot, completion);
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }

  const bool terminal_beam_already_sorted =
    params.beam_merge_policy ==
      static_cast<u32>(BeamMergePolicy::stable_run);
  if (terminal_beam_already_sorted) {
    // Stable-Run publishes a complete distance-sorted authoritative Beam on
    // every round. Avoid copying it to merge scratch, sorting it again, and
    // then serially copying 128 entries from lane zero. Exactification still
    // receives the identical ordered handle set; merge_* remains its private
    // output workspace below.
    if (threadIdx.x == 0) {
      rerank_count = min(beam_count, kPersistentMaxExact);
    }
    __syncthreads();
    for (u32 index = threadIdx.x; index < rerank_count;
         index += blockDim.x) {
      rerank_handles[index] = beam_handles[index];
      rerank_ids[index] = UINT32_MAX;
      rerank_distances[index] = beam_distances[index];
    }
  } else {
    for (u32 index = threadIdx.x; index < beam_count;
         index += blockDim.x) {
      merge_handles[index] = beam_handles[index];
      merge_distances[index] = beam_distances[index];
      merge_expanded[index] = 0;
    }
    __syncthreads();
    sort_candidates(
      merge_handles, nullptr, merge_distances, merge_expanded, beam_count);
    if (threadIdx.x == 0) {
      rerank_count = min(beam_count, kPersistentMaxExact);
      for (u32 index = 0; index < rerank_count; ++index) {
        rerank_handles[index] = merge_handles[index];
        rerank_ids[index] = UINT32_MAX;
        rerank_distances[index] = merge_distances[index];
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // Remote fixed-record headers are the source of truth for delete/upsert
    // visibility; exactify the full traversal Beam so tombstoned prefix
    // candidates can be replaced instead of returning fewer than k.
    phase_started_cycles = clock64();
  }
  __syncthreads();
  bool exact_fetch_succeeded = false;
  if constexpr (EnableAsfe) {
    exact_fetch_succeeded = exactify_into_beam_with_terminal_cache(
      params, descriptor, query, rerank_handles, rerank_ids,
      rerank_distances, rerank_count, beam_handles, beam_ids,
      beam_distances, beam_expanded, beam_count, &total_exact_reads,
      &completion.exact_snapshot_train_batches,
      &completion.exact_snapshot_train_fallbacks,
      min(kPersistentMaxExact,
          max(params.final_rerank_width,
              static_cast<u32>(descriptor.k))),
      true, merge_handles, merge_ids, merge_distances, merge_expanded,
      terminal_exact_cache);
  } else {
    exact_fetch_succeeded = exactify_into_beam(
      params, descriptor, query, rerank_handles, rerank_ids,
      rerank_distances, rerank_count, beam_handles, beam_ids,
      beam_distances, beam_expanded, beam_count, &total_exact_reads,
      &completion.exact_snapshot_train_batches,
      &completion.exact_snapshot_train_fallbacks,
      min(kPersistentMaxExact,
          max(params.final_rerank_width,
              static_cast<u32>(descriptor.k))),
      true, merge_handles, merge_ids, merge_distances, merge_expanded);
  }
  if (threadIdx.x == 0) {
    exact_phase_cycles += clock64() - phase_started_cycles;
  }
  __syncthreads();
  if constexpr (EnableAsfe) {
    // Completion publication is the query-slot reuse boundary. Keep this
    // final idempotent guard even though the terminal-aware exactifier drains
    // its train internally: it makes both the success and exact-error exits
    // safe if a future fallback returns before consuming the cache.
    drain_terminal_exact_cache_prefetch(
      params, descriptor, terminal_exact_cache);
  }

  // Transport failure is terminal even on the first route attempt; only a
  // completed-but-empty exact snapshot retains the legacy one-time reroute.
  if (!exact_fetch_succeeded || beam_count == 0) {
    if (exact_rerank_should_retry_route(
          exact_fetch_succeeded, route_attempt)) {
      for (u32 index = threadIdx.x; index < params.visited_capacity;
           index += blockDim.x) {
        visited[index] = kInvalidDeviceHandle;
      }
      if (threadIdx.x == 0) device_ring_relax(64);
      __syncthreads();
      continue;
    }
    if (threadIdx.x == 0) {
      // Empty exact rerank is a fail-stop invariant, not a normal search
      // outcome.  Inspect the already-fetched snapshots only on that path so
      // successful queries pay no extra scan.  The packed counts and first
      // rejected record reuse failure-only completion fields and distinguish
      // an empty traversal Beam, a malformed handle, and a snapshot
      // visibility failure without enlarging CompletionDescriptor.
      u32 exact_resolved = 0;
      u32 exact_equal_headers = 0;
      u32 exact_visible = 0;
      u64 first_exact_handle = kInvalidDeviceHandle;
      u64 first_exact_before = 0;
      u64 first_exact_after = 0;
      u32 first_exact_stored_incarnation = 0;
      u32 first_exact_expected_incarnation = 0;
      if (exact_fetch_succeeded) {
        for (u32 index = 0; index < rerank_count; ++index) {
          const u64 handle = rerank_handles[index];
          u64 raw = 0;
          u64 graph_offset = 0;
          u32 shard = 0;
          const bool resolved =
            resolve_handle(params, handle, raw, shard, graph_offset);
          exact_resolved += resolved ? 1u : 0u;
          const u8* record =
            params.exact_records +
            (static_cast<size_t>(descriptor.query_slot) *
               params.exact_width + index) *
              params.node_record_stride;
          const u64 before = *reinterpret_cast<const u64*>(record);
          const u64 after = *reinterpret_cast<const u64*>(
            record + params.node_record_bytes);
          exact_equal_headers += before == after ? 1u : 0u;
          const bool visible =
            resolved && exact_record_visible(params, record, handle);
          exact_visible += visible ? 1u : 0u;
          if (!visible &&
              first_exact_handle == kInvalidDeviceHandle) {
            first_exact_handle = handle;
            first_exact_before = before;
            first_exact_after = after;
            first_exact_stored_incarnation =
              *reinterpret_cast<const u32*>(
                record + params.node_incarnation_offset);
            first_exact_expected_incarnation =
              remote_incarnation(handle);
          }
        }
      }
      completion.status = -EIO;
      completion.diagnostic = make_query_diagnostic(
        exact_fetch_succeeded
          ? QueryFailureReason::exact_rerank_empty
          : QueryFailureReason::exact_fetch,
        route_snapshot_retries);
      completion.gpu_cycles = clock64() - query_started_cycles;
      completion.graph_cycles = graph_phase_cycles;
      completion.score_cycles = score_phase_cycles;
      completion.beam_cycles = beam_phase_cycles;
      completion.exact_cycles = exact_phase_cycles;
      completion.beam_selection_cycles = beam_selection_cycles;
      completion.rdma_issue_cycles = rdma_issue_cycles;
      completion.frontier_preview_cycles = frontier_preview_cycles;
      completion.frontier_prepare_cycles = frontier_prepare_cycles;
      completion.frontier_enqueue_cycles = frontier_enqueue_cycles;
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
      set_dynamic_graph_completion(completion, dynamic_graph_telemetry);
      set_frontier_completion<EnableAsfe>(
        completion, logical_expansions, critical_graph_reads,
        critical_graph_bytes, speculative_graph_reads,
        speculative_graph_bytes, speculative_arrived,
        speculative_promoted, speculative_stale,
        speculative_wasted_bytes, speculative_queue_rejects,
        issue_epochs, commit_epochs, issue_width_sum,
        issue_width_capacity_sum, commit_width_sum, max_issue_width,
        max_commit_width, critical_rob_hits, critical_misses,
        speculative_wait_cycles,
        rdma_completion_latency_ns, speculative_completion_latency_ns,
        rdma_completion_groups, speculative_completion_groups,
        core_prefetch_bytes, core_prefetch_reads, core_prefetch_arrived,
        core_prefetch_promoted, core_prefetch_stale,
        core_prefetch_queue_rejects, core_prefetch_waves,
        core_ready_waves);
      set_frontier_certificate_completion<EnableAsfe>(
        completion, frontier_reusable_certificates,
        frontier_streamed_candidate_runs, ordered_score_batches,
        ordered_score_candidates, frontier_reusable_prefix_ranks,
        frontier_reusable_full_prefix_certificates,
        frontier_reusable_issued_certificates,
        ooo_bypassed_parents,
        frontier_certificate_rejects);
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
        dynamic_code_cache_publish_successes,
        dynamic_code_cache_first_occupancies,
        dynamic_code_cache_publish_races,
        dynamic_code_cache_lookup_probe_exhaustions,
        dynamic_code_cache_publish_probe_exhaustions,
          dynamic_code_cache_lookup_probes,
          dynamic_code_cache_max_lookup_probes);
      set_beam_merge_completion(completion, beam_merge_breakdown);
      set_terminal_exact_cache_completion<EnableAsfe>(
        completion, terminal_exact_cache);
      if (exact_fetch_succeeded) {
        completion.result_count =
          (rerank_count & 0xffu) |
          ((exact_resolved & 0xffu) << 8) |
          ((exact_equal_headers & 0xffu) << 16) |
          ((exact_visible & 0xffu) << 24);
        completion.frontier_telemetry_reserved0 =
          static_cast<u32>(first_exact_handle);
        completion.frontier_telemetry_reserved1 =
          static_cast<u32>(first_exact_handle >> 32);
        completion.rdma_completion_latency_ns = first_exact_before;
        completion.speculative_completion_latency_ns = first_exact_after;
        completion.issue_width_sum =
          (static_cast<u64>(first_exact_expected_incarnation) << 32) |
          first_exact_stored_incarnation;
        completion.commit_width_sum = route_failure_reason;
      }
      set_query_trace_completion(params, query_slot, completion);
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }
  // exactify_into_beam() has already sorted the complete exact candidate run
  // and copied its valid prefix back to Beam.  Re-sorting that byte-identical
  // run here duplicated a full block-wide radix pass on every successful
  // query and cannot change either result order or visibility filtering.
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
    completion.frontier_preview_cycles = frontier_preview_cycles;
    completion.frontier_prepare_cycles = frontier_prepare_cycles;
    completion.frontier_enqueue_cycles = frontier_enqueue_cycles;
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
    set_dynamic_graph_completion(completion, dynamic_graph_telemetry);
    set_frontier_completion<EnableAsfe>(
      completion, logical_expansions, critical_graph_reads,
      critical_graph_bytes, speculative_graph_reads,
      speculative_graph_bytes, speculative_arrived,
      speculative_promoted, speculative_stale,
      speculative_wasted_bytes, speculative_queue_rejects,
      issue_epochs, commit_epochs, issue_width_sum,
      issue_width_capacity_sum, commit_width_sum, max_issue_width,
      max_commit_width, critical_rob_hits, critical_misses,
      speculative_wait_cycles,
      rdma_completion_latency_ns, speculative_completion_latency_ns,
      rdma_completion_groups, speculative_completion_groups,
      core_prefetch_bytes, core_prefetch_reads, core_prefetch_arrived,
      core_prefetch_promoted, core_prefetch_stale,
      core_prefetch_queue_rejects, core_prefetch_waves,
      core_ready_waves);
    set_frontier_certificate_completion<EnableAsfe>(
      completion, frontier_reusable_certificates,
      frontier_streamed_candidate_runs, ordered_score_batches,
      ordered_score_candidates, frontier_reusable_prefix_ranks,
      frontier_reusable_full_prefix_certificates,
      frontier_reusable_issued_certificates,
      ooo_bypassed_parents,
      frontier_certificate_rejects);
    set_terminal_exact_cache_completion<EnableAsfe>(
      completion, terminal_exact_cache);
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
      dynamic_code_cache_publish_successes,
      dynamic_code_cache_first_occupancies,
      dynamic_code_cache_publish_races,
      dynamic_code_cache_lookup_probe_exhaustions,
      dynamic_code_cache_publish_probe_exhaustions,
      dynamic_code_cache_lookup_probes,
      dynamic_code_cache_max_lookup_probes);
    set_beam_merge_completion(completion, beam_merge_breakdown);
    set_query_trace_completion(params, query_slot, completion);
    device_ring_push(params.completions, completion);
  }
  __syncthreads();
  return;
  }
}

}  // namespace gpu_search::persistent_kernel_detail
