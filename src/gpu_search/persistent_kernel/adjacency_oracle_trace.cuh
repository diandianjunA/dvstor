#pragma once

#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace gpu_search::persistent_kernel_detail {

// This file implements a deliberately expensive, sampled-only motivation
// probe.  It evaluates information that is unavailable on the production
// graph-read path after the full record has already arrived.  None of these
// helpers may be used to make a search decision.

__device__ __forceinline__ u32 adjacency_oracle_group_size(u32 index) {
  constexpr u32 sizes[kQueryAdjacencyOracleGroupCount] = {4, 8, 16, 32};
  return sizes[index];
}

__device__ __forceinline__ u32 adjacency_oracle_prefix_size(u32 index) {
  constexpr u32 sizes[kQueryAdjacencyOraclePrefixCount] = {8, 16, 32, 48, 64};
  return sizes[index];
}

__device__ __forceinline__ u32 adjacency_oracle_run_count(u32 mask) {
  return __popc(mask & ~(mask << 1u));
}

__device__ __forceinline__ bool adjacency_oracle_finite_candidate(
    u64 handle, f32 distance) {
  return handle != kInvalidDeviceHandle && isfinite(distance) &&
    distance != FLT_MAX;
}

__device__ __noinline__ f32 pq_reconstruction_edge_distance(
    const PersistentKernelParams& params, const u8* lhs_code,
    const u8* rhs_code) {
  f32 squared_distance = 0.0f;
  for (u32 subquantizer = 0; subquantizer < params.pq_subquantizers;
       ++subquantizer) {
    const f32* lhs = params.pq_centroids +
      (static_cast<size_t>(subquantizer) * 256u +
       lhs_code[subquantizer]) * params.pq_subvector_dim;
    const f32* rhs = params.pq_centroids +
      (static_cast<size_t>(subquantizer) * 256u +
       rhs_code[subquantizer]) * params.pq_subvector_dim;
    for (u32 dimension = 0; dimension < params.pq_subvector_dim;
         ++dimension) {
      const f32 difference = lhs[dimension] - rhs[dimension];
      squared_distance += difference * difference;
    }
  }
  return sqrtf(max(0.0f, squared_distance));
}

__device__ __forceinline__ f32 annulus_lower_bound_squared(
    f32 minimum_radius, f32 maximum_radius, f32 query_parent_radius) {
  f32 separation = 0.0f;
  if (query_parent_radius < minimum_radius) {
    separation = minimum_radius - query_parent_radius;
  } else if (query_parent_radius > maximum_radius) {
    separation = query_parent_radius - maximum_radius;
  }
  return separation * separation;
}

__device__ __forceinline__ f32 suffix_lower_bound_squared(
    f32 minimum_radius, f32 query_parent_radius) {
  const f32 separation =
    minimum_radius > query_parent_radius
      ? minimum_radius - query_parent_radius : 0.0f;
  return separation * separation;
}

__device__ __noinline__ void begin_adjacency_oracle_trace(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    bool trace_enabled,
    u32 search_round,
    u32 chunk_begin,
    const u64* selected_handles,
    u32 parent_count,
    const u32* neighbor_counts,
    const u32* neighbor_offsets,
    const u64* navigation_handles,
    f32* navigation_distances,
    const f32* query_lut,
    const f32* beam_distances,
    u32 beam_count,
    u32 beam_capacity,
    u32& trace_event_index) {
  if (threadIdx.x == 0) {
    trace_event_index = UINT32_MAX;
    if (trace_enabled &&
        params.query_adjacency_oracle_trace_headers != nullptr &&
        params.query_adjacency_oracle_trace_events != nullptr) {
      QueryAdjacencyOracleTraceHeader& header =
        params.query_adjacency_oracle_trace_headers[descriptor.query_slot];
      const u32 index = header.event_count++;
      if (index < params.query_rdma_trace_events_per_query) {
        trace_event_index = index;
        QueryAdjacencyOracleTraceEvent& event =
          params.query_adjacency_oracle_trace_events[
            static_cast<size_t>(descriptor.query_slot) *
              params.query_rdma_trace_events_per_query + index];
        event = {};
        event.request_id = descriptor.request_id;
        event.search_round = search_round;
        event.chunk_begin = chunk_begin;
        event.parent_count = parent_count;
        event.edge_count = neighbor_offsets[parent_count];
        event.beam_count_before = beam_count;
        event.beam_capacity = beam_capacity;
        const f32 cutoff =
          beam_count >= beam_capacity && beam_capacity != 0
            ? beam_distances[beam_capacity - 1u] : FLT_MAX;
        event.cutoff_distance_bits = __float_as_uint(cutoff);
        event.minimum_interval_safety_margin = FLT_MAX;
        for (u32 parent = 0;
             parent < kQueryBeamTurnoverTraceWidth; ++parent) {
          event.selected_best_child_rank[parent] = UINT32_MAX;
          event.selected_handles[parent] =
            parent < parent_count
              ? selected_handles[chunk_begin + parent]
              : kInvalidDeviceHandle;
          event.frontier_handles[parent] = kInvalidDeviceHandle;
          event.frontier_distance_bits[parent] = __float_as_uint(FLT_MAX);
        }
      } else {
        header.overflow = 1;
      }
    }
  }
  __syncthreads();
  if (trace_event_index == UINT32_MAX) return;

  const u32 edge_count = neighbor_offsets[parent_count];
  // A pre-transfer certificate cannot consult visited, so shadow-score every
  // static edge.  Dynamic edges would need another remote code read and are
  // conservatively treated as requiring transfer.
  for (u32 edge = threadIdx.x; edge < edge_count; edge += blockDim.x) {
    const u64 handle = navigation_handles[edge];
    u32 ordinal = 0;
    if (handle == kInvalidDeviceHandle) {
      navigation_distances[edge] = -FLT_MAX;
    } else if (static_ordinal_from_raw(params, handle, ordinal) &&
               ordinal < params.num_nodes) {
      navigation_distances[edge] = approximate_entry(
        params, query_lut,
        params.pq_codes +
          static_cast<size_t>(ordinal) * params.pq_code_bytes);
    } else {
      navigation_distances[edge] = -FLT_MAX;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    QueryAdjacencyOracleTraceEvent& event =
      params.query_adjacency_oracle_trace_events[
        static_cast<size_t>(descriptor.query_slot) *
          params.query_rdma_trace_events_per_query + trace_event_index];
    const f32 cutoff = __uint_as_float(event.cutoff_distance_bits);
    const bool beam_full =
      event.beam_capacity != 0 &&
      event.beam_count_before >= event.beam_capacity;

    for (u32 edge = 0; edge < edge_count; ++edge) {
      const u64 handle = navigation_handles[edge];
      u32 ordinal = 0;
      if (handle == kInvalidDeviceHandle) {
        ++event.invalid_decoded_count;
      } else if (!static_ordinal_from_raw(params, handle, ordinal) ||
                 ordinal >= params.num_nodes) {
        ++event.dynamic_edge_count;
      }
    }

    for (u32 group_lane = 0;
         group_lane < kQueryAdjacencyOracleGroupCount; ++group_lane) {
      const u32 group_size = adjacency_oracle_group_size(group_lane);
      for (u32 parent = 0; parent < parent_count; ++parent) {
        const u32 count = neighbor_counts[parent];
        const u32 begin = neighbor_offsets[parent];
        const u32 group_count =
          count == 0 ? 0 : (count + group_size - 1u) / group_size;
        event.total_groups[group_lane] += group_count;

        u32 parent_ordinal = 0;
        const bool parent_static =
          static_ordinal_from_raw(
            params, selected_handles[chunk_begin + parent],
            parent_ordinal) &&
          parent_ordinal < params.num_nodes;
        const u8* parent_code = parent_static
          ? params.pq_codes +
              static_cast<size_t>(parent_ordinal) * params.pq_code_bytes
          : nullptr;
        const f32 query_parent_radius = parent_static
          ? sqrtf(max(0.0f, approximate_entry(
              params, query_lut, parent_code)))
          : 0.0f;
        u32 perfect_mask = 0;
        u32 interval_mask = 0;

        for (u32 group = 0; group < group_count; ++group) {
          const u32 first = begin + group * group_size;
          const u32 limit = min(
            begin + count, first + group_size);
          bool perfect_needed = !beam_full;
          bool interval_unknown = !parent_static;
          f32 actual_minimum = FLT_MAX;
          f32 minimum_radius = FLT_MAX;
          f32 maximum_radius = 0.0f;
          for (u32 edge = first; edge < limit; ++edge) {
            const u64 child_handle = navigation_handles[edge];
            u32 child_ordinal = 0;
            const bool child_static =
              static_ordinal_from_raw(
                params, child_handle, child_ordinal) &&
              child_ordinal < params.num_nodes;
            if (!child_static) {
              perfect_needed = true;
              interval_unknown = true;
              continue;
            }
            const f32 child_distance = navigation_distances[edge];
            if (!isfinite(child_distance) ||
                child_distance == FLT_MAX ||
                child_distance <= cutoff) {
              perfect_needed = true;
            }
            actual_minimum = min(actual_minimum, child_distance);
            if (parent_static) {
              const u8* child_code = params.pq_codes +
                static_cast<size_t>(child_ordinal) * params.pq_code_bytes;
              const f32 edge_radius = pq_reconstruction_edge_distance(
                params, parent_code, child_code);
              minimum_radius = min(minimum_radius, edge_radius);
              maximum_radius = max(maximum_radius, edge_radius);
            }
          }
          bool interval_needed = !beam_full || interval_unknown;
          if (!interval_needed && minimum_radius != FLT_MAX) {
            const f32 lower_bound = annulus_lower_bound_squared(
              minimum_radius, maximum_radius, query_parent_radius);
            interval_needed = lower_bound <= cutoff;
            if (actual_minimum != FLT_MAX) {
              const f32 margin = actual_minimum - lower_bound;
              event.minimum_interval_safety_margin =
                min(event.minimum_interval_safety_margin, margin);
              if (margin < 0.0f) ++event.interval_lb_violation_count;
            }
          }
          if (perfect_needed) perfect_mask |= 1u << group;
          if (interval_needed) interval_mask |= 1u << group;
        }
        event.certificate_needed_groups[group_lane] += __popc(perfect_mask);
        event.certificate_needed_runs[group_lane] +=
          adjacency_oracle_run_count(perfect_mask);
        event.certificate_first_group_needed_parents[group_lane] +=
          (perfect_mask & 1u) != 0 ? 1u : 0u;
        event.interval_needed_groups[group_lane] += __popc(interval_mask);
        event.interval_needed_runs[group_lane] +=
          adjacency_oracle_run_count(interval_mask);
        event.interval_first_group_needed_parents[group_lane] +=
          (interval_mask & 1u) != 0 ? 1u : 0u;
      }
    }

    for (u32 prefix_lane = 0;
         prefix_lane < kQueryAdjacencyOraclePrefixCount; ++prefix_lane) {
      const u32 prefix_size = adjacency_oracle_prefix_size(prefix_lane);
      for (u32 parent = 0; parent < parent_count; ++parent) {
        const u32 count = neighbor_counts[parent];
        if (count <= prefix_size) continue;
        ++event.parents_with_tail[prefix_lane];
        const u32 tail_edge_count = count - prefix_size;
        event.total_tail_edges[prefix_lane] += tail_edge_count;
        const u32 begin = neighbor_offsets[parent] + prefix_size;
        const u32 limit = neighbor_offsets[parent] + count;

        u32 parent_ordinal = 0;
        const bool parent_static =
          static_ordinal_from_raw(
            params, selected_handles[chunk_begin + parent],
            parent_ordinal) &&
          parent_ordinal < params.num_nodes;
        const u8* parent_code = parent_static
          ? params.pq_codes +
              static_cast<size_t>(parent_ordinal) * params.pq_code_bytes
          : nullptr;
        const f32 query_parent_radius = parent_static
          ? sqrtf(max(0.0f, approximate_entry(
              params, query_lut, parent_code)))
          : 0.0f;
        bool perfect_tail_needed = !beam_full;
        bool suffix_unknown = !parent_static;
        f32 actual_minimum = FLT_MAX;
        f32 minimum_radius = FLT_MAX;
        for (u32 edge = begin; edge < limit; ++edge) {
          const u64 child_handle = navigation_handles[edge];
          u32 child_ordinal = 0;
          const bool child_static =
            static_ordinal_from_raw(
              params, child_handle, child_ordinal) &&
            child_ordinal < params.num_nodes;
          if (!child_static) {
            perfect_tail_needed = true;
            suffix_unknown = true;
            continue;
          }
          const f32 child_distance = navigation_distances[edge];
          if (!isfinite(child_distance) ||
              child_distance == FLT_MAX ||
              child_distance <= cutoff) {
            perfect_tail_needed = true;
          }
          actual_minimum = min(actual_minimum, child_distance);
          if (parent_static) {
            const u8* child_code = params.pq_codes +
              static_cast<size_t>(child_ordinal) * params.pq_code_bytes;
            minimum_radius = min(
              minimum_radius,
              pq_reconstruction_edge_distance(
                params, parent_code, child_code));
          }
        }
        bool suffix_tail_needed = !beam_full || suffix_unknown;
        if (!suffix_tail_needed && minimum_radius != FLT_MAX) {
          const f32 lower_bound = suffix_lower_bound_squared(
            minimum_radius, query_parent_radius);
          suffix_tail_needed = lower_bound <= cutoff;
          if (actual_minimum != FLT_MAX) {
            const f32 margin = actual_minimum - lower_bound;
            event.minimum_interval_safety_margin =
              min(event.minimum_interval_safety_margin, margin);
            if (margin < 0.0f) ++event.interval_lb_violation_count;
          }
        }
        if (perfect_tail_needed) {
          ++event.perfect_tail_needed_parents[prefix_lane];
          event.perfect_tail_needed_edges[prefix_lane] += tail_edge_count;
        }
        if (suffix_tail_needed) {
          ++event.suffix_interval_tail_needed_parents[prefix_lane];
          event.suffix_interval_tail_needed_edges[prefix_lane] +=
            tail_edge_count;
        }
      }
    }
  }
  __syncthreads();
}

__device__ __noinline__ void record_adjacency_oracle_post_visited(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    u32 trace_event_index,
    u32 parent_count,
    const u32* neighbor_counts,
    const u32* neighbor_offsets,
    const u64* navigation_handles,
    const f32* navigation_distances) {
  if (trace_event_index == UINT32_MAX) return;
  if (threadIdx.x == 0) {
    QueryAdjacencyOracleTraceEvent& event =
      params.query_adjacency_oracle_trace_events[
        static_cast<size_t>(descriptor.query_slot) *
          params.query_rdma_trace_events_per_query + trace_event_index];
    const f32 cutoff = __uint_as_float(event.cutoff_distance_bits);
    const bool beam_full =
      event.beam_capacity != 0 &&
      event.beam_count_before >= event.beam_capacity;
    for (u32 edge = 0; edge < event.edge_count; ++edge) {
      if (navigation_handles[edge] != kInvalidDeviceHandle) {
        ++event.visited_survivor_count;
      }
      if (adjacency_oracle_finite_candidate(
            navigation_handles[edge], navigation_distances[edge])) {
        ++event.finite_scored_count;
      }
    }
    for (u32 group_lane = 0;
         group_lane < kQueryAdjacencyOracleGroupCount; ++group_lane) {
      const u32 group_size = adjacency_oracle_group_size(group_lane);
      for (u32 parent = 0; parent < parent_count; ++parent) {
        const u32 count = neighbor_counts[parent];
        const u32 begin = neighbor_offsets[parent];
        const u32 group_count =
          count == 0 ? 0 : (count + group_size - 1u) / group_size;
        for (u32 group = 0; group < group_count; ++group) {
          const u32 first = begin + group * group_size;
          const u32 limit = min(begin + count, first + group_size);
          bool needed = false;
          for (u32 edge = first; edge < limit; ++edge) {
            if (!adjacency_oracle_finite_candidate(
                  navigation_handles[edge], navigation_distances[edge])) {
              continue;
            }
            if (!beam_full || navigation_distances[edge] <= cutoff) {
              needed = true;
              break;
            }
          }
          if (needed) ++event.post_visited_needed_groups[group_lane];
        }
      }
    }
    for (u32 prefix_lane = 0;
         prefix_lane < kQueryAdjacencyOraclePrefixCount; ++prefix_lane) {
      const u32 prefix_size = adjacency_oracle_prefix_size(prefix_lane);
      for (u32 parent = 0; parent < parent_count; ++parent) {
        const u32 count = neighbor_counts[parent];
        if (count <= prefix_size) continue;
        const u32 begin = neighbor_offsets[parent] + prefix_size;
        const u32 limit = neighbor_offsets[parent] + count;
        bool needed = false;
        for (u32 edge = begin; edge < limit; ++edge) {
          if (!adjacency_oracle_finite_candidate(
                navigation_handles[edge], navigation_distances[edge])) {
            continue;
          }
          if (!beam_full || navigation_distances[edge] <= cutoff) {
            needed = true;
            break;
          }
        }
        if (needed) {
          ++event.post_visited_tail_needed_parents[prefix_lane];
        }
      }
    }
  }
  __syncthreads();
}

__device__ __noinline__ void finish_adjacency_oracle_trace(
    const PersistentKernelParams& params,
    const QueryDescriptor& descriptor,
    u32 trace_event_index,
    u32 parent_count,
    const u32* neighbor_counts,
    const u32* neighbor_offsets,
    const u64* navigation_handles,
    const f32* navigation_distances,
    const u64* beam_handles,
    const f32* beam_distances,
    const u8* beam_expanded,
    u32 beam_count,
    u32 new_candidates_in_beam,
    u64 round_graph_cycles,
    u64 round_score_cycles,
    u64 round_beam_cycles) {
  if (trace_event_index == UINT32_MAX) return;
  if (threadIdx.x == 0) {
    QueryAdjacencyOracleTraceEvent& event =
      params.query_adjacency_oracle_trace_events[
        static_cast<size_t>(descriptor.query_slot) *
          params.query_rdma_trace_events_per_query + trace_event_index];
    event.new_candidates_in_beam = new_candidates_in_beam;
    event.round_graph_cycles = round_graph_cycles;
    event.round_score_cycles = round_score_cycles;
    event.round_beam_cycles = round_beam_cycles;
    for (u32 parent = 0;
         parent < min(parent_count, kQueryBeamTurnoverTraceWidth);
         ++parent) {
      const u32 begin = neighbor_offsets[parent];
      const u32 limit = begin + neighbor_counts[parent];
      u32 surviving_children = 0;
      u32 best_rank = UINT32_MAX;
      for (u32 edge = begin; edge < limit; ++edge) {
        if (!adjacency_oracle_finite_candidate(
              navigation_handles[edge], navigation_distances[edge])) {
          continue;
        }
        for (u32 beam = 0; beam < beam_count; ++beam) {
          if (beam_handles[beam] != navigation_handles[edge]) continue;
          ++surviving_children;
          best_rank = min(best_rank, beam);
          break;
        }
      }
      event.selected_child_in_beam_count[parent] = surviving_children;
      event.selected_best_child_rank[parent] = best_rank;
      if (surviving_children != 0) {
        event.selected_productive_mask |= 1u << parent;
      }
    }
    for (u32 beam = 0;
         beam < beam_count &&
         event.frontier_count < kQueryBeamTurnoverTraceWidth;
         ++beam) {
      if (beam_expanded[beam] != 0 ||
          !adjacency_oracle_finite_candidate(
            beam_handles[beam], beam_distances[beam])) {
        continue;
      }
      const u32 frontier = event.frontier_count++;
      event.frontier_handles[frontier] = beam_handles[beam];
      event.frontier_distance_bits[frontier] =
        __float_as_uint(beam_distances[beam]);
      bool is_new = false;
      for (u32 edge = 0; edge < event.edge_count; ++edge) {
        if (navigation_handles[edge] == beam_handles[beam]) {
          is_new = true;
          break;
        }
      }
      if (is_new) event.frontier_new_mask |= 1u << frontier;
    }
    for (u32 group_lane = 0;
         group_lane < kQueryAdjacencyOracleGroupCount; ++group_lane) {
      const u32 group_size = adjacency_oracle_group_size(group_lane);
      for (u32 parent = 0; parent < parent_count; ++parent) {
        const u32 count = neighbor_counts[parent];
        const u32 begin = neighbor_offsets[parent];
        const u32 group_count =
          count == 0 ? 0 : (count + group_size - 1u) / group_size;
        for (u32 group = 0; group < group_count; ++group) {
          const u32 first = begin + group * group_size;
          const u32 limit = min(begin + count, first + group_size);
          bool needed = false;
          for (u32 edge = first; edge < limit && !needed; ++edge) {
            if (!adjacency_oracle_finite_candidate(
                  navigation_handles[edge], navigation_distances[edge])) {
              continue;
            }
            for (u32 beam = 0; beam < beam_count; ++beam) {
              if (beam_handles[beam] == navigation_handles[edge]) {
                needed = true;
                break;
              }
            }
          }
          if (needed) ++event.final_beam_needed_groups[group_lane];
        }
      }
    }
    for (u32 prefix_lane = 0;
         prefix_lane < kQueryAdjacencyOraclePrefixCount; ++prefix_lane) {
      const u32 prefix_size = adjacency_oracle_prefix_size(prefix_lane);
      for (u32 parent = 0; parent < parent_count; ++parent) {
        const u32 count = neighbor_counts[parent];
        if (count <= prefix_size) continue;
        const u32 begin = neighbor_offsets[parent] + prefix_size;
        const u32 limit = neighbor_offsets[parent] + count;
        bool needed = false;
        for (u32 edge = begin; edge < limit && !needed; ++edge) {
          if (!adjacency_oracle_finite_candidate(
                navigation_handles[edge], navigation_distances[edge])) {
            continue;
          }
          for (u32 beam = 0; beam < beam_count; ++beam) {
            if (beam_handles[beam] == navigation_handles[edge]) {
              needed = true;
              break;
            }
          }
        }
        if (needed) {
          ++event.final_beam_tail_needed_parents[prefix_lane];
        }
      }
    }
  }
  __syncthreads();
}

}  // namespace gpu_search::persistent_kernel_detail
