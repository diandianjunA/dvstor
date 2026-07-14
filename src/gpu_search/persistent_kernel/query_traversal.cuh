__device__ u64 decode_compact_raw(const u8* source, u32 shard_bits) {
  u64 packed = 0;
  for (u32 byte = 0; byte < 5; ++byte) packed |= static_cast<u64>(source[byte]) << (8 * byte);
  if (packed == ((1ull << 40) - 1ull) || shard_bits >= 16) return 0;
  const u32 offset_bits = 40 - shard_bits;
  const u64 offset_mask = (1ull << offset_bits) - 1ull;
  const u32 shard = static_cast<u32>(packed >> offset_bits);
  const u64 offset = (packed & offset_mask) * 8;
  return (static_cast<u64>(shard) << 48) | offset;
}

__device__ void add_delta_candidates(const PersistentKernelParams& params,
                                     const QueryDescriptor& descriptor,
                                     const f32* query, const f32* query_lut,
                                     u32* beam_handles,
                                     u32* beam_ids, f32* beam_distances,
                                     u8* beam_expanded, u32& beam_count,
                                     u32 beam_capacity,
                                     const u32* selected_anchors,
                                     u32 selected_anchor_count) {
  __shared__ u32 delta_count_snapshot;
  if (threadIdx.x == 0) {
    delta_count_snapshot = min(load_cg(params.delta_count), params.delta_capacity);
  }
  __syncthreads();
  const u32 count = delta_count_snapshot;
  if (count == 0) return;
  __shared__ u32 candidate_handles[256];
  __shared__ u32 candidate_slots[256];
  __shared__ f32 candidate_distances[256];
  u32 local_slot = UINT32_MAX;
  f32 local_approximation = FLT_MAX;

  if (params.anchor_count == 0 || count <= 4096) {
    for (u32 slot = threadIdx.x; slot < count; slot += blockDim.x) {
      const DeviceDeltaRecord& record = params.delta_records[slot];
      if (!delta_visible(record, descriptor.snapshot_epoch)) continue;
      const f32 approximation = approximate_entry(
        params, query_lut,
        params.delta_pq_codes + static_cast<size_t>(slot) * params.pq_code_bytes);
      if (approximation < local_approximation) {
        local_approximation = approximation;
        local_slot = slot;
      }
    }
  } else if (selected_anchor_count != 0) {
      const u32 probe = threadIdx.x % selected_anchor_count;
      const u32 partition = threadIdx.x / selected_anchor_count;
      const u32 partitions =
        (blockDim.x - 1 - probe) / selected_anchor_count + 1;
      const u32 selected_anchor = selected_anchors[probe];
      u32 slot = selected_anchor == UINT32_MAX
        ? UINT32_MAX : load_cg(params.delta_bucket_heads + selected_anchor);
      u32 traversed = 0;
      u32 position = 0;
      while (slot != UINT32_MAX && slot < count && traversed++ < count) {
        const DeviceDeltaRecord& record = params.delta_records[slot];
        if (position % partitions == partition &&
            delta_visible(record, descriptor.snapshot_epoch)) {
          const f32 approximation = approximate_entry(
            params, query_lut,
            params.delta_pq_codes + static_cast<size_t>(slot) * params.pq_code_bytes);
          if (approximation < local_approximation) {
            local_approximation = approximation;
            local_slot = slot;
          }
        }
        slot = load_cg(params.delta_next + slot);
        ++position;
      }
  }
  candidate_slots[threadIdx.x] = local_slot;
  candidate_handles[threadIdx.x] = local_slot == UINT32_MAX
    ? UINT32_MAX
    : handle_from_raw(params, params.delta_records[local_slot].remote_node);
  candidate_distances[threadIdx.x] = local_slot == UINT32_MAX
    ? FLT_MAX
    : exact_storage_distance(params, query,
        params.delta_vectors + static_cast<size_t>(local_slot) * params.vector_bytes);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (u32 index = 0; index < min(blockDim.x, 256u); ++index) {
      const u32 handle = candidate_handles[index];
      if (handle == UINT32_MAX) continue;
      bool duplicate = false;
      for (u32 beam = 0; beam < beam_count; ++beam) {
        if (beam_handles[beam] == handle) duplicate = true;
      }
      if (!duplicate) {
        const u32 slot = candidate_slots[index];
        beam_insert(beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
                    beam_capacity, handle, params.delta_records[slot].id,
                    candidate_distances[index]);
      }
    }
  }
  __syncthreads();
}

__device__ void process_query(const PersistentKernelParams& params,
                              const QueryDescriptor& descriptor) {
  const u32 query_slot = descriptor.query_slot;
  __shared__ u64 query_started_cycles;
  if (threadIdx.x == 0) query_started_cycles = clock64();
  __syncthreads();
  CompletionDescriptor completion{
    .request_id = descriptor.request_id,
    .snapshot_epoch = descriptor.snapshot_epoch,
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
    query_lut[index] = distance;
  }
  __syncthreads();

  __shared__ u32 shared_beam_handles[kPersistentMaxBeam];
  __shared__ u32 shared_beam_ids[kPersistentMaxBeam];
  __shared__ f32 shared_beam_distances[kPersistentMaxBeam];
  __shared__ u8 shared_beam_expanded[kPersistentMaxBeam];
  __shared__ CandidateWorkspace candidate_workspace;
  u32* merge_handles = candidate_workspace.arrays.handles;
  u32* merge_ids = candidate_workspace.arrays.ids;
  f32* merge_distances = candidate_workspace.arrays.distances;
  u8* merge_expanded = candidate_workspace.arrays.expanded;
  u32* navigation_handles = params.navigation_candidate_handles +
    static_cast<size_t>(query_slot) * kPersistentMaxMergeCandidates;
  f32* navigation_distances = params.navigation_candidate_distances +
    static_cast<size_t>(query_slot) * kPersistentMaxMergeCandidates;
  u32* beam_handles = shared_beam_handles;
  u32* beam_ids = shared_beam_ids;
  f32* beam_distances = shared_beam_distances;
  u8* beam_expanded = shared_beam_expanded;
  const u32 traversal_capacity = min(kPersistentMaxBeam, params.traversal_beam_width);
  u32* visited = params.visited_hash +
    static_cast<size_t>(query_slot) * params.visited_capacity;
  for (u32 index = threadIdx.x; index < traversal_capacity; index += blockDim.x) {
    beam_handles[index] = UINT32_MAX;
    beam_ids[index] = UINT32_MAX;
    beam_distances[index] = FLT_MAX;
    beam_expanded[index] = 0;
  }
  for (u32 index = threadIdx.x; index < params.visited_capacity; index += blockDim.x) {
    visited[index] = UINT32_MAX;
  }
  __syncthreads();

  __shared__ u32 beam_count;
  __shared__ u32 rerank_handles[kPersistentMaxExact];
  __shared__ u32 rerank_ids[kPersistentMaxExact];
  __shared__ f32 rerank_distances[kPersistentMaxExact];
  __shared__ u32 rerank_count;
  __shared__ u32 total_exact_reads;
  __shared__ u32 total_exact_cache_hits;
  __shared__ u32 seed_count;
  __shared__ u32 selected_anchor_count;
  __shared__ u32 anchor_best_indices[256];
  if (params.anchor_count != 0 && params.anchor_vectors != nullptr &&
      params.anchor_handles != nullptr && params.anchor_pq_codes != nullptr) {
    constexpr u32 local_anchor_candidates = 2;
    const u32 candidates_per_thread =
      blockDim.x == kApproximateSortThreadsCompact ? 2u : 1u;
    u32 local_anchors[local_anchor_candidates];
    u32 local_handles[local_anchor_candidates];
    f32 local_distances[local_anchor_candidates];
    for (u32 index = 0; index < local_anchor_candidates; ++index) {
      local_anchors[index] = UINT32_MAX;
      local_handles[index] = UINT32_MAX;
      local_distances[index] = FLT_MAX;
    }
    for (u32 anchor = threadIdx.x; anchor < params.anchor_count; anchor += blockDim.x) {
      const u32 handle = params.anchor_handles[anchor];
      const f32 distance = approximate_entry(
        params, query_lut,
        params.anchor_pq_codes + static_cast<size_t>(anchor) * params.pq_code_bytes);
      u32 worst = 0;
      for (u32 index = 1; index < candidates_per_thread; ++index) {
        if (candidate_less(local_handles[worst], local_distances[worst],
                           local_handles[index], local_distances[index])) {
          worst = index;
        }
      }
      if (candidate_less(handle, distance,
                         local_handles[worst], local_distances[worst])) {
        local_anchors[worst] = anchor;
        local_handles[worst] = handle;
        local_distances[worst] = distance;
      }
    }
    for (u32 index = 0; index < candidates_per_thread; ++index) {
      const u32 output = threadIdx.x * candidates_per_thread + index;
      merge_handles[output] = local_handles[index];
      merge_ids[output] = local_anchors[index];
      merge_distances[output] = local_distances[index];
      merge_expanded[output] = 0;
    }
    __syncthreads();
    const u32 approximate_anchor_candidates =
      blockDim.x * candidates_per_thread;
    sort_candidates(merge_handles, merge_ids, merge_distances, merge_expanded,
                    approximate_anchor_candidates);
    if (threadIdx.x == 0) {
      u32 valid = 0;
      while (valid < approximate_anchor_candidates &&
             merge_ids[valid] != UINT32_MAX &&
             isfinite(merge_distances[valid]) &&
             merge_distances[valid] != FLT_MAX) {
        ++valid;
      }
      selected_anchor_count = min(valid, 256u);
    }
    __syncthreads();
    for (u32 index = threadIdx.x; index < selected_anchor_count;
         index += blockDim.x) {
      merge_distances[index] = exact_anchor_distance(params, query, merge_ids[index]);
      merge_expanded[index] = 0;
    }
    __syncthreads();
    sort_candidates(merge_handles, merge_ids, merge_distances, merge_expanded,
                    selected_anchor_count);
    if (threadIdx.x == 0) {
      selected_anchor_count = min(
        selected_anchor_count,
        max(min(params.entry_seed_count, traversal_capacity),
            min(params.delta_anchor_probes, kPersistentMaxAnchorProbes)));
      seed_count = min(selected_anchor_count,
                       min(params.entry_seed_count, traversal_capacity));
      for (u32 index = 0; index < selected_anchor_count; ++index) {
        anchor_best_indices[index] = merge_ids[index];
      }
    }
    __syncthreads();
    for (u32 seed = threadIdx.x; seed < seed_count; seed += blockDim.x) {
      const u32 handle = params.anchor_handles[anchor_best_indices[seed]];
      merge_handles[seed] = handle;
      merge_distances[seed] = approximate_handle(
        params, query_lut, handle, descriptor.snapshot_epoch);
      merge_expanded[seed] = 0;
    }
  } else {
    if (threadIdx.x == 0) {
      seed_count = min(params.entry_point_count, params.entry_seed_count);
      selected_anchor_count = 0;
    }
    for (u32 index = threadIdx.x; index < seed_count; index += blockDim.x) {
      const u32 handle = params.entry_points[index];
      merge_handles[index] = handle;
      merge_distances[index] = approximate_handle(
        params, query_lut, handle, descriptor.snapshot_epoch);
      merge_expanded[index] = 0;
    }
  }
  __syncthreads();
  sort_candidates(merge_handles, nullptr, merge_distances, merge_expanded,
                  seed_count);
  if (threadIdx.x == 0) {
    beam_count = min(seed_count, traversal_capacity);
    rerank_count = 0;
    total_exact_reads = 0;
    total_exact_cache_hits = 0;
    for (u32 index = 0; index < beam_count; ++index) {
      beam_handles[index] = merge_handles[index];
      beam_ids[index] = UINT32_MAX;
      beam_distances[index] = merge_distances[index];
      beam_expanded[index] = 0;
      insert_visited(visited, params.visited_capacity, beam_handles[index]);
    }
  }
  __syncthreads();
  if (beam_count == 0) {
    if (threadIdx.x == 0) {
      completion.status = -EIO;
      completion.gpu_cycles = clock64() - query_started_cycles;
      device_ring_push(params.completions, completion);
    }
    __syncthreads();
    return;
  }

  __shared__ u32 selected_handles[kPersistentMaxPrefetch];
  __shared__ u32 selected_count;
  __shared__ u32 neighbor_counts[kPersistentMaxPrefetch];
  __shared__ u32 neighbor_offsets[kPersistentMaxPrefetch + 1];
  __shared__ u32 flattened_neighbors;
  __shared__ u32 remote_reads_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 cache_hits_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 route_hits_by_lane[kPersistentMaxPrefetch];
  __shared__ u32 graph_cache_slots[kPersistentMaxPrefetch];
  __shared__ u32 total_remote_reads;
  __shared__ u32 total_remote_batches;
  __shared__ u32 total_graph_rounds;
  __shared__ u32 total_cache_hits;
  __shared__ u32 total_route_hits;
  __shared__ u32 graph_failed;
  if (threadIdx.x == 0) {
    total_remote_reads = 0;
    total_remote_batches = 0;
    total_graph_rounds = 0;
    total_cache_hits = 0;
    total_route_hits = 0;
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
          graph_cache_slots, remote_reads_by_lane, cache_hits_by_lane,
          route_hits_by_lane,
          &total_remote_batches)) {
      if (threadIdx.x == 0) graph_failed = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      graph_phase_cycles += clock64() - phase_started_cycles;
      for (u32 selected = 0; selected < selected_count; ++selected) {
        total_remote_reads += remote_reads_by_lane[selected];
        total_cache_hits += cache_hits_by_lane[selected];
        total_route_hits += route_hits_by_lane[selected];
      }
    }
    __syncthreads();
    if (graph_failed != 0) {
      for (u32 selected = warp; selected < selected_count;
           selected += blockDim.x / warp_width) {
        const u32 slot = graph_cache_slots[selected];
        if (lane_in_warp == 0 && slot != UINT32_MAX &&
            (slot & kGraphScratchBit) == 0) {
          __threadfence();
          release_graph_record(params, slot);
        }
        if (lane_in_warp == 0) graph_cache_slots[selected] = UINT32_MAX;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.remote_batches = total_remote_batches;
        completion.graph_rounds = total_graph_rounds;
        completion.cache_hits = total_cache_hits;
        completion.route_hits = total_route_hits;
        completion.exact_vectors = total_exact_reads;
        completion.exact_cache_hits = total_exact_cache_hits;
        device_ring_push(params.completions, completion);
      }
      __syncthreads();
      return;
    }

    for (u32 chunk_begin = 0; chunk_begin < selected_count;
         chunk_begin += kPersistentScoreChunk) {
      const u32 chunk_count = min(kPersistentScoreChunk,
                                  selected_count - chunk_begin);
      for (u32 local = warp; local < chunk_count;
           local += blockDim.x / warp_width) {
        const u32 selected = chunk_begin + local;
        const u32 slot = graph_cache_slots[selected];
        const u8* record = slot == UINT32_MAX ? nullptr :
          graph_record_pointer(params, descriptor.query_slot, slot);
        if (lane_in_warp == 0) {
          neighbor_counts[local] = record != nullptr && (record[1] & 1u) == 0
            ? min(static_cast<u32>(record[0]), params.graph_degree) : 0;
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
        const u32 slot = graph_cache_slots[selected];
        const u8* record = slot == UINT32_MAX ? nullptr :
          graph_record_pointer(params, descriptor.query_slot, slot);
        __syncwarp();
        const u32 count = neighbor_counts[local];
        for (u32 neighbor = lane_in_warp; neighbor < count; neighbor += warp_width) {
          const u64 raw = decode_compact_raw(record + 8 + neighbor * 5,
                                             params.graph_shard_bits);
          navigation_handles[neighbor_offsets[local] + neighbor] =
            handle_from_raw(params, raw);
        }
        __syncwarp();
        if (lane_in_warp == 0 && slot != UINT32_MAX &&
            (slot & kGraphScratchBit) == 0) {
          __threadfence();
          release_graph_record(params, slot);
        }
        if (lane_in_warp == 0) graph_cache_slots[selected] = UINT32_MAX;
      }
      __syncthreads();
      const u32 candidate_count = flattened_neighbors;
      for (u32 flat = threadIdx.x; flat < candidate_count; flat += blockDim.x) {
        const u32 handle = navigation_handles[flat];
        if (handle == UINT32_MAX ||
            !insert_visited(visited, params.visited_capacity, handle)) {
          navigation_handles[flat] = UINT32_MAX;
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
      for (u32 selected = warp; selected < selected_count;
           selected += blockDim.x / warp_width) {
        const u32 slot = graph_cache_slots[selected];
        if (lane_in_warp == 0 && slot != UINT32_MAX &&
            (slot & kGraphScratchBit) == 0) {
          __threadfence();
          release_graph_record(params, slot);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        completion.status = -EIO;
        completion.gpu_cycles = clock64() - query_started_cycles;
        completion.remote_pages = total_remote_reads;
        completion.remote_batches = total_remote_batches;
        completion.graph_rounds = total_graph_rounds;
        completion.cache_hits = total_cache_hits;
        completion.route_hits = total_route_hits;
        completion.exact_vectors = total_exact_reads;
        completion.exact_cache_hits = total_exact_cache_hits;
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
    rerank_count = 0;
    for (u32 index = 0;
         index < beam_count && rerank_count < params.final_rerank_width;
         ++index) {
      const u32 handle = merge_handles[index];
      if ((handle & kDeltaHandleBit) == 0 &&
          base_overridden(params, handle, descriptor.snapshot_epoch)) {
        continue;
      }
      rerank_handles[rerank_count] = handle;
      rerank_ids[rerank_count] = UINT32_MAX;
      rerank_distances[rerank_count] = merge_distances[index];
      ++rerank_count;
    }
    phase_started_cycles = clock64();
  }
  __syncthreads();
  exactify_into_beam(params, descriptor, query, rerank_handles, rerank_ids, rerank_distances,
                     rerank_count, beam_handles, beam_ids, beam_distances, beam_expanded,
                     beam_count, &total_exact_reads, &total_exact_cache_hits,
                     params.final_rerank_width, true, merge_handles, merge_ids,
                     merge_distances, merge_expanded);
  if (threadIdx.x == 0) {
    exact_phase_cycles += clock64() - phase_started_cycles;
  }
  __syncthreads();

  add_delta_candidates(params, descriptor, query, query_lut,
                       beam_handles, beam_ids, beam_distances, beam_expanded, beam_count,
                       params.final_rerank_width,
                       anchor_best_indices, selected_anchor_count);
  if (beam_count == 0) {
    if (threadIdx.x == 0) {
      completion.status = -EIO;
      completion.gpu_cycles = clock64() - query_started_cycles;
      completion.remote_pages = total_remote_reads;
      completion.remote_batches = total_remote_batches;
      completion.graph_rounds = total_graph_rounds;
      completion.cache_hits = total_cache_hits;
      completion.route_hits = total_route_hits;
      completion.exact_vectors = total_exact_reads;
      completion.exact_cache_hits = total_exact_cache_hits;
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
    completion.graph_rounds = total_graph_rounds;
    completion.cache_hits = total_cache_hits;
    completion.route_hits = total_route_hits;
    completion.exact_vectors = total_exact_reads;
    completion.exact_cache_hits = total_exact_cache_hits;
    device_ring_push(params.completions, completion);
  }
}

